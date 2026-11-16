"""Stage 6 - OPTIONAL Bayesian effect-size estimation (manuscript Suppl. S4.10). Not the primary inference.

The primary results are the permutation tests of Stage 7. This stage gives a model-based companion: posterior effect
sizes (group effect ASD - TD, the planned film contrast, their interaction) with 95 % HDIs, on the same models the
simulations evaluated (`simulations_multimodal_MVAR/dDTF/S13_group_film_bayes.bayes_records`), through the same
functions (`src.group_model.fit_model_with_priors`, `convergence_row`, `standardize_within_edge`;
`src.sim_group_bayes.population_effects`, `edge_effects`).

Input: Stage 5's Delta against the within-group null of each film (`stage05_delta_table_within_group.csv`), the
family edges (all 8 cross-person edges by default), standardized within edge, run per estimator.

Models (Student-t likelihood; Normal(0, 1) priors on fixed effects; half-Student-t(3, 1) on the SDs and sigma):
- main: pooled ``film * group + (film * group | edge) + (1 | dyad_id)``; if it fails the convergence gate, the
  fallback ``film * group + (group | edge) + (1 | dyad_id)``;
- check: ``film * group + (1 | edge) + (1 | dyad_id)`` (every edge gets the common effects);
- per-edge models ``film * group + (1 | dyad_id)``: NO multiplicity control (`FIT_PER_EDGE_MODELS`).
Convergence gate: R-hat < 1.01, bulk and tail ESS > 400, zero divergences (Pareto k reported, not gated).

The film effect is ``FILMS_BAYES[0] - mean(other films)`` (the first contrast of `FILM_CONTRASTS`).

Writes `06_group/<estimator>/stage06_effects.csv` and `stage06_diagnostics.csv`, and `06_group/group_model_gate.html`.
Older outputs of this stage (BH-FDR primary family, asymmetry track) are moved once to `06_group_archive_<date>/`.
"""

import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import bambi as bmb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.design import node_names
from src.group_model import convergence_row, fit_model_with_priors, standardize_within_edge
from src.io_utils import ensure_dir
from src.pipeline_config import load_stage_config
from src.sim_group_bayes import edge_effects, excludes_zero, population_effects

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this stage's own section).
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage06_group_model")

ANALYSIS_ROOT = Path(os.environ.get("SYNCC_ANALYSIS_ROOT", PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]))
INPUT_CSV = ANALYSIS_ROOT / CFG["INPUT_SUBDIR"] / CFG["INPUT_FILENAME"]
OUTPUT_DIR = ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"]
OLD_OUTPUT_MARKER = "stage06_primary_summary.csv"        # written only by the earlier (BH-FDR / asymmetry) version

ESTIMATORS = CFG["ESTIMATORS"]
GROUPS = CFG["GROUPS"]                                   # [reference, other]: group effect = GROUPS[1] - GROUPS[0]
FIRST_CONTRAST = next(iter(CFG["FILM_CONTRASTS_BAYES"].values()))
FILMS_BAYES = sorted(CFG["FILMS"], key=lambda f: -FIRST_CONTRAST[f])     # contrasted film first
assert FIRST_CONTRAST[FILMS_BAYES[0]] == 1 and all(np.isclose(FIRST_CONTRAST[f], -1 / (len(FILMS_BAYES) - 1))
                                                   for f in FILMS_BAYES[1:]), "Bayes film effect = one film vs the mean of the others"
NODES = CFG["nodes"]
ROLE = {node["name"]: node["role"] for node in NODES}
NODE_NAMES = node_names(NODES)
CROSS_EDGES = [f"{s}->{t}" for s in NODE_NAMES for t in NODE_NAMES if s != t and ROLE[s] != ROLE[t]]
FAMILY = CFG["FAMILY_EDGES"] or CROSS_EDGES
FIT_PER_EDGE_MODELS = CFG["FIT_PER_EDGE_MODELS"]
MCMC = {"draws": CFG["DRAWS"], "tune": CFG["TUNE"], "chains": CFG["CHAINS"], "target_accept": CFG["TARGET_ACCEPT"],
        "cores": CFG["CORES"], "seed": CFG["SEED"], "sd_prior_edge": "halfstudentt"}
GATE = {"rhat_max": CFG["RHAT_MAX"], "ess_min": CFG["ESS_MIN"], "pareto_k_max": np.inf}   # Pareto k reported, not gated
HDI_PROB = CFG["HDI_PROB"]

FIXED_PRIOR = bmb.Prior("Normal", mu=0, sigma=1)
SD_PRIOR = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))
PRIORS = {"Intercept": FIXED_PRIOR, "C(film, Sum)": FIXED_PRIOR, "C(group, Sum)": FIXED_PRIOR,
          "C(film, Sum):C(group, Sum)": FIXED_PRIOR, "1|dyad_id": SD_PRIOR, "1|edge": SD_PRIOR,
          "C(film, Sum)|edge": SD_PRIOR, "C(group, Sum)|edge": SD_PRIOR, "C(film, Sum):C(group, Sum)|edge": SD_PRIOR,
          "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1)}
FIXED = "dv ~ C(film, Sum) * C(group, Sum)"
MODELS = {   # label -> (formula, required group-specific prior keys); the models of S13_group_film_bayes
    "per_edge": (f"{FIXED} + (1|dyad_id)", []),
    "pooled_main": (f"{FIXED} + (C(film, Sum) * C(group, Sum) | edge) + (1|dyad_id)",
                    ["1|edge", "C(film, Sum)|edge", "C(group, Sum)|edge", "C(film, Sum):C(group, Sum)|edge"]),
    "pooled_fallback": (f"{FIXED} + (C(group, Sum) | edge) + (1|dyad_id)", ["1|edge", "C(group, Sum)|edge"]),
    "pooled_intercept_check": (f"{FIXED} + (1|edge) + (1|dyad_id)", ["1|edge"]),
}
LABELS = {"pooled_main": "pooled (film*group | edge), main", "pooled_fallback": "pooled (group | edge), fallback",
          "pooled_intercept_check": "pooled (1 | edge), check", "per_edge": "per edge, NO multiplicity control"}


def fit(label, data):
    """Fit one model; return (model, idata, gate row)."""
    formula, required = MODELS[label]
    model, idata = fit_model_with_priors(formula, data, PRIORS, required, MCMC)
    return model, idata, convergence_row(label, idata, len(data), 0, GATE["rhat_max"], GATE["ess_min"], GATE["pareto_k_max"])


def effect_rows(est, label, edge, effects, gate):
    """Rows of one model x edge (effects = `summarize_draws` dicts)."""
    return [{"estimator": est, "model": LABELS[label], "edge": edge, "effect": name, "estimate": s["estimate"],
             "hdi_low": s["hdi_low"], "hdi_high": s["hdi_high"], "hdi_excludes_0": excludes_zero(s),
             "gate_pass": gate["pass_l9"]} for name, s in effects.items()]


# ---------------------------------------------------------------------------
# Archive the outputs of the earlier version once (BH-FDR primary family, asymmetry track)
# ---------------------------------------------------------------------------
if (OUTPUT_DIR / OLD_OUTPUT_MARKER).exists():
    archive = OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_archive_{datetime.date.today().isoformat()}")
    OUTPUT_DIR.rename(archive)
    print(f"Moved the earlier stage-06 outputs to {archive}")
ensure_dir(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Fit per estimator
# ---------------------------------------------------------------------------
table = pd.read_csv(INPUT_CSV)
assert (table["null_scope"] == "within_group").all(), "stage 6 models Delta against the within-group null"
all_effects, all_gates = [], []
for est in ESTIMATORS:
    df = table[(table["estimator"] == est) & table["edge"].isin(FAMILY)]
    data, scale = standardize_within_edge(df, "delta")
    data = data.rename(columns={"delta_z": "dv"})
    data["film"] = pd.Categorical(data["film"], categories=FILMS_BAYES)
    data["group"] = pd.Categorical(data["group"], categories=GROUPS)
    data["edge"] = pd.Categorical(data["edge"], categories=FAMILY)
    print(f"\n=== {est}: {len(data)} rows ({data['dyad_id'].nunique()} dyads x films x {len(FAMILY)} edges) ===")
    rows, gates = [], []

    labels = ["pooled_main"]
    while labels:
        label = labels.pop(0)
        model, idata, gate = fit(label, data)
        gates.append(dict(gate, estimator=est, edge="all"))
        rows += effect_rows(est, label, "population", population_effects(model, idata, FILMS_BAYES, GROUPS, HDI_PROB), gate)
        for edge, effects in edge_effects(model, idata, FAMILY, FILMS_BAYES, GROUPS, data["dyad_id"].iloc[0], HDI_PROB).items():
            rows += effect_rows(est, label, edge, effects, gate)
        if label == "pooled_main" and not gate["pass_l9"]:
            labels.append("pooled_fallback")
        print(f"  {LABELS[label]}: gate {'pass' if gate['pass_l9'] else 'FAIL'}")

    model, idata, gate = fit("pooled_intercept_check", data)
    gates.append(dict(gate, estimator=est, edge="all"))
    rows += effect_rows(est, "pooled_intercept_check", "population",
                        population_effects(model, idata, FILMS_BAYES, GROUPS, HDI_PROB), gate)

    if FIT_PER_EDGE_MODELS:
        for edge in FAMILY:
            sub = data[data["edge"] == edge].reset_index(drop=True)
            model, idata, gate = fit("per_edge", sub)
            gates.append(dict(gate, estimator=est, edge=edge))
            rows += effect_rows(est, "per_edge", edge, population_effects(model, idata, FILMS_BAYES, GROUPS, HDI_PROB), gate)

    effects_df = pd.DataFrame(rows).merge(scale.rename(columns={"mean": "delta_mean", "sd": "delta_sd"}), on="edge", how="left")
    ensure_dir(OUTPUT_DIR / est)
    effects_df.to_csv(OUTPUT_DIR / est / "stage06_effects.csv", index=False)
    pd.DataFrame(gates).to_csv(OUTPUT_DIR / est / "stage06_diagnostics.csv", index=False)
    all_effects.append(effects_df)
    all_gates.append(pd.DataFrame(gates))

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
effects = pd.concat(all_effects, ignore_index=True)
gates = pd.concat(all_gates, ignore_index=True)
wide = effects.pivot_table(index=["model", "edge", "effect"], columns="estimator",
                           values=["estimate", "hdi_low", "hdi_high"], aggfunc="first", sort=False)
wide.columns = [f"{est} {val}" for val, est in wide.columns]
order = [e for e in ("dDTF", "GPDC") if e in ESTIMATORS]
wide = wide[[f"{e} {v}" for e in order for v in ("estimate", "hdi_low", "hdi_high")]].reset_index()
html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Bayesian effect sizes</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }}
  table {{ border-collapse: collapse; font-size: 0.82em; margin-bottom: 1em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.2em 0.45em; text-align: right; }}
</style></head><body>
<h1>Stage 6: optional Bayesian effect-size estimation (manuscript Suppl. S4.10)</h1>
<p><b>Not the primary inference</b> (that is Stage 7). Delta against the within-group null, standardized within edge
   (effects in SD units of each edge's Delta); family {', '.join(FAMILY)}; estimators dDTF / GPDC. Main model:
   pooled <code>film*group + (film*group|edge) + (1|dyad_id)</code> (fallback <code>(group|edge)</code> if it fails the
   gate; <code>(1|edge)</code> as a check). Per-edge models have no multiplicity control. Effects: group =
   {GROUPS[1]} - {GROUPS[0]} (mean over films), film = {FILMS_BAYES[0]} - mean of the others, interaction = film effect
   in {GROUPS[1]} minus in {GROUPS[0]}. {int(HDI_PROB * 100)} % HDI. Gate: R-hat &lt; {GATE['rhat_max']}, bulk/tail
   ESS &gt; {GATE['ess_min']}, no divergences.</p>
<h2>Convergence</h2>{gates.to_html(index=False, float_format=lambda v: f'{v:.4g}')}
<h2>Effects (dDTF / GPDC)</h2>{wide.to_html(index=False, float_format=lambda v: f'{v:.3f}')}
</body></html>"""
(OUTPUT_DIR / "group_model_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote effects, diagnostics and the gate to {OUTPUT_DIR}")
