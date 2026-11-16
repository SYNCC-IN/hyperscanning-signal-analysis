"""Stage 5 - pairing tables T (real and pseudo-dyads), Delta and robust z.

For every film, every child is paired with every caregiver who watched the same film: cell T[b, c] is the
band-averaged connectivity (coupling band 0.15-0.5 Hz, Box-Cox lambda 0.25) of the child-role rows of dyad b with the
caregiver-role rows of dyad c. The diagonal holds the real dyads, the off-diagonal cells the pseudo-dyads
(manuscript 2.5). Every cell is one MVAR fit on the analysis path shared with the simulations
(`src.surrogate.pair_tables` -> `src.sim_estimation.edge_matrices`: OLS on stacked 10 s / 50 % windows, no
detrending, p = 4), scored with every estimator in `ESTIMATORS` (GPDC, dDTF) from the same fit.

Instability (as `src.sim_inference.run_study`): a fit whose largest companion root is >= `STABILITY_MAX_ROOT`
leaves its cell NaN. A real dyad unstable at p = 4 is dropped, with its row and column, from everything computed
here and in Stage 7 (counted per film); an unstable pseudo-dyad is an empty cell, skipped in null medians, MADs and
permuted means.

Delta and robust z of every stable real dyad and edge (`src.sim_inference.delta_and_z_all`): Delta = real - median
of the dyad's pseudo-dyads, z = Delta / (1.4826 MAD), with three null scopes (`src.sim_inference.null_mask`):
'pooled' (all pseudo-dyads of the film), 'exclude' (pseudo-dyads containing neither member of the dyad) and
'within_group' (pseudo-dyads of the dyad's own group; the default for everything group-related). Dyad-level
empirical one-sided p-values (`src.sim_inference.dyad_pvalues`) are descriptive and not corrected.

A dyad that misses a film exists only in the tables of the films it has.

Writes, per film x estimator, `05_surrogate/<estimator>/<film>_pairing_table.nc` (T with dims edge x child_dyad x
cg_dyad, all 12 edges, NaN = unstable fit; per-dyad group, age, stability; the fit roots) -- Stage 7 reads these;
per null scope, `05_surrogate/stage05_delta_table_<scope>.csv` (one row per estimator x dyad x film x edge);
QC figures and `surrogate_gate.html`.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.design import assert_edges_known, node_names, window_geometry
from src.io_utils import ensure_dir, parse_case_filename
from src.pipeline_config import load_stage_config
from src.sim_inference import delta_and_z_all, dyad_pvalues, null_stats
from src.surrogate import edge_class_for, pair_tables, plot_delta_summary, plot_null_vs_real_violin

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this stage's own section).
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage05_surrogate")

ANALYSIS_ROOT = Path(os.environ.get("SYNCC_ANALYSIS_ROOT", PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]))  # env: synthetic runs
ENVELOPES_DIR = ANALYSIS_ROOT / CFG["ENVELOPES_SUBDIR"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

FILMS = CFG["FILMS"]
TARGET_SFREQ = CFG["TARGET_SFREQ"]
NODES = CFG["nodes"]
NODE_NAMES = node_names(NODES)
ROLE = {node["name"]: node["role"] for node in NODES}
assert_edges_known([(e["source"], e["target"]) for e in CFG["edge_topology"]], NODE_NAMES,
                   context="stage05 pipeline_config.json's edge_topology")

ESTIMATORS = CFG["ESTIMATORS"]
MODEL_ORDER = CFG["COMMON_MODEL_ORDER"]          # fixed for real and pseudo-dyads (manuscript 2.3)
FIT_METHOD = CFG["FIT_METHOD"]
DETREND_TYPE = CFG["DETREND_TYPE"]
WIN_LEN_S, OVERLAP_FRAC = CFG["WIN_LEN_S"], CFG["OVERLAP_FRAC"]
WIN_LEN, STEP = window_geometry(WIN_LEN_S, OVERLAP_FRAC, TARGET_SFREQ)
FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
COUPLING_BAND_HZ = tuple(CFG["COUPLING_BAND_HZ"])
BOX_COX_LAMBDA = CFG["BOX_COX_LAMBDA"]
STABILITY_MAX_ROOT = CFG["STABILITY_MAX_ROOT"]
FFDTF_ROWSUM_TOL = CFG["FFDTF_ROWSUM_TOL"]

NULL_SCOPES = CFG["NULL_SCOPES"]                 # 'pooled', 'exclude', 'within_group'
DEFAULT_SCOPE = CFG["DEFAULT_SCOPE"]             # 'within_group'
GROUP_CODES = CFG["GROUP_CODES"]                 # {'TD': 0, 'ASD': 1}; group contrasts are ASD - TD

ALL_EDGES = [(s, t) for s in NODE_NAMES for t in NODE_NAMES if s != t]          # 12 directed edges
EDGE_NAMES = [f"{s}->{t}" for s, t in ALL_EDGES]
EDGE_CLASS = {(e["source"], e["target"]): e["class"] for e in CFG["edge_topology"]}
TOPOLOGY_EDGES = [(e["source"], e["target"]) for e in CFG["edge_topology"]]
CONFIRMATORY_EDGES = [(e["source"], e["target"]) for e in CFG["edge_topology"] if e["class"] != "exploratory"]
assert WIN_LEN > MODEL_ORDER, f"win_len={WIN_LEN} must exceed p={MODEL_ORDER}"


def edge_slices(table):
    """(..., k, k) -> (..., 12) in `ALL_EDGES` order (``[target, source]`` cells)."""
    return np.stack([table[..., NODE_NAMES.index(t), NODE_NAMES.index(s)] for s, t in ALL_EDGES], axis=-1)


# ---------------------------------------------------------------------------
# 1. Per film: pairing tables (one fit per cell, all estimators), stability
# ---------------------------------------------------------------------------
cases_by_film = {film: [] for film in FILMS}
for nc_path in sorted(ENVELOPES_DIR.glob("*.nc")):
    dyad_id, film = parse_case_filename(nc_path, FILMS)
    cases_by_film[film].append(dyad_id)

rows_by_scope = {scope: [] for scope in NULL_SCOPES}
film_summaries, violin_images = [], []
for film in FILMS:
    dyad_ids = sorted(cases_by_film[film])
    envelopes = [xr.load_dataarray(ENVELOPES_DIR / f"{d}_{film}.nc") for d in dyad_ids]
    for d, env in zip(dyad_ids, envelopes):
        assert env.attrs["fs"] == TARGET_SFREQ, f"fs mismatch for {d} {film}"
    groups_str = np.array([str(env.attrs["group"]) for env in envelopes])
    ages = np.array([float(env.attrs["age_months"]) for env in envelopes])
    print(f"\n--- {film}: {len(dyad_ids)} dyads ({dict(zip(*np.unique(groups_str, return_counts=True)))}), "
          f"{len(dyad_ids) ** 2} fits ---")

    tables, roots = pair_tables(envelopes, NODES, MODEL_ORDER, WIN_LEN, STEP, FREQS, TARGET_SFREQ, ESTIMATORS,
                                COUPLING_BAND_HZ, BOX_COX_LAMBDA, DETREND_TYPE, FIT_METHOD, STABILITY_MAX_ROOT)
    real_stable = np.diag(roots) < STABILITY_MAX_ROOT
    off_diagonal = ~np.eye(len(dyad_ids), dtype=bool)
    n_pairs_unstable = int((roots[off_diagonal] >= STABILITY_MAX_ROOT).sum())
    unstable_real = [d for d, ok in zip(dyad_ids, real_stable) if not ok]
    print(f"  real dyads unstable at p={MODEL_ORDER} (dropped with row and column): {len(unstable_real)} {unstable_real}")
    print(f"  pseudo-dyads unstable (empty cells): {n_pairs_unstable} of {int(off_diagonal.sum())}")

    keep = np.flatnonzero(real_stable)
    kept_ids = [dyad_ids[i] for i in keep]
    groups = np.array([GROUP_CODES[g] for g in groups_str[keep]])
    film_summaries.append({"film": film, "n_dyads": len(dyad_ids), "n_real_unstable": len(unstable_real),
                           "unstable_real": unstable_real, "n_stable": len(keep),
                           "n_stable_by_group": {g: int((groups_str[keep] == g).sum()) for g in GROUP_CODES},
                           "n_pairs": int(off_diagonal.sum()), "n_pairs_unstable": n_pairs_unstable})

    for est in ESTIMATORS:
        table = tables[est]
        if BOX_COX_LAMBDA == -1:      # bounds hold for the untransformed estimators only
            finite = table[np.isfinite(table)]
            assert finite.min() >= -1e-9 and finite.max() <= 1 + 1e-9, f"{est} out of [0,1] in {film}"
        out_dir = ensure_dir(OUTPUT_DIR / est)
        xr.Dataset(
            {"T": (("edge", "child_dyad", "cg_dyad"), np.moveaxis(edge_slices(table), -1, 0)),
             "fit_root": (("child_dyad", "cg_dyad"), roots),
             "group": (("dyad",), groups_str), "age_months": (("dyad",), ages), "real_stable": (("dyad",), real_stable)},
            coords={"edge": EDGE_NAMES, "child_dyad": dyad_ids, "cg_dyad": dyad_ids, "dyad": dyad_ids,
                    "source": ("edge", [s for s, _ in ALL_EDGES]), "target": ("edge", [t for _, t in ALL_EDGES]),
                    "edge_class": ("edge", [edge_class_for(s, t, EDGE_CLASS) for s, t in ALL_EDGES]),
                    "cross_person": ("edge", [ROLE[s] != ROLE[t] for s, t in ALL_EDGES])},
            attrs={"estimator": est, "film": film, "model_order": MODEL_ORDER, "fit_method": FIT_METHOD,
                   "detrend_type": DETREND_TYPE, "win_len": WIN_LEN, "step": STEP, "fs": TARGET_SFREQ,
                   "coupling_band_low": COUPLING_BAND_HZ[0], "coupling_band_high": COUPLING_BAND_HZ[1],
                   "box_cox_lambda": BOX_COX_LAMBDA, "stability_max_root": STABILITY_MAX_ROOT,
                   "convention": "T[edge, b, c] = edge value of child-role rows of dyad b with caregiver-role rows "
                                 "of dyad c; diagonal = real dyads; NaN = unstable fit"},
        ).to_netcdf(out_dir / f"{film}_pairing_table.nc")

        # --- Delta, z, dyad p-values on the stable dyads (sim_inference functions) ---
        real = table[keep, keep]                                    # (n, k, k)
        pair_values = table[np.ix_(keep, keep)].copy()              # (n, n, k, k)
        pair_values[np.arange(len(keep)), np.arange(len(keep))] = np.nan
        for scope in NULL_SCOPES:
            delta, z = delta_and_z_all(real, pair_values, scope, groups)
            p_dyad = dyad_pvalues(real, pair_values, scope, groups)
            for i, dyad_id in enumerate(kept_ids):
                median, mad, n_null = null_stats(pair_values, i, scope, groups)
                for s, t in ALL_EDGES:
                    ti, si = NODE_NAMES.index(t), NODE_NAMES.index(s)
                    rows_by_scope[scope].append({
                        "estimator": est, "null_scope": scope, "dyad_id": dyad_id, "film": film,
                        "source": s, "target": t, "edge": f"{s}->{t}", "edge_class": edge_class_for(s, t, EDGE_CLASS),
                        "cross_person": ROLE[s] != ROLE[t], "group": groups_str[keep[i]], "age_months": ages[keep[i]],
                        "real_value": real[i, ti, si], "null_median": median[ti, si], "null_mad": mad[ti, si],
                        "n_null": n_null, "delta": delta[i, ti, si], "z": z[i, ti, si], "p_dyad": p_dyad[i, ti, si]})

        # --- QC violin: pooled pseudo-dyad values vs real dyads ---
        null_matrix = edge_slices(pair_values[np.isfinite(pair_values[:, :, 0, 1])])
        real_by_dyad = {d: {"band_avg": real[i], "group": groups_str[keep[i]]} for i, d in enumerate(kept_ids)}
        for delta_space, tag in ((False, "null_hist"), (True, "delta_violin")):
            fig = plot_null_vs_real_violin(TOPOLOGY_EDGES, null_matrix, real_by_dyad, ALL_EDGES, EDGE_CLASS, NODE_NAMES,
                                           est, BOX_COX_LAMBDA, f"{film} {est}: pseudo-dyads (pooled) vs real dyads",
                                           delta_space=delta_space)
            name = f"{film}_{est}_{tag}.png"
            fig.savefig(QC_DIR / name)
            plt.close(fig)
            violin_images.append(name)

# ---------------------------------------------------------------------------
# 2. Tidy tables (one per null scope, all estimators) and summary figure
# ---------------------------------------------------------------------------
tables_by_scope = {scope: pd.DataFrame(rows) for scope, rows in rows_by_scope.items()}
for scope, df in tables_by_scope.items():
    df.to_csv(OUTPUT_DIR / f"stage05_delta_table_{scope}.csv", index=False)
    print(f"Wrote {len(df)} rows to stage05_delta_table_{scope}.csv")

summary_images = []
default_df = tables_by_scope[DEFAULT_SCOPE]
for est in ESTIMATORS:
    fig = plot_delta_summary(default_df[default_df["estimator"] == est], CONFIRMATORY_EDGES,
                             f"{est}: delta by group, H2/H4 edges ({DEFAULT_SCOPE} null, pooled across films)")
    name = f"delta_summary_{est}.png"
    fig.savefig(QC_DIR / name)
    plt.close(fig)
    summary_images.append(name)

summary_lines = []
for fs_ in film_summaries:
    summary_lines.append(f"{fs_['film']}: dyads={fs_['n_dyads']}  stable real fits={fs_['n_stable']} "
                         f"{fs_['n_stable_by_group']}  real dyads dropped (unstable)={fs_['n_real_unstable']} "
                         f"{fs_['unstable_real']}  unstable pseudo-dyads={fs_['n_pairs_unstable']}/{fs_['n_pairs']}")
summary_lines.append("")
for (est, film, edge), sub in default_df[default_df["cross_person"]].groupby(["estimator", "film", "edge"], sort=False):
    by_group = "  ".join(f"{g}: delta={gdf['delta'].mean():+.4f} z={gdf['z'].mean():+.2f}" for g, gdf in sub.groupby("group"))
    summary_lines.append(f"{est} {film} {edge} ({sub['edge_class'].iloc[0]}): {by_group}")
print("\n".join(summary_lines))

# ---------------------------------------------------------------------------
# 3. HTML gate
# ---------------------------------------------------------------------------
html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 5 pairing-table gate</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 0.5em; }}
  .row img {{ max-width: 900px; border: 1px solid #ccc; }}
  pre {{ font-family: monospace; }}
</style></head><body>
<h1>Stage 5 pairing tables, Delta and z</h1>
<p>Estimators <b>{' / '.join(ESTIMATORS)}</b>; fit <b>{FIT_METHOD}</b>, p=<b>{MODEL_ORDER}</b> (real and pseudo-dyads),
   window {WIN_LEN_S:g} s / {int(OVERLAP_FRAC * 100)} %, detrend=<b>{DETREND_TYPE}</b>, coupling band
   <b>{COUPLING_BAND_HZ[0]}-{COUPLING_BAND_HZ[1]} Hz</b>, Box-Cox lambda=<b>{BOX_COX_LAMBDA}</b>, stability: largest companion
   root &lt; {STABILITY_MAX_ROOT}. Null scopes {', '.join(NULL_SCOPES)}; default for group questions: <b>{DEFAULT_SCOPE}</b>.
   Delta and z are effect sizes; the dyad-level p-values in the CSVs are descriptive and uncorrected.
   Inference: Stage 7 (permutation tests on the pairing tables).</p>
<h2>Stable fits per film</h2>
<pre>{chr(10).join(summary_lines[:len(film_summaries)])}</pre>
<h2>Group means of Delta and z, cross-person edges ({DEFAULT_SCOPE} null)</h2>
<pre>{chr(10).join(summary_lines[len(film_summaries) + 1:])}</pre>
<div class="row">{''.join(f'<img src="qc/{n}" alt="{n}">' for n in summary_images)}</div>
<h2>Pseudo-dyads (pooled) vs real dyads, per film and estimator</h2>
<div class="row">{''.join(f'<img src="qc/{n}" alt="{n}">' for n in violin_images)}</div>
</body></html>
"""
(OUTPUT_DIR / "surrogate_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote gate to {OUTPUT_DIR / 'surrogate_gate.html'}")
