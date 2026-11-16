"""Stage 3 - design matrix, descriptive order selection, and MVAR fit diagnostics.

Reads Stage 2's `02_envelopes/<dyad_id>_<film>.nc` design files and, for every case:

1. Builds the z-scored `(4, n_samples)` design matrix (`src.design.assemble_design_matrix`) and reports the
   information-criterion order `p_used` of the global 2-D signal (`src.mvar_diag.select_p_used`). This is
   DESCRIPTIVE only: the analysis order is the fixed `COMMON_MODEL_ORDER` (p = 4, manuscript 2.3) for real and
   pseudo-dyads alike, so every Delta baseline is at the same order.
2. Fits the analysis model exactly as Stages 4-5 do (`src.sim_estimation.fit_mvar`): least squares on the stacked
   10 s / 50 % windows (`FIT_METHOD='ols'`), windows fitted as cut (`DETREND_TYPE='none'`), order p = 4.
3. Runs the diagnostics (`residual_whiteness`, `ar_root_stability`) on that fit and on a global single-window fit
   of the same order, for comparison.

Writes, per dyad x film, `03_mvar/<dyad_id>_<film>_order.json`; across all cases, `03_mvar/stage03_manifest.csv`; a
window-choice sensitivity check on 1-2 dyads (gate-only); and a QC gate (`qc/*.png` + `mvar_order_gate.html`).
Each case also gets a connectivity/spectra grid figure (`src.mtmvar.plot_mvar_grid`, `PRIMARY_ESTIMATOR`, no
Box-Cox) at a fixed order and window geometry (`GRID_MODEL_ORDER`, `GRID_WIN_LEN_S`/`GRID_OVERLAP_FRAC`).

Known caveat: `src.mtmvar.mvar_criterion` works on a 2-D signal only, so the descriptive order selection uses the
global signal, not the windows.
"""

import json
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

from src.design import assemble_design_matrix, detrend_windows, node_names, rows_for_signal, window_geometry, window_stack
from src.io_utils import ensure_dir, parse_case_filename
from src.connectivity import Granger_estimator
from src.mtmvar import plot_mvar_grid
from src.sim_estimation import fit_mvar
from src.mvar_diag import (
    ar_root_stability, residual_whiteness, select_p_used,
    plot_order_curves, plot_roots_comparison, plot_acf_comparison, plot_detrend_example,
)
from src.pipeline_config import load_stage_config
from src.reporting import render_dyad_panel_mvar_order

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this
# stage's own section); only paths/values computed from PROJECT_ROOT or
# other config values stay here. See src.pipeline_config.load_stage_config.
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage03_mvar_order")

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
ENVELOPES_DIR = ANALYSIS_ROOT / CFG["ENVELOPES_SUBDIR"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

FILMS = CFG["FILMS"]
TARGET_SFREQ = CFG["TARGET_SFREQ"]  # must match Stage 2's realized design-file rate

# Node topology (single source of truth for MVAR row order -- see
# `src.design.node_names`/`rows_for_signal`).
NODES = CFG["nodes"]
NODE_NAMES = node_names(NODES)

# Order selection: global 2-D signal only (mvar_criterion caveat, see module docstring).
CRIT_TYPES = CFG["CRIT_TYPES"]
PRIMARY_CRIT = CFG["PRIMARY_CRIT"]       # parsimonious default for n ~ 150 (see Stage 3 §5 in the plan)
MAX_MODEL_ORDER = CFG["MAX_MODEL_ORDER"]      # short-segment guard, not a scientific claim

# D2: diagnostic-only sub-block orders, one per DISTINCT signal type actually
# present in NODES (not a fixed EEG/HRV pair) -- a signal with no rows is
# skipped inside select_p_used. Alternative (not implemented): drop sub-block
# diagnostics entirely.
SIGNAL_ROW_GROUPS = {signal: rows_for_signal(NODES, signal) for signal in dict.fromkeys(node["signal"] for node in NODES)}

# Analysis fit (manuscript 2.3), identical in Stages 3-5: 10 s windows, 50 % overlap, no detrending, OLS on the
# stacked windows, fixed order p = 4.
WIN_LEN_S = CFG["WIN_LEN_S"]
OVERLAP_FRAC = CFG["OVERLAP_FRAC"]
DETREND_TYPE = CFG["DETREND_TYPE"]
FIT_METHOD = CFG["FIT_METHOD"]
MODEL_ORDER = CFG["COMMON_MODEL_ORDER"]

RESIDUAL_ACF_MAX_LAG = CFG["RESIDUAL_ACF_MAX_LAG"]   # must stay well below win_len - MODEL_ORDER
MIN_WHITE_FRACTION = CFG["MIN_WHITE_FRACTION"]   # quality flag: per-variable fraction of lags within the Bartlett band

# Per-case connectivity/spectra grid figure (src.mtmvar.plot_mvar_grid): fixed model order and window geometry,
# primary estimator, for a comparable view across all cases. The grid/sensitivity plots use this stage's own
# GRID_BOX_COX_LAMBDA (default -1 = no transform) to show the untransformed values.
ESTIMATOR = CFG["PRIMARY_ESTIMATOR"]
BOX_COX_LAMBDA = CFG["GRID_BOX_COX_LAMBDA"]
GRID_MODEL_ORDER = CFG["GRID_MODEL_ORDER"]
GRID_WIN_LEN_S = CFG["GRID_WIN_LEN_S"]
GRID_OVERLAP_FRAC = CFG["GRID_OVERLAP_FRAC"]
GRID_FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
GRID_SCALE = CFG["GRID_SCALE"]

# Window-choice sensitivity check (gate-only): the analysis window (10 s/50%) against two alternatives, at the
# analysis order.
SENSITIVITY_DYADS = [tuple(pair) for pair in CFG["SENSITIVITY_DYADS"]]
SENSITIVITY_WINDOW_CONFIGS = [tuple(cfg) for cfg in CFG["SENSITIVITY_WINDOW_CONFIGS"]]
SENSITIVITY_FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
COUPLING_BAND_HZ = tuple(CFG["COUPLING_BAND_HZ"])  # band-averaged connectivity = the DV substrate
PRIMARY_EDGES = [
    (edge["source"], edge["target"]) for edge in CFG["edge_topology"] if edge["class"] != "exploratory"
]




# ---------------------------------------------------------------------------
# 1. Per-case: descriptive order selection, analysis fit (p = MODEL_ORDER), global-comparison fit, diagnostics
# ---------------------------------------------------------------------------
nc_paths = sorted(ENVELOPES_DIR.glob("*.nc"))
print(f"Stage 3: {len(nc_paths)} dyad x film design files found in {ENVELOPES_DIR}")

manifest_rows = []
gate_entries = []

for nc_path in nc_paths:
    dyad_id, film = parse_case_filename(nc_path, FILMS)
    envelopes = xr.load_dataarray(nc_path)
    design = assemble_design_matrix(envelopes, NODE_NAMES, zscore=True)
    k, n_samples = design.shape

    order = select_p_used(design, MAX_MODEL_ORDER, CRIT_TYPES, PRIMARY_CRIT, SIGNAL_ROW_GROUPS)
    p_used = order["p_used"]

    win_len, step = window_geometry(WIN_LEN_S, OVERLAP_FRAC, TARGET_SFREQ)
    assert win_len > MODEL_ORDER, f"win_len={win_len} must exceed p={MODEL_ORDER} for {dyad_id} {film}"

    stack = window_stack(design, win_len, step)
    stack_detrended = detrend_windows(stack, dtype=DETREND_TYPE)   # 'none' = windows as cut
    n_windows = stack_detrended.shape[2]

    # Analysis fit (the one Stages 4-5 use) + diagnostics.
    ar_coeffs, _, _ = fit_mvar(design, MODEL_ORDER, win_len, step, DETREND_TYPE, FIT_METHOD)
    roots, max_abs_root, stable = ar_root_stability(ar_coeffs)
    residual_acf, whiteness_summary = residual_whiteness(stack_detrended, ar_coeffs, RESIDUAL_ACF_MAX_LAG)
    n_eff_windowed = win_len - MODEL_ORDER
    band_windowed = 1.96 / np.sqrt(n_eff_windowed)

    # Global single-window comparison fit, same order and method.
    ar_coeffs_global, _, _ = fit_mvar(design, MODEL_ORDER, None, None, method=FIT_METHOD)
    roots_global, max_abs_root_global, stable_global = ar_root_stability(ar_coeffs_global)
    design_1window = design[:, :, np.newaxis]
    residual_acf_global, whiteness_summary_global = residual_whiteness(design_1window, ar_coeffs_global, RESIDUAL_ACF_MAX_LAG)
    band_global = 1.96 / np.sqrt(n_samples - MODEL_ORDER)

    quality_reasons = []
    if not stable:
        quality_reasons.append("unstable")
    if any(fraction < MIN_WHITE_FRACTION for fraction in whiteness_summary.values()):
        quality_reasons.append("residual_autocorrelation")
    quality_ok = len(quality_reasons) == 0

    record = {
        "dyad_id": dyad_id, "film": film,
        "group": str(envelopes.attrs.get("group", "")),
        "age_months": float(envelopes.attrs["age_months"]),
        "n_samples": n_samples,
        "p_used": p_used, "model_order": MODEL_ORDER, "fit_method": FIT_METHOD,
        "p_by_signal": {signal: orders[PRIMARY_CRIT] for signal, orders in order["orders_by_signal"].items()},
        "primary_crit": PRIMARY_CRIT,
        "orders_full": order["orders_full"], "orders_by_signal": order["orders_by_signal"],
        "order_at_cap": order["order_at_cap"],
        "win_len": win_len, "step": step, "step_s": step / TARGET_SFREQ, "n_windows": n_windows,
        "detrend_type": DETREND_TYPE,
        "max_abs_root": max_abs_root, "stable": stable,
        "whiteness_summary": {NODE_NAMES[c]: v for c, v in whiteness_summary.items()},
        "max_abs_root_global": max_abs_root_global, "stable_global": stable_global,
        "whiteness_summary_global": {NODE_NAMES[c]: v for c, v in whiteness_summary_global.items()},
        "quality_ok": quality_ok, "quality_reasons": quality_reasons,
    }
    (OUTPUT_DIR / f"{dyad_id}_{film}_order.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest_rows.append({
        "dyad_id": dyad_id, "film": film, "group": record["group"], "n_samples": n_samples,
        "p_used": p_used, "model_order": MODEL_ORDER, "fit_method": FIT_METHOD, "order_at_cap": order["order_at_cap"],
        **{f"p_{signal}": p for signal, p in record["p_by_signal"].items()},
        "n_windows": n_windows, "max_abs_root": max_abs_root, "stable": stable,
        "max_abs_root_global": max_abs_root_global,
        "min_white_fraction": min(whiteness_summary.values()),
        "min_white_fraction_global": min(whiteness_summary_global.values()),
        "quality_ok": quality_ok, "quality_reasons": ";".join(quality_reasons),
    })

    # --- QC figures ---
    case_title = f"{dyad_id} {film}"
    curves_by_block = {f"full ({k}-var)": order["curves_full"]}
    orders_by_block = {f"full ({k}-var)": order["orders_full"]}
    for signal, curves in order["curves_by_signal"].items():
        block_label = f"{signal} ({len(SIGNAL_ROW_GROUPS[signal])}-var)"
        curves_by_block[block_label] = curves
        orders_by_block[block_label] = order["orders_by_signal"][signal]
    fig = plot_order_curves(
        order["order_range"], curves_by_block, orders_by_block,
        MAX_MODEL_ORDER, f"{case_title}: model-order criteria (global 2-D signal)",
    )
    order_curves_path = QC_DIR / f"{dyad_id}_{film}_order_curves.png"
    fig.savefig(order_curves_path)
    plt.close(fig)

    fig = plot_roots_comparison(roots_global, roots, max_abs_root_global, max_abs_root,
                                 f"{case_title}: AR roots, global vs windowed (p={MODEL_ORDER}, {FIT_METHOD})")
    roots_path = QC_DIR / f"{dyad_id}_{film}_roots_comparison.png"
    fig.savefig(roots_path)
    plt.close(fig)

    fig = plot_acf_comparison(residual_acf_global, residual_acf, band_global, band_windowed, NODE_NAMES,
                               f"{case_title}: residual ACF, global vs windowed (p={MODEL_ORDER}, {FIT_METHOD})")
    acf_path = QC_DIR / f"{dyad_id}_{film}_acf_comparison.png"
    fig.savefig(acf_path)
    plt.close(fig)

    fig = plot_detrend_example(stack, stack_detrended, NODE_NAMES, WIN_LEN_S, COUPLING_BAND_HZ,
                                f"{case_title}: example windows, pre- vs post-detrend")
    detrend_path = QC_DIR / f"{dyad_id}_{film}_detrend_example.png"
    fig.savefig(detrend_path)
    plt.close(fig)

    fig = plot_mvar_grid(
        design, GRID_MODEL_ORDER, GRID_WIN_LEN_S, GRID_OVERLAP_FRAC, TARGET_SFREQ, DETREND_TYPE, GRID_FREQS, NODE_NAMES,
        f"{case_title}: {ESTIMATOR} grid (p={GRID_MODEL_ORDER}, window={GRID_WIN_LEN_S:g}s/{int(GRID_OVERLAP_FRAC * 100)}%)",
        COUPLING_BAND_HZ, scale=GRID_SCALE, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA, method=FIT_METHOD
    )
    mvar_grid_path = QC_DIR / f"{dyad_id}_{film}_mvar_grid.png"
    fig.savefig(mvar_grid_path)
    plt.close(fig)

    gate_entries.append({
        "dyad_id": dyad_id, "film": film, "group": record["group"],
        "p_used": p_used, "model_order": MODEL_ORDER, "n_windows": n_windows, "max_abs_root": max_abs_root, "stable": stable,
        "min_white_fraction": min(whiteness_summary.values()), "quality_ok": quality_ok,
        "order_curves": order_curves_path.name, "roots": roots_path.name,
        "acf": acf_path.name, "detrend": detrend_path.name, "mvar_grid": mvar_grid_path.name,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage03_manifest.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Run summary
# ---------------------------------------------------------------------------
print(f"\n=== Stage 3 summary ({len(manifest_df)} cases, p={MODEL_ORDER} ({FIT_METHOD}, detrend={DETREND_TYPE}), descriptive crit={PRIMARY_CRIT}, "
      f"window={WIN_LEN_S}s/{int(OVERLAP_FRAC * 100)}% overlap) ===")
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    n_stable = int(group_df["stable"].sum())
    n_white = int((group_df["min_white_fraction"] >= MIN_WHITE_FRACTION).sum())
    n_quality_ok = int(group_df["quality_ok"].sum())
    n_stable_g = int((group_df["max_abs_root_global"] < 1.0).sum())
    n_white_g = int((group_df["min_white_fraction_global"] >= MIN_WHITE_FRACTION).sum())
    print(f"{group_label}: n={n}  windowed: stable={n_stable}/{n} white={n_white}/{n} quality_ok={n_quality_ok}/{n}  "
          f"|  global: stable={n_stable_g}/{n} white={n_white_g}/{n}")

n_at_cap = int(manifest_df["order_at_cap"].sum())
if n_at_cap:
    print(f"\n{n_at_cap} case(s) had a criterion pick land at MAX_MODEL_ORDER={MAX_MODEL_ORDER}:")
    for _, row in manifest_df[manifest_df["order_at_cap"]].iterrows():
        print(f"  {row['dyad_id']} {row['film']}")

print(f"\nWrote {len(manifest_df)} order files + manifest to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Window-choice sensitivity check (gate-only)
# ---------------------------------------------------------------------------
sensitivity_results = []
for dyad_id, film in SENSITIVITY_DYADS:
    nc_path = ENVELOPES_DIR / f"{dyad_id}_{film}.nc"
    envelopes = xr.load_dataarray(nc_path)
    design = assemble_design_matrix(envelopes, NODE_NAMES, zscore=True)

    p_used = MODEL_ORDER  # the analysis order, fixed across window configs

    spectra_by_config = {}
    for win_len_s, overlap_frac in SENSITIVITY_WINDOW_CONFIGS:
        win_len, step = window_geometry(win_len_s, overlap_frac, TARGET_SFREQ)
        assert win_len > p_used, f"win_len={win_len} must exceed p_used={p_used} for sensitivity config {win_len_s}s"
        granger_estimator, _ = Granger_estimator(design, SENSITIVITY_FREQS, TARGET_SFREQ, p_used, win_len, step,
                                                 DETREND_TYPE, ESTIMATOR, BOX_COX_LAMBDA, FIT_METHOD)
        spectra_by_config[(win_len_s, overlap_frac)] = granger_estimator

    band_mask = (SENSITIVITY_FREQS >= COUPLING_BAND_HZ[0]) & (SENSITIVITY_FREQS <= COUPLING_BAND_HZ[1])

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True)
    band_table_rows = []
    for edge_idx, (source_name, target_name) in enumerate(PRIMARY_EDGES):
        source, target = NODE_NAMES.index(source_name), NODE_NAMES.index(target_name)
        axis = axes.flat[edge_idx]
        for config, granger_estimator in spectra_by_config.items():
            label = f"{config[0]:g}s/{int(config[1] * 100)}%"
            axis.plot(SENSITIVITY_FREQS, granger_estimator[target, source], label=label)
            band_table_rows.append({
                "edge": f"{source_name} -> {target_name}", "window_config": label,
                "band_avg_granger": float(granger_estimator[target, source][band_mask].mean()),
            })
        axis.axvspan(*COUPLING_BAND_HZ, color="grey", alpha=0.15)
        axis.set_title(f"{source_name} -> {target_name}", fontsize=9)
        axis.set_xlabel("Frequency (Hz)")
        axis.legend(fontsize=7)
    axes[0, 0].set_ylabel(ESTIMATOR)
    axes[1, 0].set_ylabel(ESTIMATOR)
    figure.suptitle(f"{dyad_id} {film}: window-choice sensitivity (p={p_used}, {FIT_METHOD})")
    figure.tight_layout()
    sensitivity_plot_path = QC_DIR / f"{dyad_id}_{film}_sensitivity.png"
    figure.savefig(sensitivity_plot_path)
    plt.close(figure)

    band_table = pd.DataFrame(band_table_rows)
    sensitivity_results.append({
        "dyad_id": dyad_id, "film": film, "p_used": p_used,
        "plot": sensitivity_plot_path.name, "band_table": band_table,
    })

sensitivity_table_path = OUTPUT_DIR / "sensitivity_band_averages.csv"
pd.concat([r["band_table"].assign(dyad_id=r["dyad_id"], film=r["film"]) for r in sensitivity_results]).to_csv(
    sensitivity_table_path, index=False,
)
print(f"\nWrote window-choice sensitivity check ({len(sensitivity_results)} cases) to {sensitivity_table_path}")

# ---------------------------------------------------------------------------
# 4. Interactive HTML gate
# ---------------------------------------------------------------------------
gate_by_dyad = {}
for entry in gate_entries:
    gate_by_dyad.setdefault(entry["dyad_id"], []).append(entry)
gate_dyad_ids = sorted(gate_by_dyad.keys())

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 3 windowed-MVAR gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .dyad-panel { display: none; }
  .dyad-panel.active { display: block; }
  .film-block { border-top: 1px solid #ccc; padding-top: 1em; margin-top: 1em; }
  .header-line { font-family: monospace; margin-bottom: 0.5em; }
  .badge { padding: 0.1em 0.5em; border-radius: 3px; color: white; font-size: 0.85em; }
  .badge-ok { background: #2e8b2e; }
  .badge-bad { background: #b03030; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 500px; border: 1px solid #ccc; }
  .row img.mvar-grid { max-width: 700px; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary, #sensitivity { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
</style>
</head>
<body>
<h1>Stage 3 windowed-MVAR gate</h1>
<p>analysis fit: <b>__FIT_METHOD__</b>, order <b>p=__MODEL_ORDER__</b> (fixed; the criterion order is descriptive only,
   primary_crit=__PRIMARY_CRIT__), window=<b>__WIN_LEN_S__ s / __OVERLAP_PCT__% overlap</b>, detrend=<b>__DETREND_TYPE__</b>,
   min_white_fraction=__MIN_WHITE_FRACTION__. Grid figures: estimator=<b>__ESTIMATOR__</b>,
   box_cox_lambda=<b>__BOX_COX_LAMBDA__</b> (-1 = no transform).</p>
<div id="summary">__SUMMARY__</div>
<h2>Window-choice sensitivity</h2>
<div id="sensitivity">__SENSITIVITY_TEXT__</div>
<div class="row">__SENSITIVITY_IMAGES__</div>
<h2>Per-case diagnostics</h2>
<label for="dyad-select">Dyad: </label>
<select id="dyad-select"></select>
<div id="panels">__PANELS__</div>
<script>
const dyadIds = __DYAD_IDS_JSON__;
const select = document.getElementById('dyad-select');
for (const id of dyadIds) {
  const opt = document.createElement('option');
  opt.value = id; opt.textContent = id;
  select.appendChild(opt);
}
function showDyad(id) {
  document.querySelectorAll('.dyad-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  if (panel) panel.classList.add('active');
}
select.onchange = () => showDyad(select.value);
if (dyadIds.length) showDyad(dyadIds[0]);
</script>
</body>
</html>
"""


panels_html = "\n".join(render_dyad_panel_mvar_order(dyad_id, gate_by_dyad[dyad_id]) for dyad_id in gate_dyad_ids)

summary_lines = [f"{len(manifest_df)} cases total"]
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    summary_lines.append(
        f"{group_label}: n={n}  windowed: stable={int(group_df['stable'].sum())}/{n} "
        f"white={int((group_df['min_white_fraction'] >= MIN_WHITE_FRACTION).sum())}/{n} "
        f"quality_ok={int(group_df['quality_ok'].sum())}/{n}  |  "
        f"global: stable={int((group_df['max_abs_root_global'] < 1.0).sum())}/{n} "
        f"white={int((group_df['min_white_fraction_global'] >= MIN_WHITE_FRACTION).sum())}/{n}"
    )
summary_text = "\n".join(summary_lines)

sensitivity_text_lines = []
for result in sensitivity_results:
    sensitivity_text_lines.append(f"\n{result['dyad_id']} {result['film']} (p_used={result['p_used']}):")
    sensitivity_text_lines.append(result["band_table"].to_string(index=False))
sensitivity_text = "\n".join(sensitivity_text_lines)
sensitivity_images = "\n".join(f'<img src="qc/{r["plot"]}" alt="sensitivity {r["dyad_id"]} {r["film"]}">' for r in sensitivity_results)

html = HTML_TEMPLATE.replace("__PANELS__", panels_html)
html = html.replace("__SUMMARY__", summary_text)
html = html.replace("__SENSITIVITY_TEXT__", sensitivity_text)
html = html.replace("__SENSITIVITY_IMAGES__", sensitivity_images)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
html = html.replace("__PRIMARY_CRIT__", PRIMARY_CRIT).replace("__WIN_LEN_S__", str(WIN_LEN_S))
html = html.replace("__OVERLAP_PCT__", str(int(OVERLAP_FRAC * 100))).replace("__DETREND_TYPE__", DETREND_TYPE)
html = html.replace("__MIN_WHITE_FRACTION__", str(MIN_WHITE_FRACTION))
html = html.replace("__FIT_METHOD__", FIT_METHOD).replace("__MODEL_ORDER__", str(MODEL_ORDER))
(OUTPUT_DIR / "mvar_order_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'mvar_order_gate.html'}")
