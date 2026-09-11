"""Stage 4 - Granger_estimator estimation (windowed-ACF-averaged MVAR core).

For every Stage 2 design file (`02_envelopes/<dyad_id>_<film>.nc`), reads the
matching Stage 3 order-selection record (`03_mvar/<dyad_id>_<film>_order.json`)
and estimates a directed-connectivity cube -- Granger_estimator + multivariate spectra --
at Stage 3's selected model order `p_used` and window geometry
(`win_len`/`step`/`detrend_type`), via `src.connectivity.Granger_estimator`.

Locked decisions (see `DTF_analysis_notes/pipeline_plan.md` Stage 4 -- honour,
do not silently resolve differently):

- The windowed-ACF-averaged MVAR is the estimation core, not a single global
  fit. Window geometry and `p_used` are read verbatim from Stage 3's
  `order.json`, never recomputed from a Stage-4 constant, so the windowed
  stack this stage fits is byte-for-byte identical to the one Stage 3 fit and
  diagnosed.
- The design matrix is z-scored per channel at assembly
  (`src.design.assemble_design_matrix`), reused unchanged.
- Stage-3 quality-failed cases are estimated and flagged (`quality_ok`,
  `quality_reasons` carried forward into the manifest/gate), never silently
  dropped -- whether they enter a later surrogate/group model is a Stage 5/6
  decision, not this stage's.

Open decisions, surfaced here rather than resolved silently -- see the plan
doc for the full discussion:

- Frequency grid resolution: 100 points from 0.02 Hz to just under Nyquist,
  matching Stage 3's QC grid so figures line up. Confirm before Stage 5 reads
  these cubes.
- Stage 4 writes both the full Granger_estimator cube (`Granger_estimator`) and a coupling-band
  average (`band_avg_Granger_estimator`) per case, so Stage 5 can pick its DV substrate
  without Stage 4 pre-committing.

Writes, per dyad x film, `04_Granger_estimator/<dyad_id>_<film>.npz` (Granger_estimator, spectra,
freqs, band_avg_Granger_estimator, and metadata) and a QC grid figure
(`qc/<dyad_id>_<film>_Granger_estimator_grid.png`); across all cases,
`04_Granger_estimator/stage04_manifest.csv`; a synthetic known-truth anchor
(`qc/synthetic_anchor.png`) validating the real `Granger_estimator` code path
against a known directed coupling; and an interactive QC gate
(`Granger_estimator_gate.html`).
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

from src.connectivity import Granger_estimator
from src.design import DESIGN_VARIABLES, assemble_design_matrix, window_geometry
from src.io_utils import ensure_dir, parse_case_filename
from src.mtmvar import mvar_plot
from src.mvar_diag import plot_model_order_histogram
from src.pipeline_config import load_stage_config
from src.reporting import render_dyad_panel_ffdtf
from src.surrogate import band_average_cube
from src.synthetic_mvar import edges_to_coupling, generate_var_process, plot_synthetic_anchor, summarize_coupling_strength

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this
# stage's own section); only paths/values computed from PROJECT_ROOT or
# other config values stay here. See src.pipeline_config.load_stage_config.
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage04_ffdtf")

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
ENVELOPES_DIR = ANALYSIS_ROOT / CFG["ENVELOPES_SUBDIR"]
ORDER_DIR = ANALYSIS_ROOT / CFG["ORDER_SUBDIR"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

FILMS = CFG["FILMS"]
TARGET_SFREQ = CFG["TARGET_SFREQ"]  # must match Stage 2/3's realized design-file rate (asserted against each file's attrs below)
MODEL_ORDER = CFG["MODEL_ORDER"] #"auto"  # or set to a specific integer value if not using automatic selection
# OPEN DECISION (confirm before Stage 5): 100 points, 0.02 Hz -> just under
# Nyquist, matching Stage 3's QC grid so figures line up across stages.
FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)

COUPLING_BAND_HZ = tuple(CFG["COUPLING_BAND_HZ"])
ESTIMATOR = CFG["ESTIMATOR"]  # or "ffDTF" for full-frequency DTF
BOX_COX_LAMBDA = CFG["BOX_COX_LAMBDA"]  # (x**lambda - 1) / lambda applied to the Granger_estimator cube; -1 = no transform (src.mtmvar.box_cox_transform)
PRIMARY_EDGES = [
    (edge["source"], edge["target"]) for edge in CFG["edge_topology"] if edge["class"] != "exploratory"
]
GRID_SCALE = CFG["GRID_SCALE"]
FFDTF_ROWSUM_TOL = CFG["FFDTF_ROWSUM_TOL"]

# Synthetic known-truth anchor (Stage 0 style): node 1 -> node 0 at lag 1,
# plus mild self-persistence on both nodes, run through the real
# `Granger_estimator` code path to validate it against a known answer.
ANCHOR_EDGES = [tuple(edge) for edge in CFG["ANCHOR_EDGES"]]
ANCHOR_N_NODES = CFG["ANCHOR_N_NODES"]
ANCHOR_CHAN_NAMES = CFG["ANCHOR_CHAN_NAMES"]
ANCHOR_SNR = CFG["ANCHOR_SNR"]
ANCHOR_N_SAMPLES = CFG["ANCHOR_N_SAMPLES"]
ANCHOR_SEED = CFG["ANCHOR_SEED"]
ANCHOR_FS = TARGET_SFREQ
ANCHOR_WIN_LEN_S = CFG["ANCHOR_WIN_LEN_S"]
ANCHOR_OVERLAP_FRAC = CFG["ANCHOR_OVERLAP_FRAC"]
ANCHOR_MODEL_ORDER = CFG["ANCHOR_MODEL_ORDER"]
ANCHOR_DETREND_TYPE = CFG["ANCHOR_DETREND_TYPE"]




# ---------------------------------------------------------------------------
# 1. Per-case: read Stage 3's order + window geometry, estimate Granger_estimator/spectra
# ---------------------------------------------------------------------------
nc_paths = sorted(ENVELOPES_DIR.glob("*.nc"))
print(f"Stage 4: {len(nc_paths)} dyad x film design files found in {ENVELOPES_DIR}")

manifest_rows = []
gate_entries = []

for nc_path in nc_paths:
    dyad_id, film = parse_case_filename(nc_path, FILMS)
    envelopes = xr.load_dataarray(nc_path)
    fs = envelopes.attrs["fs"]
    design = assemble_design_matrix(envelopes, zscore=True)

    order_record = json.loads((ORDER_DIR / f"{dyad_id}_{film}_order.json").read_text(encoding="utf-8"))
    if MODEL_ORDER == "auto":
        p_used = order_record["p_used"]
    else:
        p_used = MODEL_ORDER
    win_len = order_record["win_len"]
    step = order_record["step"]
    detrend_type = order_record["detrend_type"]
    quality_ok = order_record["quality_ok"]
    quality_reasons = order_record["quality_reasons"]
    group = order_record["group"]
    age_months = order_record["age_months"]

    granger_estimator, spectra = Granger_estimator(design, FREQS, fs, p_used, win_len, step, detrend_type, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA)

    granger_estimator_min = float(granger_estimator.min())
    granger_estimator_max = float(granger_estimator.max())
    row_sums = granger_estimator.sum(axis=(1, 2))
    max_rowsum_dev = float(np.max(np.abs(row_sums - 1.0)))

    band_avg_Granger_estimator = band_average_cube(granger_estimator, FREQS, COUPLING_BAND_HZ)
    edge_values = {}
    for source_name, target_name in PRIMARY_EDGES:
        source, target = DESIGN_VARIABLES.index(source_name), DESIGN_VARIABLES.index(target_name)
        edge_values[f"{source_name}->{target_name}"] = float(band_avg_Granger_estimator[target, source])

    np.savez(
        OUTPUT_DIR / f"{dyad_id}_{film}.npz",
        granger_estimator=granger_estimator, spectra=spectra, freqs=FREQS,
        var_order=p_used, win_len=win_len, step=step, detrend_type=detrend_type, fs=fs,
        variable_order=np.array(DESIGN_VARIABLES), coupling_band=np.array(COUPLING_BAND_HZ),
        band_avg_Granger_estimator=band_avg_Granger_estimator,
        dyad_id=dyad_id, film=film, group=group, age_months=age_months,
        quality_ok=quality_ok, quality_reasons=np.array(quality_reasons),
    )

    case_title = f"{dyad_id} {film}"
    mvar_plot(spectra, granger_estimator, FREQS, x_label="from ", y_label="to ", chan_names=DESIGN_VARIABLES,
              top_title=f"{case_title}: {ESTIMATOR} λ={BOX_COX_LAMBDA}(p={p_used}, window={win_len}/{step} samp)", scale=GRID_SCALE,
              fig_size=(9, 9), band_hz=COUPLING_BAND_HZ)
    grid_path = QC_DIR / f"{dyad_id}_{film}_Granger_estimator_grid.png"
    plt.gcf().savefig(grid_path)
    plt.close(plt.gcf())

    manifest_rows.append({
        "dyad_id": dyad_id, "film": film, "group": group,
        "p_used": p_used, "win_len": win_len, "step": step, "n_windows": order_record["n_windows"],
        "Granger_estimator_min": granger_estimator_min, "Granger_estimator_max": granger_estimator_max, "max_rowsum_dev": max_rowsum_dev,
        **edge_values,
        "quality_ok": quality_ok, "quality_reasons": ";".join(quality_reasons),
    })

    gate_entries.append({
        "dyad_id": dyad_id, "film": film, "group": group,
        "p_used": p_used, "win_len": win_len, "step": step,
        "Granger_estimator_min": granger_estimator_min, "Granger_estimator_max": granger_estimator_max, "max_rowsum_dev": max_rowsum_dev,
        "edge_values": edge_values, "quality_ok": quality_ok,
        "grid_image": grid_path.name,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage04_manifest.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Run summary
# ---------------------------------------------------------------------------
print(f"\n=== Stage 4 summary ({len(manifest_df)} cases) ===")
edge_columns = [f"{s}->{t}" for s, t in PRIMARY_EDGES]
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    n_in_range = int(((group_df["Granger_estimator_min"] >= 0) & (group_df["Granger_estimator_max"] <= 1)).sum())
    n_normalised = int((group_df["max_rowsum_dev"] < FFDTF_ROWSUM_TOL).sum())
    n_quality_ok = int(group_df["quality_ok"].sum())
    edge_means = " ".join(f"{col}={group_df[col].mean():.3f}" for col in edge_columns)
    print(f"{group_label}: n={n}  in_[0,1]={n_in_range}/{n}  row_normalised={n_normalised}/{n} "
          f"quality_ok={n_quality_ok}/{n}  |  mean band-avg Granger_estimator: {edge_means}")

order_hist_fig = plot_model_order_histogram(manifest_df)
order_hist_path = QC_DIR / "model_order_histogram.png"
order_hist_fig.savefig(order_hist_path)
plt.close(order_hist_fig)

print(f"\nWrote {len(manifest_df)} Granger_estimator/spectra files + manifest to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Synthetic known-truth anchor
# ---------------------------------------------------------------------------
anchor_coupling = edges_to_coupling(ANCHOR_EDGES, ANCHOR_N_NODES)
anchor_design = generate_var_process(anchor_coupling, ANCHOR_SNR, ANCHOR_N_SAMPLES, seed=ANCHOR_SEED)
anchor_win_len, anchor_step = window_geometry(ANCHOR_WIN_LEN_S, ANCHOR_OVERLAP_FRAC, ANCHOR_FS)

anchor_Granger_estimator, anchor_spectra = Granger_estimator(
    anchor_design, FREQS, ANCHOR_FS, ANCHOR_MODEL_ORDER, anchor_win_len, anchor_step, ANCHOR_DETREND_TYPE,
    ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA,
)
known_strength = summarize_coupling_strength(anchor_coupling)
recovered_strength = anchor_Granger_estimator.mean(axis=2)  # full-band average: no expected coupling band for this synthetic test

anchor_comparison_fig = plot_synthetic_anchor(
    known_strength, recovered_strength, ANCHOR_CHAN_NAMES,
    f"Synthetic anchor: node1 -> node0 (lag 1, gain {ANCHOR_EDGES[0][3]}), p={ANCHOR_MODEL_ORDER}",
)
anchor_comparison_path = QC_DIR / "synthetic_anchor.png"
anchor_comparison_fig.savefig(anchor_comparison_path)
plt.close(anchor_comparison_fig)

mvar_plot(anchor_spectra, anchor_Granger_estimator, FREQS, x_label="from ", y_label="to ", chan_names=ANCHOR_CHAN_NAMES,
          top_title=f"Synthetic anchor {ESTIMATOR} grid (p={ANCHOR_MODEL_ORDER})", scale=GRID_SCALE, fig_size=(6, 6),
          band_hz=COUPLING_BAND_HZ)
anchor_grid_path = QC_DIR / "synthetic_anchor_grid.png"
plt.gcf().savefig(anchor_grid_path)
plt.close(plt.gcf())

anchor_recovers_edge = recovered_strength[0, 1] > recovered_strength[1, 0]
print(f"\nSynthetic anchor: recovered {ESTIMATOR}[node0<-node1]={recovered_strength[0, 1]:.3f} "
      f"vs {ESTIMATOR}[node1<-node0]={recovered_strength[1, 0]:.3f}  "
      f"(true edge recovered: {anchor_recovers_edge})")
print(f"Wrote synthetic anchor figures to {anchor_comparison_path} / {anchor_grid_path}")

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
<title>Stage 4 __ESTIMATOR__ gate</title>
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
  .row img { max-width: 700px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary, #anchor-summary { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
</style>
</head>
<body>
<h1>Stage 4 __ESTIMATOR__ gate</h1>
<p>freq grid=<b>__N_FREQS__ pts, __FREQ_MIN__-__FREQ_MAX__ Hz</b>, coupling_band=<b>__COUPLING_BAND__ Hz</b>,
   ffdtf_rowsum_tol=__ROWSUM_TOL__, box_cox_lambda=<b>__BOX_COX_LAMBDA__</b> (-1 = no transform). Estimator: windowed-ACF-averaged MVAR
   (`src.connectivity.Granger_estimator`) at Stage 3's per-case `p_used`/window geometry.</p>
<h2>Synthetic known-truth anchor</h2>
<div id="anchor-summary">__ANCHOR_SUMMARY__</div>
<div class="row">
  <img src="qc/synthetic_anchor.png" alt="synthetic anchor: known vs recovered">
  <img src="qc/synthetic_anchor_grid.png" alt="synthetic anchor Granger_estimator grid">
</div>
<h2>Run summary</h2>
<div id="summary">__SUMMARY__</div>
<div class="row"><img src="qc/model_order_histogram.png" alt="model order histogram"></div>
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


panels_html = "\n".join(render_dyad_panel_ffdtf(dyad_id, gate_by_dyad[dyad_id]) for dyad_id in gate_dyad_ids)

summary_lines = [f"{len(manifest_df)} cases total"]
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    n_in_range = int(((group_df["Granger_estimator_min"] >= 0) & (group_df["Granger_estimator_max"] <= 1)).sum())
    n_normalised = int((group_df["max_rowsum_dev"] < FFDTF_ROWSUM_TOL).sum())
    n_quality_ok = int(group_df["quality_ok"].sum())
    edge_means = "  ".join(f"{col}={group_df[col].mean():.3f}" for col in edge_columns)
    summary_lines.append(
        f"{group_label}: n={n}  in_[0,1]={n_in_range}/{n}  row_normalised={n_normalised}/{n} "
        f"quality_ok={n_quality_ok}/{n}\n    mean band-avg Granger_estimator: {edge_means}"
    )
summary_text = "\n".join(summary_lines)

anchor_summary_text = (
    f"edges: {ANCHOR_EDGES}\n"
    f"known coupling strength (|gain|):\n{known_strength}\n"
    f"recovered Granger_estimator (freq-avg):\n{recovered_strength}\n"
    f"true edge (node1->node0) > reverse (node0->node1): {anchor_recovers_edge}"
)

html = HTML_TEMPLATE.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
html = html.replace("__PANELS__", panels_html)
html = html.replace("__SUMMARY__", summary_text)
html = html.replace("__ANCHOR_SUMMARY__", anchor_summary_text)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__N_FREQS__", str(len(FREQS)))
html = html.replace("__FREQ_MIN__", f"{FREQS.min():.3g}").replace("__FREQ_MAX__", f"{FREQS.max():.3g}")
html = html.replace("__COUPLING_BAND__", f"{COUPLING_BAND_HZ[0]}-{COUPLING_BAND_HZ[1]}")
html = html.replace("__ROWSUM_TOL__", str(FFDTF_ROWSUM_TOL))
(OUTPUT_DIR / "Granger_estimator_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'Granger_estimator_gate.html'}")
