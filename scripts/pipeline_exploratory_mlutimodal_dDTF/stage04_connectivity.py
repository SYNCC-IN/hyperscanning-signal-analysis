"""Stage 4 - per-case connectivity cubes (descriptive).

For every Stage 2 design file (`02_envelopes/<dyad_id>_<film>.nc`) and every estimator in `ESTIMATORS` (GPDC,
dDTF), estimates the connectivity cube and the spectral matrix with the analysis fit of the manuscript (2.3):
`src.connectivity.Granger_estimator` -> `src.sim_estimation.fit_mvar` (least squares on stacked 10 s / 50 %
windows, no detrending, fixed order p = `COMMON_MODEL_ORDER`) -> `src.sim_estimation.connectivity_cube`.

These cubes are DESCRIPTIVE (QC figures, per-case edge values). The inference uses Stage 5's pairing tables,
which are computed on the same code path.

Writes, per estimator, `04_connectivity/<estimator>/<dyad_id>_<film>.npz` (cube, spectra, freqs, band average,
metadata), a QC grid figure per case (`qc/<dyad_id>_<film>_grid.png`), `stage04_manifest.csv`, a synthetic
known-truth anchor (`qc/synthetic_anchor.png`), and an interactive gate (`connectivity_gate.html`).
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
from src.design import assemble_design_matrix, assert_edges_known, node_names, window_geometry
from src.io_utils import ensure_dir, parse_case_filename
from src.mtmvar import mvar_plot
from src.pipeline_config import load_stage_config
from src.reporting import render_dyad_panel_ffdtf
from src.surrogate import band_average_cube
from src.synthetic_mvar import edges_to_coupling, generate_var_process, plot_synthetic_anchor, summarize_coupling_strength

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this stage's own section).
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage04_connectivity")

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
ENVELOPES_DIR = ANALYSIS_ROOT / CFG["ENVELOPES_SUBDIR"]
ORDER_DIR = ANALYSIS_ROOT / CFG["ORDER_SUBDIR"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])

FILMS = CFG["FILMS"]
TARGET_SFREQ = CFG["TARGET_SFREQ"]
NODES = CFG["nodes"]
NODE_NAMES = node_names(NODES)
assert_edges_known([(e["source"], e["target"]) for e in CFG["edge_topology"]], NODE_NAMES,
                   context="stage04 pipeline_config.json's edge_topology")

ESTIMATORS = CFG["ESTIMATORS"]
MODEL_ORDER = CFG["COMMON_MODEL_ORDER"]
FIT_METHOD = CFG["FIT_METHOD"]
DETREND_TYPE = CFG["DETREND_TYPE"]
WIN_LEN, STEP = window_geometry(CFG["WIN_LEN_S"], CFG["OVERLAP_FRAC"], TARGET_SFREQ)
FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
COUPLING_BAND_HZ = tuple(CFG["COUPLING_BAND_HZ"])
BOX_COX_LAMBDA = CFG["BOX_COX_LAMBDA"]    # -1 = no transform (src.mtmvar.box_cox_transform)
PRIMARY_EDGES = [(e["source"], e["target"]) for e in CFG["edge_topology"] if e["class"] != "exploratory"]
GRID_SCALE = CFG["GRID_SCALE"]
FFDTF_ROWSUM_TOL = CFG["FFDTF_ROWSUM_TOL"]

# Synthetic known-truth anchor: node 1 -> node 0 at lag 1, mild self-persistence on both nodes.
ANCHOR_EDGES = [tuple(edge) for edge in CFG["ANCHOR_EDGES"]]
ANCHOR_N_NODES = CFG["ANCHOR_N_NODES"]
ANCHOR_CHAN_NAMES = CFG["ANCHOR_CHAN_NAMES"]
ANCHOR_SNR = CFG["ANCHOR_SNR"]
ANCHOR_N_SAMPLES = CFG["ANCHOR_N_SAMPLES"]
ANCHOR_SEED = CFG["ANCHOR_SEED"]
ANCHOR_WIN_LEN, ANCHOR_STEP = window_geometry(CFG["ANCHOR_WIN_LEN_S"], CFG["ANCHOR_OVERLAP_FRAC"], TARGET_SFREQ)
ANCHOR_MODEL_ORDER = CFG["ANCHOR_MODEL_ORDER"]
ANCHOR_DETREND_TYPE = CFG["ANCHOR_DETREND_TYPE"]

# ---------------------------------------------------------------------------
# 1. Per case and estimator: cube + spectra at the analysis fit
# ---------------------------------------------------------------------------
nc_paths = sorted(ENVELOPES_DIR.glob("*.nc"))
print(f"Stage 4: {len(nc_paths)} dyad x film design files, estimators {ESTIMATORS}, p={MODEL_ORDER}, "
      f"{FIT_METHOD}, detrend={DETREND_TYPE}")

manifest_rows, gate_entries = [], {est: [] for est in ESTIMATORS}
for est in ESTIMATORS:
    ensure_dir(OUTPUT_DIR / est / "qc")

for nc_path in nc_paths:
    dyad_id, film = parse_case_filename(nc_path, FILMS)
    envelopes = xr.load_dataarray(nc_path)
    assert envelopes.attrs["fs"] == TARGET_SFREQ, f"fs mismatch for {dyad_id} {film}"
    design = assemble_design_matrix(envelopes, NODE_NAMES, zscore=True)
    order_record = json.loads((ORDER_DIR / f"{dyad_id}_{film}_order.json").read_text(encoding="utf-8"))
    assert (order_record["fit_method"], order_record["detrend_type"], order_record["model_order"],
            order_record["win_len"], order_record["step"]) == \
        (FIT_METHOD, DETREND_TYPE, MODEL_ORDER, WIN_LEN, STEP), f"Stage 3 was run with other fit settings for {dyad_id} {film}: rerun it"

    for est in ESTIMATORS:
        cube, spectra = Granger_estimator(design, FREQS, TARGET_SFREQ, MODEL_ORDER, WIN_LEN, STEP, DETREND_TYPE,
                                          est, BOX_COX_LAMBDA, FIT_METHOD)
        if BOX_COX_LAMBDA == -1:      # bounds hold for the untransformed estimators only
            assert -1e-9 <= cube.min() and cube.max() <= 1 + 1e-9, f"{est} out of [0,1] for {dyad_id} {film}"
        band_avg = band_average_cube(cube, FREQS, COUPLING_BAND_HZ)
        edge_values = {f"{s}->{t}": float(band_avg[NODE_NAMES.index(t), NODE_NAMES.index(s)]) for s, t in PRIMARY_EDGES}
        np.savez(OUTPUT_DIR / est / f"{dyad_id}_{film}.npz", cube=cube, spectra=spectra, freqs=FREQS,
                 band_avg=band_avg, estimator=est, model_order=MODEL_ORDER, fit_method=FIT_METHOD,
                 win_len=WIN_LEN, step=STEP, detrend_type=DETREND_TYPE, fs=TARGET_SFREQ,
                 box_cox_lambda=BOX_COX_LAMBDA, variable_order=np.array(NODE_NAMES),
                 coupling_band=np.array(COUPLING_BAND_HZ), dyad_id=dyad_id, film=film,
                 group=order_record["group"], age_months=order_record["age_months"],
                 stable=order_record["stable"], max_abs_root=order_record["max_abs_root"])

        mvar_plot(spectra, cube, FREQS, x_label="from ", y_label="to ", chan_names=NODE_NAMES,
                  top_title=f"{dyad_id} {film}: {est} λ={BOX_COX_LAMBDA} (p={MODEL_ORDER}, {FIT_METHOD})",
                  scale=GRID_SCALE, fig_size=(9, 9), band_hz=COUPLING_BAND_HZ)
        grid_name = f"{dyad_id}_{film}_grid.png"
        plt.gcf().savefig(OUTPUT_DIR / est / "qc" / grid_name)
        plt.close(plt.gcf())

        row = {"estimator": est, "dyad_id": dyad_id, "film": film, "group": order_record["group"],
               "model_order": MODEL_ORDER, "n_windows": order_record["n_windows"],
               "stable": order_record["stable"], "max_abs_root": order_record["max_abs_root"],
               "value_min": float(cube.min()), "value_max": float(cube.max()), **edge_values,
               "quality_ok": order_record["quality_ok"], "quality_reasons": ";".join(order_record["quality_reasons"])}
        manifest_rows.append(row)
        gate_entries[est].append({**row, "edge_values": edge_values, "grid_image": grid_name})

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage04_manifest.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Run summary
# ---------------------------------------------------------------------------
edge_columns = [f"{s}->{t}" for s, t in PRIMARY_EDGES]
summary_lines = [f"{manifest_df[['dyad_id', 'film']].drop_duplicates().shape[0]} cases, p={MODEL_ORDER}, "
                 f"{FIT_METHOD}, detrend={DETREND_TYPE}, Box-Cox lambda={BOX_COX_LAMBDA}"]
for (est, group), group_df in manifest_df.groupby(["estimator", "group"]):
    n = len(group_df)
    edge_means = "  ".join(f"{col}={group_df[col].mean():.3f}" for col in edge_columns)
    summary_lines.append(f"{est} {group}: n={n}  stable={int(group_df['stable'].sum())}/{n}  "
                         f"quality_ok={int(group_df['quality_ok'].sum())}/{n}\n    mean band-avg: {edge_means}")
print("\n".join(summary_lines))

# ---------------------------------------------------------------------------
# 3. Synthetic known-truth anchor (same code path)
# ---------------------------------------------------------------------------
anchor_coupling = edges_to_coupling(ANCHOR_EDGES, ANCHOR_N_NODES)
anchor_design = generate_var_process(anchor_coupling, ANCHOR_SNR, ANCHOR_N_SAMPLES, seed=ANCHOR_SEED)
known_strength = summarize_coupling_strength(anchor_coupling)
anchor_lines = [f"edges: {ANCHOR_EDGES}", f"known coupling strength (|gain|):\n{known_strength}"]
for est in ESTIMATORS:
    anchor_cube, anchor_spectra = Granger_estimator(anchor_design, FREQS, TARGET_SFREQ, ANCHOR_MODEL_ORDER, ANCHOR_WIN_LEN,
                                                    ANCHOR_STEP, ANCHOR_DETREND_TYPE, est, BOX_COX_LAMBDA, FIT_METHOD)
    recovered = anchor_cube.mean(axis=2)  # full-band average: no expected coupling band for this synthetic test
    fig = plot_synthetic_anchor(known_strength, recovered, ANCHOR_CHAN_NAMES,
                                f"Synthetic anchor ({est}): node1 -> node0 (lag 1, gain {ANCHOR_EDGES[0][3]}), p={ANCHOR_MODEL_ORDER}")
    fig.savefig(OUTPUT_DIR / est / "qc" / "synthetic_anchor.png")
    plt.close(fig)
    mvar_plot(anchor_spectra, anchor_cube, FREQS, x_label="from ", y_label="to ", chan_names=ANCHOR_CHAN_NAMES,
              top_title=f"Synthetic anchor {est} grid (p={ANCHOR_MODEL_ORDER})", scale=GRID_SCALE, fig_size=(6, 6),
              band_hz=COUPLING_BAND_HZ)
    plt.gcf().savefig(OUTPUT_DIR / est / "qc" / "synthetic_anchor_grid.png")
    plt.close(plt.gcf())
    recovered_ok = bool(recovered[0, 1] > recovered[1, 0])
    anchor_lines.append(f"{est}: recovered (freq-avg)\n{recovered}\n    true edge (node1->node0) > reverse: {recovered_ok}")
    print(f"Synthetic anchor {est}: node0<-node1={recovered[0, 1]:.3f} vs node1<-node0={recovered[1, 0]:.3f} "
          f"(true edge recovered: {recovered_ok})")

# ---------------------------------------------------------------------------
# 4. Interactive HTML gate (one dyad selector; one panel set per estimator)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 4 connectivity gate</title>
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
  #summary, #anchor-summary { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
</style>
</head>
<body>
<h1>Stage 4 connectivity gate (descriptive)</h1>
<p>Estimators <b>__ESTIMATORS__</b>; fit <b>__FIT_METHOD__</b>, p=<b>__MODEL_ORDER__</b>, detrend=<b>__DETREND_TYPE__</b>,
   window __WIN_LEN__/__STEP__ samples; freq grid __N_FREQS__ pts; coupling band <b>__COUPLING_BAND__ Hz</b>;
   Box-Cox lambda=<b>__BOX_COX_LAMBDA__</b>. Same code path as the simulations (`src.sim_estimation`).
   These cubes are descriptive; inference uses the Stage 5 pairing tables.</p>
<h2>Synthetic known-truth anchor</h2>
<div id="anchor-summary">__ANCHOR_SUMMARY__</div>
<div class="row">__ANCHOR_IMAGES__</div>
<h2>Run summary</h2>
<div id="summary">__SUMMARY__</div>
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
  document.querySelectorAll('[id^="panel-' + id + '-"]').forEach(p => p.classList.add('active'));
}
select.onchange = () => showDyad(select.value);
if (dyadIds.length) showDyad(dyadIds[0]);
</script>
</body>
</html>
"""

dyad_ids = sorted(manifest_df["dyad_id"].unique())
panels = []
for est in ESTIMATORS:
    by_dyad = {}
    for entry in gate_entries[est]:
        by_dyad.setdefault(entry["dyad_id"], []).append(entry)
    panels += [render_dyad_panel_ffdtf(dyad_id, by_dyad[dyad_id], est) for dyad_id in dyad_ids]
anchor_images = "\n".join(f'<img src="{est}/qc/synthetic_anchor.png" alt="anchor {est}">'
                          f'<img src="{est}/qc/synthetic_anchor_grid.png" alt="anchor grid {est}">' for est in ESTIMATORS)
replacements = {
    "__ESTIMATORS__": " / ".join(ESTIMATORS), "__FIT_METHOD__": FIT_METHOD, "__MODEL_ORDER__": str(MODEL_ORDER),
    "__DETREND_TYPE__": DETREND_TYPE, "__WIN_LEN__": str(WIN_LEN), "__STEP__": str(STEP), "__N_FREQS__": str(len(FREQS)),
    "__COUPLING_BAND__": f"{COUPLING_BAND_HZ[0]}-{COUPLING_BAND_HZ[1]}", "__BOX_COX_LAMBDA__": str(BOX_COX_LAMBDA),
    "__ANCHOR_SUMMARY__": "\n".join(anchor_lines), "__ANCHOR_IMAGES__": anchor_images,
    "__SUMMARY__": "\n".join(summary_lines), "__PANELS__": "\n".join(panels), "__DYAD_IDS_JSON__": json.dumps(dyad_ids),
}
html = HTML_TEMPLATE
for key, value in replacements.items():
    html = html.replace(key, value)
(OUTPUT_DIR / "connectivity_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote {len(manifest_df)} cube files + manifest + gate to {OUTPUT_DIR}")
