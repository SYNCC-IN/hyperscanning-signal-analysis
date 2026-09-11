"""Stage 2 - individual-band envelopes for the interbrain ffDTF + HRV pipeline.

Reads Stage 1's `Interbrain_ffDTF_analysis/01_coverage/coverage.csv` plus the
hand-curated `included_dyads` list from `pipeline_config.json`'s "shared"
section (see that file and Stage 1's module docstring), and for every
included dyad x film, builds the four MVAR design variables -- `child:ROI`,
`cg:ROI`, `child:HRV`, `cg:HRV`:

- EEG variable (`*:ROI`): ROI-reduced fast-band amplitude envelope, using each
  participant's individual band from `band_assignments.csv`.
- HRV variable (`*:HRV`): the raw (interpolated) IBI signal, downsampled only
  -- no individualized band-pass, no Hilbert. This reverses the project note's
  original HF-envelope choice: on inspection of the real signals, the EEG
  rhythm envelopes fluctuate in a band that overlaps the raw IBI (RSA,
  ~0.2-1 Hz), whereas the HF-IBI envelope is a second-order, much slower
  signal that no longer sits in that band. Feeding the raw IBI keeps both
  modalities in a comparable band for the shared low-rate MVAR. The
  consequence (accepted explicitly, see `docs/pipeline_plan.md` Stage 2): the
  EEG side is a second-order quantity (amplitude envelope of a fast rhythm)
  while the HRV side is a first-order oscillation (the IBI itself) --
  internally consistent within each modality, but relevant to interpreting the
  exploratory cross brain-heart edges.

After downsampling, both continuous variables additionally pass through one
*shared* band-pass (`DESIGN_HIGHPASS_HZ`-`DESIGN_LOWPASS_HZ`, 2nd-order
Butterworth, `filtfilt`) -- the same filter for both, so any group delay
matches -- confirmed on inspection of the real PSDs: no interesting HRV
activity above ~0.8 Hz, plus visible VLF drift below ~0.05 Hz.

Both variables are computed on each role's whole continuous `passive_movies`
chunk (EEG: individual-band filter -> Hilbert -> downsample -> shared
band-pass; HRV: downsample -> shared band-pass), *then* segmented to a film
window taken from Stage 1's already-QC'd `film_start_s`/`film_end_s` -- so all
filter/Hilbert edge transients fall in the discarded pre/post margins and
inter-film gaps, not inside the retained window. See
`DTF_analysis_notes/pipeline_plan.md` Stage 2 and `src/design.py` for the
underlying functions.

Writes one file per dyad x film to `Interbrain_ffDTF_analysis/02_envelopes/`:
`<dyad_id>_<film>.nc` (xarray.DataArray, dims (variable, time), physical
amplitude, not z-scored -- z-scoring is a Stage 3 concern), a
`stage02_manifest.csv` (one row per included dyad x film, written or
skipped-with-reason), and a QC gate (`qc/*.png` figures + `envelopes_gate.html`
index). QC plots z-score every variable first (plotting only, never persisted)
so the EEG envelope and raw IBI -- which differ by orders of magnitude in
physical units -- are visually comparable.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemble import assemble_dyad
from src.bands import band_lookup
from src.design import node_names, roi_band_envelope, segment_signal, stack_design
from src.envelopes import (
    average_channels,
    bandpass_filter,
    downsample,
    filter_individual_band,
    hilbert_envelope,
    plot_eeg_hrv_envelopes,
    plot_signal_filtered_envelope,
)
from src.io_utils import ensure_dir, film_window, get_participant_files
from src.pipeline_config import load_stage_config
from src.psd import plot_continuous_overlay, plot_continuous_psd_band, plot_design_variable_psd
from src.reporting import render_dyad_panel_envelopes

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this
# stage's own section); only paths/values computed from PROJECT_ROOT or
# other config values stay here. See src.pipeline_config.load_stage_config.
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage02_envelopes")

DRIVE_ROOT = Path(CFG["DRIVE_ROOT"])
EEG_CLEANED_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "ICA_output" / "EEG_ICA_CLEANED"
IBI_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "IBI"

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
COVERAGE_CSV = ANALYSIS_ROOT / CFG["COVERAGE_SUBDIR"] / "coverage.csv"
# Path to the band assignments CSV file, used to determine frequency bands for the ROI.
# You should first run the exploratory spectral analysis to generate this file.
BAND_ASSIGNMENTS_PATH = PROJECT_ROOT / "Exploratory_spectral_analysis" / "04_band_assignment" / "band_assignments.csv"

OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

# ROI as config -- must match Stage 1's ROI_LABEL/ROI_CHANNELS, since the
# coverage table's roi_ok gate was computed for this ROI.
ROI_LABEL = CFG["ROI_LABEL"]
ROI_CHANNELS = CFG["ROI_CHANNELS"]  # select the right temporo-parietal channel as a proxy to right TPJ ("P7"- removed form ROI_CHANNELS list)
BAND_ROI_LABEL = ROI_LABEL  # row to read from band_assignments.csv

FILMS = CFG["FILMS"]
ROLES = CFG["ROLES"]

# Node topology (single source of truth for MVAR row order -- see
# `src.design.node_names`/`rows_for_signal`). The QC/gate plotting helpers
# below (`plot_continuous_overlay`, `plot_design_variable_psd`,
# `render_dyad_panel_envelopes`) are unchanged and still assume exactly one
# `roi_envelope` node and one `raw_ibi` node per role -- only the design
# variable computation/stacking below is generalized to iterate over `NODES`.
NODES = CFG["nodes"]
NODE_NAMES = node_names(NODES)
NODE_BY_ROLE_SIGNAL = {(node["role"], node["signal"]): node["name"] for node in NODES}
ROLE_SIGNALS_NEEDED = {}
for _node in NODES:
    ROLE_SIGNALS_NEEDED.setdefault(_node["role"], set()).add(_node["signal"])

BAND = CFG["BAND"]
EEG_FILTER_ORDER = CFG["EEG_FILTER_ORDER"]

# Set by the raw IBI (HRV_SIGNAL below), not by the EEG envelopes: the raw IBI
# carries RSA up to the top of the child HF-reference band (~1.04 Hz), so
# Nyquist must clear that -- 2.5 Hz gives Nyquist 1.25 Hz. resample_poly
# anti-aliases both signal types onto this shared rate.
TARGET_SFREQ = CFG["TARGET_SFREQ"]

# Shared post-downsample band-pass, applied identically to both the ROI envelope
# and the raw IBI on the continuous signal (before per-film segmentation), so
# both variables get exactly the same filter (and thus the same time smearing),
# Confirmed on inspection of the real PSDs: no
# interesting HRV activity above ~0.8 Hz, plus visible VLF drift below ~0.05 Hz.
DESIGN_HIGHPASS_HZ = CFG["DESIGN_HIGHPASS_HZ"]
DESIGN_LOWPASS_HZ = CFG["DESIGN_LOWPASS_HZ"]
DESIGN_FILTER_ORDER = CFG["DESIGN_FILTER_ORDER"]

# Multitaper smoothing bandwidth (Hz) for the QC PSD comparison plot -- a plain
# periodogram on a ~60 s / ~150-sample segment is too noisy to read.
DESIGN_PSD_BANDWIDTH_HZ = CFG["DESIGN_PSD_BANDWIDTH_HZ"]

# specparam's reported bandwidth is 2-sided (2*std); band_assignments.csv's
# *_bw is stored as a half-width already inflated to match that 2-sided
# value (see src/bands.py _cluster_stats). filter_individual_band's
# `bandwidth` argument is itself a half-width (cf +/- bandwidth), so passing
# fast_bw/2 makes the filter passband equal specparam's 2-sided bandwidth.
BW_CONVENTION = CFG["BW_CONVENTION"]

# HRV variable = the raw (interpolated) IBI, downsampled only -- no band-pass,
# no Hilbert (reverses the project note's HF-envelope choice, see module
# docstring). 
HRV_SIGNAL = CFG["HRV_SIGNAL"]


# "average_envelopes": filter+Hilbert each ROI channel, then average envelopes (plan default).
# "average_raw": average raw ROI channels first, then filter+Hilbert once (what demo_envelopes.py does).
ROI_REDUCTION = CFG["ROI_REDUCTION"]

# QC plots/PSDs z-score every variable first (plotting only, never persisted to
# the .nc) so the EEG envelope (uV-scale) and raw IBI (hundreds of ms) are
# visually comparable on one axis -- see module docstring.
PLOT_ZSCORE = CFG["PLOT_ZSCORE"]

# ---------------------------------------------------------------------------
# 1. Load Stage 1 outputs
# ---------------------------------------------------------------------------
INCLUDED_DYADS = CFG["included_dyads"]
assert INCLUDED_DYADS, (
    "pipeline_config.json's \"shared\".included_dyads is empty -- run Stage 1, review its QC "
    "gate/suggestion, and populate that list by hand before running Stage 2."
)
coverage_df = pd.read_csv(COVERAGE_CSV)
try:
    band_assignments = pd.read_csv(BAND_ASSIGNMENTS_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Band assignments file not found at {BAND_ASSIGNMENTS_PATH}. Please run the exploratory spectral analysis first.")

participant_files = get_participant_files(EEG_CLEANED_ROOT)
print(f"Stage 2: {len(INCLUDED_DYADS)} included dyads from {CONFIG_PATH.name}'s \"shared\".included_dyads")



# ---------------------------------------------------------------------------
# 2. Per-dyad continuous envelopes, then per-film segmentation + write
# ---------------------------------------------------------------------------
manifest_rows = []
gate_entries = []
included_dyad_meta = []

for dyad_id in INCLUDED_DYADS:
    eeg_files = participant_files[participant_files["dyad_id"] == dyad_id]
    dyad = assemble_dyad(dyad_id, eeg_files, IBI_ROOT, ROI_CHANNELS)
    print(f"Stage 2: {dyad_id} {dyad['group']} {dyad['meta']['age_months']} months" )
    role_continuous = {}
    for role in ROLES:
        needed_signals = ROLE_SIGNALS_NEEDED.get(role, set())
        entry = {"skip_reason": None}

        if "roi_envelope" in needed_signals:
            fast_cf, fast_bw = band_lookup(band_assignments, dyad_id, role, BAND_ROI_LABEL, BAND)
            if fast_cf is None:
                role_continuous[role] = {"skip_reason": f"no fast band at {BAND_ROI_LABEL} for {dyad_id} {role}"}
                continue

            eeg_entry = dyad["eeg"][role]
            roi_env, roi_env_sfreq = roi_band_envelope(
                eeg_entry["data"], eeg_entry["sfreq"], fast_cf, fast_bw / 2, EEG_FILTER_ORDER, TARGET_SFREQ, ROI_REDUCTION,
            )
            # Shared post-downsample band-pass on the continuous signal, before segmentation
            # (see DESIGN_HIGHPASS_HZ/DESIGN_LOWPASS_HZ config): identical filter for both
            # variables so any group delay matches, no interesting HRV content above ~0.8 Hz,
            # and VLF drift below ~0.05 Hz is removed.
            roi_env = bandpass_filter(roi_env, roi_env_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER)

            # Full-rate raw/filtered/envelope trace for the QC gate only (a single
            # average-raw-then-filter trace regardless of ROI_REDUCTION, since the
            # gate's purpose is a visual sanity check, not the production signal).
            raw_avg = average_channels(eeg_entry["data"])
            filtered_avg = filter_individual_band(raw_avg, eeg_entry["sfreq"], fast_cf, fast_bw / 2, EEG_FILTER_ORDER)
            envelope_avg_full = hilbert_envelope(filtered_avg)

            entry.update({
                "fast_cf": fast_cf, "fast_bw": fast_bw,
                "roi_env": roi_env, "roi_env_sfreq": roi_env_sfreq, "roi_t0": float(eeg_entry["time"][0]),
                "eeg_entry": eeg_entry, "raw_avg": raw_avg, "filtered_avg": filtered_avg,
                "envelope_avg_full": envelope_avg_full,
            })

        if "raw_ibi" in needed_signals:
            ibi_entry = dyad["ibi"][role]
            # HRV variable is the raw IBI, downsampled only -- no band-pass, no Hilbert (see module docstring).
            hrv_signal, hrv_signal_sfreq = downsample(ibi_entry["data"], ibi_entry["sfreq"], TARGET_SFREQ)
            hrv_signal = bandpass_filter(
                hrv_signal, hrv_signal_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER,
            )
            entry.update({
                "hrv_signal": hrv_signal, "hrv_signal_sfreq": hrv_signal_sfreq,
                "hrv_t0": float(ibi_entry["time"][0]), "ibi_entry": ibi_entry,
            })

        role_continuous[role] = entry

    dyad_skip_reason = next(
        (role_continuous[role]["skip_reason"] for role in ROLES if role_continuous[role]["skip_reason"]), None
    )

    dyad_qc = {}
    if dyad_skip_reason is None:
        included_dyad_meta.append({"dyad_id": dyad_id, "group": dyad["group"], **dyad["meta"]})
        films_windows = [(film, *film_window(coverage_df, dyad_id, film)) for film in FILMS]

        dyad_qc["psd_band"] = {}
        for role in ROLES:
            rc = role_continuous[role]
            fig = plot_continuous_psd_band(
                rc["raw_avg"], rc["eeg_entry"]["sfreq"], rc["fast_cf"], rc["fast_bw"] / 2,
                f"{dyad_id} {role} {ROI_LABEL} continuous PSD",
            )
            path = QC_DIR / f"{dyad_id}_{role}_continuous_psd_band.png"
            fig.savefig(path)
            plt.close(fig)
            dyad_qc["psd_band"][role] = path.name

        fig = plot_continuous_overlay(role_continuous, films_windows, f"{dyad_id} continuous ROI envelope + raw IBI + film windows")
        path = QC_DIR / f"{dyad_id}_continuous_overlay.png"
        fig.savefig(path)
        plt.close(fig)
        dyad_qc["overlay"] = path.name

    for film in FILMS:
        if dyad_skip_reason is not None:
            manifest_rows.append({
                "dyad_id": dyad_id, "film": film, "status": "skipped", "reason": dyad_skip_reason,
            })
            continue

        film_start_s, film_end_s = film_window(coverage_df, dyad_id, film)

        # Segment each node's continuous signal to this film window, keyed by
        # node name -- generalizes to any `NODES` topology, not just the
        # fixed 4-node/2-role/2-signal default.
        signal_keys = {"roi_envelope": ("roi_env", "roi_env_sfreq", "roi_t0"),
                       "raw_ibi": ("hrv_signal", "hrv_signal_sfreq", "hrv_t0")}
        node_segments = {}
        for node in NODES:
            rc = role_continuous[node["role"]]
            signal_key, sfreq_key, t0_key = signal_keys[node["signal"]]
            seg, _ = segment_signal(rc[signal_key], rc[sfreq_key], rc[t0_key], film_start_s, film_end_s)
            node_segments[node["name"]] = seg

        common_len = min(seg.size for seg in node_segments.values())
        fs = role_continuous[NODES[0]["role"]][signal_keys[NODES[0]["signal"]][1]]

        attrs = {
            "fs": fs,
            "roi_label": ROI_LABEL,
            "roi_channels": "|".join(ROI_CHANNELS),
            "child_roi_cf": role_continuous["child"]["fast_cf"],
            "child_roi_bw_half": role_continuous["child"]["fast_bw"] / 2,
            "cg_roi_cf": role_continuous["caregiver"]["fast_cf"],
            "cg_roi_bw_half": role_continuous["caregiver"]["fast_bw"] / 2,
            "eeg_filter_order": EEG_FILTER_ORDER,
            "hrv_signal": HRV_SIGNAL,
            "film": film,
            "dyad_id": dyad_id,
            "group": dyad["group"] or "",
            "age_months": dyad["meta"]["age_months"] or np.nan,
            "target_sfreq": TARGET_SFREQ,
            "zscored": 0,
            "roi_reduction": ROI_REDUCTION,
            "bw_convention": BW_CONVENTION,
            "design_highpass_hz": DESIGN_HIGHPASS_HZ,
            "design_lowpass_hz": DESIGN_LOWPASS_HZ,
            "design_filter_order": DESIGN_FILTER_ORDER,
        }

        node_segments_trimmed = {name: seg[:common_len] for name, seg in node_segments.items()}

        design = stack_design([node_segments_trimmed[name] for name in NODE_NAMES], NODE_NAMES, fs, attrs)
        out_path = OUTPUT_DIR / f"{dyad_id}_{film}.nc"

        # Role-keyed view for the QC/gate plotting helpers below
        # (`plot_design_variable_psd`, `plot_eeg_hrv_envelopes`), which are
        # unmodified and still assume one "roi"/"hrv" pair per role.
        segments_trimmed = {
            role: {
                "roi": node_segments_trimmed[NODE_BY_ROLE_SIGNAL[(role, "roi_envelope")]],
                "hrv": node_segments_trimmed[NODE_BY_ROLE_SIGNAL[(role, "raw_ibi")]],
            }
            for role in ROLES
        }
        design.to_netcdf(out_path)

        manifest_rows.append({
            "dyad_id": dyad_id, "film": film, "status": "written", "reason": "",
            "fs": fs, "n_samples": common_len,
            "child_fast_cf": role_continuous["child"]["fast_cf"], "child_fast_bw": role_continuous["child"]["fast_bw"],
            "cg_fast_cf": role_continuous["caregiver"]["fast_cf"], "cg_fast_bw": role_continuous["caregiver"]["fast_bw"],
        })

        # --- Film-level QC ---
        film_qc = {"psd_band": dyad_qc["psd_band"], "overlay": dyad_qc["overlay"], "filter_envelope": {}, "eeg_hrv": {}}
        for role in ROLES:
            rc = role_continuous[role]
            eeg_time = rc["eeg_entry"]["time"]
            mask = (eeg_time >= film_start_s) & (eeg_time <= film_end_s)
            fig = plot_signal_filtered_envelope(
                rc["raw_avg"][mask], rc["filtered_avg"][mask], rc["envelope_avg_full"][mask],
                rc["eeg_entry"]["sfreq"], f"{dyad_id} {role} {film} raw/filtered/envelope (retained window)",
            )
            path = QC_DIR / f"{dyad_id}_{film}_{role}_filter_envelope.png"
            fig.savefig(path)
            plt.close(fig)
            film_qc["filter_envelope"][role] = path.name

            roi_for_plot = zscore(segments_trimmed[role]["roi"]) if PLOT_ZSCORE else segments_trimmed[role]["roi"]
            hrv_for_plot = zscore(segments_trimmed[role]["hrv"]) if PLOT_ZSCORE else segments_trimmed[role]["hrv"]
            fig = plot_eeg_hrv_envelopes(
                roi_for_plot, fs, hrv_for_plot, fs,
                f"{dyad_id} {role} {film}: EEG {BAND} envelope and raw IBI"
                + (" (z-scored)" if PLOT_ZSCORE else ""),
            )
            path = QC_DIR / f"{dyad_id}_{film}_{role}_eeg_hrv.png"
            fig.savefig(path)
            plt.close(fig)
            film_qc["eeg_hrv"][role] = path.name

        fig = plot_design_variable_psd(
            segments_trimmed, fs, f"{dyad_id} {film} downsampled design variable PSD (aliasing check)",
            PLOT_ZSCORE, DESIGN_PSD_BANDWIDTH_HZ,
        )
        path = QC_DIR / f"{dyad_id}_{film}_design_psd.png"
        fig.savefig(path)
        plt.close(fig)
        film_qc["design_psd"] = path.name

        gate_entries.append({"dyad_id": dyad_id, "film": film, "status": "written", "qc": film_qc})

    if dyad_skip_reason is not None:
        for film in FILMS:
            gate_entries.append({"dyad_id": dyad_id, "film": film, "status": "skipped", "reason": dyad_skip_reason})

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage02_manifest.csv", index=False)

n_written = (manifest_df["status"] == "written").sum()
n_skipped = (manifest_df["status"] == "skipped").sum()
print(f"\n=== Stage 2 summary ===")
print(f"dyad x film cells: {len(manifest_df)} ({n_written} written, {n_skipped} skipped)")
if n_skipped:
    print("\nSkipped cells:")
    for _, r in manifest_df[manifest_df["status"] == "skipped"].drop_duplicates(["dyad_id", "reason"]).iterrows():
        print(f"  {r['dyad_id']} {r['film']:12s} {r['reason']}")
print(f"\nWrote {n_written} design files + manifest to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Interactive HTML gate
# ---------------------------------------------------------------------------
gate_by_dyad = {}
for entry in gate_entries:
    gate_by_dyad.setdefault(entry["dyad_id"], []).append(entry)
gate_dyad_ids = sorted(gate_by_dyad.keys())

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 2 envelopes gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .dyad-panel { display: none; }
  .dyad-panel.active { display: block; }
  .film-block { border-top: 1px solid #ccc; padding-top: 1em; margin-top: 1em; }
  .skipped { color: #a33; font-weight: 600; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 420px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
</style>
</head>
<body>
<h1>Stage 2 envelopes gate</h1>
<p>ROI: <b>__ROI_LABEL__</b> (__ROI_CHANNELS__), band: __BAND__, reduction: __ROI_REDUCTION__.</p>
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


panels_html = "\n".join(render_dyad_panel_envelopes(dyad_id, gate_by_dyad[dyad_id], ROLES) for dyad_id in gate_dyad_ids)

html = HTML_TEMPLATE.replace("__PANELS__", panels_html)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__ROI_LABEL__", ROI_LABEL).replace("__ROI_CHANNELS__", "|".join(ROI_CHANNELS))
html = html.replace("__BAND__", BAND).replace("__ROI_REDUCTION__", ROI_REDUCTION)
(OUTPUT_DIR / "envelopes_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'envelopes_gate.html'}")

# ---------------------------------------------------------------------------
# 4. Sample statistics for dyads included in the written design files
# ---------------------------------------------------------------------------
included_meta_df = pd.DataFrame(included_dyad_meta)

print(f"\n=== Dyads with written design files ({len(included_dyad_meta)}) sample statistics, by group ===")
sample_stats = {}
for group_label, group_meta in included_meta_df.groupby("group"):
    age = group_meta["age_months"].dropna()
    sex_counts = group_meta["sex"].value_counts()
    n_sexed = sex_counts.sum()
    sex_str = ", ".join(
        f"{sex_code}={count} ({100 * count / n_sexed:.1f}%)" for sex_code, count in sex_counts.items()
    )
    print(f"\n{group_label} (n={len(group_meta)}):")
    print(f"  age (months): mean={age.mean():.1f} +/- {age.std():.1f}, "
          f"range=[{age.min():.0f}, {age.max():.0f}] (n={len(age)})")
    print(f"  sex: {sex_str}")

    sample_stats[group_label] = {
        "n": len(group_meta),
        "age_months_mean": float(age.mean()),
        "age_months_std": float(age.std()),
        "age_months_min": float(age.min()),
        "age_months_max": float(age.max()),
        "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
        "sex_proportions": {str(k): float(v / n_sexed) for k, v in sex_counts.items()},
    }

sample_stats_path = OUTPUT_DIR / "stage02_sample_stats.json"
sample_stats_path.write_text(json.dumps(sample_stats, indent=2), encoding="utf-8")
print(f"\nWrote sample statistics to {sample_stats_path}")
