"""Protocol-paper figure: why EEG enters the cross-modal model as band envelopes.

Illustrates the protocol-paper paragraph arguing that cardiac interbeat
intervals fluctuate below ~1 Hz while EEG rhythms oscillate at 3-30 Hz, so
EEG is represented by the amplitude envelope of an individually defined
frequency band -- a slow signal that shares the IBI's frequency range.

For one participant x film (default: W_005 caregiver, Peppa) the figure has
two columns:

Left (time domain, shared time axis, t = 0 at film onset):
    A  raw ROI EEG (blue)
    B  EEG filtered in the individual fast band (blue) with its Hilbert
       envelope overlaid (orange)
    C  z-scored EEG envelope
    D  z-scored interbeat interval (IBI)

Right (frequency domain, multitaper PSDs over the whole film window):
    E  PSD of the raw ROI EEG, individual fast band shaded yellow
    F  PSDs of the z-scored IBI and z-scored EEG envelope overlaid

Processing mirrors Stage 2 (`stage02_envelopes.py`): band-pass + Hilbert are
applied to the whole continuous `passive_movies` chunk and only then cut to
the QC'd film window from Stage 1's `coverage.csv`, so filter/Hilbert edge
transients stay outside the plotted data. Individual band edges come from
`band_assignments.csv` with the same `fast_bw / 2` half-width convention.

Two deliberate departures from the Stage 2 design variables, both set in the
configuration block below:

- The slow signals are resampled to `SLOW_SFREQ` (default 10 Hz) rather than
  the pipeline's 2.5 Hz, so panel F shows the envelope's spectrum well above
  1 Hz instead of truncating it at the 1.25 Hz Nyquist limit.
- The shared 0.05-1 Hz design band-pass is OFF by default
  (`APPLY_DESIGN_BANDPASS`): with it on, panel F would show both signals
  below 1 Hz by construction, which would make the figure circular as an
  argument for the paragraph's claim.

Usage:
    python fig_envelope_rationale.py
    python fig_envelope_rationale.py --dyad W_030 --role child --film Brave
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemble import assemble_dyad
from src.bands import band_lookup
from src.design import roi_band_envelope, segment_signal
from src.envelopes import average_channels, bandpass_filter, downsample, filter_individual_band, hilbert_envelope
from src.io_utils import ensure_dir, film_window, get_participant_files
from src.pipeline_config import load_stage_config
from src.psd import compute_psd_multitaper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "pipeline_exploratory_mlutimodal_dDTF" / "pipeline_config.json"
CFG = load_stage_config(CONFIG_PATH, "stage02_envelopes")

DRIVE_ROOT = Path(CFG["DRIVE_ROOT"])
EEG_CLEANED_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "ICA_output" / "EEG_ICA_CLEANED"
IBI_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "IBI"

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
COVERAGE_CSV = ANALYSIS_ROOT / CFG["COVERAGE_SUBDIR"] / "coverage.csv"
BAND_ASSIGNMENTS_PATH = PROJECT_ROOT / "Exploratory_spectral_analysis" / "04_band_assignment" / "band_assignments.csv"
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "figures" / "protocol_paper")

# Default case (overridable from the command line).
DEFAULT_DYAD = "W_005"
DEFAULT_ROLE = "caregiver"
DEFAULT_FILM = "Peppa"

# Same ROI / band / filter settings as Stage 2.
ROI_LABEL = CFG["ROI_LABEL"]
ROI_CHANNELS = CFG["ROI_CHANNELS"]
BAND = CFG["BAND"]
EEG_FILTER_ORDER = CFG["EEG_FILTER_ORDER"]
ROI_REDUCTION = CFG["ROI_REDUCTION"]

# Sampling rate (Hz) of the slow signals (envelope, IBI) in panels C, D, F.
# 10 Hz -> Nyquist 5 Hz, well above the envelope's band-limited content.
SLOW_SFREQ = 10.0

# Optionally apply Stage 2's shared design band-pass to envelope and IBI.
# Off by default: see module docstring (circularity of panel F).
APPLY_DESIGN_BANDPASS = False
DESIGN_HIGHPASS_HZ = CFG["DESIGN_HIGHPASS_HZ"]
DESIGN_LOWPASS_HZ = CFG["DESIGN_LOWPASS_HZ"]
DESIGN_FILTER_ORDER = CFG["DESIGN_FILTER_ORDER"]

# Displayed time range (s, relative to film onset) for panels A-D; None shows
# the whole film. Display only -- PSDs always use the whole film window.
TIME_WINDOW_S = None

# Multitaper frequency smoothing bandwidths (Hz).
EEG_PSD_BANDWIDTH_HZ = 1.0
SLOW_PSD_BANDWIDTH_HZ = 0.2
EEG_PSD_FMIN = 0.5
EEG_PSD_FMAX = 20.0
SLOW_PSD_FMAX = 2.0
SLOW_RANGE_MARK_HZ = 1.0   # dashed reference line in panel F ("below about 1 Hz")

COLORS = {
    "eeg": "tab:blue",
    "envelope": "tab:orange",
    "ibi": "tab:red",
    "band": "tab:orange",
    "mark": "0.4",
}
FIGSIZE = (10.0, 7.0)
WIDTH_RATIOS = (2.2, 1.0)
LINEWIDTH_FAST = 0.4
LINEWIDTH_SLOW = 1.2
SAVE_FORMATS = ("pdf", "png")
SAVE_DPI = 300

# ---------------------------------------------------------------------------
# 1. Case selection
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dyad", default=DEFAULT_DYAD)
parser.add_argument("--role", default=DEFAULT_ROLE, choices=CFG["ROLES"])
parser.add_argument("--film", default=DEFAULT_FILM, choices=CFG["FILMS"])
args = parser.parse_args()
dyad_id, role, film = args.dyad, args.role, args.film

# ---------------------------------------------------------------------------
# 2. Load data and the individual band
# ---------------------------------------------------------------------------
coverage_df = pd.read_csv(COVERAGE_CSV)
band_assignments = pd.read_csv(BAND_ASSIGNMENTS_PATH)
participant_files = get_participant_files(EEG_CLEANED_ROOT)

dyad = assemble_dyad(dyad_id, participant_files[participant_files["dyad_id"] == dyad_id], IBI_ROOT, ROI_CHANNELS)
eeg_entry = dyad["eeg"][role]
ibi_entry = dyad["ibi"][role]

fast_cf, fast_bw = band_lookup(band_assignments, dyad_id, role, ROI_LABEL, BAND)
if fast_cf is None:
    raise ValueError(f"No {BAND} band at {ROI_LABEL} for {dyad_id} {role} in {BAND_ASSIGNMENTS_PATH}")
half_bw = fast_bw / 2  # same convention as Stage 2 (BW_CONVENTION)
band_lo, band_hi = fast_cf - half_bw, fast_cf + half_bw+1

film_start_s, film_end_s = film_window(coverage_df, dyad_id, film)
eeg_sfreq = eeg_entry["sfreq"]

# ---------------------------------------------------------------------------
# 3. Signals on the whole continuous chunk, then cut to the film window
# ---------------------------------------------------------------------------
# Full-rate traces for panels A, B, E.
raw = average_channels(eeg_entry["data"])
filtered = filter_individual_band(raw, eeg_sfreq, fast_cf, half_bw, EEG_FILTER_ORDER)
envelope_full = hilbert_envelope(filtered)

# Slow signals for panels C, D, F.
env_slow, env_sfreq = roi_band_envelope(
    eeg_entry["data"], eeg_sfreq, fast_cf, half_bw, EEG_FILTER_ORDER, SLOW_SFREQ, ROI_REDUCTION,
)
ibi_slow, ibi_sfreq = downsample(ibi_entry["data"], ibi_entry["sfreq"], SLOW_SFREQ)
if APPLY_DESIGN_BANDPASS:
    env_slow = bandpass_filter(env_slow, env_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER)
    ibi_slow = bandpass_filter(ibi_slow, ibi_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER)

eeg_mask = (eeg_entry["time"] >= film_start_s) & (eeg_entry["time"] <= film_end_s)
t_eeg = eeg_entry["time"][eeg_mask] - film_start_s
raw_seg = raw[eeg_mask]
filtered_seg = filtered[eeg_mask]
envelope_full_seg = envelope_full[eeg_mask]

env_seg, t_env = segment_signal(env_slow, env_sfreq, float(eeg_entry["time"][0]), film_start_s, film_end_s)
ibi_seg, t_ibi = segment_signal(ibi_slow, ibi_sfreq, float(ibi_entry["time"][0]), film_start_s, film_end_s)
n_slow = min(env_seg.size, ibi_seg.size)
env_z = zscore(env_seg[:n_slow])
ibi_z = zscore(ibi_seg[:n_slow])
t_slow = t_env[:n_slow] - film_start_s

# ---------------------------------------------------------------------------
# 4. Multitaper PSDs over the whole film window
# ---------------------------------------------------------------------------
freqs_eeg, psd_eeg = compute_psd_multitaper(
    raw_seg[np.newaxis, :], eeg_sfreq, EEG_PSD_FMIN, EEG_PSD_FMAX, EEG_PSD_BANDWIDTH_HZ,
)
psd_eeg = psd_eeg[0]

freqs_slow, psd_env = compute_psd_multitaper(
    env_z[np.newaxis, :], SLOW_SFREQ, 0.0, SLOW_PSD_FMAX, SLOW_PSD_BANDWIDTH_HZ,
)
psd_env = psd_env[0]
_, psd_ibi = compute_psd_multitaper(
    ibi_z[np.newaxis, :], SLOW_SFREQ, 0.0, SLOW_PSD_FMAX, SLOW_PSD_BANDWIDTH_HZ,
)
psd_ibi = psd_ibi[0]

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
grid = fig.add_gridspec(4, 2, width_ratios=WIDTH_RATIOS)
ax_raw = fig.add_subplot(grid[0, 0])
ax_filt = fig.add_subplot(grid[1, 0], sharex=ax_raw)
ax_env = fig.add_subplot(grid[2, 0], sharex=ax_raw)
ax_ibi = fig.add_subplot(grid[3, 0], sharex=ax_raw)
ax_psd_eeg = fig.add_subplot(grid[0:2, 1])
ax_psd_slow = fig.add_subplot(grid[2:4, 1])

# A: raw EEG
ax_raw.plot(t_eeg, raw_seg, color=COLORS["eeg"], linewidth=LINEWIDTH_FAST)
ax_raw.set_ylabel("EEG (µV)")

# B: fast-band EEG + envelope
ax_filt.plot(t_eeg, filtered_seg, color=COLORS["eeg"], linewidth=LINEWIDTH_FAST,
             label=f"{band_lo:.1f}–{band_hi:.1f} Hz")
ax_filt.plot(t_eeg, envelope_full_seg, color=COLORS["envelope"], linewidth=LINEWIDTH_SLOW, label="envelope")
ax_filt.set_ylabel("Band EEG (µV)")
ax_filt.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

# C: z-scored envelope
ax_env.plot(t_slow, env_z, color=COLORS["envelope"], linewidth=LINEWIDTH_SLOW)
ax_env.set_ylabel("Envelope (z)")

# D: z-scored IBI
ax_ibi.plot(t_slow, ibi_z, color=COLORS["ibi"], linewidth=LINEWIDTH_SLOW)
ax_ibi.set_ylabel("IBI (z)")
ax_ibi.set_xlabel("Time (s)")

for axis in (ax_raw, ax_filt, ax_env):
    axis.tick_params(labelbottom=False)
if TIME_WINDOW_S is not None:
    ax_raw.set_xlim(*TIME_WINDOW_S)
else:
    ax_raw.set_xlim(0, film_end_s - film_start_s)

# E: EEG PSD with individual fast band
ax_psd_eeg.plot(freqs_eeg, psd_eeg, color=COLORS["eeg"], linewidth=LINEWIDTH_SLOW)
ax_psd_eeg.axvspan(band_lo, band_hi, color=COLORS["band"], alpha=0.4, linewidth=0, label="selected\nEEG rhythm")
ax_psd_eeg.set_xlim(0, EEG_PSD_FMAX)
ax_psd_eeg.set_xlabel("Frequency (Hz)")
ax_psd_eeg.set_ylabel("PSD (µV²/Hz)")
ax_psd_eeg.legend(fontsize=7, frameon=False)

# F: slow-signal PSDs
ax_psd_slow.plot(freqs_slow, psd_ibi, color=COLORS["ibi"], linewidth=LINEWIDTH_SLOW, label="IBI (z)")
ax_psd_slow.plot(freqs_slow, psd_env, color=COLORS["envelope"], linewidth=LINEWIDTH_SLOW,
                 label="EEG envelope (z)")
ax_psd_slow.axvline(SLOW_RANGE_MARK_HZ, color=COLORS["mark"], linestyle="--", linewidth=0.8)
ax_psd_slow.set_xlim(0, SLOW_PSD_FMAX)
ax_psd_slow.set_xlabel("Frequency (Hz)")
ax_psd_slow.set_ylabel("PSD (z²/Hz)")
ax_psd_slow.legend(fontsize=7, frameon=False)

for letter, axis in zip("ABCDEF", (ax_raw, ax_filt, ax_env, ax_ibi, ax_psd_eeg, ax_psd_slow)):
    axis.text(-0.02, 1.0, letter, transform=axis.transAxes, ha="right", va="bottom", fontweight="bold")

out_stem = OUTPUT_DIR / f"fig_envelope_rationale_{dyad_id}_{role}_{film}"
for fmt in SAVE_FORMATS:
    fig.savefig(out_stem.with_suffix(f".{fmt}"), dpi=SAVE_DPI)
plt.close(fig)

print(f"{dyad_id} {role} {film}: {ROI_LABEL} {ROI_CHANNELS}, {BAND} band {band_lo:.2f}-{band_hi:.2f} Hz "
      f"(cf {fast_cf:.2f}), film window {film_start_s:.1f}-{film_end_s:.1f} s, "
      f"design band-pass {'on' if APPLY_DESIGN_BANDPASS else 'off'}")
print(f"Wrote {', '.join(str(out_stem.with_suffix(f'.{fmt}')) for fmt in SAVE_FORMATS)}")