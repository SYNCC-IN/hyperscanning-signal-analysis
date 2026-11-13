"""Check A4 - why does the omitted-driver heart-heart edge remain with Z in the model, and grow with recording length?

S05 'omitted_driver': a dyad-level process Z (node 4) drives cg:HRV (node 3) at lag 1 and child:HRV (node 2) at lag 2;
the hearts are not linked. The 5-node model (Z included) is fitted; in pseudo-dyads Z goes with the caregiver role.
Readout: study-level detection (pairing permutation test, pooled null) of the absent edge cg:HRV -> child:HRV.

Variants (one factor changed at a time from the default C0: p = 4, 10 s windows, 50% overlap, linear detrending per
window, least squares); all variants of one cell use the same seeds:
  C0 default; C1 constant detrending; C2 no detrending; C3a one window = the whole segment, linear detrending;
  C3b one window, no detrending (the global fit); C4 p = 8 and p = 12 (300 s only); C5 Z drives both hearts at lag 1
  (data change); C6 Z pole radius 0.5 instead of 0.80-0.92 (data change).
Per study and variant also: mean edge value in real dyads, in pseudo-dyads, and in independent dyads (no heart-heart
link, an independent Z for each person, the caregiver's Z in the model; separate random stream); and the median over
real dyads of the conditional Granger F of cg:HRV -> child:HRV from the 5-node least-squares fit
(`src.sim_estimation.conditional_gc`), at 60, 120, 300 and 600 s (the 120 and 600 s cells have no surrogate test).
Oracle: dDTF / GPDC and the population conditional GC of that edge from the true 5-node coefficients.

Usage: python CHK_A4_omitted_driver.py run|oracle|summarize [dDTF|GPDC]
Results: simulations_multimodal_MVAR/checks_2026-11/A4/
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.sim_estimation import conditional_gc, connectivity_cube, true_conditional_gc
from src.sim_generators import build_coupling, sample_node_params, simulate_dyads, zscore_designs
from src.sim_harness import run_parallel, spawn_seeds
from src import sim_progress
from src.sim_inference import real_matrices, score_study, surrogate_values, wilson_interval
from src.surrogate import band_average_cube

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PART = sys.argv[1] if len(sys.argv) > 1 else "summarize"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
CHUNK = 100
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "A4"
BASE_SEED = 20261114

N_DYADS = 25
N_STUDIES = 100
GAINS = [0.4, 0.6]
TEST_SEGMENTS = [60, 300]                 # segments with the surrogate test
GC_SEGMENTS = [60, 120, 300, 600]         # segments of the Granger check
FS = 2.5
WIN_LEN_S, OVERLAP = 10.0, 0.5
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45], [0.25, 0.45], [0.25, 0.45]])
Z_POLE_RADIUS_C6 = 0.5
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3, 4], [0, 2]        # node 4 = Z (caregiver role in pseudo-dyads)
CG_HEART, CHILD_HEART, Z_NODE, Z_CHILD = 3, 2, 4, 5   # node 5: the child's own Z (independent dyads only)
EDGES = {"cg:HRV->child:HRV": (CG_HEART, CHILD_HEART), "child:HRV->cg:HRV": (CHILD_HEART, CG_HEART)}
TARGET_EDGE = "cg:HRV->child:HRV"
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05
N_PERM = 1000
N_ORACLE = 1000                           # parameter draws of the oracle

DEFAULT = {"p": 4, "window": "10 s", "detrend": "linear", "z_lags": (1, 2), "z_radius": None}
VARIANTS = {   # label -> changes from DEFAULT; "segments" limits the cells
    "C0 default": {},
    "C1 constant detrending": {"detrend": "constant"},
    "C2 no detrending": {"detrend": "none"},
    "C3a one window, linear": {"window": "segment"},
    "C3b one window, no detrending": {"window": "global"},
    "C4 p = 8": {"p": 8, "segments": [300]},
    "C4 p = 12": {"p": 12, "segments": [300]},
    "C5 Z -> both hearts at lag 1": {"z_lags": (1, 1)},
    "C6 Z pole radius 0.5": {"z_radius": Z_POLE_RADIUS_C6},
}
PUB = {"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5,
       "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
       "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
       "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3.5,
       "legend.frameon": False, "axes.titlepad": 3}            # as in figures_manuscript.py
FPR_BAND = (0.03, 0.07)


def variant(label):
    """Settings of a variant (DEFAULT with its changes)."""
    return dict(DEFAULT, **VARIANTS[label])


def estimation(settings, seg_s, estimator):
    """Keyword arguments of `edge_matrix` for a variant and segment length."""
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    if settings["window"] == "segment":
        win_len = step = int(seg_s * FS)
    elif settings["window"] == "global":
        win_len = step = None
    return dict(p=settings["p"], win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=estimator, band_hz=BAND_HZ,
                box_cox_lambda=BOX_COX, detrend_type=settings["detrend"])


def driver_edges(settings, gain, child_driver):
    """Planted edges: Z -> cg heart and `child_driver` -> child heart, at the variant's lags."""
    lag_cg, lag_child = settings["z_lags"]
    return [(Z_NODE, CG_HEART, lag_cg, gain), (child_driver, CHILD_HEART, lag_child, gain)]


def ranges(settings, k):
    """Pole-radius and f0 ranges of the first k nodes, with the variant's Z pole radius."""
    radii = POLE_RADIUS_RANGES[:k].copy()
    if settings["z_radius"] is not None:
        radii[Z_NODE:] = settings["z_radius"]
    return radii, F0_RANGES[:k]


def simulate(settings, gain, seg_s, seed):
    """Real dyads (5 nodes, Z drives both hearts) and independent dyads (6 nodes, one Z per person), z-scored."""
    n_samples = int(seg_s * FS)
    radii, f0s = ranges(settings, 5)
    real = simulate_dyads(np.random.default_rng(seed), N_DYADS, n_samples, FS, radii, f0s,
                          driver_edges(settings, gain, Z_NODE), N_LAGS)
    radii, f0s = ranges(settings, 6)
    independent = simulate_dyads(np.random.default_rng([seed, 3]), N_DYADS, n_samples, FS, radii, f0s,
                                 driver_edges(settings, gain, Z_CHILD), N_LAGS)
    return zscore_designs(real), zscore_designs(independent)


def analyse(designs, independent, settings, seg_s, seed, with_test):
    """Readouts of one variant on one study: rate inputs, real / pseudo / independent values, Granger F."""
    est = estimation(settings, seg_s, ESTIMATOR)
    real, roots, _ = real_matrices(designs, NODES, est)
    indep, _, _ = real_matrices(independent, NODES, est)
    gc = [conditional_gc(d[NODES], CG_HEART, CHILD_HEART, settings["p"], est["win_len"], est["step"], settings["detrend"])
          for d in designs]
    out = {"real_value": float(real[:, CHILD_HEART, CG_HEART].mean()), "independent_value": float(indep[:, CHILD_HEART, CG_HEART].mean()),
           "gc_median": float(np.median(gc)), "gc_mean": float(np.mean(gc)), "max_root_real": float(roots.max())}
    if not with_test:
        return out
    values, _, n_unstable = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, None)
    stable = roots < STABILITY_MAX_ROOT
    real, values = real[stable], values[stable][:, stable]
    scored = score_study(real, values, real, "pooled", None, EDGES, DYAD_ALPHA, ALPHA, np.random.default_rng([seed, 7]), N_PERM)
    out.update({"pseudo_value": float(np.nanmean(values[:, :, CHILD_HEART, CG_HEART])), "n_unstable_pairs": n_unstable,
                "study_p": scored[TARGET_EDGE]["study_p"], "study_reject": scored[TARGET_EDGE]["study_reject"],
                "study_p_reverse": scored["child:HRV->cg:HRV"]["study_p"],
                "study_reject_reverse": scored["child:HRV->cg:HRV"]["study_reject"],
                "median_z": scored[TARGET_EDGE]["median_z"], "dyad_rate": scored[TARGET_EDGE]["dyad_rate"]})
    return out


def run_task(task):
    """One study (gain, segment, seed): every variant that applies to the segment; data simulated once per data setting."""
    gain, seg_s, seed = task["gain"], task["seg_s"], task["seed"]
    with_test = seg_s in TEST_SEGMENTS
    rows, cache = [], {}
    for label in VARIANTS:
        settings = variant(label)
        if seg_s not in settings.get("segments", GC_SEGMENTS):
            continue
        data_key = (settings["z_lags"], settings["z_radius"])
        if data_key not in cache:
            cache[data_key] = simulate(settings, gain, seg_s, seed)
        rows.append(dict(analyse(*cache[data_key], settings, seg_s, seed, with_test), estimator=ESTIMATOR, variant=label,
                         gain=gain, seg_s=seg_s, rep=task["rep"], seed=seed))
    return rows


def oracle():
    """dDTF / GPDC (raw, band-averaged) and population conditional GC of cg:HRV -> child:HRV from the true coefficients."""
    rng = np.random.default_rng([BASE_SEED, 99])
    rows = []
    for label in ("C0 default", "C5 Z -> both hearts at lag 1", "C6 Z pole radius 0.5"):
        settings = variant(label)
        radii_ranges, f0_ranges = ranges(settings, 5)
        radii, f0s = sample_node_params(rng, N_ORACLE, radii_ranges), sample_node_params(rng, N_ORACLE, f0_ranges)
        for gain in GAINS:
            values = {"dDTF": [], "GPDC": [], "GC": []}
            for d in range(N_ORACLE):
                coupling = build_coupling(radii[d], f0s[d], FS, driver_edges(settings, gain, Z_NODE), N_LAGS)
                for est in ("dDTF", "GPDC"):
                    cube = connectivity_cube(coupling, np.eye(5), FREQS, FS, est)
                    values[est].append(band_average_cube(cube, FREQS, BAND_HZ)[CHILD_HEART, CG_HEART])
                if d < 100:
                    values["GC"].append(true_conditional_gc(coupling, CG_HEART, CHILD_HEART)[0])
            rows.append({"data": label, "gain": gain, "n_draws": N_ORACLE,
                         "max_abs_dDTF": float(np.max(np.abs(values["dDTF"]))), "max_abs_GPDC": float(np.max(np.abs(values["GPDC"]))),
                         "max_abs_population_GC": float(np.max(np.abs(values["GC"]))), "n_draws_GC": len(values["GC"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary_oracle.csv", index=False)
    print(frame.to_string())


def run():
    """All studies of one estimator, saved in chunks (chunks on disk are skipped)."""
    cells = [(g, s) for g in GAINS for s in GC_SEGMENTS]
    seeds = spawn_seeds(BASE_SEED, len(cells) * N_STUDIES)
    tasks = [{"gain": g, "seg_s": s, "rep": r, "seed": seeds[i * N_STUDIES + r]}
             for i, (g, s) in enumerate(cells) for r in range(N_STUDIES)]
    folder = OUT / "raw" / ESTIMATOR
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tasks), CHUNK):
        path = folder / f"chunk_{start:05d}.csv"
        if not path.exists():
            pd.DataFrame([r for rows in run_parallel(run_task, tasks[start:start + CHUNK], N_JOBS) for r in rows]).to_csv(path, index=False)
    frame = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("chunk_*.csv"))], ignore_index=True)
    frame.to_csv(OUT / f"raw_{ESTIMATOR}.csv", index=False)
    return frame


def summarize():
    """Rate table (variant x segment x gain, both estimators), values / Granger table and the figures."""
    df = pd.concat([pd.read_csv(OUT / f"raw_{est}.csv") for est in ("dDTF", "GPDC")], ignore_index=True)
    tested = df[df.seg_s.isin(TEST_SEGMENTS)]
    rows = []
    for (est, label, seg, gain), sub in tested.groupby(["estimator", "variant", "seg_s", "gain"], sort=False):
        k, n = int(sub.study_reject.sum()), len(sub)
        low, high = wilson_interval(k, n)
        kr = int(sub.study_reject_reverse.sum())
        rows.append({"estimator": est, "variant": label, "seg_s": seg, "gain": gain, "rate": k / n, "ci_low": low, "ci_high": high,
                     "rate_reverse": kr / n, "n": n, "median_z": sub.median_z.median(),
                     "real_value": sub.real_value.mean(), "pseudo_value": sub.pseudo_value.mean(),
                     "independent_value": sub.independent_value.mean(), "gc_median": sub.gc_median.median()})
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "summary_rates.csv", index=False)
    gc = df[df.estimator == "dDTF"].groupby(["variant", "gain", "seg_s"], sort=False).agg(
        gc_median=("gc_median", "median"), gc_q25=("gc_median", lambda x: x.quantile(0.25)),
        gc_q75=("gc_median", lambda x: x.quantile(0.75)), n_studies=("rep", "size")).reset_index()
    gc.to_csv(OUT / "summary_granger.csv", index=False)
    plot(rates, gc)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(rates.round(4).to_string()); print(gc.round(5).to_string())


def plot(rates, gc):
    """fig_A4_rates: rate vs segment length per variant (rows gain, columns estimator); fig_A4_granger: median F vs length."""
    labels = list(VARIANTS)
    colours = {label: plt.cm.tab10(i) for i, label in enumerate(labels)}
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(2, 2, figsize=(7.48, 4.6), sharex=True, sharey=True)
        for i, gain in enumerate(GAINS):
            for j, est in enumerate(("dDTF", "GPDC")):
                ax = axes[i, j]
                ax.axhspan(*FPR_BAND, color="0.88", zorder=0, lw=0); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6, zorder=0)
                for label in labels:
                    sub = rates[(rates.estimator == est) & (rates.gain == gain) & (rates.variant == label)].sort_values("seg_s")
                    ax.errorbar(sub.seg_s, sub.rate, yerr=[sub.rate - sub.ci_low, sub.ci_high - sub.rate], color=colours[label],
                                marker="o" if len(sub) > 1 else "s", lw=1.4 if label == "C0 default" else 0.9, capsize=1.5,
                                elinewidth=0.6, label=label)
                ax.set_title(f"{est}, gain {gain}", loc="left"); ax.set_xticks(TEST_SEGMENTS)
                if i == 1:
                    ax.set_xlabel("segment length (s)")
                if j == 0:
                    ax.set_ylabel("rate, cg heart -> child heart")
        axes[0, 0].legend(ncol=1, loc="upper left")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fig_A4_rates.{ext}", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.6), sharey=True)
        for ax, gain in zip(axes, GAINS):
            for label in labels:
                sub = gc[(gc.gain == gain) & (gc.variant == label)].sort_values("seg_s")
                ax.plot(sub.seg_s, sub.gc_median, marker="o", color=colours[label], lw=1.4 if label == "C0 default" else 0.9, label=label)
            ax.axhline(0, color="0.3", lw=0.6); ax.set_xscale("log"); ax.set_xticks(GC_SEGMENTS); ax.set_xticklabels(GC_SEGMENTS)
            ax.set_xlabel("segment length (s)"); ax.set_title(f"conditional Granger F, cg heart -> child heart, gain {gain}", loc="left")
        axes[0].set_ylabel("median F over studies (study median over dyads)"); axes[1].legend(fontsize=6)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fig_A4_granger.{ext}", dpi=300)
        plt.close(fig)


def main():
    """Run, oracle or summarize."""
    OUT.mkdir(parents=True, exist_ok=True)
    if PART == "summarize":
        summarize()
    elif PART == "oracle":
        oracle()
    else:
        sim_progress.begin(f"CHK A4 {ESTIMATOR}", OUT, studies=N_STUDIES)
        frame = run()
        c0 = frame[(frame.variant == "C0 default") & frame.seg_s.isin(TEST_SEGMENTS)]
        sim_progress.finish(OUT, {f"C0 {s} s gain {g}": f"{sub.study_reject.mean():.3f}" for (s, g), sub in c0.groupby(["seg_s", "gain"])})


if __name__ == "__main__":
    main()
