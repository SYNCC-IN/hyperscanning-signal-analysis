"""S10 - Which coupling types are visible with raw IBI (option A) vs RSA amplitude (A', B)? (Table 2, S10)

Two-node model: child EEG envelope (node 0) and caregiver cardiac series (node 1).
Truths: 'linear' - the caregiver's RSA oscillation modulates the child's log EEG power (breathing-paced modulation);
        'amp_to_eeg' - the caregiver's RSA amplitude modulates the child's log EEG power (amplitude-only coupling);
        'none'.
Options (Table 1 of the draft): A = raw IBI + EEG envelope (shared 0.05-1 Hz band), 60 s segments, 10 s windows, band 0.15-0.5 Hz;
        A' = RSA amplitude + EEG envelope (0.01-1 Hz), 300 s segments, 30 s windows, band 0.03-0.15 Hz;
        B  = RSA amplitude + band-matched EEG envelope (0.01-0.15 Hz), same segment/window/band as A'.
Readouts: power (truth present) and FPR (truth absent) of the edge cg:cardiac -> child:EEG, reverse-edge detection.
Prediction to test: A sees 'linear' but not 'amp_to_eeg'; A'/B see 'amp_to_eeg' but not 'linear'.

Usage: python S10_coupling_type_visibility.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_10/
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.sim_cardiac import FS_ANALYSIS, represent, simulate_cardiac_dyad, simulate_eeg_envelope
from src.sim_generators import zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import run_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 10
BASE_SEED = 20260110
SETTINGS = {
    "quick": {"n_rep": 4, "n_dyads": 12, "strengths": [0.6], "seg_long_s": 300},
    "full": {"n_rep": 100, "n_dyads": 25, "strengths": [0.3, 0.6, 1.0], "seg_long_s": 300},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

OPTIONS = {"A": {"seg_s": 60, "win_s": 10.0, "band": (0.15, 0.5)},
           "Aprime": {"seg_s": None, "win_s": 30.0, "band": (0.03, 0.15)},
           "B": {"seg_s": None, "win_s": 30.0, "band": (0.03, 0.15)}}          # seg None = long segment
TRUTHS = ["none", "linear", "amp_to_eeg"]
MODEL_ORDER = 4
OVERLAP = 0.5
BOX_COX = 0.25
FREQS = np.linspace(0.01, FS_ANALYSIS / 2 - 0.01, 100)
EDGES = {"cg:cardiac->child:EEG": (1, 0), "child:EEG->cg:cardiac": (0, 1)}
NODES, CHILD_NODES = [0, 1], [0]
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
FPR_MAX, POWER_MIN = 0.10, 0.80       # [proposed]


def run_task(task):
    """One replicated study for (truth, strength): all three options on the same simulated dyads."""
    rng = np.random.default_rng(task["seed"])
    truth, strength = task["truth"], task["strength"]
    raw = []
    for _ in range(SETTINGS["n_dyads"]):
        f_cg = rng.uniform(0.20, 0.30)
        cardiac, clean_cg, log_amp_cg = simulate_cardiac_dyad(rng, f_cg + 0.15, f_cg, SETTINGS["seg_long_s"], "none", 0.0)
        eeg = simulate_eeg_envelope(rng, cardiac.shape[1], clean_cg, log_amp_cg, truth, strength)
        raw.append((eeg, cardiac[1]))
    out = []
    for option, cfg in OPTIONS.items():
        designs = []
        for eeg, ibi in raw:
            pair = np.stack([represent(eeg, option, "eeg"), represent(ibi, option, "ibi")])
            designs.append(pair if cfg["seg_s"] is None else pair[:, : int(cfg["seg_s"] * FS_ANALYSIS)])
        designs = zscore_designs(np.stack(designs))
        win_len, step = window_geometry(cfg["win_s"], OVERLAP, FS_ANALYSIS)
        est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS_ANALYSIS, estimator=ESTIMATOR, band_hz=cfg["band"], box_cox_lambda=BOX_COX)
        for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
            out.append(dict(r, option=option, truth=truth, strength=strength, rep=task["rep"]))
    return out


def main():
    """Run S10 and write all outputs to Results/<estimator>/S_10."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [("none", 0.0)] + [(t, s) for t in TRUTHS[1:] for s in SETTINGS["strengths"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"truth": t, "strength": s, "rep": i, "seed": seeds[c * SETTINGS["n_rep"] + i]} for c, (t, s) in enumerate(cells) for i in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    fwd, rev = "cg:cardiac->child:EEG", "child:EEG->cg:cardiac"
    rows = []
    for (option, truth, strength), sub in df.groupby(["option", "truth", "strength"]):
        a, b = sub[sub.edge == fwd], sub[sub.edge == rev]
        k = int(a.study_reject.sum()); lo, hi = wilson_interval(k, len(a))
        rows.append({"option": option, "truth": truth, "strength": strength, "forward_rate": k / len(a), "ci_low": lo, "ci_high": hi,
                     "reverse_rate": b.study_reject.mean(), "dyad_rate": a.dyad_rate.mean(), "median_z": a.median_z.median(), "n_rep": len(a)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, truth in zip(axes, TRUTHS[1:]):
        sub = summary[summary.truth == truth]
        for option, s2 in sub.groupby("option"):
            s2 = s2.sort_values("strength")
            ax.errorbar(s2.strength, s2.forward_rate, yerr=[(s2.forward_rate - s2.ci_low).clip(lower=0), (s2.ci_high - s2.forward_rate).clip(lower=0)], marker="o", capsize=2, label=option)
        ax.axhline(ALPHA, color="k", lw=0.6); ax.axhline(POWER_MIN, color="k", lw=0.6, ls=":")
        ax.set_title(f"truth: {truth}"); ax.set_xlabel("coupling strength")
    axes[0].set_ylabel("detection rate of cg:cardiac -> child:EEG"); axes[0].legend()
    fig.tight_layout(); fig.savefig(out_dir / "fig_coupling_type_visibility.png", dpi=150); plt.close(fig)

    top = max(SETTINGS["strengths"])
    def rate(option, truth):
        s = summary[(summary.option == option) & (summary.truth == truth) & ((summary.strength == top) | (truth == "none"))]
        return float(s.forward_rate.to_numpy()[0])
    criteria = {
        f"A detects 'linear' (power >= {POWER_MIN})": rate("A", "linear") >= POWER_MIN,
        f"A does not detect 'amp_to_eeg' (rate <= {FPR_MAX})": rate("A", "amp_to_eeg") <= FPR_MAX,
        f"A' detects 'amp_to_eeg' (power >= {POWER_MIN})": rate("Aprime", "amp_to_eeg") >= POWER_MIN,
        f"A' does not detect 'linear' (rate <= {FPR_MAX})": rate("Aprime", "linear") <= FPR_MAX,
        f"B detects 'amp_to_eeg' (power >= {POWER_MIN})": rate("B", "amp_to_eeg") >= POWER_MIN,
        f"FPR under 'none' <= {FPR_MAX} for every option": all(rate(o, "none") <= FPR_MAX for o in OPTIONS),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": {k: bool(v) for k, v in criteria.items()}}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
