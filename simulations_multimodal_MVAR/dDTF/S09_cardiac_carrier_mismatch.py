"""S09 - Carrier-frequency mismatch on the cardiac side (Table 2, S9).

Level 2. Child and caregiver IBI series are RSA oscillations with person-specific respiratory frequencies and slow drift.
The caregiver's respiratory frequency is drawn from 0.20-0.30 Hz; the child's is the caregiver's plus a gap.
Truths: 'none'; 'pulling' (caregiver -> child frequency pulling, directed phase coupling); 'amplitude' (caregiver's RSA
amplitude drives the child's RSA amplitude). Analysis: option A (raw IBI, shared 0.05-1 Hz band-pass, 2.5 Hz,
60 s segments), 2-node dDTF with the film-matched-type surrogate test (foreign child/caregiver pairs).
Factors: frequency gap, width of the dDTF coupling band.
Readouts: false-positive rate and power of the edge cg:IBI -> child:IBI (study level), reverse-edge detection, and the shift of the
surrogate null (mean null median) with the gap.

Usage: python S09_cardiac_carrier_mismatch.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_09/
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
from src.sim_cardiac import FS_ANALYSIS, represent, simulate_cardiac_dyad
from src.sim_generators import zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import run_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 9
BASE_SEED = 20260109
SETTINGS = {
    "quick": {"n_rep": 4, "n_dyads": 12, "gaps": [0.0, 0.2], "bands": ["mid"], "seg_s": 60},
    "full": {"n_rep": 100, "n_dyads": 25, "gaps": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "bands": ["narrow", "mid", "wide"], "seg_s": 60},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

TRUTHS = {"none": ("none", 0.0), "pulling": ("pulling", 0.15), "amplitude": ("amplitude", 0.6)}
BANDS = {"narrow": (0.2, 0.35), "mid": (0.15, 0.5), "wide": (0.1, 0.8)}      # [TBD: confirm]
CG_F_RANGE = (0.20, 0.30)
F_JITTER_SD = 0.02
MODEL_ORDER = 4
WIN_LEN_S, OVERLAP = 10.0, 0.5
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS_ANALYSIS / 2 - 0.02, 100)
EDGES = {"cg:IBI->child:IBI": (1, 0), "child:IBI->cg:IBI": (0, 1)}
NODES, CHILD_NODES = [0, 1], [0]
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
FPR_MAX, POWER_MIN = 0.10, 0.80       # [proposed]


def run_task(task):
    """One replicated study for (truth, gap): all bands analysed on the same designs."""
    rng = np.random.default_rng(task["seed"])
    link, strength = TRUTHS[task["truth"]]
    designs = []
    for _ in range(SETTINGS["n_dyads"]):
        f_cg = rng.uniform(*CG_F_RANGE)
        f_child = f_cg + task["gap"] + rng.normal(0, F_JITTER_SD)
        cardiac, _, _ = simulate_cardiac_dyad(rng, f_child, f_cg, SETTINGS["seg_s"], link, strength)
        designs.append(np.stack([represent(cardiac[0], "A", "ibi"), represent(cardiac[1], "A", "ibi")]))
    designs = zscore_designs(np.stack(designs))
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS_ANALYSIS)
    out = []
    for band_name in SETTINGS["bands"]:
        est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS_ANALYSIS, estimator=ESTIMATOR, band_hz=BANDS[band_name], box_cox_lambda=BOX_COX)
        for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
            out.append(dict(r, truth=task["truth"], gap=task["gap"], band=band_name, rep=task["rep"]))
    return out


def main():
    """Run S09 and write all outputs to Results/<estimator>/S_09."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [(t, g) for t in TRUTHS for g in SETTINGS["gaps"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"truth": t, "gap": g, "rep": i, "seed": seeds[c * SETTINGS["n_rep"] + i]} for c, (t, g) in enumerate(cells) for i in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    fwd, rev = "cg:IBI->child:IBI", "child:IBI->cg:IBI"
    rows = []
    for (truth, gap, band), sub in df.groupby(["truth", "gap", "band"]):
        a, b = sub[sub.edge == fwd], sub[sub.edge == rev]
        k = int(a.study_reject.sum()); lo, hi = wilson_interval(k, len(a))
        rows.append({"truth": truth, "gap": gap, "band": band, "forward_rate": k / len(a), "ci_low": lo, "ci_high": hi,
                     "reverse_rate": b.study_reject.mean(), "dyad_rate": a.dyad_rate.mean(), "median_z": a.median_z.median(),
                     "null_median_mean": a.null_median_mean.mean(), "real_mean": a.real_mean.mean(), "n_rep": len(a)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(len(SETTINGS["bands"]), 2, figsize=(10, 3.4 * len(SETTINGS["bands"])), squeeze=False)
    for row, band in enumerate(SETTINGS["bands"]):
        for truth, sub in summary[summary.band == band].groupby("truth"):
            sub = sub.sort_values("gap")
            axes[row, 0].plot(sub.gap, sub.forward_rate, marker="o", label=truth)
            axes[row, 1].plot(sub.gap, sub.null_median_mean, marker="o", label=truth)
        axes[row, 0].axhline(ALPHA, color="k", lw=0.6); axes[row, 0].set_ylabel(f"edge detection rate\nband {BANDS[band]} Hz")
        axes[row, 1].set_ylabel(f"mean surrogate-null median (Box-Cox {ESTIMATOR})")
        for col in (0, 1):
            axes[row, col].set_xlabel("carrier-frequency gap (Hz)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_gap_effect.png", dpi=150); plt.close(fig)

    mid = summary[summary.band == ("mid" if "mid" in SETTINGS["bands"] else SETTINGS["bands"][0])]
    none, pull, amp = (mid[mid.truth == t].sort_values("gap") for t in ("none", "pulling", "amplitude"))
    criteria = {
        f"no coupling: FPR <= {FPR_MAX} at every gap (mid band)": bool((none.forward_rate <= FPR_MAX).all()),
        f"pulling at gap 0: power >= {POWER_MIN} (mid band)": bool(pull.forward_rate.iloc[0] >= POWER_MIN),
        "pulling: power at the largest gap is below power at gap 0": bool(pull.forward_rate.iloc[-1] < pull.forward_rate.iloc[0]),
        f"amplitude-only link not seen by raw IBI (prediction): rate <= {FPR_MAX} at every gap": bool((amp.forward_rate <= FPR_MAX).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
