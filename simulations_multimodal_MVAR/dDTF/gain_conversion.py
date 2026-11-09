"""Conversion of the VAR coupling gain to conditional Granger causality, prediction improvement and delta at 60 s.

For every planted edge and gain of S1, S4, S5 and S12, and for a single-edge gain sweep (0.05-0.5) of the three
cross-person edge types (Level 1, same generator and per-person parameter ranges):
  - F: population conditional Granger causality of the edge in the true model (Geweke, 1984), from exact
    autocovariances (`src.sim_estimation.true_conditional_gc`); PI = 1 - exp(-F), the fractional reduction of the
    target's one-step prediction-error variance when the source's past is added. Mean and 5-95% range over
    `N_DRAWS` drawn dyads.
  - delta: for the S4 design-A edge (cg:ROI -> child:ROI, N = 25, 60 s), the pipeline's delta (real value minus the
    median of the pooled pseudo-dyad null; Box-Cox 0.25, mean over 0.15-0.5 Hz) of every dyad, for dDTF and GPDC on the
    same simulated studies: median and 25-75% range over dyads, and the median null value.
S9/S10 (Level 2) are not included: their coupling is not a VAR gain.

Usage: python gain_conversion.py [quick|full]
Results: Results/comparison/gain_conversion.csv, gain_conversion.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.sim_estimation import true_conditional_gc
from src.sim_generators import build_coupling, sample_node_params, simulate_dyads, zscore_designs
from src.sim_harness import run_parallel, spawn_seeds
from src.sim_inference import delta_and_z_all, real_matrices, surrogate_values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
N_JOBS = 4
BASE_SEED = 20260120
SETTINGS = {"quick": {"n_draws": 50, "n_studies": 2}, "full": {"n_draws": 1000, "n_studies": 100}}[MODE]
OUT = Path(__file__).resolve().parent / "Results" / "comparison"

FS = 2.5
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92], [0.80, 0.92]])   # node 4 = Z (S5)
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
GC_ORDER = 60                          # truncation order of the restricted predictor (converged by ~40)

# Delta at 60 s (S4 design A)
N_DYADS, SEG_S = 25, 60
S4_GAINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
STABILITY_MAX_ROOT = 1.0
ESTIMATORS = ["dDTF", "GPDC"]

# Single-edge gain sweep per edge type (as planted in S1/S4), for the supplementary figure
SWEEP_EDGES = {"brain->brain": (1, 0, 1), "heart->heart": (3, 2, 1), "heart->brain": (3, 0, 2)}
SWEEP_GAINS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Nodes: 0 child:ROI, 1 cg:ROI, 2 child:HRV, 3 cg:HRV, 4 Z.
# Each design: (simulation, number of nodes, planted edges (source, target, lag, gain), reported {edge_type: (source, target)})
DESIGNS = (
    [("S1", 4, [(1, 0, 1, 0.4), (3, 2, 1, 0.3), (3, 0, 2, 0.3)],
      {"brain->brain": (1, 0), "heart->heart": (3, 2), "heart->brain": (3, 0)})]
    + [("S4", 4, [(1, 0, 1, g)], {"brain->brain": (1, 0)}) for g in S4_GAINS[1:]]
    + [("S5 cascade", 4, [(1, 0, 1, g), (0, 2, 1, g)],
        {"brain->brain": (1, 0), "brain->heart (within child)": (0, 2)}) for g in (0.2, 0.4, 0.6)]
    + [("S5 common driver", 4, [(1, 3, 1, g), (1, 0, 2, g)],
        {"brain->brain (lag 2)": (1, 0), "brain->heart (within caregiver)": (1, 3)}) for g in (0.2, 0.4, 0.6)]
    + [("S5 omitted driver", 5, [(4, 3, 1, g), (4, 2, 2, g)],
        {"Z->heart (caregiver)": (4, 3), "Z->heart (child, lag 2)": (4, 2)}) for g in (0.2, 0.4, 0.6)]
    + [("S12", 4, [(1, 0, 1, f)] + ([(0, 1, 1, r)] if r > 0 else []),
        {f"brain->brain (partner edge {r})": (1, 0)} | ({f"brain->brain reverse (partner edge {f})": (0, 1)} if r > 0 else {}))
       for f, r in [(0.3, 0.0), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3), (0.2, 0.2), (0.2, 0.1)]]
    + [("S12", 4, [(3, 2, 1, f)] + ([(2, 3, 1, r)] if r > 0 else []),
        {f"heart->heart (partner edge {r})": (3, 2)} | ({f"heart->heart reverse (partner edge {f})": (2, 3)} if r > 0 else {}))
       for f, r in [(0.3, 0.0), (0.3, 0.15), (0.3, 0.3)]]
    + [("single-edge sweep", 4, [(s, t, lag, g)], {edge_type: (s, t)})
       for edge_type, (s, t, lag) in SWEEP_EDGES.items() for g in SWEEP_GAINS]
)


def gc_rows(rng):
    """F and PI (mean, 5-95%) of every reported edge of every design, over drawn dyads."""
    rows = []
    for simulation, k, edges, reported in DESIGNS:
        radii = sample_node_params(rng, SETTINGS["n_draws"], POLE_RADIUS_RANGES[:k])
        f0s = sample_node_params(rng, SETTINGS["n_draws"], F0_RANGES[:k])
        couplings = [build_coupling(radii[d], f0s[d], FS, edges, N_LAGS) for d in range(SETTINGS["n_draws"])]
        gains = {(s, t): g for s, t, _, g in edges}
        for edge_type, (source, target) in reported.items():
            gc = np.array([true_conditional_gc(c, source, target, order=GC_ORDER)[0] for c in couplings])
            pi = 1.0 - np.exp(-gc)
            rows.append(dict(simulation=simulation, edge_type=edge_type, gain=gains[source, target],
                             F_mean=gc.mean(), F_q05=np.quantile(gc, 0.05), F_q95=np.quantile(gc, 0.95),
                             PI_mean=pi.mean(), PI_q05=np.quantile(pi, 0.05), PI_q95=np.quantile(pi, 0.95)))
    return pd.DataFrame(rows)


def delta_task(task):
    """One S4 design-A study at `task['gain']`: per-dyad delta and null median of cg:ROI -> child:ROI, both estimators."""
    rng = np.random.default_rng(task["seed"])
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    designs = zscore_designs(simulate_dyads(rng, N_DYADS, int(SEG_S * FS), FS, POLE_RADIUS_RANGES[:4], F0_RANGES[:4],
                                            [(1, 0, 1, task["gain"])], N_LAGS))
    out = []
    for estimator in ESTIMATORS:
        est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=estimator, band_hz=BAND_HZ,
                   box_cox_lambda=BOX_COX)
        real, roots, _ = real_matrices(designs, NODES, est)
        values, _, _ = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng)
        keep = roots < STABILITY_MAX_ROOT
        real, values = real[keep], values[keep][:, keep]
        delta, _ = delta_and_z_all(real, values, "pooled")
        for d, n in zip(delta[:, 0, 1], real[:, 0, 1] - delta[:, 0, 1]):
            out.append(dict(estimator=estimator, gain=task["gain"], delta=d, null=n))
    return out


def delta_rows():
    """Median and 25-75% range of delta over dyads, and the median null value, per gain and estimator."""
    seeds = spawn_seeds(BASE_SEED, len(S4_GAINS) * SETTINGS["n_studies"])
    tasks = [{"gain": g, "seed": seeds[i * SETTINGS["n_studies"] + r]} for i, g in enumerate(S4_GAINS) for r in range(SETTINGS["n_studies"])]
    df = pd.DataFrame([r for recs in run_parallel(delta_task, tasks, N_JOBS) for r in recs])
    rows = []
    for gain, sub in df.groupby("gain"):
        row = {"simulation": "S4", "edge_type": "brain->brain", "gain": gain}
        for estimator, s in sub.groupby("estimator"):
            row |= {f"delta_{estimator}_med": s.delta.median(), f"delta_{estimator}_q25": s.delta.quantile(0.25),
                    f"delta_{estimator}_q75": s.delta.quantile(0.75), f"null_{estimator}_med": s.null.median()}
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(table):
    """Compact gain -> F, PI, median delta table (Supplementary Table S2)."""
    lines = ["| Simulation | Edge | Gain | F, mean (5–95%) | PI, mean (5–95%) | median Δ dDTF [IQR] | median Δ GPDC [IQR] |",
             "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        f = "—" if np.isnan(r.F_mean) else f"{r.F_mean:.3f} ({r.F_q05:.3f}–{r.F_q95:.3f})"
        pi = "—" if np.isnan(r.PI_mean) else f"{100 * r.PI_mean:.1f}% ({100 * r.PI_q05:.1f}–{100 * r.PI_q95:.1f}%)"
        dd = "—" if np.isnan(r.delta_dDTF_med) else f"{r.delta_dDTF_med:.4f} [{r.delta_dDTF_q25:.4f}, {r.delta_dDTF_q75:.4f}]"
        dg = "—" if np.isnan(r.delta_GPDC_med) else f"{r.delta_GPDC_med:.4f} [{r.delta_GPDC_q25:.4f}, {r.delta_GPDC_q75:.4f}]"
        lines.append(f"| {r.simulation} | {r.edge_type} | {r.gain:g} | {f} | {pi} | {dd} | {dg} |")
    null = table[table.simulation.eq("S4") & table.null_dDTF_med.notna()]
    lines += ["", "Median pseudo-dyad null value of the S4 edge (the level Δ is measured from), dDTF / GPDC: "
              + ", ".join(f"gain {r.gain:g}: {r.null_dDTF_med:.4f} / {r.null_GPDC_med:.4f}" for r in null.itertuples()) + "."]
    return "\n".join(lines) + "\n"


def main():
    """Compute the conversion table and write the CSV and Markdown files."""
    gc = gc_rows(np.random.default_rng(BASE_SEED))
    s4_zero = pd.DataFrame([dict(simulation="S4", edge_type="brain->brain", gain=0.0, F_mean=0.0, F_q05=0.0, F_q95=0.0,
                                 PI_mean=0.0, PI_q05=0.0, PI_q95=0.0)])
    table = pd.concat([s4_zero, gc]).merge(delta_rows(), on=["simulation", "edge_type", "gain"], how="left")
    columns = ["simulation", "edge_type", "gain", "F_mean", "F_q05", "F_q95", "PI_mean", "PI_q05", "PI_q95",
               "delta_dDTF_med", "delta_dDTF_q25", "delta_dDTF_q75", "delta_GPDC_med", "delta_GPDC_q25", "delta_GPDC_q75",
               "null_dDTF_med", "null_GPDC_med"]
    table = table[columns]
    table.to_csv(OUT / "gain_conversion.csv", index=False)
    (OUT / "gain_conversion.md").write_text(
        f"# Gain conversion ({MODE} mode: {SETTINGS['n_draws']} drawn dyads per design for F; "
        f"{SETTINGS['n_studies']} studies × {N_DYADS} dyads per gain for Δ)\n\n" + markdown_table(table))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
