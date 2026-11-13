"""Check A2 - why is the group contrast with the within-group null liberal? Fixed delta vs `renull`.

Hypothesis: `delta_and_z_all(..., mode="within_group")` subtracts each group's null median once; the label permutations
then run on this fixed delta. All dyads of a group share the estimation error of their group's null median, which
enters the observed group difference in full but partly cancels in relabelled pseudo-groups, so the permutation
distribution is too narrow. `renull=True` (`src.sim_inference`) recomputes the within-group null of every dyad under
every permutation.

Parts (each writes its own raw CSV; studies are saved in chunks, so a part can be resumed):
  s3   S3 null (no coupling), N = 16 / 25 / 50 (8 vs 8, 12 vs 13, 25 vs 25), 60 s; 1000 studies at N = 16, else 500.
       Data: S03's generator and seeds (N = 16: the extra stream of S03, i.e. the same 1000 studies; N = 25, 50: S03's
       200 studies, then a new stream). Per-edge rate and max-T family-wise rate over the 8 cross-person edges; for the
       diagnostic the observed signed group difference and the variance of its permutation distribution, per edge.
  s13  S13 scenario a (2 x 15 dyads, 3 films x 60 s): the same studies as S13 (reps 0-99) plus reps 100-199 from S13's
       seed index. Surrogates over all 30 dyads per film (the within-group blocks are S13's tables). Family-wise rate
       of the group contrast (max-T over 8 within film, per film x study) and of the film x group interaction.
  s4   Power cost, S04 design B (half the dyads with cg:ROI->child:ROI at gain g), 12 vs 13 and 25 vs 25, 60 s,
       g = 0.1, 0.2, 200 studies (new seeds).
  summarize   tables and figure from the raw CSVs.
Both versions use the same data and the same label permutations (two generators from one seed).

Usage: python CHK_A2_renull.py s3|s13|s4|summarize [dDTF|GPDC]
Results: simulations_multimodal_MVAR/checks_2026-11/A2/
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
from src.sim_generators import sample_node_params, simulate_dyads, simulate_dyads_fixed_params, zscore_designs
from src.sim_harness import run_parallel, spawn_seeds
from src import sim_progress
from src.sim_inference import (delta_and_z_all, group_mean_difference, group_permutation_maxT, label_permutations,
                               real_matrices, renull_statistics, score_study, surrogate_values, wilson_interval)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PART = sys.argv[1] if len(sys.argv) > 1 else "summarize"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
CHUNK = 200                                    # studies per saved chunk
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "A2"

FS = 2.5
SEG_S = 60
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
         "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1),
         "cg:ROI->child:HRV": (1, 2), "child:ROI->cg:HRV": (0, 3)}   # the 8 cross-person edges (the family)
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05
N_PERM = 1000

# S3 cells: N -> number of studies; seeds as in S03_null_calibration.py
S3_CELLS = {16: 1000, 25: 500, 50: 500}
S3_BASE_SEED = 20260103
S3_FULL_CELLS = [(n, s) for n in [10, 25, 50] for s in [60, 120, 300]]   # S03 main stream, 200 studies per cell
S3_N_REP = 200
S3_EXTRA_CELLS = [(10, 60), (10, 120), (10, 300), (16, 60)]               # S03 extra stream
S3_EXTRA_STRIDE = 10000
A2_BASE_SEED = 20261113                                                  # new streams of this check

# S13 scenario a
S13_BASE_SEED, S13_N_REP_MAX, S13_SCENARIO_INDEX = 20260113, 1000, 0     # scenario 'a' is the first of S13's SCENARIOS
S13_N_STUDIES, S13_N_PER_GROUP = 200, 15
FILMS = ["F1", "F2", "F3"]

# S4 design B
S4_N = [25, 50]
S4_GAINS = [0.1, 0.2]
S4_N_STUDIES = 200
S4_EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1)}
TARGET_EDGE = "cg:ROI->child:ROI"

VERSIONS = {"current": False, "renull": True}
PUB = {"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5,
       "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
       "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
       "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3.5,
       "legend.frameon": False, "axes.titlepad": 3}            # as in figures_manuscript.py
EST_COL = {"dDTF": "#1b9e9e", "GPDC": "#7a5cb8"}
FPR_BAND = (0.03, 0.07)


def estimation():
    """Keyword arguments of `edge_matrix` (the default pipeline)."""
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    return dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ,
                box_cox_lambda=BOX_COX)


def s3_seeds(n_dyads, total):
    """Seeds of the S3 studies of the N-dyad 60 s cell: S03's own seeds where S03 ran them, then a new stream."""
    if (n_dyads, SEG_S) in S3_EXTRA_CELLS:
        j = S3_EXTRA_CELLS.index((n_dyads, SEG_S))
        return spawn_seeds([S3_BASE_SEED, 1], len(S3_EXTRA_CELLS) * S3_EXTRA_STRIDE)[j * S3_EXTRA_STRIDE:j * S3_EXTRA_STRIDE + total]
    i = S3_FULL_CELLS.index((n_dyads, SEG_S))
    own = spawn_seeds(S3_BASE_SEED, len(S3_FULL_CELLS) * S3_N_REP)[i * S3_N_REP:(i + 1) * S3_N_REP]
    return own + spawn_seeds([A2_BASE_SEED, n_dyads], total)[S3_N_REP:total]


def stable_study(designs, groups):
    """Real matrices, surrogate table and groups of the dyads whose real fit is stable."""
    real, roots, _ = real_matrices(designs, NODES, estimation())
    values, _, _ = surrogate_values(designs, NODES, CHILD_NODES, estimation(), STABILITY_MAX_ROOT, None)
    stable = roots < STABILITY_MAX_ROOT
    return real[stable], values[stable][:, stable], groups[stable]


def both_versions(real, values, groups, edges, seed):
    """`score_study` with the within-group null, current and renull, on the same permutations; family = 8 edges, max-T."""
    out = {}
    for version, renull in VERSIONS.items():
        out[version] = score_study(real, values, real, "within_group", groups, edges, DYAD_ALPHA, ALPHA,
                                   np.random.default_rng([seed, 7]), N_PERM, family=EDGES, multiplicity=["maxT"],
                                   multiplicity_rng=np.random.default_rng([seed, 8]), renull=renull)
    return out


def permutation_spread(real, values, groups, seed):
    """Observed signed group difference per edge and the variance of its permutation distribution, both versions."""
    targets, sources = [t for _, t in EDGES.values()], [s for s, _ in EDGES.values()]
    delta, z = delta_and_z_all(real, values, "within_group", groups)
    d = np.where(np.isfinite(z[:, targets, sources]), delta[:, targets, sources], np.nan)
    labels = label_permutations(groups == 1, np.random.default_rng([seed, 9]), N_PERM)
    observed = group_mean_difference(d, (groups == 1)[None])[0]
    var_current = group_mean_difference(d, labels).var(axis=0, ddof=1)
    var_renull = renull_statistics(real[:, targets, sources], values[:, :, targets, sources], np.isfinite(d), labels).var(axis=0, ddof=1)
    return observed, var_current, var_renull


def run_s3(task):
    """One S3 null study: per-edge and family-wise group contrast (both versions) and the permutation spread."""
    n, seed = task["n_dyads"], task["seed"]
    designs = zscore_designs(simulate_dyads(np.random.default_rng(seed), n, int(SEG_S * FS), FS, POLE_RADIUS_RANGES,
                                            F0_RANGES, [], N_LAGS))
    real, values, groups = stable_study(designs, (np.arange(n) >= n // 2).astype(int))
    scored = both_versions(real, values, groups, EDGES, seed)
    observed, var_current, var_renull = permutation_spread(real, values, groups, seed)
    rows = []
    for e, name in enumerate(EDGES):
        row = {"estimator": ESTIMATOR, "n_dyads": n, "rep": task["rep"], "seed": seed, "edge": name, "n_stable": len(real),
               "d_obs": observed[e], "var_perm_current": var_current[e], "var_perm_renull": var_renull[e]}
        for version, out in scored.items():
            row[f"group_p_{version}"] = out[name]["group_p"]
            row[f"group_p_maxT_{version}"] = out[name]["group_p_adj_maxT"]
            row[f"any_reject_maxT_{version}"] = out[name]["any_reject_group_maxT"]
        rows.append(row)
    return rows


def s13_delta_inputs(seed):
    """Simulate one S13 scenario-a study (as `simulate_delta_table`) and return per-film real, full surrogate table, delta.

    Returns
    -------
    groups : np.ndarray, shape (30,)
    real : np.ndarray, shape (n_films, 30, 8)
        NaN for dyads whose real fit is unstable in that film.
    values : np.ndarray, shape (n_films, 30, 30, 8)
        All foreign pairs of the film (both groups); rows/columns of unstable dyads NaN.
    delta : np.ndarray, shape (n_films, 30, 8)
        Delta against the within-group null (S13's DV).
    """
    rng = np.random.default_rng(seed)
    n = 2 * S13_N_PER_GROUP
    groups = np.r_[np.zeros(S13_N_PER_GROUP, int), np.ones(S13_N_PER_GROUP, int)]
    radii, f0s = sample_node_params(rng, n, POLE_RADIUS_RANGES), sample_node_params(rng, n, F0_RANGES)
    targets, sources = [t for _, t in EDGES.values()], [s for s, _ in EDGES.values()]
    reals, tables, deltas = [], [], []
    for _ in FILMS:
        designs = zscore_designs(simulate_dyads_fixed_params(rng, radii, f0s, int(SEG_S * FS), FS, [[] for _ in range(n)], N_LAGS))
        real, roots, _ = real_matrices(designs, NODES, estimation())
        values, _, _ = surrogate_values(designs, NODES, CHILD_NODES, estimation(), STABILITY_MAX_ROOT, None)
        unstable = roots >= STABILITY_MAX_ROOT
        real[unstable] = np.nan
        values[unstable] = np.nan
        values[:, unstable] = np.nan
        delta, _ = delta_and_z_all(real, values, "within_group", groups)
        reals.append(real[:, targets, sources]); tables.append(values[:, :, targets, sources]); deltas.append(delta[:, targets, sources])
    return groups, np.stack(reals), np.stack(tables), np.stack(deltas)


def run_s13(task):
    """One S13 scenario-a study: group contrast per film (max-T over 8) and interaction (max-T over 8), both versions."""
    groups, real, values, delta = s13_delta_inputs(task["seed"])
    n_edges = len(EDGES)
    eye = np.eye(n_edges)
    contrast_map = np.vstack([eye, -eye / 2, -eye / 2])                   # F1 - mean(F2, F3)
    real_stacked = np.concatenate(list(real), axis=1)                      # (n, films x edges)
    values_stacked = np.concatenate(list(values), axis=2)
    contrast = delta[0] - (delta[1] + delta[2]) / 2
    rows = []
    for version, renull in VERSIONS.items():
        rng = np.random.default_rng([task["seed"], 7])
        tests = {f"group_{film}": (delta[f], dict(real=real[f], pair_values=values[f])) for f, film in enumerate(FILMS)}
        tests["interaction"] = (contrast, dict(real=real_stacked, pair_values=values_stacked, combine=contrast_map))
        for test, (dv, extra) in tests.items():
            res = group_permutation_maxT(dv, groups, rng, N_PERM, renull=renull, **(extra if renull else {}))
            rows.append({"estimator": ESTIMATOR, "rep": task["rep"], "seed": task["seed"], "version": version, "test": test,
                         "min_p_unadjusted": float(res["unadjusted"].min()), "min_p_maxT": float(res["maxT"].min()),
                         "any_reject_maxT": bool(np.any(res["maxT"] < ALPHA)),
                         "per_edge_rate": float(np.mean(res["unadjusted"] < ALPHA))})
    return rows


def run_s4(task):
    """One S04 design-B study: planted-edge group contrast, per edge and after max-T over 8, both versions."""
    rng = np.random.default_rng(task["seed"])
    n, gain = task["n_dyads"], task["gain"]
    half = n // 2
    d0 = simulate_dyads(rng, half, int(SEG_S * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS)
    d1 = simulate_dyads(rng, n - half, int(SEG_S * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, [(1, 0, 1, gain)], N_LAGS)
    real, values, groups = stable_study(zscore_designs(np.concatenate([d0, d1])), np.r_[np.zeros(half, int), np.ones(n - half, int)])
    scored = both_versions(real, values, groups, S4_EDGES, task["seed"])
    return [{"estimator": ESTIMATOR, "n_dyads": n, "gain": gain, "rep": task["rep"], "seed": task["seed"], "version": version,
             "edge": TARGET_EDGE, "group_p": out[TARGET_EDGE]["group_p"], "group_p_maxT": out[TARGET_EDGE]["group_p_adj_maxT"]}
            for version, out in scored.items()]


def run_chunked(func, tasks, name):
    """Run tasks in chunks of CHUNK studies, saving each chunk; chunks already on disk are skipped."""
    folder = OUT / "raw" / f"{name}_{ESTIMATOR}"
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tasks), CHUNK):
        path = folder / f"chunk_{start:05d}.csv"
        if path.exists():
            continue
        results = run_parallel(func, tasks[start:start + CHUNK], N_JOBS)
        pd.DataFrame([r for rows in results for r in rows]).to_csv(path, index=False)
    frame = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("chunk_*.csv"))], ignore_index=True)
    frame.to_csv(OUT / f"raw_{name}_{ESTIMATOR}.csv", index=False)
    return frame


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def rate_row(flags, **keys):
    """Rate with 95% Wilson interval of a boolean series."""
    k, n = int(np.sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return dict(keys, rate=k / n, ci_low=low, ci_high=high, n=n)


def summarize():
    """Rate table (cell x measure x version), diagnostic table, power table and the figure."""
    rows, diag, power = [], [], []
    for est in EST_COL:
        s3 = pd.read_csv(OUT / f"raw_s3_{est}.csv")
        for n, sub in s3.groupby("n_dyads"):
            cell = f"S3 null, {n // 2} vs {n - n // 2}"
            for version in VERSIONS:
                rows.append(rate_row(sub[f"group_p_{version}"] < ALPHA, estimator=est, cell=cell, n_per_group=n / 2,
                                     measure="per-edge rate (mean over 8 edges)", version=version))
                rows.append(rate_row(sub.groupby("rep")[f"any_reject_maxT_{version}"].first().astype(bool), estimator=est,
                                     cell=cell, n_per_group=n / 2, measure="family-wise rate, max-T over 8", version=version))
                per_edge = sub.groupby("edge").agg(var_obs=("d_obs", "var"), var_perm=(f"var_perm_{version}", "mean"))
                diag.append({"estimator": est, "cell": cell, "version": version, "n_studies": sub.rep.nunique(),
                             "var_observed": per_edge.var_obs.mean(), "mean_var_permutation": per_edge.var_perm.mean(),
                             "ratio": (per_edge.var_obs / per_edge.var_perm).mean(),
                             "ratio_min_edge": (per_edge.var_obs / per_edge.var_perm).min(),
                             "ratio_max_edge": (per_edge.var_obs / per_edge.var_perm).max()})
        s13 = pd.read_csv(OUT / f"raw_s13_{est}.csv")
        for version in VERSIONS:
            sub = s13[s13.version == version]
            group = sub[sub.test.str.startswith("group_")]
            rows.append(rate_row(group.any_reject_maxT, estimator=est, cell="S13 a, 15 vs 15, 3 films", n_per_group=15,
                                 measure="group contrast, family-wise, max-T within film (per film x study)", version=version))
            rows.append(rate_row(group.groupby("rep").any_reject_maxT.any(), estimator=est, cell="S13 a, 15 vs 15, 3 films",
                                 n_per_group=15, measure="group contrast, family-wise, any film (per study)", version=version))
            rows.append(rate_row(sub[sub.test == "interaction"].any_reject_maxT, estimator=est, cell="S13 a, 15 vs 15, 3 films",
                                 n_per_group=15, measure="interaction, family-wise, max-T over 8", version=version))
        s4 = pd.read_csv(OUT / f"raw_s4_{est}.csv")
        for (n, gain, version), sub in s4.groupby(["n_dyads", "gain", "version"]):
            for measure, column in (("power, uncorrected", "group_p"), ("power, max-T over 8", "group_p_maxT")):
                power.append(rate_row(sub[column] < ALPHA, estimator=est, cell=f"S4 design B, {n // 2} vs {n - n // 2}",
                                      gain=gain, measure=measure, version=version))
    rates, diag, power = pd.DataFrame(rows), pd.DataFrame(diag), pd.DataFrame(power)
    rates.to_csv(OUT / "summary_rates.csv", index=False)
    diag.to_csv(OUT / "summary_variance_ratio.csv", index=False)
    power.to_csv(OUT / "summary_power.csv", index=False)
    plot(rates)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rates.round(4).to_string()); print(diag.round(4).to_string()); print(power.round(4).to_string())


def plot(rates):
    """Rate vs dyads per group: current (solid) vs renull (dashed), dDTF filled, GPDC open, band 3-7%."""
    measures = ["per-edge rate (mean over 8 edges)", "family-wise rate, max-T over 8"]
    titles = ["A  per-edge rate, S3 null", "B  family-wise rate (max-T, 8 edges), S3 null"]
    s3 = rates[rates.cell.str.startswith("S3")]
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
        for ax, measure, title in zip(axes[:2], measures, titles):
            ax.axhspan(*FPR_BAND, color="0.88", zorder=0, lw=0); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6, zorder=0)
            for i, est in enumerate(EST_COL):
                for version, ls in (("current", "-"), ("renull", "--")):
                    sub = s3[(s3.estimator == est) & (s3.measure == measure) & (s3.version == version)].sort_values("n_per_group")
                    x = sub.n_per_group + (i - 0.5) * 0.3 + (0.15 if version == "renull" else -0.15)
                    ax.errorbar(x, sub.rate, yerr=[sub.rate - sub.ci_low, sub.ci_high - sub.rate], color=EST_COL[est], ls=ls,
                                marker="o", mfc=EST_COL[est] if est == "dDTF" else "white", capsize=1.5, elinewidth=0.6,
                                label=f"{est}, {version}")
            ax.set_xticks([8, 12.5, 25]); ax.set_xticklabels(["8 vs 8", "12 vs 13", "25 vs 25"])
            ax.set_xlabel("dyads per group"); ax.set_ylabel("false-positive rate"); ax.set_title(title, loc="left")
            ax.set_ylim(0, None)
        axes[0].legend(ncol=1)
        ax = axes[2]
        s13 = rates[rates.cell.str.startswith("S13") & ~rates.measure.str.contains("any film")]
        ax.axhspan(*FPR_BAND, color="0.88", zorder=0, lw=0); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6, zorder=0)
        labels = ["group (within film)", "interaction"]
        for j, key in enumerate(["group contrast", "interaction"]):
            for i, est in enumerate(EST_COL):
                for k, version in enumerate(VERSIONS):
                    r = s13[(s13.estimator == est) & s13.measure.str.startswith(key) & (s13.version == version)].iloc[0]
                    x = j + (i - 0.5) * 0.4 + (k - 0.5) * 0.18
                    bars = ax.errorbar(x, r.rate, yerr=[[r.rate - r.ci_low], [r.ci_high - r.rate]], color=EST_COL[est], marker="o",
                                       mfc=EST_COL[est] if est == "dDTF" else "white", capsize=1.5, elinewidth=0.8, ls="none")
                    bars[2][0].set_linestyle("-" if version == "current" else "--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, None)
        ax.set_ylabel("family-wise rate (max-T over 8)")
        ax.set_title("C  S13 scenario a, 15 vs 15", loc="left")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fig_A2_renull.{ext}", dpi=300)
        plt.close(fig)


def main():
    """Run one part (or summarize)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if PART == "summarize":
        summarize()
        return
    sim_progress.begin(f"CHK A2 {PART} {ESTIMATOR}", OUT, part=PART)
    if PART == "s3":
        tasks = [{"n_dyads": n, "rep": r, "seed": seed} for n, total in S3_CELLS.items() for r, seed in enumerate(s3_seeds(n, total))]
        frame = run_chunked(run_s3, tasks, "s3")
        readout = {f"N={n} per-edge current/renull": f"{(s.group_p_current < ALPHA).mean():.3f}/{(s.group_p_renull < ALPHA).mean():.3f}"
                   for n, s in frame.groupby("n_dyads")}
    elif PART == "s13":
        seeds = spawn_seeds(S13_BASE_SEED, 6 * S13_N_REP_MAX)
        tasks = [{"rep": r, "seed": seeds[S13_SCENARIO_INDEX * S13_N_REP_MAX + r]} for r in range(S13_N_STUDIES)]
        frame = run_chunked(run_s13, tasks, "s13")
        readout = {f"{t} {v}": f"{s.any_reject_maxT.mean():.3f}" for (t, v), s in frame.groupby(["test", "version"])}
    else:
        cells = [(n, g) for n in S4_N for g in S4_GAINS]
        seeds = spawn_seeds([A2_BASE_SEED, 4], len(cells) * S4_N_STUDIES)
        tasks = [{"n_dyads": n, "gain": g, "rep": r, "seed": seeds[i * S4_N_STUDIES + r]}
                 for i, (n, g) in enumerate(cells) for r in range(S4_N_STUDIES)]
        frame = run_chunked(run_s4, tasks, "s4")
        readout = {f"N={n} g={g} {v}": f"{(s.group_p < ALPHA).mean():.3f}" for (n, g, v), s in frame.groupby(["n_dyads", "gain", "version"])}
    sim_progress.finish(OUT, readout)


if __name__ == "__main__":
    main()
