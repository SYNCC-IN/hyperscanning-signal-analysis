"""Run the simulation scripts S01-S13 and the variants S04b, S06b for each connectivity estimator, then compare the estimators.

Usage:
    python simulations_multimodal_MVAR/dDTF/run_all.py quick                 # smoke test, dDTF and GPDC
    python simulations_multimodal_MVAR/dDTF/run_all.py full                  # the full protocol for both estimators (hours; see README.md)
    python simulations_multimodal_MVAR/dDTF/run_all.py quick 3 5 9           # only S03, S05, S09
    python simulations_multimodal_MVAR/dDTF/run_all.py full 4b 6b            # only the variants
    python simulations_multimodal_MVAR/dDTF/run_all.py full GPDC             # only the GPDC runs (e.g. dDTF results already exist)

Simulations run in the order of RUN_ORDER (calibration first, S13 last); each simulation runs for every estimator before
the next one starts. Each script writes to simulations_multimodal_MVAR/dDTF/Results/<estimator>/S_xx/ (S04b: S_04/partial/,
S06b: S_06/noise/).

Progress: every script prints start, per-task, summary and end lines (`src.sim_progress`); run_all adds one line per
finished run. All lines also go to Results/progress.log (`tail -f`), and Results/progress.json holds the run_all state
(current run, runs done / total, elapsed, ETA from the previous timings), the running tasks and the finished runs.
SIM_QUIET=1 switches the per-task lines off. Afterwards compare_estimators.py
writes the side-by-side tables to Results/comparison/. The timings go to Results/run_all_timing_<mode>.json
(merged with the timings already there, so runs of single estimators or scripts keep the others).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src import sim_progress  # noqa: E402

RESULTS = HERE / "Results"
PROGRESS_LOG, PROGRESS_JSON = RESULTS / "progress.log", RESULTS / "progress.json"
ESTIMATORS = ["dDTF", "GPDC"]
SCRIPTS = {"1": "S01_recovery_orientation.py", "2": "S02_envelope_vs_phase.py", "3": "S03_null_calibration.py", "4": "S04_power.py",
           "4b": "S04b_partial_coupling.py", "5": "S05_conditioning.py", "6": "S06_smoothness.py", "6b": "S06b_unequal_snr.py",
           "7": "S07_shared_stimulus.py", "8": "S08_sensitivity.py", "9": "S09_cardiac_carrier_mismatch.py",
           "10": "S10_coupling_type_visibility.py", "11": "S11_comparison.py", "12": "S12_mutual_coupling.py", "13": "S13_group_film_bayes.py"}
RUN_ORDER = ["3", "4", "4b", "1", "12", "5", "6", "6b", "7", "8", "2", "9", "10", "11", "13"]
SUMMARY_PATHS = {"4b": "S_04/partial/summary.json", "6b": "S_06/noise/summary.json"}   # default: S_<id>/summary.json


def sim_name(sim_id):
    """Display name of a simulation id, e.g. '4' -> 'S04', '4b' -> 'S04b'."""
    return f"S{int(sim_id.rstrip('b')):02d}{'b' if sim_id.endswith('b') else ''}"


def summary_path(estimator, sim_id):
    """Path of a run's summary.json."""
    return RESULTS / estimator / (SUMMARY_PATHS[sim_id] if sim_id in SUMMARY_PATHS else f"S_{int(sim_id):02d}/summary.json")


def main():
    """Run the requested scripts for the requested estimators, print a pass/fail table and build the comparison."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    estimators = [a for a in sys.argv[2:] if a in ESTIMATORS] or ESTIMATORS
    ids = [a for a in sys.argv[2:] if a not in ESTIMATORS]
    ids = [i for i in RUN_ORDER if i in ids] if ids else RUN_ORDER
    timing, status = {}, {}
    runs = [(sim_id, estimator) for sim_id in ids for estimator in estimators]
    timing_path = RESULTS / f"run_all_timing_{mode}.json"
    previous = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    start_all = time.time()
    for k, (sim_id, estimator) in enumerate(runs):
        name = sim_name(sim_id)
        remaining = [previous.get(f"{e}/{sim_name(i)}") for i, e in runs[k:]]
        eta = sim_progress.format_duration(sum(remaining)) if all(remaining) else "unknown (no previous timing)"
        sim_progress.update_state(lambda st: st.__setitem__("run_all", {
            "mode": mode, "current": f"{name} {estimator}", "runs_done": k, "runs_total": len(runs),
            "elapsed": sim_progress.format_duration(time.time() - start_all), "eta_from_previous_timings": eta}), PROGRESS_JSON)
        start = time.time()
        proc = subprocess.run([sys.executable, str(HERE / SCRIPTS[sim_id]), mode, estimator])
        timing[f"{estimator}/{name}"] = round(time.time() - start, 1)
        path = summary_path(estimator, sim_id)
        criteria = json.loads(path.read_text()).get("criteria", {}) if proc.returncode == 0 and path.exists() else None
        status[estimator, name] = "FAILED TO RUN" if criteria is None else f"{sum(criteria.values())}/{len(criteria)} criteria met"
        nxt = f"{sim_name(runs[k + 1][0])} {runs[k + 1][1]}" if k + 1 < len(runs) else "-"
        sim_progress.emit(f"[run_all] {k + 1}/{len(runs)} simulations done | elapsed {sim_progress.format_duration(time.time() - start_all)}"
                          f" | last: {name} {estimator} {status[estimator, name]} | next: {nxt}", PROGRESS_LOG)
        timing_path.write_text(json.dumps(previous | timing, indent=2))
    sim_progress.update_state(lambda st: st.__setitem__("run_all", {
        "mode": mode, "current": "-", "runs_done": len(runs), "runs_total": len(runs),
        "elapsed": sim_progress.format_duration(time.time() - start_all), "eta_from_previous_timings": "0:00:00"}), PROGRESS_JSON)
    print("       " + "".join(f"{e:<24}" for e in estimators))
    for sim_id in ids:
        name = sim_name(sim_id)
        print(f"{name:<7}" + "".join(f"{status[e, name]:<24}" for e in estimators))
    subprocess.run([sys.executable, str(HERE / "compare_estimators.py")], check=True)


if __name__ == "__main__":
    main()
