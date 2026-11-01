"""Run the simulation scripts S01-S11 for each connectivity estimator, then compare the estimators.

Usage:
    python simulations_multimodal_MVAR/dDTF/run_all.py quick                 # smoke test, dDTF and GPDC (a couple of minutes)
    python simulations_multimodal_MVAR/dDTF/run_all.py full                  # the full protocol for both estimators (hours; see README.md)
    python simulations_multimodal_MVAR/dDTF/run_all.py quick 3 5 9           # only S03, S05, S09
    python simulations_multimodal_MVAR/dDTF/run_all.py full GPDC             # only the GPDC runs (e.g. dDTF results already exist)

Each script writes to simulations_multimodal_MVAR/dDTF/Results/<estimator>/S_xx/. Afterwards compare_estimators.py
writes the side-by-side tables to Results/comparison/. The timing of this run goes to Results/run_all_timing_<mode>.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESTIMATORS = ["dDTF", "GPDC"]
SCRIPTS = {1: "S01_recovery_orientation.py", 2: "S02_envelope_vs_phase.py", 3: "S03_null_calibration.py", 4: "S04_power.py",
           5: "S05_conditioning.py", 6: "S06_smoothness.py", 7: "S07_shared_stimulus.py", 8: "S08_sensitivity.py",
           9: "S09_cardiac_carrier_mismatch.py", 10: "S10_coupling_type_visibility.py", 11: "S11_comparison.py"}


def main():
    """Run the requested scripts for the requested estimators, print a pass/fail table and build the comparison."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    estimators = [a for a in sys.argv[2:] if a in ESTIMATORS] or ESTIMATORS
    ids = [int(a) for a in sys.argv[2:] if a not in ESTIMATORS] or list(SCRIPTS)
    timing, status = {}, {}
    for estimator in estimators:
        for sim_id in ids:
            name = f"S{sim_id:02d}"
            start = time.time()
            proc = subprocess.run([sys.executable, str(HERE / SCRIPTS[sim_id]), mode, estimator], capture_output=True, text=True)
            timing[f"{estimator}/{name}"] = round(time.time() - start, 1)
            summary_path = HERE / "Results" / estimator / f"S_{sim_id:02d}" / "summary.json"
            criteria = json.loads(summary_path.read_text()).get("criteria", {}) if proc.returncode == 0 and summary_path.exists() else None
            status[estimator, name] = "FAILED TO RUN" if criteria is None else f"{sum(criteria.values())}/{len(criteria)} criteria met"
            if proc.returncode != 0:
                print(proc.stderr[-2000:])
    (HERE / "Results" / f"run_all_timing_{mode}.json").write_text(json.dumps(timing, indent=2))
    print("       " + "".join(f"{e:<24}" for e in estimators))
    for sim_id in ids:
        name = f"S{sim_id:02d}"
        print(f"{name}    " + "".join(f"{status[e, name]:<24}" for e in estimators))
    subprocess.run([sys.executable, str(HERE / "compare_estimators.py")], check=True)


if __name__ == "__main__":
    main()
