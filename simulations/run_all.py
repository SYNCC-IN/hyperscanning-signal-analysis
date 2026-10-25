"""Run the simulation scripts S01-S11 in sequence and record the run time of each.

Usage:
    python simulations/run_all.py quick            # smoke test of every script (about a minute in total)
    python simulations/run_all.py full             # the full protocol (hours; see README.md)
    python simulations/run_all.py quick 3 5 9      # only S03, S05, S09

Each script writes to simulations/Results/S_xx/. The timing of this run goes to Results/run_all_timing_<mode>.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {1: "S01_recovery_orientation.py", 2: "S02_envelope_vs_phase.py", 3: "S03_null_calibration.py", 4: "S04_power.py",
           5: "S05_conditioning.py", 6: "S06_smoothness.py", 7: "S07_shared_stimulus.py", 8: "S08_sensitivity.py",
           9: "S09_cardiac_carrier_mismatch.py", 10: "S10_coupling_type_visibility.py", 11: "S11_comparison.py"}


def main():
    """Run the requested scripts and print a pass/fail table of their acceptance criteria."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    ids = [int(a) for a in sys.argv[2:]] or list(SCRIPTS)
    timing, table = {}, []
    for sim_id in ids:
        start = time.time()
        proc = subprocess.run([sys.executable, str(HERE / SCRIPTS[sim_id]), mode], capture_output=True, text=True)
        timing[f"S{sim_id:02d}"] = round(time.time() - start, 1)
        summary_path = HERE / "Results" / f"S_{sim_id:02d}" / "summary.json"
        criteria = json.loads(summary_path.read_text()).get("criteria", {}) if proc.returncode == 0 and summary_path.exists() else None
        status = "FAILED TO RUN" if criteria is None else f"{sum(criteria.values())}/{len(criteria)} criteria met"
        table.append((f"S{sim_id:02d}", timing[f"S{sim_id:02d}"], status))
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
    (HERE / "Results").mkdir(exist_ok=True)
    (HERE / "Results" / f"run_all_timing_{mode}.json").write_text(json.dumps(timing, indent=2))
    for name, seconds, status in table:
        print(f"{name}  {seconds:8.1f} s  {status}")


if __name__ == "__main__":
    main()
