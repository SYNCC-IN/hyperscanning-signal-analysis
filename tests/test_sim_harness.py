"""Checks of `src.sim_harness`: seeds and the parallel runner."""

from src.sim_harness import spawn_seeds


def test_spawned_seeds_do_not_depend_on_how_many_are_spawned():
    """Seed i is the same whatever n, so adding replicates or cells on another stream never moves existing seeds."""
    assert spawn_seeds(20260103, 1800)[:200] == spawn_seeds(20260103, 200)
    assert spawn_seeds([20260103, 1], 50)[:10] == spawn_seeds([20260103, 1], 10)
    assert not set(spawn_seeds([20260103, 1], 100)) & set(spawn_seeds(20260103, 100))


# ---------------------------------------------------------------------------
# Progress reporting (src.sim_progress through run_parallel)
# ---------------------------------------------------------------------------
import json

import numpy as np

from src import sim_progress
from src.sim_harness import run_parallel


def draw(task):
    """A toy task with random draws (module level, so it pickles)."""
    return float(np.random.default_rng(task["seed"]).normal())


def run_toy(tmp_path, monkeypatch, quiet, n_jobs):
    """Run 40 toy tasks with reporting; return (results, printed lines)."""
    monkeypatch.setenv("SIM_QUIET", "1" if quiet else "0")
    sim_progress.begin("S99 dDTF", tmp_path, mode="quick", replicates=10)
    tasks = [{"cell": c, "rep": r, "seed": 100 * c + r} for c in range(4) for r in range(10)]
    results = run_parallel(draw, tasks, n_jobs)
    sim_progress.finish(tmp_path, {"readout": "0.1-0.2"}, {"a": True, "b": False})
    return results


def test_quiet_run_prints_only_start_summary_and_end(tmp_path, monkeypatch, capsys):
    """SIM_QUIET=1: three lines (start, summary, end); the log has the same lines; the JSON lists the finished run."""
    run_toy(tmp_path, monkeypatch, quiet=True, n_jobs=2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split("] ")[1].split(":")[0] for line in lines] == ["start", "summary", "end"]
    assert "conditions=4, tasks=40, N_JOBS=2" in lines[0] and "criteria 1/2 met" in lines[1]
    assert len((tmp_path / "progress.log").read_text().strip().splitlines()) == 3
    state = json.loads((tmp_path / "progress.json").read_text())
    assert state["finished"][-1]["label"] == "S99 dDTF" and state["running"] == {}


def test_progress_lines_and_identical_results_with_reporting_on_and_off(tmp_path, monkeypatch, capsys):
    """Per-task lines every 5% (20 of 40 tasks + final), results identical to quiet and to a plain map, in task order."""
    loud = run_toy(tmp_path, monkeypatch, quiet=False, n_jobs=2)
    lines = capsys.readouterr().out.strip().splitlines()
    task_lines = [line for line in lines if " tasks (" in line]
    assert len(task_lines) == 20 and task_lines[-1].startswith("[S99 dDTF] 40/40 tasks (100%)") and "ETA" in task_lines[0]
    quiet = run_toy(tmp_path, monkeypatch, quiet=True, n_jobs=2)
    serial = run_toy(tmp_path, monkeypatch, quiet=False, n_jobs=1)
    plain = [draw({"seed": 100 * c + r}) for c in range(4) for r in range(10)]
    assert loud == quiet == serial == plain
