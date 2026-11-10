"""Progress reporting for long simulation runs: console lines, a log file and a JSON state file.

A script calls `begin` once (label, results folder, settings), `src.sim_harness.run_parallel` reports the tasks, and
the script ends with `finish` (summary line with its main readouts, end line). Every line is printed with
``flush=True`` (so it shows under nohup, tee and background jobs) and appended to ``<results>/progress.log``
(``tail -f`` works); ``<results>/progress.json`` holds the running tasks and the finished runs and is rewritten after
every update (atomically). The environment variable SIM_QUIET=1 switches the per-task lines off; start, end and summary
lines stay on. Nothing here touches the simulations, their random streams or their results.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

STATE = {"label": None, "log": None, "json": None, "start": None, "settings": {}}


def format_duration(seconds):
    """Seconds as H:MM:SS."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def quiet():
    """True if per-task progress lines are switched off (SIM_QUIET=1)."""
    return os.environ.get("SIM_QUIET") == "1"


def emit(line, log_path=None):
    """Print a progress line (flushed) and append it to the log file.

    Parameters
    ----------
    line : str
    log_path : str or Path, optional
        Log file; default: the one set by `begin` (none if `begin` was not called).
    """
    print(line, flush=True)
    path = log_path or STATE["log"]
    if path is not None:
        with open(path, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")


def update_state(update, json_path=None):
    """Read-modify-write the JSON state file (atomic replace).

    Parameters
    ----------
    update : callable
        Function that changes the state dict in place.
    json_path : str or Path, optional
        Default: the one set by `begin` (nothing is written if neither is set).
    """
    path = Path(json_path or STATE["json"]) if (json_path or STATE["json"]) else None
    if path is None:
        return
    state = json.loads(path.read_text()) if path.exists() else {}
    update(state)
    state["updated"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, path)


def begin(label, results_root, **settings):
    """Start reporting for one simulation and estimator.

    Parameters
    ----------
    label : str
        E.g. 'S04 GPDC'.
    results_root : str or Path
        Folder of progress.log and progress.json (the simulations' Results folder).
    **settings
        Shown in the start line (mode, conditions, replicates, ...).
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    STATE.update(label=label, log=root / "progress.log", json=root / "progress.json", start=time.time(), settings=settings)


def count_conditions(tasks):
    """Number of distinct conditions among tasks (task fields except seed and replicate; 1 for non-dict tasks)."""
    return len({describe_task(task) for task in tasks}) if tasks else 0


def start_line(n_conditions, n_tasks, n_jobs):
    """Print the start line of a parallel batch (settings of `begin`, conditions, tasks, processes)."""
    if STATE["label"] is None:
        return
    settings = ", ".join(f"{k}={v}" for k, v in STATE["settings"].items())
    emit(f"[{STATE['label']}] start: {settings}{', ' if settings else ''}conditions={n_conditions}, tasks={n_tasks}, N_JOBS={n_jobs}")


def describe_task(task):
    """Short description of a task for progress lines (dict fields except seeds and replicate)."""
    if not isinstance(task, dict):
        return ""
    return ", ".join(f"{k}={v}" for k, v in task.items() if k not in ("seed", "rep"))


class TaskProgress:
    """Throttled per-task progress lines and the 'running' entry of the JSON state."""

    def __init__(self, n_tasks, every_fraction=0.05, every_seconds=60.0):
        """Set up reporting for `n_tasks` tasks: a line every `every_fraction` of the tasks or `every_seconds`."""
        self.n_tasks, self.done = n_tasks, 0
        self.start = self.last_time = time.time()
        self.step = max(1, int(every_fraction * n_tasks))
        self.next_count, self.every_seconds = self.step, every_seconds

    def update(self, task, extra_lines=()):
        """Count one finished task; emit its extra lines and, when due, a progress line."""
        self.done += 1
        if STATE["label"] is None:
            return
        for line in extra_lines:
            if not quiet():
                emit(f"[{STATE['label']}] {line}")
        now = time.time()
        if self.done < self.next_count and now - self.last_time < self.every_seconds and self.done < self.n_tasks:
            return
        self.next_count = (self.done // self.step + 1) * self.step
        self.last_time = now
        elapsed = now - self.start
        eta = elapsed / self.done * (self.n_tasks - self.done)
        last = describe_task(task)
        if not quiet():
            emit(f"[{STATE['label']}] {self.done}/{self.n_tasks} tasks ({100 * self.done / self.n_tasks:.0f}%) | "
                 f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}" + (f" | last: {last}" if last else ""))
        entry = {"tasks_done": self.done, "tasks_total": self.n_tasks, "elapsed": format_duration(elapsed),
                 "eta": format_duration(eta), "last": last}
        update_state(lambda s: s.setdefault("running", {}).__setitem__(STATE["label"], entry))


def range_text(values, decimals=3):
    """'min-max' of a numeric series (or 'n/a' when empty)."""
    values = [v for v in values if v == v]
    return f"{min(values):.{decimals}f}-{max(values):.{decimals}f}" if values else "n/a"


def finish(results_path, readouts=None, criteria=None):
    """Print the summary line (main readouts, criteria met / total) and the end line; record the run as finished.

    Parameters
    ----------
    results_path : str or Path
        Folder of the run's results.
    readouts : dict of {str: str}, optional
        Main readouts, e.g. {'FPR range (null cells)': '0.030-0.080', 'power (N=25, 60 s, gain 0.4)': '0.91'}.
    criteria : dict of {str: bool}, optional
    """
    label = STATE["label"]
    parts = [f"{k}: {v}" for k, v in (readouts or {}).items()]
    if criteria is not None:
        parts.append(f"criteria {sum(bool(v) for v in criteria.values())}/{len(criteria)} met")
    emit(f"[{label}] summary: " + " | ".join(parts))
    seconds = time.time() - STATE["start"]
    emit(f"[{label}] end: {format_duration(seconds)} | results: {results_path}")

    def record(state):
        state.get("running", {}).pop(label, None)
        state.setdefault("finished", []).append({"label": label, "time": format_duration(seconds), "seconds": round(seconds),
                                                 "results": str(results_path), "finished_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"})
    update_state(record)
