"""Small helpers shared by the simulation scripts: parallel runs (with progress reporting), seeds, result folders."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import sim_progress
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src import sim_progress


def run_parallel(func, tasks, n_jobs, result_lines=None):
    """Apply `func` to every task, in parallel processes if `n_jobs > 1`; results keep task order.

    Results are collected as they finish, and progress is reported through `src.sim_progress` (a start line, then a
    line every 5% of the tasks or every 60 s) when the script called `sim_progress.begin`. Reporting does not change
    the results.

    Parameters
    ----------
    func : callable
        Module-level function taking one task (picklable).
    tasks : list
    n_jobs : int
    result_lines : callable, optional
        ``result_lines(task, result) -> list of str``: extra lines printed for every finished task (e.g. one line
        per study, warnings); switched off with SIM_QUIET=1 like the progress lines.

    Returns
    -------
    list
    """
    sim_progress.start_line(sim_progress.count_conditions(tasks), len(tasks), n_jobs)
    progress = sim_progress.TaskProgress(len(tasks))
    lines = result_lines or (lambda task, result: [])
    if n_jobs == 1:
        results = []
        for task in tasks:
            results.append(func(task))
            progress.update(task, lines(task, results[-1]))
        return results
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(func, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            progress.update(tasks[i], lines(tasks[i], results[i]))
    return results


def spawn_seeds(base_seed, n):
    """`n` independent integer seeds derived from one base seed.

    Parameters
    ----------
    base_seed : int
    n : int

    Returns
    -------
    list of int
    """
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(base_seed).spawn(n)]


def results_dir(sim_root, sim_id, estimator):
    """Create and return ``<sim_root>/Results/<estimator>/S_<id>`` (id zero-padded to two digits).

    Parameters
    ----------
    sim_root : str or Path
        The folder holding the simulation scripts.
    sim_id : int
    estimator : str
        Connectivity estimator of the run (e.g. 'dDTF', 'GPDC'); one results tree per estimator.

    Returns
    -------
    Path
    """
    path = Path(sim_root) / "Results" / estimator / f"S_{sim_id:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rate_table(records, group_cols, flag_cols):
    """Mean of boolean/float columns by group, with the number of replications.

    Parameters
    ----------
    records : list of dict or pandas.DataFrame
    group_cols : list of str
    flag_cols : list of str

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.DataFrame(records)
    table = df.groupby(group_cols)[flag_cols].mean().reset_index()
    table["n_rep"] = df.groupby(group_cols).size().values
    return table
