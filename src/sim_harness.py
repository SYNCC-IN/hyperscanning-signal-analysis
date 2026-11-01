"""Small helpers shared by the simulation scripts: parallel runs, seeds, result folders."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def run_parallel(func, tasks, n_jobs):
    """Apply `func` to every task, in parallel processes if `n_jobs > 1`; results keep task order.

    Parameters
    ----------
    func : callable
        Module-level function taking one task (picklable).
    tasks : list
    n_jobs : int

    Returns
    -------
    list
    """
    if n_jobs == 1:
        return [func(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(func, tasks, chunksize=1))


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
