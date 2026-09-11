"""Small, generic statistics helpers with no pipeline-stage-specific logic."""

import numpy as np


def mean_and_sem(values, axis=0):
    """Mean and standard error of the mean (SEM) across realizations.

    Uses the standard error of the mean (SEM), adequate when the number of
    realizations along `axis` is in the tens-to-hundreds range.

    Parameters
    ----------
    values : np.ndarray
        Array of realizations to summarize.
    axis : int, optional
        Axis indexing independent realizations (default 0).

    Returns
    -------
    mean : np.ndarray
        Mean across `axis`.
    sem : np.ndarray
        Standard error of the mean across `axis`; the mean +/- sem gives an approximate 68% CI.
    """
    mean = values.mean(axis=axis)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
    return mean, sem
