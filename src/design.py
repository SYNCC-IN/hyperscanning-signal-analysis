"""MVAR design-variable construction for the interbrain ffDTF + HRV pipeline.

Stage 2 part: turns the continuous, ROI-selected EEG and IBI signals
assembled by `src.assemble.assemble_dyad` into the four per-dyad-per-film
design variables (`child:ROI`, `cg:ROI`, `child:HRV`, `cg:HRV`) that feed the
MVAR model in later stages. The ROI variables are individualized-band
amplitude envelopes; the HRV variables are the raw (interpolated) IBI signal,
downsampled only -- no band-pass, no Hilbert (see `scripts/stage02_envelopes.py`
for the rationale). Both are computed on the whole continuous `passive_movies`
chunk and segmented to a film window only afterwards, so filter/Hilbert edge
transients fall in the discarded margins/gaps rather than inside the retained
window -- see `scripts/stage02_envelopes.py` for the orchestration built on
top of this module.

Stage 3 part: `assemble_design_matrix` turns one Stage 2 output file into the
`(4, n_samples)` array that feeds the MVAR fit, and that Stage 4 reuses
unchanged for ffDTF -- the single source of truth for the design matrix.
`window_stack`/`detrend_windows` then cut that per-film matrix into short,
per-window-detrended segments stacked on a trials axis, feeding the
windowed-ACF-averaged fit (`src.mvar_diag.fit_mvar_avg_acf`) that is Stage 3's
default estimation path (see `DTF_analysis_notes/pipeline_plan.md` Stage 3):
short windows are closer to locally stationary for the non-stationary HRV
variable than the whole ~60 s film, and averaging the autocovariance across
many short windows (inside `src.mtmvar.count_corr`) composes them into one
stable MVAR estimate.
"""

import numpy as np
import xarray as xr
from scipy.signal import detrend as _detrend
from scipy.stats import zscore as _zscore

try:
    from .envelopes import filter_individual_band, hilbert_envelope, downsample
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.envelopes import filter_individual_band, hilbert_envelope, downsample

def node_names(nodes):
    """Node names, in MVAR row order, from a config `nodes` list.

    Parameters
    ----------
    nodes : list of dict
        `pipeline_config.json`'s `"shared".nodes`, each entry
        `{"name": str, "role": "child"|"caregiver", "signal": str}`. List
        order is the MVAR row order.

    Returns
    -------
    list of str
    """
    return [node["name"] for node in nodes]


def rows_for_signal(nodes, signal):
    """Row indices of the nodes whose `signal` field matches `signal`.

    Parameters
    ----------
    nodes : list of dict
        See `node_names`.
    signal : str
        Signal type to match, e.g. `"roi_envelope"` or `"raw_ibi"`.

    Returns
    -------
    list of int
        Indices into `nodes` (== MVAR row indices) in list order.
    """
    return [row for row, node in enumerate(nodes) if node["signal"] == signal]


def assert_edges_known(edges, names, context=""):
    """Raise a loud, edge-naming error if any (source, target) pair references an unknown node.

    Used at stage load time to validate `pipeline_config.json`'s
    `edge_topology`/`PRIMARY_FAMILY` against the actual node topology
    (`node_names(nodes)`), instead of letting a typo surface later as a bare
    `.index()` `ValueError` with no indication of which edge was at fault.

    Parameters
    ----------
    edges : list of tuple(str, str)
        `(source_name, target_name)` pairs to validate.
    names : list of str
        Known node names, e.g. `node_names(nodes)`.
    context : str, optional
        Prefix identifying the source of `edges` in the raised message (e.g.
        `"stage05 pipeline_config.json's edge_topology"`).
    """
    known = set(names)
    for source, target in edges:
        if source not in known:
            raise ValueError(f"{context}: edge {source}->{target} references unknown source node {source!r}; known nodes: {names}")
        if target not in known:
            raise ValueError(f"{context}: edge {source}->{target} references unknown target node {target!r}; known nodes: {names}")


def roi_band_envelope(roi_signals, sfreq, center_freq, bandwidth, order, target_sfreq, reduction):
    """Continuous downsampled amplitude envelope of an individualized ROI band.

    Parameters
    ----------
    roi_signals : np.ndarray, shape (n_roi_channels, n_times)
        Continuous ROI-selected signal, e.g. `dyad['eeg'][role]['data']` from
        `src.assemble.assemble_dyad`.
    sfreq : float
        Sampling frequency of `roi_signals` in Hz.
    center_freq : float
        Individualized rhythm center frequency in Hz.
    bandwidth : float
        Half-width of the individualized passband in Hz (see
        `src.envelopes.filter_individual_band`).
    order : int
        Butterworth filter order.
    target_sfreq : float
        Desired envelope sampling frequency in Hz.
    reduction : {"average_envelopes", "average_raw"}
        "average_envelopes": filter + Hilbert each ROI channel separately,
        then average the resulting envelopes. "average_raw": average the raw
        ROI channels first, then filter + Hilbert once. The two coincide for
        a single-channel ROI.

    Returns
    -------
    envelope : np.ndarray
        Downsampled instantaneous amplitude envelope.
    env_sfreq : float
        Realized envelope sampling frequency in Hz.
    """
    if reduction == "average_envelopes":
        channel_envelopes = []
        for channel_signal in roi_signals:
            filtered = filter_individual_band(channel_signal, sfreq, center_freq, bandwidth, order)
            channel_envelopes.append(hilbert_envelope(filtered))
        envelope_full = np.mean(channel_envelopes, axis=0)
    elif reduction == "average_raw":
        averaged = roi_signals.mean(axis=0)
        filtered = filter_individual_band(averaged, sfreq, center_freq, bandwidth, order)
        envelope_full = hilbert_envelope(filtered)
    else:
        raise ValueError(f"Unknown reduction {reduction!r}; expected 'average_envelopes' or 'average_raw'")

    return downsample(envelope_full, sfreq, target_sfreq)


def segment_signal(signal, fs, t0, film_start_s, film_end_s):
    """Slice a continuous downsampled signal to one film window.

    Used for all four design variables -- the ROI amplitude envelopes and
    the raw downsampled IBI alike.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Continuous downsampled signal.
    fs : float
        Sampling frequency of `signal` in Hz.
    t0 : float
        Time (s) of the source continuous chunk's first sample (i.e. the
        `time[0]` of the signal `signal` was derived from), so the
        downsampled time axis lines up with `film_start_s`/`film_end_s`,
        which are expressed in that same reference (see `coverage.csv`).
    film_start_s : float
        Film window start, in the same time reference as `t0`.
    film_end_s : float
        Film window end, in the same time reference as `t0`.

    Returns
    -------
    segment : np.ndarray
        Signal values with `film_start_s <= t <= film_end_s`.
    time : np.ndarray
        Downsampled time axis for `segment`, in the same reference as `t0`.
    """
    time = t0 + np.arange(signal.size) / fs
    mask = (time >= film_start_s) & (time <= film_end_s)
    return signal[mask], time[mask]


def stack_design(node_signals, names, fs, attrs):
    """Stack the per-node MVAR design variables into one labeled DataArray.

    Parameters
    ----------
    node_signals : list of np.ndarray, each shape (n_times,)
        Already-segmented per-node 1-D signals, in `names` order (a node's
        ROI amplitude envelope or raw downsampled IBI, per its `nodes[i]`
        config entry's `signal` field).
    names : list of str
        Node names, in the same order as `node_signals` (see
        `node_names`). Written as the `variable` coordinate.
    fs : float
        Realized common sampling frequency of every node signal, in Hz.
    attrs : dict
        Written verbatim as the returned DataArray's attrs (e.g. `fs`,
        `roi_label`, `roi_channels`, band parameters, `hrv_signal`, `film`,
        `dyad_id`, `group`, `age_months`, `target_sfreq`, `zscored`,
        `roi_reduction`, `bw_convention`).

    Returns
    -------
    xarray.DataArray, dims ("variable", "time")
        `variable` coordinate set to `names`.
    """
    lengths = {signal.size for signal in node_signals}
    if len(lengths) != 1:
        raise ValueError(f"Design variables must share one length, got {lengths}")

    n_times = node_signals[0].size
    data = np.stack(node_signals, axis=0)
    time = np.arange(n_times) / fs
    return xr.DataArray(
        data,
        dims=("variable", "time"),
        coords={"variable": names, "time": time},
        attrs=attrs,
    )


def assemble_design_matrix(envelopes, names, zscore=True):
    """Turn a Stage 2 envelope DataArray into the (k, n_samples) MVAR design matrix.

    Selects the design variables in `names` order (`xarray`'s label-based
    `.sel` raises if any is missing or misnamed -- a real error, not
    something to mask) and, by default, z-scores each variable across time
    so physically incomparable signals (e.g. uV-scale EEG envelope vs
    ms-scale raw IBI) enter the MVAR on one scale. This is the single source
    of truth for the design matrix, reused unchanged by Stage 4.

    Parameters
    ----------
    envelopes : xarray.DataArray
        Stage 2 output, dims ("variable", "time"), physical units (see
        `stack_design`).
    names : list of str
        Node names, in MVAR row order (see `node_names`).
    zscore : bool, optional
        If True (default), z-score each variable across time (``ddof=0``,
        matching Stage 2's QC z-scoring). The persisted Stage 2 file stays in
        physical units; z-scoring happens here, at design-matrix assembly.

    Returns
    -------
    np.ndarray, shape (len(names), n_samples)
        Rows in `names` order.
    """
    design = envelopes.sel(variable=names).values
    if zscore:
        design = _zscore(design, axis=1, ddof=0)
    return design


def window_stack(design, win_len, step):
    """Cut a design matrix into windows, stacked on a trials axis.

    Tiles `design` (already segmented to one film and already z-scored per
    channel over the whole film -- see `assemble_design_matrix`) into
    fixed-length windows along the time axis, discarding any trailing partial
    window rather than padding it. Applies no filtering, detrending, or other
    transform -- see `detrend_windows` for the per-window stationarisation
    step that follows this one. The caller (not this function, which does not
    know the model order `p`) is responsible for asserting ``win_len > p``.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        Design matrix, node order per `node_names(nodes)`.
    win_len : int
        Window length in samples.
    step : int
        Step between window starts, in samples (``step < win_len`` gives
        overlapping windows).

    Returns
    -------
    np.ndarray, shape (k, win_len, n_windows)
        Windows stacked on axis 2 -- the trials axis `src.mtmvar.count_corr`
        averages the autocovariance over.

    Raises
    ------
    ValueError
        If `design` has fewer than `win_len` samples, so not even one window fits.
    """
    _, n_samples = design.shape
    if n_samples < win_len:
        raise ValueError(f"design has {n_samples} samples, shorter than win_len={win_len}; cannot form one window")

    window_starts = range(0, n_samples - win_len + 1, step)
    windows = [design[:, start:start + win_len] for start in window_starts]
    return np.stack(windows, axis=2)


def detrend_windows(stack, dtype='linear'):
    """Detrend each (channel, window) time series independently, per window.

    Applied after `window_stack` and after the global per-film z-score --
    never to the whole continuous film. Removes the mean (``dtype='constant'``)
    or the mean and linear trend (``dtype='linear'``) within each window along
    the within-window time axis, resolving Stage 2's deferred "raw IBI not
    detrended / carries LF-VLF drift" caveat at the design-matrix stage. This
    removes mean + slope only, not variance, so the cross-window variance
    heterogeneity that the averaged-ACF fit (`src.mvar_diag.fit_mvar_avg_acf`)
    is meant to absorb is preserved.

    Parameters
    ----------
    stack : np.ndarray, shape (k, win_len, n_windows)
        Windowed design matrix from `window_stack`.
    dtype : {'linear', 'constant'}, optional
        Detrend type passed to `scipy.signal.detrend`. ``'linear'`` (default)
        removes mean + slope; its low-frequency attenuation edge sits at
        roughly ``1 / win_len_seconds``, so the window length must keep that
        below the coupling band of interest. ``'constant'`` removes only the
        mean/DC offset, for use if the window must be shortened enough that
        ``'linear'`` would eat into the coupling band.

    Returns
    -------
    np.ndarray, shape (k, win_len, n_windows)
        Detrended windows, same shape as `stack`.
    """
    return _detrend(stack, axis=1, type=dtype)


def window_geometry(win_len_s, overlap_frac, target_sfreq):
    """Derive integer window length/step (samples) from a length/overlap spec.

    Parameters
    ----------
    win_len_s : float
        Window length in seconds.
    overlap_frac : float
        Fractional overlap between consecutive windows (0 = none, 0.5 = half).
    target_sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    win_len : int
        Window length in samples.
    step : int
        Step between window starts, in samples.
    """
    win_len = round(win_len_s * target_sfreq)
    step = round(win_len * (1 - overlap_frac))
    return win_len, step
