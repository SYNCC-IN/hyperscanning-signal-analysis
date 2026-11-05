"""Film-matched surrogate-dyad null construction (Stage 5).

A surrogate stitches one dyad's child-role node variables with a *different*
dyad's caregiver-role node variables (same film) into a design matrix in
`node_names(nodes)` order. Both partners watched the same film but
were never in the room together, so any coupling recovered from a surrogate
reflects the shared stimulus + generic physiology, not real-time interaction
-- the null that real dyads' ffDTF is compared against
(`delta_and_z`) to isolate genuine interpersonal coupling. See
`DTF_analysis_notes/pipeline_plan.md` Stage 5 and `scripts/stage05_surrogate.py`
for the locked/open design decisions this module implements.

`Delta`/`z` stay signed everywhere -- never `abs()` or clipped -- because the
sign of a surviving effect is scientifically meaningful (e.g. negative
interpersonal HRV synchrony can be adaptive).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde, median_abs_deviation

try:
    from .connectivity import Granger_estimator, edge_value, read_edge_value
    from .design import assemble_design_matrix, detrend_windows, node_names, window_stack
    from .mvar_diag import ar_root_stability, fit_mvar_avg_acf
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.connectivity import Granger_estimator, edge_value, read_edge_value
    from src.design import assemble_design_matrix, detrend_windows, node_names, window_stack
    from src.mvar_diag import ar_root_stability, fit_mvar_avg_acf


def surrogate_pairs(dyad_ids, group_of=None):
    """All ordered (child_dyad, cg_dyad) pairs with child_dyad != cg_dyad.

    The film is fixed by the caller (surrogates are only ever built within one
    film), so this operates on the list of dyad ids present for that film.

    When ``group_of`` is given (a mapping ``dyad_id -> group label``), pairs are
    additionally restricted to partners from the *same* group -- the within-group
    null (Stage 5 D3 sensitivity, see `scripts/stage05_surrogate.py`). The
    default pooled null (L5) subtracts a *group-agnostic* per-film scalar, which
    is identical for TD and ASD and therefore cancels out of every between-group
    contrast (group main effect *and* film x group interaction): it centres the
    delta and protects a genuine group effect from being absorbed, but it removes
    nothing from the group comparisons and so cannot rule out a group-dependent
    stimulus response. Subtracting a same-group foreign-pair null instead removes
    each group's own shared-stimulus / generic-physiology baseline, so a
    surviving film x group interaction cannot be explained by the two groups
    responding to the same film differently. Foreign pairs never interacted in
    real time, so a same-group null carries no genuine coupling to absorb.
    ``group_of=None`` keeps the pooled behaviour unchanged.

    Parameters
    ----------
    dyad_ids : list of str
        Dyad ids present for one film.
    group_of : dict of {str: str}, optional
        Maps each dyad id to its group label. When provided, only ordered
        off-diagonal pairs whose two members share a group are returned. Default
        None (no group restriction).

    Returns
    -------
    list of tuple of str
        `(child_dyad_id, cg_dyad_id)`, ordered off-diagonal pairings. Length
        `N * (N - 1)` for `N` dyad ids when ``group_of`` is None; `sum_g
        n_g * (n_g - 1)` over groups when ``group_of`` is given.
    """
    pairs = [(child, cg) for child in dyad_ids for cg in dyad_ids if child != cg]
    if group_of is not None:
        pairs = [(child, cg) for child, cg in pairs if group_of[child] == group_of[cg]]
    return pairs


def assemble_surrogate_design(nodes, child_envelopes, cg_envelopes, zscore=True):
    """Stitch a foreign child with a foreign caregiver into one design matrix.

    Takes every `role == "child"` node's variable from `child_envelopes` and
    every `role == "caregiver"` node's variable from `cg_envelopes` (each a
    Stage 2 DataArray for the same film), truncates both to their common
    (shorter) time length -- the two segments are the same film but may
    differ by a sample or two -- and reassembles them in `node_names(nodes)`
    order with a fresh, shared time axis (the two source arrays' own time
    axes are dyad-specific and not meaningful once stitched). Works for any
    per-role node count/composition (L0/L5), not just one ROI + one HRV node
    per role. Z-scoring is delegated to `src.design.assemble_design_matrix`,
    so the surrogate design follows the identical per-channel z-score
    convention as a real one (single source of truth). No silent
    length-fixing beyond the documented truncate-to-min; a missing variable
    surfaces as the `.sel` `KeyError` it is.

    Parameters
    ----------
    nodes : list of dict
        `pipeline_config.json`'s `"shared".nodes` (see `src.design.node_names`).
    child_envelopes : xarray.DataArray
        Stage 2 output for the dyad supplying the child-role node variables,
        dims `("variable", "time")`.
    cg_envelopes : xarray.DataArray
        Stage 2 output for the dyad supplying the caregiver-role node
        variables, same film, same sampling frequency.
    zscore : bool, optional
        Passed through to `src.design.assemble_design_matrix` (default True).

    Returns
    -------
    np.ndarray, shape (len(nodes), n_common)
        Rows in `node_names(nodes)` order, z-scored per row when `zscore` is
        True.
    """
    names = node_names(nodes)
    child_names = [node["name"] for node in nodes if node["role"] == "child"]
    cg_names = [node["name"] for node in nodes if node["role"] == "caregiver"]

    child_part = child_envelopes.sel(variable=child_names)
    cg_part = cg_envelopes.sel(variable=cg_names)
    n_common = min(child_part.sizes["time"], cg_part.sizes["time"])

    child_values = child_part.isel(time=slice(0, n_common)).values
    cg_values = cg_part.isel(time=slice(0, n_common)).values
    fs = float(child_envelopes.attrs["fs"])

    values_by_name = dict(zip(child_names, child_values))
    values_by_name.update(zip(cg_names, cg_values))
    data = np.stack([values_by_name[name] for name in names], axis=0)
    time = np.arange(n_common) / fs
    stitched = xr.DataArray(data, dims=("variable", "time"), coords={"variable": names, "time": time})
    return assemble_design_matrix(stitched, names, zscore=zscore)


def windowed_ar_stability(design, win_len, step, p, detrend_type="linear"):
    """Companion-matrix stability of the windowed-ACF AR fit of one design matrix.

    Composition of existing diagnostics at a fixed order: `window_stack` ->
    `detrend_windows` -> `fit_mvar_avg_acf(p)` -> `ar_root_stability`. Used to
    gate surrogates and to flag real dyads at the Stage 5 common order.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix (real or surrogate).
    win_len, step : int
        Window length / step in samples.
    p : int
        MVAR model order.
    detrend_type : {'linear', 'constant'}, optional
        Per-window detrend type (default 'linear').

    Returns
    -------
    max_abs_root : float
        Largest companion-matrix eigenvalue modulus.
    is_stable : bool
        Whether every eigenvalue modulus is strictly below 1.
    """
    stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    ar_coeffs, _ = fit_mvar_avg_acf(stack, p)
    _, max_abs_root, is_stable = ar_root_stability(ar_coeffs)
    return max_abs_root, is_stable


def band_average_cube(cube, freqs, band_hz):
    """Average a (k, k, n_freqs) cube over an inclusive frequency band -> (k, k).

    Identical maths to Stage 4's script-local `band_average`, lifted into
    `src/` so both stages share one definition.

    Parameters
    ----------
    cube : np.ndarray, shape (k, k, n_freqs)
        ffDTF (or similar) cube.
    freqs : np.ndarray
        Frequency axis (Hz) matching `cube`'s last axis.
    band_hz : tuple of float
        `(low, high)` band edges in Hz, inclusive.

    Returns
    -------
    np.ndarray, shape (k, k)
        Band-averaged matrix.
    """
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    return cube[:, :, band_mask].mean(axis=2)


def delta_and_z(real_value, null_values):
    """Signed delta and z of a real edge value against its surrogate null.

    `delta = real - median(null)`; `z = (real - median(null)) / MAD(null)`. Both the
    centre and the spread are robust statistics (median, median absolute deviation)
    rather than mean/std, so a handful of extreme surrogate draws cannot dominate
    either. `null_std` (kept under its historical name for schema compatibility with
    downstream tables) is actually the MAD scaled by `scipy.stats.median_abs_deviation`'s
    `scale="normal"` (`MAD * 1.4826`), which makes it a consistent estimator of the
    standard deviation under a normal null and keeps `z` on the same scale it had
    when computed from `std(ddof=1)`. Signed on purpose -- never `abs()` or clipped,
    since the sign of what survives null subtraction is scientifically meaningful. A
    degenerate near-zero null MAD is left to surface as a large/inf `z` and is visible
    via `null_std`/`n_null` in the caller's table -- not silently patched.

    Parameters
    ----------
    real_value : float
        Real dyad's band-averaged ffDTF for one edge.
    null_values : array-like
        Surrogate null draws for the same edge.

    Returns
    -------
    dict
        Keys: `delta`, `z`, `null_median`, `null_std`, `n_null`.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_median = float(np.median(null_values))
    null_std = float(median_abs_deviation(null_values, scale="normal"))
    delta = float(real_value - null_median)
    z = delta / null_std
    return {"delta": delta, "z": z, "null_median": null_median, "null_std": null_std, "n_null": int(null_values.size)}


def real_edge_values(dyads, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz):
    """Per-dyad band-averaged connectivity value for `edge` (the real, same-dyad estimate).

    Parameters
    ----------
    dyads : list of np.ndarray, each shape (k, n_samples)
        One z-scored design matrix per dyad.
    edge : tuple of int
        `(target, source)` raw indices, `src.connectivity.edge_value`'s convention.
    freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz :
        Passed through to `src.connectivity.edge_value`.

    Returns
    -------
    np.ndarray
        One band-averaged value per dyad.
    """
    return np.array([
        edge_value(dyad, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz)
        for dyad in dyads
    ])


def surrogate_null(dyads, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz):
    """Pooled surrogate null for `edge`, over every foreign (child, caregiver) pairing.

    `surrogate_pairs` enumerates every ordered foreign pairing; each surrogate
    takes channel 0 from one dyad and channel 1 from another, so each
    channel's marginal statistics are preserved but any real interaction is
    destroyed.

    Parameters
    ----------
    dyads : list of np.ndarray, each shape (2, n_samples)
        Two-channel dyads (e.g. rough/smooth or child/caregiver proxies).
    edge : tuple of int
        `(target, source)` raw indices, `src.connectivity.edge_value`'s convention.
    freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz :
        Passed through to `src.connectivity.edge_value`.

    Returns
    -------
    np.ndarray
        One band-averaged value per foreign pairing.
    """
    dyad_ids = list(range(len(dyads)))
    values = []
    for id_a, id_b in surrogate_pairs(dyad_ids):
        design = np.stack([dyads[id_a][0], dyads[id_b][1]], axis=0)
        values.append(edge_value(design, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz))
    return np.array(values)


def edge_class_for(source_name, target_name, edge_class):
    """Return this directed edge's H2/H4/exploratory/"other" tag.

    Parameters
    ----------
    source_name, target_name : str
        Design-variable names (e.g. `"cg:ROI"`, `"child:ROI"`).
    edge_class : dict
        `{(source_name, target_name): class_label}` for the named edges;
        any edge not in this dict defaults to `"other"`.
    """
    return edge_class.get((source_name, target_name), "other")


def plot_null_vs_real_violin(edges_to_plot, null_matrix, real_by_dyad, all_edges, edge_class, names,
                              estimator, box_cox_lambda, title, delta_space=False):
    """Split violin of the surrogate null vs real dyads (TD left / ASD right), per edge.

    Density-normalised (KDE), not a raw-count histogram: the surrogate null pool (hundreds
    to thousands of draws) vastly outnumbers real dyads (tens), so a count-based plot would
    make the null dwarf the real distributions regardless of effect size. Each of the three
    densities (null, TD, ASD) is independently normalised to unit area by `gaussian_kde`
    (area, not count), then all three share one width-scale constant -- so violin width
    reflects relative density, not sample size. The null (grey, both halves, should look
    roughly symmetric about its mean) sits in the background; TD occupies the left half and
    ASD the right half of the same y-scale, for an at-a-glance group-vs-group and
    group-vs-null read.

    Parameters
    ----------
    edges_to_plot : list of tuple(str, str)
        `(source_name, target_name)` edges to render, one panel each.
    null_matrix : np.ndarray, shape (n_pairs_kept, len(all_edges))
        Pooled surrogate null draws, columns in `all_edges` order.
    real_by_dyad : dict
        `{dyad_id: {"band_avg": (4,4) array, "group": str, ...}}` for this film.
    all_edges : list of tuple(str, str)
        Column order of `null_matrix`.
    edge_class : dict
        Passed to `edge_class_for` for the panel subtitle.
    names : list of str
        Node names, in `real_by_dyad`'s `band_avg` row/column order (see
        `src.design.node_names`).
    estimator : str
        Estimator name, for the y-axis label.
    box_cox_lambda : float
        Box-Cox exponent applied upstream (-1 = none), for the y-axis label.
    title : str
        Figure title.
    delta_space : bool, optional
        If False (default), plot raw band-averaged estimator values. If True, shift every
        value in a panel by that panel's own `-null_median` before plotting -- i.e. plot
        `delta_dtf` (`delta_and_z`'s signed `real - median(null)`) instead of
        the raw value. The null violin is then centred on zero by construction; each real
        dyad's offset from zero IS its `delta_dtf`, read directly off the y-axis. Uses the
        median (matching `delta_and_z`), not the mean, so the delta shown here is exactly
        the `delta_dtf` value in the tidy table -- not a different, mean-centred quantity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    group_colors = {"TD": "tab:blue", "ASD": "tab:orange"}
    group_sides = {"TD": -1, "ASD": 1}
    n_cols = 3
    n_rows = int(np.ceil(len(edges_to_plot) / n_cols))
    half_width = 0.4  # max half-violin width (x-axis units), shared by null/TD/ASD
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
    axes_flat = list(np.atleast_1d(axes).flat)

    for panel_idx, (axis, (source_name, target_name)) in enumerate(zip(axes_flat, edges_to_plot)):
        edge_idx = all_edges.index((source_name, target_name))
        null_values = null_matrix[:, edge_idx]
        offset = np.median(null_values) if delta_space else 0.0
        null_values = null_values - offset
        values_by_group = {
            group_label: np.array([
                read_edge_value(info["band_avg"], source_name, target_name, names=names) - offset
                for info in real_by_dyad.values() if info["group"] == group_label
            ])
            for group_label in group_sides
        }

        all_values = np.concatenate([null_values] + list(values_by_group.values()))
        y_grid = np.linspace(all_values.min(), all_values.max(), 200)

        null_density = gaussian_kde(null_values)(y_grid)
        scale = half_width / null_density.max()
        axis.fill_betweenx(y_grid, -null_density * scale, null_density * scale,
                            color="lightgrey", alpha=0.6, zorder=1, label="surrogate null")
        axis.axhline(np.median(null_values), color="black", linestyle="--", linewidth=1, zorder=2, label="null median")

        for group_label, side in group_sides.items():
            values = values_by_group[group_label]
            if values.size < 2:
                continue
            density = gaussian_kde(values)(y_grid) * scale
            color = group_colors[group_label]
            axis.fill_betweenx(y_grid, 0, side * density, color=color, alpha=0.7, zorder=3, label=group_label)
            tick_x = sorted([0, side * 0.6 * half_width])
            axis.hlines(np.median(values), tick_x[0], tick_x[1], color=color, linewidth=2, zorder=4)

        axis.set_title(f"{source_name} -> {target_name} ({edge_class_for(source_name, target_name, edge_class)})", fontsize=9)
        axis.set_xlim(-half_width * 1.1, half_width * 1.1)
        axis.set_xticks([-half_width / 2, half_width / 2])
        axis.set_xticklabels(["TD", "ASD"])
        if panel_idx == 0:
            axis.legend(fontsize=6, loc="upper right")

    box_cox_suffix = "" if box_cox_lambda == -1 else f", box_cox_lambda={box_cox_lambda}"
    y_axis_label = (f"delta_dtf ({estimator}, real - null_median{box_cox_suffix})" if delta_space
                     else f"band-avg {estimator}{box_cox_suffix}")
    for axis in axes_flat[: n_rows * n_cols : n_cols]:
        axis.set_ylabel(y_axis_label)
    for axis in axes_flat[len(edges_to_plot):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_delta_summary(delta_table_df, edges_to_plot, title):
    """Per-group mean +/- SEM of `delta_dtf` for the given edges.

    Parameters
    ----------
    delta_table_df : pd.DataFrame
        The tidy Stage 5 table (or a subset), with `group`, `source`,
        `target`, `delta_dtf` columns.
    edges_to_plot : list of tuple(str, str)
        `(source_name, target_name)` edges, in display order.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    edge_labels = [f"{s}->{t}" for s, t in edges_to_plot]
    x = np.arange(len(edge_labels))
    width = 0.35
    group_labels = sorted(delta_table_df["group"].unique())
    figure, axis = plt.subplots(figsize=(7, 4))
    for i, group_label in enumerate(group_labels):
        group_df = delta_table_df[delta_table_df["group"] == group_label]
        means, sems = [], []
        for source_name, target_name in edges_to_plot:
            edge_df = group_df[(group_df["source"] == source_name) & (group_df["target"] == target_name)]
            means.append(edge_df["delta_dtf"].mean())
            sems.append(edge_df["delta_dtf"].std(ddof=1) / np.sqrt(len(edge_df)))
        offset = (i - (len(group_labels) - 1) / 2) * width
        axis.bar(x + offset, means, width=width, yerr=sems, label=group_label, capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(edge_labels, rotation=20, fontsize=8)
    axis.set_ylabel("mean delta_dtf (real - null), +/- SEM")
    axis.legend(title="group")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def compute_null(candidate_pairs, envelopes_by_dyad, win_len, step, model_order, detrend_type,
                  stability_max_root, freqs, fs, estimator, box_cox_lambda, band_hz, all_edges, nodes):
    """Estimate a surrogate null matrix for one set of candidate mismatched pairs.

    Scope-agnostic core shared by a pooled reference null and a within-group
    sensitivity null: the caller decides which pairs go in (all off-diagonal
    for the pooled null; same-group off-diagonal for a within-group null).
    Each pair is stitched (`assemble_surrogate_design`), stability-gated
    (`windowed_ar_stability`), estimated (`Granger_estimator`) and
    band-averaged; kept draws are stacked in `all_edges` column order.

    Parameters
    ----------
    candidate_pairs : list of tuple(str, str)
        Ordered `(child_dyad, cg_dyad)` pairs to attempt.
    envelopes_by_dyad : dict
        `{dyad_id: (envelopes DataArray, order_record)}` for this film.
    win_len, step : int
        Locked window geometry.
    model_order : int
        Fixed MVAR model order for every fit (real and surrogate).
    detrend_type : {'linear', 'constant'}
        Per-window detrend type.
    stability_max_root : float
        A candidate pair is excluded from the null if its max AR companion
        eigenvalue modulus is >= this.
    freqs : np.ndarray
        Frequency axis (Hz).
    fs : float
        Sampling frequency (Hz).
    estimator : str
        Estimator name passed to `Granger_estimator`.
    box_cox_lambda : float
        Box-Cox exponent passed to `Granger_estimator` (-1 = none).
    band_hz : tuple of float
        `(low, high)` band edges in Hz, passed to `band_average_cube`.
    all_edges : list of tuple(str, str)
        Column order for the returned `null_matrix`.
    nodes : list of dict
        `pipeline_config.json`'s `"shared".nodes`, passed through to
        `assemble_surrogate_design` and used to read `all_edges` off the
        resulting band-averaged cube (see `src.design.node_names`).

    Returns
    -------
    dict
        Keys: `null_matrix` (n_kept, len(all_edges)), `kept_child_dyads`,
        `kept_cg_dyads`, `n_excluded_unstable`, `n_attempted`.
    """
    names = node_names(nodes)
    null_rows, kept_child_dyads, kept_cg_dyads = [], [], []
    n_excluded_unstable = 0
    for child_dyad, cg_dyad in candidate_pairs:
        assert child_dyad != cg_dyad
        child_envelopes, _ = envelopes_by_dyad[child_dyad]
        cg_envelopes, _ = envelopes_by_dyad[cg_dyad]
        design = assemble_surrogate_design(nodes, child_envelopes, cg_envelopes, zscore=True)

        max_abs_root, _ = windowed_ar_stability(design, win_len, step, model_order, detrend_type)
        if max_abs_root >= stability_max_root:
            n_excluded_unstable += 1
            continue

        ffdtf, _ = Granger_estimator(design, freqs, fs, model_order, win_len, step, detrend_type, ESTIMATOR=estimator, box_cox_lambda=box_cox_lambda)
        band_avg = band_average_cube(ffdtf, freqs, band_hz)
        null_rows.append([read_edge_value(band_avg, s, t, names=names) for s, t in all_edges])
        kept_child_dyads.append(child_dyad)
        kept_cg_dyads.append(cg_dyad)

    return {
        "null_matrix": np.array(null_rows),
        "kept_child_dyads": kept_child_dyads,
        "kept_cg_dyads": kept_cg_dyads,
        "n_excluded_unstable": n_excluded_unstable,
        "n_attempted": len(candidate_pairs),
    }
