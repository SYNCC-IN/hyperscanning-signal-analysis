"""Signal and debug visualizations for the hyperscanning pipelines.

PSD overlays, ffDTF debug grids/matrices (passive) and H10-vs-ECG alignment
diagnostics (SECoRe). Statistical-result plots live in ``statistical_plots``.
"""
import matplotlib.pyplot as plt
import numpy as np


def _stack_psd_on_common_freq_notch(traces, fmax):
    if not traces:
        return None, None

    ref_freqs = None
    ref_len = -1
    for freqs, pxx, event, dyad_id in traces:
        keep = np.isfinite(freqs) & np.isfinite(pxx) & (freqs <= fmax) & (freqs >= 0)
        f = freqs[keep]
        if f.size > ref_len:
            ref_freqs = f
            ref_len = f.size

    if ref_freqs is None or ref_freqs.size < 2:
        return None, None

    stacked = []
    for freqs, pxx, event, dyad_id in traces:
        keep = np.isfinite(freqs) & np.isfinite(pxx) & (freqs <= fmax) & (freqs >= 0)
        f = freqs[keep]
        y = pxx[keep]
        if f.size < 2:
            continue

        y_interp = np.full(ref_freqs.shape, np.nan, dtype=float)
        overlap = (ref_freqs >= f.min()) & (ref_freqs <= f.max())
        if np.any(overlap):
            y_interp[overlap] = np.interp(ref_freqs[overlap], f, y)
        stacked.append(y_interp)

    if not stacked:
        return None, None
    return ref_freqs, np.vstack(stacked)


def _robust_ymax_from_notch(psd_by_role, analysis_channels, role_key, fmax):
    maxima = []
    for ch_name in analysis_channels:
        for freqs, pxx, event, dyad_id in psd_by_role[role_key].get(ch_name, []):
            keep = np.isfinite(freqs) & np.isfinite(pxx) & (freqs <= fmax) & (freqs >= 0)
            vals = pxx[keep]
            if vals.size:
                maxima.append(np.max(vals))

    y = np.asarray(maxima, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    ymax = q3 + 1.5 * iqr
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = np.nanmax(y)
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    return float(ymax)


def plot_role_overlay(psd_by_role, analysis_channels, rows, row_channels, max_cols, role_key, role_label, fmax=20.0, alpha=0.4):
    spectrum_maxima = []
    for ch_name in analysis_channels:
        for freqs, pxx, event, dyad_id in psd_by_role[role_key].get(ch_name, []):
            keep = freqs <= fmax
            vals = pxx[keep]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                spectrum_maxima.append(np.max(vals))

    y = np.asarray(spectrum_maxima, dtype=float)
    y = y[np.isfinite(y)]

    if y.size > 0:
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        ymax = q3 + 1.5 * iqr
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = np.nanmax(y)
    else:
        ymax = 1.0

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    fig, axes = plt.subplots(3, max_cols, figsize=(3.2 * max_cols, 8.5), squeeze=False, sharex=True, sharey=True)
    fig.suptitle(
        f"EEG PSD overlay in event window ({role_label})\\nWelch nperseg=2*fs (resolution ~0.5 Hz), ylim from spectrum maxima (Q3+1.5*IQR)",
        y=1.02,
    )

    for r, row_name in enumerate(rows):
        chs = row_channels[r]
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(chs):
                ax.axis("off")
                continue

            ch_name = chs[c]
            traces = psd_by_role[role_key].get(ch_name, [])
            if not traces:
                ax.text(0.5, 0.5, f"{ch_name}\\n(no data)", ha="center", va="center", fontsize=9, transform=ax.transAxes)
                ax.set_xlim(0, fmax)
                ax.set_ylim(0, ymax)
                ax.grid(alpha=0.2)
                continue

            for freqs, pxx, event, dyad_id in traces:
                keep = freqs <= fmax
                ax.plot(freqs[keep], pxx[keep], alpha=alpha, lw=0.8)

            ax.set_title(ch_name, fontsize=10)
            ax.set_xlim(0, fmax)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=0.25)

            if r == 2:
                ax.set_xlabel("Frequency [Hz]")
            if c == 0:
                ax.set_ylabel("PSD [uV^2/Hz]")

    plt.tight_layout()
    plt.show()


def plot_role_median_notch_ci(psd_by_role, analysis_channels, row_channels, max_cols, role_key, role_label, fmax=20.0, line_lw=1.8, fill_alpha=0.22):
    ymax = _robust_ymax_from_notch(psd_by_role, analysis_channels, role_key, fmax)

    fig, axes = plt.subplots(3, max_cols, figsize=(3.2 * max_cols, 8.5), squeeze=False, sharex=True, sharey=True)
    fig.suptitle(
        f"EEG PSD median +/- 1.57 x IQR / sqrt(n) ({role_label})\\nWelch resolution ~0.5 Hz",
        y=1.02,
    )

    rows_local = ["F", "C", "P"]
    for r, row_name in enumerate(rows_local):
        chs = row_channels[r]
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(chs):
                ax.axis("off")
                continue

            ch_name = chs[c]
            traces = psd_by_role[role_key].get(ch_name, [])
            if not traces:
                ax.text(0.5, 0.5, f"{ch_name}\\n(no data)", ha="center", va="center", fontsize=9, transform=ax.transAxes)
                ax.set_xlim(0, fmax)
                ax.set_ylim(0, ymax)
                ax.grid(alpha=0.2)
                continue

            f_ref, mat = _stack_psd_on_common_freq_notch(traces, fmax)
            if f_ref is None or mat is None:
                ax.text(0.5, 0.5, f"{ch_name}\\n(insufficient data)", ha="center", va="center", fontsize=9, transform=ax.transAxes)
                ax.set_xlim(0, fmax)
                ax.set_ylim(0, ymax)
                ax.grid(alpha=0.2)
                continue

            med = np.nanmedian(mat, axis=0)
            q1 = np.nanpercentile(mat, 25, axis=0)
            q3 = np.nanpercentile(mat, 75, axis=0)
            iqr = q3 - q1
            n_eff = np.sum(np.isfinite(mat), axis=0)

            ci_half = np.divide(
                1.57 * iqr,
                np.sqrt(np.maximum(n_eff, 1)),
                out=np.full_like(iqr, np.nan),
                where=n_eff > 0,
            )
            lower = np.clip(med - ci_half, a_min=0, a_max=None)
            upper = med + ci_half

            valid = np.isfinite(med)
            ax.plot(f_ref[valid], med[valid], lw=line_lw)
            band = np.isfinite(lower) & np.isfinite(upper)
            if np.any(band):
                ax.fill_between(f_ref[band], lower[band], upper[band], alpha=fill_alpha, linewidth=0)

            ax.set_title(f"{ch_name} (n={len(traces)})", fontsize=10)
            ax.set_xlim(0, fmax)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=0.25)
            if r == 2:
                ax.set_xlabel("Frequency [Hz]")
            if c == 0:
                ax.set_ylabel("PSD [uV^2/Hz]")

    plt.tight_layout()
    plt.show()


def debug_plot_ffdtf_grid(freqs, ff_dtf, spectra, node_names, dyad_id, event, pair_type):
    ff_abs = np.abs(ff_dtf)
    sp_abs = np.abs(spectra)
    n_nodes = ff_abs.shape[0]

    max_off = np.nanmax(ff_abs) if np.isfinite(np.nanmax(ff_abs)) else 1.0
    max_diag = np.nanmax(np.diagonal(sp_abs, axis1=0, axis2=1))
    if not np.isfinite(max_off) or max_off <= 0:
        max_off = 1.0
    if not np.isfinite(max_diag) or max_diag <= 0:
        max_diag = 1.0

    fig, axs = plt.subplots(
        n_nodes,
        n_nodes,
        figsize=(max(8, n_nodes * 0.8), max(8, n_nodes * 0.8)),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    for i in range(n_nodes):
        for j in range(n_nodes):
            ax = axs[i, j] if n_nodes > 1 else axs
            if i != j:
                y = np.real(ff_abs[i, j, :])
                ax.plot(freqs, y, lw=0.5)
                ax.fill_between(freqs, y, 0, color="skyblue", alpha=0.4)
                ax.set_ylim([0, max_off])
            else:
                y = np.real(sp_abs[i, i, :])
                ax.plot(freqs, y, lw=0.5, color=[0.7, 0.7, 0.7])
                ax.fill_between(freqs, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, max_diag])

            ax.tick_params(
                labelleft=(j == 0),
                labelbottom=(i == n_nodes - 1),
                left=(j == 0),
                bottom=(i == n_nodes - 1),
                labelsize=4,
            )
            ax.tick_params(axis="x", labelsize=6)
            if i == n_nodes - 1:
                ax.set_xlabel(node_names[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(node_names[i], fontsize=11)

    fig.suptitle(
        f"ffDTF(freq) + spectra(diag) | {dyad_id} | {event} | {pair_type}",
        fontsize=17,
    )
    plt.tight_layout()
    plt.show()


def debug_plot_ffdtf_sum_matrix(ff_dtf_sum, node_names, use_channels, dyad_id, event, pair_type, only_cross=True, sum_freq_min=0.5, sum_freq_max=3.0):
    plot_mat = ff_dtf_sum.copy()
    np.fill_diagonal(plot_mat, np.nan)

    if only_cross:
        n_ch = len(use_channels)
        mask = np.zeros_like(plot_mat, dtype=bool)
        mask[:n_ch, n_ch:] = True
        mask[n_ch:, :n_ch] = True
        plot_mat = np.where(mask, plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(plot_mat, cmap="viridis", aspect="auto")
    ax.set_title(f"ffDTF sum ({sum_freq_min:g}-{sum_freq_max:g} Hz) | {dyad_id} | {event} | {pair_type}")
    ticks = np.arange(len(node_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(node_names, rotation=90, fontsize=6)
    ax.set_yticklabels(node_names, fontsize=6)
    ax.set_xlabel("Source channel")
    ax.set_ylabel("Target channel")
    fig.colorbar(im, ax=ax, label="Summed ffDTF")
    plt.tight_layout()
    plt.show()


def plot_h10_ecg_alignment(
    t_h10,
    ibi_cg_i,
    ibi_ch_i,
    ibi_cg_ecg,
    ibi_ch_ecg,
    lag_cg,
    lag_ch,
    fs_ibi,
    dyad_id,
    save_dir=None,
):
    """Plot full-range and overlap-zoomed H10-vs-ECG alignment diagnostics."""
    lag_cg_s = lag_cg / fs_ibi
    lag_ch_s = lag_ch / fs_ibi

    start_t = min(lag_ch, lag_cg)
    t_ecg = np.arange(start_t, start_t + len(ibi_ch_ecg)) / fs_ibi

    h10_start, h10_end = float(t_h10[0]), float(t_h10[-1])
    ecg_start, ecg_end = float(t_ecg[0]), float(t_ecg[-1])
    overlap_start = max(h10_start, ecg_start)
    overlap_end = min(h10_end, ecg_end)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), dpi=100)
    axes[0].plot(t_h10, ibi_cg_i, label="H10 CG (aligned)")
    axes[0].plot(t_ecg, ibi_cg_ecg, label="ECG CG", alpha=0.8)
    axes[1].plot(t_h10, ibi_ch_i, label="H10 CH (aligned)")
    axes[1].plot(t_ecg, ibi_ch_ecg, label="ECG CH", alpha=0.8)
    axes[0].set_ylabel("IBI [ms]")
    axes[1].set_ylabel("IBI [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"CG alignment (lag={lag_cg} samples, {lag_cg_s:+.2f} s)")
    axes[1].set_title(f"CH alignment (lag={lag_ch} samples, {lag_ch_s:+.2f} s)")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Alignment check: full time range", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{dyad_id}_alignment_fullrange.png"),
            bbox_inches="tight",
            dpi=100,
        )
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), dpi=100)
    axes[0].plot(t_h10, ibi_cg_i, label="H10 CG (aligned)")
    axes[0].plot(t_ecg, ibi_cg_ecg, label="ECG CG", alpha=0.8)
    axes[1].plot(t_h10, ibi_ch_i, label="H10 CH (aligned)")
    axes[1].plot(t_ecg, ibi_ch_ecg, label="ECG CH", alpha=0.8)

    if overlap_end > overlap_start:
        axes[0].set_xlim(overlap_start, overlap_end)
        axes[1].set_xlim(overlap_start, overlap_end)
        axes[0].autoscale_view(scalex=False, scaley=True)
        axes[1].autoscale_view(scalex=False, scaley=True)

    axes[0].set_ylabel("IBI [ms]")
    axes[1].set_ylabel("IBI [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"CG alignment (lag={lag_cg} samples, {lag_cg_s:+.2f} s)")
    axes[1].set_title(f"CH alignment (lag={lag_ch} samples, {lag_ch_s:+.2f} s)")
    axes[0].legend()
    axes[1].legend()

    if overlap_end > overlap_start:
        fig.suptitle(
            f"Alignment check: zoomed to overlap [{overlap_start:.2f}, {overlap_end:.2f}] s",
            y=1.02,
        )
    else:
        fig.suptitle(
            "Alignment check: no overlap between displayed H10 and ECG ranges",
            y=1.02,
        )

    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, f"{dyad_id}_alignment_zoomed.png"),
            bbox_inches="tight",
            dpi=100,
        )
    plt.show()
