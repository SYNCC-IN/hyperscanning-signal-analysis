import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.ffdtf_stats_helpers import _split_channel_pair, prepare_group_comparison_df


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


def _default_posthoc_palette(slide_palette=None):
    base = {
        "td": "#2F97A7",
        "asd": "#A7CF00",
        "teal_dark": "#1F6A78",
        "gold": "#E6BC34",
        "gold_dark": "#9F7D1C",
        "bg_alt": "#EAF2F4",
        "grid": "#C7D6DB",
        "neutral": "#7E8F95",
        "accent": "#2ca02c",
    }
    if slide_palette is not None:
        base.update(dict(slide_palette))
    return base


def plot_boxplots_for_posthoc(wide, events, title_prefix, slide_palette=None):
    palette = _default_posthoc_palette(slide_palette)
    event_pairs = [(events[i], events[j]) for i in range(len(events)) for j in range(i + 1, len(events))]
    td_w = wide.loc[wide["group_binary"] == "TD"].copy()
    asd_w = wide.loc[wide["group_binary"] == "ASD"].copy()

    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["grid.color"] = palette["grid"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=240, constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    colors = {"TD": palette["td"], "ASD": palette["asd"]}
    offsets = {"TD": -0.16, "ASD": 0.16}
    for i, event in enumerate(events, start=1):
        for group in ("TD", "ASD"):
            vals = wide.loc[wide["group_binary"] == group, event].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[i + offsets[group]],
                widths=0.28,
                notch=True,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(colors[group])
                box.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.2)

    ax.set_xticks(range(1, len(events) + 1))
    ax.set_xticklabels(events)
    ax.set_title("Notched Boxplots: ffDTF by Movie and Group", fontsize=16)
    ax.set_xlabel("Movie", fontsize=14)
    ax.set_ylabel("ffDTF", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors["TD"], lw=8, alpha=0.55, label="TD"),
            plt.Line2D([0], [0], color=colors["ASD"], lw=8, alpha=0.55, label="ASD"),
        ],
        frameon=False,
        loc="best",
        fontsize=12,
    )

    ax2 = axes[1]
    for i, (event_a, event_b) in enumerate(event_pairs, start=1):
        diff_td = (td_w[event_b] - td_w[event_a]).to_numpy(dtype=float)
        diff_asd = (asd_w[event_b] - asd_w[event_a]).to_numpy(dtype=float)
        diff_td = diff_td[np.isfinite(diff_td)]
        diff_asd = diff_asd[np.isfinite(diff_asd)]
        for group, vals, offset in (("TD", diff_td, -0.16), ("ASD", diff_asd, 0.16)):
            if len(vals) == 0:
                continue
            bp = ax2.boxplot(
                [vals],
                positions=[i + offset],
                widths=0.28,
                notch=True,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(colors[group])
                box.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.2)

    ax2.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax2.set_xticks(range(1, len(event_pairs) + 1))
    ax2.set_xticklabels([f"{event_b}-{event_a}" for event_a, event_b in event_pairs])
    ax2.set_title("Notched Boxplots: Interaction Contrasts (DffDTF)", fontsize=16)
    ax2.set_xlabel("Movie contrast", fontsize=14)
    ax2.set_ylabel("DffDTF within dyad", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(title_prefix, fontsize=16)
    plt.show()


def plot_three_panel_movie(movie_posthoc_df, title_prefix, slide_palette=None):
    if movie_posthoc_df is None or movie_posthoc_df.empty:
        return

    palette = _default_posthoc_palette(slide_palette)
    viz_df = movie_posthoc_df.copy()
    required_cols = {
        "contrast",
        "group",
        "n_dyads",
        "median_diff",
        "p_movie_posthoc",
        "q_movie_posthoc_bh",
    }
    if not required_cols.issubset(set(viz_df.columns)):
        return

    viz_df = viz_df[["contrast", "group", "n_dyads", "median_diff", "p_movie_posthoc", "q_movie_posthoc_bh"]].copy()
    viz_df["significance"] = np.where(
        viz_df["q_movie_posthoc_bh"] < 0.05,
        "sig",
        np.where(viz_df["q_movie_posthoc_bh"] < 0.10, "trend", "ns"),
    )

    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["grid.color"] = palette["grid"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=240, constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax1 = axes[0]
    contrast_order = list(dict.fromkeys(viz_df["contrast"].tolist()))
    y_lookup = {contrast: i for i, contrast in enumerate(contrast_order)}
    y_offsets = {"TD": -0.18, "ASD": 0.18}
    for _, row in viz_df.iterrows():
        group = row["group"]
        y_coord = y_lookup[row["contrast"]] + y_offsets.get(group, 0.0)
        x_coord = row["median_diff"]
        marker_size = 35 + 10 * max(int(row["n_dyads"]), 1)
        alpha = 0.95 if row["significance"] == "sig" else (0.75 if row["significance"] == "trend" else 0.55)
        ax1.scatter(
            x_coord,
            y_coord,
            s=marker_size,
            color=palette["td"] if group == "TD" else palette["asd"],
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        q_val = row["q_movie_posthoc_bh"]
        if np.isfinite(q_val):
            ax1.text(x_coord, y_coord + 0.05, f"q={q_val:.3f}", fontsize=8, ha="center", va="bottom")

    ax1.axvline(0.0, color=palette["neutral"], linewidth=1.4, alpha=0.8, zorder=2)
    ax1.set_yticks(range(len(contrast_order)))
    ax1.set_yticklabels(contrast_order)
    ax1.set_title("Diverging Dot Plot", fontsize=16)
    ax1.set_xlabel("Median difference (event2 - event1)", fontsize=13)
    ax1.set_ylabel("Contrast", fontsize=13)
    ax1.grid(axis="x", alpha=0.25, zorder=1)

    ax2 = axes[1]
    heatmap = viz_df.pivot(index="contrast", columns="group", values="median_diff").reindex(index=contrast_order)
    heatmap = heatmap.reindex(columns=["TD", "ASD"])
    vmax = np.nanmax(np.abs(heatmap.to_numpy(dtype=float))) if heatmap.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    image = ax2.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax2.set_yticks(range(len(heatmap.index)))
    ax2.set_yticklabels(heatmap.index)
    ax2.set_xticks(range(len(heatmap.columns)))
    ax2.set_xticklabels(heatmap.columns)
    ax2.set_title("Heatmap: Median Difference", fontsize=16)
    cbar = fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Median diff", fontsize=11)

    ax3 = axes[2]
    trend_df = viz_df.copy()
    trend_df["score"] = -np.log10(np.clip(trend_df["q_movie_posthoc_bh"].to_numpy(dtype=float), 1e-12, 1.0))
    y_labels = [f"{row['contrast']} | {row['group']}" for _, row in trend_df.iterrows()]
    y_values = np.arange(len(trend_df))
    bar_colors = [
        palette["gold"] if sig == "sig" else (palette["gold_dark"] if sig == "trend" else palette["neutral"])
        for sig in trend_df["significance"]
    ]
    ax3.barh(y_values, trend_df["score"].to_numpy(dtype=float), color=bar_colors, alpha=0.9)
    ax3.axvline(-np.log10(0.10), color=palette["gold_dark"], linestyle="--", linewidth=1.2)
    ax3.axvline(-np.log10(0.05), color=palette["gold"], linestyle="--", linewidth=1.2)
    ax3.set_yticks(y_values)
    ax3.set_yticklabels(y_labels, fontsize=9)
    ax3.set_xlabel("-log10(q)", fontsize=13)
    ax3.set_title("Trend/Significance Panel (q-based)", fontsize=16)
    ax3.grid(axis="x", alpha=0.25)

    fig.suptitle(title_prefix, fontsize=17)
    plt.show()


def plot_main_effect_boxplots(wide, events, title_prefix, effect="group", slide_palette=None):
    if wide is None or wide.empty:
        return

    palette = _default_posthoc_palette(slide_palette)
    events = [event for event in events if event in wide.columns]
    if not events:
        return

    dyad_col = "dyadID" if "dyadID" in wide.columns else ("dyad_id" if "dyad_id" in wide.columns else None)
    if dyad_col is None or "group_binary" not in wide.columns:
        print("  Skipping main-effect boxplot: missing dyad/group columns in aggregated wide table.")
        return

    long_df = wide[[dyad_col, "group_binary"] + events].melt(
        id_vars=[dyad_col, "group_binary"],
        value_vars=events,
        var_name="movie",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return

    if effect == "group":
        grp_df = long_df.groupby([dyad_col, "group_binary"], as_index=False)["value"].mean()
        td_vals = grp_df.loc[grp_df["group_binary"] == "TD", "value"].to_numpy(dtype=float)
        asd_vals = grp_df.loc[grp_df["group_binary"] == "ASD", "value"].to_numpy(dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220, constrained_layout=True)
        bp = ax.boxplot(
            [td_vals[np.isfinite(td_vals)], asd_vals[np.isfinite(asd_vals)]],
            tick_labels=["TD", "ASD"],
            notch=True,
            patch_artist=True,
            showfliers=False,
        )
        colors = [palette["td"], palette["asd"]]
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.25)
            box.set_edgecolor(color)
            box.set_linewidth(1.8)
        for med in bp["medians"]:
            med.set_color("#111111")
            med.set_linewidth(2.0)

        ax.set_title(f"{title_prefix} | Main effect: Group (dyad mean over movies)")
        ax.set_xlabel("Group")
        ax.set_ylabel("ffDTF")
        ax.grid(axis="y", alpha=0.22)
        plt.show()
        return

    if effect == "movie":
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), dpi=220, constrained_layout=True)
        movie_data = [long_df.loc[long_df["movie"] == event, "value"].to_numpy(dtype=float) for event in events]
        movie_data = [vals[np.isfinite(vals)] for vals in movie_data]
        bp = ax.boxplot(
            movie_data,
            tick_labels=events,
            notch=True,
            patch_artist=True,
            showfliers=False,
        )
        base_color = palette["accent"]
        for box in bp["boxes"]:
            box.set_facecolor(base_color)
            box.set_alpha(0.22)
            box.set_edgecolor(base_color)
            box.set_linewidth(1.8)
        for med in bp["medians"]:
            med.set_color("#111111")
            med.set_linewidth(2.0)

        ax.set_title(f"{title_prefix} | Main effect: Movie (pooled over groups)")
        ax.set_xlabel("Movie")
        ax.set_ylabel("ffDTF")
        ax.grid(axis="y", alpha=0.22)
        plt.show()


def plot_movie_effect(df, save_prefix="cg_ch_movie_effect", save=True):
    movie_order = ["Peppa", "Brave", "Incredibles"]
    palette = {
        "Peppa": "#7FA8C9",
        "Brave": "#E0A458",
        "Incredibles": "#8FBF9F",
    }
    accent = "#8A5A00"

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    data = df.copy()
    if not {"dyad_id", "movie", "ff_dtf"}.issubset(data.columns):
        rename_map = {}
        for source, target in (
            ("dyad", "dyad_id"),
            ("pair_id", "dyad_id"),
            ("film", "movie"),
            ("condition", "movie"),
            ("value", "ff_dtf"),
            ("ffdtf", "ff_dtf"),
        ):
            if source in data.columns and target not in data.columns:
                rename_map[source] = target
        if rename_map:
            data = data.rename(columns=rename_map)

    required = {"dyad_id", "movie", "ff_dtf"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.loc[data["movie"].isin(movie_order), ["dyad_id", "movie", "ff_dtf"]].copy()
    data["movie"] = pd.Categorical(data["movie"], categories=movie_order, ordered=True)
    data = data.groupby(["dyad_id", "movie"], as_index=False, observed=True)["ff_dtf"].mean()

    n_dyads = data["dyad_id"].nunique()
    wide_bp = data.loc[data["movie"].isin(["Peppa", "Brave"])].pivot(index="dyad_id", columns="movie", values="ff_dtf").dropna(subset=["Peppa", "Brave"])
    diff_bp = (wide_bp["Brave"] - wide_bp["Peppa"]).rename("diff")

    rng = np.random.default_rng(2026)
    boot_n = 5000
    if len(diff_bp) > 0:
        boot_medians = np.empty(boot_n, dtype=float)
        values = diff_bp.to_numpy()
        for i in range(boot_n):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_medians[i] = np.median(sample)
        ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
        med_diff = float(np.median(values))
    else:
        ci_low, ci_high, med_diff = np.nan, np.nan, np.nan

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [2.3, 1]},
        figsize=(10, 4.5),
    )

    grouped = [data.loc[data["movie"] == movie, "ff_dtf"].dropna().to_numpy() for movie in movie_order]
    positions = np.arange(1, len(movie_order) + 1)
    bp = ax_left.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.5},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    for patch, movie in zip(bp["boxes"], movie_order):
        patch.set_facecolor(palette[movie])
        patch.set_alpha(0.55)

    for pos, movie in zip(positions, movie_order):
        vals = data.loc[data["movie"] == movie, "ff_dtf"].to_numpy()
        x_jit = pos + rng.normal(0, 0.04, size=len(vals))
        ax_left.scatter(x_jit, vals, s=18, alpha=0.45, color="#4A4A4A", edgecolors="none", zorder=3)

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(movie_order)
    ax_left.set_xlabel("Movie")
    ax_left.set_ylabel("ffDTF (cg -> ch)")
    ax_left.grid(axis="y", alpha=0.25, linewidth=0.7)

    y_min, y_max = data["ff_dtf"].min(), data["ff_dtf"].max()
    y_span = y_max - y_min if y_max > y_min else 1.0
    bracket_y = y_max + 0.08 * y_span
    h = 0.03 * y_span
    x1, x2 = 1, 2
    ax_left.plot([x1, x1, x2, x2], [bracket_y, bracket_y + h, bracket_y + h, bracket_y], color=accent, lw=1.6)
    ax_left.text((x1 + x2) / 2, bracket_y + h + 0.01 * y_span, "q = 0.021 *", ha="center", va="bottom", color=accent, fontsize=10)

    ax_left.set_ylim(y_min - 0.06 * y_span, y_max + 0.22 * y_span)
    ax_left.text(0.01, 0.98, f"n = {n_dyads}", transform=ax_left.transAxes, ha="left", va="top", fontsize=9, color="#444444")

    x0 = np.zeros(len(diff_bp), dtype=float)
    jitter = rng.normal(0, 0.035, size=len(diff_bp)) if len(diff_bp) > 0 else np.array([])
    ax_right.scatter(x0 + jitter, diff_bp.to_numpy(), s=18, alpha=0.5, color="#5A5A5A", edgecolors="none", zorder=3)
    ax_right.axhline(0.0, color="#8A8A8A", linestyle="--", linewidth=1.0, zorder=1)

    if np.isfinite(med_diff):
        ax_right.vlines(0.0, ci_low, ci_high, color="#222222", linewidth=1.2, zorder=4)
        ax_right.hlines([ci_low, ci_high], -0.05, 0.05, color="#222222", linewidth=1.0, zorder=4)
        ax_right.hlines(med_diff, -0.14, 0.14, color="#111111", linewidth=2.8, zorder=5)

    ax_right.set_xlim(-0.28, 0.28)
    ax_right.set_xticks([])
    ax_right.set_xlabel("")
    ax_right.set_ylabel("Brave - Peppa")
    ax_right.set_title("Brave - Peppa (per dyad)", fontsize=11)
    ax_right.text(0.5, 0.98, "q = 0.021 *", transform=ax_right.transAxes, ha="center", va="top", color=accent, fontsize=10)
    ax_right.grid(axis="y", alpha=0.25, linewidth=0.7)

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("cg -> ch: movie effect (pooled over groups, paired)", fontsize=12, y=1.02)
    fig.text(
        0.01,
        -0.02,
        "Node: Pz, V3 outgoing. Within-dyad repeated measures across Peppa, Brave, Incredibles.",
        ha="left",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )

    plt.tight_layout()
    if save:
        fig.savefig(f"{save_prefix}.svg", bbox_inches="tight")
        fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")

    return fig, (ax_left, ax_right)


def _node_order_from_analysis_channels(node_names, preferred_channels=None):
    preferred = [] if preferred_channels is None else [str(ch) for ch in preferred_channels]
    idx = {ch: i for i, ch in enumerate(preferred)}

    def _key(node):
        s = str(node)
        if ":" in s:
            role, ch = s.split(":", 1)
            role_rank = 0 if role == "ch" else (1 if role == "cg" else 2)
            in_pref = 0 if ch in idx else 1
            ch_rank = idx.get(ch, 10**9)
            return (role_rank, in_pref, ch_rank, ch, s)

        in_pref = 0 if s in idx else 1
        ch_rank = idx.get(s, 10**9)
        return (2, in_pref, ch_rank, s, s)

    return sorted(node_names, key=_key)


def _add_half_violin(ax, values, side="left", color="#2F97A7", alpha=0.65):
    if values is None or len(values) < 2:
        return

    vp = ax.violinplot([values], positions=[0], widths=0.9, showmeans=False, showmedians=False, showextrema=False)
    body = vp["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_linewidth(1.0)
    body.set_alpha(alpha)

    if side == "left":
        clip = Rectangle((-1.0, -1e9), 1.0, 2e9, transform=ax.transData)
    else:
        clip = Rectangle((0.0, -1e9), 1.0, 2e9, transform=ax.transData)
    body.set_clip_path(clip)


def _infer_aggregate_id_col(df_input):
    for candidate in ["dyadID", "surrogate_pair_id"]:
        if candidate in df_input.columns:
            return candidate
    return None


def _plot_split_violin_single(
    vis_df,
    group_col,
    groups,
    value_col="ff_dtf",
    channel_col="channel_pair",
    preferred_channels=None,
    left_color="#2F97A7",
    right_color="#A7CF00",
    title_prefix="Envelope ffDTF distributions per edge (cross-brain)",
    panel_suffix="",
):
    g1, g2 = groups
    if vis_df is None or vis_df.empty:
        return

    vis_df = vis_df.copy()
    vis_df[["src", "dst"]] = vis_df[channel_col].apply(lambda s: pd.Series(_split_channel_pair(s)))

    node_order = _node_order_from_analysis_channels(
        set(vis_df["src"]).union(set(vis_df["dst"])), preferred_channels=preferred_channels
    )
    n = len(node_order)
    if n == 0:
        return

    idx = {name: i for i, name in enumerate(node_order)}
    grouped = vis_df.groupby(["src", "dst", group_col])[value_col]
    g1_map = {}
    g2_map = {}
    for (src, dst, grp), vals in grouped:
        key = (idx[dst], idx[src])
        if grp == g1:
            g1_map[key] = vals.to_numpy()
        elif grp == g2:
            g2_map[key] = vals.to_numpy()

    y_vals = vis_df[value_col].to_numpy()
    y_max = np.nanpercentile(y_vals, 99.5) if len(y_vals) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = np.nanmax(y_vals) if len(y_vals) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    fig, axs = plt.subplots(
        n,
        n,
        figsize=(max(12, n * 0.72), max(12, n * 0.72)),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            ax.set_facecolor("white")
            key = (i, j)

            _add_half_violin(ax, g1_map.get(key), side="left", color=left_color, alpha=0.70)
            _add_half_violin(ax, g2_map.get(key), side="right", color=right_color, alpha=0.70)

            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(0, y_max)
            ax.set_xticks([])

            if j == 0:
                ax.tick_params(axis="y", labelsize=10, length=2)
            else:
                ax.set_yticks([])

            if i == n - 1:
                ax.set_xlabel(f"from: {node_order[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"to: {node_order[i]}", fontsize=10)

    fig.suptitle(
        f"{title_prefix}{panel_suffix}\nDirection: from column/source to row/target (column -> row)\nleft half = {g1}, right half = {g2}",
        fontsize=17,
    )
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=left_color, edgecolor="black", label=str(g1)),
        Rectangle((0, 0), 1, 1, facecolor=right_color, edgecolor="black", label=str(g2)),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_split_violin_grid(
    df_input,
    group_col,
    groups,
    target_events,
    value_col="ff_dtf",
    channel_col="channel_pair",
    event_col="event",
    filter_cross_brain=True,
    selected_channels=None,
    only_homologous=False,
    preferred_channels=None,
    left_color="#2F97A7",
    right_color="#A7CF00",
    title_prefix="Envelope ffDTF distributions per edge (cross-brain)",
    iterate_over_events=True,
    aggregate_over_events=False,
    aggregate_id_col=None,
    aggregate_func="mean",
):
    vis_df_all = prepare_group_comparison_df(
        df_input,
        group_col=group_col,
        groups=groups,
        value_col=value_col,
        event_col=event_col,
        channel_col=channel_col,
        filter_cross_brain=filter_cross_brain,
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )

    if vis_df_all.empty:
        print("Split-violin: no rows available after filtering.")
        return

    if not iterate_over_events and not aggregate_over_events:
        print("Split-violin: nothing to plot (set iterate_over_events and/or aggregate_over_events to True).")
        return

    if aggregate_over_events:
        id_col = aggregate_id_col if aggregate_id_col is not None else _infer_aggregate_id_col(vis_df_all)
        if id_col is None:
            raise ValueError(
                "aggregate_over_events=True requires aggregate_id_col, and no default id column was found."
            )

        agg_df = (
            vis_df_all.groupby([id_col, group_col, channel_col], as_index=False)[value_col]
            .agg(aggregate_func)
            .rename(columns={value_col: "__plot_value"})
        )

        _plot_split_violin_single(
            agg_df.rename(columns={"__plot_value": value_col}),
            group_col=group_col,
            groups=groups,
            value_col=value_col,
            channel_col=channel_col,
            preferred_channels=preferred_channels,
            left_color=left_color,
            right_color=right_color,
            title_prefix=title_prefix,
            panel_suffix=f", events aggregated ({aggregate_func})",
        )

    if iterate_over_events:
        event_order = [ev for ev in target_events if ev in set(vis_df_all[event_col].unique())]
        if len(event_order) == 0:
            print("Split-violin: no matching target events in dataframe.")
            return

        for ev in event_order:
            vis_df = vis_df_all.loc[vis_df_all[event_col] == ev].copy()
            if vis_df.empty:
                continue

            _plot_split_violin_single(
                vis_df,
                group_col=group_col,
                groups=groups,
                value_col=value_col,
                channel_col=channel_col,
                preferred_channels=preferred_channels,
                left_color=left_color,
                right_color=right_color,
                title_prefix=title_prefix,
                panel_suffix=f", event={ev}",
            )
