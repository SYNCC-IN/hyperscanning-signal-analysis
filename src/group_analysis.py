import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

def load_and_aggregate_data(data_dir: str):
    """
    Searches the specified directory for .npz files containing ffDTF results
    and aggregates the matrices.

    Returns:
        stacked_by_group: agregated matrices organized by group and film
        stacked_by_sex: agregated matrices organized by sex and film
        fs: sampling frequency
        channel_names: list of channel names
    """
    base_path = Path(data_dir)
    files = list(base_path.rglob("*.npz"))

    if not files:
        return None, None, None, None

    metrics = [
        "ff_dtf_g", "spectra_global",
        "ff_dtf_rev", "spectra_reversed",
        "ff_dtf_noise", "spectra_noise10pct"
    ]
    groups = ["TD", "ASD", "P", "ASD+P", "P, możliwe ASD"]
    sexes = ["B", "G"]

    stacked_by_group = {g: {} for g in groups}
    stacked_by_sex = {s: {} for s in sexes}

    for file in files:
        with np.load(file, allow_pickle=True) as data:
            matrices = {
                "ff_dtf_g": data["ff_dtf_global"],
                "spectra_global": data["spectra_global"],
                "ff_dtf_rev": data["ff_dtf_reversed"],
                "spectra_reversed": data["spectra_reversed"],
                "ff_dtf_noise": data["ff_dtf_noise10pct"],
                "spectra_noise10pct": data["spectra_noise10pct"],
            }
    groups = ["TD", "ASD", "P", "ASD+P", "P, możliwe ASD"]
    sexes = ["B", "G"]

    stacked_by_group = {g: {} for g in groups}
    stacked_by_sex = {s: {} for s in sexes}

    for file in files:
        with np.load(file, allow_pickle=True) as data:
            matrices = {
                "ff_dtf_g": data["ff_dtf_global"],
                "spectra_global": data["spectra_global"],
                "ff_dtf_rev": data["ff_dtf_reversed"],
                "spectra_reversed": data["spectra_reversed"],
                "ff_dtf_noise": data["ff_dtf_noise10pct"],
                "spectra_noise10pct": data["spectra_noise10pct"],
            }

            meta = json.loads(data["meta"].item())
            film = meta["film"]
            group = meta["child_info"]["group"]
            sex = meta["child_info"]["sex"]
            fs = int(meta['fs'])
            channel_names = meta['chan_names']

            if group in stacked_by_group:
                if film not in stacked_by_group[group]:
                    stacked_by_group[group][film] = {m: [] for m in metrics}
                for key in metrics:
                    stacked_by_group[group][film][key].append(matrices[key])

            if sex in stacked_by_sex:
                if film not in stacked_by_sex[sex]:
                    stacked_by_sex[sex][film] = {m: [] for m in metrics}
                for key in metrics:
                    stacked_by_sex[sex][film][key].append(matrices[key])

    # Stacking matrices
    for g in groups:
        for f in list(stacked_by_group[g].keys()):
            for k in metrics:
                lst = stacked_by_group[g][f][k]
                stacked_by_group[g][f][k] = np.stack(lst, axis=0) if lst else np.array([])

    for s in sexes:
        for f in list(stacked_by_sex[s].keys()):
            for k in metrics:
                lst = stacked_by_sex[s][f][k]
                stacked_by_sex[s][f][k] = np.stack(lst, axis=0) if lst else np.array([])

    if stacked_by_group and stacked_by_sex:
        films_in_groups = sorted({film for films in stacked_by_group.values() for film in films})
        films_in_sex = sorted({film for films in stacked_by_sex.values() for film in films})

        for film in films_in_groups:
            for group in stacked_by_group:
                if film in stacked_by_group[group]:
                    for metric, matrix in stacked_by_group[group][film].items():
                        print(f" Group: {group:<15} | Film: {film:<12} | {metric:<16} shape: {matrix.shape}")
        print("\n")
        for film in films_in_sex:
            for sex in stacked_by_sex:
                if film in stacked_by_sex[sex]:
                    for metric, matrix in stacked_by_sex[sex][film].items():
                        print(f" Sex: {sex:<17} | Film: {film:<12} | {metric:<16} shape: {matrix.shape}")

    return stacked_by_group, stacked_by_sex, fs, channel_names


def combine_dtf_and_spectra(dtf_data, spectra_data, channel_names):
    combined = np.copy(dtf_data)
    n_chan = len(channel_names)
    idx = np.arange(n_chan)
    combined[..., idx, idx, :] = np.real(spectra_data[..., idx, idx, :])
    return combined


def get_means(data1, data2, f_min=0.15, f_max=1.0, freq_start=0.01, freq_step=0.01):
    """
    Calculates the mean values (Spectra on diag, DTF off-diag), averaged over 
    the specified frequency band [f_min, f_max] and then averaged across subjects.
    """
    n_bins = data1.shape[-1]
    freqs = np.round(freq_start + np.arange(n_bins) * freq_step, 6)
    
    # Boolean mask covering the [0.15, 1.00] Hz band (86 bins: 0.15 to 1.00 Hz)
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    
    subj_mean_td = np.mean(data1[..., freq_mask], axis=-1)
    subj_mean_asd = np.mean(data2[..., freq_mask], axis=-1)

    group1_mean = np.mean(subj_mean_td, axis=0)
    group2_mean = np.mean(subj_mean_asd, axis=0)
    
    return group1_mean, group2_mean, subj_mean_td, subj_mean_asd


def zero_diagonal(matrix):
    res = np.copy(matrix)
    n_chan = res.shape[-1]
    idx = np.arange(n_chan)
    res[..., idx, idx] = 0
    return res


def plot_global_heatmaps(m1, m2, channel_names, title0, title1, title2, title3, save_path=None):
    """
    Plots a grid of 3 heatmaps: Group 1, Group 2, and their difference.
    X-axis: Source | Y-axis: Target
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title0, fontsize=16)
    heatmap_kwargs = {'annot': True, 'fmt': '.4f', 'xticklabels': channel_names, 'yticklabels': channel_names}

    sns.heatmap(m1, ax=axes[0], cmap='viridis', **heatmap_kwargs)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Source", fontweight='bold')
    axes[0].set_ylabel("Target", fontweight='bold')

    sns.heatmap(m2, ax=axes[1], cmap='viridis', **heatmap_kwargs)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Source", fontweight='bold')
    axes[1].set_ylabel("Target", fontweight='bold')

    diff = m1 - m2
    sns.heatmap(diff, ax=axes[2], cmap='coolwarm', center=0, **heatmap_kwargs)
    axes[2].set_title(title3)
    axes[2].set_xlabel("Source", fontweight='bold')
    axes[2].set_ylabel("Target", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_significance_stars(p_value):
    """Returns the appropriate number of stars based on the p-value."""
    if p_value < 0.001: return '***'
    elif p_value < 0.01: return '**'
    elif p_value < 0.05: return '*'
    else: return 'ns'


def plot_violins_with_stats(data1_subj, data2_subj, cond1, cond2, channel_names, title, save_path=None):
    """
    Grid of violin plots for each source-target pair, with Mann-Whitney U test results.
    Uses tight uniform Y-axis scales separately for Spectra (diagonal) and ffDTF (off-diagonal)
    """
    n_chan = data1_subj.shape[1]
    fig, axes = plt.subplots(n_chan, n_chan, figsize=(18, 18))
    fig.suptitle(title, fontsize=20)
    palette = {cond1: "#4C72B0", cond2: "#DD8452"}

    SPECTRA_YLIM = (-0.05, 1.35)
    SPECTRA_TICKS = [0.0, 0.4, 0.8, 1.2]

    FFDTF_YLIM = (-0.0005, 0.013)
    FFDTF_TICKS = [0.0, 0.005, 0.010]

    for src in range(n_chan):
        for tgt in range(n_chan):
            ax = axes[tgt, src]
            
            vals1 = data1_subj[:, tgt, src]
            vals2 = data2_subj[:, tgt, src]
            
            stat, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
            stars = get_significance_stars(p_val)
            
            df1 = pd.DataFrame({'Value': vals1, 'Group': cond1})
            df2 = pd.DataFrame({'Value': vals2, 'Group': cond2})
            df_combined = pd.concat([df1, df2], ignore_index=True)
            
            sns.violinplot(
                data=df_combined, x='Group', y='Value', ax=ax,
                palette=palette, inner="quartile", hue='Group', legend=False,
                cut=0, width=0.85
            )
            
            conn_type = "Spectra" if src == tgt else "ffDTF"
            title_text = f"{channel_names[src]} $\\rightarrow$ {channel_names[tgt]}\n{conn_type} | p={p_val:.4f}"
            
            weight = 'bold' if p_val < 0.05 else 'normal'
            ax.set_title(title_text, fontsize=11, fontweight=weight)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if src == tgt:
                ax.set_ylim(SPECTRA_YLIM)
                ax.set_yticks(SPECTRA_TICKS)
            else:
                ax.set_ylim(FFDTF_YLIM)
                ax.set_yticks(FFDTF_TICKS)

            ax.tick_params(axis='y', left=True, labelleft=False)
                
            if p_val < 0.05:
                y_max = df_combined['Value'].max()
                y_lim_top = SPECTRA_YLIM[1] if src == tgt else FFDTF_YLIM[1]
                y_lim_bot = SPECTRA_YLIM[0] if src == tgt else FFDTF_YLIM[0]
                y_range = y_lim_top - y_lim_bot
                h = y_range * 0.035
                
                x1, x2 = 0, 1
                y_line = min(y_max + h, y_lim_top - 2.5 * h)
                ax.plot([x1, x1, x2, x2], [y_line, y_line + h, y_line + h, y_line], lw=1.5, c='k')
                ax.text((x1 + x2) * 0.5, y_line + h, stars, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    axes[0, 0].tick_params(axis='y', labelleft=True)
    axes[0, 0].set_ylabel("Spectra (mean 0.15–1 Hz)", fontweight='bold')

    axes[-1, 0].tick_params(axis='y', labelleft=True)
    axes[-1, 0].set_ylabel("ffDTF (mean 0.15–1 Hz)", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def run_group_analysis(stacked_data, cond1, cond2, films, channel_names, fs, output_base_dir="group_analysis_results"):
    """
    Runs the full analysis pipeline for Global, Time-Reversed, and 10%-Noise ffDTF variants.
    Saves heatmaps and violin plots into structured directories.
    """
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("ff_dtf_g",     "spectra_global",     "01_Global",     "Global"),
        ("ff_dtf_rev",   "spectra_reversed",   "02_Reversed",   "Time-Reversed"),
        ("ff_dtf_noise", "spectra_noise10pct", "03_Noise10pct", "10% Noise Signal"),
    ]
    
    for film in films:
        print(f"[INFO] Processing film: {film}")
        film_dir = base_dir / f"{film}_{cond1}_vs_{cond2}"
        film_dir.mkdir(exist_ok=True)
        
        for dtf_key, spec_key, file_prefix, title_label in variants:
            dtf1 = stacked_data[cond1][film][dtf_key]
            spec1 = stacked_data[cond1][film][spec_key]
            combined1 = combine_dtf_and_spectra(dtf1, spec1, channel_names)
            
            dtf2 = stacked_data[cond2][film][dtf_key]
            spec2 = stacked_data[cond2][film][spec_key]
            combined2 = combine_dtf_and_spectra(dtf2, spec2, channel_names)

            # calculate means for both groups (Spectra on diag, ffDTF off-diag)
            mean1, mean2, subj1, subj2 = get_means(combined1, combined2, f_min=0.15, f_max=1.0)

            # Zero out the diagonal for plotting purposes
            m1_to_plot = zero_diagonal(mean1)
            m2_to_plot = zero_diagonal(mean2)

            # save heatmaps
            plot_global_heatmaps(
                m1_to_plot, m2_to_plot, channel_names, 
                title0=f"Film: {film} ({title_label})",
                title1=f"{cond1} Group",
                title2=f"{cond2} Group",
                title3=f"Diff ({cond1} - {cond2})",
                save_path=film_dir / f"{file_prefix}_{film}_Heatmaps.png"
            )
            
            # save violin plots
            plot_violins_with_stats(
                subj1, subj2, cond1, cond2, channel_names, 
                title=f"Film: {film} - {title_label} ffDTF and Spectra Stats",
                save_path=film_dir / f"{file_prefix}_{film}_Violins.png"
            )
            
        print(f"[INFO] Saved all plots (Global, Reversed, Noised) for {film} in {film_dir}\n")