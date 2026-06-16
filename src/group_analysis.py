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

    metrics = ["ff_dtf_g", "spectra_global", "ff_dtf_w", "spectra_windowed"]
    groups = ["TD", "ASD", "P", "ASD+P", "P, możliwe ASD"]
    sexes = ["B", "G"]

    stacked_by_group = {g: {} for g in groups}
    stacked_by_sex = {s: {} for s in sexes}

    for file in files:
        with np.load(file, allow_pickle=True) as data:
            matrices = {
                "ff_dtf_g": data["ff_dtf_global"],
                "spectra_global": data["spectra_global"],
                "ff_dtf_w": data["ff_dtf_windowed"],
                "spectra_windowed": data["spectra_windowed"]
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


def get_means(data1, data2, fs):
    """
    Calculates the mean values (Spectra on diag, DTF off-diag), averaged over 
    frequencies up to 1 Hz (subject-level) and then averaged across subjects (group-level).
    """
    idx_1hz = data1.shape[-1] // fs
    
    subj_mean_td = np.mean(data1[..., :idx_1hz], axis=-1)
    subj_mean_asd = np.mean(data2[..., :idx_1hz], axis=-1)

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
    Draws three heatmaps side by side: m1, m2, and their difference (m1 - m2).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title0, fontsize=16)
    heatmap_kwargs = {'annot': True, 'fmt': '.4f', 'xticklabels': channel_names, 'yticklabels': channel_names}

    sns.heatmap(m1, ax=axes[0], cmap='viridis', **heatmap_kwargs)
    axes[0].set_title(title1)

    sns.heatmap(m2, ax=axes[1], cmap='viridis', **heatmap_kwargs)
    axes[1].set_title(title2)

    diff = m1 - m2
    sns.heatmap(diff, ax=axes[2], cmap='coolwarm', center=0, **heatmap_kwargs)
    axes[2].set_title(title3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_windowed_heatmaps(mw1, mw2, channel_names, title0, title1, title2, title3, save_path=None):
    """
    Draws a grid of heatmaps for the windowed data (3, 4, 4).
    Rows: Time windows (time flows downwards) | Columns: Group 1, Group 2, Diff (mw1 - mw2)
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(title0, fontsize=16)
    heatmap_kwargs = {'annot': True, 'fmt': '.4f', 'xticklabels': channel_names, 'yticklabels': channel_names}
    
    for w in range(3):
        # Column 1 (mw1)
        sns.heatmap(mw1[w], ax=axes[w, 0], cmap='viridis', cbar=False, **heatmap_kwargs)
        axes[w, 0].set_ylabel(f"Window {w+1}", fontsize=12, fontweight='bold')
        if w == 0: axes[0, 0].set_title(title1, fontsize=14) # Title only in the first row
        
        # Column 2 (mw2)
        sns.heatmap(mw2[w], ax=axes[w, 1], cmap='viridis', cbar=False, **heatmap_kwargs)
        if w == 0: axes[0, 1].set_title(title2, fontsize=14)
        
        # Column 3 (mw1 - mw2)
        diff = mw1[w] - mw2[w]
        sns.heatmap(diff, ax=axes[w, 2], cmap='coolwarm', center=0, cbar=True, **heatmap_kwargs)
        if w == 0: axes[0, 2].set_title(title3, fontsize=14)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_significance_stars(p_value):
    """Returns the appropriate number of stars based on the p-value."""
    if p_value < 0.001: return '***'
    elif p_value < 0.01: return '**'
    elif p_value < 0.05: return '*'
    else: return 'ns' # not significant


def plot_violins_with_stats(data1_subj, data2_subj, cond1, cond2, channel_names, title, save_path=None):
    """
    Draws a 4x4 grid of violin plots alongside Mann-Whitney test results.
    Expects arrays with the subject dimension preserved, e.g., (29, 4, 4) and (18, 4, 4).
    """
    n_chan = data1_subj.shape[1]
    fig, axes = plt.subplots(n_chan, n_chan, figsize=(18, 18))
    fig.suptitle(title, fontsize=20)
    
    palette = {cond1: "#4C72B0", cond2: "#DD8452"}

    for i in range(n_chan):
        for j in range(n_chan):
            ax = axes[i, j]
            vals1 = data1_subj[:, i, j]
            vals2 = data2_subj[:, i, j]
            
            # Statistical Mann-Whitney test 
            stat, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
            stars = get_significance_stars(p_val)
            
            df1 = pd.DataFrame({'Value': vals1, 'Group': cond1})
            df2 = pd.DataFrame({'Value': vals2, 'Group': cond2})
            df_combined = pd.concat([df1, df2], ignore_index=True)
            
            # Draw the violin plot
            sns.violinplot(
                data=df_combined, x='Group', y='Value', ax=ax,
                palette=palette, inner="quartile", hue='Group', legend=False)
            
            conn_type = "Spectra" if i == j else "ffDTF"
            title_text = f"{channel_names[i]} $\\rightarrow$ {channel_names[j]}\n{conn_type} | p={p_val:.4f}"
            
            # Bold the title if the result is statistically significant
            weight = 'bold' if p_val < 0.05 else 'normal'
            ax.set_title(title_text, fontsize=12, fontweight=weight)
            ax.set_xlabel("")
            
            if j != 0: ax.set_ylabel("") 
                
            # Draw significance stars and lines if significant
            if p_val < 0.05:
                # Calculate where to draw the line
                y_max = df_combined['Value'].max()
                y_range = y_max - df_combined['Value'].min()
                h = y_range * 0.05 # Height of the vertical ticks
                
                # X-coordinates for groups
                x1, x2 = 0, 1
                y_line = y_max + h
                
                # Draw the horizontal line with small vertical ticks
                ax.plot([x1, x1, x2, x2], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
                ax.text((x1+x2)*.5, y_line+h, stars, ha='center', va='bottom', color='k', fontsize=16, fontweight='bold')
                ax.set_ylim(bottom=ax.get_ylim()[0], top=y_line + 4*h)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def run_group_analysis(stacked_data, cond1, cond2, films, channel_names, fs, output_base_dir="group_analysis_results"):
    """
    Runs the full analysis pipeline (Global and Windowed) for a list of films.
    Saves heatmaps and violin plots into structured directories.
    """
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for film in films:
        print(f"[INFO] Processing film: {film}")
        film_dir = base_dir / f"{film}_{cond1}_vs_{cond2}"
        film_dir.mkdir(exist_ok=True)
        
        # Data extraction for both groups and both types
        dtf1_g = stacked_data[cond1][film]["ff_dtf_g"]
        spec1_g = stacked_data[cond1][film]["spectra_global"]
        combined1_g = combine_dtf_and_spectra(dtf1_g, spec1_g, channel_names)
        
        dtf2_g = stacked_data[cond2][film]["ff_dtf_g"]
        spec2_g = stacked_data[cond2][film]["spectra_global"]
        combined2_g = combine_dtf_and_spectra(dtf2_g, spec2_g, channel_names)
        
        # Windowed
        dtf1_w = stacked_data[cond1][film]["ff_dtf_w"]
        spec1_w = stacked_data[cond1][film]["spectra_windowed"]
        combined1_w = combine_dtf_and_spectra(dtf1_w, spec1_w, channel_names)
        
        dtf2_w = stacked_data[cond2][film]["ff_dtf_w"]
        spec2_w = stacked_data[cond2][film]["spectra_windowed"]
        combined2_w = combine_dtf_and_spectra(dtf2_w, spec2_w, channel_names)
        
        # Means Calculation
        global1_mean, global2_mean, global1_subj, global2_subj = get_means(combined1_g, combined2_g, fs)
        windowed1_mean, windowed2_mean, windowed1_subj, windowed2_subj = get_means(combined1_w, combined2_w, fs)
        
        # Zero Diagonals for Heatmaps
        global1_to_plot = zero_diagonal(global1_mean)
        global2_to_plot = zero_diagonal(global2_mean)
        windowed1_to_plot = zero_diagonal(windowed1_mean)
        windowed2_to_plot = zero_diagonal(windowed2_mean)
        
        # Plotting and Saving Results
        plot_global_heatmaps(
            global1_to_plot, global2_to_plot, channel_names, 
            title0=f"Film: {film}", title1=f"{cond1} Group", title2=f"{cond2} Group", title3=f"Diff ({cond1} - {cond2})",
            save_path=film_dir / "01_Heatmaps_Global.png"
        )
        
        plot_windowed_heatmaps(
            windowed1_to_plot, windowed2_to_plot, channel_names, 
            title0=f"Film: {film}", title1=f"{cond1} Group", title2=f"{cond2} Group", title3=f"Diff ({cond1} - {cond2})",
            save_path=film_dir / "02_Heatmaps_Windowed.png"
        )
        
        plot_violins_with_stats(
            global1_subj, global2_subj, cond1, cond2, channel_names, 
            title=f"Film: {film} - Global ffDTF and Spectra Stats",
            save_path=film_dir / "03_Violins_Global.png"
        )
        
        num_windows = windowed1_subj.shape[1]
        for w in range(num_windows):
            plot_violins_with_stats(
                windowed1_subj[:, w, :, :], windowed2_subj[:, w, :, :], cond1, cond2, channel_names, 
                title=f"Film: {film} {cond1} - Window {w+1} ffDTF and Spectra Stats",
                save_path=film_dir / f"04_Violins_Window_{w+1}.png"
            )
            
        print(f"[INFO] Saved all plots for {film} in {film_dir}\n")