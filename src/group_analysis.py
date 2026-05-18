import json
from pathlib import Path
import numpy as np

def load_and_aggregate_data(data_dir: str):
    """
    Searches the specified directory for .npz files containing ffDTF results
    and aggregates the matrices.

    Returns:
        tuple: (stacked_by_group, stacked_by_sex) containing aggregated matrices.
    """
    base_path = Path(data_dir)
    files = list(base_path.rglob("*.npz"))

    if not files:
        return None, None

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

    return stacked_by_group, stacked_by_sex