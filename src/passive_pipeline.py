import json

import numpy as np
import pandas as pd

from .passive_io_helpers import build_role_lookup, discover_role_files, pairs_from_lookup


def build_real_and_surrogate_pairs(
    cleaned_signals_folder,
    target_events,
    signal_type="EEG",
    valid_dyads=None,
    process_only_smoke_dyads=False,
    smoke_max_real_dyads=4,
    include_surrogates=True,
    surrogate_random_seed=42,
    surrogate_use_all=False,
    surrogate_subset_size=50,
):
    role_files = discover_role_files(
        cleaned_signals_folder,
        target_events,
        signal_type=signal_type,
        glob_pattern="*_cleaned.nc",
        cleaned=True,
    )
    pair_lookup = build_role_lookup(role_files)
    real_pairs_all = pairs_from_lookup(pair_lookup)

    if valid_dyads is not None and len(valid_dyads) > 0:
        valid_set = set(valid_dyads)
        real_pairs_all = [x for x in real_pairs_all if x[0] in valid_set]

    if process_only_smoke_dyads:
        keep_dyads = sorted({d for d, _, _, _ in real_pairs_all})[:smoke_max_real_dyads]
        keep_set = set(keep_dyads)
        real_pairs_all = [x for x in real_pairs_all if x[0] in keep_set]

    if not real_pairs_all:
        raise RuntimeError("No real cleaned dyad/event pairs available for envelope ffDTF.")

    rng = np.random.default_rng(surrogate_random_seed)
    dyads_real = sorted({d for d, _, _, _ in real_pairs_all})
    all_surrogate_dyads = [
        (dyad_cg, dyad_ch)
        for dyad_cg in dyads_real
        for dyad_ch in dyads_real
        if dyad_cg != dyad_ch
    ]

    selected_surrogate_dyads = []
    if include_surrogates and all_surrogate_dyads:
        if surrogate_use_all:
            selected_surrogate_dyads = list(all_surrogate_dyads)
        else:
            n_pick = min(int(surrogate_subset_size), len(all_surrogate_dyads))
            idx = rng.choice(len(all_surrogate_dyads), size=n_pick, replace=False)
            selected_surrogate_dyads = [all_surrogate_dyads[i] for i in idx]

    surrogate_pairs = []
    for dyad_cg, dyad_ch in selected_surrogate_dyads:
        surrogate_id = f"S_{dyad_cg}_cg__{dyad_ch}_ch"
        for event in target_events:
            key_ch = (dyad_ch, event)
            key_cg = (dyad_cg, event)
            if key_ch not in pair_lookup or key_cg not in pair_lookup:
                continue
            roles_ch = pair_lookup[key_ch]
            roles_cg = pair_lookup[key_cg]
            if ("ch" in roles_ch) and ("cg" in roles_cg):
                surrogate_pairs.append((surrogate_id, event, roles_ch["ch"], roles_cg["cg"], dyad_cg, dyad_ch))

    return real_pairs_all, surrogate_pairs, selected_surrogate_dyads


def run_ffdtf_batch(real_pairs_all, surrogate_pairs, compute_pair_fn):
    all_rows = []
    debug_rows = []
    failed = []

    for dyad_id, event, path_ch, path_cg in real_pairs_all:
        print(f"Processing REAL {dyad_id} | {event}")
        try:
            rows, dbg = compute_pair_fn(
                path_child=path_ch,
                path_caregiver=path_cg,
                dyad_id=dyad_id,
                event=event,
                pair_type="real",
                surrogate_pair_id=dyad_id,
                surrogate_caregiver_dyad=dyad_id,
                surrogate_child_dyad=dyad_id,
            )
            all_rows.extend(rows)
            debug_rows.append(dbg)
        except Exception as exc:
            failed.append({"dyadID": dyad_id, "event": event, "pair_type": "real", "error": str(exc)})

    for surrogate_id, event, path_ch, path_cg, dyad_cg, dyad_ch in surrogate_pairs:
        print(f"Processing SURROGATE {surrogate_id} | {event}")
        try:
            rows, dbg = compute_pair_fn(
                path_child=path_ch,
                path_caregiver=path_cg,
                dyad_id=surrogate_id,
                event=event,
                pair_type="surrogate",
                surrogate_pair_id=surrogate_id,
                surrogate_caregiver_dyad=dyad_cg,
                surrogate_child_dyad=dyad_ch,
            )
            all_rows.extend(rows)
            debug_rows.append(dbg)
        except Exception as exc:
            failed.append({"dyadID": surrogate_id, "event": event, "pair_type": "surrogate", "error": str(exc)})

    ffdtf_env_df = pd.DataFrame(
        all_rows,
        columns=[
            "dyadID",
            "surrogate_pair_id",
            "surrogate_caregiver_dyad",
            "surrogate_child_dyad",
            "pair_type",
            "channel_pair",
            "edge_type",
            "ff_dtf",
            "group",
            "event",
            "n_channels_per_person",
            "model_order",
        ],
    )

    ffdtf_env_debug_df = pd.DataFrame(debug_rows)
    ffdtf_env_failed_df = pd.DataFrame(failed)
    return ffdtf_env_df, ffdtf_env_debug_df, ffdtf_env_failed_df


def extract_group_from_child_da(da_child, get_export_metadata_fn, valid_dyads_df=None):
    grp = np.nan

    child_info_attr = da_child.attrs.get("child_info", np.nan)
    if isinstance(child_info_attr, dict):
        grp = child_info_attr.get("group", np.nan)
    elif isinstance(child_info_attr, str) and child_info_attr.strip():
        try:
            decoded = json.loads(child_info_attr)
            if isinstance(decoded, dict):
                grp = decoded.get("group", np.nan)
        except Exception:
            pass

    if (isinstance(grp, str) and grp.strip() == "") or pd.isna(grp):
        meta = get_export_metadata_fn(da_child)
        grp = meta.get("child_info", {}).get("group", np.nan)

    if (isinstance(grp, str) and grp.strip() == "") or pd.isna(grp):
        dyad = da_child.attrs.get("dyad_id", np.nan)
        if valid_dyads_df is not None and isinstance(dyad, str) and not valid_dyads_df.empty:
            row = valid_dyads_df.loc[valid_dyads_df["dyadID"] == dyad, "child_group"]
            if len(row) > 0:
                grp = row.iloc[0]

    if isinstance(grp, str) and grp.strip() == "":
        grp = np.nan
    return grp
