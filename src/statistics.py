"""Statistical analysis for the hyperscanning pipelines.

Single, pipeline-neutral statistics module: low-level primitives (FDR, OLS /
partial-F, permutation and KS tests) plus the column-parameterized analysis
engine (complete-case prep, repeated-measures permutation tests, post-hoc
suites, edge-wise group comparisons). Shared by the passive (ffDTF) analysis
and, going forward, the SECoRe / HRV analysis.
"""
import hashlib
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from .passive_signal_helpers import node_to_roi


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR-adjusted q-values for ``p_values``."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.empty(n, dtype=float)
    for i, pv in enumerate(p_sorted, start=1):
        q_sorted[i - 1] = pv * n / i
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return q


def _q_to_label(q):
    """Map an FDR q-value to a significance label ('sig'/'trend'/'ns'/'n/a')."""
    if not np.isfinite(q):
        return 'n/a'
    if q < 0.05:
        return 'sig (q<0.05)'
    if q < 0.10:
        return 'trend (q<0.10)'
    return 'ns'


def ols_fit(y, X):
    """Least-squares fit; returns ``(beta, y_hat, resid, rss)``."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.dot(resid, resid))
    return beta, y_hat, resid, rss


def partial_f_from_rss(rss_reduced, rss_full, df1, df2):
    """Partial-F statistic from reduced/full residual sums of squares."""
    num = (rss_reduced - rss_full) / max(df1, 1)
    den = rss_full / max(df2, 1)
    if den <= 0:
        return np.nan
    return float(num / den)


def partial_f(y, X_full, X_reduced):
    """Partial-F comparing nested designs ``X_full`` vs ``X_reduced`` for ``y``."""
    _, _, _, rss_full = ols_fit(y, X_full)
    _, _, _, rss_reduced = ols_fit(y, X_reduced)
    df1 = X_full.shape[1] - X_reduced.shape[1]
    df2 = len(y) - X_full.shape[1]
    if df1 <= 0 or df2 <= 0:
        return np.nan
    return partial_f_from_rss(rss_reduced, rss_full, df1, df2)


def demean_within_blocks(vec, block_ids):
    """Subtract the per-block mean from ``vec`` within each block in ``block_ids``."""
    out = np.asarray(vec, dtype=float).copy()
    for b in np.unique(block_ids):
        idx = np.where(block_ids == b)[0]
        out[idx] = out[idx] - np.mean(out[idx])
    return out


def permute_within_blocks(vec, block_ids, rng):
    """Permute ``vec`` independently within each block in ``block_ids``."""
    out = np.array(vec, copy=True)
    for b in np.unique(block_ids):
        idx = np.where(block_ids == b)[0]
        out[idx] = out[idx[rng.permutation(len(idx))]]
    return out


def permute_across_units(vec, rng):
    """Return ``vec`` with its elements globally permuted."""
    idx = rng.permutation(len(vec))
    return np.asarray(vec)[idx]


def edge_seed(edge_name, base_seed):
    """Deterministic 32-bit seed derived from ``edge_name`` and ``base_seed``."""
    key = f"{int(base_seed)}|{str(edge_name)}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)


def perm_pvalue_two_sided(x, y, n_perm=5000, rng=None):
    """Two-sided permutation p-value for the difference of means ``mean(x)-mean(y)``."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])
    n_x = x.size

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:n_x]) - np.mean(pooled[n_x:])
        if abs(diff) >= abs(obs):
            count += 1

    p_val = (count + 1) / (n_perm + 1)
    return p_val, obs


def ks_statistic_two_sample(x, y):
    """Two-sample Kolmogorov-Smirnov statistic between samples ``x`` and ``y``."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if x.size == 0 or y.size == 0:
        return np.nan

    grid = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def perm_pvalue_ks(x, y, n_perm=5000, rng=None):
    """Permutation p-value for the two-sample KS statistic."""
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = x.size
    pooled = np.concatenate([x, y])

    obs = ks_statistic_two_sample(x, y)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_perm = pooled[:n_x]
        y_perm = pooled[n_x:]
        stat_perm = ks_statistic_two_sample(x_perm, y_perm)
        if stat_perm >= obs:
            count += 1

    p_val = (count + 1) / (n_perm + 1)
    return p_val, obs




def _build_edge_long_complete(edge_df, target_events):
    agg = edge_df.groupby(["dyadID", "group_binary", "event"], as_index=False)["ff_dtf"].mean()

    target_events = [str(e) for e in target_events]
    agg = agg.loc[agg["event"].isin(target_events)].copy()

    event_set = set(target_events)
    ev_by_dyad = agg.groupby("dyadID")["event"].apply(lambda s: set(map(str, s))).to_dict()
    keep_dyads = sorted([d for d, evs in ev_by_dyad.items() if evs == event_set])
    agg = agg.loc[agg["dyadID"].isin(keep_dyads)].copy()
    if agg.empty:
        return None

    g_per_dyad = agg.groupby("dyadID")["group_binary"].nunique()
    good_dyads = g_per_dyad[g_per_dyad == 1].index.tolist()
    agg = agg.loc[agg["dyadID"].isin(good_dyads)].copy()
    if agg.empty:
        return None

    group_map = agg.groupby("dyadID")["group_binary"].first().to_dict()
    rows = []
    for dyad in sorted(group_map.keys()):
        for ev in target_events:
            tmp = agg.loc[(agg["dyadID"] == dyad) & (agg["event"] == ev), "ff_dtf"]
            if len(tmp) != 1:
                return None
            rows.append(
                {
                    "dyadID": dyad,
                    "group_binary": group_map[dyad],
                    "event": ev,
                    "ff_dtf": float(tmp.iloc[0]),
                }
            )

    return pd.DataFrame(rows)


def _run_edge_rm_tests(edge_df, target_events, n_perm=5000, rng=None, min_dyads_per_group=5):
    if rng is None:
        rng = np.random.default_rng(0)

    long_df = _build_edge_long_complete(edge_df, target_events)
    if long_df is None or long_df.empty:
        return None

    labels = set(long_df["group_binary"].astype(str).unique())
    if labels != {"TD", "ASD"}:
        return {
            "ok": False,
            "reason": f"Unexpected group labels for edge: {sorted(labels)}",
            "n_td": int((long_df["group_binary"] == "TD").sum() > 0),
            "n_asd": int((long_df["group_binary"] == "ASD").sum() > 0),
            "n_dyads_total": int(long_df["dyadID"].nunique()),
        }

    y = long_df["ff_dtf"].to_numpy(dtype=float)
    g = (long_df["group_binary"].to_numpy() == "ASD").astype(float)
    events = [str(e) for e in target_events]
    m1 = (long_df["event"].to_numpy() == events[1]).astype(float)
    m2 = (long_df["event"].to_numpy() == events[2]).astype(float)
    block_ids = long_df["dyadID"].to_numpy()

    n_by_group = long_df.groupby("group_binary")["dyadID"].nunique().to_dict()
    n_td = int(n_by_group.get("TD", 0))
    n_asd = int(n_by_group.get("ASD", 0))
    if min(n_td, n_asd) < min_dyads_per_group:
        return {
            "ok": False,
            "reason": f"Insufficient dyads per group after complete-case filtering (TD={n_td}, ASD={n_asd}).",
            "n_td": n_td,
            "n_asd": n_asd,
            "n_dyads_total": int(n_td + n_asd),
        }

    dyad_summary = (
        long_df.groupby("dyadID", as_index=False)
        .agg(group_binary=("group_binary", "first"), ff_dtf_mean=("ff_dtf", "mean"))
        .sort_values("dyadID")
        .reset_index(drop=True)
    )
    y_d = dyad_summary["ff_dtf_mean"].to_numpy(dtype=float)
    g_d = (dyad_summary["group_binary"].to_numpy() == "ASD").astype(float)

    Xg0 = np.ones((len(y_d), 1), dtype=float)
    Xg1 = np.column_stack([np.ones(len(y_d), dtype=float), g_d])
    F_group_obs = partial_f(y_d, Xg1, Xg0)

    _, yhat_g0, resid_g0, _ = ols_fit(y_d, Xg0)
    perm_group = []
    for _ in range(n_perm):
        y_perm = yhat_g0 + permute_across_units(resid_g0, rng)
        perm_group.append(partial_f(y_perm, Xg1, Xg0))

    y_w = demean_within_blocks(y, block_ids)
    m1_w = demean_within_blocks(m1, block_ids)
    m2_w = demean_within_blocks(m2, block_ids)
    gm1_w = demean_within_blocks(g * m1, block_ids)
    gm2_w = demean_within_blocks(g * m2, block_ids)

    Xm0 = np.zeros((len(y_w), 0), dtype=float)
    Xm = np.column_stack([m1_w, m2_w])
    Xmi = np.column_stack([m1_w, m2_w, gm1_w, gm2_w])

    F_movie_obs = partial_f(y_w, Xm, Xm0)
    F_inter_obs = partial_f(y_w, Xmi, Xm)

    _, yhat_m0, resid_m0, _ = ols_fit(y_w, Xm0)
    perm_movie = []
    for _ in range(n_perm):
        y_perm = yhat_m0 + permute_within_blocks(resid_m0, block_ids, rng)
        perm_movie.append(partial_f(y_perm, Xm, Xm0))

    _, yhat_i0, resid_i0, _ = ols_fit(y_w, Xm)
    perm_inter = []
    for _ in range(n_perm):
        y_perm = yhat_i0 + permute_within_blocks(resid_i0, block_ids, rng)
        perm_inter.append(partial_f(y_perm, Xmi, Xm))

    def _perm_pvalue(observed, perm_stats):
        perm_stats = np.asarray(perm_stats, dtype=float)
        perm_stats = perm_stats[np.isfinite(perm_stats)]
        if not np.isfinite(observed) or perm_stats.size == 0:
            return np.nan
        return float((1 + np.sum(perm_stats >= observed)) / (perm_stats.size + 1))

    p_group = _perm_pvalue(F_group_obs, perm_group)
    p_movie = _perm_pvalue(F_movie_obs, perm_movie)
    p_inter = _perm_pvalue(F_inter_obs, perm_inter)

    return {
        "ok": True,
        "n_td": n_td,
        "n_asd": n_asd,
        "n_dyads_total": int(n_td + n_asd),
        "F_group": float(F_group_obs),
        "F_movie": float(F_movie_obs),
        "F_interaction": float(F_inter_obs),
        "p_group_perm": p_group,
        "p_movie_perm": p_movie,
        "p_interaction_perm": p_inter,
    }


def _agg_op(series, method="mean"):
    if method == "sum":
        return float(series.sum())
    return float(series.mean())


def _parse_channel_pair(cp):
    text = str(cp).strip()
    m = re.match(r"^(ch|cg):([^\-\s>]+)\s*->\s*(ch|cg):([^\s]+)$", text)
    if m is None:
        return None
    return {
        "src_role": m.group(1),
        "src_node": m.group(2),
        "dst_role": m.group(3),
        "dst_node": m.group(4),
    }


# Backwards-compatible alias; canonical implementation lives in passive_signal_helpers.


def _add_edge_parts(df):
    tmp = df.copy()
    parsed = tmp["channel_pair"].apply(_parse_channel_pair)
    tmp["src_role"] = parsed.apply(lambda x: np.nan if x is None else x["src_role"])
    tmp["src_node"] = parsed.apply(lambda x: np.nan if x is None else x["src_node"])
    tmp["dst_role"] = parsed.apply(lambda x: np.nan if x is None else x["dst_role"])
    tmp["dst_node"] = parsed.apply(lambda x: np.nan if x is None else x["dst_node"])
    tmp = tmp.dropna(subset=["src_role", "src_node", "dst_role", "dst_node"]).copy()
    return tmp


def _filter_direction_family(df, family="ch_to_cg"):
    if family == "ch_to_cg":
        return df.loc[(df["src_role"] == "ch") & (df["dst_role"] == "cg")].copy()
    if family == "cg_to_ch":
        return df.loc[(df["src_role"] == "cg") & (df["dst_role"] == "ch")].copy()
    raise ValueError(f"Unknown family: {family}")


def _run_rm_on_aggregated_keys(
    df,
    key_col,
    n_perm=5000,
    seed_base=1701,
    target_events=("Peppa", "Incredibles", "Brave"),
    min_dyads_per_group=5,
    bh_func=None,
):
    out_rows = []
    skipped_rows = []

    for stat_key, part in df.groupby(key_col):
        seed = edge_seed(str(stat_key), seed_base)
        rng = np.random.default_rng(seed)

        test_out = _run_edge_rm_tests(
            edge_df=part[["dyadID", "group_binary", "event", "ff_dtf"]].copy(),
            target_events=target_events,
            n_perm=n_perm,
            rng=rng,
            min_dyads_per_group=min_dyads_per_group,
        )

        if test_out is None or not test_out.get("ok", False):
            skipped_rows.append(
                {
                    key_col: stat_key,
                    "reason": "No valid complete-case dyads." if test_out is None else test_out.get("reason", "Unknown"),
                    "n_td": np.nan if test_out is None else test_out.get("n_td", np.nan),
                    "n_asd": np.nan if test_out is None else test_out.get("n_asd", np.nan),
                    "perm_seed": seed,
                }
            )
            continue

        out_rows.append({key_col: stat_key, "perm_seed": seed, **test_out})

    res = pd.DataFrame(out_rows)
    skipped = pd.DataFrame(skipped_rows)

    if not res.empty:
        bh = bh_fdr if bh_func is None else bh_func
        res = res.sort_values(key_col).reset_index(drop=True)
        res["q_group_fdr"] = bh(res["p_group_perm"].to_numpy())
        res["q_movie_fdr"] = bh(res["p_movie_perm"].to_numpy())
        res["q_interaction_fdr"] = bh(res["p_interaction_perm"].to_numpy())

        res["sig_group_fdr_0p05"] = res["q_group_fdr"] < 0.05
        res["sig_movie_fdr_0p05"] = res["q_movie_fdr"] < 0.05
        res["sig_interaction_fdr_0p05"] = res["q_interaction_fdr"] < 0.05

    return res, skipped


def _select_rm_targets(rm_df, key_col, q_sig=0.05, q_trend=None):
    if rm_df is None or rm_df.empty or key_col not in rm_df.columns:
        return pd.DataFrame(columns=[key_col, "target_reason"])

    rows = []
    for _, row in rm_df.iterrows():
        reasons = []
        for effect in ("group", "movie", "interaction"):
            q_val = row.get(f"q_{effect}_fdr", np.nan)
            if pd.isna(q_val):
                continue
            if q_val < q_sig:
                reasons.append(f"{effect}:sig(q={q_val:.3f})")
            elif q_trend is not None and q_sig <= q_val < q_trend:
                reasons.append(f"{effect}:trend(q={q_val:.3f})")

        if reasons:
            rows.append({key_col: row[key_col], "target_reason": "; ".join(reasons)})

    if not rows:
        return pd.DataFrame(columns=[key_col, "target_reason"])

    return pd.DataFrame(rows).drop_duplicates(subset=[key_col]).reset_index(drop=True)


def _prepare_complete_case(df_key, events):
    if df_key is None or df_key.empty:
        return pd.DataFrame()

    out = df_key.copy()
    out = out.loc[out["event"].astype(str).isin(events)].copy()
    out["event"] = out["event"].astype(str)

    event_set = set(events)
    ev_by_dyad = out.groupby("dyadID")["event"].apply(lambda s: set(map(str, s))).to_dict()
    complete_dyads = sorted([dyad for dyad, evs in ev_by_dyad.items() if evs == event_set])
    out = out.loc[out["dyadID"].isin(complete_dyads)].copy()

    stable = out.groupby("dyadID")["group_binary"].nunique()
    stable_dyads = stable[stable == 1].index.tolist()
    out = out.loc[out["dyadID"].isin(stable_dyads)].copy()
    return out


def _run_posthoc_for_key(df_agg, key_col, key_value, events, bh_func=None):
    sub = df_agg.loc[df_agg[key_col] == key_value].copy()
    sub = _prepare_complete_case(sub, events)
    if sub.empty:
        return None

    wide = (
        sub.pivot_table(
            index=["dyadID", "group_binary"],
            columns="event",
            values="ff_dtf",
            aggfunc="mean",
        ).reset_index()
    )

    missing_events = [event for event in events if event not in wide.columns]
    if missing_events:
        return None

    bh = bh_fdr if bh_func is None else bh_func
    event_pairs = list(combinations(events, 2))

    dyad_mean = wide[["dyadID", "group_binary"] + events].copy()
    dyad_mean["mean_over_movies"] = dyad_mean[events].mean(axis=1)
    td_vals = dyad_mean.loc[dyad_mean["group_binary"] == "TD", "mean_over_movies"].to_numpy(dtype=float)
    asd_vals = dyad_mean.loc[dyad_mean["group_binary"] == "ASD", "mean_over_movies"].to_numpy(dtype=float)
    td_vals = td_vals[np.isfinite(td_vals)]
    asd_vals = asd_vals[np.isfinite(asd_vals)]

    group_posthoc_df = pd.DataFrame(
        [
            {
                "contrast": "overall_mean(TD vs ASD)",
                "n_td": int(len(td_vals)),
                "n_asd": int(len(asd_vals)),
                "median_td": float(np.nanmedian(td_vals)) if len(td_vals) else np.nan,
                "median_asd": float(np.nanmedian(asd_vals)) if len(asd_vals) else np.nan,
                "mw_U": np.nan,
                "p_group_posthoc": np.nan,
            }
        ]
    )
    if min(len(td_vals), len(asd_vals)) >= 3:
        try:
            u_stat = mannwhitneyu(asd_vals, td_vals, alternative="two-sided")
            group_posthoc_df.loc[0, "mw_U"] = float(u_stat.statistic)
            group_posthoc_df.loc[0, "p_group_posthoc"] = float(u_stat.pvalue)
        except Exception:
            pass
    group_posthoc_df["q_group_posthoc_bh"] = bh(group_posthoc_df["p_group_posthoc"].to_numpy())

    movie_rows = []
    for grp in ("TD", "ASD"):
        grp_wide = wide.loc[wide["group_binary"] == grp].copy()
        for event_a, event_b in event_pairs:
            vals_a = grp_wide[event_a].to_numpy(dtype=float)
            vals_b = grp_wide[event_b].to_numpy(dtype=float)
            valid = np.isfinite(vals_a) & np.isfinite(vals_b)
            vals_a = vals_a[valid]
            vals_b = vals_b[valid]
            count = int(len(vals_a))

            p_val = np.nan
            stat_w = np.nan
            if count >= 3 and np.any(np.abs(vals_b - vals_a) > 0):
                try:
                    w_stat = wilcoxon(vals_b, vals_a, alternative="two-sided", zero_method="wilcox")
                    stat_w = float(w_stat.statistic)
                    p_val = float(w_stat.pvalue)
                except Exception:
                    pass

            movie_rows.append(
                {
                    "contrast": f"{event_b} - {event_a}",
                    "group": grp,
                    "n_dyads": count,
                    "median_diff": float(np.nanmedian(vals_b - vals_a)) if count else np.nan,
                    "wilcoxon_W": stat_w,
                    "p_movie_posthoc": p_val,
                }
            )

    movie_posthoc_df = pd.DataFrame(movie_rows)
    movie_posthoc_df["q_movie_posthoc_bh"] = bh(movie_posthoc_df["p_movie_posthoc"].to_numpy())

    interaction_rows = []
    td_wide = wide.loc[wide["group_binary"] == "TD"].copy()
    asd_wide = wide.loc[wide["group_binary"] == "ASD"].copy()
    for event_a, event_b in event_pairs:
        delta_td = (td_wide[event_b] - td_wide[event_a]).to_numpy(dtype=float)
        delta_asd = (asd_wide[event_b] - asd_wide[event_a]).to_numpy(dtype=float)
        delta_td = delta_td[np.isfinite(delta_td)]
        delta_asd = delta_asd[np.isfinite(delta_asd)]

        n_td = int(len(delta_td))
        n_asd = int(len(delta_asd))

        p_val = np.nan
        stat_u = np.nan
        if min(n_td, n_asd) >= 3:
            try:
                u_stat = mannwhitneyu(delta_asd, delta_td, alternative="two-sided")
                stat_u = float(u_stat.statistic)
                p_val = float(u_stat.pvalue)
            except Exception:
                pass

        interaction_rows.append(
            {
                "contrast": f"{event_b} - {event_a}",
                "n_td": n_td,
                "n_asd": n_asd,
                "median_delta_td": float(np.nanmedian(delta_td)) if n_td else np.nan,
                "median_delta_asd": float(np.nanmedian(delta_asd)) if n_asd else np.nan,
                "mw_U": stat_u,
                "p_interaction_posthoc": p_val,
            }
        )

    interaction_posthoc_df = pd.DataFrame(interaction_rows)
    interaction_posthoc_df["q_interaction_posthoc_bh"] = bh(interaction_posthoc_df["p_interaction_posthoc"].to_numpy())

    return {
        "wide": wide,
        "group_posthoc_df": group_posthoc_df,
        "movie_posthoc_df": movie_posthoc_df,
        "interaction_posthoc_df": interaction_posthoc_df,
    }


def run_directional_posthoc_suite(
    direction_title,
    rm_specs,
    target_events,
    q_sig=0.05,
    q_trend=None,
    mode="full",
    display_fn=None,
    plot_boxplots_fn=None,
    plot_three_panel_fn=None,
    plot_main_effect_fn=None,
):
    if mode not in {"full", "main_effect"}:
        raise ValueError(f"Unsupported mode: {mode}")

    show = print if display_fn is None else display_fn
    events = [str(event) for event in target_events]

    print(f"\n### {direction_title}")
    if q_trend is None:
        print(f"Criteria: significant only (q<{q_sig})")
    else:
        print(f"Criteria: significant q<{q_sig} or trend {q_sig}<=q<{q_trend}")

    for spec in rm_specs:
        label = spec["label"]
        rm_df = spec.get("rm_df", None)
        agg_df = spec.get("agg_df", None)
        key_col = spec["key_col"]

        if rm_df is None or agg_df is None or key_col not in agg_df.columns:
            print(f"[{label}] Missing required dataframes/columns, skipping.")
            continue

        target_df = _select_rm_targets(rm_df, key_col=key_col, q_sig=q_sig, q_trend=q_trend)
        if target_df.empty:
            print(f"[{label}] No significant{'/trend' if mode == 'full' else ''} targets.")
            continue

        print(f"[{label}] Targets to post-hoc: {len(target_df)}")
        show(target_df)

        for _, row in target_df.iterrows():
            key_value = row[key_col]
            reason = row["target_reason"]
            title = f"{direction_title} | {label} | {key_value}"
            print(f"\n- Running post-hoc for {key_value} ({reason})")

            out = _run_posthoc_for_key(agg_df, key_col=key_col, key_value=key_value, events=events)
            if out is None:
                print("  No complete-case data for post-hoc.")
                continue

            if mode == "full":
                print("  Group post-hoc (dyad mean across movies):")
                show(out["group_posthoc_df"])
                print("  Movie post-hoc (within-group paired contrasts):")
                show(out["movie_posthoc_df"].sort_values(["group", "q_movie_posthoc_bh", "contrast"]).reset_index(drop=True))
                print("  Interaction post-hoc (between-group deltas):")
                show(out["interaction_posthoc_df"].sort_values(["q_interaction_posthoc_bh", "contrast"]).reset_index(drop=True))

                if plot_boxplots_fn is not None:
                    plot_boxplots_fn(out["wide"], events=events, title_prefix=title + " | Boxplots")
                if plot_three_panel_fn is not None:
                    plot_three_panel_fn(out["movie_posthoc_df"], title_prefix=title + " | 3-panel")
                continue

            first_reason = str(reason).split(";", 1)[0].strip()
            effect = first_reason.split(":", 1)[0] if ":" in first_reason else "group"

            if effect == "group":
                print(f"  Group post-hoc (dyad mean across movies) - significant only (q<{q_sig}):")
                group_sig = out["group_posthoc_df"].loc[out["group_posthoc_df"]["q_group_posthoc_bh"] < q_sig].reset_index(drop=True)
                if group_sig.empty:
                    print("    No significant rows.")
                else:
                    show(group_sig)
                print("  Main-effect target: using group-aggregated boxplot only (no interaction panel).")
                if plot_main_effect_fn is not None:
                    plot_main_effect_fn(out["wide"], events=events, title_prefix=title + " | Boxplots", effect="group")

            elif effect == "movie":
                print(f"  Movie post-hoc (within-group paired contrasts) - significant only (q<{q_sig}):")
                movie_sig = out["movie_posthoc_df"].loc[out["movie_posthoc_df"]["q_movie_posthoc_bh"] < q_sig]
                movie_sig = movie_sig.sort_values(["group", "q_movie_posthoc_bh", "contrast"]).reset_index(drop=True)
                if movie_sig.empty:
                    print("    No significant rows.")
                else:
                    show(movie_sig)
                print("  Main-effect target: using movie-aggregated boxplot only (no interaction panel).")
                if plot_main_effect_fn is not None:
                    plot_main_effect_fn(out["wide"], events=events, title_prefix=title + " | Boxplots", effect="movie")

            else:
                print(f"  Interaction post-hoc (between-group deltas) - significant only (q<{q_sig}):")
                inter_sig = out["interaction_posthoc_df"].loc[out["interaction_posthoc_df"]["q_interaction_posthoc_bh"] < q_sig]
                inter_sig = inter_sig.sort_values(["q_interaction_posthoc_bh", "contrast"]).reset_index(drop=True)
                if inter_sig.empty:
                    print("    No significant rows.")
                else:
                    show(inter_sig)

                print("  Interaction-driven target: using grouped boxplots (group x movie).")
                if plot_boxplots_fn is not None:
                    plot_boxplots_fn(out["wide"], events=events, title_prefix=title + " | Boxplots")


def _split_channel_pair(cp):
    src, dst = cp.split("->", 1)
    return src, dst


def _apply_stats_edge_filters(df_input, channel_col="channel_pair", selected_channels=None, only_homologous=False):
    out = df_input.copy()
    if out.empty or channel_col not in out.columns:
        return out

    src_dst = out[channel_col].astype(str).str.split("->", n=1, expand=True)
    out["__src"] = src_dst[0]
    out["__dst"] = src_dst[1]
    out["__src_role"] = out["__src"].str.split(":", n=1).str[0]
    out["__dst_role"] = out["__dst"].str.split(":", n=1).str[0]
    out["__src_ch"] = out["__src"].str.split(":", n=1).str[1]
    out["__dst_ch"] = out["__dst"].str.split(":", n=1).str[1]

    if selected_channels is not None:
        selected_set = set(map(str, selected_channels))
        out = out.loc[out["__src_ch"].isin(selected_set) & out["__dst_ch"].isin(selected_set)].copy()

    if only_homologous:
        out = out.loc[
            (out["__src_ch"] == out["__dst_ch"])
            & (out["__src_role"] != out["__dst_role"])
            & out["__src_role"].isin(["ch", "cg"])
            & out["__dst_role"].isin(["ch", "cg"])
        ].copy()

    return out.drop(columns=["__src", "__dst", "__src_role", "__dst_role", "__src_ch", "__dst_ch"])


def _normalize_td_asd_label(v):
    s = str(v).strip().lower()
    if s == "td" or "typ" in s:
        return "TD"
    if s == "asd" or s.startswith("asd") or "aut" in s:
        return "ASD"
    return None


def prepare_td_asd_df(
    df_input,
    filter_cross_brain=True,
    selected_channels=None,
    only_homologous=False,
):
    out = df_input.copy()
    if "pair_type" in out.columns:
        out = out.loc[out["pair_type"] == "real"].copy()
    if filter_cross_brain and "edge_type" in out.columns:
        out = out.loc[out["edge_type"] == "cross-brain"].copy()

    out = _apply_stats_edge_filters(
        out,
        channel_col="channel_pair",
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )

    out["group_binary"] = out["group"].map(_normalize_td_asd_label)
    out = out.loc[out["group_binary"].isin(["TD", "ASD"])].copy()
    out = out.dropna(subset=["group_binary", "event", "channel_pair", "ff_dtf"])
    return out


def prepare_group_comparison_df(
    df_input,
    group_col,
    groups,
    value_col="ff_dtf",
    event_col="event",
    channel_col="channel_pair",
    filter_cross_brain=True,
    selected_channels=None,
    only_homologous=False,
):
    if df_input is None or len(df_input) == 0:
        raise RuntimeError("Input dataframe is empty.")

    test_df = df_input.copy()
    if filter_cross_brain and "edge_type" in test_df.columns:
        test_df = test_df.loc[test_df["edge_type"] == "cross-brain"].copy()

    required = {group_col, channel_col, value_col, event_col}
    missing = sorted(required - set(test_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    g1, g2 = groups
    test_df = test_df.loc[test_df[group_col].isin([g1, g2])].copy()
    test_df = test_df.dropna(subset=[group_col, channel_col, value_col, event_col])

    test_df = _apply_stats_edge_filters(
        test_df,
        channel_col=channel_col,
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )
    if test_df.empty:
        raise RuntimeError("No rows left after group filtering.")

    return test_df


def _run_edgewise_test(
    df_input,
    group_col,
    groups,
    pair_id_col,
    stat_fn,
    build_row,
    alpha=0.05,
    n_perm=5000,
    min_pairs_per_group=5,
    random_seed=123,
    aggregate_over_events=True,
    filter_cross_brain=True,
    value_col="ff_dtf",
    channel_col="channel_pair",
    selected_channels=None,
    only_homologous=False,
):
    """Shared scaffold for per-edge two-group permutation tests.

    ``stat_fn(x, y, n_perm, rng) -> (p_value, statistic)`` computes the test, and
    ``build_row(edge, x, y, statistic, p_value, g1, g2) -> dict`` builds the
    per-edge result record. Returns ``(result_df, summary)``.
    """
    g1, g2 = groups
    test_df = prepare_group_comparison_df(
        df_input,
        group_col=group_col,
        groups=groups,
        value_col=value_col,
        event_col="event",
        channel_col=channel_col,
        filter_cross_brain=filter_cross_brain,
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )

    if pair_id_col not in test_df.columns:
        return pd.DataFrame(), {
            "tested_edges": 0,
            "n_significant_fdr": 0,
            "any_significant": False,
            "message": f"Missing pair_id_col: {pair_id_col}",
        }

    if aggregate_over_events:
        pair_level = (
            test_df.groupby([pair_id_col, group_col, channel_col], as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "ff_dtf_stat"})
        )
    else:
        pair_level = test_df.rename(columns={value_col: "ff_dtf_stat"})

    rng = np.random.default_rng(random_seed)
    results = []
    for edge, grp_df in pair_level.groupby(channel_col):
        x = grp_df.loc[grp_df[group_col] == g1, "ff_dtf_stat"].to_numpy()
        y = grp_df.loc[grp_df[group_col] == g2, "ff_dtf_stat"].to_numpy()
        if x.size < min_pairs_per_group or y.size < min_pairs_per_group:
            continue

        p_val, statistic = stat_fn(x, y, n_perm=n_perm, rng=rng)
        results.append(build_row(edge, x, y, statistic, p_val, g1, g2))

    if not results:
        return pd.DataFrame(), {
            "tested_edges": 0,
            "n_significant_fdr": 0,
            "any_significant": False,
            "message": "No edges met min_pairs_per_group criterion.",
        }

    result_df = pd.DataFrame(results).sort_values("p_perm").reset_index(drop=True)
    result_df["q_fdr_bh"] = bh_fdr(result_df["p_perm"].to_numpy())
    result_df["significant_fdr"] = result_df["q_fdr_bh"] < alpha

    n_sig = int(result_df["significant_fdr"].sum())
    summary = {
        "tested_edges": int(len(result_df)),
        "n_significant_fdr": n_sig,
        "any_significant": bool(n_sig > 0),
        "message": "OK",
    }
    return result_df, summary


def run_edgewise_permutation_test(
    df_input,
    group_col,
    groups,
    pair_id_col,
    alpha=0.05,
    n_perm=5000,
    min_pairs_per_group=5,
    random_seed=123,
    aggregate_over_events=True,
    filter_cross_brain=True,
    value_col="ff_dtf",
    channel_col="channel_pair",
    selected_channels=None,
    only_homologous=False,
):
    def build_row(edge, x, y, diff_mean, p_val, g1, g2):
        return {
            "channel_pair": edge,
            f"n_{g1}": int(x.size),
            f"n_{g2}": int(y.size),
            f"mean_{g1}": float(np.mean(x)),
            f"mean_{g2}": float(np.mean(y)),
            f"diff_{g1}_minus_{g2}": float(diff_mean),
            "p_perm": float(p_val),
        }

    return _run_edgewise_test(
        df_input,
        group_col=group_col,
        groups=groups,
        pair_id_col=pair_id_col,
        stat_fn=perm_pvalue_two_sided,
        build_row=build_row,
        alpha=alpha,
        n_perm=n_perm,
        min_pairs_per_group=min_pairs_per_group,
        random_seed=random_seed,
        aggregate_over_events=aggregate_over_events,
        filter_cross_brain=filter_cross_brain,
        value_col=value_col,
        channel_col=channel_col,
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )


def run_edgewise_shape_test(
    df_input,
    group_col,
    groups,
    pair_id_col,
    alpha=0.05,
    n_perm=5000,
    min_pairs_per_group=5,
    random_seed=123,
    aggregate_over_events=True,
    filter_cross_brain=True,
    value_col="ff_dtf",
    channel_col="channel_pair",
    selected_channels=None,
    only_homologous=False,
):
    def build_row(edge, x, y, ks_stat, p_val, g1, g2):
        return {
            "channel_pair": edge,
            f"n_{g1}": int(x.size),
            f"n_{g2}": int(y.size),
            f"median_{g1}": float(np.median(x)),
            f"median_{g2}": float(np.median(y)),
            "ks_stat": float(ks_stat),
            "p_perm": float(p_val),
        }

    return _run_edgewise_test(
        df_input,
        group_col=group_col,
        groups=groups,
        pair_id_col=pair_id_col,
        stat_fn=perm_pvalue_ks,
        build_row=build_row,
        alpha=alpha,
        n_perm=n_perm,
        min_pairs_per_group=min_pairs_per_group,
        random_seed=random_seed,
        aggregate_over_events=aggregate_over_events,
        filter_cross_brain=filter_cross_brain,
        value_col=value_col,
        channel_col=channel_col,
        selected_channels=selected_channels,
        only_homologous=only_homologous,
    )
