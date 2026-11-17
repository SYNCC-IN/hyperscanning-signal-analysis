"""Stage 7 - permutation inference on the pairing tables (the primary results; manuscript 2.5, Suppl. S5.1-S5.3, S5.8).

Reads Stage 5's pairing tables (`05_surrogate/<estimator>/<film>_pairing_table.nc`) and runs, per estimator, only
functions of `src.sim_inference` and `src.perm_anova` -- the code the simulations validated:

1. Edge existence (study level), per film x edge: mean of the real dyads (diagonal of T) against within-group
   re-pairings of children and caregivers (Eq. 7; `pairing_permutation_maxT` with `groups`).
2. Mutual coupling: a pair of opposite edges is "mutual" only if both pass test 1 (max-T).
3. Dominant direction, per opposite pair: d = Delta_forward - Delta_reverse per dyad (forward = caregiver -> child;
   Delta against the within-group null); two-sided sign-flip test of mean d (`signflip_maxT`), sign of median d.
4. Group contrast ASD - TD, per film x edge: difference of group means of Delta; the within-group null is recomputed
   under every relabelling (`group_permutation_maxT(renull=True)`); unstudentized. Max-T over the family within each
   film; variants: max-T over all film x edge tests, and the group contrast of Delta averaged over films.
5. Film contrasts (within dyad), test B: mean contrast of T over the real dyads (mean of the two group means)
   against re-pairings of children and caregivers within groups (`caregiver_permutations`,
   `perm_anova.repaired_values`, centred by `all_pairs_mean`; `balanced_mean_maxT`), as in
   `simulations_multimodal_MVAR/dDTF/S13_film_tests.film_rows`. Max-T over edges x contrasts. Sensitivity: sign flip
   of the Delta contrast (`signflip_maxT`), mildly liberal (S13: 7.5-8 %).
6. Film x group interaction: group difference of the Delta contrasts, nulls recomputed under every relabelling
   (`group_permutation_maxT(renull=True, combine=...)`), max-T over edges x contrasts.
7. Optional factorial block (`FACTORIAL`): group terms G, G x F, G x E and the planned 1-df edge-structure contrasts
   G x direction / source modality / target modality (`edge_factor_contrasts`), by raw-label permutation with the
   re-nulled Delta (`relabel_renull_test`) and the unstudentized tr SS_hyp. No edge main effect, no Freedman-Lane,
   no F for group terms.
8. Covariates: no adjusted confirmatory test (none is calibrated when the groups react differently). With
   `ADD_AGE_COVARIATE`, the hybrid renull + SS_hyp test of G with age in Z is reported as exploratory.

Multiplicity: the family is fixed in the config before the run (`FAMILY_EDGES`; default all 8 cross-person edges of
one film). Within-person edges are reported descriptively only. Every test draws its rearrangements from its own
generator ``np.random.default_rng([SEED, test, estimator, film])``; one rearrangement set per family; exact
enumeration when there are at most `N_PERM` arrangements (built into the `src.sim_inference` generators). Max-T is
step-down; min-p is reported as a check.

A dyad whose real fit is unstable in a film is dropped from that film (Stage 5). Cross-film tests use the union of
dyads; a dyad absent from a film drops out of that film's columns. The film-contrast family (tests 5 and the
factorial block) uses the dyads present in every film with a nonzero weight in any configured contrast.

Writes `07_permutation/<estimator>/stage07_tests.csv` (one row per test x film/contrast x edge/pair/term), the
descriptive within-person table, and the gate `07_permutation/permutation_gate.html` (estimators side by side).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.design import node_names
from src.io_utils import ensure_dir
from src.pairing_tables import apply_film_contrasts, contrast_matrix, load_film_table, stack_films
from src.perm_anova import (all_pairs_mean, balanced_mean, balanced_mean_maxT, edge_factor_contrasts, group_terms_fn,
                            rearrangement_matrix, relabel_renull_test, repaired_values, rm_contrasts)
from src.pipeline_config import load_stage_config
from src.sim_inference import (caregiver_permutations, delta_and_z_all, family_pvalues, group_permutation_maxT,
                               pairing_permutation_maxT, renull_delta, signflip_maxT)
from src.surrogate import edge_class_for

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this stage's own section).
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage07_permutation_tests")

ANALYSIS_ROOT = Path(os.environ.get("SYNCC_ANALYSIS_ROOT", PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]))  # env: synthetic runs
INPUT_DIR = ANALYSIS_ROOT / CFG["INPUT_SUBDIR"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])

FILMS = CFG["FILMS"]
ESTIMATORS = CFG["ESTIMATORS"]
PRIMARY_ESTIMATOR = CFG["PRIMARY_ESTIMATOR"]
N_PERM = CFG["N_PERM"]
ALPHA = CFG["ALPHA"]
SEED = CFG["SEED"]
GROUP_CODES = CFG["GROUP_CODES"]                  # {'TD': 0, 'ASD': 1}: group contrasts are ASD - TD
FACTORIAL = CFG["FACTORIAL"]
ADD_AGE_COVARIATE = CFG["ADD_AGE_COVARIATE"]
FILM_CONTRASTS = CFG["FILM_CONTRASTS"]
CONTRAST_NAMES = list(FILM_CONTRASTS)
CONTRAST_WEIGHTS = np.array([[FILM_CONTRASTS[c][f] for f in FILMS] for c in CONTRAST_NAMES])

NODES = CFG["nodes"]
NODE_NAMES = node_names(NODES)
ROLE = {node["name"]: node["role"] for node in NODES}
EDGE_CLASS = {(e["source"], e["target"]): e["class"] for e in CFG["edge_topology"]}
ALL_EDGES = [f"{s}->{t}" for s in NODE_NAMES for t in NODE_NAMES if s != t]
CROSS_EDGES = [e for e in ALL_EDGES if ROLE[e.split("->")[0]] != ROLE[e.split("->")[1]]]
WITHIN_EDGES = [e for e in ALL_EDGES if e not in CROSS_EDGES]
FAMILY = CFG["FAMILY_EDGES"] or CROSS_EDGES       # fixed before the run
assert set(FAMILY) <= set(CROSS_EDGES), "within-person edges cannot be tested with pseudo-dyads; keep them out of FAMILY_EDGES"
FAMILY_LABEL = "all 8 cross-person edges of one film" if FAMILY == CROSS_EDGES else "FAMILY_EDGES: " + ", ".join(FAMILY)
NE = len(FAMILY)


def reverse(edge):
    """Opposite edge ``t->s`` of ``s->t``."""
    s, t = edge.split("->")
    return f"{t}->{s}"


# opposite pairs inside the family, forward = caregiver -> child
PAIRS = [(e, reverse(e)) for e in FAMILY if ROLE[e.split("->")[0]] == "caregiver" and reverse(e) in FAMILY]
SIGNALS = sorted({node.split(":")[1] for e in FAMILY for node in e.split("->")})
EDGE_TERMS = FAMILY == CROSS_EDGES and len(FAMILY) == 8 and len(SIGNALS) == 2   # complete 2 x 2 x 2 layout only
TESTS = {"existence": 1, "direction": 2, "group": 3, "group_24": 4, "group_mean": 5, "film_B": 6, "film_signflip": 7,
         "interaction": 8, "factorial": 9, "covariate": 10}                  # generator index of each test


def rng_for(test, est, film_index=0):
    """Generator of one test: ``default_rng([SEED, test, estimator, film])``."""
    return np.random.default_rng([SEED, TESTS[test], ESTIMATORS.index(est), film_index])


def edge_class(edge):
    """Class of an edge from the config topology ('other' if not listed)."""
    return edge_class_for(*edge.split("->"), EDGE_CLASS)


def group_codes(labels):
    """Group labels (str) -> 0/1 codes."""
    return np.array([GROUP_CODES[g] for g in labels])


def counts(labels):
    """Dyads per group."""
    return {f"n_{g}": int(np.sum(labels == g)) for g in GROUP_CODES}


def within_group_delta(real, pair_values, groups):
    """Delta of every dyad and column against its within-group null (`delta_and_z_all`), and its z."""
    delta, z = delta_and_z_all(real[:, :, None], pair_values[:, :, :, None], "within_group", groups)
    return delta[:, :, 0], z[:, :, 0]


def rows_from(res, base, names, statistic, key="edge"):
    """Tidy rows of one family of tests (`family_pvalues` output)."""
    return [dict(base, **{key: name}, statistic=float(stat), p_raw=float(res["unadjusted"][i]),
                 p_maxT=float(res["maxT"][i]), p_minP=float(res["minP"][i]))
            for i, (name, stat) in enumerate(zip(names, statistic))]


# ---------------------------------------------------------------------------
# Tests per estimator
# ---------------------------------------------------------------------------
all_rows, descriptive_rows, checklist = [], [], {}
for est in ESTIMATORS:
    print(f"\n=== {est} ===")
    films = {film: load_film_table(INPUT_DIR / est / f"{film}_pairing_table.nc", FAMILY) for film in FILMS}
    within = {film: load_film_table(INPUT_DIR / est / f"{film}_pairing_table.nc", WITHIN_EDGES)
              for film in FILMS} if WITHIN_EDGES else {}      # none with one node per person
    rows = []
    checklist[est] = []
    for f, film in enumerate(FILMS):
        t = films[film]
        groups = group_codes(t["group"])
        n_info = dict(n_dyads=len(t["ids"]), **counts(t["group"]))
        checklist[est].append(dict(film=film, n_recorded=t["n_dyads"], n_stable=len(t["ids"]), **counts(t["group"]),
                                   excluded_unstable=", ".join(t["unstable_real"]) or "none",
                                   pseudo_unstable=t["n_pairs_unstable"]))
        table = t["pair_values"].copy()
        table[np.arange(len(groups)), np.arange(len(groups))] = t["real"]
        delta, z = within_group_delta(t["real"], t["pair_values"], groups)
        common = dict(estimator=est, film=film, contrast="", family=FAMILY_LABEL, family_size=NE, **n_info)

        # 1. edge existence: pairing permutation, re-paired within groups
        res = pairing_permutation_maxT(table, rng_for("existence", est, f), N_PERM, groups)
        existence = rows_from(res, dict(common, test="1 edge existence", scope="per film",
                                        null_scope="within-group re-pairing", statistic_label="mean T of real dyads"),
                              FAMILY, np.nanmean(t["real"], axis=0))
        passed = {r["edge"]: r["p_maxT"] < ALPHA for r in existence}
        rows += existence

        # 2. mutual coupling (both opposite edges pass test 1)
        rows += [dict(common, test="2 mutual coupling", scope="per film", null_scope="from test 1", edge=f"{a} | {b}",
                      statistic_label="both edges pass test 1 (max-T)", statistic=float(passed[a] and passed[b]),
                      p_raw=np.nan, p_maxT=np.nan, p_minP=np.nan) for a, b in PAIRS]

        # 3. dominant direction: sign flip of mean d = Delta_forward - Delta_reverse
        if PAIRS:
            d = np.stack([delta[:, FAMILY.index(a)] - delta[:, FAMILY.index(b)] for a, b in PAIRS], axis=1)
            res = signflip_maxT(d, rng_for("direction", est, f), N_PERM)
            direction = rows_from(res, dict(common, test="3 dominant direction", scope="per film", family_size=len(PAIRS),
                                            null_scope="sign flip of d (within-group Delta)",
                                            statistic_label="mean d = Delta(cg->child) - Delta(child->cg)"),
                                  [f"{a} vs {b}" for a, b in PAIRS], d.mean(axis=0))
            for r, col in zip(direction, d.T):
                r["median_sign"] = "+ (cg->child)" if np.median(col) > 0 else "- (child->cg)"
            rows += direction

        # 4. group contrast ASD - TD, re-nulled within-group null, max-T within the film
        res = group_permutation_maxT(np.where(np.isfinite(z), delta, np.nan), groups, rng_for("group", est, f), N_PERM,
                                     renull=True, real=t["real"], pair_values=t["pair_values"])
        diff = delta[groups == 1].mean(axis=0) - delta[groups == 0].mean(axis=0)
        rows += rows_from(res, dict(common, test="4 group contrast", scope="per film, max-T within film",
                                    null_scope="within-group null, recomputed per relabelling",
                                    statistic_label="mean Delta ASD - TD"), FAMILY, diff)

        # within-person edges: descriptive only (cannot be tested with pseudo-dyads)
        w = within.get(film)
        w_delta = within_group_delta(w["real"], w["pair_values"], group_codes(w["group"]))[0] if w else None
        for e, edge in enumerate(WITHIN_EDGES):
            for g in GROUP_CODES:
                sel = w["group"] == g
                descriptive_rows.append({"estimator": est, "film": film, "edge": edge, "group": g, "n": int(sel.sum()),
                                         "mean_T": float(w["real"][sel, e].mean()),
                                         "mean_delta_within_group": float(w_delta[sel, e].mean())})

    # --- cross-film tests on the union of dyads ---
    st = stack_films(films, FILMS)
    groups_all = group_codes(st["group"])
    deltas = np.full_like(st["real"], np.nan)
    for f, film in enumerate(FILMS):
        rows_f = np.array([st["ids"].index(d) for d in films[film]["ids"]])
        d_f, z_f = within_group_delta(films[film]["real"], films[film]["pair_values"], group_codes(films[film]["group"]))
        deltas[rows_f, f * NE:(f + 1) * NE] = np.where(np.isfinite(z_f), d_f, np.nan)
    common = dict(estimator=est, contrast="", family_size=len(FILMS) * NE, n_dyads=len(st["ids"]), **counts(st["group"]),
                  null_scope="within-group null, recomputed per relabelling")

    # 4b. group contrast: max-T over all film x edge tests
    res = group_permutation_maxT(deltas, groups_all, rng_for("group_24", est), N_PERM, renull=True, real=st["real"],
                                 pair_values=st["pair_values"])
    diff = np.nanmean(deltas[groups_all == 1], axis=0) - np.nanmean(deltas[groups_all == 0], axis=0)
    names = [(film, edge) for film in FILMS for edge in FAMILY]
    for r in rows_from(res, dict(common, test="4 group contrast", scope="max-T over film x edge",
                                 family=f"{len(FILMS)} films x family", statistic_label="mean Delta ASD - TD"),
                       range(len(names)), diff):
        r["film"], r["edge"] = names[r["edge"]]
        rows.append(r)

    # 4c. group contrast of Delta averaged over films (a dyad missing a film is left out)
    mean_delta = deltas.reshape(len(groups_all), len(FILMS), NE).mean(axis=1)
    res = group_permutation_maxT(mean_delta, groups_all, rng_for("group_mean", est), N_PERM, renull=True, real=st["real"],
                                 pair_values=st["pair_values"], combine=np.vstack([np.eye(NE) / len(FILMS)] * len(FILMS)))
    diff = np.nanmean(mean_delta[groups_all == 1], axis=0) - np.nanmean(mean_delta[groups_all == 0], axis=0)
    rows += rows_from(res, dict(common, test="4 group contrast", scope="Delta averaged over films", film="mean",
                                family=FAMILY_LABEL, family_size=NE, n_dyads=int(np.isfinite(mean_delta[:, 0]).sum()),
                                statistic_label="mean Delta ASD - TD"), FAMILY, diff)

    # --- film contrasts: dyads present in every weighted film ---
    weighted = np.any(CONTRAST_WEIGHTS != 0, axis=0)
    used = st["present"][:, weighted].all(axis=1)
    g_used = groups_all[used]
    real_u = st["real"][used]
    table_u = st["pair_values"][np.ix_(used, used)].copy()
    table_u[np.arange(used.sum()), np.arange(used.sum())] = real_u
    contrast_names = [(c, edge) for c in CONTRAST_NAMES for edge in FAMILY]
    film_common = dict(estimator=est, film="within dyad", family=f"{len(CONTRAST_NAMES)} contrasts x family",
                       family_size=len(contrast_names), n_dyads=int(used.sum()), **counts(st["group"][used]))

    def contrast_rows(res, base, statistic):
        """Rows of a contrast x edge family."""
        out = rows_from(res, base, range(len(contrast_names)), statistic)
        for r in out:
            r["contrast"], r["edge"] = contrast_names[r["edge"]]
        return out

    # 5. test B: re-pairing children and caregivers within groups (S13_film_tests.film_rows)
    perms = caregiver_permutations(int(used.sum()), rng_for("film_B", est), N_PERM, g_used)
    centre = all_pairs_mean(table_u, g_used)[g_used]
    observed = apply_film_contrasts(real_u - centre, CONTRAST_WEIGHTS, NE)
    null = apply_film_contrasts(repaired_values(table_u, perms) - centre[None], CONTRAST_WEIGHTS, NE)
    res = balanced_mean_maxT(observed, null, g_used)
    rows += contrast_rows(res, dict(film_common, test="5 film contrast (test B)", scope="max-T over contrasts x edges",
                                    null_scope="within-group re-pairing of T",
                                    statistic_label="mean contrast of T (mean of group means)"),
                          balanced_mean(apply_film_contrasts(real_u, CONTRAST_WEIGHTS, NE), g_used))
    # 5b. sensitivity: sign flip of the Delta contrast (mildly liberal, S13: 7.5-8 %)
    delta_contrast = apply_film_contrasts(deltas[used], CONTRAST_WEIGHTS, NE)
    res = signflip_maxT(delta_contrast, rng_for("film_signflip", est), N_PERM, g_used)
    rows += contrast_rows(res, dict(film_common, test="5b film contrast, sign flip of Delta (sensitivity)",
                                    scope="max-T over contrasts x edges",
                                    null_scope="sign flip of within-group Delta contrast; mildly liberal (S13: 7.5-8%)",
                                    statistic_label="mean Delta contrast (mean of group means)"),
                          balanced_mean(delta_contrast, g_used))

    # 6. film x group interaction: group difference of the Delta contrasts, re-nulled
    all_contrast = apply_film_contrasts(deltas, CONTRAST_WEIGHTS, NE)
    res = group_permutation_maxT(all_contrast, groups_all, rng_for("interaction", est), N_PERM, renull=True,
                                 real=st["real"], pair_values=st["pair_values"],
                                 combine=contrast_matrix(CONTRAST_WEIGHTS, NE))
    diff = np.nanmean(all_contrast[groups_all == 1], axis=0) - np.nanmean(all_contrast[groups_all == 0], axis=0)
    rows += contrast_rows(res, dict(film_common, test="6 film x group interaction", scope="max-T over contrasts x edges",
                                    n_dyads=int(np.isfinite(all_contrast[:, 0]).sum()), **counts(st["group"]),
                                    null_scope="within-group null, recomputed per relabelling",
                                    statistic_label="Delta contrast ASD - TD"), diff)

    # 7. factorial block and 8. exploratory covariate: raw-label permutation, re-nulled Delta, tr SS_hyp
    if FACTORIAL or ADD_AGE_COVARIATE:
        levels = {"film": len(FILMS), "edge": NE}
        terms = {"G": rm_contrasts(levels, ()), "GxF": rm_contrasts(levels, ("film",)), "GxE": rm_contrasts(levels, ("edge",))}
        if EDGE_TERMS:
            for factor in ("direction", "source", "target"):
                terms[f"Gx{factor}"] = rm_contrasts(levels, ("edge",), {"edge": edge_factor_contrasts(FAMILY, factor)})
        complete = st["present"].all(axis=1)
        observed_labels = groups_all[complete] == 1

        def renulled(labels):
            """Re-nulled Delta of the complete-case dyads; the other dyads keep their observed group."""
            full = np.tile(groups_all == 1, (len(labels), 1))
            full[:, complete] = labels
            return renull_delta(st["real"], st["pair_values"], full)[:, complete, :]

        perms = rearrangement_matrix(int(complete.sum()), N_PERM, "permute", rng_for("factorial", est))
        base = dict(estimator=est, film="all", contrast="", n_dyads=int(complete.sum()), **counts(st["group"][complete]),
                    null_scope="raw-label permutation, within-group null recomputed per relabelling",
                    statistic_label="tr SS_hyp (unstudentized)", family_size=1, p_maxT=np.nan, p_minP=np.nan)
        if FACTORIAL:
            obs, perm = relabel_renull_test(renulled, observed_labels, perms,
                                            group_terms_fn(list(terms.values()), np.ones((int(complete.sum()), 1))), "ss_hyp")
            for i, term in enumerate(terms):
                rows.append(dict(base, test="7 factorial group terms", scope="omnibus term", edge=term,
                                 family="single test", statistic=float(obs[i]),
                                 p_raw=float(family_pvalues(obs[i:i + 1], perm[:, i:i + 1])["unadjusted"][0])))
        if ADD_AGE_COVARIATE:
            age = st["age"][complete]
            obs, perm = relabel_renull_test(renulled, observed_labels, perms,
                                            group_terms_fn([terms["G"]], np.column_stack([np.ones_like(age), age])), "ss_hyp")
            rows.append(dict(base, test="8 EXPLORATORY: G adjusted for age (hybrid renull + SS_hyp)", scope="omnibus term",
                             edge="G | age", family="exploratory, miscalibrated when groups react differently",
                             statistic=float(obs[0]),
                             p_raw=float(family_pvalues(obs, perm)["unadjusted"][0])))

    for r in rows:
        r.setdefault("edge_class", edge_class(r["edge"]) if r["edge"] in ALL_EDGES else "")
    df = pd.DataFrame(rows)
    ensure_dir(OUTPUT_DIR / est)
    df.to_csv(OUTPUT_DIR / est / "stage07_tests.csv", index=False)
    all_rows.append(df)
    print(df[df["p_maxT"] < ALPHA][["test", "scope", "film", "contrast", "edge", "statistic", "p_raw", "p_maxT"]]
          .to_string(index=False) if (df["p_maxT"] < ALPHA).any() else "no max-T rejection")

pd.DataFrame(descriptive_rows).to_csv(OUTPUT_DIR / "stage07_within_person_descriptive.csv", index=False)
results = pd.concat(all_rows, ignore_index=True)

# ---------------------------------------------------------------------------
# Gate: estimators side by side, with the interpretation guardrails (manuscript Box 2)
# ---------------------------------------------------------------------------
KEYS = ["test", "scope", "film", "contrast", "edge", "edge_class"]


def side_by_side(df):
    """One row per test cell, estimator columns (dDTF / GPDC order of the manuscript tables)."""
    order = [e for e in ("dDTF", "GPDC") if e in ESTIMATORS] + [e for e in ESTIMATORS if e not in ("dDTF", "GPDC")]
    wide = df.pivot_table(index=KEYS, columns="estimator", values=["statistic", "p_raw", "p_maxT"], aggfunc="first", sort=False)
    wide.columns = [f"{est} {val}" for val, est in wide.columns]
    cols = [f"{est} {val}" for est in order for val in ("statistic", "p_raw", "p_maxT") if f"{est} {val}" in wide.columns]
    return wide[cols].reset_index()


def html_table(df):
    """HTML table, max-T rejections in bold."""
    def fmt(col, v):
        if isinstance(v, float):
            text = "" if np.isnan(v) else f"{v:.4g}"
            return f"<b>{text}</b>" if col.endswith("p_maxT") and v < ALPHA else text
        return str(v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(c, v)}</td>" for c, v in zip(df.columns, row)) + "</tr>" for row in df.itertuples(index=False))
    return f"<table><tr>{head}</tr>{body}</table>"


sections = []
for test, sub in results.groupby("test", sort=False):
    extra = ""
    if test.startswith("3"):
        extra = html_table(sub.pivot_table(index=["film", "edge"], columns="estimator", values="median_sign", aggfunc="first").reset_index())
    sections.append(f"<h3>{test}</h3>" + html_table(side_by_side(sub)) + extra)
check_html = "".join(f"<h3>{est}</h3>" + html_table(pd.DataFrame(rows)) for est, rows in checklist.items())

GUARDRAILS = """
<ul>
<li><b>dDTF.</b> A significant reverse or indirect edge is expected when the forward edge is coupled (simulations S1, S5,
  S8). Read a dDTF edge as "the partners are coupled"; read the direction from test 3, and individual edges from GPDC
  (the primary estimator for edge-wise claims: <b>__PRIMARY__</b>). With dDTF, neither film test was calibrated in every
  simulated setting.</li>
<li><b>Dominant direction (test 3).</b> When the two signals differ systematically (child vs adult, brain vs heart), a
  significant d means a larger prediction improvement in that direction, not necessarily stronger coupling.</li>
<li><b>Within-person edges</b> cannot be tested with pseudo-dyads (both members of a pseudo-dyad keep their own
  within-person dynamics). A difference from the pseudo-dyads only shows that coupling with the partner changes the edge;
  they are listed in <code>stage07_within_person_descriptive.csv</code> and are never part of a family.</li>
<li><b>Shared drivers.</b> A driver that is not locked to the stimulus and is missing from the model (e.g. breathing)
  appears as cross-person coupling. <b>No respiration signal is recorded</b> in these data (the loaders read SVAROG
  EEG/ECG, Polar H10 IBI and Pupil Labs eye tracking only), so this cannot be checked here.</li>
<li><b>Cardiac node (option A, raw IBI).</b> The type of cardiac-brain coupling cannot be inferred from a significant edge.</li>
<li><b>Group contrasts.</b> The group contrast is tested within each film; an effect confined to one film is detected only
  within that film (the film-averaged variant dilutes it).</li>
<li><b>Covariates.</b> No covariate-adjusted test is calibrated when the groups react differently; any adjusted result is
  exploratory.</li>
</ul>"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Permutation inference gate</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }}
  table {{ border-collapse: collapse; font-size: 0.82em; margin-bottom: 1em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.2em 0.45em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #f0f0f0; padding: 0 0.3em; }}
</style></head><body>
<h1>Stage 7: permutation inference (primary results)</h1>
<p>Estimators shown as <b>dDTF / GPDC</b>; primary estimator for edge-wise claims: <b>{PRIMARY_ESTIMATOR}</b>.
   Family (fixed before the run): <b>{FAMILY_LABEL}</b>. {N_PERM} rearrangements per test (exact enumeration when there are
   fewer), alpha={ALPHA}, seed {SEED}; max-T step-down, min-p as a check (CSV). Group contrasts: ASD - TD, unstudentized.
   Film contrasts: {', '.join(f'{c} = ' + ' '.join(f'{w:+g} {f}' for f, w in FILM_CONTRASTS[c].items() if w) for c in CONTRAST_NAMES)}.
   Bold = max-T adjusted p &lt; alpha. Delta and z (Stage 5) are the effect sizes.</p>
<h2>Interpretation guardrails</h2>{GUARDRAILS.replace('__PRIMARY__', PRIMARY_ESTIMATOR)}
<h2>Reporting checklist (Box 2): stable fits and excluded dyads per film</h2>{check_html}
<h2>Results</h2>{''.join(sections)}
</body></html>"""
(OUTPUT_DIR / "permutation_gate.html").write_text(html, encoding="utf-8")
print(f"\nWrote {len(results)} test rows and the gate to {OUTPUT_DIR}")
