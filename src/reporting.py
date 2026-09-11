"""HTML-fragment renderers for the DTF pipeline stages' (stage02-stage06)
self-contained QC gate reports.

Each function takes the data rows/dicts a stage already assembled for its
gate and returns a plain HTML string fragment; the calling script embeds
these fragments into its own `HTML_TEMPLATE` and writes the final gate file.
No function here reads or writes a file itself.
"""

import pandas as pd


def render_dyad_panel_envelopes(dyad_id, entries, names):
    """Render one Stage 2 dyad's QC panel (continuous figures + per-film sections).

    Node-keyed (not role-keyed): renders one continuous PSD figure per ROI
    node, one node-QC figure per film per node (a ROI node's filter/envelope
    figure or an IBI node's raw-IBI trace), plus the shared continuous
    overlay and design-variable PSD figures -- so any per-role node count
    (zero, one, or several ROI/HRV nodes) renders correctly.

    Parameters
    ----------
    dyad_id : str
    entries : list of dict
        This dyad's `gate_entries` rows, one per film.
    names : list of str
        Node names, in display order (see `src.design.node_names`).

    Returns
    -------
    str
        HTML fragment for the dyad's panel div.
    """
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    written = [e for e in entries if e["status"] == "written"]
    if written:
        qc = written[0]["qc"]
        html.append('<div class="row">')
        for name in names:
            if name in qc["psd_band"]:
                html.append(f'<img src="qc/{qc["psd_band"][name]}" alt="{name} continuous PSD">')
        html.append(f'<img src="qc/{qc["overlay"]}" alt="continuous overlay">')
        html.append('</div>')

    for entry in entries:
        html.append(f'<div class="film-block"><h3>{entry["film"]}</h3>')
        if entry["status"] == "skipped":
            html.append(f'<p class="skipped">Skipped: {entry["reason"]}</p>')
        else:
            qc = entry["qc"]
            html.append('<div class="row">')
            for name in names:
                html.append(f'<img src="qc/{qc["node_figs"][name]}" alt="{name} QC">')
            html.append(f'<img src="qc/{qc["design_psd"]}" alt="design variable PSD aliasing check">')
            html.append('</div>')
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)


def render_dyad_panel_mvar_order(dyad_id, entries):
    """Render one Stage 3 dyad's QC panel (one film-block per case) as an HTML fragment."""
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    for entry in entries:
        badge_class = "badge-ok" if entry["quality_ok"] else "badge-bad"
        badge_text = "quality_ok" if entry["quality_ok"] else "quality_fail"
        html.append(f'<div class="film-block"><h3>{entry["film"]} (group={entry["group"]})</h3>')
        html.append(
            f'<div class="header-line">n_windows={entry["n_windows"]}  p_used={entry["p_used"]}  '
            f'max_abs_root={entry["max_abs_root"]:.3f}  stable={entry["stable"]}  '
            f'min_white_fraction={entry["min_white_fraction"]:.2f}  '
            f'<span class="badge {badge_class}">{badge_text}</span></div>'
        )
        html.append('<div class="row">')
        html.append(f'<img src="qc/{entry["order_curves"]}" alt="order curves">')
        html.append(f'<img src="qc/{entry["roots"]}" alt="AR roots global vs windowed">')
        html.append(f'<img src="qc/{entry["acf"]}" alt="residual ACF global vs windowed">')
        html.append(f'<img src="qc/{entry["detrend"]}" alt="pre vs post detrend">')
        html.append(f'<img src="qc/{entry["mvar_grid"]}" alt="ffDTF grid" class="mvar-grid">')
        html.append('</div></div>')
    html.append('</div>')
    return "\n".join(html)


def render_dyad_panel_ffdtf(dyad_id, entries):
    """Render one Stage 4 dyad's QC panel (one film-block per case) as an HTML fragment."""
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    for entry in entries:
        badge_class = "badge-ok" if entry["quality_ok"] else "badge-bad"
        badge_text = "quality_ok" if entry["quality_ok"] else "quality_fail (Stage 3)"
        edge_text = "  ".join(f"{edge}={value:.3f}" for edge, value in entry["edge_values"].items())
        html.append(f'<div class="film-block"><h3>{entry["film"]} (group={entry["group"]})</h3>')
        html.append(
            f'<div class="header-line">p_used={entry["p_used"]}  window={entry["win_len"]}/{entry["step"]} samp  '
            f'Granger_estimator range=[{entry["Granger_estimator_min"]:.3f}, {entry["Granger_estimator_max"]:.3f}]  '
            f'max_rowsum_dev={entry["max_rowsum_dev"]:.2e}  '
            f'<span class="badge {badge_class}">{badge_text}</span></div>'
        )
        html.append(f'<div class="header-line">{edge_text}</div>')
        html.append(f'<div class="row"><img src="qc/{entry["grid_image"]}" alt="Granger_estimator grid"></div>')
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)


def render_edge_table(rows):
    """Render one film's 12-edge table (real / null / delta / z) as an HTML fragment."""
    html = ['<table class="edges"><tr><th>edge</th><th>class</th><th>real</th><th>null_median</th>'
            '<th>null_std</th><th>delta_dtf</th><th>z_vs_surrogate</th></tr>']
    for row in rows:
        row_class = "other" if row["edge_class"] == "other" else "emphasis" if row["edge_class"] in ("H2_primary", "H2_reverse", "H4_primary", "H4_reverse") else ""
        html.append(
            f'<tr class="{row_class}"><td>{row["edge"]}</td><td>{row["edge_class"]}</td>'
            f'<td>{row["real"]:.4f}</td><td>{row["null_median"]:.4f}</td><td>{row["null_std"]:.4f}</td>'
            f'<td>{row["delta"]:+.4f}</td><td>{row["z"]:+.2f}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_dyad_panel_surrogate(dyad_id, entries):
    """Render one Stage 5 dyad's QC panel (one film-block per case) as an HTML fragment."""
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    for entry in entries:
        badge_class = "badge-ok" if entry["real_stable"] else "badge-bad"
        badge_text = "real_stable" if entry["real_stable"] else "real_UNSTABLE (p=4)"
        html.append(f'<div class="film-block"><h3>{entry["film"]} (group={entry["group"]})</h3>')
        html.append(
            f'<div class="header-line">max_abs_root={entry["max_abs_root"]:.3f}  '
            f'<span class="badge {badge_class}">{badge_text}</span></div>'
        )
        html.append(render_edge_table(entry["rows"]))
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)


def render_diagnostics_table(rows):
    """Render Stage 6 `stage06_diagnostics.csv` rows as an HTML table with L9 pass/fail badges."""
    html = ['<table class="diag"><tr><th>model</th><th>n</th><th>dropped</th><th>max_rhat</th>'
            '<th>min_ess_bulk</th><th>min_ess_tail</th><th>n_divergent</th><th>max_pareto_k</th>'
            '<th>loo_elpd</th><th>status</th></tr>']
    for row in rows:
        badge_class = "badge-ok" if row["pass_l9"] else "badge-bad"
        badge_text = "PASS" if row["pass_l9"] else "FAIL"
        html.append(
            f'<tr><td>{row["model"]}</td><td>{row["n_rows"]}</td><td>{row["n_dropped_unstable"]}</td>'
            f'<td>{row["max_rhat"]:.3f}</td><td>{row["min_bulk_ess"]:.0f}</td><td>{row["min_tail_ess"]:.0f}</td>'
            f'<td>{row["n_divergent"]}</td><td>{row["max_pareto_k"]:.2f}</td><td>{row["loo_elpd"]:.1f}</td>'
            f'<td><span class="badge {badge_class}">{badge_text}</span></td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_contrast_table(rows_df):
    """Render a set of Stage 6 `stage06_contrasts.csv` rows (one edge, raw units) as an HTML table."""
    html = ['<table class="contrasts"><tr><th>contrast</th><th>estimate</th><th>HDI95 low</th>'
            '<th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th></tr>']
    for _, row in rows_df.iterrows():
        html.append(
            f'<tr><td>{row["contrast"]}</td><td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td>'
            f'<td>{row["hdi_high"]:+.5f}</td><td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_primary_table(df):
    """Render Stage 6 `stage06_primary_summary.csv` (both families) as an HTML table, with BH-FDR shown for the primary family."""
    html = ['<table class="contrasts"><tr><th>label</th><th>dv</th><th>unit</th><th>estimate</th>'
            '<th>HDI95 low</th><th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th><th>bh_fdr</th></tr>']
    for _, row in df.iterrows():
        bh_text = f'{row["bh_fdr"]:.3f}' if pd.notna(row["bh_fdr"]) else "&mdash;"
        html.append(
            f'<tr><td>{row["label"]}</td><td>{row["dv"]}</td><td>{row["unit"]}</td>'
            f'<td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td><td>{row["hdi_high"]:+.5f}</td>'
            f'<td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td><td>{bh_text}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_edge_panel(edge, edge_class, contrasts_df, safe_label_fn):
    """Render one Stage 6 emphasis edge's QC panel: forest/ppcheck/loo images + its contrast table.

    Parameters
    ----------
    edge, edge_class : str
    contrasts_df : pd.DataFrame
        Full `stage06_contrasts.csv` rows (filtered here to this edge, raw unit).
    safe_label_fn : callable
        `src.io_utils.safe_label`, passed in rather than imported here to
        keep this module free of a cross-import on `io_utils` for one call.
    """
    label = safe_label_fn(edge)
    rows_df = contrasts_df[(contrasts_df["edge"] == edge) & (contrasts_df["unit"] == "raw")]
    tag = "hypothesis-generating -- no correction" if edge_class in ("exploratory",) else edge_class
    html = [f'<div class="edge-panel" id="panel-{edge}"><h2>{edge} <small>({tag})</small></h2>']
    html.append('<div class="row">'
                f'<img src="qc/{label}_forest.png" alt="{edge} forest">'
                f'<img src="qc/{label}_ppcheck.png" alt="{edge} pp_check">'
                f'<img src="qc/{label}_loo.png" alt="{edge} pareto-k">'
                '</div>')
    html.append(render_contrast_table(rows_df))
    html.append("</div>")
    return "\n".join(html)


def render_localization_table(df):
    """Render a set of D6 (Stage 6) per-edge/population interaction rows (adds edge/edge_class to render_contrast_table's columns)."""
    rows_html = ['<table class="contrasts"><tr><th>edge</th><th>edge_class</th><th>contrast</th><th>estimate</th>'
                 '<th>HDI95 low</th><th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th></tr>']
    for _, row in df.iterrows():
        rows_html.append(
            f'<tr><td>{row["edge"]}</td><td>{row["edge_class"]}</td><td>{row["contrast"]}</td>'
            f'<td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td><td>{row["hdi_high"]:+.5f}</td>'
            f'<td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td></tr>'
        )
    rows_html.append("</table>")
    return "\n".join(rows_html)
