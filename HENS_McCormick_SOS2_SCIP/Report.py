"""
Report.py

Generates a to-the-point, industrial-style PDF engineering report for a
solved HENS case: executive summary, stream/utility data, pinch summary,
the final heat-exchanger network (topology, sizing, cost), temperature
profiles, the HEN diagram, the MILP linearization error analysis (model vs.
true LMTD/area/cost, plus 3D PWL fit diagnostic plots) — followed by an
appendix with every governing equation used by the tool and a short
explanation of the solution procedure.

Usage
-----
    from Report import build_hens_report
    pdf_bytes = build_hens_report(
        results, edited_df, util_df,
        delta_tmin=delta_tmin, qh=qh, qc=qc, pinch_temp=pinch_temp,
        cost_a=cost_a, cost_b=cost_b, cost_beta=cost_beta,
        svg=svg_hen, milp_diagnostics=milp_diagnostics,
    )
    # pdf_bytes is a BytesIO ready for st.download_button / embedding.

No external binaries are required (pure-Python: reportlab + svglib).
"""

import io
import os
from datetime import datetime

# Force the non-interactive Agg backend before pyplot gets imported (here,
# via H_HP_Tester below) -- this module can be used headless
# (e.g. from a server-side Streamlit process) and must never try to open a
# GUI window. Matplotlib's backend is a process-wide singleton set on the
# first pyplot import anywhere, so if App.py (or whatever imports this
# module) hasn't already set it, this is the fallback that guarantees it.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image,
)

# FIX: this module was renamed to H_HP_Tester.py (Pre_process.py and
# Solve_extract.py both already import from that name); the old module
# name below no longer exists, so this raised ModuleNotFoundError the
# instant Report.py itself was imported. plot_3d_comparison/plot_error_surface
# also changed signature in the rename -- from taking a pre-computed "dense"
# meshgrid dict to taking (dT_lo, dT_hi, planes, beta, U) directly -- see the
# _pwl_fit_diagnostics_section() fix below for the corresponding update.
from H_HP_Tester import plot_3d_comparison, plot_error_surface

try:
    from svglib.svglib import svg2rlg
    _HAS_SVGLIB = True
except Exception:
    _HAS_SVGLIB = False


# ─────────────────────────────────────────────────────────────────────────
# Unicode font registration
#
# The report uses Δ, β, °, ², →, ✔ etc. Reportlab's built-in Helvetica/
# Courier only cover WinAnsi (Latin-1-ish), which drops Greek letters and
# arrows silently. DejaVu Sans covers all of them and ships with
# matplotlib (already a dependency here), so we register it if found and
# fall back to Helvetica/Courier (still fully functional, just missing a
# few glyphs) if it isn't available on the host.
# ─────────────────────────────────────────────────────────────────────────

_FONT_REG, _FONT_BOLD, _FONT_MONO = "Helvetica", "Helvetica-Bold", "Courier"


def _register_unicode_fonts():
    global _FONT_REG, _FONT_BOLD, _FONT_MONO
    search_dirs = ["/usr/share/fonts/truetype/dejavu"]
    try:
        import matplotlib
        search_dirs.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except Exception:
        pass

    for d in search_dirs:
        reg = os.path.join(d, "DejaVuSans.ttf")
        bold = os.path.join(d, "DejaVuSans-Bold.ttf")
        mono = os.path.join(d, "DejaVuSansMono.ttf")
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("DejaVuSans", reg))
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))
                _FONT_REG, _FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
                if os.path.exists(mono):
                    pdfmetrics.registerFont(TTFont("DejaVuSansMono", mono))
                    _FONT_MONO = "DejaVuSansMono"
                else:
                    _FONT_MONO = "DejaVuSans"
                return True
            except Exception:
                continue
    return False


_register_unicode_fonts()


# ─────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────

def _styles():
    ss = getSampleStyleSheet()
    for name in ("Normal", "BodyText", "Title", "Heading1", "Heading2", "Code"):
        ss[name].fontName = _FONT_BOLD if name in ("Title", "Heading1", "Heading2") else _FONT_REG
    ss["Code"].fontName = _FONT_MONO

    ss.add(ParagraphStyle("ReportTitle", parent=ss["Title"], fontSize=20,
                           fontName=_FONT_BOLD, spaceAfter=4,
                           textColor=colors.HexColor("#1a1a2e")))
    ss.add(ParagraphStyle("ReportSubtitle", parent=ss["Normal"], fontSize=11,
                           fontName=_FONT_REG,
                           textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                           spaceAfter=2))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, fontName=_FONT_BOLD,
                           textColor=colors.HexColor("#1a1a2e"),
                           spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, fontName=_FONT_BOLD,
                           textColor=colors.HexColor("#2a2a4e"),
                           spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.3, fontName=_FONT_REG,
                           leading=13, alignment=TA_LEFT))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8, leading=11,
                           fontName=_FONT_REG,
                           textColor=colors.HexColor("#444444")))
    ss.add(ParagraphStyle("Eq", parent=ss["Code"], fontSize=9, leading=13, fontName=_FONT_MONO,
                           leftIndent=14, spaceBefore=2, spaceAfter=6))
    ss.add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=8, fontName=_FONT_REG,
                           alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                           spaceBefore=2, spaceAfter=10))
    ss.add(ParagraphStyle("Cell", parent=ss["Normal"], fontSize=8, fontName=_FONT_REG,
                           leading=10))
    ss.add(ParagraphStyle("CellHead", parent=ss["Normal"], fontSize=8, fontName=_FONT_BOLD,
                           leading=10, textColor=colors.white))
    return ss


_TABLE_HEAD = colors.HexColor("#1a1a2e")
_TABLE_ALT = colors.HexColor("#f2f2f7")


def _table(data, col_widths=None, header=True, font_size=8, wrap=False, styles=None):
    """wrap=True renders every cell as a Paragraph so long text wraps
    inside its column instead of overflowing the page."""
    if wrap:
        ss = styles or _styles()
        cell_style = ParagraphStyle("_c", parent=ss["Cell"], fontSize=font_size, leading=font_size + 2)
        head_style = ParagraphStyle("_h", parent=ss["CellHead"], fontSize=font_size, leading=font_size + 2)
        wrapped = []
        for r_idx, row in enumerate(data):
            style = head_style if (header and r_idx == 0) else cell_style
            wrapped.append([Paragraph(str(v), style) for v in row])
        data = wrapped

    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT_REG),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, _TABLE_ALT]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEAD),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ]
    t.setStyle(TableStyle(style))
    return t


def _fmt(v, spec="{:.2f}", dash="—"):
    if v is None:
        return dash
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────

def _cover(story, ss, meta):
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Heat Exchanger Network Synthesis", ss["ReportTitle"]))
    story.append(Paragraph("Engineering Design &amp; Cost Report", ss["ReportSubtitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", color=colors.HexColor("#1a1a2e"), thickness=1))
    story.append(Spacer(1, 1 * cm))
    label_style = ParagraphStyle("_cvl", fontName=_FONT_BOLD, fontSize=10, leading=13)
    val_style = ParagraphStyle("_cvv", fontName=_FONT_REG, fontSize=10, leading=13)
    rows = [
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["\u0394Tmin", f"{meta.get('delta_tmin', '—')} \u00b0C"],
        ["Process streams", f"{meta.get('n_hot', '—')} hot / {meta.get('n_cold', '—')} cold"],
        ["Total Annualized Cost (TAC)", f"${meta.get('tac', 0):,.0f} / yr"],
        ["Design basis", meta.get("basis", "MILP superstructure + NLP exact-equation refinement")],
    ]
    rows = [[Paragraph(a, label_style), Paragraph(b, val_style)] for a, b in rows]
    t = Table(rows, colWidths=[6 * cm, 9.5 * cm])
    t.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(t)
    story.append(PageBreak())


def _executive_summary(story, ss, results, meta):
    story.append(Paragraph("1. Executive Summary", ss["H1"]))
    n_proc = len([e for e in results.get("edges", []) if e.get("Q", 0) > 1.0])
    n_util = len(results.get("util_hex_edges", []))
    txt = (
        f"This report documents the synthesized heat exchanger network (HEN) for "
        f"{meta.get('n_hot', '—')} hot and {meta.get('n_cold', '—')} cold process streams, "
        f"subject to a minimum approach temperature ΔT<sub>min</sub> = {meta.get('delta_tmin', '—')} °C. "
        f"The final network uses {n_proc} process-process exchanger(s) and {n_util} utility "
        f"exchanger(s), for a Total Annualized Cost (TAC) of ${results.get('TAC', 0):,.0f}/yr."
    )
    story.append(Paragraph(txt, ss["Body"]))
    story.append(Spacer(1, 6))

    kpi_rows = [
        ["Metric", "Value"],
        ["TAC ($/yr)", f"${results.get('TAC', 0):,.0f}"],
        ["Utility Operating Cost ($/yr)", f"${results.get('ann_util_cost', 0):,.0f}"],
        ["Process HEX Capital ($/yr)", f"${results.get('ann_cap_process', 0):,.0f}"],
        ["Utility HEX Capital ($/yr)", f"${results.get('ann_cap_util_hex', 0):,.0f}"],
        ["Hot Utility Duty (kW)", _fmt(sum(results.get("QH", [0])))],
        ["Cold Utility Duty (kW)", _fmt(sum(results.get("QC", [0])))],
        ["Pinch Temperature (°C)", _fmt(meta.get("pinch_temp"))],
        ["Min. Hot Utility, QH_min (kW)", _fmt(meta.get("qh"))],
        ["Min. Cold Utility, QC_min (kW)", _fmt(meta.get("qc"))],
    ]
    story.append(_table(kpi_rows, col_widths=[8 * cm, 6 * cm]))
    story.append(Spacer(1, 8))


def _stream_tables(story, ss, edited_df, util_df, meta):
    story.append(Paragraph("2. Process &amp; Utility Stream Data", ss["H1"]))
    story.append(Paragraph("2.1 Process Streams", ss["H2"]))
    header = ["ID", "Tin (°C)", "Tout (°C)", "CP (kW/°C)", "h (kW/m²·°C)"]
    rows = [header]
    for _, r in edited_df.iterrows():
        rows.append([
            str(r.get("Stream ID", "")), _fmt(r.get("Tin")), _fmt(r.get("Tout")),
            _fmt(r.get("CP")), _fmt(r.get("h (kW/m²·°C)"), "{:.3f}"),
        ])
    story.append(_table(rows, col_widths=[3 * cm, 3 * cm, 3 * cm, 3.2 * cm, 3.2 * cm]))
    story.append(Spacer(1, 8))

    if util_df is not None and len(util_df):
        story.append(Paragraph("2.2 Utility Streams", ss["H2"]))
        header = ["ID", "Tin (°C)", "Tout (°C)", "cp (kJ/kg·°C)", "Cost ($/kW·yr)"]
        rows = [header]
        for _, r in util_df.iterrows():
            rows.append([
                str(r.get("ID", "")), _fmt(r.get("Tin (°C)")), _fmt(r.get("Tout (°C)")),
                _fmt(r.get("cp (kJ/kg·°C)")), _fmt(r.get("Cost ($/kW·yr)")),
            ])
        story.append(_table(rows, col_widths=[2.6 * cm, 2.8 * cm, 2.8 * cm, 3.4 * cm, 3.4 * cm]))
        story.append(Spacer(1, 8))


def _pinch_section(story, ss, meta):
    story.append(Paragraph("3. Pinch Analysis Summary", ss["H1"]))
    txt = (
        f"Problem-table cascade analysis (Linnhoff &amp; Flower) at ΔT<sub>min</sub> = "
        f"{meta.get('delta_tmin', '—')} °C locates the process pinch at "
        f"{_fmt(meta.get('pinch_temp'))} °C, with minimum utility targets of "
        f"{_fmt(meta.get('qh'))} kW hot utility and {_fmt(meta.get('qc'))} kW cold utility. "
        f"These targets set the thermodynamic lower bound that the MILP superstructure "
        f"is optimized against."
    )
    story.append(Paragraph(txt, ss["Body"]))
    story.append(Spacer(1, 8))


def _network_section(story, ss, results):
    story.append(Paragraph("4. Heat Exchanger Network", ss["H1"]))

    story.append(Paragraph("4.1 Process Heat Exchangers", ss["H2"]))
    edges = [e for e in results.get("edges", []) if e.get("Q", 0) > 1.0]
    if edges:
        header = ["Hot", "Cold", "Stage", "Q (kW)", "LMTD (°C)", "Area (m²)", "CapEx ($/yr)"]
        rows = [header]
        for e in edges:
            rows.append([
                e.get("hot", ""), e.get("cold", ""), str(e.get("stage", "")),
                _fmt(e.get("Q"), "{:.1f}"), _fmt(e.get("LMTD"), "{:.1f}"),
                _fmt(e.get("Area_m2"), "{:.1f}"), _fmt(e.get("CapCost_$"), "{:,.0f}"),
            ])
        story.append(_table(rows, col_widths=[2.3 * cm, 2.3 * cm, 1.6 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 3 * cm]))
    else:
        story.append(Paragraph("No process-process exchangers are active in this network.", ss["Body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.2 Utility Heat Exchangers", ss["H2"]))
    uedges = results.get("util_hex_edges", [])
    if uedges:
        header = ["Utility", "Process Stream", "Q (kW)", "LMTD (°C)", "Area (m²)", "CapEx ($/yr)"]
        rows = [header]
        for e in uedges:
            stream = e.get("cold") if e.get("type") == "hot" else e.get("hot")
            rows.append([
                str(e.get("utility", "")), str(stream),
                _fmt(e.get("Q"), "{:.1f}"), _fmt(e.get("LMTD"), "{:.1f}"),
                _fmt(e.get("Area_m2"), "{:.1f}"), _fmt(e.get("CapCost_$"), "{:,.0f}"),
            ])
        story.append(_table(rows, col_widths=[2.6 * cm, 3.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    else:
        story.append(Paragraph("No utility exchangers are active in this network.", ss["Body"]))
    story.append(Spacer(1, 8))


def _temperature_profiles(story, ss, results):
    story.append(Paragraph("5. Stream Temperature Profiles", ss["H1"]))
    HIDs, CIDs = results.get("HIDs", []), results.get("CIDs", [])
    T_hot, T_cold = results.get("T_hot", []), results.get("T_cold", [])
    rows = [["Stream", "Temperature profile across stages (°C)"]]
    for hid, temps in zip(HIDs, T_hot):
        rows.append([hid, " → ".join(f"{t:.1f}" for t in temps)])
    for cid, temps in zip(CIDs, T_cold):
        rows.append([cid, " → ".join(f"{t:.1f}" for t in temps)])
    if len(rows) > 1:
        story.append(_table(rows, col_widths=[2.5 * cm, 13.5 * cm], font_size=7.6, wrap=True, styles=ss))
    story.append(Spacer(1, 8))


def _diagram_section(story, ss, svg):
    if not svg or not _HAS_SVGLIB:
        return
    story.append(Paragraph("6. HEN Diagram (Grid Representation)", ss["H1"]))
    try:
        drawing = svg2rlg(io.BytesIO(svg.encode("utf-8")))
        if drawing is None:
            return
        max_w = 17 * cm
        if drawing.width > 0:
            scale = min(1.0, max_w / drawing.width)
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
        story.append(drawing)
        story.append(Paragraph(
            "Grid diagram (Yee–Grossmann style): hot streams flow left→right, cold streams "
            "right→left; stage boundaries are shared by all streams.", ss["Caption"]))
    except Exception as exc:
        story.append(Paragraph(f"(Diagram could not be embedded: {exc})", ss["Small"]))
    story.append(Spacer(1, 8))


def _error_diagnostics(story, ss, milp_diagnostics):
    """
    MILP linearization error analysis: how far the MILP's own piecewise-
    linear / outer-approximation sizing (LMTD, area, cost) was from the
    exact equations evaluated at that same incumbent's own (dT1, dT2, Q) --
    i.e. the same "Model vs. True" numbers used to decide whether topology
    refinement was needed in the first place. Takes the MILP-STAGE
    diagnostics dict specifically (not the final, possibly NLP-refined,
    `results`): once the NLP has re-solved with the exact equations there
    is no more "model vs true" gap to report, so this section is only
    meaningful against the MILP incumbent that fed into it.
    """
    if not milp_diagnostics:
        return
    summ = milp_diagnostics.get("error_summary")
    edges = milp_diagnostics.get("edges") or []
    if not summ and not edges:
        return

    story.append(Paragraph("7. MILP Linearization Error Analysis", ss["H1"]))
    txt = (
        "The MILP sizes exchangers through piecewise-linear / outer-approximation fits of "
        "the true LMTD, area, and cost equations. This section reports, for the MILP's own "
        "incumbent, how far each fit's <b>Model</b> value differs from the <b>True</b> value "
        "obtained by plugging that same incumbent's own (dT1, dT2, Q) into the exact "
        "nonlinear equations -- the same check used to decide whether topology refinement "
        "was needed."
    )
    story.append(Paragraph(txt, ss["Body"]))
    story.append(Spacer(1, 6))

    milp_obj = milp_diagnostics.get("milp_obj")
    tac_true = milp_diagnostics.get("TAC_true")
    if milp_obj is not None and tac_true:
        obj_err_pct = abs(milp_obj - tac_true) / max(tac_true, 1.0) * 100
        story.append(Paragraph(
            f"Linearized MILP objective: <b>${milp_obj:,.0f}/yr</b> &nbsp;|&nbsp; "
            f"True TAC of this incumbent (exact equations): <b>${tac_true:,.0f}/yr</b> "
            f"&nbsp;|&nbsp; Objective error: <b>{obj_err_pct:.2f}%</b>", ss["Body"]))
        story.append(Spacer(1, 6))

    if summ:
        rows = [
            ["Quantity", "Max error", "Mean error"],
            ["LMTD", f"{summ.get('max_lmtd_err_pct', 0):.2f}%", f"{summ.get('mean_lmtd_err_pct', 0):.2f}%"],
            ["Area", f"{summ.get('max_area_err_pct', 0):.2f}%", f"{summ.get('mean_area_err_pct', 0):.2f}%"],
            ["Cost", f"{summ.get('max_cost_err_pct', 0):.2f}%", f"{summ.get('mean_cost_err_pct', 0):.2f}%"],
        ]
        story.append(_table(rows, col_widths=[6 * cm, 5 * cm, 5 * cm]))
        story.append(Spacer(1, 8))

    if edges and "err_lmtd_pct" in edges[0]:
        story.append(Paragraph("Per-Exchanger Breakdown", ss["H2"]))
        header = ["Exchanger", "Stage", "Q (kW)", "LMTD Model/True (°C)",
                  "Area Model/True (m²)", "Cost Model/True ($)", "Cost Err (%)"]
        rows = [header]
        for e in edges:
            rows.append([
                f"{e.get('hot_id', '?')} \u2192 {e.get('cold_id', '?')}",
                str(e.get("stage", "\u2014")),
                _fmt(e.get("Q"), "{:.1f}"),
                f"{_fmt(e.get('LMTD_model'), '{:.2f}')} / {_fmt(e.get('LMTD_true'), '{:.2f}')}",
                f"{_fmt(e.get('Area_m2_model'), '{:.1f}')} / {_fmt(e.get('Area_m2_true'), '{:.1f}')}",
                f"{_fmt(e.get('Cost_model_$'), '{:,.0f}')} / {_fmt(e.get('Cost_true_$'), '{:,.0f}')}",
                _fmt(e.get("err_cost_pct"), "{:.1f}"),
            ])
        story.append(_table(rows, col_widths=[3*cm, 1.3*cm, 1.7*cm, 3.2*cm, 3.2*cm, 3.3*cm, 1.9*cm]))
        story.append(Spacer(1, 8))


def _pwl_fit_diagnostics_section(story, ss, milp_diagnostics, max_matches=8):
    """
    Embeds the pre-solve PWL/outer-approximation fit-quality 3D plots --
    true surface vs. tangent-plane envelope, and the relative-error
    surface -- for every active process exchanger's (i,j) match. This is
    exactly the same validation Pre_process.py already runs once per
    match before the MILP is even built (H_HP_Tester.py's select_H_envelope
    validation); its per-match stats are carried through in
    milp_diagnostics["h_diag"].

    FIX (two bugs, both from the LMTD -> H redesign):
      1. This read milp_diagnostics["lmtd_diag"], a key that no longer
         exists (Solve_extract.py renamed it to "h_diag" -- see its own
         comment: "Renamed from lmtd_diag/data.lmtd_diag ... since
         lmtd_diag no longer exists there"). `.get(...) or {}` meant this
         never raised, it just silently returned `{}` every time, so this
         entire report section was permanently and silently omitted from
         every PDF, forever, with no error to flag it.
      2. Even keyed correctly, each match's diag dict here is
         `h_diag[i, j]` from Pre_process.py, which -- unlike the old
         lmtd_diag -- does NOT carry a ready-made "dense" meshgrid (its
         own comment flags this explicitly: "Report.py's 3D true-vs-
         envelope plotting will need a small update"). Regenerating the
         true 3D surfaces needs each match's `planes` (tangent-plane
         coefficients) and its (dT_lo, dT_hi) feasible range -- neither
         of which is in milp_diagnostics at all (they live only on the
         `data` namespace inside Pre_process.py, which build_hens_report()
         is never given). Rather than fabricate that data or silently
         drop the section again, this now renders what IS available --
         each active match's fit-quality stats (max/avg relative error,
         worst point, plane count, sign violations) as a table -- and
         only attempts the 3D re-plot when a caller has separately
         attached "planes"/"dT_lo"/"dT_hi" onto that match's h_diag entry
         (a natural follow-up: thread those three fields through from
         Pre_process.py's h_planes/dT1_lo/dT1_hi into h_diag[i, j] --
         U and beta are already recoverable here from milp_diagnostics'
         own "U_matrix"/"cost_beta").
    """
    if not milp_diagnostics:
        return
    h_diag = milp_diagnostics.get("h_diag") or {}
    edges = milp_diagnostics.get("edges") or []
    if not h_diag or not edges:
        return

    HIDs = milp_diagnostics.get("HIDs", [])
    CIDs = milp_diagnostics.get("CIDs", [])
    hid_to_i = {h: idx for idx, h in enumerate(HIDs)}
    cid_to_j = {c: idx for idx, c in enumerate(CIDs)}
    U_matrix = milp_diagnostics.get("U_matrix")
    cost_beta = milp_diagnostics.get("cost_beta")

    # De-duplicate by (i,j): several stages can reuse the same match.
    seen = {}
    for e in edges:
        i = hid_to_i.get(e.get("hot_id"))
        j = cid_to_j.get(e.get("cold_id"))
        if i is None or j is None:
            continue
        diag = h_diag.get((i, j))
        if diag is None:
            continue
        seen.setdefault((i, j), (e.get("hot_id"), e.get("cold_id"), diag))

    if not seen:
        return

    story.append(PageBreak())
    story.append(Paragraph("8. PWL Fit Diagnostics \u2014 H Envelope Quality", ss["H1"]))
    story.append(Paragraph(
        "For every active process match, this reports how closely the tangent-plane "
        "outer-approximation of H = (U\u00b7LMTD)^-\u03b2 (the piece that gets linearized for "
        "exchanger cost) tracks the true convex function over that match's full feasible "
        "(dT1, dT2) range. This is the same validation Pre_process.py runs before the MILP "
        "is built, summarized here for the record.", ss["Body"]))
    story.append(Spacer(1, 6))

    n_shown = 0
    items = list(seen.items())
    for (i, j), (hot_id, cold_id, diag) in items:
        if n_shown >= max_matches:
            story.append(Paragraph(
                f"({len(items) - max_matches} additional match(es) omitted for report length.)",
                ss["Body"]))
            break
        n_shown += 1

        story.append(Paragraph(f"Match: {hot_id} \u2192 {cold_id}", ss["H2"]))
        max_rel_err = diag.get("max_rel_err")
        avg_rel_err = diag.get("avg_rel_err")
        rows = [
            ["Max relative error", f"{max_rel_err*100:.3f}%" if max_rel_err is not None else "\u2014"],
            ["Avg relative error", f"{avg_rel_err*100:.3f}%" if avg_rel_err is not None else "\u2014"],
            ["Worst point (dT1, dT2)", str(diag.get("worst_point", "\u2014"))],
            ["Planes used", str(diag.get("n_planes", "\u2014"))],
            ["Sign violations", str(diag.get("n_sign_violations", "\u2014"))],
        ]
        story.append(_table(rows, col_widths=[6 * cm, 8 * cm]))
        story.append(Spacer(1, 6))

        # Optional 3D re-plot -- only possible if a caller has attached
        # "planes"/"dT_lo"/"dT_hi" onto this match's h_diag entry (not
        # supplied by Solve_extract.py's results dict today; see docstring).
        planes = diag.get("planes")
        dT_lo = diag.get("dT_lo")
        dT_hi = diag.get("dT_hi")
        if planes is None or dT_lo is None or dT_hi is None:
            story.append(Spacer(1, 4))
            continue

        beta_ij = cost_beta[i, j] if isinstance(cost_beta, dict) else cost_beta
        U_ij = (U_matrix[i][j] if U_matrix is not None else 1.0)

        for fig, caption in (
            (plot_3d_comparison(dT_lo, dT_hi, planes, beta_ij, U_ij,
                                 title=f"{hot_id} \u2192 {cold_id}",
                                 angles=((25, -60), (60, -45))),
             "True surface (viridis) vs. tangent-plane envelope (red wireframe)"),
            (plot_error_surface(dT_lo, dT_hi, planes, beta_ij, U_ij,
                                 angles=((25, -60),)),
             "Relative approximation error (%)"),
        ):
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            img_buf.seek(0)

            w_in, h_in = fig.get_size_inches()
            disp_w = 16 * cm
            disp_h = disp_w * (h_in / w_in)
            story.append(Image(img_buf, width=disp_w, height=disp_h))
            story.append(Paragraph(caption, ss["Caption"]))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 4))


# ─────────────────────────────────────────────────────────────────────────
# Appendix — equations & methodology
# ─────────────────────────────────────────────────────────────────────────

def _appendix(story, ss):
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Nomenclature", ss["H1"]))
    nomen = [
        ["Symbol", "Meaning"],
        ["i, j", "Hot process stream index / cold process stream index"],
        ["k, S", "Stage index (0-based) / number of stages"],
        ["u, v", "Hot utility index / cold utility index"],
        ["TH[i,k], TC[j,k]", "Hot / cold stream temperature at stage boundary k"],
        ["Q[i,j,k]", "Heat duty exchanged between hot i and cold j in stage k"],
        ["QH[j], QC[i]", "Hot utility duty on cold stream j / cold utility duty on hot stream i"],
        ["z[i,j,k]", "Binary: 1 if match (i,j) is active in stage k"],
        ["yHU[u,j], yCU[v,i]", "Binary: 1 if utility u (or v) is assigned to stream j (or i)"],
        ["dT1, dT2", "Approach temperatures at the hot-end / cold-end of an exchanger"],
        ["LMTD", "Log-mean temperature difference of an exchanger"],
        ["U", "Overall heat transfer coefficient"],
        ["A", "Heat transfer area"],
        ["ΔTmin", "Minimum allowed approach temperature"],
        ["cost_a, cost_b, β", "Fixed cost, area cost coefficient, and area cost exponent"],
        ["TAC", "Total Annualized Cost = utility OPEX + annualized HEX CAPEX"],
    ]
    story.append(_table(nomen, col_widths=[4.5 * cm, 11.5 * cm], font_size=8.3, wrap=True, styles=ss))

    story.append(Paragraph("Appendix B — Governing Equations", ss["H1"]))

    story.append(Paragraph("B.1 Superstructure (Yee &amp; Grossmann stage-wise model)", ss["H2"]))
    story.append(Paragraph(
        "The network is represented as a stage-wise superstructure in which every hot stream "
        "may exchange heat with every cold stream in every stage; binaries z select which "
        "matches are actually built. Overall and per-stage energy balances:", ss["Body"]))
    for eq in [
        "Overall hot balance:   Σ_j Σ_k Q[i,j,k] + QC[i] = CP_H[i]·(Tin_H[i] − Tout_H[i])   ∀ i",
        "Overall cold balance:  Σ_i Σ_k Q[i,j,k] + QH[j] = CP_C[j]·(Tout_C[j] − Tin_C[j])    ∀ j",
        "Hot stage balance:     CP_H[i]·(TH[i,k] − TH[i,k+1]) = Σ_j Q[i,j,k]                ∀ i,k",
        "Cold stage balance:    CP_C[j]·(TC[j,k] − TC[j,k+1]) = Σ_i Q[i,j,k]                ∀ j,k",
        "Inlet fixing:          TH[i,0] = Tin_H[i]  ,   TC[j,K−1] = Tin_C[j]",
        "Monotonicity:          TH[i,k] ≥ TH[i,k+1]  ,   TC[j,k] ≥ TC[j,k+1]",
        "Outlet feasibility:    TH[i,K−1] ≥ Tout_H[i]  ,   TC[j,0] ≤ Tout_C[j]",
    ]:
        story.append(Paragraph(eq, ss["Eq"]))

    story.append(Paragraph("B.2 Minimum approach temperature &amp; big-M activation", ss["H2"]))
    for eq in [
        "Q[i,j,k] ≤ Q_max[i,j]·z[i,j,k]                                    (big-M activation)",
        "TH[i,k]   − TC[j,k]   ≥ ΔTmin − dT1_hi[i,j]·(1 − z[i,j,k])        (hot-end ΔTmin)",
        "TH[i,k+1] − TC[j,k+1] ≥ ΔTmin − dT1_hi[i,j]·(1 − z[i,j,k])        (cold-end ΔTmin)",
        "dT1[i,j,k] = TH[i,k]   − TC[j,k]     (relaxed to 0 when z = 0)",
        "dT2[i,j,k] = TH[i,k+1] − TC[j,k+1]   (relaxed to 0 when z = 0)",
    ]:
        story.append(Paragraph(eq, ss["Eq"]))

    story.append(Paragraph("B.3 Exact LMTD, area and capital cost", ss["H2"]))
    story.append(Paragraph(
        "Every active exchanger is sized by the classical log-mean temperature difference, "
        "resistance-in-series area law, and a power-law capital cost:", ss["Body"]))
    for eq in [
        "LMTD = (dT1 − dT2) / ln(dT1 / dT2)         (dT1 → dT2 limit: LMTD → dT1)",
        "A = Q / (U · LMTD)",
        "Cost = cost_a + cost_b · A^β                (annualized fixed + area-scaled capital)",
        "TAC = Σ (utility opex · Q) + Σ Cost_process + Σ Cost_utility",
    ]:
        story.append(Paragraph(eq, ss["Eq"]))

    story.append(Paragraph("B.4 MILP linearization of the cost/LMTD relations", ss["H2"]))
    story.append(Paragraph(
        "Cost is a non-convex function of two decision variables (Q, LMTD), so the MILP does "
        "not embed it directly; instead it is decomposed and each nonlinear piece is replaced "
        "by a convex/concave envelope or an SOS2 (Special Ordered Set, type 2) piecewise-linear "
        "surrogate, which the solver can handle exactly as a mixed-integer program:", ss["Body"]))
    for eq in [
        "−β·ln(LMTD) is convex in (dT1,dT2)  →  outer-approximated by supporting tangent "
        "hyperplanes: NegBetaLnLMTD ≥ a0 + a1·dT1 + a2·dT2  (tightest plane binds)",
        "ln(Q) is concave  →  overestimated by tangent lines, relaxed to vacuous when z = 0",
        "A_beta = β·ln(Q) − β·ln(U) + NegBetaLnLMTD   (so that A_beta = β·ln(A))",
        "Cost(A_beta) reconstructed via a native SOS2 interpolation over a fitted grid of "
        "(A_beta, Cost) breakpoints — at most two adjacent, non-zero weights are active",
        "Utility exchanger duty→cost curves use an equivalent 1-D SOS2 breakpoint interpolation "
        "over (Q, Cost) pairs.",
    ]:
        story.append(Paragraph(eq, ss["Eq"]))
    story.append(Paragraph(
        "This fit is very close but not exact; Section 8 (if present) reports the residual gap "
        "between the MILP's linearized sizing and the true thermodynamic values at the "
        "solution found.", ss["Small"]))

    story.append(Paragraph("B.5 Stage-2 NLP — exact-equation refinement", ss["H2"]))
    story.append(Paragraph(
        "Once the MILP fixes which matches/utility assignments are active (the topology), all "
        "binaries and piecewise-linear surrogates are dropped, and the fixed-topology problem "
        "is re-optimized as a smooth NLP using the exact equations:", ss["Body"]))
    for eq in [
        "LMTD_true(dT1,dT2) = (dT1 − dT2)/ln(dT1/dT2)      (exact log-mean, used for reporting)",
        "LMTD_smooth(dT1,dT2) = [dT1·dT2·(dT1+dT2)/2]^(1/3)   (Chen 1987 — smooth, differentiable "
        "surrogate solved inside the NLP; error vs. exact LMTD is typically < 1% over normal "
        "HEN approach-temperature ranges)",
        "Cost = cost_a + cost_b · A^β    (exact power law — no SOS2/DLOG needed once continuous)",
    ]:
        story.append(Paragraph(eq, ss["Eq"]))
    story.append(Paragraph(
        "The NLP is solved with IPOPT, warm-started from the MILP's own solution.",
        ss["Small"]))

    story.append(Paragraph("Appendix C — Solution Procedure", ss["H1"]))
    for i, (title, text) in enumerate([
        ("1. Pinch targeting",
         "Classify streams, shift temperatures by ΔTmin/2, build the problem-table cascade, "
         "and locate the pinch and minimum utility targets (QH_min, QC_min)."),
        ("2. MILP superstructure synthesis",
         "Build and solve the Yee–Grossmann stage-wise MILP (Appendix B.1–B.4) to select the "
         "network topology and an approximately optimal TAC, using DLOG/SOS2 piecewise-linear "
         "surrogates for the nonlinear LMTD/area/cost relations."),
        ("3. Stage-2 NLP refinement",
         "Fix the chosen topology and re-optimize duties, temperatures, areas and costs with "
         "the exact nonlinear equations (Appendix B.5), removing all linearization error from "
         "the final reported sizing and TAC."),
        ("4. Reporting",
         "Extract the final network, temperature profiles, HEN diagram, and cost breakdown "
         "into this report."),
    ]):
        story.append(Paragraph(title, ss["H2"]))
        story.append(Paragraph(text, ss["Body"]))


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def build_hens_report(results, edited_df=None, util_df=None, *,
                       delta_tmin=None, qh=None, qc=None, pinch_temp=None,
                       cost_a=None, cost_b=None, cost_beta=None,
                       svg=None, milp_diagnostics=None):
    """
    Build the full PDF report and return it as a BytesIO buffer (ready for
    st.download_button(..., data=buf, mime="application/pdf") or embedding).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="HENS Engineering Report",
    )
    ss = _styles()

    n_hot = n_cold = None
    if edited_df is not None and len(edited_df):
        n_hot = int((edited_df["Tin"] > edited_df["Tout"]).sum())
        n_cold = int((edited_df["Tin"] < edited_df["Tout"]).sum())

    meta = dict(
        delta_tmin=delta_tmin, n_hot=n_hot, n_cold=n_cold,
        tac=results.get("TAC", 0), pinch_temp=pinch_temp, qh=qh, qc=qc,
        basis="MILP superstructure + NLP exact-equation refinement",
    )

    story = []
    _cover(story, ss, meta)
    _executive_summary(story, ss, results, meta)
    if edited_df is not None:
        _stream_tables(story, ss, edited_df, util_df, meta)
    _pinch_section(story, ss, meta)
    _network_section(story, ss, results)
    _temperature_profiles(story, ss, results)
    _diagram_section(story, ss, svg)
    _error_diagnostics(story, ss, milp_diagnostics)
    _pwl_fit_diagnostics_section(story, ss, milp_diagnostics)
    _appendix(story, ss)

    doc.build(story)
    buf.seek(0)
    return buf