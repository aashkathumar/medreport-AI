import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.units import cm
from app.models.schemas import ExplainedResult

BLUE = HexColor("#003E74")
LGREY = HexColor("#F5F5F5")

# LLM output occasionally contains Unicode dash/hyphen variants (most often
# U+2011 NON-BREAKING HYPHEN, e.g.
_UNSUPPORTED_CHARS = {
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "‒": "-",  # FIGURE DASH
    "―": "-",  # HORIZONTAL BAR
    "−": "-",  # MINUS SIGN
}


def _sanitize(text):
    if not isinstance(text, str):
        return text
    for bad, good in _UNSUPPORTED_CHARS.items():
        text = text.replace(bad, good)
    return text


def generate_report_pdf(
    explained_results: list[ExplainedResult],
    overall_summary: str,
    top_gp_topics: list[str],
    top_lifestyle_change: str,
    closing_message: str,
    user_name: str = "Patient",
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    s = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=s["Normal"], fontSize=18, textColor=BLUE,
                              fontName="Helvetica-Bold", spaceAfter=4)
    head_s = ParagraphStyle("H", parent=s["Normal"], fontSize=13, textColor=BLUE,
                             fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
    body_s = ParagraphStyle("B", parent=s["Normal"], fontSize=10, leading=14, spaceAfter=4)
    disc_s = ParagraphStyle("D", parent=s["Normal"], fontSize=8,
                             textColor=HexColor("#666666"), leading=11)

    story = [
        Paragraph("MedReport AI", title_s),
        Paragraph("Your Personalised Lab Report Explanation", body_s),
        Paragraph(f"Prepared: {datetime.now().strftime('%d %B %Y')}", body_s),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8),
        Paragraph(
            "MedReport AI is a technical prototype for educational and GP-preparation purposes only. "
            "It does not provide medical diagnosis or replace consultation with a qualified GP.",
            disc_s,
        ),
        Spacer(1, 10),
        Paragraph("Overall Summary", head_s),
        Paragraph(_sanitize(overall_summary), body_s),
    ]

    if top_gp_topics:
        story.append(Paragraph("Topics to discuss with your GP:", body_s))
        for t in top_gp_topics:
            story.append(Paragraph(f"- {_sanitize(t)}", body_s))

    story.append(Paragraph(f"<b>Top lifestyle change:</b> {_sanitize(top_lifestyle_change)}", body_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY, spaceAfter=6))
    story.append(Paragraph("Your Results, Explained", head_s))

    # ETHICS CONSTRAINT: no status/range on ExplainedResult
    # here (see schemas.py), the banner shows the raw extracted value only
    # (what was on the report), never a normal/high/low judgement about it.
    for r in explained_results:
        banner = Table(
            [[
                Paragraph(f"<b>{_sanitize(r.raw_name)}</b>", body_s),
                Paragraph(f"<b>{r.value} {r.unit}</b>", body_s),
            ]],
            colWidths=["50%", "50%"],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LGREY),
            ("ROWPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 1.5, BLUE),
        ]))
        story += [
            banner, Spacer(1, 4),
            Paragraph(_sanitize(r.what_it_measures), body_s),
        ]
        if r.lifestyle_suggestions:
            story.append(Paragraph("<b>Lifestyle suggestions:</b>", body_s))
            for sug in r.lifestyle_suggestions:
                story.append(Paragraph(f"- {_sanitize(sug)}", body_s))
        # Cite the actual retrieved reference pages, not just a source label,
        # so a reader (or an examiner) can check any explanation against the
        # NHS/MedlinePlus page it was grounded in.
        source_line = f"<i>Source: {r.source}</i>"
        for url in getattr(r, "source_urls", []) or []:
            source_line += f'<br/><font size="7">{url}</font>'

        story += [
            Paragraph(f"<b>Ask your GP:</b> {_sanitize(r.gp_question)}", body_s),
            Paragraph(source_line, disc_s),
            Spacer(1, 10),
        ]

    story += [
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceBefore=6),
        Paragraph(_sanitize(closing_message), body_s),
        Spacer(1, 6),
        Paragraph(
            "Educational tool evaluated on synthetic test data. Reference texts sourced from NHS UK "
            "(Open Government Licence v3.0) and NIH MedlinePlus (Public Domain).",
            disc_s,
        ),
    ]

    doc.build(story)
    return buf.getvalue()
