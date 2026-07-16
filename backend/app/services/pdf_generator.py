import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.units import cm
from app.models.schemas import ExplainedResult, RangeStatus

BLUE = HexColor("#003E74")
GREEN = HexColor("#2E8B57")
AMBER = HexColor("#FFA500")
RED = HexColor("#B22222")
LGREY = HexColor("#F5F5F5")

STATUS_COLOUR = {
    RangeStatus.NORMAL: GREEN, RangeStatus.LOW: RED,
    RangeStatus.HIGH: AMBER, RangeStatus.UNKNOWN: LGREY,
}
STATUS_LABEL = {
    RangeStatus.NORMAL: "Within normal range",
    RangeStatus.LOW: "Below normal range",
    RangeStatus.HIGH: "Above normal range",
    RangeStatus.UNKNOWN: "Range unknown",
}


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
            "This document is for health literacy purposes only and does not "
            "replace medical advice. Always discuss results with your GP.",
            disc_s,
        ),
        Spacer(1, 10),
        Paragraph("Overall Summary", head_s),
        Paragraph(overall_summary, body_s),
    ]

    if top_gp_topics:
        story.append(Paragraph("Topics to discuss with your GP:", body_s))
        for t in top_gp_topics:
            story.append(Paragraph(f"- {t}", body_s))

    story.append(Paragraph(f"<b>Top lifestyle change:</b> {top_lifestyle_change}", body_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY, spaceAfter=6))
    story.append(Paragraph("Your Results -- Explained", head_s))

    for r in explained_results:
        col = STATUS_COLOUR[r.status]
        label = STATUS_LABEL[r.status]
        banner = Table(
            [[
                Paragraph(f"<b>{r.raw_name}</b>", body_s),
                Paragraph(
                    f"<b>{r.value} {r.unit}</b> &nbsp; {label}",
                    ParagraphStyle("BL", parent=body_s, textColor=col),
                ),
            ]],
            colWidths=["50%", "50%"],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LGREY),
            ("ROWPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 1.5, col),
        ]))
        story += [
            banner, Spacer(1, 4),
            Paragraph(r.what_it_measures, body_s),
            Paragraph(f"<b>Your result:</b> {r.what_your_result_means}", body_s),
        ]
        if r.lifestyle_suggestions:
            story.append(Paragraph("<b>Lifestyle suggestions:</b>", body_s))
            for sug in r.lifestyle_suggestions:
                story.append(Paragraph(f"- {sug}", body_s))
        story += [
            Paragraph(f"<b>Ask your GP:</b> {r.gp_question}", body_s),
            Paragraph(f"<i>Source: {r.source}</i>", disc_s),
            Spacer(1, 10),
        ]

    story += [
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceBefore=6),
        Paragraph(closing_message, body_s),
        Spacer(1, 6),
        Paragraph(
            "MedReport AI is an educational tool. All reference ranges are "
            "sourced from NHS UK and NIH MedlinePlus.",
            disc_s,
        ),
    ]

    doc.build(story)
    return buf.getvalue()
