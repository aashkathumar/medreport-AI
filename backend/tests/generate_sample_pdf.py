"""
Generates a synthetic sample lab report PDF for local smoke testing.
Usage (from backend/): python tests/generate_sample_pdf.py
"""
from reportlab.pdfgen import canvas
import os

out_path = os.path.join(os.path.dirname(__file__), "sample_report.pdf")
c = canvas.Canvas(out_path)
c.drawString(50, 750, "Sample Blood Test Report")
c.drawString(50, 700, "Haemoglobin: 12.1 g/dL")
c.drawString(50, 680, "White Blood Cell Count: 9.8 10^9/L")
c.drawString(50, 660, "HbA1c: 48 mmol/mol")
c.drawString(50, 640, "Total Cholesterol: 6.2 mmol/L")
c.save()
print(f"Sample PDF generated at {out_path}")
