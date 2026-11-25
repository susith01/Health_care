"""
report_generator.py
Generates de-identified downloadable PDF and CSV patient risk assessment reports.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict

from fpdf import FPDF


def _csv_safe(value: Any) -> str:
    text = str(value if value is not None else "")
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def generate_csv_bytes(report: Dict[str, Any]) -> bytes:
    """Serialize a patient risk report into CSV bytes with formula-injection protection."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Field", "Value"])

    params = report["patient_params"]
    rows = [
        ("Patient ID", report["patient_id"]),
        ("Date of Prediction", report.get("created_at", datetime.now().isoformat())),
        ("Model Used", report["model_name"]),
        ("Predicted Risk Category", report["predicted_risk"]),
        ("Probabilities", str(report["probabilities"])),
        ("Age", params["age"]),
        ("Gender", params["gender"]),
        ("BMI", params["bmi"]),
        ("Blood Pressure", f"{params['systolic_bp']}/{params['diastolic_bp']} mmHg"),
        ("Blood Glucose", f"{params['glucose_level']} mg/dL"),
        ("Cholesterol", f"{params['cholesterol_level']} mg/dL"),
        ("Heart Rate", f"{params['heart_rate']} bpm"),
        ("Smoking Status", params["smoking_status"]),
        ("Physical Activity", params["physical_activity"]),
        ("Family History", params["family_history"]),
        ("Existing Conditions", params["existing_conditions"]),
        ("Recommendations", " | ".join(report["recommendations"])),
        ("Disclaimer", "Educational decision-support output only. Not a medical diagnosis."),
    ]
    for field, val in rows:
        writer.writerow([_csv_safe(field), _csv_safe(val)])
    return buf.getvalue().encode("utf-8")


def generate_pdf_bytes(report: Dict[str, Any]) -> bytes:
    """Create a structured de-identified PDF patient risk report using FPDF2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(15, 95, 168)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, "Multi-Factor Healthcare Risk Prediction System", new_x="LMARGIN", new_y="NEXT", fill=True)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)
    pdf.cell(0, 7, f"Patient ID: {report['patient_id']}  |  Model: {report['model_name']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Date: {report.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M'))}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Predicted Risk Category: {report['predicted_risk']}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9.5)
    probs = report["probabilities"]
    prob_line = "  |  ".join([f"{k}: {v*100:.1f}%" for k, v in probs.items()])
    pdf.cell(0, 7, f"Class Probabilities: {prob_line}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Patient Clinical Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    p = report["patient_params"]
    lines = [
        f"Age: {p['age']} yrs | Gender: {p['gender']} | BMI: {p['bmi']} kg/m2",
        f"Blood Pressure: {p['systolic_bp']}/{p['diastolic_bp']} mmHg | Heart Rate: {p['heart_rate']} bpm",
        f"Blood Glucose: {p['glucose_level']} mg/dL | Cholesterol: {p['cholesterol_level']} mg/dL",
        f"Smoking: {p['smoking_status']} | Activity: {p['physical_activity']}",
        f"Family History: {p['family_history']} | Existing Conditions: {p['existing_conditions']}",
    ]
    for ln in lines:
        pdf.cell(0, 6, ln, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "General Health Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    for rec in report["recommendations"]:
        pdf.multi_cell(0, 6, f"- {rec}")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        0,
        5,
        "Educational Decision-Support Disclaimer: This report is generated for academic demonstration "
        "and is not a medical diagnosis. Please consult a qualified physician for medical advice.",
    )

    return bytes(pdf.output())
