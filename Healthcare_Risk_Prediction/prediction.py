"""
prediction.py
Loads serialized scikit-learn pipelines and runs real multi-class patient risk prediction,
contributing factor analysis, and non-diagnostic educational recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from preprocessing import MODEL_FEATURES, RISK_LABELS

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_saved_metrics() -> Dict[str, Any]:
    """Load saved evaluation metrics JSON from disk."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_patient(model_name: str, patient_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run live multi-class prediction for a single patient dictionary."""
    slug = model_name.lower().replace(" ", "_")
    model_path = MODELS_DIR / f"{slug}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_models.py first."
        )

    pipeline = joblib.load(model_path)
    row_df = pd.DataFrame([{k: patient_params[k] for k in MODEL_FEATURES}])

    pred_idx = int(pipeline.predict(row_df)[0])
    probs = pipeline.predict_proba(row_df)[0]
    predicted_risk = RISK_LABELS[pred_idx]
    prob_map = {RISK_LABELS[i]: round(float(probs[i]), 4) for i in range(len(RISK_LABELS))}

    metrics = load_saved_metrics().get(model_name, {})
    feat_imp = metrics.get("feature_importance", {})
    top_factors = sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)[:6]
    contributing = [
        {
            "feature": feat,
            "value": str(patient_params.get(feat, "")),
            "importance_pct": round(imp * 100, 1),
        }
        for feat, imp in top_factors
    ]

    recommendations: List[str] = []
    if patient_params["systolic_bp"] >= 130 or patient_params["diastolic_bp"] >= 85:
        recommendations.append(
            "Blood pressure is elevated; monitor regularly, reduce sodium intake, and consult a clinician."
        )
    if patient_params["glucose_level"] >= 110:
        recommendations.append(
            "Blood glucose is above fasting reference range; discuss routine glycemic screening with a physician."
        )
    if patient_params["cholesterol_level"] >= 200:
        recommendations.append(
            "Total cholesterol is borderline or elevated; consider heart-healthy nutrition and a full lipid review."
        )
    if patient_params["bmi"] >= 25:
        recommendations.append(
            "BMI exceeds 24.9 kg/m2; gradual nutrition and physical activity adjustments can improve cardiometabolic balance."
        )
    if patient_params["smoking_status"] == "Current":
        recommendations.append(
            "Smoking substantially raises cardiovascular risk; cessation programs offer rapid preventive benefit."
        )
    if not recommendations:
        recommendations.append(
            "Parameters align well with general reference ranges; maintain regular activity and routine preventive checkups."
        )

    return {
        "model_name": model_name,
        "predicted_risk": predicted_risk,
        "probabilities": prob_map,
        "contributing_factors": contributing,
        "recommendations": recommendations,
        "disclaimer": (
            "Educational Decision-Support Warning: This prediction is not a medical diagnosis. "
            "Always consult a licensed healthcare provider."
        ),
    }
