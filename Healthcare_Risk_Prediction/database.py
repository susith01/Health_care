"""
database.py
Local SQLite persistence layer for de-identified patient records and prediction reports.
Uses strictly parameterized SQL queries to prevent SQL injection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "healthcare_records.db"


def get_connection() -> sqlite3.Connection:
    """Open a connection to the local SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create the patients and predictions tables if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL UNIQUE,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                bmi REAL NOT NULL,
                systolic_bp INTEGER NOT NULL,
                diastolic_bp INTEGER NOT NULL,
                glucose_level REAL NOT NULL,
                cholesterol_level REAL NOT NULL,
                heart_rate INTEGER NOT NULL,
                smoking_status TEXT NOT NULL,
                physical_activity TEXT NOT NULL,
                family_history TEXT NOT NULL,
                existing_conditions TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                record_source TEXT NOT NULL DEFAULT 'registered',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_risk TEXT NOT NULL,
                probabilities_json TEXT NOT NULL,
                patient_params_json TEXT NOT NULL,
                contributing_factors_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()


def insert_patient(record: Dict[str, Any]) -> None:
    """Insert a validated patient record using parameterized SQL."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patients (
                patient_id, age, gender, height_cm, weight_kg, bmi,
                systolic_bp, diastolic_bp, glucose_level, cholesterol_level,
                heart_rate, smoking_status, physical_activity, family_history,
                existing_conditions, risk_category, record_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["patient_id"],
                int(record["age"]),
                record["gender"],
                float(record["height_cm"]),
                float(record["weight_kg"]),
                float(record["bmi"]),
                int(record["systolic_bp"]),
                int(record["diastolic_bp"]),
                float(record["glucose_level"]),
                float(record["cholesterol_level"]),
                int(record["heart_rate"]),
                record["smoking_status"],
                record["physical_activity"],
                record["family_history"],
                record["existing_conditions"],
                record["risk_category"],
                record.get("record_source", "registered"),
            ),
        )
        conn.commit()


def fetch_all_patients() -> pd.DataFrame:
    """Return all stored patient records as a Pandas DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT patient_id, age, gender, height_cm, weight_kg, bmi,
                   systolic_bp, diastolic_bp, glucose_level, cholesterol_level,
                   heart_rate, smoking_status, physical_activity, family_history,
                   existing_conditions, risk_category, record_source, created_at
            FROM patients
            ORDER BY CASE WHEN record_source = 'registered' THEN 0 ELSE 1 END, id DESC
            """,
            conn,
        )


def delete_patient(patient_id: str) -> bool:
    """Delete a patient record by Patient ID."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        conn.commit()
        return cur.rowcount > 0


def save_prediction(
    patient_id: str,
    model_name: str,
    predicted_risk: str,
    probabilities: Dict[str, float],
    patient_params: Dict[str, Any],
    contributing_factors: List[Dict[str, Any]],
    recommendations: List[str],
) -> int:
    """Save a patient risk assessment to SQLite and return the report ID."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                patient_id, model_name, predicted_risk,
                probabilities_json, patient_params_json,
                contributing_factors_json, recommendations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                model_name,
                predicted_risk,
                json.dumps(probabilities),
                json.dumps(patient_params),
                json.dumps(contributing_factors),
                json.dumps(recommendations),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def fetch_all_predictions() -> List[Dict[str, Any]]:
    """Return all saved risk prediction reports ordered newest first."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    reports: List[Dict[str, Any]] = []
    for row in rows:
        reports.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "model_name": row["model_name"],
                "predicted_risk": row["predicted_risk"],
                "probabilities": json.loads(row["probabilities_json"]),
                "patient_params": json.loads(row["patient_params_json"]),
                "contributing_factors": json.loads(row["contributing_factors_json"]),
                "recommendations": json.loads(row["recommendations_json"]),
                "created_at": row["created_at"],
            }
        )
    return reports


def seed_from_dataframe_if_empty(df: pd.DataFrame) -> None:
    """Seed the SQLite patients table once from the synthetic demonstration dataset."""
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if count > 0:
        return

    for _, row in df.iterrows():
        try:
            insert_patient(
                {
                    "patient_id": str(row["patient_id"]),
                    "age": int(row["age"]),
                    "gender": str(row["gender"]),
                    "height_cm": float(row["height_cm"]),
                    "weight_kg": float(row["weight_kg"]),
                    "bmi": float(row["bmi"]),
                    "systolic_bp": int(row["systolic_bp"]),
                    "diastolic_bp": int(row["diastolic_bp"]),
                    "glucose_level": float(row["glucose_level"]),
                    "cholesterol_level": float(row["cholesterol_level"]),
                    "heart_rate": int(row["heart_rate"]),
                    "smoking_status": str(row["smoking_status"]),
                    "physical_activity": str(row["physical_activity"]),
                    "family_history": str(row["family_history"]),
                    "existing_conditions": str(row["existing_conditions"]),
                    "risk_category": str(row["risk_category"]),
                    "record_source": "synthetic_demo",
                }
            )
        except sqlite3.IntegrityError:
            continue
