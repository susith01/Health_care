"""
preprocessing.py
End-to-end leak-free healthcare data preprocessing module:
- Missing value detection & Mean / Median / Mode imputation
- Duplicate removal
- 1.5x IQR outlier detection & optional clipping
- Deterministic categorical label encoding
- StandardScaler / MinMaxScaler feature scaling
- Stratified train-test splitting
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose_level",
    "cholesterol_level",
    "heart_rate",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_status",
    "physical_activity",
    "family_history",
    "existing_conditions",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "risk_category"
RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]

CATEGORY_ORDER = [
    ["Female", "Male", "Other"],
    ["Never", "Former", "Current"],
    ["High", "Moderate", "Low"],
    ["None", "Hypertension", "Diabetes", "Heart Disease", "Multiple"],
    ["None", "Asthma", "Hypertension", "Diabetes", "Heart Disease", "Multiple"],
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the healthcare CSV dataset from disk with UTF-8 BOM support."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def audit_and_clean_dataset(
    raw_df: pd.DataFrame,
    numeric_imputation: str = "median",
    remove_duplicates: bool = True,
    clip_outliers: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Audit raw dataset quality and produce a cleaned DataFrame for inspection & CSV export.
    Note: During ML training, imputation and scaling are fitted strictly inside
    a scikit-learn Pipeline on X_train only to prevent data leakage.
    """
    df = raw_df.copy()
    raw_rows = len(df)

    duplicates_removed = 0
    if remove_duplicates:
        before = len(df)
        if "patient_id" in df.columns:
            df = df.drop_duplicates(subset=["patient_id"], keep="first")
        df = df.drop_duplicates(keep="first")
        duplicates_removed = before - len(df)

    # Keep only rows with valid target risk labels
    df = df[df[TARGET_COLUMN].isin(RISK_LABELS)].copy()

    # Missing value audit
    missing_summary: List[Dict[str, Any]] = []
    for col in NUMERIC_FEATURES + ["height_cm", "weight_kg"] + CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        miss_count = int(df[col].isna().sum())
        is_num = col in NUMERIC_FEATURES or col in ("height_cm", "weight_kg")
        if is_num:
            fill_val = (
                round(float(df[col].mean()), 2)
                if numeric_imputation == "mean"
                else round(float(df[col].median()), 2)
            )
            df[col] = df[col].fillna(fill_val)
            strategy = "Mean Imputation" if numeric_imputation == "mean" else "Median Imputation"
        else:
            mode_series = df[col].mode()
            fill_val = str(mode_series.iloc[0]) if not mode_series.empty else "None"
            df[col] = df[col].fillna(fill_val)
            strategy = "Mode Imputation"
        missing_summary.append(
            {
                "column": col,
                "missing_count": miss_count,
                "missing_pct": round((miss_count / max(1, len(df))) * 100, 1),
                "strategy": strategy,
                "fill_value": fill_val,
            }
        )

    # IQR Outlier detection
    outlier_summary: List[Dict[str, Any]] = []
    for col in NUMERIC_FEATURES:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if clip_outliers:
            df[col] = df[col].clip(lower=lower, upper=upper)
        outlier_summary.append(
            {
                "feature": col,
                "q1": round(q1, 1),
                "q3": round(q3, 1),
                "iqr": round(iqr, 1),
                "lower_bound": round(lower, 1),
                "upper_bound": round(upper, 1),
                "outlier_count": count,
            }
        )

    summary = {
        "raw_rows": raw_rows,
        "cleaned_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "total_missing": sum(m["missing_count"] for m in missing_summary),
        "missing_table": pd.DataFrame(missing_summary),
        "outlier_table": pd.DataFrame(outlier_summary),
    }
    return df, summary


def build_sklearn_preprocessor(
    numeric_imputation: str = "median",
    scaling_method: str = "standard",
) -> ColumnTransformer:
    """
    Construct a leak-free scikit-learn ColumnTransformer.
    Fitted strictly on training splits inside each model Pipeline.
    """
    scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_imputation)),
            ("scaler", scaler),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=CATEGORY_ORDER,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and multi-class target while preserving class balance."""
    valid_df = df.drop_duplicates(subset=["patient_id"], keep="first")
    valid_df = valid_df[valid_df[TARGET_COLUMN].isin(RISK_LABELS)].copy()

    X = valid_df[MODEL_FEATURES]
    y = valid_df[TARGET_COLUMN].map({lbl: idx for idx, lbl in enumerate(RISK_LABELS)})
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
