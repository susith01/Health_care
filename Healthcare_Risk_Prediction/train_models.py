"""
train_models.py
Trains and evaluates all four requested multi-class machine learning classifiers:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient Boosting Classifier
Saves trained pipelines and real hold-out test evaluation metrics to models/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    MODEL_FEATURES,
    RISK_LABELS,
    build_sklearn_preprocessor,
    load_dataset,
    stratified_train_test_split,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "healthcare_dataset.csv"
MODELS_DIR = BASE_DIR / "models"


def build_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """Return the four classifiers to benchmark."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=6, min_samples_split=6, min_samples_leaf=2, random_state=random_state
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=random_state
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def extract_feature_importance(pipeline: Pipeline) -> Dict[str, float]:
    """Extract normalized feature importance across all 12 model features."""
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        raw = np.abs(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        raw = np.linalg.norm(clf.coef_, axis=0)
    else:
        raw = np.ones(len(MODEL_FEATURES))

    total = float(np.sum(raw)) or 1.0
    return {
        feat: round(float(val / total), 4)
        for feat, val in zip(MODEL_FEATURES, raw)
    }


def train_and_evaluate_all(
    df: pd.DataFrame,
    numeric_imputation: str = "median",
    scaling_method: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train all 4 leak-free pipelines and compute real hold-out test metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    results: Dict[str, Any] = {}
    registry = build_model_registry(random_state=random_state)

    for name, estimator in registry.items():
        preprocessor = build_sklearn_preprocessor(
            numeric_imputation=numeric_imputation,
            scaling_method=scaling_method,
        )
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_ms = round((time.perf_counter() - start) * 1000, 1)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
        macro_auc = float(
            roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        )

        # Compute macro ROC curve
        fpr_grid = np.linspace(0.0, 1.0, 26)
        mean_tpr = np.zeros_like(fpr_grid)
        for c in range(3):
            fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c], y_proba[:, c])
            mean_tpr += np.interp(fpr_grid, fpr_c, tpr_c)
        mean_tpr /= 3.0

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, MODELS_DIR / f"{slug}.joblib")

        results[name] = {
            "slug": slug,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "macro_auc": round(macro_auc, 4),
            "confusion_matrix": cm,
            "roc_fpr": [round(float(x), 4) for x in fpr_grid],
            "roc_tpr": [round(float(y), 4) for y in mean_tpr],
            "feature_importance": extract_feature_importance(pipe),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "fit_ms": fit_ms,
        }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Ensure dataset/healthcare_dataset.csv exists."
        )
    dataset = load_dataset(DATASET_PATH)
    metrics = train_and_evaluate_all(dataset)
    print("Trained 4 models successfully:")
    for model_name, m in metrics.items():
        print(
            f"  - {model_name}: Accuracy={m['accuracy']*100:.2f}%, "
            f"Macro F1={m['f1_macro']*100:.2f}%, Macro AUC={m['macro_auc']:.4f}"
        )
