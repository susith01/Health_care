"""
visualization.py
Generates all 12 data-driven healthcare and machine learning charts using Plotly,
Matplotlib, and Seaborn directly from actual dataset values and real test metrics.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

RISK_COLOR_MAP = {
    "Low Risk": "#0D9488",
    "Moderate Risk": "#D97706",
    "High Risk": "#DC2626",
}


def chart_1_age_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="age",
        color="risk_category",
        nbins=15,
        title="1. Age Distribution by Risk Category",
        labels={"age": "Age (years)", "risk_category": "Risk Category"},
        color_discrete_map=RISK_COLOR_MAP,
    )
    fig.update_layout(template="plotly_white", yaxis_title="Patient Count")
    return fig


def chart_2_gender_pie(df: pd.DataFrame):
    counts = df["gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = px.pie(
        counts,
        names="Gender",
        values="Count",
        hole=0.4,
        title="2. Gender Distribution",
        color_discrete_sequence=["#0F5FA8", "#0D9488", "#64748B"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def chart_3_bmi_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="bmi",
        nbins=18,
        title="3. BMI Distribution",
        labels={"bmi": "Body Mass Index (kg/m²)"},
        color_discrete_sequence=["#0F5FA8"],
    )
    fig.update_layout(template="plotly_white", yaxis_title="Patient Count")
    return fig


def chart_4_bp_bar(df: pd.DataFrame):
    grouped = (
        df.groupby("risk_category", as_index=False)[["systolic_bp", "diastolic_bp"]]
        .mean()
        .round(1)
    )
    fig = go.Figure(
        data=[
            go.Bar(
                name="Avg Systolic BP",
                x=grouped["risk_category"],
                y=grouped["systolic_bp"],
                marker_color="#0F5FA8",
            ),
            go.Bar(
                name="Avg Diastolic BP",
                x=grouped["risk_category"],
                y=grouped["diastolic_bp"],
                marker_color="#0D9488",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="4. Average Blood Pressure by Risk Category",
        xaxis_title="Risk Category",
        yaxis_title="Blood Pressure (mmHg)",
        template="plotly_white",
    )
    return fig


def chart_5_glucose_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="glucose_level",
        color="risk_category",
        nbins=18,
        title="5. Blood Glucose Level Distribution",
        labels={"glucose_level": "Blood Glucose (mg/dL)"},
        color_discrete_map=RISK_COLOR_MAP,
    )
    fig.update_layout(template="plotly_white", yaxis_title="Patient Count")
    return fig


def chart_6_cholesterol_boxplot(df: pd.DataFrame):
    fig = px.box(
        df,
        x="risk_category",
        y="cholesterol_level",
        color="risk_category",
        title="6. Cholesterol Analysis by Risk Category (Box Plot)",
        labels={"risk_category": "Risk Category", "cholesterol_level": "Cholesterol (mg/dL)"},
        color_discrete_map=RISK_COLOR_MAP,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def chart_7_risk_pie(df: pd.DataFrame):
    counts = df["risk_category"].value_counts().reset_index()
    counts.columns = ["Risk Category", "Count"]
    fig = px.pie(
        counts,
        names="Risk Category",
        values="Count",
        hole=0.4,
        title="7. Risk Category Distribution",
        color="Risk Category",
        color_discrete_map=RISK_COLOR_MAP,
    )
    fig.update_layout(template="plotly_white")
    return fig


def chart_8_correlation_heatmap(df: pd.DataFrame):
    num_cols = [
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "glucose_level",
        "cholesterol_level",
        "heart_rate",
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("8. Clinical Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


def chart_9_accuracy_comparison(metrics: Dict[str, Any]):
    rows = [
        {
            "Algorithm": name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro F1 (%)": round(m["f1_macro"] * 100, 2),
        }
        for name, m in metrics.items()
    ]
    comp_df = pd.DataFrame(rows)
    fig = px.bar(
        comp_df,
        x="Algorithm",
        y=["Accuracy (%)", "Macro F1 (%)"],
        barmode="group",
        title="9. Algorithm Accuracy & Macro F1 Comparison",
        color_discrete_sequence=["#0F5FA8", "#0D9488"],
    )
    fig.update_layout(template="plotly_white", yaxis_range=[0, 100])
    return fig


def chart_10_confusion_matrix(model_name: str, cm: list):
    labels = ["Low Risk", "Moderate Risk", "High Risk"]
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Risk Category")
    ax.set_ylabel("Actual Risk Category")
    ax.set_title(f"10. Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def chart_11_roc_comparison(metrics: Dict[str, Any]):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline (AUC = 0.50)",
            line=dict(dash="dash", color="#94A3B8"),
        )
    )
    colors = ["#0F5FA8", "#D97706", "#0D9488", "#1E293B"]
    for (name, m), color in zip(metrics.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=m["roc_fpr"],
                y=m["roc_tpr"],
                mode="lines",
                name=f"{name} (AUC = {m['macro_auc']:.3f})",
                line=dict(width=2.2, color=color),
            )
        )
    fig.update_layout(
        title="11. Multi-Model Macro ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    return fig


def chart_12_feature_importance(model_name: str, feature_importance: Dict[str, float]):
    fi_df = (
        pd.DataFrame(
            [
                {"Feature": k, "Importance (%)": round(v * 100, 2)}
                for k, v in feature_importance.items()
            ]
        )
        .sort_values("Importance (%)", ascending=True)
    )
    fig = px.bar(
        fi_df,
        x="Importance (%)",
        y="Feature",
        orientation="h",
        title=f"12. Feature Importance — {model_name}",
        color_discrete_sequence=["#0D9488"],
    )
    fig.update_layout(template="plotly_white")
    return fig
