"""
app.py
Main 10-page Streamlit application for the Multi-Factor Healthcare Risk Prediction System.
Run locally via: streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

import database
import prediction
import preprocessing
import report_generator
import train_models
import visualization

st.set_page_config(
    page_title="Multi-Factor Healthcare Risk Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "dataset" / "healthcare_dataset.csv"

# Initialize SQLite and ensure models are trained on first run
database.init_database()
if DEFAULT_CSV.exists():
    raw_default = preprocessing.load_dataset(DEFAULT_CSV)
    clean_default, _ = preprocessing.audit_and_clean_dataset(raw_default)
    database.seed_from_dataframe_if_empty(clean_default)
    if not (BASE_DIR / "models" / "metrics.json").exists():
        train_models.train_and_evaluate_all(raw_default)

if "active_df" not in st.session_state and DEFAULT_CSV.exists():
    st.session_state.active_df = preprocessing.load_dataset(DEFAULT_CSV)
    st.session_state.dataset_name = "Synthetic Demonstration Healthcare Dataset (320 Records)"

st.sidebar.title("HealthRisk Analytics")
st.sidebar.caption("Educational Decision Support · Not a Medical Diagnosis")

PAGE = st.sidebar.radio(
    "Navigation",
    [
        "1. Home Dashboard",
        "2. Patient Registration",
        "3. Patient Data",
        "4. Data Preprocessing",
        "5. Exploratory Data Analysis",
        "6. Machine Learning Models",
        "7. Risk Prediction",
        "8. Model Performance",
        "9. Reports",
        "10. About Project",
    ],
)

metrics_data = prediction.load_saved_metrics()
patients_df = database.fetch_all_patients()

if PAGE == "1. Home Dashboard":
    st.title("Multi-Factor Healthcare Risk Prediction System")
    st.info(
        "Educational Decision-Support Notice: This platform uses clearly labeled synthetic "
        "demonstration data by default and is NOT a certified medical diagnostic tool."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SQLite Patients", len(patients_df))
    c2.metric("High Risk Patients", int((patients_df["risk_category"] == "High Risk").sum()))
    if metrics_data:
        best_name = max(metrics_data, key=lambda k: metrics_data[k]["f1_macro"])
        c3.metric("Top Model Accuracy", f"{metrics_data[best_name]['accuracy']*100:.1f}%", best_name)
    else:
        c3.metric("Top Model Accuracy", "Not trained")
    c4.metric("Saved Reports", len(database.fetch_all_predictions()))

    if metrics_data:
        st.plotly_chart(visualization.chart_9_accuracy_comparison(metrics_data), use_container_width=True)

elif PAGE == "2. Patient Registration":
    st.title("Patient Registration (SQLite)")
    with st.form("patient_reg_form"):
        col1, col2, col3 = st.columns(3)
        patient_id = col1.text_input("Patient ID", value="PAT-1001")
        age = col2.number_input("Age (years)", min_value=1, max_value=120, value=45)
        gender = col3.selectbox("Gender", ["Female", "Male", "Other"])

        height_cm = col1.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=168.0)
        weight_kg = col2.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=72.0)
        bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
        col3.metric("Auto-Calculated BMI", f"{bmi} kg/m²")

        systolic_bp = col1.number_input("Systolic BP (mmHg)", min_value=70, max_value=260, value=126)
        diastolic_bp = col2.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=82)
        heart_rate = col3.number_input("Heart Rate (bpm)", min_value=35, max_value=220, value=76)

        glucose_level = col1.number_input("Blood Glucose (mg/dL)", min_value=40.0, max_value=600.0, value=102.0)
        cholesterol_level = col2.number_input("Cholesterol (mg/dL)", min_value=80.0, max_value=600.0, value=205.0)
        smoking_status = col3.selectbox("Smoking Status", ["Never", "Former", "Current"])

        physical_activity = col1.selectbox("Physical Activity", ["High", "Moderate", "Low"])
        family_history = col2.selectbox(
            "Family History", ["None", "Hypertension", "Diabetes", "Heart Disease", "Multiple"]
        )
        existing_conditions = col3.selectbox(
            "Existing Conditions",
            ["None", "Asthma", "Hypertension", "Diabetes", "Heart Disease", "Multiple"],
        )

        submitted = st.form_submit_button("Save Patient to SQLite")
        if submitted:
            if not re.match(r"^[A-Za-z0-9_-]{3,24}$", patient_id.strip()):
                st.error("Patient ID must be 3–24 alphanumeric, hyphen, or underscore characters.")
            elif systolic_bp <= diastolic_bp:
                st.error("Systolic BP must be strictly greater than Diastolic BP.")
            else:
                best_model = max(metrics_data, key=lambda k: metrics_data[k]["f1_macro"])
                pred = prediction.predict_patient(
                    best_model,
                    {
                        "age": age,
                        "gender": gender,
                        "bmi": bmi,
                        "systolic_bp": systolic_bp,
                        "diastolic_bp": diastolic_bp,
                        "glucose_level": glucose_level,
                        "cholesterol_level": cholesterol_level,
                        "heart_rate": heart_rate,
                        "smoking_status": smoking_status,
                        "physical_activity": physical_activity,
                        "family_history": family_history,
                        "existing_conditions": existing_conditions,
                    },
                )
                try:
                    database.insert_patient(
                        {
                            "patient_id": patient_id.strip(),
                            "age": age,
                            "gender": gender,
                            "height_cm": height_cm,
                            "weight_kg": weight_kg,
                            "bmi": bmi,
                            "systolic_bp": systolic_bp,
                            "diastolic_bp": diastolic_bp,
                            "glucose_level": glucose_level,
                            "cholesterol_level": cholesterol_level,
                            "heart_rate": heart_rate,
                            "smoking_status": smoking_status,
                            "physical_activity": physical_activity,
                            "family_history": family_history,
                            "existing_conditions": existing_conditions,
                            "risk_category": pred["predicted_risk"],
                            "record_source": "registered",
                        }
                    )
                    st.success(
                        f"Saved {patient_id.strip()} (BMI {bmi}). Initial estimate ({best_model}): {pred['predicted_risk']}"
                    )
                except Exception as exc:
                    st.error(f"Could not save patient: {exc}")

elif PAGE == "3. Patient Data":
    st.title("Patient Data Repository")
    st.dataframe(patients_df, use_container_width=True)

elif PAGE == "4. Data Preprocessing":
    st.title("Data Preprocessing Pipeline")
    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded is not None:
        st.session_state.active_df = pd.read_csv(uploaded)
        st.session_state.dataset_name = f"Uploaded CSV: {uploaded.name}"

    num_imp = st.selectbox("Numeric Imputation", ["median", "mean"])
    scale_method = st.selectbox("Feature Scaling", ["standard", "minmax"])
    remove_dups = st.checkbox("Remove duplicate records", value=True)
    clip_out = st.checkbox("Clip 1.5x IQR outliers", value=True)

    cleaned_df, audit = preprocessing.audit_and_clean_dataset(
        st.session_state.active_df,
        numeric_imputation=num_imp,
        remove_duplicates=remove_dups,
        clip_outliers=clip_out,
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Raw Rows", audit["raw_rows"])
    k2.metric("Duplicates Removed", audit["duplicates_removed"])
    k3.metric("Missing Imputed", audit["total_missing"])
    k4.metric("Cleaned Rows", audit["cleaned_rows"])

    st.subheader("Missing Values Audit")
    st.dataframe(audit["missing_table"], use_container_width=True)
    st.subheader("IQR Outlier Bounds")
    st.dataframe(audit["outlier_table"], use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset (CSV)",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_healthcare_dataset.csv",
        mime="text/csv",
    )

    if st.button("Apply Pipeline & Retrain All 4 Models"):
        train_models.train_and_evaluate_all(
            st.session_state.active_df,
            numeric_imputation=num_imp,
            scaling_method=scale_method,
        )
        st.success("Retrained all 4 models using leak-free scikit-learn Pipelines.")

elif PAGE == "5. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (Charts 1–8)")
    cleaned_df, _ = preprocessing.audit_and_clean_dataset(st.session_state.active_df)
    c1, c2 = st.columns(2)
    c1.plotly_chart(visualization.chart_1_age_histogram(cleaned_df), use_container_width=True)
    c2.plotly_chart(visualization.chart_2_gender_pie(cleaned_df), use_container_width=True)
    c1.plotly_chart(visualization.chart_3_bmi_histogram(cleaned_df), use_container_width=True)
    c2.plotly_chart(visualization.chart_4_bp_bar(cleaned_df), use_container_width=True)
    c1.plotly_chart(visualization.chart_5_glucose_histogram(cleaned_df), use_container_width=True)
    c2.plotly_chart(visualization.chart_6_cholesterol_boxplot(cleaned_df), use_container_width=True)
    c1.plotly_chart(visualization.chart_7_risk_pie(cleaned_df), use_container_width=True)
    c2.pyplot(visualization.chart_8_correlation_heatmap(cleaned_df))

elif PAGE == "6. Machine Learning Models":
    st.title("Machine Learning Models")
    test_size = st.slider("Stratified Test Split Ratio", 0.15, 0.35, 0.20, 0.05)
    if st.button("Train All 4 Classifiers"):
        metrics_data = train_models.train_and_evaluate_all(
            st.session_state.active_df, test_size=test_size
        )
        st.success("Training complete.")
    for name, m in metrics_data.items():
        with st.expander(f"{name} — Accuracy {m['accuracy']*100:.2f}% | AUC {m['macro_auc']:.4f}"):
            st.json(
                {
                    "accuracy": m["accuracy"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro": m["recall_macro"],
                    "f1_macro": m["f1_macro"],
                    "macro_auc": m["macro_auc"],
                    "train_size": m["train_size"],
                    "test_size": m["test_size"],
                }
            )

elif PAGE == "7. Risk Prediction":
    st.title("Healthcare Risk Prediction")
    st.warning("Educational Warning: Predictions are NOT medical diagnoses.")
    model_choice = st.selectbox("Select Trained Model", list(metrics_data.keys()))
    col1, col2, col3 = st.columns(3)
    pid = col1.text_input("Patient ID", value="EVAL-1001")
    age = col2.number_input("Age", 1, 120, 54)
    gender = col3.selectbox("Gender", ["Male", "Female", "Other"])
    height_cm = col1.number_input("Height (cm)", 80.0, 250.0, 174.0)
    weight_kg = col2.number_input("Weight (kg)", 20.0, 300.0, 88.0)
    bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
    sbp = col3.number_input("Systolic BP", 70, 260, 142)
    dbp = col1.number_input("Diastolic BP", 40, 160, 90)
    glucose = col2.number_input("Glucose (mg/dL)", 40.0, 600.0, 138.0)
    chol = col3.number_input("Cholesterol (mg/dL)", 80.0, 600.0, 236.0)
    hr = col1.number_input("Heart Rate (bpm)", 35, 220, 84)
    smoke = col2.selectbox("Smoking", ["Never", "Former", "Current"])
    activity = col3.selectbox("Physical Activity", ["High", "Moderate", "Low"])
    fam = col1.selectbox("Family History", ["None", "Hypertension", "Diabetes", "Heart Disease", "Multiple"])
    cond = col2.selectbox("Existing Conditions", ["None", "Asthma", "Hypertension", "Diabetes", "Heart Disease", "Multiple"])

    if st.button("Predict Risk"):
        params = {
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "glucose_level": glucose,
            "cholesterol_level": chol,
            "heart_rate": hr,
            "smoking_status": smoke,
            "physical_activity": activity,
            "family_history": fam,
            "existing_conditions": cond,
        }
        res = prediction.predict_patient(model_choice, params)
        database.save_prediction(
            pid,
            res["model_name"],
            res["predicted_risk"],
            res["probabilities"],
            params,
            res["contributing_factors"],
            res["recommendations"],
        )
        st.subheader(f"Predicted Risk Category: {res['predicted_risk']}")
        st.write("Class Probabilities:", res["probabilities"])
        st.write("Top Contributing Features:", res["contributing_factors"])
        st.write("General Recommendations:")
        for r in res["recommendations"]:
            st.write(f"- {r}")

elif PAGE == "8. Model Performance":
    st.title("Model Performance Evaluation (Charts 9–12)")
    comp_rows = [
        {
            "Algorithm": k,
            "Accuracy": f"{v['accuracy']*100:.2f}%",
            "Precision (Macro)": f"{v['precision_macro']*100:.2f}%",
            "Recall (Macro)": f"{v['recall_macro']*100:.2f}%",
            "F1-Score (Macro)": f"{v['f1_macro']*100:.2f}%",
            "ROC-AUC (Macro)": f"{v['macro_auc']:.4f}",
        }
        for k, v in metrics_data.items()
    ]
    st.table(pd.DataFrame(comp_rows))
    st.plotly_chart(visualization.chart_9_accuracy_comparison(metrics_data), use_container_width=True)
    st.plotly_chart(visualization.chart_11_roc_comparison(metrics_data), use_container_width=True)
    chosen = st.selectbox("Inspect Confusion Matrix & Feature Importance", list(metrics_data.keys()))
    c1, c2 = st.columns(2)
    c1.pyplot(visualization.chart_10_confusion_matrix(chosen, metrics_data[chosen]["confusion_matrix"]))
    c2.plotly_chart(
        visualization.chart_12_feature_importance(chosen, metrics_data[chosen]["feature_importance"]),
        use_container_width=True,
    )

elif PAGE == "9. Reports":
    st.title("Patient Risk Reports (PDF / CSV)")
    reports = database.fetch_all_predictions()
    if not reports:
        st.info("No saved predictions yet. Run a prediction on Page 7 first.")
    else:
        for rep in reports:
            with st.expander(f"#{rep['id']} — {rep['patient_id']} ({rep['predicted_risk']} · {rep['model_name']})"):
                st.write(rep["probabilities"])
                col_a, col_b = st.columns(2)
                col_a.download_button(
                    "Download PDF",
                    data=report_generator.generate_pdf_bytes(rep),
                    file_name=f"risk_report_{rep['patient_id']}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{rep['id']}",
                )
                col_b.download_button(
                    "Download CSV",
                    data=report_generator.generate_csv_bytes(rep),
                    file_name=f"risk_report_{rep['patient_id']}.csv",
                    mime="text/csv",
                    key=f"csv_{rep['id']}",
                )

elif PAGE == "10. About Project":
    st.title("About Multi-Factor Healthcare Risk Prediction System")
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
