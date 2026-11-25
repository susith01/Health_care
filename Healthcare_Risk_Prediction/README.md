# Multi-Factor Healthcare Risk Prediction System

An educational clinical decision-support application analyzing multi-factor patient health parameters using four supervised machine learning algorithms (**Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, and **Gradient Boosting Classifier**), local **SQLite** storage, leak-free preprocessing, 12 visualizations, and downloadable **PDF/CSV** patient risk reports.

> **Disclaimer:** This system is strictly an educational decision-support project, not a certified medical diagnostic tool. Default records in `dataset/healthcare_dataset.csv` are clearly labeled synthetic demonstration records.

## Project Structure

```
Healthcare_Risk_Prediction/
├── app.py                  # Main 10-page Streamlit dashboard
├── database.py             # Parameterized SQLite persistence (patients & predictions)
├── preprocessing.py        # Leak-free imputation, deduplication, IQR outliers, encoding, scaling
├── train_models.py         # Trains & evaluates all 4 scikit-learn Pipelines
├── prediction.py           # Interactive multi-class risk prediction & recommendations
├── visualization.py        # 12 Matplotlib, Seaborn, and Plotly visualizations
├── report_generator.py     # De-identified PDF (FPDF2) & CSV report exports
├── requirements.txt        # Python dependencies
├── dataset/
│   └── healthcare_dataset.csv
├── models/                 # Serialized .joblib pipelines and metrics.json
├── reports/
└── README.md
```

## Windows PowerShell Setup & Execution

```powershell
# 1. Navigate to the project directory
cd Healthcare_Risk_Prediction

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install required Python libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Train all 4 machine learning models and generate real test evaluation metrics
python train_models.py

# 5. Launch the Streamlit web dashboard
streamlit run app.py
```

## Target Classification Mapping

The supervised target variable `risk_category` is explicitly defined across three discrete classes:
- `0` → **Low Risk**
- `1` → **Moderate Risk**
- `2` → **High Risk**

All four classifiers are trained natively for 3-class classification (`predict` chooses the argmax class over calibrated multi-class probabilities rather than applying arbitrary probability thresholds to a binary score).
