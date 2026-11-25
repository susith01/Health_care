import React, { useEffect, useState } from 'react';
import { BookOpen, Code2, Copy, Check, Terminal } from 'lucide-react';
import { api } from '../lib/api';
import type { ModelKey, TrainedModelsBundle } from '../types';

export function AboutProject() {
  const [tab, setTab] = useState<'docs' | 'python' | 'setup'>('docs');
  const [bundle, setBundle] = useState<TrainedModelsBundle | null>(null);
  const [pyFiles, setPyFiles] = useState<Array<{ name: string; path: string }>>([]);
  const [selectedFile, setSelectedFile] = useState<string>('app.py');
  const [fileContent, setFileContent] = useState<string>('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.getModels().then(setBundle).catch(() => {});
    api
      .getPythonFiles()
      .then((res) => {
        setPyFiles(res.files);
        if (res.files.length > 0) setSelectedFile(res.files[0].name);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (tab !== 'python' || !selectedFile) return;
    api
      .getPythonFileContent(selectedFile)
      .then((res) => setFileContent(res.content))
      .catch(() => setFileContent('# File not available'));
  }, [tab, selectedFile]);

  const handleCopy = async () => {
    if (!fileContent) return;
    await navigator.clipboard.writeText(fileContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const orderedModels: ModelKey[] = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
  ];

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            About Project &amp; Documentation
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Complete 16-section technical report, mathematical formulations, Python/Streamlit source
            modules, and Windows PowerShell execution guide.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md shrink-0">
          <button
            type="button"
            onClick={() => setTab('docs')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              tab === 'docs' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Project Documentation
          </button>
          <button
            type="button"
            onClick={() => setTab('python')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              tab === 'python' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            Python / Streamlit Code
          </button>
          <button
            type="button"
            onClick={() => setTab('setup')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              tab === 'setup' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            Windows PowerShell Guide
          </button>
        </div>
      </div>

      {tab === 'docs' && (
        <div className="space-y-6 text-sm text-slate-700 leading-relaxed">
          <section className="rounded-lg border border-slate-200 bg-white p-6 space-y-5">
            <div>
              <h2 className="text-base font-bold text-slate-900">01. Abstract</h2>
              <p className="mt-1.5">
                The Multi-Factor Healthcare Risk Prediction System is an educational clinical
                decision-support platform designed to evaluate multi-dimensional patient health
                indicators—spanning demographics, anthropometrics (BMI), hemodynamic vitals (blood
                pressure, heart rate), metabolic biomarkers (blood glucose, total cholesterol),
                lifestyle habits, and medical history—to estimate composite health risk levels (Low
                Risk, Moderate Risk, High Risk). Four supervised machine learning algorithms
                (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting) are
                trained inside a leak-free preprocessing pipeline, evaluated rigorously on
                stratified hold-out partitions, and integrated with local SQLite persistence, twelve
                interactive visualizations, and downloadable PDF/CSV reports.
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">02. Introduction</h2>
              <p className="mt-1.5">
                Chronic cardiometabolic disorders rarely stem from a single isolated measurement.
                Modest elevations in systolic blood pressure, fasting glucose, body mass index,
                sedentary lifestyle, and familial predisposition interact non-linearly to compound
                overall health risk. Machine learning provides structured methods for weighting
                multiple concurrent factors consistently and transparently for educational decision
                support.
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">03. Problem Statement</h2>
              <p className="mt-1.5">
                Traditional rule-of-thumb screening often examines individual laboratory thresholds
                in isolation (e.g., blood pressure alone or BMI alone), overlooking multi-factor
                interactions. Moreover, educational ML prototypes frequently suffer from data
                leakage (scaling before train/test splitting), fabricated accuracy numbers, or
                arbitrary probability cutoffs on binary classifiers. This project addresses those
                gaps with an end-to-end, reproducible, leak-free multi-class risk prediction
                workflow.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-1">
              <div>
                <h2 className="text-base font-bold text-slate-900">04. Existing System</h2>
                <ul className="mt-1.5 list-disc pl-4 space-y-1 text-xs sm:text-sm">
                  <li>Manual review of disconnected lab charts and vital sheets.</li>
                  <li>Single-factor risk calculators with limited extensibility.</li>
                  <li>Static spreadsheets without automated preprocessing or model comparison.</li>
                  <li>No unified patient record repository or standardized downloadable reports.</li>
                </ul>
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900">05. Proposed System</h2>
                <ul className="mt-1.5 list-disc pl-4 space-y-1 text-xs sm:text-sm">
                  <li>Integrated 10-page interactive dashboard backed by local SQLite storage.</li>
                  <li>Leak-free preprocessing pipeline (imputation, deduplication, IQR, scaling).</li>
                  <li>Four supervised classifiers evaluated with confusion matrices and ROC-AUC.</li>
                  <li>De-identified patient prediction reports exportable as PDF and CSV.</li>
                </ul>
              </div>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">06. Project Objectives</h2>
              <ol className="mt-1.5 list-decimal pl-5 space-y-1">
                <li>Capture and validate 15 structured patient health parameters in SQLite.</li>
                <li>Provide transparent, configurable data preprocessing with CSV upload/export.</li>
                <li>Deliver 12 data-driven clinical and model evaluation visualizations.</li>
                <li>Train and benchmark Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting without data leakage.</li>
                <li>Generate explainable, non-diagnostic risk estimates and downloadable PDF/CSV reports.</li>
              </ol>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">07. System Architecture</h2>
              <p className="mt-1.5">
                The platform follows a modular four-layer architecture: (1) Presentation Layer (the
                10-page interactive web dashboard + accompanying Python/Streamlit{' '}
                <code className="font-mono text-xs bg-slate-100 px-1 rounded">app.py</code>), (2)
                Data &amp; Persistence Layer (local SQLite tables{' '}
                <code className="font-mono text-xs bg-slate-100 px-1 rounded">patients</code> and{' '}
                <code className="font-mono text-xs bg-slate-100 px-1 rounded">predictions</code>),
                (3) Preprocessing &amp; Feature Engineering Layer (stratified split before fitting
                imputers, encoders, and scalers), and (4) Machine Learning &amp; Reporting Layer
                (four multi-class classifiers, ROC/AUC evaluator, and PDF/CSV report generator).
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">08. Dataset Description</h2>
              <p className="mt-1.5">
                The default dataset contains 320 clearly labeled synthetic demonstration records
                (plus controlled missing cells and duplicates in the raw CSV to demonstrate real
                data cleaning). Features include Age, Gender, Height, Weight, BMI, Systolic BP,
                Diastolic BP, Blood Glucose, Total Cholesterol, Heart Rate, Smoking Status, Physical
                Activity, Family Medical History, and Existing Medical Conditions. The 3-class
                supervised target <code className="font-mono text-xs bg-slate-100 px-1 rounded">risk_category</code>{' '}
                maps directly to <code className="font-mono text-xs bg-slate-100 px-1 rounded">Low Risk (0)</code>,{' '}
                <code className="font-mono text-xs bg-slate-100 px-1 rounded">Moderate Risk (1)</code>, and{' '}
                <code className="font-mono text-xs bg-slate-100 px-1 rounded">High Risk (2)</code>. Users
                may also upload their own properly licensed CSV dataset at any time.
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">09. Data Preprocessing</h2>
              <p className="mt-1.5">
                Exact duplicate rows are removed prior to partitioning. The dataset is split using
                stratified sampling to preserve class ratios. Strictly on the training fold, numeric
                missing values are imputed via Mean or Median, categorical missing values via Mode,
                outliers are identified via the 1.5× IQR rule, categorical variables are mapped via
                deterministic label encoding, and features are scaled via StandardScaler (Z-score)
                or Min-Max normalization.
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">10. Algorithms Used</h2>
              <ul className="mt-1.5 list-disc pl-5 space-y-1.5">
                <li>
                  <strong>Logistic Regression:</strong> Multinomial softmax linear model with L2
                  regularization providing well-calibrated baseline probabilities.
                </li>
                <li>
                  <strong>Decision Tree Classifier:</strong> Non-parametric CART classifier
                  splitting clinical features using Gini impurity minimization.
                </li>
                <li>
                  <strong>Random Forest Classifier:</strong> Bootstrap-aggregated ensemble of
                  decorrelated decision trees with random feature subsampling to curb variance.
                </li>
                <li>
                  <strong>Gradient Boosting Classifier:</strong> Stagewise additive tree ensemble
                  sequentially minimizing multinomial negative log-likelihood gradients.
                </li>
              </ul>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">11. Mathematical Explanation</h2>
              <div className="mt-2 space-y-2.5 font-mono text-xs bg-slate-50 border border-slate-200 rounded-md p-4 text-slate-800">
                <div>BMI = Weight(kg) / [Height(m)]²</div>
                <div>Z-Score Scaling: z = (x − μ_train) / σ_train</div>
                <div>Min-Max Scaling: x_norm = (x − x_min) / (x_max − x_min)</div>
                <div>IQR Outlier Fence: [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR], IQR = Q3 − Q1</div>
                <div>Softmax Logistic Regression: P(y = k | x) = exp(w_kᵀx + b_k) / Σ_j exp(w_jᵀx + b_j)</div>
                <div>Gini Impurity: Gini(D) = 1 − Σ_k p_k²</div>
                <div>Gradient Boosting Update: F_m(x) = F_(m−1)(x) + η · h_m(x), where h_m fits −∂L/∂F</div>
                <div>F1-Score: F1 = 2 × (Precision × Recall) / (Precision + Recall)</div>
              </div>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">12. Results and Discussion</h2>
              <p className="mt-1.5">
                All evaluation scores below are computed live from actual hold-out test predictions
                on the active dataset (never hardcoded or fabricated):
              </p>
              {bundle && (
                <div className="mt-3 overflow-x-auto border border-slate-200 rounded-md">
                  <table className="w-full text-left text-xs font-mono tabular-nums">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 font-sans text-slate-600">
                        <th className="py-2 px-3">Model</th>
                        <th className="py-2 px-3 text-right">Accuracy</th>
                        <th className="py-2 px-3 text-right">Macro Precision</th>
                        <th className="py-2 px-3 text-right">Macro Recall</th>
                        <th className="py-2 px-3 text-right">Macro F1</th>
                        <th className="py-2 px-3 text-right">Macro AUC</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {orderedModels.map((k) => {
                        const m = bundle.models[k];
                        return (
                          <tr key={k}>
                            <td className="py-2 px-3 font-sans font-medium text-slate-900">
                              {m.name}
                            </td>
                            <td className="py-2 px-3 text-right">{(m.accuracy * 100).toFixed(2)}%</td>
                            <td className="py-2 px-3 text-right">
                              {(m.precisionMacro * 100).toFixed(2)}%
                            </td>
                            <td className="py-2 px-3 text-right">
                              {(m.recallMacro * 100).toFixed(2)}%
                            </td>
                            <td className="py-2 px-3 text-right">{(m.f1Macro * 100).toFixed(2)}%</td>
                            <td className="py-2 px-3 text-right">{m.macroAuc.toFixed(4)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h2 className="text-base font-bold text-slate-900">13. Advantages</h2>
                <ul className="mt-1.5 list-disc pl-4 space-y-1 text-xs sm:text-sm">
                  <li>Zero data leakage through split-first preprocessing pipelines.</li>
                  <li>Side-by-side benchmarking across linear, tree, bagging, and boosting models.</li>
                  <li>Transparent feature importance and per-patient contributing factor breakdown.</li>
                  <li>Local SQLite persistence with instant de-identified PDF/CSV exports.</li>
                </ul>
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900">14. Limitations</h2>
                <ul className="mt-1.5 list-disc pl-4 space-y-1 text-xs sm:text-sm">
                  <li>Default training cohort is synthetic demonstration data, not hospital EHR data.</li>
                  <li>Does not incorporate longitudinal lab trends, medications, or imaging studies.</li>
                  <li>Not clinically validated or regulated for diagnostic or treatment decisions.</li>
                </ul>
              </div>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">15. Future Enhancements</h2>
              <p className="mt-1.5">
                Future iterations can integrate SHAP local explainability values, k-fold
                cross-validation hyperparameter search, calibration curves (Brier score), FHIR/HL7
                interoperability connectors, and multi-site prospective clinical validation studies.
              </p>
            </div>

            <div>
              <h2 className="text-base font-bold text-slate-900">16. Conclusion</h2>
              <p className="mt-1.5">
                The Multi-Factor Healthcare Risk Prediction System demonstrates how disciplined
                data preprocessing, multi-model supervised learning, rigorous hold-out evaluation,
                and clear visual analytics combine into a coherent educational healthcare decision
                support tool.
              </p>
            </div>
          </section>
        </div>
      )}

      {tab === 'python' && (
        <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {pyFiles.map((f) => (
                <button
                  key={f.name}
                  type="button"
                  onClick={() => setSelectedFile(f.name)}
                  className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                    selectedFile === f.name
                      ? 'bg-sky-700 text-white'
                      : 'bg-white border border-slate-200 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  {f.name}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-50 whitespace-nowrap"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-teal-700" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy File'}
            </button>
          </div>
          <div className="p-4 bg-slate-950 text-slate-100 overflow-x-auto max-h-[620px]">
            <pre className="text-xs font-mono leading-relaxed whitespace-pre">{fileContent}</pre>
          </div>
        </div>
      )}

      {tab === 'setup' && (
        <section className="rounded-lg border border-slate-200 bg-white p-6 space-y-5 text-sm text-slate-700">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Windows PowerShell Installation &amp; Execution Guide (Python + Streamlit)
            </h2>
            <p className="mt-1 text-xs text-slate-600">
              Complete modular Python source files are organized inside{' '}
              <code className="font-mono bg-slate-100 px-1 rounded">Healthcare_Risk_Prediction/</code>.
              Follow these steps in Windows PowerShell to run the standalone Streamlit edition locally.
            </p>
          </div>

          <div className="space-y-3 font-mono text-xs bg-slate-950 text-slate-100 rounded-md p-4 overflow-x-auto leading-relaxed">
            <div className="text-slate-400"># 1. Navigate into the Python project folder</div>
            <div>cd Healthcare_Risk_Prediction</div>
            <div className="pt-2 text-slate-400"># 2. Create and activate a Python virtual environment</div>
            <div>python -m venv venv</div>
            <div>.\venv\Scripts\Activate.ps1</div>
            <div className="pt-2 text-slate-400"># 3. Upgrade pip and install all dependencies</div>
            <div>python -m pip install --upgrade pip</div>
            <div>pip install -r requirements.txt</div>
            <div className="pt-2 text-slate-400"># 4. Train and serialize all 4 ML models (scikit-learn)</div>
            <div>python train_models.py</div>
            <div className="pt-2 text-slate-400"># 5. Launch the interactive Streamlit dashboard</div>
            <div>streamlit run app.py</div>
          </div>

          <div className="space-y-2 text-xs text-slate-600">
            <h3 className="text-sm font-semibold text-slate-900">Python Module Responsibilities</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li><code className="font-mono text-slate-900">app.py</code> — Main 10-page Streamlit dashboard connecting all modules.</li>
              <li><code className="font-mono text-slate-900">database.py</code> — Parameterized SQLite storage for patient records and predictions.</li>
              <li><code className="font-mono text-slate-900">preprocessing.py</code> — Missing value detection, Mean/Median/Mode imputation, duplicate removal, IQR outliers, encoding, scaling, and stratified train/test splitting.</li>
              <li><code className="font-mono text-slate-900">train_models.py</code> — Leak-free scikit-learn Pipelines for Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting with real metric calculation.</li>
              <li><code className="font-mono text-slate-900">prediction.py</code> — Loads serialized pipelines for interactive patient risk prediction and recommendations.</li>
              <li><code className="font-mono text-slate-900">visualization.py</code> — Generates all 12 Matplotlib, Seaborn, and Plotly charts from actual dataset values.</li>
              <li><code className="font-mono text-slate-900">report_generator.py</code> — Exports de-identified patient risk reports as PDF (FPDF) and CSV.</li>
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
