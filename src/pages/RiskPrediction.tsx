import React, { useEffect, useMemo, useState } from 'react';
import { Activity, Download, FileSpreadsheet, ShieldAlert } from 'lucide-react';
import { api } from '../lib/api';
import { downloadReportCsv, downloadReportPdf } from '../lib/reportExport';
import type {
  ExistingConditions,
  FamilyHistory,
  Gender,
  ModelKey,
  PatientRecord,
  PhysicalActivity,
  PredictionResult,
  SavedReport,
  SmokingStatus,
  TrainedModelsBundle,
} from '../types';

export function RiskPrediction() {
  const [modelsBundle, setModelsBundle] = useState<TrainedModelsBundle | null>(null);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [modelKey, setModelKey] = useState<ModelKey>('gradient_boosting');
  const [saveReport, setSaveReport] = useState(true);

  const [patientId, setPatientId] = useState('EVAL-2041');
  const [age, setAge] = useState('54');
  const [gender, setGender] = useState<Gender>('Male');
  const [heightCm, setHeightCm] = useState('174');
  const [weightKg, setWeightKg] = useState('89');
  const [systolicBp, setSystolicBp] = useState('142');
  const [diastolicBp, setDiastolicBp] = useState('91');
  const [glucoseLevel, setGlucoseLevel] = useState('138');
  const [cholesterolLevel, setCholesterolLevel] = useState('236');
  const [heartRate, setHeartRate] = useState('84');
  const [smokingStatus, setSmokingStatus] = useState<SmokingStatus>('Former');
  const [physicalActivity, setPhysicalActivity] = useState<PhysicalActivity>('Low');
  const [familyHistory, setFamilyHistory] = useState<FamilyHistory>('Heart Disease');
  const [existingConditions, setExistingConditions] = useState<ExistingConditions>('Hypertension');

  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    Promise.all([api.getModels(), api.getPatients()])
      .then(([bundle, patRes]) => {
        setModelsBundle(bundle);
        setModelKey(bundle.bestModelKey);
        setPatients(patRes.patients.slice(0, 60));
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load models.'));
  }, []);

  const bmiPreview = useMemo(() => {
    const h = Number(heightCm);
    const w = Number(weightKg);
    if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0) return '—';
    return (w / Math.pow(h / 100, 2)).toFixed(1);
  }, [heightCm, weightKg]);

  const handleLoadPatient = (pid: string) => {
    setSelectedPatientId(pid);
    const found = patients.find((p) => p.patient_id === pid);
    if (!found) return;
    setPatientId(found.patient_id);
    setAge(String(found.age));
    setGender(found.gender);
    setHeightCm(String(found.height_cm));
    setWeightKg(String(found.weight_kg));
    setSystolicBp(String(found.systolic_bp));
    setDiastolicBp(String(found.diastolic_bp));
    setGlucoseLevel(String(found.glucose_level));
    setCholesterolLevel(String(found.cholesterol_level));
    setHeartRate(String(found.heart_rate));
    setSmokingStatus(found.smoking_status);
    setPhysicalActivity(found.physical_activity);
    setFamilyHistory(found.family_history);
    setExistingConditions(found.existing_conditions);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPredicting(true);
    try {
      const res = await api.predictRisk({
        patient_id: patientId.trim() || undefined,
        model_key: modelKey,
        save_report: saveReport,
        age: Number(age),
        gender,
        height_cm: Number(heightCm),
        weight_kg: Number(weightKg),
        systolic_bp: Number(systolicBp),
        diastolic_bp: Number(diastolicBp),
        glucose_level: Number(glucoseLevel),
        cholesterol_level: Number(cholesterolLevel),
        heart_rate: Number(heartRate),
        smoking_status: smokingStatus,
        physical_activity: physicalActivity,
        family_history: familyHistory,
        existing_conditions: existingConditions,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setPredicting(false);
    }
  };

  const buildSavedReportPayload = (): SavedReport | null => {
    if (!result) return null;
    return {
      id: result.report_id ?? 0,
      patient_id: result.patient_id,
      model_key: result.model_key,
      model_name: result.model_name,
      predicted_risk: result.predicted_risk,
      probabilities: result.probabilities,
      patient_params: {
        age: Number(age),
        gender,
        height_cm: Number(heightCm),
        weight_kg: Number(weightKg),
        bmi: result.bmi,
        systolic_bp: Number(systolicBp),
        diastolic_bp: Number(diastolicBp),
        glucose_level: Number(glucoseLevel),
        cholesterol_level: Number(cholesterolLevel),
        heart_rate: Number(heartRate),
        smoking_status: smokingStatus,
        physical_activity: physicalActivity,
        family_history: familyHistory,
        existing_conditions: existingConditions,
      },
      contributing_factors: result.contributing_factors,
      recommendations: result.recommendations,
      created_at: result.created_at,
    };
  };

  const riskAccent = (risk: string) => {
    if (risk === 'High Risk') return 'text-red-700 border-red-200 bg-red-50/70';
    if (risk === 'Moderate Risk') return 'text-amber-800 border-amber-200 bg-amber-50/70';
    return 'text-teal-800 border-teal-200 bg-teal-50/70';
  };

  const flagStyle = (status: string) =>
    status === 'Elevated'
      ? 'text-red-700 font-semibold'
      : status === 'Borderline'
        ? 'text-amber-700 font-medium'
        : 'text-teal-700';

  return (
    <div className="space-y-6">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          Healthcare Risk Prediction
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Select a trained multi-class classifier and evaluate patient health parameters. Output
          categories are produced directly by the trained 3-class model.
        </p>
      </div>

      {/* Prominent Educational Warning */}
      <div className="rounded-lg border border-amber-200 bg-amber-50/70 px-4 py-3 flex items-start gap-3">
        <ShieldAlert className="w-4 h-4 text-amber-800 mt-0.5 shrink-0" />
        <p className="text-xs sm:text-sm text-amber-950 leading-relaxed">
          <span className="font-semibold">Not a Medical Diagnosis: </span>
          Predictions generated by this machine learning application are for educational decision
          support only. Always seek evaluation from a licensed physician for clinical care.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Input Form */}
        <form
          onSubmit={handlePredict}
          className="xl:col-span-5 rounded-lg border border-slate-200 bg-white p-5 space-y-4"
        >
          <div className="space-y-3 pb-3 border-b border-slate-200">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Trained Machine Learning Model
              </label>
              <select
                value={modelKey}
                onChange={(e) => setModelKey(e.target.value as ModelKey)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white font-medium text-slate-900"
              >
                {modelsBundle &&
                  (Object.keys(modelsBundle.models) as ModelKey[]).map((k) => {
                    const m = modelsBundle.models[k];
                    return (
                      <option key={k} value={k}>
                        {m.name} (Acc {(m.accuracy * 100).toFixed(1)}% · AUC {m.macroAuc.toFixed(3)})
                      </option>
                    );
                  })}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Load Existing SQLite Patient (Optional)
              </label>
              <select
                value={selectedPatientId}
                onChange={(e) => handleLoadPatient(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white text-slate-700"
              >
                <option value="">Manual entry...</option>
                {patients.map((p) => (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.patient_id} — Age {p.age}, {p.gender}, BP {p.systolic_bp}/{p.diastolic_bp} (
                    {p.record_source === 'registered' ? 'Registered' : 'Synthetic'})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block font-medium text-slate-700 mb-1">Patient ID</label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Age (years)</label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Gender</label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value as Gender)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 bg-white"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">
                Height (cm) / BMI {bmiPreview}
              </label>
              <input
                type="number"
                step="0.1"
                value={heightCm}
                onChange={(e) => setHeightCm(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Weight (kg)</label>
              <input
                type="number"
                step="0.1"
                value={weightKg}
                onChange={(e) => setWeightKg(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Heart Rate (bpm)</label>
              <input
                type="number"
                value={heartRate}
                onChange={(e) => setHeartRate(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Systolic BP (mmHg)</label>
              <input
                type="number"
                value={systolicBp}
                onChange={(e) => setSystolicBp(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Diastolic BP (mmHg)</label>
              <input
                type="number"
                value={diastolicBp}
                onChange={(e) => setDiastolicBp(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Glucose (mg/dL)</label>
              <input
                type="number"
                value={glucoseLevel}
                onChange={(e) => setGlucoseLevel(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Cholesterol (mg/dL)</label>
              <input
                type="number"
                value={cholesterolLevel}
                onChange={(e) => setCholesterolLevel(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Smoking Status</label>
              <select
                value={smokingStatus}
                onChange={(e) => setSmokingStatus(e.target.value as SmokingStatus)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 bg-white"
              >
                <option value="Never">Never</option>
                <option value="Former">Former</option>
                <option value="Current">Current</option>
              </select>
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Physical Activity</label>
              <select
                value={physicalActivity}
                onChange={(e) => setPhysicalActivity(e.target.value as PhysicalActivity)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 bg-white"
              >
                <option value="High">High</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low</option>
              </select>
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Family History</label>
              <select
                value={familyHistory}
                onChange={(e) => setFamilyHistory(e.target.value as FamilyHistory)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 bg-white"
              >
                <option value="None">None</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Diabetes">Diabetes</option>
                <option value="Heart Disease">Heart Disease</option>
                <option value="Multiple">Multiple</option>
              </select>
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Existing Conditions</label>
              <select
                value={existingConditions}
                onChange={(e) => setExistingConditions(e.target.value as ExistingConditions)}
                className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 bg-white"
              >
                <option value="None">None</option>
                <option value="Asthma">Asthma</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Diabetes">Diabetes</option>
                <option value="Heart Disease">Heart Disease</option>
                <option value="Multiple">Multiple</option>
              </select>
            </div>
          </div>

          <label className="flex items-center gap-2 text-xs text-slate-700 pt-1">
            <input
              type="checkbox"
              checked={saveReport}
              onChange={(e) => setSaveReport(e.target.checked)}
              className="rounded border-slate-300 text-sky-700"
            />
            Save prediction record to SQLite Reports archive
          </label>

          <button
            type="submit"
            disabled={predicting}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 disabled:opacity-60 transition-colors"
          >
            <Activity className="w-4 h-4" />
            {predicting ? 'Running Inference...' : 'Predict Healthcare Risk'}
          </button>
        </form>

        {/* Prediction Result Panel */}
        <div className="xl:col-span-7 space-y-5">
          {!result ? (
            <div className="rounded-lg border border-slate-200 bg-white p-10 text-center">
              <Activity className="w-9 h-9 text-slate-300 mx-auto mb-3" />
              <h2 className="text-base font-semibold text-slate-800">Ready for Risk Estimation</h2>
              <p className="mt-1 text-sm text-slate-500 max-w-md mx-auto">
                Configure patient parameters on the left and click{' '}
                <strong>Predict Healthcare Risk</strong> to compute multi-class risk probabilities,
                top contributing features, and general health recommendations.
              </p>
            </div>
          ) : (
            <>
              {/* Risk Banner & Probabilities */}
              <section className={`rounded-lg border p-5 ${riskAccent(result.predicted_risk)}`}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <div className="text-xs font-medium opacity-80">
                      Patient {result.patient_id} · {result.model_name}
                    </div>
                    <div className="mt-1 text-2xl font-bold">{result.predicted_risk}</div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => {
                        const rep = buildSavedReportPayload();
                        if (rep) downloadReportPdf(rep);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-white border border-slate-300 rounded-md hover:bg-slate-50 whitespace-nowrap"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download PDF
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const rep = buildSavedReportPayload();
                        if (rep) downloadReportCsv(rep);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-white border border-slate-300 rounded-md hover:bg-slate-50 whitespace-nowrap"
                    >
                      <FileSpreadsheet className="w-3.5 h-3.5" />
                      Download CSV
                    </button>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-3">
                  {(['Low Risk', 'Moderate Risk', 'High Risk'] as const).map((tier) => {
                    const probPct = (result.probabilities[tier] * 100).toFixed(1);
                    return (
                      <div key={tier} className="rounded-md bg-white/90 border border-slate-200/80 p-3">
                        <div className="text-xs text-slate-600">{tier}</div>
                        <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
                          {probPct}%
                        </div>
                        <div className="mt-1.5 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={
                              tier === 'High Risk'
                                ? 'h-full bg-red-600'
                                : tier === 'Moderate Risk'
                                  ? 'h-full bg-amber-500'
                                  : 'h-full bg-teal-600'
                            }
                            style={{ width: `${probPct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-3 text-[11px] text-slate-600">{result.target_definition_note}</p>
              </section>

              {/* Top Contributing Features */}
              <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-200">
                  <h2 className="text-sm font-semibold text-slate-900">
                    Important Contributing Features (Model Importance × Patient Profile)
                  </h2>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                        <th className="py-2 px-4">Clinical Feature</th>
                        <th className="py-2 px-3">Patient Value</th>
                        <th className="py-2 px-3">Reference</th>
                        <th className="py-2 px-3">Clinical Flag</th>
                        <th className="py-2 px-4 text-right">Impact Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.contributing_factors.map((f) => (
                        <tr key={f.feature}>
                          <td className="py-2 px-4 font-medium text-slate-800">{f.label}</td>
                          <td className="py-2 px-3 font-mono tabular-nums text-slate-900">
                            {f.patientValue}
                          </td>
                          <td className="py-2 px-3 text-slate-500">{f.referenceRange}</td>
                          <td className={`py-2 px-3 ${flagStyle(f.status)}`}>{f.status}</td>
                          <td className="py-2 px-4 text-right font-mono tabular-nums text-slate-700">
                            {f.contributionScore.toFixed(1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Parameter Summary & Recommendations */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <section className="rounded-lg border border-slate-200 bg-white p-5">
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">
                    Health Parameter Summary
                  </h3>
                  <div className="space-y-2 text-xs">
                    {result.parameter_summary.map((item) => (
                      <div
                        key={item.parameter}
                        className="flex items-center justify-between gap-2 pb-1.5 border-b border-slate-100 last:border-b-0"
                      >
                        <span className="text-slate-600">{item.parameter}</span>
                        <span className="font-mono tabular-nums text-slate-900">
                          {item.value}{' '}
                          <span className={`font-sans ml-1 ${flagStyle(item.status)}`}>
                            ({item.status})
                          </span>
                        </span>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="rounded-lg border border-slate-200 bg-white p-5">
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">
                    General Health Recommendations
                  </h3>
                  <ul className="space-y-2.5 text-xs text-slate-700 leading-relaxed list-disc pl-4">
                    {result.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </section>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
