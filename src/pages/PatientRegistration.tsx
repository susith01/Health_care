import React, { useMemo, useState } from 'react';
import { CheckCircle2, RotateCcw, UserPlus } from 'lucide-react';
import { api } from '../lib/api';
import type {
  ExistingConditions,
  FamilyHistory,
  Gender,
  PhysicalActivity,
  SmokingStatus,
} from '../types';

interface FormState {
  patient_id: string;
  age: string;
  gender: Gender;
  height_cm: string;
  weight_kg: string;
  systolic_bp: string;
  diastolic_bp: string;
  glucose_level: string;
  cholesterol_level: string;
  heart_rate: string;
  smoking_status: SmokingStatus;
  physical_activity: PhysicalActivity;
  family_history: FamilyHistory;
  existing_conditions: ExistingConditions;
}

const initialForm = (): FormState => ({
  patient_id: `PAT-${Math.floor(1000 + Math.random() * 9000)}`,
  age: '46',
  gender: 'Female',
  height_cm: '166',
  weight_kg: '71',
  systolic_bp: '126',
  diastolic_bp: '82',
  glucose_level: '104',
  cholesterol_level: '208',
  heart_rate: '76',
  smoking_status: 'Never',
  physical_activity: 'Moderate',
  family_history: 'Hypertension',
  existing_conditions: 'None',
});

export function PatientRegistration() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [savedNotice, setSavedNotice] = useState<{
    patientId: string;
    bmi: number;
    risk: string;
    modelName: string;
  } | null>(null);

  const bmiCalc = useMemo(() => {
    const h = Number(form.height_cm);
    const w = Number(form.weight_kg);
    if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0) {
      return { value: null as number | null, category: 'Enter valid height and weight' };
    }
    const bmi = Math.round((w / Math.pow(h / 100, 2)) * 10) / 10;
    const category =
      bmi < 18.5
        ? 'Underweight (<18.5)'
        : bmi < 25
          ? 'Normal (18.5–24.9)'
          : bmi < 30
            ? 'Overweight (25.0–29.9)'
            : 'Obese (≥30.0)';
    return { value: bmi, category };
  }, [form.height_cm, form.weight_kg]);

  const validateClient = (): Record<string, string> => {
    const errs: Record<string, string> = {};
    if (!/^[A-Za-z0-9_-]{3,24}$/.test(form.patient_id.trim())) {
      errs.patient_id = 'Use 3–24 characters (letters, numbers, hyphen, or underscore).';
    }
    const check = (key: keyof FormState, label: string, min: number, max: number) => {
      const val = Number(form[key]);
      if (!Number.isFinite(val) || form[key] === '') {
        errs[key] = `${label} is required.`;
      } else if (val < min || val > max) {
        errs[key] = `${label} must be between ${min} and ${max}.`;
      }
    };
    check('age', 'Age', 1, 120);
    check('height_cm', 'Height', 80, 250);
    check('weight_kg', 'Weight', 20, 300);
    check('systolic_bp', 'Systolic BP', 70, 260);
    check('diastolic_bp', 'Diastolic BP', 40, 160);
    if (Number(form.systolic_bp) <= Number(form.diastolic_bp)) {
      errs.systolic_bp = 'Systolic BP must be higher than Diastolic BP.';
    }
    check('glucose_level', 'Glucose Level', 40, 600);
    check('cholesterol_level', 'Cholesterol Level', 80, 600);
    check('heart_rate', 'Heart Rate', 35, 220);
    return errs;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setGlobalError(null);
    setSavedNotice(null);

    const clientErrs = validateClient();
    if (Object.keys(clientErrs).length > 0) {
      setFieldErrors(clientErrs);
      return;
    }
    setFieldErrors({});
    setSubmitting(true);

    try {
      const res = await api.createPatient({
        patient_id: form.patient_id.trim(),
        age: Number(form.age),
        gender: form.gender,
        height_cm: Number(form.height_cm),
        weight_kg: Number(form.weight_kg),
        systolic_bp: Number(form.systolic_bp),
        diastolic_bp: Number(form.diastolic_bp),
        glucose_level: Number(form.glucose_level),
        cholesterol_level: Number(form.cholesterol_level),
        heart_rate: Number(form.heart_rate),
        smoking_status: form.smoking_status,
        physical_activity: form.physical_activity,
        family_history: form.family_history,
        existing_conditions: form.existing_conditions,
      });

      setSavedNotice({
        patientId: res.patient.patient_id,
        bmi: res.patient.bmi,
        risk: res.initialPrediction.predicted_risk,
        modelName: res.initialPrediction.model_name,
      });
      setForm({
        ...form,
        patient_id: `PAT-${Math.floor(1000 + Math.random() * 9000)}`,
      });
    } catch (err) {
      const eObj = err as Error & { fieldErrors?: Record<string, string> };
      if (eObj.fieldErrors) setFieldErrors(eObj.fieldErrors);
      setGlobalError(eObj.message || 'Unable to register patient.');
    } finally {
      setSubmitting(false);
    }
  };

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    if (fieldErrors[key]) {
      setFieldErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  };

  const inputClass = (key: string) =>
    `w-full rounded-md border px-3 py-2 text-sm text-slate-900 bg-white focus:outline-none focus:ring-2 focus:ring-sky-600/20 focus:border-sky-600 ${
      fieldErrors[key] ? 'border-red-400' : 'border-slate-300'
    }`;

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Patient Registration</h1>
        <p className="mt-1 text-sm text-slate-600">
          Record de-identified patient parameters in local SQLite. BMI is calculated automatically
          and an initial risk tier is estimated by the top trained model.
        </p>
      </div>

      {savedNotice && (
        <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-teal-700 shrink-0 mt-0.5" />
          <div className="text-sm text-slate-800">
            <span className="font-semibold text-teal-900">
              Patient {savedNotice.patientId} saved to SQLite.
            </span>{' '}
            Calculated BMI: <span className="font-mono tabular-nums">{savedNotice.bmi}</span> kg/m²
            · Initial ML estimate ({savedNotice.modelName}):{' '}
            <span className="font-semibold">{savedNotice.risk}</span>.
          </div>
        </div>
      )}

      {globalError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {globalError}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6" noValidate>
        {/* Demographics & Anthropometrics */}
        <section className="rounded-lg border border-slate-200 bg-white p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              01. Patient Identifier &amp; Anthropometrics
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Use an anonymized identifier to protect patient privacy.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1" htmlFor="patient_id">
                Patient ID
              </label>
              <input
                id="patient_id"
                type="text"
                value={form.patient_id}
                onChange={(e) => update('patient_id', e.target.value)}
                className={`${inputClass('patient_id')} font-mono`}
                placeholder="PAT-1042"
              />
              {fieldErrors.patient_id && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.patient_id}</p>
              )}
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1" htmlFor="age">
                Age (years)
              </label>
              <input
                id="age"
                type="number"
                value={form.age}
                onChange={(e) => update('age', e.target.value)}
                className={`${inputClass('age')} font-mono tabular-nums`}
              />
              {fieldErrors.age && <p className="mt-1 text-xs text-red-600">{fieldErrors.age}</p>}
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1" htmlFor="gender">
                Gender
              </label>
              <select
                id="gender"
                value={form.gender}
                onChange={(e) => update('gender', e.target.value as Gender)}
                className={inputClass('gender')}
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1" htmlFor="height_cm">
                Height (cm)
              </label>
              <input
                id="height_cm"
                type="number"
                step="0.1"
                value={form.height_cm}
                onChange={(e) => update('height_cm', e.target.value)}
                className={`${inputClass('height_cm')} font-mono tabular-nums`}
              />
              {fieldErrors.height_cm && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.height_cm}</p>
              )}
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1" htmlFor="weight_kg">
                Weight (kg)
              </label>
              <input
                id="weight_kg"
                type="number"
                step="0.1"
                value={form.weight_kg}
                onChange={(e) => update('weight_kg', e.target.value)}
                className={`${inputClass('weight_kg')} font-mono tabular-nums`}
              />
              {fieldErrors.weight_kg && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.weight_kg}</p>
              )}
            </div>

            <div>
              <span className="block text-xs font-medium text-slate-700 mb-1">
                BMI (Auto-Calculated)
              </span>
              <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm flex items-center justify-between">
                <span className="font-mono font-semibold tabular-nums text-slate-900">
                  {bmiCalc.value !== null ? `${bmiCalc.value.toFixed(1)} kg/m²` : '—'}
                </span>
                <span className="text-xs text-slate-600 truncate ml-2">{bmiCalc.category}</span>
              </div>
            </div>
          </div>
        </section>

        {/* Clinical Vitals & Biomarkers */}
        <section className="rounded-lg border border-slate-200 bg-white p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              02. Clinical Vitals &amp; Laboratory Biomarkers
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Input resting cardiovascular and metabolic measurements.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="systolic_bp"
              >
                Systolic Blood Pressure (mmHg)
              </label>
              <input
                id="systolic_bp"
                type="number"
                value={form.systolic_bp}
                onChange={(e) => update('systolic_bp', e.target.value)}
                className={`${inputClass('systolic_bp')} font-mono tabular-nums`}
              />
              {fieldErrors.systolic_bp && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.systolic_bp}</p>
              )}
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="diastolic_bp"
              >
                Diastolic Blood Pressure (mmHg)
              </label>
              <input
                id="diastolic_bp"
                type="number"
                value={form.diastolic_bp}
                onChange={(e) => update('diastolic_bp', e.target.value)}
                className={`${inputClass('diastolic_bp')} font-mono tabular-nums`}
              />
              {fieldErrors.diastolic_bp && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.diastolic_bp}</p>
              )}
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="heart_rate"
              >
                Heart Rate (bpm)
              </label>
              <input
                id="heart_rate"
                type="number"
                value={form.heart_rate}
                onChange={(e) => update('heart_rate', e.target.value)}
                className={`${inputClass('heart_rate')} font-mono tabular-nums`}
              />
              {fieldErrors.heart_rate && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.heart_rate}</p>
              )}
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="glucose_level"
              >
                Blood Glucose Level (mg/dL)
              </label>
              <input
                id="glucose_level"
                type="number"
                step="0.1"
                value={form.glucose_level}
                onChange={(e) => update('glucose_level', e.target.value)}
                className={`${inputClass('glucose_level')} font-mono tabular-nums`}
              />
              {fieldErrors.glucose_level && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.glucose_level}</p>
              )}
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="cholesterol_level"
              >
                Cholesterol Level (mg/dL)
              </label>
              <input
                id="cholesterol_level"
                type="number"
                step="0.1"
                value={form.cholesterol_level}
                onChange={(e) => update('cholesterol_level', e.target.value)}
                className={`${inputClass('cholesterol_level')} font-mono tabular-nums`}
              />
              {fieldErrors.cholesterol_level && (
                <p className="mt-1 text-xs text-red-600">{fieldErrors.cholesterol_level}</p>
              )}
            </div>
          </div>
        </section>

        {/* Lifestyle & Medical History */}
        <section className="rounded-lg border border-slate-200 bg-white p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              03. Lifestyle &amp; Medical History
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Behavioral risk factors and known family or chronic conditions.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="smoking_status"
              >
                Smoking Status
              </label>
              <select
                id="smoking_status"
                value={form.smoking_status}
                onChange={(e) => update('smoking_status', e.target.value as SmokingStatus)}
                className={inputClass('smoking_status')}
              >
                <option value="Never">Never</option>
                <option value="Former">Former</option>
                <option value="Current">Current</option>
              </select>
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="physical_activity"
              >
                Physical Activity Level
              </label>
              <select
                id="physical_activity"
                value={form.physical_activity}
                onChange={(e) => update('physical_activity', e.target.value as PhysicalActivity)}
                className={inputClass('physical_activity')}
              >
                <option value="High">High</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low</option>
              </select>
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="family_history"
              >
                Family Medical History
              </label>
              <select
                id="family_history"
                value={form.family_history}
                onChange={(e) => update('family_history', e.target.value as FamilyHistory)}
                className={inputClass('family_history')}
              >
                <option value="None">None</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Diabetes">Diabetes</option>
                <option value="Heart Disease">Heart Disease</option>
                <option value="Multiple">Multiple</option>
              </select>
            </div>

            <div>
              <label
                className="block text-xs font-medium text-slate-700 mb-1"
                htmlFor="existing_conditions"
              >
                Existing Medical Conditions
              </label>
              <select
                id="existing_conditions"
                value={form.existing_conditions}
                onChange={(e) => update('existing_conditions', e.target.value as ExistingConditions)}
                className={inputClass('existing_conditions')}
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
        </section>

        <div className="flex flex-wrap items-center justify-end gap-3">
          <button
            type="button"
            onClick={() => {
              setForm(initialForm());
              setFieldErrors({});
              setGlobalError(null);
            }}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Form
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="inline-flex items-center gap-2 px-5 py-2 text-sm font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
          >
            <UserPlus className="w-4 h-4" />
            {submitting ? 'Saving to SQLite...' : 'Save Patient Record'}
          </button>
        </div>
      </form>
    </div>
  );
}
