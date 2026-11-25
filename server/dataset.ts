import fs from 'node:fs';
import path from 'node:path';
import type {
  ExistingConditions,
  FamilyHistory,
  Gender,
  PatientRecord,
  PhysicalActivity,
  RawDatasetRow,
  RiskCategory,
  SmokingStatus,
} from '../src/types.ts';

export const DATASET_COLUMNS = [
  'patient_id',
  'age',
  'gender',
  'height_cm',
  'weight_kg',
  'bmi',
  'systolic_bp',
  'diastolic_bp',
  'glucose_level',
  'cholesterol_level',
  'heart_rate',
  'smoking_status',
  'physical_activity',
  'family_history',
  'existing_conditions',
  'risk_category',
] as const;

export const SYNTHETIC_DATASET_NAME =
  'Synthetic Demonstration Healthcare Dataset (320 Records)';

export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randNormal(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function round1(val: number): number {
  return Math.round(val * 10) / 10;
}

/**
 * Computes the documented composite clinical risk tier for synthetic data generation.
 * Target classes ('Low Risk', 'Moderate Risk', 'High Risk') are defined directly
 * from multi-factor cardiometabolic risk accumulation plus realistic biological noise,
 * so ML models learn genuine multi-class decision boundaries rather than arbitrary thresholds.
 */
export function assignSyntheticRiskLabel(
  params: {
    age: number;
    bmi: number;
    systolic_bp: number;
    diastolic_bp: number;
    glucose_level: number;
    cholesterol_level: number;
    heart_rate: number;
    smoking_status: SmokingStatus;
    physical_activity: PhysicalActivity;
    family_history: FamilyHistory;
    existing_conditions: ExistingConditions;
  },
  noise: number
): RiskCategory {
  let score = 0;

  if (params.age >= 65) score += 2.2;
  else if (params.age >= 50) score += 1.3;
  else if (params.age >= 38) score += 0.5;

  if (params.bmi >= 32) score += 1.8;
  else if (params.bmi >= 27) score += 0.9;
  else if (params.bmi < 18.5) score += 0.4;

  if (params.systolic_bp >= 145 || params.diastolic_bp >= 95) score += 2.4;
  else if (params.systolic_bp >= 130 || params.diastolic_bp >= 85) score += 1.2;

  if (params.glucose_level >= 150) score += 2.3;
  else if (params.glucose_level >= 115) score += 1.1;

  if (params.cholesterol_level >= 245) score += 1.9;
  else if (params.cholesterol_level >= 205) score += 0.9;

  if (params.heart_rate >= 98) score += 1.0;
  else if (params.heart_rate >= 86) score += 0.4;

  if (params.smoking_status === 'Current') score += 1.8;
  else if (params.smoking_status === 'Former') score += 0.6;

  if (params.physical_activity === 'Low') score += 1.2;
  else if (params.physical_activity === 'Moderate') score += 0.3;
  else score -= 0.5;

  if (params.family_history === 'Multiple') score += 1.6;
  else if (params.family_history !== 'None') score += 0.8;

  if (params.existing_conditions === 'Multiple' || params.existing_conditions === 'Heart Disease') {
    score += 2.4;
  } else if (params.existing_conditions === 'Diabetes' || params.existing_conditions === 'Hypertension') {
    score += 1.5;
  } else if (params.existing_conditions === 'Asthma') {
    score += 0.6;
  }

  const finalScore = score + noise;
  if (finalScore >= 8.2) return 'High Risk';
  if (finalScore >= 4.2) return 'Moderate Risk';
  return 'Low Risk';
}

export function generateSyntheticDataset(
  count = 320,
  seed = 42
): {
  rawRows: RawDatasetRow[];
  cleanRecords: PatientRecord[];
} {
  const rng = createSeededRng(seed);
  const cleanRecords: PatientRecord[] = [];

  for (let i = 0; i < count; i++) {
    const patientId = `SYN-${String(1001 + i).padStart(4, '0')}`;
    const archetypeRoll = rng();
    // Create 3 realistic patient clusters with natural overlap
    const riskProfile = archetypeRoll < 0.35 ? 0 : archetypeRoll < 0.72 ? 1 : 2;

    const genderRoll = rng();
    const gender: Gender =
      genderRoll < 0.48 ? 'Female' : genderRoll < 0.96 ? 'Male' : 'Other';

    const ageMean = riskProfile === 0 ? 33 : riskProfile === 1 ? 51 : 64;
    const age = Math.round(clamp(randNormal(rng, ageMean, 10), 20, 86));

    const heightBase = gender === 'Male' ? 175 : gender === 'Female' ? 162 : 168;
    const height_cm = round1(clamp(randNormal(rng, heightBase, 7.5), 146, 198));

    const targetBmiMean = riskProfile === 0 ? 22.8 : riskProfile === 1 ? 27.6 : 32.1;
    const bmiSample = clamp(randNormal(rng, targetBmiMean, 3.6), 17.2, 46.5);
    const weight_kg = round1(clamp(bmiSample * Math.pow(height_cm / 100, 2), 42, 165));
    const bmi = round1(weight_kg / Math.pow(height_cm / 100, 2));

    const sbpMean = riskProfile === 0 ? 114 : riskProfile === 1 ? 131 : 151;
    const systolic_bp = Math.round(
      clamp(randNormal(rng, sbpMean + (age - 45) * 0.25, 11), 92, 205)
    );
    const diastolic_bp = Math.round(
      clamp(systolic_bp * 0.62 + randNormal(rng, 2, 5.5), 58, 125)
    );

    const gluMean = riskProfile === 0 ? 90 : riskProfile === 1 ? 118 : 162;
    const glucose_level = Math.round(
      clamp(randNormal(rng, gluMean + (bmi - 25) * 1.1, 20), 68, 310)
    );

    const cholMean = riskProfile === 0 ? 176 : riskProfile === 1 ? 214 : 254;
    const cholesterol_level = Math.round(
      clamp(randNormal(rng, cholMean + (age - 45) * 0.4, 24), 125, 360)
    );

    const hrMean = riskProfile === 0 ? 69 : riskProfile === 1 ? 78 : 89;
    const heart_rate = Math.round(clamp(randNormal(rng, hrMean, 9.5), 50, 128));

    const smokeRoll = rng();
    const smoking_status: SmokingStatus =
      riskProfile === 0
        ? smokeRoll < 0.78
          ? 'Never'
          : smokeRoll < 0.93
            ? 'Former'
            : 'Current'
        : riskProfile === 1
          ? smokeRoll < 0.45
            ? 'Never'
            : smokeRoll < 0.75
              ? 'Former'
              : 'Current'
          : smokeRoll < 0.22
            ? 'Never'
            : smokeRoll < 0.52
              ? 'Former'
              : 'Current';

    const actRoll = rng();
    const physical_activity: PhysicalActivity =
      riskProfile === 0
        ? actRoll < 0.52
          ? 'High'
          : actRoll < 0.88
            ? 'Moderate'
            : 'Low'
        : riskProfile === 1
          ? actRoll < 0.22
            ? 'High'
            : actRoll < 0.65
              ? 'Moderate'
              : 'Low'
          : actRoll < 0.1
            ? 'High'
            : actRoll < 0.38
              ? 'Moderate'
              : 'Low';

    const famRoll = rng();
    const family_history: FamilyHistory =
      riskProfile === 0
        ? famRoll < 0.74
          ? 'None'
          : famRoll < 0.85
            ? 'Hypertension'
            : famRoll < 0.95
              ? 'Diabetes'
              : 'Heart Disease'
        : riskProfile === 1
          ? famRoll < 0.36
            ? 'None'
            : famRoll < 0.58
              ? 'Hypertension'
              : famRoll < 0.78
                ? 'Diabetes'
                : famRoll < 0.91
                  ? 'Heart Disease'
                  : 'Multiple'
          : famRoll < 0.14
            ? 'None'
            : famRoll < 0.34
              ? 'Hypertension'
              : famRoll < 0.54
                ? 'Diabetes'
                : famRoll < 0.74
                  ? 'Heart Disease'
                  : 'Multiple';

    const condRoll = rng();
    const existing_conditions: ExistingConditions =
      riskProfile === 0
        ? condRoll < 0.86
          ? 'None'
          : condRoll < 0.95
            ? 'Asthma'
            : 'Hypertension'
        : riskProfile === 1
          ? condRoll < 0.38
            ? 'None'
            : condRoll < 0.62
              ? 'Hypertension'
              : condRoll < 0.8
                ? 'Diabetes'
                : condRoll < 0.92
                  ? 'Asthma'
                  : 'Heart Disease'
          : condRoll < 0.1
            ? 'None'
            : condRoll < 0.32
              ? 'Hypertension'
              : condRoll < 0.54
                ? 'Diabetes'
                : condRoll < 0.74
                  ? 'Heart Disease'
                  : 'Multiple';

    const noise = randNormal(rng, 0, 0.85);
    const risk_category = assignSyntheticRiskLabel(
      {
        age,
        bmi,
        systolic_bp,
        diastolic_bp,
        glucose_level,
        cholesterol_level,
        heart_rate,
        smoking_status,
        physical_activity,
        family_history,
        existing_conditions,
      },
      noise
    );

    cleanRecords.push({
      patient_id: patientId,
      age,
      gender,
      height_cm,
      weight_kg,
      bmi,
      systolic_bp,
      diastolic_bp,
      glucose_level,
      cholesterol_level,
      heart_rate,
      smoking_status,
      physical_activity,
      family_history,
      existing_conditions,
      risk_category,
      record_source: 'synthetic_demo',
    });
  }

  // Create raw dataset rows with controlled missing values, mild outliers, and 4 duplicates
  // so the Preprocessing module demonstrates real detection and cleaning on actual data.
  const rawRows: RawDatasetRow[] = cleanRecords.map((rec) => ({
    patient_id: rec.patient_id,
    age: rec.age,
    gender: rec.gender,
    height_cm: rec.height_cm,
    weight_kg: rec.weight_kg,
    bmi: rec.bmi,
    systolic_bp: rec.systolic_bp,
    diastolic_bp: rec.diastolic_bp,
    glucose_level: rec.glucose_level,
    cholesterol_level: rec.cholesterol_level,
    heart_rate: rec.heart_rate,
    smoking_status: rec.smoking_status,
    physical_activity: rec.physical_activity,
    family_history: rec.family_history,
    existing_conditions: rec.existing_conditions,
    risk_category: rec.risk_category,
  }));

  const missingInjections: Array<{ row: number; col: string }> = [
    { row: 11, col: 'glucose_level' },
    { row: 27, col: 'cholesterol_level' },
    { row: 44, col: 'bmi' },
    { row: 63, col: 'heart_rate' },
    { row: 89, col: 'smoking_status' },
    { row: 114, col: 'systolic_bp' },
    { row: 142, col: 'glucose_level' },
    { row: 178, col: 'physical_activity' },
    { row: 205, col: 'cholesterol_level' },
    { row: 241, col: 'diastolic_bp' },
    { row: 276, col: 'family_history' },
    { row: 299, col: 'heart_rate' },
  ];

  for (const item of missingInjections) {
    if (rawRows[item.row]) {
      rawRows[item.row][item.col] = '';
    }
  }

  // Inject 4 duplicate records at the end of the raw dataset for duplicate removal demo
  const duplicateIndices = [5, 32, 108, 215];
  for (const idx of duplicateIndices) {
    if (rawRows[idx]) {
      rawRows.push({ ...rawRows[idx] });
    }
  }

  return { rawRows, cleanRecords };
}

export function sanitizeCsvCell(val: unknown): string {
  if (val === null || val === undefined) return '';
  let str = String(val);
  // Prevent CSV formula injection
  if (/^[=+\-@\t\r]/.test(str)) {
    str = `'${str}`;
  }
  if (/[",\n\r]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function rowsToCsv(rows: RawDatasetRow[], columns: readonly string[] = DATASET_COLUMNS): string {
  const header = columns.join(',');
  const lines = rows.map((row) =>
    columns.map((col) => sanitizeCsvCell(row[col])).join(',')
  );
  return [header, ...lines].join('\n');
}

function detectCsvDelimiter(headerLine: string): string {
  const candidates = [',', ';', '\t'];
  let bestDelim = ',';
  let bestCount = -1;

  for (const delim of candidates) {
    let count = 0;
    let inQuotes = false;
    for (let i = 0; i < headerLine.length; i++) {
      const ch = headerLine[i];
      if (ch === '"') inQuotes = !inQuotes;
      else if (ch === delim && !inQuotes) count++;
    }
    if (count > bestCount) {
      bestCount = count;
      bestDelim = delim;
    }
  }
  return bestCount > 0 ? bestDelim : ',';
}

export function parseCsvText(csvText: string): {
  headers: string[];
  rows: RawDatasetRow[];
} {
  // Strip UTF-8 BOM if present (common in Windows Excel CSV exports)
  const cleanedText = csvText.replace(/^\uFEFF/, '');
  const firstLineBreak = cleanedText.search(/\r?\n/);
  const firstLine = firstLineBreak >= 0 ? cleanedText.slice(0, firstLineBreak) : cleanedText;
  const delimiter = detectCsvDelimiter(firstLine);

  const records: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < cleanedText.length; i++) {
    const ch = cleanedText[i];
    if (ch === '"') {
      if (inQuotes && cleanedText[i + 1] === '"') {
        cell += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === delimiter && !inQuotes) {
      row.push(cell.trim());
      cell = '';
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && cleanedText[i + 1] === '\n') i++;
      row.push(cell.trim());
      cell = '';
      if (row.some((v) => v.length > 0)) {
        records.push(row);
      }
      row = [];
    } else {
      cell += ch;
    }
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell.trim());
    if (row.some((v) => v.length > 0)) {
      records.push(row);
    }
  }

  if (records.length < 2) {
    return { headers: [], rows: [] };
  }

  const headers = records[0].map((h) => h.replace(/^\uFEFF/, '').trim());
  const rows: RawDatasetRow[] = [];

  for (let r = 1; r < records.length; r++) {
    const vals = records[r];
    const obj: RawDatasetRow = {};
    for (let c = 0; c < headers.length; c++) {
      const key = headers[c];
      if (!key) continue;
      obj[key] = vals[c] ?? '';
    }
    rows.push(obj);
  }

  return { headers, rows };
}

function canonicalHeaderKey(raw: string): string {
  return raw
    .replace(/^\uFEFF/, '')
    .toLowerCase()
    .replace(/\([^)]*\)/g, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

const COLUMN_ALIASES: Record<string, string[]> = {
  patient_id: ['patient_id', 'patientid', 'id', 'record_id', 'patient_no', 'patient_code', 'pid'],
  age: ['age', 'patient_age', 'age_years', 'age_yrs'],
  gender: ['gender', 'sex', 'patient_gender', 'patient_sex'],
  height_cm: ['height_cm', 'height', 'height_in_cm', 'stature'],
  weight_kg: ['weight_kg', 'weight', 'weight_in_kg', 'body_weight'],
  bmi: ['bmi', 'body_mass_index', 'bmi_value'],
  systolic_bp: [
    'systolic_bp',
    'systolic',
    'systolic_blood_pressure',
    'sbp',
    'ap_hi',
    'trestbps',
    'sys_bp',
  ],
  diastolic_bp: [
    'diastolic_bp',
    'diastolic',
    'diastolic_blood_pressure',
    'dbp',
    'ap_lo',
    'dia_bp',
  ],
  blood_pressure_combined: ['blood_pressure', 'bp', 'bloodpressure', 'resting_bp', 'restingbp'],
  glucose_level: [
    'glucose_level',
    'glucose',
    'blood_glucose',
    'blood_glucose_level',
    'fasting_glucose',
    'avg_glucose_level',
    'fastingbs',
    'sugar',
    'blood_sugar',
  ],
  cholesterol_level: [
    'cholesterol_level',
    'cholesterol',
    'total_cholesterol',
    'chol',
    'serum_cholesterol',
  ],
  heart_rate: [
    'heart_rate',
    'heartrate',
    'hr',
    'pulse',
    'pulse_rate',
    'resting_heart_rate',
    'maxhr',
    'thalach',
  ],
  smoking_status: ['smoking_status', 'smoking', 'smoker', 'smoke', 'tobacco_use'],
  physical_activity: [
    'physical_activity',
    'physical_activity_level',
    'activity',
    'activity_level',
    'exercise',
    'exercise_level',
  ],
  family_history: [
    'family_history',
    'family_medical_history',
    'fam_history',
    'hereditary_history',
  ],
  existing_conditions: [
    'existing_conditions',
    'existing_medical_conditions',
    'medical_conditions',
    'conditions',
    'comorbidities',
    'chronic_conditions',
  ],
  risk_category: [
    'risk_category',
    'risk_level',
    'risk',
    'health_risk',
    'risk_tier',
    'risk_class',
    'target',
    'outcome',
    'label',
    'class',
    'heartdisease',
    'heart_disease',
    'stroke',
    'diabetes',
    'diagnosis',
    'status',
  ],
};

function parseLooseNum(val: unknown): number | null {
  if (val === null || val === undefined) return null;
  const s = String(val).trim();
  if (!s) return null;
  const cleaned = s.replace(/[^0-9.-]/g, '');
  if (!cleaned) return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function normalizeGenderValue(val: unknown): string {
  const s = String(val ?? '').trim().toLowerCase();
  if (!s) return '';
  if (s === 'm' || s === 'male' || s === '1' || s === 'man') return 'Male';
  if (s === 'f' || s === 'female' || s === '0' || s === 'woman') return 'Female';
  if (s === 'other' || s === 'o' || s === 'non-binary' || s === '2') return 'Other';
  return '';
}

function normalizeSmokingValue(val: unknown): string {
  const s = String(val ?? '').trim().toLowerCase();
  if (!s) return '';
  if (s.includes('current') || s === 'yes' || s === '1' || s === 'true' || s === 'smokes' || s === 'smoker') {
    return 'Current';
  }
  if (s.includes('former') || s.includes('past') || s.includes('ex') || s.includes('quit')) {
    return 'Former';
  }
  if (s.includes('never') || s === 'no' || s === '0' || s === 'false' || s.includes('non')) {
    return 'Never';
  }
  return '';
}

function normalizeActivityValue(val: unknown): string {
  const s = String(val ?? '').trim().toLowerCase();
  if (!s) return '';
  if (s.includes('high') || s.includes('active') || s.includes('vigorous') || s === '3') return 'High';
  if (s.includes('mod') || s.includes('med') || s.includes('light') || s === '2' || s === '1') {
    return 'Moderate';
  }
  if (s.includes('low') || s.includes('sedent') || s.includes('inact') || s === 'no' || s === '0') {
    return 'Low';
  }
  return '';
}

function normalizeFamilyHistoryValue(val: unknown): string {
  const s = String(val ?? '').trim().toLowerCase();
  if (!s) return '';
  if (s === 'none' || s === 'no' || s === '0' || s === 'false' || s === 'n/a') return 'None';
  if (s.includes('multi') || s.includes(',') || s.includes(';')) return 'Multiple';
  if (s.includes('heart') || s.includes('cardio') || s.includes('cad')) return 'Heart Disease';
  if (s.includes('diab')) return 'Diabetes';
  if (s.includes('hyper') || s.includes('bp') || s.includes('pressure')) return 'Hypertension';
  if (s === 'yes' || s === '1' || s === 'true') return 'Heart Disease';
  return 'None';
}

function normalizeConditionsValue(val: unknown): string {
  const s = String(val ?? '').trim().toLowerCase();
  if (!s) return '';
  if (s === 'none' || s === 'no' || s === '0' || s === 'false' || s === 'healthy' || s === 'n/a') {
    return 'None';
  }
  if (s.includes('multi') || s.includes(',') || s.includes(';')) return 'Multiple';
  if (s.includes('heart') || s.includes('cardio') || s.includes('cad')) return 'Heart Disease';
  if (s.includes('diab')) return 'Diabetes';
  if (s.includes('hyper') || s.includes('bp') || s.includes('pressure')) return 'Hypertension';
  if (s.includes('asthma') || s.includes('copd') || s.includes('resp')) return 'Asthma';
  if (s === 'yes' || s === '1' || s === 'true') return 'Hypertension';
  return 'None';
}

export function normalizeUploadedCsvRows(
  headers: string[],
  rawRows: RawDatasetRow[]
): {
  normalizedRows: RawDatasetRow[];
  mappedColumns: Record<string, string>;
  derivedRiskLabels: boolean;
  warnings: string[];
} {
  const headerLookup = new Map<string, string>();
  for (const h of headers) {
    headerLookup.set(canonicalHeaderKey(h), h);
  }

  const mappedColumns: Record<string, string> = {};
  for (const [canonical, aliases] of Object.entries(COLUMN_ALIASES)) {
    for (const alias of aliases) {
      const found = headerLookup.get(alias);
      if (found) {
        mappedColumns[canonical] = found;
        break;
      }
    }
  }

  // Must recognize at least 2 clinical numeric/demographic fields to confirm it's a healthcare CSV
  const recognizedCore = [
    'age',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'blood_pressure_combined',
    'glucose_level',
    'cholesterol_level',
    'heart_rate',
  ].filter((k) => Boolean(mappedColumns[k]));

  if (recognizedCore.length < 2) {
    throw new Error(
      `Could not recognize standard healthcare columns in this CSV. Detected headers: ${headers
        .slice(0, 10)
        .join(', ')}. Include columns such as age, gender, systolic_bp, diastolic_bp, glucose_level, cholesterol_level, heart_rate, and risk_category.`
    );
  }

  // Detect whether risk_category column uses binary labels (e.g. 0/1 or Yes/No)
  const riskCol = mappedColumns.risk_category;
  const distinctRiskTokens = new Set<string>();
  if (riskCol) {
    for (const r of rawRows) {
      const tok = String(r[riskCol] ?? '').trim().toLowerCase();
      if (tok) distinctRiskTokens.add(tok);
    }
  }
  const isBinaryTarget =
    distinctRiskTokens.size === 2 &&
    [...distinctRiskTokens].every((t) =>
      ['0', '1', 'no', 'yes', 'false', 'true', 'negative', 'positive', 'low', 'high', 'normal', 'abnormal'].includes(t)
    );

  const mapRiskValue = (val: unknown): RiskCategory | null => {
    const s = String(val ?? '').trim().toLowerCase();
    if (!s) return null;
    if (isBinaryTarget) {
      if (['1', 'yes', 'true', 'positive', 'high', 'abnormal'].includes(s)) return 'High Risk';
      if (['0', 'no', 'false', 'negative', 'low', 'normal'].includes(s)) return 'Low Risk';
    }
    if (['low', 'low risk', 'low_risk', '0', 'normal', 'healthy', 'no', 'negative', 'false'].includes(s)) {
      return 'Low Risk';
    }
    if (
      ['moderate', 'moderate risk', 'moderate_risk', 'medium', 'medium risk', 'mid', 'borderline', 'elevated', '1'].includes(s)
    ) {
      return 'Moderate Risk';
    }
    if (
      ['high', 'high risk', 'high_risk', 'severe', 'critical', 'very high', '2', '3', 'yes', 'positive', 'true'].includes(s)
    ) {
      return 'High Risk';
    }
    return null;
  };

  // Detect if height is in meters (median <= 3) so we convert to cm automatically
  let heightInMeters = false;
  if (mappedColumns.height_cm) {
    const hVals = rawRows
      .map((r) => parseLooseNum(r[mappedColumns.height_cm]))
      .filter((v): v is number => v !== null && v > 0);
    if (hVals.length > 0) {
      const avgH = hVals.reduce((a, b) => a + b, 0) / hVals.length;
      if (avgH > 0.5 && avgH <= 2.6) heightInMeters = true;
    }
  }

  let derivedRiskLabels = false;
  let validTargetCount = 0;
  const warnings: string[] = [];

  const normalizedRows: RawDatasetRow[] = rawRows.map((row, idx) => {
    const getRaw = (key: string) => (mappedColumns[key] ? row[mappedColumns[key]] : undefined);

    const pid = mappedColumns.patient_id
      ? String(getRaw('patient_id') ?? '').trim() || `CSV-${String(idx + 1).padStart(4, '0')}`
      : `CSV-${String(idx + 1).padStart(4, '0')}`;

    const age = parseLooseNum(getRaw('age'));
    const gender = normalizeGenderValue(getRaw('gender'));

    let height_cm = parseLooseNum(getRaw('height_cm'));
    if (height_cm !== null && heightInMeters) {
      height_cm = round1(height_cm * 100);
    }
    const weight_kg = parseLooseNum(getRaw('weight_kg'));
    let bmi = parseLooseNum(getRaw('bmi'));
    if (bmi === null && height_cm && height_cm > 0 && weight_kg && weight_kg > 0) {
      bmi = round1(weight_kg / Math.pow(height_cm / 100, 2));
    }

    let systolic_bp = parseLooseNum(getRaw('systolic_bp'));
    let diastolic_bp = parseLooseNum(getRaw('diastolic_bp'));

    if ((systolic_bp === null || diastolic_bp === null) && mappedColumns.blood_pressure_combined) {
      const bpStr = String(getRaw('blood_pressure_combined') ?? '').trim();
      const slashMatch = bpStr.match(/(\d+(?:\.\d+)?)\s*[/\\-]\s*(\d+(?:\.\d+)?)/);
      if (slashMatch) {
        if (systolic_bp === null) systolic_bp = Number(slashMatch[1]);
        if (diastolic_bp === null) diastolic_bp = Number(slashMatch[2]);
      } else {
        const singleBp = parseLooseNum(bpStr);
        if (singleBp !== null) {
          if (singleBp >= 100 && systolic_bp === null) {
            systolic_bp = singleBp;
          } else if (singleBp > 0 && singleBp < 100 && diastolic_bp === null) {
            diastolic_bp = singleBp;
          }
        }
      }
    }

    const glucose_level = parseLooseNum(getRaw('glucose_level'));
    const cholesterol_level = parseLooseNum(getRaw('cholesterol_level'));
    const heart_rate = parseLooseNum(getRaw('heart_rate'));

    const smoking_status = normalizeSmokingValue(getRaw('smoking_status'));
    const physical_activity = normalizeActivityValue(getRaw('physical_activity'));
    const family_history = normalizeFamilyHistoryValue(getRaw('family_history'));
    const existing_conditions = normalizeConditionsValue(getRaw('existing_conditions'));

    let risk_category: RiskCategory | null = riskCol ? mapRiskValue(getRaw('risk_category')) : null;
    if (risk_category) {
      validTargetCount++;
    } else if (!riskCol) {
      derivedRiskLabels = true;
      risk_category = assignSyntheticRiskLabel(
        {
          age: age ?? 45,
          bmi: bmi ?? 25,
          systolic_bp: systolic_bp ?? 120,
          diastolic_bp: diastolic_bp ?? 80,
          glucose_level: glucose_level ?? 100,
          cholesterol_level: cholesterol_level ?? 195,
          heart_rate: heart_rate ?? 74,
          smoking_status: (smoking_status || 'Never') as SmokingStatus,
          physical_activity: (physical_activity || 'Moderate') as PhysicalActivity,
          family_history: (family_history || 'None') as FamilyHistory,
          existing_conditions: (existing_conditions || 'None') as ExistingConditions,
        },
        0
      );
    }

    return {
      patient_id: pid,
      age: age ?? '',
      gender,
      height_cm: height_cm ?? '',
      weight_kg: weight_kg ?? '',
      bmi: bmi ?? '',
      systolic_bp: systolic_bp ?? '',
      diastolic_bp: diastolic_bp ?? '',
      glucose_level: glucose_level ?? '',
      cholesterol_level: cholesterol_level ?? '',
      heart_rate: heart_rate ?? '',
      smoking_status,
      physical_activity,
      family_history,
      existing_conditions,
      risk_category: risk_category ?? '',
    };
  });

  // If a target column existed but none of its values matched known risk tokens, derive cleanly
  if (riskCol && validTargetCount < 10) {
    derivedRiskLabels = true;
    warnings.push(
      `Target column "${riskCol}" contained unrecognized labels; mapped rows using multi-factor clinical risk scoring.`
    );
    for (const r of normalizedRows) {
      if (!r.risk_category) {
        r.risk_category = assignSyntheticRiskLabel(
          {
            age: parseLooseNum(r.age) ?? 45,
            bmi: parseLooseNum(r.bmi) ?? 25,
            systolic_bp: parseLooseNum(r.systolic_bp) ?? 120,
            diastolic_bp: parseLooseNum(r.diastolic_bp) ?? 80,
            glucose_level: parseLooseNum(r.glucose_level) ?? 100,
            cholesterol_level: parseLooseNum(r.cholesterol_level) ?? 195,
            heart_rate: parseLooseNum(r.heart_rate) ?? 74,
            smoking_status: (String(r.smoking_status || 'Never')) as SmokingStatus,
            physical_activity: (String(r.physical_activity || 'Moderate')) as PhysicalActivity,
            family_history: (String(r.family_history || 'None')) as FamilyHistory,
            existing_conditions: (String(r.existing_conditions || 'None')) as ExistingConditions,
          },
          0
        );
      }
    }
  } else if (derivedRiskLabels) {
    warnings.push(
      'No risk target column was found in the uploaded CSV; derived risk tiers from clinical parameters for modeling.'
    );
  }

  return { normalizedRows, mappedColumns, derivedRiskLabels, warnings };
}

export function ensureDatasetCsvFile(): {
  rawRows: RawDatasetRow[];
  cleanRecords: PatientRecord[];
  csvPath: string;
} {
  const datasetDir = path.resolve(process.cwd(), 'Healthcare_Risk_Prediction', 'dataset');
  const csvPath = path.join(datasetDir, 'healthcare_dataset.csv');
  const { rawRows, cleanRecords } = generateSyntheticDataset(320, 42);

  if (!fs.existsSync(datasetDir)) {
    fs.mkdirSync(datasetDir, { recursive: true });
  }
  if (!fs.existsSync(csvPath)) {
    fs.writeFileSync(csvPath, rowsToCsv(rawRows, DATASET_COLUMNS), 'utf-8');
  }

  return { rawRows, cleanRecords, csvPath };
}
