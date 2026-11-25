import type {
  ExistingConditions,
  FamilyHistory,
  Gender,
  MissingColumnStat,
  OutlierColumnStat,
  PatientRecord,
  PhysicalActivity,
  PreprocessingConfig,
  PreprocessingSummary,
  RawDatasetRow,
  RiskCategory,
  SmokingStatus,
} from '../src/types.ts';
import { createSeededRng } from './dataset.ts';

export const NUMERIC_FEATURES = [
  'age',
  'height_cm',
  'weight_kg',
  'bmi',
  'systolic_bp',
  'diastolic_bp',
  'glucose_level',
  'cholesterol_level',
  'heart_rate',
] as const;

export const CATEGORICAL_FEATURES = [
  'gender',
  'smoking_status',
  'physical_activity',
  'family_history',
  'existing_conditions',
] as const;

export const MODEL_FEATURE_KEYS = [
  'age',
  'gender',
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
] as const;

export const MODEL_FEATURE_LABELS: Record<string, string> = {
  age: 'Age (years)',
  gender: 'Gender',
  bmi: 'BMI (kg/m²)',
  systolic_bp: 'Systolic BP (mmHg)',
  diastolic_bp: 'Diastolic BP (mmHg)',
  glucose_level: 'Blood Glucose (mg/dL)',
  cholesterol_level: 'Total Cholesterol (mg/dL)',
  heart_rate: 'Heart Rate (bpm)',
  smoking_status: 'Smoking Status',
  physical_activity: 'Physical Activity',
  family_history: 'Family History',
  existing_conditions: 'Existing Conditions',
};

export const RISK_CLASSES: RiskCategory[] = [
  'Low Risk',
  'Moderate Risk',
  'High Risk',
];

export const ENCODING_MAPS: Record<string, Record<string, number>> = {
  gender: { Female: 0, Male: 1, Other: 2 },
  smoking_status: { Never: 0, Former: 1, Current: 2 },
  physical_activity: { High: 0, Moderate: 1, Low: 2 },
  family_history: {
    None: 0,
    Hypertension: 1,
    Diabetes: 2,
    'Heart Disease': 3,
    Multiple: 4,
  },
  existing_conditions: {
    None: 0,
    Asthma: 1,
    Hypertension: 2,
    Diabetes: 3,
    'Heart Disease': 4,
    Multiple: 5,
  },
  risk_category: {
    'Low Risk': 0,
    'Moderate Risk': 1,
    'High Risk': 2,
  },
};

export const DEFAULT_PREPROCESSING_CONFIG: PreprocessingConfig = {
  numericImputation: 'median',
  categoricalImputation: 'mode',
  removeDuplicates: true,
  outlierStrategy: 'clip',
  scalingMethod: 'standard',
  testSize: 0.2,
  randomSeed: 42,
};

function isMissing(val: unknown): boolean {
  if (val === null || val === undefined) return true;
  if (typeof val === 'number') return Number.isNaN(val);
  const str = String(val).trim();
  return str === '' || str.toLowerCase() === 'nan' || str.toLowerCase() === 'null';
}

function toNumber(val: unknown): number | null {
  if (isMissing(val)) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function calcMean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function calcMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function calcQuantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function calcMode(values: string[], fallback: string): string {
  if (values.length === 0) return fallback;
  const freq = new Map<string, number>();
  for (const v of values) {
    freq.set(v, (freq.get(v) ?? 0) + 1);
  }
  let best = fallback;
  let bestCount = -1;
  for (const [val, cnt] of freq.entries()) {
    if (cnt > bestCount) {
      bestCount = cnt;
      best = val;
    }
  }
  return best;
}

function normalizeRiskLabel(val: unknown): RiskCategory | null {
  if (isMissing(val)) return null;
  const s = String(val).trim().toLowerCase();
  if (s === 'low' || s === 'low risk' || s === '0') return 'Low Risk';
  if (s === 'moderate' || s === 'moderate risk' || s === 'medium' || s === '1') {
    return 'Moderate Risk';
  }
  if (s === 'high' || s === 'high risk' || s === '2') return 'High Risk';
  return null;
}

function normalizeCategory<T extends string>(
  val: unknown,
  allowed: readonly T[],
  fallback: T
): T {
  if (isMissing(val)) return fallback;
  const lower = String(val).trim().toLowerCase();
  for (const item of allowed) {
    if (item.toLowerCase() === lower) return item;
  }
  return fallback;
}

export interface FittedPreprocessor {
  config: PreprocessingConfig;
  numericFillValues: Record<string, number>;
  categoricalFillValues: Record<string, string>;
  outlierBounds: Record<string, { lower: number; upper: number }>;
  scalerParams: Record<string, { mean: number; std: number; min: number; max: number }>;
  featureKeys: readonly string[];
}

export function stratifiedSplit<T>(
  items: T[],
  labels: number[],
  testSize: number,
  seed: number
): { trainItems: T[]; testItems: T[]; trainLabels: number[]; testLabels: number[] } {
  const rng = createSeededRng(seed);
  const byClass = new Map<number, number[]>();

  for (let i = 0; i < labels.length; i++) {
    const cls = labels[i];
    if (!byClass.has(cls)) byClass.set(cls, []);
    byClass.get(cls)!.push(i);
  }

  const trainIndices: number[] = [];
  const testIndices: number[] = [];

  for (const [, indices] of byClass.entries()) {
    // Fisher-Yates shuffle deterministically
    const arr = [...indices];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    const nTest =
      arr.length <= 1
        ? 0
        : Math.max(1, Math.min(arr.length - 1, Math.round(arr.length * testSize)));
    for (let i = 0; i < arr.length; i++) {
      if (i < nTest) testIndices.push(arr[i]);
      else trainIndices.push(arr[i]);
    }
  }

  if (testIndices.length === 0 && trainIndices.length > 1) {
    testIndices.push(trainIndices.pop()!);
  }

  // Shuffle combined train and test indices
  const shuffleArr = (a: number[]) => {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  };
  shuffleArr(trainIndices);
  shuffleArr(testIndices);

  return {
    trainItems: trainIndices.map((idx) => items[idx]),
    testItems: testIndices.map((idx) => items[idx]),
    trainLabels: trainIndices.map((idx) => labels[idx]),
    testLabels: testIndices.map((idx) => labels[idx]),
  };
}

/**
 * Fits imputer, IQR outlier bounds, and feature scaler strictly on the training rows
 * to guarantee zero data leakage between train and test splits.
 */
export function fitPreprocessorOnRows(
  trainRows: RawDatasetRow[],
  config: PreprocessingConfig
): FittedPreprocessor {
  const numericFillValues: Record<string, number> = {};
  const outlierBounds: Record<string, { lower: number; upper: number }> = {};

  const numericDefaults: Record<string, number> = {
    age: 45,
    height_cm: 168,
    weight_kg: 72,
    bmi: 25.5,
    systolic_bp: 122,
    diastolic_bp: 80,
    glucose_level: 102,
    cholesterol_level: 198,
    heart_rate: 75,
  };

  for (const col of NUMERIC_FEATURES) {
    const vals = trainRows
      .map((r) => toNumber(r[col]))
      .filter((v): v is number => v !== null);
    const fill =
      vals.length === 0
        ? numericDefaults[col] ?? 0
        : config.numericImputation === 'mean'
          ? calcMean(vals)
          : calcMedian(vals);
    numericFillValues[col] = Math.round(fill * 100) / 100;

    if (vals.length === 0) {
      outlierBounds[col] = { lower: fill, upper: fill };
    } else {
      const sorted = [...vals].sort((a, b) => a - b);
      const q1 = calcQuantile(sorted, 0.25);
      const q3 = calcQuantile(sorted, 0.75);
      const iqr = q3 - q1;
      outlierBounds[col] = {
        lower: Math.round((q1 - 1.5 * iqr) * 100) / 100,
        upper: Math.round((q3 + 1.5 * iqr) * 100) / 100,
      };
    }
  }

  const categoricalDefaults: Record<string, string> = {
    gender: 'Female',
    smoking_status: 'Never',
    physical_activity: 'Moderate',
    family_history: 'None',
    existing_conditions: 'None',
  };

  const categoricalFillValues: Record<string, string> = {};
  for (const col of CATEGORICAL_FEATURES) {
    const vals = trainRows
      .map((r) => (isMissing(r[col]) ? null : String(r[col]).trim()))
      .filter((v): v is string => v !== null);
    categoricalFillValues[col] = calcMode(vals, categoricalDefaults[col]);
  }

  // Transform training rows to encoded numeric matrix (unscaled) to compute scaler params
  const unscaledVectors = trainRows.map((row) =>
    extractEncodedFeatureVector(row, numericFillValues, categoricalFillValues, outlierBounds, config)
  );

  const scalerParams: Record<
    string,
    { mean: number; std: number; min: number; max: number }
  > = {};

  for (let f = 0; f < MODEL_FEATURE_KEYS.length; f++) {
    const key = MODEL_FEATURE_KEYS[f];
    const colVals = unscaledVectors.map((vec) => vec[f]);
    const mean = calcMean(colVals);
    const variance =
      colVals.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) /
      Math.max(1, colVals.length);
    const std = Math.sqrt(variance) || 1;
    let min = colVals.length > 0 ? colVals[0] : 0;
    let max = colVals.length > 0 ? colVals[0] : 0;
    for (let i = 1; i < colVals.length; i++) {
      if (colVals[i] < min) min = colVals[i];
      if (colVals[i] > max) max = colVals[i];
    }

    scalerParams[key] = {
      mean: Math.round(mean * 1000) / 1000,
      std: Math.round(std * 1000) / 1000,
      min: Math.round(min * 1000) / 1000,
      max: Math.round(max * 1000) / 1000,
    };
  }

  return {
    config,
    numericFillValues,
    categoricalFillValues,
    outlierBounds,
    scalerParams,
    featureKeys: MODEL_FEATURE_KEYS,
  };
}

function extractEncodedFeatureVector(
  row: RawDatasetRow,
  numericFillValues: Record<string, number>,
  categoricalFillValues: Record<string, string>,
  outlierBounds: Record<string, { lower: number; upper: number }>,
  config: PreprocessingConfig
): number[] {
  const getNum = (col: string): number => {
    let val = toNumber(row[col]);
    if (val === null) {
      if (col === 'bmi') {
        const h = toNumber(row.height_cm) ?? numericFillValues.height_cm;
        const w = toNumber(row.weight_kg) ?? numericFillValues.weight_kg;
        val = h > 0 ? Math.round((w / Math.pow(h / 100, 2)) * 10) / 10 : numericFillValues.bmi;
      } else {
        val = numericFillValues[col] ?? 0;
      }
    }
    if (config.outlierStrategy === 'clip' && outlierBounds[col]) {
      const { lower, upper } = outlierBounds[col];
      val = Math.max(lower, Math.min(upper, val));
    }
    return val;
  };

  const getCatCode = (col: string): number => {
    const map = ENCODING_MAPS[col];
    const raw = isMissing(row[col]) ? categoricalFillValues[col] : String(row[col]).trim();
    if (map[raw] !== undefined) return map[raw];
    // Case-insensitive lookup
    const lower = raw.toLowerCase();
    for (const [k, code] of Object.entries(map)) {
      if (k.toLowerCase() === lower) return code;
    }
    const fallback = categoricalFillValues[col];
    return map[fallback] ?? 0;
  };

  return [
    getNum('age'),
    getCatCode('gender'),
    getNum('bmi'),
    getNum('systolic_bp'),
    getNum('diastolic_bp'),
    getNum('glucose_level'),
    getNum('cholesterol_level'),
    getNum('heart_rate'),
    getCatCode('smoking_status'),
    getCatCode('physical_activity'),
    getCatCode('family_history'),
    getCatCode('existing_conditions'),
  ];
}

export function transformRowToScaledVector(
  row: RawDatasetRow,
  fitted: FittedPreprocessor
): number[] {
  const rawVec = extractEncodedFeatureVector(
    row,
    fitted.numericFillValues,
    fitted.categoricalFillValues,
    fitted.outlierBounds,
    fitted.config
  );

  return rawVec.map((val, idx) => {
    const key = MODEL_FEATURE_KEYS[idx];
    const stats = fitted.scalerParams[key];
    if (!stats) return val;
    if (fitted.config.scalingMethod === 'minmax') {
      const span = stats.max - stats.min;
      return span > 1e-9 ? (val - stats.min) / span : 0;
    }
    return stats.std > 1e-9 ? (val - stats.mean) / stats.std : 0;
  });
}

export function deduplicateRows(rows: RawDatasetRow[]): {
  uniqueRows: RawDatasetRow[];
  duplicatesRemoved: number;
} {
  const seenIds = new Set<string>();
  const seenSignatures = new Set<string>();
  const uniqueRows: RawDatasetRow[] = [];
  let duplicatesRemoved = 0;

  for (const row of rows) {
    const pid = !isMissing(row.patient_id) ? String(row.patient_id).trim() : '';
    const sig = JSON.stringify(row);
    if ((pid && seenIds.has(pid)) || seenSignatures.has(sig)) {
      duplicatesRemoved++;
      continue;
    }
    if (pid) seenIds.add(pid);
    seenSignatures.add(sig);
    uniqueRows.push(row);
  }

  return { uniqueRows, duplicatesRemoved };
}

export function runFullPreprocessingAudit(
  rawRows: RawDatasetRow[],
  datasetName: string,
  isSynthetic: boolean,
  config: PreprocessingConfig
): {
  summary: PreprocessingSummary;
  cleanedRecords: PatientRecord[];
  dedupedRawRows: RawDatasetRow[];
  validLabels: number[];
} {
  const rawRowCount = rawRows.length;

  // Step 1: Deduplicate
  let workingRows = rawRows;
  let duplicateRowsRemoved = 0;
  if (config.removeDuplicates) {
    const dedup = deduplicateRows(rawRows);
    workingRows = dedup.uniqueRows;
    duplicateRowsRemoved = dedup.duplicatesRemoved;
  }

  // Keep only rows with valid target label
  const labeledRows: RawDatasetRow[] = [];
  const validLabels: number[] = [];
  for (const r of workingRows) {
    const label = normalizeRiskLabel(r.risk_category);
    if (label) {
      labeledRows.push({ ...r, risk_category: label });
      validLabels.push(ENCODING_MAPS.risk_category[label]);
    }
  }

  // Step 2: Audit missing values across all features
  const allCheckedCols = [...NUMERIC_FEATURES, ...CATEGORICAL_FEATURES];
  const missingByColumn: MissingColumnStat[] = [];
  let totalMissingValues = 0;

  const globalNumericFills: Record<string, number> = {};
  const globalCatFills: Record<string, string> = {};
  const numDefaults: Record<string, number> = {
    age: 45,
    height_cm: 168,
    weight_kg: 72,
    bmi: 25.5,
    systolic_bp: 122,
    diastolic_bp: 80,
    glucose_level: 102,
    cholesterol_level: 198,
    heart_rate: 75,
  };
  const catDefaults: Record<string, string> = {
    gender: 'Female',
    smoking_status: 'Never',
    physical_activity: 'Moderate',
    family_history: 'None',
    existing_conditions: 'None',
  };

  for (const col of allCheckedCols) {
    let miss = 0;
    for (const r of labeledRows) {
      if (isMissing(r[col])) miss++;
    }
    totalMissingValues += miss;

    const isNum = (NUMERIC_FEATURES as readonly string[]).includes(col);
    let fillLabel = '';
    if (isNum) {
      const vals = labeledRows
        .map((r) => toNumber(r[col]))
        .filter((v): v is number => v !== null);
      const fillVal =
        vals.length === 0
          ? numDefaults[col] ?? 0
          : config.numericImputation === 'mean'
            ? calcMean(vals)
            : calcMedian(vals);
      const rounded = Math.round(fillVal * 10) / 10;
      globalNumericFills[col] = rounded;
      fillLabel = vals.length === 0 ? `${rounded} (default)` : String(rounded);
    } else {
      const vals = labeledRows
        .map((r) => (isMissing(r[col]) ? null : String(r[col]).trim()))
        .filter((v): v is string => v !== null);
      const modeVal = calcMode(vals, catDefaults[col] ?? 'None');
      globalCatFills[col] = modeVal;
      fillLabel = modeVal;
    }

    missingByColumn.push({
      column: col,
      missingCount: miss,
      missingPercent:
        labeledRows.length > 0
          ? Math.round((miss / labeledRows.length) * 1000) / 10
          : 0,
      imputationApplied: isNum
        ? config.numericImputation === 'mean'
          ? 'Mean Imputation'
          : 'Median Imputation'
        : 'Mode Imputation',
      fillValue: fillLabel,
    });
  }

  // Step 3: Outlier detection (IQR 1.5x rule)
  const outliersByColumn: OutlierColumnStat[] = [];
  let totalOutliersDetected = 0;
  const globalOutlierBounds: Record<string, { lower: number; upper: number }> = {};

  for (const col of NUMERIC_FEATURES) {
    const vals = labeledRows
      .map((r) => toNumber(r[col]) ?? globalNumericFills[col])
      .sort((a, b) => a - b);
    const q1 = calcQuantile(vals, 0.25);
    const q3 = calcQuantile(vals, 0.75);
    const iqr = q3 - q1;
    const lowerBound = Math.round((q1 - 1.5 * iqr) * 10) / 10;
    const upperBound = Math.round((q3 + 1.5 * iqr) * 10) / 10;
    globalOutlierBounds[col] = { lower: lowerBound, upper: upperBound };

    let outlierCount = 0;
    for (const v of vals) {
      if (v < lowerBound || v > upperBound) outlierCount++;
    }
    totalOutliersDetected += outlierCount;

    outliersByColumn.push({
      column: col,
      q1: Math.round(q1 * 10) / 10,
      q3: Math.round(q3 * 10) / 10,
      iqr: Math.round(iqr * 10) / 10,
      lowerBound,
      upperBound,
      outlierCount,
    });
  }

  // Build cleaned dataset records for export & preview
  const cleanedRecords: PatientRecord[] = labeledRows.map((row, idx) => {
    const cleanNum = (col: string, roundInt = false): number => {
      let v = toNumber(row[col]);
      if (v === null) v = globalNumericFills[col];
      if (config.outlierStrategy === 'clip') {
        const b = globalOutlierBounds[col];
        v = Math.max(b.lower, Math.min(b.upper, v));
      }
      return roundInt ? Math.round(v) : Math.round(v * 10) / 10;
    };

    const height_cm = cleanNum('height_cm', false);
    const weight_kg = cleanNum('weight_kg', false);
    const rawBmi = toNumber(row.bmi);
    const bmi =
      rawBmi !== null
        ? cleanNum('bmi', false)
        : Math.round((weight_kg / Math.pow(height_cm / 100, 2)) * 10) / 10;

    return {
      patient_id: !isMissing(row.patient_id)
        ? String(row.patient_id).trim()
        : `REC-${String(idx + 1).padStart(4, '0')}`,
      age: cleanNum('age', true),
      gender: normalizeCategory<Gender>(
        isMissing(row.gender) ? globalCatFills.gender : row.gender,
        ['Male', 'Female', 'Other'],
        'Female'
      ),
      height_cm,
      weight_kg,
      bmi,
      systolic_bp: cleanNum('systolic_bp', true),
      diastolic_bp: cleanNum('diastolic_bp', true),
      glucose_level: cleanNum('glucose_level', true),
      cholesterol_level: cleanNum('cholesterol_level', true),
      heart_rate: cleanNum('heart_rate', true),
      smoking_status: normalizeCategory<SmokingStatus>(
        isMissing(row.smoking_status) ? globalCatFills.smoking_status : row.smoking_status,
        ['Never', 'Former', 'Current'],
        'Never'
      ),
      physical_activity: normalizeCategory<PhysicalActivity>(
        isMissing(row.physical_activity)
          ? globalCatFills.physical_activity
          : row.physical_activity,
        ['Low', 'Moderate', 'High'],
        'Moderate'
      ),
      family_history: normalizeCategory<FamilyHistory>(
        isMissing(row.family_history) ? globalCatFills.family_history : row.family_history,
        ['None', 'Heart Disease', 'Diabetes', 'Hypertension', 'Multiple'],
        'None'
      ),
      existing_conditions: normalizeCategory<ExistingConditions>(
        isMissing(row.existing_conditions)
          ? globalCatFills.existing_conditions
          : row.existing_conditions,
        ['None', 'Hypertension', 'Diabetes', 'Asthma', 'Heart Disease', 'Multiple'],
        'None'
      ),
      risk_category: row.risk_category as RiskCategory,
      record_source: isSynthetic ? 'synthetic_demo' : 'uploaded_csv',
    };
  });

  // Step 4: Leak-free stratified train-test split & fit scaler on train split only
  const split = stratifiedSplit(
    labeledRows,
    validLabels,
    config.testSize,
    config.randomSeed
  );
  const fittedOnTrain = fitPreprocessorOnRows(split.trainItems, config);

  const countClasses = (labels: number[]): Record<RiskCategory, number> => {
    const out: Record<RiskCategory, number> = {
      'Low Risk': 0,
      'Moderate Risk': 0,
      'High Risk': 0,
    };
    for (const l of labels) {
      out[RISK_CLASSES[l]]++;
    }
    return out;
  };

  const scalingStats = MODEL_FEATURE_KEYS.map((feature) => ({
    feature: MODEL_FEATURE_LABELS[feature] ?? feature,
    ...fittedOnTrain.scalerParams[feature],
  }));

  const operationsLog = [
    {
      step: '01',
      title: 'Duplicate Record Inspection',
      detail: config.removeDuplicates
        ? `Removed ${duplicateRowsRemoved} duplicate row(s) from ${rawRowCount} raw records, leaving ${labeledRows.length} unique labeled records.`
        : `Duplicate removal disabled; retained all ${labeledRows.length} labeled records.`,
      status: 'Completed',
    },
    {
      step: '02',
      title: 'Missing Value Imputation',
      detail: `Detected ${totalMissingValues} missing cell(s). Applied ${
        config.numericImputation === 'mean' ? 'Mean' : 'Median'
      } imputation to numeric vitals and Mode imputation to categorical fields (fitted on training fold only during modeling).`,
      status: 'Completed',
    },
    {
      step: '03',
      title: 'IQR Outlier Analysis',
      detail: `Evaluated 1.5x Interquartile Range bounds across ${NUMERIC_FEATURES.length} numeric variables, flagging ${totalOutliersDetected} extreme value(s). Strategy: ${
        config.outlierStrategy === 'clip'
          ? 'Winsorized (clipped to IQR bounds)'
          : 'Retained without clipping'
      }.`,
      status: 'Completed',
    },
    {
      step: '04',
      title: 'Categorical Label Encoding',
      detail: `Mapped 5 categorical clinical attributes and the 3-class target ('Low Risk'=0, 'Moderate Risk'=1, 'High Risk'=2) to deterministic numeric codes.`,
      status: 'Completed',
    },
    {
      step: '05',
      title:
        config.scalingMethod === 'standard'
          ? 'Feature Standardization (Z-Score)'
          : 'Data Normalization (Min-Max [0, 1])',
      detail: `Fitted ${
        config.scalingMethod === 'standard'
          ? 'StandardScaler (zero mean, unit variance)'
          : 'MinMaxScaler ([0, 1] range)'
      } strictly on the ${split.trainItems.length} training records and transformed both splits to prevent data leakage.`,
      status: 'Completed',
    },
    {
      step: '06',
      title: 'Stratified Train-Test Split',
      detail: `Partitioned ${labeledRows.length} records into ${split.trainItems.length} training (${Math.round(
        (1 - config.testSize) * 100
      )}%) and ${split.testItems.length} test (${Math.round(
        config.testSize * 100
      )}%) samples while preserving class proportions (seed=${config.randomSeed}).`,
      status: 'Completed',
    },
  ];

  const summary: PreprocessingSummary = {
    datasetName,
    isSynthetic,
    rawRowCount,
    cleanedRowCount: cleanedRecords.length,
    duplicateRowsRemoved,
    totalMissingValues,
    missingByColumn,
    outliersByColumn,
    totalOutliersDetected,
    encodingMaps: ENCODING_MAPS,
    scalingStats,
    trainCount: split.trainItems.length,
    testCount: split.testItems.length,
    trainClassDistribution: countClasses(split.trainLabels),
    testClassDistribution: countClasses(split.testLabels),
    previewCleaned: cleanedRecords.slice(0, 15),
    config,
    operationsLog,
  };

  return {
    summary,
    cleanedRecords,
    dedupedRawRows: labeledRows,
    validLabels,
  };
}
