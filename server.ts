import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { createServer as createViteServer } from 'vite';
import type {
  ExistingConditions,
  FamilyHistory,
  Gender,
  ModelKey,
  OverviewStats,
  PatientRecord,
  PhysicalActivity,
  PredictionInput,
  PreprocessingConfig,
  RawDatasetRow,
  SmokingStatus,
} from './src/types.ts';
import {
  DATASET_COLUMNS,
  ensureDatasetCsvFile,
  normalizeUploadedCsvRows,
  parseCsvText,
  rowsToCsv,
  SYNTHETIC_DATASET_NAME,
} from './server/dataset.ts';
import {
  deletePatientById,
  deleteSavedReport,
  findPatientById,
  getDb,
  insertPatient,
  listAllPatients,
  listSavedReports,
  savePredictionReport,
  seedInitialPatientsIfEmpty,
} from './server/db.ts';
import { computeEdaAnalytics } from './server/eda.ts';
import {
  predictPatientRisk,
  trainAllModels,
  type TrainedRuntimeState,
} from './server/ml.ts';
import {
  DEFAULT_PREPROCESSING_CONFIG,
  runFullPreprocessingAudit,
} from './server/preprocessing.ts';

const ALLOWED_GENDERS: readonly Gender[] = ['Male', 'Female', 'Other'];
const ALLOWED_SMOKING: readonly SmokingStatus[] = ['Never', 'Former', 'Current'];
const ALLOWED_ACTIVITY: readonly PhysicalActivity[] = ['Low', 'Moderate', 'High'];
const ALLOWED_FAMILY: readonly FamilyHistory[] = [
  'None',
  'Heart Disease',
  'Diabetes',
  'Hypertension',
  'Multiple',
];
const ALLOWED_CONDITIONS: readonly ExistingConditions[] = [
  'None',
  'Hypertension',
  'Diabetes',
  'Asthma',
  'Heart Disease',
  'Multiple',
];
const ALLOWED_MODELS: readonly ModelKey[] = [
  'logistic_regression',
  'decision_tree',
  'random_forest',
  'gradient_boosting',
];

const ALLOWED_PYTHON_FILES = [
  'app.py',
  'database.py',
  'preprocessing.py',
  'train_models.py',
  'prediction.py',
  'visualization.py',
  'report_generator.py',
  'requirements.txt',
  'README.md',
] as const;

interface ActiveDatasetState {
  name: string;
  isSynthetic: boolean;
  rawRows: RawDatasetRow[];
  config: PreprocessingConfig;
  cleanedRecords: PatientRecord[];
  preprocessingSummary: ReturnType<typeof runFullPreprocessingAudit>['summary'];
  trainedRuntime: TrainedRuntimeState;
}

function validateNumericRange(
  val: unknown,
  label: string,
  min: number,
  max: number
): { ok: true; value: number } | { ok: false; error: string } {
  const num = Number(val);
  if (!Number.isFinite(num)) {
    return { ok: false, error: `${label} must be a valid number.` };
  }
  if (num < min || num > max) {
    return { ok: false, error: `${label} must be between ${min} and ${max}.` };
  }
  return { ok: true, value: num };
}

function validatePatientPayload(body: Record<string, unknown>, requirePatientId: boolean) {
  const errors: Record<string, string> = {};

  let patientId = '';
  if (requirePatientId) {
    const rawId = typeof body.patient_id === 'string' ? body.patient_id.trim() : '';
    if (!rawId) {
      errors.patient_id = 'Patient ID is required.';
    } else if (!/^[A-Za-z0-9_-]{3,24}$/.test(rawId)) {
      errors.patient_id =
        'Patient ID must be 3–24 characters using letters, numbers, hyphens, or underscores.';
    } else {
      patientId = rawId;
    }
  } else if (typeof body.patient_id === 'string' && body.patient_id.trim()) {
    patientId = body.patient_id.trim().slice(0, 24);
  }

  const ageRes = validateNumericRange(body.age, 'Age', 1, 120);
  if (!ageRes.ok) errors.age = ageRes.error;

  const heightRes = validateNumericRange(body.height_cm, 'Height (cm)', 80, 250);
  if (!heightRes.ok) errors.height_cm = heightRes.error;

  const weightRes = validateNumericRange(body.weight_kg, 'Weight (kg)', 20, 300);
  if (!weightRes.ok) errors.weight_kg = weightRes.error;

  const sbpRes = validateNumericRange(body.systolic_bp, 'Systolic BP', 70, 260);
  if (!sbpRes.ok) errors.systolic_bp = sbpRes.error;

  const dbpRes = validateNumericRange(body.diastolic_bp, 'Diastolic BP', 40, 160);
  if (!dbpRes.ok) errors.diastolic_bp = dbpRes.error;

  if (sbpRes.ok && dbpRes.ok && sbpRes.value <= dbpRes.value) {
    errors.systolic_bp = 'Systolic BP must be greater than Diastolic BP.';
  }

  const gluRes = validateNumericRange(body.glucose_level, 'Glucose Level', 40, 600);
  if (!gluRes.ok) errors.glucose_level = gluRes.error;

  const cholRes = validateNumericRange(body.cholesterol_level, 'Cholesterol Level', 80, 600);
  if (!cholRes.ok) errors.cholesterol_level = cholRes.error;

  const hrRes = validateNumericRange(body.heart_rate, 'Heart Rate', 35, 220);
  if (!hrRes.ok) errors.heart_rate = hrRes.error;

  const gender = body.gender as Gender;
  if (!ALLOWED_GENDERS.includes(gender)) {
    errors.gender = 'Please select a valid gender option.';
  }

  const smoking_status = body.smoking_status as SmokingStatus;
  if (!ALLOWED_SMOKING.includes(smoking_status)) {
    errors.smoking_status = 'Please select a valid smoking status.';
  }

  const physical_activity = body.physical_activity as PhysicalActivity;
  if (!ALLOWED_ACTIVITY.includes(physical_activity)) {
    errors.physical_activity = 'Please select a valid physical activity level.';
  }

  const family_history = body.family_history as FamilyHistory;
  if (!ALLOWED_FAMILY.includes(family_history)) {
    errors.family_history = 'Please select a valid family history option.';
  }

  const existing_conditions = body.existing_conditions as ExistingConditions;
  if (!ALLOWED_CONDITIONS.includes(existing_conditions)) {
    errors.existing_conditions = 'Please select a valid existing medical condition option.';
  }

  if (Object.keys(errors).length > 0) {
    return { valid: false as const, errors };
  }

  const heightM = (heightRes as { ok: true; value: number }).value / 100;
  const weightKg = (weightRes as { ok: true; value: number }).value;
  const bmi = Math.round((weightKg / (heightM * heightM)) * 10) / 10;

  return {
    valid: true as const,
    data: {
      patient_id: patientId,
      age: Math.round((ageRes as { ok: true; value: number }).value),
      gender,
      height_cm: Math.round((heightRes as { ok: true; value: number }).value * 10) / 10,
      weight_kg: Math.round(weightKg * 10) / 10,
      bmi,
      systolic_bp: Math.round((sbpRes as { ok: true; value: number }).value),
      diastolic_bp: Math.round((dbpRes as { ok: true; value: number }).value),
      glucose_level: Math.round((gluRes as { ok: true; value: number }).value * 10) / 10,
      cholesterol_level: Math.round((cholRes as { ok: true; value: number }).value * 10) / 10,
      heart_rate: Math.round((hrRes as { ok: true; value: number }).value),
      smoking_status,
      physical_activity,
      family_history,
      existing_conditions,
    },
  };
}

function parsePreprocessingConfig(raw: Record<string, unknown> | undefined): PreprocessingConfig {
  const numericImputation = raw?.numericImputation === 'mean' ? 'mean' : 'median';
  const removeDuplicates = raw?.removeDuplicates !== false;
  const outlierStrategy = raw?.outlierStrategy === 'keep' ? 'keep' : 'clip';
  const scalingMethod = raw?.scalingMethod === 'minmax' ? 'minmax' : 'standard';
  const rawTestSize = Number(raw?.testSize ?? DEFAULT_PREPROCESSING_CONFIG.testSize);
  const testSize =
    Number.isFinite(rawTestSize) && rawTestSize >= 0.1 && rawTestSize <= 0.4
      ? Math.round(rawTestSize * 100) / 100
      : 0.2;
  const rawSeed = Number(raw?.randomSeed ?? DEFAULT_PREPROCESSING_CONFIG.randomSeed);
  const randomSeed =
    Number.isFinite(rawSeed) && rawSeed >= 1 && rawSeed <= 99999
      ? Math.round(rawSeed)
      : 42;

  return {
    numericImputation,
    categoricalImputation: 'mode',
    removeDuplicates,
    outlierStrategy,
    scalingMethod,
    testSize,
    randomSeed,
  };
}

function bootstrapDatasetState(): ActiveDatasetState {
  getDb();
  const { rawRows, cleanRecords } = ensureDatasetCsvFile();
  seedInitialPatientsIfEmpty(cleanRecords);

  const config = { ...DEFAULT_PREPROCESSING_CONFIG };
  const audit = runFullPreprocessingAudit(rawRows, SYNTHETIC_DATASET_NAME, true, config);
  const trainedRuntime = trainAllModels(rawRows, SYNTHETIC_DATASET_NAME, true, config);

  return {
    name: SYNTHETIC_DATASET_NAME,
    isSynthetic: true,
    rawRows,
    config,
    cleanedRecords: audit.cleanedRecords,
    preprocessingSummary: audit.summary,
    trainedRuntime,
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  let state!: ActiveDatasetState;
  let stateInitialized = false;

  const ensureState = (): ActiveDatasetState => {
    if (!stateInitialized) {
      state = bootstrapDatasetState();
      stateInitialized = true;
    }
    return state;
  };

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, initialized: stateInitialized });
  });

  // Ensure SQLite seed and trained ML models are ready before handling any API request
  app.use('/api', (_req, _res, next) => {
    try {
      ensureState();
      next();
    } catch (err) {
      next(err);
    }
  });

  // --------------------------------------------------------------------------
  // Overview KPI & Summary Endpoint
  // --------------------------------------------------------------------------
  app.get('/api/overview', (_req, res) => {
    const patients = listAllPatients();
    const reports = listSavedReports();
    const registeredPatients = patients.filter((p) => p.record_source === 'registered').length;
    const syntheticPatients = patients.filter((p) => p.record_source !== 'registered').length;

    const highRiskCount = patients.filter((p) => p.risk_category === 'High Risk').length;
    const moderateRiskCount = patients.filter((p) => p.risk_category === 'Moderate Risk').length;
    const lowRiskCount = patients.filter((p) => p.risk_category === 'Low Risk').length;

    const bundle = state.trainedRuntime.bundle;
    const bestEval = bundle.models[bundle.bestModelKey];

    const modelSummary = ALLOWED_MODELS.map((key) => {
      const m = bundle.models[key];
      return {
        key: m.key,
        name: m.name,
        shortName: m.shortName,
        accuracy: m.accuracy,
        f1Macro: m.f1Macro,
        macroAuc: m.macroAuc,
      };
    });

    const payload: OverviewStats = {
      totalPatients: patients.length,
      registeredPatients,
      syntheticPatients,
      highRiskCount,
      moderateRiskCount,
      lowRiskCount,
      savedReportsCount: reports.length,
      activeDatasetName: state.name,
      isSyntheticDataset: state.isSynthetic,
      bestModel: bestEval
        ? {
            key: bestEval.key,
            name: bestEval.name,
            accuracy: bestEval.accuracy,
            f1Macro: bestEval.f1Macro,
            macroAuc: bestEval.macroAuc,
          }
        : null,
      modelSummary,
      recentReports: reports.slice(0, 6),
    };

    res.json(payload);
  });

  // --------------------------------------------------------------------------
  // Patient Registration & SQLite Records
  // --------------------------------------------------------------------------
  app.get('/api/patients', (_req, res) => {
    res.json({ patients: listAllPatients() });
  });

  app.post('/api/patients', (req, res) => {
    const validation = validatePatientPayload(req.body ?? {}, true);
    if (!validation.valid) {
      res.status(400).json({ error: 'Validation failed.', errors: validation.errors });
      return;
    }

    const existing = findPatientById(validation.data.patient_id);
    if (existing) {
      res.status(409).json({
        error: `Patient ID "${validation.data.patient_id}" already exists in SQLite.`,
        errors: { patient_id: 'This Patient ID is already registered.' },
      });
      return;
    }

    // Use the best trained model to assign an initial risk tier estimate for the new record
    const bestKey = state.trainedRuntime.bundle.bestModelKey;
    const prediction = predictPatientRisk(state.trainedRuntime, {
      ...validation.data,
      model_key: bestKey,
    });

    try {
      const created = insertPatient({
        ...validation.data,
        risk_category: prediction.predicted_risk,
        record_source: 'registered',
      });
      res.status(201).json({
        patient: created,
        initialPrediction: {
          model_name: prediction.model_name,
          predicted_risk: prediction.predicted_risk,
          probabilities: prediction.probabilities,
        },
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to save patient record.',
      });
    }
  });

  app.delete('/api/patients/:patientId', (req, res) => {
    const patientId = String(req.params.patientId ?? '').trim();
    if (!/^[A-Za-z0-9_-]{3,24}$/.test(patientId)) {
      res.status(400).json({ error: 'Invalid Patient ID format.' });
      return;
    }
    const deleted = deletePatientById(patientId);
    if (!deleted) {
      res.status(404).json({ error: 'Patient record not found.' });
      return;
    }
    res.json({ deleted: true, patient_id: patientId });
  });

  // --------------------------------------------------------------------------
  // Data Preprocessing & CSV Upload / Download
  // --------------------------------------------------------------------------
  app.get('/api/preprocessing', (_req, res) => {
    res.json(state.preprocessingSummary);
  });

  app.post('/api/preprocessing/run', (req, res) => {
    try {
      const config = parsePreprocessingConfig(req.body);
      const audit = runFullPreprocessingAudit(
        state.rawRows,
        state.name,
        state.isSynthetic,
        config
      );
      const trainedRuntime = trainAllModels(
        state.rawRows,
        state.name,
        state.isSynthetic,
        config
      );
      state = {
        ...state,
        config,
        cleanedRecords: audit.cleanedRecords,
        preprocessingSummary: audit.summary,
        trainedRuntime,
      };
      res.json({
        summary: state.preprocessingSummary,
        models: state.trainedRuntime.bundle,
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Preprocessing failed.',
      });
    }
  });

  app.post('/api/dataset/upload', (req, res) => {
    const csvText = typeof req.body?.csvText === 'string' ? req.body.csvText : '';
    const rawFilename = typeof req.body?.filename === 'string' ? req.body.filename : 'uploaded.csv';
    const cleanFilename = rawFilename.replace(/[^A-Za-z0-9._ -]/g, '').slice(0, 64) || 'uploaded.csv';

    if (!csvText || csvText.trim().length === 0) {
      res.status(400).json({ error: 'The selected CSV file is empty.' });
      return;
    }
    if (csvText.length > 10_000_000) {
      res.status(400).json({ error: 'CSV file is too large. Please upload a file under 10 MB.' });
      return;
    }

    const parsed = parseCsvText(csvText);
    if (parsed.headers.length === 0 || parsed.rows.length === 0) {
      res.status(400).json({
        error: 'Could not read CSV rows. Ensure the file has a header row and comma-, semicolon-, or tab-separated data rows.',
      });
      return;
    }
    if (parsed.rows.length < 10) {
      res.status(400).json({
        error: `Uploaded CSV has only ${parsed.rows.length} row(s). Please include at least 10 patient records for stratified ML evaluation.`,
      });
      return;
    }

    try {
      const normalized = normalizeUploadedCsvRows(parsed.headers, parsed.rows);
      const datasetName = `Uploaded CSV: ${cleanFilename} (${normalized.normalizedRows.length} rows)`;
      const audit = runFullPreprocessingAudit(
        normalized.normalizedRows,
        datasetName,
        false,
        state.config
      );
      const trainedRuntime = trainAllModels(
        normalized.normalizedRows,
        datasetName,
        false,
        state.config
      );

      state = {
        ...state,
        name: datasetName,
        isSynthetic: false,
        rawRows: normalized.normalizedRows,
        cleanedRecords: audit.cleanedRecords,
        preprocessingSummary: audit.summary,
        trainedRuntime,
      };

      res.json({
        summary: state.preprocessingSummary,
        models: state.trainedRuntime.bundle,
        mappedColumns: normalized.mappedColumns,
        warnings: normalized.warnings,
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Failed to process uploaded CSV dataset.',
      });
    }
  });

  app.get('/api/dataset/sample-template', (_req, res) => {
    const sampleRows = state.cleanedRecords.slice(0, 20).map((r) => ({
      patient_id: r.patient_id,
      age: r.age,
      gender: r.gender,
      height_cm: r.height_cm,
      weight_kg: r.weight_kg,
      bmi: r.bmi,
      systolic_bp: r.systolic_bp,
      diastolic_bp: r.diastolic_bp,
      glucose_level: r.glucose_level,
      cholesterol_level: r.cholesterol_level,
      heart_rate: r.heart_rate,
      smoking_status: r.smoking_status,
      physical_activity: r.physical_activity,
      family_history: r.family_history,
      existing_conditions: r.existing_conditions,
      risk_category: r.risk_category,
    }));
    const csv = rowsToCsv(sampleRows, DATASET_COLUMNS);
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="sample_healthcare_template.csv"');
    res.send(csv);
  });

  app.post('/api/dataset/reset', (_req, res) => {
    state = bootstrapDatasetState();
    res.json({
      summary: state.preprocessingSummary,
      models: state.trainedRuntime.bundle,
    });
  });

  app.get('/api/dataset/download-cleaned', (_req, res) => {
    const rows: RawDatasetRow[] = state.cleanedRecords.map((r) => ({
      patient_id: r.patient_id,
      age: r.age,
      gender: r.gender,
      height_cm: r.height_cm,
      weight_kg: r.weight_kg,
      bmi: r.bmi,
      systolic_bp: r.systolic_bp,
      diastolic_bp: r.diastolic_bp,
      glucose_level: r.glucose_level,
      cholesterol_level: r.cholesterol_level,
      heart_rate: r.heart_rate,
      smoking_status: r.smoking_status,
      physical_activity: r.physical_activity,
      family_history: r.family_history,
      existing_conditions: r.existing_conditions,
      risk_category: r.risk_category,
    }));
    const csv = rowsToCsv(rows, DATASET_COLUMNS);
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="cleaned_healthcare_dataset.csv"');
    res.send(csv);
  });

  // --------------------------------------------------------------------------
  // Exploratory Data Analysis (EDA) Endpoint
  // --------------------------------------------------------------------------
  app.get('/api/eda', (_req, res) => {
    const eda = computeEdaAnalytics(state.cleanedRecords, state.name, state.isSynthetic);
    res.json(eda);
  });

  // --------------------------------------------------------------------------
  // Machine Learning Models & Evaluation
  // --------------------------------------------------------------------------
  app.get('/api/models', (_req, res) => {
    res.json(state.trainedRuntime.bundle);
  });

  app.post('/api/models/train', (req, res) => {
    try {
      const config = parsePreprocessingConfig(req.body);
      const audit = runFullPreprocessingAudit(
        state.rawRows,
        state.name,
        state.isSynthetic,
        config
      );
      const trainedRuntime = trainAllModels(
        state.rawRows,
        state.name,
        state.isSynthetic,
        config
      );
      state = {
        ...state,
        config,
        cleanedRecords: audit.cleanedRecords,
        preprocessingSummary: audit.summary,
        trainedRuntime,
      };
      res.json(state.trainedRuntime.bundle);
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Model training failed.',
      });
    }
  });

  // --------------------------------------------------------------------------
  // Live Patient Risk Prediction
  // --------------------------------------------------------------------------
  app.post('/api/predict', (req, res) => {
    const body = req.body ?? {};
    const modelKey = body.model_key as ModelKey;
    if (!ALLOWED_MODELS.includes(modelKey)) {
      res.status(400).json({ error: 'Please select a valid machine learning model.' });
      return;
    }

    const validation = validatePatientPayload(body, false);
    if (!validation.valid) {
      res.status(400).json({ error: 'Validation failed.', errors: validation.errors });
      return;
    }

    const predInput: PredictionInput = {
      ...validation.data,
      model_key: modelKey,
      save_report: body.save_report !== false,
    };

    const result = predictPatientRisk(state.trainedRuntime, predInput);

    if (predInput.save_report) {
      const saved = savePredictionReport({
        patient_id: result.patient_id,
        model_key: result.model_key,
        model_name: result.model_name,
        predicted_risk: result.predicted_risk,
        probabilities: result.probabilities,
        patient_params: {
          age: validation.data.age,
          gender: validation.data.gender,
          height_cm: validation.data.height_cm,
          weight_kg: validation.data.weight_kg,
          bmi: result.bmi,
          systolic_bp: validation.data.systolic_bp,
          diastolic_bp: validation.data.diastolic_bp,
          glucose_level: validation.data.glucose_level,
          cholesterol_level: validation.data.cholesterol_level,
          heart_rate: validation.data.heart_rate,
          smoking_status: validation.data.smoking_status,
          physical_activity: validation.data.physical_activity,
          family_history: validation.data.family_history,
          existing_conditions: validation.data.existing_conditions,
        },
        contributing_factors: result.contributing_factors,
        recommendations: result.recommendations,
      });
      result.report_id = saved.id;
      result.created_at = saved.created_at;
    }

    res.json(result);
  });

  // --------------------------------------------------------------------------
  // Saved Risk Reports (SQLite)
  // --------------------------------------------------------------------------
  app.get('/api/reports', (_req, res) => {
    res.json({ reports: listSavedReports() });
  });

  app.delete('/api/reports/:id', (req, res) => {
    const id = Number(req.params.id);
    if (!Number.isInteger(id) || id <= 0) {
      res.status(400).json({ error: 'Invalid report ID.' });
      return;
    }
    const ok = deleteSavedReport(id);
    if (!ok) {
      res.status(404).json({ error: 'Report not found.' });
      return;
    }
    res.json({ deleted: true, id });
  });

  // --------------------------------------------------------------------------
  // Python/Streamlit Project Bundle Viewer (Strict Allowlist & Path Confinement)
  // --------------------------------------------------------------------------
  app.get('/api/python-files', (_req, res) => {
    res.json({
      files: ALLOWED_PYTHON_FILES.map((name) => ({
        name,
        path: `Healthcare_Risk_Prediction/${name}`,
      })),
    });
  });

  app.get('/api/python-files/:filename', (req, res) => {
    const filename = String(req.params.filename ?? '');
    if (!(ALLOWED_PYTHON_FILES as readonly string[]).includes(filename)) {
      res.status(404).json({ error: 'File not found in project bundle.' });
      return;
    }
    const baseDir = path.resolve(process.cwd(), 'Healthcare_Risk_Prediction');
    const resolved = path.resolve(baseDir, filename);
    if (!resolved.startsWith(baseDir + path.sep)) {
      res.status(403).json({ error: 'Access denied.' });
      return;
    }
    if (!fs.existsSync(resolved)) {
      res.status(404).json({ error: 'Project file has not been generated yet.' });
      return;
    }
    const content = fs.readFileSync(resolved, 'utf-8');
    res.json({ filename, content });
  });

  // --------------------------------------------------------------------------
  // Vite Dev Middleware or Production Static Serving
  // --------------------------------------------------------------------------
  let viteServerPromise: Promise<Awaited<ReturnType<typeof createViteServer>>> | null = null;

  if (process.env.NODE_ENV !== 'production') {
    // Start Vite without opening a separate HMR WebSocket port (24678) or file watchers
    viteServerPromise = createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        watch: null,
      },
      appType: 'spa',
    });

    app.use((req, res, next) => {
      viteServerPromise!
        .then((vite) => vite.middlewares(req, res, next))
        .catch(next);
    });
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const PORT = Number(process.env.PORT) || 3000;
  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`Healthcare Risk Prediction System running on http://localhost:${PORT}`);
    // Warm up SQLite seed and train ML models right after port 3000 is open
    setImmediate(() => {
      try {
        ensureState();
      } catch (err) {
        console.error('Background state bootstrap failed:', err);
      }
    });
  });

  const shutdown = () => {
    server.close(() => {
      if (viteServerPromise) {
        viteServerPromise
          .then((vite) => vite.close())
          .finally(() => process.exit(0));
      } else {
        process.exit(0);
      }
    });
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

startServer();
