export type Gender = 'Male' | 'Female' | 'Other';
export type SmokingStatus = 'Never' | 'Former' | 'Current';
export type PhysicalActivity = 'Low' | 'Moderate' | 'High';
export type FamilyHistory = 'None' | 'Heart Disease' | 'Diabetes' | 'Hypertension' | 'Multiple';
export type ExistingConditions = 'None' | 'Hypertension' | 'Diabetes' | 'Asthma' | 'Heart Disease' | 'Multiple';
export type RiskCategory = 'Low Risk' | 'Moderate Risk' | 'High Risk';

export type ModelKey =
  | 'logistic_regression'
  | 'decision_tree'
  | 'random_forest'
  | 'gradient_boosting';

export interface PatientRecord {
  id?: number;
  patient_id: string;
  age: number;
  gender: Gender;
  height_cm: number;
  weight_kg: number;
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
  risk_category: RiskCategory;
  record_source: 'synthetic_demo' | 'registered' | 'uploaded_csv';
  created_at?: string;
}

export interface RawDatasetRow {
  [key: string]: string | number | null | undefined;
}

export interface MissingColumnStat {
  column: string;
  missingCount: number;
  missingPercent: number;
  imputationApplied: string;
  fillValue: string;
}

export interface OutlierColumnStat {
  column: string;
  q1: number;
  q3: number;
  iqr: number;
  lowerBound: number;
  upperBound: number;
  outlierCount: number;
}

export interface PreprocessingConfig {
  numericImputation: 'mean' | 'median';
  categoricalImputation: 'mode';
  removeDuplicates: boolean;
  outlierStrategy: 'clip' | 'keep';
  scalingMethod: 'standard' | 'minmax';
  testSize: number;
  randomSeed: number;
}

export interface PreprocessingSummary {
  datasetName: string;
  isSynthetic: boolean;
  rawRowCount: number;
  cleanedRowCount: number;
  duplicateRowsRemoved: number;
  totalMissingValues: number;
  missingByColumn: MissingColumnStat[];
  outliersByColumn: OutlierColumnStat[];
  totalOutliersDetected: number;
  encodingMaps: Record<string, Record<string, number>>;
  scalingStats: Array<{
    feature: string;
    mean: number;
    std: number;
    min: number;
    max: number;
  }>;
  trainCount: number;
  testCount: number;
  trainClassDistribution: Record<RiskCategory, number>;
  testClassDistribution: Record<RiskCategory, number>;
  previewCleaned: PatientRecord[];
  config: PreprocessingConfig;
  operationsLog: Array<{
    step: string;
    title: string;
    detail: string;
    status: string;
  }>;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface FeatureImportanceItem {
  feature: string;
  label: string;
  importance: number;
}

export interface ModelEvaluation {
  key: ModelKey;
  name: string;
  shortName: string;
  description: string;
  hyperparameters: Record<string, string | number>;
  accuracy: number;
  precisionMacro: number;
  recallMacro: number;
  f1Macro: number;
  precisionWeighted: number;
  recallWeighted: number;
  f1Weighted: number;
  macroAuc: number;
  perClassMetrics: Record<
    RiskCategory,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
      auc: number;
    }
  >;
  confusionMatrix: number[][];
  classLabels: RiskCategory[];
  macroRocCurve: RocPoint[];
  perClassRocCurves: Record<RiskCategory, RocPoint[]>;
  featureImportance: FeatureImportanceItem[];
  trainSize: number;
  testSize: number;
  trainingTimeMs: number;
}

export interface TrainedModelsBundle {
  trainedAt: string;
  datasetName: string;
  isSynthetic: boolean;
  config: PreprocessingConfig;
  bestModelKey: ModelKey;
  models: Record<ModelKey, ModelEvaluation>;
}

export interface PredictionInput {
  patient_id?: string;
  model_key: ModelKey;
  save_report?: boolean;
  age: number;
  gender: Gender;
  height_cm: number;
  weight_kg: number;
  systolic_bp: number;
  diastolic_bp: number;
  glucose_level: number;
  cholesterol_level: number;
  heart_rate: number;
  smoking_status: SmokingStatus;
  physical_activity: PhysicalActivity;
  family_history: FamilyHistory;
  existing_conditions: ExistingConditions;
}

export interface ContributingFactor {
  feature: string;
  label: string;
  patientValue: string;
  referenceRange: string;
  status: 'Elevated' | 'Borderline' | 'Normal';
  contributionScore: number;
}

export interface PredictionResult {
  report_id?: number;
  patient_id: string;
  model_key: ModelKey;
  model_name: string;
  predicted_risk: RiskCategory;
  probabilities: Record<RiskCategory, number>;
  confidence: number;
  bmi: number;
  contributing_factors: ContributingFactor[];
  parameter_summary: Array<{
    parameter: string;
    value: string;
    reference: string;
    status: 'Normal' | 'Borderline' | 'Elevated';
  }>;
  recommendations: string[];
  target_definition_note: string;
  disclaimer: string;
  created_at: string;
}

export interface SavedReport {
  id: number;
  patient_id: string;
  model_key: ModelKey;
  model_name: string;
  predicted_risk: RiskCategory;
  probabilities: Record<RiskCategory, number>;
  patient_params: {
    age: number;
    gender: Gender;
    height_cm: number;
    weight_kg: number;
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
  };
  contributing_factors: ContributingFactor[];
  recommendations: string[];
  created_at: string;
}

export interface HistogramBin {
  range: string;
  count: number;
  lowRisk: number;
  moderateRisk: number;
  highRisk: number;
}

export interface BoxPlotStat {
  category: RiskCategory;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  count: number;
}

export interface EdaData {
  datasetName: string;
  isSynthetic: boolean;
  totalRecords: number;
  ageHistogram: HistogramBin[];
  genderDistribution: Array<{ name: string; value: number; percentage: number }>;
  bmiHistogram: HistogramBin[];
  bloodPressureByRisk: Array<{
    risk: RiskCategory;
    avgSystolic: number;
    avgDiastolic: number;
    count: number;
  }>;
  glucoseHistogram: HistogramBin[];
  cholesterolBoxPlot: BoxPlotStat[];
  riskDistribution: Array<{ name: RiskCategory; value: number; percentage: number }>;
  correlationMatrix: {
    features: string[];
    labels: string[];
    matrix: number[][];
  };
}

export interface OverviewStats {
  totalPatients: number;
  registeredPatients: number;
  syntheticPatients: number;
  highRiskCount: number;
  moderateRiskCount: number;
  lowRiskCount: number;
  savedReportsCount: number;
  activeDatasetName: string;
  isSyntheticDataset: boolean;
  bestModel: {
    key: ModelKey;
    name: string;
    accuracy: number;
    f1Macro: number;
    macroAuc: number;
  } | null;
  modelSummary: Array<{
    key: ModelKey;
    name: string;
    shortName: string;
    accuracy: number;
    f1Macro: number;
    macroAuc: number;
  }>;
  recentReports: SavedReport[];
}
