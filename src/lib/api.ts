import type {
  EdaData,
  OverviewStats,
  PatientRecord,
  PredictionInput,
  PredictionResult,
  PreprocessingConfig,
  PreprocessingSummary,
  SavedReport,
  TrainedModelsBundle,
} from '../types';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`) as Error & {
      fieldErrors?: Record<string, string>;
    };
    if (data.errors) err.fieldErrors = data.errors;
    throw err;
  }
  return data as T;
}

export const api = {
  getOverview: () => fetchJson<OverviewStats>('/api/overview'),
  getPatients: () => fetchJson<{ patients: PatientRecord[] }>('/api/patients'),
  createPatient: (payload: Record<string, unknown>) =>
    fetchJson<{
      patient: PatientRecord;
      initialPrediction: {
        model_name: string;
        predicted_risk: string;
        probabilities: Record<string, number>;
      };
    }>('/api/patients', { method: 'POST', body: JSON.stringify(payload) }),
  deletePatient: (patientId: string) =>
    fetchJson<{ deleted: boolean }>(`/api/patients/${encodeURIComponent(patientId)}`, {
      method: 'DELETE',
    }),

  getPreprocessing: () => fetchJson<PreprocessingSummary>('/api/preprocessing'),
  runPreprocessing: (config: PreprocessingConfig) =>
    fetchJson<{ summary: PreprocessingSummary; models: TrainedModelsBundle }>(
      '/api/preprocessing/run',
      { method: 'POST', body: JSON.stringify(config) }
    ),
  uploadDatasetCsv: (filename: string, csvText: string) =>
    fetchJson<{
      summary: PreprocessingSummary;
      models: TrainedModelsBundle;
      mappedColumns?: Record<string, string>;
      warnings?: string[];
    }>('/api/dataset/upload', { method: 'POST', body: JSON.stringify({ filename, csvText }) }),
  resetDataset: () =>
    fetchJson<{ summary: PreprocessingSummary; models: TrainedModelsBundle }>(
      '/api/dataset/reset',
      { method: 'POST' }
    ),

  getEda: () => fetchJson<EdaData>('/api/eda'),
  getModels: () => fetchJson<TrainedModelsBundle>('/api/models'),
  trainModels: (config: PreprocessingConfig) =>
    fetchJson<TrainedModelsBundle>('/api/models/train', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  predictRisk: (input: PredictionInput) =>
    fetchJson<PredictionResult>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(input),
    }),

  getReports: () => fetchJson<{ reports: SavedReport[] }>('/api/reports'),
  deleteReport: (id: number) =>
    fetchJson<{ deleted: boolean }>(`/api/reports/${id}`, { method: 'DELETE' }),

  getPythonFiles: () =>
    fetchJson<{ files: Array<{ name: string; path: string }> }>('/api/python-files'),
  getPythonFileContent: (filename: string) =>
    fetchJson<{ filename: string; content: string }>(
      `/api/python-files/${encodeURIComponent(filename)}`
    ),
};
