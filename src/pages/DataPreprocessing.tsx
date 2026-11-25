import React, { useEffect, useRef, useState } from 'react';
import { Download, Play, RotateCcw, Upload } from 'lucide-react';
import { api } from '../lib/api';
import type { PreprocessingConfig, PreprocessingSummary } from '../types';

export function DataPreprocessing() {
  const [summary, setSummary] = useState<PreprocessingSummary | null>(null);
  const [config, setConfig] = useState<PreprocessingConfig>({
    numericImputation: 'median',
    categoricalImputation: 'mode',
    removeDuplicates: true,
    outlierStrategy: 'clip',
    scalingMethod: 'standard',
    testSize: 0.2,
    randomSeed: 42,
  });
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .getPreprocessing()
      .then((data) => {
        setSummary(data);
        setConfig(data.config);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load preprocessing state.'))
      .finally(() => setLoading(false));
  }, []);

  const handleRunPipeline = async () => {
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.runPreprocessing(config);
      setSummary(res.summary);
      setNotice('Preprocessing pipeline executed and all four ML models were retrained without data leakage.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run preprocessing.');
    } finally {
      setRunning(false);
    }
  };

  const handleUploadCsv = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const text = await file.text();
      const res = await api.uploadDatasetCsv(file.name, text);
      setSummary(res.summary);
      const mappedCount = res.mappedColumns ? Object.keys(res.mappedColumns).length : 0;
      const warnSuffix =
        res.warnings && res.warnings.length > 0 ? ` Note: ${res.warnings.join(' ')}` : '';
      setNotice(
        `Loaded "${file.name}" (${res.summary.cleanedRowCount} clean records, ${mappedCount} columns matched) and retrained all 4 ML models.${warnSuffix}`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CSV upload failed.');
    } finally {
      setRunning(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleResetDataset = async () => {
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.resetDataset();
      setSummary(res.summary);
      setConfig(res.summary.config);
      setNotice('Restored default synthetic demonstration dataset and retrained all four models.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Reset failed.');
    } finally {
      setRunning(false);
    }
  };

  if (loading || !summary) {
    return (
      <div className="p-8 text-sm text-slate-500">Loading data preprocessing audit...</div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Data Preprocessing Studio
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Active Dataset: <span className="font-medium text-slate-900">{summary.datasetName}</span>
            {summary.isSynthetic && (
              <span> · Clearly labeled synthetic demonstration data</span>
            )}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Supports comma, semicolon, or tab-delimited CSVs (UTF-8 / Excel BOM), flexible column
            names (e.g., Age, Sex, Blood Pressure, Glucose, Cholesterol, Heart Rate, Risk Level),
            and 10+ records.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv,text/plain"
            onChange={handleUploadCsv}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={running}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
          >
            <Upload className="w-3.5 h-3.5" />
            {running ? 'Uploading...' : 'Upload CSV Dataset'}
          </button>
          <a
            href="/api/dataset/sample-template"
            download="sample_healthcare_template.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Sample CSV
          </a>
          <a
            href="/api/dataset/download-cleaned"
            download="cleaned_healthcare_dataset.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Download Cleaned CSV
          </a>
          {!summary.isSynthetic && (
            <button
              type="button"
              onClick={handleResetDataset}
              disabled={running}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Restore Default Dataset
            </button>
          )}
        </div>
      </div>

      {notice && (
        <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4 text-sm text-teal-900">
          {notice}
        </div>
      )}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Pipeline Controls */}
      <section className="rounded-lg border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Pipeline Configuration</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Imputers and scalers are fitted exclusively on the training partition after stratified
              splitting to eliminate data leakage.
            </p>
          </div>
          <button
            type="button"
            onClick={handleRunPipeline}
            disabled={running}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            {running ? 'Running Pipeline...' : 'Apply Preprocessing & Retrain'}
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 pt-2">
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Numeric Imputation
            </label>
            <select
              value={config.numericImputation}
              onChange={(e) =>
                setConfig((c) => ({ ...c, numericImputation: e.target.value as 'mean' | 'median' }))
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white"
            >
              <option value="median">Median Imputation</option>
              <option value="mean">Mean Imputation</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Categorical Imputation
            </label>
            <input
              type="text"
              readOnly
              value="Mode Imputation"
              className="w-full rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Outlier Handling (1.5x IQR)
            </label>
            <select
              value={config.outlierStrategy}
              onChange={(e) =>
                setConfig((c) => ({ ...c, outlierStrategy: e.target.value as 'clip' | 'keep' }))
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white"
            >
              <option value="clip">Winsorize (Clip to IQR Bounds)</option>
              <option value="keep">Detect &amp; Retain Values</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Feature Scaling / Normalization
            </label>
            <select
              value={config.scalingMethod}
              onChange={(e) =>
                setConfig((c) => ({ ...c, scalingMethod: e.target.value as 'standard' | 'minmax' }))
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white"
            >
              <option value="standard">StandardScaler (Z-Score)</option>
              <option value="minmax">Min-Max Normalization [0, 1]</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Stratified Test Split ({Math.round(config.testSize * 100)}%)
            </label>
            <select
              value={String(config.testSize)}
              onChange={(e) => setConfig((c) => ({ ...c, testSize: Number(e.target.value) }))}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs bg-white font-mono"
            >
              <option value="0.15">85% Train / 15% Test</option>
              <option value="0.2">80% Train / 20% Test</option>
              <option value="0.25">75% Train / 25% Test</option>
              <option value="0.3">70% Train / 30% Test</option>
            </select>
          </div>
        </div>

        <label className="inline-flex items-center gap-2 text-xs text-slate-700 pt-1">
          <input
            type="checkbox"
            checked={config.removeDuplicates}
            onChange={(e) => setConfig((c) => ({ ...c, removeDuplicates: e.target.checked }))}
            className="rounded border-slate-300 text-sky-700 focus:ring-sky-600"
          />
          Remove exact duplicate rows and duplicate Patient IDs before splitting
        </label>
      </section>

      {/* Audit KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs text-slate-500">Raw Rows</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-slate-900">
            {summary.rawRowCount}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs text-slate-500">Duplicates Removed</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-sky-700">
            {summary.duplicateRowsRemoved}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs text-slate-500">Missing Values Imputed</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-amber-700">
            {summary.totalMissingValues}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs text-slate-500">IQR Outliers Detected</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-slate-900">
            {summary.totalOutliersDetected}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs text-slate-500">Train / Test Split</div>
          <div className="mt-1 text-2xl font-bold font-mono tabular-nums text-teal-700">
            {summary.trainCount} / {summary.testCount}
          </div>
        </div>
      </div>

      {/* Step-by-step Operations Log */}
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-base font-semibold text-slate-900 mb-3">
          Executed Preprocessing Operations &amp; Rationale
        </h2>
        <div className="divide-y divide-slate-100">
          {summary.operationsLog.map((op) => (
            <div key={op.step} className="py-3 flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
              <span className="font-mono text-xs font-semibold text-sky-700 shrink-0">
                {op.step}.
              </span>
              <div className="flex-1">
                <div className="text-sm font-medium text-slate-900">{op.title}</div>
                <p className="text-xs text-slate-600 mt-0.5 leading-relaxed">{op.detail}</p>
              </div>
              <span className="text-xs font-medium text-teal-700 shrink-0">{op.status}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Missing Values & Outliers Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              Missing Value Detection &amp; Imputation Summary
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                  <th className="py-2 px-4">Column</th>
                  <th className="py-2 px-3 text-right">Missing</th>
                  <th className="py-2 px-3">Strategy</th>
                  <th className="py-2 px-4 text-right">Fill Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {summary.missingByColumn.map((m) => (
                  <tr key={m.column}>
                    <td className="py-2 px-4 font-mono text-slate-800">{m.column}</td>
                    <td
                      className={`py-2 px-3 text-right font-mono tabular-nums ${
                        m.missingCount > 0 ? 'text-amber-700 font-semibold' : 'text-slate-500'
                      }`}
                    >
                      {m.missingCount} ({m.missingPercent}%)
                    </td>
                    <td className="py-2 px-3 text-slate-600">{m.imputationApplied}</td>
                    <td className="py-2 px-4 text-right font-mono tabular-nums text-slate-800">
                      {m.fillValue}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              IQR Outlier Detection Bounds (1.5 × IQR)
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                  <th className="py-2 px-4">Feature</th>
                  <th className="py-2 px-3 text-right">Q1 – Q3</th>
                  <th className="py-2 px-3 text-right">IQR Bounds</th>
                  <th className="py-2 px-4 text-right">Outliers</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {summary.outliersByColumn.map((o) => (
                  <tr key={o.column}>
                    <td className="py-2 px-4 font-mono text-slate-800">{o.column}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                      {o.q1} – {o.q3}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                      [{o.lowerBound}, {o.upperBound}]
                    </td>
                    <td
                      className={`py-2 px-4 text-right font-mono tabular-nums ${
                        o.outlierCount > 0 ? 'text-amber-700 font-semibold' : 'text-slate-500'
                      }`}
                    >
                      {o.outlierCount}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Label Encoding & Feature Scaling Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">
            Deterministic Label Encoding Mapping
          </h2>
          <div className="space-y-3 text-xs">
            {Object.entries(summary.encodingMaps).map(([col, mapping]) => (
              <div key={col} className="pb-2.5 border-b border-slate-100 last:border-b-0">
                <div className="font-mono font-semibold text-slate-800 mb-1">{col}</div>
                <div className="text-slate-600 flex flex-wrap gap-x-3 gap-y-1">
                  {Object.entries(mapping).map(([label, code]) => (
                    <span key={label}>
                      {label} → <strong className="font-mono text-slate-900">{code}</strong>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              Training-Fold Feature Scaling Statistics (Leak-Free)
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                  <th className="py-2 px-4">Feature</th>
                  <th className="py-2 px-3 text-right">Mean</th>
                  <th className="py-2 px-3 text-right">Std Dev</th>
                  <th className="py-2 px-4 text-right">Min / Max</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {summary.scalingStats.map((s) => (
                  <tr key={s.feature}>
                    <td className="py-2 px-4 text-slate-800">{s.feature}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {s.mean.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {s.std.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-right font-mono tabular-nums text-slate-500">
                      {s.min} / {s.max}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Stratified Split Distribution */}
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900 mb-3">
          Stratified Train-Test Class Balance Verification
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div>
            <div className="font-medium text-slate-700 mb-2">
              Training Partition (n = {summary.trainCount})
            </div>
            <div className="space-y-1.5 font-mono tabular-nums">
              {Object.entries(summary.trainClassDistribution).map(([tier, cnt]) => (
                <div key={tier} className="flex justify-between">
                  <span className="font-sans text-slate-600">{tier}</span>
                  <span>
                    {cnt} ({((cnt / Math.max(1, summary.trainCount)) * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="font-medium text-slate-700 mb-2">
              Hold-Out Test Partition (n = {summary.testCount})
            </div>
            <div className="space-y-1.5 font-mono tabular-nums">
              {Object.entries(summary.testClassDistribution).map(([tier, cnt]) => (
                <div key={tier} className="flex justify-between">
                  <span className="font-sans text-slate-600">{tier}</span>
                  <span>
                    {cnt} ({((cnt / Math.max(1, summary.testCount)) * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
