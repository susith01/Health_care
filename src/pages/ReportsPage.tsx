import React, { useEffect, useState } from 'react';
import { Download, FileSpreadsheet, FileText, Trash2 } from 'lucide-react';
import { api } from '../lib/api';
import { downloadReportCsv, downloadReportPdf } from '../lib/reportExport';
import type { SavedReport } from '../types';

export function ReportsPage() {
  const [reports, setReports] = useState<SavedReport[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getReports()
      .then((res) => {
        setReports(res.reports);
        if (res.reports.length > 0) setSelectedId(res.reports[0].id);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load saved reports.'))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (id: number) => {
    try {
      await api.deleteReport(id);
      setReports((prev) => {
        const next = prev.filter((r) => r.id !== id);
        if (selectedId === id) {
          setSelectedId(next[0]?.id ?? null);
        }
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete report.');
    }
  };

  const activeReport = reports.find((r) => r.id === selectedId) ?? null;

  const riskColor = (r: string) =>
    r === 'High Risk'
      ? 'text-red-700 font-semibold'
      : r === 'Moderate Risk'
        ? 'text-amber-700 font-semibold'
        : 'text-teal-700 font-semibold';

  return (
    <div className="space-y-6">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          Patient Risk Reports
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          De-identified patient risk assessment reports persisted in SQLite. Download structured
          PDF or CSV summaries for educational review.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-8 text-sm text-slate-500">Loading saved reports...</div>
      ) : reports.length === 0 ? (
        <div className="rounded-lg border border-slate-200 bg-white p-10 text-center">
          <FileText className="w-9 h-9 text-slate-300 mx-auto mb-3" />
          <h2 className="text-base font-semibold text-slate-800">No Risk Reports Saved Yet</h2>
          <p className="mt-1 text-sm text-slate-500 max-w-md mx-auto">
            Navigate to the <strong>Risk Prediction</strong> page and run a patient evaluation with
            "Save prediction record" enabled to populate downloadable PDF and CSV reports here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left list */}
          <div className="lg:col-span-5 rounded-lg border border-slate-200 bg-white overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-200 text-xs font-medium text-slate-600">
              Saved Assessments ({reports.length})
            </div>
            <div className="divide-y divide-slate-100 max-h-[600px] overflow-y-auto">
              {reports.map((rep) => {
                const isSelected = rep.id === selectedId;
                return (
                  <div
                    key={rep.id}
                    className={`px-4 py-3 flex items-center justify-between gap-3 transition-colors ${
                      isSelected ? 'bg-sky-50/60' : 'hover:bg-slate-50'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => setSelectedId(rep.id)}
                      className="text-left flex-1 min-w-0"
                    >
                      <div className="flex items-center gap-2 text-xs">
                        <span className="font-mono font-semibold text-slate-900">
                          {rep.patient_id}
                        </span>
                        <span className="text-slate-300" aria-hidden="true">·</span>
                        <span className={riskColor(rep.predicted_risk)}>{rep.predicted_risk}</span>
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-500 truncate">
                        {rep.model_name} · {new Date(rep.created_at).toLocaleString()}
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(rep.id)}
                      className="p-1.5 text-slate-400 hover:text-red-600 transition-colors shrink-0"
                      title="Delete report"
                      aria-label={`Delete report ${rep.patient_id}`}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right detail preview */}
          {activeReport && (
            <section className="lg:col-span-7 rounded-lg border border-slate-200 bg-white p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 border-b border-slate-200 pb-4">
                <div>
                  <div className="text-xs text-slate-500">
                    De-Identified Risk Report #{activeReport.id} ·{' '}
                    {new Date(activeReport.created_at).toLocaleString()}
                  </div>
                  <h2 className="mt-1 text-lg font-bold text-slate-900">
                    Patient <span className="font-mono">{activeReport.patient_id}</span> —{' '}
                    <span className={riskColor(activeReport.predicted_risk)}>
                      {activeReport.predicted_risk}
                    </span>
                  </h2>
                  <div className="mt-1 text-xs text-slate-600">
                    Model: <span className="font-medium text-slate-800">{activeReport.model_name}</span>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={() => downloadReportPdf(activeReport)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 transition-colors whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download PDF
                  </button>
                  <button
                    type="button"
                    onClick={() => downloadReportCsv(activeReport)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    Download CSV
                  </button>
                </div>
              </div>

              {/* Probabilities */}
              <div className="grid grid-cols-3 gap-3 text-xs">
                {(['Low Risk', 'Moderate Risk', 'High Risk'] as const).map((tier) => (
                  <div key={tier} className="rounded-md bg-slate-50 p-3">
                    <div className="text-slate-500">{tier}</div>
                    <div className="mt-1 text-base font-bold font-mono tabular-nums text-slate-900">
                      {(activeReport.probabilities[tier] * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* Clinical snapshot */}
              <div>
                <h3 className="text-xs font-semibold text-slate-800 mb-2">
                  Clinical Parameters Snapshot
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-2 gap-x-4 text-xs border border-slate-200 rounded-md p-3.5">
                  <div>
                    <span className="text-slate-500">Age: </span>
                    <span className="font-mono text-slate-900">{activeReport.patient_params.age} yrs</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Gender: </span>
                    <span className="text-slate-900">{activeReport.patient_params.gender}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">BMI: </span>
                    <span className="font-mono text-slate-900">
                      {activeReport.patient_params.bmi} kg/m²
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">BP: </span>
                    <span className="font-mono text-slate-900">
                      {activeReport.patient_params.systolic_bp}/
                      {activeReport.patient_params.diastolic_bp} mmHg
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Glucose: </span>
                    <span className="font-mono text-slate-900">
                      {activeReport.patient_params.glucose_level} mg/dL
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Cholesterol: </span>
                    <span className="font-mono text-slate-900">
                      {activeReport.patient_params.cholesterol_level} mg/dL
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Heart Rate: </span>
                    <span className="font-mono text-slate-900">
                      {activeReport.patient_params.heart_rate} bpm
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Smoking: </span>
                    <span className="text-slate-900">
                      {activeReport.patient_params.smoking_status}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Activity: </span>
                    <span className="text-slate-900">
                      {activeReport.patient_params.physical_activity}
                    </span>
                  </div>
                  <div className="col-span-2 sm:col-span-3">
                    <span className="text-slate-500">History / Conditions: </span>
                    <span className="text-slate-900">
                      Family: {activeReport.patient_params.family_history} · Existing:{' '}
                      {activeReport.patient_params.existing_conditions}
                    </span>
                  </div>
                </div>
              </div>

              {/* Contributing Factors */}
              <div>
                <h3 className="text-xs font-semibold text-slate-800 mb-2">
                  Top Contributing Factors
                </h3>
                <div className="space-y-1.5 text-xs">
                  {activeReport.contributing_factors.map((f) => (
                    <div
                      key={f.feature}
                      className="flex items-center justify-between py-1 border-b border-slate-100 last:border-b-0"
                    >
                      <span className="text-slate-700">{f.label}</span>
                      <span className="font-mono tabular-nums text-slate-900">
                        {f.patientValue} ({f.status})
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div>
                <h3 className="text-xs font-semibold text-slate-800 mb-2">
                  General Recommendations
                </h3>
                <ul className="list-disc pl-4 space-y-1.5 text-xs text-slate-600 leading-relaxed">
                  {activeReport.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>

              <p className="text-[11px] text-slate-500 pt-2 border-t border-slate-200">
                Privacy &amp; Clinical Notice: Reports store de-identified Patient IDs without
                names, addresses, or national identifiers. Output is for educational decision support
                only and does not constitute medical advice.
              </p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
