import React, { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ArrowRight, FileText, ShieldAlert, UserPlus } from 'lucide-react';
import { api } from '../lib/api';
import type { OverviewStats } from '../types';

export type NavPageId =
  | 'home'
  | 'register'
  | 'patients'
  | 'preprocessing'
  | 'eda'
  | 'models'
  | 'predict'
  | 'performance'
  | 'reports'
  | 'about';

function riskTextColor(risk: string): string {
  if (risk === 'High Risk') return 'text-red-700 font-semibold';
  if (risk === 'Moderate Risk') return 'text-amber-700 font-semibold';
  return 'text-teal-700 font-semibold';
}

export function HomeDashboard({ onNavigate }: { onNavigate: (page: NavPageId) => void }) {
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    api
      .getOverview()
      .then((data) => {
        if (active) setOverview(data);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'Unable to load overview.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-28 rounded-lg border border-slate-200 bg-slate-50 animate-pulse" />
          ))}
        </div>
        <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 animate-pulse" />
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-5 text-sm text-red-800">
        {error || 'Dashboard metrics could not be loaded.'}
      </div>
    );
  }

  const riskBreakdown = [
    {
      label: 'Low Risk',
      count: overview.lowRiskCount,
      pct:
        overview.totalPatients > 0
          ? ((overview.lowRiskCount / overview.totalPatients) * 100).toFixed(1)
          : '0.0',
      barClass: 'bg-teal-600',
    },
    {
      label: 'Moderate Risk',
      count: overview.moderateRiskCount,
      pct:
        overview.totalPatients > 0
          ? ((overview.moderateRiskCount / overview.totalPatients) * 100).toFixed(1)
          : '0.0',
      barClass: 'bg-amber-500',
    },
    {
      label: 'High Risk',
      count: overview.highRiskCount,
      pct:
        overview.totalPatients > 0
          ? ((overview.highRiskCount / overview.totalPatients) * 100).toFixed(1)
          : '0.0',
      barClass: 'bg-red-600',
    },
  ];

  const modelBarData = overview.modelSummary.map((m) => ({
    name: m.shortName,
    Accuracy: Math.round(m.accuracy * 1000) / 10,
    'Macro F1': Math.round(m.f1Macro * 1000) / 10,
    isBest: overview.bestModel?.key === m.key,
  }));

  return (
    <div className="space-y-8">
      {/* Header & Non-Diagnostic Educational Notice */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-6">
        <div className="space-y-1.5 max-w-2xl">
          <p className="text-xs font-medium text-sky-700">
            Clinical Analytics &amp; Supervised Risk Stratification
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
            Multi-Factor Healthcare Risk Prediction System
          </h1>
          <p className="text-sm text-slate-600 leading-relaxed">
            Analyze multi-factor patient vitals, metabolic biomarkers, lifestyle habits, and clinical
            history across four trained supervised machine learning models.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => onNavigate('register')}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <UserPlus className="w-4 h-4 text-slate-600" />
            Register Patient
          </button>
          <button
            type="button"
            onClick={() => onNavigate('predict')}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 transition-colors whitespace-nowrap shrink-0"
          >
            Run Risk Prediction
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Educational Disclaimer */}
      <div className="rounded-lg border border-sky-200 bg-sky-50/70 px-4 py-3 flex items-start gap-3">
        <ShieldAlert className="w-4 h-4 text-sky-800 mt-0.5 shrink-0" />
        <div className="text-xs sm:text-sm text-slate-700 leading-relaxed">
          <span className="font-semibold text-slate-900">Educational Decision-Support Notice: </span>
          This platform is an academic machine learning demonstration, not a certified medical
          diagnostic device.{' '}
          {overview.isSyntheticDataset && (
            <span>
              The active training dataset ({overview.activeDatasetName}) is clearly labeled synthetic
              demonstration data, not real patient records.
            </span>
          )}
        </div>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">Total SQLite Patient Records</div>
          <div className="mt-2 text-3xl font-bold text-slate-900 font-mono tabular-nums">
            {overview.totalPatients}
          </div>
          <div className="mt-2 text-xs text-slate-500">
            <span>{overview.registeredPatients} registered</span>
            <span className="mx-1.5" aria-hidden="true">·</span>
            <span>{overview.syntheticPatients} synthetic demo</span>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">High-Risk Stratification</div>
          <div className="mt-2 text-3xl font-bold text-red-700 font-mono tabular-nums">
            {overview.highRiskCount}
          </div>
          <div className="mt-2 text-xs text-slate-500">
            <span>
              {overview.totalPatients > 0
                ? ((overview.highRiskCount / overview.totalPatients) * 100).toFixed(1)
                : '0.0'}
              % of cohort
            </span>
            <span className="mx-1.5" aria-hidden="true">·</span>
            <span>{overview.moderateRiskCount} moderate</span>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">Top Evaluated Model</div>
          <div className="mt-2 text-3xl font-bold text-teal-700 font-mono tabular-nums">
            {overview.bestModel ? `${(overview.bestModel.accuracy * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="mt-2 text-xs text-slate-600 truncate">
            {overview.bestModel ? (
              <>
                <span className="font-medium text-slate-800">{overview.bestModel.name}</span>
                <span className="mx-1.5 text-slate-400" aria-hidden="true">·</span>
                <span className="font-mono tabular-nums">
                  AUC {overview.bestModel.macroAuc.toFixed(3)}
                </span>
              </>
            ) : (
              'Models pending training'
            )}
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">Saved Risk Reports</div>
          <div className="mt-2 text-3xl font-bold text-slate-900 font-mono tabular-nums">
            {overview.savedReportsCount}
          </div>
          <div className="mt-2 text-xs text-slate-500">
            <button
              type="button"
              onClick={() => onNavigate('reports')}
              className="text-sky-700 hover:underline font-medium"
            >
              Export PDF or CSV reports
            </button>
          </div>
        </div>
      </div>

      {/* Primary Analytics Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 rounded-lg border border-slate-200 bg-white p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-4 mb-1">
              <h2 className="text-base font-semibold text-slate-900">
                Test-Set Model Accuracy &amp; Macro F1 Comparison
              </h2>
              <button
                type="button"
                onClick={() => onNavigate('performance')}
                className="text-xs font-medium text-sky-700 hover:underline whitespace-nowrap shrink-0"
              >
                Full Evaluation
              </button>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Computed directly from stratified hold-out test predictions using a leak-free
              preprocessing pipeline.
            </p>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelBarData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#334155' }} />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 11, fill: '#64748B' }}
                  unit="%"
                />
                <Tooltip
                  formatter={(v: unknown) => [`${v}%`, '']}
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#CBD5E1',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="Accuracy" radius={[3, 3, 0, 0]} maxBarSize={38}>
                  {modelBarData.map((entry, idx) => (
                    <Cell
                      key={idx}
                      fill={entry.isBest ? '#0F5FA8' : '#38BDF8'}
                    />
                  ))}
                </Bar>
                <Bar dataKey="Macro F1" fill="#0D9488" radius={[3, 3, 0, 0]} maxBarSize={38} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-5 rounded-lg border border-slate-200 bg-white p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-1">
              <h2 className="text-base font-semibold text-slate-900">Cohort Risk Distribution</h2>
              <button
                type="button"
                onClick={() => onNavigate('eda')}
                className="text-xs font-medium text-sky-700 hover:underline whitespace-nowrap shrink-0"
              >
                Explore 12 Charts
              </button>
            </div>
            <p className="text-xs text-slate-500 mb-5">
              Distribution of Low, Moderate, and High Risk labels across all stored records.
            </p>

            <div className="space-y-4">
              {riskBreakdown.map((item) => (
                <div key={item.label} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-slate-800">{item.label}</span>
                    <span className="font-mono tabular-nums text-xs text-slate-600">
                      {item.count} patients ({item.pct}%)
                    </span>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${item.barClass}`}
                      style={{ width: `${Math.min(100, Number(item.pct))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-200 grid grid-cols-2 gap-3 text-xs">
            <button
              type="button"
              onClick={() => onNavigate('preprocessing')}
              className="text-left p-2.5 rounded-md border border-slate-200 hover:bg-slate-50 transition-colors"
            >
              <div className="font-semibold text-slate-800">Preprocessing Studio</div>
              <div className="text-slate-500 mt-0.5">Upload CSV, impute, scale</div>
            </button>
            <button
              type="button"
              onClick={() => onNavigate('about')}
              className="text-left p-2.5 rounded-md border border-slate-200 hover:bg-slate-50 transition-colors"
            >
              <div className="font-semibold text-slate-800">Documentation &amp; Python</div>
              <div className="text-slate-500 mt-0.5">16 sections + Streamlit code</div>
            </button>
          </div>
        </div>
      </div>

      {/* Recent Assessments Table */}
      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Recent Risk Predictions</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              De-identified assessment log persisted in local SQLite.
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('reports')}
            className="text-xs font-medium text-sky-700 hover:underline whitespace-nowrap shrink-0"
          >
            View All Reports
          </button>
        </div>

        {overview.recentReports.length === 0 ? (
          <div className="p-8 text-center">
            <FileText className="w-8 h-8 text-slate-300 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-700">No saved risk predictions yet</p>
            <p className="text-xs text-slate-500 mt-1 mb-4">
              Run a patient risk evaluation to generate calibrated class probabilities and a
              downloadable report.
            </p>
            <button
              type="button"
              onClick={() => onNavigate('predict')}
              className="px-4 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 transition-colors"
            >
              Create First Prediction
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-xs font-medium text-slate-600">
                  <th className="py-2.5 px-5">Patient ID</th>
                  <th className="py-2.5 px-4">Predicted Category</th>
                  <th className="py-2.5 px-4">Model Used</th>
                  <th className="py-2.5 px-4 text-right">BMI</th>
                  <th className="py-2.5 px-4 text-right">Blood Pressure</th>
                  <th className="py-2.5 px-5 text-right">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {overview.recentReports.map((rep) => (
                  <tr key={rep.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-5 font-mono text-xs font-medium text-slate-900">
                      {rep.patient_id}
                    </td>
                    <td className={`py-2.5 px-4 text-xs ${riskTextColor(rep.predicted_risk)}`}>
                      {rep.predicted_risk}
                    </td>
                    <td className="py-2.5 px-4 text-xs text-slate-600">{rep.model_name}</td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                      {rep.patient_params.bmi.toFixed(1)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                      {rep.patient_params.systolic_bp}/{rep.patient_params.diastolic_bp}
                    </td>
                    <td className="py-2.5 px-5 text-right font-mono tabular-nums text-xs text-slate-500">
                      {new Date(rep.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
