import React, { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CholesterolBoxPlotChart, CorrelationHeatmap } from '../components/Charts';
import { api } from '../lib/api';
import type { EdaData } from '../types';

const GENDER_COLORS = ['#0F5FA8', '#0D9488', '#64748B'];
const RISK_COLORS: Record<string, string> = {
  'Low Risk': '#0D9488',
  'Moderate Risk': '#D97706',
  'High Risk': '#DC2626',
};

export function ExploratoryAnalysis() {
  const [eda, setEda] = useState<EdaData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getEda()
      .then(setEda)
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to compute EDA charts.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="p-8 text-sm text-slate-500">Computing dataset visualizations...</div>;
  }

  if (error || !eda) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        {error || 'Exploratory Data Analysis unavailable.'}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          Exploratory Data Analysis (Charts 1–8)
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          All visualizations are generated from actual values in{' '}
          <span className="font-medium text-slate-900">{eda.datasetName}</span> (n ={' '}
          <span className="font-mono tabular-nums">{eda.totalRecords}</span>). Charts 9–12 (Model
          Comparison, Confusion Matrix, ROC Curves, and Feature Importance) appear on the Model
          Performance page.
        </p>
      </div>

      {/* Row 1: Chart 1 (Age Histogram) & Chart 2 (Gender Pie) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            01. Age Distribution — Stacked Histogram
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Patient age brackets segmented by risk tier, showing elevated risk prevalence in older
            cohorts.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eda.ageHistogram} margin={{ top: 8, right: 12, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="range"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  label={{ value: 'Age Group (years)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#64748B' }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#64748B' }}
                  label={{ value: 'Patients', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#64748B' }}
                />
                <Tooltip contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }} />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11.5px' }} />
                <Bar dataKey="lowRisk" stackId="a" name="Low Risk" fill="#0D9488" />
                <Bar dataKey="moderateRisk" stackId="a" name="Moderate Risk" fill="#D97706" />
                <Bar dataKey="highRisk" stackId="a" name="High Risk" fill="#DC2626" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="lg:col-span-5 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            02. Gender Distribution — Pie Chart
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Demographic gender breakdown across the active cleaned cohort.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={eda.genderDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={48}
                  outerRadius={86}
                  paddingAngle={2}
                  label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(1)}%`}
                >
                  {eda.genderDistribution.map((entry, i) => (
                    <Cell key={entry.name} fill={GENDER_COLORS[i % GENDER_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: unknown) => [`${v} patients`, 'Count']}
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }}
                />
                <Legend verticalAlign="bottom" height={24} wrapperStyle={{ fontSize: '11.5px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Row 2: Chart 3 (BMI Histogram) & Chart 4 (Blood Pressure Grouped Bar) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            03. BMI Distribution — Histogram
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Frequency across WHO BMI bands; higher BMI ranges (≥30 kg/m²) align strongly with
            Moderate and High Risk tiers.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eda.bmiHistogram} margin={{ top: 8, right: 12, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="range"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  label={{ value: 'BMI Category (kg/m²)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#64748B' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748B' }} />
                <Tooltip contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }} />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11.5px' }} />
                <Bar dataKey="count" name="Total Patients" fill="#0F5FA8" radius={[3, 3, 0, 0]} maxBarSize={42} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            04. Blood Pressure Analysis by Risk Tier — Grouped Bar Chart
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Mean Systolic and Diastolic blood pressure (mmHg) demonstrating progressive elevation
            from Low Risk to High Risk.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eda.bloodPressureByRisk} margin={{ top: 8, right: 12, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="risk" tick={{ fontSize: 11.5, fill: '#334155' }} />
                <YAxis
                  domain={[0, 180]}
                  tick={{ fontSize: 11, fill: '#64748B' }}
                  label={{ value: 'mmHg', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#64748B' }}
                />
                <Tooltip
                  formatter={(v: unknown) => [`${v} mmHg`, '']}
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11.5px' }} />
                <Bar dataKey="avgSystolic" name="Avg Systolic BP" fill="#0F5FA8" radius={[3, 3, 0, 0]} maxBarSize={44} />
                <Bar dataKey="avgDiastolic" name="Avg Diastolic BP" fill="#0D9488" radius={[3, 3, 0, 0]} maxBarSize={44} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Row 3: Chart 5 (Glucose Histogram) & Chart 6 (Cholesterol Box Plot) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            05. Blood Glucose Level Distribution — Histogram
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Distribution of glucose readings (mg/dL) segmented across clinical glycemic ranges
            (normal &lt;100, borderline 100–125, elevated ≥126).
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eda.glucoseHistogram} margin={{ top: 8, right: 12, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="range"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  label={{ value: 'Blood Glucose (mg/dL)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#64748B' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748B' }} />
                <Tooltip contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }} />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11.5px' }} />
                <Bar dataKey="lowRisk" stackId="g" name="Low Risk" fill="#0D9488" />
                <Bar dataKey="moderateRisk" stackId="g" name="Moderate Risk" fill="#D97706" />
                <Bar dataKey="highRisk" stackId="g" name="High Risk" fill="#DC2626" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            06. Cholesterol Analysis by Risk Category — Box Plot
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-3">
            Quartile spread (Min, Q1, Median, Q3, Max, and Mean) of total cholesterol (mg/dL) across
            each risk classification.
          </p>
          <CholesterolBoxPlotChart data={eda.cholesterolBoxPlot} />
        </section>
      </div>

      {/* Row 4: Chart 7 (Risk Pie) & Chart 8 (Correlation Heatmap) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-4 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            07. Risk Category Distribution — Pie Chart
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Class balance across the three supervised target categories used for stratified training.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={eda.riskDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={44}
                  outerRadius={82}
                  paddingAngle={2}
                  label={({ percent }) => `${((percent ?? 0) * 100).toFixed(1)}%`}
                >
                  {eda.riskDistribution.map((entry) => (
                    <Cell key={entry.name} fill={RISK_COLORS[entry.name] || '#0F5FA8'} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: unknown) => [`${v} patients`, 'Count']}
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }}
                />
                <Legend verticalAlign="bottom" height={24} wrapperStyle={{ fontSize: '11.5px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="lg:col-span-8 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            08. Multi-Factor Pearson Correlation Heatmap
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Pairwise linear correlations among clinical vitals, lifestyle factors, and encoded Risk
            Tier (0 = Low, 1 = Moderate, 2 = High).
          </p>
          <CorrelationHeatmap
            labels={eda.correlationMatrix.labels}
            matrix={eda.correlationMatrix.matrix}
          />
        </section>
      </div>
    </div>
  );
}
