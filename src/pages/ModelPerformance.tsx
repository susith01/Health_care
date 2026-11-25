import React, { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ConfusionMatrixHeatmap, RocComparisonChart } from '../components/Charts';
import { api } from '../lib/api';
import type { ModelKey, TrainedModelsBundle } from '../types';

const MODEL_ORDER: ModelKey[] = [
  'logistic_regression',
  'decision_tree',
  'random_forest',
  'gradient_boosting',
];

export function ModelPerformance() {
  const [bundle, setBundle] = useState<TrainedModelsBundle | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelKey>('gradient_boosting');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getModels()
      .then((data) => {
        setBundle(data);
        setSelectedModel(data.bestModelKey);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load evaluation metrics.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="p-8 text-sm text-slate-500">Loading model performance evaluation...</div>;
  }

  if (error || !bundle) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        {error || 'Model evaluation data unavailable.'}
      </div>
    );
  }

  const comparisonBarData = MODEL_ORDER.map((k) => {
    const m = bundle.models[k];
    return {
      name: m.shortName,
      Accuracy: Math.round(m.accuracy * 1000) / 10,
      Precision: Math.round(m.precisionMacro * 1000) / 10,
      Recall: Math.round(m.recallMacro * 1000) / 10,
      'F1-Score': Math.round(m.f1Macro * 1000) / 10,
    };
  });

  const activeEval = bundle.models[selectedModel];
  const featureImpData = activeEval.featureImportance.map((fi) => ({
    label: fi.label,
    importancePct: Math.round(fi.importance * 1000) / 10,
  }));

  return (
    <div className="space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          Model Performance Evaluation (Charts 9–12)
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Actual hold-out test evaluation metrics across all four supervised algorithms (Test n ={' '}
          <span className="font-mono tabular-nums">{activeEval.testSize}</span> · Stratified Split{' '}
          {Math.round((1 - bundle.config.testSize) * 100)}/{Math.round(bundle.config.testSize * 100)}
          ).
        </p>
      </div>

      {/* Model Comparison Table */}
      <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-900">
            Four-Algorithm Evaluation Comparison Table
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                <th className="py-2.5 px-5">Algorithm</th>
                <th className="py-2.5 px-3 text-right">Accuracy</th>
                <th className="py-2.5 px-3 text-right">Macro Precision</th>
                <th className="py-2.5 px-3 text-right">Macro Recall</th>
                <th className="py-2.5 px-3 text-right">Macro F1</th>
                <th className="py-2.5 px-3 text-right">Weighted F1</th>
                <th className="py-2.5 px-3 text-right">Macro AUC</th>
                <th className="py-2.5 px-5 text-right">Fit Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
              {MODEL_ORDER.map((k) => {
                const m = bundle.models[k];
                const isBest = k === bundle.bestModelKey;
                return (
                  <tr
                    key={k}
                    className={isBest ? 'bg-sky-50/40' : 'hover:bg-slate-50/80'}
                  >
                    <td className="py-3 px-5 font-sans font-semibold text-slate-900">
                      {m.name}
                      {isBest && (
                        <span className="ml-2 text-[11px] font-medium text-teal-700">
                          · Top Performer
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right font-semibold text-slate-900">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {(m.precisionMacro * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {(m.recallMacro * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-semibold text-sky-800">
                      {(m.f1Macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {(m.f1Weighted * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-semibold text-teal-700">
                      {m.macroAuc.toFixed(4)}
                    </td>
                    <td className="py-3 px-5 text-right text-slate-500">
                      {m.trainingTimeMs} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Chart 9 & Chart 11 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            09. Algorithm Accuracy &amp; Metric Comparison — Bar Chart
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Direct comparison of Accuracy, Macro Precision, Macro Recall, and Macro F1-Score (%)
            computed on identical test samples.
          </p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonBarData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" tick={{ fontSize: 11.5, fill: '#334155' }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11, fill: '#64748B' }} />
                <Tooltip
                  formatter={(v: unknown) => [`${v}%`, '']}
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={34} wrapperStyle={{ fontSize: '11.5px' }} />
                <Bar dataKey="Accuracy" fill="#0F5FA8" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Precision" fill="#38BDF8" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Recall" fill="#0D9488" radius={[2, 2, 0, 0]} />
                <Bar dataKey="F1-Score" fill="#1E293B" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            11. Multi-Model ROC Curve Comparison (Macro OvR)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            One-vs-Rest macro-averaged Receiver Operating Characteristic curves across all three
            risk classes, with trapezoidal Area Under Curve (AUC).
          </p>
          <RocComparisonChart models={bundle.models} />
        </section>
      </div>

      {/* Model Switcher for Charts 10 & 12 */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="text-sm font-semibold text-slate-900">
          Inspect Per-Model Confusion Matrix &amp; Feature Importance
        </div>
        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-md">
          {MODEL_ORDER.map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setSelectedModel(k)}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                selectedModel === k
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {bundle.models[k].shortName}
            </button>
          ))}
        </div>
      </div>

      {/* Chart 10 & Chart 12 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-5 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            10. Confusion Matrix Heatmap — {activeEval.name}
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Cell values show exact sample counts and row recall percentages for each target risk
            tier.
          </p>
          <ConfusionMatrixHeatmap
            matrix={activeEval.confusionMatrix}
            labels={activeEval.classLabels}
            modelName={activeEval.name}
          />
        </section>

        <section className="lg:col-span-7 rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">
            12. Feature Importance — Horizontal Bar Chart ({activeEval.name})
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 mb-4">
            Normalized relative contribution (%) of each clinical parameter learned by the selected
            model on the training split.
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImpData}
                layout="vertical"
                margin={{ top: 4, right: 24, left: 16, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" horizontal={false} />
                <XAxis
                  type="number"
                  unit="%"
                  tick={{ fontSize: 11, fill: '#64748B' }}
                />
                <YAxis
                  dataKey="label"
                  type="category"
                  width={150}
                  tick={{ fontSize: 11, fill: '#334155' }}
                />
                <Tooltip
                  formatter={(v: unknown) => [`${v}%`, 'Relative Importance']}
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#CBD5E1', fontSize: '12px' }}
                />
                <Bar dataKey="importancePct" name="Importance (%)" fill="#0D9488" radius={[0, 3, 3, 0]} maxBarSize={18} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  );
}
