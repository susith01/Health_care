import React, { useEffect, useState } from 'react';
import { Play } from 'lucide-react';
import { ConfusionMatrixHeatmap } from '../components/Charts';
import { api } from '../lib/api';
import type { ModelKey, PreprocessingConfig, TrainedModelsBundle } from '../types';

const MODEL_ORDER: ModelKey[] = [
  'logistic_regression',
  'decision_tree',
  'random_forest',
  'gradient_boosting',
];

export function MachineLearningModels() {
  const [bundle, setBundle] = useState<TrainedModelsBundle | null>(null);
  const [selectedKey, setSelectedKey] = useState<ModelKey>('gradient_boosting');
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
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getModels()
      .then((data) => {
        setBundle(data);
        setSelectedKey(data.bestModelKey);
        setConfig(data.config);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load models.'))
      .finally(() => setLoading(false));
  }, []);

  const handleRetrain = async () => {
    setTraining(true);
    setError(null);
    try {
      const updated = await api.trainModels(config);
      setBundle(updated);
      setSelectedKey(updated.bestModelKey);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed.');
    } finally {
      setTraining(false);
    }
  };

  if (loading || !bundle) {
    return <div className="p-8 text-sm text-slate-500">Loading trained machine learning models...</div>;
  }

  const activeModel = bundle.models[selectedKey];

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Machine Learning Models
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Four supervised multi-class classifiers trained on{' '}
            <span className="font-medium text-slate-900">{bundle.datasetName}</span> using a
            leak-free stratified train/test pipeline.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <select
            aria-label="Test split ratio"
            value={String(config.testSize)}
            onChange={(e) => setConfig((c) => ({ ...c, testSize: Number(e.target.value) }))}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-mono"
          >
            <option value="0.15">85/15 Split</option>
            <option value="0.2">80/20 Split</option>
            <option value="0.25">75/25 Split</option>
            <option value="0.3">70/30 Split</option>
          </select>
          <select
            aria-label="Feature scaling method"
            value={config.scalingMethod}
            onChange={(e) =>
              setConfig((c) => ({ ...c, scalingMethod: e.target.value as 'standard' | 'minmax' }))
            }
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-xs"
          >
            <option value="standard">StandardScaler</option>
            <option value="minmax">MinMax [0, 1]</option>
          </select>
          <button
            type="button"
            onClick={handleRetrain}
            disabled={training}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            {training ? 'Training 4 Models...' : 'Train All 4 Models'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* 4 Model Selector Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {MODEL_ORDER.map((key) => {
          const m = bundle.models[key];
          const isSelected = key === selectedKey;
          const isBest = key === bundle.bestModelKey;
          return (
            <button
              key={key}
              type="button"
              onClick={() => setSelectedKey(key)}
              className={`text-left rounded-lg border p-4 transition-colors ${
                isSelected
                  ? 'border-sky-700 bg-sky-50/40'
                  : 'border-slate-200 bg-white hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold text-slate-900">{m.name}</span>
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {isBest ? (
                  <span className="text-teal-700 font-medium">Highest Macro F1</span>
                ) : (
                  <span>Evaluated on test split</span>
                )}
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 pt-3 border-t border-slate-200/80 text-xs font-mono tabular-nums">
                <div>
                  <div className="text-[11px] font-sans text-slate-500">Accuracy</div>
                  <div className="font-semibold text-slate-900">
                    {(m.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-sans text-slate-500">Macro F1</div>
                  <div className="font-semibold text-slate-900">
                    {(m.f1Macro * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-sans text-slate-500">AUC</div>
                  <div className="font-semibold text-slate-900">{m.macroAuc.toFixed(3)}</div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Model Detailed Diagnostic Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 rounded-lg border border-slate-200 bg-white p-6 space-y-5">
          <div>
            <div className="text-xs text-slate-500">
              Trained on {activeModel.trainSize} samples · Evaluated on {activeModel.testSize}{' '}
              hold-out samples · Fit time {activeModel.trainingTimeMs} ms
            </div>
            <h2 className="mt-1 text-lg font-bold text-slate-900">{activeModel.name}</h2>
            <p className="mt-1 text-sm text-slate-600 leading-relaxed">{activeModel.description}</p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="rounded-md bg-slate-50 p-3">
              <div className="text-xs text-slate-500">Test Accuracy</div>
              <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
                {(activeModel.accuracy * 100).toFixed(2)}%
              </div>
            </div>
            <div className="rounded-md bg-slate-50 p-3">
              <div className="text-xs text-slate-500">Macro Precision</div>
              <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
                {(activeModel.precisionMacro * 100).toFixed(2)}%
              </div>
            </div>
            <div className="rounded-md bg-slate-50 p-3">
              <div className="text-xs text-slate-500">Macro Recall</div>
              <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
                {(activeModel.recallMacro * 100).toFixed(2)}%
              </div>
            </div>
            <div className="rounded-md bg-slate-50 p-3">
              <div className="text-xs text-slate-500">Macro ROC-AUC</div>
              <div className="mt-1 text-lg font-bold font-mono tabular-nums text-teal-700">
                {activeModel.macroAuc.toFixed(4)}
              </div>
            </div>
          </div>

          {/* Per-class Classification Report */}
          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-2.5">
              Per-Class Classification Report (Hold-Out Test Set)
            </h3>
            <div className="overflow-x-auto border border-slate-200 rounded-md">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                    <th className="py-2 px-3">Risk Class</th>
                    <th className="py-2 px-3 text-right">Precision</th>
                    <th className="py-2 px-3 text-right">Recall</th>
                    <th className="py-2 px-3 text-right">F1-Score</th>
                    <th className="py-2 px-3 text-right">OvR AUC</th>
                    <th className="py-2 px-3 text-right">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                  {activeModel.classLabels.map((cls) => {
                    const stats = activeModel.perClassMetrics[cls];
                    return (
                      <tr key={cls}>
                        <td className="py-2 px-3 font-sans font-medium text-slate-800">{cls}</td>
                        <td className="py-2 px-3 text-right">{(stats.precision * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-right">{(stats.recall * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-right">{(stats.f1 * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-right">{stats.auc.toFixed(3)}</td>
                        <td className="py-2 px-3 text-right text-slate-600">{stats.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="pt-2 border-t border-slate-200">
            <h3 className="text-xs font-semibold text-slate-700 mb-2">Model Hyperparameters</h3>
            <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-slate-600">
              {Object.entries(activeModel.hyperparameters).map(([k, v]) => (
                <span key={k}>
                  <span className="font-mono text-slate-500">{k}</span>:{' '}
                  <strong className="font-mono text-slate-900">{String(v)}</strong>
                </span>
              ))}
            </div>
          </div>
        </section>

        <section className="lg:col-span-5 rounded-lg border border-slate-200 bg-white p-6 space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Confusion Matrix — {activeModel.name}
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Diagonal counts indicate exact classifications; off-diagonal counts highlight
              boundary misclassifications.
            </p>
          </div>
          <ConfusionMatrixHeatmap
            matrix={activeModel.confusionMatrix}
            labels={activeModel.classLabels}
            modelName={activeModel.name}
          />
          <div className="pt-4 border-t border-slate-200 space-y-2 text-xs text-slate-600">
            <div className="font-semibold text-slate-800">Leak-Free Pipeline Guarantee</div>
            <p className="leading-relaxed">
              Stratified partitioning occurs before computing numeric medians/means, categorical
              modes, IQR bounds, or feature scalers. All preprocessing parameters are learned
              exclusively on the {activeModel.trainSize}-record training set and applied unchanged to
              the {activeModel.testSize}-record test set.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
