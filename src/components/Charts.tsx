import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { BoxPlotStat, ModelEvaluation, ModelKey, RiskCategory } from '../types';

/**
 * Chart 6: Custom SVG Box Plot for Cholesterol Distribution by Risk Category
 * Renders genuine Min, Q1, Median, Q3, Max, and Mean markers from actual dataset quartiles.
 */
export function CholesterolBoxPlotChart({ data }: { data: BoxPlotStat[] }) {
  const valid = data.filter((d) => d.count > 0);
  if (valid.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-sm text-slate-500">
        No cholesterol records available for box plot rendering.
      </div>
    );
  }

  const globalMin = Math.floor(Math.min(...valid.map((d) => d.min)) / 20) * 20;
  const globalMax = Math.ceil(Math.max(...valid.map((d) => d.max)) / 20) * 20;
  const span = Math.max(40, globalMax - globalMin);

  const width = 520;
  const height = 270;
  const pad = { top: 24, right: 24, bottom: 44, left: 56 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const yScale = (val: number) => pad.top + plotH - ((val - globalMin) / span) * plotH;
  const ticks = 5;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) =>
    Math.round(globalMin + (span * i) / ticks)
  );

  const colors: Record<RiskCategory, { fill: string; stroke: string }> = {
    'Low Risk': { fill: '#CCFBF1', stroke: '#0D9488' },
    'Moderate Risk': { fill: '#FEF3C7', stroke: '#D97706' },
    'High Risk': { fill: '#FEE2E2', stroke: '#DC2626' },
  };

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-64"
        role="img"
        aria-label="Cholesterol Box Plot by Risk Category"
      >
        {yTicks.map((t) => {
          const y = yScale(t);
          return (
            <g key={t}>
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={y}
                y2={y}
                stroke="#E2E8F0"
                strokeDasharray="3 3"
              />
              <text
                x={pad.left - 8}
                y={y + 4}
                textAnchor="end"
                fontSize="11"
                fill="#64748B"
                fontFamily="IBM Plex Mono, monospace"
              >
                {t}
              </text>
            </g>
          );
        })}

        <text
          x={14}
          y={pad.top + plotH / 2}
          transform={`rotate(-90, 14, ${pad.top + plotH / 2})`}
          textAnchor="middle"
          fontSize="11"
          fill="#475569"
        >
          Cholesterol (mg/dL)
        </text>

        {valid.map((item, idx) => {
          const colW = plotW / valid.length;
          const cx = pad.left + colW * idx + colW / 2;
          const boxW = Math.min(64, colW * 0.45);
          const yMin = yScale(item.min);
          const yQ1 = yScale(item.q1);
          const yMed = yScale(item.median);
          const yQ3 = yScale(item.q3);
          const yMax = yScale(item.max);
          const yMean = yScale(item.mean);
          const palette = colors[item.category];

          return (
            <g key={item.category}>
              {/* Whisker line */}
              <line
                x1={cx}
                x2={cx}
                y1={yMax}
                y2={yMin}
                stroke={palette.stroke}
                strokeWidth="1.5"
              />
              {/* Min & Max caps */}
              <line
                x1={cx - boxW / 3}
                x2={cx + boxW / 3}
                y1={yMax}
                y2={yMax}
                stroke={palette.stroke}
                strokeWidth="1.5"
              />
              <line
                x1={cx - boxW / 3}
                x2={cx + boxW / 3}
                y1={yMin}
                y2={yMin}
                stroke={palette.stroke}
                strokeWidth="1.5"
              />
              {/* IQR Box */}
              <rect
                x={cx - boxW / 2}
                y={yQ3}
                width={boxW}
                height={Math.max(2, yQ1 - yQ3)}
                fill={palette.fill}
                stroke={palette.stroke}
                strokeWidth="1.8"
                rx="2"
              />
              {/* Median line */}
              <line
                x1={cx - boxW / 2}
                x2={cx + boxW / 2}
                y1={yMed}
                y2={yMed}
                stroke="#0F172A"
                strokeWidth="2.2"
              />
              {/* Mean diamond */}
              <circle cx={cx} cy={yMean} r="3.5" fill="#0F5FA8" />
              {/* X-axis label */}
              <text
                x={cx}
                y={height - 20}
                textAnchor="middle"
                fontSize="11.5"
                fontWeight="600"
                fill="#1E293B"
              >
                {item.category}
              </text>
              <text
                x={cx}
                y={height - 6}
                textAnchor="middle"
                fontSize="10"
                fill="#64748B"
                fontFamily="IBM Plex Mono, monospace"
              >
                Med {item.median} · IQR {Math.round((item.q3 - item.q1) * 10) / 10}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-slate-600">
        <span>Box = Q1 to Q3 (IQR)</span>
        <span aria-hidden="true">·</span>
        <span>Dark bar = Median</span>
        <span aria-hidden="true">·</span>
        <span>Blue marker = Mean</span>
        <span aria-hidden="true">·</span>
        <span>Whiskers = Min / Max</span>
      </div>
    </div>
  );
}

/**
 * Chart 8: Interactive Pearson Correlation Heatmap
 */
export function CorrelationHeatmap({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  const cellBg = (val: number): string => {
    const abs = Math.min(1, Math.abs(val));
    if (val >= 0) {
      // Medical blue scale for positive correlation
      const alpha = 0.08 + abs * 0.78;
      return `rgba(15, 95, 168, ${alpha.toFixed(2)})`;
    }
    // Teal/coral scale for negative correlation
    const alpha = 0.08 + abs * 0.72;
    return `rgba(13, 148, 136, ${alpha.toFixed(2)})`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs font-mono tabular-nums">
        <thead>
          <tr>
            <th className="p-1.5 text-left font-sans font-medium text-slate-500">Feature</th>
            {labels.map((lbl) => (
              <th
                key={lbl}
                className="p-1.5 text-center font-sans font-medium text-slate-600 whitespace-nowrap"
              >
                {lbl}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((rowLabel, i) => (
            <tr key={rowLabel}>
              <td className="p-1.5 font-sans font-medium text-slate-700 whitespace-nowrap">
                {rowLabel}
              </td>
              {matrix[i].map((val, j) => {
                const darkText = Math.abs(val) < 0.55;
                return (
                  <td
                    key={`${i}-${j}`}
                    style={{ backgroundColor: cellBg(val) }}
                    className={`p-1.5 text-center border border-white text-[11px] ${
                      darkText ? 'text-slate-900' : 'text-white font-medium'
                    }`}
                    title={`${rowLabel} vs ${labels[j]}: r = ${val.toFixed(2)}`}
                  >
                    {val.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
        <span>Negative correlation (Teal) · Zero (Light) · Positive correlation (Medical Blue)</span>
        <span className="font-mono">r ∈ [-1.00, +1.00]</span>
      </div>
    </div>
  );
}

/**
 * Chart 10: Confusion Matrix Heatmap (3x3)
 */
export function ConfusionMatrixHeatmap({
  matrix,
  labels,
  modelName,
}: {
  matrix: number[][];
  labels: RiskCategory[];
  modelName: string;
}) {
  const maxVal = Math.max(1, ...matrix.flat());
  const total = matrix.flat().reduce((a, b) => a + b, 0);

  return (
    <div className="w-full">
      <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
        <span>Rows: Actual Risk Class · Columns: Predicted Risk Class</span>
        <span className="font-mono tabular-nums">Test n = {total}</span>
      </div>
      <div className="overflow-x-auto">
        <table
          className="w-full border-collapse text-sm"
          aria-label={`Confusion matrix for ${modelName}`}
        >
          <thead>
            <tr>
              <th className="p-2 text-left text-xs font-medium text-slate-500 border-b border-slate-200">
                Actual \ Predicted
              </th>
              {labels.map((lbl) => (
                <th
                  key={lbl}
                  className="p-2 text-center text-xs font-semibold text-slate-700 border-b border-slate-200 whitespace-nowrap"
                >
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => {
              const rowSum = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={labels[rIdx]}>
                  <td className="p-2 text-xs font-semibold text-slate-700 whitespace-nowrap border-r border-slate-200">
                    {labels[rIdx]}
                  </td>
                  {row.map((cell, cIdx) => {
                    const isDiag = rIdx === cIdx;
                    const intensity = cell / maxVal;
                    const bg = isDiag
                      ? `rgba(13, 148, 136, ${(0.12 + intensity * 0.72).toFixed(2)})`
                      : cell > 0
                        ? `rgba(220, 38, 38, ${(0.1 + intensity * 0.5).toFixed(2)})`
                        : '#F8FAFC';
                    const textDark = isDiag ? intensity < 0.55 : intensity < 0.65;
                    const pct = rowSum > 0 ? ((cell / rowSum) * 100).toFixed(0) : '0';

                    return (
                      <td
                        key={`${rIdx}-${cIdx}`}
                        style={{ backgroundColor: bg }}
                        className={`p-3 text-center border border-white font-mono tabular-nums ${
                          textDark ? 'text-slate-900' : 'text-white'
                        }`}
                      >
                        <div className="text-base font-semibold">{cell}</div>
                        <div className="text-[11px] opacity-85">{pct}%</div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Chart 11: Multi-Model Macro ROC Curve Comparison
 */
export function RocComparisonChart({
  models,
}: {
  models: Record<ModelKey, ModelEvaluation>;
}) {
  const keys: ModelKey[] = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
  ];

  // All macroRocCurve arrays share the same 26-point uniform FPR grid [0.00 .. 1.00]
  const baseCurve = models.logistic_regression.macroRocCurve;
  const chartData = baseCurve.map((pt, idx) => ({
    fpr: pt.fpr,
    baseline: pt.fpr,
    logistic_regression: models.logistic_regression.macroRocCurve[idx]?.tpr ?? pt.fpr,
    decision_tree: models.decision_tree.macroRocCurve[idx]?.tpr ?? pt.fpr,
    random_forest: models.random_forest.macroRocCurve[idx]?.tpr ?? pt.fpr,
    gradient_boosting: models.gradient_boosting.macroRocCurve[idx]?.tpr ?? pt.fpr,
  }));

  const palette: Record<ModelKey, string> = {
    logistic_regression: '#0F5FA8',
    decision_tree: '#D97706',
    random_forest: '#0D9488',
    gradient_boosting: '#1E293B',
  };

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 6 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 11, fill: '#64748B' }}
            label={{
              value: 'False Positive Rate (1 − Specificity)',
              position: 'insideBottom',
              offset: -12,
              fontSize: 11,
              fill: '#475569',
            }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 11, fill: '#64748B' }}
            label={{
              value: 'True Positive Rate (Sensitivity)',
              angle: -90,
              position: 'insideLeft',
              offset: 12,
              fontSize: 11,
              fill: '#475569',
            }}
          />
          <Tooltip
            formatter={(val: unknown, name: unknown) => [
              typeof val === 'number' ? val.toFixed(3) : String(val),
              String(name),
            ]}
            labelFormatter={(label) => `FPR: ${Number(label).toFixed(2)}`}
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#CBD5E1',
              fontSize: '12px',
            }}
          />
          <Legend verticalAlign="top" height={34} wrapperStyle={{ fontSize: '11.5px' }} />
          <Line
            type="linear"
            dataKey="baseline"
            name="Random Classifier (AUC = 0.50)"
            stroke="#94A3B8"
            strokeDasharray="5 5"
            dot={false}
            strokeWidth={1.5}
          />
          {keys.map((k) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              name={`${models[k].shortName} (AUC = ${models[k].macroAuc.toFixed(3)})`}
              stroke={palette[k]}
              strokeWidth={2.2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
