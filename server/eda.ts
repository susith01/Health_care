import type { BoxPlotStat, EdaData, HistogramBin, PatientRecord, RiskCategory } from '../src/types.ts';
import { calcQuantile, ENCODING_MAPS, RISK_CLASSES } from './preprocessing.ts';

function buildHistogram(
  records: PatientRecord[],
  getter: (r: PatientRecord) => number,
  bins: Array<{ label: string; min: number; max: number }>
): HistogramBin[] {
  return bins.map((b) => {
    let lowRisk = 0;
    let moderateRisk = 0;
    let highRisk = 0;
    for (const r of records) {
      const val = getter(r);
      if (val >= b.min && val < b.max) {
        if (r.risk_category === 'Low Risk') lowRisk++;
        else if (r.risk_category === 'Moderate Risk') moderateRisk++;
        else highRisk++;
      }
    }
    return {
      range: b.label,
      count: lowRisk + moderateRisk + highRisk,
      lowRisk,
      moderateRisk,
      highRisk,
    };
  });
}

function pearsonCorrelation(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  return den > 1e-9 ? Math.round((num / den) * 100) / 100 : 0;
}

export function computeEdaAnalytics(
  records: PatientRecord[],
  datasetName: string,
  isSynthetic: boolean
): EdaData {
  const totalRecords = records.length;

  // 1. Age Distribution Histogram
  const ageHistogram = buildHistogram(records, (r) => r.age, [
    { label: '20–29', min: 0, max: 30 },
    { label: '30–39', min: 30, max: 40 },
    { label: '40–49', min: 40, max: 50 },
    { label: '50–59', min: 50, max: 60 },
    { label: '60–69', min: 60, max: 70 },
    { label: '70–79', min: 70, max: 80 },
    { label: '80+', min: 80, max: 150 },
  ]);

  // 2. Gender Distribution Pie
  const genderCounts: Record<string, number> = { Female: 0, Male: 0, Other: 0 };
  for (const r of records) {
    genderCounts[r.gender] = (genderCounts[r.gender] ?? 0) + 1;
  }
  const genderDistribution = Object.entries(genderCounts)
    .filter(([, count]) => count > 0)
    .map(([name, value]) => ({
      name,
      value,
      percentage: totalRecords > 0 ? Math.round((value / totalRecords) * 1000) / 10 : 0,
    }));

  // 3. BMI Distribution Histogram
  const bmiHistogram = buildHistogram(records, (r) => r.bmi, [
    { label: '<18.5', min: 0, max: 18.5 },
    { label: '18.5–22.9', min: 18.5, max: 23 },
    { label: '23–24.9', min: 23, max: 25 },
    { label: '25–29.9', min: 25, max: 30 },
    { label: '30–34.9', min: 30, max: 35 },
    { label: '35+', min: 35, max: 100 },
  ]);

  // 4. Blood Pressure Analysis by Risk Category (Grouped Bar Chart)
  const bloodPressureByRisk = RISK_CLASSES.map((risk) => {
    const group = records.filter((r) => r.risk_category === risk);
    const avgSystolic =
      group.length > 0
        ? Math.round((group.reduce((s, r) => s + r.systolic_bp, 0) / group.length) * 10) / 10
        : 0;
    const avgDiastolic =
      group.length > 0
        ? Math.round((group.reduce((s, r) => s + r.diastolic_bp, 0) / group.length) * 10) / 10
        : 0;
    return { risk, avgSystolic, avgDiastolic, count: group.length };
  });

  // 5. Glucose Level Distribution Histogram
  const glucoseHistogram = buildHistogram(records, (r) => r.glucose_level, [
    { label: '<90', min: 0, max: 90 },
    { label: '90–109', min: 90, max: 110 },
    { label: '110–125', min: 110, max: 126 },
    { label: '126–159', min: 126, max: 160 },
    { label: '160–199', min: 160, max: 200 },
    { label: '200+', min: 200, max: 600 },
  ]);

  // 6. Cholesterol Analysis Box Plot by Risk Category
  const cholesterolBoxPlot: BoxPlotStat[] = RISK_CLASSES.map((category: RiskCategory) => {
    const vals = records
      .filter((r) => r.risk_category === category)
      .map((r) => r.cholesterol_level)
      .sort((a, b) => a - b);
    if (vals.length === 0) {
      return { category, min: 0, q1: 0, median: 0, q3: 0, max: 0, mean: 0, count: 0 };
    }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return {
      category,
      min: Math.round(vals[0] * 10) / 10,
      q1: Math.round(calcQuantile(vals, 0.25) * 10) / 10,
      median: Math.round(calcQuantile(vals, 0.5) * 10) / 10,
      q3: Math.round(calcQuantile(vals, 0.75) * 10) / 10,
      max: Math.round(vals[vals.length - 1] * 10) / 10,
      mean: Math.round(mean * 10) / 10,
      count: vals.length,
    };
  });

  // 7. Risk Category Distribution Pie
  const riskDistribution = RISK_CLASSES.map((name) => {
    const value = records.filter((r) => r.risk_category === name).length;
    return {
      name,
      value,
      percentage: totalRecords > 0 ? Math.round((value / totalRecords) * 1000) / 10 : 0,
    };
  });

  // 8. Correlation Heatmap (10 Clinical Variables + Risk Target)
  const corrDefs: Array<{ key: string; label: string; fn: (r: PatientRecord) => number }> = [
    { key: 'age', label: 'Age', fn: (r) => r.age },
    { key: 'bmi', label: 'BMI', fn: (r) => r.bmi },
    { key: 'systolic_bp', label: 'Sys BP', fn: (r) => r.systolic_bp },
    { key: 'diastolic_bp', label: 'Dia BP', fn: (r) => r.diastolic_bp },
    { key: 'glucose_level', label: 'Glucose', fn: (r) => r.glucose_level },
    { key: 'cholesterol_level', label: 'Chol.', fn: (r) => r.cholesterol_level },
    { key: 'heart_rate', label: 'HR', fn: (r) => r.heart_rate },
    {
      key: 'smoking_status',
      label: 'Smoking',
      fn: (r) => ENCODING_MAPS.smoking_status[r.smoking_status] ?? 0,
    },
    {
      key: 'physical_activity',
      label: 'Inactivity',
      fn: (r) => ENCODING_MAPS.physical_activity[r.physical_activity] ?? 1,
    },
    {
      key: 'risk_category',
      label: 'Risk Tier',
      fn: (r) => ENCODING_MAPS.risk_category[r.risk_category] ?? 0,
    },
  ];

  const vectors = corrDefs.map((def) => records.map(def.fn));
  const matrix: number[][] = corrDefs.map((_, i) =>
    corrDefs.map((_, j) => (i === j ? 1 : pearsonCorrelation(vectors[i], vectors[j])))
  );

  return {
    datasetName,
    isSynthetic,
    totalRecords,
    ageHistogram,
    genderDistribution,
    bmiHistogram,
    bloodPressureByRisk,
    glucoseHistogram,
    cholesterolBoxPlot,
    riskDistribution,
    correlationMatrix: {
      features: corrDefs.map((d) => d.key),
      labels: corrDefs.map((d) => d.label),
      matrix,
    },
  };
}
