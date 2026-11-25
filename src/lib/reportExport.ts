import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import type { SavedReport } from '../types';

function csvEscape(val: unknown): string {
  if (val === null || val === undefined) return '';
  let str = String(val);
  if (/^[=+\-@\t\r]/.test(str)) {
    str = `'${str}`;
  }
  if (/[",\n\r]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function downloadReportCsv(report: SavedReport): void {
  const rows: Array<[string, string]> = [
    ['Report Type', 'Multi-Factor Healthcare Risk Assessment (Educational Decision Support)'],
    ['Patient ID', report.patient_id],
    ['Date of Prediction', new Date(report.created_at).toLocaleString()],
    ['Machine Learning Model', report.model_name],
    ['Predicted Risk Category', report.predicted_risk],
    ['Low Risk Probability', `${(report.probabilities['Low Risk'] * 100).toFixed(1)}%`],
    ['Moderate Risk Probability', `${(report.probabilities['Moderate Risk'] * 100).toFixed(1)}%`],
    ['High Risk Probability', `${(report.probabilities['High Risk'] * 100).toFixed(1)}%`],
    ['Age (years)', String(report.patient_params.age)],
    ['Gender', report.patient_params.gender],
    ['Height (cm)', String(report.patient_params.height_cm)],
    ['Weight (kg)', String(report.patient_params.weight_kg)],
    ['BMI (kg/m2)', String(report.patient_params.bmi)],
    [
      'Blood Pressure (mmHg)',
      `${report.patient_params.systolic_bp}/${report.patient_params.diastolic_bp}`,
    ],
    ['Blood Glucose (mg/dL)', String(report.patient_params.glucose_level)],
    ['Cholesterol (mg/dL)', String(report.patient_params.cholesterol_level)],
    ['Heart Rate (bpm)', String(report.patient_params.heart_rate)],
    ['Smoking Status', report.patient_params.smoking_status],
    ['Physical Activity', report.patient_params.physical_activity],
    ['Family Medical History', report.patient_params.family_history],
    ['Existing Medical Conditions', report.patient_params.existing_conditions],
    [
      'Top Contributing Factors',
      report.contributing_factors
        .map((f) => `${f.label}: ${f.patientValue} (${f.status})`)
        .join(' | '),
    ],
    ['General Recommendations', report.recommendations.join(' | ')],
    [
      'Medical Disclaimer',
      'Educational decision-support output only. Not a medical diagnosis. Consult a licensed healthcare professional.',
    ],
  ];

  const csvContent = ['Field,Value', ...rows.map(([k, v]) => `${csvEscape(k)},${csvEscape(v)}`)].join(
    '\n'
  );
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `health_risk_report_${report.patient_id}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadReportPdf(report: SavedReport): void {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const margin = 42;
  let y = 46;

  // Header
  doc.setFillColor(15, 95, 168);
  doc.rect(0, 0, doc.internal.pageSize.getWidth(), 78, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.text('Multi-Factor Healthcare Risk Prediction System', margin, 34);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.text('Patient Risk Assessment Report — Educational Decision Support', margin, 54);

  y = 102;
  doc.setTextColor(15, 23, 42);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text(`Patient ID: ${report.patient_id}`, margin, y);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.setTextColor(71, 85, 105);
  doc.text(
    `Date: ${new Date(report.created_at).toLocaleString()}   |   Model: ${report.model_name}`,
    margin,
    y + 16
  );

  // Risk summary banner
  y += 32;
  const riskColor: [number, number, number] =
    report.predicted_risk === 'High Risk'
      ? [185, 28, 28]
      : report.predicted_risk === 'Moderate Risk'
        ? [180, 83, 9]
        : [21, 128, 61];
  doc.setDrawColor(...riskColor);
  doc.setFillColor(248, 250, 252);
  doc.roundedRect(margin, y, doc.internal.pageSize.getWidth() - margin * 2, 54, 4, 4, 'FD');
  doc.setTextColor(...riskColor);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.text(`Predicted Risk Category: ${report.predicted_risk}`, margin + 14, y + 22);
  doc.setTextColor(51, 65, 85);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.text(
    `Class Probabilities — Low Risk: ${(report.probabilities['Low Risk'] * 100).toFixed(1)}%   |   Moderate Risk: ${(report.probabilities['Moderate Risk'] * 100).toFixed(1)}%   |   High Risk: ${(report.probabilities['High Risk'] * 100).toFixed(1)}%`,
    margin + 14,
    y + 40
  );

  y += 72;
  doc.setTextColor(15, 23, 42);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('1. Patient Health Parameters (De-Identified)', margin, y);

  autoTable(doc, {
    startY: y + 8,
    margin: { left: margin, right: margin },
    head: [['Clinical Parameter', 'Recorded Value', 'Clinical Parameter', 'Recorded Value']],
    body: [
      ['Age', `${report.patient_params.age} years`, 'Gender', report.patient_params.gender],
      [
        'Height / Weight',
        `${report.patient_params.height_cm} cm / ${report.patient_params.weight_kg} kg`,
        'BMI',
        `${report.patient_params.bmi} kg/m²`,
      ],
      [
        'Blood Pressure',
        `${report.patient_params.systolic_bp}/${report.patient_params.diastolic_bp} mmHg`,
        'Heart Rate',
        `${report.patient_params.heart_rate} bpm`,
      ],
      [
        'Blood Glucose',
        `${report.patient_params.glucose_level} mg/dL`,
        'Total Cholesterol',
        `${report.patient_params.cholesterol_level} mg/dL`,
      ],
      [
        'Smoking Status',
        report.patient_params.smoking_status,
        'Physical Activity',
        report.patient_params.physical_activity,
      ],
      [
        'Family History',
        report.patient_params.family_history,
        'Existing Conditions',
        report.patient_params.existing_conditions,
      ],
    ],
    theme: 'grid',
    headStyles: { fillColor: [15, 95, 168], fontSize: 9 },
    styles: { fontSize: 8.8, cellPadding: 5 },
  });

  const afterParamsY =
    (doc as unknown as { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y + 140;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text('2. Top Contributing Factors', margin, afterParamsY + 22);

  autoTable(doc, {
    startY: afterParamsY + 30,
    margin: { left: margin, right: margin },
    head: [['Factor', 'Patient Value', 'Reference Range', 'Clinical Flag', 'Weight Score']],
    body: report.contributing_factors.map((f) => [
      f.label,
      f.patientValue,
      f.referenceRange,
      f.status,
      String(f.contributionScore),
    ]),
    theme: 'striped',
    headStyles: { fillColor: [13, 148, 136], fontSize: 9 },
    styles: { fontSize: 8.8, cellPadding: 4.5 },
  });

  const afterFactorsY =
    (doc as unknown as { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ??
    afterParamsY + 140;

  let recY = afterFactorsY + 24;
  if (recY > 700) {
    doc.addPage();
    recY = 50;
  }

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text('3. General Health Recommendations', margin, recY);
  recY += 16;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.2);
  doc.setTextColor(51, 65, 85);
  const maxWidth = doc.internal.pageSize.getWidth() - margin * 2;
  for (const rec of report.recommendations) {
    const lines = doc.splitTextToSize(`• ${rec}`, maxWidth);
    doc.text(lines, margin, recY);
    recY += lines.length * 13 + 4;
  }

  recY += 10;
  doc.setDrawColor(203, 213, 225);
  doc.line(margin, recY, doc.internal.pageSize.getWidth() - margin, recY);
  recY += 14;
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(8.2);
  doc.setTextColor(100, 116, 139);
  const disclaimerLines = doc.splitTextToSize(
    'Disclaimer: This report is produced by an educational machine learning decision-support tool. It does not constitute a medical diagnosis, clinical advice, or treatment plan. Always consult a qualified physician for medical decisions.',
    maxWidth
  );
  doc.text(disclaimerLines, margin, recY);

  doc.save(`health_risk_report_${report.patient_id}.pdf`);
}
