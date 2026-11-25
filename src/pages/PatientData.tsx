import React, { useEffect, useMemo, useState } from 'react';
import { Download, Search, Trash2 } from 'lucide-react';
import { api } from '../lib/api';
import type { PatientRecord } from '../types';

export function PatientData() {
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<'all' | 'Low Risk' | 'Moderate Risk' | 'High Risk'>('all');
  const [sourceFilter, setSourceFilter] = useState<'all' | 'registered' | 'synthetic_demo'>('all');
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const loadPatients = () => {
    setLoading(true);
    api
      .getPatients()
      .then((res) => setPatients(res.patients))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load patient records.'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadPatients();
  }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return patients.filter((p) => {
      if (riskFilter !== 'all' && p.risk_category !== riskFilter) return false;
      if (sourceFilter !== 'all' && p.record_source !== sourceFilter) return false;
      if (
        q &&
        !p.patient_id.toLowerCase().includes(q) &&
        !p.existing_conditions.toLowerCase().includes(q) &&
        !p.gender.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [patients, search, riskFilter, sourceFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const paged = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleDelete = async (patientId: string) => {
    try {
      await api.deletePatient(patientId);
      setPatients((prev) => prev.filter((p) => p.patient_id !== patientId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed.');
    }
  };

  const handleExportCsv = () => {
    const cols: Array<keyof PatientRecord> = [
      'patient_id',
      'age',
      'gender',
      'height_cm',
      'weight_kg',
      'bmi',
      'systolic_bp',
      'diastolic_bp',
      'glucose_level',
      'cholesterol_level',
      'heart_rate',
      'smoking_status',
      'physical_activity',
      'family_history',
      'existing_conditions',
      'risk_category',
      'record_source',
    ];
    const escape = (v: unknown) => {
      let s = String(v ?? '');
      if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const csv = [
      cols.join(','),
      ...filtered.map((row) => cols.map((c) => escape(row[c])).join(',')),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sqlite_patient_records.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const riskColor = (r: string) =>
    r === 'High Risk'
      ? 'text-red-700 font-semibold'
      : r === 'Moderate Risk'
        ? 'text-amber-700 font-semibold'
        : 'text-teal-700 font-semibold';

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Patient Data Repository</h1>
          <p className="mt-1 text-sm text-slate-600">
            Browse, search, and filter patient records stored in local SQLite. Synthetic
            demonstration records are explicitly labeled.
          </p>
        </div>
        <button
          type="button"
          onClick={handleExportCsv}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
        >
          <Download className="w-4 h-4" />
          Export Filtered CSV
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Search & Filter Toolbar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="Search by Patient ID, gender, or condition..."
            className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-sky-600/20 focus:border-sky-600"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
            {(['all', 'Low Risk', 'Moderate Risk', 'High Risk'] as const).map((tier) => (
              <button
                key={tier}
                type="button"
                onClick={() => {
                  setRiskFilter(tier);
                  setPage(1);
                }}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  riskFilter === tier
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {tier === 'all' ? 'All Tiers' : tier}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
            {(
              [
                ['all', 'All Sources'],
                ['registered', 'Registered'],
                ['synthetic_demo', 'Synthetic Demo'],
              ] as const
            ).map(([val, lbl]) => (
              <button
                key={val}
                type="button"
                onClick={() => {
                  setSourceFilter(val);
                  setPage(1);
                }}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  sourceFilter === val
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Data Table */}
      <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between text-xs text-slate-500">
          <span>
            Showing <strong className="text-slate-800 font-mono tabular-nums">{paged.length}</strong>{' '}
            of <strong className="text-slate-800 font-mono tabular-nums">{filtered.length}</strong>{' '}
            matching records
          </span>
          <span className="font-mono tabular-nums">
            Page {currentPage} / {totalPages}
          </span>
        </div>

        {loading ? (
          <div className="p-8 text-center text-sm text-slate-500">Loading patient records...</div>
        ) : paged.length === 0 ? (
          <div className="p-8 text-center text-sm text-slate-500">
            No patient records match your current search or filter criteria.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 font-medium">
                  <th className="py-2.5 px-4">Patient ID</th>
                  <th className="py-2.5 px-3">Source</th>
                  <th className="py-2.5 px-3 text-right">Age</th>
                  <th className="py-2.5 px-3">Gender</th>
                  <th className="py-2.5 px-3 text-right">BMI</th>
                  <th className="py-2.5 px-3 text-right">BP (mmHg)</th>
                  <th className="py-2.5 px-3 text-right">Glucose</th>
                  <th className="py-2.5 px-3 text-right">Chol.</th>
                  <th className="py-2.5 px-3 text-right">HR</th>
                  <th className="py-2.5 px-3">Smoking</th>
                  <th className="py-2.5 px-3">Conditions</th>
                  <th className="py-2.5 px-3">Risk Category</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {paged.map((p) => (
                  <tr key={p.patient_id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-4 font-mono font-medium text-slate-900 whitespace-nowrap">
                      {p.patient_id}
                    </td>
                    <td className="py-2.5 px-3 text-slate-500 whitespace-nowrap">
                      {p.record_source === 'registered' ? 'Registered' : 'Synthetic Demo'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                      {p.age}
                    </td>
                    <td className="py-2.5 px-3 text-slate-700">{p.gender}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                      {p.bmi.toFixed(1)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700 whitespace-nowrap">
                      {p.systolic_bp}/{p.diastolic_bp}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                      {p.glucose_level}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                      {p.cholesterol_level}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                      {p.heart_rate}
                    </td>
                    <td className="py-2.5 px-3 text-slate-700">{p.smoking_status}</td>
                    <td className="py-2.5 px-3 text-slate-700 whitespace-nowrap">
                      {p.existing_conditions}
                    </td>
                    <td className={`py-2.5 px-3 whitespace-nowrap ${riskColor(p.risk_category)}`}>
                      {p.risk_category}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {p.record_source === 'registered' ? (
                        <button
                          type="button"
                          onClick={() => handleDelete(p.patient_id)}
                          className="p-1 text-slate-400 hover:text-red-600 transition-colors"
                          title="Delete registered record"
                          aria-label={`Delete ${p.patient_id}`}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      ) : (
                        <span className="text-slate-300">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between text-xs">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1.5 border border-slate-300 rounded text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              Previous
            </button>
            <span className="text-slate-600 font-mono tabular-nums">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1.5 border border-slate-300 rounded text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
