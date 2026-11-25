import Database from 'better-sqlite3';
import path from 'node:path';
import type {
  ContributingFactor,
  ModelKey,
  PatientRecord,
  RiskCategory,
  SavedReport,
} from '../src/types.ts';

const DB_PATH = path.resolve(process.cwd(), 'healthcare_records.db');

let dbInstance: Database.Database | null = null;

export function getDb(): Database.Database {
  if (dbInstance) return dbInstance;

  const db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');

  db.exec(`
    CREATE TABLE IF NOT EXISTS patients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id TEXT NOT NULL UNIQUE,
      age INTEGER NOT NULL,
      gender TEXT NOT NULL,
      height_cm REAL NOT NULL,
      weight_kg REAL NOT NULL,
      bmi REAL NOT NULL,
      systolic_bp INTEGER NOT NULL,
      diastolic_bp INTEGER NOT NULL,
      glucose_level REAL NOT NULL,
      cholesterol_level REAL NOT NULL,
      heart_rate INTEGER NOT NULL,
      smoking_status TEXT NOT NULL,
      physical_activity TEXT NOT NULL,
      family_history TEXT NOT NULL,
      existing_conditions TEXT NOT NULL,
      risk_category TEXT NOT NULL,
      record_source TEXT NOT NULL DEFAULT 'registered',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id TEXT NOT NULL,
      model_key TEXT NOT NULL,
      model_name TEXT NOT NULL,
      predicted_risk TEXT NOT NULL,
      probabilities_json TEXT NOT NULL,
      patient_params_json TEXT NOT NULL,
      contributing_factors_json TEXT NOT NULL,
      recommendations_json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  dbInstance = db;
  return db;
}

export function seedInitialPatientsIfEmpty(records: PatientRecord[]): void {
  const db = getDb();
  const countRow = db.prepare('SELECT COUNT(*) as total FROM patients').get() as {
    total: number;
  };
  if (countRow.total > 0) return;

  const insertStmt = db.prepare(`
    INSERT OR IGNORE INTO patients (
      patient_id, age, gender, height_cm, weight_kg, bmi,
      systolic_bp, diastolic_bp, glucose_level, cholesterol_level,
      heart_rate, smoking_status, physical_activity, family_history,
      existing_conditions, risk_category, record_source
    ) VALUES (
      ?, ?, ?, ?, ?, ?,
      ?, ?, ?, ?,
      ?, ?, ?, ?,
      ?, ?, ?
    )
  `);

  const insertMany = db.transaction((items: PatientRecord[]) => {
    for (const p of items) {
      insertStmt.run(
        p.patient_id,
        p.age,
        p.gender,
        p.height_cm,
        p.weight_kg,
        p.bmi,
        p.systolic_bp,
        p.diastolic_bp,
        p.glucose_level,
        p.cholesterol_level,
        p.heart_rate,
        p.smoking_status,
        p.physical_activity,
        p.family_history,
        p.existing_conditions,
        p.risk_category,
        p.record_source
      );
    }
  });

  insertMany(records);
}

export function listAllPatients(): PatientRecord[] {
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT id, patient_id, age, gender, height_cm, weight_kg, bmi,
              systolic_bp, diastolic_bp, glucose_level, cholesterol_level,
              heart_rate, smoking_status, physical_activity, family_history,
              existing_conditions, risk_category, record_source, created_at
       FROM patients
       ORDER BY CASE WHEN record_source = 'registered' THEN 0 ELSE 1 END, id DESC`
    )
    .all() as PatientRecord[];
  return rows;
}

export function findPatientById(patientId: string): PatientRecord | undefined {
  const db = getDb();
  return db
    .prepare('SELECT * FROM patients WHERE patient_id = ?')
    .get(patientId) as PatientRecord | undefined;
}

export function insertPatient(record: PatientRecord): PatientRecord {
  const db = getDb();
  const stmt = db.prepare(`
    INSERT INTO patients (
      patient_id, age, gender, height_cm, weight_kg, bmi,
      systolic_bp, diastolic_bp, glucose_level, cholesterol_level,
      heart_rate, smoking_status, physical_activity, family_history,
      existing_conditions, risk_category, record_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  stmt.run(
    record.patient_id,
    record.age,
    record.gender,
    record.height_cm,
    record.weight_kg,
    record.bmi,
    record.systolic_bp,
    record.diastolic_bp,
    record.glucose_level,
    record.cholesterol_level,
    record.heart_rate,
    record.smoking_status,
    record.physical_activity,
    record.family_history,
    record.existing_conditions,
    record.risk_category,
    record.record_source
  );

  return findPatientById(record.patient_id)!;
}

export function deletePatientById(patientId: string): boolean {
  const db = getDb();
  const res = db
    .prepare('DELETE FROM patients WHERE patient_id = ?')
    .run(patientId);
  return res.changes > 0;
}

export function savePredictionReport(input: {
  patient_id: string;
  model_key: ModelKey;
  model_name: string;
  predicted_risk: RiskCategory;
  probabilities: Record<RiskCategory, number>;
  patient_params: SavedReport['patient_params'];
  contributing_factors: ContributingFactor[];
  recommendations: string[];
}): SavedReport {
  const db = getDb();
  const stmt = db.prepare(`
    INSERT INTO predictions (
      patient_id, model_key, model_name, predicted_risk,
      probabilities_json, patient_params_json,
      contributing_factors_json, recommendations_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const info = stmt.run(
    input.patient_id,
    input.model_key,
    input.model_name,
    input.predicted_risk,
    JSON.stringify(input.probabilities),
    JSON.stringify(input.patient_params),
    JSON.stringify(input.contributing_factors),
    JSON.stringify(input.recommendations)
  );

  const row = db
    .prepare('SELECT * FROM predictions WHERE id = ?')
    .get(Number(info.lastInsertRowid)) as {
    id: number;
    patient_id: string;
    model_key: ModelKey;
    model_name: string;
    predicted_risk: RiskCategory;
    probabilities_json: string;
    patient_params_json: string;
    contributing_factors_json: string;
    recommendations_json: string;
    created_at: string;
  };

  return {
    id: row.id,
    patient_id: row.patient_id,
    model_key: row.model_key,
    model_name: row.model_name,
    predicted_risk: row.predicted_risk,
    probabilities: JSON.parse(row.probabilities_json),
    patient_params: JSON.parse(row.patient_params_json),
    contributing_factors: JSON.parse(row.contributing_factors_json),
    recommendations: JSON.parse(row.recommendations_json),
    created_at: row.created_at,
  };
}

export function listSavedReports(): SavedReport[] {
  const db = getDb();
  const rows = db
    .prepare('SELECT * FROM predictions ORDER BY id DESC LIMIT 100')
    .all() as Array<{
    id: number;
    patient_id: string;
    model_key: ModelKey;
    model_name: string;
    predicted_risk: RiskCategory;
    probabilities_json: string;
    patient_params_json: string;
    contributing_factors_json: string;
    recommendations_json: string;
    created_at: string;
  }>;

  return rows.map((row) => ({
    id: row.id,
    patient_id: row.patient_id,
    model_key: row.model_key,
    model_name: row.model_name,
    predicted_risk: row.predicted_risk,
    probabilities: JSON.parse(row.probabilities_json),
    patient_params: JSON.parse(row.patient_params_json),
    contributing_factors: JSON.parse(row.contributing_factors_json),
    recommendations: JSON.parse(row.recommendations_json),
    created_at: row.created_at,
  }));
}

export function deleteSavedReport(id: number): boolean {
  const db = getDb();
  const res = db.prepare('DELETE FROM predictions WHERE id = ?').run(id);
  return res.changes > 0;
}
