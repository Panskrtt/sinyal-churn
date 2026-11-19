"""Lapisan data: SQLite3 + penyimpanan hasil model. Dipakai oleh app.py dan manage.py."""
import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import ml

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('retensi','analis','admin')),
  aktif INTEGER NOT NULL DEFAULT 1, dibuat TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS customers(
  customer_id TEXT PRIMARY KEY, gender TEXT, senior_citizen TEXT, partner TEXT, dependents TEXT,
  tenure REAL, phone_service TEXT, multiple_lines TEXT, internet_service TEXT, online_security TEXT,
  online_backup TEXT, device_protection TEXT, tech_support TEXT, streaming_tv TEXT, streaming_movies TEXT,
  contract TEXT, paperless_billing TEXT, payment_method TEXT, monthly_charges REAL, total_charges REAL,
  churn_aktual TEXT, layanan_tambahan INTEGER, prob REAL, segmen INTEGER, diperbarui TEXT,
  prob_model REAL, skor_fuzzy REAL, keanggotaan TEXT);
CREATE INDEX IF NOT EXISTS idx_c_prob ON customers(prob);
CREATE TABLE IF NOT EXISTS followups(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
  user_id INTEGER NOT NULL REFERENCES users(id), aksi TEXT NOT NULL, status TEXT NOT NULL,
  catatan TEXT NOT NULL, dibuat TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS riwayat_prediksi(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL, prob REAL NOT NULL,
  sumber TEXT NOT NULL, waktu TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_r_cust ON riwayat_prediksi(customer_id);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY, nama TEXT, ukuran INTEGER, churn_rate REAL, tenure REAL, tagihan REAL,
  layanan REAL, ringkasan TEXT, rekomendasi TEXT);
CREATE TABLE IF NOT EXISTS model_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, waktu TEXT NOT NULL, user_id INTEGER, model TEXT,
  n_data INTEGER, metrics TEXT NOT NULL);
"""

COLMAP = [("customerID", "customer_id"), ("gender", "gender"), ("SeniorCitizen", "senior_citizen"),
          ("Partner", "partner"), ("Dependents", "dependents"), ("tenure", "tenure"),
          ("PhoneService", "phone_service"), ("MultipleLines", "multiple_lines"),
          ("InternetService", "internet_service"), ("OnlineSecurity", "online_security"),
          ("OnlineBackup", "online_backup"), ("DeviceProtection", "device_protection"),
          ("TechSupport", "tech_support"), ("StreamingTV", "streaming_tv"),
          ("StreamingMovies", "streaming_movies"), ("Contract", "contract"),
          ("PaperlessBilling", "paperless_billing"), ("PaymentMethod", "payment_method"),
          ("MonthlyCharges", "monthly_charges"), ("TotalCharges", "total_charges"),
          ("Churn", "churn_aktual"), ("jumlah_layanan_tambahan", "layanan_tambahan")]
TO_DB = dict(COLMAP)
TO_CSV = {v: k for k, v in COLMAP}
NOW = lambda: datetime.now().isoformat(timespec="seconds")


def connect(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(customers)")}
    for col, typ in (("prob_model", "REAL"), ("skor_fuzzy", "REAL"), ("keanggotaan", "TEXT")):
        if col not in have:  # migrasi dari versi lama
            conn.execute(f"ALTER TABLE customers ADD COLUMN {col} {typ}")
    conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('threshold','0.4')")
    conn.commit()


def get_threshold(conn):
    return float(conn.execute("SELECT value FROM settings WHERE key='threshold'").fetchone()[0])


# ------------------------------------------------------------ pelanggan
def upsert_customers(conn, d):
    """d: DataFrame hasil ml.prepare(). Data lama dengan ID sama diperbarui, riwayat tindak lanjut tetap."""
    cols = [db for _, db in COLMAP]
    sql = (f"INSERT INTO customers({','.join(cols)},diperbarui) VALUES({','.join('?' * len(cols))},?) "
           f"ON CONFLICT(customer_id) DO UPDATE SET " +
           ",".join(f"{c}=excluded.{c}" for c in cols[1:] if c != "churn_aktual") +
           ",churn_aktual=COALESCE(excluded.churn_aktual,customers.churn_aktual),diperbarui=excluded.diperbarui")
    rows = []
    for rec in d[[csv for csv, _ in COLMAP]].astype(object).where(lambda x: x.notna(), None).itertuples(index=False):
        rows.append([None if v is None else (v.item() if hasattr(v, "item") else v) for v in rec] + [NOW()])
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def customers_df(conn, ids=None):
    q = "SELECT * FROM customers"
    df = pd.read_sql_query(q, conn)
    if ids is not None:
        df = df[df["customer_id"].isin(ids)]
    df = df.rename(columns=TO_CSV)
    return df.reset_index(drop=True)


# ------------------------------------------------------------ model
def model_path(data_dir):
    return os.path.join(data_dir, "model.joblib")


def load_bundle(data_dir):
    p = model_path(data_dir)
    return joblib.load(p) if os.path.exists(p) else None


def score_and_store(conn, bundle, d, sumber, scores=None, segs=None, memb=None):
    """Hitung skor + segmen untuk baris d, simpan ke customers dan riwayat_prediksi.
    scores/segs/memb (dari ml.train) dipakai untuk data latih agar skor berasal dari prediksi out-of-fold."""
    r = ml.score_all(bundle, d)
    ids = d["customerID"].tolist()
    prob, pm, ps, seg, U = r["prob"].copy(), r["model"].copy(), r["fuzzy"].copy(), r["seg"].copy(), r["U"].tolist()
    for n, i in enumerate(ids):
        if scores is not None and i in scores["prob"]:
            prob[n], pm[n], ps[n] = scores["prob"][i], scores["model"][i], scores["fuzzy"][i]
        if segs is not None and i in segs:
            seg[n], U[n] = segs[i], memb[i]
    conn.executemany(
        "UPDATE customers SET prob=?, prob_model=?, skor_fuzzy=?, segmen=?, keanggotaan=? WHERE customer_id=?",
        [(float(prob[n]), float(pm[n]), float(ps[n]), int(seg[n]),
          json.dumps([round(float(x), 4) for x in U[n]]), i) for n, i in enumerate(ids)])
    t = NOW()
    conn.executemany("INSERT INTO riwayat_prediksi(customer_id,prob,sumber,waktu) VALUES(?,?,?,?)",
                     [(i, float(prob[n]), sumber, t) for n, i in enumerate(ids)])
    conn.commit()


def train_and_store(conn, data_dir, user_id=None):
    full = ml.prepare(customers_df(conn))
    lab = full[full["Churn"].notna()]
    if len(lab) < 300 or lab["Churn"].nunique() < 2 or lab["Churn"].value_counts().min() < 30:
        raise ValueError("Data berlabel belum cukup untuk melatih model. "
                         "Dibutuhkan minimal 300 pelanggan berlabel dengan masing-masing kelas churn minimal 30.")
    bundle, metrics, scores, segs, memb, profiles = ml.train(full)
    joblib.dump(bundle, model_path(data_dir))
    score_and_store(conn, bundle, full, "latih ulang", scores=scores, segs=segs, memb=memb)
    conn.execute("DELETE FROM segments")
    for i, p in enumerate(profiles):
        conn.execute("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?)",
                     (i, f"Segmen {chr(65 + i)}", p["ukuran"], p["churn_rate"], p["tenure"], p["tagihan"],
                      p["layanan"], p["ringkasan"], json.dumps(p["rekomendasi"], ensure_ascii=False)))
    metrics["trained_at"] = bundle["trained_at"]
    conn.execute("INSERT INTO model_runs(waktu,user_id,model,n_data,metrics) VALUES(?,?,?,?,?)",
                 (NOW(), user_id, metrics["selected"], metrics["n"], json.dumps(metrics)))
    conn.commit()
    return metrics


def latest_metrics(conn):
    r = conn.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None, None
    return r, json.loads(r["metrics"])


def seed_users(conn):
    from werkzeug.security import generate_password_hash
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
        return False
    for nama, email, pw, role in [("Admin Sistem", "admin@sinyal.id", "admin123", "admin"),
                                  ("Analis Data", "analis@sinyal.id", "analis123", "analis"),
                                  ("Tim Retensi", "retensi@sinyal.id", "retensi123", "retensi")]:
        conn.execute("INSERT INTO users(nama,email,password_hash,role,dibuat) VALUES(?,?,?,?,?)",
                     (nama, email, generate_password_hash(pw), role, NOW()))
    conn.commit()
    return True
