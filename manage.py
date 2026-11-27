"""Perintah pengelolaan.

    python manage.py init     buat tabel, muat dataset contoh, latih model (aman dijalankan ulang)
    python manage.py ensure   jalankan init hanya bila database/model belum siap (dipakai entrypoint container)

Database dipilih lewat DATABASE_URL di .env atau variabel lingkungan; bila kosong dipakai SQLite
di folder instance/.
"""
import os
import sys
import time

import pandas as pd

import dbconn
import ml
import service

dbconn.load_env()
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SINYAL_DATA_DIR", os.path.join(BASE, "instance"))
SEED_CSV = os.path.join(BASE, "data", "telco.csv")


def db_url(data_dir=DATA_DIR):
    return os.environ.get("DATABASE_URL") or "sqlite:///" + os.path.join(data_dir, "sinyal.db")


def init(csv_path=SEED_CSV, data_dir=DATA_DIR, url=None):
    url = url or db_url(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    service.ensure_database(url)
    conn = service.connect(url)
    service.init_schema(conn)
    if service.seed_users(conn):
        if os.environ.get("APP_ENV", "").lower() == "production":
            print("Akun awal dibuat menggunakan kata sandi dari variabel lingkungan.")
        else:
            print("Akun awal dibuat: admin@sinyal.id / analis@sinyal.id / retensi@sinyal.id (kata sandi: <peran>123)")
            print("Segera ganti kata sandi lewat menu Pengguna.")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    errs = ml.validate(df)
    if errs:
        sys.exit("Dataset tidak valid: " + " ".join(errs))
    n = service.upsert_customers(conn, ml.prepare(df))
    print(f"Database: {'MySQL/MariaDB' if service.is_mysql(url) else 'SQLite'}. {n} pelanggan dimuat. "
          "Melatih model (sekitar 1 menit)...")
    t = time.time()
    m = service.train_and_store(conn, data_dir)
    print(f"Selesai dalam {time.time() - t:.0f} detik. Model terpilih: {m['selected']}, {m['k_selected']} segmen.")
    conn.close()


def ready(data_dir=DATA_DIR, url=None):
    url = url or db_url(data_dir)
    if not os.path.exists(service.model_path(data_dir)):
        return False
    try:
        service.ensure_database(url)
        conn = service.connect(url)
        n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        r = conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        conn.close()
        return n > 0 and r > 0
    except Exception:
        return False


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "init":
            init()
        elif cmd == "ensure":
            init() if not ready() else print("Database dan model sudah siap.")
        else:
            print(__doc__)
    except RuntimeError as e:
        sys.exit(f"Galat: {e}")
