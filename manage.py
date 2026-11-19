"""Perintah pengelolaan.  Pemakaian:  python manage.py init"""
import os
import sys
import time

import pandas as pd

import ml
import service

DATA_DIR = os.environ.get("SINYAL_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance"))
DB = os.path.join(DATA_DIR, "sinyal.db")
SEED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telco.csv")


def init(csv_path=SEED_CSV, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    conn = service.connect(os.path.join(data_dir, "sinyal.db"))
    service.init_schema(conn)
    if service.seed_users(conn):
        print("Akun awal dibuat: admin@sinyal.id / analis@sinyal.id / retensi@sinyal.id (kata sandi: <peran>123)")
        print("Segera ganti kata sandi lewat menu Pengguna.")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    errs = ml.validate(df)
    if errs:
        sys.exit("Dataset tidak valid: " + " ".join(errs))
    n = service.upsert_customers(conn, ml.prepare(df))
    print(f"{n} pelanggan dimuat. Melatih model (sekitar 1 menit)...")
    t = time.time()
    m = service.train_and_store(conn, data_dir)
    print(f"Selesai dalam {time.time() - t:.0f} detik. Model terpilih: {m['selected']}, "
          f"{m['k_selected']} segmen.")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init()
    else:
        print(__doc__)
