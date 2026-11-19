#!/bin/sh
set -e
DATA="${SINYAL_DATA_DIR:-/data}"

# Jalankan inisialisasi (database + pelatihan model) hanya pada start pertama.
if [ ! -f "$DATA/sinyal.db" ] || [ ! -f "$DATA/model.joblib" ]; then
  echo "Inisialisasi pertama: membuat database dan melatih model (sekitar 1 menit)..."
  python manage.py init
fi

exec gunicorn app:app --bind "0.0.0.0:${PORT:-8000}" --workers 1 --threads 4 --timeout 300
