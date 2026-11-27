#!/bin/sh
set -e

# Inisialisasi (tabel, dataset contoh, pelatihan model) hanya bila belum siap.
python manage.py ensure

exec gunicorn app:app --bind "0.0.0.0:${PORT:-8000}" --workers 1 --threads 4 --timeout 300
