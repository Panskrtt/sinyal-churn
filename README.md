# Sinyal: dashboard peringatan dini churn pelanggan

Flask + SQLite3, dengan komponen fuzzy dan siap dijalankan di Docker.

- **Klasifikasi churn**: Regresi Logistik, Random Forest, Gradient Boosting (dipilih otomatis dari AUC validasi silang 5 lipatan).
- **Fuzzy C-Means** (`fcm.py`): segmentasi pelanggan dengan derajat keanggotaan di tiap segmen.
- **Fuzzy Sugeno orde nol** (`fuzzy_sugeno.py`): indeks risiko dari 10 aturan JIKA-MAKA; konsekuen dipelajari dari data.
- **Skor risiko gabungan** = (1 - a) x probabilitas model + a x indeks Sugeno; a dipilih dari AUC out-of-fold.
- K-Prototypes (`kprototypes.py`) dan K-Means dipertahankan sebagai pembanding klasterisasi.

## Menjalankan dengan Docker (disarankan)
```bash
cp .env.example .env        # lalu isi SECRET_KEY
docker compose up --build
```
Buka http://localhost:8000. Start pertama butuh sekitar 1 menit untuk membuat database dan melatih model;
cek progres dengan `docker compose logs -f`. Data (SQLite, model, kunci sesi) disimpan di volume `sinyal-data`,
jadi tetap ada setelah container dibuat ulang.

Perintah berguna:
```bash
docker compose down            # hentikan (data tetap)
docker compose down -v         # hentikan dan HAPUS data
docker compose exec sinyal python -m unittest discover -s tests -v
```
Berkas `.env` jangan di-commit (sudah ada di `.gitignore`). Bila diakses lewat HTTPS, set `SESSION_COOKIE_SECURE=1`.

## Menjalankan tanpa Docker
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py init              # buat database + latih model
python -m flask --app app run --port 8080
```

## Akun awal (segera ganti lewat menu Pengguna)
| Peran | Email | Kata sandi |
|---|---|---|
| Admin sistem | admin@sinyal.id | admin123 |
| Analis data | analis@sinyal.id | analis123 |
| Tim retensi | retensi@sinyal.id | retensi123 |

## Pemetaan 10 use case
| Use case | Halaman | Peran |
|---|---|---|
| Login | /masuk | semua |
| Unggah data | /unggah | analis, admin |
| Latih ulang model | /model | analis, admin |
| Atur ambang probabilitas | /model | analis, admin |
| Lihat dashboard ringkasan | / | semua |
| Filter daftar pelanggan | /pelanggan | semua |
| Lihat profil segmen | /segmen dan /pelanggan/<id> | semua |
| Catat tindak lanjut | /pelanggan/<id> | semua |
| Ekspor laporan (CSV/Excel) | tombol di /pelanggan | semua |
| Kelola hak akses | /pengguna | admin |

Halaman tambahan: `/fuzzy` (penjelasan dan parameter fuzzy), `/health` (pemeriksaan kesehatan untuk Docker).

## Struktur
- `app.py` rute Flask, otorisasi per peran, CSRF.
- `ml.py` pembersihan data, klasifikasi, klasterisasi, gabungan fuzzy, penjelasan, saran aksi.
- `fcm.py`, `fuzzy_sugeno.py`, `kprototypes.py` algoritma yang ditulis sendiri (hanya bergantung pada NumPy/SciPy).
- `service.py` skema SQLite dan penyimpanan hasil model. `manage.py` perintah `init`.
- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh` pengemasan.
- `tests/` tes black-box UT-01 sampai UT-16 dan uji unit komponen fuzzy.

## Tes
```bash
python manage.py init
python -m unittest discover -s tests -v
```
