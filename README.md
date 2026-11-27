# Sinyal

Dashboard peringatan dini churn pelanggan berbasis Flask. Aplikasi memberi skor risiko, mengelompokkan pelanggan, dan membantu tim retensi mencatat tindak lanjut.

## Fitur

- Prediksi churn menggunakan beberapa model klasifikasi dan validasi silang.
- Segmentasi pelanggan dengan Fuzzy C-Means.
- Skor gabungan model dan Fuzzy Sugeno, dengan penjelasan faktor risiko.
- Daftar pelanggan dengan filter, profil, rekomendasi aksi, dan ekspor CSV/Excel.
- Tiga peran: Tim retensi, Analis data, dan Admin sistem.
- Penyimpanan SQLite untuk pengembangan lokal atau MySQL/MariaDB untuk deployment.

## Menjalankan secara lokal

Gunakan Python 3.12. SQLite dipakai secara default, jadi server database tidak diperlukan.

```powershell
Copy-Item .env.example .env
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py init
python -m flask --app app run --port 8080
```

Buka http://127.0.0.1:8080. Untuk pengembangan lokal, `.env.example` menggunakan `APP_ENV=development` dan `DATABASE_URL` kosong, sehingga database dibuat di `instance/`.

Akun lokal:

| Peran | Email | Kata sandi |
|---|---|---|
| Admin sistem | `admin@sinyal.id` | `admin123` |
| Analis data | `analis@sinyal.id` | `analis123` |
| Tim retensi | `retensi@sinyal.id` | `retensi123` |

Kredensial tersebut hanya untuk development/test. Jangan gunakan di deployment publik.

## Deploy ke Railway

Railway menggunakan Railpack untuk mendeteksi dan membangun aplikasi Python.

1. Push source code ke repository GitHub. Jangan sertakan `.env`, `.venv/`, atau `instance/`.
2. Di Railway, buat project dengan layanan **MySQL**, lalu tambahkan layanan aplikasi dari repository GitHub.
3. Pada pengaturan layanan aplikasi, pilih **Railpack** sebagai builder, biarkan Build Command kosong, dan isi Start Command:

   ```sh
   python manage.py ensure && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
   ```

4. Tambahkan variabel berikut pada layanan aplikasi. Ganti `MySQL` pada referensi variabel jika nama layanan database berbeda.

   | Variabel | Nilai |
   |---|---|
   | `DATABASE_URL` | `${{MySQL.MYSQL_URL}}` |
   | `APP_ENV` | `production` |
   | `SINYAL_DATA_DIR` | `/data` |
   | `SECRET_KEY` | Kunci acak minimal 32 byte |
   | `SESSION_COOKIE_SECURE` | `1` |
   | `INITIAL_ADMIN_PASSWORD` | Kata sandi unik, minimal 12 karakter |
   | `INITIAL_ANALYST_PASSWORD` | Kata sandi unik, minimal 12 karakter |
   | `INITIAL_RETENTION_PASSWORD` | Kata sandi unik, minimal 12 karakter |

   Buat nilai `SECRET_KEY` dengan `python -c "import secrets; print(secrets.token_hex(32))"`. Simpan semua nilai rahasia hanya di Railway Variables, jangan commit ke Git.

5. Tambahkan persistent volume ke layanan aplikasi pada path `/data`. Atur health check path ke `/health` dengan timeout 300 detik.
6. Deploy aplikasi. Proses pertama menyiapkan tabel, memuat dataset contoh, dan melatih model; proses ini bisa memerlukan beberapa menit.
7. Setelah deployment sehat, aktifkan **Generate Domain** pada Networking. Buka domain Railway dan masuk menggunakan email akun di atas dengan kata sandi produksi yang kamu atur.

Database MySQL sebaiknya tetap privat. Aktifkan backup sebelum memasukkan data pelanggan sungguhan.

## Halaman per peran

| Peran | Halaman utama | Fokus |
|---|---|---|
| Tim retensi | Ringkasan, Pelanggan, Segmen, Tindak lanjut | Antrean pelanggan berisiko dan pencatatan aksi retensi |
| Analis data | Ringkasan, Pelanggan, Segmen, Logika fuzzy, Unggah data, Model, Tindak lanjut | Data, pelatihan, evaluasi model, dan analisis fuzzy |
| Admin sistem | Semua halaman analis dan retensi, ditambah Pengguna | Pengelolaan akun dan hak akses |

## Tes

```powershell
python -m unittest tests.test_dbconn tests.test_fuzzy -v
```

Tes black-box memerlukan database dan model lokal yang sudah disiapkan. Jangan arahkan tes MySQL ke database produksi.

## Struktur proyek

- `app.py`: rute Flask, login, otorisasi peran, dan CSRF.
- `ml.py`, `fcm.py`, `fuzzy_sugeno.py`, `kprototypes.py`: pemrosesan data dan algoritma.
- `dbconn.py`, `service.py`, `manage.py`: koneksi, skema, penyimpanan, dan inisialisasi.
- `templates/`, `static/`: antarmuka aplikasi.
- `data/telco.csv`: dataset contoh untuk inisialisasi pertama.
- `tests/`: tes unit dan tes black-box.
