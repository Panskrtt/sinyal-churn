# Sinyal: dashboard peringatan dini churn pelanggan

Flask + MySQL/MariaDB (atau SQLite), dengan komponen fuzzy dan siap di-deploy ke Railway.

- **Klasifikasi churn**: Regresi Logistik, Random Forest, Gradient Boosting (dipilih otomatis dari AUC validasi silang 5 lipatan).
- **Fuzzy C-Means** (`fcm.py`): segmentasi pelanggan dengan derajat keanggotaan di tiap segmen.
- **Fuzzy Sugeno orde nol** (`fuzzy_sugeno.py`): indeks risiko dari 10 aturan JIKA-MAKA; konsekuen dipelajari dari data.
- **Skor risiko gabungan** = (1 - a) x probabilitas model + a x indeks Sugeno; a dipilih dari AUC out-of-fold.
- K-Prototypes (`kprototypes.py`) dan K-Means dipertahankan sebagai pembanding klasterisasi.

## Database: MySQL (XAMPP) atau SQLite
Aplikasi memilih database dari `DATABASE_URL` di berkas `.env`:

| Pilihan | Nilai `DATABASE_URL` |
|---|---|
| MySQL XAMPP | `mysql://root:@127.0.0.1:3306/sinyal` |
| SQLite (tanpa server) | kosongkan |
### Menjalankan dengan MySQL XAMPP
1. Buka **XAMPP Control Panel**, klik **Start** pada **MySQL**. Apache tidak wajib; nyalakan hanya bila ingin melihat tabel lewat phpMyAdmin (http://localhost/phpmyadmin).
2. Siapkan `.env`:
   ```powershell
   copy .env.example .env
   ```
   Pastikan barisnya `DATABASE_URL=mysql://root:@127.0.0.1:3306/sinyal` (root tanpa kata sandi adalah bawaan XAMPP; bila kamu sudah memberi kata sandi: `mysql://root:KATASANDI@127.0.0.1:3306/sinyal`).
3. Pasang library (ada tambahan PyMySQL) dan inisialisasi:
   ```powershell
   pip install -r requirements.txt
   python manage.py init
   python -m flask --app app run --port 8080
   ```
   Database `sinyal` dan semua tabel dibuat otomatis. Buka http://127.0.0.1:8080.

Masalah umum:
- `Tidak dapat terhubung ke MySQL`: MySQL di XAMPP belum Start, atau port-nya bukan 3306 (lihat angka Port di Control Panel, lalu sesuaikan URL).
- `Access denied for user 'root'`: kata sandi root di `DATABASE_URL` tidak cocok.
- Karakter khusus pada kata sandi harus ditulis dengan persen-encoding di URL (`@` menjadi `%40`).

Catatan: model terlatih (`model.joblib`) dan kunci sesi disimpan sebagai berkas di folder `instance/`, sedangkan data pelanggan, pengguna, dan riwayat ada di database.

## Deploy publik di Railway

Workspace perlu tersedia di repositori GitHub yang bisa dihubungkan ke Railway. Railway memakai `Dockerfile` di root untuk membangun aplikasi di server Railway; Docker tidak perlu dipasang atau dijalankan di komputer lokal. Railway menyediakan database MySQL privat.

1. Buat project Railway, tambahkan service **MySQL**, lalu tambahkan service aplikasi dari repositori GitHub ini.
2. Pada service aplikasi, atur variabel berikut. Jika nama service database bukan `MySQL`, sesuaikan nama di referensi `DATABASE_URL`.

   | Variabel | Nilai |
   |---|---|
   | `DATABASE_URL` | `${{MySQL.MYSQL_URL}}` |
   | `APP_ENV` | `production` |
   | `SECRET_KEY` | Kunci acak minimal 32 byte; buat lokal dengan `python -c "import secrets; print(secrets.token_hex(32))"` |
   | `SESSION_COOKIE_SECURE` | `1` |
   | `INITIAL_ADMIN_PASSWORD` | Kata sandi acak minimal 12 karakter |
   | `INITIAL_ANALYST_PASSWORD` | Kata sandi acak minimal 12 karakter |
   | `INITIAL_RETENTION_PASSWORD` | Kata sandi acak minimal 12 karakter |

   Masukkan semua kata sandi dan `SECRET_KEY` langsung di Railway Variables, jangan commit ke Git atau kirim lewat chat. Ketiga kata sandi bootstrap hanya digunakan saat database belum memiliki pengguna.
3. Tambahkan persistent volume ke service aplikasi pada path `/data`. Ini menyimpan model terlatih dan kunci sesi melewati redeploy.
4. Atur health check service ke path `/health` (timeout 300 detik), lalu deploy. Saat deploy pertama, aplikasi mengisi database dengan dataset contoh dan melatih model.
5. Setelah deploy sehat, aktifkan **Generate Domain** pada Networking service aplikasi. Railway menyediakan domain HTTPS; login menggunakan akun yang dibuat dari kata sandi bootstrap di atas.

Railway mengisi variabel `PORT` otomatis. Database MySQL harus tetap privat; aplikasi mengaksesnya melalui `MYSQL_URL` internal. Aktifkan backup database sebelum memakai data pelanggan sungguhan.

## Menjalankan tanpa XAMPP dan tanpa Docker
Kosongkan `DATABASE_URL` di `.env`, lalu:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py init
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

Halaman tambahan: `/fuzzy` (penjelasan dan parameter fuzzy), `/health` (pemeriksaan kesehatan untuk Railway).

## Halaman per aktor
Halaman operasional pelanggan tetap dapat dipakai bersama, tetapi menu dan ringkasan awal disesuaikan dengan pekerjaan tiap peran:

| Peran | Halaman utama | Fokus ringkasan |
|---|---|---|
| Tim retensi | Ringkasan, Pelanggan, Segmen, Tindak lanjut | Antrean pelanggan berisiko dan pencatatan aksi retensi |
| Analis data | Ringkasan, Pelanggan, Segmen, Logika fuzzy, Unggah data, Model, Tindak lanjut | Kesiapan data, status model, evaluasi, dan analisis fuzzy |
| Admin sistem | Semua halaman analis dan retensi, ditambah Pengguna | Ikhtisar akun per peran dan pengelolaan hak akses |

Tim retensi tidak melihat menu Logika fuzzy, Unggah data, atau Model. Unggah data dan Model dibatasi di server untuk analis/admin; halaman Logika fuzzy tetap dapat dibuka tim retensi melalui URL langsung. Menu Pengguna dan pengelolaannya hanya tersedia untuk admin.

## Struktur
- `app.py` rute Flask, otorisasi per peran, CSRF.
- `ml.py` pembersihan data, klasifikasi, klasterisasi, gabungan fuzzy, penjelasan, saran aksi.
- `fcm.py`, `fuzzy_sugeno.py`, `kprototypes.py` algoritma yang ditulis sendiri (hanya bergantung pada NumPy/SciPy).
- `dbconn.py` koneksi database (SQLite atau MySQL). `service.py` skema dan penyimpanan hasil model. `manage.py` perintah `init` dan `ensure`.
- `Dockerfile`, `docker-entrypoint.sh` untuk build Railway.
- `tests/` tes black-box UT-01 sampai UT-16 dan uji unit komponen fuzzy.

## Tes
```bash
python manage.py init
python -m unittest discover -s tests -v
```
Secara bawaan tes memakai salinan SQLite di folder sementara. Untuk menguji MySQL, buat database khusus tes
(`sinyal_test`), isi `TEST_DATABASE_URL`, jalankan `init` pada database itu, lalu jalankan tes
(petunjuk lengkap di kepala `tests/test_blackbox.py`). Jangan arahkan tes ke database aplikasimu.
