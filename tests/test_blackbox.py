"""Black-box testing UT-01..UT-14 (sesuai Bab 3.8.2 laporan). Jalankan:  python -m unittest tests.test_blackbox -v

Tes memakai salinan database di folder sementara, jadi data asli tidak berubah.
Butuh database yang sudah diinisialisasi:  python manage.py init
"""
import io
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["SINYAL_AUTOCREATE"] = "0"
from app import create_app  # noqa: E402

SRC = os.environ.get("SINYAL_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance"))
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "telco.csv")


class BB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        for f in ("sinyal.db", "model.joblib"):
            shutil.copy(os.path.join(SRC, f), cls.tmp)
        cls.app = create_app(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def client(self, email=None, pw=None):
        c = self.app.test_client()
        with c.session_transaction() as s:
            s["_csrf"] = "tok"
        if email:
            c.post("/masuk", data={"email": email, "password": pw, "csrf_token": "tok"})
            with c.session_transaction() as s:  # login membersihkan sesi, jadi token CSRF diganti
                s["_csrf"] = "tok"
        return c

    def post(self, c, url, **data):
        return c.post(url, data=dict(data, csrf_token="tok"), follow_redirects=True)

    def admin(self):
        return self.client("admin@sinyal.id", "admin123")

    def analis(self):
        return self.client("analis@sinyal.id", "analis123")

    # UT-01 / UT-02  (FR-13)
    def test_ut01_login_benar(self):
        c = self.client()
        r = c.post("/masuk", data={"email": "retensi@sinyal.id", "password": "retensi123", "csrf_token": "tok"}, follow_redirects=True)
        self.assertIn("pelanggan aktif berada di atas ambang", r.text)

    def test_ut02_login_salah(self):
        c = self.client()
        r = c.post("/masuk", data={"email": "retensi@sinyal.id", "password": "salah", "csrf_token": "tok"}, follow_redirects=True)
        self.assertIn("Email atau kata sandi salah", r.text)
        self.assertEqual(c.get("/").status_code, 302)  # tetap ditolak

    # UT-03 / UT-04  (FR-01)
    def test_ut03_unggah_valid(self):
        head, rows = open(CSV, encoding="utf-8").read().splitlines()[0], open(CSV, encoding="utf-8").read().splitlines()[1:6]
        body = "\n".join([head] + [r.replace(r.split(",")[0], "UJI-" + r.split(",")[0], 1) for r in rows])
        c = self.analis()
        r = c.post("/unggah", data={"csrf_token": "tok", "berkas": (io.BytesIO(body.encode()), "u.csv")},
                   content_type="multipart/form-data", follow_redirects=True)
        self.assertIn("5 pelanggan disimpan (5 baru, 0 diperbarui)", r.text)

    def test_ut04_unggah_tidak_valid(self):
        c = self.analis()
        r = c.post("/unggah", data={"csrf_token": "tok", "berkas": (io.BytesIO(b"customerID,tenure\nA,1\n"), "u.csv")},
                   content_type="multipart/form-data", follow_redirects=True)
        self.assertIn("Kolom wajib tidak ditemukan", r.text)
        head = open(CSV, encoding="utf-8").read().splitlines()[0]
        bad = head + "\nX1,Male,0,Yes,No,abc,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No"
        r = c.post("/unggah", data={"csrf_token": "tok", "berkas": (io.BytesIO(bad.encode()), "u.csv")},
                   content_type="multipart/form-data", follow_redirects=True)
        self.assertIn("non-angka", r.text)

    # UT-05  (FR-11)
    def test_ut05_latih_ulang(self):
        r = self.post(self.analis(), "/model/latih")
        self.assertIn("Model dilatih ulang", r.text)

    # UT-06 / UT-07  (FR-10)
    def test_ut06_ambang_valid(self):
        c = self.analis()
        r = self.post(c, "/model/ambang", ambang="0,5")
        self.assertIn("Ambang disimpan", r.text)
        self.assertIn("0,50", r.text)

    def test_ut07_ambang_tidak_valid(self):
        for v in ("1,5", "-0,1", "abc", ""):
            r = self.post(self.analis(), "/model/ambang", ambang=v)
            self.assertIn("antara 0 dan 1", r.text, v)

    # UT-08  (FR-06, FR-02, FR-07)
    def test_ut08_dashboard(self):
        r = self.client("retensi@sinyal.id", "retensi123").get("/")
        self.assertIn("<svg", r.text)
        self.assertIn("Faktor yang berkaitan dengan churn", r.text)
        self.assertIn("Segmen pelanggan", r.text)

    # UT-09  (FR-03, FR-04)
    def test_ut09_filter(self):
        c = self.client("retensi@sinyal.id", "retensi123")
        r = c.get("/pelanggan?kontrak=Two year&tampil=semua&status=semua")
        self.assertIn("2 tahun", r.text)
        self.assertNotIn("<td>bulanan</td>", r.text)
        r2 = c.get("/pelanggan?zzz=1&q=TIDAK-ADA")
        self.assertIn("Tidak ada pelanggan yang cocok", r2.text)

    # UT-10  (FR-05, FR-08)
    def test_ut10_profil_segmen(self):
        r = self.client("retensi@sinyal.id", "retensi123").get("/pelanggan/7590-VHVEG")
        self.assertIn("Segmen", r.text)
        self.assertIn("Saran aksi retensi", r.text)
        self.assertIn("Faktor pendorong risiko", r.text)

    # UT-11 / UT-12  (FR-12)
    def test_ut11_tindak_lanjut(self):
        c = self.client("retensi@sinyal.id", "retensi123")
        r = self.post(c, "/pelanggan/7590-VHVEG/tindak-lanjut", aksi="Telepon pelanggan", status="Direncanakan", catatan="Hubungi besok pagi")
        self.assertIn("Tindak lanjut disimpan", r.text)
        self.assertIn("Hubungi besok pagi", r.text)

    def test_ut12_tindak_lanjut_kosong(self):
        c = self.client("retensi@sinyal.id", "retensi123")
        r = self.post(c, "/pelanggan/7590-VHVEG/tindak-lanjut", aksi="Telepon pelanggan", status="Direncanakan", catatan="   ")
        self.assertIn("Catatan tindak lanjut wajib diisi", r.text)

    # UT-13  (FR-09)
    def test_ut13_ekspor(self):
        c = self.client("retensi@sinyal.id", "retensi123")
        r = c.get("/ekspor?format=csv&kontrak=Month-to-month")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment; filename=pelanggan-", r.headers["Content-Disposition"])
        self.assertIn("ID pelanggan", r.text)
        x = c.get("/ekspor?format=xlsx")
        self.assertEqual(x.data[:2], b"PK")

    # UT-14  (kelola hak akses, NFR-04)
    def test_ut14_tambah_akun(self):
        c = self.admin()
        r = self.post(c, "/pengguna", nama="Uji Coba", email="uji@sinyal.id", password="rahasia123", role="retensi")
        self.assertIn("uji@sinyal.id", r.text)
        self.assertIn("ditambahkan", r.text)
        self.assertEqual(self.client("uji@sinyal.id", "rahasia123").get("/").status_code, 200)

    # UT-15 / UT-16  (komponen fuzzy)
    def test_ut15_halaman_logika_fuzzy(self):
        r = self.client("retensi@sinyal.id", "retensi123").get("/fuzzy")
        self.assertIn("Fuzzy C-Means untuk segmentasi", r.text)
        self.assertIn("Basis aturan", r.text)
        self.assertEqual(r.text.count("<polyline"), 8)

    def test_ut16_profil_menampilkan_keanggotaan_dan_aturan(self):
        r = self.client("retensi@sinyal.id", "retensi123").get("/pelanggan/7590-VHVEG")
        self.assertIn("Derajat keanggotaan segmen", r.text)
        self.assertIn("Aturan Sugeno yang paling aktif", r.text)

    def test_health_tanpa_login(self):
        r = self.app.test_client().get("/health")
        self.assertEqual((r.status_code, r.text), (200, "ok"))

    # keamanan tambahan (NFR-04)
    def test_otorisasi_per_peran(self):
        c = self.client("retensi@sinyal.id", "retensi123")
        for u in ("/unggah", "/model", "/pengguna"):
            self.assertEqual(c.get(u).status_code, 403, u)
        self.assertEqual(self.analis().get("/pengguna").status_code, 403)

    def test_csrf_dan_login_wajib(self):
        c = self.client("admin@sinyal.id", "admin123")
        self.assertEqual(c.post("/model/ambang", data={"ambang": "0.3"}).status_code, 400)
        self.assertEqual(self.app.test_client().get("/pelanggan").status_code, 302)


if __name__ == "__main__":
    unittest.main(verbosity=2)
