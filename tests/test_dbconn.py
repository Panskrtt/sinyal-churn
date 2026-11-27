"""Uji lapisan database tanpa server MySQL.  Jalankan:  python -m unittest tests.test_dbconn -v"""
import os
import re
import sys
import unittest
from decimal import Decimal
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dbconn  # noqa: E402
import service  # noqa: E402


def tables(ddl):
    out = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)\((.*?)\)(?:\s*ENGINE|\s*;|\s*$)", ddl, re.S):
        cols = set()
        for part in re.split(r",\s*(?![^()]*\))", m.group(2)):
            w = part.strip().replace("`", "").split()
            if w and w[0].upper() not in ("KEY", "FOREIGN", "PRIMARY", "UNIQUE"):
                cols.add(w[0])
        out[m.group(1)] = cols
    return out


class DbLayerTests(unittest.TestCase):
    def test_skema_sqlite_dan_mysql_punya_tabel_dan_kolom_sama(self):
        a, b = tables(service.SCHEMA_SQLITE), tables(service.SCHEMA_MYSQL)
        self.assertEqual(set(a), set(b))
        self.assertGreaterEqual(len(a), 7)
        for t in a:
            self.assertEqual(a[t], b[t], f"kolom tabel {t} berbeda")

    def test_semua_kolom_colmap_ada_di_skema_mysql(self):
        cols = tables(service.SCHEMA_MYSQL)["customers"]
        for _, db in service.COLMAP:
            self.assertIn(db, cols)

    def test_terjemahan_placeholder(self):
        self.assertEqual(dbconn.MySQLConn.translate("SELECT * FROM t WHERE a=? AND b LIKE ?"),
                         "SELECT * FROM t WHERE a=%s AND b LIKE %s")
        self.assertEqual(dbconn.MySQLConn.translate("SELECT '100%' WHERE x=?"), "SELECT '100%%' WHERE x=%s")

    def test_parse_url(self):
        p = dbconn.parse_mysql("mysql://root:@127.0.0.1:3306/sinyal")
        self.assertEqual((p["user"], p["password"], p["host"], p["port"], p["db"]), ("root", "", "127.0.0.1", 3306, "sinyal"))
        p = dbconn.parse_mysql("mysql+pymysql://sinyal:p%40ss@db/sinyal")
        self.assertEqual((p["password"], p["host"], p["port"]), ("p@ss", "db", 3306))
        with self.assertRaises(RuntimeError):
            dbconn.parse_mysql("mysql://root@localhost/bad-name;drop")

    def test_row_dan_decimal(self):
        r = dbconn.Row(["a", "b"], [1, "x"])
        self.assertEqual((r[0], r["b"], list(r), len(r)), (1, "x", [1, "x"], 2))
        self.assertEqual(dict(r), {"a": 1, "b": "x"})
        self.assertEqual(dbconn.conv(Decimal("12")), 12)
        self.assertIsInstance(dbconn.conv(Decimal("12")), int)
        self.assertAlmostEqual(dbconn.conv(Decimal("0.2654")), 0.2654)

    def test_sqlite_tetap_default_dan_dialek(self):
        import tempfile
        d = tempfile.mkdtemp()
        c = dbconn.connect("sqlite:///" + os.path.join(d, "x.db"))
        self.assertEqual(c.dialect, "sqlite")
        service.init_schema(c)
        service.init_schema(c)  # idempoten
        self.assertEqual(service.get_threshold(c), 0.4)
        c.close()

    def test_seed_produksi_mewajibkan_kata_sandi_bootstrap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = dbconn.connect("sqlite:///" + os.path.join(d, "x.db"))
            service.init_schema(c)
            with patch.dict(os.environ, {"APP_ENV": "production"}, clear=False):
                for key in ("INITIAL_ADMIN_PASSWORD", "INITIAL_ANALYST_PASSWORD", "INITIAL_RETENTION_PASSWORD"):
                    os.environ.pop(key, None)
                with self.assertRaisesRegex(RuntimeError, "wajib diatur"):
                    service.seed_users(c)
            c.close()

    def test_seed_produksi_menggunakan_kata_sandi_rahasia(self):
        import tempfile
        from werkzeug.security import check_password_hash

        passwords = {
            "APP_ENV": "production",
            "INITIAL_ADMIN_PASSWORD": "initial-admin-secret",
            "INITIAL_ANALYST_PASSWORD": "initial-analyst-secret",
            "INITIAL_RETENTION_PASSWORD": "initial-retention-secret",
        }
        with tempfile.TemporaryDirectory() as d:
            c = dbconn.connect("sqlite:///" + os.path.join(d, "x.db"))
            service.init_schema(c)
            with patch.dict(os.environ, passwords, clear=False):
                self.assertTrue(service.seed_users(c))
            user = c.execute("SELECT password_hash FROM users WHERE email=?", ("admin@sinyal.id",)).fetchone()
            self.assertTrue(check_password_hash(user["password_hash"], passwords["INITIAL_ADMIN_PASSWORD"]))
            c.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
