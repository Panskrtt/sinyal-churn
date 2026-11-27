"""Koneksi database: SQLite (bawaan) atau MySQL/MariaDB.

Pilih lewat DATABASE_URL:
    sqlite:///instance/sinyal.db                      (default bila kosong)
    mysql://root:@127.0.0.1:3306/sinyal               (XAMPP: user root tanpa kata sandi)
    mysql://user:password@host:3306/sinyal             (Railway atau server MySQL lain)
Kode aplikasi memakai antarmuka yang sama untuk keduanya: execute(), executemany(), commit(), dan
baris hasil yang bisa diakses lewat indeks (row[0]) maupun nama (row["kolom"]).
"""
import os
import re
import sqlite3
from decimal import Decimal
from urllib.parse import unquote, urlparse


def load_env(path=None):
    """Baca berkas .env sederhana (KEY=VALUE) ke os.environ tanpa menimpa variabel yang sudah ada."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def is_mysql(url):
    return str(url).startswith("mysql")


def parse_mysql(url):
    u = urlparse(url)
    db = (u.path or "/").lstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_]+", db or ""):
        raise RuntimeError("Nama database pada DATABASE_URL harus berupa huruf, angka, atau garis bawah.")
    return dict(host=u.hostname or "127.0.0.1", port=u.port or 3306, user=unquote(u.username or "root"),
                password=unquote(u.password or ""), db=db)


# ------------------------------------------------------------------ baris & kursor (khusus MySQL)
class Row:
    """Mirip sqlite3.Row: row[0], row["nama"], keys(), iterasi nilai."""
    __slots__ = ("_keys", "_vals", "_idx")

    def __init__(self, keys, vals):
        self._keys, self._vals = list(keys), list(vals)
        self._idx = {k: i for i, k in enumerate(self._keys)}

    def __getitem__(self, i):
        return self._vals[i] if isinstance(i, int) else self._vals[self._idx[i]]

    def keys(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._vals)

    def __len__(self):
        return len(self._vals)

    def __repr__(self):
        return f"Row({dict(zip(self._keys, self._vals))})"


def conv(v):
    """MySQL mengembalikan SUM/AVG sebagai Decimal; ubah ke int/float agar sama dengan SQLite."""
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    return v


class _Cur:
    def __init__(self, cur):
        self._c = cur
        self.rowcount = cur.rowcount
        self.lastrowid = cur.lastrowid
        self.description = cur.description

    def _keys(self):
        return [d[0] for d in self._c.description]

    def fetchone(self):
        t = self._c.fetchone()
        return None if t is None else Row(self._keys(), [conv(x) for x in t])

    def fetchall(self):
        keys = self._keys()
        return [Row(keys, [conv(x) for x in t]) for t in self._c.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class MySQLConn:
    dialect = "mysql"

    def __init__(self, raw):
        self.raw = raw

    @staticmethod
    def translate(sql):
        """Placeholder gaya SQLite (?) -> gaya PyMySQL (%s); tanda % literal digandakan."""
        return sql.replace("%", "%%").replace("?", "%s")

    def execute(self, sql, params=()):
        cur = self.raw.cursor()
        cur.execute(self.translate(sql), tuple(params))
        return _Cur(cur)

    def executemany(self, sql, seq):
        cur = self.raw.cursor()
        cur.executemany(self.translate(sql), [tuple(x) for x in seq])
        return cur

    def executescript(self, script):
        for stmt in (s.strip() for s in script.split(";")):
            if stmt:
                self.execute(stmt)

    def commit(self):
        self.raw.commit()

    def close(self):
        self.raw.close()


class SQLiteConn(sqlite3.Connection):
    dialect = "sqlite"


def _friendly(p, e):
    return RuntimeError(
        f"Tidak dapat terhubung ke MySQL di {p['host']}:{p['port']} sebagai '{p['user']}': {e}. "
        "Pastikan server MySQL aktif dan DATABASE_URL benar.")


def _pymysql():
    try:
        import pymysql
        return pymysql
    except ImportError as e:
        raise RuntimeError("Paket PyMySQL belum terpasang. Jalankan: pip install -r requirements.txt") from e


def ensure_database(url):
    """Buat database MySQL bila belum ada (butuh hak CREATE; kegagalan hak akses diabaikan)."""
    if not is_mysql(url):
        return
    p, pymysql = parse_mysql(url), _pymysql()
    try:
        raw = pymysql.connect(host=p["host"], port=p["port"], user=p["user"], password=p["password"], charset="utf8mb4")
    except pymysql.err.OperationalError as e:
        raise _friendly(p, e) from e
    try:
        with raw.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{p['db']}` CHARACTER SET utf8mb4")
        raw.commit()
    except Exception:
        pass
    finally:
        raw.close()


def connect(url):
    url = str(url)
    if is_mysql(url):
        p, pymysql = parse_mysql(url), _pymysql()
        try:
            raw = pymysql.connect(host=p["host"], port=p["port"], user=p["user"], password=p["password"],
                                  database=p["db"], charset="utf8mb4", autocommit=False)
        except pymysql.err.OperationalError as e:
            raise _friendly(p, e) from e
        return MySQLConn(raw)
    path = url[len("sqlite:///"):] if url.startswith("sqlite:///") else url
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, factory=SQLiteConn)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
