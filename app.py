"""Sinyal: dashboard peringatan dini churn pelanggan (Flask + SQLite3)."""
import csv
import hmac
import io
import json
import os
import re
import secrets
from datetime import date, datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import (Flask, Response, abort, flash, g, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import fuzzy_sugeno as fs
import ml
import service

ROLES = {"retensi": "Tim retensi", "analis": "Analis data", "admin": "Admin sistem"}
AKSI = ["Telepon pelanggan", "Tawarkan kontrak lebih panjang", "Tawarkan potongan harga",
        "Tawarkan layanan tambahan", "Ajak beralih ke pembayaran otomatis", "Lainnya"]
STATUS_TL = ["Direncanakan", "Sudah dilakukan", "Berhasil dipertahankan", "Tetap berhenti"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENTS = sorted(ml.ALLOWED["PaymentMethod"])
PER_PAGE = 25
FACTORS = [("contract", "Jenis kontrak"), ("internet_service", "Layanan internet"),
           ("payment_method", "Metode pembayaran"), ("layanan_tambahan", "Jumlah layanan tambahan")]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app(data_dir=None):
    app = Flask(__name__)
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.environ.get("SINYAL_DATA_DIR", os.path.join(base, "instance"))
    os.makedirs(data_dir, exist_ok=True)
    key = os.environ.get("SECRET_KEY")
    if not key:
        kp = os.path.join(data_dir, "secret.key")
        if not os.path.exists(kp):
            with open(kp, "w") as f:
                f.write(secrets.token_hex(32))
        with open(kp) as f:
            key = f.read().strip()
    app.config.update(SECRET_KEY=key, DATA_DIR=data_dir, DATABASE=os.path.join(data_dir, "sinyal.db"),
                      MAX_CONTENT_LENGTH=10 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE") == "1")
    boot = service.connect(app.config["DATABASE"])
    service.init_schema(boot)
    service.seed_users(boot)
    boot.close()
    app._bundle = None
    app._bundle_mtime = None

    # ------------------------------------------------------------ util
    def db():
        if "db" not in g:
            g.db = service.connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        c = g.pop("db", None)
        if c:
            c.close()

    def bundle():
        p = service.model_path(app.config["DATA_DIR"])
        if not os.path.exists(p):
            return None
        m = os.path.getmtime(p)
        if app._bundle is None or app._bundle_mtime != m:
            app._bundle, app._bundle_mtime = service.load_bundle(app.config["DATA_DIR"]), m
        return app._bundle

    def csrf_token():
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_hex(16)
        return session["_csrf"]

    @app.before_request
    def guard():
        if request.method == "POST":
            tok = request.form.get("csrf_token", "")
            if not tok or not hmac.compare_digest(tok, session.get("_csrf", "")):
                abort(400)
        g.user = None
        if session.get("uid"):
            u = db().execute("SELECT * FROM users WHERE id=? AND aktif=1", (session["uid"],)).fetchone()
            if u:
                g.user = u
            else:
                session.clear()

    @app.after_request
    def headers(r):
        r.headers["X-Content-Type-Options"] = "nosniff"
        r.headers["X-Frame-Options"] = "DENY"
        r.headers["Referrer-Policy"] = "same-origin"
        return r

    def login_required(f):
        @wraps(f)
        def w(*a, **k):
            if not g.user:
                return redirect(url_for("login", next=request.full_path.rstrip("?")))
            return f(*a, **k)
        return w

    def roles(*allowed):
        def deco(f):
            @wraps(f)
            def w(*a, **k):
                if g.user["role"] not in allowed:
                    abort(403)
                return f(*a, **k)
            return login_required(w)
        return deco

    def threshold():
        return service.get_threshold(db())

    # ------------------------------------------------------------ template
    def idn(n, d=0):
        if n is None:
            return "–"
        s = f"{n:,.{d}f}"
        return s.replace(",", "§").replace(".", ",").replace("§", ".")

    app.jinja_env.filters["idn"] = idn
    app.jinja_env.filters["pct"] = lambda v, d=0: "–" if v is None else idn(v * 100, d) + "%"
    app.jinja_env.filters["tgl"] = lambda s: (s or "")[:16].replace("T", " ")
    app.jinja_env.filters["vid"] = lambda v: ml.VAL_ID.get(v, v)
    app.jinja_env.globals.update(csrf_token=csrf_token, ROLES=ROLES)

    @app.context_processor
    def ctx():
        return dict(user=g.get("user"), thr=threshold() if g.get("user") else None)

    def level(p):
        return 0 if p is None else 1 if p < 0.2 else 2 if p < 0.4 else 3 if p < 0.6 else 4

    app.jinja_env.globals["level"] = level

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def err(e):
        msg = {400: ("Permintaan tidak valid", "Muat ulang halaman lalu coba lagi."),
               403: ("Akses ditolak", "Peran akunmu tidak punya izin untuk membuka halaman ini."),
               404: ("Halaman tidak ditemukan", "Periksa alamatnya, atau kembali ke ringkasan."),
               413: ("Berkas terlalu besar", "Ukuran maksimum berkas adalah 10 MB.")}[e.code]
        return render_template("error.html", kode=e.code, judul=msg[0], isi=msg[1]), e.code

    @app.route("/health")
    def health():
        db().execute("SELECT 1").fetchone()
        return Response("ok", mimetype="text/plain")

    # ------------------------------------------------------------ UC1 login
    @app.route("/masuk", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            if u and u["aktif"] and check_password_hash(u["password_hash"], request.form.get("password", "")):
                session.clear()
                session["uid"] = u["id"]
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
            flash("Email atau kata sandi salah.", "error")
        return render_template("login.html")

    @app.route("/keluar", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ------------------------------------------------------------ UC5 dashboard
    def churn_by(col):
        rows = db().execute(f"SELECT {col} AS g, COUNT(*) AS n, AVG(churn_aktual='Yes') AS r FROM customers "
                            f"WHERE churn_aktual IS NOT NULL GROUP BY {col} ORDER BY {'g' if col == 'layanan_tambahan' else 'r DESC'}").fetchall()
        return [dict(g=str(r["g"]), n=r["n"], r=r["r"]) for r in rows]

    @app.route("/")
    @login_required
    def dashboard():
        c, t = db(), threshold()
        total = c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        scored = c.execute("SELECT COUNT(*) FROM customers WHERE prob IS NOT NULL").fetchone()[0]
        if not total or not scored:
            return render_template("dashboard.html", kosong=True, total=total)
        aktif = c.execute("SELECT COUNT(*) FROM customers WHERE churn_aktual IS NULL OR churn_aktual='No'").fetchone()[0]
        berisiko = c.execute("SELECT COUNT(*) FROM customers WHERE (churn_aktual IS NULL OR churn_aktual='No') "
                             "AND prob>=?", (t,)).fetchone()[0]
        rate = c.execute("SELECT AVG(churn_aktual='Yes') FROM customers WHERE churn_aktual IS NOT NULL").fetchone()[0]
        probs = np.array([r[0] for r in c.execute(
            "SELECT prob FROM customers WHERE prob IS NOT NULL AND (churn_aktual IS NULL OR churn_aktual='No')")])
        counts, _ = np.histogram(probs, bins=20, range=(0, 1))
        mx = max(counts.max(), 1)
        bars = [dict(x=i * 36, h=round(150 * n / mx, 1), n=int(n), hot=(i / 20) >= t - 1e-9, lo=i * 5, hi=i * 5 + 5)
                for i, n in enumerate(counts)]
        segs = c.execute("""SELECT s.*, (SELECT COUNT(*) FROM customers c WHERE c.segmen=s.id AND prob>=?
                            AND (churn_aktual IS NULL OR churn_aktual='No')) AS berisiko FROM segments s""", (t,)).fetchall()
        factors = [dict(judul=j, rows=churn_by(col)) for col, j in FACTORS]
        fmax = max(r["r"] for f in factors for r in f["rows"]) or 1
        return render_template("dashboard.html", kosong=False, total=total, aktif=aktif, berisiko=berisiko,
                               rate=rate, bars=bars, segs=segs, factors=factors, fmax=fmax, tx=round(t * 720, 1))

    # ------------------------------------------------------------ UC6 filter daftar + UC9 ekspor
    SORTS = {"prob": "prob DESC", "tenure": "tenure ASC", "tagihan": "monthly_charges DESC"}

    def build_filter(args):
        t, where, params = threshold(), ["1=1"], []
        q = args.get("q", "").strip()
        if q:
            where.append("customer_id LIKE ?")
            params.append(f"%{q}%")
        for key, col, ok in (("kontrak", "contract", CONTRACTS), ("internet", "internet_service", INTERNET),
                             ("bayar", "payment_method", PAYMENTS)):
            if args.get(key) in ok:
                where.append(f"{col}=?")
                params.append(args[key])
        if args.get("segmen", "").isdigit():
            where.append("segmen=?")
            params.append(int(args["segmen"]))
        status = args.get("status", "aktif")
        if status == "aktif":
            where.append("(churn_aktual IS NULL OR churn_aktual='No')")
        elif status == "berhenti":
            where.append("churn_aktual='Yes'")
        if args.get("tampil", "risiko") == "risiko":
            where.append("prob>=?")
            params.append(t)
        order = SORTS.get(args.get("urut", "prob"), SORTS["prob"])
        return " AND ".join(where), params, order

    @app.route("/pelanggan")
    @login_required
    def customers():
        where, params, order = build_filter(request.args)
        n = db().execute(f"SELECT COUNT(*) FROM customers WHERE {where}", params).fetchone()[0]
        pages = max(1, -(-n // PER_PAGE))
        page = min(max(request.args.get("hal", "1").isdigit() and int(request.args.get("hal", "1")) or 1, 1), pages)
        rows = db().execute(f"SELECT * FROM customers WHERE {where} ORDER BY {order}, customer_id "
                            f"LIMIT {PER_PAGE} OFFSET {(page - 1) * PER_PAGE}", params).fetchall()
        segs = db().execute("SELECT id, nama FROM segments ORDER BY id").fetchall()
        args = {k: v for k, v in request.args.items() if k != "hal" and v != ""}
        return render_template("customers.html", rows=rows, n=n, page=page, pages=pages, segs=segs,
                               contracts=CONTRACTS, internet=INTERNET, payments=PAYMENTS, args=args,
                               f=request.args, seg_names={s["id"]: s["nama"] for s in segs})

    def safe(v):
        if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
            return "'" + v
        return v

    @app.route("/ekspor")
    @login_required
    def export():
        where, params, order = build_filter(request.args)
        t = threshold()
        rows = db().execute(f"SELECT c.*, s.nama AS seg FROM customers c LEFT JOIN segments s ON s.id=c.segmen "
                            f"WHERE {where.replace('customer_id', 'c.customer_id')} ORDER BY {order.replace('tenure', 'c.tenure')}", params).fetchall()
        head = ["ID pelanggan", "Skor risiko gabungan", "Probabilitas model", "Indeks fuzzy Sugeno", "Status", "Segmen", "Kontrak", "Layanan internet",
                "Metode pembayaran", "Lama berlangganan (bulan)", "Tagihan bulanan (USD)", "Layanan tambahan",
                "Saran aksi"]
        out = []
        for r in rows:
            d = {TO: r[DB] for TO, DB in [(k, v) for k, v in service.COLMAP]}
            rec = " | ".join(ml.recommend(d)) if r["prob"] is not None and r["prob"] >= t else ""
            status = "Sudah berhenti" if r["churn_aktual"] == "Yes" else ("Berisiko" if (r["prob"] or 0) >= t else "Aman")
            out.append([safe(r["customer_id"]), None if r["prob"] is None else round(r["prob"], 4),
                        None if r["prob_model"] is None else round(r["prob_model"], 4),
                        None if r["skor_fuzzy"] is None else round(r["skor_fuzzy"], 4), status,
                        r["seg"] or "", r["contract"], r["internet_service"], r["payment_method"], r["tenure"],
                        r["monthly_charges"], r["layanan_tambahan"], rec])
        name = f"pelanggan-{date.today():%Y%m%d}"
        if request.args.get("format") == "xlsx":
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Pelanggan"
            ws.append(head)
            for r in out:
                ws.append(r)
            buf = io.BytesIO()
            wb.save(buf)
            return Response(buf.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            headers={"Content-Disposition": f"attachment; filename={name}.xlsx"})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(head)
        w.writerows(out)
        return Response("\ufeff" + buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}.csv"})

    # ------------------------------------------------------------ UC7 profil segmen + UC8 tindak lanjut
    @app.route("/pelanggan/<cid>")
    @login_required
    def customer(cid):
        r = db().execute("SELECT * FROM customers WHERE customer_id=?", (cid,)).fetchone()
        if not r:
            abort(404)
        row = {csv_: r[db_] for csv_, db_ in service.COLMAP}
        seg = db().execute("SELECT * FROM segments WHERE id=?", (r["segmen"],)).fetchone() if r["segmen"] is not None else None
        up = down = []
        rules, alpha = [], None
        b = bundle()
        if b and r["prob"] is not None:
            up, down = ml.explain(b, row)
            rules = fs.explain(pd.DataFrame([row]), b["sugeno"]["z"])
            alpha = b["sugeno"]["alpha"]
        memb = json.loads(r["keanggotaan"]) if r["keanggotaan"] else []
        seg_all = db().execute("SELECT id, nama FROM segments ORDER BY id").fetchall()
        t = threshold()
        hist = db().execute("SELECT prob,sumber,waktu FROM riwayat_prediksi WHERE customer_id=? ORDER BY id DESC LIMIT 8", (cid,)).fetchall()
        fu = db().execute("SELECT f.*, u.nama FROM followups f JOIN users u ON u.id=f.user_id "
                          "WHERE customer_id=? ORDER BY f.id DESC", (cid,)).fetchall()
        return render_template("customer.html", r=r, seg=seg, up=up, down=down, hist=hist, fu=fu,
                               rules=rules, alpha=alpha, memb=list(zip([s["nama"] for s in seg_all], memb)),
                               recs=ml.recommend(row), seg_recs=json.loads(seg["rekomendasi"]) if seg else [],
                               aksi=AKSI, status_tl=STATUS_TL, berisiko=(r["prob"] or 0) >= t)

    @app.route("/pelanggan/<cid>/tindak-lanjut", methods=["POST"])
    @login_required
    def add_followup(cid):
        if not db().execute("SELECT 1 FROM customers WHERE customer_id=?", (cid,)).fetchone():
            abort(404)
        catatan = request.form.get("catatan", "").strip()
        aksi, status = request.form.get("aksi"), request.form.get("status")
        if not catatan:
            flash("Catatan tindak lanjut wajib diisi.", "error")
        elif aksi not in AKSI or status not in STATUS_TL:
            flash("Pilih aksi dan status dari daftar yang tersedia.", "error")
        else:
            db().execute("INSERT INTO followups(customer_id,user_id,aksi,status,catatan,dibuat) VALUES(?,?,?,?,?,?)",
                         (cid, g.user["id"], aksi, status, catatan[:1000], service.NOW()))
            db().commit()
            flash("Tindak lanjut disimpan.", "ok")
        return redirect(url_for("customer", cid=cid) + "#tindak-lanjut")

    @app.route("/tindak-lanjut")
    @login_required
    def followups():
        rows = db().execute("SELECT f.*, u.nama FROM followups f JOIN users u ON u.id=f.user_id "
                            "ORDER BY f.id DESC LIMIT 200").fetchall()
        return render_template("followups.html", rows=rows)

    @app.route("/segmen")
    @login_required
    def segments():
        t = threshold()
        segs = db().execute("""SELECT s.*, (SELECT COUNT(*) FROM customers c WHERE c.segmen=s.id AND prob>=?
                               AND (churn_aktual IS NULL OR churn_aktual='No')) AS berisiko FROM segments s ORDER BY id""", (t,)).fetchall()
        return render_template("segments.html", segs=[dict(s, rec=json.loads(s["rekomendasi"])) for s in segs])

    @app.route("/fuzzy")
    @login_required
    def fuzzy():
        run, m = service.latest_metrics(db())
        colors = ["#0B7A83", "#DB6B2A", "#C7372F"]
        plots = []
        for var, sets in fs.mf_curves().items():
            lo, hi = sets["_range"]
            lines = []
            for i, (lab, pts) in enumerate((k, v) for k, v in sets.items() if k != "_range"):
                poly = " ".join(f"{10 + (x - lo) / (hi - lo) * 340:.1f},{120 - mu * 100:.1f}" for x, mu in pts)
                lines.append(dict(label=lab, poly=poly, color=colors[i % 3]))
            ticks = [(round(10 + (v - lo) / (hi - lo) * 340, 1), f"{v:g}") for v in np.linspace(lo, hi, 5)]
            plots.append(dict(judul=fs.VAR_LABEL[var], lines=lines, ticks=ticks))
        return render_template("fuzzy.html", m=m, run=run, plots=plots)

    # ------------------------------------------------------------ UC2 unggah
    @app.route("/unggah", methods=["GET", "POST"])
    @roles("analis", "admin")
    def upload():
        if request.method == "POST":
            f = request.files.get("berkas")
            if not f or not f.filename:
                flash("Pilih berkas CSV terlebih dahulu.", "error")
                return redirect(url_for("upload"))
            if not f.filename.lower().endswith(".csv"):
                flash("Format berkas harus CSV.", "error")
                return redirect(url_for("upload"))
            raw = f.read()
            df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.StringIO(raw.decode(enc)), dtype=str, keep_default_na=False)
                    break
                except Exception:
                    continue
            if df is None:
                flash("Berkas tidak dapat dibaca sebagai CSV.", "error")
                return redirect(url_for("upload"))
            errs = ml.validate(df)
            if errs:
                for e in errs[:6]:
                    flash(e, "error")
                return redirect(url_for("upload"))
            d = ml.prepare(df)
            existing = {r[0] for r in db().execute("SELECT customer_id FROM customers")}
            baru = sum(1 for i in d["customerID"] if i not in existing)
            service.upsert_customers(db(), d)
            b = bundle()
            if b:
                service.score_and_store(db(), b, d, "unggah")
                flash(f"{len(d)} pelanggan disimpan ({baru} baru, {len(d) - baru} diperbarui) dan sudah diberi skor.", "ok")
            else:
                flash(f"{len(d)} pelanggan disimpan. Latih model agar skor risiko muncul.", "ok")
            return redirect(url_for("upload"))
        total = db().execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        lab = db().execute("SELECT COUNT(*) FROM customers WHERE churn_aktual IS NOT NULL").fetchone()[0]
        return render_template("upload.html", total=total, lab=lab, kolom=ml.REQUIRED)

    # ------------------------------------------------------------ UC3 latih ulang + UC4 ambang + evaluasi
    @app.route("/model")
    @roles("analis", "admin")
    def model():
        run, m = service.latest_metrics(db())
        t = threshold()
        cm = None
        if m:
            q = "SELECT SUM(churn_aktual='Yes' AND prob>=?), SUM(churn_aktual='No' AND prob>=?), " \
                "SUM(churn_aktual='Yes' AND prob<?), SUM(churn_aktual='No' AND prob<?) " \
                "FROM customers WHERE churn_aktual IS NOT NULL AND prob IS NOT NULL"
            tp, fp, fn, tn = [x or 0 for x in db().execute(q, (t, t, t, t)).fetchone()]
            cm = dict(tp=tp, fp=fp, fn=fn, tn=tn)
        return render_template("model.html", run=run, m=m, cm=cm)

    @app.route("/model/latih", methods=["POST"])
    @roles("analis", "admin")
    def retrain():
        try:
            m = service.train_and_store(db(), app.config["DATA_DIR"], g.user["id"])
            flash(f"Model dilatih ulang dengan {idn(m['n'])} pelanggan. Model terpilih: {m['selected']}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("model"))

    @app.route("/model/ambang", methods=["POST"])
    @roles("analis", "admin")
    def set_threshold():
        raw = request.form.get("ambang", "").strip().replace(",", ".")
        try:
            v = float(raw)
            if not 0 <= v <= 1 or v != v:
                raise ValueError
        except ValueError:
            flash("Ambang harus berupa angka antara 0 dan 1, misalnya 0,4.", "error")
            return redirect(url_for("model"))
        db().execute("UPDATE settings SET value=? WHERE key='threshold'", (str(v),))
        db().commit()
        n = db().execute("SELECT COUNT(*) FROM customers WHERE prob>=? AND (churn_aktual IS NULL OR churn_aktual='No')", (v,)).fetchone()[0]
        flash(f"Ambang disimpan. Sekarang {idn(n)} pelanggan aktif berstatus berisiko.", "ok")
        return redirect(url_for("model"))

    # ------------------------------------------------------------ UC10 kelola hak akses
    @app.route("/pengguna", methods=["GET", "POST"])
    @roles("admin")
    def users():
        if request.method == "POST":
            nama = request.form.get("nama", "").strip()
            email = request.form.get("email", "").strip().lower()
            pw, role = request.form.get("password", ""), request.form.get("role")
            if not nama or not EMAIL_RE.match(email):
                flash("Isi nama dan alamat email yang valid.", "error")
            elif len(pw) < 8:
                flash("Kata sandi minimal 8 karakter.", "error")
            elif role not in ROLES:
                flash("Pilih peran dari daftar.", "error")
            elif db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
                flash("Email itu sudah terdaftar.", "error")
            else:
                db().execute("INSERT INTO users(nama,email,password_hash,role,dibuat) VALUES(?,?,?,?,?)",
                             (nama, email, generate_password_hash(pw), role, service.NOW()))
                db().commit()
                flash(f"Akun {email} ditambahkan.", "ok")
            return redirect(url_for("users"))
        return render_template("users.html", rows=db().execute("SELECT * FROM users ORDER BY id").fetchall())

    @app.route("/pengguna/<int:uid>", methods=["POST"])
    @roles("admin")
    def update_user(uid):
        u = db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            abort(404)
        role, aktif, pw = request.form.get("role", u["role"]), request.form.get("aktif"), request.form.get("password", "")
        if role not in ROLES:
            abort(400)
        if uid == g.user["id"] and (role != "admin" or aktif == "0"):
            flash("Kamu tidak bisa menurunkan peran atau menonaktifkan akunmu sendiri.", "error")
            return redirect(url_for("users"))
        if pw and len(pw) < 8:
            flash("Kata sandi baru minimal 8 karakter.", "error")
            return redirect(url_for("users"))
        db().execute("UPDATE users SET role=?, aktif=? WHERE id=?", (role, 0 if aktif == "0" else 1, uid))
        if pw:
            db().execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(pw), uid))
        db().commit()
        flash(f"Akun {u['email']} diperbarui.", "ok")
        return redirect(url_for("users"))

    return app


app = create_app() if os.environ.get("SINYAL_AUTOCREATE", "1") == "1" else None

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
