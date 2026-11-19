"""Pipeline sains data: pembersihan, klasifikasi churn, klasterisasi K-Prototypes, penjelasan."""
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import fuzzy_sugeno as fs
from fcm import FuzzyCMeans, memberships as fcm_memberships
from kprototypes import KPrototypes

YN = {"Yes", "No"}
ALLOWED = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {"0", "1", "Yes", "No"},
    "Partner": YN, "Dependents": YN, "PhoneService": YN, "PaperlessBilling": YN,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {"Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"},
}
REQUIRED = ["customerID", "tenure", "MonthlyCharges", "TotalCharges"] + list(ALLOWED)
ADDON = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
         "StreamingTV", "StreamingMovies"]

NUM_CLF = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_CLF = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
           "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
           "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM_CLF + CAT_CLF
SUGENO_NAME = "Fuzzy Sugeno (aturan)"
NUM_CLU = ["tenure", "MonthlyCharges", "jumlah_layanan_tambahan"]
CAT_CLU = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
           "SeniorCitizen", "Partner", "Dependents"]

LABEL = {
    "tenure": "Lama berlangganan", "MonthlyCharges": "Tagihan bulanan", "TotalCharges": "Total tagihan",
    "gender": "Jenis kelamin", "SeniorCitizen": "Pelanggan lansia", "Partner": "Punya pasangan",
    "Dependents": "Punya tanggungan", "PhoneService": "Layanan telepon", "MultipleLines": "Saluran ganda",
    "InternetService": "Layanan internet", "OnlineSecurity": "Keamanan online",
    "OnlineBackup": "Cadangan online", "DeviceProtection": "Proteksi perangkat",
    "TechSupport": "Dukungan teknis", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming film",
    "Contract": "Jenis kontrak", "PaperlessBilling": "Tagihan tanpa kertas",
    "PaymentMethod": "Metode pembayaran",
}
VAL_ID = {"Yes": "ya", "No phone service": "tanpa layanan telepon", "No internet service": "tanpa internet",
          "Male": "laki-laki", "Female": "perempuan", "Month-to-month": "bulanan", "One year": "1 tahun", "Two year": "2 tahun",
          "Fiber optic": "fiber optic", "DSL": "DSL", "No": "tidak",
          "Electronic check": "electronic check", "Mailed check": "cek pos",
          "Bank transfer (automatic)": "transfer otomatis", "Credit card (automatic)": "kartu kredit otomatis"}


# ---------------------------------------------------------------- data
def validate(df):
    """Kembalikan daftar pesan galat. Daftar kosong berarti data valid."""
    errs = []
    if df is None or len(df) == 0:
        return ["File tidak berisi baris data."]
    cols = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = cols
    miss = [c for c in REQUIRED if c not in cols]
    if miss:
        return ["Kolom wajib tidak ditemukan: " + ", ".join(miss) + "."]
    if df["customerID"].isna().any() or (df["customerID"].astype(str).str.strip() == "").any():
        errs.append("Kolom customerID memiliki nilai kosong.")
    for c in ["tenure", "MonthlyCharges"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            errs.append(f"Kolom {c} berisi nilai non-angka atau kosong ({int(v.isna().sum())} baris).")
        elif (v < 0).any():
            errs.append(f"Kolom {c} berisi nilai negatif.")
    tc = df["TotalCharges"].astype(str).str.strip().replace("", np.nan)
    tc = tc.where(tc != "nan", np.nan)
    if pd.to_numeric(tc, errors="coerce").isna().sum() > tc.isna().sum():
        errs.append("Kolom TotalCharges berisi nilai non-angka.")
    for c, ok in ALLOWED.items():
        bad = set(df[c].astype(str).str.strip().unique()) - ok
        if bad:
            errs.append(f"Kolom {c} berisi nilai tidak dikenal: {', '.join(sorted(bad)[:3])}.")
    if "Churn" in cols:
        bad = set(df["Churn"].dropna().astype(str).str.strip().unique()) - YN
        if bad:
            errs.append("Kolom Churn hanya boleh berisi Yes atau No.")
    return errs


def prepare(df):
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    d["customerID"] = d["customerID"].astype(str).str.strip()
    d = d.drop_duplicates("customerID", keep="last").reset_index(drop=True)
    d["tenure"] = pd.to_numeric(d["tenure"], errors="coerce")
    d["MonthlyCharges"] = pd.to_numeric(d["MonthlyCharges"], errors="coerce")
    tc = d["TotalCharges"].astype(str).str.strip().replace("", np.nan)
    d["TotalCharges"] = pd.to_numeric(tc, errors="coerce").fillna(d["tenure"] * d["MonthlyCharges"])
    for c in ALLOWED:
        d[c] = d[c].astype(str).str.strip()
    d["SeniorCitizen"] = d["SeniorCitizen"].map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
    d["jumlah_layanan_tambahan"] = (d[ADDON] == "Yes").sum(axis=1)
    if "Churn" in d:
        d["Churn"] = d["Churn"].astype("object").where(d["Churn"].isin(YN), None)
    else:
        d["Churn"] = None
    return d


# ---------------------------------------------------------------- klasifikasi
def _pipe(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM_CLF),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_CLF)])
    return Pipeline([("pre", pre), ("clf", model)])


def _candidates(seed):
    return [
        ("Regresi Logistik", lambda: LogisticRegression(max_iter=2000), True),
        ("Regresi Logistik (bobot kelas)", lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), False),
        ("Random Forest", lambda: RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=seed), True),
        ("Random Forest (bobot kelas)", lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed), False),
        ("Gradient Boosting", lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed), True),
        ("Baseline (tebakan proporsi kelas)", lambda: DummyClassifier(strategy="stratified", random_state=seed), False),
    ]


def _fold_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(y, p), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
                accuracy=accuracy_score(y, yhat))


def evaluate_classifiers(d, seed=42):
    X, y = d[FEATURES], (d["Churn"] == "Yes").astype(int).values
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows, oofs = [], {}
    for name, make, deployable in _candidates(seed):
        oof, folds = np.zeros(len(d)), []
        for tr, te in skf.split(X, y):
            m = _pipe(make()).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            folds.append(_fold_metrics(y[te], p))
        agg = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
        agg.update(name=name, deployable=deployable, auc_std=float(np.std([f["auc"] for f in folds])),
                   recall_std=float(np.std([f["recall"] for f in folds])))
        rows.append(agg)
        oofs[name] = oof
    # Fuzzy Sugeno: konsekuen dipelajari ulang pada tiap lipatan latih
    oof, folds = np.zeros(len(d)), []
    for tr, te in skf.split(X, y):
        z = fs.fit(d.iloc[tr], y[tr])
        oof[te] = fs.score(d.iloc[te], z)
        folds.append(_fold_metrics(y[te], oof[te]))
    agg = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    agg.update(name=SUGENO_NAME, deployable=False, auc_std=float(np.std([f["auc"] for f in folds])),
               recall_std=float(np.std([f["recall"] for f in folds])))
    rows.append(agg)
    oofs[SUGENO_NAME] = oof
    return rows, oofs, y


def threshold_table(y, p, grid=(0.2, 0.3, 0.4, 0.5, 0.6)):
    out = []
    for t in grid:
        yhat = (p >= t).astype(int)
        out.append(dict(t=t, precision=float(precision_score(y, yhat, zero_division=0)),
                        recall=float(recall_score(y, yhat, zero_division=0)),
                        f1=float(f1_score(y, yhat, zero_division=0)), flagged=int(yhat.sum())))
    return out


# ---------------------------------------------------------------- klasterisasi
def _cat_maps(d):
    return {c: {v: i for i, v in enumerate(sorted(d[c].unique()))} for c in CAT_CLU}


def _encode(d, maps):
    return np.column_stack([d[c].map(maps[c]).fillna(-1).astype(int).values for c in CAT_CLU])


def _gower(num, cat, ranges):
    D = np.zeros((len(num), len(num)))
    for j in range(num.shape[1]):
        D += np.abs(num[:, j][:, None] - num[:, j][None, :]) / ranges[j]
    for j in range(cat.shape[1]):
        D += cat[:, j][:, None] != cat[:, j][None, :]
    return D / (num.shape[1] + cat.shape[1])


def _assoc(labels, churn):
    """Chi-square dan Cramer's V antara kluster dan churn (data berlabel saja)."""
    m = pd.Series(churn).notna().values
    tab = pd.crosstab(pd.Series(labels)[m].values, pd.Series(churn)[m].values)
    chi2, p, _, _ = chi2_contingency(tab, correction=False)
    v = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
    return float(p), v


def _fuzzy_X(d, prep, w=0.5):
    """Matriks untuk FCM: numerik terstandar + one-hot kategorikal berbobot w.
    w = 0,5 menyamakan bobot ketidakcocokan kategori dengan gamma pada K-Prototypes."""
    Z = (d[NUM_CLU].astype(float).values - prep["mu"]) / prep["sd"]
    cols = [Z]
    for c in CAT_CLU:
        M = np.zeros((len(d), len(prep["maps"][c])))
        codes = d[c].map(prep["maps"][c]).fillna(-1).astype(int).values
        ok = codes >= 0
        M[np.where(ok)[0], codes[ok]] = w
        cols.append(M)
    return np.hstack(cols)


def evaluate_clustering(d, ks=range(2, 8), seed=42, sample=1500):
    raw = d[NUM_CLU].astype(float).values
    mu, sd = raw.mean(0), raw.std(0)
    Z = (raw - mu) / sd
    maps = _cat_maps(d)
    prep = dict(mu=mu, sd=sd, maps=maps)
    C = _encode(d, maps)
    XF = _fuzzy_X(d, prep)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d), min(len(d), sample), replace=False)
    ranges = raw.max(0) - raw.min(0)
    ranges[ranges == 0] = 1
    G = _gower(raw[idx], C[idx], ranges)
    rows, fitted, fitted_fcm = [], {}, {}
    for k in ks:
        kp = KPrototypes(k, random_state=seed).fit(Z, C)
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        fc = FuzzyCMeans(k, random_state=seed).fit(XF)
        r = {"k": k, "fcm_pc": fc.partition_coefficient(), "fcm_pe": fc.partition_entropy(), "fcm_xb": fc.xie_beni()}
        for tag, lab in (("kproto", kp.labels_), ("kmeans", km.labels_), ("fcm", fc.U_.argmax(1))):
            sub = lab[idx]
            r[f"{tag}_sil"] = float(silhouette_score(G, sub, metric="precomputed")) if len(set(sub)) > 1 else None
            p, v = _assoc(lab, d["Churn"].values)
            r[f"{tag}_p"], r[f"{tag}_v"] = p, v
        rows.append(r)
        fitted[k], fitted_fcm[k] = kp, fc
    return rows, fitted, fitted_fcm, prep


def recommend(row, prob=None):
    """Aturan sederhana untuk saran aksi retensi (satu pelanggan)."""
    r = []
    if row["Contract"] == "Month-to-month" and row["tenure"] <= 12:
        r.append("Tawarkan peralihan ke kontrak 1 tahun dengan potongan tagihan di bulan-bulan awal.")
    elif row["Contract"] == "Month-to-month":
        r.append("Tawarkan kontrak 1 atau 2 tahun dengan harga lebih rendah sebagai penghargaan masa berlangganan.")
    if row["PaymentMethod"] == "Electronic check":
        r.append("Ajak beralih ke pembayaran otomatis (transfer bank atau kartu kredit).")
    if row["InternetService"] == "Fiber optic" and row["TechSupport"] == "No":
        r.append("Tawarkan paket dukungan teknis untuk pelanggan fiber optic.")
    if row["InternetService"] != "No" and row["jumlah_layanan_tambahan"] <= 1:
        r.append("Kenalkan layanan tambahan (keamanan online atau cadangan) lewat masa percobaan gratis.")
    if row["tenure"] <= 6:
        r.append("Jadwalkan panggilan sambutan dalam 90 hari pertama untuk memastikan layanan berjalan baik.")
    if row["MonthlyCharges"] >= 80 and row["InternetService"] == "Fiber optic":
        r.append("Tinjau paket yang dipakai dan tawarkan penyesuaian harga bila ada paket yang lebih sesuai.")
    return r or ["Pertahankan komunikasi rutin. Belum ada pemicu risiko yang menonjol."]


def _segment_profiles(d, labels):
    rows = []
    for c in sorted(set(labels)):
        s = d[labels == c]
        lab = s["Churn"].dropna()
        modes = {}
        for col in CAT_CLU:
            vc = s[col].value_counts(normalize=True)
            modes[col] = (vc.index[0], float(vc.iloc[0]))
        con, net, pay = modes["Contract"][0], modes["InternetService"][0], modes["PaymentMethod"][0]
        net_txt = "tanpa internet" if net == "No" else f"internet {VAL_ID.get(net, net)}"
        summary = (f"Mayoritas kontrak {VAL_ID.get(con, con)}, {net_txt}, "
                   f"bayar dengan {VAL_ID.get(pay, pay)}. Rata-rata berlangganan {s['tenure'].mean():.0f} bulan "
                   f"dengan tagihan {s['MonthlyCharges'].mean():.0f} per bulan.")
        proto = {k: modes[k][0] for k in CAT_CLU}
        proto.update(tenure=s["tenure"].median(), MonthlyCharges=s["MonthlyCharges"].median(),
                     jumlah_layanan_tambahan=int(round(s["jumlah_layanan_tambahan"].median())),
                     TechSupport=s["TechSupport"].mode().iloc[0])
        rows.append(dict(ukuran=int(len(s)), churn_rate=float((lab == "Yes").mean()) if len(lab) else None,
                         tenure=float(s["tenure"].mean()), tagihan=float(s["MonthlyCharges"].mean()),
                         layanan=float(s["jumlah_layanan_tambahan"].mean()), ringkasan=summary,
                         modes={k: [v[0], v[1]] for k, v in modes.items()}, rekomendasi=recommend(proto)))
    return rows


# ---------------------------------------------------------------- pelatihan penuh
BLEND_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def train(d, seed=42, k_range=(3, 5)):
    """d: data hasil prepare() dengan label Churn terisi."""
    d = d[d["Churn"].notna()].reset_index(drop=True)
    clf_rows, oofs, y = evaluate_classifiers(d, seed)
    best = max((r for r in clf_rows if r["deployable"]), key=lambda r: r["auc"])
    make = {n: m for n, m, _ in _candidates(seed)}[best["name"]]
    final = _pipe(make()).fit(d[FEATURES], y)
    oof_m, oof_s = oofs[best["name"]], oofs[SUGENO_NAME]

    # skor gabungan = (1 - a) * probabilitas model + a * indeks Sugeno; a dipilih dari AUC out-of-fold
    blend = [dict(alpha=a, auc=float(roc_auc_score(y, (1 - a) * oof_m + a * oof_s))) for a in BLEND_GRID]
    alpha = max(blend, key=lambda r: r["auc"])["alpha"]
    oof = (1 - alpha) * oof_m + alpha * oof_s
    z = fs.fit(d, y)

    clu_rows, _, fitted_fcm, prep = evaluate_clustering(d, seed=seed)
    ok = [r for r in clu_rows if k_range[0] <= r["k"] <= k_range[1]]
    k_sel = min(ok, key=lambda r: r["fcm_xb"])["k"]          # indeks Xie-Beni terkecil
    fc = fitted_fcm[k_sel]
    raw_lab = fc.U_.argmax(1)
    # urutkan segmen: Segmen 0 (A) = tingkat churn tertinggi
    rate = [(d["Churn"][raw_lab == c] == "Yes").mean() for c in range(k_sel)]
    order = {int(c): int(i) for i, c in enumerate(np.argsort(rate)[::-1])}
    labels = np.array([order[int(c)] for c in raw_lab])
    U = np.zeros_like(fc.U_)
    for c, r in order.items():
        U[:, r] = fc.U_[:, c]
    profiles = _segment_profiles(d, labels)

    non = d[d["Churn"] == "No"]
    ref = {c: float(non[c].median()) for c in NUM_CLF}
    ref.update({c: non[c].mode().iloc[0] for c in CAT_CLF})

    bundle = dict(clf=final, name=best["name"], ref=ref, k=k_sel,
                  sugeno=dict(z=z, alpha=alpha),
                  seg=dict(mu=prep["mu"], sd=prep["sd"], maps=prep["maps"], centers=fc.centers_, m=fc.m, order=order),
                  trained_at=datetime.now().isoformat(timespec="seconds"))
    sugeno_row = next(r for r in clf_rows if r["name"] == SUGENO_NAME)
    p_seg, v_seg = _assoc(labels, d["Churn"].values)
    metrics = dict(n=len(d), churn_rate=float(y.mean()), clf=clf_rows, selected=best["name"],
                   thresholds=threshold_table(y, oof), clu=clu_rows, k_selected=k_sel, k_range=list(k_range),
                   segment_method="Fuzzy C-Means", segment_p=p_seg, segment_v=v_seg,
                   fcm=dict(pc=fc.partition_coefficient(), pe=fc.partition_entropy(), xb=fc.xie_beni(), m=fc.m),
                   sugeno=dict(auc=sugeno_row["auc"], alpha=alpha, blend=blend,
                               rules=[dict(aturan=n, z=float(v)) for n, v in zip(fs.RULE_NAMES, z)]))
    ids = d["customerID"].tolist()
    scores = dict(prob=dict(zip(ids, oof)), model=dict(zip(ids, oof_m)), fuzzy=dict(zip(ids, oof_s)))
    return bundle, metrics, scores, dict(zip(ids, labels)), dict(zip(ids, U.tolist())), profiles


def predict_proba(bundle, d):
    return bundle["clf"].predict_proba(d[FEATURES])[:, 1]


def assign_segments(bundle, d):
    """Kembalikan (segmen dominan, matriks derajat keanggotaan terurut sesuai Segmen A, B, ...)."""
    sg = bundle["seg"]
    U_raw = fcm_memberships(_fuzzy_X(d, sg), sg["centers"], sg["m"])
    U = np.zeros_like(U_raw)
    for c, r in sg["order"].items():
        U[:, r] = U_raw[:, c]
    return U.argmax(axis=1), U


def score_all(bundle, d):
    pm = predict_proba(bundle, d)
    ps = fs.score(d, bundle["sugeno"]["z"])
    a = bundle["sugeno"]["alpha"]
    labels, U = assign_segments(bundle, d)
    return dict(prob=(1 - a) * pm + a * ps, model=pm, fuzzy=ps, seg=labels, U=U)


def explain(bundle, row):
    """Faktor pendorong: selisih probabilitas bila fitur diganti dengan nilai pelanggan 'tipikal bertahan'."""
    ref = bundle["ref"]
    base = pd.DataFrame([{f: row[f] for f in FEATURES}])
    alts = []
    for f in FEATURES:
        a = base.copy()
        a[f] = ref[f]
        alts.append(a)
    p0 = float(bundle["clf"].predict_proba(base)[:, 1][0])
    pa = bundle["clf"].predict_proba(pd.concat(alts, ignore_index=True))[:, 1]
    out = []
    for f, p in zip(FEATURES, pa):
        delta = p0 - float(p)
        if abs(delta) >= 0.005:
            val = row[f]
            if isinstance(val, (int, float, np.integer, np.floating)):
                val = f"{val:.0f} bulan" if f == "tenure" else "USD " + f"{val:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")
            out.append(dict(fitur=LABEL[f], nilai=VAL_ID.get(val, val), delta=delta))
    up = sorted([o for o in out if o["delta"] > 0], key=lambda o: -o["delta"])[:4]
    down = sorted([o for o in out if o["delta"] < 0], key=lambda o: o["delta"])[:3]
    return up, down
