"""Indeks risiko churn dengan Fuzzy Sugeno orde nol.

Antesedan (himpunan fuzzy dan aturan) ditentukan oleh perancang sistem; konsekuen konstan z_r
dipelajari dari data berlabel dengan kuadrat terkecil berbatas [0,1] (seperti tahap LSE pada ANFIS).

    w_r(x)  = min(derajat keanggotaan tiap premis aturan r)           (operator DAN = min)
    s(x)    = sum_r w_r(x) * z_r / sum_r w_r(x)                        (rata-rata terbobot Sugeno)
"""
import numpy as np
from scipy.optimize import lsq_linear


def trap(x, a, b, c, d):
    """Fungsi keanggotaan trapesium. a=b atau c=d menghasilkan bahu tegak."""
    x = np.asarray(x, float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (d - x) / (d - c) if d > c else (x <= d).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


# himpunan fuzzy: variabel -> {label: (a, b, c, d)}
MFS = {
    "tenure": {"baru": (0, 0, 6, 18), "menengah": (6, 18, 30, 42), "lama": (30, 48, 72, 72)},
    "MonthlyCharges": {"rendah": (0, 0, 30, 50), "sedang": (35, 55, 75, 95), "tinggi": (70, 90, 200, 200)},
    "jumlah_layanan_tambahan": {"sedikit": (0, 0, 1, 3), "banyak": (2, 4, 6, 6)},
}
VAR_LABEL = {"tenure": "Lama berlangganan (bulan)", "MonthlyCharges": "Tagihan bulanan (USD)",
             "jumlah_layanan_tambahan": "Jumlah layanan tambahan"}

# (nama aturan, fungsi pengaktifan). m = dict derajat keanggotaan fuzzy; crisp = 0/1.
RULES = [
    ("Kontrak bulanan DAN pelanggan baru", lambda m: np.minimum(m["bulanan"], m["tenure.baru"])),
    ("Kontrak bulanan DAN lama berlangganan menengah", lambda m: np.minimum(m["bulanan"], m["tenure.menengah"])),
    ("Kontrak bulanan DAN pelanggan lama", lambda m: np.minimum(m["bulanan"], m["tenure.lama"])),
    ("Kontrak 1 tahun", lambda m: m["satu_tahun"]),
    ("Kontrak 2 tahun", lambda m: m["dua_tahun"]),
    ("Fiber optic DAN tagihan tinggi DAN layanan tambahan sedikit",
     lambda m: np.minimum(np.minimum(m["fiber"], m["tagihan.tinggi"]), m["layanan.sedikit"])),
    ("Bayar electronic check DAN kontrak bulanan", lambda m: np.minimum(m["echeck"], m["bulanan"])),
    ("Layanan tambahan banyak DAN pelanggan lama", lambda m: np.minimum(m["layanan.banyak"], m["tenure.lama"])),
    ("Tanpa internet DAN tagihan rendah", lambda m: np.minimum(m["tanpa_internet"], m["tagihan.rendah"])),
    ("Tagihan sedang DAN pelanggan baru", lambda m: np.minimum(m["tagihan.sedang"], m["tenure.baru"])),
]
RULE_NAMES = [r[0] for r in RULES]


def _memberships(d):
    m = {}
    for lab, p in MFS["tenure"].items():
        m[f"tenure.{lab}"] = trap(d["tenure"].values, *p)
    for lab, p in MFS["MonthlyCharges"].items():
        m[f"tagihan.{lab}"] = trap(d["MonthlyCharges"].values, *p)
    for lab, p in MFS["jumlah_layanan_tambahan"].items():
        m[f"layanan.{lab}"] = trap(d["jumlah_layanan_tambahan"].values, *p)
    m["bulanan"] = (d["Contract"].values == "Month-to-month").astype(float)
    m["satu_tahun"] = (d["Contract"].values == "One year").astype(float)
    m["dua_tahun"] = (d["Contract"].values == "Two year").astype(float)
    m["fiber"] = (d["InternetService"].values == "Fiber optic").astype(float)
    m["tanpa_internet"] = (d["InternetService"].values == "No").astype(float)
    m["echeck"] = (d["PaymentMethod"].values == "Electronic check").astype(float)
    return m


def firing(d):
    """Matriks (n, jumlah_aturan) kekuatan pengaktifan w."""
    m = _memberships(d)
    return np.column_stack([f(m) for _, f in RULES])


def normalized(W):
    s = W.sum(axis=1, keepdims=True)
    return W / np.where(s == 0, 1, s)


def fit(d, y):
    """Pelajari konsekuen z (tiap aturan) dengan kuadrat terkecil berbatas [0,1]."""
    Phi = normalized(firing(d))
    return lsq_linear(Phi, np.asarray(y, float), bounds=(0.0, 1.0)).x


def score(d, z):
    W = firing(d)
    s = W.sum(axis=1)
    out = normalized(W) @ z
    return np.where(s == 0, float(np.mean(z)), out)


def explain(row_df, z, top=3):
    """Aturan yang paling aktif untuk satu pelanggan."""
    W = normalized(firing(row_df))[0]
    idx = np.argsort(-W)[:top]
    return [dict(aturan=RULE_NAMES[i], bobot=float(W[i]), z=float(z[i])) for i in idx if W[i] > 0.01]


def mf_curves():
    """Titik kurva keanggotaan untuk digambar (variabel -> label -> [(x, mu)])."""
    rng = {"tenure": (0, 72), "MonthlyCharges": (18, 120), "jumlah_layanan_tambahan": (0, 6)}
    out = {}
    for var, sets in MFS.items():
        lo, hi = rng[var]
        xs = np.linspace(lo, hi, 120)
        out[var] = {lab: [(float(x), float(trap(x, *p))) for x in xs] for lab, p in sets.items()}
        out[var]["_range"] = (lo, hi)
    return out
