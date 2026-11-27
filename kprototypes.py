"""K-Prototypes (Huang, 1998) untuk data campuran numerik + kategorikal.

Jarak objek x ke prototipe c:
    d(x, c) = sum_j (x_j - c_j)^2  +  gamma * sum_l [x_l != c_l]
Pusat numerik = rata-rata, pusat kategorikal = modus (nilai terbanyak).
"""
import numpy as np


class KPrototypes:
    def __init__(self, n_clusters=3, gamma=None, n_init=5, max_iter=50, random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def _dist(self, Xn, Xc, Cn, Cc):
        # (n, k): jarak Euclidean kuadrat numerik + gamma * ketidakcocokan kategorikal
        d = ((Xn[:, None, :] - Cn[None, :, :]) ** 2).sum(axis=2)
        mism = np.zeros_like(d)
        for j in range(Xc.shape[1]):
            mism += (Xc[:, j][:, None] != Cc[:, j][None, :])
        return d + self.gamma * mism

    @staticmethod
    def _mode(col, n_cat):
        return np.bincount(col, minlength=n_cat).argmax()

    def _run(self, Xn, Xc, rng):
        n = Xn.shape[0]
        k = self.n_clusters
        idx = rng.choice(n, k, replace=False)
        Cn, Cc = Xn[idx].copy(), Xc[idx].copy()
        labels = np.full(n, -1)
        for _ in range(self.max_iter):
            D = self._dist(Xn, Xc, Cn, Cc)
            new = D.argmin(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                m = labels == c
                if not m.any():  # kluster kosong: ambil titik terjauh
                    far = D[np.arange(n), labels].argmax()
                    Cn[c], Cc[c] = Xn[far], Xc[far]
                    continue
                Cn[c] = Xn[m].mean(axis=0)
                for j in range(Xc.shape[1]):
                    Cc[c, j] = self._mode(Xc[m, j], self.n_cats_[j])
        D = self._dist(Xn, Xc, Cn, Cc)
        labels = D.argmin(axis=1)
        cost = D[np.arange(n), labels].sum()
        return cost, labels, Cn, Cc

    def fit(self, Xn, Xc):
        Xn = np.asarray(Xn, float)
        Xc = np.asarray(Xc, int)
        self.n_cats_ = [int(Xc[:, j].max()) + 1 for j in range(Xc.shape[1])]
        if self.gamma is None:  # saran Huang: 0.5 * rata-rata simpangan baku numerik
            self.gamma = 0.5 * float(Xn.std(axis=0).mean())
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            res = self._run(Xn, Xc, rng)
            if best is None or res[0] < best[0]:
                best = res
        self.cost_, self.labels_, self.centers_num_, self.centers_cat_ = best
        return self

    def predict(self, Xn, Xc):
        D = self._dist(np.asarray(Xn, float), np.asarray(Xc, int), self.centers_num_, self.centers_cat_)
        return D.argmin(axis=1)
