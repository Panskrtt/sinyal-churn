"""Fuzzy C-Means (Bezdek, 1981).

Tiap objek punya derajat keanggotaan u_ik di setiap kluster, dengan sum_k u_ik = 1.
Fungsi tujuan:  J = sum_i sum_k u_ik^m * ||x_i - c_k||^2
Pembaruan:      c_k = sum_i u_ik^m x_i / sum_i u_ik^m
                u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
"""
import numpy as np


def sqdist(X, C):
    d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
    return np.maximum(d, 1e-12)


def memberships(X, C, m=2.0):
    inv = sqdist(X, C) ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, eps=1e-5, max_iter=300, n_init=5, random_state=42):
        self.n_clusters, self.m, self.eps = n_clusters, m, eps
        self.max_iter, self.n_init, self.random_state = max_iter, n_init, random_state

    def _run(self, X, rng):
        U = rng.random((len(X), self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)
        for _ in range(self.max_iter):
            W = U ** self.m
            C = (W.T @ X) / W.sum(axis=0)[:, None]
            U_new = memberships(X, C, self.m)
            done = np.abs(U_new - U).max() < self.eps
            U = U_new
            if done:
                break
        J = float(((U ** self.m) * sqdist(X, C)).sum())
        return J, U, C

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.random_state)
        best = min((self._run(X, rng) for _ in range(self.n_init)), key=lambda r: r[0])
        self.J_, self.U_, self.centers_ = best
        return self

    def predict_memberships(self, X):
        return memberships(np.asarray(X, float), self.centers_, self.m)

    def partition_coefficient(self):          # makin besar makin tegas (maks 1)
        return float((self.U_ ** 2).sum(axis=1).mean())

    def partition_entropy(self):              # makin kecil makin tegas
        U = np.clip(self.U_, 1e-12, 1)
        return float(-(U * np.log(U)).sum(axis=1).mean())

    def xie_beni(self):                       # makin kecil makin baik
        C = self.centers_
        d = sqdist(C, C)
        np.fill_diagonal(d, np.inf)
        return float(self.J_ / (len(self.U_) * d.min()))
