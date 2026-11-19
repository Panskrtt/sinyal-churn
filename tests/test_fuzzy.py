"""Uji unit komponen fuzzy.  Jalankan:  python -m unittest tests.test_fuzzy -v"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy_sugeno as fs  # noqa: E402
import ml  # noqa: E402
from fcm import FuzzyCMeans  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "telco.csv")


class FuzzyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = ml.prepare(pd.read_csv(CSV, dtype=str, keep_default_na=False))
        cls.y = (cls.d["Churn"] == "Yes").astype(int).values

    def test_fungsi_keanggotaan(self):
        self.assertEqual(float(fs.trap(0, 0, 0, 6, 18)), 1.0)      # bahu kiri tegak
        self.assertAlmostEqual(float(fs.trap(12, 0, 0, 6, 18)), 0.5)
        self.assertEqual(float(fs.trap(30, 0, 0, 6, 18)), 0.0)

    def test_fcm_keanggotaan_berjumlah_satu(self):
        X = np.random.default_rng(0).normal(size=(300, 4))
        f = FuzzyCMeans(3).fit(X)
        self.assertTrue(np.allclose(f.U_.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(f.predict_memberships(X).sum(axis=1), 1.0))
        self.assertTrue(0 < f.partition_coefficient() <= 1)

    def test_sugeno_skor_dalam_0_1_dan_semua_pelanggan_terjangkau(self):
        self.assertTrue((fs.firing(self.d).sum(axis=1) > 0).all())
        z = fs.fit(self.d, self.y)
        s = fs.score(self.d, z)
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        self.assertGreater(np.corrcoef(s, self.y)[0, 1], 0.3)     # skor memang berkaitan dengan churn


if __name__ == "__main__":
    unittest.main(verbosity=2)
