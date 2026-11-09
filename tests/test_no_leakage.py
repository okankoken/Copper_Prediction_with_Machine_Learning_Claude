"""Leakage and consistency tests.  Run:  python -m unittest discover -s tests -t . -v"""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from copperlab import config as C
from copperlab import dekad as K
from copperlab.data import apply_publication_lags, copper_dekads, exogenous_raw
from copperlab.experts import run_origin, training_rows, tune
from copperlab.features import broad_features, own_features, sentiment_features, targets
from copperlab.news import dekad_sentiment

RUN_DATE = date(2026, 11, 10)


class TestDekads(unittest.TestCase):
    def test_roundtrip_and_boundaries(self):
        d = pd.DatetimeIndex(["2024-02-10", "2024-02-11", "2024-02-20", "2024-02-21", "2024-02-29", "2024-03-01"])
        n = K.from_dates(d)
        self.assertEqual([K.parts(int(x))[2] for x in n], [1, 2, 2, 3, 3, 1])
        self.assertEqual(int(n[-1]) - int(n[-2]), 1)
        self.assertEqual(K.end(int(n[4])), pd.Timestamp("2024-02-29"))
        self.assertEqual(K.ref_month(K.make(2024, 2, 3)), pd.Period("2024-02", "M"))
        self.assertEqual(K.ref_month(K.make(2024, 2, 2)), pd.Period("2024-01", "M"))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cu = copper_dekads(RUN_DATE)
        cls.months = pd.period_range(K.ref_month(int(cls.cu.index[0])), K.ref_month(int(cls.cu.index[-1])), freq="M")
        cls.raw = exogenous_raw(cls.months)
        cls.exo = apply_publication_lags(cls.raw, cls.months)
        cls.news = dekad_sentiment(int(cls.cu.index[-1]))
        cls.F, cls.Y = own_features(cls.cu), targets(cls.cu)
        cls.B = broad_features(cls.exo, cls.cu.index)
        cls.S = sentiment_features(cls.news, cls.cu.index)

    def _perturb_future(self, t):
        cu2 = self.cu.copy()
        fut = cu2.index > t
        cu2.loc[fut, ["price", "close", "f3m"]] *= 3.0
        cu2.loc[fut, "stock"] *= 0.1
        exo2 = self.exo.copy()
        later = exo2.index > K.ref_month(t)
        exo2.loc[later] = exo2.loc[later] * 5
        S2 = self.S.copy()
        S2.loc[S2.index > t] = S2.loc[S2.index > t] * -7
        return own_features(cu2), broad_features(exo2, cu2.index), S2, targets(cu2)


class TestFeatures(Base):
    def test_origin_is_last_completed_dekad(self):
        self.assertEqual(K.label(int(self.cu.index[-1])), "2026-10-3")

    def test_features_do_not_change_when_future_is_removed(self):
        for t in (K.make(2013, 6, 2), K.make(2018, 3, 3), K.make(2022, 11, 1), K.make(2026, 9, 2)):
            cu_t = self.cu.loc[:t]
            months_t = self.months[self.months <= K.ref_month(t)]
            exo_t = apply_publication_lags(self.raw.loc[:K.ref_month(t)], months_t)
            pd.testing.assert_series_equal(own_features(cu_t).loc[t], self.F.loc[t], check_names=False)
            pd.testing.assert_series_equal(broad_features(exo_t, cu_t.index).loc[t], self.B.loc[t], check_names=False)
            s_t = sentiment_features(self.news[self.news.index <= t], cu_t.index)
            pd.testing.assert_series_equal(s_t.loc[t], self.S.loc[t], check_names=False)

    def test_macro_uses_only_completed_months(self):
        t = K.make(2020, 6, 2)          # 11-20 June: June is not finished, so May is the latest month
        name = C.SERIES[0].name + "__c3"
        self.assertEqual(self.B.loc[t, name], self.B.loc[K.make(2020, 5, 3), name])


class TestExperts(Base):
    def test_training_targets_are_realised_at_origin(self):
        t = K.make(2021, 5, 2)
        for h in C.HORIZONS:
            rows = training_rows(self.F.index, self.Y[h], t, h)
            self.assertLessEqual(int(rows.max()) + h, t)
            self.assertTrue(self.Y[h].loc[rows].notna().all())

    def test_future_data_cannot_move_a_past_forecast(self):
        t = K.make(2021, 8, 3)
        base, _ = run_origin(t, self.F, self.B, self.S, self.Y)
        moved, _ = run_origin(t, *self._perturb_future(t))
        a = pd.DataFrame(base).set_index(["horizon", "model"])["yhat"]
        b = pd.DataFrame(moved).set_index(["horizon", "model"])["yhat"]
        pd.testing.assert_series_equal(a, b)

    def test_future_data_cannot_change_tuning(self):
        T = K.make(2019, 12, 3)
        self.assertEqual(tune(T, 9, self.F, self.B, self.S, self.Y), tune(T, 9, *self._perturb_future(T)))

    def test_forecast_is_mean_of_ml_experts(self):
        recs, _ = run_origin(K.make(2023, 4, 1), self.F, self.B, self.S, self.Y)
        p = pd.DataFrame(recs).pivot(index="horizon", columns="model", values="yhat")
        ml = p[[m for m in C.ML_EXPERTS if m in p]].mean(axis=1)
        np.testing.assert_allclose(p[C.FORECAST], ml)


if __name__ == "__main__":
    unittest.main()
