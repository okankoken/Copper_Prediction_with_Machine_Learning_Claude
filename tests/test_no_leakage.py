"""Leakage and consistency tests.  Run:  python -m unittest discover -s tests -t . -v"""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from copperlab import config as C
from copperlab.combine import combine, expert_weights
from copperlab.data import apply_publication_lags, copper_monthly, exogenous_raw
from copperlab.experts import run_origin, tune_lgbm
from copperlab.features import broad_features, own_features, sentiment_features, targets
from copperlab.news import monthly_sentiment

RUN_DATE = date(2026, 11, 9)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cu = copper_monthly(RUN_DATE)
        cls.raw = exogenous_raw(cls.cu.index)
        cls.exo = apply_publication_lags(cls.raw, cls.cu.index)
        cls.sent = monthly_sentiment(cls.cu.index[-1])
        cls.F, cls.B, cls.Y = own_features(cls.cu), broad_features(cls.exo), targets(cls.cu)
        cls.S = sentiment_features(cls.sent, cls.cu.index)

    def _perturb_future(self, t):
        cu2 = self.cu.copy()
        future = cu2.index > t
        cu2.loc[future, ["price", "close", "f3m"]] *= 3.0
        cu2.loc[future, "stock"] *= 0.1
        exo2 = self.exo.copy()
        exo2.loc[exo2.index > t] = exo2.loc[exo2.index > t] * 5
        S2 = self.S.copy()
        S2.loc[S2.index > t] = S2.loc[S2.index > t] * -7
        return own_features(cu2), broad_features(exo2), S2, targets(cu2)


class TestFeatures(Base):
    def test_features_do_not_change_when_future_is_removed(self):
        for t in [pd.Period(m, "M") for m in ("2013-06", "2018-03", "2022-11", "2026-09")]:
            cu_t = self.cu.loc[:t]
            exo_t = apply_publication_lags(self.raw.loc[:t], cu_t.index)
            pd.testing.assert_series_equal(own_features(cu_t).loc[t], self.F.loc[t], check_names=False)
            pd.testing.assert_series_equal(broad_features(exo_t).loc[t], self.B.loc[t], check_names=False)
            s_t = sentiment_features(self.sent[self.sent.index <= t], cu_t.index)
            pd.testing.assert_series_equal(s_t.loc[t], self.S.loc[t], check_names=False)

    def test_publication_lag_is_applied(self):
        t = pd.Period("2020-06", "M")
        for s in C.SERIES:
            src = self.raw[s.name].loc[t - s.lag]
            if pd.notna(src):
                self.assertAlmostEqual(self.exo.loc[t, s.name], src, msg=s.name)

    def test_running_month_is_excluded(self):
        self.assertEqual(self.cu.index[-1], pd.Period("2026-10", "M"))


class TestExperts(Base):
    def test_training_targets_are_realised_at_origin(self):
        t = pd.Period("2021-05", "M")
        for r in run_origin(t, self.F, self.B, self.S, self.Y):
            self.assertLessEqual(r["train_last"] + r["horizon"], t)

    def test_future_data_cannot_move_a_past_forecast(self):
        t = pd.Period("2021-08", "M")
        base = pd.DataFrame(run_origin(t, self.F, self.B, self.S, self.Y)).set_index(["horizon", "model"])["yhat"]
        moved = pd.DataFrame(run_origin(t, *self._perturb_future(t))).set_index(["horizon", "model"])["yhat"]
        pd.testing.assert_series_equal(base, moved)

    def test_future_data_cannot_change_tuning(self):
        T = pd.Period("2019-12", "M")
        self.assertEqual(tune_lgbm(T, 3, self.F, self.B, self.S, self.Y),
                         tune_lgbm(T, 3, *self._perturb_future(T)))


class TestCombination(unittest.TestCase):
    def _synthetic(self, n=200, h=1, seed=0):
        rng = np.random.default_rng(seed)
        origins = pd.period_range("2005-01", periods=n, freq="M")
        y = rng.normal(0, 0.05, n)
        rows = []
        for i, t in enumerate(origins):
            for m in C.MODELS:
                noise = {C.ANCHOR: 0.04, "enet": 0.01}.get(m, 0.08)
                rows.append({"origin": t, "horizon": h, "target": t + h, "model": m,
                             "yhat": 0.0 if m == "rw" else y[i] + rng.normal(0, noise), "y": y[i]})
        return pd.DataFrame(rows)

    @staticmethod
    def _wide(preds):
        wide = preds.pivot_table(index="origin", columns="model", values="yhat")
        meta = preds.groupby("origin")[["y", "target"]].first()
        return wide.join(meta).reset_index()

    def test_unmatured_outcomes_do_not_affect_weights(self):
        preds = self._synthetic()
        t = preds["origin"].unique()[120]
        _, w1 = combine(preds)
        bad = preds.copy()
        bad.loc[(bad["target"] > t) & (bad["model"] == "enet"), "yhat"] += 10.0
        _, w2 = combine(bad)
        a = w1[w1.origin == t].filter(like="w_").iloc[0]
        b = w2[w2.origin == t].filter(like="w_").iloc[0]
        pd.testing.assert_series_equal(a, b)

    def test_expert_must_beat_anchor_to_get_weight(self):
        w, _ = expert_weights(self._wide(self._synthetic()), 1)
        self.assertGreater(w["enet"], 0.5)
        self.assertEqual(w["lgbm"], 0.0)
        self.assertEqual(w["rw"], 0.0)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_too_little_evidence_means_anchor_only(self):
        w, _ = expert_weights(self._wide(self._synthetic(n=40)), 6)   # needs 72 matured, has 40
        self.assertEqual(w[C.ANCHOR], 1.0)

    def test_missing_expert_gets_no_weight(self):
        preds = self._synthetic()
        preds = preds[~((preds["model"] == "enet") & (preds["origin"] == preds["origin"].max()))]
        _, w = combine(preds)
        self.assertEqual(w.sort_values("origin")["w_enet"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
