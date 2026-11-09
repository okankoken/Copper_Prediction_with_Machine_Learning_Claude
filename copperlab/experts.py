"""Walk-forward ML experts at dekad resolution, with time-series hyperparameter tuning.

`run_origin(t, ...)` simulates exactly what could be done at the end of dekad t:
  * features use data dated <= end of t (monthly macro data with publication lags),
  * horizon-h models train only on dekads s with s + h <= t (targets realised at t),
  * hyperparameters are tuned once a year (at each December's last dekad) with
    purged time-series cross-validation (gap = h-1) on data available then, at
    the horizons in TUNE_HORIZONS; every horizon uses the nearest tuned one.
Inputs enter the models directly (no PCA), so each input's effect can be read off.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import config as C
from . import dekad as K
from .features import OWN_FEATURES, SENT_FEATURES

ENET_L1 = (0.1, 0.5, 0.9, 1.0)
LGBM_SPACE = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.01, 0.03, 0.06],
    "num_leaves": [3, 7, 15],
    "min_child_samples": [10, 20, 40],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    "colsample_bytree": [0.3, 0.6, 0.9],
    "subsample": [0.7, 1.0],
}
LGBM_DEFAULT = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 20,
                "reg_lambda": 5.0, "colsample_bytree": 0.6, "subsample": 0.8}
ENET_DEFAULT = {"alpha": 0.01, "l1_ratio": 0.5}


def _lgbm(**params) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(subsample_freq=1, random_state=C.RANDOM_STATE, n_jobs=1, verbose=-1, **params)


def purged_cv(n: int, h: int, max_splits: int) -> TimeSeriesSplit:
    """Expanding-window CV, 1-year (36-dekad) validation blocks, (h-1)-dekad purge gap
    so no training label overlaps a validation period; at least 2 years of training."""
    gap = h - 1
    splits = int(max(2, min(max_splits, (n - gap - 72) // 36)))
    return TimeSeriesSplit(n_splits=splits, gap=gap, test_size=36)


def tune_point(t: int) -> int:
    """Last dekad of the latest December at or before t."""
    y, m, k = K.parts(t)
    return K.make(y if (m == 12 and k == 3) else y - 1, 12, 3)


def nearest_tuned(h: int) -> int:
    return min(C.TUNE_HORIZONS, key=lambda x: (abs(x - h), x))


class Design:
    """Inputs available at origin t, with standardisation fitted on rows <= t."""

    def __init__(self, t: int, F: pd.DataFrame, B: pd.DataFrame, S: pd.DataFrame):
        avail = F.index[F.index <= t]
        own = F.loc[avail, list(OWN_FEATURES)]
        self.rows = avail[own.notna().all(axis=1).values]
        self.ok = t in self.rows
        if not self.ok:
            return
        Bt = B.loc[self.rows]
        cov = Bt.notna().mean()
        self.macro = cov[(cov >= C.COVERAGE) & Bt.loc[t].notna()].index.tolist()
        self.X = pd.concat([own.loc[self.rows], Bt[self.macro], S.reindex(self.rows)[list(SENT_FEATURES)]], axis=1)
        self.linear = list(OWN_FEATURES) + self.macro
        self.tree = self.linear + list(SENT_FEATURES)
        self.news = list(OWN_FEATURES) + list(SENT_FEATURES)

    def z(self, cols, train_rows, rows):
        tr = self.X.loc[train_rows, cols]
        mu, sd = tr.mean(), tr.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return ((tr - mu) / sd).fillna(0.0), ((self.X.loc[rows, cols] - mu) / sd).fillna(0.0)


def training_rows(rows: pd.Index, y: pd.Series, t: int, h: int) -> pd.Index:
    cand = rows[(rows + h) <= t]
    return cand[y.loc[cand].notna().values]


def tune(T: int, h: int, F, B, S, Y) -> dict:
    """Best ElasticNet (alpha, l1_ratio) and LightGBM parameters at tuning point T, horizon h."""
    d = Design(T, F, B, S)
    if not d.ok:
        return {"enet": dict(ENET_DEFAULT), "lgbm": dict(LGBM_DEFAULT)}
    cand = training_rows(d.rows, Y[h], T, h)
    if len(cand) < C.MIN_TRAIN:
        return {"enet": dict(ENET_DEFAULT), "lgbm": dict(LGBM_DEFAULT)}
    y = Y[h].loc[cand].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xl, _ = d.z(d.linear, cand, cand)
        e = ElasticNetCV(l1_ratio=list(ENET_L1), n_alphas=30, cv=purged_cv(len(cand), h, 4), max_iter=5000,
                         random_state=C.RANDOM_STATE).fit(Xl.values, y)
        search = RandomizedSearchCV(_lgbm(), LGBM_SPACE, n_iter=C.LGBM_TUNE_ITER, cv=purged_cv(len(cand), h, 3),
                                    scoring="neg_mean_absolute_error", random_state=C.RANDOM_STATE, n_jobs=1)
        search.fit(d.X.loc[cand, d.tree], y)
    return {"enet": {"alpha": float(e.alpha_), "l1_ratio": float(e.l1_ratio_)}, "lgbm": dict(search.best_params_)}


def _enet(params: dict) -> ElasticNet:
    return ElasticNet(**params, max_iter=5000, random_state=C.RANDOM_STATE)


def run_origin(t: int, F, B, S, Y, params: dict | None = None, details: bool = False):
    """Forecasts of every model at origin t for all horizons; optionally coefficients
    and SHAP contributions for the input-effects page."""
    d = Design(t, F, B, S)
    if not d.ok:
        return ([], {}) if details else []
    params = params or {}
    eod, spread = float(F.loc[t, "eod_gap"]), float(F.loc[t, "spread_3m"])
    news_ok = bool(d.X.loc[t, list(SENT_FEATURES)].notna().all())
    records, info = [], {}
    for h in C.HORIZONS:
        y = Y[h]
        cand = training_rows(d.rows, y, t, h)
        if len(cand) < C.MIN_TRAIN:
            continue
        ytr = y.loc[cand].values
        p = params.get(nearest_tuned(h), {"enet": ENET_DEFAULT, "lgbm": LGBM_DEFAULT})
        preds = {"rw": 0.0, "forward": eod + spread * min(h, C.FORWARD_TENOR) / C.FORWARD_TENOR}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtr, Xt = d.z(d.linear, cand, [t])
            enet = _enet(p["enet"]).fit(Xtr.values, ytr)
            preds["enet"] = float(enet.predict(Xt.values)[0])

            gbm = _lgbm(**p["lgbm"]).fit(d.X.loc[cand, d.tree], ytr)
            preds["lgbm"] = float(gbm.predict(d.X.loc[[t], d.tree])[0])

            news_rows = cand[d.X.loc[cand, list(SENT_FEATURES)].notna().all(axis=1).values]
            news = None
            if news_ok and len(news_rows) >= C.MIN_TRAIN_NEWS:
                Ntr, Nt = d.z(d.news, news_rows, [t])
                news = _enet(p["enet"]).fit(Ntr.values, y.loc[news_rows].values)
                preds["enet_news"] = float(news.predict(Nt.values)[0])

        preds[C.FORECAST] = float(np.mean([preds[m] for m in C.ML_EXPERTS if m in preds]))
        actual = y.loc[t]
        for model, value in preds.items():
            records.append({"origin": t, "horizon": h, "target": t + h, "model": model, "yhat": value,
                            "y": float(actual) if pd.notna(actual) else np.nan, "n_train": len(cand)})

        if h in C.DRIVER_HORIZONS:
            coef = pd.Series(enet.coef_, index=d.linear)
            row = {"coef": coef}
            if details:
                n_models = 2 + (news is not None)
                contrib = (coef * Xt.iloc[0]).reindex(d.tree).fillna(0.0)
                shap = gbm.predict(d.X.loc[[t], d.tree], pred_contrib=True)[0][:-1]
                contrib = contrib + pd.Series(shap, index=d.tree)
                if news is not None:
                    contrib = contrib.add((pd.Series(news.coef_, index=d.news) * Nt.iloc[0]), fill_value=0.0)
                row["contrib"] = contrib / n_models
                row["gain"] = pd.Series(gbm.booster_.feature_importance("gain"), index=d.tree)
            info[h] = row
    return records, info
