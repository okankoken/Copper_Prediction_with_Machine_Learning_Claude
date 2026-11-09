"""Walk-forward experts with time-series hyperparameter tuning.

`run_origin(t, ...)` simulates exactly what could be done at the end of month t:
  * features use data dated <= t (publication lags already applied),
  * PCA factors are fitted on feature rows <= t (unsupervised, no targets),
  * horizon-h models train only on rows s with s + h <= t (targets realised at t),
  * hyperparameters are tuned with purged time-series cross-validation
    (TimeSeriesSplit with gap = h) on that same training data only.
LightGBM is re-tuned once a year (every December origin) and the chosen
parameters are used for the following 12 origins; ElasticNet is tuned at every
origin. The production forecast is `run_origin` at the latest origin.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import config as C
from .features import OWN_FEATURES, SENT_FEATURES, series_of

ENET_L1 = (0.1, 0.5, 0.9, 1.0)
LGBM_SPACE = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.01, 0.03, 0.06],
    "num_leaves": [3, 7, 15],
    "min_child_samples": [8, 15, 25],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "subsample": [0.7, 1.0],
}
LGBM_DEFAULT = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 15,
                "reg_lambda": 5.0, "colsample_bytree": 0.8, "subsample": 0.8}


def _scale(train: pd.DataFrame, *others: pd.DataFrame):
    mu = train.mean()
    sd = train.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return [(train - mu) / sd] + [(o - mu) / sd for o in others]


def _lgbm(**params) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(subsample_freq=1, random_state=C.RANDOM_STATE, n_jobs=1, verbose=-1, **params)


def factor_block(B: pd.DataFrame, rows: pd.PeriodIndex, t: pd.Period):
    """PCA factors of the broad panel using only rows <= t."""
    Bt = B.loc[rows]
    coverage = Bt.notna().mean()
    cols = coverage[(coverage >= C.FACTOR_MIN_COVERAGE) & Bt.loc[t].notna()].index.tolist()
    if len(cols) < C.N_FACTORS:
        return pd.DataFrame(index=rows), None
    (Z,) = _scale(Bt[cols])
    Z = Z.fillna(0.0)
    pca = PCA(n_components=C.N_FACTORS, random_state=C.RANDOM_STATE).fit(Z.values)
    names = [f"pc{i + 1}" for i in range(C.N_FACTORS)]
    pcs = pd.DataFrame(pca.transform(Z.values), index=rows, columns=names)
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=names)
    return pcs, loadings


def design(t: pd.Period, F: pd.DataFrame, B: pd.DataFrame, S: pd.DataFrame):
    """Feature matrix for all rows <= t with complete own features."""
    avail = F.index[F.index <= t]
    own = F.loc[avail, list(OWN_FEATURES)]
    rows = avail[own.notna().all(axis=1).values]
    if t not in rows:
        return None, None, rows
    pcs, loadings = factor_block(B, rows, t)
    X = pd.concat([own.loc[rows], pcs, S.reindex(rows)[list(SENT_FEATURES)]], axis=1)
    return X, loadings, rows


def training_rows(rows: pd.PeriodIndex, y: pd.Series, t: pd.Period, h: int) -> pd.PeriodIndex:
    cand = rows[(rows + h) <= t]
    return cand[y.loc[cand].notna().values]


def purged_cv(n: int, h: int, max_splits: int) -> TimeSeriesSplit:
    """Expanding-window CV with 12-month validation blocks and an (h-1)-row purge gap,
    so no training label overlaps a validation month. At least 24 training rows."""
    gap = h - 1
    splits = int(max(2, min(max_splits, (n - gap - 24) // 12)))
    return TimeSeriesSplit(n_splits=splits, gap=gap, test_size=12)


def tune_point(t: pd.Period) -> pd.Period:
    """Last December at or before t (LightGBM is re-tuned once a year)."""
    return pd.Period(f"{t.year if t.month == 12 else t.year - 1}-12", "M")


def tune_lgbm(T: pd.Period, h: int, F, B, S, Y) -> dict:
    X, _, rows = design(T, F, B, S)
    if X is None:
        return dict(LGBM_DEFAULT)
    cand = training_rows(rows, Y[h], T, h)
    if len(cand) < C.MIN_TRAIN:
        return dict(LGBM_DEFAULT)
    cols = list(OWN_FEATURES) + [c for c in X.columns if c.startswith("pc")] + list(SENT_FEATURES)
    cv = purged_cv(len(cand), h, 3)
    search = RandomizedSearchCV(_lgbm(), LGBM_SPACE, n_iter=C.LGBM_TUNE_ITER, cv=cv,
                                scoring="neg_mean_absolute_error", random_state=C.RANDOM_STATE, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X.loc[cand, cols], Y[h].loc[cand].values)
    return dict(search.best_params_)


def _enet(Xtr: pd.DataFrame, ytr, Xt: pd.DataFrame, h: int):
    Xs, Xts = _scale(Xtr, Xt)
    cv = purged_cv(len(Xtr), h, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ElasticNetCV(l1_ratio=list(ENET_L1), n_alphas=30, cv=cv, max_iter=20000,
                         random_state=C.RANDOM_STATE).fit(Xs.values, ytr)
    return float(m.predict(Xts.values)[0]), m


def run_origin(t: pd.Period, F, B, S, Y, lgbm_params: dict | None = None, details: bool = False):
    X, loadings, rows = design(t, F, B, S)
    if X is None:
        return ([], {}) if details else []
    pc_cols = [c for c in X.columns if c.startswith("pc")]
    base_cols = list(OWN_FEATURES) + pc_cols
    lgbm_cols = base_cols + list(SENT_FEATURES)
    news_cols = list(OWN_FEATURES) + list(SENT_FEATURES)

    records, info = [], {}
    for h in C.HORIZONS:
        y = Y[h]
        cand = training_rows(rows, y, t, h)
        if len(cand) < C.MIN_TRAIN:
            continue
        ytr = y.loc[cand].values
        eom, spread = float(F.loc[t, "eom_gap"]), float(F.loc[t, "spread_3m"])
        preds = {
            "rw": 0.0,
            "rw_spot": eom,
            # month-end price moved along the LME cash -> 3-month curve, flat beyond the tenor
            "forward": eom + spread * min(h, C.FORWARD_TENOR) / C.FORWARD_TENOR,
        }
        preds["enet"], enet_m = _enet(X.loc[cand, base_cols], ytr, X.loc[[t], base_cols], h)

        params = (lgbm_params or {}).get(h, LGBM_DEFAULT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gbm = _lgbm(**params).fit(X.loc[cand, lgbm_cols], ytr)
            preds["lgbm"] = float(gbm.predict(X.loc[[t], lgbm_cols])[0])

        # news expert: only months where sentiment exists (2017+)
        news_rows = cand[X.loc[cand, list(SENT_FEATURES)].notna().all(axis=1).values]
        if len(news_rows) >= C.MIN_TRAIN_NEWS and X.loc[t, list(SENT_FEATURES)].notna().all():
            preds["enet_news"], news_m = _enet(X.loc[news_rows, news_cols], y.loc[news_rows].values,
                                               X.loc[[t], news_cols], h)
        else:
            news_m = None

        actual = y.loc[t]
        for model, value in preds.items():
            records.append({
                "origin": t, "horizon": h, "target": t + h, "model": model,
                "yhat": value, "y": float(actual) if pd.notna(actual) else np.nan,
                "n_train": len(cand), "train_first": cand[0], "train_last": cand[-1],
            })
        if details:
            info[h] = {
                "enet_coef": pd.Series(enet_m.coef_, index=base_cols),
                "enet_alpha": float(enet_m.alpha_), "enet_l1": float(enet_m.l1_ratio_),
                "news_coef": pd.Series(news_m.coef_, index=news_cols) if news_m is not None else None,
                "lgbm_params": params,
                "lgbm_gain": pd.Series(gbm.booster_.feature_importance("gain"), index=lgbm_cols),
                "loadings": loadings,
            }
    return (records, info) if details else records


def pc_labels(loadings: pd.DataFrame | None, top: int = 3) -> dict[str, str]:
    """Human-readable description of each factor from its largest loadings."""
    if loadings is None:
        return {}
    labels = {}
    for pc in loadings.columns:
        s = loadings[pc]
        best = s.abs().sort_values(ascending=False).index[:top]
        labels[pc] = ", ".join(f"{'+' if s[c] > 0 else '-'}{series_of(c)}" for c in best)
    return labels
