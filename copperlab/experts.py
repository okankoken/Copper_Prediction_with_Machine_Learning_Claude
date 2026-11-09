"""Walk-forward experts.

`run_origin(t, ...)` simulates exactly what could be done at the end of month t:
  * features use data dated <= t (publication lags already applied),
  * PCA factors are fitted on feature rows <= t (unsupervised, no targets),
  * horizon-h models train only on rows s with s + h <= t, i.e. targets that
    had already been realised at t.
The production forecast is simply `run_origin` at the latest origin, so the
backtest and production use the same code path.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from . import config as C
from .features import OWN_FEATURES, series_of

ALPHAS = np.logspace(0, 4, 25)


def _scale(train: pd.DataFrame, *others: pd.DataFrame):
    mu = train.mean()
    sd = train.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return [(train - mu) / sd] + [(o - mu) / sd for o in others]


def _gbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, random_state=C.RANDOM_STATE, n_jobs=1, verbose=-1,
    )


def factor_block(B: pd.DataFrame, rows: pd.PeriodIndex, t: pd.Period):
    """PCA factors of the broad panel using only rows <= t."""
    Bt = B.loc[rows]
    coverage = Bt.notna().mean()
    cols = coverage[(coverage >= C.FACTOR_MIN_COVERAGE) & Bt.loc[t].notna()].index.tolist()
    if len(cols) < C.N_FACTORS:
        return pd.DataFrame(index=rows), None, cols
    (Z,) = _scale(Bt[cols])
    Z = Z.fillna(0.0)
    pca = PCA(n_components=C.N_FACTORS, random_state=C.RANDOM_STATE).fit(Z.values)
    names = [f"pc{i + 1}" for i in range(C.N_FACTORS)]
    pcs = pd.DataFrame(pca.transform(Z.values), index=rows, columns=names)
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=names)
    loadings.attrs["explained"] = pca.explained_variance_ratio_.tolist()
    return pcs, loadings, cols


def run_origin(t: pd.Period, F: pd.DataFrame, B: pd.DataFrame, Y: pd.DataFrame, details: bool = False):
    avail = F.index[F.index <= t]
    own = F.loc[avail, list(OWN_FEATURES)]
    rows = avail[own.notna().all(axis=1).values]
    if t not in rows:
        return ([], {}) if details else []
    pcs, loadings, _ = factor_block(B, rows, t)
    X = pd.concat([own.loc[rows], pcs], axis=1)
    own_cols = list(OWN_FEATURES)
    fac_cols = own_cols + list(pcs.columns)

    records, info = [], {}
    for h in C.HORIZONS:
        y = Y[h]
        cand = rows[(rows + h) <= t]
        cand = cand[y.loc[cand].notna().values]
        if len(cand) < C.MIN_TRAIN:
            continue
        ytr = y.loc[cand].values
        eom, spread = float(F.loc[t, "eom_gap"]), float(F.loc[t, "spread_3m"])
        # forward: month-end price moved along the LME cash -> 3-month curve, flat beyond the tenor
        preds = {"rw": 0.0, "rw_spot": eom, "forward": eom + spread * min(h, C.FORWARD_TENOR) / C.FORWARD_TENOR}

        Xtr, Xt = _scale(X.loc[cand, own_cols], X.loc[[t], own_cols])
        own_m = RidgeCV(alphas=ALPHAS).fit(Xtr.values, ytr)
        preds["own_ridge"] = float(own_m.predict(Xt.values)[0])

        Xtr_f, Xt_f = _scale(X.loc[cand, fac_cols], X.loc[[t], fac_cols])
        fac_m = RidgeCV(alphas=ALPHAS).fit(Xtr_f.values, ytr)
        preds["factor_ridge"] = float(fac_m.predict(Xt_f.values)[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gbm = _gbm().fit(X.loc[cand, fac_cols], ytr)
            preds["gbm"] = float(gbm.predict(X.loc[[t], fac_cols])[0])

        actual = y.loc[t]
        for model, value in preds.items():
            records.append({
                "origin": t, "horizon": h, "target": t + h, "model": model,
                "yhat": value, "y": float(actual) if pd.notna(actual) else np.nan,
                "n_train": len(cand), "train_first": cand[0], "train_last": cand[-1],
            })
        if details:
            info[h] = {
                "factor_coef": pd.Series(fac_m.coef_, index=fac_cols),
                "factor_intercept": float(fac_m.intercept_),
                "factor_alpha": float(fac_m.alpha_),
                "x_scaled": Xt_f.iloc[0],
                "x_raw": X.loc[t, fac_cols],
                "gbm_importance": pd.Series(gbm.booster_.feature_importance("gain"), index=fac_cols),
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
        labels[pc] = ", ".join(f"{'+' if s[c] > 0 else '-'}{series_of(c)}({c.split('__')[1]})" for c in best)
    return labels
