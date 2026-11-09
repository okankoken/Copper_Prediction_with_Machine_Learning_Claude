"""Earn-your-weight combination and empirical (conformal-style) intervals.

At origin t and horizon h only forecasts whose target month is <= t (already
realised) are used, both for expert weights and for interval widths.
Until enough such forecasts exist the combination is 100% the anchor rule:
an expert must prove itself before it moves the published number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def expert_weights(hist: pd.DataFrame, h: int, models=C.MODELS) -> tuple[dict, int]:
    """Earn-your-weight: inverse-MSE weights over the last max(COMBO_LOOKBACK, 12*h)
    matured forecasts, restricted to the anchor (month-end price moved along the
    LME forward curve, the strongest simple benchmark) plus experts whose trailing MSE beat it. Anything
    that has not beaten the anchor recently gets exactly 0.

    h-month forecasts made in consecutive months overlap, so n of them hold only
    about n/h independent outcomes. Weights are only estimated once that
    effective sample reaches COMBO_MIN_EFFECTIVE."""
    need = C.COMBO_MIN_EFFECTIVE * h
    hist = hist.dropna(subset=["y", *models]).sort_values("target").tail(max(C.COMBO_LOOKBACK, need))
    if len(hist) < need:
        return {m: float(m == C.ANCHOR) for m in models}, len(hist)
    mse = {m: float(np.mean((hist[m] - hist["y"]) ** 2)) for m in models}
    eligible = [m for m in models if m == C.ANCHOR or mse[m] < mse[C.ANCHOR]]
    inv = {m: 1.0 / max(mse[m], 1e-12) for m in eligible}
    total = sum(inv.values())
    return {m: inv.get(m, 0.0) / total for m in models}, len(hist)


def combine(preds: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = preds.pivot_table(index=["horizon", "origin"], columns="model", values="yhat").reset_index()
    meta = preds.groupby(["horizon", "origin"], as_index=False).agg(target=("target", "first"), y=("y", "first"))
    wide = wide.merge(meta, on=["horizon", "origin"])
    models = [m for m in C.MODELS if m in wide.columns]

    combo_rows, weight_rows = [], []
    for h, g in wide.groupby("horizon"):
        g = g.sort_values("origin").reset_index(drop=True)
        for _, row in g.iterrows():
            t = row["origin"]
            hist = g[(g["target"] <= t)]
            w, n_w = expert_weights(hist, int(h), models)
            yhat = float(sum(w[m] * row[m] for m in models))
            combo_rows.append({"horizon": h, "origin": t, "target": row["target"], "y": row["y"],
                               "yhat": yhat, "n_weight_obs": n_w})
            weight_rows.append({"horizon": h, "origin": t, "n_weight_obs": n_w, **{f"w_{m}": w[m] for m in models}})
    combo = pd.DataFrame(combo_rows)
    weights = pd.DataFrame(weight_rows)
    combo = add_intervals(combo, Y)
    return combo, weights


def add_intervals(combo: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of log return = yhat + quantiles of matured combination errors,
    re-centred on their median so the predictive median equals the point forecast
    (no hidden drift from the 2015-2026 copper uptrend enters the bands).

    Fallback (too few matured errors): the distribution of past h-month copper
    returns realised by t, centred the same way.
    """
    out = []
    for h, g in combo.groupby("horizon"):
        g = g.sort_values("origin").copy()
        g["err"] = g["y"] - g["yhat"]
        qcols = {q: [] for q in C.QUANTILES}
        p_up, method, n_err = [], [], []
        for _, row in g.iterrows():
            t = row["origin"]
            matured = g.loc[(g["target"] <= t), "err"].dropna().values
            if len(matured) >= C.CONFORMAL_MIN:
                errs, m = matured, "empirical_errors"
            else:
                past = Y[h]
                errs, m = past[(past.index + h) <= t].dropna().values, "historical_returns"
                if len(errs) == 0:
                    errs, m = np.zeros(1), "no_history"
            sims = row["yhat"] + errs - np.median(errs)
            for q in C.QUANTILES:
                qcols[q].append(float(np.quantile(sims, q)))
            p_up.append(float(np.mean(sims > 0)))
            method.append(m)
            n_err.append(len(errs))
        for q in C.QUANTILES:
            g[f"q{int(q * 100):02d}"] = qcols[q]
        g["p_up"] = p_up
        g["interval_method"] = method
        g["n_interval_obs"] = n_err
        out.append(g)
    return pd.concat(out, ignore_index=True)
