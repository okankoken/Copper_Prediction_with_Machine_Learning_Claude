"""Earn-your-weight combination.

At origin t and horizon h only forecasts whose target month is <= t (already
realised) are used. Each expert is compared with the anchor rule (month-end
price moved along the LME forward curve) on the matured forecasts they share:
  * it needs at least 12*h of them (h-month forecasts overlap, so n forecasts
    hold only ~n/h independent outcomes; 12 independent outcomes are required),
  * only the most recent max(COMBO_LOOKBACK, 12*h) are used,
  * it gets weight only if its MSE is below the anchor's on that sample,
    weight proportional to MSE(anchor) / MSE(expert); the anchor has weight 1.
Experts without a forecast at t (e.g. the news expert before 2020) get 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def expert_weights(hist: pd.DataFrame, h: int, models=C.MODELS, available=None) -> tuple[dict, int]:
    need = C.COMBO_MIN_EFFECTIVE * h
    look = max(C.COMBO_LOOKBACK, need)
    hist = hist.sort_values("target")
    score = {C.ANCHOR: 1.0}
    n_anchor = int(hist[["y", C.ANCHOR]].notna().all(axis=1).sum())
    for m in models:
        if m == C.ANCHOR or (available is not None and not available.get(m, False)):
            continue
        both = hist.dropna(subset=["y", m, C.ANCHOR]).tail(look)
        if len(both) < need:
            continue
        mse_m = float(np.mean((both[m] - both["y"]) ** 2))
        mse_a = float(np.mean((both[C.ANCHOR] - both["y"]) ** 2))
        if mse_m < mse_a:
            score[m] = mse_a / max(mse_m, 1e-12)
    total = sum(score.values())
    return {m: score.get(m, 0.0) / total for m in models}, n_anchor


def combine(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = preds[preds["model"].isin(C.MODELS)]
    wide = base.pivot_table(index=["horizon", "origin"], columns="model", values="yhat").reset_index()
    meta = base.groupby(["horizon", "origin"], as_index=False).agg(target=("target", "first"), y=("y", "first"))
    wide = wide.merge(meta, on=["horizon", "origin"])
    models = [m for m in C.MODELS if m in wide.columns]

    combo_rows, weight_rows = [], []
    for h, g in wide.groupby("horizon"):
        g = g.sort_values("origin").reset_index(drop=True)
        for _, row in g.iterrows():
            t = row["origin"]
            avail = {m: pd.notna(row[m]) for m in models}
            w, n_w = expert_weights(g[g["target"] <= t], int(h), models, avail)
            yhat = float(sum(w[m] * row[m] for m in models if w[m] > 0))
            combo_rows.append({"horizon": h, "origin": t, "target": row["target"], "y": row["y"], "yhat": yhat})
            weight_rows.append({"horizon": h, "origin": t, "n_weight_obs": n_w, **{f"w_{m}": w[m] for m in models}})
    return pd.DataFrame(combo_rows), pd.DataFrame(weight_rows)
