"""Out-of-sample evaluation in USD/t, with a Diebold-Mariano test against the random walk."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def dm_test(err_model: np.ndarray, err_bench: np.ndarray, h: int) -> float:
    """One-sided p-value that the model has lower absolute error than the benchmark.

    Newey-West variance with h-1 lags (overlapping forecasts) and the
    Harvey-Leybourne-Newbold small-sample correction.
    """
    d = np.abs(err_model) - np.abs(err_bench)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan
    dc = d - d.mean()
    var = np.mean(dc * dc)
    for k in range(1, h):
        if k < n:
            var += 2 * (1 - k / h) * np.mean(dc[k:] * dc[:-k])
    if var <= 0:
        var = np.mean(dc * dc)
    stat = d.mean() / np.sqrt(var / n)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    return float(stats.t.cdf(stat * corr, df=n - 1))


def _price_errors(yhat, y, p0):
    f = p0 * np.exp(yhat)
    a = p0 * np.exp(y)
    return f, a


def evaluate(preds: pd.DataFrame, combo: pd.DataFrame, origin_price: pd.Series) -> pd.DataFrame:
    """Metrics per (model, horizon) on the common matured sample."""
    c = combo[["horizon", "origin", "target", "y", "yhat"]].assign(model="combo")
    allp = pd.concat([preds[["horizon", "origin", "target", "y", "yhat", "model"]], c], ignore_index=True)
    allp = allp.dropna(subset=["y"])
    allp["p0"] = allp["origin"].map(origin_price)
    rows = []
    for h, g in allp.groupby("horizon"):
        bench = g[g["model"] == C.BENCHMARK].set_index("origin")
        fb, ab = _price_errors(bench["yhat"], bench["y"], bench["p0"])
        eb = (fb - ab)
        for m, gm in g.groupby("model"):
            gm = gm.set_index("origin").loc[bench.index]
            f, a = _price_errors(gm["yhat"], gm["y"], gm["p0"])
            e = f - a
            sign_ok = np.sign(gm["yhat"]) == np.sign(gm["y"])
            has_dir = gm["yhat"].abs() > 1e-12
            rows.append({
                "horizon": h, "model": m, "n": len(gm),
                "first_origin": str(gm.index.min()), "last_origin": str(gm.index.max()),
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))),
                "mape_pct": float(np.mean(np.abs(e) / a) * 100), "bias": float(np.mean(e)),
                "directional_acc_pct": float(sign_ok[has_dir].mean() * 100) if has_dir.any() else np.nan,
                "skill_vs_rw_pct": float((1 - np.mean(np.abs(e)) / np.mean(np.abs(eb))) * 100),
                "dm_p_value": np.nan if m == C.BENCHMARK else dm_test(e.values, eb.values, h),
            })
    return pd.DataFrame(rows)


def interval_quality(combo: pd.DataFrame) -> pd.DataFrame:
    g = combo.dropna(subset=["y"])
    rows = []
    for h, gh in g.groupby("horizon"):
        emp = gh[gh["interval_method"] == "empirical_errors"]
        use = emp if len(emp) >= 12 else gh
        pin = []
        for q in C.QUANTILES:
            col = f"q{int(q * 100):02d}"
            u = use["y"] - use[col]
            pin.append(np.mean(np.maximum(q * u, (q - 1) * u)))
        rows.append({
            "horizon": h, "n": len(use), "method": "empirical_errors" if use is emp else "mixed",
            "coverage_80_pct": float(((use["y"] >= use["q10"]) & (use["y"] <= use["q90"])).mean() * 100),
            "coverage_90_pct": float(((use["y"] >= use["q05"]) & (use["y"] <= use["q95"])).mean() * 100),
            "coverage_50_pct": float(((use["y"] >= use["q25"]) & (use["y"] <= use["q75"])).mean() * 100),
            "mean_pinball_log": float(np.mean(pin)),
            "brier_up": float(np.mean((use["p_up"] - (use["y"] > 0)) ** 2)),
        })
    return pd.DataFrame(rows)


def verdict(skill: float, p: float) -> tuple[str, str]:
    if np.isnan(skill):
        return "unknown", "Yetersiz veri"
    if skill > 0 and not np.isnan(p) and p < 0.10:
        return "beats_rw", "Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor"
    if skill > 0:
        return "weak_edge", "Rastgele yürüyüşten biraz iyi, fark istatistiksel olarak kanıtlanmadı"
    return "no_edge", "Rastgele yürüyüşü geçemiyor; tahmin 'fiyat aynı kalır' varsayımına yakın okunmalı"
