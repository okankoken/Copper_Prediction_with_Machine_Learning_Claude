"""Out-of-sample evaluation in USD/t, with Diebold-Mariano tests."""
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


def evaluate(allp: pd.DataFrame, origin_price: pd.Series) -> pd.DataFrame:
    """Metrics per (model, horizon). Each model is scored on the origins where it has a
    forecast, against the random walk and the anchor rule on exactly those origins."""
    allp = allp.dropna(subset=["y"]).copy()
    allp["p0"] = allp["origin"].map(origin_price)
    allp["f"] = allp["p0"] * np.exp(allp["yhat"])
    allp["a"] = allp["p0"] * np.exp(allp["y"])
    allp["e"] = allp["f"] - allp["a"]
    rows = []
    for h, g in allp.groupby("horizon"):
        bench = g[g["model"] == C.BENCHMARK].set_index("origin")["e"]
        anchor = g[g["model"] == C.ANCHOR].set_index("origin")["e"]
        for m, gm in g.groupby("model"):
            gm = gm.set_index("origin")
            idx = gm.index.intersection(bench.index).intersection(anchor.index)
            gm = gm.loc[idx]
            e, eb, ea = gm["e"].values, bench.loc[idx].values, anchor.loc[idx].values
            has_dir = gm["yhat"].abs() > 1e-12
            hit = (np.sign(gm["yhat"]) == np.sign(gm["y"]))[has_dir]
            rows.append({
                "horizon": h, "model": m, "n": len(gm),
                "first_origin": str(idx.min()), "last_origin": str(idx.max()),
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))),
                "mape_pct": float(np.mean(np.abs(e) / gm["a"]) * 100), "bias": float(np.mean(e)),
                "directional_acc_pct": float(hit.mean() * 100) if len(hit) else np.nan,
                "skill_vs_rw_pct": float((1 - np.mean(np.abs(e)) / np.mean(np.abs(eb))) * 100),
                "skill_vs_anchor_pct": float((1 - np.mean(np.abs(e)) / np.mean(np.abs(ea))) * 100),
                "dm_p_vs_rw": np.nan if m == C.BENCHMARK else dm_test(e, eb, h),
                "dm_p_vs_anchor": np.nan if m == C.ANCHOR else dm_test(e, ea, h),
            })
    return pd.DataFrame(rows)


def pairwise(allp: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Skill of model a over model b on their common matured origins (used for the news test)."""
    d = allp.dropna(subset=["y"])
    rows = []
    for h, g in d.groupby("horizon"):
        x = g[g["model"] == a].set_index("origin")
        z = g[g["model"] == b].set_index("origin")
        idx = x.index.intersection(z.index)
        if len(idx) < 10:
            continue
        ea = np.abs(np.exp(x.loc[idx, "yhat"]) - np.exp(x.loc[idx, "y"]))
        eb = np.abs(np.exp(z.loc[idx, "yhat"]) - np.exp(z.loc[idx, "y"]))
        rows.append({"horizon": h, "n": len(idx), "skill_pct": float((1 - ea.mean() / eb.mean()) * 100),
                     "dm_p": dm_test(ea.values, eb.values, int(h))})
    return pd.DataFrame(rows)


def verdict(skill: float, p: float) -> str:
    if np.isnan(skill):
        return "Yetersiz veri"
    if skill > 0 and not np.isnan(p) and p < 0.10:
        return "Anlamlı şekilde daha iyi"
    if skill > 0:
        return "Biraz daha iyi (kanıtlanmadı)"
    return "Daha iyi değil"
