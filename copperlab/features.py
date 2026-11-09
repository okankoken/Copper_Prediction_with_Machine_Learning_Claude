"""Feature construction. Every feature at month t uses only data dated <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

OWN_FEATURES = (
    "r1", "r3", "r6", "r12",      # own log returns of the monthly average price
    "ma12_gap",                   # distance from the 12-month mean (mean reversion / trend)
    "vol6",                       # recent volatility
    "eom_gap",                    # month-end close vs monthly average (averaging effect)
    "spread_3m",                  # 3-month vs cash: contango (+) / backwardation (-)
    "stock_c3", "stock_c12",      # LME copper inventory changes
    "stock_z24",                  # inventory level vs its last 2 years
)

OWN_LABELS_TR = {
    "r1": "Bakır 1 aylık getiri",
    "r3": "Bakır 3 aylık getiri",
    "r6": "Bakır 6 aylık getiri",
    "r12": "Bakır 12 aylık getiri",
    "ma12_gap": "12 aylık ortalamadan sapma",
    "vol6": "6 aylık oynaklık",
    "eom_gap": "Ay sonu fiyatı / ay ortalaması",
    "spread_3m": "3 ay vadeli - nakit farkı (contango)",
    "stock_c3": "LME stok 3 aylık değişim",
    "stock_c12": "LME stok 12 aylık değişim",
    "stock_z24": "LME stok seviyesi (24 ay z-skoru)",
}


def own_features(cu: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(cu["price"])
    f = pd.DataFrame(index=cu.index)
    for k in (1, 3, 6, 12):
        f[f"r{k}"] = lp - lp.shift(k)
    f["ma12_gap"] = lp - lp.rolling(12).mean()
    f["vol6"] = f["r1"].rolling(6).std()
    f["eom_gap"] = np.log(cu["close"] / cu["price"])
    f["spread_3m"] = np.log(cu["f3m"] / cu["price"])
    ls = np.log(cu["stock"])
    f["stock_c3"] = ls - ls.shift(3)
    f["stock_c12"] = ls - ls.shift(12)
    f["stock_z24"] = (ls - ls.rolling(24).mean()) / ls.rolling(24).std()
    return f[list(OWN_FEATURES)]


def broad_features(exo: pd.DataFrame) -> pd.DataFrame:
    """3- and 12-month changes of every exogenous series (log changes for positive levels)."""
    out = {}
    for s in C.SERIES:
        x = exo[s.name]
        if s.kind == "log":
            x = np.log(x.where(x > 0))
        out[f"{s.name}__c3"] = x - x.shift(3)
        out[f"{s.name}__c12"] = x - x.shift(12)
    return pd.DataFrame(out, index=exo.index)


def targets(cu: pd.DataFrame) -> pd.DataFrame:
    """y_h(t) = log(P[t+h] / P[t]) of the monthly average cash price."""
    lp = np.log(cu["price"])
    return pd.DataFrame({h: lp.shift(-h) - lp for h in C.HORIZONS}, index=cu.index)


def series_of(broad_col: str) -> str:
    return broad_col.split("__")[0]
