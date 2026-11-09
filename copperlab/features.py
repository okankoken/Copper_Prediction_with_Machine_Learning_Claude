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


SENT_FEATURES = ("news_tone", "news_tone_c3", "news_attention_c3", "finbert_net", "finbert_net_c3")

SENT_LABELS_TR = {
    "news_tone": "Haber tonu (GDELT)",
    "news_tone_c3": "Haber tonu 3 aylık değişim",
    "news_attention_c3": "Haber ilgisi 3 aylık değişim",
    "finbert_net": "Başlık duyarlılığı (FinBERT)",
    "finbert_net_c3": "Başlık duyarlılığı 3 aylık değişim",
}


def sentiment_features(sent: pd.DataFrame, index: pd.PeriodIndex) -> pd.DataFrame:
    """Monthly news features; NaN where no news history exists (before 2017)."""
    s = sent.reindex(index) if sent is not None and len(sent) else pd.DataFrame(index=index)
    out = pd.DataFrame(index=index)
    tone = s.get("news_tone", pd.Series(np.nan, index=index))
    att = s.get("news_attention", pd.Series(np.nan, index=index))
    fb = s.get("finbert_net", pd.Series(np.nan, index=index))
    out["news_tone"] = tone
    out["news_tone_c3"] = tone - tone.shift(3)
    out["news_attention_c3"] = np.log(att.where(att > 0)) - np.log(att.where(att > 0)).shift(3)
    out["finbert_net"] = fb
    out["finbert_net_c3"] = fb - fb.shift(3)
    return out[list(SENT_FEATURES)]


def series_of(broad_col: str) -> str:
    return broad_col.split("__")[0]
