"""Features at dekad resolution. Every feature of dekad t uses only data dated <= end of t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import dekad as K

OWN_FEATURES = (
    "r1", "r3", "r9", "r18", "r36",   # own log returns over 1 dekad, 1, 3, 6, 12 months
    "ma36_gap",                       # distance from the 12-month mean
    "vol18",                          # volatility of dekad returns over 6 months
    "eod_gap",                        # last close vs dekad average
    "spread_3m",                      # 3-month vs cash: contango (+) / backwardation (-)
    "stock_c9", "stock_c36",          # LME copper stock change over 3 and 12 months
    "stock_z72",                      # stock level vs the last 2 years
)

OWN_TR = {
    "r1": "Bakır son 10 günlük getiri", "r3": "Bakır 1 aylık getiri", "r9": "Bakır 3 aylık getiri",
    "r18": "Bakır 6 aylık getiri", "r36": "Bakır 12 aylık getiri", "ma36_gap": "Bakır 12 aylık ortalamadan sapma",
    "vol18": "Bakır oynaklığı", "eod_gap": "Dönem sonu fiyatı / dönem ortalaması",
    "spread_3m": "3 ay vadeli - nakit farkı", "stock_c9": "LME bakır stoku 3 aylık değişim",
    "stock_c36": "LME bakır stoku 12 aylık değişim", "stock_z72": "LME bakır stoku seviyesi",
}

SENT_FEATURES = ("news_tone", "news_tone_c9", "news_attention_c9", "finbert_net", "finbert_net_c9")

SENT_TR = {
    "news_tone": "Haber tonu (GDELT)", "news_tone_c9": "Haber tonu 3 aylık değişim",
    "news_attention_c9": "Bakır haberlerine ilgi 3 aylık değişim", "finbert_net": "Başlık duyarlılığı (FinBERT)",
    "finbert_net_c9": "Başlık duyarlılığı 3 aylık değişim",
}


def own_features(cu: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(cu["price"])
    f = pd.DataFrame(index=cu.index)
    for k in (1, 3, 9, 18, 36):
        f[f"r{k}"] = lp - lp.shift(k)
    f["ma36_gap"] = lp - lp.rolling(36).mean()
    f["vol18"] = f["r1"].rolling(18).std()
    f["eod_gap"] = np.log(cu["close"] / cu["price"])
    f["spread_3m"] = np.log(cu["f3m"] / cu["price"])
    ls = np.log(cu["stock"])
    f["stock_c9"] = ls - ls.shift(9)
    f["stock_c36"] = ls - ls.shift(36)
    f["stock_z72"] = (ls - ls.rolling(72).mean()) / ls.rolling(72).std()
    return f[list(OWN_FEATURES)]


def broad_monthly(exo: pd.DataFrame) -> pd.DataFrame:
    """3- and 12-month changes of every exogenous series (log changes for positive levels)."""
    out = {}
    for s in C.SERIES:
        x = exo[s.name]
        if s.kind == "log":
            x = np.log(x.where(x > 0))
        out[f"{s.name}__c3"] = x - x.shift(3)
        out[f"{s.name}__c12"] = x - x.shift(12)
    return pd.DataFrame(out, index=exo.index)


def broad_features(exo: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Monthly macro changes mapped to dekads: dekad t sees the last month completed by its end."""
    bm = broad_monthly(exo)
    refs = [K.ref_month(int(n)) for n in index]
    out = bm.reindex(refs)
    out.index = index
    return out


def sentiment_features(news_d: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    """Dekad news features; NaN where no news history exists (before 2017)."""
    s = news_d.reindex(index) if news_d is not None and len(news_d) else pd.DataFrame(index=index)
    nan = pd.Series(np.nan, index=index)
    tone = s.get("news_tone", nan)
    att = s.get("news_attention", nan)
    fb = s.get("finbert_net", nan).rolling(3, min_periods=2).mean()   # ~1 month of headlines
    la = np.log(att.where(att > 0))
    out = pd.DataFrame(index=index)
    out["news_tone"] = tone
    out["news_tone_c9"] = tone - tone.shift(9)
    out["news_attention_c9"] = la - la.shift(9)
    out["finbert_net"] = fb
    out["finbert_net_c9"] = fb - fb.shift(9)
    return out[list(SENT_FEATURES)]


def targets(cu: pd.DataFrame) -> pd.DataFrame:
    """y_h(t) = log(P[t+h] / P[t]) of the dekad average cash price."""
    lp = np.log(cu["price"])
    return pd.DataFrame({h: lp.shift(-h) - lp for h in C.HORIZONS}, index=cu.index)


def describe(feature: str) -> tuple[str, str, str]:
    """(series key, Turkish label, family) of any model input."""
    if feature in OWN_TR:
        return feature, OWN_TR[feature], "copper"
    if feature in SENT_TR:
        return feature, SENT_TR[feature], "news"
    name, tr = feature.split("__")
    fam = next(s.family for s in C.SERIES if s.name == name)
    span = {"c3": "3 aylık değişim", "c12": "12 aylık değişim"}[tr]
    return name, f"{C.SERIES_TR.get(name, name)} ({span})", fam
