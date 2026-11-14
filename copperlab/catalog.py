"""Data catalog for the metadata page: every input series with its provider, coverage and
publication lag, plus the copper price data, the news data and the model's data volume."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .data import _read_daily, exogenous_raw

# provider, link
PROVIDERS = {
    "lme": ("LME – London Metal Exchange (Westmetall üzerinden)", "https://www.westmetall.com/en/markdaten.php"),
    "fred": ("FRED – St. Louis Fed", "https://fred.stlouisfed.org/"),
    "eurostat": ("Eurostat", "https://ec.europa.eu/eurostat/"),
    "ecb": ("Avrupa Merkez Bankası (ECB)", "https://www.ecb.europa.eu/stats/"),
    "boj": ("Japonya Merkez Bankası (BoJ)", "https://www.stat-search.boj.or.jp/"),
    "meti": ("Japonya Ekonomi Bakanlığı (METI)", "https://www.meti.go.jp/english/statistics/"),
    "worldbank": ("Dünya Bankası – Pink Sheet", "https://www.worldbank.org/en/research/commodity-markets"),
    "nbs": ("Çin İstatistik Bürosu (NBS, chinadata.live)", "https://chinadata.live/"),
    "bcrp": ("Peru Merkez Bankası (BCRP)", "https://estadisticas.bcrp.gob.pe/"),
    "epu": ("Economic Policy Uncertainty (Baker, Bloom, Davis)", "https://www.policyuncertainty.com/"),
    "gpr": ("Jeopolitik Risk Endeksi (Caldara, Iacoviello)", "https://www.matteoiacoviello.com/gpr.htm"),
}
FILE_PROVIDER = {
    "market/lme_copper_daily.csv": "lme", "market/lme_other_metals_daily.csv": "lme",
    "macro/fred_macro_monthly.csv": "fred", "market/worldbank_commodities_monthly.csv": "worldbank",
    "mining/peru_copper_cost_drivers_monthly.csv": "bcrp",
    "risk/economic_policy_uncertainty_monthly.csv": "epu", "risk/geopolitical_risk_monthly.csv": "gpr",
}
SERIES_PROVIDER = {"ecb_rate": "ecb", "japan_ppi": "boj", "japan_industrial_production": "meti"}


def provider_of(s) -> str:
    if s.name in SERIES_PROVIDER:
        return SERIES_PROVIDER[s.name]
    if s.file in FILE_PROVIDER:
        return FILE_PROVIDER[s.file]
    if s.file == "macro/global_macro_monthly.csv":
        return "eurostat"
    if s.file.startswith(("macro/china", "market/china")):
        return "nbs"
    return "fred"


def lag_tr(lag: int) -> str:
    return {0: "Ay kapanınca", 1: "1 ay gecikmeli", 2: "2 ay gecikmeli"}.get(lag, f"{lag} ay gecikmeli")


def series_table(start: str = "2008-01") -> pd.DataFrame:
    months = pd.period_range(start, pd.Timestamp.today().to_period("M") - 1, freq="M")
    raw = exogenous_raw(months)
    rows = []
    for s in C.SERIES:
        x = raw[s.name].dropna()
        x = x[x.index >= pd.Period(start, "M")]
        span = pd.period_range(x.index.min(), x.index.max(), freq="M") if len(x) else []
        p = provider_of(s)
        rows.append({
            "series": s.name, "label": C.SERIES_TR.get(s.name, s.name), "family": s.family,
            "family_tr": C.FAMILY_TR.get(s.family, s.family), "provider": PROVIDERS[p][0], "url": PROVIDERS[p][1],
            "frequency": "Günlük → aylık ort." if s.daily else "Aylık", "file": f"data/raw/{s.file}",
            "column": s.column, "transform": "log değişim" if s.kind == "log" else "fark",
            "publication": lag_tr(s.lag),
            "first": str(x.index.min()) if len(x) else None, "last": str(x.index.max()) if len(x) else None,
            "observations": int(len(x)),
            "completeness_pct": round(100 * len(x) / len(span), 1) if len(span) else 0.0,
            "last_value": float(x.iloc[-1]) if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


def copper_table() -> pd.DataFrame:
    d = _read_daily(C.COPPER_FILE)
    rows = []
    for col, label in (("cash_settlement_usd_per_ton", "LME bakır nakit fiyatı (hedef)"),
                       ("copper_3_month_usd_per_ton", "LME bakır 3 ay vadeli fiyat"),
                       ("copper_stock_ton", "LME bakır stokları")):
        x = d.dropna(subset=[col])
        rows.append({"series": col, "label": label, "family_tr": "Bakırın kendisi", "provider": PROVIDERS["lme"][0],
                     "url": PROVIDERS["lme"][1], "frequency": "Günlük", "file": f"data/raw/{C.COPPER_FILE}",
                     "first": x["date"].min().date().isoformat(), "last": x["date"].max().date().isoformat(),
                     "observations": int(len(x)), "last_value": float(x[col].iloc[-1])})
    return pd.DataFrame(rows)


def news_summary(heads: pd.DataFrame, removed: pd.Series, news_d: pd.DataFrame) -> dict:
    from .news import GNEWS_SETS, NEWS, _headlines
    allh = _headlines()
    out = {
        "gdelt": {"provider": "GDELT Project – DOC 2.0 API", "url": "https://www.gdeltproject.org/",
                  "query": '"copper price" (İngilizce)', "measures": "günlük haber tonu ve haber hacmi"},
        "google_news": {"provider": "Google News RSS", "url": "https://news.google.com/",
                        "searches": {k: list(v) for k, v in GNEWS_SETS.items()},
                        "headlines_downloaded": int(len(allh)), "headlines_kept": int(len(heads)),
                        "removed": {k: int(v) for k, v in removed.items()},
                        "first": str(pd.to_datetime(allh["date"]).min().date()) if len(allh) else None,
                        "last": str(pd.to_datetime(allh["date"]).max().date()) if len(allh) else None,
                        "top_sources": heads["domain"].value_counts().head(12).to_dict() if len(heads) else {}},
        "finbert": {"provider": "ProsusAI/finbert (Hugging Face)", "url": "https://huggingface.co/ProsusAI/finbert"},
        "dekads_with_news": int(news_d["news_tone"].notna().sum()) if "news_tone" in news_d else 0,
    }
    tone_years = sorted(f.stem for f in (NEWS / "tone").glob("*.json")) if (NEWS / "tone").exists() else []
    out["gdelt"]["years"] = tone_years
    return out


def write(heads: pd.DataFrame, removed: pd.Series, news_d: pd.DataFrame, panel_shape: tuple, extra: dict) -> None:
    series_table().to_csv(C.OUT / "data_catalog.csv", index=False)
    copper_table().to_csv(C.OUT / "data_catalog_copper.csv", index=False)
    summary = {"news": news_summary(heads, removed, news_d),
               "panel": {"rows": int(panel_shape[0]), "columns": int(panel_shape[1])}, **extra}
    (C.OUT / "data_catalog.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
