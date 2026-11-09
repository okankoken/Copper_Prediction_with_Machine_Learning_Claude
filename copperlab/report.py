"""Plain Turkish run report (outputs/REPORT.md)."""
from __future__ import annotations

import pandas as pd

from . import config as C

MODEL_TR = {
    "main": "Ana tahmin",
    "ml_model": "ML modeli (3 ML uzmanının ortalaması)",
    "rw": "Fiyat aynı kalır (rastgele yürüyüş)",
    "rw_spot": "Ay sonu fiyatı kalır",
    "forward": "LME vadeli eğrisi",
    "enet": "ElasticNet (bakır + makro faktörler)",
    "lgbm": "LightGBM (bakır + makro + haber)",
    "enet_news": "ElasticNet (bakır + haber duyarlılığı)",
}


def _fmt(x, nd=0):
    if pd.isna(x):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def write_report(out, metrics, news_test, manifest):
    o = manifest["forecast_origin"]
    L = [f"# Bakır tahmin raporu — başlangıç {o}", "",
         f"Oluşturulma {manifest['generated_at_utc']} UTC · kod `{manifest['code_commit']}` · "
         f"{manifest['runtime_seconds']} sn", "",
         f"{o} LME nakit ortalaması: **{_fmt(manifest['origin_price_usd_t'])} USD/t**", "",
         "## Tahmin", "",
         "| Ufuk | Ay | Ana tahmin | ML modeli | Ana tahmin isabeti* | ML isabeti* |",
         "|---|---|---:|---:|---|---|"]
    for _, r in out.iterrows():
        L.append(f"| H{r.horizon} | {r.forecast_month} | {_fmt(r.main_usd_t)} | {_fmt(r.ml_usd_t)} | "
                 f"%{_fmt(r.main_mape_pct, 1)} hata · {r.main_verdict} | %{_fmt(r.ml_mape_pct, 1)} hata · {r.ml_verdict} |")
    L += ["", "*Geriye dönük testte ortalama mutlak yüzde hata ve 'fiyat aynı kalır' tahminine göre karar.", ""]

    L += ["## Modellerin geriye dönük karnesi", "",
          "Değer = 'fiyat aynı kalır' tahminine göre hatadaki azalma (%). Pozitif daha iyi.", "",
          "| Model | " + " | ".join(f"H{h}" for h in C.HORIZONS) + " |",
          "|---|" + "---:|" * len(C.HORIZONS)]
    for m in ["main", "ml_model", "forward", "rw_spot", "enet", "lgbm", "enet_news"]:
        g = metrics[metrics["model"] == m].set_index("horizon")
        if g.empty:
            continue
        L.append(f"| {MODEL_TR[m]} | " + " | ".join(_fmt(g["skill_vs_rw_pct"].get(h), 1) for h in C.HORIZONS) + " |")
    L.append("")

    if not news_test.empty:
        L += ["## Haber duyarlılığı işe yarıyor mu?", "",
              "Aynı dönemde (2020 sonrası) haberli ElasticNet ile habersiz ElasticNet karşılaştırması.", "",
              "| Ufuk | Gözlem | Haberin katkısı (%) | p |", "|---|---:|---:|---:|"]
        for r in news_test.itertuples():
            L.append(f"| H{r.horizon} | {r.n} | {_fmt(r.skill_pct, 1)} | {_fmt(r.dm_p, 3)} |")
        L.append("")

    L += ["## Sınırlar", "",
          "- Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir.",
          "- Haber verisi 2017'den başlar; haberli model daha kısa geçmişle değerlendirilir.",
          "- Bu çalışma yatırım tavsiyesi değildir.", ""]
    (C.OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
