"""Plain Turkish run report (outputs/REPORT.md)."""
from __future__ import annotations

import pandas as pd

from . import config as C

MODEL_TR = {
    "ml": "ML tahmini (3 modelin ortalaması)",
    "rw": "Fiyat aynı kalır",
    "forward": "LME vadeli eğrisi",
    "enet": "ElasticNet (bakır + makro)",
    "lgbm": "LightGBM (bakır + makro + haber)",
    "enet_news": "ElasticNet (bakır + haber)",
}


def _fmt(x, nd=0):
    if pd.isna(x):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def write_report(out, metrics, news_test, effects, manifest):
    L = [f"# Bakır tahmini — son veri: {manifest['forecast_origin_tr']}", "",
         f"Oluşturulma {manifest['generated_at_utc']} UTC · kod `{manifest['code_commit']}` · "
         f"{manifest['runtime_seconds']} sn", "",
         f"Son 10 günlük dönem ortalaması: **{_fmt(manifest['origin_price_usd_t'])} USD/t**", "",
         "## Tahmin (10 günlük dönem ortalamaları)", "",
         "| Dönem | Tahmin USD/t | Değişim | Geçmiş hata (MAPE) | Yön isabeti |", "|---|---:|---:|---:|---:|"]
    for _, r in out.iterrows():
        L.append(f"| {r.period_tr} | {_fmt(r.forecast_usd_t)} | {_fmt(r.change_pct, 1)}% | "
                 f"%{_fmt(r.backtest_mape_pct, 1)} | %{_fmt(r.direction_hit_pct, 0)} |")
    L.append("")

    L += ["## Geriye dönük karne", "", "'Fiyat aynı kalır' tahminine göre hatadaki azalma (%), pozitif = daha iyi.", "",
          "| Model | 1 ay | 3 ay | 6 ay | 12 ay |", "|---|---:|---:|---:|---:|"]
    for m in ["ml", "enet", "lgbm", "enet_news", "forward"]:
        g = metrics[metrics["model"] == m].set_index("horizon")
        if not g.empty:
            L.append(f"| {MODEL_TR[m]} | " + " | ".join(_fmt(g["skill_vs_rw_pct"].get(h), 1) for h in (3, 9, 18, 36)) + " |")
    L.append("")

    if not news_test.empty:
        L += ["## Haber duyarlılığı katkısı", "", "Haberli ElasticNet'in habersiz ElasticNet'e göre hata azalması (%), "
              "aynı dönemde.", "", "| Ufuk | Gözlem | Katkı | p |", "|---|---:|---:|---:|"]
        for r in news_test[news_test["horizon"].isin([3, 9, 18, 36])].itertuples():
            L.append(f"| {r.horizon // 3} ay | {r.n} | {_fmt(r.skill_pct, 1)} | {_fmt(r.dm_p, 3)} |")
        L.append("")

    L += ["## Tahmini en çok etkileyen girdiler (3 ay)", ""]
    e = effects[effects["horizon_dekads"] == 9]
    for title, part in (("Yukarı itenler", e.nlargest(8, "contribution_usd")),
                        ("Aşağı çekenler", e.nsmallest(8, "contribution_usd"))):
        L.append(f"**{title}:** " + "; ".join(f"{r.label} ({_fmt(r.contribution_usd)} $)" for r in part.itertuples()
                                              if abs(r.contribution_usd) > 0.5))
        L.append("")

    L += ["## Sınırlar", "", "- Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir.",
          "- Haber verisi 2017'den başlar.", "- Yatırım tavsiyesi değildir.", ""]
    (C.OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
