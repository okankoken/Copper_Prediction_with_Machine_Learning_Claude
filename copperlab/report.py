"""Plain Turkish run report (outputs/REPORT.md)."""
from __future__ import annotations

import pandas as pd

from . import config as C

MODEL_TR = {
    "rw": "Rastgele yürüyüş (fiyat aynı kalır)",
    "rw_spot": "Ay sonu fiyatı sabit kalır",
    "forward": "LME vadeli eğrisi (ay sonu + contango)",
    "ml_scenario": "ML senaryosu (3 ML uzmanının ortalaması)",
    "own_ridge": "Bakırın kendi sinyalleri (Ridge)",
    "factor_ridge": "Bakır + makro faktörler (Ridge)",
    "gbm": "Bakır + makro faktörler (LightGBM)",
    "combo": "Kombinasyon (yayımlanan)",
}


def _fmt(x, nd=0):
    if pd.isna(x):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def reference_comparison(out: pd.DataFrame) -> pd.DataFrame | None:
    ref_path = C.DATA_REF / "copper_production_forecast_latest.csv"
    if not ref_path.exists():
        return None
    ref = pd.read_csv(ref_path)
    if "final_forecast_price_usd_per_ton" not in ref or "horizon" not in ref:
        return None
    ref = ref[["horizon", "final_forecast_price_usd_per_ton", "forecast_origin"]].rename(
        columns={"final_forecast_price_usd_per_ton": "original_usd_t", "forecast_origin": "original_origin"})
    m = out[["horizon", "forecast_month", "forecast_usd_t", "p10_usd_t", "p90_usd_t"]].merge(ref, on="horizon")
    m["inside_80_band"] = (m["original_usd_t"] >= m["p10_usd_t"]) & (m["original_usd_t"] <= m["p90_usd_t"])
    return m


def write_report(out, metrics, intervals, drivers, manifest):
    L = []
    o = manifest["forecast_origin"]
    p0 = manifest["origin_price_usd_t"]
    L += [f"# Bakır tahmin raporu — başlangıç {o}", "",
          f"Oluşturulma: {manifest['generated_at_utc']} UTC · kod `{manifest['code_commit']}` · "
          f"süre {manifest['runtime_seconds']} sn", "",
          f"Başlangıç fiyatı ({o} LME nakit ortalaması): **{_fmt(p0)} USD/t**", ""]

    L += ["## Tahmin", "",
          "| Ufuk | Ay | Tahmin | %80 aralık | Yukarı olasılığı | Değerlendirme |",
          "|---|---|---:|---|---:|---|"]
    for _, r in out.iterrows():
        L.append(f"| H{r.horizon} | {r.forecast_month} | {_fmt(r.forecast_usd_t)} | "
                 f"{_fmt(r.p10_usd_t)} – {_fmt(r.p90_usd_t)} | %{_fmt(r.prob_above_origin * 100)} | {r.verdict_tr} |")
    L.append("")

    L += ["## ML senaryosu (yayımlanan tahmine dahil değil)", "",
          "Üç ML uzmanının eşit ağırlıklı ortalaması. Geçmişte vadeli eğri kuralından daha kötü sonuç verdiği "
          "için ana tahmine katılmaz; bilgi amaçlı gösterilir.", "",
          "| Ufuk | ML senaryosu | Backtest MAPE | Vadeli eğriye göre |", "|---|---:|---:|---:|"]
    for _, r in out.iterrows():
        L.append(f"| H{r.horizon} | {_fmt(r.ml_scenario_usd_t)} | %{_fmt(r.ml_scenario_mape_pct, 2)} | "
                 f"%{_fmt(r.ml_scenario_vs_anchor_pct, 1)} |")
    L.append("")

    L += ["## Model güvenilirliği (walk-forward backtest)", "",
          "Beceri = rastgele yürüyüşe göre ortalama mutlak hatadaki azalma. "
          "p < 0,10 ise fark istatistiksel olarak anlamlı kabul edilir (Diebold-Mariano).", "",
          "| Ufuk | Gözlem | Kombinasyon MAPE | RW MAPE | Beceri | DM p | En iyi uzman |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for h, g in metrics.groupby("horizon"):
        g = g.set_index("model")
        best = g.drop(index=["combo", "ml_scenario"], errors="ignore")["mae"].idxmin()
        c = g.loc["combo"]
        L.append(f"| H{h} | {int(c.n)} | %{_fmt(c.mape_pct, 2)} | %{_fmt(g.loc['rw'].mape_pct, 2)} | "
                 f"%{_fmt(c.skill_vs_rw_pct, 1)} | {_fmt(c.dm_p_value, 3)} | {MODEL_TR[best]} |")
    L.append("")

    L += ["## Aralıkların kalibrasyonu", "",
          "İyi kalibre edilmiş bir %80 aralık gerçekleşmelerin yaklaşık %80'ini kapsamalı.", "",
          "| Ufuk | Gözlem | %50 kapsama | %80 kapsama | %90 kapsama |", "|---|---:|---:|---:|---:|"]
    for _, r in intervals.iterrows():
        L.append(f"| H{r.horizon} | {r.n} | %{_fmt(r.coverage_50_pct)} | %{_fmt(r.coverage_80_pct)} | "
                 f"%{_fmt(r.coverage_90_pct)} |")
    L.append("")

    L += ["## Tahmini etkileyen başlıca girdiler (makro faktör Ridge uzmanı)", "",
          "Katkı = katsayı × standartlaştırılmış güncel değer, log-getiri cinsinden; USD karşılığı "
          "yaklaşık değerdir. Nedensellik değil, modelin koşullu ilişkisidir.", ""]
    for h in (1, 3, 6, 12):
        d = drivers[drivers["horizon"] == h]
        if d.empty:
            continue
        d = d.reindex(d["contribution_log"].abs().sort_values(ascending=False).index).head(5)
        L.append(f"**H{h}:** " + "; ".join(f"{r.label} ({'+' if r.contribution_usd_approx >= 0 else ''}"
                                          f"{_fmt(r.contribution_usd_approx)} USD/t)" for r in d.itertuples()))
        L.append("")

    comp = reference_comparison(out)
    if comp is not None:
        L += ["## Orijinal projenin tahmini ile karşılaştırma", "",
              f"Orijinal proje tahmin başlangıcı: {comp['original_origin'].iloc[0]}. "
              "Orijinal dosya yalnızca okunur kopya olarak alındı.", "",
              "| Ufuk | Bu proje | Orijinal | Fark | Orijinal bu projenin %80 aralığında mı? |",
              "|---|---:|---:|---:|---|"]
        for r in comp.itertuples():
            L.append(f"| H{r.horizon} | {_fmt(r.forecast_usd_t)} | {_fmt(r.original_usd_t)} | "
                     f"{_fmt(r.original_usd_t - r.forecast_usd_t)} | {'Evet' if r.inside_80_band else 'Hayır'} |")
        L.append("")

    L += ["## Sınırlar", "",
          "- Veriler güncel sürümdür (revizyonlar sonrası); geçmişte yayımlanmış ilk değerler değildir.",
          "- Yayın gecikmeleri varsayımdır: piyasa verileri 0, risk endeksleri 1–2, resmi makro veriler 2 ay.",
          "- Çakışan çok aylık getiriler nedeniyle uzun ufuklarda bağımsız gözlem sayısı görünenden azdır.",
          "- Bu çalışma yatırım tavsiyesi değildir.", ""]
    (C.OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
