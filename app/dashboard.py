"""Streamlit dashboard:  streamlit run app/dashboard.py --server.port 8501"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copperlab.report import MODEL_TR, reference_comparison  # noqa: E402

OUT = ROOT / "outputs"
COPPER = "#c4622d"
BAND = "196,98,45"
INK = "#8f9bb3"   # readable on both light and dark Streamlit themes
GREY = "#8a8a8a"

st.set_page_config(page_title="Bakır Tahmin Laboratuvarı", page_icon="🟠", layout="wide")


@st.cache_data
def load(stamp: float):
    d = {
        "fc": pd.read_csv(OUT / "forecast_latest.csv"),
        "metrics": pd.read_csv(OUT / "metrics_by_horizon.csv"),
        "cal": pd.read_csv(OUT / "interval_calibration.csv"),
        "combo": pd.read_csv(OUT / "backtest_combination.csv"),
        "preds": pd.read_csv(OUT / "backtest_expert_predictions.csv"),
        "weights": pd.read_csv(OUT / "expert_weights_history.csv"),
        "drivers": pd.read_csv(OUT / "drivers_latest.csv"),
        "panel": pd.read_csv(ROOT / "data" / "panel" / "copper_monthly_panel.csv"),
        "manifest": json.loads((OUT / "run_manifest.json").read_text()),
    }
    d["panel"]["date"] = pd.PeriodIndex(d["panel"]["month"], freq="M").to_timestamp("M")
    return d


def tr(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def month_end(s):
    return pd.PeriodIndex(s.astype(str), freq="M").to_timestamp("M")


def layout(fig, h=430, y="USD/t"):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title=y,
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


if not (OUT / "forecast_latest.csv").exists():
    st.error("Çıktı bulunamadı. Önce `python -m copperlab all` çalıştırın.")
    st.stop()

D = load((OUT / "forecast_latest.csv").stat().st_mtime)
fc, M, man = D["fc"], D["metrics"], D["manifest"]
p0 = man["origin_price_usd_t"]
fc["date"] = pd.to_datetime(fc["forecast_date"])

st.title("🟠 Bakır Tahmin Laboratuvarı")
st.caption(f"LME bakır nakit fiyatı · aylık ortalama · H1–H12 · tahmin başlangıcı **{man['forecast_origin']}** · "
           f"oluşturulma {man['generated_at_utc']} UTC · kod `{man['code_commit']}`")

t1, t2, t3, t4, t5, t6 = st.tabs(["Tahmin", "Güvenilirlik", "Backtest", "ML uzmanları ne diyor?",
                                  "Orijinal proje ile", "Yöntem"])

# ---------------------------------------------------------------- Tahmin
with t1:
    cols = st.columns(5)
    cols[0].metric(f"{man['forecast_origin']} ortalama", f"{tr(p0)} $")
    for c, h in zip(cols[1:], (1, 3, 6, 12)):
        r = fc[fc.horizon == h].iloc[0]
        c.metric(f"H{h} · {r.forecast_month}", f"{tr(r.forecast_usd_t)} $",
                 f"{r.change_from_origin_pct:+.1f}%".replace(".", ","), delta_color="off",
                 help=f"%80 aralık: {tr(r.p10_usd_t)} – {tr(r.p90_usd_t)} · yukarı olasılığı %{tr(r.prob_above_origin*100)}")

    years = st.slider("Gösterilen geçmiş (yıl)", 2, 17, 6)
    hist = D["panel"][D["panel"]["date"] >= D["panel"]["date"].max() - pd.DateOffset(years=years)]
    x0 = pd.concat([pd.Series([hist["date"].iloc[-1]]), fc["date"]])

    def band(lo, hi, alpha, name):
        return go.Scatter(x=list(x0) + list(x0[::-1]),
                          y=[p0] + list(fc[hi]) + list(fc[lo][::-1]) + [p0],
                          fill="toself", fillcolor=f"rgba({BAND},{alpha})", line=dict(width=0),
                          name=name, hoverinfo="skip")

    fig = go.Figure()
    fig.add_trace(band("p05_usd_t", "p95_usd_t", 0.12, "%90 aralık"))
    fig.add_trace(band("p10_usd_t", "p90_usd_t", 0.20, "%80 aralık"))
    fig.add_trace(band("p25_usd_t", "p75_usd_t", 0.32, "%50 aralık"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Gerçekleşen (aylık ort.)",
                             line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=x0, y=[p0] + list(fc["forecast_usd_t"]), name="Tahmin (medyan)",
                             line=dict(color=COPPER, width=3), mode="lines+markers"))
    st.plotly_chart(layout(fig, 470), use_container_width=True)

    ml_w = fc[["weight_own_ridge", "weight_factor_ridge", "weight_gbm"]].sum(axis=1)
    if (ml_w < 0.05).all():
        st.info("Tahmin çizgisi düz görünüyorsa bu bir hata değil: son değerlendirme penceresinde hiçbir ML "
                "uzmanı **ay sonu fiyatının korunacağı** kuralını yenemedi. Sistem bu yüzden ağırlığı o kurala "
                "veriyor. Asıl bilgi, aralıkların genişliğinde ve yukarı/aşağı olasılığında.")
    else:
        st.info("ML uzmanlarının bu aydaki toplam ağırlığı ufka göre: " +
                ", ".join(f"H{h} %{tr(w * 100)}" for h, w in zip(fc.horizon, ml_w)))

    show = fc[["horizon", "forecast_month", "forecast_usd_t", "p10_usd_t", "p50_usd_t", "p90_usd_t",
               "prob_above_origin", "backtest_mape_pct", "skill_vs_rw_pct", "dm_p_value", "verdict_tr"]].copy()
    show.columns = ["Ufuk", "Ay", "Tahmin", "P10", "P50", "P90", "Yukarı olasılığı", "Backtest MAPE %",
                    "RW'ye göre beceri %", "DM p", "Değerlendirme"]
    st.dataframe(show.style.format({"Tahmin": "{:,.0f}", "P10": "{:,.0f}", "P50": "{:,.0f}", "P90": "{:,.0f}",
                                    "Yukarı olasılığı": "{:.0%}", "Backtest MAPE %": "{:.2f}",
                                    "RW'ye göre beceri %": "{:+.1f}", "DM p": "{:.3f}"}),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Güvenilirlik
with t2:
    st.subheader("Her uzman rastgele yürüyüşe göre ne kadar iyi?")
    st.caption("Beceri = 1 − MAE(model) / MAE(fiyat aynı kalır). Pozitif = daha iyi. "
               f"Aynı {int(M[M.horizon == 1].n.iloc[0])}–{int(M[M.horizon == 12].n.iloc[0])} gözlemlik ortak örneklem.")
    order = ["combo", "rw_spot", "own_ridge", "factor_ridge", "gbm"]
    colors = {"combo": COPPER, "rw_spot": INK, "own_ridge": "#5b8db8", "factor_ridge": "#7aa974", "gbm": "#a77dc2"}
    fig = go.Figure()
    for m in order:
        g = M[M.model == m]
        fig.add_trace(go.Bar(x=g.horizon, y=g.skill_vs_rw_pct, name=MODEL_TR[m], marker_color=colors[m]))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=[f"H{i}" for i in range(1, 13)])
    st.plotly_chart(layout(fig, 420, "Beceri %"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kombinasyon: istatistiksel anlamlılık (Diebold–Mariano, tek yönlü)**")
        cm = M[M.model == "combo"][["horizon", "n", "mape_pct", "skill_vs_rw_pct", "dm_p_value", "directional_acc_pct"]]
        cm.columns = ["Ufuk", "Gözlem", "MAPE %", "Beceri %", "p", "Yön isabeti %"]
        st.dataframe(cm.style.format({"MAPE %": "{:.2f}", "Beceri %": "{:+.1f}", "p": "{:.3f}",
                                      "Yön isabeti %": "{:.1f}"}), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Aralıklar ne kadar dürüst? (gerçekleşen kapsama oranı)**")
        cal = D["cal"]
        fig = go.Figure()
        for col, nom, name in (("coverage_50_pct", 50, "%50"), ("coverage_80_pct", 80, "%80"), ("coverage_90_pct", 90, "%90")):
            fig.add_trace(go.Scatter(x=cal.horizon, y=cal[col], mode="lines+markers", name=f"{name} aralık"))
            fig.add_hline(y=nom, line_dash="dot", line_color=GREY)
        fig.update_yaxes(range=[30, 100])
        st.plotly_chart(layout(fig, 330, "Kapsama %"), use_container_width=True)

# ---------------------------------------------------------------- Backtest
with t3:
    h = st.select_slider("Ufuk", options=list(range(1, 13)), value=1, format_func=lambda x: f"H{x}")
    c = D["combo"][D["combo"].horizon == h].dropna(subset=["y"]).copy()
    c["date"] = month_end(c["target"])
    base = D["panel"].set_index("month")["price"]
    c["p0"] = c["origin"].map(base)
    for col in ("yhat", "y", "q10", "q90"):
        c[col + "_usd"] = c["p0"] * np.exp(c[col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(c.date) + list(c.date[::-1]), y=list(c.q90_usd) + list(c.q10_usd[::-1]),
                             fill="toself", fillcolor=f"rgba({BAND},0.18)", line=dict(width=0), name="%80 aralık"))
    fig.add_trace(go.Scatter(x=c.date, y=c.y_usd, name="Gerçekleşen", line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=c.date, y=c.yhat_usd, name=f"{h} ay önce yapılan tahmin", line=dict(color=COPPER)))
    st.plotly_chart(layout(fig, 430), use_container_width=True)
    inside = ((c.y >= c.q10) & (c.y <= c.q90)).mean()
    st.caption(f"Bu ufukta gerçekleşmelerin %{tr(inside*100)}'i %80 aralığın içinde kaldı ({len(c)} tahmin).")

    st.markdown("**Kombinasyon ağırlıkları zaman içinde**")
    w = D["weights"][D["weights"].horizon == h].copy()
    w["date"] = month_end(w["origin"])
    fig = go.Figure()
    for m in ["rw", "rw_spot", "own_ridge", "factor_ridge", "gbm"]:
        fig.add_trace(go.Scatter(x=w.date, y=w[f"w_{m}"], stackgroup="one", name=MODEL_TR[m],
                                 line=dict(width=0.5, color=colors.get(m, GREY))))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(layout(fig, 300, "Ağırlık"), use_container_width=True)

# ---------------------------------------------------------------- Drivers
with t4:
    h = st.select_slider("Ufuk ", options=list(range(1, 13)), value=3, format_func=lambda x: f"H{x}")
    w_fac = fc.loc[fc.horizon == h, "weight_factor_ridge"].iloc[0]
    st.warning(f"Bu bölüm **makro faktör Ridge** uzmanının ne söylediğini gösterir. Bu uzmanın H{h} "
               f"kombinasyonundaki ağırlığı şu an %{tr(w_fac * 100)}. Ağırlık, uzman son değerlendirme "
               "penceresinde ay sonu fiyatı kuralını yenerse artar. Katkılar nedensellik değil, "
               "modelin koşullu ilişkisidir.")
    d = D["drivers"][D["drivers"].horizon == h].copy()
    d = d.reindex(d.contribution_usd_approx.abs().sort_values().index).tail(12)
    expert = fc.loc[fc.horizon == h, "expert_factor_ridge_usd_t"].iloc[0]
    st.metric(f"Makro faktör uzmanının H{h} tahmini", f"{tr(expert)} $",
              f"{(expert / p0 - 1) * 100:+.1f}%".replace(".", ","), delta_color="off")
    fig = go.Figure(go.Bar(x=d.contribution_usd_approx, y=d.label, orientation="h",
                           marker_color=[COPPER if v > 0 else "#5b8db8" for v in d.contribution_usd_approx]))
    st.plotly_chart(layout(fig, 460, ""), use_container_width=True)
    st.caption("Katkı ≈ başlangıç fiyatı × katsayı × standartlaştırılmış güncel değer (USD/t).")

# ---------------------------------------------------------------- Original
with t5:
    comp = reference_comparison(fc)
    if comp is None:
        st.info("Orijinal projenin tahmin dosyası bulunamadı.")
    else:
        st.caption(f"Orijinal proje (`Copper_Prediction_with_Machine_Learning`) tahmin başlangıcı "
                   f"{comp.original_origin.iloc[0]}. Dosya yalnızca okunur kopya olarak alındı.")
        fig = go.Figure()
        x = pd.to_datetime(fc["forecast_date"])
        fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(fc.p90_usd_t) + list(fc.p10_usd_t[::-1]),
                                 fill="toself", fillcolor=f"rgba({BAND},0.2)", line=dict(width=0),
                                 name="Bu proje: %80 aralık"))
        fig.add_trace(go.Scatter(x=x, y=fc.forecast_usd_t, name="Bu proje: medyan", line=dict(color=COPPER, width=3)))
        fig.add_trace(go.Scatter(x=x, y=comp.original_usd_t, name="Orijinal proje", mode="lines+markers",
                                 line=dict(color=INK, dash="dash")))
        st.plotly_chart(layout(fig, 430), use_container_width=True)
        st.dataframe(comp.rename(columns={"horizon": "Ufuk", "forecast_month": "Ay", "forecast_usd_t": "Bu proje",
                                          "original_usd_t": "Orijinal", "inside_80_band": "%80 aralıkta mı?"})
                     [["Ufuk", "Ay", "Bu proje", "Orijinal", "%80 aralıkta mı?"]]
                     .style.format({"Bu proje": "{:,.0f}", "Orijinal": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Method
with t6:
    st.markdown((ROOT / "docs" / "METHOD.md").read_text(encoding="utf-8"))
