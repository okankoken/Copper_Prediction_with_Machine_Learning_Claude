"""Lean dashboard:  streamlit run app/dashboard.py --server.port 8501"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copperlab.report import MODEL_TR  # noqa: E402

OUT = ROOT / "outputs"
COPPER, ML, INK, GRID = "#d0702f", "#3fa7a0", "#8f9bb3", "rgba(128,128,128,0.18)"

st.set_page_config(page_title="LME Bakır Tahmini", page_icon="🟠", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1200px} "
            "[data-testid=stMetricValue]{font-size:1.9rem}</style>", unsafe_allow_html=True)


@st.cache_data
def load(stamp: float):
    d = {}
    for k, f in {"fc": "forecast_latest.csv", "metrics": "metrics_by_horizon.csv",
                 "news": "news_sentiment_monthly.csv", "imp": "feature_importance_latest.csv",
                 "newstest": "news_value_test.csv"}.items():
        try:
            d[k] = pd.read_csv(OUT / f)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
    d["panel"] = pd.read_csv(ROOT / "data" / "panel" / "copper_monthly_panel.csv", usecols=["month", "price"])
    d["panel"]["date"] = pd.PeriodIndex(d["panel"]["month"], freq="M").to_timestamp("M")
    d["manifest"] = json.loads((OUT / "run_manifest.json").read_text())
    return d


def tr(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def style(fig, h):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID)
    return fig


if not (OUT / "forecast_latest.csv").exists():
    st.error("Çıktı yok. Önce `./run.sh` çalıştırın.")
    st.stop()

D = load((OUT / "forecast_latest.csv").stat().st_mtime)
fc, M, man = D["fc"], D["metrics"], D["manifest"]
p0 = man["origin_price_usd_t"]
fc["date"] = pd.to_datetime(fc["forecast_date"])

st.markdown("## 🟠 LME Bakır Tahmini")
st.caption(f"Aylık ortalama nakit fiyatı, USD/ton · son veri **{man['forecast_origin']}** · "
           f"güncelleme {man['generated_at_utc'][:16].replace('T', ' ')} UTC")

# ---- KPIs
cols = st.columns(5)
cols[0].metric(f"{man['forecast_origin']} ortalaması", tr(p0))
for c, h in zip(cols[1:], (1, 3, 6, 12)):
    r = fc[fc.horizon == h].iloc[0]
    c.metric(f"{h} ay sonra · {r.forecast_month}", tr(r.main_usd_t),
             f"{(r.main_usd_t / p0 - 1) * 100:+.1f}%".replace(".", ","))
    pct = f"{(r.ml_usd_t / p0 - 1) * 100:+.1f}%".replace(".", ",")
    c.caption(f"ML modeli: **{tr(r.ml_usd_t)}** ({pct})")

# ---- main chart
hist = D["panel"][D["panel"]["date"] >= D["panel"]["date"].max() - pd.DateOffset(years=5)]
x0 = [hist["date"].iloc[-1], *fc["date"]]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Gerçekleşen", line=dict(color=INK, width=2)))
fig.add_trace(go.Scatter(x=x0, y=[p0, *fc["main_usd_t"]], name="Ana tahmin", mode="lines+markers",
                         line=dict(color=COPPER, width=3)))
fig.add_trace(go.Scatter(x=x0, y=[p0, *fc["ml_usd_t"]], name="ML modeli", mode="lines+markers",
                         line=dict(color=ML, width=2, dash="dash")))
fig.update_yaxes(title=None, tickformat=",.0f")
st.plotly_chart(style(fig, 420), use_container_width=True)

# ---- accuracy + news
left, right = st.columns(2)
with left:
    st.markdown("**Hangi model ne kadar isabetli?**")
    st.caption("Geriye dönük testte, 'fiyat aynı kalır' tahminine göre hatadaki azalma (%). Üstte olan daha iyi.")
    fig = go.Figure()
    for m, color, dash in (("main", COPPER, None), ("ml_model", ML, "dash"), ("lgbm", "#8b6cc4", "dot"),
                           ("enet", "#5b8db8", "dot"), ("enet_news", "#c4a23a", "dot")):
        g = M[M.model == m].sort_values("horizon")
        if not g.empty:
            fig.add_trace(go.Scatter(x=g.horizon, y=g.skill_vs_rw_pct, name=MODEL_TR[m].split(" (")[0],
                                     line=dict(color=color, dash=dash, width=3 if m == "main" else 2)))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[f"{h} ay" for h in range(1, 13)])
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(style(fig, 300), use_container_width=True)

with right:
    st.markdown("**Bakır haberlerinin havası**")
    if "news" in D and len(D["news"]):
        nw = D["news"].copy()
        nw["date"] = pd.PeriodIndex(nw["month"], freq="M").to_timestamp("M")
        nw = nw[nw["date"] >= nw["date"].max() - pd.DateOffset(years=5)]
        col = "finbert_net" if "finbert_net" in nw and nw["finbert_net"].notna().any() else "news_tone"
        last = nw.dropna(subset=[col]).iloc[-1]
        z = (last[col] - nw[col].mean()) / nw[col].std()
        mood = "olumlu" if z > 0.5 else "olumsuz" if z < -0.5 else "nötr"
        st.caption(f"{'FinBERT başlık duyarlılığı' if col == 'finbert_net' else 'GDELT haber tonu'} · "
                   f"son ay ({last['month']}) 5 yıllık ortalamaya göre **{mood}**")
        fig = go.Figure(go.Bar(x=nw["date"], y=nw[col] - nw[col].mean(),
                               marker_color=[COPPER if v > 0 else "#5b8db8" for v in nw[col] - nw[col].mean()]))
        st.plotly_chart(style(fig, 270), use_container_width=True)
    else:
        st.info("Haber verisi henüz yok. `./run.sh news` ile indirilir.")

# ---- details
with st.expander("Tahmin tablosu"):
    t = fc[["forecast_month", "main_usd_t", "ml_usd_t", "main_mape_pct", "ml_mape_pct",
            "main_direction_pct", "ml_direction_pct"]].copy()
    t.columns = ["Ay", "Ana tahmin", "ML modeli", "Ana tahmin hata %", "ML hata %",
                 "Ana tahmin yön isabeti %", "ML yön isabeti %"]
    st.dataframe(t.style.format({c: "{:,.0f}" for c in ("Ana tahmin", "ML modeli")} |
                                {c: "{:.1f}" for c in t.columns[3:]}), hide_index=True, use_container_width=True)

with st.expander("ML modeli neye bakıyor?"):
    if "imp" in D:
        h = st.select_slider("Ufuk", options=list(range(1, 13)), value=3, format_func=lambda x: f"{x} ay")
        im = D["imp"][(D["imp"].horizon == h) & (D["imp"].model == "lgbm")].nlargest(10, "importance")[::-1]
        fig = go.Figure(go.Bar(x=im["importance"], y=im["label"], orientation="h", marker_color=ML))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 330), use_container_width=True)
        st.caption("Ayarlanmış LightGBM modelinde her girdinin kazanç payı (son tahmin).")

with st.expander("Nasıl çalışıyor?"):
    st.markdown(
        "- **Ana tahmin**, geçmişte en isabetli çıkan yöntemi kullanır. Bir model ancak geriye dönük testte "
        "LME vadeli eğrisinden (ay sonu fiyatı + 3 ay vadeli fark) daha az hata yaptıysa ağırlık alır.\n"
        "- **ML modeli** üç modelin ortalamasıdır: ElasticNet (bakır + 55 makro seriden faktörler), "
        "LightGBM (aynısı + haber duyarlılığı) ve haber duyarlılığıyla ElasticNet. "
        "Hiperparametreler zaman serisi çapraz doğrulamasıyla ayarlanır.\n"
        "- **Haber duyarlılığı**: GDELT'ten bakır piyasası haberleri; başlıklar FinBERT ile puanlanır (2017+).\n"
        "- Her şey 2015'ten beri ay ay, yalnızca o gün bilinen veriyle yeniden yapılarak test edilir.\n"
        "- Yatırım tavsiyesi değildir.")
