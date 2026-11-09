"""Forecast page: one forecast line, 36 ten-day periods."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COPPER, CSS, DOWN, INK, UP, data, pct, style, tr
from copperlab import dekad as K

st.markdown(CSS, unsafe_allow_html=True)
D = data()
fc, M, man = D["fc"], D["metrics"], D["manifest"]
p0 = man["origin_price_usd_t"]
fc["date"] = pd.to_datetime(fc["period_end"])

st.markdown("## 🟠 LME Bakır Tahmini")
st.caption(f"Nakit fiyatın 10 günlük dönem ortalaması, USD/ton · son veri **{man['forecast_origin_tr']}** · "
           f"güncelleme {man['generated_at_utc'][:16].replace('T', ' ')} UTC")

cols = st.columns(5)
cols[0].metric("Şimdi", tr(p0), help=man["forecast_origin_tr"])
for c, (h, name) in zip(cols[1:], ((3, "1 ay"), (9, "3 ay"), (18, "6 ay"), (36, "12 ay"))):
    r = fc[fc.horizon_dekads == h].iloc[0]
    c.metric(name, tr(r.forecast_usd_t), pct(r.change_pct), help=r.period_tr)

panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
hist = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(years=3)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Gerçekleşen", line=dict(color=INK, width=2)))
fig.add_trace(go.Scatter(x=[hist["date"].iloc[-1], *fc["date"]], y=[p0, *fc["forecast_usd_t"]], name="Tahmin",
                         mode="lines+markers", marker=dict(size=5), line=dict(color=COPPER, width=3),
                         text=["", *fc["period_tr"]], hovertemplate="%{text}: %{y:,.0f}<extra></extra>"))
fig.update_yaxes(tickformat=",.0f")
st.plotly_chart(style(fig, 430), use_container_width=True)

ml = M[M.model == "ml"].set_index("horizon")
rw = M[M.model == "rw"].set_index("horizon")
acc = " · ".join(f"{n}: %{tr(ml.loc[h, 'mape_pct'], 1)} hata, yön %{tr(ml.loc[h, 'directional_acc_pct'])}"
                 for h, n in ((3, "1 ay"), (9, "3 ay"), (18, "6 ay"), (36, "12 ay")) if h in ml.index)
better = [n for h, n in ((3, "1 ay"), (9, "3 ay"), (18, "6 ay"), (36, "12 ay"))
          if h in ml.index and ml.loc[h, "skill_vs_rw_pct"] > 0]
st.caption(f"**Geçmiş başarı** (2015'ten beri her 10 günde bir yeniden yapılan tahminler): {acc}. "
           + (f"'Fiyat aynı kalır' tahmininden daha iyi olduğu ufuklar: {', '.join(better)}."
              if better else "Hiçbir ufukta 'fiyat aynı kalır' tahmininden daha az hata yapmadı; "
              "tahmin bir yön görüşüdür, kesinlik değil."))

left, right = st.columns([3, 2])
with left:
    with st.expander("36 dönemlik tahmin tablosu"):
        t = fc[["period_tr", "forecast_usd_t", "change_pct", "backtest_mape_pct", "direction_hit_pct"]].copy()
        t.columns = ["Dönem", "Tahmin (USD/t)", "Değişim %", "Geçmiş hata %", "Yön isabeti %"]
        st.dataframe(t.style.format({"Tahmin (USD/t)": "{:,.0f}", "Değişim %": "{:+.1f}",
                                     "Geçmiş hata %": "{:.1f}", "Yön isabeti %": "{:.0f}"}),
                     hide_index=True, use_container_width=True, height=420)
with right:
    if "news" in D and len(D["news"]):
        nw = D["news"].copy()
        col = "finbert_net" if "finbert_net" in nw and nw["finbert_net"].notna().sum() > 30 else "news_tone"
        nw = nw.dropna(subset=[col])
        nw["date"] = pd.to_datetime(nw["date"])
        nw = nw[nw["date"] >= nw["date"].max() - pd.DateOffset(years=3)]
        dev = nw[col].rolling(3, min_periods=1).mean() - nw[col].mean()
        st.markdown("**Bakır haberlerinin havası**")
        st.caption("FinBERT başlık duyarlılığı" if col == "finbert_net" else "GDELT haber tonu")
        fig = go.Figure(go.Bar(x=nw["date"], y=dev, marker_color=[UP if v > 0 else DOWN for v in dev]))
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(style(fig, 180), use_container_width=True)
    else:
        st.caption("Haber verisi henüz yok.")

st.caption("Yatırım tavsiyesi değildir.")
