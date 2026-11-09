"""Forecast page: one forecast line, 36 ten-day periods."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COPPER, DOWN, NAVY, SLATE, UP, data, hero, kpi_cards, note, section, style, tr
from copperlab import dekad as K

D = data()
fc, M, man = D["fc"], D["metrics"], D["manifest"]
p0 = man["origin_price_usd_t"]
fc["date"] = pd.to_datetime(fc["period_end"])

hero("LME bakır nakit fiyatı · 10 günlük dönem ortalamaları · 12 ay ileri makine öğrenmesi tahmini")

cards = [{"label": "Son dönem", "value": f"{tr(p0)} $", "sub": man["forecast_origin_tr"], "main": True}]
for h, name in ((3, "1 ay sonra"), (9, "3 ay sonra"), (18, "6 ay sonra"), (36, "12 ay sonra")):
    r = fc[fc.horizon_dekads == h].iloc[0]
    cards.append({"label": name, "value": f"{tr(r.forecast_usd_t)} $", "delta": r.change_pct, "sub": r.period_tr})
kpi_cards(cards)

section("Geçmiş fiyat ve 12 aylık tahmin", "Gri: gerçekleşen 10 günlük ortalamalar · Bakır rengi: modelin tahmini")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
years = st.radio("Geçmiş", [1, 3, 5, 10], index=1, horizontal=True, format_func=lambda y: f"{y} yıl",
                 label_visibility="collapsed")
hist = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(years=years)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Gerçekleşen", line=dict(color=SLATE, width=2),
                         hovertemplate="%{y:,.0f} $<extra></extra>"))
fig.add_trace(go.Scatter(x=[hist["date"].iloc[-1], *fc["date"]], y=[p0, *fc["forecast_usd_t"]], name="Tahmin",
                         mode="lines+markers", marker=dict(size=5, color=COPPER), line=dict(color=COPPER, width=3),
                         text=["", *fc["period_tr"]], hovertemplate="%{text}: %{y:,.0f} $<extra></extra>"))
fig.add_vrect(x0=hist["date"].iloc[-1], x1=fc["date"].iloc[-1], fillcolor="rgba(200,105,58,0.06)", line_width=0)
fig.update_yaxes(tickformat=",.0f")
with st.container(border=True):
    st.plotly_chart(style(fig, 440), use_container_width=True)

ml = M[M.model == "ml"].set_index("horizon")
spans = ((3, "1 ay"), (9, "3 ay"), (18, "6 ay"), (36, "12 ay"))
better = [n for h, n in spans if h in ml.index and ml.loc[h, "skill_vs_rw_pct"] > 0]
note("<b>Bu tahmin ne kadar güvenilir?</b> 2015'ten beri her 10 günde bir, yalnızca o gün bilinen veriyle yeniden "
     "yapılan tahminlerin ortalama hatası: " +
     " · ".join(f"{n} <b>%{tr(ml.loc[h, 'mape_pct'], 1)}</b>" for h, n in spans if h in ml.index) +
     ". Yön isabeti %" + tr(ml.loc[[h for h, _ in spans if h in ml.index], "directional_acc_pct"].mean()) +
     ". 'Fiyat aynı kalır' tahmininden daha iyi olduğu ufuklar: " + (", ".join(better) if better else "yok") + ".")

left, right = st.columns([3, 2])
with left:
    section("36 dönemlik tahmin")
    t = fc[["period_tr", "forecast_usd_t", "change_pct", "backtest_mape_pct", "direction_hit_pct"]].copy()
    t.columns = ["Dönem", "Tahmin (USD/t)", "Değişim %", "Geçmiş hata %", "Yön isabeti %"]
    st.dataframe(t, hide_index=True, use_container_width=True, height=360,
                 column_config={"Tahmin (USD/t)": st.column_config.NumberColumn(format="%.0f"),
                                "Değişim %": st.column_config.NumberColumn(format="%+.1f"),
                                "Geçmiş hata %": st.column_config.NumberColumn(format="%.1f"),
                                "Yön isabeti %": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                  format="%.0f")})
with right:
    section("Bakır haberlerinin havası", "Son 3 yıl, 1 aylık ortalama")
    if "news" in D and len(D["news"]):
        nw = D["news"].copy()
        col = "finbert_net" if "finbert_net" in nw and nw["finbert_net"].notna().sum() > 30 else "news_tone"
        nw = nw.dropna(subset=[col])
        nw["date"] = pd.to_datetime(nw["date"])
        nw = nw[nw["date"] >= nw["date"].max() - pd.DateOffset(years=3)]
        dev = nw[col].rolling(3, min_periods=1).mean() - nw[col].mean()
        fig = go.Figure(go.Bar(x=nw["date"], y=dev, marker_color=[UP if v > 0 else DOWN for v in dev]))
        fig.update_yaxes(showticklabels=False)
        with st.container(border=True):
            st.plotly_chart(style(fig, 300), use_container_width=True)
        st.caption(("FinBERT başlık duyarlılığı" if col == "finbert_net" else "GDELT haber tonu") +
                   " · yeşil olumlu, kırmızı olumsuz")
    else:
        st.caption("Haber verisi henüz yok.")
