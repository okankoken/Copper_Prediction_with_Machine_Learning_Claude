"""Drivers page: which inputs push copper up or down."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import DOWN, MUTED, UP, data, hero, kpi_cards, note, section, style, tr

D = data()
man = D["manifest"]
hero("Etkenler · hangi veri bakırı yukarı, hangisi aşağı itiyor?")
if "effects" not in D:
    st.info("Etki tablosu henüz yok.")
    st.stop()
E = D["effects"]

months = st.radio("Ufuk", [1, 3, 6, 12], index=1, horizontal=True, format_func=lambda m: f"{m} ay sonrası")
e = E[E["months"] == months].copy()
by = e.groupby(["series", "family_tr"], as_index=False).agg(usd=("contribution_usd", "sum"), label=("label", "first"))
by["name"] = by["label"].str.replace(r" \((3|12) aylık değişim\)$", "", regex=True)
fam = by.groupby("family_tr", as_index=False)["usd"].sum()
fam = fam.reindex(fam["usd"].abs().sort_values(ascending=False).index)

up_ = by.nlargest(1, "usd").iloc[0]
down_ = by.nsmallest(1, "usd").iloc[0]
kpi_cards([
    {"label": "Toplam yukarı itiş", "value": f"+{tr(by.loc[by.usd > 0, 'usd'].sum())} $", "sub": "pozitif katkılar"},
    {"label": "Toplam aşağı çekiş", "value": f"−{tr(-by.loc[by.usd < 0, 'usd'].sum())} $", "sub": "negatif katkılar"},
    {"label": "En güçlü yukarı", "value": up_["name"], "sub": f"+{tr(up_['usd'])} $"},
    {"label": "En güçlü aşağı", "value": down_["name"], "sub": f"−{tr(-down_['usd'])} $"},
])

left, right = st.columns([3, 2])
with left:
    section("Bugünkü tahmine etkisi",
            f"Son veri ({man['forecast_origin_tr']}) ile her verinin {months} ay sonrası tahminine katkısı, USD/ton")
    top = by.reindex(by["usd"].abs().sort_values(ascending=False).index).head(14)[::-1]
    fig = go.Figure(go.Bar(x=top["usd"], y=top["name"], orientation="h",
                           marker_color=[UP if v > 0 else DOWN for v in top["usd"]], customdata=top["family_tr"],
                           hovertemplate="%{y} (%{customdata}): %{x:+,.0f} $<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(tickformat="+,.0f", ticksuffix=" $")
    with st.container(border=True):
        st.plotly_chart(style(fig, 440), use_container_width=True)
with right:
    section("Gruplara göre", "Aynı gruptaki verilerin toplam katkısı")
    f2 = fam[fam["usd"].abs() >= 0.5][::-1]
    fig = go.Figure(go.Bar(x=f2["usd"], y=f2["family_tr"], orientation="h",
                           marker_color=[UP if v > 0 else DOWN for v in f2["usd"]],
                           hovertemplate="%{y}: %{x:+,.0f} $<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(tickformat="+,.0f", ticksuffix=" $")
    with st.container(border=True):
        st.plotly_chart(style(fig, 440), use_container_width=True)

section("Geçmişte tutarlı ilişkiler",
        "2015'ten beri her 10 günde bir yeniden eğitilen ElasticNet modeli. Çubuk uzunsa model o veriyi sık ve hep "
        "aynı yönde kullandı. Sağ (yeşil): bu veri artınca bakır sonradan genelde yükseldi. Sol (kırmızı): düştü.")
h = e.dropna(subset=["positive_pct"]).copy()
h["score"] = h["consistency"] * (h["direction"].map({"+": 1, "-": -1}))
top = h.reindex(h["score"].abs().sort_values(ascending=False).index).head(14)[::-1]
fig = go.Figure(go.Bar(
    x=top["score"] * 100, y=top["label"], orientation="h",
    marker_color=[UP if v > 0 else DOWN for v in top["score"]],
    customdata=top[["selected_pct", "positive_pct"]].values,
    hovertemplate="%{y}<br>modelin kullandığı dönemler: %{customdata[0]:.0f}%"
                  "<br>bunların pozitif olanı: %{customdata[1]:.0f}%<extra></extra>"))
fig.add_vline(x=0, line_color=MUTED, line_width=1)
fig.update_xaxes(range=[-100, 100], ticksuffix="%", title="Tutarlılık (−100 hep aşağı, +100 hep yukarı)")
with st.container(border=True):
    st.plotly_chart(style(fig, 440), use_container_width=True)
note("Bunlar istatistiksel ilişkilerdir, nedensellik kanıtı değildir. Birbirine bağlı veriler (ör. faizler ve kurlar) "
     "birbirinin yerine geçebilir; bu yüzden tek bir verinin etkisi tek başına yorumlanmamalı.")
