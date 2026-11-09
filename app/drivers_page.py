"""Drivers page: which inputs push copper up or down."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import CSS, DOWN, UP, data, style, tr

st.markdown(CSS, unsafe_allow_html=True)
D = data()
man = D["manifest"]
if "effects" not in D:
    st.info("Etki tablosu henüz yok.")
    st.stop()
E = D["effects"]

st.markdown("## ⚖️ Bakırı ne yukarı, ne aşağı itiyor?")
months = st.radio("Ufuk", [1, 3, 6, 12], index=1, horizontal=True, format_func=lambda m: f"{m} ay sonrası")
e = E[E["months"] == months].copy()

# ---------------------------------------------------------------- today
st.markdown("#### Bugünkü tahmine etkisi")
st.caption(f"Son veri ({man['forecast_origin_tr']}) ile her verinin {months} ay sonrası tahminine katkısı, USD/ton. "
           "Yeşil yukarı, kırmızı aşağı. Aynı serinin 3 ve 12 aylık değişimleri toplandı.")
by = e.groupby(["series", "family_tr"], as_index=False).agg(usd=("contribution_usd", "sum"),
                                                             label=("label", "first"))
by["name"] = by["label"].str.replace(r" \((3|12) aylık değişim\)$", "", regex=True)
top = by.reindex(by["usd"].abs().sort_values(ascending=False).index).head(14)[::-1]
fig = go.Figure(go.Bar(x=top["usd"], y=top["name"], orientation="h",
                       marker_color=[UP if v > 0 else DOWN for v in top["usd"]],
                       customdata=top["family_tr"], hovertemplate="%{y} (%{customdata}): %{x:+,.0f} $<extra></extra>"))
fig.add_vline(x=0, line_color="#8f9bb3", line_width=1)
fig.update_xaxes(tickformat="+,.0f", ticksuffix=" $")
st.plotly_chart(style(fig, 430), use_container_width=True)

fam = by.groupby("family_tr", as_index=False)["usd"].sum()
fam = fam.reindex(fam["usd"].abs().sort_values(ascending=False).index)
st.caption("Gruplara göre toplam: " + " · ".join(
    f"{r.family_tr} {'+' if r.usd >= 0 else '−'}{tr(abs(r.usd))} USD" for r in fam.itertuples() if abs(r.usd) >= 1))

# ---------------------------------------------------------------- history
st.markdown("#### Geçmişte tutarlı ilişkiler")
st.caption("2015'ten beri her 10 günde bir yeniden eğitilen ElasticNet modeli. Çubuk ne kadar uzunsa, model o veriyi "
           "o kadar sık ve hep aynı yönde kullandı. **Sağ (yeşil):** bu veri artınca bakır sonraki dönemde genelde "
           "yükseldi. **Sol (kırmızı):** düştü. İlişkidir, nedensellik kanıtı değildir.")
h = e.dropna(subset=["positive_pct"]).copy()
h["score"] = h["consistency"] * (h["direction"].map({"+": 1, "-": -1}))
top = h.reindex(h["score"].abs().sort_values(ascending=False).index).head(14)[::-1]
fig = go.Figure(go.Bar(
    x=top["score"] * 100, y=top["label"], orientation="h",
    marker_color=[UP if v > 0 else DOWN for v in top["score"]],
    customdata=top[["selected_pct", "positive_pct"]].values,
    hovertemplate="%{y}<br>modelin kullandığı dönemler: %{customdata[0]:.0f}%"
                  "<br>bunların pozitif olanı: %{customdata[1]:.0f}%<extra></extra>"))
fig.add_vline(x=0, line_color="#8f9bb3", line_width=1)
fig.update_xaxes(range=[-100, 100], ticksuffix="%", title="Tutarlılık (−100 hep aşağı, +100 hep yukarı)")
st.plotly_chart(style(fig, 430), use_container_width=True)
