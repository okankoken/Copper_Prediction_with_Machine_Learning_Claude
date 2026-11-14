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

def zigzag(values: list[float], threshold: float = 0.012) -> list[int]:
    """Indices of turning points: a new peak/trough is confirmed once the price has moved
    `threshold` (1.2%) the other way. Always includes the first and last point."""
    v, n = values, len(values)
    piv, trend, ext = [0], 0, 0
    for i in range(1, n):
        if trend == 0:
            if v[i] >= v[0] * (1 + threshold):
                trend, ext = 1, i
            elif v[i] <= v[0] * (1 - threshold):
                trend, ext = -1, i
        elif trend == 1:
            if v[i] > v[ext]:
                ext = i
            elif v[i] <= v[ext] * (1 - threshold):
                piv.append(ext)
                trend, ext = -1, i
        else:
            if v[i] < v[ext]:
                ext = i
            elif v[i] >= v[ext] * (1 + threshold):
                piv.append(ext)
                trend, ext = 1, i
    if trend != 0 and piv[-1] != ext:
        piv.append(ext)
    if piv[-1] != n - 1:
        piv.append(n - 1)
    return piv


def money(v):
    return f"{tr(v)} $"


section("Geçmiş fiyat ve 12 aylık tahmin",
        "▲ yeşil: bir önceki 10 günlük döneme göre yükseliş · ▼ kırmızı: düşüş · gölgeli alanlar: tahmindeki "
        "yükseliş/düşüş evreleri ve toplam değişimleri · etiketler: önemli zirve ve dipler")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
c1, c2 = st.columns([3, 1])
years = c1.radio("Geçmiş", [1, 3, 5, 10], index=0, horizontal=True, format_func=lambda y: f"{y} yıl",
                 label_visibility="collapsed")
view = c2.radio("Görünüm", ["Çizgi", "Sütun"], horizontal=True, label_visibility="collapsed")
hist = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(years=years)]

s = pd.concat([pd.DataFrame({"date": hist["date"], "price": hist["price"], "kind": "real",
                             "label": [K.label_tr(int(n)) for n in hist["dekad"]]}),
               pd.DataFrame({"date": fc["date"], "price": fc["forecast_usd_t"], "kind": "fc", "label": fc["period_tr"]})],
              ignore_index=True)
s["chg"] = s["price"].pct_change() * 100
s["up"] = s["chg"].fillna(0) >= 0
real, fut = s[s["kind"] == "real"], s[s["kind"] == "fc"]

fig = go.Figure()
if view == "Sütun":
    for part, name, op in ((real, "Gerçekleşen", 0.45), (fut, "Tahmin", 0.95)):
        fig.add_trace(go.Bar(x=part["date"], y=part["price"], name=name, opacity=op, showlegend=False,
                             marker=dict(color=[UP if u else DOWN for u in part["up"]],
                                         line=dict(color=COPPER if name == "Tahmin" else "rgba(0,0,0,0)", width=1)),
                             customdata=part[["label", "chg"]].values,
                             hovertemplate="%{customdata[0]}<br>%{y:,.0f} $ (%{customdata[1]:+.1f}%)<extra>"
                                           + name + "</extra>"))
    fig.update_layout(bargap=0.12)
else:
    fig.add_trace(go.Scatter(x=real["date"], y=real["price"], name="Gerçekleşen", line=dict(color=SLATE, width=2),
                             customdata=real[["label", "chg"]].values,
                             hovertemplate="%{customdata[0]}<br>%{y:,.0f} $ (%{customdata[1]:+.1f}%)<extra></extra>"))
    fig.add_trace(go.Scatter(x=[real["date"].iloc[-1], *fut["date"]], y=[p0, *fut["price"]], name="Tahmin",
                             line=dict(color=COPPER, width=3), mode="lines"))
    fig.add_trace(go.Scatter(x=s["date"], y=s["price"], mode="markers", showlegend=False,
                             marker=dict(symbol=["triangle-up" if u else "triangle-down" for u in s["up"]], size=8,
                                         color=[UP if u else DOWN for u in s["up"]]),
                             customdata=s[["label", "chg"]].values,
                             hovertemplate="%{customdata[0]}<br>%{y:,.0f} $ (%{customdata[1]:+.1f}%)<extra></extra>"))

# rising / falling phases of the forecast, with their total change
fvals = [p0, *fut["price"]]
fdates = [real["date"].iloc[-1], *fut["date"]]
piv = zigzag(fvals)
for a, b in zip(piv[:-1], piv[1:]):
    ch = (fvals[b] / fvals[a] - 1) * 100
    if abs(ch) < 0.3:
        continue
    short = b - a < 4
    fig.add_vrect(x0=fdates[a], x1=fdates[b], line_width=0,
                  fillcolor="rgba(26,158,110,0.08)" if ch > 0 else "rgba(214,69,69,0.08)")
    if short:
        continue
    fig.add_annotation(x=fdates[a] + (fdates[b] - fdates[a]) / 2, y=0.02, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"{'↗' if ch > 0 else '↘'} {ch:+.1f}%".replace(".", ","),
                       font=dict(color=UP if ch > 0 else DOWN, size=12))

# important highs and lows: history and forecast extremes + forecast turning points
marks = {}
for part, tag in ((real, "geçmiş"), (fut, "tahmin")):
    hi, lo = part["price"].idxmax(), part["price"].idxmin()
    marks[hi] = (f"▲ {tag} zirvesi", True)
    marks[lo] = (f"▼ {tag} dibi", False)
for i in piv[1:-1]:
    j = fut.index[i - 1]
    marks.setdefault(j, ("▲ tepe" if fvals[i] >= fvals[i - 1] else "▼ dip", fvals[i] >= fvals[i - 1]))
for j, (txt, top) in marks.items():
    r = s.loc[j]
    fig.add_annotation(x=r["date"], y=r["price"], text=f"{txt}<br><b>{money(r['price'])}</b><br>{r['label']}",
                       showarrow=True, arrowhead=2, arrowsize=0.8, ax=0, ay=-48 if top else 48,
                       font=dict(size=10, color=UP if top else DOWN), bgcolor="rgba(255,255,255,0.85)",
                       bordercolor=UP if top else DOWN, borderwidth=1)

fig.add_vline(x=real["date"].iloc[-1], line_dash="dot", line_color=SLATE)
lo_y, hi_y = s["price"].min(), s["price"].max()
fig.update_yaxes(tickformat=",.0f", range=[lo_y - (hi_y - lo_y) * 0.25, hi_y + (hi_y - lo_y) * 0.25])
fig.update_xaxes(showgrid=True, gridcolor="rgba(100,116,139,0.12)", dtick="M1" if years <= 1 else "M3",
                 tickformat="%b %Y")
fig.update_layout(hovermode="closest")
with st.container(border=True):
    st.plotly_chart(style(fig, 500), width="stretch")

# ---------------------------------------------------------------- the 10-day periods, made visible
AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
fc["month_tr"] = [f"{AYLAR[int(p.split('-')[1]) - 1]} {p.split('-')[0]}" for p in fc["period"]]
fc["part_tr"] = [p.split(" ")[0] for p in fc["period_tr"]]

section("Önümüzdeki 10 günlük dönemler", "Her ay üç döneme ayrılır: 1–10 · 11–20 · 21–ay sonu")
kpi_cards([{"label": r.period_tr, "value": f"{tr(r.forecast_usd_t)} $", "delta": r.change_pct,
            "sub": "son döneme göre"} for r in fc.head(3).itertuples()])

section("12 ay boyunca 10 günlük tahmin", "Her sütun bir 10 günlük dönemin ortalama fiyat tahmini · "
        "yeşil: son dönemin üstü, kırmızı: altı")
fig = go.Figure(go.Bar(
    x=[fc["month_tr"], fc["part_tr"]], y=fc["forecast_usd_t"],
    marker_color=[UP if v >= p0 else DOWN for v in fc["forecast_usd_t"]],
    text=[f"{v / 1000:.2f}".replace(".", ",") + "k" for v in fc["forecast_usd_t"]], textposition="outside",
    textfont=dict(size=10), customdata=fc[["period_tr", "change_pct"]].values,
    hovertemplate="%{customdata[0]}<br>%{y:,.0f} $ (%{customdata[1]:+.1f}%)<extra></extra>"))
fig.add_hline(y=p0, line_dash="dot", line_color=SLATE,
              annotation_text=f"son dönem {tr(p0)} $", annotation_position="top left")
lo, hi = min(fc["forecast_usd_t"].min(), p0), max(fc["forecast_usd_t"].max(), p0)
fig.update_yaxes(range=[lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.45], tickformat=",.0f")
fig.update_xaxes(tickfont=dict(size=10))
fig.update_layout(hovermode="closest", bargap=0.15)
with st.container(border=True):
    st.plotly_chart(style(fig, 400), width="stretch")


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
    st.dataframe(t, hide_index=True, width="stretch", height=360,
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
            st.plotly_chart(style(fig, 300), width="stretch")
        st.caption(("FinBERT başlık duyarlılığı" if col == "finbert_net" else "GDELT haber tonu") +
                   " · yeşil olumlu, kırmızı olumsuz")
    else:
        st.caption("Haber verisi henüz yok.")
