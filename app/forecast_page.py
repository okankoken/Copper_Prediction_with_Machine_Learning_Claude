"""Forecast page: one forecast line, 36 ten-day periods."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COPPER, DOWN, NAVY, SLATE, TEAL, UP, data, hero, kpi_cards, note, section, style, tr
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

section("Geçmiş bakır fiyatı + 12 aylık tahmin",
        "10 günlük dönem ortalamaları · ◆ tahminin aylık ortalaması · alttaki kaydırıcıyla dönemi seçebilirsiniz")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
span = st.radio("Dönem", ["1Y", "3Y", "5Y", "10Y", "Tümü"], index=0, horizontal=True, label_visibility="collapsed")
last_real = panel["date"].max()

# forecast monthly averages: day-weighted mean of each month's three 10-day periods (the current month
# combines its actual first period with the forecast ones), shown as one diamond in mid-month
pts = pd.concat([pd.DataFrame({"dekad": panel["dekad"].astype(int), "price": panel["price"], "fc": False}),
                 pd.DataFrame({"dekad": [K.make(*map(int, p.split("-"))) for p in fc["period"]],
                               "price": fc["forecast_usd_t"], "fc": True})], ignore_index=True)
pts["days"] = [(K.end(n) - K.start(n)).days + 1 for n in pts["dekad"]]
pts["month"] = [K.start(n).to_period("M") for n in pts["dekad"]]
mon = pts.groupby("month").apply(lambda g: pd.Series({
    "avg": (g["price"] * g["days"]).sum() / g["days"].sum(), "fc": bool(g["fc"].any()),
    "mixed": bool(g["fc"].any() and not g["fc"].all()), "n": len(g)}), include_groups=False)
mon = mon[(mon["n"] == 3) & mon["fc"]]
AYLAR_TAM = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim",
             "Kasım", "Aralık"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=panel["date"], y=panel["price"], name="Gerçekleşen LME bakır", mode="lines",
                         line=dict(color=TEAL, width=2.2), hovertemplate="Gerçekleşen: <b>%{y:,.0f} $</b><extra></extra>"))
fig.add_trace(go.Scatter(x=[last_real, *fc["date"]], y=[p0, *fc["forecast_usd_t"]], name="10 günlük tahmin",
                         mode="lines+markers", line=dict(color=COPPER, width=2.5, dash="dash"),
                         marker=dict(size=5, color=COPPER), text=["son gerçekleşen", *fc["period_tr"]],
                         hovertemplate="Tahmin (%{text}): <b>%{y:,.0f} $</b><extra></extra>"))
fig.add_trace(go.Scatter(x=[per.start_time + pd.Timedelta(days=14) for per in mon.index], y=mon["avg"],
                         name="Aylık tahmin ortalaması", mode="markers",
                         marker=dict(symbol="diamond", size=10, color=NAVY, line=dict(color="white", width=1)),
                         text=[f"{AYLAR_TAM[per.month - 1]} {per.year} ortalaması" + (" (gerçek + tahmin)" if r.mixed else "")
                               for per, r in mon.iterrows()],
                         hovertemplate="%{text}: <b>%{y:,.0f} $</b><extra></extra>"))
fig.add_vline(x=last_real, line_dash="dot", line_color=COPPER, line_width=1.5)
for idx, txt, color, ay in ((fc["forecast_usd_t"].idxmax(), "Tahmin zirvesi", UP, -38),
                            (fc["forecast_usd_t"].idxmin(), "Tahmin dibi", DOWN, 38)):
    r = fc.loc[idx]
    ax = -70 if idx >= len(fc) - 6 else 0      # keep labels near the right edge inside the plot
    fig.add_annotation(x=r["date"], y=r["forecast_usd_t"], showarrow=True, arrowhead=2, ax=ax, ay=ay,
                       text=f"{txt}: <b>{tr(r['forecast_usd_t'])} $</b>", font=dict(size=10, color=color),
                       bgcolor="rgba(255,255,255,0.9)", bordercolor=color, borderwidth=1)

years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(span)
x0 = last_real - pd.DateOffset(years=years) if years else panel["date"].min()
x1 = fc["date"].max() + pd.Timedelta(days=20)
view = pd.concat([panel.loc[panel["date"] >= x0, "price"], fc["forecast_usd_t"]])
pad = (view.max() - view.min()) * 0.12
fig.update_xaxes(range=[x0, x1], rangeslider=dict(visible=True, thickness=0.08, bgcolor="#eef3f8"),
                 showgrid=True, gridcolor="rgba(100,116,139,0.10)")
fig.update_yaxes(title_text="USD / ton", tickformat=",.0f", range=[view.min() - pad, view.max() + pad])
fig = style(fig, 520)
# legend top-left so it never sits under Plotly's toolbar (top-right); a new period button resets the zoom
fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                  hovermode="x unified", hoverdistance=60, spikedistance=-1, dragmode=False,
                  hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#dfe7f1"),
                  margin=dict(l=8, r=8, t=56, b=8), uirevision=span)
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                 spikedash="dot", spikecolor="#94a3b8", hoverformat="%d %b %Y")
with st.container(border=True):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False,
                                                  "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})

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
