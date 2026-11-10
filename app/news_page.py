"""News page: copper news sentiment, its link to later prices, and recent headlines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import COPPER, DOWN, UP, data, hero, kpi_cards, note, section, style, tr
from copperlab import dekad as K

D = data()
hero("Haberler · GDELT haber tonu ve Google News başlıklarının FinBERT duyarlılık analizi")

if "news" not in D or D["news"].empty:
    st.info("Haber verisi henüz yok.")
    st.stop()

nw = D["news"].copy()
nw["date"] = pd.to_datetime(nw["date"])
has_fb = "finbert_net" in nw and nw["finbert_net"].notna().sum() > 20
first, last = nw["date"].min(), nw["date"].max()
n_heads = int(nw["finbert_n"].sum()) if "finbert_n" in nw else 0
n_arts = int(nw["news_articles"].sum()) if "news_articles" in nw else 0
recent = nw.dropna(subset=["news_tone"]).tail(3)["news_tone"].mean()
kpi_cards([
    {"label": "Kapsanan dönem", "value": f"{first:%Y} – {last:%Y}", "sub": f"{first:%b %Y} → {last:%b %Y}"},
    {"label": "Haber sayısı (GDELT)", "value": tr(n_arts), "sub": "\"copper price\" geçen haberler"},
    {"label": "FinBERT başlığı", "value": tr(n_heads), "sub": "Google News başlıkları"},
    {"label": "Son ay haber tonu", "value": tr(recent, 2).replace("-", "−"),
     "sub": "ortalama " + tr(nw["news_tone"].mean(), 2).replace("-", "−")},
])

# ---------------------------------------------------------------- 1) sentiment vs price
section("Haberlerin havası ve bakır fiyatı", "Çubuklar 1 aylık ortalama · yeşil ortalamadan olumlu, kırmızı olumsuz")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
panel = panel[panel["date"] >= first]
series = {"news_tone": "GDELT haber tonu"}
if "finbert_supply_net" in nw and nw["finbert_supply_net"].notna().sum() > 20:
    series = {"finbert_supply_net": "Madencilik / Şili / Peru haberleri", **series}
if has_fb:
    series = {"finbert_net": "Piyasa başlıkları (FinBERT)", **series}
pick = st.radio("Ölçü", list(series), format_func=series.get, horizontal=True, label_visibility="collapsed")
x = nw.set_index("date")[pick].rolling(3, min_periods=1).mean().dropna()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=x.index, y=x - x.mean(), name=series[pick],
                     marker_color=[UP if v > 0 else DOWN for v in x - x.mean()], opacity=0.75), secondary_y=False)
fig.add_trace(go.Scatter(x=panel["date"], y=panel["price"], name="LME bakır (USD/t)",
                         line=dict(color=COPPER, width=2)), secondary_y=True)
fig.update_yaxes(title_text="ortalamaya göre", showticklabels=False, secondary_y=False)
fig.update_yaxes(tickformat=",.0f", secondary_y=True, showgrid=False)
with st.container(border=True):
    st.plotly_chart(style(fig, 380), use_container_width=True)

# ---------------------------------------------------------------- 2) does it lead prices?
left, right = st.columns(2)
with left:
    section("Haber, sonraki fiyatı ne kadar söylüyor?",
            "Bugünkü haber havası ile sonraki 1–12 ayın bakır getirisi arasındaki korelasyon")
    p = panel.set_index("dekad")
    lp = np.log(p["price"])
    s = (D["news"].set_index("dekad")[pick].rolling(3, min_periods=1).mean()
         if pick in D["news"] else pd.Series(dtype=float))
    rows = []
    for months in range(1, 13):
        fut = lp.shift(-3 * months) - lp
        both = pd.concat([s, fut], axis=1).dropna()
        if len(both) > 30:
            rows.append((months, both.iloc[:, 0].corr(both.iloc[:, 1])))
    if rows:
        c = pd.DataFrame(rows, columns=["months", "corr"])
        fig = go.Figure(go.Bar(x=c["months"], y=c["corr"], marker_color=[UP if v > 0 else DOWN for v in c["corr"]]))
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[f"{m} ay" for m in range(1, 13)])
        fig.update_yaxes(range=[-0.5, 0.5])
        with st.container(border=True):
            st.plotly_chart(style(fig, 280), use_container_width=True)
        st.caption("±0,2'nin altı zayıf ilişki demektir.")
with right:
    section("Modele eklemek işe yaradı mı?", "Haberli ElasticNet ile habersiz ElasticNet, aynı dönemde")
    nt = D.get("newstest")
    if nt is not None and len(nt):
        nt = nt[nt["horizon"].isin([3, 9, 18, 36])]
        kpi_cards([{"label": f"{r.horizon // 3} ay", "value": f"{tr(r.skill_pct, 1)}%".replace("-", "−"),
                    "sub": ("daha iyi" if r.skill_pct > 0 else "daha kötü") +
                           (" · anlamlı" if r.dm_p < 0.10 and r.skill_pct > 0 else "") + f" · {r.n} tahmin"}
                   for r in nt.itertuples()])
    else:
        note("Henüz yeterli haber geçmişi yok (en az ~3 yıl gerekiyor). Gece çalışması tamamlanınca burada "
             "haberli modelin habersiz modele göre başarısı görünecek.")

# ---------------------------------------------------------------- 3) headlines
section("Son başlıklar ve FinBERT yorumu")
hd = D.get("headlines")
if hd is not None and len(hd):
    hd = hd.copy()
    hd["date"] = pd.to_datetime(hd["date"]).dt.date
    tag = {"positive": "🟢 olumlu", "negative": "🔴 olumsuz", "neutral": "⚪ nötr"}
    hd["Duyarlılık"] = hd["label"].map(tag)
    hd["Konu"] = hd["topic"].map({"market": "Piyasa", "supply": "Madencilik · Şili · Peru"})         if "topic" in hd else "Piyasa"
    c1, c2 = st.columns(2)
    f = c1.radio("Duyarlılık", ["Hepsi", "🟢 olumlu", "🔴 olumsuz", "⚪ nötr"], horizontal=True, key="hf")
    t = c2.radio("Konu", ["Hepsi", "Piyasa", "Madencilik · Şili · Peru"], horizontal=True, key="ht")
    if f != "Hepsi":
        hd = hd[hd["Duyarlılık"] == f]
    if t != "Hepsi":
        hd = hd[hd["Konu"] == t]
    st.dataframe(hd[["date", "Duyarlılık", "Konu", "title", "domain", "url"]].head(80).rename(
        columns={"date": "Tarih", "title": "Başlık", "domain": "Kaynak", "url": "Bağlantı"}),
        hide_index=True, use_container_width=True, height=380,
        column_config={"Bağlantı": st.column_config.LinkColumn(display_text="aç")})
else:
    note("Başlıklar henüz puanlanmadı; gece çalışması tamamlanınca burada görünecek.")
