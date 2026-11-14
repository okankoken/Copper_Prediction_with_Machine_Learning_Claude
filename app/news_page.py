"""News page: copper news, its likely price impact, its link to later prices, and recent headlines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import COPPER, DOWN, MUTED, UP, data, hero, kpi_cards, note, section, style, tr
from copperlab import dekad as K

D = data()
hero("Haberler · bakır başlıklarının fiyata olası etkisi, GDELT haber tonu ve FinBERT")

if "news" not in D or D["news"].empty:
    st.info("Haber verisi henüz yok.")
    st.stop()

nw = D["news"].copy()
nw["date"] = pd.to_datetime(nw["date"])
first, last = nw["date"].min(), nw["date"].max()
n_kept = int(nw["finbert_n"].sum()) if "finbert_n" in nw else 0
removed = D.get("filter")
n_removed = int(removed["count"].sum()) if removed is not None and len(removed) else 0
hd_all = D.get("headlines")
recent = hd_all[pd.to_datetime(hd_all["date"]) >= pd.to_datetime(hd_all["date"]).max() - pd.Timedelta(days=30)] \
    if hd_all is not None and len(hd_all) else pd.DataFrame()
n_up = int((recent.get("direction") == "up").sum()) if len(recent) else 0
n_down = int((recent.get("direction") == "down").sum()) if len(recent) else 0
kpi_cards([
    {"label": "Kapsanan dönem", "value": f"{first:%Y} – {last:%Y}", "sub": f"{first:%b %Y} → {last:%b %Y}"},
    {"label": "İlgili başlık", "value": tr(n_kept), "sub": f"{tr(n_removed)} alakasız/tekrar başlık elendi"},
    {"label": "Son 30 gün · yukarı", "value": str(n_up), "sub": "fiyatı yukarı itecek haber"},
    {"label": "Son 30 gün · aşağı", "value": str(n_down), "sub": "fiyatı aşağı çekecek haber"},
])

# ---------------------------------------------------------------- 1) news vs price
section("Haberlerin bakır fiyatına etkisi ve fiyat", "Çubuklar 1 aylık ortalama · yeşil fiyatı yukarı, kırmızı aşağı "
        "itecek haberlerin ağır bastığı dönemler")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
panel = panel[panel["date"] >= first]
series = {}
if "impact_net" in nw and nw["impact_net"].notna().sum() > 20:
    series["impact_net"] = "Bakır fiyat etkisi (kural tabanlı)"
if "finbert_net" in nw and nw["finbert_net"].notna().sum() > 20:
    series["finbert_net"] = "Genel finans tonu (FinBERT)"
series["news_tone"] = "GDELT haber tonu"
pick = st.radio("Ölçü", list(series), format_func=series.get, horizontal=True, label_visibility="collapsed")
x = nw.set_index("date")[pick].rolling(3, min_periods=1).mean().dropna()
base = 0.0 if pick == "impact_net" else x.mean()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=x.index, y=x - base, name=series[pick],
                     marker_color=[UP if v > 0 else DOWN for v in x - base], opacity=0.75), secondary_y=False)
fig.add_trace(go.Scatter(x=panel["date"], y=panel["price"], name="LME bakır (USD/t)",
                         line=dict(color=COPPER, width=2)), secondary_y=True)
fig.update_yaxes(showticklabels=False, secondary_y=False)
fig.update_yaxes(tickformat=",.0f", secondary_y=True, showgrid=False)
with st.container(border=True):
    st.plotly_chart(style(fig, 380), use_container_width=True)

# ---------------------------------------------------------------- 2) does it lead prices / help the model?
left, right = st.columns(2)
with left:
    section("Haber, sonraki fiyatı ne kadar söylüyor?",
            "Seçili ölçü ile sonraki 1–12 ayın bakır getirisi arasındaki korelasyon")
    p = panel.set_index("dekad")
    lp = np.log(p["price"])
    s = D["news"].set_index("dekad")[pick].rolling(3, min_periods=1).mean() if pick in D["news"] else pd.Series(dtype=float)
    rows = []
    for months in range(1, 13):
        both = pd.concat([s, lp.shift(-3 * months) - lp], axis=1).dropna()
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
        note("Henüz yeterli haber geçmişi yok.")

# ---------------------------------------------------------------- 3) headlines
section("Son başlıklar ve fiyata olası etkisi",
        "Sadece bakırla ilgili başlıklar. Etki kuralları: fiyat yönü · arz kesintisi/grev (yukarı) · üretim/stok "
        "artışı (aşağı) · talep. Kural yoksa ve haber fiyatla ilgiliyse FinBERT tonu kullanılır.")
if hd_all is not None and len(hd_all):
    hd = hd_all.copy()
    hd["date"] = pd.to_datetime(hd["date"]).dt.date
    hd["Etki"] = hd["direction"].map({"up": "▲ yukarı", "down": "▼ aşağı", "neutral": "– nötr"})
    hd["Konu"] = hd["topic"].map({"market": "Piyasa", "supply": "Madencilik · Şili · Peru"})
    hd["FinBERT"] = hd["label"].map({"positive": "olumlu", "negative": "olumsuz", "neutral": "nötr"})
    c1, c2 = st.columns(2)
    f = c1.radio("Etki", ["Hepsi", "▲ yukarı", "▼ aşağı", "– nötr"], horizontal=True, key="hf")
    t = c2.radio("Konu", ["Hepsi", "Piyasa", "Madencilik · Şili · Peru"], horizontal=True, key="ht")
    if f != "Hepsi":
        hd = hd[hd["Etki"] == f]
    if t != "Hepsi":
        hd = hd[hd["Konu"] == t]
    st.dataframe(hd[["date", "Etki", "reason", "title", "Konu", "domain", "FinBERT", "url"]].head(100).rename(
        columns={"date": "Tarih", "reason": "Gerekçe", "title": "Başlık", "domain": "Kaynak", "url": "Bağlantı"}),
        hide_index=True, use_container_width=True, height=420,
        column_config={"Bağlantı": st.column_config.LinkColumn(display_text="aç"),
                       "Başlık": st.column_config.TextColumn(width="large")})
    if removed is not None and len(removed):
        st.caption("Elenen başlıklar: " + " · ".join(f"{r.reason} {tr(r.count)}" for r in removed.itertuples()))
else:
    note("Başlıklar henüz puanlanmadı.")
