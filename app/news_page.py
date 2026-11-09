"""News page: copper news sentiment, its link to later prices, and recent headlines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import COPPER, CSS, DOWN, INK, UP, data, style, tr
from copperlab import dekad as K

st.markdown(CSS, unsafe_allow_html=True)
D = data()
st.markdown("## 📰 Bakır haberleri")

if "news" not in D or D["news"].empty:
    st.info("Haber verisi henüz yok.")
    st.stop()

nw = D["news"].copy()
nw["date"] = pd.to_datetime(nw["date"])
has_fb = "finbert_net" in nw and nw["finbert_net"].notna().sum() > 20
first, last = nw["date"].min(), nw["date"].max()
n_heads = int(nw["finbert_n"].sum()) if "finbert_n" in nw else 0
st.caption(f"Kaynak: GDELT küresel haber veritabanı, İngilizce \"copper price\" haberleri · "
           f"{first:%b %Y} – {last:%b %Y}" + (f" · FinBERT ile puanlanan {tr(n_heads)} başlık" if n_heads else ""))

# ---------------------------------------------------------------- 1) sentiment vs price
st.markdown("#### Haberlerin havası ve bakır fiyatı")
panel = D["panel"].copy()
panel["date"] = [K.end(int(n)) for n in panel["dekad"]]
panel = panel[panel["date"] >= first]
series = {"news_tone": "GDELT haber tonu"}
if has_fb:
    series = {"finbert_net": "FinBERT başlık duyarlılığı", **series}
pick = st.radio("Ölçü", list(series), format_func=series.get, horizontal=True)
x = nw.set_index("date")[pick].rolling(3, min_periods=1).mean().dropna()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=x.index, y=x - x.mean(), name=series[pick],
                     marker_color=[UP if v > 0 else DOWN for v in x - x.mean()], opacity=0.75), secondary_y=False)
fig.add_trace(go.Scatter(x=panel["date"], y=panel["price"], name="LME bakır (USD/t)",
                         line=dict(color=COPPER, width=2)), secondary_y=True)
fig.update_yaxes(title_text="ortalamaya göre", showticklabels=False, secondary_y=False)
fig.update_yaxes(tickformat=",.0f", secondary_y=True, showgrid=False)
st.plotly_chart(style(fig, 360), use_container_width=True)
st.caption("Çubuklar 1 aylık (3 dönemlik) ortalama; yeşil = ortalamadan olumlu, kırmızı = olumsuz.")

# ---------------------------------------------------------------- 2) does it lead prices?
left, right = st.columns(2)
with left:
    st.markdown("#### Haber, sonraki fiyatı ne kadar söylüyor?")
    p = panel.set_index("dekad")
    lp = np.log(p["price"])
    s = p[pick] if pick in p else pd.Series(dtype=float)
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
        st.plotly_chart(style(fig, 260), use_container_width=True)
        st.caption("Bugünkü haber havası ile sonraki 1–12 ayın bakır getirisi arasındaki korelasyon. "
                   "±0,2'nin altı zayıf ilişki demektir.")
with right:
    st.markdown("#### Modele eklemek işe yaradı mı?")
    nt = D.get("newstest")
    if nt is not None and len(nt):
        nt = nt[nt["horizon"].isin([3, 9, 18, 36])]
        for r in nt.itertuples():
            verdict = "daha iyi" if r.skill_pct > 0 else "daha kötü"
            sig = " (anlamlı)" if r.dm_p < 0.10 and r.skill_pct > 0 else ""
            st.markdown(f"- **{r.horizon // 3} ay:** haberli model %{tr(abs(r.skill_pct), 1)} {verdict}{sig} "
                        f"· {r.n} tahmin")
        st.caption("Haberli ElasticNet ile habersiz ElasticNet aynı dönemde karşılaştırıldı.")
    else:
        st.caption("Henüz yeterli haber geçmişi yok (en az ~3 yıl gerekiyor).")

# ---------------------------------------------------------------- 3) headlines
st.markdown("#### Son başlıklar ve FinBERT yorumu")
hd = D.get("headlines")
if hd is not None and len(hd):
    hd = hd.copy()
    hd["date"] = pd.to_datetime(hd["date"]).dt.date
    tag = {"positive": "🟢 olumlu", "negative": "🔴 olumsuz", "neutral": "⚪ nötr"}
    hd["Duyarlılık"] = hd["label"].map(tag)
    f = st.radio("Göster", ["Hepsi", "🟢 olumlu", "🔴 olumsuz", "⚪ nötr"], horizontal=True, key="hf")
    if f != "Hepsi":
        hd = hd[hd["Duyarlılık"] == f]
    st.dataframe(hd[["date", "Duyarlılık", "title", "domain", "url"]].head(80).rename(
        columns={"date": "Tarih", "title": "Başlık", "domain": "Kaynak", "url": "Bağlantı"}),
        hide_index=True, use_container_width=True, height=380,
        column_config={"Bağlantı": st.column_config.LinkColumn(display_text="aç")})
else:
    st.caption("Başlıklar henüz puanlanmadı; gece çalışması tamamlanınca burada görünecek.")
