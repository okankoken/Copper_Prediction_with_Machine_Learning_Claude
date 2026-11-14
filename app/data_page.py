"""Metadata page: how much data there is and where every piece comes from."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COPPER, NAVY, TEAL, data, hero, kpi_cards, note, section, style, tr

D = data()
hero("Veri kaynakları · modelin kullandığı her verinin kaynağı, dönemi ve miktarı")
cat, cu, meta = D.get("catalog"), D.get("catalog_cu"), D.get("catalog_json", {})
if cat is None or cu is None:
    st.info("Veri kataloğu henüz oluşturulmadı; bir sonraki `run.cmd run` ile gelecek.")
    st.stop()

news = meta.get("news", {})
gn = news.get("google_news", {})
model = meta.get("model", {})
total_points = int(cu["observations"].sum() + cat["observations"].sum() + gn.get("headlines_downloaded", 0))
kpi_cards([
    {"label": "Veri kaynağı", "value": str(cat["provider"].nunique() + 2), "sub": "kurum + GDELT + Google News", "main": True},
    {"label": "Veri serisi", "value": str(len(cat) + len(cu)), "sub": f"{len(cu)} bakır + {len(cat)} piyasa/makro"},
    {"label": "Dönem", "value": f"{cu['first'].min()[:4]} – {cu['last'].max()[:4]}",
     "sub": f"son veri {cu['last'].max()}"},
    {"label": "Toplam veri noktası", "value": tr(total_points), "sub": "günlük + aylık + haber başlığı"},
    {"label": "Haber başlığı", "value": tr(gn.get("headlines_downloaded", 0)),
     "sub": f"{tr(gn.get('headlines_kept', 0))} ilgili başlık kullanıldı"},
])

# ---------------------------------------------------------------- providers
section("Kaynaklar", "Her kurumdan kaç seri geldiği ve hangi dönemi kapsadığı")
prov = cat.groupby(["provider", "url"], as_index=False).agg(
    series=("series", "size"), groups=("family_tr", lambda x: ", ".join(sorted(set(x)))),
    first=("first", "min"), last=("last", "max"), observations=("observations", "sum"),
    frequency=("frequency", lambda x: ", ".join(sorted(set(x)))))
lme = cu.iloc[0]
lme_other = cat[cat["provider"] == lme["provider"]]
prov = pd.concat([pd.DataFrame([{
    "provider": lme["provider"], "url": lme["url"], "series": len(cu) + len(lme_other),
    "groups": "Bakır fiyatı, vadeli fiyat, stoklar, diğer metaller ve stokları", "first": cu["first"].min()[:7],
    "last": cu["last"].max()[:7], "observations": int(cu["observations"].sum() + lme_other["observations"].sum()),
    "frequency": "Günlük"}]),
    prov[prov["provider"] != lme["provider"]]], ignore_index=True)
if gn:
    prov = pd.concat([prov, pd.DataFrame([
        {"provider": news["gdelt"]["provider"], "url": news["gdelt"]["url"], "series": 2,
         "groups": "Haber tonu, haber hacmi", "first": (news["gdelt"].get("years") or ["–"])[0],
         "last": (news["gdelt"].get("years") or ["–"])[-1], "observations": None, "frequency": "Günlük"},
        {"provider": gn["provider"], "url": gn["url"], "series": 1, "groups": "Bakır haber başlıkları (FinBERT ile)",
         "first": (gn.get("first") or "–")[:7], "last": (gn.get("last") or "–")[:7],
         "observations": gn.get("headlines_downloaded"), "frequency": "Günlük"}])], ignore_index=True)
st.dataframe(prov.rename(columns={"provider": "Kaynak", "url": "Bağlantı", "series": "Seri", "groups": "İçerik",
                                  "first": "Başlangıç", "last": "Son", "observations": "Gözlem",
                                  "frequency": "Sıklık"}),
             hide_index=True, width="stretch",
             column_config={"Bağlantı": st.column_config.LinkColumn(display_text="site"),
                            "Gözlem": st.column_config.NumberColumn(format="%d")})

# ---------------------------------------------------------------- coverage chart
section("Zaman kapsamı", "Her serinin modelde kullanılan dönemi (2008'den itibaren)")
fam = st.multiselect("Grup", sorted(cat["family_tr"].unique()), default=[], placeholder="Tüm gruplar")
c = cat if not fam else cat[cat["family_tr"].isin(fam)]
c = c.sort_values(["family_tr", "first"], ascending=[False, False])
start = pd.PeriodIndex(c["first"], freq="M").to_timestamp()
end = pd.PeriodIndex(c["last"], freq="M").to_timestamp("M")
palette = [NAVY, TEAL, COPPER, "#7a5195", "#ef5675", "#4c956c", "#ffa600", "#58508d", "#2f4b7c", "#a05195",
           "#665191", "#d45087", "#f95d6a", "#ff7c43"]
colors = {f: palette[i % len(palette)] for i, f in enumerate(sorted(cat["family_tr"].unique()))}
fig = go.Figure()
for f in c["family_tr"].unique():
    m = (c["family_tr"] == f).values
    fig.add_trace(go.Bar(y=c["label"][m], x=(end[m] - start[m]).days * 86400000, base=start[m], orientation="h",
                         name=f, marker_color=colors[f],
                         customdata=c[["observations", "completeness_pct", "provider"]][m].values,
                         hovertemplate="%{y}<br>%{base|%Y-%m} → %{x}<br>%{customdata[0]} gözlem · "
                                       "%{customdata[1]}% dolu<br>%{customdata[2]}<extra></extra>"))
fig.update_xaxes(type="date")
fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.08))
with st.container(border=True):
    st.plotly_chart(style(fig, max(320, 16 * len(c) + 80)), width="stretch")
note("Çin verileri %86–92 dolu görünür: Çin İstatistik Bürosu Ocak ve Şubat'ı tek rakam olarak yayımlar. "
     "Eksik aylar en fazla 2 ay ileri taşınır; gelecekteki bir değer hiçbir zaman geriye doldurulmaz.")

# ---------------------------------------------------------------- full table
section("Tüm seriler", "Kaynak, yayın gecikmesi ve son değer")
q = st.text_input("Ara", placeholder="ör. Çin, faiz, FRED", label_visibility="collapsed")
t = cat.copy()
if q:
    t = t[t.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
st.dataframe(t[["label", "family_tr", "provider", "frequency", "publication", "first", "last", "observations",
                "completeness_pct", "last_value", "transform", "file", "url"]].rename(columns={
                    "label": "Seri", "family_tr": "Grup", "provider": "Kaynak", "frequency": "Sıklık",
                    "publication": "Ne zaman bilinir", "first": "Başlangıç", "last": "Son", "observations": "Gözlem",
                    "completeness_pct": "Doluluk %", "last_value": "Son değer", "transform": "Modelde",
                    "file": "Dosya", "url": "Bağlantı"}),
             hide_index=True, width="stretch", height=420,
             column_config={"Doluluk %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                            "Son değer": st.column_config.NumberColumn(format="%.2f"),
                            "Bağlantı": st.column_config.LinkColumn(display_text="site")})
st.caption("'Ne zaman bilinir': model bir ayın verisini ancak bu kadar gecikmeyle kullanır (resmi istatistikler "
           "geç yayımlandığı için). 'Modelde': serinin 3 ve 12 aylık değişimi girdi olarak kullanılır.")

# ---------------------------------------------------------------- copper + news + model
left, right = st.columns(2)
with left:
    section("Bakır fiyat verisi", "LME günlük verisi, 10 günlük dönem ortalamasına çevrilir")
    st.dataframe(cu[["label", "first", "last", "observations", "last_value"]].rename(columns={
        "label": "Seri", "first": "Başlangıç", "last": "Son", "observations": "Gün", "last_value": "Son değer"}),
        hide_index=True, width="stretch",
        column_config={"Son değer": st.column_config.NumberColumn(format="%.0f")})
    if model:
        section("Modelin veri hacmi")
        kpi_cards([
            {"label": "10 günlük dönem", "value": tr(meta.get("panel", {}).get("rows", 0)),
             "sub": f"eğitim {model.get('training_start', '')[:7]} başlar"},
            {"label": "Model girdisi", "value": str(model["inputs_copper"] + model["inputs_macro"] + model["inputs_news"]),
             "sub": f"{model['inputs_copper']} bakır · {model['inputs_macro']} makro · {model['inputs_news']} haber"},
            {"label": "Geriye dönük tahmin", "value": tr(model["predictions"]),
             "sub": f"{model['backtest_origins']} başlangıç × {model['horizons']} dönem"},
        ])
with right:
    section("Haber verisi")
    if gn:
        removed = gn.get("removed", {})
        st.markdown(
            f"- **GDELT:** {news['gdelt']['query']} haberlerinin {news['gdelt']['measures']} "
            f"({', '.join(news['gdelt'].get('years', [])[:1] + news['gdelt'].get('years', [])[-1:])} arası, "
            f"{len(news['gdelt'].get('years', []))} yıl)\n"
            f"- **Google News:** {tr(gn['headlines_downloaded'])} başlık indirildi, "
            f"**{tr(gn['headlines_kept'])}** bakırla ilgili başlık kullanıldı\n"
            f"- **Elenen:** " + ", ".join(f"{k} {tr(v)}" for k, v in removed.items()) + "\n"
            f"- **Duyarlılık modeli:** [{news['finbert']['provider']}]({news['finbert']['url']}) + bakıra özel "
            f"fiyat etkisi kuralları")
        with st.expander("Google News aramaları"):
            for k, v in gn.get("searches", {}).items():
                st.markdown(f"**{'Piyasa' if k == 'gnews' else 'Madencilik · Şili · Peru'}:** " + " · ".join(v))
        src = pd.Series(gn.get("top_sources", {})).sort_values()
        if len(src):
            fig = go.Figure(go.Bar(x=src.values, y=src.index, orientation="h", marker_color=TEAL))
            with st.container(border=True):
                st.plotly_chart(style(fig, 330), width="stretch")
            st.caption("Kullanılan başlıkların en sık geldiği haber kaynakları")
