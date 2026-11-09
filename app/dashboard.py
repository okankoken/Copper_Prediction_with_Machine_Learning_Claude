"""Dashboard entry point:  streamlit run app/dashboard.py --server.port 8501"""
from __future__ import annotations

import streamlit as st

from common import APP_NAME, data, page_setup, tr

st.set_page_config(page_title=APP_NAME, page_icon="🟠", layout="wide", initial_sidebar_state="expanded")
page_setup()

nav = st.navigation([st.Page("forecast_page.py", title="Tahmin", icon="📈", default=True),
                     st.Page("drivers_page.py", title="Etkenler", icon="⚖️"),
                     st.Page("news_page.py", title="Haberler", icon="📰")])

D = data()
man = D["manifest"]
with st.sidebar:
    st.markdown('<div class="side-brand">🟠 CopperCast</div>'
                '<div class="side-sub">LME bakır tahmini · 10 günlük dönemler · 12 ay</div>', unsafe_allow_html=True)
    news = man.get("news_dekads") or ["–", "–"]
    for title, value in (("Son veri", man["forecast_origin_tr"]),
                         ("Son dönem ortalaması", f"{tr(man['origin_price_usd_t'])} USD/t"),
                         ("Güncelleme", man["generated_at_utc"][:16].replace("T", " ") + " UTC"),
                         ("Model", "ElasticNet + LightGBM + haberli ElasticNet"),
                         ("Haber verisi", f"{news[0][:7]} → {news[1][:7]}")):
        st.markdown(f'<div class="side-card"><b>{title}</b>{value}</div>', unsafe_allow_html=True)
    st.caption("⚠️ Analiz amaçlıdır, yatırım tavsiyesi değildir.")

nav.run()
