"""Dashboard entry point:  streamlit run app/dashboard.py --server.port 8501"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LME Bakır Tahmini", page_icon="🟠", layout="wide", initial_sidebar_state="collapsed")
nav = st.navigation([st.Page("forecast_page.py", title="Tahmin", icon="📈", default=True),
                     st.Page("drivers_page.py", title="Etkenler", icon="⚖️"),
                     st.Page("news_page.py", title="Haberler", icon="📰")])
nav.run()
