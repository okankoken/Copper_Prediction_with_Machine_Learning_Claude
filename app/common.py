"""Shared loading and styling for the dashboard pages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
COPPER, INK, GRID = "#d0702f", "#8f9bb3", "rgba(128,128,128,0.18)"
UP, DOWN = "#2e9d6a", "#d1495b"
CSS = ("<style>.block-container{padding-top:2rem;max-width:1150px} "
       "[data-testid=stMetricValue]{font-size:1.45rem} [data-testid=stMetricLabel] p{font-size:.85rem}</style>")


def tr(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, nd=1):
    return f"{x:+.{nd}f}%".replace(".", ",")


@st.cache_data
def load(stamp: float) -> dict:
    d = {}
    for k, f in {"fc": "forecast_latest.csv", "metrics": "metrics_by_horizon.csv", "news": "news_sentiment.csv",
                 "effects": "input_effects.csv", "newstest": "news_value_test.csv",
                 "headlines": "news_headlines_recent.csv", "headcounts": "news_headlines_monthly_counts.csv"}.items():
        try:
            d[k] = pd.read_csv(OUT / f)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
    d["panel"] = pd.read_csv(ROOT / "data" / "panel" / "copper_dekad_panel.csv",
                             usecols=["dekad", "label", "price", "news_tone", "finbert_net"])
    d["manifest"] = json.loads((OUT / "run_manifest.json").read_text())
    return d


def data() -> dict:
    if not (OUT / "forecast_latest.csv").exists():
        st.error("Çıktı yok. Önce `./run.sh` çalıştırın.")
        st.stop()
    return load((OUT / "forecast_latest.csv").stat().st_mtime)


def style(fig, h):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID)
    return fig
