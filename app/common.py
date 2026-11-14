"""Shared data loading, palette and layout helpers for the dashboard pages."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

APP_NAME = "Copper Prediction with Machine Learning Claude"

# palette: deep blue / teal with a copper accent, on a cool light background
NAVY, TEAL, COPPER = "#1f5f8b", "#2a9d8f", "#c8693a"
SLATE, MUTED, GRID = "#475569", "#64748b", "rgba(100,116,139,0.16)"
UP, DOWN = "#1a9e6e", "#d64545"

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1180px}}
[data-testid=stSidebar] {{background: linear-gradient(180deg, #e6eef7 0%, #f4f7fb 100%);
                         border-right: 1px solid #dbe4ef}}
.hero h1 {{font-size: 2.3rem; line-height: 1.15; font-weight: 800; margin: 0 0 .35rem 0; padding: 0;
           background: linear-gradient(90deg, {NAVY} 0%, {TEAL} 65%, {COPPER} 100%);
           -webkit-background-clip: text; background-clip: text; color: transparent}}
.hero p {{color: {MUTED}; margin: 0 0 1.1rem 0; font-size: .95rem}}
.kpis {{display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: .75rem;
        margin: .2rem 0 1.2rem 0}}
.kpi {{background: #fff; border: 1px solid #dfe7f1; border-radius: 14px; padding: .8rem 1rem;
       box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 18px rgba(31,95,139,.06)}}
.kpi.main {{border-top: 4px solid {NAVY}}}
.kpi .l {{font-size: .74rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .05em}}
.kpi .v {{font-size: 1.55rem; font-weight: 750; color: #12202f; margin: .15rem 0}}
.kpi .s {{font-size: .76rem; color: {MUTED}}}
.badge {{display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: .76rem;
         font-weight: 650; margin-right: .35rem}}
.badge.up {{background: #e3f5ee; color: {UP}}}
.badge.down {{background: #fbe7e7; color: {DOWN}}}
.badge.flat {{background: #eef2f7; color: {MUTED}}}
.sec {{margin: 1.2rem 0 .5rem 0; padding-bottom: .35rem; border-bottom: 2px solid #dfe7f1}}
.sec h3 {{margin: 0; padding: 0; font-size: 1.12rem; color: #12202f}}
.sec p {{margin: .25rem 0 0 0; color: {MUTED}; font-size: .85rem}}
.side-brand {{font-weight: 800; font-size: 1.02rem; color: {NAVY}; margin-bottom: .1rem}}
.side-sub {{color: {MUTED}; font-size: .8rem; margin-bottom: .7rem}}
.side-card {{background: #fff; border: 1px solid #dbe4ef; border-radius: 10px; padding: .45rem .7rem;
             margin: .3rem 0; font-size: .82rem; color: #1c2836}}
.side-card b {{display: block; color: {MUTED}; font-weight: 600; font-size: .7rem; text-transform: uppercase;
               letter-spacing: .04em}}
.note {{background: #fff; border: 1px solid #dfe7f1; border-left: 4px solid {TEAL}; border-radius: 10px;
        padding: .7rem .9rem; color: #1c2836; font-size: .88rem; margin: .4rem 0 1rem 0}}
</style>
"""


def tr(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, nd=1):
    return f"{x:+.{nd}f}%".replace(".", ",")


@st.cache_data
def load(stamp: float) -> dict:
    d = {}
    for k, f in {"fc": "forecast_latest.csv", "metrics": "metrics_by_horizon.csv", "news": "news_sentiment.csv",
                 "effects": "input_effects.csv", "newstest": "news_value_test.csv",
                 "headlines": "news_headlines_recent.csv", "headcounts": "news_headlines_monthly_counts.csv",
                 "filter": "news_filter_summary.csv"}.items():
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


def page_setup():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(subtitle: str):
    st.markdown(f'<div class="hero"><h1>{APP_NAME}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    cap = f"<p>{caption}</p>" if caption else ""
    st.markdown(f'<div class="sec"><h3>{title}</h3>{cap}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi_cards(cards: list[dict]):
    """cards: dicts with label, value, optional delta (float, %), sub, main (bool)."""
    parts = []
    for c in cards:
        badge = ""
        if c.get("delta") is not None:
            d = c["delta"]
            cls = "up" if d > 0.05 else "down" if d < -0.05 else "flat"
            arrow = "▲" if d > 0.05 else "▼" if d < -0.05 else "■"
            badge = f'<span class="badge {cls}">{arrow} {pct(d)}</span>'
        main = " main" if c.get("main") else ""
        parts.append(f'<div class="kpi{main}"><div class="l">{html.escape(c["label"])}</div>'
                     f'<div class="v"{" style=font-size:1.05rem" if len(c["value"]) > 12 else ""}>'
                     f'{html.escape(c["value"])}</div>'
                     f'<div class="s">{badge}{html.escape(c.get("sub", ""))}</div></div>')
    st.markdown(f'<div class="kpis">{"".join(parts)}</div>', unsafe_allow_html=True)


def style(fig, h):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=8, r=8, t=16, b=8),
                      hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1c2836"))
    fig.update_xaxes(showgrid=False, linecolor="#dfe7f1")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
