"""Equities that carry information about copper demand or supply, from Yahoo Finance.

Daily adjusted closes since 2008 are cached in data/raw/equities_extra/stocks_daily.csv;
the model uses their monthly averages like any other market series (known at month end).
"""
from __future__ import annotations

import time

import pandas as pd

from . import config as C

FILE = C.STOCKS_FILE

STOCKS = C.STOCKS


def fetch(start: str = "2008-01-01", log=print) -> pd.DataFrame:
    """Download all tickers (sequentially, with retries) and write the daily cache."""
    import yfinance as yf

    cols = {}
    for name, (ticker, _, _) in STOCKS.items():
        for attempt in range(3):
            try:
                s = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)["Close"]
                s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
                if len(s.dropna()):
                    cols[name] = s.dropna()
                    break
            except Exception as e:  # network / cache hiccups: retry
                log(f"  {ticker}: {type(e).__name__}, retry {attempt + 1}")
            time.sleep(2)
        else:
            log(f"  {ticker}: no data")
    d = pd.DataFrame(cols)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d = d.sort_index()
    out = C.DATA_RAW / FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    d.rename_axis("date").to_csv(out, lineterminator="\n")
    log(f"stocks: {len(cols)}/{len(STOCKS)} tickers, {d.index.min().date()} -> {d.index.max().date()}")
    return d
