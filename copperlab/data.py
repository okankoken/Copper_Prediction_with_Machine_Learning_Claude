"""Build the monthly panel from the raw snapshot.

Index: pandas PeriodIndex (freq="M"). Only completed months are kept.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import config as C


class OriginGuardError(RuntimeError):
    pass


def _month_index(values) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(pd.Series(values).astype(str), format="mixed"), freq="M")


def _read_daily(rel: str) -> pd.DataFrame:
    d = pd.read_csv(C.DATA_RAW / rel, parse_dates=["date"]).sort_values("date")
    d["month"] = d["date"].dt.to_period("M")
    return d


def _read_monthly(rel: str) -> pd.DataFrame:
    d = pd.read_csv(C.DATA_RAW / rel)
    date_col = d.columns[0]
    d.index = _month_index(d[date_col])
    d = d.drop(columns=[date_col]).select_dtypes("number")
    return d.groupby(level=0).mean()


def copper_monthly(run_date: date | None = None) -> pd.DataFrame:
    """Monthly LME copper: average cash price (the target), month-end close,
    average 3-month price and average LME stock. Drops the running month."""
    run_date = run_date or date.today()
    d = _read_daily(C.COPPER_FILE).dropna(subset=["cash_settlement_usd_per_ton"])
    g = d.groupby("month")
    m = pd.DataFrame({
        "price": g["cash_settlement_usd_per_ton"].mean(),
        "close": g["cash_settlement_usd_per_ton"].last(),
        "f3m": g["copper_3_month_usd_per_ton"].mean(),
        "stock": g["copper_stock_ton"].mean(),
        "trading_days": g.size(),
        "last_day": g["date"].max(),
    })
    running = pd.Period(run_date, "M")
    m = m[m.index < running]
    # A month is complete only if data reach its last few calendar days.
    last = m.index[-1]
    if m.loc[last, "last_day"] < last.end_time.normalize() - pd.Timedelta(days=5):
        m = m.iloc[:-1]
    return m


def check_origin(cu: pd.DataFrame, run_date: date | None = None) -> pd.Period:
    """Forecast origin must be the month just before the run month."""
    run_date = run_date or date.today()
    expected = pd.Period(run_date, "M") - 1
    origin = cu.index[-1]
    if origin != expected:
        raise OriginGuardError(
            f"Copper data end at {origin}, expected the completed month {expected}. "
            "Refresh LME data before producing a forecast (backtest-only runs may pass --allow-stale)."
        )
    return origin


def exogenous_raw(index: pd.PeriodIndex) -> pd.DataFrame:
    """Raw monthly values of every registry series, aligned to `index` (no lag yet)."""
    cache: dict[str, pd.DataFrame] = {}
    cols = {}
    for s in C.SERIES:
        if s.file not in cache:
            if s.daily:
                d = _read_daily(s.file)
                cache[s.file] = d.groupby("month").mean(numeric_only=True)
            else:
                cache[s.file] = _read_monthly(s.file)
        cols[s.name] = cache[s.file][s.column]
    raw = pd.DataFrame(cols)
    full = pd.period_range(min(raw.index.min(), index.min()), index.max(), freq="M")
    return raw.reindex(full)


def apply_publication_lags(raw: pd.DataFrame, index: pd.PeriodIndex) -> pd.DataFrame:
    """Value visible at month t = value of month t-lag, forward-filled at most FFILL_LIMIT months.

    Values dated after the origin are never used: shifting is strictly backward
    and forward filling only carries past observations.
    """
    known = {}
    for s in C.SERIES:
        x = raw[s.name].shift(s.lag).ffill(limit=C.FFILL_LIMIT)
        known[s.name] = x
    return pd.DataFrame(known).reindex(index)


def build_panel(run_date: date | None = None, allow_stale: bool = False):
    cu = copper_monthly(run_date)
    if not allow_stale:
        check_origin(cu, run_date)
    raw = exogenous_raw(cu.index)
    exo = apply_publication_lags(raw, cu.index)
    exo = exo.where(~np.isinf(exo))
    return cu, exo
