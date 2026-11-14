"""Copper at dekad resolution + monthly exogenous data with publication lags."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from . import dekad as K


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


def copper_dekads(run_date: date | None = None) -> pd.DataFrame:
    """Per dekad: average cash price (the target), last close, average 3-month price and stock.
    Only dekads that ended before run_date and whose data reach their last days are kept."""
    run_date = run_date or date.today()
    d = _read_daily(C.COPPER_FILE).dropna(subset=["cash_settlement_usd_per_ton"])
    d["dekad"] = K.from_dates(d["date"])
    g = d.groupby("dekad")
    cu = pd.DataFrame({
        "price": g["cash_settlement_usd_per_ton"].mean(),
        "close": g["cash_settlement_usd_per_ton"].last(),
        "f3m": g["copper_3_month_usd_per_ton"].mean(),
        "stock": g["copper_stock_ton"].mean(),
        "trading_days": g.size(),
        "last_day": g["date"].max(),
    })
    cu = cu[cu.index <= K.last_complete(run_date)]
    # complete = at most one business day (a holiday) missing at the end of the dekad
    last = int(cu.index[-1])
    missing = np.busday_count((cu.loc[last, "last_day"] + pd.Timedelta(days=1)).date(),
                              (K.end(last) + pd.Timedelta(days=1)).date())
    if missing > 1:
        cu = cu.iloc[:-1]
    full = pd.RangeIndex(int(cu.index[0]), int(cu.index[-1]) + 1)
    cu = cu.reindex(full)
    cu[["price", "close", "f3m", "stock"]] = cu[["price", "close", "f3m", "stock"]].ffill()  # holiday dekads
    cu.index.name = "dekad"
    return cu


def check_origin(cu: pd.DataFrame, run_date: date | None = None) -> int:
    run_date = run_date or date.today()
    expected = K.last_complete(run_date)
    origin = int(cu.index[-1])
    if origin != expected:
        raise OriginGuardError(
            f"Copper data end at {K.label(origin)}, expected the completed dekad {K.label(expected)}. "
            "Refresh LME data before producing a forecast (backtest-only runs may pass --allow-stale).")
    return origin


def exogenous_raw(months: pd.PeriodIndex) -> pd.DataFrame:
    """Raw monthly values of every registry series (no lag yet)."""
    cache: dict[str, pd.DataFrame] = {}
    cols = {}
    for s in C.SERIES:
        if s.file not in cache:
            if s.daily:
                cache[s.file] = _read_daily(s.file).groupby("month").mean(numeric_only=True)
            else:
                cache[s.file] = _read_monthly(s.file)
        cols[s.name] = cache[s.file][s.column]
    raw = pd.DataFrame(cols)
    full = pd.period_range(min(raw.index.min(), months.min()), months.max(), freq="M")
    return raw.reindex(full)


def apply_publication_lags(raw: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Value visible after month t closes = value of month t-lag (forward-filled at most FFILL_LIMIT)."""
    known = {s.name: raw[s.name].shift(s.lag).ffill(limit=C.FFILL_LIMIT) for s in C.SERIES}
    return pd.DataFrame(known).reindex(months)


def build_panel(run_date: date | None = None, allow_stale: bool = False):
    cu = copper_dekads(run_date)
    if not allow_stale:
        check_origin(cu, run_date)
    months = pd.period_range(K.ref_month(int(cu.index[0])), K.ref_month(int(cu.index[-1])), freq="M")
    raw = exogenous_raw(months)
    exo = apply_publication_lags(raw, months)
    exo = exo.where(~np.isinf(exo))
    return cu, exo
