"""Dekads: a month split into three periods (days 1-10, 11-20, 21-end).

A dekad is an integer id = year*36 + (month-1)*3 + (k-1), so "h dekads later" is
plain integer addition and "realised by origin t" is `s + h <= t`.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def from_dates(d: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(d)
    k = np.minimum((d.day - 1) // 10, 2)
    return (d.year * 36 + (d.month - 1) * 3 + k).to_numpy()


def make(year: int, month: int, k: int) -> int:
    return year * 36 + (month - 1) * 3 + (k - 1)


def parts(n: int) -> tuple[int, int, int]:
    return n // 36, (n % 36) // 3 + 1, n % 3 + 1


def start(n: int) -> pd.Timestamp:
    y, m, k = parts(n)
    return pd.Timestamp(y, m, 1 + 10 * (k - 1))


def end(n: int) -> pd.Timestamp:
    y, m, k = parts(n)
    return pd.Timestamp(y, m, 1).to_period("M").end_time.normalize() if k == 3 else pd.Timestamp(y, m, 10 * k)


def label(n: int) -> str:
    y, m, k = parts(n)
    return f"{y}-{m:02d}-{k}"


def label_tr(n: int) -> str:
    aylar = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
    y, m, k = parts(n)
    span = {1: "1-10", 2: "11-20", 3: f"21-{end(n).day}"}[k]
    return f"{span} {aylar[m - 1]} {y}"


def ref_month(n: int) -> pd.Period:
    """Last calendar month completed by the end of dekad n."""
    y, m, k = parts(n)
    p = pd.Period(year=y, month=m, freq="M")
    return p if k == 3 else p - 1


def last_complete(run_date: date) -> int:
    """Last dekad that ended before run_date."""
    n = int(from_dates(pd.DatetimeIndex([pd.Timestamp(run_date)]))[0])
    return n - 1
