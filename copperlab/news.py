"""Copper news sentiment from GDELT (2017 onwards).

Two independent signals per month:
  * GDELT tone      : GDELT's own document tone for copper-market articles, and the
                      share of global coverage those articles take (attention).
  * FinBERT headline: up to 250 most relevant English headlines per month, scored
                      with ProsusAI/finbert (positive - negative probability).

Everything fetched is cached under data/news/, so monthly updates only fetch the
new month. News is known by the end of its month, so it has no publication lag.
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from . import config as C

NEWS = C.ROOT / "data" / "news"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
# A single phrase: GDELT rejects (HTTP 429) heavier OR-queries far more often.
QUERY = '"copper price" sourcelang:english'
START = pd.Period("2017-01", "M")
HEADERS = {"User-Agent": "copperlab-research/0.2"}
PAUSE = 45.0                  # GDELT throttles hard; one request per ~45 s is reliable


def _get(params: dict, log=print) -> dict:
    for attempt in range(10):
        time.sleep(PAUSE)
        try:
            r = requests.get(API, params={**params, "query": QUERY, "format": "json"}, headers=HEADERS, timeout=120)
        except requests.RequestException as e:
            log(f"  {params.get('mode')} {params.get('startdatetime')}: {type(e).__name__}, retry {attempt + 1}")
            time.sleep(90)
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            return r.json() if r.text.strip() else {}
        log(f"  {params.get('mode')} {params.get('startdatetime')}: HTTP {r.status_code}, retry {attempt + 1}")
        time.sleep(90)   # rate limited (HTTP 429) or transient error
    raise RuntimeError(f"GDELT request failed: {params}")


def _span(p: pd.Period) -> dict:
    return {"startdatetime": p.start_time.strftime("%Y%m%d%H%M%S"),
            "enddatetime": (p + 1).start_time.strftime("%Y%m%d%H%M%S")}


def _quarters(last: pd.Period) -> list[pd.Period]:
    return list(pd.period_range(START.asfreq("Q"), last.asfreq("Q"), freq="Q"))


def fetch(last: pd.Period, log=print) -> None:
    """Fetch what is not cached yet: daily tone/volume one calendar year per request,
    the 250 most relevant headlines one quarter per request. The newest year/quarter is
    always refreshed."""
    for kind in ("tone", "volume", "headlines"):
        (NEWS / kind).mkdir(parents=True, exist_ok=True)
    for year in range(START.year, last.year + 1):
        end = min(pd.Period(f"{year}-12", "M"), last)
        span = {"startdatetime": f"{year}0101000000",
                "enddatetime": (end + 1).start_time.strftime("%Y%m%d%H%M%S")}
        for kind, mode in (("tone", "timelinetone"), ("volume", "timelinevolraw")):
            path = NEWS / kind / f"{year}.json"
            if path.exists() and year != last.year:
                continue
            path.write_text(json.dumps(_get({"mode": mode, **span}, log)), encoding="utf-8")
        log(f"timeline {year} ok")
    for q in _quarters(last):
        path = NEWS / "headlines" / f"{q}.json"
        if path.exists() and q != last.asfreq("Q"):
            continue
        span = {"startdatetime": q.asfreq("M", "s").start_time.strftime("%Y%m%d%H%M%S"),
                "enddatetime": (min(q.asfreq("M", "e"), last) + 1).start_time.strftime("%Y%m%d%H%M%S")}
        path.write_text(json.dumps(_get({"mode": "artlist", "maxrecords": 250, "sort": "hybridrel", **span}, log)),
                        encoding="utf-8")
        log(f"headlines {q} ok")


def _timeline(kind: str) -> pd.DataFrame:
    rows = []
    for path in sorted((NEWS / kind).glob("*.json")) if (NEWS / kind).exists() else []:
        js = json.loads(path.read_text(encoding="utf-8") or "{}")
        for series in js.get("timeline", []):
            for d in series.get("data", []):
                rows.append({"series": series.get("series"), "date": d["date"], "value": d["value"],
                             "norm": d.get("norm")})
    t = pd.DataFrame(rows, columns=["series", "date", "value", "norm"])
    return t.drop_duplicates(["series", "date"], keep="last")


def _headlines() -> pd.DataFrame:
    rows = []
    for f in sorted((NEWS / "headlines").glob("*.json")) if (NEWS / "headlines").exists() else []:
        for a in json.loads(f.read_text(encoding="utf-8") or "{}").get("articles", []):
            seen = str(a.get("seendate") or "")
            if len(seen) >= 8 and a.get("url"):
                rows.append({"url": a["url"], "date": pd.Timestamp(seen[:8]),
                             "title": (a.get("title") or "").strip(), "domain": a.get("domain")})
    return pd.DataFrame(rows, columns=["url", "date", "title", "domain"]).drop_duplicates("url")


def score_headlines(log=print) -> pd.DataFrame:
    """FinBERT probabilities for every cached headline, cached by URL in data/news/finbert_scores.csv."""
    out_path = NEWS / "finbert_scores.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["url"])
    t = _headlines()
    todo = t[~t["url"].isin(set(done["url"])) & (t["title"].str.len() > 15)]
    if len(todo):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
        labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        probs = []
        with torch.no_grad():
            for i in range(0, len(todo), 64):
                enc = tok(todo["title"].iloc[i:i + 64].tolist(), padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
                probs.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
                if i % 3200 == 0:
                    log(f"finbert {i}/{len(todo)}")
        P = np.vstack(probs)
        todo = todo[["url", "title"]].copy()
        for j, lab in enumerate(labels):
            todo[f"p_{lab}"] = P[:, j]
        done = pd.concat([done, todo], ignore_index=True)
        done.to_csv(out_path, index=False)
    return done


def dekad_sentiment(last: int | None = None) -> pd.DataFrame:
    """One row per dekad: GDELT tone, attention (copper-price share of all coverage) and
    FinBERT net headline sentiment (positive - negative probability)."""
    from . import dekad as K
    out = pd.DataFrame(index=pd.Index([], name="dekad", dtype="int64"))
    tone, vol = _timeline("tone"), _timeline("volume")
    if not tone.empty:
        tone["dekad"] = K.from_dates(pd.to_datetime(tone["date"], format="%Y%m%dT%H%M%SZ"))
        out = out.join(tone.groupby("dekad")["value"].mean().rename("news_tone"), how="outer")
    if not vol.empty:
        arts = vol[vol["series"].str.contains("Article Count", na=False)].copy()
        arts["dekad"] = K.from_dates(pd.to_datetime(arts["date"], format="%Y%m%dT%H%M%SZ"))
        g = arts.groupby("dekad")
        out = out.join(pd.DataFrame({"news_articles": g["value"].sum(),
                                     "news_attention": g["value"].sum() / g["norm"].sum().clip(lower=1) * 1e4}),
                       how="outer")
    scores = NEWS / "finbert_scores.csv"
    if scores.exists():
        fb = pd.read_csv(scores)
        if {"p_positive", "p_negative"} <= set(fb.columns):
            fb = fb.drop(columns=[c for c in ("date", "month") if c in fb.columns]).merge(
                _headlines()[["url", "date"]], on="url", how="inner")
            fb["dekad"] = K.from_dates(fb["date"])
            fb["net"] = fb["p_positive"] - fb["p_negative"]
            g = fb.groupby("dekad")
            out = out.join(pd.DataFrame({"finbert_net": g["net"].mean(), "finbert_n": g.size()}), how="outer")
    if last is not None:
        out = out[out.index <= last]
    return out.sort_index()


if __name__ == "__main__":
    last = pd.Period(date.today(), "M") - 1
    fetch(last)
    score_headlines()
    print(dekad_sentiment().tail())
