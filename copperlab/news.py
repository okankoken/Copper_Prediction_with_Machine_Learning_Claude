"""Copper news sentiment (2017 onwards).

Two independent signals:
  * GDELT tone      : GDELT's own document tone for "copper price" articles, and the
                      share of global coverage those articles take (attention).
  * Headlines       : English copper-market headlines from Google News RSS searches
                      (several queries per month), scored with ProsusAI/finbert
                      (positive - negative probability). GDELT's article-list API
                      rejects our requests (HTTP 429), so headlines come from Google News.

Everything fetched is cached under data/news/, so monthly updates only fetch the
new month. News is known by the end of its month, so it has no publication lag.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date

# FinBERT weights (~0.5 GB) live outside /home, which is small on the VM
if os.path.isdir("/var/tmp/train_hf"):
    os.environ.setdefault("HF_HOME", "/var/tmp/train_hf")

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
    for attempt in range(6):
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


def _save(path, params, log) -> bool:
    """Fetch one request into `path`; on repeated failure log it and leave it for the next round."""
    try:
        path.write_text(json.dumps(_get(params, log)), encoding="utf-8")
        return True
    except RuntimeError as e:
        log(f"  skipped for now: {e}")
        return False


def fetch(last: pd.Period, log=print) -> int:
    """Returns the number of requests still missing after this round."""
    """GDELT daily tone/volume, one calendar year per request; the newest year is always refreshed."""
    missing = 0
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
            missing += not _save(path, {"mode": mode, **span}, log)
        log(f"timeline {year} done")
    return missing


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


GNEWS = "https://news.google.com/rss/search"
# folder -> searches. "gnews": copper market in general; "gnews_supply": mining and the two
# largest producers (Chile, Peru), English-language coverage incl. local outlets' English editions.
GNEWS_SETS = {
    "gnews": ('"copper price"', '"copper prices"', '"LME copper"', '"copper market"', '"copper demand"',
              '"copper supply"'),
    "gnews_supply": ('copper site:mining.com', 'copper Chile', 'copper Peru', 'Codelco', 'Escondida copper',
                     '"Las Bambas" OR Antamina OR Cerrejon copper', 'copper mine strike OR outage'),
}


def fetch_google_news(last: pd.Period, log=print, folder: str = "gnews") -> int:
    """Google News headlines per month and query, cached in data/news/<folder>/<YYYY-MM>.json.
    The newest month is always refreshed. Returns the number of failed searches."""
    import feedparser
    queries = GNEWS_SETS[folder]
    out = NEWS / folder
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for p in pd.period_range(START, last, freq="M"):
        path = out / f"{p}.json"
        if path.exists() and p != last:
            continue
        a, b = p.start_time.strftime("%Y-%m-%d"), (p + 1).start_time.strftime("%Y-%m-%d")
        items = []
        for q in queries:
            try:
                r = requests.get(GNEWS, params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US",
                                                "ceid": "US:en"}, headers=HEADERS, timeout=30)
                r.raise_for_status()
            except requests.RequestException as e:
                failed += 1
                log(f"  gnews {p} {q}: {type(e).__name__}")
                time.sleep(10)
                continue
            for e in feedparser.parse(r.text).entries:
                title = e.get("title", "")
                src = e.get("source", {}).get("title", "") if isinstance(e.get("source"), dict) else ""
                if src and title.endswith(f" - {src}"):
                    title = title[: -len(src) - 3]
                items.append({"url": e.get("link"), "title": title.strip(), "domain": src,
                              "published": e.get("published", ""), "query": q})
            time.sleep(1.5)
        path.write_text(json.dumps({"articles": items}), encoding="utf-8")
        log(f"{folder} {p}: {len(items)} headlines")
    return failed


def _headlines() -> pd.DataFrame:
    rows = []
    for folder in GNEWS_SETS:
        for f in sorted((NEWS / folder).glob("*.json")) if (NEWS / folder).exists() else []:
            for a in json.loads(f.read_text(encoding="utf-8") or "{}").get("articles", []):
                d = pd.to_datetime(a.get("published"), errors="coerce", utc=True)
                if a.get("url") and pd.notna(d) and len(a.get("title", "")) > 15:
                    rows.append({"url": a["url"], "date": d.tz_convert(None).normalize(), "title": a["title"],
                                 "domain": a.get("domain"), "topic": "supply" if folder != "gnews" else "market"})
    for f in sorted((NEWS / "headlines").glob("*.json")) if (NEWS / "headlines").exists() else []:
        for a in json.loads(f.read_text(encoding="utf-8") or "{}").get("articles", []):
            seen = str(a.get("seendate") or "")
            if len(seen) >= 8 and a.get("url"):
                rows.append({"url": a["url"], "date": pd.Timestamp(seen[:8]),
                             "title": (a.get("title") or "").strip(), "domain": a.get("domain")})
    return pd.DataFrame(rows, columns=["url", "date", "title", "domain", "topic"]).drop_duplicates("url")


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
    fb = scored_headlines()          # relevant, de-duplicated, with copper price-impact scores
    if len(fb):
        fb["dekad"] = K.from_dates(pd.to_datetime(fb["date"]))
        g = fb.groupby("dekad")
        out = out.join(pd.DataFrame({"finbert_net": g["net"].mean(), "finbert_n": g.size(),
                                     "impact_net": g["impact"].mean(),
                                     "impact_n": g["impact"].apply(lambda x: int((x.abs() > 0.15).sum()))}),
                       how="outer")
    if last is not None:
        out = out[out.index <= last]
    return out.sort_index()


def scored_headlines(with_removed: bool = False):
    """Relevant, de-duplicated headlines with FinBERT tone and copper price impact
    (see copperlab.headlines). With with_removed=True also returns the removal counts."""
    from .headlines import clean, score_frame
    cols = ["date", "title", "domain", "topic", "url", "label", "net", "impact", "direction", "reason"]
    empty = pd.DataFrame(columns=cols)
    scores = NEWS / "finbert_scores.csv"
    fb = pd.read_csv(scores) if scores.exists() else pd.DataFrame()
    if not {"p_positive", "p_negative", "p_neutral"} <= set(fb.columns):
        return (empty, pd.Series(dtype=int)) if with_removed else empty
    fb = fb[["url", "p_positive", "p_negative", "p_neutral"]].merge(_headlines(), on="url", how="inner")
    fb["net"] = fb["p_positive"] - fb["p_negative"]
    fb["label"] = fb[["p_positive", "p_negative", "p_neutral"]].idxmax(axis=1).str.replace("p_", "", regex=False)
    kept, removed = clean(fb)
    out = score_frame(kept).sort_values("date", ascending=False)[cols]
    return (out, removed) if with_removed else out


if __name__ == "__main__":
    last = pd.Period(date.today(), "M") - 1
    for folder in GNEWS_SETS:
        fetch_google_news(last, folder=folder)
    score_headlines()
    print(dekad_sentiment().tail())
