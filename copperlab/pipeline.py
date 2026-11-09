"""End-to-end run at dekad resolution: data + news -> yearly tuning -> walk-forward -> forecast + input effects."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import dekad as K
from .data import build_panel
from .experts import run_origin, tune, tune_point
from .features import broad_features, describe, own_features, sentiment_features, targets
from .metrics import evaluate, pairwise
from .news import dekad_sentiment
from .snapshot import sha256


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def effects_table(coef_hist: pd.DataFrame, latest: dict, p0: float) -> pd.DataFrame:
    """Per input and horizon: today's contribution to the forecast (USD/t, ML average) and how
    consistently the ElasticNet has assigned it the same sign across all backtest origins."""
    rows = []
    for h in C.DRIVER_HORIZONS:
        hist = coef_hist[coef_hist["horizon"] == h]
        piv = hist.pivot_table(index="origin", columns="feature", values="coef")
        contrib = latest.get(h, {}).get("contrib", pd.Series(dtype=float))
        last_coef = latest.get(h, {}).get("coef", pd.Series(dtype=float))
        for feat in sorted(set(piv.columns) | set(contrib.index)):
            key, label, fam = describe(feat)
            col = piv[feat] if feat in piv else pd.Series(dtype=float)
            used = col.dropna()
            nz = used[used.abs() > 1e-10]
            rows.append({
                "horizon_dekads": h, "months": h // 3, "feature": feat, "series": key, "label": label,
                "family": fam, "family_tr": C.FAMILY_TR.get(fam, fam),
                "contribution_log": float(contrib.get(feat, 0.0)),
                "contribution_usd": float(p0 * (np.exp(contrib.get(feat, 0.0)) - 1)),
                "coef_latest": float(last_coef.get(feat, np.nan)),
                "origins_seen": int(len(used)),
                "selected_pct": float(len(nz) / len(used) * 100) if len(used) else np.nan,
                "positive_pct": float((nz > 0).mean() * 100) if len(nz) else np.nan,
                "mean_coef": float(used.mean()) if len(used) else np.nan,
            })
    t = pd.DataFrame(rows)
    sel = t["selected_pct"].fillna(0) / 100
    agree = (t["positive_pct"].fillna(50) - 50).abs() / 50
    t["consistency"] = sel * agree
    t["direction"] = np.where(t["positive_pct"] >= 50, "+", "-")
    return t


def run(run_date: date | None = None, allow_stale: bool = False, n_jobs: int = C.N_JOBS, log=print) -> dict:
    t0 = time.time()
    run_date = run_date or date.today()
    cu, exo = build_panel(run_date, allow_stale=allow_stale)
    origin = int(cu.index[-1])
    news_d = dekad_sentiment(origin)
    F, B, Y = own_features(cu), broad_features(exo, cu.index), targets(cu)
    S = sentiment_features(news_d, cu.index)

    C.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([cu.drop(columns=["last_day"]), F, S, B], axis=1)
    panel.insert(0, "label", [K.label(int(n)) for n in panel.index])
    panel.to_csv(C.PANEL_DIR / "copper_dekad_panel.csv", index_label="dekad")

    first = K.make(*C.FIRST_ORIGIN)
    origins = [int(n) for n in F.index if first <= n <= origin]

    points = sorted({tune_point(t) for t in origins})
    jobs = [(T, h) for T in points for h in C.TUNE_HORIZONS]
    log(f"tuning: {len(points)} yearly points x {len(C.TUNE_HORIZONS)} horizons")
    tuned = Parallel(n_jobs=n_jobs)(delayed(tune)(T, h, F, B, S, Y) for T, h in jobs)
    params: dict = {}
    for (T, h), p in zip(jobs, tuned):
        params.setdefault(T, {})[h] = p

    log(f"walk-forward: {len(origins)} origins x {len(C.HORIZONS)} horizons")
    hist = [t for t in origins if t != origin]
    results = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(run_origin)(t, F, B, S, Y, params[tune_point(t)]) for t in hist)
    last_records, latest = run_origin(origin, F, B, S, Y, params[tune_point(origin)], details=True)
    preds = pd.DataFrame([r for rec, _ in results for r in rec] + last_records)
    coef_hist = pd.DataFrame([{"origin": t, "horizon": h, "feature": f, "coef": c}
                              for t, (_, inf) in zip(hist + [origin], results + [(None, latest)])
                              for h, row in inf.items() for f, c in row["coef"].items()])

    metrics = evaluate(preds, cu["price"])
    news_test = pairwise(preds, "enet_news", "enet")

    p0 = float(cu.loc[origin, "price"])
    cur = preds[preds["origin"] == origin].pivot(index="horizon", columns="model", values="yhat")
    M = metrics.set_index(["model", "horizon"])
    hz = [int(h) for h in cur.index]
    out = pd.DataFrame({
        "forecast_origin": K.label(origin), "origin_label_tr": K.label_tr(origin),
        "horizon_dekads": hz,
        "period": [K.label(origin + h) for h in hz], "period_tr": [K.label_tr(origin + h) for h in hz],
        "period_start": [K.start(origin + h).date().isoformat() for h in hz],
        "period_end": [K.end(origin + h).date().isoformat() for h in hz],
        "origin_price_usd_t": p0,
        "forecast_usd_t": p0 * np.exp(cur.loc[hz, C.FORECAST].values),
    })
    out["change_pct"] = (out["forecast_usd_t"] / p0 - 1) * 100
    for m in C.ML_EXPERTS:
        if m in cur.columns:
            out[f"{m}_usd_t"] = p0 * np.exp(cur.loc[hz, m].values)
    for col, src in (("backtest_n", "n"), ("backtest_mape_pct", "mape_pct"), ("direction_hit_pct", "directional_acc_pct"),
                     ("skill_vs_rw_pct", "skill_vs_rw_pct"), ("skill_vs_forward_pct", "skill_vs_anchor_pct"),
                     ("dm_p_vs_rw", "dm_p_vs_rw")):
        out[col] = [M.loc[(C.FORECAST, h), src] if (C.FORECAST, h) in M.index else np.nan for h in hz]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["generated_at_utc"] = generated

    effects = effects_table(coef_hist, latest, p0)
    tuning = pd.DataFrame([{"tune_point": K.label(T), "horizon_dekads": h,
                            **{f"enet_{k}": v for k, v in p["enet"].items()},
                            **{f"lgbm_{k}": v for k, v in p["lgbm"].items()}}
                           for T, d in params.items() for h, p in d.items()])

    C.OUT.mkdir(parents=True, exist_ok=True)
    C.ARCHIVE.mkdir(parents=True, exist_ok=True)
    keep = {"archive", "REPORT.md"}
    for f in C.OUT.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()
    out.to_csv(C.OUT / "forecast_latest.csv", index=False)
    out.to_csv(C.ARCHIVE / f"forecast_{K.label(origin)}_{datetime.now():%Y%m%dT%H%M%S}.csv", index=False)
    preds.assign(origin_label=preds["origin"].map(K.label)).to_csv(C.OUT / "backtest_predictions.csv", index=False)
    metrics.to_csv(C.OUT / "metrics_by_horizon.csv", index=False)
    news_test.to_csv(C.OUT / "news_value_test.csv", index=False)
    effects.to_csv(C.OUT / "input_effects.csv", index=False)
    tuning.to_csv(C.OUT / "tuning.csv", index=False)
    nd = news_d.copy()
    nd.insert(0, "label", [K.label(int(n)) for n in nd.index])
    nd["date"] = [K.end(int(n)).date().isoformat() for n in nd.index]
    nd.to_csv(C.OUT / "news_sentiment.csv", index_label="dekad")

    snap = C.ROOT / "data" / "snapshot_manifest.json"
    manifest = {
        "generated_at_utc": generated, "run_date": run_date.isoformat(),
        "forecast_origin": K.label(origin), "forecast_origin_tr": K.label_tr(origin), "origin_price_usd_t": p0,
        "panel_dekads": [K.label(int(cu.index[0])), K.label(origin)],
        "backtest_origins": [K.label(origins[0]), K.label(origin)], "n_backtest_origins": len(origins),
        "news_dekads": [K.label(int(news_d.index.min())), K.label(int(news_d.index.max()))] if len(news_d) else None,
        "news_headlines_scored": int(news_d["finbert_n"].sum()) if "finbert_n" in news_d else 0,
        "n_predictions": int(len(preds)), "code_commit": _git_commit(),
        "config": {k: getattr(C, k) for k in ("HORIZONS", "FIRST_ORIGIN", "MIN_TRAIN", "MIN_TRAIN_NEWS", "COVERAGE",
                                               "TUNE_HORIZONS", "LGBM_TUNE_ITER", "FORWARD_TENOR", "FFILL_LIMIT",
                                               "ML_EXPERTS", "FORECAST", "DRIVER_HORIZONS")},
        "series": [asdict(s) for s in C.SERIES],
        "snapshot": json.loads(snap.read_text()) if snap.exists() else None,
        "panel_sha256": sha256(C.PANEL_DIR / "copper_dekad_panel.csv"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (C.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    from .report import write_report
    write_report(out, metrics, news_test, effects, manifest)
    return manifest
