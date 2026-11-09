"""End-to-end run: panel + news -> tuning -> walk-forward experts -> combination -> forecast."""
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
from .combine import combine
from .data import build_panel
from .experts import pc_labels, run_origin, tune_lgbm, tune_point
from .features import (OWN_LABELS_TR, SENT_LABELS_TR, broad_features, own_features, sentiment_features,
                       targets)
from .metrics import evaluate, pairwise, verdict
from .news import monthly_sentiment
from .snapshot import sha256

PERIOD_COLS = ("origin", "target", "train_first", "train_last")


def _to_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in PERIOD_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def add_ml_model(preds: pd.DataFrame) -> pd.DataFrame:
    """The ML model = equal-weight average of the ML experts available at each origin."""
    ml = preds[preds["model"].isin(C.ML_EXPERTS)]
    keys = ["target", "y", "n_train", "train_first", "train_last"]
    sc = ml.groupby(["origin", "horizon"], as_index=False).agg(
        **{k: (k, "first") for k in keys}, yhat=("yhat", "mean"))
    sc["model"] = C.SCENARIO
    return pd.concat([preds, sc[preds.columns]], ignore_index=True)


def label(feat: str, pcl: dict) -> str:
    if feat in OWN_LABELS_TR:
        return OWN_LABELS_TR[feat]
    if feat in SENT_LABELS_TR:
        return SENT_LABELS_TR[feat]
    return f"Makro faktör {feat.upper()} ({pcl.get(feat, '')})"


def importance_table(info: dict) -> pd.DataFrame:
    rows = []
    for h, d in info.items():
        pcl = pc_labels(d["loadings"])
        gain = d["lgbm_gain"] / max(d["lgbm_gain"].sum(), 1e-12)
        for feat, g in gain.items():
            rows.append({"horizon": h, "model": "lgbm", "feature": feat, "label": label(feat, pcl),
                         "group": "news" if feat in SENT_LABELS_TR else "copper" if feat in OWN_LABELS_TR else "macro",
                         "importance": float(g)})
        coef = d["enet_coef"]
        tot = max(coef.abs().sum(), 1e-12)
        for feat, c in coef.items():
            rows.append({"horizon": h, "model": "enet", "feature": feat, "label": label(feat, pcl),
                         "group": "copper" if feat in OWN_LABELS_TR else "macro", "importance": float(abs(c) / tot),
                         "sign": float(np.sign(c))})
    return pd.DataFrame(rows)


def run(run_date: date | None = None, allow_stale: bool = False, n_jobs: int = C.N_JOBS, log=print) -> dict:
    t0 = time.time()
    run_date = run_date or date.today()
    cu, exo = build_panel(run_date, allow_stale=allow_stale)
    origin = cu.index[-1]
    sent = monthly_sentiment(origin)
    F, B, Y = own_features(cu), broad_features(exo), targets(cu)
    S = sentiment_features(sent, cu.index)

    C.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([cu.drop(columns=["last_day"]), F, S, B], axis=1)
    panel.index = panel.index.astype(str)
    panel.to_csv(C.PANEL_DIR / "copper_monthly_panel.csv", index_label="month")

    origins = list(F.index[F.index >= pd.Period(C.FIRST_ORIGIN, "M")])

    # 1) yearly LightGBM tuning (purged time-series CV on data available at each December)
    points = sorted({tune_point(t) for t in origins})
    log(f"tuning LightGBM: {len(points)} yearly points x {len(C.HORIZONS)} horizons")
    tuned = Parallel(n_jobs=n_jobs)(delayed(tune_lgbm)(T, h, F, B, S, Y) for T in points for h in C.HORIZONS)
    params = {}
    for (T, h), p in zip([(T, h) for T in points for h in C.HORIZONS], tuned):
        params.setdefault(T, {})[h] = p

    # 2) walk-forward experts
    log(f"walk-forward: {len(origins)} origins")
    hist = [t for t in origins if t != origin]
    chunks = Parallel(n_jobs=n_jobs)(delayed(run_origin)(t, F, B, S, Y, params[tune_point(t)]) for t in hist)
    last_records, info = run_origin(origin, F, B, S, Y, params[tune_point(origin)], details=True)
    preds = add_ml_model(pd.DataFrame([r for ch in chunks for r in ch] + last_records))

    # 3) combination + evaluation
    combo, weights = combine(preds)
    allp = pd.concat([preds[["horizon", "origin", "target", "y", "yhat", "model"]],
                      combo[["horizon", "origin", "target", "y", "yhat"]].assign(model="main")], ignore_index=True)
    metrics = evaluate(allp, cu["price"])
    news_test = pairwise(allp, "enet_news", "enet")

    # 4) forecast table
    p0 = float(cu.loc[origin, "price"])
    cur = allp[allp["origin"] == origin].pivot(index="horizon", columns="model", values="yhat")
    wl = weights[weights["origin"] == origin].set_index("horizon")
    M = metrics.set_index(["model", "horizon"])
    hz = list(cur.index)
    out = pd.DataFrame({
        "forecast_origin": str(origin), "horizon": hz,
        "forecast_month": [str(origin + h) for h in hz],
        "forecast_date": [(origin + h).end_time.date().isoformat() for h in hz],
        "origin_price_usd_t": p0,
        "main_usd_t": p0 * np.exp(cur.loc[hz, "main"].values),
        "ml_usd_t": p0 * np.exp(cur.loc[hz, C.SCENARIO].values),
    })
    for m in C.MODELS:
        if m in cur.columns:
            out[f"expert_{m}_usd_t"] = p0 * np.exp(cur.loc[hz, m].values)
        out[f"weight_{m}"] = wl.loc[hz, f"w_{m}"].values if f"w_{m}" in wl.columns else 0.0
    for tag, m in (("main", "main"), ("ml", C.SCENARIO)):
        out[f"{tag}_backtest_n"] = [M.loc[(m, h), "n"] for h in hz]
        out[f"{tag}_mape_pct"] = [M.loc[(m, h), "mape_pct"] for h in hz]
        out[f"{tag}_direction_pct"] = [M.loc[(m, h), "directional_acc_pct"] for h in hz]
        out[f"{tag}_skill_vs_rw_pct"] = [M.loc[(m, h), "skill_vs_rw_pct"] for h in hz]
        out[f"{tag}_dm_p_vs_rw"] = [M.loc[(m, h), "dm_p_vs_rw"] for h in hz]
    out["main_verdict"] = [verdict(s, p) for s, p in zip(out["main_skill_vs_rw_pct"], out["main_dm_p_vs_rw"])]
    out["ml_verdict"] = [verdict(s, p) for s, p in zip(out["ml_skill_vs_rw_pct"], out["ml_dm_p_vs_rw"])]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["generated_at_utc"] = generated

    tuning = pd.DataFrame([{"tune_point": str(T), "horizon": h, **p} for T, d in params.items() for h, p in d.items()])
    importance = importance_table(info)

    # 5) write
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.ARCHIVE.mkdir(parents=True, exist_ok=True)
    for stale in ("backtest_combination.csv", "interval_calibration.csv", "drivers_latest.csv"):
        (C.OUT / stale).unlink(missing_ok=True)
    out.to_csv(C.OUT / "forecast_latest.csv", index=False)
    out.to_csv(C.ARCHIVE / f"forecast_{origin}_{datetime.now():%Y%m%dT%H%M%S}.csv", index=False)
    _to_str(preds).to_csv(C.OUT / "backtest_expert_predictions.csv", index=False)
    _to_str(combo).to_csv(C.OUT / "backtest_main.csv", index=False)
    _to_str(weights).to_csv(C.OUT / "expert_weights_history.csv", index=False)
    metrics.to_csv(C.OUT / "metrics_by_horizon.csv", index=False)
    news_test.to_csv(C.OUT / "news_value_test.csv", index=False)
    tuning.to_csv(C.OUT / "lgbm_tuning.csv", index=False)
    importance.to_csv(C.OUT / "feature_importance_latest.csv", index=False)
    s_out = sent.copy()
    s_out.index = s_out.index.astype(str)
    s_out.to_csv(C.OUT / "news_sentiment_monthly.csv", index_label="month")

    snap = C.ROOT / "data" / "snapshot_manifest.json"
    manifest = {
        "generated_at_utc": generated, "run_date": run_date.isoformat(),
        "forecast_origin": str(origin), "origin_price_usd_t": p0,
        "panel_months": [str(cu.index[0]), str(origin)],
        "backtest_origins": [str(origins[0]), str(origin)],
        "news_months": [str(sent.index.min()), str(sent.index.max())] if len(sent) else None,
        "n_expert_predictions": int(len(preds)),
        "code_commit": _git_commit(),
        "config": {k: getattr(C, k) for k in ("HORIZONS", "FIRST_ORIGIN", "MIN_TRAIN", "MIN_TRAIN_NEWS",
                                               "N_FACTORS", "FACTOR_MIN_COVERAGE", "COMBO_LOOKBACK",
                                               "COMBO_MIN_EFFECTIVE", "LGBM_TUNE_ITER", "FFILL_LIMIT",
                                               "MODELS", "ML_EXPERTS", "BENCHMARK", "ANCHOR")},
        "series": [asdict(s) for s in C.SERIES],
        "snapshot": json.loads(snap.read_text()) if snap.exists() else None,
        "panel_sha256": sha256(C.PANEL_DIR / "copper_monthly_panel.csv"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (C.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    from .report import write_report
    write_report(out, metrics, news_test, manifest)
    return manifest
