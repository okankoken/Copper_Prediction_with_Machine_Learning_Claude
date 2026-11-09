"""End-to-end run: panel -> walk-forward experts -> combination -> metrics -> forecast."""
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
from .experts import pc_labels, run_origin
from .features import OWN_LABELS_TR, broad_features, own_features, targets
from .metrics import evaluate, interval_quality, verdict
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


def drivers_table(info: dict, price0: float) -> pd.DataFrame:
    rows = []
    for h, d in info.items():
        labels = pc_labels(d["loadings"])
        contrib = d["factor_coef"] * d["x_scaled"]
        for feat, v in contrib.items():
            rows.append({
                "horizon": h, "feature": feat,
                "label": OWN_LABELS_TR.get(feat, f"Makro faktör {feat.upper()}: {labels.get(feat, '')}"),
                "group": "copper_own" if feat in OWN_LABELS_TR else "macro_factor",
                "value": float(d["x_raw"][feat]), "z": float(d["x_scaled"][feat]),
                "coef_log": float(d["factor_coef"][feat]), "contribution_log": float(v),
                "contribution_usd_approx": float(price0 * v),
                "gbm_gain_share": float(d["gbm_importance"][feat] / max(d["gbm_importance"].sum(), 1e-12)),
            })
    return pd.DataFrame(rows)


def run(run_date: date | None = None, allow_stale: bool = False, n_jobs: int = C.N_JOBS) -> dict:
    t0 = time.time()
    run_date = run_date or date.today()
    cu, exo = build_panel(run_date, allow_stale=allow_stale)
    F, B, Y = own_features(cu), broad_features(exo), targets(cu)

    C.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([cu.drop(columns=["last_day"]), F, B], axis=1)
    panel.index = panel.index.astype(str)
    panel.to_csv(C.PANEL_DIR / "copper_monthly_panel.csv", index_label="month")

    origin = cu.index[-1]
    origins = F.index[F.index >= pd.Period(C.FIRST_ORIGIN, "M")]
    hist_origins = [t for t in origins if t != origin]
    chunks = Parallel(n_jobs=n_jobs)(delayed(run_origin)(t, F, B, Y) for t in hist_origins)
    last_records, info = run_origin(origin, F, B, Y, details=True)
    preds = pd.DataFrame([r for ch in chunks for r in ch] + last_records)

    combo, weights = combine(preds, Y)
    metrics = evaluate(preds, combo, cu["price"])
    intervals = interval_quality(combo)

    # ---- production forecast = the walk-forward step at the latest origin
    p0 = float(cu.loc[origin, "price"])
    fc = combo[combo["origin"] == origin].sort_values("horizon").copy()
    wl = weights[weights["origin"] == origin].set_index("horizon")
    ex = preds[preds["origin"] == origin].pivot(index="horizon", columns="model", values="yhat")
    cm = metrics[metrics["model"] == "combo"].set_index("horizon")
    out = pd.DataFrame({
        "forecast_origin": str(origin),
        "horizon": fc["horizon"].values,
        "forecast_month": fc["target"].astype(str).values,
        "forecast_date": [t.end_time.date().isoformat() for t in fc["target"]],
        "origin_price_usd_t": p0,
        "forecast_usd_t": p0 * np.exp(fc["yhat"].values),
    })
    for q in C.QUANTILES:
        out[f"p{int(q * 100):02d}_usd_t"] = p0 * np.exp(fc[f"q{int(q * 100):02d}"].values)
    out["change_from_origin_pct"] = (out["forecast_usd_t"] / p0 - 1) * 100
    out["prob_above_origin"] = fc["p_up"].values
    out["interval_method"] = fc["interval_method"].values
    for m in C.MODELS:
        out[f"expert_{m}_usd_t"] = p0 * np.exp(ex.loc[out["horizon"], m].values)
        out[f"weight_{m}"] = wl.loc[out["horizon"], f"w_{m}"].values
    out["weight_obs"] = wl.loc[out["horizon"], "n_weight_obs"].values
    out["backtest_n"] = cm.loc[out["horizon"], "n"].values
    out["backtest_mape_pct"] = cm.loc[out["horizon"], "mape_pct"].values
    out["skill_vs_rw_pct"] = cm.loc[out["horizon"], "skill_vs_rw_pct"].values
    out["dm_p_value"] = cm.loc[out["horizon"], "dm_p_value"].values
    v = [verdict(s, p) for s, p in zip(out["skill_vs_rw_pct"], out["dm_p_value"])]
    out["verdict"] = [a for a, _ in v]
    out["verdict_tr"] = [b for _, b in v]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["generated_at_utc"] = generated

    drivers = drivers_table(info, p0)

    # ---- write
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.ARCHIVE.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.OUT / "forecast_latest.csv", index=False)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out.to_csv(C.ARCHIVE / f"forecast_{origin}_{stamp}.csv", index=False)
    _to_str(preds).to_csv(C.OUT / "backtest_expert_predictions.csv", index=False)
    _to_str(combo).to_csv(C.OUT / "backtest_combination.csv", index=False)
    _to_str(weights).to_csv(C.OUT / "expert_weights_history.csv", index=False)
    metrics.to_csv(C.OUT / "metrics_by_horizon.csv", index=False)
    intervals.to_csv(C.OUT / "interval_calibration.csv", index=False)
    drivers.to_csv(C.OUT / "drivers_latest.csv", index=False)

    snap = C.ROOT / "data" / "snapshot_manifest.json"
    manifest = {
        "generated_at_utc": generated,
        "run_date": run_date.isoformat(),
        "forecast_origin": str(origin),
        "origin_price_usd_t": p0,
        "panel_months": [str(cu.index[0]), str(origin)],
        "backtest_origins": [str(origins[0]), str(origin)],
        "n_expert_predictions": int(len(preds)),
        "code_commit": _git_commit(),
        "config": {k: getattr(C, k) for k in ("HORIZONS", "FIRST_ORIGIN", "MIN_TRAIN", "N_FACTORS",
                                               "FACTOR_MIN_COVERAGE", "COMBO_LOOKBACK", "COMBO_MIN_EFFECTIVE",
                                               "CONFORMAL_MIN", "QUANTILES", "FFILL_LIMIT", "MODELS",
                                               "BENCHMARK", "ANCHOR")},
        "series": [asdict(s) for s in C.SERIES],
        "snapshot": json.loads(snap.read_text()) if snap.exists() else None,
        "panel_sha256": sha256(C.PANEL_DIR / "copper_monthly_panel.csv"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (C.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    from .report import write_report
    write_report(out, metrics, intervals, drivers, manifest)
    return manifest
