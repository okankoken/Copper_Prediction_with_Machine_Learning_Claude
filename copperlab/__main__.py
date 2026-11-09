"""CLI:  python -m copperlab snapshot | run [--allow-stale] [--run-date YYYY-MM-DD] | all"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from . import config as C


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="copperlab", description="LME copper H1-H12 probabilistic forecaster")
    p.add_argument("command", choices=["snapshot", "news", "run", "all"])
    p.add_argument("--allow-stale", action="store_true", help="skip the forecast-origin guardrail")
    p.add_argument("--run-date", type=date.fromisoformat, default=None)
    p.add_argument("--jobs", type=int, default=C.N_JOBS)
    a = p.parse_args(argv)

    if a.command in ("snapshot", "all"):
        from .snapshot import take_snapshot
        m = take_snapshot()
        print(f"snapshot: {len(m['files'])} files copied from {m['source_project']}")
    if a.command in ("news", "all"):
        import pandas as pd
        from . import news
        last = pd.Period(a.run_date or date.today(), "M") - 1
        news.fetch(last)
        news.score_headlines()
        print(f"news: sentiment through {news.monthly_sentiment(last).index.max()}")
    if a.command in ("run", "all"):
        from .pipeline import run
        m = run(a.run_date, allow_stale=a.allow_stale, n_jobs=a.jobs)
        print(f"forecast origin {m['forecast_origin']}  price {m['origin_price_usd_t']:.1f} USD/t  "
              f"({m['n_expert_predictions']} expert predictions, {m['runtime_seconds']} s)")
        print(f"outputs -> {C.OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
