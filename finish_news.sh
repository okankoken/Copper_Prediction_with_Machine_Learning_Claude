#!/usr/bin/env bash
# After the market-headline download: supply headlines -> FinBERT -> retrain -> tests -> dashboard -> commit -> push.
cd "$(dirname "$0")"
PY=/home/train/miniconda3/bin/python
export PYTHONPATH=$PWD
until grep -qE "GNEWS\+FINBERT DONE|Traceback" logs/gnews.log; do sleep 20; done
echo "[$(date)] market headlines done: $(tail -2 logs/gnews.log | head -1)"
$PY -u - <<'PY'
import pandas as pd
from copperlab import news
print("supply failed searches:", news.fetch_google_news(pd.Period("2026-11", "M"), folder="gnews_supply"), flush=True)
news.score_headlines()
d = news.dekad_sentiment()
print("headlines scored:", int(d["finbert_n"].sum()), "supply:", int(d["finbert_supply_n"].sum()), flush=True)
PY
echo "[$(date)] supply headlines + FinBERT done"
./run.sh run && echo "[$(date)] run ok" || { echo "[$(date)] RUN FAILED"; exit 1; }
./run.sh test 2>&1 | tail -3
./restart_dashboard.sh
git add -A -- outputs data/panel data/snapshot_manifest.json data/news copperlab app overnight.sh finish_news.sh .gitignore
git commit -q -m "News sentiment: GDELT tone + Google News headlines (market, mining/Chile/Peru) scored with FinBERT

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date)] committed"
git push -q origin main && echo "[$(date)] pushed" || echo "[$(date)] PUSH FAILED"
echo "[$(date)] FINISH DONE"
