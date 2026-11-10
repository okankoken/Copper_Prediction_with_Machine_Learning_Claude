#!/usr/bin/env bash
# News download in rounds (GDELT throttles hard) -> FinBERT -> retrain -> tests -> dashboard -> commit -> push.
cd "$(dirname "$0")"
PY=/home/train/miniconda3/bin/python
for round in 1 2 3 4 5 6 7 8; do
  echo "[$(date)] news round $round"
  $PY -u -m copperlab news >> logs/news_fetch.log 2>&1 && { echo "[$(date)] news complete"; break; }
  tail -1 logs/news_fetch.log
  sleep 600
done
./run.sh run && echo "[$(date)] run ok" || { echo "[$(date)] RUN FAILED"; exit 1; }
./run.sh test 2>&1 | tail -3
./restart_dashboard.sh
git add -A -- outputs data/panel data/snapshot_manifest.json data/news/finbert_scores.csv copperlab overnight.sh 2>/dev/null
git commit -q -m "Forecast run with news sentiment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date)] committed"
git push -q origin main && echo "[$(date)] pushed to GitHub" || echo "[$(date)] PUSH FAILED"
echo "[$(date)] OVERNIGHT DONE"
