#!/usr/bin/env bash
# Wait for the news download, then: FinBERT scoring -> retrain -> tests -> dashboard -> commit outputs.
cd "$(dirname "$0")"
PY=/home/train/miniconda3/bin/python
while pgrep -f "^$PY -u -m copperlab.news" >/dev/null; do sleep 60; done
echo "[$(date)] download finished"; tail -3 logs/news_fetch.log
$PY -u -m copperlab news && echo "[$(date)] news + FinBERT ok" || { echo "[$(date)] NEWS STEP FAILED"; }
./run.sh run && echo "[$(date)] run ok" || { echo "[$(date)] RUN FAILED"; exit 1; }
./run.sh test 2>&1 | tail -3
./restart_dashboard.sh
git add -A -- outputs data/panel data/snapshot_manifest.json data/news/finbert_scores.csv 2>/dev/null
git commit -q -m "Forecast run with news sentiment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date)] committed"
git push -q origin main && echo "[$(date)] pushed to GitHub" || echo "[$(date)] PUSH FAILED"
echo "[$(date)] OVERNIGHT DONE"
