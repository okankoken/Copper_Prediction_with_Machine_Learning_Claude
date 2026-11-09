#!/usr/bin/env bash
# Restart the Streamlit dashboard in the background (logs/dashboard.log).
cd "$(dirname "$0")"
pids=$(pgrep -f "^/home/train/miniconda3/bin/python -m streamlit run app/dashboard.py")
[ -n "$pids" ] && kill $pids && sleep 1
mkdir -p logs
nohup ./run.sh dashboard > logs/dashboard.log 2>&1 &
for i in $(seq 1 30); do curl -s -o /dev/null http://127.0.0.1:8501/ && echo "dashboard up" && exit 0; sleep 1; done
echo "dashboard did not start"; tail -20 logs/dashboard.log; exit 1
