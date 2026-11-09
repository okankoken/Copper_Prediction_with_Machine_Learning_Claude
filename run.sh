#!/usr/bin/env bash
# ./run.sh            snapshot + backtest + forecast + report
# ./run.sh test       leakage / consistency tests
# ./run.sh dashboard  Streamlit on port 8501
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-/home/train/miniconda3/bin/python}"

case "${1:-all}" in
  all)       "$PY" -m copperlab all "${@:2}" ;;
  run)       "$PY" -m copperlab run "${@:2}" ;;
  test)      "$PY" -m unittest discover -s tests -t . -v ;;
  dashboard) "$PY" -m streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" \
               --server.headless true --browser.gatherUsageStats false ;;
  *) echo "usage: $0 [all|run|test|dashboard]"; exit 1 ;;
esac
