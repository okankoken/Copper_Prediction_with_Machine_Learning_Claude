@echo off
rem Copper Prediction with Machine Learning Claude - Windows runner
rem   run.cmd setup      create .venv and install everything (first time only)
rem   run.cmd update     data from VM + news + model + tests + publish to GitHub (the monthly job)
rem   run.cmd data       refresh LME / macro data from the source project on the VM
rem   run.cmd stocks     refresh the equities (NVIDIA, miners, cables...) from Yahoo Finance
rem   run.cmd news       fetch new headlines and GDELT tone, score with FinBERT
rem   run.cmd run        tuning + backtest + forecast (~5-10 min)
rem   run.cmd test       leakage and consistency tests
rem   run.cmd dashboard  Streamlit at http://localhost:8510
rem   run.cmd publish    commit outputs and push to GitHub (Streamlit Cloud updates itself)
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "HF_HOME=%~dp0.hf_cache"
set "PYTHONIOENCODING=utf-8"
set "PYTHONWARNINGS=ignore"
if not defined COPPER_N_JOBS set "COPPER_N_JOBS=12"
if not exist logs mkdir logs

if "%~1"=="" goto usage
if /i "%~1"=="setup" goto setup
if not exist "%PY%" (echo .venv yok, once: run.cmd setup & exit /b 1)
if /i "%~1"=="update" goto update
if /i "%~1"=="data" ("%PY%" -m copperlab snapshot & exit /b %errorlevel%)
if /i "%~1"=="stocks" ("%PY%" -m copperlab stocks & exit /b %errorlevel%)
if /i "%~1"=="news" ("%PY%" -m copperlab news & exit /b 0)
if /i "%~1"=="run" ("%PY%" -m copperlab run & exit /b %errorlevel%)
if /i "%~1"=="test" ("%PY%" -m unittest discover -s tests -t . & exit /b %errorlevel%)
if /i "%~1"=="dashboard" ("%PY%" -m streamlit run app\dashboard.py --server.address localhost --server.port 8510 --server.headless true & exit /b 0)
if /i "%~1"=="publish" goto publish
goto usage

:setup
py -3.13 -m venv .venv || exit /b 1
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements-pipeline.txt --extra-index-url https://download.pytorch.org/whl/cpu
exit /b %errorlevel%

:update
echo [%date% %time%] update start
"%PY%" -m copperlab snapshot || (echo DATA FAILED & exit /b 1)
"%PY%" -m copperlab stocks || (echo STOCKS FAILED & exit /b 1)
"%PY%" -m copperlab news
"%PY%" -m copperlab run || (echo RUN FAILED & exit /b 1)
"%PY%" -m unittest discover -s tests -t . || (echo TESTS FAILED - not publishing & exit /b 1)
call :publish
echo [%date% %time%] update done
exit /b 0

:publish
git add -A -- outputs data/panel data/raw data/snapshot_manifest.json
git diff --cached --quiet && (echo nothing to publish & exit /b 0)
git commit -q -m "Forecast update %date%" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin main && echo pushed to GitHub
exit /b %errorlevel%

:usage
echo usage: run.cmd [setup^|update^|data^|news^|run^|test^|dashboard^|publish]
exit /b 1
