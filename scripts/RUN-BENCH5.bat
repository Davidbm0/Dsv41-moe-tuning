@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
python llama_server_bench5.py %* > bench5_run.log 2>&1
set RC=%errorlevel%
type bench5_run.log
echo.
echo === exit code %RC% | log complet: bench5_run.log ===
pause
