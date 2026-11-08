@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
python llama_server_bench6.py --new %* > bench6_run.log 2>&1
set RC=%errorlevel%
type bench6_run.log
echo.
echo === exit code %RC% | log complet: bench6_run.log ===
pause
