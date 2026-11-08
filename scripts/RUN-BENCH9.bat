@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo Rulez bench9 (mask1, lazy/reasoning despartite)...
echo Log complet: bench9_run.log
python llama_server_bench9.py --new %* > bench9_run.log 2>&1
echo.
echo Exit code: %ERRORLEVEL%  (detalii in bench9_run.log)
pause
