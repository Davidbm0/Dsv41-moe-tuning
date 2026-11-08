@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo Rulez bench10 (2x2 mask x lazy, reasoning OFF)...
echo Log complet: bench10_run.log
python llama_server_bench10.py --new %* > bench10_run.log 2>&1
echo.
echo Exit code: %ERRORLEVEL%  (detalii in bench10_run.log)
pause
