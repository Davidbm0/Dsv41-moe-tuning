@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo Rulez bench8 (lazy-mode on, decode coerent 512 tok)...
echo Log complet: bench8_run.log
python llama_server_bench8.py --new %* > bench8_run.log 2>&1
echo.
echo Exit code: %ERRORLEVEL%  (detalii in bench8_run.log)
pause
