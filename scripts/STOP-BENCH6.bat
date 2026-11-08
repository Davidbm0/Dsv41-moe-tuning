@echo off
taskkill /F /IM llama-server.exe 2>nul
taskkill /F /IM llama-moe-trace.exe 2>nul
echo Oprit servere llama. (Strata trebuie repornit manual.)
pause
