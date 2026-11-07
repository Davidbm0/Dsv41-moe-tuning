@echo off
taskkill /F /IM llama-server.exe 2>nul
taskkill /F /IM llama-moe-trace.exe 2>nul
taskkill /F /IM strata.exe 2>nul
taskkill /F /IM strata-vision.exe 2>nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*bench5*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Bench5 oprit.
