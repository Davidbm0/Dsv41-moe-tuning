@echo off
chcp 65001 >nul
echo Omor llama-server / llama-moe-trace / strata ramasi pe GPU...
powershell -NoProfile -Command "Get-Process llama-server,llama-moe-trace,strata-vision,strata -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('kill ' + $_.ProcessName + ' pid ' + $_.Id); Stop-Process -Id $_.Id -Force }"
pause
