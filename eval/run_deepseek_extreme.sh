#!/usr/bin/env bash
# run_deepseek_extreme.sh — eval DeepSeek V4.1 Flash IQ2_XS pe suite-ul extreme P1-P5
# Flow: opreste Strata -> porneste llama-server DeepSeek pe 8081 (split automat pe ambele GPU-uri)
#       -> coding eval (pass@1, single step) -> kill llama-server -> reporneste Strata
# Ruleaza prin run_deepseek_visible.bat (fereastra vizibila cu progress live).
# 04.11: foloseste llama-fork (build thecodacus, fused DSV4 HC ops + moe-cache-slots) —
#        build-ul vechi din llama.cpp\ nu are HC fused => ~2.5 t/s.
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM="$DIR/llama-fork/build/bin/llama-server.exe"
GGUF="${DSV41_MODEL:-CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf}"
PORT=8081
SETUP="DeepSeek-V4.1-IQ2XS_Extreme_pass1"
LOG="$DIR/deepseek_extreme_$(date +%d.%H%M).log"

cd "$DIR" || exit 1

cleanup () {
  echo "[cleanup] omor llama-server..."
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"
  sleep 3
  if curl -s -m 5 "http://127.0.0.1:8080/health" | grep -q ok; then
    echo "[cleanup] Strata e deja sus — nu fac nimic."
  else
    echo "[cleanup] repornesc Strata (run-iq3_xxs.bat)..."
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '${STRATA_DIR:-CHANGE_ME\Strata}\run-iq3_xxs.bat'"
  fi
  echo "[cleanup] gata. Rezultate: results/$SETUP*.json, log: $LOG"
}
trap cleanup EXIT INT TERM

cat <<'BANNER'
============================================================
  DEEPSEEK V4.1 Flash IQ2_XS — EVAL P1-P5 (pass@1)
  NU inchide aceasta fereastra. NU porni Strata manual.
  GPU-ul va parea "gol" (5-10% util) — E NORMAL: la acest
  model urias expertii ruleaza pe CPU/RAM, nu pe placa video.
  Fara activitate pe GPU nu inseamna ca s-a blocat.
  Timp estimat: ~70-110 min/problema, 5 probleme = 6-9 h.
============================================================
BANNER

# 1. Oprire Strata (eliberam 5080; Gemma de pe 3080 ramane pornita = creier de rezerva)
echo "[1/4] opresc Strata..."
powershell -NoProfile -Command "Get-Process strata,strata-vision -ErrorAction SilentlyContinue | Stop-Process -Force"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'Strata' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 4

# verificare: Strata trebuie sa fie cu adevarat jos inainte de DeepSeek
if curl -s -m 5 "http://127.0.0.1:8080/health" | grep -q ok; then
  echo "FATAL: Strata inca ruleaza pe 8080 dupa incercarea de oprire."
  echo "Inchide-o manual (tray / task manager) si ruleaza din nou acest script."
  exit 4
fi

# 2. Pornire DeepSeek — llama-fork: expert cache pe GPU (moe-cache-slots auto = umple VRAM-ul liber)
echo "[2/4] pornesc DeepSeek V4.1 Flash IQ2_XS (incarcare din SSD — 2-10 min)..."
MOE=(-ot "ffn_.*_exps.=CPU" --n-gpu-layers 99 --fit on --fit-target 512)
"$LLM" -m "$GGUF" --host 127.0.0.1 --port $PORT \
  -c 32768 -ctk q8_0 -ctv q8_0 --ubatch-size 1024 \
  --moe-cache-slots auto --moe-cache-margin 512 --moe-cache-borrow 512 \
  --split-mode none --main-gpu 0 \
  --no-jinja --chat-template chatml \
  "${MOE[@]}" --reasoning on --reasoning-format deepseek \
  > "$DIR/serve_deepseek.out" 2>&1 &
LLM_PID=$!

echo "   astept health pe :$PORT ..."
for i in $(seq 1 120); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q ok; then break; fi
  if ! kill -0 $LLM_PID 2>/dev/null; then
    echo "FATAL: llama-server a murat la pornire. Ultimele 30 linii:"; tail -30 "$DIR/serve_deepseek.out"; exit 2
  fi
  sleep 10
done
if ! curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q ok; then
  echo "FATAL: health nu a devenit ok in 20 min. Vezi serve_deepseek.out"; exit 3
fi
MODEL_ID=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "   server ok. model: DeepSeek IQ2_XS pe port $PORT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 3. Heartbeat: la fiecare 30s afiseaza ultima linie din logul serverului (tokeni generati + t/s)
( while :; do sleep 30; tail -n1 "$DIR/serve_deepseek.out" 2>/dev/null; done ) &
HB=$!

# 4. Eval: suite P (P1-P5), pass@1 single step
echo "[3/4] rulez eval P1-P5 (pass@1, max_tokens 16384, think). Progress per problema mai jos..."
python coding_eval_extreme.py --port $PORT --model "$MODEL_ID" --setup "$SETUP" \
  --mode think --max-tokens 16384 --http-timeout 7200 --max-repairs 0 --suite p 2>&1 | tee "$LOG"
kill $HB 2>/dev/null

echo "[4/4] eval terminat."
