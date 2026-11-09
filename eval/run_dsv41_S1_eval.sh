#!/usr/bin/env bash
# run_dsv41_S1_eval.sh — eval extreme P1-P5 pe DeepSeek V4.1-Flash cu RETETA CÂȘTIGATOARE din bench10 S1
# (build-readahead, mask1 + lazy on, reasoning OFF).
#
# Server ruleaza exact configul S1 din bench10 (llama-forks/llama_server_bench10.py):
#   --moe-cache-mask 1 --moe-cache-dyn 32 --moe-cache-margin 2048 --moe-cache-slots 8
#   -b 4096 -ub 1024 --lazy-mode on --no-sched-async-cpu --reasoning off
#   + env GGML_CUDA_REGISTER_HOST=1, GGML_SCHED_PREFETCH_EXPERTS=2, GGML_SCHED_READAHEAD=16
#   + --reasoning-budget 768 --reasoning-format deepseek --skip-chat-parsing
#   -ngl 99 -ncmoe 99 --load-mode mmap -c 4096 -fa on -np 1, port 8090
#
# Eval: coding_eval_extreme.py, single step (pass@1) + 4 self-repairuri (pass@5), timp per problema.
#
# Folosire:  bash run_dsv41_S1_eval.sh [--no-kill] [--max-repairs N] [--problems p1,p3] [--nothink]
# Default:  THINK single-pas (reasoning ON, budget nelimitat, max_tokens 16384, ctx 24576, 0 repairuri)
#           = echivalent testul M5 din 5.11.  --nothink = rețeta bench10 S1 (reasoning off, 6144 tok, 4 repairuri)
# Oprire:    Ctrl+C (serverul e omorât la ieșire)
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

FORK="${DSV41_FORK_DIR:-CHANGE_ME/llama-forks}"
SERVER="$FORK/build-readahead/bin/llama-server.exe"
MODEL="${DSV41_MODEL:-CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf}"
PORT=8090
# DEFAULT = echivalent M5 original (think, single pas, budget nelimitat) — comparare egala cu testul 5.11
MAXTOK=16384
MAXREPAIRS=0
PROBLEMS="p1,p2,p3,p4,p5"
MODE=think
REASON=on
BUDGET=-1
CTX0=24576
KILL_FLASH=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-kill)     KILL_FLASH=0 ;;
    --max-repairs) MAXREPAIRS="$2"; shift ;;
    --problems)    PROBLEMS="$2"; shift ;;
    --ctx)         CTX0="$2"; shift ;;
    --max-tokens)  MAXTOK="$2"; shift ;;
    --nothink)     MODE=nothink; REASON=off; BUDGET=768; MAXTOK=6144; CTX0=8192; MAXREPAIRS=4 ;;
    --think)       MODE=think; REASON=on; BUDGET=-1 ;;
    --m5)          MODE=think; REASON=on; BUDGET=-1; MAXTOK=16384; CTX0=24576; MAXREPAIRS=0 ;;
    *) echo "Argument necunoscut: $1"; exit 1 ;;
  esac
  shift
done
SETUP="DSV41-S1"   # server: reteta bench10 S1 (mask1 + lazy on); default: think single-pas (echivalent M5)

echo "=== PREFLIGHT ==="
[ -x "$SERVER" ] || { echo "FATAL: llama-server lipseste: $SERVER"; exit 1; }
[ -f "$MODEL" ]  || { echo "FATAL: model GGUF lipseste: $MODEL"; exit 1; }
[ -f "coding_eval_extreme.py" ] || { echo "FATAL: coding_eval_extreme.py lipseste in $DIR"; exit 1; }

PY=python
command -v "$PY" >/dev/null 2>&1 || PY=python3
"$PY" --version || { echo "FATAL: python negasit"; exit 1; }

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
free_ram_gb() { powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB" 2>/dev/null | tr -d '\r' | tr ',' '.'; }

SRV_PID=""
cleanup() {
  # $! in git-bash NU e PID-ul Windows — omorim dupa nume, nu dupa PID
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  echo "[cleanup] llama-server omor (dupa nume)"
  if [ "$KILL_FLASH" = "1" ]; then
    echo "[cleanup] Repornesc Strata (8080 Flash-Next) ca sa revina chat-ul..."
    powershell -NoProfile -Command "Start-Process -FilePath \"${STRATA_DIR:-CHANGE_ME\Strata}\.venv\Scripts\python.exe\" -ArgumentList \"${STRATA_DIR:-CHANGE_ME\Strata}\serve\server.py --engine strata --config ${STRATA_DIR:-CHANGE_ME\Strata}\strata-iq3_xxs.json --port 8080 --open\" -WindowStyle Hidden" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

if [ "$KILL_FLASH" = "1" ]; then
  echo "[1] Opresc Strata + SUPERVISORUL lui (python server.py — altfel il reporneste in ~45 s) + llama-server ramase..."
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'Strata.*serve.server.py' } | ForEach-Object { Write-Host ('  kill supervisor pid ' + \$_.ProcessId); Stop-Process -Id \$_.ProcessId -Force }; Get-Process llama-server,strata-vision,strata -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('  kill ' + \$_.ProcessName + ' pid ' + \$_.Id); Stop-Process -Id \$_.Id -Force }" 2>/dev/null
  echo "  astept eliberare VRAM (max 60 s)..."
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    sleep 5
    V=$(vram_used)
    [ -n "$V" ] && [ "$V" -le 1500 ] && { echo "  CUDA0 liber: $V MiB"; break; }
  done
fi

echo "[gate] VRAM / RAM:"
V=$(vram_used)
if [ -n "$V" ] && [ "$V" -gt 1500 ]; then
  echo "  FATAL: CUDA0 foloseste $V MiB (>1500). GPU-ul e ocupat (joc/Strata) -> serverul ar muri la cudaMalloc OOM."
  echo "  Inchide ce ocupa GPU-ul si re-ruleaza (sau --no-kill daca ai deja liber)."
  exit 1
fi
echo "  CUDA0: ${V:-?} MiB"
R=$(free_ram_gb)
echo "  RAM liber: ${R:-?} GB"
if [ -n "$R" ]; then
  ok=$(awk -v r="$R" 'BEGIN{print (r>=20.0)?1:0}')
  [ "$ok" = "1" ] || { echo "  FATAL: sub 20 GB RAM liber — mmap + offload ar thrashui."; exit 1; }
fi

echo "[2] Pornesc llama-server cu reteta bench10 S1 (mask1 + lazy on, reasoning $REASON)..."
export CUDA_VISIBLE_DEVICES=0   # FORTEZ 5080 doar — fork-ul e compilat sm_120, pe 3080 (sm_86) crapa la warmup
export GGML_CUDA_REGISTER_HOST=1
export GGML_SCHED_PREFETCH_EXPERTS=2
export GGML_SCHED_READAHEAD=16

start_server() {
  # --skip-chat-parsing + --reasoning-budget se pun DOAR cu reasoning off (reteta bench10 S1).
  # Cu reasoning ON ele lasau <think> in campul content (reasoning_chars=0, gandire in cod) -> fix 09.11
  local RFLAGS=(--reasoning "$REASON" --reasoning-format deepseek)
  if [ "$REASON" = off ]; then
    RFLAGS+=(--reasoning-budget "$BUDGET" --skip-chat-parsing)
  else
    # think: exact ca testul original 5.11 (llama-fork): chatml fara jinja.
    # Template-ul jinja Unsloth + budget -1 lasa modelul in bucla de gandire pe p1/p3/p5.
    RFLAGS+=(--no-jinja --chat-template chatml)
  fi
  "$SERVER" -m "$MODEL" -ngl 99 -ncmoe 99 --load-mode mmap -c "$CTX" -fa on \
    -b 4096 -ub 1024 -np 1 \
    "${RFLAGS[@]}" \
    --moe-cache-mask 1 --moe-cache-dyn 32 --moe-cache-margin "$MARGIN" --moe-cache-slots 8 \
    --layer-major 1 --lazy-mode on --no-sched-async-cpu \
    -dev CUDA0 \
    --port "$PORT" -v > "S1_eval_server.log" 2>&1 &
  SRV_PID=$!
  echo "  server pid $SRV_PID (ctx $CTX, margin $MARGIN), log: S1_eval_server.log"
}

READY=0
for attempt in 1 2; do
  if [ "$attempt" = "1" ]; then CTX=$CTX0; MARGIN=2048
  else CTX=4096; MARGIN=3072; mv -f S1_eval_server.log S1_eval_server_try1.log 2>/dev/null
       echo "  [retry] a doua incercare: ctx 4096 + margin 3072 (fallbackul oficial din bench10)"
  fi
  start_server
  echo "[3] Astept /health 200 (max 30 min; stall 10 min in log = hang)..."
  N=0; LASTSIZE=-1; STALL=0
  for _ in $(seq 1 360); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "  serverul a murat la pornire (incercare $attempt). Ultimele 15 linii:"
      tail -15 S1_eval_server.log
      break
    fi
    if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then READY=1; break; fi
    N=$((N+1))
    if [ $((N % 12)) -eq 0 ]; then
      SZ=$(wc -c < S1_eval_server.log)
      [ "$SZ" = "$LASTSIZE" ] && STALL=$((STALL+1)) || STALL=0
      LASTSIZE=$SZ
      echo "    ... $(($N*5)) s | RAM liber: $(free_ram_gb) GB | log ${SZ}B stall ${STALL}m | $(tail -1 S1_eval_server.log | cut -c1-90)"
      [ "$STALL" -ge 10 ] && { echo "  HANG: log nemișcat 10 min — tai serverul"; break; }
    fi
    sleep 5
  done
  [ "$READY" = "1" ] && break
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  sleep 6
done
[ "$READY" = "1" ] || { echo "  FATAL: serverul nu a supravietuit/nu e gata in 2 incercari."; tail -25 S1_eval_server.log; exit 1; }
echo "  server sus pe :$PORT"

MODEL_ID=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ -z "$MODEL_ID" ] && { echo "  FATAL: /v1/models intoarce nimic."; exit 1; }
echo "  model id: $MODEL_ID"

echo "[4] Warmup (un 'Hi' ca sa iasa expertii la suprafata cache-ului)..."
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}" > S1_eval_warmup.json
grep -q '"choices"' S1_eval_warmup.json || { echo "  FATAL: warmup esuat:"; head -c 400 S1_eval_warmup.json; echo; exit 1; }
echo "  warmup OK"

# [4b] PROBE de stabilitate (C): o generare LUNGA (cat in eval) inainte de eval.
# If the server dies here, we know immediately (mid-generation crash) instead of
# burning all 5 problems on connection refused.
#
# [5] EVAL cu WATCHDOG (B): daca serverul crapa in timpul unei probleme,
# repornim (warmup + rehydratare cache MoE) si reincercam problema o data.
# A 2-a repornire de la al doilea crash incolo foloseste fallback-ul ctx 4096 + margin 3072.
RESTARTS=0
restart_server() {
  RESTARTS=$((RESTARTS+1))
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  sleep 6
  if [ "$RESTARTS" -ge 2 ]; then CTX=4096; MARGIN=3072; echo "  [watchdog] repornire $RESTARTS cu fallback ctx 4096 + margin 3072"
  else echo "  [watchdog] repornire $RESTARTS cu acelasi config S1"; fi
  start_server
  for _ in $(seq 1 360); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "  [watchdog] serverul a murat la pornire"; return 1; }
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && break
    sleep 5
  done
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}" > S1_eval_warmup.json
  grep -q '"choices"' S1_eval_warmup.json || { echo "  [watchdog] warmup post-restart esuat"; return 1; }
  echo "  [watchdog] server reluat, warmup OK"
  return 0
}

echo "[4b] Stability probe: generare lunga ($MAXTOK tokeni) — cautam crash mid-generation..."
curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long, complete, fully-documented Python module: a transmission-line loss calculator (skin effect, dielectric loss, conductor roughness models) with many helper functions and docstrings. Reply only with code in one python block.\"}],\"max_tokens\":$MAXTOK,\"temperature\":0.1}" > S1_eval_probe.json
if grep -q '"choices"' S1_eval_probe.json; then
  "$PY" - <<'EOF'
import json
j = json.load(open("S1_eval_probe.json"))
c = j["choices"][0]
u = j.get("usage", {})
print(f"  probe OK: finish={c.get('finish_reason')} completion_tokens={u.get('completion_tokens')}")
EOF
else
  echo "  probe a ESUAT (fara 'choices'). Ultimele 10 linii din server log (fara CUDA Graph):"
  grep -a -v "CUDA Graph" S1_eval_server.log | tail -10
fi
if ! curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
  echo "  probe: serverul NU mai e health dupa generarea lunga -> crash mid-generation confirmat."
  echo "  incerc sa repornesc serverul inainte de eval..."
  restart_server || { echo "  FATAL: nu pot reporni serverul dupa probe."; exit 1; }
fi

# [5] EVAL cu WATCHDOG (B): daca serverul crapa in timpul unei probleme,
# repornim (warmup + rehydratare cache MoE) si reincercam problema o data.
# Folosim restart_server definit mai sus (si ea numara RESTARTS -> a 2-a repornire = fallback).

STAMP=$(date +%d.%H%M)
echo "[5] Eval P1-P5 (mode $MODE, max_tokens $MAXTOK, pass@1 + $MAXREPAIRS self-repairuri, single step per incercare, watchdog)"
for P in p1 p2 p3 p4 p5; do
  case ",$PROBLEMS," in *",$P,"*) ;; *) echo "  [SKIP] $P"; continue ;; esac
  echo "==================== $P ===================="
  for PTRY in 1 2; do
    "$PY" coding_eval_extreme.py --port "$PORT" --model "$MODEL_ID" \
      --setup "$SETUP" --mode "$MODE" \
      --max-tokens "$MAXTOK" --http-timeout 2400 --max-repairs "$MAXREPAIRS" --problems "$P" \
      2>&1 | tee "S1_eval_${P}_${STAMP}.log"
    if ! curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
      echo "  [watchdog] server mort in timpul lui $P (health dat dupa problema)"
      grep -a "slot decode token\|n_gen = " S1_eval_server.log | tail -2
      restart_server || { echo "  [watchdog] nu pot reporni — $P ramane neterminata, trec mai departe"; break; }
    else
      break
    fi
  done
  # fiecare apel per-problema suprascrie results/DSV41-S1_nothink.json -> pastram copie per P
  cp -f "results/${SETUP}_${MODE}.json" "results/${SETUP}_${MODE}_${P}.json" 2>/dev/null
done

echo "[6] Rezumat per problema:"
for P in p1 p2 p3 p4 p5; do
  grep -h "^$P" "S1_eval_${P}_${STAMP}.log" 2>/dev/null
done
echo "JSON: results/${SETUP}_${MODE}.json | server log: S1_eval_server.log"
echo "[gata] Omor serverul (trap cleanup)..."
