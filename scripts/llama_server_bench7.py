#!/usr/bin/env python3
"""
llama_server_bench5.py — runda mica de confirmare mask2, derivata din bench4 (structura care a mers).

Ce testeaza (hipoteza confirmata din PR #22 al fork-ului: mask2+dyn32 = 18.4 t/s pe un 3060):
  --moe-cache-mask 2 forteaza routerul pe experții din cache GPU; doar top-2 citesc din RAM.
  Asta taie citirile/token de la ~2.17 GB la ~0.74 GB -> decode mult peste 2.74 t/s din bench4.

Fixuri fata de bench4:
  1. TOATE configurile au --moe-cache-mask 2 (in bench4 lipsea din aproape tot).
  2. --moe-cache-margin 2048 (bench4: margin 768/1024 -> pp buffer de 1653-1765 MiB nu mai incapea
     dupa ce cache-ul umflase CUDA0 -> OOM in generare la B/S/G).
  3. -b 256 -ub 128 din start (graph de prefill ~0.8 GB in loc de 1.65 GB). Fallback: margin 3072.
  4. Fara re-trace implicit (llama-moe-trace nu accepta --no-jinja -> bench4_trace a crapat);
     profilul se refoloseste din profile2.csv (aceeasi bin + acelasi model). --trace forateaza re-trace corect.
  5. Summary incremental bench7_summary.json, gate VRAM/RAM identic bench4.

Rulezi:  RUN-BENCH5.bat  (sau python llama_server_bench5.py [--only a,b] [--dry] [--no-kill]
                         [--trace] [--no-prefill] [--fit-target N])
Opriti:  STOP-BENCH5.bat
"""
import os, sys, json, time, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_NAME = "build-readahead" if "--new" in sys.argv else "build-dsv41"
BIN = os.path.join(HERE, BIN_NAME, "bin")
SERVER = os.path.join(BIN, "llama-server.exe")
TRACE = os.path.join(BIN, "llama-moe-trace.exe")
MODEL = os.environ.get("DSV41_MODEL", "CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")
PROFILE = os.path.join(HERE, "profile5.csv")
PROFILE_SRC = os.path.join(HERE, "profile2.csv")   # profil valid din bench2 (aceeasi bin/model)
PORT = 8090
N_PRED = 1536
REASON_BUDGET = 768
VRAM_GATE_MAX_MIB = 1200
RAM_GATE_MIN_GB = 20.0

SHORT_PROMPT = ("Write a Python function solve_microstrip_width(z_target, h_mm, er, t_oz) that finds the "
                "microstrip trace width for a target 50 Ohm impedance using Hammerstad-Jensen equations. "
                "Explain the role of the effective dielectric constant.")
LONG_PROMPT = None

def long_prompt():
    global LONG_PROMPT
    if LONG_PROMPT is None:
        base = ("Signal integrity review notes: impedance, reflections, termination, via stubs, "
                "return loss, insertion loss, skew, PDN resonance. ")
        LONG_PROMPT = base * 120
    return LONG_PROMPT

# ---- matricea: fix mask2, marimi controlate ca sa nu OOM (model 5440 + cache + graph < 16302) ----
def C(extra, env=None): return (extra, env or {})

MASK = ["--moe-cache-mask", "2", "--moe-cache-dyn", "32", "--moe-cache-margin", "2048"]

CONFIGS = [
    # I = F (41 t/s prefill) + weights-in-ram 60 (README: page-cache holding two groups;
    #     noi avem 83 GB liberi, Codacus a testat cu ~50)
    ("I_wir60",          C(MASK + ["--moe-cache-slots", "8", "-b", "4096", "-ub", "1024",
                                   "--weights-in-ram", "60"])),
    # J = ub 512 -> 7 ubatches pentru un prompt de 3271; lm3 are ce grupa (la F cu 4 ub n-a ajutat)
    ("J_ub512_lm3",      C(MASK + ["--moe-cache-slots", "8", "-b", "4096", "-ub", "512"])),
    # K = combinatia I+J
    ("K_wir60_ub512_lm3", C(MASK + ["--moe-cache-slots", "8", "-b", "4096", "-ub", "512",
                                    "--weights-in-ram", "60"])),
    # L = recipe-ul exact din README fork: lazy-mode on + no-sched-async-cpu (vs auto al nostru)
    ("L_recipe_exact",   C(MASK + ["--moe-cache-slots", "8", "-b", "4096", "-ub", "1024",
                                   "--lazy-mode", "on", "--no-sched-async-cpu"])),
]
LM3 = {"J_ub512_lm3", "K_wir60_ub512_lm3"}
NO_THINK = set()

# ---------------- helpers (identice bench4) ----------------
def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""

def free_ram_gb():
    out = sh(["powershell", "-NoProfile", "-Command",
              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"]).strip()
    try: return float(out.replace(",", "."))
    except ValueError: return -1.0

def vram_used_mib():
    out = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    vals = []
    for line in out.splitlines():
        try: vals.append(int(line.strip()))
        except ValueError: pass
    return vals

def gpu_holders():
    out = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    return [l.strip() for l in out.splitlines() if l.strip()]

def port_busy(port):
    out = sh(["netstat", "-ano"])
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            return line.split()[-1]
    return None

def kill_pid(pid):
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)

def stop_flash():
    print("[1] Opresc Strata/Flash + llama-server ramase (kill pe NUME, nu pe port)...")
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-Process llama-server,strata-vision,strata -ErrorAction SilentlyContinue | "
              "ForEach-Object { $_.Id }"])
    for pid in out.split():
        print(f"  kill llama-server/strata-vision/strata pid {pid}")
        kill_pid(pid)
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
              "Where-Object { $_.CommandLine -like '*Strata*' } | ForEach-Object { $_.ProcessId }"])
    for pid in out.split():
        print(f"  kill wrapper python Strata pid {pid}")
        kill_pid(pid)
    netstat = sh(["netstat", "-ano"])
    for port in (8080, 8081, 8082, PORT):
        for line in netstat.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                print(f"  kill ascultator pe port {port} pid {pid}")
                kill_pid(pid)
    print("  astept eliberare VRAM (max 60 s)...")
    for i in range(12):
        time.sleep(5)
        v = vram_used_mib()
        if v and v[0] <= VRAM_GATE_MAX_MIB:
            print(f"  CUDA0 liber: {v[0]} MiB dupa {(i+1)*5} s")
            return True
    return False

def gate_vram():
    v = vram_used_mib()
    if not v:
        print("  GATE: nvidia-smi nu intoarce date -> nu pot verifica VRAM. Rulez oricum.")
        return True
    print(f"  VRAM used: CUDA0={v[0]} MiB" + (f", CUDA1={v[1]} MiB" if len(v) > 1 else ""))
    holders = gpu_holders()
    if holders:
        print("  procese pe GPU: " + " | ".join(holders))
    if v[0] > VRAM_GATE_MAX_MIB:
        print(f"  FATAL: CUDA0 foloseste {v[0]} MiB (>{VRAM_GATE_MAX_MIB}).")
        print("  5080 nu e liber -> bench-ul ar muri la 'cudaMalloc failed: out of memory' ca bench2/3.")
        print("  Inchide manual ce ocupa GPU-ul (Strata/jocuri) si re-ruleaza.")
        return False
    return True

def gate_ram():
    ram = free_ram_gb()
    print(f"  RAM liber: {ram:.1f} GB")
    if ram < RAM_GATE_MIN_GB:
        print(f"  FATAL: doar {ram:.1f} GB RAM liber (<{RAM_GATE_MIN_GB}) — mmap + offload ar thrashui.")
        return False
    return True

def ensure_profile():
    """Profile5 din profile2 (valid, aceeasi bin/model). Fara --no-jinja in trace (trace-ul nu-l accepta)."""
    if os.path.exists(PROFILE):
        return True
    if os.path.exists(PROFILE_SRC):
        import shutil
        shutil.copy(PROFILE_SRC, PROFILE)
        print(f"  profile5.csv creat din profile2.csv ({os.path.getsize(PROFILE)} bytes)")
        return True
    return False

def run_trace():
    """Re-trace corect: llama-moe-trace NU accepta --no-jinja (aici a crapat bench4_trace)."""
    print("\n[2] Profil routing (llama-moe-trace, 2 prompturi, fara --no-jinja)...")
    prompts = [
        ("code", "Write a Python function that simulates a transactional key-value store with savepoints and deadlock detection."),
        ("si",   "A 50 Ohm microstrip on FR4 (er=4.4, h=1.6 mm) needs about 2.8 mm width. Explain briefly."),
    ]
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1")
    parts = []
    for tag, p in prompts:
        out_csv = os.path.join(HERE, f"profile5_{tag}.csv")
        log = os.path.join(HERE, f"bench7_trace_{tag}.log")
        env2 = dict(env, MOE_TRACE_OUT=out_csv)
        cmd = [TRACE, "-m", MODEL, "-dev", "CUDA0", "-ngl", "99", "-ncmoe", "99",
               "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
               "-b", "512", "-ub", "256", "--temp", "0.7", "-n", "1024", "-p", p]
        with open(log, "wb") as lf:
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1200, env=env2)
            except subprocess.TimeoutExpired:
                print(f"  trace {tag}: TIMEOUT 1200s")
        if os.path.exists(out_csv):
            parts.append(out_csv)
            print(f"  profil {tag}: {os.path.getsize(out_csv)} bytes")
        else:
            print(f"  profil {tag}: LIPSA (vezi bench7_trace_{tag}.log)")
    if parts:
        with open(PROFILE, "w") as out:
            for i, fpath in enumerate(parts):
                with open(fpath) as fh:
                    lines = fh.readlines()
                    out.writelines(lines if i == 0 else lines[1:])
        print(f"  => {PROFILE} ({os.path.getsize(PROFILE)} bytes)")
        return True
    return False

def chat_body(prompt, n_predict, think_on):
    b = {"messages": [{"role": "user", "content": prompt}],
         "max_tokens": n_predict, "temperature": 0.7}
    if think_on:
        b["reasoning_effort"] = "high"
    return json.dumps(b).encode()

def post_chat(prompt, n_predict, think_on=True, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=chat_body(prompt, n_predict, think_on),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    u = d.get("usage", {})
    timings = d.get("timings", {})
    content = d["choices"][0]["message"].get("content", "") or ""
    reasoning = d["choices"][0]["message"].get("reasoning_content", "") or ""
    return dt, u.get("completion_tokens", 0), u.get("prompt_tokens", 0), content, reasoning, timings

def wait_ready(proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False

def tail_log(path, n=25):
    try:
        with open(path, "rb") as f:
            txt = f.read().decode("utf-8", "ignore")
        return [l.strip() for l in txt.splitlines() if l.strip()][-n:]
    except Exception:
        return []

def extract_info(txt):
    cache_line, offload, mask_line = "", "", ""
    for line in txt.splitlines():
        if "moe_cache_build" in line:
            cache_line = line.split("I moe_cache_build:")[-1].strip()
        if "offloaded" in line and "expert" in line.lower():
            offload = line.strip()[-120:]
        if "moe_cache_mask" in line or "mask" in line.lower() and "cache" in line.lower():
            mask_line = line.strip()[-120:]
    return cache_line, offload, mask_line

def build_cmd(name, extra, fit_target):
    think_off = name in NO_THINK
    budget = REASON_BUDGET
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
           "-b", "256", "-ub", "128", "-np", "1",          # FIX bench4: -b 512 -> pp graph 1.65 GB
           "--reasoning", "off" if think_off else "on",
           "--reasoning-budget", str(budget),
           "--reasoning-format", "deepseek",
           "--skip-chat-parsing",
           "--port", str(PORT), "-v"]
    if BIN_NAME == "build-readahead":
        # flags din fork-ul actualizat (docs README: layer-major + readahead)
        cmd += ["--layer-major", "3" if name in LM3 else "1"]
    if fit_target:
        cmd += ["--fit", "on", "--fit-target", str(fit_target)]
    if not any(a == "-dev" for a in extra):
        cmd += ["-dev", "CUDA0"]
    cmd += extra
    return cmd

def measure_once(name, cmd, env, log, think_on, do_prefill):
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_ready(proc):
                code = proc.poll()
                tail = tail_log(log, 25)
                reason = "OOM" if any("out of memory" in l or "failed to allocate" in l for l in tail) \
                         else ("template" if any("chat template" in l for l in tail) else f"server exit {code}")
                return None, reason, tail
            try:
                post_chat("Hi.", 8, think_on=think_on)
            except Exception as e:
                print(f"  warmup esuat: {e}")
            gens, samples = [], ""
            for i in range(3):
                try:
                    dt, ctok, ptok, content, reasoning, tim = post_chat(SHORT_PROMPT, N_PRED, think_on=think_on)
                except Exception as e:
                    print(f"  run {i+1} a crapat: {e}")
                    break
                if ctok:
                    g = tim.get("predicted_per_second") or (ctok / dt if dt else 0)
                    gens.append(g)
                    print(f"  run {i+1}: {ctok} tok in {dt:.1f}s = {g:.1f} t/s server "
                          f"(think {len(reasoning)} chars)")
                    samples = content[:200]
            prefill_tps = 0.0
            if do_prefill:
                try:
                    dt, ctok, ptok, _, _, tim = post_chat(long_prompt(), 16, think_on=think_on)
                    if dt > 0:
                        prefill_tps = tim.get("prompt_per_second") or (ptok / dt)
                        print(f"  prefill: {ptok} tok in {dt:.1f}s = {prefill_tps:.1f} t/s")
                except Exception as e:
                    print(f"  prefill esuat: {e}")
            if not gens:
                tail = tail_log(log, 25)
                reason = "OOM in generare" if any("out of memory" in l for l in tail) else "no gen measured"
                return None, reason, tail
            with open(log, "rb") as f:
                txt = f.read().decode("utf-8", "ignore")
            cache_line, offload, mask_line = extract_info(txt)
            return ({"gen_tps": round(sum(gens)/len(gens), 2),
                     "prefill_tps": round(prefill_tps, 2),
                     "cache": cache_line, "offload": offload,
                     "sample": samples[:160]}, "OK", [])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill_pid(proc.pid)
            time.sleep(6)

def run_cfg(name, extra, env_extra, fit_target, do_prefill):
    print(f"\n[TEST] {name}: {' '.join(extra) if extra else '(fara cache)'}")
    v = vram_used_mib()
    if v and v[0] > VRAM_GATE_MAX_MIB:
        print(f"  GATE per-config: CUDA0={v[0]} MiB > {VRAM_GATE_MAX_MIB} -> reincerc gate")
        if not gate_vram():
            return {"gen_tps": None, "prefill_tps": None, "fail": "gate VRAM inainte de config"}
    # prefetch 2 seturi (README: =6 da dublu-buffer dar costa ~6 tensori/strat VRAM;
    # cu cache 8-12 slots + margin 2048, 2 e compromisul sigur pe 16 GB)
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1", GGML_SCHED_PREFETCH_EXPERTS="2")
    if BIN_NAME == "build-readahead":
        env["GGML_SCHED_READAHEAD"] = "16"
    env.update(env_extra)
    think_off = name in NO_THINK
    log = os.path.join(HERE, f"bench7_{name}.log")
    cmd = build_cmd(name, extra, fit_target)
    result, reason, tail = measure_once(name, cmd, env, log, think_on=not think_off, do_prefill=do_prefill)
    # fallback OOM: margin 3072 (lasam mai mult loc liber pe CUDA0 pentru pp graph)
    if result is None and "OOM" in reason:
        print("  fallback OOM: reiau cu --moe-cache-margin 3072")
        cmd2 = list(cmd)
        for i, a in enumerate(cmd2):
            if a == "--moe-cache-margin" and i + 1 < len(cmd2):
                cmd2[i+1] = "3072"
        log2 = os.path.join(HERE, f"bench7_{name}_m3072.log")
        result, reason2, tail = measure_once(name, cmd2, env, log2, think_on=not think_off, do_prefill=do_prefill)
        if result:
            result["note"] = "reusit cu margin 3072 (fallback OOM)"
    if result is None:
        return {"gen_tps": None, "prefill_tps": None, "fail": reason, "last_log": tail[-12:]}
    return result

def main():
    known = {"--only", "--no-kill", "--trace", "--dry", "--no-prefill", "--fit-target", "--new"}
    for a in sys.argv[1:]:
        if a.startswith("-") and a not in known and not (sys.argv[sys.argv.index(a)-1] in known):
            print("Folosire: python llama_server_bench5.py [--only nume1,nume2] [--dry] [--no-kill] [--trace] [--no-prefill] [--fit-target N]")
            print("  --dry         printeaza comenzile + starile GPU/RAM, nu atinge GPU-ul")
            print("  --no-kill     nu opri Strata (presupune ca tu l-ai oprit manual)")
            print("  --trace       re-face profilul cu llama-moe-trace (altfel refoloseste profile2.csv)")
            print("  --no-prefill  sare peste masurarea prefill (ruleaza mai repede)")
            print("  --fit-target N  adauga --fit on --fit-target N la toate serverele")
            return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    fit_target = None
    if "--fit-target" in sys.argv:
        fit_target = int(sys.argv[sys.argv.index("--fit-target") + 1])
    do_prefill = "--no-prefill" not in sys.argv

    print("=== PREFLIGHT ===")
    ok_model = os.path.exists(MODEL)
    print(f"  model GGUF: {'OK' if ok_model else 'LIPSA ' + MODEL}")
    ok_bin = os.path.exists(SERVER)
    print(f"  llama-server: {'OK' if ok_bin else 'LIPSA ' + SERVER}")
    busy = port_busy(PORT)
    print(f"  port {PORT}: {'OCUPAT de pid ' + busy + ' — FATAL' if busy else 'liber'}")
    prof = "profile5 existent" if os.path.exists(PROFILE) else \
           ("din profile2.csv" if os.path.exists(PROFILE_SRC) else "LIPSA -> D se sari")
    print(f"  profil routing: {prof}")
    if not ok_model or not ok_bin or busy:
        print("FATAL: preflight esuat.")
        return 1

    if "--dry" in sys.argv:
        print("\n=== DRY RUN (nu ating GPU-ul) ===")
        print(f"  VRAM used: {vram_used_mib()} MiB | RAM liber: {free_ram_gb():.1f} GB")
        for name, (extra, env_extra) in CONFIGS:
            print(f"  {name}: {' '.join(build_cmd(name, extra, fit_target))}")
        return 0

    if "--no-kill" not in sys.argv:
        stop_flash()
    print("\n[gate] VRAM:")
    if not gate_vram():
        return 1
    print("[gate] RAM:")
    if not gate_ram():
        return 1

    if "--trace" in sys.argv:
        run_trace()
    ensure_profile()
    if not os.path.exists(PROFILE):
        print("  profile5.csv lipseste -> configul D va fi sarit")

    results = {}
    for name, (extra, env_extra) in CONFIGS:
        if only and not any(o.lower() in name.lower() for o in only):
            print(f"\n[SKIP] {name}")
            continue
        if "--moe-cache-profile" in extra and not os.path.exists(PROFILE):
            print(f"\n[SKIP] {name} (profile5.csv lipseste)")
            results[name] = {"gen_tps": None, "prefill_tps": None, "fail": "profile lipseste"}
            continue
        try:
            results[name] = run_cfg(name, extra, env_extra, fit_target, do_prefill)
        except Exception as e:
            print(f"  {name}: EXCEPTIE {e}")
            results[name] = {"gen_tps": None, "prefill_tps": None, "fail": f"exceptie {e}"}
        with open(os.path.join(HERE, "bench7_summary.json"), "w") as f:
            json.dump(results, f, indent=1)
        v = vram_used_mib()
        if v and v[0] > VRAM_GATE_MAX_MIB:
            print(f"  post-config CUDA0={v[0]} MiB — mai mult decat gate; kill residue")
            out = sh(["powershell", "-NoProfile", "-Command",
                      "Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object { $_.Id }"])
            for pid in out.split():
                kill_pid(pid)
            time.sleep(8)

    print("\n============ REZUMAT bench5 (mask2, warm server, gate VRAM/RAM) ============")
    print(f"{'config':24s} {'gen t/s':>8s} {'prefill':>8s}  notes")
    for name, _ in CONFIGS:
        r = results.get(name)
        if r and r.get("gen_tps"):
            note = (r.get("note") or r.get("cache") or "")[:55]
            print(f"{name:24s} {r['gen_tps']:>8.1f} {r.get('prefill_tps', 0):>8.1f}  {note}")
        elif r:
            print(f"{name:24s} {'ESUAT':>8s} {'':>8s}  {r.get('fail', '?')}")
        else:
            print(f"{name:24s} {'SKIP':>8s}")
    print("\nReference: bench4 max 2.74 | bench1 I_slots8 (mask2) 8.93 | Codacus 3060 mask2 warm 18.4")
    print("Summary: bench7_summary.json | loguri: bench7_*.log")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(2)
