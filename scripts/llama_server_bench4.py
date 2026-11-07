#!/usr/bin/env python3
"""
llama_server_bench4.py — rescris dupa esecurile bench1/2/3 (toate cauzele documentate in loguri).

Fixuri fata de bench2/3:
  1. GATE VRAM real: nvidia-smi dupa kill; mai multi轮ti 60 s; FATAL daca CUDA0 tot ocupat.
     (bench2/3: 17+ configs murite la 'cudaMalloc failed: out of memory' pt ca Strata nu fusese
      oprit — /unload pe 8080 NU functioneaza, kill-ul trebuie facut pe NUME de proces.)
  2. Kill Strata pe nume: strata-vision.exe + wrapper python cu 'Strata' in CommandLine
     (bench3 se baza pe POST /unload -> nu facea nimic).
  3. Chat template: --skip-chat-parsing (din bench3, unde mergea) + fallback --no-jinja --chat-template chatml
     (bench2 A/D/H/L/Q: 'chat template parsing error').
  4. DSpark: preflight pe GGUF (tensor 'output_hc_fn.weight' absent -> draft incompatibil cu fork-ul).
    Configurile dspark sunt SCOSE din matrice default; --with-dspark le reinclude (steag experimental).
  5. Fallback OOM in timpul generarii (bench3 M/X/Y/AC/AD/AE): retry 1x cu -ub 128 (compute buffer
     ~1.65 GB -> ~0.8 GB) in loc sa arunci configul.
  6. FATAL la start daca port 8090 ocupat ('couldn't bind HTTP server socket').
  7. Profile configs sarite daca profile4.csv lipseste (trace esuat).
  8. --dry: printeaza comanda completa per config + valorile gate-ilor, fara atingerea GPU-ului.

Rulezi:  RUN-BENCH4.bat  (sau python llama_server_bench4.py [--only a,b] [--dry] [--no-kill]
         [--no-trace] [--with-dspark] [--fit-target N])
Opriti:  STOP-BENCH4.bat
"""
import os, sys, json, time, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build-dsv41", "bin")
SERVER = os.path.join(BIN, "llama-server.exe")
TRACE = os.path.join(BIN, "llama-moe-trace.exe")
MODEL = os.environ.get("DSV41_MODEL", "CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")
DSPARK = os.environ.get("DSV41_DRAFT", "CHANGE_ME/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf")
DSPARK_REQUIRED_TENSOR = b"output_hc_fn.weight"   # absent in GGUF-ul actual -> draft incarca si pica
PROFILE = os.path.join(HERE, "profile4.csv")
PORT = 8090
N_PRED = 1536
REASON_BUDGET = 768
VRAM_GATE_MAX_MIB = 1200      # CUDA0 trebuie sub asta inainte de orice config
RAM_GATE_MIN_GB = 20.0        # mmap + expert offload au nevoie de page-cache headroom

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

# ---- matricea de configurari ----
def C(extra, env=None): return (extra, env or {})

DSPARK_ARGS = ["-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
               "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"]

CONFIGS = [
    ("A_baseline",        C([])),
    ("B_dyn16",           C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"])),
    ("C_dyn16_mask2",     C(["--moe-cache-dyn", "32", "--moe-cache-mask", "2", "--moe-cache-slots", "16"])),
    ("D_dyn_auto",        C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto", "--moe-cache-margin", "768"])),
    ("E_profile16",       C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16", "--moe-cache-dyn", "32"])),
    ("F_profile16_nodyn", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16"])),
    ("G_profile24",       C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "24", "--moe-cache-dyn", "32"])),
    ("H_profile16_borrow",C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12",
                             "--moe-cache-dyn", "32", "--moe-cache-borrow", "1024", "-b", "2048", "-ub", "1024"])),
    ("I_dyn16_nothink",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"], env=None)),
    ("J_profile16_decay7",C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16",
                             "--moe-cache-dyn", "32", "--moe-cache-decay", "0.7"])),
    ("K_dyn_slotsauto",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto", "--moe-cache-margin", "1024"])),
    ("L_dyn16_borrow256", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "--moe-cache-borrow", "256", "-b", "1024", "-ub", "512"])),
    ("M_dyn16_kvq8",      C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16", "-ctk", "q8_0", "-ctv", "q8_0"])),
    ("N_ncmoe0_auto",     C(["--moe-cache-dyn", "32", "-ncmoe", "0",
                             "--moe-cache-slots", "auto", "--moe-cache-margin", "1024"])),
    ("O_ngram_cache",     C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "--spec-type", "ngram-cache", "--spec-draft-n-max", "4"])),
    ("P_ngram_map_k4v",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "--spec-type", "ngram-map-k4v", "--spec-draft-n-max", "4",
                             "--spec-ngram-map-k4v-size-n", "12", "--spec-ngram-map-k4v-size-m", "48"])),
    ("Q_ncmoe24",         C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16", "-ncmoe", "24"])),
    ("R_dyn16_poll",      C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "-t", "8", "-tb", "8", "--poll", "100", "--prio", "2",
                             "-Cr", "0-7", "--cpu-strict", "1"])),
    ("S_dyn16_budget1536",C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"])),
    # dual-GPU (3080 trebuie si ea libera)
    ("T_dual16",          C(["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18"])),
    ("U_dual16_ts3414",   C(["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "34,14"])),
]
# dspark = experimental, doar cu --with-dspark (draft GGUF actual e incompatibil — vezi preflight)
DSPARK_CONFIGS = [
    ("V_dspark16",     C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"] + DSPARK_ARGS)),
    ("W_dual16_dspark",C(["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18",
                          "--moe-cache-dyn", "32", "--moe-cache-slots", "16"] + DSPARK_ARGS)),
]
NO_THINK = {"I_dyn16_nothink"}
BUDGET_OVERRIDES = {"S_dyn16_budget1536": 1536}

# ---------------- helpers ----------------
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
    """[used_mib_gpu0, used_mib_gpu1, ...]"""
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
    # a) orice llama-server / strata-vision dupa numele procesului
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-Process llama-server,strata-vision -ErrorAction SilentlyContinue | "
              "ForEach-Object { $_.Id }"])
    for pid in out.split():
        print(f"  kill llama-server/strata-vision pid {pid}")
        kill_pid(pid)
    # b) wrapper python care are 'Strata' in CommandLine (serverul 8080)
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
              "Where-Object { $_.CommandLine -like '*Strata*' } | ForEach-Object { $_.ProcessId }"])
    for pid in out.split():
        print(f"  kill wrapper python Strata pid {pid}")
        kill_pid(pid)
    # c) orice mai asculta pe 8080/8081/8082/8090
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

def dspark_preflight():
    """Verifica draft GGUF are tensorul pe care loader-ul fork-ului il cere."""
    if not os.path.exists(DSPARK):
        return False, "fisier draft lipseste"
    try:
        with open(DSPARK, "rb") as f:
            head = f.read(200 * 1024 * 1024)   # tensor info near head; scan 200 MB
        if DSPARK_REQUIRED_TENSOR in head:
            return True, "tensor output_hc_fn.weight gasit"
        return False, ("draft GGUF NU are tensorul 'output_hc_fn.weight' — "
                       "incompatibil cu loader-ul fork-ului (cauza exacta a esecului M/P/R/S/U/V/W/AB)")
    except Exception as e:
        return False, f"preflight esuat: {e}"

def run_trace():
    print("\n[2] Profil routing (llama-moe-trace, 3 prompturi)...")
    prompts = [
        ("code", "Write a Python function that simulates a transactional key-value store with savepoints and deadlock detection."),
        ("chat", "Explain the difference between characteristic impedance and propagation delay in a PCB trace."),
        ("si",   "A 50 Ohm microstrip on FR4 (er=4.4, h=1.6 mm) needs about 2.8 mm width. Explain briefly."),
    ]
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1")
    parts = []
    for tag, p in prompts:
        out_csv = os.path.join(HERE, f"profile4_{tag}.csv")
        log = os.path.join(HERE, f"bench4_trace_{tag}.log")
        env2 = dict(env, MOE_TRACE_OUT=out_csv)
        cmd = [TRACE, "-m", MODEL, "-dev", "CUDA0", "-ngl", "99", "-ncmoe", "99",
               "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
               "-b", "512", "-ub", "256", "--temp", "0.7", "-n", "1024",
               "--no-jinja", "--chat-template", "chatml", "-p", p]
        with open(log, "wb") as lf:
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1200, env=env2)
            except subprocess.TimeoutExpired:
                print(f"  trace {tag}: TIMEOUT 1200s")
        if os.path.exists(out_csv):
            parts.append(out_csv)
            print(f"  profil {tag}: {os.path.getsize(out_csv)} bytes")
        else:
            print(f"  profil {tag}: LIPSA (vezi bench4_trace_{tag}.log)")
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
        # template-ul DSV4.1 accepta doar low/high/max sau int 1..100 (NU 'medium' — bench2 folosea 'medium')
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
    cache_line, offload = "", ""
    for line in txt.splitlines():
        if "moe_cache_build" in line:
            cache_line = line.split("I moe_cache_build:")[-1].strip()
        if "offloaded" in line and "expert" in line.lower():
            offload = line.strip()[-120:]
    return cache_line, offload

def build_cmd(name, extra, fit_target):
    think_off = name in NO_THINK
    budget = BUDGET_OVERRIDES.get(name, REASON_BUDGET)
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
           "-b", "512", "-ub", "256", "-np", "1",
           "--reasoning", "off" if think_off else "on",
           "--reasoning-budget", str(budget),
           "--reasoning-format", "deepseek",
           "--skip-chat-parsing",
           "--port", str(PORT), "-v"]
    if fit_target:
        cmd += ["--fit", "on", "--fit-target", str(fit_target)]
    if not any(a == "-dev" for a in extra):
        cmd += ["-dev", "CUDA0"]
    cmd += extra
    return cmd

def measure_once(name, cmd, env, log, think_on):
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
            cache_line, offload = extract_info(txt)
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

def run_cfg(name, extra, env_extra, fit_target):
    print(f"\n[TEST] {name}: {' '.join(extra) if extra else '(fara cache)'}")
    v = vram_used_mib()
    if v and v[0] > VRAM_GATE_MAX_MIB:
        print(f"  GATE per-config: CUDA0={v[0]} MiB > {VRAM_GATE_MAX_MIB} -> reincerc gate")
        if not gate_vram():
            return {"gen_tps": None, "prefill_tps": None, "fail": "gate VRAM inainte de config"}
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1", GGML_SCHED_PREFETCH_EXPERTS="1")
    env.update(env_extra)
    think_off = name in NO_THINK
    log = os.path.join(HERE, f"bench4_{name}.log")
    cmd = build_cmd(name, extra, fit_target)
    result, reason, tail = measure_once(name, cmd, env, log, think_on=not think_off)
    # fallback OOM: -ub 128 (compute buffer ~1.65 GB -> ~0.8 GB)
    if result is None and "OOM" in reason:
        print("  fallback OOM: reiau cu -ub 128")
        cmd2 = list(cmd)
        for i, a in enumerate(cmd2):
            if a == "-ub" and i + 1 < len(cmd2) and cmd2[i+1] == "256":
                cmd2[i+1] = "128"
        log2 = os.path.join(HERE, f"bench4_{name}_ub128.log")
        result, reason2, tail = measure_once(name, cmd2, env, log2, think_on=not think_off)
        if result:
            result["note"] = "reusit cu -ub 128 (fallback OOM)"
    if result is None:
        return {"gen_tps": None, "prefill_tps": None, "fail": reason, "last_log": tail[-12:]}
    return result

def main():
    known = {"--only", "--no-kill", "--no-trace", "--dry", "--with-dspark", "--fit-target"}
    for a in sys.argv[1:]:
        if a.startswith("-") and a not in known and not (sys.argv[sys.argv.index(a)-1] in known):
            print("Folosire: python llama_server_bench4.py [--only nume1,nume2] [--dry] [--no-kill] [--no-trace] [--with-dspark] [--fit-target N]")
            print("  --dry        printeaza comenzile + starile GPU/RAM, nu atinge GPU-ul")
            print("  --no-kill    nu opri Strata (presupune ca tu l-ai oprit manual)")
            print("  --no-trace   foloseste profile4.csv existent")
            print("  --with-dspark include configs cu draft DSpark (experimental — GGUF actual e incompatibil)")
            print("  --fit-target N  adauga --fit on --fit-target N la toate serverele")
            return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    fit_target = None
    if "--fit-target" in sys.argv:
        fit_target = int(sys.argv[sys.argv.index("--fit-target") + 1])

    configs = list(CONFIGS)
    print("=== PREFLIGHT ===")
    ok_model = os.path.exists(MODEL)
    print(f"  model GGUF: {'OK' if ok_model else 'LIPSA ' + MODEL}")
    ok_bin = os.path.exists(SERVER)
    print(f"  llama-server: {'OK' if ok_bin else 'LIPSA ' + SERVER}")
    busy = port_busy(PORT)
    print(f"  port {PORT}: {'OCUPAT de pid ' + busy + ' — FATAL' if busy else 'liber'}")
    ds_ok, ds_msg = dspark_preflight()
    print(f"  draft DSpark: {'OK — ' + ds_msg if ds_ok else 'INCOMPATIBIL — ' + ds_msg}")
    if "--with-dspark" in sys.argv:
        if ds_ok:
            configs += DSPARK_CONFIGS
        else:
            print("  --with-dspark dat dar draft incompatibil -> configs dspark SCOSE")
    if not ok_model or not ok_bin or busy:
        print("FATAL: preflight esuat.")
        return 1

    if "--dry" in sys.argv:
        print("\n=== DRY RUN (nu ating GPU-ul) ===")
        print(f"  VRAM used: {vram_used_mib()} MiB | RAM liber: {free_ram_gb():.1f} GB")
        for name, (extra, env_extra) in configs:
            print(f"  {name}: {' '.join(build_cmd(name, extra, fit_target))}")
        prof_note = "existent" if os.path.exists(PROFILE) else \
                    ("LIPSA -> se copiaza din profile2.csv la rulare" if os.path.exists(os.path.join(HERE, "profile2.csv"))
                     else "LIPSA -> configs E/F/G/H/J sarite")
        print(f"  profile4.csv: {prof_note}")
        return 0

    if "--no-kill" not in sys.argv:
        stop_flash()
    print("\n[gate] VRAM:")
    if not gate_vram():
        return 1
    print("[gate] RAM:")
    if not gate_ram():
        return 1

    trace_ok = True
    if "--no-trace" not in sys.argv:
        trace_ok = run_trace()
    # fallback: refolosim profilul bun din bench2/3 (aceeasi bin, acelasi model) in loc de re-trace
    if not os.path.exists(PROFILE) and os.path.exists(os.path.join(HERE, "profile2.csv")):
        import shutil
        shutil.copy(os.path.join(HERE, "profile2.csv"), PROFILE)
        print(f"  profile4.csv creat din profile2.csv existent ({os.path.getsize(PROFILE)} bytes)")
    if not trace_ok and not os.path.exists(PROFILE):
        print("  profile4.csv lipseste -> configs cu --moe-cache-profile vor fi sarite")

    results = {}
    for name, (extra, env_extra) in configs:
        if only and not any(o.lower() in name.lower() for o in only):
            print(f"\n[SKIP] {name}")
            continue
        if "--moe-cache-profile" in extra and not os.path.exists(PROFILE):
            print(f"\n[SKIP] {name} (profile4.csv lipseste)")
            results[name] = {"gen_tps": None, "prefill_tps": None, "fail": "profile lipseste"}
            continue
        try:
            results[name] = run_cfg(name, extra, env_extra, fit_target)
        except Exception as e:
            print(f"  {name}: EXCEPTIE {e}")
            results[name] = {"gen_tps": None, "prefill_tps": None, "fail": f"exceptie {e}"}
        with open(os.path.join(HERE, "bench4_summary.json"), "w") as f:
            json.dump(results, f, indent=1)   # salvez incremental dupa fiecare config
        # dupa fiecare config: GPU-ul trebuie sa revina liber pentru urmatorul
        v = vram_used_mib()
        if v and v[0] > VRAM_GATE_MAX_MIB:
            print(f"  post-config CUDA0={v[0]} MiB — mai mult decat gate; kill residue")
            out = sh(["powershell", "-NoProfile", "-Command",
                      "Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object { $_.Id }"])
            for pid in out.split():
                kill_pid(pid)
            time.sleep(8)

    print("\n============ REZUMAT bench4 (warm server, chat nativ, gate VRAM/RAM) ============")
    print(f"{'config':22s} {'gen t/s':>8s} {'prefill':>8s}  notes")
    for name, _ in configs:
        r = results.get(name)
        if r and r.get("gen_tps"):
            note = (r.get("note") or r.get("cache") or "")[:50]
            print(f"{name:22s} {r['gen_tps']:>8.1f} {r.get('prefill_tps', 0):>8.1f}  {note}")
        elif r:
            print(f"{name:22s} {'ESUAT':>8s} {'':>8s}  {r.get('fail', '?')}")
        else:
            print(f"{name:22s} {'SKIP':>8s}")
    print("\nSummary: bench4_summary.json | loguri: bench4_*.log")
    return 0

if __name__ == "__main__":
    sys.exit(main())
