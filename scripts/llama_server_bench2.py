#!/usr/bin/env python3
"""
llama_server_bench2.py — runda 2, completă. Reguli noi față de runda 1:
  1. chat template NATIV (jinja din GGUF) + --reasoning on + budget 768.
     (runda 1: --no-jinja + chatml => output gunoi => t/s comparate pe gunoi)
  2. /v1/chat/completions (nu /completion raw)
  3. profil real din 3 prompturi (code/chat/long) mergute in profile.csv
  4. DSpark draft (Q8_0, Markov head) pe 3080 (-devd CUDA1) — nu pe 5080 lang model
  5. masura si PREFILL pe prompt lung (~1500 tokeni), nu doar decode
  6. extrage din log: moe_cache_build (sloturi reale), accept rate speculative

Ordinea: stop Flash-Next -> trace (profil) -> pentru fiecare config: server,
health, warmup, 3x decode cronometrat, 1x prefill lung -> sumar.

Rulezi:  RUN-BENCH2.bat   (sau: python llama_server_bench2.py [--only nume1,nume2])
"""
import os, sys, json, time, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build-dsv41", "bin")
SERVER = os.path.join(BIN, "llama-server.exe")
TRACE = os.path.join(BIN, "llama-moe-trace.exe")
MODEL = os.environ.get("DSV41_MODEL", "CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")
DSPARK = os.environ.get("DSV41_DRAFT", "CHANGE_ME/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf")
PROFILE = os.path.join(HERE, "profile2.csv")
PORT = 8090
N_PRED = 1536
REASON_BUDGET = 768

SHORT_PROMPT = ("Write a Python function solve_microstrip_width(z_target, h_mm, er, t_oz) that finds the "
                "microstrip trace width for a target 50 Ohm impedance using Hammerstad-Jensen equations. "
                "Explain the role of the effective dielectric constant.")
LONG_PROMPT = None  # construit mai jos

def long_prompt():
    global LONG_PROMPT
    if LONG_PROMPT is None:
        base = ("Signal integrity review notes: impedance, reflections, termination, via stubs, "
                "return loss, insertion loss, skew, PDN resonance. ")
        LONG_PROMPT = base * 120  # ~1500-1900 tokeni
    return LONG_PROMPT

# ---- matricea de configurari ----
# (nume, extra-args-server, env-extra)
def C(extra, env=None): return (extra, env or {})

CONFIGS = [
    ("A_baseline",     C([], {"GGML_CUDA_REGISTER_HOST": "1"})),
    ("B_dyn16",        C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("C_dyn16_mask2",  C(["--moe-cache-dyn", "32", "--moe-cache-mask", "2", "--moe-cache-slots", "16"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("D_dyn_auto",     C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto", "--moe-cache-margin", "768"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("E_profile16",    C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16",
                          "--moe-cache-dyn", "32"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("F_profile16_nodynam", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("G_profile24",    C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "24",
                          "--moe-cache-dyn", "32"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("H_profile16_borrow", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16",
                              "--moe-cache-dyn", "32", "--moe-cache-borrow", "1024",
                              "-b", "2048", "-ub", "1024"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("I_dspark16",     C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                          "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("J_dspark16_mask2", C(["--moe-cache-dyn", "32", "--moe-cache-mask", "2", "--moe-cache-slots", "16",
                          "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("K_dspark_profile16", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16",
                          "--moe-cache-dyn", "32",
                          "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("L_dual16",       C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                          "-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "3"})),
    ("M_dual16_dspark", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                          "-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18",
                          "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "3"})),
    ("N_dyn16_nothink", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("O_profile16_decay7", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "16",
                          "--moe-cache-dyn", "32", "--moe-cache-decay", "0.7"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("P_dspark_nmax3", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                          "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", "3",
                          "--spec-draft-p-min", "0.3"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("Q_dual16_ts3414", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                          "-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "34,14"],
                         {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "3"})),

    # ---- runda 3: 15 configurari extra orientate pe viteza ----
    ("R_dyn_slotsauto", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto",
                           "--moe-cache-margin", "1024"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("S_dyn16_mask1",   C(["--moe-cache-dyn", "32", "--moe-cache-mask", "1", "--moe-cache-slots", "16"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("T_dyn16_decay5",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "--moe-cache-decay", "0.5"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("U_dyn16_borrow256", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "--moe-cache-borrow", "256", "-b", "1024", "-ub", "512"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("V_dyn16_kvq8",    C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "-ctk", "q8_0", "-ctv", "q8_0"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("W_dyn16_ncmoe0",  C(["--moe-cache-dyn", "32", "-ncmoe", "0",
                           "--moe-cache-slots", "auto", "--moe-cache-margin", "1024"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("X_ngram_cache",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "--spec-type", "ngram-cache", "--spec-draft-n-max", "4"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("Y_ngram_map_k4v", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "--spec-type", "ngram-map-k4v", "--spec-draft-n-max", "4",
                           "--spec-ngram-map-k4v-size-n", "12", "--spec-ngram-map-k4v-size-m", "48"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("Z_dspark_nmax4",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                           "--spec-type", "draft-dspark", "--spec-draft-n-max", "4",
                           "--spec-draft-n-min", "2", "--spec-draft-p-min", "0.45"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("AA_dspark_kvq8",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                           "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
                           "-ctk", "q8_0", "-ctv", "q8_0"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("AB_dspark_ncmoed", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99", "-ncmoed", "99",
                           "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("AC_dual16_prefetch3", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                               "-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "32,16"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "3",
                           "GGML_CUDA_REGISTER_HOST_CUDA1": "1"})),
    ("AD_ncmoe24",      C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16", "-ncmoe", "24"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("AE_dyn16_poll",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                           "-t", "8", "-tb", "8", "--poll", "100", "--prio", "2",
                           "-Cr", "0-7", "--cpu-strict", "1"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
    ("AF_dspark_nothink", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "16",
                             "-md", DSPARK, "-devd", "CUDA1", "-ngld", "99",
                             "--spec-type", "draft-dspark", "--spec-draft-n-max", "2"],
                          {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"})),
]

def free_ram_gb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"],
                         capture_output=True, text=True).stdout.strip()
    try: return float(out.replace(",", "."))
    except ValueError: return -1.0

def pids_on_port(port):
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pids.add(line.split()[-1])
    return pids

def kill_pid(pid):
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)

def stop_flash():
    print("[1] Opresc Flash-Next (8080/8081) + llama-server ramase...")
    for port in (8080, 8081, 8082):
        for pid in pids_on_port(port):
            cl = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 f"(Get-Process -Id {pid}).ProcessName"],
                                capture_output=True, text=True).stdout
            if "python" in cl.lower() or "llama" in cl.lower() or "strata" in cl.lower():
                print(f"  kill port {port}: pid {pid} ({cl.strip()})")
                kill_pid(pid)
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "llama-server" in line:
            kill_pid(line.split()[-1])
    print("  astept 10 s eliberare RAM/VRAM...")
    time.sleep(10)
    print(f"  RAM liber acum: {free_ram_gb():.1f} GB")

def run_trace():
    """Profil real: 3 prompturi (code / chat / SI-lecture), mergute in profile2.csv."""
    print("\n[2] Profil routing (llama-moe-trace, 3 prompturi)...")
    prompts = [
        ("code", "Write a Python function that simulates a transactional key-value store with savepoints and deadlock detection."),
        ("chat", "Explain the difference between characteristic impedance and propagation delay in a PCB trace."),
        ("si",   "A 50 Ohm microstrip on FR4 (er=4.4, h=1.6 mm) needs about 2.8 mm width. Explain briefly."),
    ]
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1")
    parts = []
    for tag, p in prompts:
        out_csv = os.path.join(HERE, f"profile_{tag}.csv")
        log = os.path.join(HERE, f"bench_trace_{tag}.log")
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
            print(f"  profil {tag}: LIPSA (vezi bench_trace_{tag}.log)")
    if parts:
        with open(PROFILE, "w") as out:
            for i, f in enumerate(parts):
                with open(f) as fh:
                    lines = fh.readlines()
                    if i == 0 or not lines:
                        out.writelines(lines)
                    else:
                        out.writelines(lines[1:])  # sari header
        print(f"  => {PROFILE} ({os.path.getsize(PROFILE)} bytes)")

def chat_body(prompt, n_predict):
    return json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.7,
        "reasoning_effort": "medium",
    }).encode()

def post_chat(prompt, n_predict, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=chat_body(prompt, n_predict),
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

def run_cfg(name, extra, env_extra):
    print(f"\n[TEST] {name}: {' '.join(extra) if extra else '(fara cache)'}")
    log = os.path.join(HERE, f"bench2_{name}.log")
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1")
    env.update(env_extra)
    think_off = name in ("N_dyn16_nothink", "AF_dspark_nothink")
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "8192", "-fa", "on",
           "-b", "512", "-ub", "256", "-np", "1",
           "--reasoning", "off" if think_off else "on",
           "--reasoning-budget", str(REASON_BUDGET),
           "--reasoning-format", "deepseek",
           "--port", str(PORT), "-v"]
    if not any(a == "-dev" for a in extra):
        cmd += ["-dev", "CUDA0"]
    cmd += extra
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_ready(proc):
                print("  server NU a ajuns ready -> skip")
                return None
            # warmup
            try:
                post_chat("Hi.", 8)
            except Exception as e:
                print(f"  warmup esuat: {e}")
            gens, prompts_t, samples = [], [], ""
            for i in range(3):
                try:
                    dt, ctok, ptok, content, reasoning, tim = post_chat(SHORT_PROMPT, N_PRED)
                except Exception as e:
                    print(f"  run {i+1} a crapat: {e}")
                    break
                if ctok:
                    # timings de pe server (predictive decode real, fara prefill in medie)
                    g = tim.get("predicted_per_second") or (ctok / dt if dt else 0)
                    gens.append(g)
                    prompts_t.append(tim.get("prompt_per_second") or 0)
                    print(f"  run {i+1}: {ctok} tok in {dt:.1f}s = {g:.1f} t/s server "
                          f"(wall {ctok/dt:.1f}) (think {len(reasoning)} chars)")
                    samples = content[:200]
            # prefill lung (masuram prompt_per_second: n_predict mic)
            prefill_tps = 0.0
            try:
                dt, ctok, ptok, _, _, tim = post_chat(long_prompt(), 16)
                if dt > 0:
                    prefill_tps = tim.get("prompt_per_second") or (ptok / dt)
                    print(f"  prefill: {ptok} tok prompt in {dt:.1f}s = {prefill_tps:.1f} t/s")
            except Exception as e:
                print(f"  prefill esuat: {e}")
            if not gens:
                return None
            # extras din log: sloturi reale cache + accept rate spec
            cache_line, accept = "", ""
            try:
                with open(log, "rb") as f:
                    txt = f.read().decode("utf-8", "ignore")
                for line in txt.splitlines():
                    if "moe_cache_build" in line:
                        cache_line = line.split("I moe_cache_build:")[-1].strip()
                    if "speculative" in line.lower() and "accept" in line.lower():
                        accept = line.strip()[-120:]
            except Exception:
                pass
            return {"gen_tps": round(sum(gens)/len(gens), 2),
                    "prefill_tps": round(prefill_tps, 2),
                    "cache": cache_line, "spec_accept": accept,
                    "sample": samples[:160]}
        finally:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: kill_pid(proc.pid)
            time.sleep(6)

def main():
    known = {"--only", "--no-kill", "--no-trace"}
    for a in sys.argv[1:]:
        if a.startswith("-") and a not in known and not (sys.argv[sys.argv.index(a)-1] in known):
            print("Folosire: python llama_server_bench2.py [--only nume1,nume2] [--no-kill] [--no-trace]")
            print("  --only   ruleaza doar configurile care contin unul din nume (ex: --only dspark)")
            print("  --no-kill nu opri Flash-Next/strata inainte (daca rulezi manual cu GPU-ul liber)")
            print("  --no-trace sare peste profilarea moe-trace (foloseste profile2.csv existent)")
            return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--no-kill" not in sys.argv:
        stop_flash()
    if "--no-trace" not in sys.argv:
        run_trace()
    results = {}
    for name, (extra, env_extra) in CONFIGS:
        if only and not any(o.lower() in name.lower() for o in only):
            print(f"\n[SKIP] {name}")
            continue
        try:
            results[name] = run_cfg(name, extra, env_extra)
        except Exception as e:
            print(f"  {name}: EXCEPTIE {e}")
            results[name] = None
        with open(os.path.join(HERE, "bench2_summary.json"), "w") as f:
            json.dump(results, f, indent=1)  # salvezi incremental
    print("\n================ REZUMAT runda 2 (warm server, chat nativ, think budget 768) ===============")
    print(f"{'config':22s} {'gen t/s':>8s} {'prefill t/s':>11s}  cache")
    for name, _ in CONFIGS:
        r = results.get(name)
        if r:
            print(f"{name:22s} {r['gen_tps']:>8.1f} {r['prefill_tps']:>11.1f}  {r['cache'][:60]}")
        else:
            print(f"{name:22s} {'ESUAT':>8s}")
    print("\nSummary: bench2_summary.json | loguri: bench2_*.log")

if __name__ == "__main__":
    main()
