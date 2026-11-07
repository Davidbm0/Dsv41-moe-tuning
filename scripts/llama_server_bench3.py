#!/usr/bin/env python3
"""
llama_server_bench3.py — runda 3, ~30 configs, pe binarul REBUILD-UIT (00:23, jinja patch).
Schimbari fata de bench2:
  1. salvam ULTIMELE 25 linii din log cand serverul pica -> de ce a dat null
  2. extract "moe cache slots" / "offloaded N/M" din log pentru orice arm
  3. guard RAM liber inainte de fiecare arm (modelul e 266 GB, mmap => page cache)
  4. --smoke = ruleaza doar A_baseline (verificare ca masuratorile apar real)
  5. profilul moe-trace se ruleaza o singura data, inainte de batch (--no-trace il sare)

Rulezi:  RUN-BENCH3.bat  sau  python llama_server_bench3.py [--smoke] [--only sub1,sub2] [--no-trace]
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
LONG_PROMPT = None

def long_prompt():
    global LONG_PROMPT
    if LONG_PROMPT is None:
        base = ("Signal integrity review notes: impedance, reflections, termination, via stubs, "
                "return loss, insertion loss, skew, PDN resonance. ")
        LONG_PROMPT = base * 120
    return LONG_PROMPT

def C(extra, env=None): return (extra, env or {})

# ---- 31 configurari ----
DS = ["-md", DSPARK, "-devd", "CUDA1", "-ngld", "99", "--spec-type", "draft-dspark"]
DUAL = ["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18"]
E1 = {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "6"}
E3 = {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "3"}

CONFIGS = [
    # G1 — cache dinamic / slots
    ("A_baseline",      C(["--moe-cache-dyn", "32"], E1)),
    ("B_dyn16",         C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto", "--moe-cache-margin", "3072"], E1)),
    ("C_dyn16_mask2",   C(["--moe-cache-dyn", "32", "--moe-cache-mask", "2", "--moe-cache-slots", "12"], E1)),
    ("D_dyn16_mask1",   C(["--moe-cache-dyn", "32", "--moe-cache-mask", "1", "--moe-cache-slots", "12"], E1)),
    ("E_dyn_slotsauto", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "auto", "--moe-cache-margin", "3072"], E1)),
    ("F_dyn16_decay5",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12", "--moe-cache-decay", "0.5"], E1)),
    ("G_dyn16_borrow",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                           "--moe-cache-borrow", "256", "-b", "1024", "-ub", "512"], E1)),
    ("H_ncmoe24",       C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12", "-ncmoe", "24"], E1)),
    ("I_ncmoe0_auto",   C(["--moe-cache-dyn", "32", "-ncmoe", "0",
                           "--moe-cache-slots", "auto", "--moe-cache-margin", "3072"], E1)),
    # G2 — profil routing (profile2.csv din moe-trace)
    ("J_profile16",       C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12", "--moe-cache-dyn", "32"], E1)),
    ("K_profile16_nodyn", C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12"], E1)),
    ("L_profile24",       C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "24", "--moe-cache-dyn", "32"], E1)),
    ("M_profile16_decay7",C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12",
                             "--moe-cache-dyn", "32", "--moe-cache-decay", "0.7"], E1)),
    ("N_profile16_borrow",C(["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12",
                             "--moe-cache-dyn", "32", "--moe-cache-borrow", "1024", "-b", "2048", "-ub", "1024"], E1)),
    # G3 — KV quant
    ("O_dyn16_kvq8",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12", "-ctk", "q8_0", "-ctv", "q8_0"], E1)),
    ("P_dspark_kvq8", C(DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                              "--spec-draft-n-max", "2", "-ctk", "q8_0", "-ctv", "q8_0"], E1)),
    # G4 — dspark (draft pe 3080)
    ("Q_dspark16",        C(DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12", "--spec-draft-n-max", "2"], E1)),
    ("R_dspark16_mask2",  C(DS + ["--moe-cache-dyn", "32", "--moe-cache-mask", "2", "--moe-cache-slots", "12",
                                  "--spec-draft-n-max", "2"], E1)),
    ("S_dspark_profile16",C(DS + ["--moe-cache-profile", PROFILE, "--moe-cache-slots", "12",
                                  "--moe-cache-dyn", "32", "--spec-draft-n-max", "2"], E1)),
    ("T_dspark_nmax3",    C(DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                                  "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.3"], E1)),
    ("U_dspark_nmax4",    C(DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                                  "--spec-draft-n-max", "4", "--spec-draft-n-min", "2", "--spec-draft-p-min", "0.45"], E1)),
    ("V_dspark_ncmoed99", C(DS + ["-ncmoed", "99", "--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                                  "--spec-draft-n-max", "2"], E1)),
    ("W_dspark_nothink",  C(DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12", "--spec-draft-n-max", "2"], E1)),
    # G5 — ngram (fara draft model)
    ("X_ngram_cache",   C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                           "--spec-type", "ngram-cache", "--spec-draft-n-max", "4"], E1)),
    ("Y_ngram_map_k4v", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                           "--spec-type", "ngram-map-k4v", "--spec-draft-n-max", "4",
                           "--spec-ngram-map-k4v-size-n", "12", "--spec-ngram-map-k4v-size-m", "48"], E1)),
    # G6 — dual GPU (layer split 5080+3080)
    ("Z_dual16",          C(DUAL + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12"], E3)),
    ("AA_dual16_ts3414",  C(["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "34,14",
                             "--moe-cache-dyn", "32", "--moe-cache-slots", "12"], E3)),
    ("AB_dual16_dspark",  C(DUAL + DS + ["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                                         "--spec-draft-n-max", "2"], E3)),
    # G7 — CPU / scheduling / reasoning
    ("AC_dyn16_poll",  C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12",
                          "-t", "8", "-tb", "8", "--poll", "100", "--prio", "2",
                          "-Cr", "0-7", "--cpu-strict", "1"], E1)),
    ("AD_dyn16_nothink", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12"], E1)),
    ("AE_dyn16_budget1536", C(["--moe-cache-dyn", "32", "--moe-cache-slots", "12"], E1)),
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
    print("[1] Opresc Flash-Next (8080/8081/8082) + llama-server ramase...")
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
    # strata.exe/strata-vision.exe nu ascultă pe port — trebuie omorâți pe nume
    for img in ("strata.exe", "strata-vision.exe"):
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {img}"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if img.replace(".exe", "") in line.lower():
                parts = line.split()
                if parts[-1].isdigit():
                    print(f"  kill {img}: pid {parts[-1]}")
                    kill_pid(parts[-1])
    print("  astept 10 s eliberare RAM/VRAM...")
    time.sleep(10)
    print(f"  RAM liber acum: {free_ram_gb():.1f} GB")

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
        out_csv = os.path.join(HERE, f"profile_{tag}.csv")
        log = os.path.join(HERE, f"bench3_trace_{tag}.log")
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
            print(f"  profil {tag}: LIPSA (vezi bench3_trace_{tag}.log)")
    if parts:
        with open(PROFILE, "w") as out:
            for i, f in enumerate(parts):
                with open(f) as fh:
                    lines = fh.readlines()
                    if i == 0 or not lines:
                        out.writelines(lines)
                    else:
                        out.writelines(lines[1:])
        print(f"  => {PROFILE} ({os.path.getsize(PROFILE)} bytes)")

def chat_body(prompt, n_predict, think_on=True, budget=REASON_BUDGET):
    b = {"messages": [{"role": "user", "content": prompt}],
         "max_tokens": n_predict, "temperature": 0.7}
    if think_on:
        # template-ul DSV4.1 accepta doar low/high/max sau int 1..100 (NU 'medium')
        b["reasoning_effort"] = "high"
    return json.dumps(b).encode()

def post_chat(prompt, n_predict, timeout=1800, think_on=True):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=chat_body(prompt, n_predict, think_on),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    u = d.get("usage", {})
    timings = d.get("timings", {})
    # numaram TOT tokenurile generate (reasoning inclus) — altfel masuram 14 tok din 1550
    n_gen = max(u.get("completion_tokens", 0) or 0,
                int(timings.get("predicted", 0) or 0),
                len((d["choices"][0]["message"].get("reasoning_content") or "").split()) * 1)
    content = d["choices"][0]["message"].get("content", "") or ""
    reasoning = d["choices"][0]["message"].get("reasoning_content", "") or ""
    return dt, n_gen, u.get("prompt_tokens", 0), content, reasoning, timings

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
    cache_line, offload, slots = "", "", ""
    for line in txt.splitlines():
        if "moe_cache_build" in line:
            cache_line = line.split("I moe_cache_build:")[-1].strip()
        if "offloaded" in line and "expert" in line.lower():
            offload = line.strip()[-120:]
        if "moe cache" in line.lower() and "slot" in line.lower():
            slots = line.strip()[-120:]
    return cache_line, offload, slots

def unload_strata():
    # Strata renaste automat la fiecare meu rand de agent -> il descarcam inainte de fiecare arm
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/unload",
                                     data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        time.sleep(4)
    except Exception:
        pass

def run_cfg(name, extra, env_extra):
    print(f"\n[TEST] {name}: {' '.join(extra) if extra else '(fara cache)'}")
    unload_strata()
    ram = free_ram_gb()
    if ram < 20.0:
        print(f"  GUARD: doar {ram:.1f} GB RAM liber (<20) -> SKIP (mmap needs page-cache headroom)")
        return None
    log = os.path.join(HERE, f"bench3_{name}.log")
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1")
    env.update(env_extra)
    think_off = name in ("AD_dyn16_nothink", "W_dspark_nothink")
    budget = 1536 if name == "AE_dyn16_budget1536" else REASON_BUDGET
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
           "-b", "512", "-ub", "256", "-np", "1",
           "--reasoning", "off" if think_off else "on",
           "--reasoning-budget", str(budget),
           "--reasoning-format", "deepseek",
           "--skip-chat-parsing",
           "--port", str(PORT), "-v"]
    if not any(a == "-dev" for a in extra):
        cmd += ["-dev", "CUDA0"]
    cmd += extra
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_ready(proc):
                code = proc.poll()
                print(f"  server NU a ajuns ready (exit={code}) -> ultimele linii din log:")
                for l in tail_log(log, 12):
                    print(f"    | {l[:150]}")
                return {"gen_tps": None, "prefill_tps": None, "fail": f"server exit {code}",
                        "last_log": tail_log(log, 25)}
            try:
                post_chat("Hi.", 8, think_on=not think_off)
            except Exception as e:
                print(f"  warmup esuat: {e}")
            gens, prompts_t, samples = [], [], ""
            for i in range(3):
                try:
                    dt, ctok, ptok, content, reasoning, tim = post_chat(
                        SHORT_PROMPT, N_PRED, think_on=not think_off)
                except Exception as e:
                    print(f"  run {i+1} a crapat: {e}")
                    break
                if ctok >= 50:
                    g = tim.get("predicted_per_second") or (ctok / dt if dt else 0)
                    gens.append(g)
                    prompts_t.append(tim.get("prompt_per_second") or 0)
                    print(f"  run {i+1}: {ctok} tok in {dt:.1f}s = {g:.1f} t/s server "
                          f"(think {len(reasoning)} chars)")
                    samples = content[:200]
                    if i == 0 and dt > 420 and ctok < 100:
                        print("  ABORT: arm patologic (prea lent, aruncă 25 min degeaba)")
                        cl = extract_info(open(log, 'rb').read().decode('utf-8', 'ignore'))[0]
                        return {"gen_tps": round(g, 2), "prefill_tps": 0,
                                "fail": f"pathological: {ctok} tok in {dt:.0f}s",
                                "cache": cl}
                else:
                    print(f"  run {i+1}: doar {ctok} tok in {dt:.1f}s -> aruncat (nu e gen real)")
            prefill_tps = 0.0
            try:
                dt, ctok, ptok, _, _, tim = post_chat(long_prompt(), 16, think_on=not think_off)
                if dt > 0:
                    prefill_tps = tim.get("prompt_per_second") or (ptok / dt)
                    print(f"  prefill: {ptok} tok prompt in {dt:.1f}s = {prefill_tps:.1f} t/s")
            except Exception as e:
                print(f"  prefill esuat: {e}")
            cache_line, offload, slots = "", "", ""
            try:
                with open(log, "rb") as f:
                    txt = f.read().decode("utf-8", "ignore")
                cache_line, offload, slots = extract_info(txt)
            except Exception:
                pass
            if not gens:
                print("  NU avem niciun gen masurat -> ultimele linii:")
                for l in tail_log(log, 12):
                    print(f"    | {l[:150]}")
                return {"gen_tps": None, "prefill_tps": None, "fail": "no gen measured",
                        "last_log": tail_log(log, 25)}
            return {"gen_tps": round(sum(gens)/len(gens), 2),
                    "prefill_tps": round(prefill_tps, 2),
                    "ram_gb": round(ram, 1),
                    "cache": cache_line, "offload": offload, "slots": slots,
                    "sample": samples[:160]}
        finally:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: kill_pid(proc.pid)
            time.sleep(6)

def main():
    smoke = "--smoke" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--no-kill" not in sys.argv:
        stop_flash()
    if "--no-trace" not in sys.argv and not smoke:
        run_trace()
    results = {}
    for name, (extra, env_extra) in CONFIGS:
        if smoke and name != "A_baseline":
            continue
        if only and not any(o.lower() in name.lower() for o in only):
            print(f"\n[SKIP] {name}")
            continue
        try:
            results[name] = run_cfg(name, extra, env_extra)
        except Exception as e:
            print(f"  {name}: EXCEPTIE {e}")
            results[name] = None
        with open(os.path.join(HERE, "bench3_summary.json"), "w") as f:
            prev = {}
            try:
                with open(os.path.join(HERE, "bench3_summary.json")) as g:
                    prev = json.load(g)
            except Exception:
                pass
            prev.update(results)
            json.dump(prev, f, indent=1)
    print("\n================ REZUMAT runda 3 (binar rebuild, chat nativ) ===============")
    print(f"{'config':22s} {'gen t/s':>8s} {'prefill t/s':>11s}  cache")
    for name, _ in CONFIGS:
        r = results.get(name)
        if r and r.get("gen_tps"):
            print(f"{name:22s} {r['gen_tps']:>8.1f} {r['prefill_tps']:>11.1f}  "
                  f"{(r.get('cache') or r.get('slots') or '')[:60]}")
        elif r:
            print(f"{name:22s} {'ESUAT':>8s}  ({r.get('fail','?')})")
    print("\nSummary: bench3_summary.json | loguri: bench3_*.log")

if __name__ == "__main__":
    main()
