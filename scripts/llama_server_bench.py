#!/usr/bin/env python3
"""
llama_server_bench.py — test CORECT (warm server), ordinea impusa:
  1. opreste Flash-Next (8080) + llama ramase, asteapta eliberare RAM
  2. llama-moe-trace -> profile.csv (profil de routing, o data)
  3. pentru fiecare config: llama-server pe 8090, health poll, warmup,
     3 cereri cronometrate /completion, media t/s; log -v ca sa vedem
     "moe_cache_build" (cache activ?) 
  4. sumar in bench_server_summary.txt

De ce server si nu cli: README fork: "Benchmark on a warm server... not one
request per fresh start." + llama-cli cu -p intra in chat interactiv si atarna.
"""
import os, sys, json, time, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build-dsv41", "bin")
SERVER = os.path.join(BIN, "llama-server.exe")
TRACE = os.path.join(BIN, "llama-moe-trace.exe")
MODEL = os.environ.get("DSV41_MODEL", "CHANGE_ME/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")
PROFILE = os.path.join(HERE, "profile.csv")
PORT = 8090
N_PRED = 128
PROMPT = ("A 50 Ohm microstrip on FR4 (er=4.4, h=1.6 mm) needs a width of about 2.8 mm. "
          "Explain briefly how characteristic impedance depends on trace width and substrate height.")

CONFIGS = [
    ("I_slots8",       ["--moe-cache-slots", "8", "--moe-cache-margin", "512",
                        "--moe-cache-dyn", "32", "--moe-cache-mask", "2"]),
    ("J_dual16",       ["--moe-cache-slots", "16", "--moe-cache-margin", "512",
                        "--moe-cache-dyn", "32", "--moe-cache-mask", "2",
                        "-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "30,18"]),
    ("K_profile8",     ["--moe-cache-profile", PROFILE, "--moe-cache-slots", "8",
                        "--moe-cache-margin", "512", "--moe-cache-dyn", "32"]),
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
    print("[1] Opresc Flash-Next (8080/8081) + llama ramase...")
    for port in (8080, 8081):
        for pid in pids_on_port(port):
            cl = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 f"(Get-Process -Id {pid}).ProcessName"],
                                 capture_output=True, text=True).stdout
            if "python" in cl.lower() or "llama" in cl.lower():
                print(f"  kill port {port}: pid {pid} ({cl.strip()})")
                kill_pid(pid)
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "llama-server" in line:
            pid = line.split()[-1]
            print(f"  kill llama-server ramas: pid {pid}")
            kill_pid(pid)
    print("  astept 8 s eliberare RAM/VRAM...")
    time.sleep(8)
    print(f"  RAM liber acum: {free_ram_gb():.1f} GB")

def run_trace():
    print("\n[2] Profil routing (llama-moe-trace, o singura data)...")
    log = os.path.join(HERE, "bench_trace.log")
    env = dict(os.environ, MOE_TRACE_OUT=PROFILE,
               GGML_CUDA_REGISTER_HOST="1", GGML_SCHED_PREFETCH_EXPERTS="1")
    cmd = [TRACE, "-m", MODEL, "-dev", "CUDA0", "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
           "-b", "512", "-ub", "256", "--temp", "0.7", "-n", "512", "-p", PROMPT]
    with open(log, "wb") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1800, env=env)
    print(f"  profil: {PROFILE} ({os.path.getsize(PROFILE) if os.path.exists(PROFILE) else 0} bytes)")

def wait_ready(proc, logf, timeout=900):
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

def completion(n_predict):
    body = json.dumps({"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    tim = d.get("timings", {})
    return tim.get("predicted_per_second", 0.0), tim.get("prompt_per_second", 0.0), dt, d.get("content", "")

def run_cfg(name, extra):
    print(f"\n[TEST] {name}: {' '.join(extra) if extra else '(fara cache)'}")
    log = os.path.join(HERE, f"bench_srv_{name}.log")
    env = dict(os.environ, GGML_CUDA_REGISTER_HOST="1", GGML_SCHED_PREFETCH_EXPERTS="0")
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "99",
           "--load-mode", "mmap", "--lazy-mode", "auto", "-c", "4096", "-fa", "on",
           "-b", "512", "-ub", "256", "-np", "1",
           "--no-jinja", "--chat-template", "chatml", "--port", str(PORT), "-v"]
    if not any(a == "-dev" for a in extra):
        cmd += ["-dev", "CUDA0"]
    cmd += extra
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_ready(proc, lf):
                print("  server NU a ajuns ready -> skip")
                return None
            # warmup scurt
            try:
                completion(8)
            except Exception as e:
                print(f"  warmup esuat: {e}")
            gens, prompts = [], []
            content = ""
            for i in range(3):
                try:
                    g, p, dt, content = completion(N_PRED)
                except Exception as e:
                    print(f"  run {i+1} a crapat: {e}")
                    break
                gens.append(g); prompts.append(p)
                print(f"  run {i+1}: gen {g:.1f} t/s | prompt {p:.1f} t/s")
            if not gens:
                return None
            return {"gen": round(sum(gens)/len(gens), 2), "prompt": round(sum(prompts)/len(prompts), 2),
                    "sample": content[:160]}
        finally:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: kill_pid(proc.pid)
            time.sleep(5)

def main():
    if "--trace-only" in sys.argv:
        stop_flash(); run_trace(); return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    stop_flash()
    run_trace()
    results = {}
    for name, extra in CONFIGS:
        if only and not any(o.lower() in name.lower() for o in only):
            print(f"\n[SKIP] {name}")
            continue
        results[name] = run_cfg(name, extra)
    print("\n================ REZUMAT (warm server) ===================")
    print(f"{'config':24s} {'gen t/s':>8s} {'prompt t/s':>10s}")
    for name, _ in CONFIGS:
        r = results.get(name)
        if r: print(f"{name:24s} {r['gen']:>8.1f} {r['prompt']:>10.1f}")
        else: print(f"{name:24s} {'ESUAT':>8s}")
    with open(os.path.join(HERE, "bench_server_summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    last = results.get(CONFIGS[-1][0]) or {}
    print("\nSample output ultimul run:", str(last.get("sample", "?"))[:120])

if __name__ == "__main__":
    main()
