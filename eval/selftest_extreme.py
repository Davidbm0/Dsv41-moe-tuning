#!/usr/bin/env python3
"""Self-test: asserturile din coding_eval_extreme.py trebuie sa TREACA pe implementarile
de referinta (cele din verify_extreme_engineering.py). Daca un assert pica aici,
assert-ul e gresit, nu modelul."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coding_eval_extreme import build_problems, run_tests

REF = {}

REF["p1_dijkstra_parity"] = '''
import math, heapq, collections
def parity_turn_dijkstra(n, edges, start, dst):
    adj = collections.defaultdict(list)
    for u, v, w, c in edges:
        adj[u].append((v, w, c))
    INF = float("inf")
    dist = {}
    pq = [(0, start, 0, None, 0)]
    while pq:
        cost, u, step, lc, run = heapq.heappop(pq)
        if u == dst:
            return cost
        key = (u, step, lc, run)
        if dist.get(key, INF) < cost:
            continue
        for v, w, c in adj[u]:
            nstep = step + 1
            if lc is not None and c == lc and run >= 2:
                continue
            add = w * (2 if nstep % 2 == 0 else 1)
            if lc is not None and lc != c:
                add += abs(lc - c) * 3
            nrun = run + 1 if lc is not None and c == lc else 1
            nkey = (v, nstep, c, nrun)
            ncost = cost + add
            if ncost < dist.get(nkey, INF):
                dist[nkey] = ncost
                heapq.heappush(pq, (ncost, v, nstep, c, nrun))
    return -1
'''

REF["p2_microstrip"] = '''
import math
def z0_microstrip(w, h_mm, er, t_mm):
    u = w / h_mm
    if t_mm > 0:
        th = t_mm / h_mm
        du1 = th / math.pi * math.log(1.0 + (4.0*math.e) / (th * (1.0/math.tanh(math.sqrt(6.517*u)))**2))
        du_r = 0.5 * du1 * (1.0 + 1.0/math.cosh(math.sqrt(er-1.0)))
    else:
        du1 = du_r = 0.0
    u1, ur = u + du1, u + du_r
    a = 1.0 + (1.0/49.0)*math.log((ur**4 + (ur/52.0)**2)/(ur**4 + 0.432)) + (1.0/18.7)*math.log(1.0 + (ur/18.1)**3)
    b = 0.564 * ((er-0.9)/(er+3.0))**0.053
    eeff = (er+1.0)/2.0 + ((er-1.0)/2.0) * (1.0 + 10.0/ur)**(-a*b)
    f_u = lambda x: 6.0 + (2.0*math.pi - 6.0)*math.exp(-(30.666/x)**0.7528)
    z0fs = (376.730313668/(2.0*math.pi)) * math.log(f_u(u1)/u1 + math.sqrt(1.0 + (2.0/u1)**2))
    return z0fs / math.sqrt(eeff), eeff
def solve_microstrip_width(z_target, h_mm, er, t_oz):
    t_mm = t_oz * 0.034798
    lo, hi = 0.01, 20.0
    for _ in range(200):
        mid = (lo+hi)/2
        z, _ = z0_microstrip(mid, h_mm, er, t_mm)
        if z > z_target: lo = mid
        else: hi = mid
        if hi-lo < 1e-9: break
    w = (lo+hi)/2
    z, eeff = z0_microstrip(w, h_mm, er, t_mm)
    delay = 1e12 * math.sqrt(eeff) / 2.99792458e11
    return {"w_mm": round(w,4), "w_mils": round(w*39.3700787,4),
            "eeff": round(eeff,4), "delay_ps_per_mm": round(delay,4)}
'''

REF["p3_tx_engine"] = '''
import collections
def run_tx_engine(ops):
    out = []
    store = {}
    locks = {}
    active = set()
    uncommitted = {}
    acquired_order = {}
    savepoints = {}
    waiters = collections.defaultdict(collections.deque)
    def release(tx, keys):
        for k in keys:
            if locks.get(k) == tx:
                del locks[k]
                while waiters[k]:
                    process(waiters[k].popleft())
    def cycle(tx_a, tx_b):
        seen = set()
        stack = [tx_b]
        while stack:
            t = stack.pop()
            if t == tx_a: return True
            if t in seen: continue
            seen.add(t)
            for k, wq in waiters.items():
                for _op in wq:
                    if _op[1] == t and locks.get(k) is not None:
                        stack.append(locks.get(k))
        return False
    def process(op):
        kind = op[0]
        if kind == "BEGIN":
            active.add(op[1]); uncommitted.setdefault(op[1], {}); acquired_order.setdefault(op[1], []); savepoints.setdefault(op[1], [])
            out.append("OK"); return True
        if kind == "SET":
            _, tx, key, val = op
            if key not in locks or locks[key] == tx:
                locks[key] = tx
                if key not in acquired_order[tx]: acquired_order[tx].append(key)
                uncommitted[tx][key] = val
                out.append("OK"); return True
            other = locks[key]
            if cycle(tx, other):
                out.append("DEADLOCK_ABORT")
                for k in acquired_order.get(tx, []):
                    if locks.get(k) == tx:
                        del locks[k]
                        while waiters[k]:
                            nxt = waiters[k].popleft()
                            if nxt[1] != tx:
                                process(nxt)
                active.discard(tx)
                uncommitted.pop(tx, None)
                return False
            waiters[key].append(op)
            out.append("WAITING"); return False
        if kind == "SAVEPOINT":
            _, tx, sp = op
            savepoints[tx].append((sp, dict(uncommitted[tx]), set(acquired_order[tx])))
            out.append("OK"); return True
        if kind == "ROLLBACK_TO":
            _, tx, sp = op
            snap = None
            for (name, writes, lockset) in savepoints[tx]:
                if name == sp: snap = (writes, lockset); break
            if snap:
                writes, lockset = snap
                uncommitted[tx] = dict(writes)
                keep = [k for k in acquired_order[tx] if k in lockset]
                released = [k for k in acquired_order[tx] if k not in lockset]
                acquired_order[tx] = keep
                savepoints[tx] = savepoints[tx][: [s[0] for s in savepoints[tx]].index(sp)+1]
                out.append("OK")
                release(tx, released)
                return True
            out.append("OK"); return True
        if kind == "COMMIT":
            _, tx = op
            store.update(uncommitted.get(tx, {}))
            uncommitted[tx] = {}
            active.discard(tx)
            keys = acquired_order.get(tx, []); acquired_order[tx] = []
            out.append("COMMITTED")
            release(tx, keys)
            return True
        if kind == "GET":
            out.append(store.get(op[1])); return True
    for op in ops:
        process(op)
    return out
'''

REF["p4_pdn"] = '''
import math
def analyze_pdn_impedance(caps, f_start, f_stop, num_points):
    freqs = [f_start * (f_stop/f_start)**(k/(num_points-1)) for k in range(num_points)]
    Zs = []
    for f in freqs:
        w = 2*math.pi*f
        Y = sum(1.0/(esr + 1j*(w*esl - 1.0/(w*c))) for (c, esr, esl) in caps)
        Zs.append(abs(1.0/Y))
    peaks = [k for k in range(1, num_points-1) if Zs[k] > Zs[k-1] and Zs[k] > Zs[k+1]]
    best = max(peaks, key=lambda k: Zs[k])
    return {"max_anti_res_freq_mhz": round(freqs[best]/1e6, 2),
            "max_anti_res_z_mohm": round(Zs[best]*1e3, 2),
            "min_z_mohm": round(min(Zs)*1e3, 2)}
'''

REF["p5_ast"] = '''
def optimize_and_execute(ast_code, initial_regs):
    regs = dict(initial_regs)
    def get(x): return regs.get(x, 0) if isinstance(x, str) else x
    def run(stmts):
        for s in stmts:
            op = s["op"]
            if op == "SET": regs[s["reg"]] = get(s["val"])
            elif op in ("ADD","SUB","MUL"):
                a, b = get(s["a"]), get(s["b"])
                regs[s["dst"]] = a+b if op=="ADD" else (a-b if op=="SUB" else a*b)
            elif op == "REPEAT":
                for _ in range(int(get(s["n"]))): run(s["body"])
            elif op == "IF":
                if regs.get(s["reg"], 0) > s["gt"]: run(s["body"])
    run(ast_code["body"])
    return regs
'''

fail = 0
PROBLEMS = build_problems("ref14")
for name, prompt, asserts in PROBLEMS:
    ok, err = run_tests(REF[name], asserts)
    print(f"{'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -> {err[:300]}"))
    fail += not ok
print("SELFTEST:", "totul OK — asserturile harness-ului sunt corecte" if fail == 0 else f"{fail} asserturi problematice")
sys.exit(1 if fail else 0)
