# Prompts + asserturi pentru suita S1-S5 (P3-Class Stateful Systems).
# Asserturile sunt exact cele din document, validate 5/5 de verify_s_suite.py.

S1_PROMPT = '''Write a Python function `simulate_rtos(events: list[tuple]) -> list[str]` that simulates a single-core preemptive RTOS scheduler with TRANSITIVE priority inheritance.

Rules:
1. Lower numerical priority value = HIGHER priority (1 preempts 5).
2. Task states: READY, RUNNING, BLOCKED, COMPLETED.
3. At any moment the CPU executes the READY task with highest effective priority. Tie-break: the task in READY state the longest (FIFO). A RUNNING task is NOT preempted by a newly awakened task of the EXACT SAME effective priority.
4. eff_prio(T) = min(base_prio(T), min(eff_prio(W) for all W waiting on ANY mutex held by T)). Inheritance propagates transitively across chains. On release of mutex M by T: (a) grant M to the waiter on M with highest effective priority (tie: FIFO wait order), waiter becomes READY; (b) recompute all effective priorities, T demoted back.
5. Events, after EACH one append the currently RUNNING task_id (or "IDLE"):
- ("SPAWN", task_id, base_prio): task enters READY.
- ("LOCK", mutex_id): RUNNING task requests mutex. Free -> acquire, keep running. Held -> RUNNING task becomes BLOCKED on it, inheritance propagates immediately, scheduler picks next.
- ("UNLOCK", mutex_id): RUNNING task releases it. Grant to best waiter, recompute, reschedule (unlocks can cause immediate preemption).
- ("EXIT",): RUNNING task finishes, releases all held mutexes in alphabetical order of mutex_id (waking waiters), becomes COMPLETED.

Return list of RUNNING task_id after each event. Reply with ONLY the complete function in one ```python block.'''

S1_ASSERTS = '''def test_s1_rtos():
    events = [
        ("SPAWN", "T_LOW", 10),
        ("LOCK", "M_A"),
        ("SPAWN", "T_MED", 5),
        ("LOCK", "M_B"),
        ("LOCK", "M_A"),
        ("SPAWN", "T_HIGH", 1),
        ("LOCK", "M_B"),
        ("SPAWN", "T_NOISE", 2),
        ("UNLOCK", "M_A"),
        ("UNLOCK", "M_B"),
        ("EXIT",),
        ("EXIT",),
        ("EXIT",),
        ("EXIT",)
    ]
    expected = [
        "T_LOW", "T_LOW", "T_MED", "T_MED", "T_LOW",
        "T_HIGH", "T_LOW", "T_LOW", "T_MED", "T_HIGH",
        "T_NOISE", "T_MED", "T_LOW", "IDLE"
    ]
    assert simulate_rtos(events) == expected
test_s1_rtos()'''


S2_PROMPT = '''Write a Python function `simulate_verilog_netlist(gates, stimuli, probe_times) -> list[dict]`: event-driven 3-valued ('0','1','X') digital logic simulator with INERTIAL delay glitch cancellation.

Rules:
1. All nets start at 'X' at t=-1.
2. gates: (gate_id, gate_type, in_nets, out_net, inertial_delay_ps). NOT: 0->1, 1->0, X->X. AND: '0' if any input '0'; '1' if all '1'; else 'X'. OR: '1' if any '1'; '0' if all '0'; else 'X'. XOR: 'X' if any 'X'; else parity of '1's.
3. Inertial delay: when an input of gate G changes at time t, compute val_new vs val_current of out_net:
- val_new == val_current: CANCEL any pending event for G (pulse shorter than delay, absorbed).
- val_new != val_current: if G already has a pending event for the SAME val_new, keep it (do not push later); if pending for a DIFFERENT value, cancel it and schedule (t+delay, out_net, val_new); if none, schedule it.
4. Delta cycles: if multiple events at the same timestamp t, apply ALL net updates for t BEFORE re-evaluating fanout gates at t.
5. stimuli: (time_ps, net, value) sorted by time.
6. For each probe time t_p return {net: value} reflecting all changes processed up to and including t == t_p. A probe at time t_p < next_event_time reflects state before that event.

Reply with ONLY the complete function in one ```python block.'''

S2_ASSERTS = '''def test_s2_glitch_sim():
    gates = [
        ("G1", "AND", ["A", "B"], "Y", 10),
        ("G2", "NOT", ["Y"], "Z", 5)
    ]
    stimuli = [
        (0, "A", "0"),
        (5, "B", "1"),
        (20, "A", "1"),
        (26, "A", "0"),
        (40, "A", "1")
    ]
    probes = [9, 10, 15, 32, 50, 55]
    res = simulate_verilog_netlist(gates, stimuli, probes)
    assert res[0]["Y"] == "X" and res[0]["Z"] == "X"
    assert res[1]["Y"] == "0" and res[1]["Z"] == "X"
    assert res[2]["Y"] == "0" and res[2]["Z"] == "1"
    assert res[3]["Y"] == "0" and res[3]["Z"] == "1"
    assert res[4]["Y"] == "1" and res[4]["Z"] == "1"
    assert res[5]["Y"] == "1" and res[5]["Z"] == "0"
test_s2_glitch_sim()'''


S3_PROMPT = '''Write a Python function `simulate_moesi(num_cores, cache_capacity, ops) -> dict`: multi-core MOESI snooping cache coherence (M/O/E/S/I) with per-core fully-associational LRU cache.

Rules:
1. RAM: defaultdict 0 for all addresses. Each core cache holds at most cache_capacity addresses, LRU managed.
2. READ hits and WRITE hits update recency (MRU).
3. Eviction: when a core ALLOCATES a line and cache is full (len == capacity), FIRST evict LRU address. If evicted line is 'M' or 'O': write back to RAM (ram[addr]=val) and bus_writebacks += 1. If an 'O' line is evicted and written back, other cores holding 'S' KEEP their 'S' (RAM now clean). Evicted line becomes 'I' and is deleted.
4. ("READ", core, addr):
- Hit (M/O/E/S): return cached value, update LRU, no bus traffic.
- Miss: evict LRU if full, then snoop:
  A) no core has addr: read RAM, state 'E'.
  B) some core has 'M': it supplies data, M->O, requester gets 'S'. No write-back.
  C) some core has 'O' (others may have S): O core supplies and STAYS 'O', requester 'S'.
  D) some core has 'E': E core supplies, E->S, requester 'S'.
  E) only 'S' cores: read RAM, requester 'S'.
5. ("WRITE", core, addr, val):
- Hit: if 'M' or 'E': update, state 'M', LRU. If 'O' or 'S': broadcast Invalidate (all OTHER cores holding addr -> 'I', removed), update, state 'M', LRU.
- Miss: evict LRU if full; RWITM: all other cores holding addr (M/O/E/S) invalidate WITHOUT write-back; requester sets val, state 'M'.

Return {'reads': [values in order], 'ram': {addr:val non-zero}, 'bus_writebacks': int, 'final_states': {core: {addr: (state,val)}}}.
Reply with ONLY the complete function in one ```python block.'''

S3_ASSERTS = '''def test_s3_moesi():
    ops = [
        ("READ", 0, 100),
        ("WRITE", 0, 100, 42),
        ("READ", 1, 100),
        ("READ", 2, 100),
        ("READ", 0, 200),
        ("READ", 0, 100),
        ("READ", 0, 300),
        ("READ", 0, 400),
        ("WRITE", 1, 100, 99),
        ("READ", 2, 100)
    ]
    res = simulate_moesi(3, 2, ops)
    assert res["reads"] == [0, 42, 42, 0, 42, 0, 0, 99]
    assert res["bus_writebacks"] == 1
    assert res["ram"] == {100: 42}
    assert res["final_states"][1][100] == ("O", 99)
    assert res["final_states"][2][100] == ("S", 99)
    assert 100 not in res["final_states"][0]
test_s3_moesi()'''


S4_PROMPT = '''Write a Python function `autoroute_pcb_grid(grid_w, grid_h, obstacles, nets) -> dict`: single-layer Manhattan PCB router with iterative PathFinder rip-up & reroute.

Rules:
1. Cells (x,y), 0<=x<grid_w, 0<=y<grid_h. Moves orthogonal [(-1,0),(1,0),(0,-1),(0,1)].
2. Blocked: obstacles AND terminal pins (src,dst) of OTHER nets are permanently impassable.
3. history_cost[c] cumulative, init 0.
4. Step cost entering unoccupied adjacent cell c while routing net N: cost = 1 + history_cost[c] + (10 if c currently used by any OTHER routed net else 0). Dijkstra. Deterministic tie-break: among equal total cost paths choose lexicographically smaller sequence of (x,y).
5. Max 20 passes. In each pass, process nets in given order:
- If net already routed and NONE of its cells overlap other nets: skip.
- Else remove it from routed_paths and Dijkstra-route src->dst, store path.
- END of pass: find cells shared by >=2 nets (collisions). If zero: converged, terminate. Else for each collision cell: history_cost[c] += 4 AND rip up (delete) ALL nets passing through c.

Return {'converged': bool, 'passes_used': int (1-indexed), 'total_wirelength': sum(len(path)-1), 'paths': {net: [(x,y)...]}}.
Reply with ONLY the complete function in one ```python block.'''

S4_ASSERTS = '''def test_s4_pcb_router():
    obstacles = [(0, 1), (2, 1), (1, 0)]
    nets = [
        ("NET1", (0, 2), (2, 2)),
        ("NET2", (1, 1), (1, 3))
    ]
    res = autoroute_pcb_grid(5, 5, obstacles, nets)
    assert res["converged"] is True
    assert res["passes_used"] >= 2
    assert (1, 2) in res["paths"]["NET2"]
    assert (1, 2) not in res["paths"]["NET1"]
test_s4_pcb_router()'''


S5_PROMPT = '''Write a Python function `simulate_raft(num_nodes, events) -> dict`: deterministic Raft leader election, log replication, partitions and the Figure-8 commit rule. Nodes 0..N-1.

Per-node state: current_term=0, voted_for=None, role="FOLLOWER", log=1-indexed list of (term,cmd), commit_index=0.
Partitions: RPCs only within the same partition group.
Events:
- ("PARTITION", groups): connectivity = sublists, e.g. [[0,1],[2,3,4]].
- ("HEAL",): full connectivity restored.
- ("TIMEOUT", node): node: current_term+=1, role="CANDIDATE", voted_for=self, sends RequestVote to all REACHABLE peers in order 0..N-1. Peer P: if cand_term > P.current_term: P.current_term=cand_term, role="FOLLOWER", voted_for=None. P votes for C iff cand_term==P.current_term AND (P.voted_for is None or ==C) AND (cand_last_term, cand_last_idx) >= (P_last_term, P_last_idx) (empty log = (0,0)). Strict majority (> N//2) votes -> "LEADER", immediately sends heartbeats to all reachable peers (heartbeat: if leader.term > P.term then P.term=leader.term, role=FOLLOWER, voted_for=None).
- ("CLIENT_CMD", leader, cmd): if leader not LEADER, ignore. Else append (leader.current_term, cmd) to its log.
- ("SYNC", leader): leader replicates log to all REACHABLE followers by overwriting conflicting suffixes and appending missing entries (followers' logs become identical to leader's; also bring follower term up to leader term). Figure-8 invariant: leader.commit_index = highest index N such that a majority of ALL cluster nodes store a matching entry at index N AND log[N].term == leader.current_term. A leader MUST NOT commit entries from older terms by counting replicas. If commit advances, propagate commit_index to reachable followers (capped at their log length).

Return {'terms': [...], 'roles': [...], 'commit_indices': [...], 'logs': {node: [(term,cmd)...]}}.
Reply with ONLY the complete function in one ```python block.'''

S5_ASSERTS = '''def test_s5_raft_fig8():
    events = [
        ("TIMEOUT", 0),
        ("PARTITION", [[0, 1], [2, 3, 4]]),
        ("CLIENT_CMD", 0, "SET_X=1"),
        ("SYNC", 0),
        ("TIMEOUT", 2),
        ("HEAL",),
        ("PARTITION", [[0, 1, 2], [3, 4]]),
        ("TIMEOUT", 0),
        ("SYNC", 0),
        ("CLIENT_CMD", 0, "SET_Y=2"),
        ("SYNC", 0)
    ]
    res = simulate_raft(5, events)
    assert res["commit_indices"][0] == 2
    assert res["logs"][2] == [(1, "SET_X=1"), (3, "SET_Y=2")]
test_s5_raft_fig8()'''


def build_s_problems():
    return [
        ("s1_rtos", S1_PROMPT, S1_ASSERTS),
        ("s2_glitch", S2_PROMPT, S2_ASSERTS),
        ("s3_moesi", S3_PROMPT, S3_ASSERTS),
        ("s4_router", S4_PROMPT, S4_ASSERTS),
        ("s5_raft", S5_PROMPT, S5_ASSERTS),
    ]
