# dsv41-moe-tuning — tuning a 763B MoE on one RTX 5080 (16 GB VRAM, 96 GB DDR5)

Benchmark scripts + measured results for running **DeepSeek V4.1-Flash** (1017B-class MoE,
GSQ-RCO quantization) through a llama.cpp fork with a **GPU expert cache**, on a desktop
that cannot hold the model in VRAM.

The point of this repo: MoE decode speed on a small GPU is a **memory-bandwidth and
cache-hit-rate problem**, not a FLOPS problem. These benches measure which knobs
(`--moe-cache-mask`, slots, margin, prefetch) move the needle, and by how much.

**Hardware:** AMD Ryzen 7 7800X3D · RTX 5080 16 GB · 96 GB DDR5-5600 (2x48) · NVMe Gen5
**Fork:** [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp), branch `fable5/deepseek41`
**Model:** DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XS GGUF (~266 GB, 6 shards)

## The knobs (what each bench proved)

| Knob | Effect | Evidence |
|---|---|---|
| `--moe-cache-mask 2` | **3x decode** (2.7 → 7.8 t/s): router restricted to cached experts except top-2 | bench4 vs bench5, same slots |
| `--moe-cache-mask 2` + prefill | **8x slower prefill** (4.3 → 0.5 t/s) on the old build; fixed by `-b` > `-ub` + `--layer-major` + `GGML_SCHED_READAHEAD` on the readahead build (41 t/s) | bench5 vs bench6 |
| `--moe-cache-slots` | more cached experts/layer = higher hit rate; 16 slots without borrow OOMs CUDA0 | bench4 (7 pass / 14 fail) |
| `--moe-cache-margin` | reserves VRAM for the prefill graph; 512 vs 2048 changes both hit rate and OOM | bench1 vs bench5 |
| `GGML_SCHED_PREFETCH_EXPERTS` | overlaps expert uploads with compute (env var) | bench5 setup |
| `--layer-major` / readahead (newer fork commits) | fixes the prefill side; not in the Oct-28 build used here | fork PR, pending rebuild |

## Results (decode t/s, same model, same GPU)

| Run | Config | Decode | Prefill |
|---|---|---|---|
| Strata native engine (`cfg_M`) | 76 GB RAM expert store + DSpark | **15.1** | 728 |
| bench1 `I_slots8` | mask2, 8 slots, margin 512 | **8.93** | — |
| warm-server `E_dyn32_mask2` | mask2, 7 slots auto | 7.23 | 15.2 |
| bench5 `B_mask2_slots12` | mask2, 12 slots, margin 2048 | 7.83 | 0.53 |
| bench4 `K/L` (no mask) | 8–16 slots, no mask | 2.74 | 2.1 |
| bench4 `A_baseline` | no cache | 2.24 | — |
| **bench6 `F` (readahead build)** | mask2, 8 slots, `-b 4096 -ub 1024`, `--layer-major 1` | **8.3** | **41.0** |
| bench6 `E` (readahead, `-b 256`) | control, same as bench5 A | 7.3 | 3.4 |
| bench6 `G` (`--layer-major 3`) | 4 ubatches — nothing to group | 6.8 | 38.8 |
| bench7 `I`/`K` (`--weights-in-ram 60`) | **Linux-only knob** — `load_weights_ram: --weights-in-ram needs Linux` on Windows, no effect | bench7 logs |
| bench7 `J`/`K` (`-ub 512 --layer-major 3`) | worse prefill than F — lm3 needs many ubatches per batch | 6.8–7.5 | 12.1 |
| **bench7 `L` (`--lazy-mode on --no-sched-async-cpu`)** | **18.4 / 21.2 / 22.7 t/s** across 3 warm runs (server-measured) — see caveat | 20.8 | 41.6 |

### What bench7 proved

- `--weights-in-ram` is a no-op on Windows (Linux-only). Dead knob.
- `-ub 512` + `--layer-major 3` *hurts* prefill (41 → 12 t/s). The winning prefill recipe
  stays `-b 4096 -ub 1024 --layer-major 1`.
- `--lazy-mode on --no-sched-async-cpu` produced 18–22 t/s decode — the fork's headline
  number, on our hardware. **Caveat:** all three runs hit the 1536-token cap in a
  low-entropy repetition loop ("Need X. Need Y. …"), where the router keeps picking the
  same experts → near-100% cache hit rate → no swaps. Coherent runs (bench6 F) still
  decode at 8–9 t/s. Warm-up is visible: run1 18.4 → run3 22.7 t/s.
- The speed itself is real (server `eval time`, not a client artifact): 1536 tokens in
  67–83 s. The open question is how much of it survives coherent output.

### bench8 results (lazy-mode on, coherent-decode attempt)

| Config | decode avg | prefill | finish_reason |
|---|---|---|---|
| bench8 `M` (L recipe, mask 2, 8 slots) | **18.7** (15.2 / 19.8 / 21.0) | 43.1 | length ×3 |
| bench8 `N` (mask 2, 12 slots) | 19.6 (15.4 / 21.1 / 22.3) | 42.1 | length ×3 |
| bench8 `O` (**mask 1**, 8 slots) | **22.7** (18.8 / 24.0 / 25.2) | 43.1 | length ×3 |

### What bench8 proved

- `--lazy-mode on --no-sched-async-cpu` is the real decode lever: 8–9 → 18–25 t/s on our box.
- **mask 1 beats mask 2** once lazy-mode is on (22.7 vs 18.7). Forcing the router into cache
  residency is no longer needed; letting it roam a warm cache is faster.
- 12 slots ≈ 8 slots (19.6 vs 18.7). More warm experts don't buy much.
- Warm-up curve is consistent: run1 ~15 → run3 ~21–25 t/s.
- **But**: all 9 runs ended `finish=length` — the model rambles inside the `<think>` block
  ("We need answer. Need recall formulas…") and never closes it. The 512 tokens are
  low-entropy reasoning filler, so the expert router keeps picking the same experts →
  cache stays hot → speed is the *warm-cache ceiling*, measured on degenerate text.
  bench6 F (lazy-mode auto) stopped naturally at 249–390 tokens at 7–9 t/s.
- Open confound: does lazy-mode *itself* degrade the output, or does degenerate output
  simply enable the speed? bench9 separates the two.

### bench9 results + verdict

| Config | decode avg | prefill | finish | output quality |
|---|---|---|---|---|
| `P` lazy on + mask 1 + **reasoning OFF** | **15.2** (10.5 / 16.2 / 18.7) | 42.6 | length (512 cap) | **coherent markdown**, real HJ formulas |
| `Q` lazy auto + mask 1 + reasoning ON | 7.8 (6.8 / 8.3 / 8.4) | 43.4 | stop ×3 | **garbage** ("?< 是与其他意思…") |
| `R` lazy on + mask 1 + budget 256 | 18.5 (12.2 / 20.0 / 23.2) | 43.6 | length | rambling "Need answer. Need…" |

Verdict:
- **lazy-mode on is a real speed knob, not a degeneracy artifact.** P generated a coherent
  Hammerstad-Jensen writeup (correct ε_eff discussion, proper LaTeX) at 16–18.7 t/s warm.
  `finish=length` on P is just the 512-token cap cutting a long answer mid-section
  ("## Python implementation"), not a loop.
- **The degeneracy in bench8 came from reasoning ON + lazy on**, not from lazy itself.
  `--reasoning-budget 256` did NOT force the think block closed (R still rambles) — budget
  appears ineffective with this template.
- **mask 1 + lazy auto = garbage output** (Q). Mask 1 with only 8 slots forces the router into
  8/128 experts per layer; quality collapses. Mask 2 (bench6 F) was coherent. Mask 1 is too
  aggressive at this cache width.
- Caveat: `finish_reason` alone is not a quality gate — Q stopped at EOS and was still garbage.
  Content inspection is required.

### bench10 (queued)

`scripts/llama_server_bench10.py` + `RUN-BENCH10.bat` — clean 2×2 {mask 1, mask 2} × {lazy on, lazy auto},
all with **server-side reasoning OFF**, decode 1024 tok (full answer incl. code), samples recorded as
head+tail so loop-vs-complete-answer is visible in the summary:
- `S1` mask 1 + lazy on (P at 1024 tok — does the answer finish with finish=stop?)
- `S2` mask 2 + lazy on (same speed as S1 with better quality?)
- `S3` mask 1 + lazy auto (was Q's garbage from reasoning ON or from mask 1 itself?)
- `S4` mask 2 + lazy auto (coherent control, reasoning OFF)

### What bench6 proved (old build vs readahead build, same model/GPU)

- Prefill: **0.53 → 41 t/s (77x)**. The fix is `-b 4096 -ub 1024` + `GGML_SCHED_READAHEAD=16`
  + `--layer-major 1`: fewer ubatches means fewer full passes over the expert weights,
  and the readahead threads overlap SSD reads with compute.
- `--layer-major 3` does **not** help at only 4 ubatches per prompt (it needs many
  ubatches per batch, as in the fork README's `-b 32768` test).
- Decode is unchanged by the new build (~7–9 t/s) — matching the fork README
  ("decode speed did not change"). The decode plateau is the synchronous expert swap.
- A seeded `--moe-cache-profile` helped on the old build (+20%) but is redundant once
  readahead is on.
- Best individual runs: 9.1 t/s decode (F run3, H run3).

Bandwidth math: without mask the router pulls ~2.2 GB of expert weights per token;
with mask2 it is ~0.7 GB/token. At 7.8 t/s that is ~5.8 GB/s — far below the DDR5-5600
ceiling (~90 GB/s theoretical), so the fork's synchronous expert-swap overhead, not RAM
bandwidth, is the current limit.

## Scripts

`scripts/llama_server_bench*.py` — self-contained benches (preflight, VRAM gate,
incremental JSON summary, per-config logs). Each takes `--dry` (print commands, no GPU),
`--only <tags>`, `--no-kill`.

Point them at your own model via env vars instead of editing the file:

```bash
export DSV41_MODEL=/path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf
python scripts/llama_server_bench5.py --dry
```

`local-fixes.patch` — two local patches needed to serve the ISTA GGUF chat template
(`dict()` jinja builtin + a context include); apply on top of the fork checkout.

## Reproduce

1. Clone the fork branch, apply `local-fixes.patch`, build with CMake + CUDA
   (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` for Blackwell).
2. Get the GGUF (split shards; the Engram sidecar loads automatically from the same dir).
3. Set `DSV41_MODEL`, run `--dry` first, then the bench. Close any other GPU user first
   (the scripts refuse to start if the GPU is busy — that gate is intentional).

## Caveats

- IQ2_XS output quality is poor at these configs; speed numbers are comparable across
  configs, samples are not a quality benchmark.
- `--moe-cache-mask` deliberately changes routing (speed vs quality tradeoff).
- The DSpark draft GGUF published for this model has a tensor-layout mismatch
  (`output_hc_fn.weight` missing) — speculative decoding is not wired in here.
- Results are from one machine; the VRAM gate values assume 16 GB.

`analysis/` contains the two write-ups (warm-server analysis, VRAM/SSD pull analysis)
in Romanian. `results/` has the raw JSON summaries.
