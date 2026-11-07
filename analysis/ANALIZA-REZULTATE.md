# Analiza finală — warm server bench (RUN-LLAMA-SERVER-BENCH.bat, 16:29-16:40)

## Rezultate (server cald, warmup + 3 cereri, chatml, REGISTER_HOST+PREFETCH=1)

| Config | Gen t/s | Prompt t/s | Cache activ? |
|---|---|---|---|
| H_warm_fara_cache | 3.03 | 6.98 | nu |
| **E_dyn32_mask2** | **7.23** | **15.21** | DA: 40 straturi x 7 slots, 2591 MiB în VRAM |
| G_profile_only | 2.30 | 7.38 | NU — profile.csv lipsește |
| F_profile_dyn_mask | 3.24 | 6.91 | NU — profile.csv lipsește |

## Concluzii

1. **Cache MoE confirmat funcțional: E = 2.4x peste H** (7.23 vs 3.03 gen; 15.2 vs 7.0 prompt). Log E arată clar:
   - `moe_cache_build: expert cache: 40 layers x 7 slots, 2591.37 MiB uploaded to CUDA0`
   - re-electări dyn la fiecare 32 tokeni (276→355→411 uploads, 0 skipped)
   - switch cache: 2364 ms (asta e penalizarea re-encărcării)
2. **G și F au fost aruncate de facto**: `llama-moe-trace` nu acceptă `--no-jinja` (eroare invalid argument) => profile.csv nu s-a creat => server: `cannot open profile ... expert cache disabled`. G/F = practic H cu overhead. Cauza reparată în script.
3. **E e încă sub Strata cfg_M** (7.2 vs 9-15 tok/s), dar acum la ~80% din baseline cu doar 7 slots/strat (2.6 GB din ~14 GB VRAM liberi). Sloturi mai multe = hit-rate mai mare.
4. **Calitatea output-ului e proastă la TOATE** (inclusiv H fără cache) — chatml + IQ2_XS 2.3 bpw. `--moe-cache-mask 2` schimbă deliberat routingul ("it changes the output" — zis în help). Deci viteză vs calitate.
5. Comparabil cu testele reci A-D: E (7.23) ≈ 1.3x peste C (5.5) — metoda caldă contează.

## Următorul pas (reparat în script, de rerulat)
- trace fixat (fără --no-jinja; moe-trace oricum folosește prompt simplu) => profile.csv real
- `--only G,F` ca să nu reruleze H/E
- apoi variantă cu slots mai mari: `--moe-cache-dyn 32 --moe-cache-mask 2` + profile + eventual LLAMA_ARG slot count mai mare (40x7=2.6 GB; se poate urca spre ~8-10 GB)

## Comparație finală cu Strata cfg_M
- Strata: LFU + budget 12000 + SSD 76 GB => 9-15 tok/s, calitate OK (template nativ)
- llama.cpp fork: expert cache VRAM => 7.2 tok/s decode / 15.2 prefill, calitate slabă cu mask
- Prefill-ul fork-ului (15 t/s) e punctul forte real pentru HLE/roade lungi.
