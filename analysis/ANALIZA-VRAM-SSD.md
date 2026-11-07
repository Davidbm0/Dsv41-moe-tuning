# VRAM max + SSD pull — analiza (2026-11-06)

## VRAM: cache-ul AUTO umple deja ce poate; marja e singura care crește
Codul (src/llama-model.cpp ~2287): slots auto = (VRAM liber - margin - fixed) / per_slot.
- E a primit 7 slots/strat = 2591 MiB => per_slot = 370 MiB/strat (40 straturi, expert IQ2_XS ~9.3 MB)
- CUDA0 model buffer = 5440 MiB (model non-expert pe GPU) + compute 339 + KV 20 + cache 2591 + marja 512 + pool CUDA => ~14.3/16.3 GB folositi. 5080 e deja aproape plin.
- Marja default 512 MiB e conservatoare. Cu --moe-cache-margin 128 + --moe-cache-slots 24: 24*370 MB = 8.9 GB cache => total ~15.1 GB VRAM. Asta e "15.5 GB"-ul cerut.
- 3080 (10 GB) e complet nefolosit. K_dual_slots: -dev CUDA0,CUDA1 -sm layer -ts 30,18 + slots auto => cache pe ambele GPU. Build-ul are GGML_SCHED_MAX_COPIES=4 (deja ok in CMakeCache).

## SSD: llama.cpp NU are SSD expert cache ca Strata
Fork-ul foloseste mmap + page cache OS. "SSD pull" = rata la care page cache-ul aduce expertii lipsa.
Masurat pe T705 (fisier GGUF real, 12 GB offset):
- 1 thread: 1.91 GB/s
- 4 thread: 2.08 GB/s
- 8 thread: 1.99 GB/s  => ~2 GB/s, NU scaleaza cu thread-uri
Suspicios de mic pentru Gen5 (T705 = 12-14 GB/s specificatie). De verificat: grad de umplere disc, temperatura, slot.

## De ce conteaza: matematica decode
6 experti/token x 40 straturi x 9.3 MB = 2.4 GB citiri per token generat.
- Doar din SSD @ 2 GB/s => 0.83 tok/s (asta a fost D cu 0.0)
- RAM 96 GB page cache ~ 37% din expertii calzi => H = 3.0 t/s
- + cache VRAM 7 slots => E = 7.2 t/s
- + cache VRAM 24 slots (8.9 GB) => hit-rate mult mai mare => potential 10-12 t/s
SSD-ul e doar backstop; RAM+VRAM fac catldura.

## Configuri noi in script (I/J/K)
I_slots16: 16 slots (5.9 GB cache)
J_slots24: 24 slots (8.9 GB cache, ~15.1 GB VRAM total)
K_dual_slots: auto pe ambele GPU (5080+3080, ~19 GB cache potential)
Toate cu margin 128, dyn32, mask2.
