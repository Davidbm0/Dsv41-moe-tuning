# Coding eval — DeepSeek V4.1-Flash prin 3 interfete de serving (acelasi GGUF, aceeasi masina)

Harness: `coding_eval_extreme.py` — 5 probleme (P1 dijkstra+paritate, P2 microstrip Hammerstad,
P3 tx-engine cu deadlock/rollback, P4 PDN anti-rezonanta, P5 AST optimizer) + 5 extensii
e1–e5 in runul M5. Ground truth CPU-only (`verify_extreme_engineering.py`, asserturi validate
prin `selftest_extreme.py`). Scor = teste rulate real, nu auto-raport. Pass@1 = prima
incercare, single step, fara self-repair.

Model: DeepSeek-V4.1-Flash-1017 GSQ-RCO **IQ2_XS** (acelasi fisier in toate cele 3 runuri).
Hardware: RTX 5080 16 GB + 96 GB DDR5-5600. Hyperparametri think identici: temp 0.6 /
top_p 0.95 / min_p 0, max_tokens 16384, np 1.

## Cele 3 interfete

| Interfata | Server | Detaliu |
|---|---|---|
| **Strata engine** | `strata-dsv41.exe --serve` (:8081) | pack 2.25bpw, ctx 32768, expertii intr-un store de RAM dedicat (CPU lane 61–64%), DSpark auto |
| **llama.cpp build vechi** | llama-server (build Oct-28, fara HC fused) | expertii routati pe CPU prin `-ot`, ~2.5 t/s |
| **llama.cpp build readahead** | llama-server din fork (bench10 recipe) | GPU expert cache: mask1, 8 slots, lazy on, `-b 4096 -ub 1024`, readahead 16 |

## Rezultate

| Run | Interfata | pass@1 | finish | Decode in eval |
|---|---|---|---|---|
| M5 (5.11) | **Strata engine** | **10/10** (5 P + 5 extensii) | stop ×10 | 8.7–11.5 t/s |
| Extreme (4.11) | llama.cpp build vechi | **0/5** | stop, dar 1–22 tokeni output (degenerat) | ~0.1–2.5 t/s |
| S1 think (9.11) | llama.cpp readahead | **2/5** | p2/p4 stop · p1/p3/p5 **length** | 14.5–20 t/s |
| S1 nothink (9.11) | llama.cpp readahead | 2/5 (4/5 pass@5) | — | 9–22 t/s |
| Strata-DSV41 (4.11) | Strata engine (prima incercare, max_tokens 8192) | 1/1 (p5) | stop | 3.7–6.1 t/s |

Detaliu S1 think (9.11, single pas):

| Prob | Rezultat | Tok total | finish | Gandire (chars) | t/s |
|---|---|---|---|---|---|
| p1 dijkstra | FAIL | 16384 | length | 62 909 | 20.0 |
| p2 microstrip | PASS @1 | 6076 | stop | 14 091 | 14.5 |
| p3 tx_engine | FAIL | 16384 | length | 65 001 | 18.6 |
| p4 pdn | PASS @1 | 3553 | stop | 10 314 | 17.0 |
| p5 ast | FAIL | 16384 | length | 68 479 | 19.8 |

Comparativ, M5 pe Strata engine (aceleasi 5 P1–P5, toate pass):

| Prob | Tok total | finish | Gandire (chars) | t/s |
|---|---|---|---|---|
| p1 dijkstra | 2024 | stop | 5 997 | 10.5 |
| p2 microstrip | 7272 | stop | 16 108 | 11.5 |
| p3 tx_engine | 13471 | stop | 50 008 | 11.4 |
| p4 pdn | 3514 | stop | 10 458 | 8.7 |
| p5 ast | 6152 | stop | 24 656 | 11.2 |

## Ce arata comparatia

1. **Viteza in llama.cpp fork e reala si a castigat la decode.** Build-ul vechi: 0/5, ~2.5 t/s,
   output degenerat (1–22 tokeni). Build-ul readahead cu recipe-ul bench10: 17.8 mediu /
   21.4 t/s warm (reasoning OFF) — de ~8x peste build-ul vechi si peste 15.1 t/s decode al
   Strata engine pe acelasi GGUF. In eval (think), llama readahead a facut 14.5–20 t/s vs
   8.7–11.5 t/s prin Strata — adica ~1.7x mai rapid la decode.

2. **Dar la viteza mai mare, calitatea a scazut sub Strata.** Strata engine pe acelasi model:
   10/10 pass@1, gandirea se inchide natural in toate runurile (`finish=stop` ×10, inclusiv
   p3 cu 50k chars de reasoning). llama.cpp readahead, de ~2x mai rapid: 2/5. Pe 3 din 5
   probleme modelul **intră în buclă de gândire care nu se mai închide** — 63–68k caractere
   (~18–19k tokeni), taie la plafonul de 16384 (`finish=length`) și nu apucă să scrie cod.

3. **Limitarea e in serving, nu in model.** Același IQ2_XS, aceiași hyperparametri. Diferența
   vine de la lantul de serving: template (jinja Unsloth vs chatml), interactiunea reasoning ON ×
   lazy-mode, si `--reasoning-budget` care nu forcheaza inchiderea blocului de think (demonstrat
   in bench9 R, budget 256). Gandirea din S1 ruleaza corect separat (`reasoning_content` populat,
   10–68k chars/problema) — contaminarea initiala venea din `--skip-chat-parsing`, eliminata la rerun.

4. **Prefill ramâne mult sub Strata.** Cel mai bun prefill masurat in fork: 41.8 t/s (bench10 S1).
   Strata engine pe acelasi GGUF: 728 t/s (cfg_M) — de 17x. La chat real (prompturi mari +
   istoric lung), diferenta de prefill se simte mai mult decat cea de decode.

5. **Confundari onesti:** comparatia M5 (Strata) vs S1 (llama) nu izoleaza un singur factor —
   template + build + plasarea expertilor (RAM store vs GPU cache) se schimba simultan. Fixul
   candidat (`--no-jinja --chat-template chatml` in S1, aplicat in script) nu a fost rerulat.

## Concluzie

Knob-urile de viteza din fork (moe-cache-mask, lazy-mode, readahead) functioneaza — 2.2 →
17.8 t/s mediu de-a lungul bench-urilor — dar in coding eval real, aceeasi interfață llama.cpp
de ~1.7x mai rapidă la decode ia **2/5** fata de **10/10** prin Strata engine pe acelasi model:
gândirea nelimitată + template-ul implicit produc 3/5 esecuri prin epuizarea bugetului de
tokeni în thinking, nu prin eroare de algoritm. Speed-up fără regression de calitate — în
forma actuală a fork-ului, nu se obțin.

## Fisiere

- `results/DSV41-M5_think.json` — reference run 5.11 prin Strata engine (10/10 pass@1, 10 probleme)
- `results/DSV41-S1_think*.json` — run 9.11 prin llama.cpp readahead, think (2/5)
- `results/DSV41-S1_nothink*.json` — run 9.11 prin llama.cpp readahead, nothink (2/5 @1, 4/5 @5)
- `results/DeepSeek-V4.1-IQ2XS_Extreme_pass1_think.json` — run 4.11 prin llama.cpp build vechi (0/5)
- `results/Strata-DSV41-IQ2XS_think.json` — run 4.11 prin Strata engine, p5 (1/1)
- `coding_eval_extreme.py`, `s_suite.py`, `verify_extreme_*.py`, `selftest_extreme.py` — harness + ground truth
- `run_deepseek_extreme.sh` — server recipe build vechi (llama.cpp)
- `run_dsv41_S1_eval.sh` — server recipe readahead (think default, `--nothink` pentru bench10 recipe)

Căile locale sunt inlocuite cu variabile de mediu (`DSV41_MODEL`, `DSV41_FORK_DIR`, `STRATA_DIR`).
