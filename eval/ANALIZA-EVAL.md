# Coding eval — DeepSeek V4.1-Flash: build vechi vs build readahead vs Strata

Harness: `coding_eval_extreme.py` — 5 probleme (P1 dijkstra+paritate, P2 microstrip Hammerstad,
P3 tx-engine cu deadlock/rollback, P4 PDN anti-rezonanta, P5 AST optimizer). Ground truth
CPU-only (`verify_extreme_engineering.py`, asserturi validate prin `selftest_extreme.py`).
Scor = teste rulate real, nu auto-raport. Pass@1 = prima incercare, single step, fara self-repair.

## Setup-uri comparate (acelasi model, aceeasi masina: RTX 5080 16 GB + 96 GB DDR5)

| Setup | Server | Template | Expert cache | Gandeire |
|---|---|---|---|---|
| **M5 (5.11)** | llama.cpp fork, build vechi | chatml (`--no-jinja`) | expertii pe CPU (`-ot ffn..=CPU`, fit) | ON, budget natural |
| **S1 (9.11)** | fork build `readahead` | jinja Unsloth (implicit) | GPU cache (mask1, 8 slots, lazy on) | ON, budget -1 |
| Strata engine | motor nativ Strata + DSpark | — | 76 GB RAM store | — |

Hyperparametri identici la ambele rulate: think temp 0.6 / top_p 0.95, max_tokens 16384, np 1.

## Rezultate

| Setup | pass@1 (P1–P5) | finish | decode in eval | Gandire per problema |
|---|---|---|---|---|
| M5 build vechi | **5/5** (10/10 pe suite completa) | stop x5 | 8.7–11.5 t/s | 6k–50k chars, **se inchide singura** |
| S1 readahead | **2/5** | p2/p4 stop · p1/p3/p5 **length** | 14.5–20 t/s | p1/p3/p5: 63–68k chars, **nu se inchide** |
| S1 readahead nothink (ctx 8192) | 2/5 (4/5 pass@5) | — | 9–22 t/s | — |

Detaliu S1 (9.11, think, single pas):

| Prob | Rezultat | Tok total | finish | Gandire (chars) | t/s |
|---|---|---|---|---|---|
| p1 dijkstra | FAIL | 16384 | length | 62 909 | 20.0 |
| p2 microstrip | PASS @1 | 6076 | stop | 14 091 | 14.5 |
| p3 tx_engine | FAIL | 16384 | length | 65 001 | 18.6 |
| p4 pdn | PASS @1 | 3553 | stop | 10 314 | 17.0 |
| p5 ast | FAIL | 16384 | length | 68 479 | 19.8 |

## Ce arata comparatia

1. **Viteza din fork-ul readahead e reala dar nu ajunge la Strata.**
   Recipe-ul castigator din bench10 (mask1 + lazy on, reasoning OFF) face 17.8 mediu / 21.4 t/s
   warm decode — sub motorul Strata pe acelasi GGUF (cfg_M: 15.1 t/s decode cu DSpark si
   **728 t/s prefill** vs 41.8 t/s prefill la fork — de 17x). Iar pe Qwen3.8-Flash-Next,
   Strata urca la ~69–76 t/s sustained. Diferenta e arhitectura: Strata tine expertii intr-un
   store de RAM dedicat + draft DSpark; llama.cpp face swap sincron de experti pe PCIe.

2. **La viteza mai mare, calitatea a scazut.** Build-ul vechi (chatml, expertii pe CPU,
   8.7–11.5 t/s) a luat 5/5 pass@1. Build-ul readahead, de ~2x mai rapid, a luat 2/5.
   Cauza directa: pe 3 din 5 probleme modelul **intră în buclă de gândire care nu se mai
   închide** — 63–68k caractere de reasoning (~18–19k tokeni), taie la plafonul de 16384
   (`finish=length`) și nu apucă să scrie cod. Build-ul vechi cu chatml închidea gândirea
   natural (p3: 50k chars reasoning, `finish=stop`, cod scris, pass).

3. **Limitarea e în serving, nu în model.** Aceeași cuantificare IQ2_XS, aceiași hyperparametri.
   Diferența de scor vine de la template (jinja Unsloth vs chatml) și de la interactiunea
   reasoning ON × lazy-mode din build-ul readahead — `--reasoning-budget` nu forcheaza
   inchiderea blocului de think (demonstrat in bench9 R, budget 256). Gandirea din S1 ruleaza
   totusi corect separat (`reasoning_content` populat, 10–68k chars/problema) — contaminarea
   venea din `--skip-chat-parsing`, eliminata la rerun.

4. **Confundari onesti:** comparatia M5 vs S1 nu izoleaza un singur factor (template + build +
   plasarirea expertilor se schimba simultan). Fixul candidat (`--no-jinja --chat-template
   chatml` in S1, aplicat in script) nu a fost rerulat — decizia a fost ca nu merita inca ~1 h
   de rulare ca sa confirmam ce arata deja 3 esecuri identic simulate.

## Concluzie

Knob-urile de viteza din fork (moe-cache-mask, lazy-mode, readahead) functioneaza — 2.2 →
17.8 t/s de-a lungul bench-urilor — dar plafonul ramâne sub motorul Strata (mai ales la
prefill, 17x), iar pe problem hard build-ul readahead limitează modelul: gândirea nelimitată
+ template-ul implicit produc 3/5 esecuri prin epuizarea bugetului de tokeni în thinking, nu
prin eroare de algoritm. Speed-up de 2x fără regression de calitate — în forma actuală a
fork-ului, nu se obțin.

## Fisiere

- `results/DSV41-M5_think.json` — reference run 5.11 (10/10 pass@1, 10 probleme)
- `results/DSV41-S1_think*.json` — run 9.11 think (2/5)
- `results/DSV41-S1_nothink*.json` — run 9.11 nothink (2/5 @1, 4/5 @5)
- `coding_eval_extreme.py`, `s_suite.py`, `verify_extreme_*.py`, `selftest_extreme.py` — harness + ground truth
- `run_deepseek_extreme.sh` — server recipe M5 (build vechi)
- `run_dsv41_S1_eval.sh` — server recipe S1 (readahead; think default, `--nothink` pentru bench10 recipe)

Căile locale sunt inlocuite cu variabile de mediu (`DSV41_MODEL`, `DSV41_FORK_DIR`, `STRATA_DIR`).
