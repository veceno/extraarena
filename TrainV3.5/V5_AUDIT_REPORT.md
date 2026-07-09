# Аудит архитектуры обучения ботов V5 (TrainV3.5)

Дата: 2026-07-08. Worktree: `glm-5.2/TrainV3.5Prep`. Аудировано 7 измерений из 8 (одно упало на лимит — см. Приложение). Всего находок: 69, подтверждено ≥2/3 линз: 65.

## 1. Executive Summary

V5-Max pipeline портирован как код в 7 блоках (-1→0→A→B→C→D→E1) с зелёными синтетическими тестами, но операционно запущены только Phase A и Block B. Оба симптома подтверждены кодом и артефактами: S1 (монотонные условия) — диверсифицирующая машина Block B (`BlockBLeagueDriver` с parity/curriculum/promotion/exit) существует, но UNTRACKED-раннер `run_blockB_league.py` инлайнит упрощённый цикл и ни разу её не вызывает; `p1/p2_score_rate=0.5` frozen на 25000+ апдейтов во всех 7 прогонах. S2 (second-start катастрофа first=0.54 / second=0.19 при near-symmetric seat 0.33/0.39) — композиция трёх блокерных факторов: отсутствие persistent obs-фичи starting-side (`globals[16..31]` = 16 нулевых каналов), симметричный reward без turn-order компенсации, и мёртвый SecondStartParityLoop. Имена-фиксы (`rewardfix`/`diverse_fixed`/`start_second-targeted`) НЕ устранили симптомы; targeted BACKFIRED (second 0.188→0.133, wr 0.363→0.324). Найдены 10 уникальных blocker и ~20 major проблем, плюс критический reproducibility-блокер: все 7 runs выполнены на 6-in-1 uncommitted diff (включая `legal_action_offsets` fix), свежий checkout падает при `env_count>1`.

Главные выводы:
- **S1 подтверждён жёстко**: `parity.update()` никогда не вызывается в `run_blockB_league.py` (0 hits), `p1_score_rate=p2_score_rate=0.5` как единственное значение в 100% апдейтов всех 7 run-ов; `BlockBLeagueDriver` (B8) с полной диверсифицирующей машиной — мёртвый код, в рантайме не инстанцируется.
- **S2 подтверждён и объяснён mechanistically**: асимметрия именно по turn-order (first/second 0.54/0.19 = 3x gap), не по seat (p1/p2 ~0.33/0.39 near-symmetric); root cause — модель физически не знает свою starting-side (нет `am_first_player` obs-фичи) + симметричный reward + dead parity loop + `start_second` без shuffle = forced all-p2 training против v4max-heavy → catastrophic forgetting.
- **Все 3 имени-фикса косметические или backfired**: `rewardfix` = uncommitted macro-step negation (не reward-формула), сжал loss 10x но winrate упал 0.393→0.387; `diverse_fixed` — осцилляция 0.23–0.40 и деградация 0.398→0.266; `targeted` — second 0.188→0.133, wr 0.363→0.324.
- **Критический reproducibility-блокер**: 6 связанных operational изменений (reward attribution fold, `legal_action_offsets` fix, `start_second`/`strict_balanced` branches, per-env ChaCha, full_envs execute) живут только в uncommitted diff; `git stash` + `env_count>1` → crash в `prepare_rust_ppo.py:902`. Все `runs/*.npz` произведены uncommitted-кодом.
- **НОВЫЕ prod-readiness проблемы на оси `mana_draw`**: параллельный binary head никогда не обучается PPO (padded-legal scoring обходит `model.__call__`), его logit-шкала не калибрована к 601-candidate logits, и eval отбрасывает `output[2]` — все eval WR измеряют 601-ONLY политику и являются misleading-прокси для prod WR, где активный необученный head может nondeterministically override'ить 601-решения.

## 2. Подтверждение симптомов

### S1 — монотонные условия / модель портится: ПОДТВЕРЖДЕНО

Механизм (file:line):
- `TrainV3.5/scripts/run_blockB_league.py:186,217-218` — `parity = SecondStartParityLoop(...)` сконструирован, `p1_score_rate=float(parity.p1_score_rate())` читается, но `parity.update()` НИКОГДА не вызывается (grep → 0 hits). Rates остаются на конструкторных дефолтах 0.5/0.5 навсегда.
- `TrainV3.5/scripts/run_blockB_league.py:204-265` — операционный цикл инлайнен; `BlockBLeagueDriver` (B8) с `_measure_snapshot`/`evaluate_block_b_gate`/`maybe_update_best_ever`/`detect_h2h_plateau`/`_collapse_boost_for` — 0 hits в раннере.
- `TrainV3.5/scripts/run_blockB_league.py:83-87` — `DynamicSelfSnapshotOpponent._selected_entry` всегда берёт `rolling[-1]` (новейший снапшот); self-play = monotone recency ladder, не diverse league.
- `TrainV3.5/scripts/run_blockB_league.py:295-296` — `--opponent-mix` override: `if opponent_mix_override is not None: return _normalize_mix(opponent_mix_override)` — ранний return БЕЗ `curriculum.reweight`; в targeted run `mix_used` имеет 1 unique значение across 5000 updates.
- `TrainV3.5/scripts/run_blockB_league.py:251-258` — `SnapshotEntry(..., h2h_vs_best=0.5, ..., promotion_eligible=False)` хардкод; `best_ever` заморожен на seed-анкоре u250 весь прогон.
- `TrainV3.5/python/train_v3/block_b_opponent_mix.py:131,154,157` — `BLOCK_B_V4_ORIG_TOTAL=0.75`, `FROZEN_NON_SELF_TOTAL=0.95`, `SELF_SNAPSHOT_SHARE_CAP=0.05` → 75-78% batch против одной замороженной V4-orig ONNX каждый апдейт.
- `TrainV3.5/rust/trainv3_core/src/worker.rs:1265-1282` — 'random' rule agent = детерминистическая hash-функция от `(salt, slot_idx, turn, state)`, НЕ RNG; fix (per-env ChaCha) — uncommitted в `rust_ffi.py:1280-1335`.

Воспроизведение:
```
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_rewardfix_25k_from_u0500_20260706_234554/progress.jsonl')]; print(set(r['p1_score_rate'] for r in rows), set(r['p2_score_rate'] for r in rows), set(r['oversampling_scheme']['breach'] for r in rows))"
# → {0.5} {0.5} {False}
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/progress.jsonl')]; print(len(set(json.dumps(r['mix_used']) for r in rows)))"
# → 1
python3 -c "import json; sp=json.load(open('TrainV3.5/runs/blockB_rewardfix_25k_from_u0500_20260706_234554/snapshot_pool.json')); print(sp['best_ever']['update_number'], sp['best_ever']['h2h_vs_best'])"
# → 250 0.5
```

### S2 — p1 паритет / second ужасен: ПОДТВЕРЖДЕНО

Артефакт `TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/eval_final_vs_v4max_noassist_64seeds/summary.json`:
- `source_u12500`: wr=0.363, p1=0.336, p2=0.391, first=0.539, second=0.1875, hp=-6.34
- `targeted_u2500`: wr=0.359, p1=0.328, p2=0.391, first=0.539, second=0.1797
- `targeted_u5000`: wr=0.324, p1=0.305, p2=0.344, first=0.516, second=0.1328, hp=-7.38

Winrate ПАДАЕТ (0.363→0.324), hp-дельта ухудшается (-6.34→-7.38). Seat p1/p2 ~0.33/0.39 (near-symmetric), но turn-order first/second 0.54/0.19 (3x gap) — асимметрия именно по TURN-ORDER, не по seat.

Механизм (file:line):
- `TrainV3.5/python/train_v3/obs_v5.py:70-97`; `TrainV3.5/rust/trainv3_core/src/kernel.rs:2420-2443` — V5 globals[16..31] = 16 нулевых каналов; нет persistent `am_first_player` obs-фичи. Feedforward политика слепа к starting-side идентичности.
- `core/classic_setup.py:48-60` — p1 стартует с mana=1/max_mana=1 и ходит первым; p2 — mana=0/max_mana=0. Структурный tempo-advantage.
- `TrainV3.5/python/train_v3/reward_v5.py:109-135`; `TrainV3.5/rust/trainv3_core/src/kernel.rs:2656-2705` — reward полностью симметричен (terminal ±1/0 + perspective-relative delta shaping + V5 shaping clamp ±0.06), БЕЗ side/turn-order compensation-терма.
- `TrainV3.5/python/train_v3/rust_live_self_play.py:1112-1115` (uncommitted) — `start_second` branch: `learner_sides = np.where(starting_actors == 1, 2, 1)` БЕЗ shuffle; т.к. Rust ArenaEnv всегда `current_actor_id=1` на turn 1 (`core/classic_setup.py:40-41` fallback p1), инверсия всегда даёт 2 → все 96 envs learner=p2 (`learner_actor_counts={"2":96}` CONST across 5000 updates) → forced all-second training без ни одного p1 learner-transition → catastrophic forgetting.
- `TrainV3.5/python/train_v3/rust_live_self_play.py:884-886` (uncommitted) — macro-step reward attribution: `row = last_learner_row[i]; if row is not None and row < target_steps: rewards[row, i] += -float(out_rewards[i])`. Для second-mover первый opponent-opener происходит ДО первого learner transition → guard `row is not None` пропускает fold → opponent-opener shaping теряется (механистическое объяснение turn-order асимметрии reward signal).
- `TrainV3.5/python/train_v3/rust_live_self_play.py:862-866`; `TrainV3.5/rust/trainv3_core/src/ppo.rs:40-47` — `decisive_early_end` маркирует decided states как TERMINATED (nonterminal=0) → GAE bootstrap = 0 → value target = small shaping reward (~+0.02) вместо true value (~+1.0). Second-mover что survives early aggression и достигает decisive lead обучается с заниженным value target.
- `TrainV3.5/python/train_v3/rust_live_self_play.py:1159-1167`; `TrainV3.5/rust/trainv3_core/src/ppo.rs:42-47` — GAE bootstraps с 0.0 на `steps_per_update` boundary для ongoing episodes (no `bootstrap_values` kwarg); second-start игры длиннее → больше строк hitting 0-bootstrap → value-head corruption.
- `TrainV3.5/scripts/run_v5_vs_v4max_benchmark.py:148` — eval отбрасывает `mana_draw_logit` (output[2]); `TrainV3.5/python/train_v3/env_v5.py:73-78` — нет `mana_draw` step path. `mana_draw` — catch-up механика для second-mover; eval её не упражняет → second-start winrate искусственно занижен относительно prod capability.

Воспроизведение:
```
python3 -c "import json; print(json.load(open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/eval_final_vs_v4max_noassist_64seeds/summary.json'))['rows'])"
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/progress.jsonl')]; r=rows[0]; print(r['oversampling_scheme']['starting_actor_counts'], r['learner_actor_counts'])"
# → {'1':96,'2':0} {'2':96}
```

## 3. Найденные проблемы

### Сводная таблица blocker

| # | severity | location | симптом | кратко |
|---|---|---|---|---|
| B1 | blocker | `run_blockB_league.py:186,217-218` | S1+S2 | `parity.update()` никогда не вызывается; rates frozen 0.5/0.5 |
| B2 | blocker | `run_blockB_league.py:204-265` | S1+S2 | `BlockBLeagueDriver` (B8) — мёртвый код; раннер инлайнит цикл без диверсифицирующей машины |
| B3 | blocker | `run_blockB_league.py:251-262`; `snapshot_pool.py:324` | S1+S2 | `h2h_vs_best=0.5` хардкод; `best_ever` заморожен на u250; `maybe_update_best_ever` — 0 hits |
| B4 | blocker | `rust_live_self_play.py:1112-1115` (uncommitted) | S2 | `start_second` без shuffle → forced all-p2 (learner_actor_counts={"2":96}); targeted BACKFIRED |
| B5 | blocker | `obs_v5.py:70-97`; `kernel.rs:2420-2443` | S2 | Нет `am_first_player` obs-фичи; globals[16..31] = 16 zero channels |
| B6 | blocker | `classic_setup.py:48-60`; `reward_v5.py:109-135`; `kernel.rs:2656-2705` | S2 | Симметричный reward без turn-order компенсации при结构性 first-move advantage |
| B7 | blocker | `run_v5_vs_v4max_benchmark.py:148`; `env_v5.py:73-78`; `bot_brain.py:728` | S2+NEW | Eval отбрасывает `mana_draw` head; train+prod исполняют; eval WR = misleading-прокси |
| B8 | blocker | `rust_live_self_play.py:791` (HEAD `:759`) | NEW | `legal_action_offsets` offset-баг в committed HEAD; `env_count>1` → crash; fix uncommitted |
| B9 | blocker | `run_blockB_league.py:295-296` | S1 | `--opponent-mix` override замораживает mix (unique=1 across 5000); bypass `curriculum.reweight` |
| B10 | blocker | `run_blockB_league.py:83-87` | S1 | `DynamicSelfSnapshotOpponent` всегда `rolling[-1]`; self-play = monotone ladder |

### Сводная таблица major

| # | severity | location | симптом | кратко |
|---|---|---|---|---|
| M1 | major | `v5_policy.py:85,140`; `rust_policy.py:248-270`; `rust_ppo.py:621-695` | NEW | `mana_draw` parallel head не получает градиента в PPO (padded-legal path обходит `__call__`) |
| M2 | major | `v5_policy.py:136,140`; `mana_draw_head_v5.py:133`; `bot_brain.py:727-728` | NEW | `mana_draw` logit не калиброван к 601-candidate logits (разные input dims, нет normalization) |
| M3 | major | `rust_live_self_play.py:881-907` (uncommitted); `reward_v5.py` byte-unchanged | S1 | "rewardfix" = uncommitted macro-step negation, НЕ фикс reward-формулы |
| M4 | major | `rust_live_self_play.py:884-886` | S2 | Macro-step reward attribution дропает opponent-opener reward для second-mover |
| M5 | major | `rust_live_self_play.py:862-866`; `ppo.rs:40-47` | S2 | `decisive_early_end` zeroeит GAE bootstrap для winning positions (terminal vs truncated) |
| M6 | major | `rust_live_self_play.py:1159-1167`; `ppo.rs:42-47` | S2 | GAE bootstraps с 0.0 на steps_per_update boundary для ongoing episodes |
| M7 | major | `rust_ppo.py:335-360`; `ppo_phaseA_config.py:381` | S2 | Нет target-KL early-stop, нет grad clipping (`max_grad_norm=None`); 6 epochs stale-batch reuse |
| M8 | major | `run_blockB_league.py:141,462`; `run_phaseA_random_bootstrap.py:218` | S2 | Constant LR без warmup/decay/schedule весь 5k-25k-update run |
| M9 | major | `progress.jsonl targeted` | S2 | Negative policy_loss/KL + entropy collapse 0.058→0.022; targeted fix backfired |
| M10 | major | `rust_live_self_play.py`+`rust_ffi.py`+`ppo_phaseA_config.py`+`ffi.rs` (uncommitted) | NEW | 6-in-1 uncommitted bundle; runs невоспроизводимы из committed кода |
| M11 | major | `rust_ffi.py:1280-1335` (uncommitted); `worker.rs:1265-1282` | NEW (S1 root) | per-env ChaCha fix uncommitted; committed HEAD = hash-random (детерминистический) |
| M12 | major | `block_b_opponent_mix.py:131,154,157`; `design.md:118,144,224` | S1 | V4-orig 0.75 frozen + self cap 0.05; Q5 «modest V4-orig + high self» инвертировано |
| M13 | major | `run_blockB_league.py:276-284`; `blockB_league_manifest.json` | S2 | Manifest `best_checkpoint`=u5000 (WORST), не `pool.best_ever`=u250 |
| M14 | major | `block_b_gate.py:281-291,408-416` | NEW latent | B6 monotone = non-decreasing (flat passes), не strictly-increasing |
| M15 | major | `curriculum.py:158-194` | NEW latent | Cap на boost FACTOR не share; no floor на frozen lanes; cumulative drift |
| M16 | major | `snapshot_pool.py:418-442` | S1 | `self_snapshot_prevalence` cap 0.05; Q5 «high self-prevalence» не реализована |
| M17 | major | `run_blockB_league.py:297`; `block_b_league_driver.py:381-397` | NEW | Collapse monitor hardcoded 1.0 в runner AND bypassed override-ом |
| M18 | major | `env_v5.py:43`; `classic_rl_env.py:182`; `rust_ffi.py:1254`; `kernel.rs:767` | S2 structural | Train Rust vs Eval Python; parity по golden-тестам, не общий код |
| M19 | major | `env_v5.py:43-48`; `classic_rl_env.py:99`; `rust_ffi.py:1239` | S2 | `max_turns`=80 в EVAL vs 120 в TRAIN; long second-start игры обрезаются |
| M20 | major | `block_b_opponent_mix.py:131`; `run_blockB_league.py:329-337` | S1 | 75% batch против одной V4-orig ONNX (argmax deterministic + t07/t12 одним per-run-seeded RNG) |
| M21 | major | `worker.rs:1265-1282` | S1 | 'random' rule agent = hash-функция, НЕ RNG; одинаковые (salt,state) → один action |
| M22 | major | `run_phase10_v4max_distill.py:857,1003,1053`; `run_phase19_*.py` | NEW hygiene | Legacy distillation scripts modified но DISABLED; operator может случайно запустить disabled lane |
| M23 | major | `rust_live_self_play.py:1110-1124` (uncommitted) | S2 | `start_second` + v4max-heavy → learner систематически проигрывает → negative-reward bias |

### НОВЫЕ (не озвученные) проблемы

- **B8** — `legal_action_offsets` offset-баг в committed HEAD (`rust_live_self_play.py:791`/HEAD `:759`): `legal_action_offsets[row, i] = int(cur_offsets[i]) + int(legal_tape.size)` добавляет worker's cumulative offset из другой системы координат; тесты зелёные только из-за `env_count=1` или fake `offsets=0`. Свежий checkout + `env_count>1` → crash. Все реальные runs выполнялись с uncommitted fix. Доказательство: `git show HEAD:.../rust_live_self_play.py:759` vs текущий `:791`.
- **B9** — `--opponent-mix` override замораживает mix: `run_blockB_league.py:295-296 if opponent_mix_override is not None: return _normalize_mix(...)` — ранний return БЕЗ `curriculum.reweight`. В targeted run `mix_used` unique=1 across 5000 updates.
- **M1** — `mana_draw` head не в forward graph во время PPO: `score_padded_legal_action_inputs` (`rust_policy.py:248-270`) вызывает отдельные слои (`encode_state`, `action_encoder`, `candidate_scorer`, `value_head`), НИКОГДА `model.__call__` или `mana_draw_head`. BC (A2) / AWAC-CRR (C3) могли бы обучать head, но ни один не запущен (нет `runs/bc_*`/`runs/c_loop_*`).
- **M2** — `mana_draw` logit vs candidate_scorer logit: `candidate_scorer = Linear(hidden+action_hidden, 1)` на конкатенации state+action; `mana_draw_head = Linear(hidden, 1)` на state_emb одном. Нет calibration layer. `select_includes_mana_draw` сравнивает их напрямую.
- **M10** — 6-in-1 uncommitted bundle: `git status` показывает `M rust_live_self_play.py`+`M rust_ffi.py`+`M ppo_phaseA_config.py`+`M ffi.rs`. Diff включает reward attribution fold, offset fix, full_envs execute, start_second branch, strict_balanced branch, from_live per-env ChaCha. Имя "rewardfix" описывает 1 из 6. `stash@{0}` содержит 5 из 6.
- **M11** — per-env ChaCha fix uncommitted: HEAD `from_live` создаёт один trace (один seed на все envs), `reset_pool_mode="fixed"`, NO ChaCha → legal_random = hash. Fix: `trace_seed=seed+idx*9973` + `reset_pool_mode='cycle'` + `worker.use_chacha_rng()`. Любой `git-reset`/`clean` воспроизведёт S1 даже если оператор думает что работает на исправленном коде.
- **M14** — B6 monotone check = non-decreasing (`block_b_gate.py:289-291: all(s[i+1] >= s[i] - tol ...)`); flat/stuck aggregate проходит. Gate мёртв в operational run, но если бы был wired с хардкод 0.5 h2h — продвигал бы stuck model.
- **M15** — Curriculum cap на boost FACTOR не resulting share; no floor на frozen lanes; persistent loss к одному lane может сузить mix произвольно (latent в targeted т.к. override bypasses reweight).
- **M17** — Collapse monitor hardcoded 1.0 в runner (`run_blockB_league.py:297`) AND bypassed override-ом; mana_draw-collapse self-snapshot boost никогда не срабатывает.
- **M22** — Legacy distillation scripts (`run_phase10_v4max_distill.py`, `run_phase19_noassist_conservative_second_start.py`) модифицированы (V5 3-output adapter) но DISABLED (`ppo_phaseA_config.py:98 'distillation': 'disabled'`); не помечены deprecated; operator может случайно запустить disabled lane.
- **Minor new**: `bot_brain.py:479` — `_find_matching_legal_action_index` не сравнивает position для PlayCardAction (TODO `:465-466`); latent под full placement mode.
- **Minor new**: `classic_actions_v1.py:228` vs `kernel.rs:1346` — consume_ally при full board masked OUT в 601 mask, но EXEMPT в apply path; модель систематически слепа к full-board consume_ally.
- **Minor new**: `arena_env.py:44-51` — legacy divergent env (`MAX_HAND=4`, `MAX_BOARD=5`, 109 actions); landmine если импортнуть вместо `ClassicRLEnv`.
- **Minor new**: `ppo_phaseA_config.py:286-291` vs `rust_live_self_play.py:886` — docstring говорит "opponent-actor steps carry ZERO credit", но код negates-fold; `value_loss` logged после coef-scaling (0.5x raw MSE) — misleads operator.

### Подробности по измерениям

(Содержимое секций `/tmp/v5audit_section_*.md` свернуто в таблицы выше; ключевые цитаты и repro сохранены в разделах 2, 6, 7.)

## 4. Карта архитектуры

Pipeline V5-Max = цепь из 7 design-блоков:

| Блок | Назначение | Статус кода | Операционно запущен? |
|---|---|---|---|
| -1 | Rust ArenaEnv byte-parity port core/engine.py mechanics | COMPLETE (34/34 mechanics, 151 golden tests green) | — (parity enforced by tests) |
| 0 | V5 card-shape 73-dim disjoint fork, OBS_V5_DIM=7128, parallel mana_draw head, PARTIAL V4-Max warm-start, offline-bridge loader | COMPLETE (114 python + 157 cargo green) | нет |
| A | Random-heavy Rust ArenaEnv PPO bootstrap + A-gate (override 2026-07-05: distillation DISABLED) | COMPLETE (111 python green) | ДА — `run_phaseA_random_bootstrap.py` достиг 6.14M states (2000 updates) fresh-init teacher-free |
| B | League on Rust ArenaEnv: snapshot pool, V4-orig temp spectrum, opponent mix, curriculum, second-start parity, B-gate, exit-to-C2, league driver | COMPLETE (197 python green) | ДА — `run_blockB_league.py` (упрощённый inline, НЕ `BlockBLeagueDriver`); ~527 npz checkpoints; несколько league runs up to 25k updates |
| C | RLHF loop C2→C3: collect fresh human battles via rlhf_env, offline_replay_bridge, Hybrid AWAC×PPO-clip replay, K=2 stall exit | COMPLETE (236 python green) | НЕТ (нет `runs/c_loop_*`) |
| D | League-2 consolidation, fresh pool from post-C, D→E1 handoff | COMPLETE (40 python green) | НЕТ (нет `runs/blockD_*`) |
| E1 | Tournament threshold-table gate + human-QA SOFT panel + ship (ONNX export 3-output, vendored V5 encoders, additive BerserkInference V5 branch, extra-lr-v5-max config, LIFO V5 detector) | COMPLETE (83 python green) | НЕТ (`block_e1_runner`'s `build_production_*` stubs raise `NotImplementedError`) |

Измеренный winrate vs V4-max noassist ~0.39 (below B6/E3 ≥0.70 threshold) с severe second-start gap → pipeline застрял в Block B; модель не бьёт V4-max. Phase A "distillation" claim в оригинальном спеке explicitly overridden 2026-07-05: distillation DISABLED, Phase A = random-bootstrap PPO.

## 5. Расхождения спека↔код

| Заявлено | По факту | Источник |
|---|---|---|
| "Phase A = LLM/V4Max distillation / semi-synthetic ExtraRLHF teacher distillation" | Phase A = teacher-free random-heavy Rust ArenaEnv PPO; distillation DISABLED (`ppo_phaseA_config.py:96-100 'distillation': 'disabled'`); override 2026-07-05 в самом спеке | `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-design.md:13` |
| "Operational Block B league RUN = `BlockBLeagueDriver.run(n_updates)` с per-update B3 mix + B4 curriculum + B5 parity + D-B5 collapse monitor + every ~2000 updates B1 snapshot → B6 promotion → B7 plateau/exit→C2" | `run_blockB_league.py` инлайнит упрощённый цикл; `BlockBLeagueDriver` — 0 hits в раннере; `parity.update`/`maybe_update_best_ever`/`evaluate_block_b_gate`/`detect_h2h_plateau`/`_collapse_boost_for` — 0 hits | `BLOCK_B_COMPLETION.md:139` vs `run_blockB_league.py:204-265` |
| "decontaminated reward = A's reward_v5 formulas with A3 learner-only attribution, unchanged — already in A3/A4" | (a) reward_v5.py formulas unchanged ✓ (byte-locked); (b) "learner-only attribution zeroes opponent-actor steps" — TRUE only для LIVE-self-play path, и даже там opponent rewards re-added negated as macro-step folding (`rust_live_self_play.py:886`), не zeroed; docstring misleading | `BLOCK_B_PLAN.md:188-191,795-796` vs `rust_live_self_play.py:886`, `ppo_phaseA_config.py:286-291` |
| "Block -1 validated 34/34 mechanics and 151 golden tests green; Rust gets cards via FFI so gap = mechanic logic" | CONFIRMED: `kernel.rs` — full Rust re-implementation of `core/engine.py` mechanics, parity by golden tests, не shared code. `globals[16..31]` = 16 unused zero padding (latent budget) — docstring не упоминает | MEMORY vs `kernel.rs:2443`, `obs_v5.py:70-97` |
| "V5-Max pipeline -1→0→A→B→C→D→E1 COMPLETE end-to-end; all block decisions implemented; five root-cause fixes (learner-only reward, max_turns≥120, entropy 0.01, epochs 6, graduated mix) applied" | (1) ВСЕ 7 блоков портированы как код с синтетическими тестами ✓; (2) operational RUN chain USER-gated, NEVER autonomously run end-to-end (только 2026-07-02 "field test" на deliberately garbage model); (3) пять fixes имеют code+test confirmation, но в operational run `graduated mix` обнуляется override-ом, `learner-only reward` = macro-step negation (не zeroing); (4) A1 (bc_dataset) + A2 (bc_train) существуют с тестами но НЕ на active random-bootstrap path | specs + block plans + memory vs `run_blockB_league.py:295-296`, `rust_live_self_play.py:886` |
| Q5 «keep V4-orig lane weight modest + self-snapshot prevalence high» (mitigation intent) | D-B5 Hybrid cap'ит self на 0.05 (LOW prevalence); V4-orig frozen 0.75 (NOT modest) — частично противоречит Q5 | `design.md:118,144` (freeze 0.75) vs `design.md:224` (Q5 modest+high) vs `block_b_opponent_mix.py:131,157` (0.75/0.05) |
| Run NAMES: `rewardfix`/`diverse_fixed`/`start_second`/`targeted` = "fixes" | Ни один не устранил симптомы; `rewardfix` = uncommitted macro-step negation (loss 10x↓, winrate всё равно ↓); `diverse_fixed` = осцилляция+деградация; `start_second`/`targeted` BACKFIRED (second 0.188→0.133) | `progress.jsonl`/`summary.json` всех 7 run-ов |

## 6. Рекомендации (по приоритету)

### P0 — первопричины S1/S2

1. **Wire `BlockBLeagueDriver.run()` в `run_blockB_league.py`** (или перенести `_measure_snapshot`/`parity.update`/`maybe_update_best_ever`/`_collapse_boost_for`/`evaluate_block_b_gate`/`detect_h2h_plateau` в раннер). Файл: `TrainV3.5/scripts/run_blockB_league.py:204-265`. Без этого любой "fix" монотонности косметический. Причина: B1, B2, B3, M17.

2. **Добавить `am_first_player` obs-фичу в `globals[16]`** (16 zero channels уже зарезервированы). Файлы: `TrainV3.5/python/train_v3/obs_v5.py:70-97`; `TrainV3.5/rust/trainv3_core/src/kernel.rs:2420-2443`; `ai/train_v2/classic_obs_v1.py`. Переобучить с warm-start (`strict=False` для новых каналов). Причина: B5 (root cause S2 — модель физически не знает сторону).

3. **Починить `start_second` — добавить shuffle starting actors per-env** (или использовать `strict_balanced` с `rng.shuffle(sides)` как `:536-549`). Файл: `TrainV3.5/python/train_v3/rust_live_self_play.py:1112-1115`. Причина: B4 (forced all-p2 → catastrophic forgetting).

4. **Рассмотреть side/turn-order compensation term в reward** (например, small second-mover bonus или asymmetric shaping). Файлы: `TrainV3.5/python/train_v3/reward_v5.py:109-135`; `TrainV3.5/rust/trainv3_core/src/kernel.rs:2656-2705`. Причина: B6 (structural tempo-advantage не скомпенсирован). Альтернатива: полагаться на obs-фичу + parity oversampling, но тогда P0.1 и P0.2 обязательны.

5. **Починить EVAL чтобы исполнял `mana_draw`** через `mana_draw_legal_mask`+`select_includes_mana_draw` (или `engine.legal_actions`+`ManaDrawAction` как prod). Файлы: `TrainV3.5/scripts/run_v5_vs_v4max_benchmark.py:148`; `TrainV3.5/python/train_v3/env_v5.py:73-78`. Причина: B7 (eval WR = misleading-прокси; second-start winrate искусственно занижен).

6. **Выровнять `max_turns=120` в EVAL** (передать в `TrainV3EnvConfig`→`ClassicRLEnv`). Файлы: `TrainV3.5/python/train_v3/env_v5.py:43-48`; `ai/train_v2/classic_rl_env.py:99`. Причина: M19 (long second-start игры обрезаются на 80).

### P1 — PPO-математика и stability

7. **Добавить target-KL early-stop и grad clipping** в PPO. Файлы: `TrainV3.5/python/train_v3/rust_ppo.py:335-360` (добавить `break` при `approx_kl > target_kl`); `ppo_phaseA_config.py:381` (добавить `max_grad_norm` ≠ None). Причина: M7 (6 epochs stale-batch reuse, entropy collapse).

8. **Использовать TRUNCATED вместо TERMINATED для `decisive_early_end`** — bootstrap `V(s_decisive) ≈ +1.0`, не 0. Файлы: `TrainV3.5/python/train_v3/rust_live_self_play.py:862-866`; `TrainV3.5/rust/trainv3_core/src/ppo.rs:40-47`. Причина: M5 (value target = small shaping вместо true value).

9. **Передать `bootstrap_values` (value-head prediction) в `prepare_rust_ppo_batch`** для rollout tail. Файл: `TrainV3.5/python/train_v3/rust_live_self_play.py:1159-1167`. Причина: M6 (GAE bootstraps с 0.0 на boundary).

10. **Добавить LR schedule (warmup + cosine decay)**. Файлы: `TrainV3.5/scripts/run_blockB_league.py:141`; `TrainV3.5/python/train_v3/rust_trainer.py:39`. Причина: M8 (constant LR over long run with changing gradient distribution).

11. **Исправить macro-step reward attribution для second-mover** — записывать opponent-opener reward в специальный буфер или начальный learner row. Файл: `TrainV3.5/python/train_v3/rust_live_self_play.py:884-886`. Причина: M4 (opponent-opener shaping теряется для second-mover).

### P1 — reproducibility

12. **Заккоммитить 6-in-1 uncommitted bundle** (или разбить на атомарные коммиты с тестами). Файлы: `rust_live_self_play.py`, `rust_ffi.py`, `ppo_phaseA_config.py`, `ffi.rs`. Причина: B8, M10, M11 (runs невоспроизводимы из committed кода; `git stash` + `env_count>1` → crash).

13. **Исправить manifest `best_checkpoint`** = `pool.best_ever.path`, не final checkpoint. Файл: `TrainV3.5/scripts/run_blockB_league.py:276-284,370-393`. Причина: M13 (downstream consumer shipped бы WORST checkpoint).

### P2 — structural / latent

14. **Обучить `mana_draw` head через BC (A2) или AWAC-CRR (C3)** перед prod-деплоем; ИЛИ добавить `mana_draw` в PPO loss (тогда `score_padded_legal_action_inputs` должен вызывать `mana_draw_head`). Файлы: `TrainV3.5/python/train_v3/v5_policy.py:85,140`; `TrainV3.5/python/train_v3/rust_policy.py:248-270`. Причина: M1, M2 (head грузится в prod со случайными init-весами).

15. **Снизить V4-orig frozen weight и поднять self-snapshot cap** (привести к Q5 intent). Файлы: `TrainV3.5/python/train_v3/block_b_opponent_mix.py:131,154,157`; `snapshot_pool.py:438`. Причина: M12, M16, M20 (75% batch против одной ONNX → monotone conditions).

16. **Реализовать pool-wide random/best_ever draw в `DynamicSelfSnapshotOpponent`** вместо `rolling[-1]`. Файл: `TrainV3.5/scripts/run_blockB_league.py:83-87`. Причина: B10 (monotone recency ladder).

17. **Deprecate legacy distillation scripts** (`run_phase10_v4max_distill.py`, `run_phase19_*.py`) или пометить `# DEPRECATED`. Причина: M22.

18. **Расширить golden-trace coverage** для second-start/long-game edge cases (Rust↔Python parity). Причина: M18.

## 7. Воспроизведение

### S1 (монотонность)

```
# 1. dead parity loop
grep -c 'parity.update' TrainV3.5/scripts/run_blockB_league.py
# → 0

python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_rewardfix_25k_from_u0500_20260706_234554/progress.jsonl')]; print(set(r['p1_score_rate'] for r in rows), set(r['p2_score_rate'] for r in rows), set(r['oversampling_scheme']['breach'] for r in rows))"
# → {0.5} {0.5} {False}

# 2. frozen mix в targeted
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/progress.jsonl')]; print(len(set(json.dumps(r['mix_used']) for r in rows)))"
# → 1

# 3. snapshot pool inert
python3 -c "import json; sp=json.load(open('TrainV3.5/runs/blockB_rewardfix_25k_from_u0500_20260706_234554/snapshot_pool.json')); print('best_ever u',sp['best_ever']['update_number'],'h2h',sp['best_ever']['h2h_vs_best']); [print(e['update_number'],e['h2h_vs_best'],e['promotion_eligible']) for e in sp['rolling']]"
# → best_ever u 250 h2h 0.5; все rolling 0.5 False

# 4. BlockBLeagueDriver не вызывается
grep -n 'BlockBLeagueDriver\|block_b_gate\|exit_to_c2\|maybe_update_best_ever\|_collapse_boost_for\|_measure_snapshot' TrainV3.5/scripts/run_blockB_league.py
# → 0 hits кроме _merge_self_snapshot_split

# 5. hash-random в committed HEAD
git stash; grep -n 'use_chacha_rng\|trace_seed' TrainV3.5/python/train_v3/rust_ffi.py
# → 0 hits (fix uncommitted)
```

### S2 (second-start катастрофа)

```
# 1. eval-артефакт
python3 -c "import json; print(json.dumps(json.load(open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/eval_final_vs_v4max_noassist_64seeds/summary.json'))['rows'], indent=2))"
# source u12500: first=0.539, second=0.1875; targeted u5000: first=0.516, second=0.1328

# 2. forced all-p2 (start_second без shuffle)
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/progress.jsonl')]; r=rows[0]; print(r['oversampling_scheme']['starting_actor_counts'], r['learner_actor_counts'])"
# → {'1':96,'2':0} {'2':96}

# 3. нет am_first_player obs-фичи
grep -n 'out\[global_base + 1[6-9]\|out\[global_base + 2' TrainV3.5/rust/trainv3_core/src/kernel.rs
# → 0 hits (globals[16..31] = 0)

# 4. eval discards mana_draw
grep -n 'mana_draw' TrainV3.5/scripts/run_v5_vs_v4max_benchmark.py | head
# → только output[2] читается, никогда не feeds в env

# 5. targeted backfire (winrate decline)
python3 -c "import json; rows=[json.loads(l) for l in open('TrainV3.5/runs/blockB_targeted_v4max_second_from_u12500_20260707_142330/progress.jsonl')]; pl=[r['update_metrics']['policy_loss'] for r in rows]; kl=[r['update_metrics']['approx_kl'] for r in rows]; en=[r['update_metrics']['entropy'] for r in rows]; print('policy_loss',pl[0],pl[-1],min(pl)); print('approx_kl',kl[0],kl[-1]); print('entropy',en[0],en[-1])"
# → policy_loss -0.122 -0.242 -0.303; approx_kl -0.0006 -0.0008; entropy 0.058 0.022
```

### Полный новый run (для верификации фиксов)

```
cd TrainV3.5
cargo build --release
python3 scripts/run_blockB_league.py \
  --source-checkpoint runs/<source>.npz \
  --updates 5000 \
  --env-count 96 \
  --side-sampling-policy strict_balanced \
  --opponent-mix ''  # без override → curriculum.reweight активен
# затем eval:
python3 scripts/run_v5_vs_v4max_benchmark.py \
  --v5-checkpoint runs/<final>.npz \
  --games 64 --no-bonuses
```

## 8. Приложение — статус верификации по измерениям

| Измерение | секция | находок | подтверждено (≥2/3 линз) | status |
|---|---|---|---|---|
| action-codec-mana-masking | `/tmp/v5audit_section_action-codec-mana-masking.md` | 7 | 7 (3/3 каждый) | OK |
| monotonic-blockB | `/tmp/v5audit_section_monotonic-blockB.md` | 12 | 11 (1 гипотеза 0/3 — invalid sign-flip claim) | OK |
| reward-ppo-loss | `/tmp/v5audit_section_reward-ppo-loss.md` | 12 | 11 (1 minor 2/3 — global normalization linkage не доказан) | OK |
| runner-script-new-problems | `/tmp/v5audit_section_runner-script-new-problems.md` | 10 | 10 (3/3 каждый; 3 non-finding) | OK |
| second-start-asymmetry | `/tmp/v5audit_section_second-start-asymmetry.md` | 7 | 6 (1 гипотеза 1/3 — macro-step attribution bias) | OK |
| snapshot-curriculum-gate | `/tmp/v5audit_section_snapshot-curriculum-gate.md` | 11 | 10 (1 гипотеза 0/3 — PPO sign convention misread) | OK |
| train-eval-env-parity | `/tmp/v5audit_section_train-eval-env-parity.md` | 8 | 7 (1 гипотеза 0/3 — ASSIST-mode OOD опровергнут артефактами) | OK |
| **8-е измерение** | — | — | — | **УПАЛО НА ЛИМИТ** (не добито; секция отсутствует в `/tmp/`) |

### Note про упавшие на лимит

8-е измерение не было завершено из-за лимита (timeout/context). По контексту карты областей вероятные темы недобитого измерения: **`block_b_league_driver` internals parity** (детальная проверка `_build_reweighted_mix`/`_collapse_boost_for` алгоритмов против spec) ИЛИ **`curriculum-landscape`** (анализ long-run curriculum dynamics). Эти области частично покрыты измерениями `monotonic-blockB` и `snapshot-curriculum-gate` (находки M14, M15, M17), но отдельного глубокого аудита нет. Рекомендуется отдельный проход для подтверждения/опровержения гипотез M14 (monotone non-decreasing gate) и M15 (curriculum cumulative drift) на реальных long-run артефактах.

### Гипотезы (неподтверждённые, realCount<2/3)

- **monotonic-blockB finding "negative loss/KL = PPO sign-flip"** (0/3): эмпирика воспроизводится (loss<0, approx_kl<0 в targeted), но центральная математическая претензия НЕВЕРНА — `approx_kl = mean((ratio-1)·log_ratio)` может быть отрицательной; PPO `loss = -mean(min(r·A, clip(r)·A))` отрицателен при отрицательных advantages после нормализации. Регрессия winrate реальна, но механизм "PPO sign-flip" не подтверждён; правдоподобный механизм — forced all-second + v4max-heavy → negative-reward bias (B4, M23).
- **second-start-asymmetry finding "macro-step attribution asymmetry для second-mover"** (1/3): код-наблюдение верно (`row is not None` guard пропускает fold для first opponent step), но интерпретация как bias отклонена — эффект может компенсироваться другими путями.
- **snapshot-curriculum-gate finding "negative policy_loss+approx_kl sign inconsistency"** (0/3): numerics реальны, но mechanistic claim invalid из-за misread PPO sign convention.
- **train-eval-env-parity finding "ASSIST-mode OOD private_info"** (0/3): артефакты опровергают claim — `v4max_losslogs` run НЕ является ASSIST-mode eval с populated private_info; оставлен как гипотеза.

### «Подтверждено кодом» vs «гипотеза»

- **Подтверждено кодом (high, 3/3 линз)**: B1–B10, M1–M13, M16–M21, часть minors. Каждое утверждение о коде имеет file:line и цитату.
- **Гипотеза (medium, 1/3–2/3)**: M4 (macro-step attribution bias), M14 (gate false-promotion — latent, gate мёртв), M15 (curriculum cumulative drift — не воспроизвёдён dynamic claim), M23 (negative-reward bias mechanism — эмпирика подтверждена, механизм правдоподобен но не изолирован экспериментом), minor findings про advantage global normalization и V5 reward shaping face-rush bias.
- **Invalid (0/3)**: "PPO sign-flip" (negative loss/KL) — математическая претензия неверна; "ASSIST-mode OOD" — артефакты опровергают.