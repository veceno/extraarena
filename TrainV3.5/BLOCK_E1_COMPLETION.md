# Block E1 — V5-Max Tournament + Ship (the FINAL V5-Max stage) — IN-WORKTREE COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max Block E1 in-worktree components — the V5 ONNX exporter (E1) + the V5 inference guard + export tests (E2) + the E3 tournament harness + final-acceptance gate (E3) + the E4 human-QA panel SOFT gate (E4) + the E5 ship component + prod wiring (E5: vendored V5 encoders into `ai/train_v2/`, additive `BerserkInference` V5 branch, `extra-lr-v5-max` config profile + 4-top-tier retarget, LIFO V5 kind detector) + the E-E12 thin CLI runner composing E1–E5 (the runner). 6 components across two INDEPENDENT tracks (export E1/E2; tournament E3/E4) converging at ship (E5) + composed by the runner (E-E12); the operational RUN (A4 rust_live_self_play GameRunner + model_mlx candidate_loader + rlhf_env MCP c2_client + JsonScorecardClient + the real MLX/Rust wiring) is USER-run per E-E12 — the in-worktree code is built + synthetic-tested regardless.
**Status:** ALL 6 IN-WORKTREE COMPONENTS COMPLETE (2026-07-02). Each done + independently re-verified green via the ultracode workflow (implementer + 4 refute-by-default verifiers + fix + fresh re-verify) with 0 blocker/major remaining on each.
**Pipeline position:** `Block -1 -> 0 -> A -> B -> C -> D (COMPLETE) -> [D-league RUN -> D->E1 handoff] -> E1 (this, COMPLETE) -> [USER-run operational ship: load BERSERK_BRAIN with the V5 profile]`. **E1 IS THE FINAL V5-MAX STAGE** — there is no Block E2/F; after E1 the pipeline is shipped + the operational deploy is USER-run.

**Spec anchor** (`docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-design.md`): Block E1 = "Tournament + ship" — the winner of the E3 gauntlet (threshold-table gate) is exported to ONNX (3-output head: logits/value/mana_draw_logit) + bundled as `extra-lr-v5-max` + the V5 sidecar kind detector is registered LIFO ahead of the V4 detector + the prod wiring is verified. Ship is GATED on E2 (parity + fallback-guard) green + E3 (winner passes the threshold table).

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `a24039bb` | plan | `BLOCK_E1_PLAN.md` — Tournament + ship (solo-draft from a grounding fan-out + verify-only workflow; PASS-WITH-FINDINGS, findings merged) |
| `81e133ee` | decisions | Block E1 DECISIONS-CONFIRMED — E-E1..E-E14 (AskUserQuestion 2026-07-02) |
| `4dcfc7ad` | E1 | `export_onnx_v5.py` — V5 ONNX exporter (3-output head: logits/value/mana_draw_logit; split-encoder mirror of the training forward; `model_version="v5_split_encoder_onnx_v1"` + `mana_draw_head:true` + `format:"v5"` + `obs_dim:7128` sidecar; opset 17) |
| `2f95fb9e` | E2 | `v5_inference_guard.py` (vendored to `ai/train_v2/`) + `test_export_onnx_v5.py` — parity + fallback-guard (the SPEC :174 last-resort: a V5 ONNX producing NaN/garbage logits MUST raise RuntimeError, NOT silent rule-based fallback) |
| `597ddd6b` | E3 | `e1_tournament.py` — V5 tournament harness + final-acceptance gate (`run_e1_tournament` + `select_e1_winner` + `E1TournamentConfig` + `E1CandidateReport.passed()` threshold table; reads 11 run-artifact metadata fields from `loaded["metadata"]`) |
| `f239bce8` | E4 | `e1_human_qa_panel.py` — V5 human-QA panel driver, SOFT gate (`run_e1_human_qa_panel(candidates, *, c2_client, scorecard_client, min_reviewers, min_battles, battles_per_series=1000)`; `McpCollectionClient` + `ReviewerScorecardClient`/`JsonScorecardClient` Protocols; `HumanQAVerdict`; never aborts ship) |
| `ca31692a` | E5 | `e1_ship.py` (ship component) + PROD WIRING — vendored V5 encoders into `ai/train_v2/` (`obs_v5`+`v5_contracts`+`mana_draw_head_v5`+`v5_inference_guard`, the one `from .contracts`->`from ai.train_v2.v5_contracts` rewrite is the only intentional divergence; codec-sync invariant test guards byte-faithfulness) + additive `BerserkInference` V5 branch in `ai/bot_brain.py` (V4 byte-unchanged) + `extra-lr-v5-max` config profile + 4-top-tier retarget in `infrastructure/config.py` + LIFO V5 kind detector (`register_v5_kind_detector`, idempotent) |
| `0c111e1c` | E-E12 | `block_e1_runner.py` + `test_block_e1_runner.py` — the thin E-E12 CLI runner composing E1–E5 READ-ONLY (injectable `run_e1_pipeline` + thin `main` argparse CLI; operational pieces injected; SOFT panel gate; candidate.json writer for the release bundle) |

Combined suite (final, independently re-confirmed after the runner): **Block E1 python 83 passed** (runner 13 + ship 19 + export+guard + tournament + qa-panel), 0 failed. **Prod regression tests green** — `tests/test_regression_spec.py` + `tests/test_arena_frontend_regressions.py` strengthened (top tiers → V5, non-top tiers stay V4; strengthens, not weakens). **Rust golden_kernel 47 passed / 0 failed** (Block E1 made ZERO `.rs`/`.toml` edits — the cargo baseline is unchanged from Block -1/0/A/B/C/D).

---

## 2. DECISIONS-CONFIRMED (AskUserQuestion 2026-07-02, `BLOCK_E1_PLAN.md:72-78`)

| Decision | Choice | Meaning |
|---|---|---|
| **E-E1** (export entry shape) | **NEW file `export_onnx_v5.py`** | a dedicated V5 exporter (NOT extending the V4 export path); 3-output head, split-encoder mirror |
| **E-E2** (prod inference path) | **Extend `BerserkInference`** | additive V5 branch in `ai/bot_brain.py` (`_get_action_v5` + `_validate_v5_contract` + `_V5_FORMAT="v5"` + relaxed format gate); V4 `_get_action_train_v2_classic` + `_validate_train_v2_contract` byte-unchanged |
| **E-E3** (parity tolerance) | **1e-4 measure-then-pin** | measure the actual train↔ONNX divergence on a fixed batch, pin the observed max as the per-tensor rtol/atol (not a guessed constant) |
| **E-E4** (tournament GameRunner) | **LIVE Rust H2H + pre-baked secondary** | the primary GameRunner is the A4 `rust_live_self_play` (real engine H2H); a pre-baked secondary for offline replay fallback |
| **E-E5** (h2h_vs_self_snapshot gate) | **hard gate trending AND latest>=0.52** | the self-snapshot lane passes iff the trending series is non-decreasing across >=5 snapshots AND the latest h2h_vs_self_snapshot >= 0.52 |
| **E-E6** (no_bonus retarget) | **retarget no_bonus to self-snapshot >=0.70 each + V4-max secondary advisory** | the no_bonus benchmark is the self-snapshot lane (>=0.70 per side), with V4-max as a secondary advisory (not the primary gate) |
| **E-E7** (v4max/end_turn hard gates) | **hard gates >=0.95** | the v4max + end_turn lanes are hard gates at >=0.95 (no softness on the frozen-field lanes) |
| **E-E8** (human-QA panel) | **Hybrid (component + USER-run, soft gate)** | the panel is built as a component + synthetic-tested; the operational c2_client/scorecard_client are USER-run; it is a SOFT gate (never aborts ship) |
| **E-E9** (tournament harness) | **NEW `e1_tournament.py`** | a dedicated V5 tournament harness (NOT extending the Block-D league driver) |
| **E-E10** (ONNX head shape) | **ONNX 3-output head wired** | the ONNX exports 3 outputs (logits, value, mana_draw_logit); the prod `_get_action_v5` runs `session.run(["logits","value","mana_draw_logit"], ...)` |
| **E-E11** (profile scope) | **max-only `extra-lr-v5-max`** | ONE V5 profile (`extra-lr-v5-max`); only the 4 top tiers retarget; the 8 non-top tiers stay V4 (`extra-lr-v4-{micro,lite,opti}`) |
| **E-E12** (runner) | **Hybrid (thin runner + USER-run)** | `block_e1_runner.py` is a thin CLI shell composing E1–E5; the operational RUN (GameRunner/candidate_loader/c2_client/scorecard_client/mana_draw_baseline) is USER-executed; factory stubs raise NotImplementedError |
| **E-E13** (sidecar identity + detector) | **`model_version="v5_split_encoder_onnx_v1"` + `mana_draw_head:true` + `format:"v5"` + `obs_dim:7128`; LIFO detector load-bearing** | the V5 sidecar fingerprint; the V5 kind detector MUST run FIRST (LIFO insert-at-0) ahead of the V4 `_sidecar_kind_detector` (a V5 sidecar ALSO satisfies the V4 detector's `inputs`/`action_feature_dim` OR-branches, so without the V5 detector first it would misclassify as `action_onnx`) |
| **E-E14** (faithful-layer bonus) | **faithful-layer + base-1456 path identity + faithful LAYER-output bit-close bonus** | the E3 threshold table includes a faithful-layer bonus: base-1456 path identity (the non-top-tier V4 path is byte-identical) + a faithful LAYER-output bit-close bonus |

---

## 3. Components (what was built)

### E1. `export_onnx_v5.py` — V5 ONNX exporter (3-output head) — `4dcfc7ad`
The V5 exporter mirrors the training-time split-encoder forward and exports a 3-output ONNX (logits, value, mana_draw_logit). `export_v5_checkpoint_to_onnx(checkpoint_path, output_path, *, opset=17, placement_mode=None) -> str` returns the ONNX path + writes the `.onnx.json` sidecar carrying `model_version="v5_split_encoder_onnx_v1"` + `mana_draw_head:true` + `format:"v5"` + `obs_dim:7128` (the E-E13 fingerprint). opset 17.

### E2. `v5_inference_guard.py` + `test_export_onnx_v5.py` — parity + fallback-guard — `2f95fb9e`
`_assert_v5_logits_finite_legal(logits, legal_mask) -> int` raises `RuntimeError` on NaN/inf OR no-legal-candidate (the SPEC :174 last-resort prod safety: a malformed V5 ONNX MUST raise, NOT silently fall back to a rule-based action). The export tests cover parity (train↔ONNX, E-E3 measure-then-pin) + the guard.

### E3. `e1_tournament.py` — V5 tournament harness + final-acceptance gate — `597ddd6b`
`run_e1_tournament(config, *, game_runner, candidate_loader, side_runner=None) -> list[E1CandidateReport]` plays the `UNIFIED_GAUNTLET_ROSTER` (v4max, random, end_turn, best_self_snapshot, *EXPLOIT_AGENT_KINDS) via the injected `game_runner`; reads 11 run-artifact metadata fields from `loaded["metadata"]` (throughput, entropy, max_abs_kl, no_bonus_p1/p2/second, no_assist_score_rate, exploit_resistance_score_rate, h2h_vs_self_snapshot_history, p1_p2_gap, human_qa_verdict). `select_e1_winner(reports) -> Optional[E1CandidateReport]` returns the passer with the highest h2h_vs_v4_orig (ties break by candidate_path); None when no passer = NO-SHIP. `E1CandidateReport.passed()` is the threshold-table verdict (E-E5 trending+latest>=0.52, E-E6 no_bonus self-snapshot>=0.70 each, E-E7 v4max/end_turn>=0.95, E-E14 faithful-layer bonus). `E1TournamentConfig` (frozen: candidate_set + mana_draw_baseline REQUIRED; games_per_opponent=20, gauntlet_roster=UNIFIED_GAUNTLET_ROSTER, throughput_floor=12000, entropy_floor=0.70, max_abs_kl_floor=0.12). `make_default_candidate_loader(policy)` builds the production loader (`load_checkpoint(path, policy)` from `ai.train_v2.model_mlx`).

### E4. `e1_human_qa_panel.py` — V5 human-QA panel, SOFT gate — `f239bce8`
`run_e1_human_qa_panel(candidates: E1CandidateSet, *, c2_client: McpCollectionClient, scorecard_client: ReviewerScorecardClient, min_reviewers, min_battles, battles_per_series=1000) -> Dict[str, HumanQAVerdict]`. Iterates `_iter_candidate_paths` (post-D, post-C3, post-B; Nones dropped; dedup). `McpCollectionClient` Protocol (`rlhf_env/components/c2_collection_driver.py:46`): start_series/next_battle/list_v5_groups/get_v5_dataset_summary/get_v5_trace/list_battles. `ReviewerScorecardClient` Protocol (`e1_human_qa_panel.py:144`): submit_scorecard/list_scorecards; `JsonScorecardClient(path)` file-backed. `HumanQAVerdict` frozen (candidate_path, n_battles, n_reviewers, mean_difficulty_score, n_harder_than_baseline, verdict, freeform_notes, stop_condition_met). **SOFT gate (E-E8): a harder verdict is a soft pass, easier/inconclusive is a soft warn — it NEVER raises/blocks; the runner wraps it in try/except so a panel error does NOT abort ship.**

### E5. `e1_ship.py` + PROD WIRING — `ca31692a`
The ship component + the (committed-source, NOT call-time) prod edits. `ship_v5_winner(winner_report, *, onnx_export_fn: Callable[[str,str],str], bundle_config) -> ShipResult` (frozen): (a) NO-SHIP guard — `winner_report.passed()` must be True (else `RuntimeError`); (b) export ONNX into `candidate_dir/extra-lr-v5-max.onnx`; (c) `build_release_bundle(bundle_config)` (REUSED-AS-IS, format-agnostic); (d) `register_v5_kind_detector()` (LIFO, idempotent — `if v5_detector not in registry._detectors`); (e) verify prod wiring (`extra-lr-v5-max` in `BOT_MODEL_PROFILES` + 4 top tiers `brain_profile=="extra-lr-v5-max"` + `BOT_DIFFICULTY_PROFILES` derived `obs_dim==7128`); (f) verify the vendored fallback guard imports; return `ShipResult`. **ship_v5_winner does NOT write `candidate.json` — the runner does (build_release_bundle raises FileNotFoundError without it).**

**Vendoring decision (the plan's one unpinned gap, resolved):** the live `_get_action_v5` path needs the V5 obs encoder (7128-dim), not just `mana_draw_head_v5`. Verified the full set `{obs_v5, contracts→v5_contracts, mana_draw_head_v5, v5_inference_guard}` imports ONLY `core.*` + `ai.train_v2.*` (ZERO `train_v3.*` deps) → vendored cleanly into `ai/train_v2/` (mirroring the Block-0 `v5_card_shape_v1.py` precedent). `contracts.py` renamed to `v5_contracts.py` (avoid generic-name collision); `obs_v5`'s `from .contracts` rewritten to `from ai.train_v2.v5_contracts`. The codec-sync invariant test (`test_vendored_obs_v5_byte_faithful_to_train_v3`) guards byte-faithfulness (the one-line import rewrite is the ONLY intentional divergence).

**`ai/bot_brain.py` additive V5 branch (LIVE prod, 279 ins/25 del):** `_V5_FORMAT="v5"` (:38); relaxed format gate (:287) `if profile_format not in (_TRAIN_V2_FORMAT, _V5_FORMAT):` (unknown formats STILL SKIP); `_validate_v5_contract` classmethod (obs_dim 7128, 3-tuple logits/value/mana_draw_logit, mana_draw_head truthy); `v5` elif in `get_action` (after V4, before the raise); `_get_action_v5` (lazy-imports `ai.train_v2.{obs_v5,mana_draw_head_v5,classic_actions_1,v5_inference_guard}` + `core.actions.ManaDrawAction`; `session.run(["logits","value","mana_draw_logit"], ...)`; `_assert_v5_logits_finite_legal` whose `RuntimeError` PROPAGATES via `except RuntimeError: raise` BEFORE the generic `except Exception: return _legal_fallback` [SPEC :174 last-resort, NOT swallowed]; wires `mana_draw_legal_mask` + `select_includes_mana_draw`, returns the ManaDrawAction index when legal+logit-higher, else decodes the 601-best). **V4 `_validate_train_v2_contract` + `_get_action_train_v2_classic` byte-unchanged** (the 25 deletions are the V4 session-dict relocated into an `else` branch, content-identical — V4 behavior provably unchanged; verified by the regression tests + the V4-path-unchanged test).

**`infrastructure/config.py` additive edits:** new `extra-lr-v5-max` profile (model_path `ai/models/extra-lr-v5-max.onnx`, format `v5`, obs_dim 7128, action_feature_dim 171, max_candidate_actions 601, mana_draw_head True, placement_mode append_only, verify_mask False); retarget ONLY the 4 top tiers (tier_hard_4500, tier_hard_plus_6000, tier_max_minus_7500, tier_max_9000) `brain_profile` from `extra-lr-v4-max` to `extra-lr-v5-max`; the 8 non-top tiers stay `extra-lr-v4-{micro,lite,opti}`; `BOT_DIFFICULTY_PROFILES` derivation auto-propagates (NO edit).

**V5 kind detector (LIFO, load-bearing — E-E13):** `register_v5_kind_detector(registry=None)` registers `v5_detector` via `registry.register_detector(v5_detector)` (insert-at-0, sits AHEAD of `_sidecar_kind_detector`). `v5_detector` returns `"v5"` for V5 sidecars (`model_version=="v5_split_encoder_onnx_v1"` OR `obs_dim==7128`+`mana_draw_head`+`format=="v5"`), None for V4 (delegates to `_sidecar_kind_detector` → `action_onnx`). Idempotent guard. Does NOT re-register the already-taken V5 factory slot (`policy_adapters.py:411` `_factory_v5_real`).

### E-E12. `block_e1_runner.py` + `test_block_e1_runner.py` — the thin CLI runner — `0c111e1c`
The composition shell. `run_e1_pipeline(manifest, *, game_runner, candidate_loader, c2_client, scorecard_client, mana_draw_baseline, bundle_config, min_reviewers, min_battles, no_bonus_benchmark_json_path=None, battles_per_series=1000, run_panel=True, onnx_export_fn=export_v5_checkpoint_to_onnx) -> Optional[ShipResult]` — the INJECTABLE core: (a) `build_e1_candidate_set_from_manifest` (reconstruct `E1CandidateSet` from flat `candidate_paths` [post-D, post-C3, post-B]; None/short/empty+best_ever-fallback; accepts dict OR `BlockDLeagueManifest`); (b) `E1TournamentConfig`; (c) `run_e1_tournament`; (d) `select_e1_winner` (None → NO-SHIP → return None, NOT raise); (e) `run_e1_human_qa_panel` wrapped in try/except (SOFT — log + continue, never aborts ship); (f) `write_candidate_json` (the release bundle REQUIRES it; BEFORE `ship_v5_winner`); (g) `ship_v5_winner`. ALL operational pieces injected (NO MLX/Rust/ONNX/rlhf_env constructed by the runner). `main(argv)` thin argparse CLI; 4 `build_production_*` factory stubs raise NotImplementedError referencing the operational wiring (A4 `rust_live_self_play`, `model_mlx.load_checkpoint`, rlhf_env MCP, `JsonScorecardClient`) — the real RUN is USER-executed per E-E12. sys.path uses `TrainV3.5/python` (NOT the broken `TrainV3` path in `run_v5_acceptance.py:16`). 13 synthetic tests green (candidate-set reconstruction ×5, candidate.json write, ship-a-passer, no-ship-on-no-passer, no-ship-on-no-candidates, panel-soft-does-not-block-ship, skip-panel, composition-order, main CLI parse).

---

## 4. Constraints honored

1. **Prod runtime must not import `TrainV3.5` AND must not import `rlhf_env`.** The live hot path `ai/bot_brain.py` imports ONLY `ai.train_v2.*` (vendored V5) + `core.*` — ZERO `train_v3`/`rlhf_env` imports. `e1_ship.py`/`block_e1_runner.py` are TrainV3.5-side so MAY import `rlhf_env.policy_adapters` + `ai.train_v2.release_bundle` + `infrastructure.config`. Verified by `test_no_forbidden_import_in_bot_brain` + grep over `ai/ webapp/ infrastructure/` (0 prod imports of `block_e1_runner`).
2. **Frozen-classic Python files byte-frozen** for V4-orig ONNX — NOT monkeypatched at runtime.
3. **V4 path byte-unchanged.** `ai/bot_brain.py` V4 `_validate_train_v2_contract` + `_get_action_train_v2_classic` byte-unchanged; the 8 non-top tiers unchanged; the regression tests assert non-top tiers stay V4.
4. **Rust NOT touched.** ZERO `.rs`/`.toml` edits in Block E1 (`git diff --stat 4dcfc7ad^..0c111e1c -- '*.rs' '*.toml'` empty); 157 cargo tests green baseline unchanged.
5. **ONNX fallback guard (SPEC :174) = last-resort prod safety.** A V5 ONNX producing NaN/garbage logits MUST raise `RuntimeError`, NOT silently fall back to a rule-based action. Wired into `_get_action_v5` via the vendored `v5_inference_guard`; `except RuntimeError: raise` BEFORE the generic `except Exception: return _legal_fallback`.
6. **LIFO V5 detector load-bearing (E-E13).** Without the V5 detector first, a V5 sidecar would misclassify as `action_onnx` via the V4 detector's `inputs`/`action_feature_dim` OR-branches.
7. **mana_draw wiring mandatory.** Shipping a V5 ONNX with a mana_draw head WITHOUT wiring it silently drops mana_draw = ship-correctness blocker. `_get_action_v5` wires `mana_draw_legal_mask` + `select_includes_mana_draw` + returns the `ManaDrawAction` index when legal+logit-higher.
8. **Additive-only prod edits.** V4 path byte-unchanged, non-top tiers unchanged, `BOT_DIFFICULTY_PROFILES` derivation untouched.
9. **NO-SHIP returns None (not raise)** in the runner; `ship_v5_winner` itself raises on a non-passer (the runner never reaches ship with a None winner — it short-circuits at `select_e1_winner` None first).

---

## 5. Ultracode discipline (per-component)

Each of the 6 components followed the DURABLE per-component pattern: **solo-draft + verify-only** for the plan; **implementer + 4 refute-by-default critics + fix + FRESH re-verify** for each component. The FRESH re-verify re-runs (not resume) so critics re-read the revised files. ASCII-only workflow scripts (`.claude_tmp_*.js`, NO function declarations, NO backtick template literals, string-concat prompts, single-quote-open/single-quote-close), `node --check` + hazard scan, invoke via `scriptPath`, delete temp before commit. The DURABLE lesson applied: single-quote strings close with single-quote; `node --check` is NOT a reliable gate for the Workflow parser (scan for `+ '` lines); avoid nested-quote concatenation; no `python3 -c` inline.

**finalGreen=true** on each component's workflow (verdicts via the schema's `overall="PASS"` field, not text parsing). 0 blocker/major across all verify + re-verify rounds. The one minor finding on the runner (the `build_e1_candidate_set_from_manifest` docstring claimed "None preserved in slot" but a real manifest drops Nones — functionally harmless since downstream consumers use ordered+deduped paths, not slot labels) was fixed with a docstring-accuracy clarification.

---

## 6. Transition — the operational ship (USER-run)

Block E1 is the FINAL V5-Max stage. The in-worktree code is COMPLETE + synthetic-tested. The operational deploy is **USER-run** with prod-safety checkout (per E-E8 + E-E12 = Hybrid):

1. **D-league RUN** (USER): D3 seed pool -> BlockDLeagueDriver consolidation -> D->E1 handoff carrying `BlockDLeagueManifest.candidate_paths` + `best_ever_path`.
2. **E1 RUN** (USER): `block_e1_runner.py main` with the operational factories wired (A4 `rust_live_self_play` GameRunner, `model_mlx.load_checkpoint` candidate_loader, rlhf_env MCP c2_client, `JsonScorecardClient`, production-measured `ManaDrawBaseline`) -> E3 gauntlet -> E4 panel (SOFT) -> `ship_v5_winner` -> `extra-lr-v5-max` release bundle.
3. **Prod deploy** (USER, prod-safety checkout): load `BERSERK_BRAIN` with the `extra-lr-v5-max` profile; the ONNX fallback guard (SPEC :174) is the last-resort defense against a malformed V5 ONNX.

The V5-Max pipeline (`Block -1 -> 0 -> A -> B -> C -> D -> E1`) is now COMPLETE end-to-end in the in-worktree code. The remaining work is operational (USER-run runs + deploy), not in-worktree code.