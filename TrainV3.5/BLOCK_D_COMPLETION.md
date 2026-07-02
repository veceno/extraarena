# Block D — V5 League-2 consolidation (post-C self-play settle + D->E1 handoff) — IN-WORKTREE COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max Block D in-worktree components — the consolidation opponent-mix builder (D1) + the C->D handoff (D3: fresh pool seeded from post-C + E1 candidate threading) + the Block-D league driver (D2: subclass overriding `_build_reweighted_mix` + `run()`, emits the D->E1 handoff). 3 components, dependency-ordered (D1 -> D3 -> D2), synthetic-testable; the operational league RUN (D3.build_block_d_seed_pool -> BlockDLeagueDriver.run -> d2_manifest) is USER-run glue gated on the A-gate / C-loop PASS carrying a `CLoopManifest.best_ever_path` — the in-worktree code is built + synthetic-tested regardless.
**Status:** ALL 3 IN-WORKTREE COMPONENTS COMPLETE (2026-07-02). Each done + independently re-verified green via the ultracode workflow (implementer + 4 refute-by-default verifiers + fix + fresh re-verify) with 0 blocker/major remaining on each.
**Pipeline position:** `Block -1 -> 0 -> A -> B -> C -> D (this, COMPLETE) -> [D-league RUN: D3 seed pool -> D2 consolidation -> D->E1 handoff] -> E1 (tournament + ship)`.

**Spec anchor** (`docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-design.md:130-131`): Block D = "Short self-play among {best post-C checkpoint, post-B/seed anchors, V4-orig spectrum, exploit-lanes} to settle post-C and prevent overfit to the last replay batch. Promotion by external bench." Block D is online self-play on the Online path (`:164`), NOT the offline/human paths.

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `cdc58602` | plan | `BLOCK_D_PLAN.md` — League-2 consolidation (solo-draft from a 5-agent ultracode grounding fan-out + verify-only workflow: FAIL->revised, 1 blocker + 2 majors + 9 minors merged surgically) |
| `a07b9828` | decisions | Block D DECISIONS-CONFIRMED — D-D1..D-D4 (AskUserQuestion 2026-07-02) |
| `87695d24` | D1 | `block_d_opponent_mix.py` — the consolidation opponent-mix composition (D-D1 load-bearing: self+v5_snapshot 0.50, V4-orig 0.30, exploit 0.15, tail 0.05; explicit group shares; always sums to 1.0 incl. the degenerate pure-self-play branch) |
| `072328cf` | D3 | `c_to_d_handoff.py` — C->D seed pool + E1 candidate threading (D-D2 fresh pool seeded from post-C; `E1CandidateSet` frozen dataclass; `build_block_d_seed_pool` + `thread_e1_candidates`) |
| `4a7a2098` | D2 | `block_d_league_driver.py` — Block-D consolidation driver + D->E1 handoff (`BlockDLeagueDriver(BlockBLeagueDriver)` overrides `_build_reweighted_mix` + `run()`; `BlockDLeagueManifest` exited_to_e1 + candidate_paths + aggregate_history) |

Combined suite (final, independently re-confirmed after D2): **python 40 passed** (D1 12 + D3 11 + D2 17), 0 failed. Sanity D2+B8+D1+D3 = 52 passed. **Rust golden_kernel 47 passed / 0 failed** (Block D made ZERO `.rs`/`.toml` edits — `git diff --stat cdc58602^..4a7a2098 -- '*.rs' '*.toml'` is empty; the cargo baseline is unchanged from Block -1/0/B/C).

---

## 2. DECISIONS-CONFIRMED (AskUserQuestion 2026-07-02, `BLOCK_D_PLAN.md:49`)

| Decision | Choice | Meaning |
|---|---|---|
| **D-D1** (the Block-D opponent mix, load-bearing) | **Consolidation ~0.50** | self+v5_snapshot 0.50, V4-orig 0.30, exploit 0.15, tail 0.05 (explicit group shares; the OPPOSITE of B3's 0.05-cap weak-learner-vs-frozen-field profile). D1 `self_share_target=0.50`. |
| **D-D2** (pool coexistence, Block-B Q10) | **Fresh pool seeded from post-C** | new `SnapshotPool`; `set_seed_anchor(post-C)` = immutable seed + inaugural best-ever; post-B peers as rolling non-anchors (FIFO-eligible sparring partners, NOT permanent anchors). Block-D best-ever tracks Block-D improvement, not Block-B history. |
| **D-D3** (Block-D exit condition, D->E1) | **Fixed short schedule** | run small `n_updates`, B6-promote at `snapshot_cadence`, exit->E1 at schedule end carrying `pool.best_ever.path`; NO plateau early-exit (a plateau early-exit is redundant for a short consolidation). `exit_mode="fixed_schedule"` (default). |
| **D-D4** (per-lane-loss reweight) | **Off / no reweight** | no-op curriculum `cap=0.0`; mix shape fixed by D-D1. `curriculum_off=True` (default). |

All four = the recommended defaults. Execution D1->D3->D2 proceeded with these frozen.

---

## 3. Components (what was built)

### D1. `block_d_opponent_mix.py` — the consolidation opponent-mix composition (D-D1, load-bearing) — `87695d24`
The load-bearing new code of Block D. B3 `build_block_b_opponent_mix` (`block_b_opponent_mix.py:276-325`) hard-codes a Block-B profile (self+v5_snapshot capped at 0.05, V4-orig 0.75, exploit 0.15, tail 0.05) — a weak-learner-vs-frozen-field league. Block D is the OPPOSITE: the post-C learner is already strong and should settle against its strong PEERS (self/peer-HEAVY ~0.50) with V4-orig spectrum + exploit-lanes as smaller PROBES.

- Public API: `build_block_d_opponent_mix(pool, *, self_share_target=0.50, v4_orig_share=0.30, exploit_share=0.15, tail_share=0.05, collapse_boost=1.0) -> list[(name,weight)]`; `build_block_d_mix_string`; `parse_block_d_opponent_mix` (= `parse_block_b_opponent_mix` re-exported — `BLOCK_D_IDENTITIES` IS `BLOCK_B_IDENTITIES`, the SAME 11-name set, so no new dispatch identities / no A4 edit); `collapse_reweight_boost` (re-export); `BLOCK_D_IDENTITIES`; `BLOCK_D_MAX_SELF_SHARE=0.95`.
- `self_share_target` is applied DIRECTLY (NOT via `pool.self_snapshot_prevalence_weight()` — that B1 method is the 0->0.05 ramp for B3's weak-learner regime; Block D's pool is pre-seeded with post-C + post-B anchors by D3, so the peer field is populated from update 1 and `self_share_target` applies as-is). The within-group B3 frozen RATIOS are reused verbatim (V4-orig 0.40/0.20/0.15, exploit 0.05/0.05/0.05, tail 0.03/0.01/0.01). Self-snapshot split keeps `_self_snapshot_split` (half `self`, half `v5_snapshot`).
- **The mix ALWAYS sums to 1.0** (`test_mix_sums_to_one` iterates `self_share_target` in {0,0.25,0.5,0.75,1.0} x prevalence in {0.0,0.05} x `collapse_boost` in {0.5,1.0,2.0,4.0}). The load-bearing degenerate fix: when `non_self_target_total == 0.0` (pure-self-play, `self_share_target == 1.0` with all non-self shares 0) the boost is MEANINGLESS (no non-self lanes to compress), so `self_snapshot_weight = 1.0` is FORCED (NOT `self_share_target * collapse_boost`, which would sum to `collapse_boost != 1.0` and breach the contract for every `collapse_boost != 1.0`). Round-2 verify caught a REAL major an earlier draft introduced here (an unbounded `self_share_target * collapse_boost` summed to 2.0 at sst=1.0, cb=2.0) — the fix + the extended `collapse_boost` grid that hid it validated the durable fresh-re-verify pattern.
- D1 emits BOTH `self` and `v5_snapshot`; `v5_snapshot` is NOT dispatchable, so D2's `_build_reweighted_mix` override MUST wrap the result in `_merge_self_snapshot_split(...)`.
- Tests: 12 SYNTHETIC tests green (mix sums to 1.0 across the grid; self+v5_snapshot share == `self_share_target` exact; group-share assertion fires on a bad sum; within-group ratios preserved; `collapse_boost` raises self-share capped at 0.95; `test_block_d_mix_not_block_b_frozen` consolidation-vs-frozen-field guard at a full pool; `test_pool_independence_d1_does_not_read_prevalence`; identity guards `collapse_reweight_boost is _b3_collapse_reweight_boost`, `parse_block_d_opponent_mix is _b3_parse_block_b_opponent_mix`, `BLOCK_D_IDENTITIES is BLOCK_B_IDENTITIES`).

### D3. `c_to_d_handoff.py` — C->D seed pool + E1 candidate threading (D-D2) — `072328cf`
Sits BETWEEN the C-loop driver (C4) and the Block-D consolidation driver (D2): the C-loop emits a `CLoopManifest` carrying `best_ever_path`; D3 turns that into (a) a FRESH `SnapshotPool` seeded from the post-C best-ever (D-D2) and (b) the threaded `E1CandidateSet` handed to E1 at the Block-D exit.

- `E1CandidateSet` (frozen @dataclass, `design.md:134`): `{post_d_path: str|None, post_c3_best_path: str|None, post_b_path: str|None}` all default None; `with_post_d(path) -> E1CandidateSet` via `dataclasses.replace` (the original is never mutated — D2 builds the post-D-filled copy at Block-D exit). SINGLE-TYPED: D2 passes the `E1CandidateSet` OBJECT (NOT raw strings) when filling post-D.
- `build_block_d_seed_pool(c_manifest, *, post_b_anchor_paths=None, post_c3_best_path=None, h2h_vs_best=0.0, p1_p2_gap=0.0) -> SnapshotPool`: constructs a FRESH `SnapshotPool`; `set_seed_anchor(SnapshotEntry(path=c_manifest.best_ever_path, update_number=0, h2h_vs_best=..., p1_p2_gap=..., promotion_eligible=True))` (role OMITTED — `set_seed_anchor` reconstructs with SEED_ROLE internally; `update_number=0` is the REQUIRED field with no default) so post-C is the immutable seed + inaugural best-ever; `add_snapshot(..., role="rolling")` for each post-B peer (FIFO-eligible sparring partners, NOT anchors — INTENTIONAL: the pool non-anchor slot count is bounded). Raises `ValueError` if `c_manifest.best_ever_path is None` (C-loop skipped case — surfacing, NOT silent).
- `thread_e1_candidates(c_manifest, post_b_path=None) -> E1CandidateSet`: captures post-C3 (`c_manifest.best_ever_path`) + post-B; post-D None until D2 fills it.
- Tests: 11 SYNTHETIC tests green (post-C is seed anchor + inaugural best-ever; `test_seed_pool_threads_h2h_and_p1p2_baseline_into_anchor_and_best_ever` — the load-bearing-for-promotion baseline that a hardcoded-0.0 regression would miss; post-B are rolling non-anchors NOT anchors; seed-anchor immutability `RuntimeError` on a second `set_seed_anchor`; None `best_ever_path` surfaces `ValueError`; `E1CandidateSet` frozen + `with_post_d` returns a NEW set with the original unchanged; defaults all-None).

### D2. `block_d_league_driver.py` — Block-D consolidation driver + D->E1 handoff — `4a7a2098`
`BlockDLeagueDriver(BlockBLeagueDriver)` subclass overriding TWO surfaces (the inherited `run()` cannot be used verbatim — it builds `BlockBLeagueManifest` at `:562` and early-returns on B7 `exit_fires` setting `exited_to_c2` at `:624-628`; none of the `exited_to_e1` / `candidate_paths` / `block_d_schedule_complete` semantics D2 requires):

- **`_build_reweighted_mix` override** (`block_b_league_driver.py:400-424`): `build_block_d_opponent_mix(self.pool, self_share_target=self.self_share_target, collapse_boost=boost)` -> `curriculum.reweight(mix, cap=0.0 when curriculum_off [D-D4 OFF], cap=0.25 otherwise)` -> `_merge_self_snapshot_split(reweighted)` (IMPORTED from B8 `:670-700`, NOT redefined) so `v5_snapshot` is merged into `self` BEFORE the mix reaches A4 (without this merge A4 `resolve_opponent_dispatch` raises on `v5_snapshot` — absent from `POLICY_OPPONENT_KINDS`/`BLOCK_B_POLICY_OPPONENT_KINDS`/`RULE_AGENT_CODES`). `collapse_boost` threads from the inherited `_collapse_boost_for(_learner_mana_draw_rate())` (the D-B5 mana_draw-collapse monitor; 1.0 when no data/baseline).
- **`run()` override** (`:550-633`): COPIES the B8 inline per-update loop body (`:564-628`; B8 has NO factored `_run_update` helper and `block_b_league_driver.py` is not editable per plan section 4, so the loop is duplicated and kept in sync — divergence risk noted, regression guard `test_d2_loop_matches_b8_per_update_steps`). `_snapshot_step` + `_measure_snapshot` are genuinely INHERITED (called, not copied). BUT (a) constructs `BlockDLeagueManifest`; (b) under D-D3=fixed-schedule, calls `detect_h2h_plateau` per snapshot (inside the inherited `_snapshot_step`) for monitoring but does NOT act on `exit_fires` — the loop runs to `n_updates` completion, `exit_verdict['reason']=='block_d_schedule_complete'`; (c) at schedule end sets `exited_to_e1=True`, `candidate_paths` (post-D first, post-C3, post-B; Nones dropped), `aggregate_history` (fresh-seeded, NOT carried from CLoopManifest), `h2h_history`, `best_ever_path`. Under `exit_mode="plateau"` the B7 `below_target_exits=False` reading fires `reason='plateau_at_or_above_dominance_target'` (`exit_to_c2.py:232-236`) -> early D->E1 exit (NOT `dominant_plateau_e1_path`).
- **`__init__` additions** over B8: `self_share_target` (D-D1, default 0.50; validated in [0.0,1.0]); `exit_mode` ("fixed_schedule"|"plateau", D-D3, default "fixed_schedule"; validated); `curriculum_off` (D-D4, default True); `e1_candidate_set` (D3 `E1CandidateSet | None`). `below_target_exits` wired via `super().__init__`: `False` when `exit_mode="plateau"` (flipped at/above reading), `DEFAULT_BELOW_TARGET_EXITS` otherwise (the inherited `_snapshot_step:528` `detect_h2h_plateau` reads `self.below_target_exits`). The B8 `__init__` already sets `self._aggregate_history=[]`, `self._h2h_history=[]`, `self._last_rollout=None`, `self._opponent_policies=None` — inherited, not reset. `e1_candidate_set.with_post_d(best_path)` fills post-D at exit (frozen -> new set, reassigned to `self.e1_candidate_set`).
- **`BlockDLeagueManifest`**: a FRESH dataclass mirroring `BlockBLeagueManifest` fields + RENAMED `exited_to_c2`->`exited_to_e1` + NEW `candidate_paths` (the E1 tournament set, `design.md:134`) + `aggregate_history` (fresh-seeded `[]`). `to_dict()` covers all 11 keys.
- **Pool pre-seed requirement** (D-D2): the caller MUST `pool.set_seed_anchor(post-C entry)` BEFORE `run()` so the inherited first-snapshot `set_seed_anchor` branch (`:508-509`) is skipped and post-C (not the first Block-D snapshot) is the seed anchor — `test_pre_seed_skips_first_snapshot_seed_anchor` regression guard.
- Tests: 17 SYNTHETIC tests green (subclass-of-B8; mix uses D1 not B3 + no `v5_snapshot` reaches A4; `curriculum_off` -> `reweight` cap=0.0 (else cap=0.25); fixed-schedule exit fires at end with `block_d_schedule_complete` + `candidate_paths` (NOT a plateau exit, loop runs to completion); plateau mode early-exit with `plateau_at_or_above_dominance_target` (NOT `dominant_plateau_e1_path`); pre-seed skips first-snapshot seed anchor; B6 gate reused (NOT A5 a_gate); loop matches B8 per-update steps; `candidate_paths` threads the E1 set (post-D first); manifest `to_dict()` keys; invalid `exit_mode`/`self_share_target` raise; `aggregate_history` fresh-seeded).

---

## 4. Constraints honored (frozen-classic guard, `BLOCK_D_PLAN.md:88-94`)

- `block_b_league_driver.py` / `block_b_opponent_mix.py` NOT modified (B8/B3 completed + verified) — D1 is a NEW sibling module, D2 is a NEW subclass; the B8 `_build_reweighted_mix` + `run()` surfaces are overridden (not edited); `_merge_self_snapshot_split` is IMPORTED (not redefined).
- `c_to_d_handoff.py` NOT modified by D1/D2 (D3 owns it; D2 imports `E1CandidateSet` READ-ONLY).
- `classic_obs_v1`/`classic_actions_v1`/`classic_card_shape_v1`/`classic_rl_env.py`/`reward_v5.py` byte-frozen for V4-orig ONNX — NOT touched.
- `v5_trace.py` NOT modified + NOT imported into the D1/D2/D3 code path.
- `core`/`state`/`league_v5`/`gauntlet_v5`/`opponents_v5`/`rust_ffi`/`rust_ppo`/`rust_live_self_play` consumed READ-ONLY — D2 feeds `opponent_mix_parsed=` the D1 mix the same way B8 does (bypassing `parse_v5_opponent_mix`).
- NO Rust edit (confirmed: `git diff --stat cdc58602^..4a7a2098 -- '*.rs' '*.toml'` empty; cargo golden_kernel 47 passed / 0 failed).
- NO TrainV3.5-into-prod import.

---

## 5. Ultracode workflow discipline (DURABLE)

- **Plan-writer**: solo-draft + verify-only (5-agent grounding fan-out -> 4 refute-by-default critics + Finalize). Stalling fix from Block B. The Block D plan was solo-drafted, verify-only-merged (1 blocker + 2 majors + 9 minors), THEN D-D1..D-D4 presented via AskUserQuestion.
- **Per-component implementer + 4 refute-by-default verifiers + fix + FRESH re-verify** (re-run verify FRESH after edits, not resume, so critics re-read revised files). D1: round-1 PASS-WITH-FINDINGS (minors) -> round-2 FAIL (major I introduced: unbounded degenerate branch summed to 2.0) -> round-3 PASS-WITH-FINDINGS (cosmetic). D3: PASS-WITH-FINDINGS (2 minor) on first round. D2: round-1 PASS-WITH-FINDINGS (1 cosmetic temp-file cleanup) -> round-2 4/4 PASS.
- **ASCII-only workflow scripts** (`.claude_tmp_d{N}.js`), `node --check`, invoke via `scriptPath`, DELETE TEMP BEFORE COMMIT. No function declarations, backtick template literals, string-concat prompts (the `\\'` nested-quote hazard that broke D2's first parse -> replaced with quote-free instructions).

---

## 6. Transition to E1 (tournament + ship)

**Spec** (`docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-design.md:133-158`): Block E1 = Tournament + ship.
- **Candidates**: post-D, post-C3 (best C-iteration), post-B (if C added nothing) — exactly the `E1CandidateSet` D3 threads + D2 fills (`candidate_paths`, post-D first).
- **Eval**: Rust ArenaEnv gauntlet (retargeted to best self-snapshot) + a human-QA panel in rlhf_env (operationalizes "субъективно сложнее для человека").
- **Final acceptance thresholds** (adapt `run_v5_acceptance`/`gauntlet_v5`):
  - Absolute strength anchor (vs frozen V4-orig, asymmetric): H2H >= 0.70 (user's original 70-80% band, retargeted to V4-orig).
  - Sanity: H2H vs Random ~1.0; H2H vs end_turn ~1.0.
  - Progression (vs best self-snapshot): monotone improvement, ~0.52-0.55 over prior best (beating your own best by 70% is ill-defined); the plateau rule is the promotion driver.
  - no_bonus p1/p2/second (vs best self-snapshot) >= 0.70 each; no_assist_min_score_rate >= 0.55 (calibrate up); exploit_resistance_min_score_rate >= 0.50; p1_p2_max_score_gap <= 0.12; mana_draw usage-rate in [0.5x, 1.5x] human baseline; min_entropy/max_abs_kl >= 0.70 / <= 0.12; min_e2e_tps >= 12000; human-QA difficulty (subjective panel).
- **Ship**: export best -> ONNX + sidecar; register via `_derive_kind_from_sidecar`; deploy to prod arena (argmax); marker `extra-lr-v5-max`. Snapshots/temperature variants feed the next iteration's league pool.
- **Integrity footprint** (`:171-189`): ONNX fallback guard (`policy_factory` raises `RuntimeError` on silent rule-based fallback); **tighten V5 export-parity tolerance** (the existing `LOGIT_TOL=1e-3` is too loose — a real V5 export bug <1e-3 slips); warm-start parity (V5 601-scorer + base-1456 forward-pass bit-close to V4-Max immediately after load); rlhf_env codec sync (Block -1); V5 ONNX<->MLX export within tightened tolerance.

**Next step**: author `BLOCK_E1_PLAN.md` via ultracode (solo-draft + verify-only workflow), then present E-E* decisions via AskUserQuestion, then execute the E1 components (implementer + 4 verifiers + fix + fresh re-verify per component), commit + push, then E1_COMPLETION.md + ship.

**Open items carried into E1 planning**:
- The D-league RUN (D3 seed pool -> D2 consolidation -> D->E1 handoff) is USER-run operational glue gated on the A-gate / C-loop PASS carrying a `CLoopManifest.best_ever_path` — the same operational pattern as Block C (C4 was the loop driver but the operational rlhf RUN was USER-run per D-C0/D-C9). A thin `block_d_runner.py` wrapper for repeatability is an E1-prep operational artifact, NOT a Block-D component.
- D-D6 "short" duration (`n_updates` / `snapshot_cadence` sized for ~3-5 B6 promote checks) is an implementer default surfaced as confirm-only — the user may veto the "short" order of magnitude at E1-prep time.
- The latent-loader 'full' vs 'append_only' finding flagged from Block A for Block C is resolved (C0 `loader_append_only_fix`); E1 ships the chosen checkpoint, so the loader mode of the SHIPPED artifact is an E1 export concern.