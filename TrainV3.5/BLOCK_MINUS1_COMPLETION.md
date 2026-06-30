# Block -1 — Rust ArenaEnv 1-in-1 parity port — COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** port the legacy Rust training kernel (`TrainV3.5/rust/trainv3_core/`) to be byte-parity with the real game (`core/` + frozen `ai/train_v2/classic_*`), so V5 rollouts/league/acceptance run on a faithful environment.
**Status:** ✅ **COMPLETE** (2026-06-30). Validated PASS by focused source-vs-source re-verification.
**Pipeline position:** `Block -1 (freeze + Rust parity) → 0 (foundation) → A → B → C → D → E1`. This closes **Q1** (Rust ArenaEnv parity) of the V5-Max handoff open questions.

---

## 1. Commits (chronological)

| Commit | Phase | Summary |
|---|---|---|
| `662fbd5a` | P1 + P2 + #14 | weighted draw + RNG + state fields; board cap 5 + mana_draw action + parallel binary head; recorded-outcome RNG seed |
| `a16b6f52` | P3 | cleanup rebirth + crime_and_punishment + consume_ally |
| `7e1b5b14` | P4 | attack cleave + instant_kill + freeze + armor_X_Y range roll |
| `852e968c` | P5 | 5 new mechanic applicators + card52 playable BOTH sides (TAMHP) + additive `classic_actions_v1` mask_targets (user-authorized) |
| `9a020816` | P6 | mechanic-driven card15 (`battlecry_damage_\d+_random`) + card24 (`shield_refresh`) — remove card_id hardcodes; `choice_rolls` RNG stream |
| `f563a317` | P7 | terminal sudden-death + max_turns/truncation; BTreeMap for deterministic serde |
| `4fa33034` | P8 | `card_shape_v5.rs` (34-classic mirror) + CS-1 doc; escalated 2 frozen-classic violations |
| `9e936f47` | 5-slot obs | port obs encoder to 5-slot (catch Rust up to current Python); re-encode 5 fixtures |
| `69bfff34` | P9 + fixup | validation fixup (7 kernel fixes) + 6 missed mechanics + consume_ally MASK revert |
| `3339e870` | final fixes | `card_shape number_after_prefix` MAJOR + `apply_random_battlecry_damage` guard minor |

Suite: **151 tests green** (104 lib + 47 integration + 0 failed) vs baseline 26.

---

## 2. What was ported

**Engine dynamics (kernel.rs):**
- Weighted draw + anti-stuck + overdraw-to-discard (Phase 1); FIFO sites replaced.
- `apply_mana_draw` (Phase 2): cost = `MANA_DRAW_BASE=2*(count+1)` → 2/4/6…; **MD-5 = Python refunds mana on fatigue** (count not incremented). Parallel binary head (not a 602nd candidate) via `mana_draw_legal` flag + separate FFI channel.
- Board cap 5 (`GAME_BOARD_CAP`); codec layout stays `NUM_BOARD=7`.
- rebirth (pre-pass each cleanup iteration), crime_and_punishment (direct hp subtraction, bypasses armor/reflect), consume_ally (player-chosen target, bypasses cap) — Phase 3.
- cleave (unconditional), instant_kill, freeze, armor_X_Y `randint` roll — Phase 4.
- 5 new mechanic applicators (card47 aoe_silence, card48 team_wide_shield, card52 TAMHP ×3 modes) — Phase 5.
- mechanic-driven card15 random-battlecry-damage + card24 shield_refresh (hardcodes removed) — Phase 6.
- sudden-death (no threshold, fires t1 when enabled, direct hp subtraction), max_turns/truncation — Phase 7.
- 6 missed mechanics (aoe_damage_2, heal_target_3/5, delete_target, choose_shield_damage, cast_random_spell) — Phase 9. `card.level=1` always in training env confirmed.
- 7 validation fixes: att_pos /8→/6 (frozen `classic_actions_v1.py:589` uses `(_NUM_BOARD+1)`=6.0), mana_drain two-stage (immediate + pending end-turn), roll_range degenerate-guard stream order, cleave unconditional, check_game_over no-resurrection, TAMHP Full-mode post-insertion indexing, truncation propagated through `advance_rule_until_actor`.

**Codecs (frozen-classic mirrors — Rust side, authorized to fix):**
- `card_shape.rs` mirrors frozen `classic_card_shape_v1.py` (64-float, 34 one-hot mechanics, 17 scalar channels at offsets 47..63).
- `card_shape_v5.rs` = 34-entry classic mirror (audit premise corrected: no separate V5 mechanics list exists in Python — new families are mechanic-STRING logic, not one-hots).
- 5-slot obs encoder (catch Rust up to current Python's 5 board slots/side).

**RNG — recorded-outcome protocol** (user-chosen strategy "Recorded-outcome + Rust-own RNG"):
5 streams: `draw_picks`, `reshuffle_orders`, `randint_rolls`, `choice_rolls`, `sample_rolls` (new for `random.sample`).
- Python `golden_trace.py` records; Rust `DrawRng::Recorded` replays; Live uses `StdRng` (ChaCha).
- `roll_choice` n==1 stream-desync fixed (Recorded pops `choice_rolls` even when n==1; Live short-circuits).

---

## 3. Key architecture decisions (do not re-derive)

- **action_mask parity target = frozen `classic_actions_v1._mask_play_actions` (NOT `engine.py:get_legal_actions`).** These two legitimately differ: the frozen mask (line 228) has NO consume_ally exemption at full board; `engine.py:1228` DOES exempt consume_ally. Rust's MASK matches the frozen codec; Rust's APPLY path (`apply_play_card`, kernel.rs:1335) matches `engine.py`. **Prior fixup wrongly conflated them** (added exemption to mask + monkeypatched frozen codec to hide it) — reverted.
- **Snapshot-replay:** Rust kernel receives pre-built states (doesn't construct fresh games). Python bakes init effects (`start_mana`, t1 sudden death) into the initial snapshot before FFI handoff; `golden_trace` records initial post-reset; trace-driven Rust reproduces them. → init-time sudden death is **by-design NOT fixed** (adding an unused Rust init hook would be dead code inconsistent with the pattern).
- **`apply_action_unchecked` / `apply_decoded`** = mask-bypassing engine apply path, **TEST-ONLY** (only `golden_kernel.rs:1678` calls it; NOT in worker/ffi real step path). Used for mask-illegal-but-engine-legal actions (consume_ally at full board, cast_random_spell forced).
- **BTreeMap (not HashMap)** for deterministic serde serialization (sudden_death fields, pending_mana_drain).
- **`card.level = 1` always** in training env (`core/state.py:94` + `converter.py:208-209`). cast_random_spell scaling: dmg=4, heal=5, freeze_count=1, buff=2.
- **Self-referential fixture failure mode:** fixtures re-encoded from Rust's own output hide real divergences; **source-vs-source (Python real game) is the only oracle.** Broad validation used refute-by-default verifiers reading Python directly.

---

## 4. Validation (two rounds + focused re-validation)

**Round 1** (10-dim adversarial, source-vs-source) found `9e936f47`'s "BLOCK -1 COMPLETE" was premature → Phase 9 (`69bfff34`) resolved 2 MAJOR + 1 missed-mechanic.

**Round 2** (8-dim) found 3 more: obs card_shape scalar extractor (MAJOR, fixed `3339e870`), random-battlecry guard (minor, fixed `3339e870`), init-time sudden death (minor, by-design, documented).

**Focused re-validation** (`wz5ggvrcj`, 3 refute-by-default source verifiers) — **PASS**, all `real_divergence=false, severity=none`:
- `obs_codec_fix`: `number_after_prefix` correct; scalar[4]=0.2/0.3/0.5 (cards 14/36/35) = Python; other 16 channels unaffected (disjoint/single-prefix families → `max==single value`); `scalar_second_after_prefix` preserved.
- `game_logic_fix`: guard removal correct (always roll_choice + apply_damage → consume_shield, as `effects.py:67-83`); card15 unchanged.
- `completeness_regression`: full 50-card / 17-scalar-channel catalog sweep byte-matches Python oracle; Phase-9 mechanics + consume_ally + mana_drain apply-layer untouched.

**Pre-existing edge (NOT a `3339e870` regression):** a hypothetical `cleave_<single>` mechanic (no real card) would yield Rust 0.3 vs Python 0.0 (Python `^cleave_(\d+)_\d+` requires trailing `_digits`). Out-of-scope hardening, deferred until a card with that form is added.

---

## 5. Catalog coverage

`ai/cards.json` = **50 cards**, IDs 1..52 (IDs 2 and 9 absent). card52 (TAMHP) is the highest. **34/34 catalog mechanics** have a Rust apply branch or are correctly out-of-kernel-scope (completeness critic confirmed). Frozen-classic Python files (`classic_obs_v1`, `classic_actions_v1`, `classic_card_shape_v1`) **untouched** (byte-locked guard held); `card_shape.rs` is the Rust mirror, authorized to fix.

---

## 6. Handoff to Block 0 (foundation)

Per `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-handoff.md`:

- **Q1 (Rust ArenaEnv parity) — CLOSED by this work.** ✅
- **Q2 (`HAND_CAP` value) — OPEN.** Confirm ruleset max hand size (currently hard-capped 4) to parameterize lifted `_NUM_HAND`.
- **Q3 (V4 warm-start fidelity) — OPEN.** Does V5's fused architecture allow clean 601-scorer + base-1456 transfer from V4-Max (`update_1190.npz`), or partial? Needs a probe (instantiate V5 policy, load npz, forward-pass frozen-obs subset, compare logits to V4-Max ONNX base-1456 within export tolerance).
- **Q4 (mana_draw usage baseline) — OPEN, deferred to Phase A.** Define `mana_draw_count / eligible_turns` distribution over pilot battles (can't close now — no pilot yet; define metric + `[0.5×,1.5×]` A-gate band placeholder).
- **Q5 (V4-orig blind-lane bias) — OPEN, decide at plan time.** V4-orig never draws and is a Phase-B lane → learner could over-fit to "opponent never draws." Mitigation: modest V4-orig lane weight + high self-snapshot prevalence + monitor mana_draw collapse. Set thresholds in the plan.

**Next:** close Q2–Q5 → produce Block 0 implementation plan (the `writing-plans` terminal step) → then Block 0 foundation execution: `v5_card_shape_v1`, `encode_observation_v5`, `mana_draw` head, warm-start loader, offline-bridge — gated by tests (spec §6).

---

## 7. Session artifacts (workflow scripts)

`TrainV3.5/python/train_v3/blockminus1_{phase9_port,validate,revalidate}.workflow.js` — Ultracode orchestration scripts used for the Phase 9 port + two validation rounds. Committed for methodology reproducibility (contain hardcoded absolute worktree paths — re-target before re-running elsewhere).

See project-memory `extra-lr-v5-blockminus1-port-progress.md` (durable cross-session tracker) and `extra-lr-v5-blockminus1-parity-audit.md` (the 42-gap audit + 8-phase plan).