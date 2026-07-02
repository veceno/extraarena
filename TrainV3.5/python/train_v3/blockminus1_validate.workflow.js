export const meta = {
  name: 'blockminus1-validate',
  description: 'Block -1 final sign-off: broad adversarial validation of Rust ArenaEnv parity vs Python real game, reading source directly (not self-referential fixtures), refute-by-default',
  phases: [
    { title: 'Validate', detail: 'parallel source-reading verifiers across game-logic / mask / codec / RNG / state-matcher / terminal / completeness' },
    { title: 'Adjudicate', detail: 'triage confirmed findings into blocker/major/minor' },
  ],
}

const ROOT = '/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep'
const T35 = `${ROOT}/TrainV3.5`

const COMMON = `Worktree root: ${ROOT} (ai/, core/, ai/cards.json). TrainV3.5/ subdir: rust/trainv3_core/ + python/train_v3/.
HEAD commit 69bfff34 = "Block -1 validation fixup + Phase 9 — 6 missed mechanics + consume_ally mask revert". This is the FINAL sign-off validation for Block -1 (Rust ArenaEnv 1-in-1 with the real game).
- Read the Rust source AND the Python source-of-truth DIRECTLY. Do NOT trust golden fixtures as oracle — the prior validation found fixtures were self-referential (re-encoded from Rust's own output), hiding real divergences. The Python real game (core/engine.py, core/effects.py, core/state.py, core/converter.py) + frozen classic codec (ai/train_v2/classic_obs_v1.py, classic_actions_v1.py, classic_card_shape_v1.py) are the only oracles.
- Frozen-classic files are BYTE-LOCKED (classic_obs_v1/classic_actions_v1/classic_card_shape_v1) — Rust must match them byte-for-behavior. The ONLY authorized exception was the additive card52 TAMHP mask branch (Phase 5); any other divergence from frozen-classic is a finding.
- Run cargo to confirm green state: \`cd ${T35} && cargo test --quiet -p trainv3_core 2>&1 | grep "test result"\` (baseline: 103 lib + 47 integration + 0 failed).
- Default to real_divergence=true if you cannot confirm byte-parity. Be concrete with file:line evidence on BOTH sides. A green cargo test does NOT prove parity (fixtures can be self-referential) — only source-vs-source comparison does.`

const V = { type: 'object', required: ['real_divergence', 'severity', 'evidence', 'recommended_fix'], properties: {
  real_divergence: { type: 'boolean' },
  severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
  file: { type: 'string' }, line: { type: 'string' },
  py_file: { type: 'string' }, py_line: { type: 'string' },
  evidence: { type: 'string' }, recommended_fix: { type: 'string' },
} }

const DIMS = [
  { key: 'game_logic', prompt: `Game-logic parity. Sweep kernel.rs apply_play_effects / apply_play_card / apply_attack / apply_end_turn / cleanup_dead_units / apply_deathrattle_effects / apply_damage_modifiers / apply_regen vs core/engine.py + core/effects.py. REFUTE: any mechanic apply branch where Rust's behavior differs from Python (damage path, heal clamp, shield consume, deathrattle fire conditions, graveyard order, board-cap guards, target mapping 1..7 minion / 8 hero / 9..15 friendly). Check the 6 newly-ported mechanics (aoe_damage_X, heal_target_X, delete_target, choose_shield_damage, cast_random_spell) AND the older ones (mana_drain two-stage, consume_ally, cleave, instant_kill, lifesteal, freeze, regen, shield_refresh, rebirth, crime_and_punishment, TAMHP, team_wide_shield, aoe_silence, aoe_freeze, battlecry_*). Report any divergence with both file:line.` },
  { key: 'action_mask', prompt: `Action mask parity. Read kernel.rs build_action_mask / mask_play_actions / mask_attack_actions / mask_targets_for_card / apply_placement_mode vs frozen classic_actions_v1.py (build_action_mask, _mask_play_actions, _mask_attack_actions, _mask_targets_for_card, _apply_placement_mode). REFUTE byte-for-behavior: the consume_ally mask (must have NO exemption at full board, matching classic_actions_v1.py:228), board-full guard cap, target exposure (1..7 enemy minion, 8 hero, 9..15 friendly), choose_shield_damage no-target slot, placement_mode masking, attack target taunt/bypass logic. Confirm GAME_BOARD_CAP==5 == _NUM_BOARD==5. Any mask divergence from the frozen codec is a finding.` },
  { key: 'action_codec', prompt: `Action codec + action_features parity. Read kernel.rs encode_action_features / action_codec.rs vs frozen classic_actions_v1.py (encode_action_features, the 601-candidate layout, _NUM_BOARD/_NUM_HAND/_NUM_ATTACK_TARGETS/_ATTACK_BASE). REFUTE: att_pos channel 141 must be (att_pos+1)/(_NUM_BOARD+1) = /6.0 (the prior MAJOR bug was /8.0 — confirm it's now /6.0 at kernel.rs:1845); all other action_features channels; the 601-candidate space layout; MAX_CANDIDATE_ACTIONS. Any divergence is a finding.` },
  { key: 'obs_codec', prompt: `Observation codec parity. Read kernel.rs encode_card_slots_v1 (5-slot) / encode_card_shape_v1 / card_shape.rs / card_shape_v5.rs vs frozen classic_obs_v1.py (_encode_card_slots, OBS_DIM=1456, _CARD_SLOTS=20, 5 own + 5 enemy board + 4 hand + 4 padding) + classic_card_shape_v1.py (MECHANICS_LIST 34, per-card shape, board_pos /8.0 hand_pos /5.0). REFUTE: the 5-slot layout matches Python exactly (zone summaries at offset 1312), per-card mechanic one-hot + scalars byte-match, OBS_DIM. Confirm the index-47 desk_freeze/damage overlap is shared with Python (accepted-as-frozen, deferred to Block 0 — NOT a new finding). Any NEW obs divergence is a finding.` },
  { key: 'rng', prompt: `Recorded-outcome RNG parity. Read kernel.rs DrawRng (Recorded/Live), roll_range, roll_choice, roll_sample, draw_one_from_deck/compute_draw_weights vs golden_trace.py RecordingRng + core/engine.py weighted draw. REFUTE: stream consumption order matches Python's random.randint/random.choice/random.sample/random.random calls; the n==1 roll_choice pop-for-sync (kernel.rs:228-256); roll_range min>=max pop-before-guard; roll_sample empty-pop-before-guard (mirrors Python's \`if unfrozen_enemies:\`); draw_picks/reshuffle_orders for multi-card draws; Live vs Recorded distribution validity. Any stream desync or distribution divergence is a finding.` },
  { key: 'state_matcher_fixtures', prompt: `State-transition matcher + fixture modality. Read golden_kernel.rs (assert_trace_transitions_match full-matcher, assert_trace_state_transitions_match, assert_trace_state_transitions_match_forced) + the fixture generators (gen_phase7_fixtures.py, gen_phase9_fixtures.py, gen_e2e_oracle_fixture.py, regen_action_fixtures.py, regen_obs5_fixtures.py). REFUTE: are the full-matcher fixtures' action_features/mask/obs hashes derived from the PYTHON oracle (not re-encoded from Rust's own output)? Does the forced matcher (apply_action_unchecked) only bypass the mask for engine-legal-mask-illegal actions (consume_ally full board, cast_random_spell)? Is apply_action_unchecked reachable from the real worker/ffi step path (it must NOT be — only tests)? Are existing fixtures still parity-valid (the regen'd attack_cleanup/scripted_basic/taunt_attack)? Any self-referential fixture or mask-bypass leak is a finding.` },
  { key: 'terminal_truncation', prompt: `Terminal / truncation / sudden-death parity. Read kernel.rs check_game_over (no else-branch), apply_sudden_death, max_turns/truncated in worker.rs advance_rule_until_actor + step + ffi truncation propagation vs core/engine.py _check_game_over, sudden_death_turns_by_player, core/state.py fields, rollout_worker.py. REFUTE: draw/p1_win/p2_win conditions, no resurrection of terminal states, sudden-death fires every turn from t1 when enabled (damage escalation), truncated = turn>max_turns (strict gt, independent of terminated), truncation propagated through advance_rule_until_actor (the prior fix added this — confirm it's correct and not double-resetting). Any divergence is a finding.` },
  { key: 'completeness', prompt: `Completeness critic — final sweep. Cross-check the FULL ai/cards.json 50-card / 34-mechanic catalog against ALL Rust apply branches (kernel.rs) + core/effects.py + core/engine.py. The Phase 9 completeness verifier claimed all 34 are covered (6 new: aoe_damage_2, heal_target_3/5, delete_target, choose_shield_damage, cast_random_spell). REFUTE that claim: independently enumerate every catalog mechanic, find its Python handler file:line, find its Rust apply branch file:line, and confirm behavioral parity. Look specifically for: dynamic-regex families (battlecry_/spell_/heal_/damage_/aura_/armor_/regen_/reflect_/start_mana_) where Python's regex matches a form Rust doesn't; any @register_effect with no Rust branch that IS reachable from a catalog card; any mechanic referenced in mask/requires_target but dropped in apply. Also check: are there out-of-catalog Python handlers (spell_damage, summon, buff_all, etc.) that a catalog card could trigger via a mechanic string (check ai/cards.json mechanics arrays, not just the handler registry)? Report any REAL gap with both file:line, or confirm complete.` },
]

phase('Validate')
const results = await parallel(DIMS.map(d => () =>
  agent(`${COMMON}\n\n${d.prompt}\n\nReturn the Verdict schema. Default real_divergence=true if you cannot confirm byte-parity from source-vs-source.`,
    { label: `v:${d.key}`, phase: 'Validate', schema: V, effort: 'high' })
    .then(v => v ? { ...v, concern: d.key } : null)
)).then(rs => rs.filter(Boolean))

const confirmed = results.filter(v => v.real_divergence && v.severity !== 'none')
const blockers = confirmed.filter(v => v.severity === 'blocker' || v.severity === 'major')
log(`Validate: ${results.length} dimensions, ${confirmed.length} confirmed divergences (${blockers.length} blocker/major): ${confirmed.map(c => c.concern + ':' + c.severity).join(', ') || 'none'}`)

phase('Adjudicate')
return {
  status: blockers.length === 0 ? (confirmed.length === 0 ? 'PASS' : 'PASS-WITH-MINOR') : 'FAIL',
  results,
  confirmed,
  summary: results.map(r => `${r.concern}: ${r.real_divergence ? r.severity : 'ok'}`).join('\n'),
}