export const meta = {
  name: 'blockminus1-revalidate',
  description: 'Block -1 focused re-validation after the two final fixes (card_shape scalar extractor + random-battlecry guard): confirm fixes correct + no regression',
  phases: [{ title: 'Revalidate', detail: 'parallel refute-by-default source verifiers: obs_codec fix + game_logic fix + completeness regression' }],
}

const ROOT = '/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep'
const T35 = `${ROOT}/TrainV3.5`

const COMMON = `Worktree root: ${ROOT}. TrainV3.5/ subdir: rust/trainv3_core/ + python/train_v3/. HEAD = 3339e870 (final validation fixes on top of 69bfff34 Phase 9). Cargo baseline: 104 lib + 47 integration + 0 failed.
- Read Rust source AND Python source-of-truth DIRECTLY (not fixtures). Python real game: core/engine.py, core/effects.py, core/state.py, core/converter.py. Frozen classic codec: ai/train_v2/classic_obs_v1.py, classic_actions_v1.py, classic_card_shape_v1.py (byte-locked — Rust must match them).
- The prior broad validation (8 dims) found 3 issues: obs_codec MAJOR (card_shape number_after_prefix bug), game_logic MINOR (apply_random_battlecry_damage amount<=0 guard), terminal_truncation MINOR (init-time sudden death, documented as by-design snapshot-replay like start_mana). 3339e870 fixed the first two. Your job: confirm the two fixes are CORRECT and introduced NO new divergence, and that the card_shape change (shared scalar extractor) didn't regress any OTHER scalar channel.
- Default real_divergence=true if you cannot confirm byte-parity from source-vs-source. Green cargo does not prove parity.`

const V = { type: 'object', required: ['real_divergence', 'severity', 'evidence', 'recommended_fix'], properties: {
  real_divergence: { type: 'boolean' },
  severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
  file: { type: 'string' }, line: { type: 'string' },
  py_file: { type: 'string' }, py_line: { type: 'string' },
  evidence: { type: 'string' }, recommended_fix: { type: 'string' },
} }

const DIMS = [
  { key: 'obs_codec_fix', prompt: `Re-verify the card_shape.rs number_after_prefix fix (commit 3339e870). Read card_shape.rs:174 (the new iterate-all-prefixes-keep-max version) vs Python classic_card_shape_v1.py:38 + 86-103 (the regex + scalar_max-over-mechanics logic). REFUTE: (1) does Rust scalar[4] (offset 51) now yield 0.2/0.5/0.3 for battlecry_heal_hero_2/target_5/target_3 matching Python? (2) does the fix preserve correct values for the plain form battlecry_heal_5 (0.5) and all OTHER scalar channels (damage, heal, aoe_damage, battlecry_damage, buff, armor, reflect, regen, aura_atk, cleave, mana_gain, mana_drain, start_mana, summon)? (3) does the max-across-prefixes ever yield a WRONG (higher-than-Python) value for any real mechanic — i.e. is there a mechanic+prefix-list combo where two prefixes both parse to different ints and max picks the wrong one vs Python's single regex match? (4) does scalar_second_after_prefix (index 1, used for buff out[6] and cleave out[12]) still parse correctly? Cross-check each scalar channel's mechanic forms in ai/cards.json. Any new divergence from the fix is a finding.` },
  { key: 'game_logic_fix', prompt: `Re-verify the kernel.rs apply_random_battlecry_damage guard removal (commit 3339e870, kernel.rs:2053-2071). Read the new version vs Python core/effects.py:67-83 _apply_random_battlecry_damage. REFUTE: (1) does Rust now ALWAYS build targets=opponent.board+[hero], roll_choice (consuming choice_rolls), and apply_damage (consuming shield) regardless of amount — matching Python? (2) is there any current catalog card (card 15 battlecry_damage_1_random, amount=1) where the removal changes behavior? (3) does the removed guard affect any OTHER caller or the card15 golden fixture (golden_trace has card15)? Confirm card15 fixture/test still passes (cargo green says yes, but verify the logic path). Any divergence is a finding.` },
  { key: 'completeness_regression', prompt: `Regression + completeness sweep after 3339e870. The card_shape number_after_prefix change is GLOBAL (shared by all scalar extractors). Sweep the full ai/cards.json 50-card / 34-mechanic catalog and confirm: (1) for EVERY catalog card, every mechanic's contribution to EVERY scalar channel (offset 47..63) in Rust (card_shape.rs mechanic_scalars) matches Python (classic_card_shape_v1.py _encode_mechanics_cached) — specifically check cards that have marked forms (battlecry_heal_hero_2 card14, battlecry_heal_target_5 card35, battlecry_heal_target_3 card36) AND cards with range forms (aura_atk_1_3, damage_1_5, armor_1_3, regen_X_Y, reflect_X_Y, start_mana_1_5) that converter normalizes. (2) confirm the 6 Phase-9 mechanics + consume_ally + mana_drain are still fully covered in the APPLY layer (kernel.rs) — 3339e870 only touched card_shape.rs + the random-battlecry guard, so apply parity should be unchanged, but confirm no accidental edit. (3) any NEW gap introduced by the fix? Report any divergence with both file:line, or confirm clean.` },
]

phase('Revalidate')
const results = await parallel(DIMS.map(d => () =>
  agent(`${COMMON}\n\n${d.prompt}\n\nReturn the Verdict schema. Default real_divergence=true if you cannot confirm byte-parity.`,
    { label: `rv:${d.key}`, phase: 'Revalidate', schema: V, effort: 'high' })
    .then(v => v ? { ...v, concern: d.key } : null)
)).then(rs => rs.filter(Boolean))

const confirmed = results.filter(v => v.real_divergence && v.severity !== 'none')
const blockers = confirmed.filter(v => v.severity === 'blocker' || v.severity === 'major')
log(`Revalidate: ${results.length} dims, ${confirmed.length} confirmed (${blockers.length} blocker/major): ${confirmed.map(c => c.concern+':'+c.severity).join(', ') || 'none'}`)

return {
  status: blockers.length === 0 ? (confirmed.length === 0 ? 'PASS' : 'PASS-WITH-MINOR') : 'FAIL',
  results,
  confirmed,
}