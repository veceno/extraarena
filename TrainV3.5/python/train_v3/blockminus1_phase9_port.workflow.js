export const meta = {
  name: 'blockminus1-phase9-port',
  description: 'Block -1 Phase 9: revert consume_ally mask divergence + port 6 missed mechanics (aoe_damage_2, heal_target_3/5, delete_target, choose_shield_damage, cast_random_spell) into Rust kernel w/ recorded-outcome RNG, then adversarially verify parity vs Python effects.py',
  phases: [
    { title: 'Port', detail: 'one agent: consume_ally mask revert + 6 mechanic apply branches + sample_rolls RNG + fixtures + tests + cargo green' },
    { title: 'Verify', detail: 'parallel refute-by-default verifiers, one per mechanic + RNG + completeness critic' },
    { title: 'Fix', detail: 'apply verified divergences, re-run cargo' },
  ],
}

const ROOT = '/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep'
const T35 = `${ROOT}/TrainV3.5`

const COMMON = `Worktree root: ${ROOT} (contains ai/, core/, ai/cards.json). TrainV3.5/ subdir contains rust/trainv3_core/ + python/train_v3/.
- Run cargo: \`cd ${T35} && cargo test --quiet -p trainv3_core 2>&1 | grep "test result"\` (and full output for failures).
- Run python generators: \`cd ${ROOT} && PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.<module>\`.
- TrainV3.5/ is gitignored/untracked at repo root (EXPECTED — do NOT fix git tracking). Prod runtime must not import TrainV3.5 — do NOT add any TrainV3.5 import to prod bot code or webapp.
- FROZEN-classic Python files are BYTE-LOCKED: ai/train_v2/classic_obs_v1.py, classic_actions_v1.py, classic_card_shape_v1.py. Do NOT modify them (the ONLY authorized exception was the additive card52 TAMHP mask branch, already done in Phase 5). The action_mask parity target IS classic_actions_v1._mask_play_actions — Rust's mask MUST match it byte-for-byte, NOT core/engine.py:get_legal_actions (those two are a known pre-existing Python-side divergence: the engine exempts consume_ally from the board-full guard, the frozen mask does NOT).
- Python real game (core/engine.py, core/effects.py, core/state.py, ai/train_v2/classic_*) is the source of truth for game LOGIC. classic_actions_v1 is the source of truth for the action MASK.
- Use BTreeMap (not HashMap) for any new serde-serialized state field (deterministic JSON for the golden matcher).
- The prior fixup left these UNCOMMITTED in the worktree (keep the good ones, fix the bad one): 7 kernel.rs fixes (att_pos /6.0, mana_drain, roll_range, cleave, check_game_over, TAMHP, + the WRONG consume_ally mask change), truncation fix (worker.rs/ffi.rs/rust_ffi.py — GOOD, keep), regen'd action_features fixtures (scripted_basic/taunt_attack/attack_cleanup — GOOD, keep), e2e_oracle fixture + test (GOOD, keep), mana_drain fixture + test (GOOD, keep), consume_ally_full fixture + test + gen_phase7_fixtures.py monkeypatch (the monkeypatch MUST be removed — see task).
- Current cargo baseline (from prior fixup, with the WRONG consume_ally mask still in place): 103 lib pass + 46 integration pass + 0 failed. Your changes must keep it green (or improve).`

const PORT_REPORT_SCHEMA = {
  type: 'object',
  required: ['ok', 'cargo_lib_passed', 'cargo_integration_passed', 'cargo_integration_failed', 'changes', 'frozen_python_modified', 'unsure'],
  properties: {
    ok: { type: 'boolean' },
    cargo_lib_passed: { type: 'integer' },
    cargo_integration_passed: { type: 'integer' },
    cargo_integration_failed: { type: 'integer' },
    cargo_failed_test_names: { type: 'array', items: { type: 'string' } },
    changes: { type: 'array', items: { type: 'object', required: ['file', 'summary'], properties: { file: { type: 'string' }, lines: { type: 'string' }, summary: { type: 'string' } } } },
    new_fixtures: { type: 'array', items: { type: 'string' } },
    new_tests: { type: 'array', items: { type: 'string' } },
    frozen_python_modified: { type: 'array', items: { type: 'string' } },
    card_level_finding: { type: 'string' },
    unsure: { type: 'array', items: { type: 'string' } },
  },
}

const PORT_PROMPT = `${COMMON}

You are the Port agent. Do ALL kernel.rs edits sequentially (single agent — no parallel kernel.rs writes). The work has TWO parts.

=================================================================
PART A — REVERT the consume_ally MASK divergence (correctness fix)
=================================================================
The prior fixup added a consume_ally exemption to TWO guards in kernel.rs. One is CORRECT, one is WRONG.

1. MASK guard (kernel.rs ~line 814, fn mask_play_actions): currently
   \`if is_warrior && me.board.len() >= GAME_BOARD_CAP && !card.has_mechanic("consume_ally") { continue; }\`
   The frozen codec classic_actions_v1._mask_play_actions (line 228) is:
   \`if is_warrior and len(me.board) >= _NUM_BOARD: continue\`  — NO consume_ally exemption.
   => REVERT the mask: remove \`&& !card.has_mechanic("consume_ally")\` so Rust's action_mask matches the frozen codec byte-for-byte. Fix the comment (lines ~808-813) — it wrongly claims to mirror engine.py:1228; the MASK mirrors the frozen classic_actions_v1._mask_play_actions (no exemption), the APPLY path mirrors engine.py:1228 (with exemption). Do NOT touch the apply path.

2. APPLY guard (kernel.rs ~line 1195, fn apply_play_card): currently
   \`if card.is_warrior() && player.board.len() >= GAME_BOARD_CAP && !card.has_mechanic("consume_ally") { return Err("board_full"); }\`
   This matches core/engine.py:1228 (\`if len(player.board) >= 5 and "consume_ally" not in card.mechanics: continue\`). => KEEP as-is. CORRECT.

3. Lib unit test (kernel.rs test module): the prior fixup added \`mask_enables_consume_ally_play_when_board_full\` asserting the consume play bit ==1.0 at board==5. After the mask revert, the frozen codec says that bit is 0.0 (masked OUT). => INVERT: rename to \`mask_masks_out_consume_ally_play_when_board_full\` and assert the consume play bit ==0.0 at board==5 (parity with frozen codec). The consume_ally play is still ACCEPTED by the apply path (game logic), just not exposed by the model mask.

4. gen_phase7_fixtures.py: REMOVE the runtime monkeypatch entirely — delete the \`_mask_play_actions_with_consume_exempt\` function (~lines 46-68), the \`_orig_mask_play_actions = _ca._mask_play_actions\` line (~71), and the install/restore lines in gen_consume_ally_full (~160 install, ~202 restore; remove the try/finally, keep the body). The frozen file classic_actions_v1.py MUST NOT be monkeypatched. Then REWRITE gen_consume_ally_full so it does NOT rely on the mask to find the consume play (the frozen mask masks it out at board==5). Instead construct the play action_id directly: action_id = _PLAY_BASE + hand_idx*_PLAY_STRIDE + pos*_NUM_PLAY_TARGETS + target_code, where hand_idx = index of the Канеки (card_id 20) in me.hand, pos = board.len().min(NUM_PLAY_POS-1) (append), target_code = 9 (consume friendly board[0]). Call env.step(action_id) directly — the engine apply path exempts consume_ally (engine.py:1228) so the play succeeds even though the mask disables it. Keep the post-conditions (board stays 5, Канеки on board with consumed stats). The mana_drain generator (gen_mana_drain) needs NO monkeypatch — leave it unchanged (it does not use the consume mask).

5. REGENERATE the consume_ally_full fixture: \`cd ${ROOT} && PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase7_fixtures\` (regenerates both consume_ally_full + mana_drain). Confirm the consume_ally_full fixture's recorded mask now has the consume play bit DISABLED (0.0) — matching the reverted Rust mask. The existing state-transition test \`rust_kernel_apply_action_matches_consume_ally_full_python_transitions\` in golden_kernel.rs is STATE-ONLY (assert_trace_state_transitions_match, does NOT compare mask) — it forces the action via apply path, so it stays valid and green.

=================================================================
PART B — PORT 6 missed mechanics (Python applies, Rust apply_play_effects silently drops)
=================================================================
All 6 have registered/dynamic Python handlers in core/effects.py and NO apply branch in Rust apply_play_effects (Rust only handles their targeting/masking, then no-ops). Add the apply branches. For each, mirror the Python EXACTLY (damage path, shield/armor/reflect, heal clamp, graveyard/no-deathrattle, target mapping). Source of truth file:line given.

B1. aoe_damage_X (card 10 = aoe_damage_2, potion "Импульс Бездны"; dynamic aoe_damage_{dmg} registered effects.py:433-450):
    - Python: \`for unit in opponent.board: apply_damage(unit, dmg)\` — ALL enemy minions, NOT the enemy hero. dmg parsed from the mechanic suffix. Uses apply_damage (goes through shield/armor/reflect modifiers — NOT direct hp subtraction).
    - Rust: parse dmg from mechanic via regex \`aoe_damage_(\\d+)\`; for each enemy minion apply the SAME damage path Rust uses for attacks (the one that respects shield/armor/reflect, e.g. apply_damage_to_unit or the modifier pipeline). No target needed (mask: no target slot).

B2. (battlecry_|spell_)?heal_target_X (card 36 = battlecry_heal_target_3 "Юни", card 35 = battlecry_heal_target_5 "Фрирен"; dynamic regex effects.py:1441-1461):
    - Python: \`if match and target_id:\` heal_amount = parsed int. Find target in owner.board (FRIENDLY unit) by instance_id → apply_heal(unit, heal_amount); if not found, check owner.hero → apply_heal(owner.hero, heal_amount). apply_heal clamps to max_hp (heal cannot exceed max_hp).
    - Rust: parse heal_amount from \`heal_target_(\\d+)\` (strip battlecry_/spell_ prefix). target_code maps to a FRIENDLY target (friendly minion or own hero) — verify the target-code→friendly mapping in Rust mask_targets_for_card / apply_damage_to_play_target and reuse it. apply_heal = min(hp + amount, max_hp). If target is the owner hero, heal the hero. Requires target (mask exposes friendly targets).

B3. delete_target (card 13, potion "Черная Дыра"; effects.py:811-830):
    - Python: \`if not target_id: return\`. Find enemy unit by instance_id. \`if consume_shield(unit, "delete_target"): return\` — shield BLOCKS the delete AND is consumed. Else \`opponent.board.remove(unit); opponent.graveyard.append(unit)\` — NO deathrattle fires (remove, not kill). Targets enemy minions ONLY (loop is opponent.board, NOT hero — if target_id is hero, nothing happens).
    - Rust: target_code maps to enemy minion. If target unit has shield → consume shield, return (no remove). Else remove from opponent.board → push to opponent.graveyard, NO deathrattle chain. Requires target (enemy minion).

B4. choose_shield_damage (card 21, warrior "Геральт"; effects.py:958-986):
    - Python: \`if target_id:\` find enemy unit by instance_id → apply_damage(unit, 3); elif target is enemy hero → apply_damage(hero, 3). \`else:\` (no target) → if "shield" not in card.mechanics: card.mechanics.append("shield") — grants shield to the PLAYED CARD itself (the warrior unit on board).
    - Rust: the mask exposes BOTH the no-target slot (base+0) AND target slots (per mask_targets_for_card / the has_csd path at kernel.rs:820,829). In apply_play_effects: if target_code == 0 (no target) → add "shield" to the just-played card's mechanics (the card is on board at \`position\` — use the position param passed to apply_play_effects to find it; add "shield" if not present). If target_code != 0 → apply 3 damage to the enemy target (unit or hero) via the damage-modifier path. Verify the mask already exposes both branches correctly (it should — the has_csd handling exists); only the APPLY branch is missing.

B5. cast_random_spell (card 26, warrior "Мидория"; effects.py:838-951) — COMPLEX, needs recorded-outcome RNG:
    - Python: \`level = card.level\`. \`spell_choice = random.randint(1, 4)\`.
      * spell 1 Texas Smash: dmg = 4 + (level-1); possible_targets = opponent.board + [opponent.hero]; target = random.choice(possible_targets); apply_damage(target, dmg).
      * spell 2 Recovery: heal = 5 + (level-1); apply_heal(owner.hero, heal).
      * spell 3 Blackwhip: freeze_count = 2 if level>=5 else 1; unfrozen_enemies = [u for u in opponent.board if not u.is_frozen]; targets_to_freeze = random.sample(unfrozen_enemies, min(freeze_count, len(unfrozen_enemies))); for each: if consume_shield(target, "blackwhip"): continue; else target.is_frozen = True.
      * spell 4 Full Cowl: buff = 2 + ((level-1)//2); card.attack += buff; card.hp += buff; card.max_hp += buff (buffs the played card itself).
    - FIRST: determine \`card.level\` in the training env. Check core/state.py CardInstance.level field + how the env sets it (core/engine.py apply_start_game_effects / deck construction / ai/train_v2/classic_rl_env.py). Record the finding in card_level_finding. If level is always 1 in training, scaling is dmg=4, heal=5, freeze_count=1, buff=2 — but mirror whatever Python actually does for ANY level the env can produce.
    - RNG protocol: existing recorded-outcome streams are draw_picks/reshuffle_orders/randint_rolls/choice_rolls (see kernel.rs DrawRng::Recorded + golden_trace.py RecordingRng). cast_random_spell needs: randint(1,4) [→ randint_rolls], then conditionally random.choice [→ choice_rolls, 0-based index into opponent.board+[hero]] (spell 1) OR random.sample [→ NEW stream sample_rolls] (spell 3). ADD \`sample_rolls: VecDeque<Vec<i32>>\` to DrawRng::Recorded with #[serde(default)] — each entry is the list of selected 0-based indices into the unfrozen_enemies list, in selection order. Extend golden_trace.py RecordingRng to record random.sample → step["sample_rolls"] (list of index-lists), parallel to choice_rolls. Rust roll_sample helper: Recorded pops the next index-list; Live uses a deterministic sample over the population (use the same StdRng the worker uses — match Python's *distribution*, not its MT19937 stream; recorded-outcome fixtures pin byte-parity, live training just needs a valid sample).
    - Rust apply branch: pop spell_choice from randint_rolls (Recorded) or gen_range(1..=4) (Live). Then branch on spell exactly as Python. For spell 1 choice: pop choice_rolls index into opponent.board+[hero] (Recorded) or rng pick (Live). For spell 3 sample: pop sample_rolls index-list (Recorded) or rng sample (Live). For each frozen target: consume_shield (skip if shield) else set is_frozen. For spell 4: buff the played card (find via position param) attack/hp/max_hp.
    - New fixture golden_trace_cast_random_spell.json exercising all 4 spells (you may need a deck with multiple card 26 and enough turns; use single-card-type decks where possible for draw determinism + recorded-outcome for the spell RNG). Use verify_mask=False for RNG-stream fixtures (isolate from _verify_mask clone re-exec — established pattern). Add a state-transition test (or full-matcher if recorded-outcome makes obs deterministic) in golden_kernel.rs. Use PlacementMode consistent with the fixture's env_config.

For ALL 6 mechanics: also confirm the OBSERVATION layer (card_shape.rs MECHANICS_LIST + mechanic_scalars) already encodes them as one-hot/scalar (the cards-rescan said it does) — do NOT change the frozen classic card_shape; if card_shape_v5.rs or card_shape.rs is missing a flag, that is a SEPARATE obs gap to note in unsure[] (do NOT fix frozen-classic files). The task here is the APPLY layer.

After all edits: run \`cd ${T35} && cargo test --quiet -p trainv3_core 2>&1\` and capture the test-result lines. Fix compile errors and any NEW test failures you introduced. The 3 prior FIX-1 attack-fixture failures should already be fixed by the prior regen (they were green at 46/0); do not regress them. Report the final cargo counts.

Return the PortReport schema. Be precise with file:line in changes[]. List EVERY frozen Python file you touched in frozen_python_modified[] (should be EMPTY — you must not edit any). If you edited classic_actions_v1.py STOP and report it in unsure[].`

const VERIFY_CONCERNS = [
  { key: 'consume_ally_revert', prompt: `Verify the consume_ally MASK revert. Read kernel.rs mask_play_actions (~line 814) and confirm it now matches frozen classic_actions_v1._mask_play_actions (ai/train_v2/classic_actions_v1.py:228) BYTE-FOR-BEHAVIOR: \`if is_warrior && board.len() >= GAME_BOARD_CAP { continue; }\` with NO consume_ally exemption. Confirm the APPLY path (kernel.rs ~1195 apply_play_card) STILL has the consume_ally exemption (matches core/engine.py:1228). Confirm gen_phase7_fixtures.py has NO monkeypatch on classic_actions_v1._mask_play_actions. Confirm the lib unit test now asserts the consume play bit ==0.0 at board==5 (masks OUT, per frozen codec). Try to REFUTE: is there ANY consume_ally scenario where Rust's mask now diverges from frozen classic_actions_v1? Is the consume_ally_full fixture's recorded mask consistent with the frozen codec (bit 0.0)?` },
  { key: 'aoe_damage', prompt: `Verify aoe_damage_X parity. Read the Rust apply branch vs core/effects.py:433-450. REFUTE: does Rust hit ALL enemy minions and NOT the hero? Does it use the damage-modifier path (shield/armor/reflect) not direct hp-=? Is dmg parsed from the mechanic suffix correctly? Any edge case (empty board, armor, shield) where it diverges?` },
  { key: 'heal_target', prompt: `Verify heal_target_X parity (cards 35/36). Read the Rust apply branch vs core/effects.py:1441-1461. REFUTE: does it heal a FRIENDLY unit (owner.board) or owner hero by the parsed amount? Does apply_heal clamp to max_hp (min(hp+amt, max_hp))? Is the target_code→friendly-target mapping correct (friendly minion vs own hero)? Does it handle the not-found-in-board→hero fallback? Any divergence?` },
  { key: 'delete_target', prompt: `Verify delete_target parity (card 13). Read the Rust apply branch vs core/effects.py:811-830. REFUTE: does shield BLOCK the delete AND get consumed (consume_shield)? Does it remove the enemy minion to graveyard with NO deathrattle chain? Does it target enemy minions ONLY (not hero — hero target is a no-op)? Any divergence (e.g. firing deathrattle, not consuming shield, hitting hero)?` },
  { key: 'choose_shield_damage', prompt: `Verify choose_shield_damage parity (card 21). Read the Rust apply branch vs core/effects.py:958-986. REFUTE: with target → 3 damage to enemy unit/hero via the damage-modifier path? WITHOUT target (target_code 0) → grant "shield" to the PLAYED CARD's mechanics (the warrior on board), only if not already present? Does the mask expose BOTH the no-target slot and target slots? Any divergence (e.g. shielding the wrong card, wrong damage path, mask missing a branch)?` },
  { key: 'cast_random_spell', prompt: `Verify cast_random_spell parity (card 26) vs core/effects.py:838-951. REFUTE: is spell_choice = randint(1,4) consumed from randint_rolls (Recorded) / gen_range(1..=4) (Live)? Spell 1: dmg = 4+(level-1), target = choice over opponent.board+[hero] from choice_rolls/rng, apply_damage? Spell 2: heal owner.hero by 5+(level-1) (apply_heal clamp)? Spell 3: freeze_count = 2 if level>=5 else 1, targets = sample of unfrozen_enemies from sample_rolls/rng, consume_shield skips else is_frozen=True? Spell 4: buff = 2+((level-1)//2) to the played card's attack/hp/max_hp? Is \`card.level\` correct (the port agent's card_level_finding)? Does the sample_rolls recorded stream + golden_trace RecordingRng record/replay correctly? Any divergence in scaling, target list order, shield handling, or RNG stream mapping?` },
  { key: 'rng_protocol', prompt: `Verify the recorded-outcome RNG protocol extension (sample_rolls). Read kernel.rs DrawRng::Recorded + roll_sample + golden_trace.py RecordingRng. REFUTE: is sample_rolls a VecDeque<Vec<i32>> with #[serde(default)]? Does golden_trace record random.sample as index-lists in step["sample_rolls"]? Does Recorded pop-in-order while Live uses a valid sample? Does the golden state matcher stay symmetric for existing fixtures (absent sample_rolls → empty default on both sides)? Is there any stream desync (e.g. popping when the spell branch doesn't call sample, or min>=max degenerate cases)?` },
  { key: 'completeness', prompt: `Completeness critic. Cross-check the FULL mechanic catalog (ai/cards.json 50 cards, 34 distinct mechanics) against the Rust apply branches (kernel.rs apply_play_effects/apply_attack/apply_end_turn/cleanup/apply_damage_modifiers/apply_regen) and core/effects.py + core/engine.py. The port just added 6 (aoe_damage_2, heal_target_3/5, delete_target, choose_shield_damage, cast_random_spell). Are there ANY OTHER mechanics Python applies that Rust still silently drops? Re-examine the cards-rescan table (it found exactly these 6 + confirmed mana_drain was already ported). Look for: any battlecry_/spell_/heal_/damage_ dynamic regex family with a Python handler and no Rust branch; any registered @register_effect with no Rust apply branch; any mechanic referenced only in mask_targets_for_card/requires_target but dropped in apply. Report any NEW gap with file:line evidence, or confirm complete.` },
]

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['concern', 'real_divergence', 'severity', 'evidence', 'recommended_fix'],
  properties: {
    concern: { type: 'string' },
    real_divergence: { type: 'boolean' },
    severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
    file: { type: 'string' },
    line: { type: 'string' },
    evidence: { type: 'string' },
    recommended_fix: { type: 'string' },
  },
}

phase('Port')
const port = await agent(PORT_PROMPT, { label: 'port', phase: 'Port', schema: PORT_REPORT_SCHEMA, effort: 'high' })
log(`Port done: ok=${port.ok}, cargo lib=${port.cargo_lib_passed} int=${port.cargo_integration_passed}/${port.cargo_integration_failed}, unsure=${port.unsure.length}, frozen_modified=${port.frozen_python_modified.length}, level=${port.card_level_finding}`)

phase('Verify')
const verifyResults = await parallel(VERIFY_CONCERNS.map(c => () =>
  agent(`${COMMON}\n\n${c.prompt}\n\nReturn the Verdict schema. Default to real_divergence=true if you cannot confirm parity. Be concrete with file:line evidence.`,
    { label: `verify:${c.key}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' })
    .then(v => v ? { ...v, concern: c.key } : null)
)).then(rs => rs.filter(Boolean))

const confirmed = verifyResults.filter(v => v.real_divergence && v.severity !== 'none')
log(`Verify: ${verifyResults.length} verifiers, ${confirmed.length} confirmed divergences: ${confirmed.map(c => c.concern + '(' + c.severity + ')').join(', ')}`)

if (confirmed.length === 0) {
  return { status: 'PASS', port, verifyResults, confirmed: [] }
}

phase('Fix')
const fixPrompt = `${COMMON}

The Port agent produced these changes (PortReport):
${JSON.stringify(port, null, 2)}

Adversarial verifiers CONFIRMED these divergences (refute-by-default, so these survived scrutiny):
${JSON.stringify(confirmed, null, 2)}

Fix EACH confirmed divergence in kernel.rs (and golden_trace.py / fixtures / tests as needed). Do NOT touch frozen-classic Python files. After fixing, run \`cd ${T35} && cargo test --quiet -p trainv3_core 2>&1 | grep "test result"\` and report the final counts + what you changed (file:line). Return a text report: ok, cargo counts, per-divergence fix summary, any remaining unsure.`

const fix = await agent(fixPrompt, { label: 'fix', phase: 'Fix', effort: 'high' })
return { status: 'FIXED', port, verifyResults, confirmed, fix }