use trainv3_core::ffi::{
    trainv3_worker_action_features_f16_len, trainv3_worker_action_features_f16_ptr,
    trainv3_worker_action_features_len, trainv3_worker_action_mask_len, trainv3_worker_encode,
    trainv3_worker_episode_lengths_len, trainv3_worker_episode_lengths_ptr,
    trainv3_worker_episode_returns_len, trainv3_worker_episode_returns_ptr, trainv3_worker_free,
    trainv3_worker_from_trace_json, trainv3_worker_from_trace_json_pool_with_options_v2,
    trainv3_worker_from_trace_json_pool_with_options_v3,
    trainv3_worker_from_trace_json_with_options, trainv3_worker_from_trace_json_with_options_v2,
    trainv3_worker_from_trace_json_with_options_v3, trainv3_worker_from_trace_json_with_options_v4,
    trainv3_worker_from_trace_json_with_options_v5, trainv3_worker_from_trace_json_with_options_v6,
    trainv3_worker_legal_action_counts_len, trainv3_worker_observation_v1_len,
    trainv3_worker_observation_v1_ptr, trainv3_worker_observation_v5_len,
    trainv3_worker_observation_v5_ptr, trainv3_worker_reset, trainv3_worker_reset_flags_len,
    trainv3_worker_reset_flags_ptr, trainv3_worker_reset_indices, trainv3_worker_rewards_len,
    trainv3_worker_rewards_ptr, trainv3_worker_rollout_action_tape,
    trainv3_worker_rollout_action_tape_pre_step, trainv3_worker_rollout_broadcast_action_ids,
    trainv3_worker_rollout_broadcast_action_ids_pre_step, trainv3_worker_step,
    trainv3_worker_step_auto_reset, trainv3_worker_terminal_observation_v1_len,
    trainv3_worker_terminal_observation_v5_len, trainv3_worker_terminal_observation_valid_len,
};
use trainv3_core::kernel::{
    compute_reward_components_v5, hash_f32_le, DrawRng, GoldenSnapshot, GoldenTrace, KernelConfig,
    RewardComponentsV5, RolloutKernel,
};
use trainv3_core::worker::{
    ActionFeatureOutput, ActionMaskOutput, BatchedRolloutWorker, DiagnosticOutput,
    ObservationOutput, TerminalObservationOutput, WorkerRng,
};
use trainv3_core::OBS_DIM_V1;

#[test]
fn rust_kernel_matches_python_initial_golden_snapshot() {
    let raw = include_str!("fixtures/golden_trace_seed123.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let snapshot = &trace.initial;

    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    let out = kernel.encode_snapshot(&snapshot.state, snapshot.state.current_turn_owner_id);

    assert_eq!(out.legal_ids(), snapshot.legal_ids);
    assert_eq!(hash_f32_le(&out.action_mask), snapshot.mask_sha256_f32_le);
    assert_eq!(
        hash_f32_le(&out.action_features),
        snapshot.action_features_sha256_f32_le
    );
    assert_eq!(hash_f32_le(&out.observation_v1), snapshot.obs_sha256_f32_le);
    assert_eq!(
        hash_f32_le(&out.observation_v5),
        snapshot
            .obs_v5_sha256_f32_le
            .as_deref()
            .expect("v5 hash exists")
    );
}

#[test]
fn rust_kernel_matches_all_python_golden_snapshots_and_rewards() {
    let raw = include_str!("fixtures/golden_trace_seed123.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));

    assert_snapshot_matches(&kernel, "initial", &trace.initial);
    for step in &trace.steps {
        assert_snapshot_matches(&kernel, &format!("step {} pre", step.t), &step.pre);
        assert_snapshot_matches(&kernel, &format!("step {} post", step.t), &step.post);
        assert_reward_components_close(
            compute_reward_components_v5(&step.pre.state, &step.post.state, step.acting_player_id),
            step.reward_components_v5,
        );
    }
}

#[test]
fn rust_kernel_apply_action_matches_scripted_python_transitions() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_extended_combat_python_transitions() {
    for raw in [
        include_str!("fixtures/golden_trace_targeted_potion.json"),
        include_str!("fixtures/golden_trace_taunt_attack.json"),
        include_str!("fixtures/golden_trace_attack_cleanup.json"),
    ] {
        let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
        assert_trace_transitions_match(&trace);
    }
}

#[test]
fn rust_kernel_matches_python_e2e_oracle_multi_mechanic_with_attack() {
    // True end-to-end Python-oracle fixture generated ENTIRELY from
    // `golden_trace.build_golden_trace` with an EXPLICIT deck override
    // (`--p1-deck-ids` / `--p2-deck-ids`), so re-simulation is legal and
    // deterministic — unlike the legacy fixtures, whose state trajectory was
    // frozen against the default deck pool (Phase 5 added card 51 and shifted
    // the default-deck draw outcomes, so they can only be RE-ENCODED, never
    // re-simulated).
    //
    // Scenario: multi-mechanic state with a minion attack + two ported
    // mechanics. p1 plays Наофуми (card 30, `taunt`); p2 plays Крипер (card
    // 34, `deathrattle_aoe_damage_2`); at step t6 p1 takes action_id 545 =
    // _ATTACK_BASE(545) + att_idx(0)*_NUM_ATTACK_TARGETS(8) + tcode(0) → p1
    // board[0] attacks p2 board[0], so att_pos = 0 (>= 0). This locks in the
    // kernel-fix att_pos divisor (/6.0, mirroring
    // ai/train_v2/classic_actions_v1.py:589) via the FULL matcher
    // (assert_trace_transitions_match), which compares obs_v1 + obs_v5 +
    // mask + action_features + reward byte-level — the modality closure.
    //
    // Single-card-type decks (p1 all 30, p2 all 34) make the end-of-turn draw
    // deterministic across RNG modalities: the drawn CARD is always 30 / 34
    // regardless of which deck index the weighted draw picks, and the deck
    // zone is encoded in obs only as a SUMMARY (count + mean/max stats +
    // mechanic tallies — classic_obs_v1._encode_one_zone), so the obs / mask
    // / action_features are identical under both Python MT19937 and the
    // `WorkerRng::Deterministic` zero-RNG this matcher uses.
    let raw = include_str!("fixtures/golden_trace_e2e_oracle.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_deathrattle_cleanup_current_python() {
    // Task #14 (DW-7): the deathrattle_cleanup fixture uses a MULTI-card deck
    // [37,27,28,41,40,42,29,38], so the Rust `WorkerRng::Deterministic`
    // (zero-RNG → weighted_choice_idx always picks index 0) DIVERGES from
    // Python's seeded MT19937 weighted draw at every end-of-turn draw. The
    // recorded-outcome RNG protocol closes this gap: the fixture now carries
    // per-step `draw_picks` + `reshuffle_orders`, and the replay builds a
    // `DrawRng::Recorded` per step so Rust applies Python's exact draw
    // outcomes.
    //
    // State-only parity (assert_trace_state_transitions_match): asserts
    // post-state JSON equality only — the Rust observation encoder is at
    // OLD-Python parity (board 7 slots/side, hand at slots 16-19) while
    // current Python `classic_obs_v1._encode_card_slots` uses 5 board
    // slots/side (hand at slots 12-15, `_CARD_SLOTS=20`). That observation-
    // codec gap is Phase 8 / CS-1 scope and is orthogonal to the draw RNG
    // parity verified here.
    let raw = include_str!("fixtures/golden_trace_deathrattle_cleanup.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_weighted_draw_python_transitions() {
    // Phase 1 (Block -1) parity: verifies the no-FIFO weighted draw logic
    // (compute_draw_weights / weighted_choice_idx / draw_one_from_deck /
    // reset_to_base_state) reproduces Python's state transitions exactly.
    //
    // This asserts POST-STATE parity (every card in every zone, mana,
    // mana_draw_count_this_turn, turn owner/number) — i.e. the gameplay
    // state the draw logic owns. It deliberately does NOT assert obs_v1 /
    // obs_v5 / mask / action_features parity: the Rust observation encoder
    // is at OLD-Python parity (board 7 slots/side, hand at slots 16-19) while
    // the current Python `classic_obs_v1._encode_card_slots` uses 5 board
    // slots/side (hand at slots 12-15, `_CARD_SLOTS=20` with 4 trailing zero
    // slots). That observation-codec gap is Phase 8 / CS-1 scope and is
    // tracked separately; it is orthogonal to the draw logic verified here.
    let raw = include_str!("fixtures/golden_trace_weighted_draw.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_mana_draw_python_transitions() {
    // Phase 2 (Block -1) parity: verifies the mana_draw action (parallel
    // binary head) reproduces Python's state transitions exactly. The fixture
    // exercises a SUCCESSFUL mana_draw: p2 (mana_per_turn=3) plays a 0-cost
    // warrior, then a 1-cost potion (→ graveyard), then takes mana_draw —
    // the draw reshuffles the 1-card graveyard into the deck and draws it
    // (deterministic single-card pick, index 0 under zero-RNG). This covers
    // MD-1/2 (cost = MANA_DRAW_BASE*(count+1), deduct before draw), MD-3
    // (parallel binary head, separate from the 601 mask), MD-5 (mana
    // deducted before draw; on draw-failure mana is refunded — the success
    // path here confirms the deduct+count+draw order), and AC-FFI-1 (board
    // cap 5 game rule vs NUM_BOARD=7 codec layout). The fixture also records
    // `mana_draw_legal` per step (true at steps 2 and 3) and `mana_draw_taken`
    // (true at step 3); the replay threads `mana_draw_taken` into
    // `apply_action` as the `mana_draw_flag` argument. State-transition only
    // (obs parity deferred to Phase 8).
    let raw = include_str!("fixtures/golden_trace_mana_draw.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_rebirth_python_transitions() {
    // Phase 3 (REBIRTH-1): card 50 Бан (rebirth_1, 8 mana 3/7) is killed by
    // Сайтама's instant_kill attack. The rebirth pre-pass in
    // cleanup_dead_units_for_player resurrects Бан with 1 HP and consumes the
    // rebirth_1 mechanic string. Multi-card deck (8× Бан vs 8× Сайтама);
    // recorded-outcome RNG handles draw parity. State-transition only.
    let raw = include_str!("fixtures/golden_trace_rebirth.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_crime_and_punishment_python_transitions() {
    // Phase 3 (CAP-1): card 49 Достоевский hero (crime_and_punishment_2).
    // When a friendly minion (Скелет) dies to Сайтама's instant_kill, CAP
    // deals 2 damage to the opponent hero via DIRECT hp subtraction (not
    // apply_damage — bypasses armor/reflect). The fixture verifies p2 hero
    // hp 30→28. Multi-card deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_cap.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_consume_ally_python_transitions() {
    // Phase 3 (CLN-3): card 20 Канеки Кен (consume_ally, 3 mana 2/2) consumes
    // a friendly Скелет (2/1). The consumed ally is REMOVED from board and sent
    // to graveyard; its attack/hp/max_hp are added to the played card
    // (2+2=4, 2+1=3). The consumed ally's deathrattle does NOT fire. Multi-card
    // deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_consume_ally.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_consume_ally_full_python_transitions() {
    // CLN-3 full-board consume_ally parity: board.len()==5 + a consume_ally
    // warrior (Канеки, card 20) in hand. The frozen codec mask
    // (classic_actions_v1._mask_play_actions:228) does NOT exempt consume_ally,
    // so the consume play is masked OUT at a full board (Rust's action_mask
    // matches the frozen codec byte-for-byte — the prior incorrect exemption
    // was reverted). The engine apply path (core/engine.py:1228 + kernel.rs
    // apply_play_card board-full guard) STILL exempts consume_ally, so the
    // play applies when FORCED. This test forces the action via
    // apply_action_unchecked (mask-bypassing) and compares post-state JSON
    // only (no mask comparison). The fixture is generated without monkey-
    // patching the frozen codec: the consume play action_id is constructed
    // directly and force-applied via the engine apply path in the generator.
    let raw = include_str!("fixtures/golden_trace_consume_ally_full.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match_forced(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_mana_drain_python_transitions() {
    // Phase 7 FIX 3 (mana_drain two-stage): card 12 Кража Маны
    // (mana_drain_2) vs an opponent at mana=1. Immediate drain: opponent
    // mana 1→0 (core/effects.py:540-580). Pending drain: 1 scheduled for the
    // opponent's next turn via pending_mana_drain_by_player (core/state.py:183
    // field). End-turn applies the pending: opponent mana 10→9
    // (core/engine.py:700-703). The fixture's pre/post state payloads
    // include pending_mana_drain_by_player so the cross-step scheduled drain
    // is captured. State-transition only.
    let raw = include_str!("fixtures/golden_trace_mana_drain.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_cast_random_spell_python_transitions() {
    // Phase 9 (CRS-1): card 26 Мидория (cast_random_spell, 5 mana 5/5). On play,
    // the engine rolls random.randint(1,4) to pick one of 4 spells and applies
    // it: 1) Texas Smash — 4+(level-1) damage to a random enemy minion
    // (random.choice); 2) Recovery — 5+(level-1) heal to owner hero; 3) Black
    // Whip — freeze_count = 2 if level>=5 else 1, freeze random.sample enemy
    // minions (consume_shield blocks); 4) Full Cowl — 2+((level-1)/2) buff to
    // the played card's attack/hp/max_hp. level is always 1 in the classic
    // training env (deck_from_card_ids defaults level=1) → scaling is dmg=4,
    // heal=5, freeze_count=1, buff=2. The fixture exercises all 4 spells across
    // 9 casts (seed 400, 43 steps). The recorded-outcome RNG protocol threads
    // randint_rolls (the 1..=4 spell pick), choice_rolls (spell 1 target) and
    // sample_rolls (spell 3 freeze targets). All plays are mask-legal (no
    // target required → no-target slot), so apply_action_unchecked (mask-
    // bypassing but mask-legal here) applies cleanly. State-transition only.
    let raw = include_str!("fixtures/golden_trace_cast_random_spell.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match_forced(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_cleave_python_transitions() {
    // Phase 4 (TA-1 cleave): card 23 Сукуна (cleave_1_2, 7/7) attacks the
    // middle of 3 Скелет lvl5 (3/2). The main target dies to 7 damage;
    // cleave splashes 1 damage to each living neighbour (indices 0 and 2),
    // leaving them at 1 HP. `cleave_X_Y` parses X=1 for the splash (the _Y
    // hit-count is the potion-battlecry form, unused for warrior attack
    // cleave). The counter-attack (3) drops Сукуна to 4 HP. Multi-card
    // deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_cleave.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_instant_kill_python_transitions() {
    // Phase 4 (TA-2 instant_kill): card 25 Сайтама (instant_kill, 10/10)
    // attacks Атакующий Титан lvl10 (15/15). Normal damage would leave 5 HP;
    // instant_kill sets hp=0 → Титан dies (and Сайтама dies to the 15
    // counter). Without instant_kill Титан would survive at 5 HP, so this
    // fixture distinguishes the mechanic. instant_kill_used is set (one-shot)
    // but NOT serialized into the state payload (skip_serializing), so the
    // matcher verifies the kill effect, not the flag. Multi-card deck;
    // recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_instant_kill.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_freeze_python_transitions() {
    // Phase 4 (TA-3 freeze): card 19 Саб-Зиро (battlecry_freeze, 3/4) is
    // played targeting an enemy Скелет → is_frozen=True (shield would block,
    // not exercised here). The next end-turn thaws it (is_frozen=False,
    // is_ready=False — skips one activation). The fixture verifies both the
    // frozen post-state of the play step and the thawed/skipped post-state
    // of the following end-turn. Multi-card deck; recorded-outcome RNG.
    // State-transition only.
    let raw = include_str!("fixtures/golden_trace_freeze.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_armor_python_transitions() {
    // Phase 4 (TA-4 armor_X_Y range roll + recorded-outcome RNG extension):
    // Даркнесс hero (card 5) with `armor_1_3` injected post-conversion (the
    // converter collapses armor_X_Y → armor_X at deck-construction, so the
    // fixture re-injects the range form to exercise the engine's randint
    // path). Сукуна (7 atk) attacks the hero → apply_damage_modifiers rolls
    // randint(1,3)=1 → hero takes 6 (45→39). The fixture's step 4 carries
    // `randint_rolls=[1]`; Rust's `roll_range` pops that recorded value via
    // `DrawRng::Recorded.randint_rolls`. verify_mask=False on this fixture
    // avoids phantom randint rolls from `_verify_mask` clone-execution.
    // Multi-card deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_armor.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_aoe_silence_python_transitions() {
    // Phase 5 (AOE-SILENCE-1): card 47 Солдатик (cost 7, 4/5, aoe_silence).
    // p1 fills board with 3 Сукуна (card 23, cleave_1_2 — a passive warrior
    // attack mechanic, no play target required). p2 plays Солдатик →
    // aoe_silence strips `mechanics` from up to 3 enemy minions that have
    // mechanics (all 3 Сукуна → cleave_1_2 removed, mechanics=[]). Mirrors
    // core/effects.py `effect_aoe_silence` (limit=3, candidates =
    // `[u for u in opponent.board if u.mechanics]`, `unit.mechanics = []`).
    // Status flags (is_frozen/is_ready/instant_kill_used) are NOT touched.
    // Multi-card deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_aoe_silence.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_team_wide_shield_python_transitions() {
    // Phase 5 (TWS-1/2): card 48 Соул Гудман (cost 7, 2/4, team_wide_shield).
    // p1 plays 3 Скелет (card 27, no mechanics), then plays Соул Гудман →
    // team_wide_shield grants `shield` to up to 3 friendly minions EXCLUDING
    // the just-played card (TWS-2 self-exclusion — mirrors core/effects.py
    // `effect_team_wide_shield`:
    // `targets = [u for u in owner.board if u.instance_id != card.instance_id]`).
    // After: 3 Скелет gain `shield`; Соул Гудман does NOT have `shield`.
    // NOTE: the audit note TWS-2 in the task brief speculated Python "includes"
    // the played card; Python in fact SELF-EXCLUDES it (code+comment confirm),
    // and Rust matches Python (the frozen oracle). Multi-card deck; recorded-
    // outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_team_wide_shield.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_tamhp_python_transitions() {
    // Phase 5 (TAMHP-1/2/3, AC-FFI-4): card 52 Криста Ленц (cost 2, 1/2,
    // target_ally_max_hp_plus_universal_1). Played by BOTH players (user
    // decision: "playable everywhere"). Exercises BOTH target families:
    //   - friendly minion target (code 9): p1 and p2 each play Криста
    //     targeting their Скелет → Скелет.max_hp += 1 (hp UNCHANGED — direct
    //     increase, NO clamp via heal_card; audit risk note honored).
    //   - own hero target (code 16): p1 plays Криста targeting own hero →
    //     hero.max_hp += 1 (hp unchanged).
    // Mirrors core/effects.py `target_ally_max_hp_plus_universal_N` handler.
    // The Rust `requires_target` + `mask_targets_for_card` additive branch
    // (friendly-minion codes 9..15 + own-hero code 16) makes card52 playable
    // on both sides. Multi-card deck; recorded-outcome RNG. State-transition
    // only.
    let raw = include_str!("fixtures/golden_trace_tamhp.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_random_battlecry_python_transitions() {
    // Phase 6 (AC-FFI-2, TA-5): card 15 Тока Киришима (cost 2, 2/1, warrior,
    // `battlecry_damage_1_random`). Python `_apply_random_battlecry_damage`
    // (core/effects.py line 67-83) builds `targets = list(opponent.board) +
    // [opponent.hero]` and picks one via `random.choice`, dealing 1 damage.
    // The chosen index is recorded in the new `choice_rolls` stream; Rust
    // replays it via `roll_choice` (mechanic-driven, NOT card_id). p1 has 2
    // Скелет on board → targets list has 3 entries; the recorded roll picks
    // index 1 (2nd Скелет) which dies (hp 1→0, removed). Multi-card deck;
    // recorded-outcome RNG (choice_rolls). State-transition only.
    let raw = include_str!("fixtures/golden_trace_random_battlecry.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_shield_refresh_python_transitions() {
    // Phase 6 (TWS-3, AC-FFI-3): card 24 Годжо Сатору (cost 9, 5/6, warrior,
    // `["shield", "shield_refresh"]`). Python `core/engine.py::_handle_end_turn`
    // (line 728) refreshes `shield` at the start of the owner's turn when the
    // unit has `shield_refresh` and no current `shield`. The fixture consumes
    // the shield via an enemy Скелет attack (shield gone, hp unchanged), then
    // ends the enemy turn → owner's turn start re-adds `shield` (mechanic-
    // driven via `has_mechanic("shield_refresh")`, NOT card_id == 24). No
    // choice_rolls (no random.choice mechanic exercised). State-transition only.
    let raw = include_str!("fixtures/golden_trace_shield_refresh.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_sudden_death_python_transitions() {
    // Phase 7 (WD-1): sudden-death modifier. The env enables
    // `sudden_death_enabled` with damage_start=1, damage_step=1. The active
    // player's hero takes escalating damage at the start of each of their
    // turns: damage = 1 + (turn_count-1)*1, applied via DIRECT hp subtraction
    // (bypasses armor/reflect — mirrors core/engine.py
    // `_apply_start_turn_mode_effects` line 837 `player.hero.hp -= damage`).
    // The init-time tick (p1 @ turn 1) is baked into the recorded initial
    // state (p1 hp 45→44). The 8 end_turn steps then apply: p2@2(1), p1@3(2),
    // p2@4(2), p1@5(3), p2@6(3), p1@7(4), p2@8(4), p1@9(5) → final p1=30,
    // p2=35. The fixture records `sudden_death_turns_by_player`={1:5,2:4} and
    // `sudden_death_last_applied_turn_by_player`={1:9,2:8}; Rust's
    // `KernelState` serializes these fields so the state-transition matcher
    // verifies the per-player escalating counter, not just the hp deltas.
    // Multi-card deck; recorded-outcome RNG. State-transition only.
    let raw = include_str!("fixtures/golden_trace_sudden_death.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    assert_trace_state_transitions_match(&trace);
}

#[test]
fn rust_kernel_apply_action_matches_truncation_python_transitions() {
    // Phase 7 (WD-2): max_turns truncation. The env config sets max_turns=4.
    // Python `ClassicRLEnv.step` returns `truncated = st.turn_number >
    // self._max_turns` (strictly greater-than). The 4 end_turn steps advance
    // turn_number 1→2→3→4→5; step 3 crosses the boundary (turn 5 > 4) →
    // truncated=True, terminated=False (status still "ongoing"). This test
    // asserts BOTH post-state JSON equality AND that Rust's `apply_action`
    // returns `truncated` matching the fixture's recorded `truncated` flag at
    // every step (the generic state-transition matcher only compares state
    // JSON, so the truncated flag is checked explicitly here). No
    // sudden-death. Multi-card deck; recorded-outcome RNG. State-transition
    // + truncated-flag parity.
    let raw = include_str!("fixtures/golden_trace_truncation.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    for step in &trace.steps {
        let mut draw_rng = DrawRng::recorded(
            step.draw_picks.clone(),
            step.reshuffle_orders.clone(),
            step.randint_rolls.clone(),
            step.choice_rolls.clone(),
        );
        let actual = kernel
            .apply_action(
                &step.pre.state,
                step.acting_player_id,
                step.action_id,
                step.mana_draw_taken,
                &mut draw_rng,
            )
            .expect("scripted action applies");
        let actual_json = serde_json::to_string(&actual.state).expect("serialize actual state");
        let expected_json =
            serde_json::to_string(&step.post.state).expect("serialize expected state");
        assert_eq!(actual_json, expected_json, "step {} post-state JSON mismatch", step.t);
        assert_eq!(
            actual.truncated, step.truncated,
            "step {} truncated flag mismatch",
            step.t
        );
        assert_eq!(
            actual.terminated, step.terminated,
            "step {} terminated flag mismatch",
            step.t
        );
    }
}

#[test]
fn batched_rollout_worker_matches_python_trace_with_internal_history() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let mut worker = BatchedRolloutWorker::from_snapshots(
        config,
        &[trace.initial.clone(), trace.initial.clone()],
    );

    let initial = worker.encode_all();
    assert_eq!(initial.env_count, 2);
    assert_snapshot_slice_matches(
        &initial.observation_v5,
        0,
        6480,
        &trace.initial.obs_v5_sha256_f32_le,
    );
    assert_snapshot_slice_matches(
        &initial.observation_v5,
        1,
        6480,
        &trace.initial.obs_v5_sha256_f32_le,
    );

    for step in &trace.steps {
        let out = worker
            .step(&[step.action_id, step.action_id])
            .expect("batch step applies");
        assert_eq!(out.env_count, 2);
        assert!(
            (out.rewards[0] - step.reward).abs() < 1e-6,
            "step {} env 0 reward",
            step.t
        );
        assert!(
            (out.rewards[1] - step.reward).abs() < 1e-6,
            "step {} env 1 reward",
            step.t
        );
        assert_snapshot_slice_matches(
            &out.observation_v1,
            0,
            1456,
            &Some(step.post.obs_sha256_f32_le.clone()),
        );
        assert_snapshot_slice_matches(
            &out.observation_v1,
            1,
            1456,
            &Some(step.post.obs_sha256_f32_le.clone()),
        );
        assert_snapshot_slice_matches(
            &out.observation_v5,
            0,
            6480,
            &step.post.obs_v5_sha256_f32_le,
        );
        assert_snapshot_slice_matches(
            &out.observation_v5,
            1,
            6480,
            &step.post.obs_v5_sha256_f32_le,
        );
        assert_snapshot_slice_matches(
            &out.action_mask,
            0,
            601,
            &Some(step.post.mask_sha256_f32_le.clone()),
        );
        assert_snapshot_slice_matches(
            &out.action_features,
            0,
            601 * 171,
            &Some(step.post.action_features_sha256_f32_le.clone()),
        );
        assert_compact_action_features_match_dense(&out, 0);
        assert_compact_action_features_match_dense(&out, 1);
    }
}

#[test]
fn trace_pool_slots_keep_their_seeded_v5_info_modes() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let mut hidden_trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let mut visible_trace = hidden_trace.clone();
    hidden_trace.env_config.enemy_hand_known = false;
    visible_trace.env_config.enemy_hand_known = true;

    let config = KernelConfig::from_trace_config(&hidden_trace.env_config);
    let mut worker = BatchedRolloutWorker::from_snapshot_pool_with_trace_configs(
        config,
        &[
            (
                hidden_trace.initial.clone(),
                hidden_trace.env_config.clone(),
            ),
            (
                visible_trace.initial.clone(),
                visible_trace.env_config.clone(),
            ),
        ],
        2,
        ActionFeatureOutput::DenseAndLegal,
        ObservationOutput::V1AndV5,
        ActionMaskOutput::Dense,
        TerminalObservationOutput::Full,
        DiagnosticOutput::Full,
        true,
    )
    .expect("compatible trace pool constructs");

    let hidden_flag = OBS_DIM_V1 + 3;
    let visible_flag = 6480 + OBS_DIM_V1 + 3;
    let initial = worker.encode_all();
    assert_eq!(initial.observation_v5[hidden_flag], 0.0);
    assert_eq!(initial.observation_v5[visible_flag], 1.0);

    worker.reset_indices(&[0]).expect("slot reset succeeds");
    let cycled = worker.encode_all();
    assert_eq!(cycled.observation_v5[hidden_flag], 1.0);
    assert_eq!(cycled.observation_v5[visible_flag], 1.0);
}

#[test]
fn trace_pool_slots_keep_their_seeded_v5_assist_modes() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let mut no_assist_trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let mut assisted_trace = no_assist_trace.clone();
    no_assist_trace.env_config.assembler_enabled = false;
    no_assist_trace.env_config.desirerer_enabled = false;
    no_assist_trace.env_config.teacher_hint_available = false;
    no_assist_trace.env_config.assist_profile_id = 0;
    assisted_trace.env_config.assembler_enabled = true;
    assisted_trace.env_config.assembler_strength = 0.9;
    assisted_trace.env_config.desirerer_enabled = true;
    assisted_trace.env_config.desirerer_strength = 0.7;
    assisted_trace.env_config.teacher_hint_available = true;
    assisted_trace.env_config.assist_profile_id = 3;

    let config = KernelConfig::from_trace_config(&no_assist_trace.env_config);
    let worker = BatchedRolloutWorker::from_snapshot_pool_with_trace_configs(
        config,
        &[
            (
                no_assist_trace.initial.clone(),
                no_assist_trace.env_config.clone(),
            ),
            (
                assisted_trace.initial.clone(),
                assisted_trace.env_config.clone(),
            ),
        ],
        2,
        ActionFeatureOutput::DenseAndLegal,
        ObservationOutput::V1AndV5,
        ActionMaskOutput::Dense,
        TerminalObservationOutput::Full,
        DiagnosticOutput::Full,
        true,
    )
    .expect("compatible trace pool constructs");

    let first = OBS_DIM_V1;
    let second = 6480 + OBS_DIM_V1;
    let initial = worker.encode_all();
    assert_eq!(initial.observation_v5[first + 9], 0.0);
    assert_eq!(initial.observation_v5[second + 9], 1.0);
    assert_eq!(initial.observation_v5[second + 10], 0.9);
    assert_eq!(initial.observation_v5[second + 11], 1.0);
    assert_eq!(initial.observation_v5[second + 12], 0.7);
    assert_eq!(initial.observation_v5[second + 13], 1.0);
    assert_eq!(initial.observation_v5[second + 14], 3.0 / 16.0);
}

#[test]
fn trace_pool_rejects_incompatible_shared_settings() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let mut incompatible = trace.clone();
    incompatible.env_config.mana_per_turn = trace.env_config.mana_per_turn + 1;

    let config = KernelConfig::from_trace_config(&trace.env_config);
    let err = BatchedRolloutWorker::from_snapshot_pool_with_trace_configs(
        config,
        &[
            (trace.initial.clone(), trace.env_config.clone()),
            (
                incompatible.initial.clone(),
                incompatible.env_config.clone(),
            ),
        ],
        2,
        ActionFeatureOutput::DenseAndLegal,
        ObservationOutput::V1AndV5,
        ActionMaskOutput::Dense,
        TerminalObservationOutput::Full,
        DiagnosticOutput::Full,
        false,
    )
    .expect_err("mixed mana_per_turn must be rejected");

    assert!(err.contains("mana_per_turn"), "{err}");
}

#[test]
fn batched_rollout_worker_resets_to_initial_snapshots_after_steps() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let mut worker = BatchedRolloutWorker::from_snapshots(
        config,
        &[trace.initial.clone(), trace.initial.clone()],
    );

    let initial = worker.encode_all();
    for step in &trace.steps {
        worker
            .step(&[step.action_id, step.action_id])
            .expect("batch step applies");
    }

    worker.reset_all();
    let reset = worker.encode_all();
    assert_eq!(reset.env_count, initial.env_count);
    assert_eq!(reset.rewards, vec![0.0, 0.0]);
    assert_eq!(reset.terminated, initial.terminated);
    assert_eq!(
        hash_f32_le(&reset.observation_v1),
        hash_f32_le(&initial.observation_v1)
    );
    assert_eq!(
        hash_f32_le(&reset.observation_v5),
        hash_f32_le(&initial.observation_v5)
    );
    assert_eq!(
        hash_f32_le(&reset.action_mask),
        hash_f32_le(&initial.action_mask)
    );
    assert_eq!(
        hash_f32_le(&reset.action_features),
        hash_f32_le(&initial.action_features)
    );
}

#[test]
fn batched_rollout_worker_resets_selected_envs_only() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let mut worker = BatchedRolloutWorker::from_snapshots(
        config,
        &[
            trace.initial.clone(),
            trace.initial.clone(),
            trace.initial.clone(),
        ],
    );

    let initial = worker.encode_all();
    worker
        .step(&[trace.steps[0].action_id, 0, trace.steps[0].action_id])
        .expect("mixed batch step applies");

    worker.reset_indices(&[0, 2]).expect("selected envs reset");
    let reset = worker.encode_all();
    assert_eq!(
        hash_slice(&reset.observation_v5, 0, 6480),
        hash_slice(&initial.observation_v5, 0, 6480)
    );
    assert_ne!(
        hash_slice(&reset.observation_v5, 1, 6480),
        hash_slice(&initial.observation_v5, 1, 6480)
    );
    assert_eq!(
        hash_slice(&reset.observation_v5, 2, 6480),
        hash_slice(&initial.observation_v5, 2, 6480)
    );
    assert!(worker.reset_indices(&[3]).is_err());
}

#[test]
fn batched_rollout_worker_auto_resets_terminated_envs_after_step() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let mut lethal = trace.steps[4].pre.clone();
    lethal.state.p2.hero.hp = 1;
    lethal.state.p2.hero.max_hp = 1;
    let mut no_reset_worker =
        BatchedRolloutWorker::from_snapshots(config, &[lethal.clone(), trace.initial.clone()]);
    let mut worker = BatchedRolloutWorker::from_snapshots(config, &[lethal, trace.initial.clone()]);

    let initial = worker.encode_all();
    let terminal_expected = no_reset_worker
        .step(&[trace.steps[4].action_id, trace.steps[0].action_id])
        .expect("plain batch step applies");
    let out = worker
        .step_auto_reset(&[trace.steps[4].action_id, trace.steps[0].action_id])
        .expect("auto-reset batch step applies");
    assert_eq!(out.terminated, vec![true, false]);
    assert_eq!(out.reset_flags, vec![true, false]);
    assert_eq!(out.terminal_observation_valid, vec![true, false]);
    assert_eq!(out.episode_lengths, vec![1, 1]);
    assert!((out.episode_returns[0] - terminal_expected.rewards[0]).abs() < 1e-6);
    assert!((out.episode_returns[1] - trace.steps[0].reward).abs() < 1e-6);
    assert_eq!(
        hash_slice(&out.observation_v5, 0, 6480),
        hash_slice(&initial.observation_v5, 0, 6480)
    );
    assert_eq!(
        hash_slice(&out.terminal_observation_v5, 0, 6480),
        hash_slice(&terminal_expected.observation_v5, 0, 6480)
    );
    assert_ne!(
        hash_slice(&out.terminal_observation_v5, 0, 6480),
        hash_slice(&out.observation_v5, 0, 6480)
    );
    assert_eq!(
        hash_slice(&out.terminal_observation_v5, 1, 6480),
        hash_zeroes(6480)
    );
    assert_ne!(
        hash_slice(&out.observation_v5, 1, 6480),
        hash_slice(&initial.observation_v5, 1, 6480)
    );

    let encoded = worker.encode_all();
    assert_eq!(
        hash_slice(&encoded.observation_v5, 0, 6480),
        hash_slice(&initial.observation_v5, 0, 6480)
    );
    assert_eq!(encoded.episode_lengths, vec![0, 1]);
    assert!((encoded.episode_returns[0] - 0.0).abs() < 1e-6);
    assert!((encoded.episode_returns[1] - trace.steps[0].reward).abs() < 1e-6);
}

#[test]
fn ffi_worker_exposes_batched_tensor_buffers() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let worker = unsafe { trainv3_worker_from_trace_json(raw.as_ptr(), raw.len(), 2) };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_eq!(obs_v5.len(), 2 * 6480);
        assert_snapshot_slice_matches(&obs_v5, 0, 6480, &trace.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&obs_v5, 1, 6480, &trace.initial.obs_v5_sha256_f32_le);
    }

    let actions = [trace.steps[0].action_id, trace.steps[0].action_id];
    let stepped = unsafe { trainv3_worker_step(worker, actions.as_ptr(), actions.len()) };
    assert_eq!(stepped, 0);
    unsafe {
        let obs_v1 = std::slice::from_raw_parts(
            trainv3_worker_observation_v1_ptr(worker),
            trainv3_worker_observation_v1_len(worker),
        );
        assert_snapshot_slice_matches(
            &obs_v1,
            0,
            1456,
            &Some(trace.steps[0].post.obs_sha256_f32_le.clone()),
        );
        let rewards = std::slice::from_raw_parts(
            trainv3_worker_rewards_ptr(worker),
            trainv3_worker_rewards_len(worker),
        );
        assert_eq!(rewards.len(), 2);
        assert!((rewards[0] - trace.steps[0].reward).abs() < 1e-6);
        assert!((rewards[1] - trace.steps[0].reward).abs() < 1e-6);
        assert_eq!(trainv3_worker_action_features_len(worker), 2 * 601 * 171);

        let reset = trainv3_worker_reset(worker);
        assert_eq!(reset, 0);
        let reset_obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        let reset_env0_hash = hash_slice(&reset_obs_v5, 0, 6480);
        let reset_env1_hash = hash_slice(&reset_obs_v5, 1, 6480);
        assert_snapshot_slice_matches(&reset_obs_v5, 0, 6480, &trace.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&reset_obs_v5, 1, 6480, &trace.initial.obs_v5_sha256_f32_le);

        let mixed_actions = [trace.steps[0].action_id, 0];
        assert_eq!(
            trainv3_worker_step(worker, mixed_actions.as_ptr(), mixed_actions.len()),
            0
        );
        let reset_indices = [1_usize];
        assert_eq!(
            trainv3_worker_reset_indices(worker, reset_indices.as_ptr(), reset_indices.len()),
            0
        );
        let selected_reset_obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_ne!(hash_slice(&selected_reset_obs_v5, 0, 6480), reset_env0_hash);
        assert_eq!(hash_slice(&selected_reset_obs_v5, 1, 6480), reset_env1_hash);
        let invalid_indices = [2_usize];
        assert_eq!(
            trainv3_worker_reset_indices(worker, invalid_indices.as_ptr(), invalid_indices.len()),
            -3
        );
        let auto_actions = [0_usize, 0_usize];
        assert_eq!(
            trainv3_worker_step_auto_reset(worker, auto_actions.as_ptr(), auto_actions.len()),
            0
        );
        let reset_flags = std::slice::from_raw_parts(
            trainv3_worker_reset_flags_ptr(worker),
            trainv3_worker_reset_flags_len(worker),
        );
        assert_eq!(reset_flags, &[0, 0]);
        let episode_returns = std::slice::from_raw_parts(
            trainv3_worker_episode_returns_ptr(worker),
            trainv3_worker_episode_returns_len(worker),
        );
        let episode_lengths = std::slice::from_raw_parts(
            trainv3_worker_episode_lengths_ptr(worker),
            trainv3_worker_episode_lengths_len(worker),
        );
        assert_eq!(episode_returns.len(), 2);
        assert_eq!(episode_lengths, &[2, 1]);
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_seeds_env_slots_from_trace_json_pool() {
    let raw_a = include_str!("fixtures/golden_trace_scripted_basic.json");
    let raw_b = include_str!("fixtures/golden_trace_seed123.json");
    let trace_a: GoldenTrace = serde_json::from_str(raw_a).expect("fixture parses");
    let trace_b: GoldenTrace = serde_json::from_str(raw_b).expect("fixture parses");
    let pool = format!("[{raw_a},{raw_b}]");
    let worker = unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v2(pool.as_ptr(), pool.len(), 4, 0, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_eq!(obs_v5.len(), 4 * 6480);
        assert_snapshot_slice_matches(&obs_v5, 0, 6480, &trace_a.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&obs_v5, 1, 6480, &trace_b.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&obs_v5, 2, 6480, &trace_a.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&obs_v5, 3, 6480, &trace_b.initial.obs_v5_sha256_f32_le);
        assert_ne!(hash_slice(&obs_v5, 0, 6480), hash_slice(&obs_v5, 1, 6480));
        assert_eq!(trainv3_worker_action_features_len(worker), 0);
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_rolls_out_action_tape_in_one_call() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let env_count = 2;
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v2(raw.as_ptr(), raw.len(), env_count, 0, 1)
    };
    assert!(!worker.is_null());

    let mut actions = Vec::new();
    for step in &trace.steps {
        actions.extend(std::iter::repeat(step.action_id).take(env_count));
    }
    let rollout = unsafe {
        trainv3_worker_rollout_action_tape(
            worker,
            actions.as_ptr(),
            actions.len(),
            trace.steps.len(),
            0,
        )
    };
    assert_eq!(rollout, 0);

    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        let rewards = std::slice::from_raw_parts(
            trainv3_worker_rewards_ptr(worker),
            trainv3_worker_rewards_len(worker),
        );
        assert_eq!(obs_v5.len(), trace.steps.len() * env_count * 6480);
        assert_eq!(rewards.len(), trace.steps.len() * env_count);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);

        for (step_idx, step) in trace.steps.iter().enumerate() {
            for env_idx in 0..env_count {
                let row_idx = step_idx * env_count + env_idx;
                assert_snapshot_slice_matches(
                    obs_v5,
                    row_idx,
                    6480,
                    &step.post.obs_v5_sha256_f32_le,
                );
                assert!(
                    (rewards[row_idx] - step.reward).abs() < 1e-6,
                    "step {step_idx} env {env_idx} reward"
                );
            }
        }
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_rolls_out_broadcast_action_ids_in_one_call() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let env_count = 2;
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v2(raw.as_ptr(), raw.len(), env_count, 0, 1)
    };
    assert!(!worker.is_null());

    let actions = trace
        .steps
        .iter()
        .map(|step| step.action_id)
        .collect::<Vec<_>>();
    let rollout = unsafe {
        trainv3_worker_rollout_broadcast_action_ids(worker, actions.as_ptr(), actions.len(), 0)
    };
    assert_eq!(rollout, 0);

    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        let rewards = std::slice::from_raw_parts(
            trainv3_worker_rewards_ptr(worker),
            trainv3_worker_rewards_len(worker),
        );
        assert_eq!(obs_v5.len(), trace.steps.len() * env_count * 6480);
        assert_eq!(rewards.len(), trace.steps.len() * env_count);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);

        for (step_idx, step) in trace.steps.iter().enumerate() {
            for env_idx in 0..env_count {
                let row_idx = step_idx * env_count + env_idx;
                assert_snapshot_slice_matches(
                    obs_v5,
                    row_idx,
                    6480,
                    &step.post.obs_v5_sha256_f32_le,
                );
                assert!(
                    (rewards[row_idx] - step.reward).abs() < 1e-6,
                    "step {step_idx} env {env_idx} reward"
                );
            }
        }
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_rolls_out_pre_step_action_tape_for_ppo_batches() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let env_count = 2;
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v5(
            raw.as_ptr(),
            raw.len(),
            env_count,
            0,
            1,
            1,
            1,
            1,
        )
    };
    assert!(!worker.is_null());

    let mut actions = Vec::new();
    for step in &trace.steps {
        actions.extend(std::iter::repeat(step.action_id).take(env_count));
    }
    let rollout = unsafe {
        trainv3_worker_rollout_action_tape_pre_step(
            worker,
            actions.as_ptr(),
            actions.len(),
            trace.steps.len(),
            0,
        )
    };
    assert_eq!(rollout, 0);

    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        let rewards = std::slice::from_raw_parts(
            trainv3_worker_rewards_ptr(worker),
            trainv3_worker_rewards_len(worker),
        );
        assert_eq!(obs_v5.len(), trace.steps.len() * env_count * 6480);
        assert_eq!(rewards.len(), trace.steps.len() * env_count);
        assert_eq!(trainv3_worker_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_action_mask_len(worker), 0);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v5_len(worker), 0);

        for env_idx in 0..env_count {
            assert_snapshot_slice_matches(
                obs_v5,
                env_idx,
                6480,
                &trace.initial.obs_v5_sha256_f32_le,
            );
        }
        for (step_idx, step) in trace.steps.iter().enumerate() {
            for env_idx in 0..env_count {
                let row_idx = step_idx * env_count + env_idx;
                assert!(
                    (rewards[row_idx] - step.reward).abs() < 1e-6,
                    "step {step_idx} env {env_idx} reward"
                );
            }
        }
        for env_idx in 0..env_count {
            let row_idx = env_count + env_idx;
            assert_snapshot_slice_matches(
                obs_v5,
                row_idx,
                6480,
                &trace.steps[0].post.obs_v5_sha256_f32_le,
            );
        }
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_rolls_out_broadcast_pre_step_action_ids_for_ppo_batches() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let env_count = 2;
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v5(
            raw.as_ptr(),
            raw.len(),
            env_count,
            0,
            1,
            1,
            1,
            1,
        )
    };
    assert!(!worker.is_null());

    let actions = trace
        .steps
        .iter()
        .map(|step| step.action_id)
        .collect::<Vec<_>>();
    let rollout = unsafe {
        trainv3_worker_rollout_broadcast_action_ids_pre_step(
            worker,
            actions.as_ptr(),
            actions.len(),
            0,
        )
    };
    assert_eq!(rollout, 0);

    unsafe {
        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        let rewards = std::slice::from_raw_parts(
            trainv3_worker_rewards_ptr(worker),
            trainv3_worker_rewards_len(worker),
        );
        assert_eq!(obs_v5.len(), trace.steps.len() * env_count * 6480);
        assert_eq!(rewards.len(), trace.steps.len() * env_count);
        assert_eq!(trainv3_worker_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_action_mask_len(worker), 0);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v5_len(worker), 0);

        for env_idx in 0..env_count {
            assert_snapshot_slice_matches(
                obs_v5,
                env_idx,
                6480,
                &trace.initial.obs_v5_sha256_f32_le,
            );
        }
        for (step_idx, step) in trace.steps.iter().enumerate() {
            for env_idx in 0..env_count {
                let row_idx = step_idx * env_count + env_idx;
                assert!(
                    (rewards[row_idx] - step.reward).abs() < 1e-6,
                    "step {step_idx} env {env_idx} reward"
                );
            }
        }
        for env_idx in 0..env_count {
            let row_idx = env_count + env_idx;
            assert_snapshot_slice_matches(
                obs_v5,
                row_idx,
                6480,
                &trace.steps[0].post.obs_v5_sha256_f32_le,
            );
        }
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_can_omit_standalone_v1_observation_output() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v3(raw.as_ptr(), raw.len(), 2, 0, 1, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        assert_eq!(trainv3_worker_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_observation_v5_len(worker), 2 * 6480);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);

        let obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_snapshot_slice_matches(&obs_v5, 0, 6480, &trace.initial.obs_v5_sha256_f32_le);
        assert_snapshot_slice_matches(&obs_v5, 1, 6480, &trace.initial.obs_v5_sha256_f32_le);
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_can_omit_dense_action_mask_output() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v4(raw.as_ptr(), raw.len(), 2, 0, 1, 0, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        assert_eq!(trainv3_worker_action_mask_len(worker), 0);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);
        assert_eq!(trainv3_worker_legal_action_counts_len(worker), 2);
        assert_eq!(trainv3_worker_observation_v5_len(worker), 2 * 6480);
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_can_omit_terminal_observation_output() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v5(raw.as_ptr(), raw.len(), 2, 0, 1, 0, 0, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        assert_eq!(trainv3_worker_terminal_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v5_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_valid_len(worker), 2);
        assert_eq!(trainv3_worker_observation_v5_len(worker), 2 * 6480);
        trainv3_worker_free(worker);
    }
}

#[test]
fn pre_step_action_tape_preallocates_fixed_rollout_buffers() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let env_count = 2;
    let snapshots = vec![trace.initial.clone(); env_count];
    let mut worker = BatchedRolloutWorker::from_snapshots_with_outputs(
        config,
        &snapshots,
        ActionFeatureOutput::LegalOnly,
        ObservationOutput::V5Only,
        ActionMaskOutput::LegalOnly,
        TerminalObservationOutput::None,
        DiagnosticOutput::None,
    );
    let mut actions = Vec::new();
    for step in &trace.steps {
        actions.extend(std::iter::repeat(step.action_id).take(env_count));
    }

    let out = worker
        .rollout_action_tape_pre_step(&actions, trace.steps.len(), false)
        .expect("pre-step action tape rollout succeeds");

    let rows = trace.steps.len() * env_count;
    assert_eq!(out.observation_v5.len(), rows * 6480);
    assert_eq!(out.observation_v5.capacity(), rows * 6480);
    assert_eq!(out.selected_local_indices.len(), rows);
    assert_eq!(out.selected_local_indices.capacity(), rows);
    assert_eq!(out.rewards.capacity(), rows);
    assert_eq!(out.terminated.capacity(), rows);
    assert_eq!(out.legal_action_counts.capacity(), rows);
    assert_eq!(out.legal_action_offsets.capacity(), rows);
}

#[test]
fn ffi_worker_can_omit_reset_and_episode_diagnostic_output() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let worker = unsafe {
        trainv3_worker_from_trace_json_with_options_v6(raw.as_ptr(), raw.len(), 2, 0, 1, 1, 1, 1, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        assert_eq!(trainv3_worker_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_action_mask_len(worker), 0);
        assert_eq!(trainv3_worker_action_features_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v1_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_v5_len(worker), 0);
        assert_eq!(trainv3_worker_reset_flags_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_valid_len(worker), 0);
        assert_eq!(trainv3_worker_episode_returns_len(worker), 0);
        assert_eq!(trainv3_worker_episode_lengths_len(worker), 0);
        assert_eq!(trainv3_worker_observation_v5_len(worker), 2 * 6480);
        assert_eq!(trainv3_worker_legal_action_counts_len(worker), 2);
    }

    let actions = [1_usize, 1_usize];
    let stepped = unsafe { trainv3_worker_step(worker, actions.as_ptr(), actions.len()) };
    assert_eq!(stepped, 0);
    unsafe {
        assert_eq!(trainv3_worker_rewards_len(worker), 2);
        assert_eq!(trainv3_worker_reset_flags_len(worker), 0);
        assert_eq!(trainv3_worker_terminal_observation_valid_len(worker), 0);
        assert_eq!(trainv3_worker_episode_returns_len(worker), 0);
        assert_eq!(trainv3_worker_episode_lengths_len(worker), 0);
        trainv3_worker_free(worker);
    }
}

#[test]
fn ffi_worker_cycles_trace_pool_snapshots_on_reset() {
    let raw_a = include_str!("fixtures/golden_trace_scripted_basic.json");
    let raw_b = include_str!("fixtures/golden_trace_seed123.json");
    let trace_a: GoldenTrace = serde_json::from_str(raw_a).expect("fixture parses");
    let trace_b: GoldenTrace = serde_json::from_str(raw_b).expect("fixture parses");
    let pool = format!("[{raw_a},{raw_b}]");
    let worker = unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v3(pool.as_ptr(), pool.len(), 2, 0, 1, 1)
    };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        let initial_obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_snapshot_slice_matches(
            initial_obs_v5,
            0,
            6480,
            &trace_a.initial.obs_v5_sha256_f32_le,
        );
        assert_snapshot_slice_matches(
            initial_obs_v5,
            1,
            6480,
            &trace_b.initial.obs_v5_sha256_f32_le,
        );

        assert_eq!(trainv3_worker_reset(worker), 0);
        let reset_once_obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_snapshot_slice_matches(
            reset_once_obs_v5,
            0,
            6480,
            &trace_b.initial.obs_v5_sha256_f32_le,
        );
        assert_snapshot_slice_matches(
            reset_once_obs_v5,
            1,
            6480,
            &trace_a.initial.obs_v5_sha256_f32_le,
        );

        assert_eq!(trainv3_worker_reset(worker), 0);
        let reset_twice_obs_v5 = std::slice::from_raw_parts(
            trainv3_worker_observation_v5_ptr(worker),
            trainv3_worker_observation_v5_len(worker),
        );
        assert_snapshot_slice_matches(
            reset_twice_obs_v5,
            0,
            6480,
            &trace_a.initial.obs_v5_sha256_f32_le,
        );
        assert_snapshot_slice_matches(
            reset_twice_obs_v5,
            1,
            6480,
            &trace_b.initial.obs_v5_sha256_f32_le,
        );
        trainv3_worker_free(worker);
    }
}

#[test]
fn batched_rollout_worker_exposes_compact_legal_action_features() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let worker = BatchedRolloutWorker::from_snapshots(
        config,
        &[trace.initial.clone(), trace.initial.clone()],
    );

    let out = worker.encode_all();
    assert_eq!(out.legal_action_counts.len(), 2);
    assert_eq!(
        out.legal_action_ids.len(),
        out.legal_action_counts.iter().sum::<usize>()
    );
    assert_eq!(
        out.legal_action_features.len(),
        out.legal_action_ids.len() * 171
    );
    assert!(out.legal_action_features.len() < out.action_features.len());
    assert_compact_action_features_match_dense(&out, 0);
    assert_compact_action_features_match_dense(&out, 1);
}

#[test]
fn batched_rollout_worker_can_omit_dense_action_features_for_legal_only_output() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let trace: GoldenTrace = serde_json::from_str(raw).expect("fixture parses");
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let dense_worker = BatchedRolloutWorker::from_snapshots(
        config.clone(),
        &[trace.initial.clone(), trace.initial.clone()],
    );
    let legal_only_worker = BatchedRolloutWorker::from_snapshots_with_action_feature_output(
        config,
        &[trace.initial.clone(), trace.initial.clone()],
        ActionFeatureOutput::LegalOnly,
    );

    let dense = dense_worker.encode_all();
    let legal_only = legal_only_worker.encode_all();

    assert!(legal_only.action_features.is_empty());
    assert_eq!(legal_only.action_mask, dense.action_mask);
    assert_eq!(legal_only.legal_action_counts, dense.legal_action_counts);
    assert_eq!(legal_only.legal_action_ids, dense.legal_action_ids);
    assert_eq!(
        legal_only.legal_action_features,
        dense.legal_action_features
    );
}

#[test]
fn ffi_worker_exposes_float16_action_feature_buffer_when_requested() {
    let raw = include_str!("fixtures/golden_trace_scripted_basic.json");
    let worker =
        unsafe { trainv3_worker_from_trace_json_with_options(raw.as_ptr(), raw.len(), 2, 1) };
    assert!(!worker.is_null());

    let encoded = unsafe { trainv3_worker_encode(worker) };
    assert_eq!(encoded, 0);
    unsafe {
        let features_f16 = std::slice::from_raw_parts(
            trainv3_worker_action_features_f16_ptr(worker),
            trainv3_worker_action_features_f16_len(worker),
        );
        assert_eq!(features_f16.len(), 2 * 601 * 171);
        assert_eq!(features_f16[128], 0x3c00);
        trainv3_worker_free(worker);
    }
}

fn hash_slice(values: &[f32], env_idx: usize, width: usize) -> String {
    let start = env_idx * width;
    let end = start + width;
    hash_f32_le(&values[start..end])
}

fn hash_zeroes(width: usize) -> String {
    hash_f32_le(&vec![0.0_f32; width])
}

fn assert_snapshot_slice_matches(
    values: &[f32],
    env_idx: usize,
    width: usize,
    expected: &Option<String>,
) {
    let expected = expected.as_deref().expect("expected hash exists");
    let start = env_idx * width;
    let end = start + width;
    assert_eq!(
        hash_f32_le(&values[start..end]),
        expected,
        "env {env_idx} width {width}"
    );
}

fn assert_compact_action_features_match_dense(
    out: &trainv3_core::worker::BatchTensorOutput,
    env_idx: usize,
) {
    let dense_action_start = env_idx * 601;
    let dense_feature_start = env_idx * 601 * 171;
    let compact_offset = out.legal_action_offsets[env_idx];
    assert_eq!(
        compact_offset,
        out.legal_action_counts.iter().take(env_idx).sum::<usize>()
    );
    let count = out.legal_action_counts[env_idx];
    let ids = &out.legal_action_ids[compact_offset..compact_offset + count];
    let compact = &out.legal_action_features[compact_offset * 171..(compact_offset + count) * 171];

    let expected_ids: Vec<usize> = out.action_mask[dense_action_start..dense_action_start + 601]
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if *value > 0.0 { Some(idx) } else { None })
        .collect();
    assert_eq!(ids, expected_ids.as_slice());

    for (compact_row, action_id) in ids.iter().copied().enumerate() {
        let dense_start = dense_feature_start + action_id * 171;
        let compact_start = compact_row * 171;
        assert_eq!(
            &compact[compact_start..compact_start + 171],
            &out.action_features[dense_start..dense_start + 171],
            "env {env_idx} action {action_id}"
        );
    }
}

fn assert_trace_transitions_match(trace: &GoldenTrace) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));

    for step in &trace.steps {
        // Deterministic (zero) RNG: the frozen fixtures' draws are all
        // deck[0], and a zero-RNG makes weighted_choice_idx pick index 0,
        // so the recorded post-states reproduce exactly. Wrapped in
        // DrawRng::Live (task #14: apply_action now takes &mut DrawRng).
        let mut rng = WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let actual = kernel
            .apply_action(&step.pre.state, step.acting_player_id, step.action_id, step.mana_draw_taken, &mut draw_rng)
            .expect("scripted action applies");
        let actual_out = kernel.encode_snapshot_with_history(
            &actual.state,
            actual.state.current_turn_owner_id,
            &step.post.history_events,
        );
        assert_eq!(
            hash_f32_le(&actual_out.observation_v1),
            step.post.obs_sha256_f32_le,
            "step {} post obs_v1",
            step.t
        );
        assert_eq!(
            hash_f32_le(&actual_out.action_mask),
            step.post.mask_sha256_f32_le,
            "step {} post mask",
            step.t
        );
        assert_eq!(
            hash_f32_le(&actual_out.action_features),
            step.post.action_features_sha256_f32_le,
            "step {} post action_features",
            step.t
        );
        assert_eq!(
            hash_f32_le(&actual_out.observation_v5),
            step.post
                .obs_v5_sha256_f32_le
                .as_deref()
                .expect("v5 hash exists"),
            "step {} post obs_v5",
            step.t
        );
        assert_reward_components_close(actual.reward_components_v5, step.reward_components_v5);
    }
}

/// State-only transition parity: asserts that applying each step's recorded
/// action to its recorded pre-state produces a post-state that serializes to
/// the exact same JSON as the fixture's recorded post-state. This verifies
/// the gameplay-state logic the kernel owns (draw, cleanup, mechanics) without
/// depending on the observation encoder, which is at OLD-Python parity and
/// diverges from current Python `classic_obs_v1` (Phase 8 / CS-1 scope).
///
/// Recorded-outcome RNG (task #14 / DW-7): each step's `draw_picks` +
/// `reshuffle_orders` are replayed via a `DrawRng::Recorded`. For pre-#14
/// fixtures (no `draw_picks` field — `#[serde(default)]` → empty) the
/// `Recorded` stream is empty and `next_pick` falls back to 0 (matching the
/// old `Deterministic` zero-RNG idx-0 behaviour for single-card decks), and
/// `next_order` returns `None` (identity reshuffle — matches zero-RNG
/// `shuffle` for single-card graveyards). So this matcher stays backward-
/// compatible with the weighted_draw / mana_draw fixtures.
fn assert_trace_state_transitions_match(trace: &GoldenTrace) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    for step in &trace.steps {
        let mut draw_rng = DrawRng::recorded(
            step.draw_picks.clone(),
            step.reshuffle_orders.clone(),
            step.randint_rolls.clone(),
            step.choice_rolls.clone(),
        );
        let actual = kernel
            .apply_action(&step.pre.state, step.acting_player_id, step.action_id, step.mana_draw_taken, &mut draw_rng)
            .expect("scripted action applies");
        let actual_json = serde_json::to_string(&actual.state).expect("serialize actual state");
        let expected_json = serde_json::to_string(&step.post.state).expect("serialize expected state");
        assert_eq!(
            actual_json, expected_json,
            "step {} post-state JSON mismatch",
            step.t
        );
    }
}

/// Forced state-transition matcher: applies each step's action via
/// `apply_action_unchecked` (mask-bypassing engine apply path) instead of the
/// mask-checked `apply_action`. Used by the consume_ally_full parity test:
/// the frozen codec mask masks the consume_ally play OUT at a full board (no
/// exemption — Rust's action_mask matches the frozen codec byte-for-byte),
/// but the engine apply path exempts consume_ally (core/engine.py:1228 +
/// kernel.rs apply_play_card board-full guard), so the play applies when
/// forced. Compares post-state JSON only (no mask comparison). Includes the
/// `sample_rolls` recorded-outcome stream (Phase 9 cast_random_spell).
fn assert_trace_state_transitions_match_forced(trace: &GoldenTrace) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    for step in &trace.steps {
        let mut draw_rng = DrawRng::recorded_with_sample(
            step.draw_picks.clone(),
            step.reshuffle_orders.clone(),
            step.randint_rolls.clone(),
            step.choice_rolls.clone(),
            step.sample_rolls.clone(),
        );
        let actual = kernel
            .apply_action_unchecked(&step.pre.state, step.acting_player_id, step.action_id, &mut draw_rng)
            .expect("forced scripted action applies");
        let actual_json = serde_json::to_string(&actual.state).expect("serialize actual state");
        let expected_json = serde_json::to_string(&step.post.state).expect("serialize expected state");
        assert_eq!(
            actual_json, expected_json,
            "step {} post-state JSON mismatch (forced)",
            step.t
        );
    }
}

fn assert_snapshot_matches(kernel: &RolloutKernel, label: &str, snapshot: &GoldenSnapshot) {
    let out = kernel.encode_snapshot_with_history(
        &snapshot.state,
        snapshot.state.current_turn_owner_id,
        &snapshot.history_events,
    );
    assert_eq!(out.legal_ids(), snapshot.legal_ids, "{label} legal ids");
    assert_eq!(
        hash_f32_le(&out.action_mask),
        snapshot.mask_sha256_f32_le,
        "{label} mask"
    );
    assert_eq!(
        hash_f32_le(&out.action_features),
        snapshot.action_features_sha256_f32_le,
        "{label} action features"
    );
    assert_eq!(
        hash_f32_le(&out.observation_v1),
        snapshot.obs_sha256_f32_le,
        "{label} obs v1"
    );
    assert_eq!(
        hash_f32_le(&out.observation_v5),
        snapshot
            .obs_v5_sha256_f32_le
            .as_deref()
            .expect("v5 hash exists"),
        "{label} obs v5"
    );
}

fn assert_reward_components_close(actual: RewardComponentsV5, expected: RewardComponentsV5) {
    assert!((actual.hp_potential_delta - expected.hp_potential_delta).abs() < 1e-6);
    assert!((actual.board_power_delta - expected.board_power_delta).abs() < 1e-6);
    assert!((actual.my_board_power - expected.my_board_power).abs() < 1e-6);
    assert!((actual.enemy_board_power - expected.enemy_board_power).abs() < 1e-6);
    assert!((actual.board_power_ratio - expected.board_power_ratio).abs() < 1e-6);
    assert_eq!(actual.board_under_0_7, expected.board_under_0_7);
    assert_eq!(actual.own_board_wiped, expected.own_board_wiped);
    assert_eq!(actual.my_board_count_delta, expected.my_board_count_delta);
    assert_eq!(
        actual.enemy_board_count_delta,
        expected.enemy_board_count_delta
    );
}
