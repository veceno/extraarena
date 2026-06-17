from battle_engine import BattleEngine
from types import SimpleNamespace
from infrastructure.match_modes import (
    ClassicParams,
    EXTRA_ARENA_ROTATING_IDS,
    MODE_CONFIGS,
    ModeConfig,
    ROTATION_ANCHOR_EPOCH_SECONDS,
    ROTATION_INTERVAL_SECONDS,
    build_extra_arena_widget_payload,
    get_current_extra_arena_mode,
    resolve_canonical_mode_id,
    resolve_mode_config,
    serialize_mode_config,
)


def test_blitz_is_classic_preset_with_modifiers():
    config = resolve_mode_config("extra_arena:blitz")

    assert config.available is True
    assert config.ruleset == "classic"
    assert config.classic.turn_duration_seconds == 5
    assert config.classic.mana_per_turn == 2
    assert config.classic.hero_health_multiplier == 0.5
    assert config.rewards.enabled is True


def test_friendly_and_training_disable_rewards_without_new_ruleset():
    friendly = resolve_mode_config("friendly")
    training = resolve_mode_config("training")

    assert friendly.ruleset == "classic"
    assert training.ruleset == "classic"
    assert friendly.rewards.enabled is False
    assert training.rewards.trophies is False


def test_draft_is_known_but_not_available_until_ruleset_exists():
    config = resolve_mode_config("extra_arena:draft")

    assert config.ruleset == "draft"
    assert config.available is False


def test_mode_aliases_normalize_to_canonical_ids():
    assert resolve_mode_config("blitz").mode_id == "extra_arena:blitz"
    assert resolve_mode_config("ranked").mode_id == "classic"
    assert resolve_mode_config(None).mode_id == "classic"


def test_extraarena_aliases_resolve():
    assert resolve_mode_config("extraarena:blitz").mode_id == "extra_arena:blitz"
    assert resolve_mode_config("extraarena:spellstorm").mode_id == "extra_arena:spellstorm"
    assert resolve_mode_config("extraarena:blitzkrieg").mode_id == "extra_arena:blitzkrieg"
    assert resolve_mode_config("extraarena:sudden_death").mode_id == "extra_arena:sudden_death"
    assert resolve_mode_config("extraarena:powermax").mode_id == "extra_arena:powermax"


def test_new_modes_resolve_to_classic_ruleset():
    for mid in EXTRA_ARENA_ROTATING_IDS:
        cfg = resolve_mode_config(mid)
        assert cfg.ruleset == "classic", f"{mid} should be classic"
        assert cfg.available is True, f"{mid} should be available"


def test_train_v2_bot_safe_mode_only_allows_classic_and_training():
    from infrastructure import match_modes

    assert match_modes.is_train_v2_bot_safe_mode("classic") is True
    assert match_modes.is_train_v2_bot_safe_mode("training") is True
    assert match_modes.is_train_v2_bot_safe_mode(None) is True
    assert match_modes.is_train_v2_bot_safe_mode("ranked") is True
    assert match_modes.is_train_v2_bot_safe_mode("extra_arena:spellstorm") is False
    assert match_modes.is_train_v2_bot_safe_mode("extraarena:powermax") is False
    assert match_modes.is_train_v2_bot_safe_mode("friendly") is False
    assert match_modes.is_train_v2_bot_safe_mode("extra_arena:draft") is False
    assert match_modes.is_train_v2_bot_safe_mode("unknown") is False


def test_classic_has_no_new_flags():
    classic = resolve_mode_config("classic")
    assert classic.classic.spells_free is False
    assert classic.classic.summon_ready_on_play is False
    assert classic.classic.sudden_death_enabled is False
    assert classic.classic.bots_allowed is True


def test_battle_engine_uses_resolved_mode_params():
    engine = BattleEngine(game_mode="extra_arena:blitz")

    assert engine.game_mode == "extra_arena:blitz"
    assert engine.ruleset == "classic"
    assert engine.turn_duration == 5
    assert engine.mode_config.classic.mana_per_turn == 2


def test_card_level_modes_can_force_deck_levels():
    engine = BattleEngine(game_mode="classic")
    engine.mode_config = ModeConfig(
        mode_id="test:max-levels",
        ruleset="classic",
        label="Test Max Levels",
        classic=ClassicParams(card_level_mode="max"),
    )

    adjusted = engine._apply_card_level_mode({1: 3, 2: 7}, [1, "2:copy", "bad", 3])

    assert adjusted[1] == 10
    assert adjusted[2] == 10
    assert adjusted[3] == 10


def test_mode_config_serializes_for_api_payloads():
    payload = serialize_mode_config(resolve_mode_config("friendly"))

    assert payload["mode_id"] == "friendly"
    assert payload["ruleset"] == "classic"
    assert payload["rewards"]["enabled"] is False


def test_rotation_returns_current_modifier():
    now = ROTATION_ANCHOR_EPOCH_SECONDS + 100
    rot = get_current_extra_arena_mode(now, EXTRA_ARENA_ROTATING_IDS)
    assert rot is not None
    assert rot.mode_id in EXTRA_ARENA_ROTATING_IDS
    assert rot.next_rotation_at > now
    assert rot.seconds_to_rotation > 0
    assert rot.cycle_index == 0


def test_rotation_cycle_index_increases():
    now = ROTATION_ANCHOR_EPOCH_SECONDS + ROTATION_INTERVAL_SECONDS * 3 + 10
    rot = get_current_extra_arena_mode(now, EXTRA_ARENA_ROTATING_IDS)
    assert rot is not None
    assert rot.cycle_index == 3


def test_rotation_disabled_mode_still_in_same_cycle():
    enabled = ["extra_arena:blitzkrieg", "extra_arena:sudden_death", "extra_arena:powermax"]
    now = ROTATION_ANCHOR_EPOCH_SECONDS + 100
    rot = get_current_extra_arena_mode(now, enabled)
    assert rot is not None
    # Cycle 0 is planned as blitzkrieg from the full rotation list; disabling
    # spellstorm must not shift the current cycle to sudden_death/powermax.
    assert rot.mode_id == "extra_arena:blitzkrieg"
    # next_rotation_at should still be on the boundary, not shifted
    assert rot.next_rotation_at == ROTATION_ANCHOR_EPOCH_SECONDS + ROTATION_INTERVAL_SECONDS


def test_rotation_fallback_only_when_planned_mode_is_disabled():
    now = ROTATION_ANCHOR_EPOCH_SECONDS + ROTATION_INTERVAL_SECONDS * 3 + 10
    # Cycle 3 is planned as powermax in the full list.
    planned_enabled = [
        "extra_arena:blitzkrieg",
        "extra_arena:sudden_death",
        "extra_arena:powermax",
    ]
    rot = get_current_extra_arena_mode(now, planned_enabled)
    assert rot is not None
    assert rot.mode_id == "extra_arena:powermax"

    planned_disabled = ["extra_arena:blitzkrieg", "extra_arena:sudden_death"]
    fallback = get_current_extra_arena_mode(now, planned_disabled)
    assert fallback is not None
    assert fallback.mode_id == planned_disabled[3 % len(planned_disabled)]
    assert fallback.next_rotation_at == rot.next_rotation_at


def test_rotation_no_enabled_returns_none():
    rot = get_current_extra_arena_mode(ROTATION_ANCHOR_EPOCH_SECONDS + 100, [])
    assert rot is None


def test_resolve_canonical_extra_arena_generic():
    canonical = resolve_canonical_mode_id("extra_arena", now=ROTATION_ANCHOR_EPOCH_SECONDS + 10, enabled_mode_ids=EXTRA_ARENA_ROTATING_IDS)
    assert canonical.startswith("extra_arena:")


def test_resolve_canonical_extra_arena_no_enabled_stays_unresolved():
    assert resolve_canonical_mode_id(
        "extra_arena",
        now=ROTATION_ANCHOR_EPOCH_SECONDS + 10,
        enabled_mode_ids=[],
    ) == "extra_arena"


def test_resolve_canonical_classic_passthrough():
    assert resolve_canonical_mode_id("classic") == "classic"
    assert resolve_canonical_mode_id("training") == "training"
    assert resolve_canonical_mode_id("friendly") == "friendly"


def test_unknown_and_case_variant_modes_are_rejected_not_classic():
    assert resolve_mode_config("training").mode_id == "training"
    assert resolve_mode_config("TRAINING").available is False
    assert resolve_mode_config("clasisc").available is False
    assert resolve_mode_config("clasisc").mode_id == "clasisc"


def test_legacy_aliases_still_map_to_classic():
    assert resolve_mode_config("").mode_id == "classic"
    assert resolve_mode_config("normal").mode_id == "classic"
    assert resolve_mode_config("ranked").mode_id == "classic"


def test_powermax_sets_card_level_mode_max():
    cfg = resolve_mode_config("extra_arena:powermax")
    assert cfg.classic.card_level_mode == "max"


def test_spellstorm_has_spells_free():
    cfg = resolve_mode_config("extra_arena:spellstorm")
    assert cfg.classic.spells_free is True


def test_blitzkrieg_has_summon_ready():
    cfg = resolve_mode_config("extra_arena:blitzkrieg")
    assert cfg.classic.summon_ready_on_play is True
    assert cfg.classic.turn_duration_seconds == 10


def test_sudden_death_enabled():
    cfg = resolve_mode_config("extra_arena:sudden_death")
    assert cfg.classic.sudden_death_enabled is True


def test_sudden_death_serializes_current_turn_damage_for_active_player_only():
    engine = BattleEngine(game_mode="extra_arena:sudden_death")
    engine._arena = SimpleNamespace(
        state=SimpleNamespace(
            current_turn_owner_id=101,
            sudden_death_turns_by_player={101: 3, 202: 2},
        )
    )

    payload = engine._serialize_sudden_death_state(
        SimpleNamespace(user_id=101),
        SimpleNamespace(user_id=202),
    )

    assert payload["player_turn_damage"] == 3
    assert payload["opponent_turn_damage"] is None
    assert payload["player_next_damage"] == 4
    assert payload["opponent_next_damage"] == 3


def test_extra_arena_widget_payload_uses_minimal_public_contract():
    now = ROTATION_ANCHOR_EPOCH_SECONDS + 123

    payload = build_extra_arena_widget_payload(
        now,
        EXTRA_ARENA_ROTATING_IDS,
        extra_arena_enabled=True,
    )

    assert payload["enabled"] is True
    assert payload["mode_id"] == "extra_arena:blitzkrieg"
    assert payload["label"] == "ExtraArena Blitzkrieg"
    assert payload["seconds_to_rotation"] == ROTATION_INTERVAL_SECONDS - 123
    assert payload["next_rotation_at"] == ROTATION_ANCHOR_EPOCH_SECONDS + ROTATION_INTERVAL_SECONDS
    assert payload["rotation_interval_seconds"] == ROTATION_INTERVAL_SECONDS
    assert "description" not in payload


def test_extra_arena_widget_payload_reports_disabled_state_without_mode_details():
    payload = build_extra_arena_widget_payload(
        ROTATION_ANCHOR_EPOCH_SECONDS + 123,
        EXTRA_ARENA_ROTATING_IDS,
        extra_arena_enabled=False,
    )

    assert payload == {
        "enabled": False,
        "mode_id": None,
        "label": None,
        "seconds_to_rotation": None,
        "next_rotation_at": None,
        "rotation_interval_seconds": ROTATION_INTERVAL_SECONDS,
    }
