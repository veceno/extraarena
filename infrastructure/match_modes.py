from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Literal


Ruleset = Literal["classic", "draft"]
CardLevelMode = Literal["normal", "disabled", "max"]


@dataclass(frozen=True)
class RewardParams:
    """Economy switches shared by all battle rulesets."""

    enabled: bool = True
    trophies: bool = True
    coins: bool = True
    stars: bool = True
    win_counter: bool = True


@dataclass(frozen=True)
class ClassicParams:
    """Parameters for the classic ruleset and its modifier presets."""

    turn_duration_seconds: int = 25
    mana_per_turn: int = 1
    hero_health_multiplier: float = 1.0
    card_level_mode: CardLevelMode = "normal"
    bot_turn_delay_range: tuple[float, float] = (4.0, 6.0)
    bot_hard_turn_delay_range: tuple[float, float] = (1.5, 2.5)
    bot_action_gap_range: tuple[float, float] = (0.4, 0.8)
    bot_emergency_threshold_seconds: float = 5.0
    # ExtraArena classic modifiers
    spells_free: bool = False
    summon_ready_on_play: bool = False
    sudden_death_enabled: bool = False
    sudden_death_damage_start: int = 1
    sudden_death_damage_step: int = 1
    overdraw_to_discard: bool = False
    bots_allowed: bool = True


@dataclass(frozen=True)
class ModeConfig:
    """Resolved battle mode configuration.

    `mode_id` is the matchmaking bucket, while `ruleset` chooses the battle
    implementation. For example, blitz is classic with different params.
    """

    mode_id: str
    ruleset: Ruleset
    label: str
    available: bool = True
    classic: ClassicParams = field(default_factory=ClassicParams)
    rewards: RewardParams = field(default_factory=RewardParams)


NO_REWARDS = RewardParams(
    enabled=False,
    trophies=False,
    coins=False,
    stars=False,
    win_counter=False,
)


MODE_CONFIGS: dict[str, ModeConfig] = {
    "classic": ModeConfig(
        mode_id="classic",
        ruleset="classic",
        label="Classic",
    ),
    # Legacy blitz — preserved for history and backward compatibility
    "extra_arena:blitz": ModeConfig(
        mode_id="extra_arena:blitz",
        ruleset="classic",
        label="ExtraArena Blitz",
        classic=ClassicParams(
            turn_duration_seconds=5,
            mana_per_turn=2,
            hero_health_multiplier=0.5,
            card_level_mode="normal",
            bot_turn_delay_range=(0.3, 0.8),
            bot_hard_turn_delay_range=(0.3, 0.8),
            bot_action_gap_range=(0.05, 0.15),
            bot_emergency_threshold_seconds=2.0,
        ),
    ),
    "training": ModeConfig(
        mode_id="training",
        ruleset="classic",
        label="Training",
        rewards=NO_REWARDS,
    ),
    "friendly": ModeConfig(
        mode_id="friendly",
        ruleset="classic",
        label="Friendly",
        rewards=NO_REWARDS,
    ),
    "extra_arena:draft": ModeConfig(
        mode_id="extra_arena:draft",
        ruleset="draft",
        label="ExtraArena Draft",
        available=False,
        rewards=NO_REWARDS,
    ),
    # New rotating ExtraArena modifiers
    "extra_arena:powermax": ModeConfig(
        mode_id="extra_arena:powermax",
        ruleset="classic",
        label="ExtraArena PowerMax",
        classic=ClassicParams(
            card_level_mode="max",
            bots_allowed=True,
        ),
    ),
    "extra_arena:blitzkrieg": ModeConfig(
        mode_id="extra_arena:blitzkrieg",
        ruleset="classic",
        label="ExtraArena Blitzkrieg",
        classic=ClassicParams(
            turn_duration_seconds=10,
            summon_ready_on_play=True,
            hero_health_multiplier=1.0,
            bots_allowed=True,
        ),
    ),
    "extra_arena:spellstorm": ModeConfig(
        mode_id="extra_arena:spellstorm",
        ruleset="classic",
        label="ExtraArena Spellstorm",
        classic=ClassicParams(
            spells_free=True,
            bots_allowed=True,
        ),
    ),
    "extra_arena:sudden_death": ModeConfig(
        mode_id="extra_arena:sudden_death",
        ruleset="classic",
        label="ExtraArena Sudden Death",
        classic=ClassicParams(
            sudden_death_enabled=True,
            bots_allowed=True,
        ),
    ),
}


MODE_ALIASES: dict[str, str] = {
    "": "classic",
    "ranked": "classic",
    "normal": "classic",
    "blitz": "extra_arena:blitz",
    "draft": "extra_arena:draft",
    # extraarena variants
    "extraarena": "extra_arena",
    "extraarena:blitz": "extra_arena:blitz",
    "extraarena:draft": "extra_arena:draft",
    "extraarena:powermax": "extra_arena:powermax",
    "extraarena:blitzkrieg": "extra_arena:blitzkrieg",
    "extraarena:spellstorm": "extra_arena:spellstorm",
    "extraarena:sudden_death": "extra_arena:sudden_death",
    "extra_arena:blitzkrieg": "extra_arena:blitzkrieg",
    "extra_arena:spellstorm": "extra_arena:spellstorm",
    "extra_arena:sudden_death": "extra_arena:sudden_death",
    "extra_arena:powermax": "extra_arena:powermax",
}


# IDs eligible for ExtraArena rotation (all current classic-ruleset modifiers)
EXTRA_ARENA_ROTATING_IDS: list[str] = [
    "extra_arena:blitzkrieg",
    "extra_arena:spellstorm",
    "extra_arena:sudden_death",
    "extra_arena:powermax",
]

# Rotation settings
ROTATION_ENABLED: bool = True
ROTATION_INTERVAL_SECONDS: int = 7200  # 2 hours
ROTATION_ANCHOR_EPOCH_SECONDS: int = 1767225600  # 2026-01-01T00:00:00Z


@dataclass(frozen=True)
class RotationState:
    """Result of ExtraArena mode rotation resolution."""

    mode_id: str
    label: str
    description: str
    next_rotation_at: int
    seconds_to_rotation: int
    cycle_index: int


def _is_extra_arena_generic(mode_id: str) -> bool:
    """Return True for any unresolved generic ExtraArena input."""
    raw = mode_id.strip().lower()
    return raw in ("extra_arena", "extraarena")


def normalize_mode_id(mode_id: str | None) -> str:
    raw = str(mode_id or "classic").strip()
    return MODE_ALIASES.get(raw, raw)


def resolve_mode_config(mode_id: str | None) -> ModeConfig:
    normalized = normalize_mode_id(mode_id)
    config = MODE_CONFIGS.get(normalized)
    if config is not None:
        return config
    return ModeConfig(
        mode_id=normalized,
        ruleset="classic",
        label=f"Unknown mode: {normalized}",
        available=False,
        rewards=NO_REWARDS,
    )


def is_mode_available(mode_id: str | None) -> bool:
    return resolve_mode_config(mode_id).available


def mode_unavailable_payload(mode_id: str | None) -> dict[str, str]:
    config = resolve_mode_config(mode_id)
    return {
        "error": "mode_unavailable",
        "mode": config.mode_id,
        "message": f"{config.label} is not available yet",
    }


def should_apply_rewards(mode_id: str | None) -> bool:
    return resolve_mode_config(mode_id).rewards.enabled


def is_train_v2_bot_safe_mode(mode_id: str | None) -> bool:
    config = resolve_mode_config(mode_id)
    return config.mode_id in {"classic", "training"}


def serialize_mode_config(config: ModeConfig) -> dict[str, object]:
    return asdict(config)


def _enabled_in_list(mode_id: str, enabled_set: set[str]) -> bool:
    return mode_id in enabled_set


def get_current_extra_arena_mode(
    now: float,
    enabled_mode_ids: list[str],
) -> RotationState | None:
    """Deterministic rotation of ExtraArena modifiers.

    Args:
        now: current UNIX timestamp.
        enabled_mode_ids: list of enabled modifier IDs from DB/static config.

    Returns:
        RotationState for the active modifier, or None if no modifiers enabled.
    """
    enabled = [m for m in enabled_mode_ids if m in EXTRA_ARENA_ROTATING_IDS]
    if not enabled:
        return None

    cycle_index = math.floor((now - ROTATION_ANCHOR_EPOCH_SECONDS) / ROTATION_INTERVAL_SECONDS)
    if cycle_index < 0:
        cycle_index = 0

    # First choose the planned mode from the full rotation list. Disabling a
    # non-current mode must not shift the active mode for this cycle.
    planned_id = EXTRA_ARENA_ROTATING_IDS[cycle_index % len(EXTRA_ARENA_ROTATING_IDS)]
    if planned_id in enabled:
        chosen_id = planned_id
    else:
        # If the planned mode is disabled, choose a stable replacement within
        # the same cycle without changing the timer boundary.
        chosen_id = enabled[cycle_index % len(enabled)]

    cfg = MODE_CONFIGS.get(chosen_id)
    if cfg is None:
        return None

    next_rotation_at = ROTATION_ANCHOR_EPOCH_SECONDS + (cycle_index + 1) * ROTATION_INTERVAL_SECONDS
    seconds_to_rotation = max(0, int(next_rotation_at - now))

    descriptions = {
        "extra_arena:powermax": "Все карты на максимальном уровне — только умение решает.",
        "extra_arena:blitzkrieg": "Ускоренные ходы и существа готовы к бою сразу.",
        "extra_arena:spellstorm": "Заклинания бесплатные — впереди великий хаос.",
        "extra_arena:sudden_death": "Каждый ход герой теряет здоровье — скорость решает всё.",
    }

    return RotationState(
        mode_id=cfg.mode_id,
        label=cfg.label,
        description=descriptions.get(cfg.mode_id, "Особый режим с уникальными правилами."),
        next_rotation_at=next_rotation_at,
        seconds_to_rotation=seconds_to_rotation,
        cycle_index=cycle_index,
    )


def build_extra_arena_widget_payload(
    now: float,
    enabled_mode_ids: list[str],
    *,
    extra_arena_enabled: bool = True,
) -> dict[str, object]:
    """Return the compact public ExtraArena rotation payload used by widgets."""
    if not extra_arena_enabled:
        return {
            "enabled": False,
            "mode_id": None,
            "label": None,
            "seconds_to_rotation": None,
            "next_rotation_at": None,
            "rotation_interval_seconds": ROTATION_INTERVAL_SECONDS,
        }

    rot = get_current_extra_arena_mode(now, enabled_mode_ids)
    if rot is None:
        return {
            "enabled": False,
            "mode_id": None,
            "label": None,
            "seconds_to_rotation": None,
            "next_rotation_at": None,
            "rotation_interval_seconds": ROTATION_INTERVAL_SECONDS,
        }

    return {
        "enabled": True,
        "mode_id": rot.mode_id,
        "label": rot.label,
        "seconds_to_rotation": rot.seconds_to_rotation,
        "next_rotation_at": rot.next_rotation_at,
        "rotation_interval_seconds": ROTATION_INTERVAL_SECONDS,
    }


def resolve_canonical_mode_id(
    mode_id: str | None,
    *,
    now: float | None = None,
    enabled_mode_ids: list[str] | None = None,
) -> str:
    """Resolve any user-facing mode string into the canonical mode_id.

    Generic ``extra_arena`` / ``extraarena`` inputs resolve to the currently
    active rotating modifier.  ``classic``, ``training``, ``friendly`` pass
    through without rotation.
    """
    raw = str(mode_id or "classic").strip()
    normalized = normalize_mode_id(raw)
    if _is_extra_arena_generic(normalized):
        effective_now = now if now is not None else time.time()
        effective_enabled = enabled_mode_ids if enabled_mode_ids is not None else EXTRA_ARENA_ROTATING_IDS
        rot = get_current_extra_arena_mode(effective_now, effective_enabled)
        if rot is None:
            return "extra_arena"
        return rot.mode_id
    return normalized


def get_extra_arena_mode_list() -> list[dict[str, object]]:
    """Return list of modifier metadata for the UI."""
    return [
        {
            "mode_id": m,
            "label": MODE_CONFIGS[m].label,
            "description": {
                "extra_arena:powermax": "Все карты на максимальном уровне.",
                "extra_arena:blitzkrieg": "Существа готовы после выставления. Ход 10 сек.",
                "extra_arena:spellstorm": "Заклинания бесплатные.",
                "extra_arena:sudden_death": "Герой теряет здоровье каждый ход.",
            }.get(m, ""),
        }
        for m in EXTRA_ARENA_ROTATING_IDS
        if m in MODE_CONFIGS
    ]
