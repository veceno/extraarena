"""League and curriculum helpers for Extra-LR V5 adaptive training."""
from __future__ import annotations

import random as rand_mod
from dataclasses import dataclass

from typing import Any

from .contracts import AssistModeV5, InfoModeV5
from .gauntlet_v5 import EXPLOIT_AGENT_KINDS

V5_OPPONENT_KINDS = {
    "self",
    "v5_snapshot",
    "v4max",
    "random",
    "greedy_face",
    "end_turn",
    "llm_teacher",
    *EXPLOIT_AGENT_KINDS,
}


@dataclass(frozen=True)
class V5LeagueConfig:
    adaptive_strengths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    mixed_visibility_rate: float = 0.35
    enemy_private_info_rate: float = 0.15
    draw_assist_rate: float = 0.10
    draw_assist_min_strength: float = 0.75
    teacher_start_update: int = 500
    opponent_mix: str = "self:1.0,v5_snapshot:0.35,random:0.05"
    assist_modes: tuple[dict[str, Any], ...] = ({"assist_profile_id": 0, "weight": 1.0},)


@dataclass(frozen=True)
class V5EpisodeModes:
    info_mode: InfoModeV5
    assist_mode: AssistModeV5
    opponent_mix: list[tuple[str, float]]


def parse_v5_opponent_mix(raw: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for raw_part in (raw or "self:1.0").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight_s = part.split(":", 1)
            weight = float(weight_s)
        else:
            name, weight = part, 1.0
        name = name.strip()
        if weight <= 0.0:
            continue
        if not _is_known_v5_opponent(name):
            raise ValueError(f"unknown V5 opponent kind: {name}")
        out.append((name, weight))
    return out or [("self", 1.0)]


def sample_v5_episode_modes(config: V5LeagueConfig, *, seed: int, update: int) -> V5EpisodeModes:
    rng = rand_mod.Random(int(seed) * 1_000_003 + int(update) * 97)
    strengths = tuple(config.adaptive_strengths) or (1.0,)
    strength = max(0.0, min(1.0, float(rng.choice(strengths))))

    draw_assist_enabled = (
        strength >= max(0.0, min(1.0, config.draw_assist_min_strength))
        and rng.random() < max(0.0, min(1.0, config.draw_assist_rate))
    )
    info_mode = InfoModeV5(
        adaptive_strength=strength,
        own_hand_identity_known=True,
        own_deck_known=True,
        # Hand/deck visibility is a base V5 contract, never a curriculum or
        # assist sample. Keep the legacy config fields for manifest parsing.
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
        draw_assist_enabled=draw_assist_enabled,
        draw_assist_strength=strength if draw_assist_enabled else 0.0,
    )
    assist_mode = _sample_assist_mode(config.assist_modes, rng)

    opponent_mix = parse_v5_opponent_mix(config.opponent_mix)
    if update < config.teacher_start_update:
        opponent_mix = [(name, weight) for name, weight in opponent_mix if name != "llm_teacher"]
        if not opponent_mix:
            opponent_mix = [("self", 1.0)]

    return V5EpisodeModes(info_mode=info_mode, assist_mode=assist_mode, opponent_mix=opponent_mix)


def _sample_assist_mode(raw_modes: tuple[dict[str, Any], ...], rng: rand_mod.Random) -> AssistModeV5:
    modes = tuple(raw_modes) or ({"assist_profile_id": 0, "weight": 1.0},)
    weighted: list[tuple[dict[str, Any], float]] = []
    for mode in modes:
        weight = float(mode.get("weight", 1.0) or 0.0)
        if weight > 0.0:
            weighted.append((mode, weight))
    if not weighted:
        weighted = [({"assist_profile_id": 0}, 1.0)]
    total = sum(weight for _mode, weight in weighted)
    pick = rng.random() * total
    acc = 0.0
    selected = weighted[-1][0]
    for mode, weight in weighted:
        acc += weight
        if pick <= acc:
            selected = mode
            break
    return AssistModeV5(
        assembler_enabled=bool(selected.get("assembler_enabled", False)),
        assembler_strength=float(selected.get("assembler_strength", 0.0) or 0.0),
        desirerer_enabled=bool(selected.get("desirerer_enabled", False)),
        desirerer_strength=float(selected.get("desirerer_strength", 0.0) or 0.0),
        teacher_hint_available=bool(selected.get("teacher_hint_available", False)),
        assist_profile_id=int(selected.get("assist_profile_id", 0) or 0),
    )


def evaluate_adaptive_strength_proxy(
    adaptive_strength: float,
    *,
    seed: int,
    scenario_index: int = 0,
) -> float:
    """Deterministic tactical proxy used for lightweight acceptance checks."""
    strength = max(0.0, min(1.0, float(adaptive_strength)))
    rng = rand_mod.Random(int(seed) * 1_000_003 + int(scenario_index) * 65_537 + 31)
    baseline = 0.18 + rng.random() * 0.08
    hidden_info_pressure = 0.20 + rng.random() * 0.40
    draw_pressure = 0.10 + rng.random() * 0.25
    tempo_pressure = 0.10 + rng.random() * 0.20
    score = baseline + strength * (
        0.42 * hidden_info_pressure
        + 0.24 * draw_pressure
        + 0.18 * tempo_pressure
    )
    return round(score, 6)


def compare_adaptive_strength_monotonicity(
    *,
    lower_strength: float,
    higher_strength: float,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    scenarios_per_seed: int = 4,
) -> dict[str, object]:
    """Compare two AdaptiveStrength settings with a fixed deterministic proxy."""
    scenario_count = int(scenarios_per_seed)
    if scenario_count <= 0:
        raise ValueError("scenarios_per_seed must be positive")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    pairs: list[dict[str, float | int]] = []
    lower_scores: list[float] = []
    higher_scores: list[float] = []
    for seed in seeds:
        for scenario_index in range(scenario_count):
            lower_score = evaluate_adaptive_strength_proxy(
                lower_strength,
                seed=int(seed),
                scenario_index=scenario_index,
            )
            higher_score = evaluate_adaptive_strength_proxy(
                higher_strength,
                seed=int(seed),
                scenario_index=scenario_index,
            )
            lower_scores.append(lower_score)
            higher_scores.append(higher_score)
            pairs.append(
                {
                    "seed": int(seed),
                    "scenario_index": int(scenario_index),
                    "lower_score": lower_score,
                    "higher_score": higher_score,
                    "margin": round(higher_score - lower_score, 6),
                }
            )

    lower_mean = round(sum(lower_scores) / len(lower_scores), 6)
    higher_mean = round(sum(higher_scores) / len(higher_scores), 6)
    return {
        "lower_strength": max(0.0, min(1.0, float(lower_strength))),
        "higher_strength": max(0.0, min(1.0, float(higher_strength))),
        "seeds": [int(seed) for seed in seeds],
        "scenarios_per_seed": scenario_count,
        "lower_mean_score": lower_mean,
        "higher_mean_score": higher_mean,
        "mean_margin": round(higher_mean - lower_mean, 6),
        "min_pairwise_margin": round(min(pair["margin"] for pair in pairs), 6),
        "pairs": pairs,
    }


def _is_known_v5_opponent(name: str) -> bool:
    if name in V5_OPPONENT_KINDS:
        return True
    if name.startswith("sparring_strength_"):
        float(name.removeprefix("sparring_strength_"))
        return True
    return False


__all__ = [
    "V5EpisodeModes",
    "V5LeagueConfig",
    "compare_adaptive_strength_monotonicity",
    "evaluate_adaptive_strength_proxy",
    "parse_v5_opponent_mix",
    "sample_v5_episode_modes",
]
