#!/usr/bin/env python3
"""Paired V5 H2H/gauntlet benchmark with optional ExtraLR V1 auxiliaries.

The local ``ai/model_benchmark`` package is intentionally gitignored in the
main checkout.  This runner keeps its classic ONNX adapters and reporting, but
uses the audited worktree's V5 environment and learned auxiliary adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


WORKTREE = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path("/Users/laveqox/Documents/ExtraArenaRaS")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "TrainV3.5" / "python"))

import ai  # noqa: E402

if str(MAIN_ROOT / "ai") not in ai.__path__:
    ai.__path__.append(str(MAIN_ROOT / "ai"))

import mlx.core as mx  # noqa: E402

from ai.model_benchmark.config import (  # noqa: E402
    BenchmarkConfig,
    ModelSpec,
    default_model_specs,
    filter_available_specs,
)
from ai.model_benchmark.policies import create_policy  # noqa: E402
from ai.model_benchmark.reporting import write_report_artifacts  # noqa: E402
from ai.model_benchmark.scenarios import ScenarioSpec, build_level_map  # noqa: E402
from ai.train_v2.classic_actions_v1 import decode_action  # noqa: E402
from ai.train_v2.classic_rl_env import _load_cards_db  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint  # noqa: E402
from core.actions import ManaDrawAction, PlayCardAction  # noqa: E402
from rlhf_env.components.deck_builder import (  # noqa: E402
    CardCatalog,
    build_random_arena_deck,
)
from train_v3.aux_inference import (  # noqa: E402
    AssemblerV1,
    CARD_CATALOG,
    CardOptimumV1,
    ForcedDrawRandom,
    HERO_CARD_IDS,
)
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.mana_draw_head_v5 import (  # noqa: E402
    mana_draw_legal_mask,
    select_includes_mana_draw,
)
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_BASE_CHECKPOINT = (
    WORKTREE
    / "TrainV3.5/runs/blockB_from_phaseA_p2accepted100_parallel_20260714_210400"
    / "checkpoints/extra_lr_v5_blockB_league_update_29250.npz"
)
DEFAULT_AUX_DIR = (
    WORKTREE
    / "TrainV3.5/runs/phase_c_aux_v1_u29250_h299_projectionfix_20260727/models"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class V5NpzPolicy:
    def __init__(self, checkpoint: Path, *, name: str | None = None):
        self.checkpoint = checkpoint.resolve()
        self.name = name or f"extra-lr-v5-{checkpoint.stem}"
        self.model = create_v5_policy(
            policy_kind="v5_split_encoder",
            hidden_dim=256,
            action_hidden_dim=128,
        )
        self.loaded = load_checkpoint(str(self.checkpoint), self.model)

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select_action(self, env: TrainV3ClassicEnv, player_id: int):
        obs = env.observe(player_id).astype(np.float32)
        mask = env.action_mask(player_id).astype(np.float32)
        features = env.action_features(player_id, include_preview=False).astype(np.float32)
        md_legal = mana_draw_legal_mask(env.env._env.state, player_id)
        output = self.model(
            mx.array(obs[None, :]),
            mx.array(features[None, :, :]),
            mana_draw_legal=mx.array([md_legal]),
        )
        logits = output[0] if isinstance(output, tuple) else output
        mana_draw_logit = output[2] if isinstance(output, tuple) and len(output) >= 3 else None
        if mana_draw_logit is None:
            mx.eval(logits)
        else:
            mx.eval(logits, mana_draw_logit)
        logits_np = np.asarray(logits, dtype=np.float32)[0]
        masked = np.where(mask.astype(bool), logits_np, -1.0e9)
        action_id = int(np.argmax(masked))
        if mana_draw_logit is not None and select_includes_mana_draw(
            float(np.asarray(mana_draw_logit, dtype=np.float32).reshape(-1)[0]),
            float(masked[action_id]),
            bool(md_legal),
        ):
            return ManaDrawAction()
        return action_id


@dataclass(frozen=True)
class AuxAblationMode:
    name: str
    assembler_enabled: bool
    cardoptimum_enabled: bool


AUX_ABLATION_MODES = {
    "none": AuxAblationMode("none", False, False),
    "assembler_only": AuxAblationMode("assembler_only", True, False),
    "cardoptimum_only": AuxAblationMode("cardoptimum_only", False, True),
    "both": AuxAblationMode("both", True, True),
}


@dataclass
class AuxBundle:
    assembler: AssemblerV1
    cardoptimum: CardOptimumV1

    @classmethod
    def load(cls, directory: Path) -> "AuxBundle":
        directory = directory.resolve()
        return cls(
            assembler=AssemblerV1(directory / "extra_lr_assembler_v1.npz"),
            cardoptimum=CardOptimumV1(directory / "extra_lr_cardoptimum_v1.npz"),
        )


def _action_kind(env: TrainV3ClassicEnv, player_id: int, action: Any) -> str:
    if isinstance(action, ManaDrawAction):
        return "mana_draw"
    decoded = decode_action(env.env._env.state, player_id, int(action))
    if decoded is None:
        return "unknown"
    kind = str(decoded.to_dict().get("type") or "unknown")
    return kind if kind in {"end_turn", "play_card", "attack"} else "unknown"


def _action_to_json(state: Any, player_id: int, action: Any) -> dict[str, Any] | None:
    if hasattr(action, "to_dict"):
        data = action.to_dict()
        return data if isinstance(data, dict) else {"repr": repr(data)}
    decoded = decode_action(state, int(player_id), int(action))
    if decoded is None:
        return None
    data = decoded.to_dict() if hasattr(decoded, "to_dict") else None
    return data if isinstance(data, dict) else {"repr": repr(decoded)}


def _card_snapshot(card: Any) -> dict[str, Any]:
    return {
        "card_id": int(getattr(card, "card_id", 0) or 0),
        "name": str(getattr(card, "name", "") or ""),
        "attack": int(getattr(card, "attack", 0) or 0),
        "hp": int(getattr(card, "hp", 0) or 0),
        "max_hp": int(getattr(card, "max_hp", 0) or 0),
        "mana_cost": int(getattr(card, "mana_cost", 0) or 0),
        "is_ready": bool(getattr(card, "is_ready", False)),
    }


def _player_snapshot(player: Any) -> dict[str, Any]:
    return {
        "hero_hp": int(getattr(getattr(player, "hero", None), "hp", 0) or 0),
        "mana": int(getattr(player, "mana", 0) or 0),
        "max_mana": int(getattr(player, "max_mana", 0) or 0),
        "mana_draw_count_this_turn": int(
            getattr(player, "mana_draw_count_this_turn", 0) or 0
        ),
        "hand": [_card_snapshot(card) for card in list(getattr(player, "hand", []) or [])],
        "deck_count": len(list(getattr(player, "deck", []) or [])),
        "board": [_card_snapshot(card) for card in list(getattr(player, "board", []) or [])],
        "graveyard_count": len(list(getattr(player, "graveyard", []) or [])),
    }


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "turn_number": int(getattr(state, "turn_number", 0) or 0),
        "current_player_id": int(getattr(state, "current_turn_owner_id", 0) or 0),
        "p1": _player_snapshot(getattr(state, "p1", None)),
        "p2": _player_snapshot(getattr(state, "p2", None)),
    }


def _default_deck_for_seed(seed: int) -> list[int]:
    probe = TrainV3ClassicEnv(TrainV3EnvConfig(seed=seed))
    episode_rng = random.Random(seed)
    deck_rng = random.Random(episode_rng.randint(0, 2**31 - 1))
    return [int(card_id) for card_id in probe.env._generate_default_deck(deck_rng)]


def _assembler_search_space(seed: int, *, candidates: int = 256) -> tuple[list[int], list[list[int]]]:
    rng = random.Random(seed ^ 0x5A55E4B1)
    heroes = sorted(HERO_CARD_IDS)
    warriors = sorted(
        card_id
        for card_id, card in CARD_CATALOG.items()
        if card.get("card_type") == "warrior"
    )
    potions = sorted(
        card_id
        for card_id, card in CARD_CATALOG.items()
        if card.get("card_type") == "potion"
    )
    pool_heroes = rng.sample(heroes, min(4, len(heroes)))
    pool_warriors = rng.sample(warriors, min(18, len(warriors)))
    pool_potions = rng.sample(potions, min(6, len(potions)))
    pool_ids = [*pool_heroes, *pool_warriors, *pool_potions]
    pool = CardCatalog(
        cards={card_id: CARD_CATALOG[card_id] for card_id in pool_ids},
        heroes=pool_heroes,
        warriors=pool_warriors,
        potions=pool_potions,
    )
    allowed_pool = sorted(pool_ids)
    decks: set[tuple[int, ...]] = set()
    target = max(1, int(candidates))
    while len(decks) < target:
        deck = tuple(
            build_random_arena_deck(
                pool,
                rng=random.Random(rng.getrandbits(64)),
            )
        )
        decks.add(tuple(int(card_id) for card_id in deck))
    return allowed_pool, [list(deck) for deck in sorted(decks)]


def _player(state: Any, player_id: int) -> Any:
    return state.p1 if int(state.p1.user_id) == int(player_id) else state.p2


def _action_triggers_battlecry_draw(
    env: TrainV3ClassicEnv,
    player_id: int,
    action: Any,
) -> bool:
    if isinstance(action, ManaDrawAction):
        return False
    decoded = decode_action(env.env._env.state, player_id, int(action))
    if not isinstance(decoded, PlayCardAction):
        return False
    player = _player(env.env._env.state, player_id)
    if not 0 <= int(decoded.hand_index) < len(player.hand):
        return False
    return "battlecry_draw_card" in list(player.hand[int(decoded.hand_index)].mechanics)


def _arm_cardoptimum_draw(
    *,
    env: TrainV3ClassicEnv,
    aux: AuxBundle,
    rng: ForcedDrawRandom,
    candidate_player_id: int,
    actor_id: int,
    action: Any,
    action_kind: str,
) -> dict[str, Any] | None:
    draw_player_id: int | None = None
    reason = ""
    if int(actor_id) == int(candidate_player_id) and isinstance(action, ManaDrawAction):
        draw_player_id = candidate_player_id
        reason = "mana_draw"
    elif (
        int(actor_id) != int(candidate_player_id)
        and action_kind == "end_turn"
    ):
        draw_player_id = candidate_player_id
        reason = "turn_start_draw"
    elif (
        int(actor_id) == int(candidate_player_id)
        and _action_triggers_battlecry_draw(env, actor_id, action)
    ):
        draw_player_id = candidate_player_id
        reason = "battlecry_draw"
    if draw_player_id is None:
        return None
    choice = aux.cardoptimum.choose(env.env._env.state, draw_player_id)
    if not choice:
        return None
    armed = rng.arm(
        _player(env.env._env.state, draw_player_id),
        int(choice["selected_card_id"]),
    )
    return {
        "reason": reason,
        "armed": bool(armed),
        "selected_card_id": int(choice["selected_card_id"]),
        "top_score": float(choice["ranked_options"][0]["score"]),
    }


def _explicit_info_mode() -> InfoModeV5:
    return InfoModeV5(
        adaptive_strength=1.0,
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
        draw_assist_enabled=False,
        draw_assist_strength=0.0,
    )


def _run_game(
    spec: ScenarioSpec,
    *,
    candidate: V5NpzPolicy,
    v5_policies: dict[str, V5NpzPolicy],
    opponent_policies: dict[str, Any],
    model_map: dict[str, ModelSpec],
    config: BenchmarkConfig,
    aux: AuxBundle,
    ablation: AuxAblationMode,
    log_events: bool,
) -> dict[str, Any]:
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=spec.seed,
            max_turns=config.max_turns,
            verify_mask=False,
            placement_mode="append_only",
            # The model sees the exact same observation contract in every arm.
            # This isolates the two external mechanisms from assist-profile bits.
            info_mode=_explicit_info_mode(),
            assist_mode=AssistModeV5(),
            history_limit=20,
        )
    )
    candidate_player_id = 1 if spec.p1_name == candidate.name else 2
    opponent_player_id = 2 if candidate_player_id == 1 else 1
    opponent_deck = _default_deck_for_seed(spec.seed)
    candidate_deck = list(opponent_deck)
    assembler_info: dict[str, Any] | None = None
    if ablation.assembler_enabled:
        allowed_pool, candidates = _assembler_search_space(spec.seed)
        selection = aux.assembler.select(
            candidates=candidates,
            opponent_deck_ids=opponent_deck,
            allowed_pool_ids=allowed_pool,
            candidate_levels=(
                spec.p1_levels if candidate_player_id == 1 else spec.p2_levels
            ),
            opponent_levels=(
                spec.p2_levels if candidate_player_id == 1 else spec.p1_levels
            ),
        )
        candidate_deck = list(selection.deck_ids)
        assembler_info = asdict(selection)
    p1_deck = candidate_deck if candidate_player_id == 1 else opponent_deck
    p2_deck = candidate_deck if candidate_player_id == 2 else opponent_deck
    env.reset(
        p1_deck_ids=p1_deck,
        p2_deck_ids=p2_deck,
        p1_levels=spec.p1_levels,
        p2_levels=spec.p2_levels,
        starting_player_id=spec.starting_player_id,
        seed=spec.seed,
    )

    opponent_name = spec.opponent_model
    if opponent_name not in v5_policies:
        if opponent_name not in opponent_policies:
            opponent_policies[opponent_name] = create_policy(
                model_map[opponent_name].policy_spec()
            )
        opponent = opponent_policies[opponent_name]
    else:
        opponent = v5_policies[opponent_name]
    candidate.reset(spec.seed * 2 + 1)
    opponent.reset(spec.seed * 2 + 2)

    # Install the transparent wrapper in every arm. CardOptimum activation is
    # therefore the only paired difference in how a draw is selected.
    forced_rng = ForcedDrawRandom(env.env._env._rng)
    env.env._env._rng = forced_rng
    action_counts = {
        spec.p1_name: {
            "end_turn": 0,
            "play_card": 0,
            "attack": 0,
            "mana_draw": 0,
            "unknown": 0,
        },
        spec.p2_name: {
            "end_turn": 0,
            "play_card": 0,
            "attack": 0,
            "mana_draw": 0,
            "unknown": 0,
        },
    }
    invalid = {spec.p1_name: 0, spec.p2_name: 0}
    aux_stats = {
        "cardoptimum_calls": 0,
        "forced_draws": 0,
    }
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    steps = 0
    error = None
    events: list[dict[str, Any]] = []
    try:
        while not terminated and not truncated and steps < config.max_steps:
            player_id = env.current_player_id()
            actor_name = spec.p1_name if player_id == 1 else spec.p2_name
            if actor_name in v5_policies:
                action = v5_policies[actor_name].select_action(env, player_id)
            else:
                action = opponent.select_action(env.env, player_id)
            action_kind = _action_kind(env, player_id, action)
            action_counts[actor_name][action_kind] += 1
            if log_events:
                state_before = _state_snapshot(env.env._env.state)
                action_json = _action_to_json(env.env._env.state, player_id, action)
            cardoptimum = None
            armed = False
            if ablation.cardoptimum_enabled:
                cardoptimum = _arm_cardoptimum_draw(
                    env=env,
                    aux=aux,
                    rng=forced_rng,
                    candidate_player_id=candidate_player_id,
                    actor_id=player_id,
                    action=action,
                    action_kind=action_kind,
                )
                if cardoptimum is not None:
                    aux_stats["cardoptimum_calls"] += 1
                    armed = bool(cardoptimum["armed"])
            if isinstance(action, ManaDrawAction):
                _obs, reward, terminated, truncated, last_info = env.step_core_action(action)
            else:
                _obs, reward, terminated, truncated, last_info = env.step(int(action))
            if armed and forced_rng._forced_random is None:
                aux_stats["forced_draws"] += 1
            forced_rng.clear()
            if last_info.get("invalid_action"):
                invalid[actor_name] += 1
            steps += 1
            if log_events:
                events.append(
                    {
                        "step": steps,
                        "actor_name": actor_name,
                        "player_id": int(player_id),
                        "action_kind": action_kind,
                        "action": action_json,
                        "reward": float(reward),
                        "invalid_action": bool(last_info.get("invalid_action")),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "cardoptimum": cardoptimum,
                        "state_before": state_before,
                        "state_after": _state_snapshot(env.env._env.state),
                    }
                )
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"

    state = env.env._env.state
    winner_id = env.env.winner_id()
    winner_name = (
        spec.p1_name
        if winner_id == 1
        else spec.p2_name
        if winner_id == 2
        else None
    )
    underlying_status = getattr(state.status, "value", str(state.status))
    timed_out = not terminated and not truncated and steps >= config.max_steps
    result = {
        **asdict(spec),
        "benchmark_mode": ablation.name,
        "assembler_enabled": ablation.assembler_enabled,
        "cardoptimum_enabled": ablation.cardoptimum_enabled,
        "candidate_player_id": candidate_player_id,
        "opponent_player_id": opponent_player_id,
        "winner_id": winner_id,
        "winner_name": winner_name,
        "draw": winner_id is None and underlying_status == "draw",
        "timed_out": timed_out,
        "truncated": bool(truncated),
        "turns": int(getattr(state, "turn_number", 0)),
        "steps": steps,
        "p1_hp": int(state.p1.hero.hp),
        "p2_hp": int(state.p2.hero.hp),
        "invalid_actions": invalid,
        "action_counts": action_counts,
        "status": "max_steps" if timed_out else underlying_status,
        "underlying_status": underlying_status,
        "error": error,
        "information_contract": {
            "history_events": 20,
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": True,
            "enemy_deck_known": True,
            "enemy_deck_order_known": True,
            "draw_assist_enabled": False,
            "draw_assist_strength": 0.0,
            "assembler_assist_bit": False,
            "desirerer_assist_bit": False,
            "assist_profile_id": 0,
        },
        "candidate_deck_ids": candidate_deck,
        "opponent_deck_ids": opponent_deck,
        "assembler": assembler_info,
        "timestamp_duo": None,
        "aux_stats": aux_stats,
    }
    if log_events:
        result["events"] = events
        result["final_state"] = _state_snapshot(state)
    return result


def _paired_even_scenarios(
    *,
    candidate: ModelSpec,
    opponents: list[ModelSpec],
    config: BenchmarkConfig,
    card_ids: tuple[int, ...],
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    game_index = 0
    levels = build_level_map(
        card_ids,
        base_card_level=config.base_card_level,
        delta=0,
        min_level=config.min_card_level,
        max_level=config.max_card_level,
    )
    for opponent_index, opponent in enumerate(opponents):
        for seed_index in range(config.games_per_scenario):
            seed = int(config.base_seed) + opponent_index * 100_003 + seed_index
            for focal_as_p1 in (True, False):
                p1 = candidate if focal_as_p1 else opponent
                p2 = opponent if focal_as_p1 else candidate
                family = "focal_as_p1_even" if focal_as_p1 else "focal_as_p2_even"
                for starting_player_id in (1, 2):
                    scenarios.append(
                        ScenarioSpec(
                            scenario_id=(
                                f"{candidate.name}__vs__{opponent.name}__paired_even"
                                f"__seed{seed}__p{'1' if focal_as_p1 else '2'}"
                                f"__start{starting_player_id}"
                            ),
                            family=family,
                            focal_model=candidate.name,
                            opponent_model=opponent.name,
                            p1_name=p1.name,
                            p2_name=p2.name,
                            p1_level_delta=0,
                            p2_level_delta=0,
                            p1_effective_level=config.base_card_level,
                            p2_effective_level=config.base_card_level,
                            p1_levels=dict(levels),
                            p2_levels=dict(levels),
                            seed=seed,
                            starting_player_id=starting_player_id,
                            game_index=game_index,
                            ranked_game=True,
                        )
                    )
                    game_index += 1
    return scenarios


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _wilson_interval(wins: int, draws: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    score = wins + 0.5 * draws
    p = score / games
    z = 1.959963984540054
    denominator = 1.0 + z * z / games
    centre = p + z * z / (2.0 * games)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
    return (centre - spread) / denominator, (centre + spread) / denominator


def _v5_h2h(results: list[dict[str, Any]], candidate_name: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        buckets[row["opponent_model"]].append(row)
    output = []
    for opponent, rows in sorted(buckets.items()):
        wins = sum(row["winner_name"] == candidate_name for row in rows)
        draws = sum(bool(row["draw"]) for row in rows)
        losses = len(rows) - wins - draws
        first = [
            row
            for row in rows
            if row["candidate_player_id"] == row["starting_player_id"]
        ]
        second = [row for row in rows if row not in first]

        def score(subset: list[dict[str, Any]]) -> float:
            if not subset:
                return 0.0
            return (
                sum(row["winner_name"] == candidate_name for row in subset)
                + 0.5 * sum(bool(row["draw"]) for row in subset)
            ) / len(subset)

        ci_low, ci_high = _wilson_interval(wins, draws, len(rows))
        output.append(
            {
                "opponent": opponent,
                "scenario": "paired_even",
                "games": len(rows),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_rate": score(rows),
                "score_ci95": [ci_low, ci_high],
                "first_score_rate": score(first),
                "second_score_rate": score(second),
                "seat_p1_score_rate": score(
                    [row for row in rows if row["candidate_player_id"] == 1]
                ),
                "seat_p2_score_rate": score(
                    [row for row in rows if row["candidate_player_id"] == 2]
                ),
                "invalid_actions": sum(
                    row["invalid_actions"].get(candidate_name, 0) for row in rows
                ),
                "mana_draw_count": sum(
                    row["action_counts"][candidate_name].get("mana_draw", 0)
                    for row in rows
                ),
                "cardoptimum_forced_draws": sum(
                    int(row["aux_stats"].get("forced_draws", 0)) for row in rows
                ),
            }
        )
    return output


def _run_mode(
    *,
    mode: str,
    output_dir: Path,
    candidate: V5NpzPolicy,
    v5_policies: dict[str, V5NpzPolicy],
    opponents: list[ModelSpec],
    model_map: dict[str, ModelSpec],
    scenarios: list[ScenarioSpec],
    config: BenchmarkConfig,
    aux: AuxBundle,
    ablation: AuxAblationMode,
    log_events: bool,
) -> dict[str, Any]:
    if mode != ablation.name:
        raise ValueError(f"mode mismatch: {mode!r} != {ablation.name!r}")
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    opponent_policies: dict[str, Any] = {}
    results = []
    print(f"{mode}: {len(scenarios)} paired battles", flush=True)
    for index, scenario in enumerate(scenarios, start=1):
        if (
            index == 1
            or index == len(scenarios)
            or index % max(1, len(scenarios) // 40) == 0
        ):
            print(
                f"{mode} progress: {index}/{len(scenarios)} "
                f"{scenario.opponent_model}",
                flush=True,
            )
        results.append(
            _run_game(
                scenario,
                candidate=candidate,
                v5_policies=v5_policies,
                opponent_policies=opponent_policies,
                model_map=model_map,
                config=config,
                aux=aux,
                ablation=ablation,
                log_events=log_events,
            )
        )
    payload = {
        "config": {
            **_jsonable(asdict(config)),
            "paired_seed_seat_and_initiative": True,
            "scenario": "even_level_only",
            "benchmark_mode": mode,
            "assembler_enabled": ablation.assembler_enabled,
            "cardoptimum_enabled": ablation.cardoptimum_enabled,
            "metronome_enabled": False,
            "timestamp_enabled": False,
            "candidate_checkpoint": str(candidate.checkpoint),
            "candidate_checkpoint_sha256": _sha256(candidate.checkpoint),
            "auxiliary_artifacts": {
                "assembler": {
                    "path": str(aux.assembler.artifact),
                    "sha256": _sha256(Path(aux.assembler.artifact)),
                },
                "cardoptimum": {
                    "path": str(aux.cardoptimum.artifact),
                    "sha256": _sha256(Path(aux.cardoptimum.artifact)),
                },
            },
        },
        "models": [_jsonable(asdict(spec)) for spec in [model_map[candidate.name], *opponents]],
        "v5_checkpoint_metadata": candidate.loaded.get("metadata", {}),
        "total_battles": len(results),
        "error_count": sum(bool(row["error"]) for row in results),
        "events_logged": bool(log_events),
        "results": _jsonable(results),
    }
    (mode_dir / "raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    write_report_artifacts(payload, mode_dir)
    h2h = _v5_h2h(results, candidate.name)
    (mode_dir / "v5_h2h.json").write_text(
        json.dumps(h2h, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"payload": payload, "h2h": h2h}


def _comparison(
    no_aux: list[dict[str, Any]],
    with_aux: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    left = {row["opponent"]: row for row in no_aux}
    right = {row["opponent"]: row for row in with_aux}
    rows = []
    for opponent in sorted(set(left) | set(right)):
        before = left.get(opponent, {})
        after = right.get(opponent, {})
        before_score = float(before.get("score_rate", 0.0))
        after_score = float(after.get("score_rate", 0.0))
        rows.append(
            {
                "opponent": opponent,
                "games_per_mode": int(after.get("games", before.get("games", 0))),
                "no_aux_score_rate": before_score,
                "with_aux_score_rate": after_score,
                "delta_percentage_points": (after_score - before_score) * 100.0,
                "no_aux_record": [
                    int(before.get("wins", 0)),
                    int(before.get("losses", 0)),
                    int(before.get("draws", 0)),
                ],
                "with_aux_record": [
                    int(after.get("wins", 0)),
                    int(after.get("losses", 0)),
                    int(after.get("draws", 0)),
                ],
                "forced_draws": int(after.get("cardoptimum_forced_draws", 0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--base-name",
        default="extra-lr-v5-postB-preV5-u29250",
    )
    parser.add_argument(
        "--extra-v5-opponent",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Add another V5 NPZ opponent; may be supplied multiple times.",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Restrict the run to exact opponent names after the base, extra-V5, "
            "and classic rosters are assembled. May be supplied multiple times."
        ),
    )
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds-per-opponent",
        "--games-per-scenario",
        dest="seeds_per_opponent",
        type=int,
        default=64,
    )
    parser.add_argument("--seed", type=int, default=71324001)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "none",
            "assembler_only",
            "cardoptimum_only",
            "both",
            "no_aux",
            "with_aux",
        ),
        default=None,
    )
    parser.add_argument(
        "--modes",
        default=None,
        help=(
            "Comma-separated ablation arms. Canonical values: "
            "none,assembler_only,cardoptimum_only,both."
        ),
    )
    parser.add_argument("--log-events", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = V5NpzPolicy(
        args.checkpoint.resolve(),
        name=args.candidate_name or f"extra-lr-v5-{args.checkpoint.stem}",
    )
    base = V5NpzPolicy(
        args.base_checkpoint.resolve(),
        name=str(args.base_name),
    )
    candidate_spec = ModelSpec(
        candidate.name,
        "v5_npz",
        candidate.checkpoint,
        ranked=True,
    )
    base_spec = ModelSpec(
        base.name,
        "v5_npz",
        base.checkpoint,
        ranked=False,
    )
    classic_opponents = [
        replace(spec, ranked=False, game_multiplier=1.0)
        for spec in default_model_specs(repo_root=MAIN_ROOT)
        if "v4.1" not in spec.name.lower() and "v41" not in spec.name.lower()
    ]
    classic_opponents = filter_available_specs(
        classic_opponents,
        repo_root=MAIN_ROOT,
        strict=False,
    )
    extra_v5_policies: list[V5NpzPolicy] = []
    for raw in args.extra_v5_opponent:
        if "=" not in raw:
            parser.error("--extra-v5-opponent must use NAME=CHECKPOINT")
        name, path_text = raw.split("=", 1)
        path = Path(path_text).expanduser().resolve()
        if not name.strip():
            parser.error("--extra-v5-opponent name cannot be empty")
        if not path.exists():
            parser.error(f"extra V5 checkpoint not found: {path}")
        extra_v5_policies.append(V5NpzPolicy(path, name=name.strip()))
    extra_specs = [
        ModelSpec(policy.name, "v5_npz", policy.checkpoint, ranked=False)
        for policy in extra_v5_policies
    ]
    opponents = [base_spec, *extra_specs, *classic_opponents]
    if args.opponent:
        requested_opponents = set(args.opponent)
        available_opponents = {spec.name for spec in opponents}
        missing_opponents = sorted(requested_opponents - available_opponents)
        if missing_opponents:
            parser.error(
                "unknown --opponent name(s): "
                + ", ".join(missing_opponents)
                + "; available: "
                + ", ".join(sorted(available_opponents))
            )
        opponents = [
            spec for spec in opponents if spec.name in requested_opponents
        ]
    models = [candidate_spec, *opponents]
    if len({spec.name for spec in models}) != len(models):
        parser.error("candidate/opponent model names must be unique")
    model_map = {spec.name: spec for spec in models}
    v5_policies = {
        policy.name: policy
        for policy in [candidate, base, *extra_v5_policies]
    }
    config = BenchmarkConfig(
        games_per_scenario=max(1, int(args.seeds_per_opponent)),
        base_seed=int(args.seed),
        base_card_level=4,
        starting_player_ids=(1, 2),
        strict_models=False,
        fail_on_error=False,
        workers=1,
        max_steps=max(1, int(args.max_steps)),
        max_turns=max(1, int(args.max_turns)),
        exclude_families=(),
    )
    card_ids = tuple(sorted(int(card_id) for card_id in _load_cards_db().keys()))
    scenarios = _paired_even_scenarios(
        candidate=candidate_spec,
        opponents=opponents,
        config=config,
        card_ids=card_ids,
    )
    aux = AuxBundle.load(args.aux_dir.resolve())
    if args.mode is not None and args.modes is not None:
        parser.error("use either --mode or --modes, not both")
    aliases = {"no_aux": "none", "with_aux": "both"}
    if args.modes is not None:
        selected_modes = [
            aliases.get(item.strip(), item.strip())
            for item in args.modes.split(",")
            if item.strip()
        ]
    elif args.mode in (None, "all"):
        selected_modes = list(AUX_ABLATION_MODES)
    else:
        selected_modes = [aliases.get(args.mode, args.mode)]
    unknown_modes = [
        mode for mode in selected_modes if mode not in AUX_ABLATION_MODES
    ]
    if unknown_modes:
        parser.error(f"unknown ablation modes: {unknown_modes}")
    selected_modes = list(dict.fromkeys(selected_modes))

    runs: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        runs[mode] = _run_mode(
            mode=mode,
            output_dir=output_dir,
            candidate=candidate,
            v5_policies=v5_policies,
            opponents=opponents,
            model_map=model_map,
            scenarios=scenarios,
            config=config,
            aux=aux,
            ablation=AUX_ABLATION_MODES[mode],
            log_events=bool(args.log_events),
        )
    if "none" in runs and "both" in runs:
        comparison = _comparison(
            runs["none"]["h2h"],
            runs["both"]["h2h"],
        )
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison, ensure_ascii=False), flush=True)
    else:
        only = next(iter(runs.values()))
        print(json.dumps(only["h2h"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
