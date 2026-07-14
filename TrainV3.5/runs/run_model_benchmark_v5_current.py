#!/usr/bin/env python3
"""Run a V5 NPZ checkpoint through the gitignored ai/model_benchmark suite."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np


# This script lives at ``<worktree>/TrainV3.5/runs``.  ``parents[2]`` is the
# actual worktree root; ``parents[3]`` points at its ``.claude`` parent and
# makes the standalone benchmark fail to import the worktree's ``ai`` package.
WORKTREE = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path("/Users/laveqox/Documents/ExtraArenaRaS")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "TrainV3.5" / "python"))

# ai/model_benchmark is a local gitignored layer present only in the main
# checkout. Extend the worktree's ai package path so all train_v2 imports still
# resolve to the audited worktree code.
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
from ai.model_benchmark.scenarios import generate_scenarios  # noqa: E402
from ai.train_v2.classic_actions_v1 import decode_action  # noqa: E402
from ai.train_v2.classic_rl_env import _load_cards_db  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint  # noqa: E402
from core.actions import ManaDrawAction  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.mana_draw_head_v5 import (  # noqa: E402
    mana_draw_legal_mask,
    select_includes_mana_draw,
)
from train_v3.v5_policy import create_v5_policy  # noqa: E402


class V5NpzPolicy:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.name = f"extra-lr-v5-{checkpoint.stem}"
        self.model = create_v5_policy(
            policy_kind="v5_split_encoder",
            hidden_dim=256,
            action_hidden_dim=128,
        )
        self.loaded = load_checkpoint(str(checkpoint), self.model)

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


def _run_game(
    spec,
    v5: V5NpzPolicy,
    opponents: dict[str, Any],
    model_map: dict[str, ModelSpec],
    config,
    *,
    log_events: bool,
):
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=spec.seed,
            max_turns=config.max_turns,
            verify_mask=False,
            placement_mode="append_only",
            info_mode=InfoModeV5(),
            assist_mode=AssistModeV5(),
            history_limit=20,
        )
    )
    env.reset(
        p1_levels=spec.p1_levels,
        p2_levels=spec.p2_levels,
        starting_player_id=spec.starting_player_id,
        seed=spec.seed,
    )
    opponent_name = spec.opponent_model
    if opponent_name not in opponents:
        opponents[opponent_name] = create_policy(model_map[opponent_name].policy_spec())
    opponent = opponents[opponent_name]
    v5.reset(spec.seed * 2 + 1)
    opponent.reset(spec.seed * 2 + 2)
    action_counts = {
        spec.p1_name: {"end_turn": 0, "play_card": 0, "attack": 0, "mana_draw": 0, "unknown": 0},
        spec.p2_name: {"end_turn": 0, "play_card": 0, "attack": 0, "mana_draw": 0, "unknown": 0},
    }
    invalid = {spec.p1_name: 0, spec.p2_name: 0}
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    steps = 0
    error = None
    events: list[dict[str, Any]] = []
    try:
        while not terminated and not truncated and steps < config.max_steps:
            player_id = env.current_player_id()
            actor_name = spec.p1_name if player_id == 1 else spec.p2_name
            if actor_name == v5.name:
                action = v5.select_action(env, player_id)
            else:
                action = opponent.select_action(env.env, player_id)
            action_kind = _action_kind(env, player_id, action)
            action_counts[actor_name][action_kind] += 1
            if log_events:
                state_before = _state_snapshot(env.env._env.state)
                action_json = _action_to_json(env.env._env.state, player_id, action)
            if isinstance(action, ManaDrawAction):
                _obs, reward, terminated, truncated, last_info = env.step_core_action(action)
            else:
                _obs, reward, terminated, truncated, last_info = env.step(int(action))
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
                        "state_before": state_before,
                        "state_after": _state_snapshot(env.env._env.state),
                    }
                )
    except Exception as exc:  # preserve model_benchmark's raw error contract
        error = f"{exc.__class__.__name__}: {exc}"

    state = env.env._env.state
    winner_id = env.env.winner_id()
    winner_name = spec.p1_name if winner_id == 1 else spec.p2_name if winner_id == 2 else None
    underlying_status = getattr(state.status, "value", str(state.status))
    timed_out = not terminated and not truncated and steps >= config.max_steps
    result = {
        **asdict(spec),
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
    }
    if log_events:
        result["events"] = events
        result["final_state"] = _state_snapshot(state)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _v5_h2h(results: list[dict[str, Any]], v5_name: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        group = "even" if row["family"].endswith("_even") else "v5_minus3"
        buckets[(row["opponent_model"], group)].append(row)
    output = []
    for (opponent, group), rows in sorted(buckets.items()):
        wins = sum(row["winner_name"] == v5_name for row in rows)
        draws = sum(bool(row["draw"]) for row in rows)
        first = [
            row for row in rows
            if (1 if row["p1_name"] == v5_name else 2) == row["starting_player_id"]
        ]
        second = [row for row in rows if row not in first]
        score = lambda subset: (
            sum(row["winner_name"] == v5_name for row in subset)
            + 0.5 * sum(bool(row["draw"]) for row in subset)
        ) / len(subset) if subset else 0.0
        output.append(
            {
                "opponent": opponent,
                "scenario": group,
                "games": len(rows),
                "wins": wins,
                "draws": draws,
                "losses": len(rows) - wins - draws,
                "score_rate": score(rows),
                "first_score_rate": score(first),
                "second_score_rate": score(second),
                "invalid_actions": sum(row["invalid_actions"].get(v5_name, 0) for row in rows),
                "mana_draw_count": sum(row["action_counts"][v5_name].get("mana_draw", 0) for row in rows),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-scenario", type=int, default=8)
    parser.add_argument("--seed", type=int, default=71324001)
    parser.add_argument(
        "--log-events",
        action="store_true",
        help="Store per-action state/action records in raw.json for behavioral analysis.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v5 = V5NpzPolicy(checkpoint)
    v5_spec = ModelSpec(v5.name, "v5_npz", checkpoint, ranked=True)
    opponents = [
        replace(spec, ranked=False)
        for spec in default_model_specs(repo_root=MAIN_ROOT)
        if "v4.1" not in spec.name.lower() and "v41" not in spec.name.lower()
    ]
    opponents = filter_available_specs(opponents, repo_root=MAIN_ROOT, strict=False)
    specs = [v5_spec, *opponents]
    config = BenchmarkConfig(
        games_per_scenario=args.games_per_scenario,
        base_seed=args.seed,
        base_card_level=4,
        starting_player_ids=(1, 2),
        strict_models=False,
        fail_on_error=False,
        workers=1,
        max_steps=300,
        max_turns=120,
    )
    card_ids = tuple(sorted(int(card_id) for card_id in _load_cards_db().keys()))
    scenarios = [
        replace(spec, ranked_game=True)
        for spec in generate_scenarios(specs, config, card_ids=card_ids)
    ]
    model_map = {spec.name: spec for spec in specs}
    opponent_policies: dict[str, Any] = {}
    results = []
    print(f"model_benchmark V5 plan: {len(specs)} models, {len(scenarios)} battles", flush=True)
    for index, scenario in enumerate(scenarios, start=1):
        if index == 1 or index == len(scenarios) or index % max(1, len(scenarios) // 20) == 0:
            print(f"Progress: {index}/{len(scenarios)} {scenario.scenario_id}", flush=True)
        results.append(
            _run_game(
                scenario,
                v5,
                opponent_policies,
                model_map,
                config,
                log_events=bool(args.log_events),
            )
        )

    payload = {
        "config": _jsonable(asdict(config)),
        "models": [_jsonable(asdict(spec)) for spec in specs],
        "v5_checkpoint_metadata": v5.loaded.get("metadata", {}),
        "total_battles": len(results),
        "error_count": sum(bool(row["error"]) for row in results),
        "events_logged": bool(args.log_events),
        "results": _jsonable(results),
    }
    raw_path = output_dir / "raw.json"
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_artifacts(payload, output_dir)
    h2h = _v5_h2h(results, v5.name)
    (output_dir / "v5_h2h.json").write_text(
        json.dumps(h2h, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(h2h, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
