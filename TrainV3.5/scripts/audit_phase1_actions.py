#!/usr/bin/env python3
"""Audit V5 top actions in lost Phase 1 diagnostic games."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))
SCRIPTS = ROOT / "TrainV3" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase1_runtime_acceptance_bench as bench  # noqa: E402
from ai.train_v2.classic_actions_v1 import _get_me_enemy  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint  # noqa: E402
from train_v3.contracts import AssistModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_AUDIT_LANES = (
    "face_rush",
    "anti_draw_greed",
    "board_control",
    "greedy_trade",
    "punish_empty_board",
    "anti_hand_leak_overfit",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", default=",".join(DEFAULT_AUDIT_LANES))
    parser.add_argument("--seeds", default=",".join(str(v) for v in range(91000, 91008)))
    parser.add_argument("--deck-ids", default=",".join(str(v) for v in bench.DEFAULT_DECK_IDS))
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=48)
    parser.add_argument("--max-records-per-lane", type=int, default=8)
    parser.add_argument("--turns-before-loss", type=int, default=4)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve()
    lanes = tuple(item.strip() for item in args.lanes.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    deck_ids = tuple(int(item.strip()) for item in args.deck_ids.split(",") if item.strip())

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    loaded = load_checkpoint(str(checkpoint), model)

    records: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []
    records_by_lane = {lane: 0 for lane in lanes}
    for lane in lanes:
        for seed in seeds:
            for v5_player_id in (1, 2):
                for starting_player_id in (1, 2):
                    result = audit_game(
                        model=model,
                        lane=lane,
                        seed=seed,
                        max_steps=args.max_steps,
                        deck_ids=deck_ids,
                        v5_player_id=v5_player_id,
                        starting_player_id=starting_player_id,
                        top_k=args.top_k,
                    )
                    game_summaries.append(result["summary"])
                    if result["summary"]["winner_name"] != "v5":
                        lane_records = select_critical_records(
                            result["decision_records"],
                            limit=int(args.turns_before_loss),
                        )
                        remaining_lane = max(0, int(args.max_records_per_lane) - int(records_by_lane[lane]))
                        selected = lane_records[:remaining_lane]
                        records.extend(selected)
                        records_by_lane[lane] += len(selected)
                    if records_by_lane[lane] >= int(args.max_records_per_lane) or len(records) >= args.max_records:
                        break
                if records_by_lane[lane] >= int(args.max_records_per_lane) or len(records) >= args.max_records:
                    break
            if records_by_lane[lane] >= int(args.max_records_per_lane) or len(records) >= args.max_records:
                break
        if len(records) >= args.max_records:
            break
    records = records[: args.max_records]
    report = {
        "schema": "extra_lr_v5_phase1_action_audit_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": loaded.get("metadata", {}),
        "lanes": list(lanes),
        "seeds": list(seeds),
        "deck_ids": list(deck_ids),
        "top_k": int(args.top_k),
        "max_records": int(args.max_records),
        "max_records_per_lane": int(args.max_records_per_lane),
        "records_by_lane": records_by_lane,
        "turns_before_loss": int(args.turns_before_loss),
        "summary": summarize_games(game_summaries),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "summary": report["summary"]}, sort_keys=True))
    return 0


def audit_game(
    *,
    model: Any,
    lane: str,
    seed: int,
    max_steps: int,
    deck_ids: tuple[int, ...],
    v5_player_id: int,
    starting_player_id: int,
    top_k: int,
) -> dict[str, Any]:
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=int(seed),
            verify_mask=False,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
            info_mode=bench._info_mode(),
            assist_mode=AssistModeV5(),
        )
    )
    env.reset(
        p1_deck_ids=deck_ids,
        p2_deck_ids=deck_ids,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=int(starting_player_id),
        seed=int(seed),
    )
    game_index = (int(seed) - bench.DEFAULT_SEEDS[0]) * 4 + (int(v5_player_id) - 1) * 2 + (int(starting_player_id) - 1)
    lane_rng = random.Random(bench._resolve_random_seed(int(seed), int(game_index), "seed"))
    decision_records: list[dict[str, Any]] = []
    invalid_actions = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        current = env.current_player_id()
        if current == int(v5_player_id):
            action_id, top_actions, legal_action_count = score_v5_action(model, env, current, top_k=top_k)
            decision_records.append(
                {
                    "lane": lane,
                    "seed": int(seed),
                    "step": int(steps),
                    "turn_number": int(env.env._env.state.turn_number),
                    "v5_player_id": int(v5_player_id),
                    "starting_player_id": int(starting_player_id),
                    "v5_started": int(starting_player_id) == int(v5_player_id),
                    "selected_action_id": int(action_id),
                    "selected_action": describe_action(int(action_id)),
                    "state": state_summary(env, current),
                    "legal_action_count": int(legal_action_count),
                    "top_actions": top_actions,
                }
            )
        else:
            action_id = bench.select_lane_action(
                lane,
                env,
                current,
                seed=int(seed),
                step=int(steps),
                exploit_policy="state_score",
                legal_source="classic_legal_action_ids",
                rng=lane_rng,
            )
        _obs, _reward, terminated, truncated, info = env.step(int(action_id))
        invalid_actions += int(bool(info.get("invalid_action")))
        if terminated or truncated:
            break
    state = env.env._env.state
    winner = env.env.winner_id()
    opponent_id = 2 if int(v5_player_id) == 1 else 1
    v5_hp = int(state.p1.hero.hp if int(v5_player_id) == 1 else state.p2.hero.hp)
    opp_hp = int(state.p2.hero.hp if int(v5_player_id) == 1 else state.p1.hero.hp)
    summary = {
        "lane": lane,
        "seed": int(seed),
        "v5_player_id": int(v5_player_id),
        "starting_player_id": int(starting_player_id),
        "winner": winner,
        "winner_name": "v5" if winner == int(v5_player_id) else "opponent" if winner == opponent_id else None,
        "steps": int(steps),
        "hp_margin": int(v5_hp - opp_hp),
        "invalid_actions": int(invalid_actions),
        "decision_count": len(decision_records),
    }
    for record in decision_records:
        record["game_result"] = summary
    return {"summary": summary, "decision_records": decision_records}


def select_critical_records(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if not records:
        return []
    ranked = sorted(records, key=record_priority, reverse=True)
    return ranked[: max(1, int(limit))]


def record_priority(record: dict[str, Any]) -> float:
    state = record.get("state") or {}
    board_deficit = float(state.get("enemy_board_power", 0.0)) - float(state.get("my_board_power", 0.0))
    hp_deficit = float(state.get("enemy_hp", 0.0)) - float(state.get("my_hp", 0.0))
    legal_richness = min(6.0, float(len(record.get("top_actions") or [])))
    selected = record.get("selected_action") or {}
    selected_kind = str(selected.get("kind", ""))
    passive_penalty = 4.0 if selected_kind == "end" and board_deficit > 0.0 else 0.0
    return max(0.0, board_deficit) * 1.5 + max(0.0, hp_deficit) * 0.75 + legal_richness + passive_penalty


def score_v5_action(model: Any, env: TrainV3ClassicEnv, player_id: int, *, top_k: int) -> tuple[int, list[dict[str, Any]], int]:
    obs = env.observe(player_id).astype(np.float32)
    mask = env.action_mask(player_id).astype(np.float32)
    action_features = env.action_features(player_id, include_preview=False).astype(np.float32)
    logits, _value = model(mx.array(obs[None, :]), mx.array(action_features[None, :, :]))
    mx.eval(logits)
    logits_np = np.array(logits, dtype=np.float32)[0]
    legal = np.flatnonzero(mask.astype(bool))
    if legal.size == 0:
        return 0, [], 0
    legal_logits = logits_np[legal]
    order = np.argsort(-legal_logits)[: max(1, int(top_k))]
    top_actions = []
    for rank, local_idx in enumerate(order, start=1):
        action_id = int(legal[int(local_idx)])
        top_actions.append(
            {
                "rank": int(rank),
                "action_id": action_id,
                "action": describe_action(action_id),
                "logit": float(legal_logits[int(local_idx)]),
            }
        )
    return int(top_actions[0]["action_id"]), top_actions, int(legal.size)


def state_summary(env: TrainV3ClassicEnv, player_id: int) -> dict[str, Any]:
    state = env.env._env.state
    me, enemy = _get_me_enemy(state, player_id)
    return {
        "current_player_id": int(state.current_turn_owner_id),
        "turn_number": int(state.turn_number),
        "my_hp": int(me.hero.hp),
        "enemy_hp": int(enemy.hero.hp),
        "my_mana": int(me.mana),
        "enemy_mana": int(enemy.mana),
        "my_hand": [card_summary(card) for card in me.hand],
        "my_board": [card_summary(card) for card in me.board],
        "enemy_board": [card_summary(card) for card in enemy.board],
        "my_board_power": float(bench.board_power(me.board)),
        "enemy_board_power": float(bench.board_power(enemy.board)),
    }


def card_summary(card: Any) -> dict[str, Any]:
    return {
        "card_id": int(getattr(card, "card_id", -1)),
        "type": str(getattr(getattr(card, "card_type", ""), "name", getattr(card, "card_type", ""))).split(".")[-1].lower(),
        "mana": int(getattr(card, "mana_cost", 0)),
        "atk": int(getattr(card, "attack", 0)),
        "hp": int(getattr(card, "hp", 0)),
        "level": int(getattr(card, "level", 0)),
        "mechanics": [str(value) for value in getattr(card, "mechanics", ())],
    }


def describe_action(action_id: int) -> dict[str, Any]:
    kind, a, b = bench.decode_compact_action(int(action_id))
    target = "enemy_hero" if kind == "attack" and b == 7 else b
    if kind == "play":
        target = "none" if b == 0 else "enemy_hero" if b == 8 else b
    return {"kind": kind, "source_index": int(a), "target": target}


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    by_lane: dict[str, dict[str, Any]] = {}
    for lane in sorted({game["lane"] for game in games}):
        lane_games = [game for game in games if game["lane"] == lane]
        wins = sum(1 for game in lane_games if game["winner_name"] == "v5")
        by_lane[lane] = {
            "games": len(lane_games),
            "score_rate": wins / len(lane_games) if lane_games else 0.0,
            "avg_hp_margin": sum(float(game["hp_margin"]) for game in lane_games) / len(lane_games) if lane_games else 0.0,
        }
    wins = sum(1 for game in games if game["winner_name"] == "v5")
    return {
        "games": len(games),
        "score_rate": wins / len(games) if games else 0.0,
        "by_lane": by_lane,
    }


if __name__ == "__main__":
    raise SystemExit(main())
