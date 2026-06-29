#!/usr/bin/env python3
"""Experimental runtime Phase 1 bench for V5 against deterministic exploit lanes.

This script is source-controlled so the acceptance harness is no longer a
one-off side-channel artifact. Treat it as experimental until it reproduces the
historical `phase1_acceptance_runtime_bench_u0325.json` numbers closely enough.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
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

from ai.train_v2.classic_actions_v1 import _get_me_enemy  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


LANES = (
    "legal_random",
    "face_rush",
    "board_control",
    "greedy_trade",
    "stall",
    "punish_empty_board",
    "anti_draw_greed",
    "anti_hand_leak_overfit",
)
DEFAULT_DECK_IDS = (1, 37, 38, 40, 41, 42, 27, 28, 29)
DEFAULT_SEEDS = tuple(range(91000, 91008))
THRESHOLDS = {
    "invalid_action_max": 0,
    "p1_p2_max_score_gap": 0.12,
    "random_min_score_rate": 0.45,
    "scenario_min_score_rate": 0.42,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--deck-ids", default=",".join(str(v) for v in DEFAULT_DECK_IDS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--lanes", default=",".join(LANES), help="Comma-separated subset of Phase 1 lanes.")
    parser.add_argument("--exploit-policy", choices=("state_score", "simple"), default="state_score")
    parser.add_argument("--opponent-deck-mode", choices=("mirror", "default"), default="mirror")
    parser.add_argument(
        "--legal-source",
        choices=("classic_legal_action_ids", "action_mask"),
        default="classic_legal_action_ids",
        help="Source for opponent legal action ordering. classic_legal_action_ids mirrors train_v2 policies.",
    )
    parser.add_argument(
        "--random-seed-mode",
        choices=("seed", "seed_plus_game_index", "seed_times_100_plus_game_index"),
        default="seed",
        help="Seed scheme for the legal_random opponent policy.",
    )
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve()
    deck_ids = tuple(int(v.strip()) for v in args.deck_ids.split(",") if v.strip())
    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
    lanes = tuple(v.strip() for v in args.lanes.split(",") if v.strip())
    unknown_lanes = sorted(set(lanes) - set(LANES))
    if unknown_lanes:
        raise ValueError(f"unknown lanes: {unknown_lanes}")
    output = args.output
    if output is None:
        update = _checkpoint_update(checkpoint)
        output = checkpoint.parents[1] / f"phase1_acceptance_runtime_bench_u{update:04d}.json"
    output = output.resolve()

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    loaded = load_checkpoint(str(checkpoint), model)
    metadata = dict(loaded.get("metadata", {}))

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for lane in lanes:
        for seed in seeds:
            for v5_player_id in (1, 2):
                for starting_player_id in (1, 2):
                    row = run_game(
                        model=model,
                        lane=lane,
                        seed=seed,
                        max_steps=int(args.max_steps),
                        deck_ids=deck_ids,
                        v5_player_id=v5_player_id,
                        starting_player_id=starting_player_id,
                        exploit_policy=args.exploit_policy,
                        opponent_deck_mode=args.opponent_deck_mode,
                        legal_source=args.legal_source,
                        random_seed_mode=args.random_seed_mode,
                    )
                    rows.append(row)
                    print(
                        "PHASE1_ACCEPTANCE_GAME",
                        json.dumps(
                            {
                                "lane": lane,
                                "seed": seed,
                                "v5_player_id": v5_player_id,
                                "starting_player_id": starting_player_id,
                                "winner": row["winner_name"],
                                "steps": row["steps"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    report = build_report(
        checkpoint=checkpoint,
        checkpoint_update=_checkpoint_update(checkpoint),
        checkpoint_metadata=metadata,
        deck_ids=deck_ids,
        seeds=seeds,
        max_steps=int(args.max_steps),
        exploit_policy=args.exploit_policy,
        opponent_deck_mode=args.opponent_deck_mode,
        legal_source=args.legal_source,
        random_seed_mode=args.random_seed_mode,
        lanes=lanes,
        rows=rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE1_ACCEPTANCE_REPORT", output)
    print(json.dumps(report["acceptance"], sort_keys=True), flush=True)
    return 0 if report["acceptance"]["phase1_pass"] else 2


def run_game(
    *,
    model: Any,
    lane: str,
    seed: int,
    max_steps: int,
    deck_ids: tuple[int, ...],
    v5_player_id: int,
    starting_player_id: int,
    exploit_policy: str,
    opponent_deck_mode: str,
    legal_source: str,
    random_seed_mode: str,
) -> dict[str, Any]:
    opponent_id = 2 if int(v5_player_id) == 1 else 1
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=int(seed),
            verify_mask=False,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
            info_mode=_info_mode(),
            assist_mode=AssistModeV5(),
        )
    )
    p1_deck_ids = deck_ids if int(v5_player_id) == 1 or opponent_deck_mode == "mirror" else None
    p2_deck_ids = deck_ids if int(v5_player_id) == 2 or opponent_deck_mode == "mirror" else None
    env.reset(
        p1_deck_ids=p1_deck_ids,
        p2_deck_ids=p2_deck_ids,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=int(starting_player_id),
        seed=int(seed),
    )
    game_index = (int(seed) - DEFAULT_SEEDS[0]) * 4 + (int(v5_player_id) - 1) * 2 + (int(starting_player_id) - 1)
    lane_rng = random.Random(_resolve_random_seed(int(seed), int(game_index), str(random_seed_mode)))
    invalid_actions = 0
    v5_invalid_fallbacks = 0
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        current = env.current_player_id()
        if current == int(v5_player_id):
            action_id, fallback = select_v5_action(model, env, current)
            v5_invalid_fallbacks += int(fallback)
        else:
            action_id = select_lane_action(
                lane,
                env,
                current,
                seed=int(seed),
                step=int(steps),
                exploit_policy=exploit_policy,
                legal_source=legal_source,
                rng=lane_rng,
            )
        _obs, _reward, terminated, truncated, info = env.step(int(action_id))
        invalid_actions += int(bool(info.get("invalid_action")))
        if terminated or truncated:
            break

    state = env.env._env.state
    winner = env.env.winner_id()
    v5_hp = int(state.p1.hero.hp if int(v5_player_id) == 1 else state.p2.hero.hp)
    opp_hp = int(state.p2.hero.hp if int(v5_player_id) == 1 else state.p1.hero.hp)
    return {
        "lane": lane,
        "seed": int(seed),
        "v5_player_id": int(v5_player_id),
        "opponent_player_id": int(opponent_id),
        "starting_player_id": int(starting_player_id),
        "v5_started": int(starting_player_id) == int(v5_player_id),
        "winner": winner,
        "winner_name": "v5" if winner == int(v5_player_id) else "opponent" if winner == int(opponent_id) else None,
        "v5_win": winner == int(v5_player_id),
        "draw": winner is None,
        "steps": int(steps),
        "turns": int(state.turn_number),
        "p1_hp": int(state.p1.hero.hp),
        "p2_hp": int(state.p2.hero.hp),
        "v5_hp": v5_hp,
        "opponent_hp": opp_hp,
        "hp_margin": v5_hp - opp_hp,
        "invalid_actions": int(invalid_actions),
        "v5_invalid_fallbacks": int(v5_invalid_fallbacks),
    }


def select_v5_action(model: Any, env: TrainV3ClassicEnv, player_id: int) -> tuple[int, bool]:
    obs = env.observe(player_id).astype(np.float32)
    mask = env.action_mask(player_id).astype(np.float32)
    action_features = env.action_features(player_id, include_preview=False).astype(np.float32)
    logits, _value = model(mx.array(obs[None, :]), mx.array(action_features[None, :, :]))
    mx.eval(logits)
    logits_np = np.array(logits, dtype=np.float32)[0]
    masked = np.where(mask.astype(bool), logits_np, -1.0e9)
    action_id = int(np.argmax(masked))
    if mask[action_id] == 1.0:
        return action_id, False
    legal = np.flatnonzero(mask == 1.0)
    return (int(legal[0]) if legal.size else 0), True


def select_lane_action(
    lane: str,
    env: TrainV3ClassicEnv,
    player_id: int,
    *,
    seed: int,
    step: int,
    exploit_policy: str,
    legal_source: str,
    rng: random.Random,
) -> int:
    legal = opponent_legal_action_ids(env, player_id, legal_source=legal_source)
    if not legal:
        return 0
    if lane == "legal_random":
        return int(rng.choice(legal))
    legal_np = np.asarray(legal, dtype=np.int64)
    if exploit_policy == "simple":
        selected = select_simple_lane_action(lane, legal_np)
        return int(selected if selected is not None else legal_np[0])
    state = env.env._env.state
    me, enemy = _get_me_enemy(state, player_id)
    best_id = int(legal_np[0])
    best_score = float("-inf")
    for action_id in legal_np:
        score = exploit_action_score(str(lane), int(action_id), me, enemy)
        if score > best_score:
            best_score = score
            best_id = int(action_id)
    return best_id


def opponent_legal_action_ids(env: TrainV3ClassicEnv, player_id: int, *, legal_source: str) -> list[int]:
    if legal_source == "classic_legal_action_ids":
        return [int(action_id) for action_id in env.env.legal_action_ids(player_id)]
    if legal_source == "action_mask":
        return [int(action_id) for action_id in np.flatnonzero(env.action_mask(player_id).astype(np.float32) == 1.0)]
    raise ValueError("legal_source must be classic_legal_action_ids or action_mask")


def _resolve_random_seed(seed: int, game_index: int, mode: str) -> int:
    if mode == "seed":
        return int(seed)
    if mode == "seed_plus_game_index":
        return int(seed) + int(game_index)
    if mode == "seed_times_100_plus_game_index":
        return int(seed) * 100 + int(game_index)
    raise ValueError("unknown random_seed_mode")


def select_simple_lane_action(lane: str, legal: np.ndarray) -> int | None:
    decoded = [(int(action_id), decode_compact_action(int(action_id))) for action_id in legal]
    if lane == "stall":
        return 0 if any(action_id == 0 for action_id, _action in decoded) else int(legal[0])
    if lane in {"face_rush", "punish_empty_board"}:
        for action_id, (kind, _a, target_code) in decoded:
            if kind == "attack" and target_code == 7:
                return action_id
        return first_kind(decoded, "play") or int(legal[0])
    if lane in {"board_control", "greedy_trade"}:
        for action_id, (kind, _a, target_code) in decoded:
            if kind == "attack" and target_code > 0:
                return action_id
        return first_kind(decoded, "play") or int(legal[0])
    if lane in {"anti_draw_greed", "anti_hand_leak_overfit"}:
        return first_kind(decoded, "play") or first_kind(decoded, "attack") or int(legal[0])
    return int(legal[0])


def first_kind(decoded: list[tuple[int, tuple[str, int, int]]], kind: str) -> int | None:
    for action_id, action in decoded:
        if action[0] == kind:
            return action_id
    return None


def exploit_action_score(lane: str, action_id: int, me: Any, enemy: Any) -> float:
    kind, a, b = decode_compact_action(action_id)
    tie = -float(action_id) * 0.000001
    if lane == "face_rush":
        score = face_rush_score(kind, a, b, me, enemy)
    elif lane == "board_control":
        score = board_control_score(kind, a, b, me, enemy)
    elif lane == "greedy_trade":
        score = greedy_trade_score(kind, a, b, me, enemy)
    elif lane == "stall":
        score = stall_score(kind, a, b, me, enemy)
    elif lane == "punish_empty_board":
        score = punish_empty_board_score(kind, a, b, me, enemy)
    elif lane == "anti_draw_greed":
        score = anti_draw_greed_score(kind, a, b, me, enemy)
    elif lane == "anti_hand_leak_overfit":
        score = anti_hand_leak_overfit_score(kind, a, b, me, enemy)
    else:
        score = 0.0
    return float(score + tie)


def decode_compact_action(action_id: int) -> tuple[str, int, int]:
    if action_id == 0:
        return ("end", -1, -1)
    if 1 <= action_id <= 544:
        flat = action_id - 1
        pos_idx, target_code = divmod(flat, 17)
        hand_idx, _pos_idx = divmod(pos_idx, 8)
        return ("play", hand_idx, target_code)
    if 545 <= action_id <= 600:
        flat = action_id - 545
        attacker_idx, target_code = divmod(flat, 8)
        return ("attack", attacker_idx, target_code)
    return ("unknown", -1, -1)


def face_rush_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        return 600.0 + (1000.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack) * 8.0
    if kind == "attack":
        return 120.0 + trade_score(me, enemy, a, b) * 0.25
    if kind == "play":
        return 180.0 + hand_play_pressure(me, a) + mana_spend_score(me, a)
    if kind == "end":
        return -50.0
    return -1000.0


def board_control_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    if kind == "attack" and b < 7:
        trade = trade_score(me, enemy, a, b)
        return (520.0 + trade) if trade > 0.0 else (230.0 + trade * 0.5)
    if kind == "play":
        return 260.0 + board_development_score(me, a) + mana_spend_score(me, a)
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        return 100.0 + (900.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack)
    if kind == "end":
        return -20.0
    return -100.0


def greedy_trade_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    if kind == "attack" and b < 7:
        trade = trade_score(me, enemy, a, b)
        return (560.0 + trade * 1.6) if trade > 0.0 else (70.0 + trade * 0.35)
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        return 410.0 + (950.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack) * 5.0
    if kind == "play":
        return 190.0 + board_development_score(me, a) + mana_spend_score(me, a)
    if kind == "end":
        return -40.0
    return -100.0


def stall_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    enemy_pressure = board_power(enemy.board)
    if kind == "attack" and b < 7 and enemy_pressure > 0.0:
        return 420.0 + trade_score(me, enemy, a, b)
    if kind == "play" and enemy_pressure > 20.0:
        return 240.0 + defensive_play_score(me, a) + mana_spend_score(me, a)
    if kind == "end":
        return 120.0
    if kind == "play":
        return 70.0 + defensive_play_score(me, a)
    if kind == "attack" and b == 7:
        return 20.0 + float(board_attack(me, a))
    return -80.0


def punish_empty_board_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    if not enemy.board:
        if kind == "attack" and b == 7:
            attack = board_attack(me, a)
            return 650.0 + (1000.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack) * 8.0
        if kind == "play":
            return 320.0 + hand_play_pressure(me, a) + mana_spend_score(me, a)
        if kind == "attack":
            return 100.0
        if kind == "end":
            return -100.0
    low_board_pressure = len(enemy.board) <= 1
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        pressure_bonus = 240.0 if low_board_pressure else 40.0
        return 300.0 + pressure_bonus + (1000.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack) * 7.0
    if kind == "play":
        return 300.0 + hand_play_pressure(me, a) + mana_spend_score(me, a)
    if kind == "attack" and b < 7:
        trade = trade_score(me, enemy, a, b)
        clear_bonus = 80.0 if len(enemy.board) <= 2 else 0.0
        return 260.0 + clear_bonus + trade * 0.55
    if kind == "end":
        return -80.0
    return -80.0


def anti_draw_greed_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    my_pressure = board_power(me.board)
    enemy_pressure = board_power(enemy.board)
    pressure_deficit = max(0.0, enemy_pressure - my_pressure)
    hand_overflow = max(0.0, float(len(me.hand)) - 3.0)
    unused_mana = float(max(0, int(getattr(me, "mana", 0))))
    if kind == "play":
        pressure_penalty = -120.0 if pressure_deficit > 18.0 else 0.0
        return (
            360.0
            + board_development_score(me, a)
            + mana_spend_score(me, a) * 1.4
            + hand_overflow * 32.0
            + pressure_penalty
        )
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        return 170.0 + (1000.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack) * 3.0
    if kind == "attack" and b < 7:
        trade = trade_score(me, enemy, a, b)
        pressure_bonus = pressure_deficit * 4.0 if pressure_deficit > 8.0 else 0.0
        return 420.0 + trade * 0.75 + pressure_bonus
    if kind == "end":
        return -20.0 - hand_overflow * 12.0 - unused_mana * 3.0 - (60.0 if pressure_deficit > 10.0 else 0.0)
    return -50.0


def anti_hand_leak_overfit_score(kind: str, a: int, b: int, me: Any, enemy: Any) -> float:
    enemy_pressure = board_power(enemy.board)
    if kind == "attack" and b < 7:
        trade = trade_score(me, enemy, a, b)
        return (430.0 + trade) if trade > 35.0 or enemy_pressure > 12.0 else (120.0 + trade * 0.4)
    if kind == "play":
        return 360.0 + defensive_play_score(me, a) + mana_spend_score(me, a) * 0.8
    if kind == "attack" and b == 7:
        attack = board_attack(me, a)
        return 120.0 + (950.0 if attack >= int(enemy.hero.hp) else 0.0) + float(attack)
    if kind == "end":
        return -20.0
    return -100.0


def trade_score(me: Any, enemy: Any, attacker_index: int, target_code: int) -> float:
    if attacker_index < 0 or attacker_index >= len(me.board) or target_code < 0 or target_code >= len(enemy.board):
        return -1000.0
    attacker = me.board[attacker_index]
    target = enemy.board[target_code]
    attack = effective_attack(attacker, me.board, me.hero)
    target_attack = effective_attack(target, enemy.board, enemy.hero)
    kills_target = attack >= int(target.hp)
    attacker_survives = int(attacker.hp) > target_attack
    target_value = card_power(target)
    attacker_value = card_power(attacker)
    score = target_value * 5.0 - attacker_value * 1.25
    score += 120.0 + target_value * 3.0 if kills_target else -(80.0 + float(target.hp) * 4.0)
    score += 45.0 + float(attacker.hp) if attacker_survives else -(55.0 + attacker_value)
    return score


def hand_play_pressure(me: Any, hand_index: int) -> float:
    card = _hand_card(me, hand_index)
    if card is None:
        return -1000.0
    return card_power(card) + (20.0 if has_mechanic(card, "charge") else 0.0)


def board_development_score(me: Any, hand_index: int) -> float:
    card = _hand_card(me, hand_index)
    if card is None:
        return -1000.0
    return (
        card_power(card)
        + (25.0 if _card_type_name(card) == "warrior" else 0.0)
        + (10.0 if has_mechanic(card, "taunt") else 0.0)
    )


def defensive_play_score(me: Any, hand_index: int) -> float:
    card = _hand_card(me, hand_index)
    if card is None:
        return -1000.0
    return (
        float(card.hp) * 4.0
        + float(card.attack)
        + (45.0 if has_mechanic(card, "taunt") else 0.0)
        + (20.0 if has_mechanic(card, "shield") else 0.0)
    )


def mana_spend_score(me: Any, hand_index: int) -> float:
    card = _hand_card(me, hand_index)
    if card is None:
        return -1000.0
    spare = abs(float(me.mana) - float(card.mana_cost))
    return float(card.mana_cost) * 8.0 - spare


def board_attack(me: Any, attacker_index: int) -> int:
    if attacker_index < 0 or attacker_index >= len(me.board):
        return 0
    return effective_attack(me.board[attacker_index], me.board, me.hero)


def board_power(board: Any) -> float:
    return sum(card_power(card) for card in board)


def card_power(card: Any) -> float:
    return (
        float(max(0, int(card.attack)) * max(0, int(card.hp)))
        + float(max(0, int(getattr(card, "level", 0))))
        + (3.0 if has_mechanic(card, "taunt") else 0.0)
        + (4.0 if has_mechanic(card, "shield") else 0.0)
        + (2.0 if has_mechanic(card, "charge") else 0.0)
    )


def effective_attack(card: Any, board: Any, hero: Any) -> int:
    bonus = 0
    for aura in board:
        if getattr(aura, "instance_id", None) == getattr(card, "instance_id", None):
            continue
        bonus += sum(parse_aura_atk(value) or 0 for value in getattr(aura, "mechanics", ()))
    bonus += sum(parse_aura_atk(value) or 0 for value in getattr(hero, "mechanics", ()))
    return int(card.attack) + int(bonus)


def parse_aura_atk(mechanic: str) -> int | None:
    text = str(mechanic)
    if not text.startswith("aura_atk_"):
        return None
    rest = text.removeprefix("aura_atk_")
    if "_" in rest:
        return None
    try:
        return int(rest)
    except ValueError:
        return None


def has_mechanic(card: Any, mechanic: str) -> bool:
    return any(str(value) == mechanic for value in getattr(card, "mechanics", ()))


def _hand_card(me: Any, hand_index: int) -> Any | None:
    if hand_index < 0 or hand_index >= len(me.hand):
        return None
    return me.hand[hand_index]


def _card_type_name(card: Any) -> str:
    raw = getattr(card, "card_type", "")
    name = getattr(raw, "name", raw)
    return str(name).split(".")[-1].lower()


def build_report(
    *,
    checkpoint: Path,
    checkpoint_update: int,
    checkpoint_metadata: dict[str, Any],
    deck_ids: tuple[int, ...],
    seeds: tuple[int, ...],
    max_steps: int,
    exploit_policy: str,
    opponent_deck_mode: str,
    legal_source: str,
    random_seed_mode: str,
    lanes: tuple[str, ...],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    by_lane = {lane: summarize([row for row in rows if row["lane"] == lane]) for lane in lanes}
    overall = summarize(rows)
    random_pass = (
        by_lane["legal_random"]["score_rate"] >= THRESHOLDS["random_min_score_rate"]
        if "legal_random" in by_lane
        else True
    )
    scenario_pass = all(
        by_lane[lane]["score_rate"] >= THRESHOLDS["scenario_min_score_rate"]
        for lane in lanes
        if lane != "legal_random"
    )
    invalid_pass = overall["invalid_actions"] <= THRESHOLDS["invalid_action_max"]
    side_gap_pass = overall["p1_p2_score_gap"] <= THRESHOLDS["p1_p2_max_score_gap"]
    return {
        "schema": "extra_lr_v5_phase1_acceptance_runtime_bench_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(elapsed_seconds),
        "phase": str(
            checkpoint_metadata.get("trainv3_phase")
            or (checkpoint_metadata.get("config") or {}).get("phase")
            or "unknown"
        ),
        "model_name": "extra-lr-v5-adaptive",
        "checkpoint": str(checkpoint),
        "checkpoint_update": int(checkpoint_update),
        "checkpoint_metadata": {
            "run_name": checkpoint_metadata.get("run_name"),
            "trace_manifest_id": checkpoint_metadata.get("trace_manifest_id"),
            "resume_source_update": checkpoint_metadata.get("resume_source_update"),
            "optimizer_learning_rate": checkpoint_metadata.get("optimizer_learning_rate"),
        },
        "obs_v5_dim": 6480,
        "deck_ids": list(deck_ids),
        "seeds": list(seeds),
        "max_steps": int(max_steps),
        "lanes": list(lanes),
        "mode": {
            "info_mode": asdict(_info_mode()),
            "assist_mode": asdict(AssistModeV5()),
            "exploit_policy": str(exploit_policy),
            "opponent_deck_mode": str(opponent_deck_mode),
            "legal_source": str(legal_source),
            "random_seed_mode": str(random_seed_mode),
        },
        "thresholds": dict(THRESHOLDS),
        "notes": (
            "Scenario policies are deterministic Python oracle eval mirrors of Rust training lane "
            "preferences; no assist/private enemy info enabled."
        ),
        "rows": rows,
        "by_lane": by_lane,
        "overall": overall,
        "acceptance": {
            "invalid_action_pass": bool(invalid_pass),
            "legal_random_score_pass": bool(random_pass),
            "scenario_score_pass": bool(scenario_pass),
            "side_gap_pass": bool(side_gap_pass),
            "phase1_pass": bool(invalid_pass and random_pass and scenario_pass and side_gap_pass),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(1 for row in rows if row["winner_name"] == "v5")
    draws = sum(1 for row in rows if row["winner_name"] is None)
    p1 = [row for row in rows if row["v5_player_id"] == 1]
    p2 = [row for row in rows if row["v5_player_id"] == 2]
    first = [row for row in rows if row["v5_started"]]
    second = [row for row in rows if not row["v5_started"]]
    p1_score = score_rate(p1)
    p2_score = score_rate(p2)
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "winrate": wins / total if total else 0.0,
        "score_rate": (wins + 0.5 * draws) / total if total else 0.0,
        "p1_score_rate": p1_score,
        "p2_score_rate": p2_score,
        "p1_p2_score_gap": abs(p1_score - p2_score),
        "first_score_rate": score_rate(first),
        "second_score_rate": score_rate(second),
        "avg_hp_margin": mean(row["hp_margin"] for row in rows),
        "avg_steps": mean(row["steps"] for row in rows),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "v5_invalid_fallbacks": sum(int(row["v5_invalid_fallbacks"]) for row in rows),
    }


def score_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for row in rows if row["winner_name"] == "v5")
    draws = sum(1 for row in rows if row["winner_name"] is None)
    return (wins + 0.5 * draws) / len(rows)


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _info_mode() -> InfoModeV5:
    return InfoModeV5(
        adaptive_strength=1.0,
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=False,
        enemy_deck_known=False,
        enemy_deck_order_known=False,
        draw_assist_enabled=False,
        draw_assist_strength=0.0,
    )


def _checkpoint_update(path: Path) -> int:
    stem = path.stem
    marker = "update_"
    if marker not in stem:
        return 0
    return int(stem.rsplit(marker, 1)[1])


if __name__ == "__main__":
    raise SystemExit(main())
