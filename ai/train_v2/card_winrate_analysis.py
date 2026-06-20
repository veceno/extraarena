"""
Card/deck win-rate analytics for TrainV2 ONNX candidates.

This is an evaluation-only helper: it runs deterministic matches and aggregates
which visible cards, played cards, and card pairs correlate with candidate wins.
It does not affect training or production inference.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ai.train_v2.classic_actions_v1 import decode_action
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.onnx_policy import OnnxActionPolicy
from ai.train_v2.policies import EndTurnPolicy, GreedyFacePolicy, RandomLegalPolicy
from core.state import CardInstance


TRAINV2_ONNX: dict[str, str] = {
    "trainv2_0251": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/best_update_0251.onnx",
    "trainv2_0348": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0348.onnx",
    "trainv2_0700": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/best_update_0700.onnx",
    "trainv2_0800": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/update_0800.onnx",
}

DEFAULT_OPPONENTS = "trainv2_0251,trainv2_0348,greedy_face"
LEGACY_OPPONENTS = {"legacy_max", "legacy_medium", "legacy_random_biggest"}


def _make_policy(name: str, *, seed: int = 0):
    if name == "random":
        return RandomLegalPolicy(seed=seed)
    if name == "end_turn":
        return EndTurnPolicy()
    if name == "greedy_face":
        return GreedyFacePolicy()
    if name in LEGACY_OPPONENTS:
        raise ValueError(f"legacy opponents are unsupported in the v4 bot pipeline: {name}")
    if name in TRAINV2_ONNX:
        return OnnxActionPolicy(TRAINV2_ONNX[name], mode="argmax", seed=seed, verify_mask=False)
    raise ValueError(f"unknown opponent: {name}")


def _card_key(card: CardInstance) -> str:
    return f"{card.card_id}:{card.name}"


def _player_cards(env: ClassicRLEnv, player_id: int) -> list[CardInstance]:
    st = env._env.state
    player = st.p1 if st.p1.user_id == player_id else st.p2
    cards = [player.hero]
    cards.extend(player.hand)
    cards.extend(player.board)
    cards.extend(player.deck)
    cards.extend(player.graveyard)
    return cards


def _dedupe_cards(cards: list[CardInstance]) -> list[str]:
    return sorted({_card_key(card) for card in cards if card.card_id})


def _find_card_by_instance(env: ClassicRLEnv, instance_id: str | None) -> CardInstance | None:
    if not instance_id:
        return None
    st = env._env.state
    for player in (st.p1, st.p2):
        if str(player.hero.instance_id) == str(instance_id):
            return player.hero
        for zone in (player.hand, player.board, player.deck, player.graveyard):
            for card in zone:
                if str(card.instance_id) == str(instance_id):
                    return card
    return None


def _describe_candidate_action(env: ClassicRLEnv, player_id: int, action_id: int) -> dict[str, Any]:
    st = env._env.state
    me = st.p1 if st.p1.user_id == player_id else st.p2
    action = decode_action(st, player_id, action_id)
    if action is None:
        return {"type": "invalid"}

    data = action.to_dict()
    action_type = data.get("type")
    if action_type == "play_card":
        hand_index = int(data.get("hand_index", -1))
        card = me.hand[hand_index] if 0 <= hand_index < len(me.hand) else None
        target = _find_card_by_instance(env, data.get("target_id"))
        return {
            "type": "play_card",
            "card": _card_key(card) if card else None,
            "target": _card_key(target) if target else None,
        }

    if action_type == "attack":
        attacker = _find_card_by_instance(env, data.get("attacker_id"))
        target = _find_card_by_instance(env, data.get("target_id"))
        return {
            "type": "attack",
            "card": _card_key(attacker) if attacker else None,
            "target": _card_key(target) if target else "enemy_hero",
            "target_is_hero": bool(data.get("target_is_hero")),
        }

    return {"type": action_type or "unknown"}


def _record_counter(counter: dict[str, Counter], key: str, won: bool) -> None:
    counter[key]["games"] += 1
    if won:
        counter[key]["wins"] += 1


def _rank(counter: dict[str, Counter], *, total_games: int, total_wins: int, min_games: int, limit: int) -> list[dict]:
    baseline = total_wins / total_games if total_games else 0.0
    rows = []
    for key, c in counter.items():
        games = int(c["games"])
        if games < min_games:
            continue
        wins = int(c["wins"])
        winrate = wins / games if games else 0.0
        rows.append(
            {
                "key": key,
                "games": games,
                "wins": wins,
                "winrate": round(winrate, 4),
                "lift": round(winrate - baseline, 4),
            }
        )
    rows.sort(key=lambda r: (r["lift"], r["games"], r["winrate"]), reverse=True)
    return rows[:limit]


def analyze_cards(
    *,
    model: str,
    opponents: list[str],
    games: int,
    seed: int,
    max_steps: int,
    output: str | None,
    min_games: int,
    limit: int,
) -> dict:
    candidate = OnnxActionPolicy(model, mode="argmax", seed=seed, verify_mask=False)
    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")

    initial_cards: dict[str, Counter] = defaultdict(Counter)
    initial_pairs: dict[str, Counter] = defaultdict(Counter)
    played_cards: dict[str, Counter] = defaultdict(Counter)
    played_pairs: dict[str, Counter] = defaultdict(Counter)
    attacking_cards: dict[str, Counter] = defaultdict(Counter)
    matchup_rows: list[dict] = []

    total_games = 0
    total_wins = 0
    total_draws = 0

    for opponent_name in opponents:
        opponent = _make_policy(opponent_name, seed=seed + 1000)
        matchup_games = 0
        matchup_wins = 0
        matchup_draws = 0

        for game_idx in range(games):
            for candidate_player_id in (1, 2):
                game_seed = seed + game_idx
                if hasattr(candidate, "reset"):
                    candidate.reset(game_seed * 2 + candidate_player_id)
                if hasattr(opponent, "reset"):
                    opponent.reset(game_seed * 3 + candidate_player_id)

                env.reset(seed=game_seed)
                start_cards = _dedupe_cards(_player_cards(env, candidate_player_id))
                episode_played: set[str] = set()
                episode_attackers: set[str] = set()

                for _ in range(max_steps):
                    cp = env.current_player_id()
                    if cp == candidate_player_id:
                        action_id = candidate.select_action(env, cp)
                        desc = _describe_candidate_action(env, cp, action_id)
                        if desc.get("type") == "play_card" and desc.get("card"):
                            episode_played.add(desc["card"])
                        elif desc.get("type") == "attack" and desc.get("card"):
                            episode_attackers.add(desc["card"])
                    else:
                        action_id = opponent.select_action(env, cp)

                    _, _, terminated, truncated, _ = env.step(action_id)
                    if terminated or truncated:
                        break

                winner = env.winner_id()
                won = winner == candidate_player_id
                draw = winner is None

                total_games += 1
                matchup_games += 1
                if won:
                    total_wins += 1
                    matchup_wins += 1
                if draw:
                    total_draws += 1
                    matchup_draws += 1

                for card in start_cards:
                    _record_counter(initial_cards, card, won)
                for left, right in itertools.combinations(start_cards, 2):
                    _record_counter(initial_pairs, f"{left} + {right}", won)
                for card in episode_played:
                    _record_counter(played_cards, card, won)
                for left, right in itertools.combinations(sorted(episode_played), 2):
                    _record_counter(played_pairs, f"{left} + {right}", won)
                for card in episode_attackers:
                    _record_counter(attacking_cards, card, won)

        matchup_rows.append(
            {
                "opponent": opponent_name,
                "games": matchup_games,
                "wins": matchup_wins,
                "draws": matchup_draws,
                "winrate": round(matchup_wins / matchup_games, 4) if matchup_games else 0.0,
            }
        )

    result = {
        "model": model,
        "opponents": opponents,
        "games_per_opponent_per_side": games,
        "total_games": total_games,
        "total_wins": total_wins,
        "total_draws": total_draws,
        "overall_winrate": round(total_wins / total_games, 4) if total_games else 0.0,
        "matchups": matchup_rows,
        "top_initial_cards": _rank(initial_cards, total_games=total_games, total_wins=total_wins, min_games=min_games, limit=limit),
        "top_initial_pairs": _rank(initial_pairs, total_games=total_games, total_wins=total_wins, min_games=min_games, limit=limit),
        "top_played_cards": _rank(played_cards, total_games=total_games, total_wins=total_wins, min_games=min_games, limit=limit),
        "top_played_pairs": _rank(played_pairs, total_games=total_games, total_wins=total_wins, min_games=min_games, limit=limit),
        "top_attackers": _rank(attacking_cards, total_games=total_games, total_wins=total_wins, min_games=min_games, limit=limit),
    }

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    for row in rows:
        print(
            f"  {row['winrate'] * 100:5.1f}% "
            f"lift={row['lift'] * 100:+5.1f}pp "
            f"games={row['games']:4d}  {row['key']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cards/decks correlated with TrainV2 candidate wins")
    parser.add_argument("--model", required=True, help="Candidate TrainV2 ONNX path")
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--games", type=int, default=50, help="Seeds per opponent; both sides are played")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--output", default="")
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    result = analyze_cards(
        model=args.model,
        opponents=[item.strip() for item in args.opponents.split(",") if item.strip()],
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        output=args.output or None,
        min_games=args.min_games,
        limit=args.limit,
    )

    print(
        f"Analyzed {result['total_games']} games | "
        f"overall winrate={result['overall_winrate'] * 100:.1f}% | "
        f"draws={result['total_draws']}"
    )
    print("\nMatchups")
    for row in result["matchups"]:
        print(f"  {row['opponent']}: {row['winrate'] * 100:5.1f}% ({row['wins']}/{row['games']}, draws={row['draws']})")
    _print_table("Top initial deck cards", result["top_initial_cards"])
    _print_table("Top initial deck pairs", result["top_initial_pairs"])
    _print_table("Top played cards", result["top_played_cards"])
    _print_table("Top played pairs", result["top_played_pairs"])
    _print_table("Top attacking cards", result["top_attackers"])


if __name__ == "__main__":
    main()
