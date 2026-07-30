#!/usr/bin/env python3
"""Generate authoritative synthetic data for ExtraLR auxiliary models.

The generator deliberately does not emit Metronome labels: human decision
latency is human-only.  It produces:

* complete V5-vs-V5 battles for TimeStamp simulator pretraining;
* controlled, side/initiative-balanced matchup cells for Assembler;
* matched counterfactual card-injection branches for CardOptimum.

Runs are deterministic, shardable, append-only, and resumable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "TrainV3.5" / "runs"
for path in (ROOT, ROOT / "TrainV3.5" / "python", RUNS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.actions import ManaDrawAction  # noqa: E402
from core.engine import HAND_CAP  # noqa: E402
from core.state import GameStatus  # noqa: E402
from rlhf_env.components.deck_builder import (  # noqa: E402
    CardCatalog,
    build_random_arena_deck,
    load_catalog,
    validate_deck,
)
from run_model_benchmark_v5_current import V5NpzPolicy, _action_kind  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402


SCHEMA_VERSION = "extra_lr_aux_synthetic_v1"
GENERATOR_VERSION = "1.0.0"
BALANCED_GAMES_PER_CELL = 20  # 5 RNG seeds x 2 seats x 2 initiatives


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _split_for(key: str) -> str:
    bucket = _stable_int("split", key) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _json_dump(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _card_snapshot(card: Any) -> dict[str, Any]:
    return {
        "card_id": int(card.card_id),
        "level": int(card.level),
        "attack": int(card.attack),
        "hp": int(card.hp),
        "max_hp": int(card.max_hp),
        "mana_cost": int(card.mana_cost),
        "ready": bool(card.is_ready),
        "frozen": bool(card.is_frozen),
        "skip_count": int(card.skip_count),
    }


def _visible_player_snapshot(player: Any, *, expose_private: bool) -> dict[str, Any]:
    row = {
        "hero": _card_snapshot(player.hero),
        "mana": int(player.mana),
        "max_mana": int(player.max_mana),
        "mana_draw_count_this_turn": int(player.mana_draw_count_this_turn),
        "hand_count": len(player.hand),
        "deck_count": len(player.deck),
        "graveyard": [_card_snapshot(card) for card in player.graveyard],
        "board": [_card_snapshot(card) for card in player.board],
    }
    if expose_private:
        row["hand"] = [_card_snapshot(card) for card in player.hand]
        row["remaining_deck"] = [_card_snapshot(card) for card in player.deck]
    return row


def _state_snapshot(env: TrainV3ClassicEnv, actor_id: int) -> dict[str, Any]:
    state = env.env._env.state
    actor = state.p1 if actor_id == 1 else state.p2
    opponent = state.p2 if actor_id == 1 else state.p1
    return {
        "information_mode": "actor_private_opponent_public_v1",
        "turn_number": int(state.turn_number),
        "actor_id": int(actor_id),
        "actor": _visible_player_snapshot(actor, expose_private=True),
        "opponent": _visible_player_snapshot(opponent, expose_private=False),
    }


def _subcatalog(catalog: CardCatalog, rng: random.Random) -> CardCatalog:
    hero_ids = rng.sample(catalog.heroes, min(4, len(catalog.heroes)))
    warrior_ids = rng.sample(catalog.warriors, min(18, len(catalog.warriors)))
    potion_ids = rng.sample(catalog.potions, min(6, len(catalog.potions)))
    ids = [*hero_ids, *warrior_ids, *potion_ids]
    return CardCatalog(
        cards={card_id: catalog.cards[card_id] for card_id in ids},
        heroes=hero_ids,
        warriors=warrior_ids,
        potions=potion_ids,
    )


def _levels_for(deck: Iterable[int], catalog: CardCatalog, rng: random.Random) -> dict[int, int]:
    center = rng.choice((3, 5, 8))
    result: dict[int, int] = {}
    for card_id in deck:
        if catalog.card(card_id).get("card_type") == "hero":
            result[int(card_id)] = rng.choice((1, 2))
        else:
            result[int(card_id)] = max(1, min(10, center + rng.randint(-2, 2)))
    return result


def _cell_spec(cell_index: int, base_seed: int, catalog: CardCatalog) -> dict[str, Any]:
    rng = random.Random(_stable_int("cell", base_seed, cell_index))
    pool = _subcatalog(catalog, rng)
    candidate = build_random_arena_deck(pool, rng=random.Random(rng.getrandbits(64)))
    opponent = build_random_arena_deck(pool, rng=random.Random(rng.getrandbits(64)))
    while opponent == candidate:
        opponent = build_random_arena_deck(pool, rng=random.Random(rng.getrandbits(64)))
    for deck in (candidate, opponent):
        ok, reason = validate_deck(deck, catalog)
        if not ok:
            raise RuntimeError(f"invalid generated deck: {reason}")
    pool_ids = sorted(pool.card_ids)
    pool_id = hashlib.sha256(",".join(map(str, pool_ids)).encode()).hexdigest()[:16]
    return {
        "cell_id": f"cell-{cell_index:06d}",
        "pool_id": pool_id,
        "allowed_pool_ids": pool_ids,
        "candidate_deck_ids": candidate,
        "opponent_deck_ids": opponent,
        "candidate_levels": _levels_for(candidate, catalog, rng),
        "opponent_levels": _levels_for(opponent, catalog, rng),
        "split": _split_for(pool_id),
    }


def _new_env(seed: int, max_turns: int) -> TrainV3ClassicEnv:
    return TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=seed,
            max_turns=max_turns,
            verify_mask=False,
            placement_mode="append_only",
            info_mode=InfoModeV5(),
            assist_mode=AssistModeV5(),
            history_limit=20,
        )
    )


def _step_policy(env: TrainV3ClassicEnv, policy: V5NpzPolicy) -> tuple[bool, bool, dict[str, Any], str]:
    actor_id = env.current_player_id()
    action = policy.select_action(env, actor_id)
    kind = _action_kind(env, actor_id, action)
    if isinstance(action, ManaDrawAction):
        _obs, _reward, terminated, truncated, info = env.step_core_action(action)
    else:
        _obs, _reward, terminated, truncated, info = env.step(int(action))
    return bool(terminated), bool(truncated), dict(info), kind


def _utility(winner_id: int | None, actor_id: int, status: Any) -> float:
    if winner_id is None:
        return 0.5 if status != GameStatus.ONGOING else 0.0
    return 1.0 if winner_id == actor_id else 0.0


def _inject_candidate(env: TrainV3ClassicEnv, actor_id: int, card_id: int) -> None:
    state = env.env._env.state
    actor = state.p1 if actor_id == 1 else state.p2
    index = next(index for index, card in enumerate(actor.deck) if int(card.card_id) == int(card_id))
    card = actor.deck.pop(index)
    card.skip_count = 0
    actor.hand.append(card)
    state.arena_engine = env.env._env
    env.env._cache.set_state(state, env.current_player_id())


def _counterfactual_row(
    env: TrainV3ClassicEnv,
    policy: V5NpzPolicy,
    *,
    battle_id: str,
    split: str,
    checkpoint_hash: str,
    branch_repeats: int,
    max_steps: int,
    branch_seed_base: int,
) -> dict[str, Any] | None:
    actor_id = env.current_player_id()
    state = env.env._env.state
    actor = state.p1 if actor_id == 1 else state.p2
    if len(actor.hand) >= HAND_CAP or len(actor.deck) < 2:
        return None
    candidate_ids = sorted({int(card.card_id) for card in actor.deck})
    if len(candidate_ids) > 6:
        rng = random.Random(_stable_int("candidate_sample", battle_id, state.turn_number, actor_id))
        candidate_ids = sorted(rng.sample(candidate_ids, 6))
    state_key = f"{battle_id}:turn-{state.turn_number}:actor-{actor_id}"
    scores: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        returns: list[float] = []
        errors: list[str] = []
        for repeat in range(branch_repeats):
            branch = copy.deepcopy(env)
            continuation_seed = _stable_int(branch_seed_base, state_key, repeat) % (2**31 - 1)
            branch.env._env._rng = random.Random(continuation_seed)
            branch.env._env.state.arena_engine = branch.env._env
            try:
                _inject_candidate(branch, actor_id, candidate_id)
                terminated = truncated = False
                steps = 0
                while not terminated and not truncated and steps < max_steps:
                    terminated, truncated, info, _kind = _step_policy(branch, policy)
                    if info.get("invalid_action"):
                        raise RuntimeError(f"invalid continuation action: {info.get('error')}")
                    steps += 1
                winner_id = branch.env.winner_id()
                status = branch.env._env.state.status
                returns.append(_utility(winner_id, actor_id, status))
            except Exception as exc:  # preserve branch failures instead of inventing labels
                errors.append(f"{exc.__class__.__name__}: {exc}")
        mean = sum(returns) / len(returns) if returns else None
        variance = (
            sum((value - mean) ** 2 for value in returns) / len(returns)
            if mean is not None and returns
            else None
        )
        scores.append(
            {
                "card_id": candidate_id,
                "expected_return": mean,
                "return_std": math.sqrt(variance) if variance is not None else None,
                "valid_repeats": len(returns),
                "errors": errors,
            }
        )
    valid = [row for row in scores if row["expected_return"] is not None]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda row: (-float(row["expected_return"]), int(row["card_id"])))
    best = valid[0]
    return {
        "schema": "extra_lr_cardoptimum_counterfactual_v1",
        "migration_from": "extra-sublr-desirerer-v1",
        "state_id": hashlib.sha256(state_key.encode()).hexdigest()[:24],
        "battle_id": battle_id,
        # Keep every state derived from the same allowed pool/matchup lineage
        # in one split. Splitting by battle_id leaks near-identical states from
        # a balanced matchup cell across train/validation/test.
        "split": split,
        "checkpoint_sha256": checkpoint_hash,
        "label_policy": "matched_full_rollout_card_injection",
        "branch_repeats": branch_repeats,
        "candidate_pool_ids": candidate_ids,
        "candidate_scores": scores,
        "best_card_id": int(best["card_id"]),
        "best_expected_return": float(best["expected_return"]),
        "state": _state_snapshot(env, actor_id),
    }


def _battle_spec(global_index: int, base_seed: int, catalog: CardCatalog) -> dict[str, Any]:
    cell_index, within_cell = divmod(global_index, BALANCED_GAMES_PER_CELL)
    repeat, permutation = divmod(within_cell, 4)
    candidate_seat = 1 if permutation < 2 else 2
    starting_player = 1 if permutation % 2 == 0 else 2
    cell = _cell_spec(cell_index, base_seed, catalog)
    seed = _stable_int("battle", base_seed, cell_index, repeat) % (2**31 - 1)
    return {
        **cell,
        "global_index": global_index,
        "battle_id": f"syn-{global_index:08d}",
        "repeat": repeat,
        "candidate_seat": candidate_seat,
        "starting_player_id": starting_player,
        "seed": seed,
    }


def _run_battle(
    spec: dict[str, Any],
    policy: V5NpzPolicy,
    *,
    checkpoint_hash: str,
    max_steps: int,
    max_turns: int,
    collect_counterfactual: bool,
    branch_repeats: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate_seat = int(spec["candidate_seat"])
    if candidate_seat == 1:
        p1_deck, p2_deck = spec["candidate_deck_ids"], spec["opponent_deck_ids"]
        p1_levels, p2_levels = spec["candidate_levels"], spec["opponent_levels"]
    else:
        p1_deck, p2_deck = spec["opponent_deck_ids"], spec["candidate_deck_ids"]
        p1_levels, p2_levels = spec["opponent_levels"], spec["candidate_levels"]
    env = _new_env(int(spec["seed"]), max_turns)
    env.reset(
        p1_deck_ids=p1_deck,
        p2_deck_ids=p2_deck,
        p1_levels=p1_levels,
        p2_levels=p2_levels,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=int(spec["starting_player_id"]),
        seed=int(spec["seed"]),
    )
    policy.reset(int(spec["seed"]))
    terminated = truncated = False
    steps = 0
    invalid = 0
    error: str | None = None
    action_counts: Counter[str] = Counter()
    counterfactual: dict[str, Any] | None = None
    cf_target_step = 5 + (_stable_int("cf_step", spec["battle_id"]) % 16)
    started = time.monotonic()
    try:
        while not terminated and not truncated and steps < max_steps:
            if collect_counterfactual and counterfactual is None and steps >= cf_target_step:
                counterfactual = _counterfactual_row(
                    env,
                    policy,
                    battle_id=str(spec["battle_id"]),
                    split=str(spec["split"]),
                    checkpoint_hash=checkpoint_hash,
                    branch_repeats=branch_repeats,
                    max_steps=max_steps,
                    branch_seed_base=int(spec["seed"]),
                )
            terminated, truncated, info, kind = _step_policy(env, policy)
            action_counts[kind] += 1
            invalid += int(bool(info.get("invalid_action")))
            steps += 1
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
    winner_id = env.env.winner_id()
    status = env.env._env.state.status
    candidate_won = winner_id == candidate_seat
    score = 1.0 if candidate_won else (0.5 if winner_id is None and status != GameStatus.ONGOING else 0.0)
    row = {
        "schema": "extra_lr_v5_selfplay_battle_v1",
        "battle_id": spec["battle_id"],
        "global_index": int(spec["global_index"]),
        "cell_id": spec["cell_id"],
        "pool_id": spec["pool_id"],
        "split": spec["split"],
        "allowed_pool_ids": spec["allowed_pool_ids"],
        "candidate_deck_ids": spec["candidate_deck_ids"],
        "opponent_deck_ids": spec["opponent_deck_ids"],
        "candidate_levels": spec["candidate_levels"],
        "opponent_levels": spec["opponent_levels"],
        "candidate_seat": candidate_seat,
        "starting_player_id": int(spec["starting_player_id"]),
        "seed": int(spec["seed"]),
        "repeat": int(spec["repeat"]),
        "winner_id": winner_id,
        "candidate_score": score,
        "steps": steps,
        "turns": int(env.env._env.state.turn_number),
        "status": getattr(status, "value", str(status)),
        "completed": bool(terminated and error is None),
        "truncated": bool(truncated or (not terminated and steps >= max_steps)),
        "invalid_actions": invalid,
        "action_counts": dict(action_counts),
        "mana_draw_count": int(action_counts["mana_draw"]),
        "simulation_wall_ms": round((time.monotonic() - started) * 1000, 3),
        "checkpoint_sha256": checkpoint_hash,
        "error": error,
    }
    return row, counterfactual


def _existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(int(json.loads(line)["global_index"]))
    return result


def generate(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = (ROOT / "ai" / "cards.json").resolve()
    catalog = load_catalog(catalog_path)
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = _sha256(checkpoint)
    policy = V5NpzPolicy(checkpoint)
    battles_path = output_dir / f"battles.shard-{args.shard_index:02d}.jsonl"
    cardopt_path = output_dir / f"cardoptimum.shard-{args.shard_index:02d}.jsonl"
    progress_path = output_dir / f"progress.shard-{args.shard_index:02d}.json"
    completed_indices = _existing_indices(battles_path)
    assigned = [
        index for index in range(args.total_battles)
        if index % args.shard_count == args.shard_index
    ]
    started_at = _utc_now()
    with battles_path.open("a", encoding="utf-8") as battles_file, cardopt_path.open("a", encoding="utf-8") as cardopt_file:
        for position, global_index in enumerate(assigned, start=1):
            if global_index in completed_indices:
                continue
            spec = _battle_spec(global_index, args.seed, catalog)
            collect_cf = args.cardoptimum_every > 0 and global_index % args.cardoptimum_every == 0
            battle, cardoptimum = _run_battle(
                spec,
                policy,
                checkpoint_hash=checkpoint_hash,
                max_steps=args.max_steps,
                max_turns=args.max_turns,
                collect_counterfactual=collect_cf,
                branch_repeats=args.branch_repeats,
            )
            battles_file.write(_json_dump(battle) + "\n")
            battles_file.flush()
            if cardoptimum is not None:
                cardopt_file.write(_json_dump(cardoptimum) + "\n")
                cardopt_file.flush()
            completed_indices.add(global_index)
            if len(completed_indices) % args.progress_every == 0 or position == len(assigned):
                payload = {
                    "schema": SCHEMA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "assigned_battles": len(assigned),
                    "completed_battles": len(completed_indices),
                    "progress": len(completed_indices) / max(1, len(assigned)),
                    "started_at": started_at,
                    "updated_at": _utc_now(),
                    "pid": os.getpid(),
                }
                _atomic_json(progress_path, payload)
                print(_json_dump(payload), flush=True)


def _wilson_interval(score_sum: float, count: int, z: float = 1.96) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    p = score_sum / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def merge(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    battles = _read_jsonl(output_dir.glob("battles.shard-*.jsonl"))
    cardoptimum = _read_jsonl(output_dir.glob("cardoptimum.shard-*.jsonl"))
    battles_by_id = {str(row["battle_id"]): row for row in battles}
    missing_lineage = [
        str(row["battle_id"])
        for row in cardoptimum
        if str(row["battle_id"]) not in battles_by_id
    ]
    if missing_lineage:
        raise ValueError(
            "CardOptimum rows reference missing source battles: "
            + ", ".join(missing_lineage[:10])
        )
    cardoptimum = [
        {
            **row,
            "split": battles_by_id[str(row["battle_id"])]["split"],
            "lineage_pool_id": battles_by_id[str(row["battle_id"])]["pool_id"],
            "lineage_cell_id": battles_by_id[str(row["battle_id"])]["cell_id"],
        }
        for row in cardoptimum
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in battles:
        by_cell[str(row["cell_id"])].append(row)
    assembler_rows: list[dict[str, Any]] = []
    for cell_id, rows in sorted(by_cell.items()):
        example = rows[0]
        usable = [
            row for row in rows
            if row["completed"] and not row["truncated"] and not row["error"] and row["invalid_actions"] == 0
        ]
        score_sum = sum(float(row["candidate_score"]) for row in usable)
        low, high = _wilson_interval(score_sum, len(usable))
        assembler_rows.append(
            {
                "schema": "extra_lr_assembler_matchup_v1",
                "migration_from": "extra-sublr-assembler-v1",
                "cell_id": cell_id,
                "pool_id": example["pool_id"],
                "split": example["split"],
                "allowed_pool_ids": example["allowed_pool_ids"],
                "candidate_deck_ids": example["candidate_deck_ids"],
                "opponent_deck_ids": example["opponent_deck_ids"],
                "candidate_levels": example["candidate_levels"],
                "opponent_levels": example["opponent_levels"],
                "expected_matchup_score": score_sum / len(usable) if usable else None,
                "score_ci95": [low, high],
                "usable_battles": len(usable),
                "scheduled_battles": len(rows),
                "paired_rng_repeats": len({row["repeat"] for row in rows}),
                "covers_both_seats": {row["candidate_seat"] for row in rows} == {1, 2},
                "covers_both_initiatives": {row["starting_player_id"] for row in rows} == {1, 2},
                "checkpoint_sha256": example["checkpoint_sha256"],
                "label_policy": "paired_seed_side_and_initiative_balanced_full_outcome",
            }
        )
    timestamp_rows = [
        {
            "schema": "extra_lr_timestamp_simulation_v1",
            "battle_id": row["battle_id"],
            "split": row["split"],
            "mono_deck_ids": row["candidate_deck_ids"],
            "opponent_deck_ids": row["opponent_deck_ids"],
            "mono_levels": row["candidate_levels"],
            "opponent_levels": row["opponent_levels"],
            "starting_player_relative": "first" if row["candidate_seat"] == row["starting_player_id"] else "second",
            "turns": row["turns"],
            "actions": row["steps"],
            "duration_seconds": None,
            "wall_clock_label_available": False,
            "completed": row["completed"],
            "censored": bool(row["truncated"] or row["error"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "label_policy": "simulator_turns_and_actions_only",
        }
        for row in battles
    ]
    outputs = {
        "assembler_matchups.jsonl": assembler_rows,
        "cardoptimum_counterfactual.jsonl": cardoptimum,
        "timestamp_simulations.jsonl": timestamp_rows,
    }
    for filename, rows in outputs.items():
        path = output_dir / filename
        path.write_text("".join(_json_dump(row) + "\n" for row in rows), encoding="utf-8")
    checkpoint = args.checkpoint.resolve()
    catalog = (ROOT / "ai" / "cards.json").resolve()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = None
    split_counts = Counter(row["split"] for row in battles)
    candidate_first = [
        float(row["candidate_score"])
        for row in battles
        if row["candidate_seat"] == row["starting_player_id"]
    ]
    candidate_second = [
        float(row["candidate_score"])
        for row in battles
        if row["candidate_seat"] != row["starting_player_id"]
    ]
    candidate_p1 = [
        float(row["candidate_score"]) for row in battles if row["candidate_seat"] == 1
    ]
    candidate_p2 = [
        float(row["candidate_score"]) for row in battles if row["candidate_seat"] == 2
    ]
    informative_cardoptimum = sum(
        len({
            score["expected_return"]
            for score in row["candidate_scores"]
            if score["expected_return"] is not None
        }) > 1
        for row in cardoptimum
    )
    covered_pool_cards = {
        int(card_id) for row in battles for card_id in row["allowed_pool_ids"]
    }
    covered_deck_cards = {
        int(card_id)
        for row in battles
        for card_id in [*row["candidate_deck_ids"], *row["opponent_deck_ids"]]
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_at": _utc_now(),
        "git_commit": git_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "catalog": str(catalog),
        "catalog_sha256": _sha256(catalog),
        "ruleset": "python TrainV3ClassicEnv/current production mechanics",
        "total_battles": len(battles),
        "complete_clean_battles": sum(
            row["completed"] and not row["truncated"] and not row["error"] and row["invalid_actions"] == 0
            for row in battles
        ),
        "assembler_matchup_cells": len(assembler_rows),
        "cardoptimum_states": len(cardoptimum),
        "timestamp_rows": len(timestamp_rows),
        "metronome_rows": 0,
        "metronome_policy": "human_only_no_synthetic_labels",
        "split_policy": "compositional hash; Assembler by allowed pool, battle-derived rows by lineage",
        "split_counts": dict(split_counts),
        "balancing": "5 paired RNG seeds x both candidate seats x both starting players per matchup cell",
        "quality": {
            "error_battles": sum(bool(row["error"]) for row in battles),
            "truncated_battles": sum(bool(row["truncated"]) for row in battles),
            "invalid_action_battles": sum(int(row["invalid_actions"]) > 0 for row in battles),
            "unique_pools": len({row["pool_id"] for row in battles}),
            "unique_candidate_decks": len({
                tuple(row["candidate_deck_ids"]) for row in battles
            }),
            "unique_opponent_decks": len({
                tuple(row["opponent_deck_ids"]) for row in battles
            }),
            "covered_pool_card_ids": len(covered_pool_cards),
            "covered_deck_card_ids": len(covered_deck_cards),
            "candidate_first_score_rate": (
                sum(candidate_first) / len(candidate_first) if candidate_first else None
            ),
            "candidate_second_score_rate": (
                sum(candidate_second) / len(candidate_second) if candidate_second else None
            ),
            "candidate_p1_score_rate": (
                sum(candidate_p1) / len(candidate_p1) if candidate_p1 else None
            ),
            "candidate_p2_score_rate": (
                sum(candidate_p2) / len(candidate_p2) if candidate_p2 else None
            ),
            "assembler_min_usable_battles_per_cell": min(
                (row["usable_battles"] for row in assembler_rows), default=0
            ),
            "assembler_fully_balanced_cells": sum(
                row["covers_both_seats"] and row["covers_both_initiatives"]
                for row in assembler_rows
            ),
            "cardoptimum_informative_states": informative_cardoptimum,
            "cardoptimum_informative_fraction": (
                informative_cardoptimum / len(cardoptimum) if cardoptimum else 0.0
            ),
            "cardoptimum_branch_errors": sum(
                len(score["errors"])
                for row in cardoptimum
                for score in row["candidate_scores"]
            ),
        },
        "readiness": {
            "assembler": "ready_for_bootstrap_training",
            "timestamp": "ready_for_simulator_pretraining; human seconds calibration still required",
            "cardoptimum": (
                "candidate_ready"
                if informative_cardoptimum >= 1000
                else "bootstrap_only; expand informative counterfactual states before final training"
            ),
            "metronome": "awaiting human-only timing rows",
        },
        "files": {},
    }
    for filename in outputs:
        path = output_dir / filename
        manifest["files"][filename] = {"sha256": _sha256(path), "rows": len(outputs[filename])}
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-battles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=5_052_027)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--cardoptimum-every", type=int, default=10)
    parser.add_argument("--branch-repeats", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.total_battles <= 0:
        parser.error("--total-battles must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard configuration")
    if args.branch_repeats <= 0:
        parser.error("--branch-repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.merge:
        merge(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
