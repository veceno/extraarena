#!/usr/bin/env python3
"""Train the four confirmed ExtraLR V1 auxiliary models.

The models are deliberately small, auditable ridge regressors.  They are
bootstrap artifacts, not replacements for the no-assist V5 policy:

* Assembler scores a candidate deck against an opponent deck and allowed pool.
* CardOptimum scores every card in a matched counterfactual candidate set.
* Metronome predicts log human decision latency from human-visible state only.
* TimeStamp predicts turns from simulator data and calibrates wall-clock
  duration on completed human battles (Mono plus an optional Duo candidate).

Every split is lineage/group based.  Synthetic timing is never used for
Metronome, rejected actions are never labels, and the TimeStamp wall-clock
calibrator excludes obvious background-idle battles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from train_v3.aux_inference import metronome_features_from_trace


ROOT = Path(__file__).resolve().parents[2]
CARD_CATALOG = {
    int(card["id"]): card
    for card in json.loads((ROOT / "ai" / "cards.json").read_text())
}
CARD_IDS = tuple(
    sorted(CARD_CATALOG)
)
CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
ACTION_TYPES = ("attack", "play_card", "mana_draw", "end_turn")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(key: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _balanced_group_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Deterministically allocate whole groups while keeping row counts useful."""
    counts = Counter(str(row["group_id"]) for row in rows)
    target = max(1, round(sum(counts.values()) * 0.10))
    ordered = sorted(
        counts,
        key=lambda group_id: hashlib.sha256(
            f"aux-split:{group_id}".encode()
        ).digest(),
    )
    result: dict[str, str] = {}
    test_rows = validation_rows = 0
    for group_id in ordered:
        if test_rows < target:
            result[group_id] = "test"
            test_rows += counts[group_id]
        elif validation_rows < target:
            result[group_id] = "validation"
            validation_rows += counts[group_id]
        else:
            result[group_id] = "train"
    return result


def _quantile(values: Iterable[float], q: float) -> float:
    materialized = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(materialized, q)) if materialized.size else 0.0


def _deck_vector(
    deck_ids: Iterable[int], levels: dict[str, Any] | dict[int, Any] | None = None
) -> np.ndarray:
    out = np.zeros(len(CARD_IDS) * 2, dtype=np.float64)
    levels = levels or {}
    for card_id_raw in deck_ids:
        card_id = int(card_id_raw)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        out[index] += 1.0
        level = levels.get(str(card_id), levels.get(card_id, 1))
        out[len(CARD_IDS) + index] = max(out[len(CARD_IDS) + index], float(level) / 10.0)
    return out


def _pool_vector(card_ids: Iterable[int]) -> np.ndarray:
    out = np.zeros(len(CARD_IDS), dtype=np.float64)
    for card_id_raw in card_ids:
        index = CARD_INDEX.get(int(card_id_raw))
        if index is not None:
            out[index] = 1.0
    return out


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    weights = (
        np.ones(z.shape[0], dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    weighted = design * np.sqrt(weights)[:, None]
    target = y * np.sqrt(weights)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ target)
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "intercept": np.asarray([coef[0]], dtype=np.float32),
        "coef": coef[1:].astype(np.float32),
    }


def _predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (
        np.asarray(x, dtype=np.float64) - model["feature_mean"].astype(np.float64)
    ) / model["feature_scale"].astype(np.float64)
    return z @ model["coef"].astype(np.float64) + float(model["intercept"][0])


def _save_model(
    out_dir: Path,
    name: str,
    model: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    np.savez_compressed(npz_path, **model)
    payload = {
        **manifest,
        "artifact": str(npz_path),
        "artifact_sha256": _sha256(npz_path),
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return npz_path, json_path


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(pred) - np.asarray(y)
    return {
        "rows": int(len(y)),
        "mae": float(np.mean(np.abs(error))) if len(y) else 0.0,
        "median_ae": float(np.median(np.abs(error))) if len(y) else 0.0,
        "rmse": float(np.sqrt(np.mean(error**2))) if len(y) else 0.0,
    }


def _assembler_features(row: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            _deck_vector(row["candidate_deck_ids"], row.get("candidate_levels")),
            _deck_vector(row["opponent_deck_ids"], row.get("opponent_levels")),
            _pool_vector(row["allowed_pool_ids"]),
            np.asarray(
                [
                    len(set(row["candidate_deck_ids"])) / 9.0,
                    len(set(row["opponent_deck_ids"])) / 9.0,
                    float(row.get("usable_battles", 0)) / 20.0,
                ]
            ),
        ]
    )


def _train_assembler(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    x = np.stack([_assembler_features(row) for row in rows])
    y = np.asarray([row["expected_matchup_score"] for row in rows], dtype=np.float64)
    split = np.asarray([row["split"] for row in rows])
    train = split == "train"
    test = split == "test"
    weight = np.asarray(
        [max(0.1, float(row.get("usable_battles", 1)) / 20.0) for row in rows]
    )
    model = _fit_ridge(x[train], y[train], alpha=18.0, sample_weight=weight[train])
    pred = np.clip(_predict(model, x[test]), 0.0, 1.0)
    metrics = _regression_metrics(y[test], pred)
    metrics["baseline_mae"] = float(np.mean(np.abs(y[test] - y[train].mean())))
    readiness = (
        "bootstrap_ready"
        if metrics["rows"] >= 40 and metrics["mae"] < metrics["baseline_mae"]
        else "research_only"
    )
    _save_model(
        out_dir,
        "extra_lr_assembler_v1",
        model,
        {
            "schema": "extra_lr_aux_model_v1",
            "model": "ExtraLR Assembler V1",
            "target": "paired_seed_expected_matchup_score",
            "feature_schema": "candidate_deck_bag+levels,opponent_deck_bag+levels,allowed_pool",
            "split": "provided compositional pool split",
            "metrics": metrics,
            "readiness": readiness,
        },
    )
    return {"metrics": metrics, "readiness": readiness}


def _state_scalars(state: dict[str, Any]) -> np.ndarray:
    actor = state.get("actor") or {}
    opponent = state.get("opponent") or {}

    def board_stats(player: dict[str, Any]) -> tuple[float, float, float, float]:
        board = player.get("board") or []
        return (
            len(board) / 5.0,
            sum(float(card.get("attack", 0)) for card in board) / 50.0,
            sum(float(card.get("hp", 0)) for card in board) / 50.0,
            sum(bool(card.get("ready")) for card in board) / 5.0,
        )

    actor_hero = actor.get("hero") or {}
    opponent_hero = opponent.get("hero") or {}
    actor_board = board_stats(actor)
    opponent_board = board_stats(opponent)
    return np.asarray(
        [
            math.log1p(float(state.get("turn_number", 0))) / 4.0,
            float(actor_hero.get("hp", 0)) / max(float(actor_hero.get("max_hp", 1)), 1.0),
            float(opponent_hero.get("hp", 0)) / max(float(opponent_hero.get("max_hp", 1)), 1.0),
            float(actor.get("mana", 0)) / 10.0,
            float(actor.get("max_mana", 0)) / 10.0,
            float(opponent.get("mana", 0)) / 10.0,
            float(opponent.get("max_mana", 0)) / 10.0,
            float(actor.get("hand_count", len(actor.get("hand") or []))) / 4.0,
            float(opponent.get("hand_count", 0)) / 4.0,
            float(actor.get("deck_count", len(actor.get("remaining_deck") or []))) / 9.0,
            float(opponent.get("deck_count", 0)) / 9.0,
            *actor_board,
            *opponent_board,
            float(actor.get("mana_draw_count_this_turn", 0)) / 2.0,
        ],
        dtype=np.float64,
    )


def _cardopt_features(state: dict[str, Any], score: dict[str, Any]) -> np.ndarray:
    card_id = int(score["card_id"])
    one_hot = np.zeros(len(CARD_IDS), dtype=np.float64)
    if card_id in CARD_INDEX:
        one_hot[CARD_INDEX[card_id]] = 1.0
    actor = state.get("actor") or {}
    card = next(
        (
            card
            for card in actor.get("remaining_deck") or []
            if int(card.get("card_id", -1)) == card_id
        ),
        {},
    )
    stats = np.asarray(
        [
            float(card.get("level", 1)) / 10.0,
            float(card.get("attack", 0)) / 30.0,
            float(card.get("hp", 0)) / 30.0,
            float(card.get("mana_cost", 0)) / 10.0,
            float(card.get("skip_count", 0)) / 3.0,
        ]
    )
    base = _state_scalars(state)
    return np.concatenate([base, one_hot, stats, base[:7] * stats.mean()])


def _train_cardoptimum(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    examples: list[tuple[str, str, np.ndarray, float, float]] = []
    informative_states: set[str] = set()
    for row in rows:
        scores = row.get("candidate_scores") or []
        values = [float(score["expected_return"]) for score in scores]
        informative = bool(values and max(values) - min(values) > 1.0e-8)
        if informative:
            informative_states.add(row["state_id"])
        for score in scores:
            examples.append(
                (
                    row["state_id"],
                    row["split"],
                    _cardopt_features(row["state"], score),
                    float(score["expected_return"]),
                    1.0 if informative else 0.2,
                )
            )
    x = np.stack([row[2] for row in examples])
    y = np.asarray([row[3] for row in examples])
    split = np.asarray([row[1] for row in examples])
    weight = np.asarray([row[4] for row in examples])
    train = split == "train"
    model = _fit_ridge(x[train], y[train], alpha=12.0, sample_weight=weight[train])

    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        if row[1] == "test" and row[0] in informative_states:
            by_state[row[0]].append(index)
    regrets: list[float] = []
    top1 = 0
    for indices in by_state.values():
        truth = y[indices]
        pred = _predict(model, x[indices])
        choice = int(np.argmax(pred))
        regrets.append(float(np.max(truth) - truth[choice]))
        top1 += int(truth[choice] >= np.max(truth) - 1.0e-8)
    metrics = {
        "counterfactual_states": len(rows),
        "informative_states": len(informative_states),
        "test_informative_states": len(by_state),
        "top1_accuracy": top1 / len(by_state) if by_state else 0.0,
        "mean_ranking_regret": float(np.mean(regrets)) if regrets else 0.0,
        "p90_ranking_regret": _quantile(regrets, 0.9),
    }
    readiness = "bootstrap_only" if len(informative_states) < 1000 else "candidate_ready"
    _save_model(
        out_dir,
        "extra_lr_cardoptimum_v1",
        model,
        {
            "schema": "extra_lr_aux_model_v1",
            "model": "ExtraLR CardOptimum V1",
            "target": "matched_counterfactual_expected_return",
            "feature_schema": "human_visible_state+candidate_card_identity_and_stats",
            "split": "state lineage split",
            "metrics": metrics,
            "readiness": readiness,
        },
    )
    return {"metrics": metrics, "readiness": readiness}


def _human_action_features(row: dict[str, Any]) -> np.ndarray:
    return metronome_features_from_trace(row)


def _extract_human_rows(
    freeze_dir: Path, selection: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timing_rows: list[dict[str, Any]] = []
    battle_rows: list[dict[str, Any]] = []
    for selected in selection["battle_rows"]:
        group_id, battle_id = selected["group_id"], selected["battle_id"]
        group_dir = freeze_dir / "sessions" / group_id
        meta = json.loads(
            (group_dir / "battles" / battle_id / "v5" / "meta.json").read_text()
        )
        log = json.loads((group_dir / "battles" / f"{battle_id}.json").read_text())
        trace_path = group_dir / "battles" / battle_id / "v5" / "actions.jsonl"
        trace_rows = _read_jsonl(trace_path)
        for row in trace_rows:
            latency = row.get("human_decision_time_ms")
            if (
                row.get("decision_source") == "human"
                and row.get("accepted") is True
                and isinstance(latency, (int, float))
            ):
                timing_rows.append(
                    {
                        "group_id": group_id,
                        "battle_id": battle_id,
                        "action_type": row.get("action_type"),
                        "latency_ms": float(latency),
                        "idle_or_censored": not (100.0 <= float(latency) <= 25000.0),
                        "features": _human_action_features(row).tolist(),
                    }
                )
        first_pre = (trace_rows[0].get("pre_state") or {}) if trace_rows else {}
        starting_uid = first_pre.get("current_turn_owner_id")
        p1_uid = meta.get("p1_user_id")
        # meta spans the user-visible battle interval and is therefore the
        # TimeStamp wall-clock target.  The compact battle log may subtract a
        # server restart/background gap; retain that as an audit field instead
        # of silently replacing the wall-clock label.
        duration = float(meta.get("duration_seconds") or log.get("duration_seconds") or 0.0)
        battle_rows.append(
            {
                "group_id": group_id,
                "battle_id": battle_id,
                "mono_deck_ids": [int(card["card_id"]) for card in meta["p1_deck"]],
                "mono_levels": {str(card["card_id"]): int(card["level"]) for card in meta["p1_deck"]},
                "opponent_deck_ids": [int(card["card_id"]) for card in meta["p2_deck"]],
                "opponent_levels": {str(card["card_id"]): int(card["level"]) for card in meta["p2_deck"]},
                "starting_player_relative": "first" if starting_uid == p1_uid else "second",
                "turns": int(meta.get("turns") or 0),
                "duration_seconds": duration,
                "active_log_duration_seconds": float(log.get("duration_seconds") or 0.0),
                "completed": meta.get("status") in {"p1_win", "p2_win", "draw"},
                "idle_or_censored": not (10.0 <= duration <= 1800.0),
            }
        )
    return timing_rows, battle_rows


def _train_metronome(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    usable = [row for row in rows if not row["idle_or_censored"]]
    x = np.asarray([row["features"] for row in usable], dtype=np.float64)
    y = np.log1p(np.asarray([row["latency_ms"] for row in usable], dtype=np.float64))
    split_map = _balanced_group_splits(usable)
    split = np.asarray([split_map[row["group_id"]] for row in usable])
    train = split == "train"
    test = split == "test"
    model = _fit_ridge(x[train], y[train], alpha=22.0)
    pred_log = _predict(model, x[test])
    pred_ms = np.expm1(pred_log)
    true_ms = np.expm1(y[test])
    residual = y[train] - _predict(model, x[train])
    p50_shift = _quantile(residual, 0.5)
    p90_shift = _quantile(residual, 0.9)
    pred_p50 = np.expm1(pred_log + p50_shift)
    pred_p90 = np.expm1(pred_log + p90_shift)
    baseline_ms = float(np.median(np.expm1(y[train])))
    metrics = {
        **_regression_metrics(true_ms, pred_ms),
        "baseline_median_mae": float(np.mean(np.abs(true_ms - baseline_ms))),
        "log_mae": float(np.mean(np.abs(pred_log - y[test]))) if np.any(test) else 0.0,
        "p50_coverage": float(np.mean(true_ms <= pred_p50)) if np.any(test) else 0.0,
        "p90_coverage": float(np.mean(true_ms <= pred_p90)) if np.any(test) else 0.0,
        "raw_human_rows": len(rows),
        "usable_human_rows": len(usable),
        "censored_or_idle_rows": len(rows) - len(usable),
        "groups": len({row["group_id"] for row in usable}),
        "participant_key_available": False,
    }
    model["residual_log_quantiles"] = np.asarray(
        [p50_shift, p90_shift, _quantile(residual, 0.99)], dtype=np.float32
    )
    readiness = "candidate_ready" if len(usable) >= 5000 and metrics["p90_coverage"] >= 0.75 else "calibration_needed"
    _save_model(
        out_dir,
        "extra_lr_metronome_v1",
        model,
        {
            "schema": "extra_lr_aux_model_v1",
            "model": "ExtraLR Metronome V1",
            "target": "log1p(human_decision_time_ms)",
            "feature_schema": "human_visible_state+legal_complexity+action_context",
            "split": "group/session proxy; participant key unavailable",
            "safety_sampling": "sample log-time distribution; clamp only below 100ms and above 25s",
            "metrics": metrics,
            "readiness": readiness,
        },
    )
    return {"metrics": metrics, "readiness": readiness}


def _timestamp_features(row: dict[str, Any], *, duo: bool) -> np.ndarray:
    parts = [
        _deck_vector(row["mono_deck_ids"], row.get("mono_levels")),
        np.asarray([float(row.get("starting_player_relative") == "first")]),
    ]
    if duo:
        parts.append(_deck_vector(row["opponent_deck_ids"], row.get("opponent_levels")))
    return np.concatenate(parts)


def _deck_summary(
    deck_ids: Iterable[int], levels: dict[str, Any] | dict[int, Any] | None
) -> np.ndarray:
    ids = [int(card_id) for card_id in deck_ids]
    levels = levels or {}
    cards = [CARD_CATALOG[card_id] for card_id in ids if card_id in CARD_CATALOG]
    nonheroes = [card for card in cards if card.get("card_type") != "hero"]
    level_values = np.asarray(
        [float(levels.get(str(card_id), levels.get(card_id, 1))) for card_id in ids],
        dtype=np.float64,
    )
    mana = np.asarray([float(card.get("mana_cost", 0)) for card in nonheroes])
    attack = np.asarray([float(card.get("base_attack", 0)) for card in nonheroes])
    hp = np.asarray([float(card.get("base_hp", 0)) for card in nonheroes])
    hero_hp = next(
        (
            float(card.get("base_hp", 0))
            for card in cards
            if card.get("card_type") == "hero"
        ),
        0.0,
    )
    return np.asarray(
        [
            hero_hp / 50.0,
            float(np.mean(level_values)) / 10.0,
            float(np.std(level_values)) / 5.0,
            float(np.max(level_values)) / 10.0,
            float(np.mean(mana)) / 10.0,
            float(np.std(mana)) / 5.0,
            float(np.mean(attack)) / 20.0,
            float(np.mean(hp)) / 20.0,
            sum(card.get("card_type") == "potion" for card in cards) / 3.0,
        ],
        dtype=np.float64,
    )


def _duration_calibration_features(
    row: dict[str, Any],
    predicted_log_turns: float,
    *,
    duo: bool,
    include_deck_summary: bool,
) -> np.ndarray:
    parts = [
        np.asarray(
            [
                float(predicted_log_turns),
                float(row.get("starting_player_relative") == "first"),
            ]
        )
    ]
    if include_deck_summary:
        parts.append(
            _deck_summary(row["mono_deck_ids"], row.get("mono_levels"))
        )
        if duo:
            parts.append(
                _deck_summary(
                    row["opponent_deck_ids"], row.get("opponent_levels")
                )
            )
    return np.concatenate(parts)


def _train_timestamp(
    simulator_rows: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    simulator = [row for row in simulator_rows if row.get("completed") and not row.get("censored")]
    results: dict[str, Any] = {}
    mono_test_mae = float("inf")
    for duo in (False, True):
        label = "duo" if duo else "mono"
        xs = np.stack([_timestamp_features(row, duo=duo) for row in simulator])
        ys = np.log1p(np.asarray([row["turns"] for row in simulator], dtype=np.float64))
        splits = np.asarray([row["split"] for row in simulator])
        train = splits == "train"
        test = splits == "test"
        turn_model = _fit_ridge(xs[train], ys[train], alpha=35.0)
        turn_pred = np.expm1(_predict(turn_model, xs[test]))
        turn_true = np.expm1(ys[test])
        turn_metrics = _regression_metrics(turn_true, turn_pred)
        turn_metrics["baseline_median_mae"] = float(
            np.mean(np.abs(turn_true - np.median(np.expm1(ys[train]))))
        )

        usable_human = [
            row for row in human_rows if row["completed"] and not row["idle_or_censored"]
        ]
        hx = np.stack([_timestamp_features(row, duo=duo) for row in usable_human])
        hturn_pred = _predict(turn_model, hx)
        hy = np.log1p(
            np.asarray([row["duration_seconds"] for row in usable_human], dtype=np.float64)
        )
        human_split_map = _balanced_group_splits(usable_human)
        hsplits = np.asarray(
            [human_split_map[row["group_id"]] for row in usable_human]
        )
        htrain = hsplits == "train"
        hvalidation = hsplits == "validation"
        htest = hsplits == "test"
        candidates: list[tuple[float, bool, float, np.ndarray]] = []
        for include_summary in (False, True):
            candidate_x = np.stack(
                [
                    _duration_calibration_features(
                        row,
                        predicted,
                        duo=duo,
                        include_deck_summary=include_summary,
                    )
                    for row, predicted in zip(
                        usable_human, hturn_pred, strict=True
                    )
                ]
            )
            for alpha in (1.0, 10.0, 50.0, 200.0, 1000.0):
                candidate_model = _fit_ridge(
                    candidate_x[htrain], hy[htrain], alpha=alpha
                )
                validation_pred = np.expm1(
                    _predict(candidate_model, candidate_x[hvalidation])
                )
                validation_true = np.expm1(hy[hvalidation])
                validation_mae = float(
                    np.mean(np.abs(validation_true - validation_pred))
                )
                candidates.append(
                    (validation_mae, include_summary, alpha, candidate_x)
                )
        validation_mae, include_summary, duration_alpha, calibration_x = min(
            candidates, key=lambda item: item[0]
        )
        calibration_fit = ~htest
        duration_model = _fit_ridge(
            calibration_x[calibration_fit],
            hy[calibration_fit],
            alpha=duration_alpha,
        )
        duration_pred_log = _predict(duration_model, calibration_x[htest])
        duration_pred = np.expm1(duration_pred_log)
        duration_true = np.expm1(hy[htest])
        residual = hy[calibration_fit] - _predict(
            duration_model, calibration_x[calibration_fit]
        )
        q50, q90 = _quantile(residual, 0.5), _quantile(residual, 0.9)
        duration_metrics = {
            **_regression_metrics(duration_true, duration_pred),
            "baseline_median_mae": float(
                np.mean(
                    np.abs(
                            duration_true
                        - np.median(np.expm1(hy[calibration_fit]))
                    )
                )
            ),
            "p50_coverage": float(
                np.mean(duration_true <= np.expm1(duration_pred_log + q50))
            ),
            "p90_coverage": float(
                np.mean(duration_true <= np.expm1(duration_pred_log + q90))
            ),
            "human_rows": len(usable_human),
            "human_censored_or_idle": len(human_rows) - len(usable_human),
            "selected_validation_mae": validation_mae,
            "selected_alpha": duration_alpha,
            "selected_features": (
                "predicted_turns+start+deck_summary"
                if include_summary
                else "predicted_turns+start"
            ),
        }
        if not duo:
            mono_test_mae = duration_metrics["mae"]
        beats_baseline = (
            duration_metrics["mae"] < duration_metrics["baseline_median_mae"]
        )
        ship = (
            duration_metrics["rows"] >= 20
            and beats_baseline
            and (not duo or duration_metrics["mae"] <= mono_test_mae * 1.05)
        )
        combined = {
            **{f"turn_{key}": value for key, value in turn_model.items()},
            **{f"duration_{key}": value for key, value in duration_model.items()},
            "duration_residual_log_quantiles": np.asarray(
                [q50, q90, _quantile(residual, 0.99)], dtype=np.float32
            ),
        }
        _save_model(
            out_dir,
            f"extra_lr_timestamp_v1_{label}",
            combined,
            {
                "schema": "extra_lr_aux_model_v1",
                "model": f"ExtraLR TimeStamp V1 {label.title()}",
                "targets": ["simulator_turns", "human_wall_clock_seconds"],
                "feature_schema": (
                    "turn_model(both_decks+levels+starting_player); "
                    f"duration_calibrator({duration_metrics['selected_features']})"
                    if duo
                    else "turn_model(user_deck+levels+starting_player+opponent_population=v5); "
                    f"duration_calibrator({duration_metrics['selected_features']})"
                ),
                "split": "simulator lineage plus human group split",
                "turn_metrics": turn_metrics,
                "duration_metrics": duration_metrics,
                "readiness": "candidate_ready" if ship else "optional_not_shipped",
            },
        )
        results[label] = {
            "turn_metrics": turn_metrics,
            "duration_metrics": duration_metrics,
            "readiness": "candidate_ready" if ship else "optional_not_shipped",
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-data", type=Path, required=True)
    parser.add_argument("--human-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    aux = args.aux_data.resolve()
    freeze = args.human_freeze.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets_dir = output / "datasets"
    models_dir = output / "models"

    source_manifest = json.loads((aux / "dataset_manifest.json").read_text())
    selection = json.loads((freeze / "selection_manifest.json").read_text())
    assembler_rows = _read_jsonl(aux / "assembler_matchups.jsonl")
    cardoptimum_rows = _read_jsonl(aux / "cardoptimum_counterfactual.jsonl")
    timestamp_rows = _read_jsonl(aux / "timestamp_simulations.jsonl")
    timing_rows, human_battles = _extract_human_rows(freeze, selection)
    _write_jsonl(datasets_dir / "metronome_human.jsonl", timing_rows)
    _write_jsonl(datasets_dir / "timestamp_human.jsonl", human_battles)

    summary = {
        "schema": "extra_lr_aux_training_run_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            "aux_manifest": str(aux / "dataset_manifest.json"),
            "aux_manifest_sha256": _sha256(aux / "dataset_manifest.json"),
            "human_selection": str(freeze / "selection_manifest.json"),
            "human_selection_sha256": _sha256(freeze / "selection_manifest.json"),
            "target_checkpoint_sha256": source_manifest["checkpoint_sha256"],
        },
        "datasets": {
            "assembler_rows": len(assembler_rows),
            "cardoptimum_states": len(cardoptimum_rows),
            "metronome_rows": len(timing_rows),
            "timestamp_simulator_rows": len(timestamp_rows),
            "timestamp_human_rows": len(human_battles),
        },
        "models": {
            "assembler": _train_assembler(assembler_rows, models_dir),
            "cardoptimum": _train_cardoptimum(cardoptimum_rows, models_dir),
            "metronome": _train_metronome(timing_rows, models_dir),
            "timestamp": _train_timestamp(timestamp_rows, human_battles, models_dir),
        },
    }
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    (output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
