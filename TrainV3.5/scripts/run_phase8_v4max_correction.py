#!/usr/bin/env python3
"""Run Extra-LR V5 Phase 8 V4-max correction training.

This phase is intentionally benchmark-driven: V4.1 is excluded, trace scenarios
are built around V4-max deck distributions, and the run metadata carries the
external 75% V4-max score target.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3" / "python"))

from ai.train_v2.classic_rl_env import ClassicRLEnv
from train_v3.aux_models import AssemblerCandidate, DeckMatchupEvaluator, load_assembler_dataset
from train_v3.league_v5 import V5LeagueConfig
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
from train_v3.trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool
from train_v3.v5_artifacts import read_manifest_json, write_manifest_json
from train_v3.v5_policy import create_v5_policy


DEFAULT_RESUME = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase4_mixed_assist_private_refresh_after_handicap_20260606_145409"
    / "checkpoints"
    / "trainv3_rust_legal_update_1000.npz"
)
DEFAULT_ASSEMBLER_DATASET = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase5_aux_models_from_phase4_20260606_231349"
    / "aux"
    / "assembler.jsonl"
)
HERO_CARD_IDS = frozenset({1, 3, 4, 5, 6, 7})


def main() -> int:
    run_name = _env_str("PHASE8_RUN_NAME", "phase8_v4max_correction_s1_assist")
    env_count = _env_int("PHASE8_ENV_COUNT", 8192)
    steps_per_update = _env_int("PHASE8_STEPS_PER_UPDATE", 32)
    updates = _env_int("PHASE8_UPDATES", 1200)
    minibatch_size = _env_int("PHASE8_MINIBATCH_SIZE", 8192)
    checkpoint_every = _env_int("PHASE8_CHECKPOINT_EVERY", 100)
    hidden_dim = _env_int("PHASE8_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE8_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE8_LR", 0.00018)
    entropy_coef = _env_float("PHASE8_ENTROPY_COEF", 0.024)
    max_grad_norm = _env_optional_float("PHASE8_MAX_GRAD_NORM", 0.5)
    seed = _env_int("PHASE8_SEED", 88001)
    trace_seed_count = _env_int("PHASE8_TRACE_SEED_COUNT", 96)
    resume_checkpoint = _env_path("PHASE8_RESUME_CHECKPOINT", DEFAULT_RESUME)
    assembler_dataset = _env_path("PHASE8_ASSEMBLER_DATASET", DEFAULT_ASSEMBLER_DATASET)
    target_v4max_score = _env_float("PHASE8_TARGET_V4MAX_SCORE", 0.75)
    out_root = Path(os.environ.get("PHASE8_OUT_ROOT", ROOT / "TrainV3" / "runs")).resolve()
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    trace_dir = run_dir / "trace_pool"
    manifest_path = run_dir / "trace_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase8_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    v4_deck_bank = _build_v4max_deck_bank(seed=seed, count=trace_seed_count)
    counter_decks = _build_counter_decks(v4_deck_bank, assembler_dataset=assembler_dataset)
    config_snapshot = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "updates": updates,
        "minibatch_size": minibatch_size,
        "checkpoint_every": checkpoint_every,
        "hidden_dim": hidden_dim,
        "action_hidden_dim": action_hidden_dim,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "max_grad_norm": max_grad_norm,
        "adaptive_strengths": [1.0],
        "seed": seed,
        "trace_seed_count": trace_seed_count,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "assembler_dataset": str(assembler_dataset) if assembler_dataset is not None else None,
        "target_v4max_score": target_v4max_score,
        "v4max_model_path": str(ROOT / "ai" / "models" / "extra-lr-v4-max.onnx"),
        "v4_1_included": False,
        "trace_policy": "v4max_deck_bank_full_info_assist_s1",
        "v4_deck_bank_size": len(v4_deck_bank),
        "counter_deck_count": len(counter_decks),
        "counter_decks": counter_decks[:16],
        "library_path": str(library_path),
    }
    (run_dir / "phase8_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE8_RUN_DIR", run_dir, flush=True)
    print("PHASE8_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    manifest = generate_v5_trace_pool(
        _phase8_scenarios(
            v4_deck_bank=v4_deck_bank,
            counter_decks=counter_decks,
            seed=seed,
        ),
        trace_dir,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_manifest_json(manifest, manifest_path)
    loaded_manifest = read_manifest_json(manifest_path)
    print(
        "PHASE8_TRACE_MANIFEST",
        manifest_path,
        loaded_manifest["manifest_id"],
        "traces",
        len(loaded_manifest["traces"]),
        flush=True,
    )

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    resume_metadata: dict[str, Any] = {}
    optimizer_restored = False
    if resume_checkpoint is not None:
        from ai.train_v2.model_mlx import load_checkpoint

        loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=optimizer)
        resume_metadata = dict(loaded.get("metadata", {}))
        optimizer_restored = bool(loaded.get("optimizer_restored", False))
        print(
            "PHASE8_RESUME",
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "optimizer_restored": optimizer_restored,
                    "source_update": resume_metadata.get("update"),
                    "source_total_env_transitions": resume_metadata.get("total_env_transitions"),
                    "source_run_name": resume_metadata.get("run_name"),
                    "source_model_name": resume_metadata.get("model_name"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    train_config = RustPPOTrainingConfig(
        run_name=run_name,
        v5_league_config=V5LeagueConfig(
            adaptive_strengths=(1.0,),
            mixed_visibility_rate=1.0,
            enemy_private_info_rate=1.0,
            draw_assist_rate=1.0,
            draw_assist_min_strength=1.0,
            teacher_start_update=999999,
            opponent_mix="self:1.0,v5_snapshot:0.5,greedy_face:0.05,random:0.02,llm_teacher:0.02",
            assist_modes=(
                {
                    "assembler_enabled": True,
                    "assembler_strength": 1.0,
                    "desirerer_enabled": True,
                    "desirerer_strength": 1.0,
                    "teacher_hint_available": True,
                    "assist_profile_id": 15,
                    "weight": 1.0,
                },
            ),
        ),
        curriculum_metadata={
            "phase": "phase8_v4max_correction",
            "machine": "macbook_pro_m4_pro_24gb",
            "target_v4max_score": target_v4max_score,
            "benchmark_gate": "v5_s1_assist_vs_v4max_balanced",
            "v4_1_included": False,
            "assist_policy": "full_assist_s1",
            "private_info_policy": "enemy_hand_deck_known",
            "teacher_hint_policy": "offline_flag_only",
            "opponent_execution_policy": "scheduled_metadata_trace_pool_not_online_v4",
            "trace_policy": "v4max_deck_bank_counterpick_curriculum",
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "resume_source_update": resume_metadata.get("update", 0),
            "resume_optimizer_restored": optimizer_restored,
            "v4max_deck_bank_size": len(v4_deck_bank),
            "counter_deck_count": len(counter_decks),
        },
        updates=updates,
        env_count=env_count,
        steps_per_update=steps_per_update,
        epochs=1,
        minibatch_size=minibatch_size,
        clip_epsilon=0.16,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_every=checkpoint_every,
        metrics_path=run_dir / "metrics.jsonl",
        league_manifest_path=run_dir / "league_manifest.json",
        trace_manifest_path=manifest_path,
        v5_runtime_mode_source="league_schedule",
        policy_scoring_backend="padded",
        policy_selection_backend="rust",
        full_batch_eval=False,
        diagnostic_mode="none",
        seed=seed,
    )

    started = time.perf_counter()
    result = train_rust_ppo_trace_files([], model, optimizer, train_config, library_path=library_path)
    elapsed = time.perf_counter() - started
    metrics = result["metrics"]
    collect_tps = [int(item["env_transitions"]) / float(item["collect_seconds"]) for item in metrics]
    summary = {
        "status": "ok",
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_source_update": resume_metadata.get("update", 0),
        "resume_optimizer_restored": optimizer_restored,
        "trace_manifest_id": result["trace_manifest_id"],
        "updates": updates,
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "total_env_transitions": int(result["total_env_transitions"]),
        "elapsed_seconds": elapsed,
        "end_to_end_transitions_per_second": int(result["total_env_transitions"]) / elapsed,
        "mean_collect_transitions_per_second": sum(collect_tps) / len(collect_tps),
        "max_abs_approx_kl": max(abs(float(item["approx_kl"])) for item in metrics),
        "min_entropy": min(float(item["entropy"]) for item in metrics),
        "last_loss": float(metrics[-1]["loss"]),
        "last_approx_kl": float(metrics[-1]["approx_kl"]),
        "last_entropy": float(metrics[-1]["entropy"]),
        "dense_bytes_any": max(int(item["stored_dense_feature_bytes"]) for item in metrics),
        "next_observation_bytes_any": max(int(item["stored_next_observation_bytes"]) for item in metrics),
        "terminal_observation_bytes_any": max(int(item["stored_terminal_observation_bytes"]) for item in metrics),
        "checkpoint_path": result["checkpoint_path"],
        "league_manifest_path": result["league_manifest_path"],
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "target_v4max_score": target_v4max_score,
        "max_rss_mb": _rss_mb(),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE8_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _phase8_scenarios(
    *,
    v4_deck_bank: list[list[int]],
    counter_decks: list[list[int]],
    seed: int,
) -> list[V5TraceScenario]:
    visibility = (
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": True,
            "enemy_deck_known": True,
            "enemy_deck_order_known": False,
        },
    )
    draw_assist = ({"draw_assist_enabled": True, "draw_assist_strength": 1.0},)
    assist = (
        {
            "assembler_enabled": True,
            "assembler_strength": 1.0,
            "desirerer_enabled": True,
            "desirerer_strength": 1.0,
            "teacher_hint_available": True,
            "assist_profile_id": 15,
            "weight": 1.0,
        },
    )
    scenarios: list[V5TraceScenario] = []
    scenario_seed = int(seed) + 70_000
    for idx, v4_deck in enumerate(v4_deck_bank):
        counter = counter_decks[idx % len(counter_decks)] if counter_decks else v4_deck
        seeds = (scenario_seed + idx * 29, scenario_seed + idx * 29 + 1)
        scenarios.append(
            V5TraceScenario(
                scenario_key=f"phase8_v4max_counter_p1_{idx:03d}",
                seeds=seeds,
                steps=14,
                p1_deck_ids=tuple(counter),
                p2_deck_ids=tuple(v4_deck),
                adaptive_strengths=(1.0,),
                visibility_modes=visibility,
                draw_assist_modes=draw_assist,
                assist_modes=assist,
            )
        )
        scenarios.append(
            V5TraceScenario(
                scenario_key=f"phase8_v4max_counter_p2_{idx:03d}",
                seeds=seeds,
                steps=14,
                p1_deck_ids=tuple(v4_deck),
                p2_deck_ids=tuple(counter),
                adaptive_strengths=(1.0,),
                visibility_modes=visibility,
                draw_assist_modes=draw_assist,
                assist_modes=assist,
            )
        )
    return scenarios


def _build_v4max_deck_bank(*, seed: int, count: int) -> list[list[int]]:
    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")
    seen: set[tuple[int, ...]] = set()
    decks: list[list[int]] = []
    probe = 0
    while len(decks) < int(count) and probe < int(count) * 8:
        current_seed = int(seed) + probe
        env.reset(seed=current_seed, starting_player_id=1)
        state = env._env.state
        for player in (state.p1, state.p2):
            deck = _normalize_deck(_player_card_pool_ids(player), opponent_deck_ids=())
            key = tuple(deck)
            if key not in seen:
                seen.add(key)
                decks.append(deck)
                if len(decks) >= int(count):
                    break
        probe += 1
    if not decks:
        raise RuntimeError("failed to build V4-max deck bank")
    return decks


def _build_counter_decks(v4_deck_bank: list[list[int]], *, assembler_dataset: Path | None) -> list[list[int]]:
    evaluator = DeckMatchupEvaluator()
    candidates = _load_assembler_candidates(assembler_dataset)
    if not candidates:
        return [list(deck) for deck in v4_deck_bank]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for opponent in v4_deck_bank:
        normalized_candidates = [
            AssemblerCandidate(
                deck_ids=_normalize_deck(candidate.deck_ids, opponent_deck_ids=opponent),
                metadata=dict(candidate.metadata),
            )
            for candidate in candidates
        ]
        best = evaluator.search_best(opponent, normalized_candidates)
        key = tuple(best.deck_ids)
        if key not in seen:
            seen.add(key)
            out.append(list(best.deck_ids))
    return out or [list(deck) for deck in v4_deck_bank]


def _load_assembler_candidates(path: Path | None) -> list[AssemblerCandidate]:
    if path is None or not path.exists():
        return []
    seen: set[tuple[int, ...]] = set()
    candidates: list[AssemblerCandidate] = []
    for row in load_assembler_dataset(path):
        deck = tuple(int(card_id) for card_id in row.candidate_deck_ids if int(card_id) > 0)
        if deck in seen:
            continue
        seen.add(deck)
        candidates.append(AssemblerCandidate(deck_ids=list(deck), metadata={"source_run": row.source_run}))
    return candidates


def _normalize_deck(deck_ids: Iterable[int], *, opponent_deck_ids: Iterable[int]) -> list[int]:
    raw = [int(card_id) for card_id in deck_ids if int(card_id) > 0]
    hero_ids = [card_id for card_id in raw if card_id in HERO_CARD_IDS]
    if hero_ids:
        hero_id = hero_ids[0]
        non_heroes = [card_id for card_id in raw if card_id not in HERO_CARD_IDS]
    else:
        opponent_heroes = [int(card_id) for card_id in opponent_deck_ids if int(card_id) in HERO_CARD_IDS]
        hero_id = opponent_heroes[0] if opponent_heroes else 1
        non_heroes = raw
    if not non_heroes:
        non_heroes = [37, 38, 40, 41, 42, 27, 28, 29]
    return [hero_id, *non_heroes[:8]]


def _player_card_pool_ids(player) -> list[int]:
    return [
        int(card.card_id)
        for card in [player.hero, *player.hand, *player.deck, *player.board, *player.graveyard]
        if int(card.card_id) > 0
    ]


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default.resolve() if default is not None else None
    return Path(value).expanduser().resolve()


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024


if __name__ == "__main__":
    raise SystemExit(main())
