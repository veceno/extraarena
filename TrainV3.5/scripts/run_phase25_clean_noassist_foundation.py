#!/usr/bin/env python3
"""Run a clean-room V5 no-assist foundation training pass.

Phase25 deliberately excludes earlier V5 data because prior no-assist runs were
contaminated by deck-assist leakage. It starts from a fresh model by default,
uses Rust for the rollout/training hot path, and records a narrow no-assist
contract in every artifact.
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

from train_v3.league_v5 import V5LeagueConfig
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
from train_v3.trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool
from train_v3.v5_artifacts import read_manifest_json, write_manifest_json
from train_v3.v5_policy import create_v5_policy


DEFAULT_DECK_POOL: tuple[tuple[int, ...], ...] = (
    (1, 37, 38, 40, 41, 42, 27, 28, 29),
    (1, 40, 43, 29, 31, 25, 27, 37, 28),
    (1, 32, 26, 16, 40, 33, 46, 28, 29),
    (1, 17, 23, 19, 8, 42, 28, 44, 43),
    (1, 32, 18, 45, 28, 16, 46, 38, 36),
    (1, 20, 42, 31, 21, 13, 27, 28, 38),
    (1, 41, 45, 31, 21, 40, 39, 43, 15),
)


def main() -> int:
    run_name = _env_str("PHASE25_RUN_NAME", "phase25_clean_noassist_foundation")
    env_count = _env_int("PHASE25_ENV_COUNT", 24576)
    steps_per_update = _env_int("PHASE25_STEPS_PER_UPDATE", 32)
    updates = _env_int("PHASE25_UPDATES", 5000)
    minibatch_size = _env_int("PHASE25_MINIBATCH_SIZE", 8192)
    checkpoint_every = _env_int("PHASE25_CHECKPOINT_EVERY", 50)
    hidden_dim = _env_int("PHASE25_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE25_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE25_LR", 2.0e-4)
    entropy_coef = _env_float("PHASE25_ENTROPY_COEF", 0.03)
    clip_epsilon = _env_float("PHASE25_CLIP_EPSILON", 0.16)
    max_grad_norm = _env_optional_float("PHASE25_MAX_GRAD_NORM", 0.5)
    seed = _env_int("PHASE25_SEED", 25001)
    trace_seed_count = _env_int("PHASE25_TRACE_SEED_COUNT", 16)
    strengths = _env_float_tuple("PHASE25_ADAPTIVE_STRENGTHS", (1.0,))
    deck_pool = _env_deck_pool("PHASE25_DECK_POOL", DEFAULT_DECK_POOL)
    deck_pair_mode = _env_str("PHASE25_DECK_PAIR_MODE", "ordered")
    level_modes = _env_level_modes("PHASE25_LEVEL_MODES_JSON")
    reuse_trace_manifest_path = _env_path("PHASE25_TRACE_MANIFEST_PATH")
    policy_padding_mode = _env_str("PHASE25_POLICY_PADDING_MODE", "bucketed")
    policy_bucket_max_padding_ratio = _env_float("PHASE25_POLICY_BUCKET_MAX_PADDING_RATIO", 1.35)
    policy_bucket_min_rows = _env_int("PHASE25_POLICY_BUCKET_MIN_ROWS", 2048)
    ppo_minibatch_plan = _env_str("PHASE25_PPO_MINIBATCH_PLAN", "contiguous")
    log_selected_trace_paths = _env_bool("PHASE25_LOG_SELECTED_TRACE_PATHS", False)
    resume_checkpoint = _env_path("PHASE25_RESUME_CHECKPOINT")
    out_root = Path(os.environ.get("PHASE25_OUT_ROOT", ROOT / "TrainV3" / "runs")).resolve()
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    trace_dir = run_dir / "trace_pool"
    manifest_path = run_dir / "trace_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase25_clean_noassist_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    deck_pairs = build_phase25_deck_pairs(deck_pool, mode=deck_pair_mode)
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
        "clip_epsilon": clip_epsilon,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "trace_seed_count": trace_seed_count,
        "trace_manifest_path": str(reuse_trace_manifest_path) if reuse_trace_manifest_path is not None else str(manifest_path),
        "trace_manifest_reused": reuse_trace_manifest_path is not None,
        "adaptive_strengths": list(strengths),
        "deck_pair_mode": deck_pair_mode,
        "deck_pool": [list(deck) for deck in deck_pool],
        "deck_pair_count": len(deck_pairs),
        "level_modes": list(level_modes),
        "policy_padding_mode": policy_padding_mode,
        "policy_bucket_max_padding_ratio": policy_bucket_max_padding_ratio,
        "policy_bucket_min_rows": policy_bucket_min_rows,
        "ppo_minibatch_plan": ppo_minibatch_plan,
        "log_selected_trace_paths": log_selected_trace_paths,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "library_path": str(library_path),
        "clean_room_noassist": True,
        "contaminated_prior_data_excluded": True,
        "v4_1_included": False,
    }
    (run_dir / "phase25_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE25_RUN_DIR", run_dir, flush=True)
    print("PHASE25_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    if reuse_trace_manifest_path is not None:
        manifest_path = reuse_trace_manifest_path
    else:
        manifest = generate_v5_trace_pool(
            build_phase25_trace_scenarios(
                seed=seed,
                trace_seed_count=trace_seed_count,
                adaptive_strengths=strengths,
                deck_pairs=deck_pairs,
                level_modes=level_modes,
            ),
            trace_dir,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        write_manifest_json(manifest, manifest_path)
    loaded_manifest = read_manifest_json(manifest_path)
    print(
        "PHASE25_TRACE_MANIFEST",
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
            "PHASE25_RESUME",
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "optimizer_restored": optimizer_restored,
                    "source_update": resume_metadata.get("update"),
                    "source_run_name": resume_metadata.get("run_name"),
                    "source_model_name": resume_metadata.get("model_name"),
                    "warning": "resume disables strict clean-room-from-random semantics",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    train_config = RustPPOTrainingConfig(
        run_name=run_name,
        v5_league_config=V5LeagueConfig(
            adaptive_strengths=strengths,
            mixed_visibility_rate=0.0,
            enemy_private_info_rate=0.0,
            draw_assist_rate=0.0,
            draw_assist_min_strength=1.0,
            teacher_start_update=999999,
            opponent_mix="self:1.0,v5_snapshot:0.25,random:0.02",
            assist_modes=(
                {
                    "assembler_enabled": False,
                    "assembler_strength": 0.0,
                    "desirerer_enabled": False,
                    "desirerer_strength": 0.0,
                    "teacher_hint_available": False,
                    "assist_profile_id": 0,
                    "weight": 1.0,
                },
            ),
        ),
        curriculum_metadata={
            "phase": "phase25_clean_noassist_foundation",
            "clean_room_noassist": True,
            "contaminated_prior_data_excluded": True,
            "deck_pair_mode": deck_pair_mode,
            "deck_pair_count": len(deck_pairs),
            "deck_pool": [list(deck) for deck in deck_pool],
            "level_modes": list(level_modes),
            "private_info_policy": "enemy_hidden_only",
            "draw_assist_policy": "off",
            "assist_policy": "off",
            "teacher_hint_policy": "off",
            "v4_1_included": False,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "resume_source_update": resume_metadata.get("update", 0),
            "resume_optimizer_restored": optimizer_restored,
        },
        updates=updates,
        env_count=env_count,
        steps_per_update=steps_per_update,
        epochs=1,
        minibatch_size=minibatch_size,
        clip_epsilon=clip_epsilon,
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
        policy_padding_mode=policy_padding_mode,
        policy_bucket_max_padding_ratio=policy_bucket_max_padding_ratio,
        policy_bucket_min_rows=policy_bucket_min_rows,
        ppo_minibatch_plan=ppo_minibatch_plan,
        log_selected_trace_paths=log_selected_trace_paths,
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
        "trace_manifest_id": result["trace_manifest_id"],
        "updates": updates,
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "total_env_transitions": int(result["total_env_transitions"]),
        "elapsed_seconds": elapsed,
        "end_to_end_transitions_per_second": int(result["total_env_transitions"]) / elapsed,
        "mean_collect_transitions_per_second": sum(collect_tps) / len(collect_tps),
        "last_loss": float(metrics[-1]["loss"]),
        "last_approx_kl": float(metrics[-1]["approx_kl"]),
        "last_entropy": float(metrics[-1]["entropy"]),
        "max_abs_approx_kl": max(abs(float(item["approx_kl"])) for item in metrics),
        "min_entropy": min(float(item["entropy"]) for item in metrics),
        "checkpoint_path": result["checkpoint_path"],
        "league_manifest_path": result["league_manifest_path"],
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "clean_room_noassist": True,
        "contaminated_prior_data_excluded": True,
        "max_rss_mb": _rss_mb(),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE25_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def build_phase25_trace_scenarios(
    *,
    seed: int,
    trace_seed_count: int,
    adaptive_strengths: tuple[float, ...],
    deck_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    level_modes: tuple[dict[str, int | str], ...],
) -> list[V5TraceScenario]:
    if int(trace_seed_count) <= 0:
        raise ValueError("trace_seed_count must be positive")
    if not deck_pairs:
        raise ValueError("deck_pairs must not be empty")
    seeds = tuple(int(seed) + 125_000 + idx * 17 for idx in range(int(trace_seed_count)))
    visibility = (
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": False,
            "enemy_deck_known": False,
            "enemy_deck_order_known": False,
        },
    )
    draw = ({"draw_assist_enabled": False, "draw_assist_strength": 0.0},)
    assist = (
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 0,
        },
    )
    scenarios: list[V5TraceScenario] = []
    for pair_idx, (p1_deck, p2_deck) in enumerate(deck_pairs):
        for choose in ("first", "last"):
            scenarios.append(
                V5TraceScenario(
                    scenario_key=f"phase25_clean_noassist_pair_{pair_idx:03d}_{choose}",
                    seeds=seeds,
                    steps=16,
                    p1_deck_ids=tuple(p1_deck),
                    p2_deck_ids=tuple(p2_deck),
                    adaptive_strengths=adaptive_strengths,
                    visibility_modes=visibility,
                    draw_assist_modes=draw,
                    assist_modes=assist,
                    level_modes=level_modes,
                    choose=choose,
                )
            )
    return scenarios


def build_phase25_deck_pairs(
    deck_pool: tuple[tuple[int, ...], ...],
    *,
    mode: str,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    pool = tuple(tuple(deck) for deck in deck_pool)
    if not pool:
        raise ValueError("deck_pool must not be empty")
    mode = str(mode).strip().lower()
    if mode == "mirror":
        return tuple((deck, deck) for deck in pool)
    if mode == "cycle":
        pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for idx, deck in enumerate(pool):
            other = pool[(idx + 1) % len(pool)]
            pairs.append((deck, deck))
            pairs.append((deck, other))
            pairs.append((other, deck))
        return tuple(_dedupe_pairs(pairs))
    if mode == "ordered":
        return tuple((p1, p2) for p1 in pool for p2 in pool)
    raise ValueError("deck_pair_mode must be mirror, cycle, or ordered")


def _dedupe_pairs(
    pairs: Iterable[tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return tuple(out)


def _env_deck_pool(name: str, default: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    decks: list[tuple[int, ...]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        deck = tuple(int(part.strip()) for part in chunk.split(",") if part.strip())
        if len(deck) < 2:
            raise ValueError(f"{name} contains a deck with fewer than two ids")
        decks.append(deck)
    if not decks:
        raise ValueError(f"{name} must contain at least one deck")
    return tuple(decks)


def _env_level_modes(name: str) -> tuple[dict[str, int | str], ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return (
            {"p1_level": 1, "p2_level": 1, "label": "equal_l1"},
            {"p1_level": 1, "p2_level": 2, "label": "p1_l1_vs_p2_l2"},
            {"p1_level": 2, "p2_level": 1, "label": "p1_l2_vs_p2_l1"},
        )
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{name} must be a non-empty JSON list")
    return tuple(_normalize_level_mode(item) for item in parsed)


def _normalize_level_mode(value: Any) -> dict[str, int | str]:
    if not isinstance(value, dict):
        raise ValueError("each level mode must be an object")
    p1 = max(1, min(10, int(value.get("p1_level", 1))))
    p2 = max(1, min(10, int(value.get("p2_level", 1))))
    label = str(value.get("label") or f"p1_l{p1}_vs_p2_l{p2}")
    return {"p1_level": p1, "p2_level": p2, "label": label}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(default)
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024


if __name__ == "__main__":
    raise SystemExit(main())
