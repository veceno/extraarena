#!/usr/bin/env python3
"""Build V5 auxiliary sub-model datasets from completed training artifacts."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3.5" / "python"))

from train_v3.aux_models import (
    AssemblerCandidate,
    DeckMatchupEvaluator,
    DrawAssistController,
    build_assembler_rows_from_v5_trace,
    build_desirerer_rows_from_v5_trace,
    evaluate_assembler_baseline,
    evaluate_desirerer_baseline,
    load_assembler_dataset,
    load_desirerer_dataset,
    save_assembler_dataset_with_manifest,
    save_desirerer_dataset_with_manifest,
)
from train_v3.trace_factory_v5 import load_v5_trace_pool_manifest, resolve_v5_trace_paths
from train_v3.v5_artifacts import read_manifest_json


DEFAULT_SOURCE_RUN = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase4_mixed_assist_private_refresh_after_handicap_20260606_145409"
)


def main() -> int:
    run_name = _env_str("PHASE5_RUN_NAME", "phase5_aux_models_from_phase4")
    source_run = _env_path("PHASE5_SOURCE_RUN", DEFAULT_SOURCE_RUN)
    max_traces = _env_int("PHASE5_MAX_TRACES", 0)
    out_root = Path(os.environ.get("PHASE5_OUT_ROOT", ROOT / "TrainV3.5" / "runs")).resolve()

    source_manifest_path = source_run / "trace_manifest.json"
    source_summary_path = source_run / "run_summary.json"
    source_league_path = source_run / "league_manifest.json"
    source_checkpoint = _source_checkpoint(source_run)
    source_manifest = load_v5_trace_pool_manifest(source_manifest_path)
    source_manifest_id = str(source_manifest["manifest_id"])
    trace_paths = resolve_v5_trace_paths(source_manifest)
    if max_traces > 0:
        trace_paths = trace_paths[:max_traces]

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    aux_dir = run_dir / "aux"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase5_aux_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    config = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run": str(source_run),
        "source_trace_manifest": str(source_manifest_path),
        "source_trace_manifest_id": source_manifest_id,
        "source_checkpoint": str(source_checkpoint),
        "max_traces": max_traces,
        "traces_selected": len(trace_paths),
        "stage": "aux_submodels_dataset_eval",
        "assembler_label_policy": "weak_trace_outcome_label",
        "desirerer_label_policy": "trace_step_next_turn_delta",
    }
    (run_dir / "phase5_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE5_RUN_DIR", run_dir, flush=True)
    print("PHASE5_CONFIG", json.dumps(config, sort_keys=True), flush=True)

    assembler_rows = []
    desirerer_rows = []
    draw_assist_trace_count = 0
    private_info_trace_count = 0
    assist_profile_counts: dict[str, int] = {}
    for idx, trace_path in enumerate(trace_paths, start=1):
        trace = read_manifest_json(trace_path)
        source = str(trace_path)
        assembler_rows.extend(build_assembler_rows_from_v5_trace(trace, source_run=source))
        desirerer_rows.extend(build_desirerer_rows_from_v5_trace(trace, source_run=source))
        env_config = trace.get("env_config") if isinstance(trace.get("env_config"), dict) else {}
        draw_assist_trace_count += int(bool(env_config.get("draw_assist_enabled")))
        private_info_trace_count += int(bool(env_config.get("enemy_hand_known")) or bool(env_config.get("enemy_deck_known")))
        profile = str(int(env_config.get("assist_profile_id", 0) or 0))
        assist_profile_counts[profile] = assist_profile_counts.get(profile, 0) + 1
        if idx % 512 == 0 or idx == len(trace_paths):
            print(
                "PHASE5_PROGRESS",
                json.dumps(
                    {
                        "traces_processed": idx,
                        "traces_total": len(trace_paths),
                        "assembler_rows": len(assembler_rows),
                        "desirerer_rows": len(desirerer_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assembler_dataset, assembler_manifest = save_assembler_dataset_with_manifest(
        assembler_rows,
        aux_dir / "assembler.jsonl",
        aux_dir / "assembler_manifest.json",
        source_manifest_ids=(source_manifest_id,),
    )
    desirerer_dataset, desirerer_manifest = save_desirerer_dataset_with_manifest(
        desirerer_rows,
        aux_dir / "desirerer.jsonl",
        aux_dir / "desirerer_manifest.json",
        source_manifest_ids=(source_manifest_id,),
    )

    assembler_eval = evaluate_assembler_baseline(load_assembler_dataset(assembler_dataset))
    desirerer_eval = evaluate_desirerer_baseline(load_desirerer_dataset(desirerer_dataset))
    summary = {
        "status": "ok",
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_trace_manifest_id": source_manifest_id,
        "source_summary_path": str(source_summary_path) if source_summary_path.exists() else "",
        "source_league_manifest_path": str(source_league_path) if source_league_path.exists() else "",
        "traces_processed": len(trace_paths),
        "draw_assist_trace_count": draw_assist_trace_count,
        "private_info_trace_count": private_info_trace_count,
        "assist_profile_counts": assist_profile_counts,
        "assembler_dataset_path": str(assembler_dataset),
        "assembler_manifest_path": str(assembler_manifest),
        "assembler_rows": len(assembler_rows),
        "assembler_eval": assembler_eval,
        "assembler_search_smoke": _assembler_search_smoke(assembler_rows),
        "desirerer_dataset_path": str(desirerer_dataset),
        "desirerer_manifest_path": str(desirerer_manifest),
        "desirerer_rows": len(desirerer_rows),
        "desirerer_eval": desirerer_eval,
        "draw_controller_smoke": _draw_controller_smoke(desirerer_rows),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE5_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _assembler_search_smoke(rows: list[Any]) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    first = rows[0]
    candidates = [
        AssemblerCandidate(deck_ids=list(row.candidate_deck_ids), metadata={"source_run": row.source_run})
        for row in rows[: min(64, len(rows))]
    ]
    best = DeckMatchupEvaluator().search_best(first.opponent_deck_ids, candidates)
    return {
        "status": "ok",
        "opponent_deck_ids": list(first.opponent_deck_ids),
        "candidate_count": len(candidates),
        "best_deck_ids": list(best.deck_ids),
        "best_score": float(best.score),
    }


def _draw_controller_smoke(rows: list[Any]) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    row = rows[0]
    return DrawAssistController().choose_draw(
        deck_ids=row.deck_ids,
        hand_ids=row.hand_ids,
        board_power_ratio=float(row.state_summary.get("board_power_ratio", 1.0) or 1.0),
        draw_assist_enabled=True,
        draw_assist_strength=max(0.1, float(row.draw_assist_strength)),
    )


def _source_checkpoint(source_run: Path) -> Path:
    summary_path = source_run / "run_summary.json"
    if summary_path.exists():
        summary = read_manifest_json(summary_path)
        checkpoint = Path(str(summary.get("checkpoint_path", "")))
        if checkpoint.exists():
            return checkpoint
    checkpoints = sorted((source_run / "checkpoints").glob("*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {source_run}")
    return checkpoints[-1]


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default.resolve()
    return Path(value).expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
