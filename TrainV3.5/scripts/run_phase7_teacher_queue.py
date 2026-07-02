#!/usr/bin/env python3
"""Build an offline LLM-teacher queue from completed V5 training traces."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3.5" / "python"))

from train_v3.llm_teacher import OpenAICompatibleTeacherConfig, OpenAICompatibleTeacherClient
from train_v3.trace_factory_v5 import load_v5_trace_pool_manifest, resolve_v5_trace_paths
from train_v3.v5_artifacts import read_manifest_json


DEFAULT_SOURCE_RUN = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase6_noassist_entropy_recovery_20260607_112603"
)
DEFAULT_AUX_RUN = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase5_aux_models_from_phase4_20260606_231349"
)
DEFAULT_ACCEPTANCE_RUN = ROOT / "TrainV3.5" / "runs" / "acceptance_v5_20260607_124618"


def main() -> int:
    run_name = _env_str("PHASE7_RUN_NAME", "phase7_teacher_queue_from_recovery")
    source_run = _env_path("PHASE7_SOURCE_RUN", DEFAULT_SOURCE_RUN)
    aux_run = _env_path("PHASE7_AUX_RUN", DEFAULT_AUX_RUN)
    acceptance_run = _env_path("PHASE7_ACCEPTANCE_RUN", DEFAULT_ACCEPTANCE_RUN)
    max_states = _env_int("PHASE7_MAX_STATES", 256)
    max_traces = _env_int("PHASE7_MAX_TRACES", 0)
    call_teacher = _env_bool("PHASE7_CALL_TEACHER", False)
    out_root = Path(os.environ.get("PHASE7_OUT_ROOT", ROOT / "TrainV3.5" / "runs")).resolve()

    teacher_config = OpenAICompatibleTeacherConfig(
        base_url=_env_str("PHASE7_TEACHER_BASE_URL", "https://api.deepseek.com"),
        api_key_env=_env_str("PHASE7_TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        model=_env_str("PHASE7_TEACHER_MODEL", "deepseek-reasoner"),
        prompt_version=_env_str("PHASE7_TEACHER_PROMPT_VERSION", "v5-hard-state-v1"),
        timeout_seconds=float(_env_str("PHASE7_TEACHER_TIMEOUT_SECONDS", "30")),
        max_retries=_env_int("PHASE7_TEACHER_MAX_RETRIES", 2),
        cache_dir=None,
    )

    source_manifest_path = source_run / "trace_manifest.json"
    source_summary_path = source_run / "run_summary.json"
    source_checkpoint = _source_checkpoint(source_run)
    source_manifest = load_v5_trace_pool_manifest(source_manifest_path)
    source_manifest_id = str(source_manifest["manifest_id"])
    trace_paths = resolve_v5_trace_paths(source_manifest)
    if max_traces > 0:
        trace_paths = trace_paths[:max_traces]

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase7_teacher_queue_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    requests = _select_teacher_requests(
        trace_paths=trace_paths,
        max_states=max_states,
        model=teacher_config.model,
        base_url=teacher_config.base_url,
        prompt_version=teacher_config.prompt_version,
    )
    queue_path = run_dir / "teacher_requests.jsonl"
    queue_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests),
        encoding="utf-8",
    )

    preferences_path = ""
    label_error_count = 0
    label_errors_path = ""
    labeled_count = 0
    if call_teacher:
        preferences_path, labeled_count, label_errors_path, label_error_count = _label_requests(
            run_dir,
            queue_path,
            teacher_config,
        )

    summary = {
        "status": "ok",
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_summary_path": str(source_summary_path) if source_summary_path.exists() else "",
        "source_trace_manifest_id": source_manifest_id,
        "aux_run": str(aux_run),
        "assembler_manifest_path": str(aux_run / "aux" / "assembler_manifest.json")
        if (aux_run / "aux" / "assembler_manifest.json").exists()
        else "",
        "desirerer_manifest_path": str(aux_run / "aux" / "desirerer_manifest.json")
        if (aux_run / "aux" / "desirerer_manifest.json").exists()
        else "",
        "acceptance_report_path": str(acceptance_run / "acceptance_report.json")
        if (acceptance_run / "acceptance_report.json").exists()
        else "",
        "teacher_base_url": teacher_config.base_url.rstrip("/"),
        "teacher_api_key_env": teacher_config.api_key_env,
        "teacher_model": teacher_config.model,
        "teacher_prompt_version": teacher_config.prompt_version,
        "call_teacher": call_teacher,
        "queue_path": str(queue_path),
        "queued_states": len(requests),
        "labeled_count": labeled_count,
        "label_error_count": label_error_count,
        "teacher_label_errors_path": label_errors_path,
        "teacher_preferences_path": preferences_path,
        "hardness": _hardness_summary(requests),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE7_RUN_DIR", run_dir, flush=True)
    print("PHASE7_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _select_teacher_requests(
    *,
    trace_paths: list[Path],
    max_states: int,
    model: str,
    base_url: str,
    prompt_version: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace_path in trace_paths:
        trace = read_manifest_json(trace_path)
        env_config = trace.get("env_config") if isinstance(trace.get("env_config"), dict) else {}
        for step in trace.get("steps", []):
            if not isinstance(step, dict):
                continue
            pre = step.get("pre")
            if not isinstance(pre, dict):
                continue
            legal_action_ids = [int(item) for item in pre.get("legal_ids", []) if int(item) >= 0]
            state_sha256 = str(pre.get("state_sha256", ""))
            if not state_sha256 or state_sha256 in seen or len(legal_action_ids) < 2:
                continue
            state = pre.get("state")
            if not isinstance(state, dict):
                continue
            state_summary = _state_summary(
                state=state,
                acting_player_id=int(step.get("acting_player_id", 0) or 0),
                legal_action_ids=legal_action_ids,
                reward=float(step.get("reward", step.get("base_reward", 0.0)) or 0.0),
                env_config=env_config,
            )
            hardness_score = _hardness_score(state_summary)
            seen.add(state_sha256)
            candidates.append(
                {
                    "state_sha256": state_sha256,
                    "legal_action_ids": legal_action_ids,
                    "state_summary": state_summary,
                    "hardness_score": hardness_score,
                    "source_trace_path": str(trace_path),
                    "source_step_t": int(step.get("t", 0) or 0),
                    "teacher_model": model,
                    "teacher_provider_base_url": base_url.rstrip("/"),
                    "prompt_version": prompt_version,
                    "request_cache_key": _request_cache_key(
                        state_sha256=state_sha256,
                        model=model,
                        base_url=base_url,
                        prompt_version=prompt_version,
                    ),
                }
            )
    candidates.sort(key=lambda row: (-float(row["hardness_score"]), row["state_sha256"]))
    return candidates[:max(0, int(max_states))]


def _label_requests(
    run_dir: Path,
    queue_path: Path,
    teacher_config: OpenAICompatibleTeacherConfig,
) -> tuple[str, int, str, int]:
    client = OpenAICompatibleTeacherClient(
        OpenAICompatibleTeacherConfig(
            base_url=teacher_config.base_url,
            api_key_env=teacher_config.api_key_env,
            model=teacher_config.model,
            prompt_version=teacher_config.prompt_version,
            timeout_seconds=teacher_config.timeout_seconds,
            max_retries=teacher_config.max_retries,
            cache_dir=run_dir / "teacher_cache",
        )
    )
    out = run_dir / "teacher_preferences.jsonl"
    errors_out = run_dir / "teacher_label_errors.jsonl"
    rows = []
    errors = []
    lines = [line for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            row = client.label_state(
                state_sha256=str(request["state_sha256"]),
                legal_action_ids=[int(item) for item in request["legal_action_ids"]],
                state_summary=dict(request["state_summary"]),
            )
            rows.append(row.to_dict())
            out.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        except Exception as exc:
            errors.append(
                {
                    "state_sha256": str(request.get("state_sha256", "")),
                    "request_cache_key": str(request.get("request_cache_key", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            errors_out.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in errors),
                encoding="utf-8",
            )
        if idx % 16 == 0 or idx == len(lines):
            print(
                "PHASE7_LABEL_PROGRESS",
                json.dumps(
                    {
                        "processed": idx,
                        "total": len(lines),
                        "labeled": len(rows),
                        "errors": len(errors),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return str(out), len(rows), str(errors_out) if errors else "", len(errors)


def _state_summary(
    *,
    state: dict[str, Any],
    acting_player_id: int,
    legal_action_ids: list[int],
    reward: float,
    env_config: dict[str, Any],
) -> dict[str, Any]:
    me_key = "p1" if acting_player_id == 1 else "p2"
    enemy_key = "p2" if acting_player_id == 1 else "p1"
    me = state.get(me_key, {}) if isinstance(state.get(me_key), dict) else {}
    enemy = state.get(enemy_key, {}) if isinstance(state.get(enemy_key), dict) else {}
    my_board_power = _cards_power(me.get("board", []))
    enemy_board_power = _cards_power(enemy.get("board", []))
    return {
        "acting_player_id": acting_player_id,
        "turn_number": int(state.get("turn_number", 0) or 0),
        "legal_action_count": len(legal_action_ids),
        "legal_action_ids": legal_action_ids[:64],
        "reward": reward,
        "my_hero_hp": _hero_hp(me),
        "enemy_hero_hp": _hero_hp(enemy),
        "hp_delta": _hero_hp(me) - _hero_hp(enemy),
        "my_board_count": len(me.get("board", []) if isinstance(me.get("board", []), list) else []),
        "enemy_board_count": len(enemy.get("board", []) if isinstance(enemy.get("board", []), list) else []),
        "my_board_power": my_board_power,
        "enemy_board_power": enemy_board_power,
        "board_power_ratio": my_board_power / max(enemy_board_power, 1.0),
        "my_hand_count": len(me.get("hand", []) if isinstance(me.get("hand", []), list) else []),
        "enemy_hand_count": len(enemy.get("hand", []) if isinstance(enemy.get("hand", []), list) else []),
        "adaptive_strength": float(env_config.get("adaptive_strength", 0.0) or 0.0),
        "draw_assist_enabled": bool(env_config.get("draw_assist_enabled")),
        "enemy_private_known": bool(env_config.get("enemy_hand_known")) or bool(env_config.get("enemy_deck_known")),
        "assist_profile_id": int(env_config.get("assist_profile_id", 0) or 0),
    }


def _hardness_score(summary: dict[str, Any]) -> float:
    legal_complexity = min(float(summary["legal_action_count"]) / 8.0, 1.0)
    hp_pressure = min(max(-float(summary["hp_delta"]), 0.0) / 30.0, 1.0)
    board_pressure = min(max(float(summary["enemy_board_power"]) - float(summary["my_board_power"]), 0.0) / 60.0, 1.0)
    negative_reward = min(max(-float(summary["reward"]), 0.0) / 2.0, 1.0)
    return round(0.35 * legal_complexity + 0.25 * hp_pressure + 0.25 * board_pressure + 0.15 * negative_reward, 6)


def _hardness_summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["hardness_score"]) for row in requests]
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
    }


def _request_cache_key(*, state_sha256: str, model: str, base_url: str, prompt_version: str) -> str:
    raw = json.dumps(
        {
            "state_sha256": state_sha256,
            "model": model,
            "base_url": base_url.rstrip("/"),
            "prompt_version": prompt_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cards_power(cards: Any) -> float:
    if not isinstance(cards, list):
        return 0.0
    total = 0.0
    for card in cards:
        if isinstance(card, dict):
            total += max(0, int(card.get("attack", 0) or 0)) * max(0, int(card.get("hp", 0) or 0))
    return total


def _hero_hp(player: dict[str, Any]) -> int:
    hero = player.get("hero")
    return int(hero.get("hp", 0) or 0) if isinstance(hero, dict) else 0


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default.resolve()
    return Path(value).expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
