#!/usr/bin/env python3
"""Deterministically replay and replace nonterminal V5 benchmark scenarios.

The benchmark is symmetric at the seed level (four seat/start cells).  A rare
cell can hit the base max-step or max-turn limit even when its three siblings
finish normally.  This tool reruns only the affected seed/opponent groups with
larger limits, replaces only the matching scenario IDs, preserves the original
raw artifact, and regenerates all derived reports with an audit trail.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "TrainV3.5" / "runs"
MAIN_ROOT = Path("/Users/laveqox/Documents/ExtraArenaRaS")
for path in (ROOT, ROOT / "TrainV3.5" / "python", RUNS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ai  # noqa: E402

if str(MAIN_ROOT / "ai") not in ai.__path__:
    ai.__path__.append(str(MAIN_ROOT / "ai"))

from ai.model_benchmark.reporting import write_report_artifacts  # noqa: E402
from run_model_benchmark_v5_current import _v5_h2h  # noqa: E402


TERMINAL_STATUSES = {"p1_win", "p2_win", "draw"}


def _nonterminal(row: dict[str, Any]) -> bool:
    return bool(
        row.get("error")
        or row.get("timed_out")
        or row.get("truncated")
        or row.get("status") not in TERMINAL_STATUSES
    )


def _adjudicate_persistent_draw(
    row: dict[str, Any],
    *,
    max_steps: int,
    max_turns: int,
) -> dict[str, Any]:
    if row.get("error"):
        raise ValueError("execution errors cannot be adjudicated as draws")
    if not (
        row.get("timed_out")
        or row.get("truncated")
        or row.get("status") in {"max_steps", "ongoing"}
    ):
        raise ValueError(f"unsupported nonterminal status: {row.get('status')!r}")
    return {
        **row,
        "status": "draw",
        "winner_id": None,
        "winner_name": None,
        "draw": True,
        "timed_out": False,
        "truncated": False,
        "adjudicated_draw": True,
        "adjudication_reason": "persistent_nonterminal_after_extended_replay",
        "adjudication_limits": {
            "max_steps": max_steps,
            "max_turns": max_turns,
        },
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:100]


def _model_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in payload["models"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one model named {name!r}, got {len(matches)}")
    return matches[0]


def _command_for_group(
    *,
    payload: dict[str, Any],
    candidate: dict[str, Any],
    opponent_name: str,
    seed: int,
    output_dir: Path,
    runner: Path,
    max_steps: int,
    max_turns: int,
) -> list[str]:
    opponent = _model_by_name(payload, opponent_name)
    v5_opponents = [
        row
        for row in payload["models"]
        if not row.get("ranked") and row.get("kind") == "v5_npz"
    ]
    if opponent.get("kind") == "v5_npz":
        base = opponent
    elif v5_opponents:
        base = v5_opponents[0]
    else:
        raise ValueError("repair runner requires one non-candidate V5 checkpoint")
    aux_artifacts = payload["config"]["auxiliary_artifacts"]
    aux_dir = Path(aux_artifacts["assembler"]["path"]).resolve().parent
    command = [
        sys.executable,
        str(runner),
        "--checkpoint",
        str(Path(candidate["model_path"]).resolve()),
        "--candidate-name",
        str(candidate["name"]),
        "--base-checkpoint",
        str(Path(base["model_path"]).resolve()),
        "--base-name",
        str(base["name"]),
        "--opponent",
        opponent_name,
        "--aux-dir",
        str(aux_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        str(payload["config"]["benchmark_mode"]),
        "--seeds-per-opponent",
        "1",
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--max-turns",
        str(max_turns),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=RUNS / "run_model_benchmark_v5_current.py",
    )
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--max-turns", type=int, default=480)
    args = parser.parse_args()

    raw_path = args.raw.resolve()
    mode_dir = raw_path.parent
    original_path = mode_dir / "raw.pre_terminal_repair.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    affected = [row for row in payload["results"] if _nonterminal(row)]
    if not affected:
        print(json.dumps({"status": "no_repair_needed", "raw": str(raw_path)}))
        return
    if original_path.exists():
        raise FileExistsError(f"preserved original already exists: {original_path}")
    if any(row.get("error") for row in affected):
        raise ValueError("automatic repair refuses benchmark execution errors")

    candidate_rows = [row for row in payload["models"] if row.get("ranked")]
    if len(candidate_rows) != 1:
        raise ValueError(f"expected one ranked candidate, got {len(candidate_rows)}")
    candidate = candidate_rows[0]
    groups = sorted(
        {(str(row["opponent_model"]), int(row["seed"])) for row in affected}
    )
    replacement_by_scenario: dict[str, dict[str, Any]] = {}
    group_audits: list[dict[str, Any]] = []
    repair_root = mode_dir / "terminal_repairs"
    repair_root.mkdir(parents=True, exist_ok=True)
    for opponent_name, seed in groups:
        group_dir = repair_root / f"{_slug(opponent_name)}__seed{seed}"
        command = _command_for_group(
            payload=payload,
            candidate=candidate,
            opponent_name=opponent_name,
            seed=seed,
            output_dir=group_dir,
            runner=args.runner.resolve(),
            max_steps=args.max_steps,
            max_turns=args.max_turns,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        mode = str(payload["config"]["benchmark_mode"])
        repair_raw = group_dir / mode / "raw.json"
        repair_payload = json.loads(repair_raw.read_text(encoding="utf-8"))
        repair_rows = {
            str(row["scenario_id"]): row for row in repair_payload["results"]
        }
        targets = [
            row for row in affected
            if row["opponent_model"] == opponent_name and int(row["seed"]) == seed
        ]
        for old in targets:
            scenario_id = str(old["scenario_id"])
            if scenario_id not in repair_rows:
                raise ValueError(f"repair omitted scenario {scenario_id}")
            replacement = repair_rows[scenario_id]
            if _nonterminal(replacement):
                replacement = _adjudicate_persistent_draw(
                    replacement,
                    max_steps=args.max_steps,
                    max_turns=args.max_turns,
                )
            replacement_by_scenario[scenario_id] = replacement
        group_audits.append(
            {
                "opponent": opponent_name,
                "seed": seed,
                "target_scenarios": [row["scenario_id"] for row in targets],
                "command": command,
                "repair_raw": str(repair_raw),
                "stdout_tail": completed.stdout.strip().splitlines()[-1:],
            }
        )

    original_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    old_by_scenario = {
        str(row["scenario_id"]): row for row in affected
    }
    payload["results"] = [
        replacement_by_scenario.get(str(row["scenario_id"]), row)
        for row in payload["results"]
    ]
    remaining = [row for row in payload["results"] if _nonterminal(row)]
    if remaining:
        raise ValueError(f"{len(remaining)} nonterminal rows remain after repair")
    payload["config"]["terminal_retry_policy"] = (
        f"rerun exact nonterminal seed/opponent cells at "
        f"max_steps={args.max_steps},max_turns={args.max_turns}"
    )
    payload["config"]["terminal_repairs_count"] = len(replacement_by_scenario)
    payload["error_count"] = sum(bool(row.get("error")) for row in payload["results"])
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema": "extra_lr_benchmark_terminal_repair_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "raw": str(raw_path),
        "original_raw": str(original_path),
        "repair_limits": {
            "max_steps": args.max_steps,
            "max_turns": args.max_turns,
        },
        "repaired_scenarios": len(replacement_by_scenario),
        "adjudicated_draws": sum(
            bool(row.get("adjudicated_draw"))
            for row in replacement_by_scenario.values()
        ),
        "groups": group_audits,
        "replacements": [
            {
                "scenario_id": scenario_id,
                "original": old_by_scenario[scenario_id],
                "replacement": replacement_by_scenario[scenario_id],
            }
            for scenario_id in sorted(replacement_by_scenario)
        ],
    }
    (mode_dir / "terminal_repair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report_artifacts(payload, mode_dir)
    h2h = _v5_h2h(payload["results"], str(candidate["name"]))
    (mode_dir / "v5_h2h.json").write_text(
        json.dumps(h2h, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "repaired",
                "raw": str(raw_path),
                "repaired_scenarios": len(replacement_by_scenario),
                "groups": len(groups),
                "remaining_nonterminal": 0,
                "adjudicated_draws": sum(
                    bool(row.get("adjudicated_draw"))
                    for row in replacement_by_scenario.values()
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
