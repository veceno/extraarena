#!/usr/bin/env python3
"""Run a conservative, resumable Phase-C AWAC/CRR update from u29250.

The authoritative lane is the frozen human corpus.  A small, explicitly
lower-rate semi-synthetic supplement is replayed first: all ten Luna battles
plus a side/outcome-balanced set of complex MiniMax battles.  Each source is
deep-validated and pinned by SHA-256 before it can enter training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "TrainV3.5" / "python", ROOT / "TrainV3.5" / "runs"):
    if str(path) not in os.sys.path:
        os.sys.path.insert(0, str(path))

from rlhf_env.components.v5_trace_validate import validate_v5_trace  # noqa: E402
from train_v3.awac_crr_replay import AwacCrrReplay  # noqa: E402
from train_v3.offline_replay_bridge import (  # noqa: E402
    build_offline_replay_batch,
    make_policy_fn_from_checkpoint,
)


TARGET_ALIAS = "extra-lr-v5-postB-preV5-u29250"
TARGET_WEIGHTS_HASH = "d09ed1941aeb707e"
TARGET_CATALOG_HASH = "2d4e28c0f7740975"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _stable_order(value: str) -> bytes:
    return hashlib.sha256(f"phase-c-v1:{value}".encode()).digest()


def _human_shards(
    freeze_dir: Path, size: int, *, output_dir: Path
) -> list[list[Path]]:
    selection = json.loads((freeze_dir / "selection_manifest.json").read_text())
    frozen_root = output_dir / "human_groups"
    frozen_root.mkdir(parents=True, exist_ok=True)
    groups: list[Path] = []
    for row in selection["battle_rows"]:
        group_id, battle_id = row["group_id"], row["battle_id"]
        source_group = freeze_dir / "sessions" / group_id
        source_manifest = json.loads((source_group / "manifest.json").read_text())
        battle_result = next(
            battle
            for battle in source_manifest.get("battles_results") or []
            if battle.get("battle_id") == battle_id
        )
        frozen_group = frozen_root / f"{group_id}--{battle_id}"
        frozen_group.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            frozen_group / "manifest.json",
            {
                **source_manifest,
                "battle_ids": [battle_id],
                "battles_results": [battle_result],
                "results": {
                    **(source_manifest.get("results") or {}),
                    "battles_finished": 1,
                    "battles_planned": 1,
                },
                "frozen_phase_c_selection": True,
                "source_group_id": group_id,
            },
        )
        for name, target in (
            ("battles", source_group / "battles"),
            ("catalog.json", source_group / "catalog.json"),
        ):
            link = frozen_group / name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"unexpected non-symlink in human freeze: {link}")
            link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
        groups.append(frozen_group)
    groups.sort(key=lambda path: _stable_order(path.name))
    return _chunk(groups, size)


def _luna_group_ids(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text())
    return {str(row["group_id"]) for row in payload["sources"]}


def _select_semi_synthetic(
    sessions_dir: Path,
    luna_manifest: Path,
    *,
    minimax_battles: int,
    output: Path,
) -> list[Path]:
    luna_ids = _luna_group_ids(luna_manifest)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    luna: list[dict[str, Any]] = []
    for group_dir in sessions_dir.iterdir():
        manifest_path = group_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for battle in manifest.get("battles_results") or []:
            battle_id = str(battle.get("battle_id") or "")
            v5_dir = group_dir / "battles" / battle_id / "v5"
            meta_path = v5_dir / "meta.json"
            actions_path = v5_dir / "actions.jsonl"
            log_path = group_dir / "battles" / f"{battle_id}.json"
            if not (meta_path.exists() and actions_path.exists() and log_path.exists()):
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            bot_policy = meta.get("bot_policy") or {}
            if not (
                battle.get("v5_trace_ok") is True
                and battle.get("degraded") is not True
                and meta.get("p1_actor_type") == "llm"
                and meta.get("p2_actor_type") == "bot"
                and meta.get("battle_tag") == "llm-vs-rl"
                and bot_policy.get("weights_hash") == TARGET_WEIGHTS_HASH
                and meta.get("status") in {"p1_win", "p2_win"}
            ):
                continue
            validation = validate_v5_trace(
                v5_dir,
                log_path,
                expected_catalog_hash=TARGET_CATALOG_HASH,
                expected_card_count=50,
            )
            if not validation["ok"]:
                continue
            row = {
                "group_id": group_dir.name,
                "battle_id": battle_id,
                "status": meta["status"],
                "turns": int(meta.get("turns") or 0),
                "actions_sha256": _sha256(actions_path),
                "meta_sha256": _sha256(meta_path),
                "source": "luna" if group_dir.name in luna_ids else "minimax_m3",
            }
            if group_dir.name in luna_ids:
                luna.append(row)
            elif 12 <= row["turns"] <= 60:
                pools[row["status"]].append(row)

    per_outcome = minimax_battles // 2
    selected_minimax: list[dict[str, Any]] = []
    for status in ("p1_win", "p2_win"):
        candidates = sorted(
            pools[status],
            key=lambda row: (-row["turns"], _stable_order(row["group_id"])),
        )
        selected_minimax.extend(candidates[:per_outcome])
    selected = sorted(luna, key=lambda row: row["group_id"]) + sorted(
        selected_minimax, key=lambda row: row["group_id"]
    )
    if len(luna) != len(luna_ids):
        raise RuntimeError(f"Luna validation mismatch: expected {len(luna_ids)}, got {len(luna)}")
    if len(selected_minimax) != per_outcome * 2:
        raise RuntimeError(
            f"MiniMax selection shortfall: expected {per_outcome * 2}, got {len(selected_minimax)}"
        )
    frozen_root = output.parent / "semi_synthetic_groups"
    frozen_root.mkdir(parents=True, exist_ok=True)
    frozen_groups: list[Path] = []
    for row in selected:
        source_group = sessions_dir / row["group_id"]
        source_manifest = json.loads((source_group / "manifest.json").read_text())
        battle_result = next(
            battle
            for battle in source_manifest.get("battles_results") or []
            if battle.get("battle_id") == row["battle_id"]
        )
        frozen_group = frozen_root / f"{row['group_id']}--{row['battle_id']}"
        frozen_group.mkdir(parents=True, exist_ok=True)
        pruned_manifest = {
            **source_manifest,
            "battle_ids": [row["battle_id"]],
            "battles_results": [battle_result],
            "results": {
                **(source_manifest.get("results") or {}),
                "battles_finished": 1,
                "battles_planned": 1,
            },
            "frozen_phase_c_selection": True,
            "source_group_id": row["group_id"],
        }
        _atomic_json(frozen_group / "manifest.json", pruned_manifest)
        for name, target in (
            ("battles", source_group / "battles"),
            ("catalog.json", source_group / "catalog.json"),
        ):
            link = frozen_group / name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"unexpected non-symlink in semi freeze: {link}")
            link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
        frozen_groups.append(frozen_group)

    payload = {
        "schema": "extra_lr_phase_c_semi_synthetic_selection_v1",
        "target_alias": TARGET_ALIAS,
        "target_weights_hash": TARGET_WEIGHTS_HASH,
        "catalog_hash": TARGET_CATALOG_HASH,
        "selection_policy": {
            "luna": "all 10 deep-valid battles",
            "minimax_m3": (
                f"{per_outcome} wins + {per_outcome} losses; highest-turn completed "
                "battles in [12,60]"
            ),
            "training_weight": "lower learning rate before authoritative human replay",
        },
        "battles": len(selected),
        "luna_battles": len(luna),
        "minimax_battles": len(selected_minimax),
        "rows": selected,
        "frozen_group_dirs": [str(path) for path in frozen_groups],
    }
    _atomic_json(output, payload)
    return frozen_groups


def _chunk(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[index : index + size] for index in range(0, len(paths), size)]


def _clear_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--human-freeze", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--luna-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human-battles-per-shard", type=int, default=24)
    parser.add_argument("--semi-battles-per-shard", type=int, default=24)
    parser.add_argument("--minimax-battles", type=int, default=50)
    parser.add_argument("--human-lr", type=float, default=2.5e-5)
    parser.add_argument("--semi-lr", type=float, default=6.25e-6)
    parser.add_argument("--minibatch-size", type=int, default=128)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    freeze = args.human_freeze.resolve()
    sessions = args.sessions.resolve()
    luna_manifest = args.luna_manifest.resolve()
    output = args.output.resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidate = checkpoints / "extra_lr_v5_phaseC_candidate_h299.npz"
    temporary = checkpoints / ".candidate-next.npz"
    progress_path = output / "training_progress.json"
    semi_selection_path = output / "semi_synthetic_selection.json"

    validation = json.loads((freeze / "validation_report.json").read_text())
    if validation.get("checked") != 299 or validation.get("ok") != 299:
        raise RuntimeError(f"human freeze failed validation gate: {validation}")
    human_shards = _human_shards(
        freeze, args.human_battles_per_shard, output_dir=output
    )
    semi_groups = _select_semi_synthetic(
        sessions,
        luna_manifest,
        minimax_battles=args.minimax_battles,
        output=semi_selection_path,
    )
    lanes = [
        *[
            {
                "key": f"semi-{index:02d}",
                "lane": "semi_synthetic_llm",
                "groups": shard,
                "sources": ("llm",),
                "lr": args.semi_lr,
            }
            for index, shard in enumerate(
                _chunk(semi_groups, args.semi_battles_per_shard), 1
            )
        ],
        *[
            {
                "key": f"human-{index:02d}",
                "lane": "human_authoritative",
                "groups": shard,
                "sources": ("human",),
                "lr": args.human_lr,
            }
            for index, shard in enumerate(human_shards, 1)
        ],
    ]
    selected_semi_count = len(
        json.loads(semi_selection_path.read_text())["rows"]
    )
    if sum(len(lane["groups"]) for lane in lanes if lane["lane"] == "semi_synthetic_llm") != selected_semi_count:
        raise RuntimeError("semi-synthetic shard accounting mismatch")

    progress: dict[str, Any]
    if progress_path.exists() and candidate.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {
            "schema": "extra_lr_phase_c_training_progress_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_sha256": _sha256(checkpoint),
            "human_freeze": str(freeze),
            "human_selection_sha256": _sha256(freeze / "selection_manifest.json"),
            "semi_selection": str(semi_selection_path),
            "semi_selection_sha256": _sha256(semi_selection_path),
            "config": {
                "human_lr": args.human_lr,
                "semi_lr": args.semi_lr,
                "minibatch_size": args.minibatch_size,
                "epochs": 1,
                "max_grad_norm": 1.0,
                "freeze_faithful": True,
                "train_value_head": True,
            },
            "completed": [],
            "updates": [],
        }
    completed = set(progress["completed"])
    # A long battle can make the dense (steps x games x 601 x 171) tensor much
    # larger than a battle-count estimate suggests.  Resuming with a smaller
    # shard size must not replay already completed groups merely because shard
    # numbering changed.  Filter by the persisted exact group accounting and
    # give every remaining shard a content-derived key.
    completed_groups: dict[str, set[str]] = defaultdict(set)
    for update in progress["updates"]:
        completed_groups[str(update["lane"])].update(update["groups"])
    remaining_lanes: list[dict[str, Any]] = []
    for lane in lanes:
        remaining_groups = [
            path
            for path in lane["groups"]
            if path.name not in completed_groups[lane["lane"]]
        ]
        if not remaining_groups:
            continue
        lane = {**lane, "groups": remaining_groups}
        content = ",".join(path.name for path in remaining_groups)
        lane["key"] = (
            f"{lane['lane']}-"
            f"{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        )
        remaining_lanes.append(lane)
    lanes = remaining_lanes
    current = candidate if candidate.exists() and completed else checkpoint
    trainer = AwacCrrReplay()

    for lane in lanes:
        if lane["key"] in completed:
            continue
        started = time.time()
        policy_fn = make_policy_fn_from_checkpoint(current)
        batch = build_offline_replay_batch(
            policy_fn,
            group_dirs=lane["groups"],
            strict=True,
            accepted_decision_sources=lane["sources"],
        )
        if (
            lane["lane"] == "semi_synthetic_llm"
            and batch.num_games != len(lane["groups"])
        ):
            raise RuntimeError(
                f"{lane['key']} exact-count gate failed: selected="
                f"{len(lane['groups'])}, ingested={batch.num_games}"
            )
        if batch.num_games <= 0 or batch.skipped_rows:
            raise RuntimeError(
                f"{lane['key']} ingestion gate failed: games={batch.num_games}, "
                f"skipped={batch.skipped_rows}"
            )
        metrics = trainer.run(
            batch,
            checkpoint_path=current,
            epochs=1,
            minibatch_size=args.minibatch_size,
            lr=lane["lr"],
            max_grad_norm=1.0,
            freeze_faithful=True,
            train_value_head=True,
            seed=29250 + len(progress["updates"]),
            save_checkpoint_path=temporary,
        )
        if metrics.status != "trained" or not temporary.exists():
            raise RuntimeError(f"{lane['key']} training failed: {metrics}")
        os.replace(temporary, candidate)
        current = candidate
        record = {
            "key": lane["key"],
            "lane": lane["lane"],
            "groups": [path.name for path in lane["groups"]],
            "games": batch.num_games,
            "rows": batch.num_rows,
            "mana_draw_rows": batch.mana_draw_row_count,
            "skipped_rows": batch.skipped_rows,
            "lr": lane["lr"],
            "elapsed_seconds": round(time.time() - started, 3),
            "metrics": {
                "loss_before": metrics.extra.get("loss_before"),
                "loss_after": metrics.extra.get("loss_after"),
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "mana_draw_bce": metrics.mana_draw_bce,
                "entropy": metrics.entropy,
                "approx_kl": metrics.approx_kl,
                "clip_fraction": metrics.clip_fraction,
                "num_updates": metrics.num_updates,
            },
            "candidate_sha256": _sha256(candidate),
        }
        progress["completed"].append(lane["key"])
        progress["updates"].append(record)
        progress["candidate"] = str(candidate)
        progress["candidate_sha256"] = record["candidate_sha256"]
        _atomic_json(progress_path, progress)
        print("PHASE_C_UPDATE", json.dumps(record, ensure_ascii=False), flush=True)
        del batch, policy_fn, metrics
        _clear_mlx()

    progress["status"] = "trained"
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    progress["totals"] = {
        "games": sum(row["games"] for row in progress["updates"]),
        "rows": sum(row["rows"] for row in progress["updates"]),
        "mana_draw_rows": sum(row["mana_draw_rows"] for row in progress["updates"]),
        "optimizer_updates": sum(
            row["metrics"]["num_updates"] for row in progress["updates"]
        ),
    }
    human_games = sum(
        row["games"]
        for row in progress["updates"]
        if row["lane"] == "human_authoritative"
    )
    semi_games = sum(
        row["games"]
        for row in progress["updates"]
        if row["lane"] == "semi_synthetic_llm"
    )
    if human_games != 299 or semi_games != selected_semi_count:
        raise RuntimeError(
            f"final lane accounting failed: human={human_games}/299, "
            f"semi={semi_games}/{selected_semi_count}"
        )
    _atomic_json(output / "run_summary.json", progress)
    _atomic_json(progress_path, progress)
    print("PHASE_C_RESULT", json.dumps(progress["totals"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
