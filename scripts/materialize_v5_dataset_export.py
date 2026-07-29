#!/usr/bin/env python3
"""Materialize a production V5 NDJSON export into the canonical disk layout.

The admin export is intentionally transport-oriented: its first NDJSON record
is a header and every following record is one complete battle bundle.  Offline
training consumes the files-only ``rlhf_v5_storage_v1`` layout instead.  This
tool bridges the two formats without ever publishing a partial dataset.

Example::

    python scripts/materialize_v5_dataset_export.py \
      extraarena_v5_dataset_20260728_120000.jsonl \
      rlhf_env/sessions/production-human-2026-07-28
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, TextIO
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlhf_env.components.v5_trace_validate import (
    validate_v5_metronome_contract,
    validate_v5_timestamp_contract,
    validate_v5_trace,
)


EXPORT_FORMAT = "extraarena_v5_dataset_export_v1"
STORAGE_SCHEMA = "rlhf_v5_storage_v1"
MATERIALIZED_FORMAT = "extraarena_v5_materialized_dataset_v1"
PSEUDONYMIZED_RECORD_ID_SCHEME = "random_per_export_record_ids_v1"
TERMINAL_STATUSES = frozenset({"p1_win", "p2_win", "draw", "stalemate"})
MANIFEST_STATUS = {
    "p1_win": "P1_WIN",
    "p2_win": "P2_WIN",
    "draw": "DRAW",
    "stalemate": "STALEMATE",
}
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PSEUDONYMIZED_RECORD_ID_RE = re.compile(r"^record_[0-9a-f]{32}$")
REQUIRED_TIMING_KEYS = frozenset(
    {
        "human_decision_time_ms",
        "human_decision_time_raw_ms",
        "decision_time_censored",
        "decision_censor_reason",
        "metronome_prediction_ms",
        "metronome_applied_ms",
        "metronome_fallback_used",
    }
)
HEADER_REQUIRED_KEYS = frozenset(
    {
        "record_type",
        "format",
        "format_version",
        "storage_schema",
        "created_at",
        "privacy",
        "include_players",
        "record_id_scheme",
        "days",
        "limit_battles",
        "battle_count",
        "skipped_invalid",
    }
)
HEADER_ALLOWED_KEYS = HEADER_REQUIRED_KEYS | frozenset(
    {
        "current_catalog_hash",
        "current_card_count",
        "notes",
    }
)
BATTLE_ENVELOPE_KEYS = frozenset(
    {
        "record_type",
        "battle_id",
        "storage_schema",
        "status",
        "finished_at",
        "meta",
        "turns",
        "actions",
    }
)
PARTICIPANT_ID_KEYS = frozenset(
    {
        "userid",
        "playerid",
        "participantid",
        "actoruserid",
        "actinguserid",
        "currentturnownerid",
        "winneruserid",
        "loseruserid",
        "p1userid",
        "p2userid",
        "owneruserid",
        "sourceuserid",
        "targetuserid",
        "actorid",
        "ownerid",
        "currentplayer",
        "currentplayerid",
        "startingplayerid",
        "winnerid",
        "loserid",
    }
)
PARTICIPANT_ID_LIST_KEYS = frozenset(
    {
        "playerids",
        "readyuserids",
        "waitingforuserids",
    }
)
SENSITIVE_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearertoken",
        "bottoken",
        "clientsecret",
        "connectionstring",
        "cookie",
        "credential",
        "credentials",
        "databasepassword",
        "databasedsn",
        "databaseurl",
        "dsn",
        "idtoken",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessioncookie",
        "sessiontoken",
        "token",
    }
)
RAW_IDENTITY_FRAGMENTS = (
    "telegramid",
    "telegramuser",
    "chatid",
    "username",
    "firstname",
    "lastname",
    "phone",
    "email",
)
CREDENTIAL_URI_RE = re.compile(
    r"(?i)^[a-z][a-z0-9+.-]*://[^/\s@:]+:[^/\s@]+@"
)


class MaterializationError(ValueError):
    """The export is not safe and complete enough to publish."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_component(value: Any, *, field: str) -> str:
    component = str(value or "")
    if (
        not SAFE_COMPONENT_RE.fullmatch(component)
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
    ):
        raise MaterializationError(
            f"{field}={component!r} is not a safe path component"
        )
    return component


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _validate_private_bundle(
    value: Any,
    *,
    battle_id: str,
    path: str = "bundle",
) -> None:
    """Reject raw identities and credentials before anything is published."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if (
                normalized in SENSITIVE_FIELD_NAMES
                or normalized.endswith("password")
                or normalized.endswith("secret")
                or normalized.endswith("apikey")
                or normalized.endswith("authtoken")
                or normalized.endswith("accesstoken")
                or normalized.endswith("refreshtoken")
            ):
                raise MaterializationError(
                    f"battle {battle_id}: {child_path} contains a "
                    "forbidden sensitive field"
                )
            if (
                normalized in PARTICIPANT_ID_KEYS
                and nested not in (None, 1, 2)
            ):
                raise MaterializationError(
                    f"battle {battle_id}: {child_path} contains a raw "
                    "participant identity"
                )
            if normalized in PARTICIPANT_ID_LIST_KEYS and (
                not isinstance(nested, list)
                or any(item not in (1, 2) for item in nested)
            ):
                raise MaterializationError(
                    f"battle {battle_id}: {child_path} contains raw "
                    "participant identities"
                )
            if (
                "userid" in normalized
                and normalized not in PARTICIPANT_ID_KEYS
            ) or any(
                fragment in normalized
                for fragment in RAW_IDENTITY_FRAGMENTS
            ):
                raise MaterializationError(
                    f"battle {battle_id}: {child_path} contains a "
                    "forbidden identity field"
                )
            _validate_private_bundle(
                nested,
                battle_id=battle_id,
                path=child_path,
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_private_bundle(
                nested,
                battle_id=battle_id,
                path=f"{path}[{index}]",
            )
    elif isinstance(value, str) and CREDENTIAL_URI_RE.match(value.strip()):
        raise MaterializationError(
            f"battle {battle_id}: {path} contains a credential URI"
        )


def _load_record(raw: str, *, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MaterializationError(
            f"line {line_number}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MaterializationError(
            f"line {line_number}: NDJSON record must be an object"
        )
    return value


def _iter_records(handle: TextIO) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_number, raw in enumerate(handle, 1):
        if not raw.strip():
            continue
        yield line_number, _load_record(raw, line_number=line_number)


def _validate_header(header: Mapping[str, Any], *, line_number: int) -> int:
    if not HEADER_REQUIRED_KEYS.issubset(header) or not set(
        header
    ).issubset(HEADER_ALLOWED_KEYS):
        raise MaterializationError(
            f"line {line_number}: header fields do not match the "
            "V5 transport allowlist"
        )
    if header.get("record_type") != "header":
        raise MaterializationError(
            f"line {line_number}: first record must be record_type='header'"
        )
    if header.get("format") != EXPORT_FORMAT:
        raise MaterializationError(
            f"line {line_number}: format must be {EXPORT_FORMAT!r}"
        )
    if header.get("format_version") != 1:
        raise MaterializationError(
            f"line {line_number}: unsupported format_version="
            f"{header.get('format_version')!r}"
        )
    if header.get("storage_schema") != STORAGE_SCHEMA:
        raise MaterializationError(
            f"line {line_number}: storage_schema must be {STORAGE_SCHEMA!r}"
        )
    if (
        header.get("privacy") != "side_pseudonyms_p1_1_p2_2"
        or header.get("include_players") is not False
        or header.get("record_id_scheme")
        != PSEUDONYMIZED_RECORD_ID_SCHEME
    ):
        raise MaterializationError(
            f"line {line_number}: V5 training export must use fixed side "
            "pseudonyms, include_players=false and export-local random "
            "record IDs"
        )
    if "notes" in header and header.get("notes") != (
        "Each following line is one complete terminal "
        "rlhf_v5_storage_v1 battle bundle."
    ):
        raise MaterializationError(
            f"line {line_number}: non-canonical notes field"
        )
    battle_count = header.get("battle_count")
    if (
        not isinstance(battle_count, int)
        or isinstance(battle_count, bool)
        or battle_count < 1
    ):
        raise MaterializationError(
            f"line {line_number}: battle_count must be a positive integer"
        )
    return battle_count


def _parse_timestamp(value: Any, *, field: str, battle_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MaterializationError(
            f"battle {battle_id}: meta.{field} is required for TimeStamp"
        )
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterializationError(
            f"battle {battle_id}: meta.{field} must be a valid timezone-aware "
            "ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MaterializationError(
            f"battle {battle_id}: meta.{field} must be a valid timezone-aware "
            "ISO-8601 timestamp"
        )
    return parsed.astimezone(timezone.utc)


def _require_timestamp_contract(meta: Mapping[str, Any], *, battle_id: str) -> None:
    started_at = _parse_timestamp(
        meta.get("started_at"),
        field="started_at",
        battle_id=battle_id,
    )
    finished_at = _parse_timestamp(
        meta.get("finished_at"),
        field="finished_at",
        battle_id=battle_id,
    )
    if finished_at < started_at:
        raise MaterializationError(
            f"battle {battle_id}: meta.finished_at precedes meta.started_at"
        )

    duration = meta.get("duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        raise MaterializationError(
            f"battle {battle_id}: meta.duration_seconds must be finite and >= 0"
        )
    duration_value = float(duration)
    wall_duration = (finished_at - started_at).total_seconds()
    # The authoritative label is monotonic, while the ISO timestamps use the
    # wall clock. Permit ordinary NTP drift but reject a corrupted label that
    # is orders of magnitude away from the recorded battle interval.
    duration_tolerance = max(60.0, 0.10 * max(duration_value, wall_duration))
    if abs(duration_value - wall_duration) > duration_tolerance:
        raise MaterializationError(
            f"battle {battle_id}: meta.duration_seconds is inconsistent with "
            "started_at/finished_at"
        )

    start_metadata = meta.get("start_metadata")
    if (
        not isinstance(start_metadata, dict)
        or start_metadata.get("client_ready_anchored") is not True
    ):
        raise MaterializationError(
            f"battle {battle_id}: TimeStamp start must be anchored to client_ready"
        )

    features = meta.get("timestamp_features")
    required_features = {
        "p1_deck_size",
        "p2_deck_size",
        "starting_player",
        "duration_seconds",
        "turns",
    }
    if not isinstance(features, dict):
        raise MaterializationError(
            f"battle {battle_id}: meta.timestamp_features must be an object"
        )
    missing = sorted(required_features - set(features))
    if missing:
        raise MaterializationError(
            f"battle {battle_id}: meta.timestamp_features missing {missing}"
        )
    feature_duration = features.get("duration_seconds")
    if (
        not isinstance(feature_duration, (int, float))
        or isinstance(feature_duration, bool)
        or not math.isfinite(float(feature_duration))
        or abs(float(feature_duration) - duration_value) > 1e-3
    ):
        raise MaterializationError(
            f"battle {battle_id}: timestamp_features.duration_seconds must "
            "match meta.duration_seconds"
        )

    feature_turns = features.get("turns")
    meta_turns = meta.get("turns")
    if (
        not isinstance(feature_turns, int)
        or isinstance(feature_turns, bool)
        or feature_turns < 1
        or not isinstance(meta_turns, int)
        or isinstance(meta_turns, bool)
        or meta_turns != feature_turns
    ):
        raise MaterializationError(
            f"battle {battle_id}: timestamp_features.turns must be a positive "
            "integer matching meta.turns"
        )

    starting_player = features.get("starting_player")
    if starting_player not in {"p1", "p2"} or (
        meta.get("starting_player") is not None
        and meta.get("starting_player") != starting_player
    ):
        raise MaterializationError(
            f"battle {battle_id}: timestamp_features.starting_player is invalid "
            "or inconsistent with meta.starting_player"
        )

    for player in ("p1", "p2"):
        deck = meta.get(f"{player}_deck")
        if not isinstance(deck, list) or not deck:
            raise MaterializationError(
                f"battle {battle_id}: meta.{player}_deck must be non-empty"
            )
        deck_size = features.get(f"{player}_deck_size")
        if (
            not isinstance(deck_size, int)
            or isinstance(deck_size, bool)
            or deck_size != len(deck)
        ):
            raise MaterializationError(
                f"battle {battle_id}: timestamp_features.{player}_deck_size "
                f"must match meta.{player}_deck"
            )


def _require_metronome_contract(
    actions: list[dict[str, Any]], *, battle_id: str
) -> None:
    for position, action in enumerate(actions, 1):
        missing = sorted(REQUIRED_TIMING_KEYS - set(action))
        if missing:
            raise MaterializationError(
                f"battle {battle_id} action {position}: timing fields missing {missing}"
            )

        source = action.get("decision_source")
        control = action.get("control_source")
        action_type = action.get("action_type")
        if source not in {"bot", "rl", "llm"}:
            continue
        if control == "timeout" or action_type in {"surrender", "draw", "stalemate"}:
            continue

        fallback = action.get("metronome_fallback_used")
        predicted = action.get("metronome_prediction_ms")
        applied = action.get("metronome_applied_ms")
        if not isinstance(fallback, bool):
            raise MaterializationError(
                f"battle {battle_id} action {position}: automated action must "
                "record metronome_fallback_used as bool"
            )
        if (
            not isinstance(applied, (int, float))
            or isinstance(applied, bool)
            or not math.isfinite(float(applied))
            or float(applied) < 0.0
        ):
            raise MaterializationError(
                f"battle {battle_id} action {position}: automated action must "
                "record a finite metronome_applied_ms"
            )
        if not fallback and (
            not isinstance(predicted, (int, float))
            or isinstance(predicted, bool)
            or not math.isfinite(float(predicted))
            or float(predicted) < 0.0
        ):
            raise MaterializationError(
                f"battle {battle_id} action {position}: non-fallback automated "
                "action must record metronome_prediction_ms"
            )


def _require_pseudonymized_record_ids(
    value: Any,
    *,
    battle_id: str,
) -> None:
    """Fail closed if a declared battle/match ID is not an opaque export ID."""

    def check(candidate: Any, *, field: str) -> None:
        if candidate is None:
            return
        if (
            not isinstance(candidate, str)
            or PSEUDONYMIZED_RECORD_ID_RE.fullmatch(candidate) is None
        ):
            raise MaterializationError(
                f"battle {battle_id}: {field} must be an export-local "
                "opaque record identifier"
            )

    def walk(nested: Any, path: str) -> None:
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                key_text = str(key)
                normalized_key = re.sub(
                    r"[^a-z0-9]",
                    "",
                    key_text.strip().lower(),
                )
                child_path = f"{path}.{key_text}"
                if normalized_key.endswith(("battleid", "matchid")):
                    check(child, field=child_path)
                elif normalized_key.endswith(("battleids", "matchids")):
                    if not isinstance(child, list):
                        raise MaterializationError(
                            f"battle {battle_id}: {child_path} must be a list "
                            "of export-local opaque record identifiers"
                        )
                    for index, candidate in enumerate(child):
                        check(
                            candidate,
                            field=f"{child_path}[{index}]",
                        )
                else:
                    walk(child, child_path)
        elif isinstance(nested, list):
            for index, child in enumerate(nested):
                walk(child, f"{path}[{index}]")

    walk(value, "bundle")


def _validate_bundle(
    record: Mapping[str, Any],
    *,
    line_number: int,
    seen_battle_ids: set[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(record) != BATTLE_ENVELOPE_KEYS:
        raise MaterializationError(
            f"line {line_number}: battle envelope fields do not match "
            "the V5 transport allowlist"
        )
    if record.get("record_type") != "battle":
        raise MaterializationError(
            f"line {line_number}: expected record_type='battle'"
        )
    battle_id = _safe_component(record.get("battle_id"), field="battle_id")
    if PSEUDONYMIZED_RECORD_ID_RE.fullmatch(battle_id) is None:
        raise MaterializationError(
            f"line {line_number}: battle_id must be an export-local opaque "
            "record identifier"
        )
    if battle_id in seen_battle_ids:
        raise MaterializationError(f"line {line_number}: duplicate battle_id={battle_id!r}")
    if record.get("storage_schema") != STORAGE_SCHEMA:
        raise MaterializationError(
            f"battle {battle_id}: storage_schema must be {STORAGE_SCHEMA!r}"
        )

    status = record.get("status")
    if status not in TERMINAL_STATUSES:
        raise MaterializationError(
            f"battle {battle_id}: non-terminal status={status!r}"
        )
    finished_at = record.get("finished_at")
    if not isinstance(finished_at, str) or not finished_at.strip():
        raise MaterializationError(
            f"battle {battle_id}: finished_at is required"
        )

    meta = record.get("meta")
    turns = record.get("turns")
    actions = record.get("actions")
    if not isinstance(meta, dict):
        raise MaterializationError(f"battle {battle_id}: meta must be an object")
    if not isinstance(turns, list) or not turns or not all(
        isinstance(row, dict) for row in turns
    ):
        raise MaterializationError(
            f"battle {battle_id}: turns must be a non-empty list of objects"
        )
    if not isinstance(actions, list) or not actions or not all(
        isinstance(row, dict) for row in actions
    ):
        raise MaterializationError(
            f"battle {battle_id}: actions must be a non-empty list of objects"
        )

    if meta.get("schema_version") != STORAGE_SCHEMA:
        raise MaterializationError(
            f"battle {battle_id}: meta.schema_version must be {STORAGE_SCHEMA!r}"
        )
    if str(meta.get("battle_id") or "") != battle_id:
        raise MaterializationError(
            f"battle {battle_id}: meta.battle_id does not match the bundle"
        )
    _require_pseudonymized_record_ids(
        {
            "battle_id": battle_id,
            "meta": meta,
            "turns": turns,
            "actions": actions,
        },
        battle_id=battle_id,
    )
    if meta.get("status") != status:
        raise MaterializationError(
            f"battle {battle_id}: meta.status={meta.get('status')!r} "
            f"does not match status={status!r}"
        )
    _validate_private_bundle(
        {
            "meta": meta,
            "turns": turns,
            "actions": actions,
        },
        battle_id=battle_id,
    )

    actor_types = (meta.get("p1_actor_type"), meta.get("p2_actor_type"))
    if "human" not in actor_types:
        raise MaterializationError(
            f"battle {battle_id}: production collection only accepts "
            "human-vs-bot or human-vs-human battles"
        )

    seqs = [row.get("seq") for row in actions]
    if seqs != list(range(1, len(actions) + 1)):
        raise MaterializationError(
            f"battle {battle_id}: action seq must be contiguous 1..N"
        )

    auxiliary_issues = [
        *validate_v5_timestamp_contract(meta),
        *validate_v5_metronome_contract(actions),
    ]
    if auxiliary_issues:
        raise MaterializationError(
            f"battle {battle_id}: {auxiliary_issues[0]}"
        )
    seen_battle_ids.add(battle_id)
    return battle_id, dict(meta), list(turns), list(actions)


def _write_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _collection_class(meta: Mapping[str, Any]) -> str:
    p1_type = meta.get("p1_actor_type")
    p2_type = meta.get("p2_actor_type")
    if p1_type == "human" and p2_type == "human":
        return "human-vs-human"
    return "human-vs-bot"


def _battle_result(
    *,
    battle_id: str,
    meta: Mapping[str, Any],
    turns: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    p1_uid = meta.get("p1_user_id")
    p2_uid = meta.get("p2_user_id")
    winner = meta.get("winner_user_id")
    loser = None
    if winner == p1_uid:
        loser = p2_uid
    elif winner == p2_uid:
        loser = p1_uid
    return {
        "battle_id": battle_id,
        "winner_user_id": winner,
        "loser_user_id": loser,
        "status": MANIFEST_STATUS[status],
        "turns": int(meta.get("turns") or len(turns)),
        "duration_seconds": float(meta["duration_seconds"]),
        "battle_tag": str(
            meta.get("battle_tag") or _collection_class(meta)
        ),
        "collection_class": _collection_class(meta),
        "p1_actor_type": meta.get("p1_actor_type"),
        "p2_actor_type": meta.get("p2_actor_type"),
        "v5_dir": f"battles/{battle_id}/v5",
        "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
        "v5_trace_ok": True,
        "validation_scope": "v5_trace_without_legacy_battle_log",
        "finished_at": meta.get("finished_at"),
    }


def _build_manifest(
    *,
    group_id: str,
    header: Mapping[str, Any],
    input_name: str,
    battle_results: list[dict[str, Any]],
) -> dict[str, Any]:
    p1_wins = sum(row["status"] == "P1_WIN" for row in battle_results)
    p2_wins = sum(row["status"] == "P2_WIN" for row in battle_results)
    draws = sum(row["status"] in {"DRAW", "STALEMATE"} for row in battle_results)
    total = len(battle_results)
    return {
        "manifest_version": "1.0",
        "schema_version": STORAGE_SCHEMA,
        "storage_schema": STORAGE_SCHEMA,
        "materialized_format": MATERIALIZED_FORMAT,
        "group_id": group_id,
        "created_at": str(header.get("created_at") or _utc_now_iso()),
        "finished_at": _utc_now_iso(),
        "spec": {
            "source_format": EXPORT_FORMAT,
            "source_file": input_name,
            "privacy": header.get("privacy"),
            "include_players": bool(header.get("include_players", False)),
            "record_id_scheme": header.get("record_id_scheme"),
            "days": header.get("days"),
            "limit_battles": header.get("limit_battles"),
            "source_skipped_invalid": int(header.get("skipped_invalid") or 0),
            "collection_classes": ["human-vs-bot", "human-vs-human"],
        },
        "env": {
            "materializer": "scripts/materialize_v5_dataset_export.py",
            "deep_validator": "rlhf_env.components.v5_trace_validate.validate_v5_trace",
            "validation_scope": "v5_trace_without_legacy_battle_log",
        },
        "results": {
            "battles_planned": total,
            "battles_finished": total,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "winrate_p1": round(p1_wins / total, 4),
            "winrate_p2": round(p2_wins / total, 4),
            "avg_turns": round(
                sum(row["turns"] for row in battle_results) / total, 2
            ),
            "avg_duration_seconds": round(
                sum(row["duration_seconds"] for row in battle_results) / total,
                3,
            ),
        },
        "battle_ids": [row["battle_id"] for row in battle_results],
        "battles_results": battle_results,
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publish_directory(temp_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.is_symlink():
        raise MaterializationError("refusing to replace a symlink output path")
    if output_dir.exists() and not overwrite:
        raise MaterializationError(
            f"output already exists: {output_dir} (use --overwrite)"
        )

    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid4().hex}"
        )
        output_dir.rename(backup)
    try:
        temp_dir.rename(output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        _remove_path(backup)


def materialize_export(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    group_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize *input_path* and return the published manifest.

    The complete dataset is built and deep-validated in a sibling temporary
    directory.  The destination becomes visible only after every battle has
    passed validation.
    """

    source = Path(input_path)
    target = Path(output_dir)
    if not source.is_file():
        raise MaterializationError(f"input export not found: {source}")
    source_resolved = source.resolve()
    target_resolved = target.resolve(strict=False)
    if source_resolved == target_resolved or target_resolved in source_resolved.parents:
        raise MaterializationError(
            "input export must not be inside the output directory"
        )
    if target.is_symlink():
        raise MaterializationError("refusing to write through an output symlink")
    if target.exists() and not overwrite:
        raise MaterializationError(
            f"output already exists: {target} (use --overwrite)"
        )

    resolved_group_id = _safe_component(
        group_id
        or f"production_v5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        field="group_id",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=str(target.parent),
        )
    )
    try:
        (temp_dir / "battles").mkdir()
        with source.open("r", encoding="utf-8") as handle:
            records = iter(_iter_records(handle))
            try:
                header_line, header = next(records)
            except StopIteration as exc:
                raise MaterializationError("export is empty") from exc
            expected_battles = _validate_header(
                header, line_number=header_line
            )

            seen_battle_ids: set[str] = set()
            battle_results: list[dict[str, Any]] = []
            for line_number, record in records:
                battle_id, meta, turns, actions = _validate_bundle(
                    record,
                    line_number=line_number,
                    seen_battle_ids=seen_battle_ids,
                )
                v5_dir = temp_dir / "battles" / battle_id / "v5"
                v5_dir.mkdir(parents=True)
                _write_json(v5_dir / "meta.json", meta)
                _write_jsonl(v5_dir / "turns.jsonl", turns)
                _write_jsonl(v5_dir / "actions.jsonl", actions)

                report = validate_v5_trace(v5_dir)
                if not report.get("ok"):
                    issues = "; ".join(str(issue) for issue in report["issues"][:8])
                    if len(report.get("issues") or []) > 8:
                        issues += "; ..."
                    raise MaterializationError(
                        f"battle {battle_id}: canonical V5 validation failed: {issues}"
                    )
                battle_results.append(
                    _battle_result(
                        battle_id=battle_id,
                        meta=meta,
                        turns=turns,
                        status=str(record["status"]),
                    )
                )

        if len(battle_results) != expected_battles:
            raise MaterializationError(
                f"header battle_count={expected_battles}, "
                f"but found {len(battle_results)} battle records"
            )

        manifest = _build_manifest(
            group_id=resolved_group_id,
            header=header,
            input_name=source.name,
            battle_results=battle_results,
        )
        _write_json(temp_dir / "manifest.json", manifest)
        _publish_directory(temp_dir, target, overwrite=overwrite)
        return manifest
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize extraarena_v5_dataset_export_v1 NDJSON into "
            "canonical rlhf_v5_storage_v1 files."
        )
    )
    parser.add_argument("input", type=Path, help="NDJSON admin export")
    parser.add_argument("output", type=Path, help="new dataset/group directory")
    parser.add_argument(
        "--group-id",
        help="safe manifest group id (default: generated production_v5 timestamp)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output directory after validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = materialize_export(
            args.input,
            args.output,
            group_id=args.group_id,
            overwrite=args.overwrite,
        )
    except (MaterializationError, OSError) as exc:
        print(f"materialization failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "group_id": manifest["group_id"],
                "battle_count": len(manifest["battle_ids"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
