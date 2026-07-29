"""Safe, training-oriented dataset administration for the ExtraRLHF MCP.

The arena MCP historically managed only headless battle groups.  Production
collection now also emits V5 battle bundles, Nemesis records and ReturnClock
session telemetry.  This module provides one deliberately narrow boundary for
those datasets:

* every readable/writable artifact is confined to ``datasets_dir``;
* production exports are pseudonymized and published atomically with mode 0600;
* validation is schema-aware and returns split/readiness diagnostics;
* no tool accepts a raw privacy salt or an ``include_players`` escape hatch.

The helpers are independent from the JSON-RPC layer so they can be exercised
directly by tests and reused by future trainers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping

from core.nemesis_dataset import (
    NEMESIS_EXPORT_FORMAT,
    NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME,
    NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME,
    NEMESIS_SCHEMA,
    validate_nemesis_record,
    write_nemesis_export,
)
from core.v5_dataset import (
    V5_ACTIONS_VERSION,
    V5_CARD_PARAMS_SCHEMA,
    V5_CARD_SHAPE_VERSION,
    V5_DECK_PARAMS_SCHEMA,
    V5_OBS_VERSION,
    V5_STORAGE_SCHEMA,
    V5_VISIBILITY,
)
from infrastructure.returnclock_dataset import (
    DATASET_FORMAT as RETURNCLOCK_DATASET_FORMAT,
    DATASET_VERSION as RETURNCLOCK_DATASET_VERSION,
    FEATURE_COLUMNS as RETURNCLOCK_FEATURE_COLUMNS,
)
from scripts.export_nemesis_dataset import _load_records as load_nemesis_records
from scripts.materialize_v5_dataset_export import (
    EXPORT_FORMAT as V5_EXPORT_FORMAT,
    MATERIALIZED_FORMAT as V5_MATERIALIZED_FORMAT,
    MaterializationError,
    materialize_export,
)


_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_USER_HASH_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PSEUDONYMIZED_RECORD_ID_RE = re.compile(r"^record_[0-9a-f]{32}$")
_NATIVE_OPAQUE_BATTLE_ID_RE = re.compile(r"^b_[0-9a-f]{10,32}$")
_NATIVE_OPAQUE_MATCH_ID_RE = re.compile(r"^m_[0-9a-f]{12,32}$")
_V5_PSEUDONYMIZED_RECORD_ID_SCHEME = "random_per_export_record_ids_v1"
_V5_NATIVE_RECORD_ID_SCHEME = "native_opaque_record_ids_v1"
_V5_PARTICIPANT_ID_KEYS = frozenset(
    {
        "user_id",
        "player_id",
        "participant_id",
        "actor_user_id",
        "acting_user_id",
        "current_turn_owner_id",
        "winner_user_id",
        "loser_user_id",
        "p1_user_id",
        "p2_user_id",
        "owner_user_id",
        "source_user_id",
        "target_user_id",
        "actor_id",
        "owner_id",
        "current_player",
        "current_player_id",
        "starting_player_id",
        "winner_id",
        "loser_id",
    }
)
_V5_PARTICIPANT_ID_LIST_KEYS = frozenset(
    {
        "player_ids",
        "ready_user_ids",
        "waiting_for_user_ids",
    }
)
_V5_PARTICIPANT_ID_KEYS_NORMALIZED = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower())
    for key in _V5_PARTICIPANT_ID_KEYS
)
_V5_PARTICIPANT_ID_LIST_KEYS_NORMALIZED = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower())
    for key in _V5_PARTICIPANT_ID_LIST_KEYS
)
_RETURNCLOCK_EXAMPLE_KEYS = frozenset(
    {
        "record_type",
        "dataset_format",
        "dataset_version",
        "user_id_hash",
        "prediction_cutoff_at",
        "features",
        "label",
        "post_cutoff",
    }
)
_RETURNCLOCK_POST_CUTOFF_KEYS = frozenset(
    {
        "notification_decision_count",
        "provider_accepted_count",
        "notification_sent_count",
        "notification_opened_count",
        "notification_channels",
        "treatment_arms",
        "assignments",
        "notification_attributed",
        "treatment_assigned",
        "organic_candidate",
    }
)
_RETURNCLOCK_ASSIGNMENT_KEYS = frozenset(
    {
        "experiment_id",
        "treatment_arm",
        "assignment_probability",
        "decision",
        "decision_source",
        "policy_version",
        "model_version",
    }
)
_RETURNCLOCK_HEADER_KEYS = frozenset(
    {
        "record_type",
        "format",
        "format_version",
        "generated_at",
        "dataset_end",
        "ingested_before",
        "cutoff_start",
        "horizon_hours",
        "min_analytics_version",
        "sessionization_gap_minutes",
        "feature_columns",
        "label_columns",
        "post_cutoff_namespace",
        "user_id_scheme",
        "pseudonymization_key_id",
        "privacy_note",
        "meaningful_session",
        "summary",
    }
)
_RETURNCLOCK_SPLIT_HEADER_KEYS = frozenset(
    {
        "split_name",
        "split_assignment_basis",
        "source_sha256",
        "source_summary",
    }
)
_RETURNCLOCK_SPLIT_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "source",
        "source_sha256",
        "pseudonymization_key_id",
        "assignment_basis",
        "strategy",
        "requested_fractions",
        "training_filter",
        "source_example_count",
        "example_count",
        "excluded_treated_count",
        "excluded_temporal_boundary_count",
        "user_count",
        "splits",
        "feature_columns",
        "post_cutoff_excluded_from_features",
    }
)
_RETURNCLOCK_SPLIT_ENTRY_KEYS = frozenset(
    {
        "file",
        "sha256",
        "example_count",
        "organic_example_count",
        "treated_example_count",
        "user_count",
        "first_cutoff_min",
        "first_cutoff_max",
        "row_cutoff_min",
        "row_cutoff_max",
    }
)
_NEMESIS_SPLIT_FORMAT = "extraarena_nemesis_split_v1"
_NEMESIS_SPLIT_NAMES = ("train", "validation", "test")
_NEMESIS_SPLIT_REGIMES = (
    "lite_deck_grouped",
    "standard_player_disjoint",
    "standard_chronological",
    "standard_deck_grouped",
)
_NEMESIS_SPLIT_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "created_at",
        "source",
        "source_sha256",
        "source_format",
        "source_schema_version",
        "source_battle_count",
        "privacy",
        "catalog",
        "requested_fractions",
        "algorithms",
        "exclusions",
        "artifacts",
        "feature_contract",
        "training_readiness",
    }
)
_NEMESIS_SPLIT_ENTRY_KEYS = frozenset(
    {
        "file",
        "sha256",
        "example_count",
        "group_count",
        "player_group_count",
        "feature_cutoff_min",
        "feature_cutoff_max",
    }
)
_NEMESIS_HEADER_KEYS = frozenset(
    {
        "record_type",
        "format",
        "format_version",
        "schema_version",
        "created_at",
        "battle_count",
        "identity_scheme",
        "include_players",
        "player_group_scheme",
        "record_id_scheme",
    }
)
_V5_MATERIALIZED_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "schema_version",
        "storage_schema",
        "materialized_format",
        "group_id",
        "created_at",
        "finished_at",
        "spec",
        "env",
        "results",
        "battle_ids",
        "battles_results",
    }
)
_V5_MATERIALIZED_SPEC_KEYS = frozenset(
    {
        "source_format",
        "source_file",
        "privacy",
        "include_players",
        "record_id_scheme",
        "days",
        "limit_battles",
        "source_skipped_invalid",
        "collection_classes",
        "current_catalog_hash",
        "current_card_count",
    }
)
_V5_MATERIALIZED_ENV_KEYS = frozenset(
    {
        "materializer",
        "deep_validator",
        "validation_scope",
    }
)
_V5_MATERIALIZED_RESULT_KEYS = frozenset(
    {
        "battles_planned",
        "battles_finished",
        "p1_wins",
        "p2_wins",
        "draws",
        "winrate_p1",
        "winrate_p2",
        "avg_turns",
        "avg_duration_seconds",
    }
)
_V5_MATERIALIZED_BATTLE_RESULT_KEYS = frozenset(
    {
        "battle_id",
        "winner_user_id",
        "loser_user_id",
        "status",
        "turns",
        "duration_seconds",
        "battle_tag",
        "collection_class",
        "p1_actor_type",
        "p2_actor_type",
        "v5_dir",
        "v5_meta_path",
        "v5_trace_ok",
        "validation_scope",
        "finished_at",
    }
)
_V5_STATUS_TO_MANIFEST = {
    "p1_win": "P1_WIN",
    "p2_win": "P2_WIN",
    "draw": "DRAW",
    "stalemate": "STALEMATE",
}
_DATASET_KINDS = frozenset(
    {
        "returnclock",
        "returnclock_split",
        "nemesis",
        "nemesis_split",
        "v5_export",
        "v5_materialized",
        "unknown",
    }
)


class DatasetToolboxError(ValueError):
    """A dataset operation is unsafe or violates a public contract."""


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatasetToolboxError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetToolboxError(
                    f"line {line_number}: dataset record must be an object"
                )
            yield line_number, value


def _first_record(path: Path) -> dict[str, Any]:
    try:
        return next(_iter_jsonl(path))[1]
    except StopIteration as exc:
        raise DatasetToolboxError("dataset is empty") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bundle(path: Path) -> str:
    """Hash an artifact directory by relative path and file content digest."""

    digest = hashlib.sha256()
    for item in sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        ),
        key=lambda candidate: str(candidate.relative_to(path)),
    ):
        relative = str(item.relative_to(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_canonical_utc_timestamp(value: Any) -> bool:
    parsed = _parse_utc_timestamp(value)
    if parsed is None or not isinstance(value, str):
        return False
    return value == parsed.isoformat().replace("+00:00", "Z")


def _finite_number(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _returnclock_feature_issues(
    features: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    def require_int(
        key: str,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int | None:
        value = features.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            issues.append(key)
            return None
        return value

    def require_nonnegative_optional(key: str) -> float | None:
        value = features.get(key)
        if value is None:
            return None
        parsed = _finite_number(value)
        if parsed is None or parsed < 0.0:
            issues.append(key)
            return None
        return parsed

    timezone_name = features.get("timezone")
    if (
        not isinstance(timezone_name, str)
        or not timezone_name.strip()
        or len(timezone_name) > 128
    ):
        issues.append("timezone")
    if not isinstance(features.get("timezone_known"), bool):
        issues.append("timezone_known")
    require_int("local_weekday", 0, 6)
    require_int("local_hour", 0, 23)
    for key in ("local_hour_sin", "local_hour_cos"):
        value = _finite_number(features.get(key))
        if value is None or not -1.000001 <= value <= 1.000001:
            issues.append(key)
    sessions = [
        require_int("sessions_1d"),
        require_int("sessions_7d"),
        require_int("sessions_28d"),
    ]
    if all(value is not None for value in sessions) and not (
        sessions[0] <= sessions[1] <= sessions[2]
    ):
        issues.append("sessions_windows_not_monotonic")
    for key in (
        "hours_since_previous_session",
        "median_gap_hours_28d",
        "gap_iqr_hours_28d",
    ):
        require_nonnegative_optional(key)
    recent_hours = features.get("recent_local_start_hours")
    if (
        not isinstance(recent_hours, list)
        or len(recent_hours) > 8
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 23
            for value in recent_hours
        )
    ):
        issues.append("recent_local_start_hours")
    for key in (
        "last_session_duration_seconds",
        "last_session_screen_count",
        "last_session_battles",
        "last_session_cases",
        "notifications_24h",
        "notifications_7d",
    ):
        require_int(key)
    if (
        isinstance(features.get("notifications_24h"), int)
        and not isinstance(features.get("notifications_24h"), bool)
        and isinstance(features.get("notifications_7d"), int)
        and not isinstance(features.get("notifications_7d"), bool)
        and features["notifications_24h"] > features["notifications_7d"]
    ):
        issues.append("notification_windows_not_monotonic")
    for key in ("last_session_source", "last_session_entrypoint"):
        value = features.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 128
        ):
            issues.append(key)
    if not isinstance(features.get("last_session_end_inferred"), bool):
        issues.append("last_session_end_inferred")
    return issues


_RAW_IDENTITY_FIELD_FRAGMENTS = (
    "telegram_id",
    "telegram_user",
    "chat_id",
    "username",
    "first_name",
    "last_name",
    "phone",
    "email",
)
_SENSITIVE_FIELD_NAMES = frozenset(
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
_CREDENTIAL_URI_RE = re.compile(
    r"(?i)^[a-z][a-z0-9+.-]*://[^/\s@:]+:[^/\s@]+@"
)


def _normalized_field_name(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _is_raw_identity_field(key: Any, *, allow_seat_user_id: bool) -> bool:
    key_text = _normalized_field_name(key)
    if key_text in {
        "useridhash",
        "useridscheme",
        "pseudonymizationkeyid",
    }:
        return False
    if not allow_seat_user_id and "userid" in key_text:
        return True
    return any(
        _normalized_field_name(fragment) in key_text
        for fragment in _RAW_IDENTITY_FIELD_FRAGMENTS
    )


def _contains_raw_user_id(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _is_raw_identity_field(key, allow_seat_user_id=False):
                return True
            if _contains_raw_user_id(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_user_id(item) for item in value)
    return False


def _v5_sensitive_data_issues(value: Any, *, path: str) -> list[str]:
    """Return paths containing credentials or secret-bearing fields.

    V5 traces deliberately allow forward-compatible metadata.  That makes an
    exact nested allowlist impractical, but it must not become a route for
    publishing connection strings, API credentials or authentication state.
    Only field paths are returned so validation reports cannot echo a secret.
    """

    issues: list[str] = []

    def walk(nested: Any, nested_path: str) -> None:
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                key_text = str(key)
                child_path = f"{nested_path}.{key_text}"
                normalized_key = _normalized_field_name(key)
                if (
                    normalized_key in _SENSITIVE_FIELD_NAMES
                    or normalized_key.endswith("password")
                    or normalized_key.endswith("secret")
                    or normalized_key.endswith("apikey")
                    or normalized_key.endswith("authtoken")
                    or normalized_key.endswith("accesstoken")
                    or normalized_key.endswith("refreshtoken")
                ):
                    issues.append(f"{child_path}_sensitive_field_forbidden")
                walk(child, child_path)
        elif isinstance(nested, list):
            for index, child in enumerate(nested):
                walk(child, f"{nested_path}[{index}]")
        elif isinstance(nested, str) and _CREDENTIAL_URI_RE.match(
            nested.strip()
        ):
            issues.append(f"{nested_path}_credential_uri_forbidden")

    walk(value, path)
    return issues


def _record_id_privacy_issues(
    value: Any,
    *,
    path: str,
    scheme: str,
) -> list[str]:
    """Validate all structurally declared battle/match IDs for one artifact."""

    issues: list[str] = []

    def is_valid(candidate: Any, normalized_key: str) -> bool:
        if candidate is None:
            return True
        if not isinstance(candidate, str):
            return False
        if scheme == _V5_PSEUDONYMIZED_RECORD_ID_SCHEME:
            return _PSEUDONYMIZED_RECORD_ID_RE.fullmatch(candidate) is not None
        if scheme == _V5_NATIVE_RECORD_ID_SCHEME:
            matcher = (
                _NATIVE_OPAQUE_MATCH_ID_RE
                if normalized_key.endswith("matchid")
                else _NATIVE_OPAQUE_BATTLE_ID_RE
            )
            return matcher.fullmatch(candidate) is not None
        return False

    def walk(nested: Any, nested_path: str) -> None:
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                key_text = str(key)
                normalized_key = _normalized_field_name(key)
                child_path = f"{nested_path}.{key_text}"
                if normalized_key.endswith(("battleid", "matchid")):
                    if not is_valid(child, normalized_key):
                        issues.append(
                            f"{child_path}_record_id_not_opaque"
                        )
                elif normalized_key.endswith(("battleids", "matchids")):
                    singular_key = normalized_key[:-1]
                    if not isinstance(child, list):
                        issues.append(
                            f"{child_path}_record_id_list_invalid"
                        )
                    else:
                        for index, candidate in enumerate(child):
                            if not is_valid(candidate, singular_key):
                                issues.append(
                                    f"{child_path}[{index}]_record_id_not_opaque"
                                )
                else:
                    walk(child, child_path)
        elif isinstance(nested, list):
            for index, child in enumerate(nested):
                walk(child, f"{nested_path}[{index}]")

    walk(value, path)
    return issues


def _expected_v5_battle_result(
    battle_id: str,
    meta: Mapping[str, Any],
    turns: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reconstruct the canonical manifest row from authoritative trace data."""

    status = _V5_STATUS_TO_MANIFEST.get(str(meta.get("status") or ""))
    turns_count = meta.get("turns")
    duration = meta.get("duration_seconds")
    if (
        status is None
        or not isinstance(turns_count, int)
        or isinstance(turns_count, bool)
        or turns_count < 1
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        return None

    p1_user_id = meta.get("p1_user_id")
    p2_user_id = meta.get("p2_user_id")
    winner_user_id = meta.get("winner_user_id")
    loser_user_id = None
    if winner_user_id == p1_user_id:
        loser_user_id = p2_user_id
    elif winner_user_id == p2_user_id:
        loser_user_id = p1_user_id

    collection_class = (
        "human-vs-human"
        if (
            meta.get("p1_actor_type") == "human"
            and meta.get("p2_actor_type") == "human"
        )
        else "human-vs-bot"
    )
    return {
        "battle_id": battle_id,
        "winner_user_id": winner_user_id,
        "loser_user_id": loser_user_id,
        "status": status,
        "turns": turns_count,
        "duration_seconds": float(duration),
        "battle_tag": str(meta.get("battle_tag") or collection_class),
        "collection_class": collection_class,
        "p1_actor_type": meta.get("p1_actor_type"),
        "p2_actor_type": meta.get("p2_actor_type"),
        "v5_dir": f"battles/{battle_id}/v5",
        "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
        "v5_trace_ok": True,
        "validation_scope": "v5_trace_without_legacy_battle_log",
        "finished_at": meta.get("finished_at"),
    }


def _expected_v5_results_summary(
    battle_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(battle_results)
    p1_wins = sum(row["status"] == "P1_WIN" for row in battle_results)
    p2_wins = sum(row["status"] == "P2_WIN" for row in battle_results)
    draws = sum(
        row["status"] in {"DRAW", "STALEMATE"}
        for row in battle_results
    )
    return {
        "battles_planned": total,
        "battles_finished": total,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "winrate_p1": round(p1_wins / total, 4) if total else 0.0,
        "winrate_p2": round(p2_wins / total, 4) if total else 0.0,
        "avg_turns": (
            round(
                sum(row["turns"] for row in battle_results) / total,
                2,
            )
            if total
            else 0.0
        ),
        "avg_duration_seconds": (
            round(
                sum(
                    row["duration_seconds"]
                    for row in battle_results
                )
                / total,
                3,
            )
            if total
            else 0.0
        ),
    }


def _manifest_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return bool(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return actual == expected


def _catalog_hash_matches(actual: Any, expected_sha256: str) -> bool:
    value = str(actual or "").strip().lower()
    expected = str(expected_sha256 or "").strip().lower()
    return bool(
        len(value) >= 16
        and re.fullmatch(r"[0-9a-f]+", value)
        and expected.startswith(value)
    )


def _v5_privacy_issues(
    meta: Mapping[str, Any],
    turns: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    require_meta_seats: bool = True,
) -> list[str]:
    """Prove that a V5 bundle uses only the authoritative seat IDs 1/2."""

    issues: list[str] = []
    if (
        require_meta_seats
        and (
            meta.get("p1_user_id") != 1
            or meta.get("p2_user_id") != 2
        )
    ):
        issues.append("meta_seat_ids_not_pseudonymized")

    def safe_seat_id(value: Any) -> bool:
        return bool(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in {1, 2}
            or isinstance(value, str)
            and value.strip().lower() in {"1", "2", "p1", "p2"}
        )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            p1 = value.get("p1")
            p2 = value.get("p2")
            if isinstance(p1, Mapping) and "user_id" in p1:
                if p1.get("user_id") != 1:
                    issues.append(f"{path}.p1.user_id_not_1")
            if isinstance(p2, Mapping) and "user_id" in p2:
                if p2.get("user_id") != 2:
                    issues.append(f"{path}.p2.user_id_not_2")
            seats = value.get("seats")
            if isinstance(seats, Mapping):
                seat_p1 = seats.get("p1")
                seat_p2 = seats.get("p2")
                if (
                    isinstance(seat_p1, Mapping)
                    and seat_p1.get("participant_id") != 1
                ):
                    issues.append(
                        f"{path}.seats.p1.participant_id_not_1"
                    )
                if (
                    isinstance(seat_p2, Mapping)
                    and seat_p2.get("participant_id") != 2
                ):
                    issues.append(
                        f"{path}.seats.p2.participant_id_not_2"
                    )
            for key, nested in value.items():
                key_text = str(key)
                normalized_key = _normalized_field_name(key)
                child_path = f"{path}.{key_text}"
                if _is_raw_identity_field(
                    key,
                    allow_seat_user_id=True,
                ):
                    issues.append(f"{child_path}_pii_field_forbidden")
                if (
                    normalized_key in _V5_PARTICIPANT_ID_KEYS_NORMALIZED
                    or normalized_key.endswith("userid")
                ):
                    if nested is not None and not safe_seat_id(nested):
                        issues.append(
                            f"{child_path}_outside_pseudonymous_seats"
                        )
                if (
                    normalized_key
                    in _V5_PARTICIPANT_ID_LIST_KEYS_NORMALIZED
                    or normalized_key.endswith("userids")
                    or normalized_key.endswith("playerids")
                    or normalized_key.endswith("participantids")
                ):
                    if not isinstance(nested, list) or any(
                        not safe_seat_id(item)
                        for item in nested
                    ):
                        issues.append(
                            f"{child_path}_outside_pseudonymous_seats"
                        )
                walk(nested, child_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{path}[{index}]")

    walk(meta, "meta")
    walk(turns, "turns")
    walk(actions, "actions")
    return issues


def _v5_policy_issues(meta: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if value.get("degraded") is True:
                issues.append(f"{path}.degraded")
            warnings = value.get("policy_warnings")
            if isinstance(warnings, list) and warnings:
                issues.append(f"{path}.policy_warnings")
            for key, nested in value.items():
                walk(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{path}[{index}]")

    walk(meta, "meta")
    model_provenance = meta.get("model_provenance")
    for side in ("p1", "p2"):
        if meta.get(f"{side}_actor_type") not in {"bot", "rl", "llm"}:
            continue
        provenance: Any = None
        if isinstance(model_provenance, Mapping):
            provenance = model_provenance.get(side)
        if provenance is None:
            provenance = (
                meta.get("bot_policy")
                if side == "p2"
                else meta.get("p1_policy")
            )
        if not isinstance(provenance, Mapping):
            issues.append(f"{side}_model_provenance_missing")
            continue
        weights_hash = provenance.get("weights_hash")
        if (
            not isinstance(weights_hash, str)
            or not re.fullmatch(r"[0-9a-fA-F]{16,64}", weights_hash)
        ):
            issues.append(f"{side}_weights_hash_missing_or_invalid")
    return issues


def _v5_current_ruleset_issues(
    meta: Mapping[str, Any],
    turns: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[str]:
    expected_meta = {
        "schema_version": V5_STORAGE_SCHEMA,
        "visibility": V5_VISIBILITY,
        "actions_version": V5_ACTIONS_VERSION,
        "obs_version": V5_OBS_VERSION,
        "card_shape_version": V5_CARD_SHAPE_VERSION,
        "card_params_schema": V5_CARD_PARAMS_SCHEMA,
        "deck_params_schema": V5_DECK_PARAMS_SCHEMA,
        "ruleset": "classic",
    }
    issues = [
        f"meta_{key}_mismatch"
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    ]

    snapshots: list[tuple[str, Any]] = [
        (f"turns[{index}]", state)
        for index, state in enumerate(turns)
    ]
    for action_index, action in enumerate(actions):
        snapshots.extend(
            [
                (
                    f"actions[{action_index}].pre_state",
                    action.get("pre_state"),
                ),
                (
                    f"actions[{action_index}].post_state",
                    action.get("post_state"),
                ),
            ]
        )
    for label, snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        for side in ("p1", "p2"):
            player = snapshot.get(side)
            value = (
                player.get("mana_draw_count_this_turn")
                if isinstance(player, Mapping)
                else None
            )
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                issues.append(
                    f"{label}.{side}.mana_draw_count_this_turn"
                )
    return issues


def _format_for_header(header: Mapping[str, Any]) -> str:
    value = str(header.get("format") or "")
    if value == RETURNCLOCK_DATASET_FORMAT:
        return "returnclock"
    if value == NEMESIS_EXPORT_FORMAT:
        return "nemesis"
    if value == V5_EXPORT_FORMAT:
        return "v5_export"
    return "unknown"


class DatasetToolbox:
    """Schema-aware dataset export, inventory, validation and materialization."""

    def __init__(
        self,
        datasets_dir: str | Path,
        *,
        returnclock_salt_env: str = "RETURNCLOCK_DATASET_SALT",
        returnclock_salt_key_id_env: str = (
            "RETURNCLOCK_DATASET_SALT_KEY_ID"
        ),
        production_enabled: bool = False,
        database_factory: Callable[[Any], Any] | None = None,
        settings_factory: Callable[[], Any] | None = None,
        cards_path: str | Path | None = None,
    ) -> None:
        requested_root = Path(datasets_dir).expanduser()
        if requested_root.is_symlink():
            raise DatasetToolboxError("datasets_dir must not be a symlink")
        self.root = requested_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.returnclock_salt_env = str(returnclock_salt_env).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", self.returnclock_salt_env):
            raise DatasetToolboxError("invalid ReturnClock salt environment name")
        self.returnclock_salt_key_id_env = str(
            returnclock_salt_key_id_env
        ).strip()
        if not re.fullmatch(
            r"[A-Z][A-Z0-9_]{2,127}",
            self.returnclock_salt_key_id_env,
        ):
            raise DatasetToolboxError(
                "invalid ReturnClock salt key ID environment name"
            )
        self.production_enabled = bool(production_enabled)
        if (database_factory is None) != (settings_factory is None):
            raise DatasetToolboxError(
                "database_factory and settings_factory must be provided together"
            )
        self._database_factory = database_factory
        self._settings_factory = settings_factory
        self.cards_path = (
            Path(cards_path).expanduser().resolve()
            if cards_path is not None
            else Path(__file__).resolve().parents[2] / "ai" / "cards.json"
        )
        self._catalog_payload: dict[str, Any] | None = None
        self.current_catalog_hash: str | None = None
        self.current_card_count: int | None = None
        try:
            raw_catalog = self.cards_path.read_bytes()
            parsed_catalog = json.loads(raw_catalog)
            if (
                not isinstance(parsed_catalog, list)
                or not parsed_catalog
                or not all(
                    isinstance(card, dict)
                    and isinstance(card.get("id"), int)
                    and not isinstance(card.get("id"), bool)
                    for card in parsed_catalog
                )
            ):
                raise ValueError("cards.json must be a non-empty card list")
            cards = {
                str(card["id"]): card
                for card in parsed_catalog
            }
            if len(cards) != len(parsed_catalog):
                raise ValueError("cards.json contains duplicate card ids")
            self.current_catalog_hash = hashlib.sha256(
                raw_catalog
            ).hexdigest()
            self.current_card_count = len(cards)
            self._catalog_payload = {
                "schema": "rlhf_v5_catalog_v1",
                "catalog_hash": self.current_catalog_hash,
                "cards": cards,
            }
        except (OSError, ValueError, json.JSONDecodeError):
            self._catalog_payload = None

    def status(self) -> dict[str, Any]:
        inventory = self.list_artifacts(limit=1000)
        by_kind: dict[str, int] = {}
        for artifact in inventory["artifacts"]:
            artifact_kind = str(artifact.get("kind") or "unknown")
            by_kind[artifact_kind] = by_kind.get(artifact_kind, 0) + 1
        blockers: list[str] = []
        if not self.production_enabled:
            blockers.append("production_data_disabled")
        salt_configured = len(
            os.getenv(self.returnclock_salt_env, "").encode("utf-8")
        ) >= 32
        key_id = os.getenv(
            self.returnclock_salt_key_id_env,
            "",
        ).strip()
        key_id_configured = bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}",
                key_id,
            )
        )
        if not salt_configured:
            blockers.append("returnclock_salt_missing_or_weak")
        if not key_id_configured:
            blockers.append("returnclock_salt_key_id_missing")
        catalog_ready = bool(
            self._catalog_payload is not None
            and self.current_catalog_hash
            and self.current_card_count == 50
        )
        if not catalog_ready:
            blockers.append("current_50_card_catalog_unavailable")
        return {
            "datasets_dir": str(self.root),
            "production_data_enabled": self.production_enabled,
            "returnclock_salt_configured": salt_configured,
            "returnclock_salt_key_id_configured": key_id_configured,
            "current_catalog_ready": catalog_ready,
            "current_catalog_hash": self.current_catalog_hash,
            "current_card_count": self.current_card_count,
            "artifact_count": inventory["count"],
            "artifacts_by_kind": by_kind,
            "natural_return_model_supported": True,
            "causal_send_policy_ready": False,
            "causal_blocker": (
                "randomized no-send/control assignment data is required"
            ),
            "blockers": blockers,
        }

    # ------------------------------------------------------------------
    # Safe path boundary
    # ------------------------------------------------------------------
    def resolve(
        self,
        value: str | Path,
        *,
        must_exist: bool = False,
        expect: str | None = None,
    ) -> Path:
        raw = Path(value).expanduser()
        lexical = raw if raw.is_absolute() else self.root / raw
        lexical = Path(os.path.abspath(lexical))

        # ``self.root`` is canonical, while callers can legitimately receive
        # an equivalent absolute spelling from the OS.  The common macOS case
        # is ``/var/...`` for a root canonicalized to ``/private/var/...``.
        # Locate an ancestor which resolves to the datasets root instead of
        # comparing the two lexical spellings.  Requiring an alias for the
        # root itself (rather than merely a symlink to some nested directory)
        # keeps the original confinement boundary intact.
        lexical_root = next(
            (
                ancestor
                for ancestor in (lexical, *lexical.parents)
                if ancestor.resolve(strict=False) == self.root
            ),
            None,
        )
        if lexical_root is None:
            raise DatasetToolboxError(
                "dataset path must stay inside datasets_dir"
            )
        relative = lexical.relative_to(lexical_root)
        if not relative.parts:
            raise DatasetToolboxError("dataset path must name an artifact")

        current = lexical_root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise DatasetToolboxError(
                    f"dataset path crosses symlink: {current.name}"
                )
        resolved = lexical.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise DatasetToolboxError(
                "dataset path resolves outside datasets_dir"
            ) from exc
        if must_exist and not resolved.exists():
            raise DatasetToolboxError(f"dataset artifact not found: {value}")
        if expect == "file" and resolved.exists() and not resolved.is_file():
            raise DatasetToolboxError(f"dataset artifact is not a file: {value}")
        if expect == "dir" and resolved.exists() and not resolved.is_dir():
            raise DatasetToolboxError(
                f"dataset artifact is not a directory: {value}"
            )
        return resolved

    def _prepare_output(
        self,
        value: str | Path,
        *,
        suffix: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        destination = self.resolve(value)
        if suffix and destination.suffix.lower() != suffix:
            raise DatasetToolboxError(
                f"output must use the {suffix} suffix"
            )
        if destination.exists() and not overwrite:
            raise DatasetToolboxError(
                f"output already exists: {destination.relative_to(self.root)}"
            )
        if suffix and destination.exists() and not destination.is_file():
            raise DatasetToolboxError("dataset output must be a file")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        return destination

    # ------------------------------------------------------------------
    # Inventory and inspection
    # ------------------------------------------------------------------
    def _detect_path_kind(self, path: Path) -> str:
        if path.is_dir():
            manifest = path / "manifest.json"
            if not manifest.is_file() or manifest.is_symlink():
                return "unknown"
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return "unknown"
            if payload.get("materialized_format") == V5_MATERIALIZED_FORMAT:
                return "v5_materialized"
            if payload.get("format") == "extraarena_returnclock_split_v1":
                return "returnclock_split"
            if payload.get("format") == _NEMESIS_SPLIT_FORMAT:
                return "nemesis_split"
            return "unknown"
        if path.suffix.lower() not in _JSONL_SUFFIXES:
            return "unknown"
        try:
            return _format_for_header(_first_record(path))
        except (OSError, DatasetToolboxError):
            return "unknown"

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        if kind is not None and kind not in _DATASET_KINDS - {"unknown"}:
            raise DatasetToolboxError(
                f"kind must be one of {sorted(_DATASET_KINDS - {'unknown'})}"
            )
        bounded_limit = max(1, min(int(limit), 1000))
        candidates: list[Path] = []
        for path in self.root.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix.lower() in _JSONL_SUFFIXES:
                candidates.append(path)
            elif (
                path.is_dir()
                and (path / "manifest.json").is_file()
                and not (path / "manifest.json").is_symlink()
            ):
                candidates.append(path)
        candidates.sort(
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
        artifacts: list[dict[str, Any]] = []
        for path in candidates:
            detected = self._detect_path_kind(path)
            if kind is not None and detected != kind:
                continue
            stat_result = path.stat()
            artifacts.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(self.root)),
                    "kind": detected,
                    "is_directory": path.is_dir(),
                    "size_bytes": (
                        stat_result.st_size
                        if path.is_file()
                        else sum(
                            item.stat().st_size
                            for item in path.rglob("*")
                            if item.is_file() and not item.is_symlink()
                        )
                    ),
                    "modified_at_ns": stat_result.st_mtime_ns,
                }
            )
            if len(artifacts) >= bounded_limit:
                break
        return {
            "datasets_dir": str(self.root),
            "count": len(artifacts),
            "limit": bounded_limit,
            "artifacts": artifacts,
        }

    def inspect_artifact(self, path: str | Path) -> dict[str, Any]:
        artifact = self.resolve(path, must_exist=True)
        kind = self._detect_path_kind(artifact)
        base: dict[str, Any] = {
            "path": str(artifact),
            "relative_path": str(artifact.relative_to(self.root)),
            "kind": kind,
        }
        if artifact.is_dir():
            manifest_path = artifact / "manifest.json"
            if manifest_path.is_symlink():
                raise DatasetToolboxError(
                    "dataset manifest must not be a symlink"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = [
                item
                for item in artifact.rglob("*")
                if item.is_file() and not item.is_symlink()
            ]
            if kind == "returnclock_split":
                split_rows = manifest.get("splits")
                safe_splits: dict[str, Any] = {}
                if isinstance(split_rows, dict):
                    for split_name in ("train", "validation", "test"):
                        entry = split_rows.get(split_name)
                        if not isinstance(entry, dict):
                            continue
                        safe_splits[split_name] = {
                            key: entry.get(key)
                            for key in (
                                "file",
                                "sha256",
                                "example_count",
                                "organic_example_count",
                                "treated_example_count",
                                "user_count",
                                "first_cutoff_min",
                                "first_cutoff_max",
                                "row_cutoff_min",
                                "row_cutoff_max",
                            )
                        }
                safe_manifest = {
                    "format": manifest.get("format"),
                    "format_version": manifest.get("format_version"),
                    "source_sha256": manifest.get("source_sha256"),
                    "pseudonymization_key_id": manifest.get(
                        "pseudonymization_key_id"
                    ),
                    "assignment_basis": manifest.get("assignment_basis"),
                    "training_filter": manifest.get("training_filter"),
                    "source_example_count": manifest.get(
                        "source_example_count"
                    ),
                    "example_count": manifest.get("example_count"),
                    "excluded_treated_count": manifest.get(
                        "excluded_treated_count"
                    ),
                    "excluded_temporal_boundary_count": manifest.get(
                        "excluded_temporal_boundary_count"
                    ),
                    "user_count": manifest.get("user_count"),
                    "post_cutoff_excluded_from_features": manifest.get(
                        "post_cutoff_excluded_from_features"
                    ),
                    "splits": safe_splits,
                }
            elif kind == "nemesis_split":
                artifacts = manifest.get("artifacts")
                safe_artifacts: dict[str, Any] = {}
                if isinstance(artifacts, dict):
                    for regime in _NEMESIS_SPLIT_REGIMES:
                        regime_rows = artifacts.get(regime)
                        if not isinstance(regime_rows, dict):
                            continue
                        safe_artifacts[regime] = {
                            split_name: {
                                key: split_entry.get(key)
                                for key in _NEMESIS_SPLIT_ENTRY_KEYS
                            }
                            for split_name in _NEMESIS_SPLIT_NAMES
                            if isinstance(
                                split_entry := regime_rows.get(split_name),
                                dict,
                            )
                        }
                safe_manifest = {
                    "format": manifest.get("format"),
                    "format_version": manifest.get("format_version"),
                    "source_sha256": manifest.get("source_sha256"),
                    "source_battle_count": manifest.get(
                        "source_battle_count"
                    ),
                    "privacy": manifest.get("privacy"),
                    "catalog": manifest.get("catalog"),
                    "exclusions": self._bounded_nemesis_exclusions(
                        manifest.get("exclusions")
                    ),
                    "artifacts": safe_artifacts,
                    "training_readiness": manifest.get(
                        "training_readiness"
                    ),
                }
            else:
                results = manifest.get("results")
                safe_results = (
                    {
                        key: results.get(key)
                        for key in _V5_MATERIALIZED_RESULT_KEYS
                    }
                    if isinstance(results, dict)
                    else None
                )
                safe_manifest = {
                    "group_id": manifest.get("group_id"),
                    "storage_schema": manifest.get("storage_schema"),
                    "materialized_format": manifest.get(
                        "materialized_format"
                    ),
                    "finished_at": manifest.get("finished_at"),
                    "privacy": (manifest.get("spec") or {}).get(
                        "privacy"
                    ),
                    "include_players": bool(
                        (manifest.get("spec") or {}).get(
                            "include_players", False
                        )
                    ),
                    "results": safe_results,
                }
            return {
                **base,
                "size_bytes": sum(item.stat().st_size for item in files),
                "file_count": len(files),
                "sha256": _sha256_bundle(artifact),
                "mode": oct(stat.S_IMODE(artifact.stat().st_mode)),
                "manifest_sha256": _sha256_file(manifest_path),
                "manifest_mode": oct(
                    stat.S_IMODE(manifest_path.stat().st_mode)
                ),
                "manifest": safe_manifest,
            }

        header = _first_record(artifact)
        record_count = sum(1 for _ in _iter_jsonl(artifact))
        if kind == "nemesis":
            safe_header = {
                key: header.get(key)
                for key in _NEMESIS_HEADER_KEYS
            }
        elif kind == "returnclock":
            summary = header.get("summary")
            safe_header = {
                key: header.get(key)
                for key in (
                    "record_type",
                    "format",
                    "format_version",
                    "generated_at",
                    "dataset_end",
                    "ingested_before",
                    "cutoff_start",
                    "horizon_hours",
                    "min_analytics_version",
                    "sessionization_gap_minutes",
                    "feature_columns",
                    "label_columns",
                    "post_cutoff_namespace",
                    "user_id_scheme",
                    "pseudonymization_key_id",
                    "privacy_note",
                    "meaningful_session",
                )
            }
            safe_header["summary"] = (
                {
                    key: summary.get(key)
                    for key in (
                        "example_count",
                        "observed_returns",
                        "complete_no_return_horizons",
                        "right_censored",
                        "treated_intervals",
                        "organic_candidates",
                        "excluded_legacy_sessions",
                        "excluded_unfinished_sessions",
                        "excluded_nonmeaningful_sessions",
                        "inferred_stale_session_ends",
                    )
                }
                if isinstance(summary, dict)
                else None
            )
        elif kind == "v5_export":
            safe_header = {
                key: header.get(key)
                for key in (
                    "record_type",
                    "format",
                    "format_version",
                    "storage_schema",
                    "created_at",
                    "privacy",
                    "include_players",
                    "days",
                    "limit_battles",
                    "battle_count",
                    "skipped_invalid",
                    "current_catalog_hash",
                    "current_card_count",
                )
            }
        else:
            safe_header = {
                "record_type": header.get("record_type"),
                "format": header.get("format"),
                "format_version": header.get("format_version"),
            }
        return {
            **base,
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
            "mode": oct(stat.S_IMODE(artifact.stat().st_mode)),
            "record_count": record_count,
            "data_record_count": max(0, record_count - 1),
            "header": safe_header,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _require_current_catalog(self) -> tuple[str, int, dict[str, Any]]:
        if (
            self._catalog_payload is None
            or self.current_catalog_hash is None
            or self.current_card_count != 50
        ):
            raise DatasetToolboxError(
                "current 50-card ai/cards.json catalog is unavailable"
            )
        return (
            self.current_catalog_hash,
            self.current_card_count,
            self._catalog_payload,
        )

    @staticmethod
    def _safe_v5_dir(root: Path, battle_id: Any) -> Path:
        component = str(battle_id or "")
        if (
            not _SAFE_COMPONENT_RE.fullmatch(component)
            or component in {".", ".."}
        ):
            raise DatasetToolboxError(
                f"unsafe battle_id in V5 manifest: {component!r}"
            )
        if root.is_symlink():
            raise DatasetToolboxError(
                "materialized V5 root must not be a symlink"
            )
        current = root
        for part in ("battles", component, "v5"):
            current = current / part
            if current.is_symlink():
                raise DatasetToolboxError(
                    f"materialized V5 path crosses symlink: {part}"
                )
        resolved = current.resolve(strict=False)
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise DatasetToolboxError(
                "materialized V5 path escapes artifact root"
            ) from exc
        return resolved

    @staticmethod
    def _write_private_json(path: Path, payload: Any) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        with temporary.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)

    def _attach_current_catalog(self, materialized: Path) -> None:
        current_hash, _, catalog_payload = self._require_current_catalog()
        manifest_path = materialized / "manifest.json"
        if manifest_path.is_symlink():
            raise DatasetToolboxError("manifest symlink is forbidden")
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetToolboxError(f"invalid V5 manifest: {exc}") from exc
        battle_ids = manifest.get("battle_ids")
        if not isinstance(battle_ids, list) or not battle_ids:
            raise DatasetToolboxError("V5 manifest has no battle_ids")
        for battle_id in battle_ids:
            v5_dir = self._safe_v5_dir(materialized, battle_id)
            meta_path = v5_dir / "meta.json"
            if meta_path.is_symlink():
                raise DatasetToolboxError(
                    f"battle {battle_id}: meta symlink is forbidden"
                )
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetToolboxError(
                    f"battle {battle_id}: invalid meta.json: {exc}"
                ) from exc
            if not _catalog_hash_matches(
                meta.get("catalog_hash"),
                current_hash,
            ):
                raise DatasetToolboxError(
                    f"battle {battle_id}: stale or missing catalog_hash"
                )
            source_catalog_hash = str(meta.get("catalog_hash"))
            if source_catalog_hash != current_hash:
                meta["source_catalog_hash"] = source_catalog_hash
            meta["catalog_hash"] = current_hash
            meta["catalog_path"] = "catalog.json"
            meta["catalog_path_base"] = "group_dir"
            self._write_private_json(meta_path, meta)
        self._write_private_json(
            materialized / "catalog.json",
            catalog_payload,
        )
        spec = manifest.setdefault("spec", {})
        if not isinstance(spec, dict):
            raise DatasetToolboxError("V5 manifest spec must be an object")
        spec["current_catalog_hash"] = current_hash
        spec["current_card_count"] = self.current_card_count
        self._write_private_json(manifest_path, manifest)

    def validate_artifact(self, path: str | Path) -> dict[str, Any]:
        artifact = self.resolve(path, must_exist=True)
        kind = self._detect_path_kind(artifact)
        if kind == "returnclock":
            return self._validate_returnclock(artifact)
        if kind == "returnclock_split":
            return self._validate_returnclock_split(artifact)
        if kind == "nemesis":
            return self._validate_nemesis(artifact)
        if kind == "nemesis_split":
            return self._validate_nemesis_split(artifact)
        if kind == "v5_export":
            return self._validate_v5_export(artifact)
        if kind == "v5_materialized":
            return self._validate_v5_materialized(artifact)
        return {
            "ok": False,
            "training_ready": False,
            "kind": "unknown",
            "path": str(artifact),
            "issues": ["unsupported_or_unrecognized_dataset_format"],
        }

    def _validate_returnclock(self, path: Path) -> dict[str, Any]:
        issues: list[str] = []
        records = _iter_jsonl(path)
        try:
            _, header = next(records)
        except StopIteration:
            return {
                "ok": False,
                "training_ready": False,
                "kind": "returnclock",
                "path": str(path),
                "issues": ["empty_dataset"],
            }
        if header.get("record_type") != "header":
            issues.append("first_record_must_be_header")
        allowed_header_keys = (
            _RETURNCLOCK_HEADER_KEYS
            | (
                _RETURNCLOCK_SPLIT_HEADER_KEYS
                if header.get("split_name") is not None
                else frozenset()
            )
        )
        if set(header) != allowed_header_keys:
            issues.append("header_allowlist_mismatch")
        if header.get("format") != RETURNCLOCK_DATASET_FORMAT:
            issues.append("incompatible_format")
        if header.get("format_version") != RETURNCLOCK_DATASET_VERSION:
            issues.append("incompatible_format_version")
        pseudonymization_key_id = header.get("pseudonymization_key_id")
        if not isinstance(
            pseudonymization_key_id, str
        ) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}",
            pseudonymization_key_id,
        ):
            issues.append("pseudonymization_key_id_missing_or_invalid")
        feature_columns = header.get("feature_columns")
        if feature_columns != list(RETURNCLOCK_FEATURE_COLUMNS):
            issues.append("feature_allowlist_mismatch")
        if header.get("label_columns") != [
            "target_observed",
            "right_censored",
            "time_to_return_minutes",
            "observation_window_minutes",
        ]:
            issues.append("label_columns_mismatch")
        if (
            header.get("post_cutoff_namespace")
            != "excluded_from_model_features"
        ):
            issues.append("post_cutoff_namespace_mismatch")
        if header.get("user_id_scheme") != "hmac-sha256-truncated-128":
            issues.append("user_id_scheme_mismatch")
        if header.get("privacy_note") != (
            "user_id_hash is pseudonymous and intended only for grouped "
            "splits inside access-controlled training storage"
        ):
            issues.append("privacy_note_mismatch")
        if _contains_raw_user_id(header):
            issues.append("header_raw_user_id")
        meaningful_session = header.get("meaningful_session")
        if meaningful_session != {
            "duration_seconds_gte": 120,
            "screen_count_gte": 2,
            "or_battle_or_case": True,
        }:
            issues.append("meaningful_session_contract_mismatch")
        if header.get("sessionization_gap_minutes") != 30:
            issues.append("sessionization_gap_mismatch")
        generated_at = _parse_utc_timestamp(header.get("generated_at"))
        if generated_at is None:
            issues.append("generated_at_invalid")
        elif not _is_canonical_utc_timestamp(header.get("generated_at")):
            issues.append("generated_at_not_canonical_utc")
        min_analytics_version = header.get("min_analytics_version")
        if (
            not isinstance(min_analytics_version, int)
            or isinstance(min_analytics_version, bool)
            or min_analytics_version < 2
        ):
            issues.append("min_analytics_version_below_v2")
        horizon_hours = header.get("horizon_hours")
        if (
            not isinstance(horizon_hours, int)
            or isinstance(horizon_hours, bool)
            or not 1 <= horizon_hours <= 31 * 24
        ):
            issues.append("horizon_hours_invalid")
            horizon_minutes: float | None = None
        else:
            horizon_minutes = float(horizon_hours * 60)
        dataset_end = _parse_utc_timestamp(header.get("dataset_end"))
        if dataset_end is None:
            issues.append("dataset_end_invalid")
        elif not _is_canonical_utc_timestamp(header.get("dataset_end")):
            issues.append("dataset_end_not_canonical_utc")
        ingested_before = _parse_utc_timestamp(
            header.get("ingested_before")
        )
        if ingested_before is None:
            issues.append("ingested_before_invalid")
        elif not _is_canonical_utc_timestamp(
            header.get("ingested_before")
        ):
            issues.append("ingested_before_not_canonical_utc")
        elif (
            dataset_end is not None
            and ingested_before < dataset_end
        ):
            issues.append("ingested_before_precedes_dataset_end")
        cutoff_start_raw = header.get("cutoff_start")
        cutoff_start = (
            None
            if cutoff_start_raw is None
            else _parse_utc_timestamp(cutoff_start_raw)
        )
        if cutoff_start_raw is not None and cutoff_start is None:
            issues.append("cutoff_start_invalid")
        elif cutoff_start_raw is not None and not (
            _is_canonical_utc_timestamp(cutoff_start_raw)
        ):
            issues.append("cutoff_start_not_canonical_utc")
        if (
            cutoff_start is not None
            and dataset_end is not None
            and cutoff_start > dataset_end
        ):
            issues.append("cutoff_start_after_dataset_end")

        example_count = 0
        users: set[str] = set()
        cutoffs: set[str] = set()
        organic_user_first_cutoffs: dict[str, str] = {}
        example_keys: set[tuple[str, str]] = set()
        row_fingerprints: set[str] = set()
        observed = censored = treated = organic = complete_no_return = 0
        previous_sort_key: tuple[str, str] | None = None
        label_keys = {
            "target_observed",
            "right_censored",
            "time_to_return_minutes",
            "observation_window_minutes",
        }
        for line_number, row in records:
            example_count += 1
            if set(row) != _RETURNCLOCK_EXAMPLE_KEYS:
                issues.append(f"line_{line_number}:example_allowlist")
            if row.get("record_type") != "example":
                issues.append(f"line_{line_number}:record_type")
                continue
            if (
                row.get("dataset_format") != RETURNCLOCK_DATASET_FORMAT
                or row.get("dataset_version") != RETURNCLOCK_DATASET_VERSION
            ):
                issues.append(f"line_{line_number}:schema")
            user_hash = row.get("user_id_hash")
            if not isinstance(user_hash, str) or not _USER_HASH_RE.fullmatch(
                user_hash
            ):
                issues.append(f"line_{line_number}:user_id_hash")
            else:
                users.add(user_hash)
            cutoff = row.get("prediction_cutoff_at")
            cutoff_at = _parse_utc_timestamp(cutoff)
            if cutoff_at is None:
                issues.append(f"line_{line_number}:prediction_cutoff_at")
                cutoff = ""
            else:
                cutoff = str(cutoff)
                if not _is_canonical_utc_timestamp(cutoff):
                    issues.append(
                        f"line_{line_number}:prediction_cutoff_not_canonical_utc"
                    )
                cutoffs.add(cutoff)
                if cutoff_start is not None and cutoff_at < cutoff_start:
                    issues.append(
                        f"line_{line_number}:cutoff_before_cutoff_start"
                    )
                if dataset_end is not None and cutoff_at > dataset_end:
                    issues.append(
                        f"line_{line_number}:cutoff_after_dataset_end"
                    )
            sort_key = (cutoff, str(user_hash or ""))
            if previous_sort_key is not None and sort_key < previous_sort_key:
                issues.append(f"line_{line_number}:not_chronologically_sorted")
            previous_sort_key = sort_key
            example_key = (str(user_hash or ""), cutoff)
            if example_key in example_keys:
                issues.append(f"line_{line_number}:duplicate_example_key")
            example_keys.add(example_key)
            row_fingerprint = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if row_fingerprint in row_fingerprints:
                issues.append(f"line_{line_number}:duplicate_example_row")
            row_fingerprints.add(row_fingerprint)

            features = row.get("features")
            if (
                not isinstance(features, dict)
                or set(features) != set(feature_columns or [])
                or len(features) != len(feature_columns or [])
            ):
                issues.append(f"line_{line_number}:features_allowlist")
            elif feature_issues := _returnclock_feature_issues(features):
                issues.extend(
                    f"line_{line_number}:feature_{issue}"
                    for issue in feature_issues
                )
            label = row.get("label")
            if not isinstance(label, dict) or set(label) != label_keys:
                issues.append(f"line_{line_number}:label_contract")
                label = {}
            target_observed = label.get("target_observed")
            right_censored = label.get("right_censored")
            if not isinstance(target_observed, bool):
                issues.append(f"line_{line_number}:target_observed_type")
            if not isinstance(right_censored, bool):
                issues.append(f"line_{line_number}:right_censored_type")
            if target_observed is True and right_censored is True:
                issues.append(
                    f"line_{line_number}:observed_and_censored"
                )
            observation = _finite_number(
                label.get("observation_window_minutes")
            )
            if (
                observation is None
                or observation < 0.0
                or (
                    horizon_minutes is not None
                    and observation > horizon_minutes + 1e-6
                )
            ):
                issues.append(
                    f"line_{line_number}:observation_window_minutes"
                )
            return_minutes = label.get("time_to_return_minutes")
            if target_observed is True:
                parsed_return = _finite_number(return_minutes)
                if (
                    parsed_return is None
                    or parsed_return < 0.0
                    or observation is None
                    or abs(parsed_return - observation) > 1e-3
                ):
                    issues.append(
                        f"line_{line_number}:observed_return_label"
                    )
            elif return_minutes is not None:
                issues.append(
                    f"line_{line_number}:unobserved_return_must_be_null"
                )
            if (
                target_observed is False
                and right_censored is True
                and horizon_minutes is not None
                and observation is not None
                and observation >= horizon_minutes - 1e-6
            ):
                issues.append(
                    f"line_{line_number}:censored_window_not_partial"
                )
            if (
                target_observed is False
                and right_censored is True
                and cutoff_at is not None
                and dataset_end is not None
                and observation is not None
            ):
                expected_censor_minutes = round(
                    (dataset_end - cutoff_at).total_seconds() / 60.0,
                    3,
                )
                if (
                    expected_censor_minutes < 0.0
                    or abs(observation - expected_censor_minutes) > 1e-3
                ):
                    issues.append(
                        f"line_{line_number}:censor_time_mismatch"
                    )
            if (
                target_observed is False
                and right_censored is False
                and horizon_minutes is not None
                and (
                    observation is None
                    or abs(observation - horizon_minutes) > 1e-3
                )
            ):
                issues.append(
                    f"line_{line_number}:complete_negative_window"
                )
            if (
                cutoff_at is not None
                and dataset_end is not None
                and observation is not None
                and cutoff_at + timedelta(minutes=observation)
                > dataset_end + timedelta(seconds=1)
            ):
                issues.append(
                    f"line_{line_number}:label_after_dataset_end"
                )
            post_cutoff = row.get("post_cutoff")
            if (
                not isinstance(post_cutoff, dict)
                or set(post_cutoff) != _RETURNCLOCK_POST_CUTOFF_KEYS
            ):
                issues.append(f"line_{line_number}:post_cutoff_contract")
                post_cutoff = {}
            for count_key in (
                "notification_decision_count",
                "provider_accepted_count",
                "notification_sent_count",
                "notification_opened_count",
            ):
                count = post_cutoff.get(count_key)
                if (
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                ):
                    issues.append(
                        f"line_{line_number}:post_cutoff_{count_key}"
                    )
            for list_key in (
                "notification_channels",
                "treatment_arms",
            ):
                values = post_cutoff.get(list_key)
                if (
                    not isinstance(values, list)
                    or not all(
                        isinstance(value, str) and bool(value.strip())
                        for value in values
                    )
                    or values != sorted(set(values))
                ):
                    issues.append(
                        f"line_{line_number}:post_cutoff_{list_key}"
                    )
            for bool_key in (
                "notification_attributed",
                "treatment_assigned",
                "organic_candidate",
            ):
                if not isinstance(post_cutoff.get(bool_key), bool):
                    issues.append(
                        f"line_{line_number}:post_cutoff_{bool_key}"
                    )
            assignments = post_cutoff.get("assignments")
            if not isinstance(assignments, list):
                issues.append(
                    f"line_{line_number}:post_cutoff_assignments"
                )
                assignments = []
            for assignment_index, assignment in enumerate(assignments):
                prefix = (
                    f"line_{line_number}:assignment_{assignment_index}"
                )
                if (
                    not isinstance(assignment, dict)
                    or set(assignment) != _RETURNCLOCK_ASSIGNMENT_KEYS
                ):
                    issues.append(f"{prefix}:allowlist")
                    continue
                for nullable_text_key in ("experiment_id", "model_version"):
                    value = assignment.get(nullable_text_key)
                    if value is not None and (
                        not isinstance(value, str) or not value.strip()
                    ):
                        issues.append(
                            f"{prefix}:{nullable_text_key}"
                        )
                for text_key in (
                    "treatment_arm",
                    "decision",
                    "decision_source",
                    "policy_version",
                ):
                    value = assignment.get(text_key)
                    if not isinstance(value, str) or not value.strip():
                        issues.append(f"{prefix}:{text_key}")
                probability = assignment.get("assignment_probability")
                parsed_probability = (
                    None
                    if probability is None
                    else _finite_number(probability)
                )
                if (
                    probability is not None
                    and (
                        parsed_probability is None
                        or not 0.0 <= parsed_probability <= 1.0
                    )
                ):
                    issues.append(f"{prefix}:assignment_probability")
            if (
                isinstance(post_cutoff.get("notification_decision_count"), int)
                and not isinstance(
                    post_cutoff.get("notification_decision_count"), bool
                )
                and len(assignments)
                != post_cutoff.get("notification_decision_count")
            ):
                issues.append(
                    f"line_{line_number}:assignment_count_mismatch"
                )
            expected_treatment_arms = sorted(
                {
                    str(assignment.get("treatment_arm"))
                    for assignment in assignments
                    if isinstance(assignment, dict)
                    and assignment.get("treatment_arm") not in {None, ""}
                }
            )
            if post_cutoff.get("treatment_arms") != expected_treatment_arms:
                issues.append(
                    f"line_{line_number}:treatment_arms_mismatch"
                )
            expected_treatment_assigned = any(
                isinstance(assignment, dict)
                and str(assignment.get("decision") or "")
                .strip()
                .lower()
                not in {"", "skip", "no_send", "control"}
                for assignment in assignments
            )
            if (
                post_cutoff.get("treatment_assigned")
                is not expected_treatment_assigned
            ):
                issues.append(
                    f"line_{line_number}:treatment_assigned_mismatch"
                )
            organic_candidate = post_cutoff.get("organic_candidate")
            expected_organic = not bool(
                post_cutoff.get("provider_accepted_count")
                or post_cutoff.get("notification_sent_count")
                or post_cutoff.get("notification_opened_count")
                or post_cutoff.get("notification_attributed")
                or post_cutoff.get("treatment_assigned")
            )
            if (
                isinstance(organic_candidate, bool)
                and organic_candidate != expected_organic
            ):
                issues.append(
                    f"line_{line_number}:organic_candidate_inconsistent"
                )
            if _contains_raw_user_id(row):
                issues.append(f"line_{line_number}:raw_user_id")
            observed += target_observed is True
            censored += right_censored is True
            complete_no_return += (
                target_observed is False and right_censored is False
            )
            organic += organic_candidate is True
            treated += organic_candidate is False
            if (
                organic_candidate is True
                and isinstance(user_hash, str)
                and _USER_HASH_RE.fullmatch(user_hash)
                and cutoff_at is not None
            ):
                previous_first = organic_user_first_cutoffs.get(user_hash)
                if previous_first is None or cutoff < previous_first:
                    organic_user_first_cutoffs[user_hash] = cutoff
            if len(issues) >= 100:
                issues.append("issue_limit_reached")
                break

        summary = header.get("summary")
        if not isinstance(summary, dict):
            issues.append("header_summary_missing")
            summary = {}
        expected_summary_keys = {
            "example_count",
            "observed_returns",
            "complete_no_return_horizons",
            "right_censored",
            "treated_intervals",
            "organic_candidates",
            "excluded_legacy_sessions",
            "excluded_unfinished_sessions",
            "excluded_nonmeaningful_sessions",
            "inferred_stale_session_ends",
        }
        if set(summary) != expected_summary_keys:
            issues.append("header_summary_allowlist_mismatch")
        expected_counts = {
            "example_count": example_count,
            "observed_returns": observed,
            "complete_no_return_horizons": complete_no_return,
            "right_censored": censored,
            "treated_intervals": treated,
            "organic_candidates": organic,
        }
        for key, expected in expected_counts.items():
            if summary.get(key) != expected:
                issues.append(f"header_summary_mismatch:{key}")
        for key in (
            "excluded_legacy_sessions",
            "excluded_unfinished_sessions",
            "excluded_nonmeaningful_sessions",
            "inferred_stale_session_ends",
        ):
            value = summary.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                issues.append(f"header_summary_invalid:{key}")
        organic_first_cutoff_cohorts = set(
            organic_user_first_cutoffs.values()
        )
        # The mandatory materializer publishes train/validation/test.  Raw
        # readiness must therefore guarantee the same minimum preconditions:
        # three user groups and three strictly ordered first-cutoff cohorts.
        grouped_split = len(organic_user_first_cutoffs) >= 3
        temporal_split = len(organic_first_cutoff_cohorts) >= 3
        ok = not issues
        natural_return_training_ready = bool(
            ok
            and example_count > 0
            and organic > 0
            and grouped_split
            and temporal_split
        )
        return {
            "ok": ok,
            "training_ready": natural_return_training_ready,
            "training_ready_scope": "natural_return_observational",
            "natural_return_training_ready": (
                natural_return_training_ready
            ),
            "causal_notification_training_ready": False,
            "kind": "returnclock",
            "path": str(path),
            "sha256": _sha256_file(path),
            "issues": issues,
            "summary": {
                **expected_counts,
                "distinct_users": len(users),
                "distinct_cutoffs": len(cutoffs),
                "organic_distinct_users": len(
                    organic_user_first_cutoffs
                ),
                "organic_distinct_first_cutoff_cohorts": len(
                    organic_first_cutoff_cohorts
                ),
                "grouped_user_split_possible": grouped_split,
                "temporal_split_possible": temporal_split,
                "treated_share": (
                    round(treated / example_count, 6)
                    if example_count
                    else 0.0
                ),
                "right_censored_share": (
                    round(censored / example_count, 6)
                    if example_count
                    else 0.0
                ),
                "pseudonymization_key_id": pseudonymization_key_id,
            },
        }

    def _validate_returnclock_split(self, path: Path) -> dict[str, Any]:
        manifest_path = path / "manifest.json"
        issues: list[str] = []
        if manifest_path.is_symlink():
            return {
                "ok": False,
                "training_ready": False,
                "kind": "returnclock_split",
                "path": str(path),
                "issues": ["manifest_symlink_forbidden"],
            }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "training_ready": False,
                "kind": "returnclock_split",
                "path": str(path),
                "issues": [f"manifest:{exc}"],
            }
        if manifest.get("format") != "extraarena_returnclock_split_v1":
            issues.append("incompatible_split_manifest")
        if set(manifest) != _RETURNCLOCK_SPLIT_MANIFEST_KEYS:
            issues.append("split_manifest_allowlist_mismatch")
        if _contains_raw_user_id(manifest):
            issues.append("split_manifest_raw_user_id")
        if manifest.get("format_version") != 1:
            issues.append("incompatible_split_manifest_version")
        if manifest.get("assignment_basis") != "user_first_prediction_cutoff":
            issues.append("unsupported_assignment_basis")
        strategy = manifest.get("strategy")
        if (
            not isinstance(strategy, dict)
            or set(strategy)
            != {"grouped_by", "ordered_by", "assignment_basis"}
        ):
            issues.append("split_strategy_contract")
        elif strategy != {
            "grouped_by": "user_id_hash",
            "ordered_by": "prediction_cutoff_at",
            "assignment_basis": "user_first_prediction_cutoff",
        }:
            issues.append("split_strategy_values")
        requested_fractions = manifest.get("requested_fractions")
        if (
            not isinstance(requested_fractions, dict)
            or set(requested_fractions)
            != {"train", "validation", "test"}
        ):
            issues.append("split_requested_fractions_contract")
        else:
            parsed_fractions = [
                _finite_number(requested_fractions.get(name))
                for name in ("train", "validation", "test")
            ]
            if (
                any(
                    value is None or value <= 0.0 or value >= 1.0
                    for value in parsed_fractions
                )
                or abs(sum(value or 0.0 for value in parsed_fractions) - 1.0)
                > 1e-9
            ):
                issues.append("split_requested_fractions_values")
        if manifest.get("training_filter") != {
            "field": "post_cutoff.organic_candidate",
            "equals": True,
        }:
            issues.append("split_training_filter")
        if manifest.get("post_cutoff_excluded_from_features") is not True:
            issues.append("post_cutoff_feature_guard_missing")
        split_manifest_rows = manifest.get("splits")
        if (
            not isinstance(split_manifest_rows, dict)
            or set(split_manifest_rows)
            != {"train", "validation", "test"}
        ):
            issues.append("split_entries_contract")

        source: Path | None = None
        source_header: dict[str, Any] = {}
        source_examples: list[dict[str, Any]] = []
        source_relative = manifest.get("source")
        if isinstance(source_relative, str):
            try:
                source = self.resolve(
                    source_relative,
                    must_exist=True,
                    expect="file",
                )
                source_validation = self._validate_returnclock(source)
                if not source_validation.get("ok"):
                    issues.extend(
                        f"source:{item}"
                        for item in (
                            source_validation.get("issues") or []
                        )[:30]
                    )
                if _sha256_file(source) != manifest.get("source_sha256"):
                    issues.append("source_sha256_mismatch")
                source_rows = list(_iter_jsonl(source))
                if source_rows:
                    source_header = source_rows[0][1]
                    source_examples = [
                        row for _, row in source_rows[1:]
                    ]
            except DatasetToolboxError as exc:
                issues.append(f"source:{exc}")
        else:
            issues.append("source_missing")

        def row_fingerprint(row: Mapping[str, Any]) -> str:
            return json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        source_organic_examples = [
            row
            for row in source_examples
            if (row.get("post_cutoff") or {}).get(
                "organic_candidate"
            )
            is True
        ]
        source_fingerprints = Counter(
            row_fingerprint(row) for row in source_organic_examples
        )
        seen_users: set[str] = set()
        all_users: set[str] = set()
        seen_fingerprints: Counter[str] = Counter()
        total_rows = 0
        split_reports: dict[str, dict[str, Any]] = {}
        first_cutoff_bounds: list[tuple[str, str, str]] = []
        row_cutoff_bounds: list[tuple[str, str, str]] = []
        expected_key_id = manifest.get("pseudonymization_key_id")
        for split_name in ("train", "validation", "test"):
            entry = (manifest.get("splits") or {}).get(split_name)
            if not isinstance(entry, dict):
                issues.append(f"{split_name}:manifest_entry_missing")
                continue
            if set(entry) != _RETURNCLOCK_SPLIT_ENTRY_KEYS:
                issues.append(f"{split_name}:manifest_entry_allowlist")
            file_name = entry.get("file")
            if not isinstance(file_name, str):
                issues.append(f"{split_name}:file_missing")
                continue
            if file_name != f"{split_name}.jsonl":
                issues.append(f"{split_name}:unsafe_file_name")
                continue
            try:
                split_path = self.resolve(
                    path.relative_to(self.root) / file_name,
                    must_exist=True,
                    expect="file",
                )
            except DatasetToolboxError as exc:
                issues.append(f"{split_name}:{exc}")
                continue
            report = self._validate_returnclock(split_path)
            split_reports[split_name] = report
            if not report.get("ok"):
                issues.extend(
                    f"{split_name}:{item}"
                    for item in (report.get("issues") or [])[:30]
                )
            if _sha256_file(split_path) != entry.get("sha256"):
                issues.append(f"{split_name}:sha256_mismatch")

            users: set[str] = set()
            cutoffs: list[str] = []
            organic_rows = 0
            user_first_cutoffs: dict[str, str] = {}
            split_fingerprints: Counter[str] = Counter()
            has_cross_split_duplicate = False
            header: dict[str, Any] | None = None
            for _, row in _iter_jsonl(split_path):
                if row.get("record_type") == "header":
                    if header is not None:
                        issues.append(f"{split_name}:multiple_headers")
                    header = row
                    continue
                user_hash = str(row.get("user_id_hash") or "")
                cutoff = str(row.get("prediction_cutoff_at") or "")
                users.add(user_hash)
                cutoffs.append(cutoff)
                organic_rows += int(
                    (row.get("post_cutoff") or {}).get(
                        "organic_candidate"
                    )
                    is True
                )
                if (
                    (row.get("post_cutoff") or {}).get(
                        "organic_candidate"
                    )
                    is not True
                ):
                    issues.append(
                        f"{split_name}:nonorganic_training_row"
                    )
                previous = user_first_cutoffs.get(user_hash)
                if previous is None or cutoff < previous:
                    user_first_cutoffs[user_hash] = cutoff
                fingerprint = row_fingerprint(row)
                split_fingerprints[fingerprint] += 1
                if seen_fingerprints[fingerprint] > 0:
                    has_cross_split_duplicate = True
                seen_fingerprints[fingerprint] += 1
            if header is None:
                issues.append(f"{split_name}:header_missing")
            else:
                if header.get("split_name") != split_name:
                    issues.append(f"{split_name}:header_split_name")
                if (
                    header.get("split_assignment_basis")
                    != "user_first_prediction_cutoff"
                ):
                    issues.append(
                        f"{split_name}:header_assignment_basis"
                    )
                if (
                    header.get("pseudonymization_key_id")
                    != expected_key_id
                ):
                    issues.append(
                        f"{split_name}:pseudonymization_key_id_mismatch"
                    )
                if (
                    source is not None
                    and header.get("source_sha256")
                    != _sha256_file(source)
                ):
                    issues.append(
                        f"{split_name}:header_source_sha256_mismatch"
                    )
                if (
                    source_header
                    and header.get("feature_columns")
                    != source_header.get("feature_columns")
                ):
                    issues.append(
                        f"{split_name}:feature_columns_mismatch"
                    )
                if (
                    source_header
                    and header.get("source_summary")
                    != source_header.get("summary")
                ):
                    issues.append(
                        f"{split_name}:source_summary_mismatch"
                    )
            if (
                has_cross_split_duplicate
                or any(
                    count > 1
                    for count in split_fingerprints.values()
                )
            ):
                issues.append(f"{split_name}:duplicate_rows")
            if users & seen_users:
                issues.append(f"{split_name}:user_leakage")
            seen_users.update(users)
            all_users.update(users)
            total_rows += len(cutoffs)
            if entry.get("example_count") != len(cutoffs):
                issues.append(f"{split_name}:example_count_mismatch")
            if entry.get("user_count") != len(users):
                issues.append(f"{split_name}:user_count_mismatch")
            if entry.get("organic_example_count") != organic_rows:
                issues.append(
                    f"{split_name}:organic_example_count_mismatch"
                )
            if entry.get("treated_example_count") != (
                len(cutoffs) - organic_rows
            ):
                issues.append(
                    f"{split_name}:treated_example_count_mismatch"
                )
            if cutoffs and user_first_cutoffs:
                actual_anchor_min = min(user_first_cutoffs.values())
                actual_anchor_max = max(user_first_cutoffs.values())
                actual_row_min = min(cutoffs)
                actual_row_max = max(cutoffs)
                expected_bounds = {
                    "first_cutoff_min": actual_anchor_min,
                    "first_cutoff_max": actual_anchor_max,
                    "row_cutoff_min": actual_row_min,
                    "row_cutoff_max": actual_row_max,
                }
                for key, expected in expected_bounds.items():
                    if entry.get(key) != expected:
                        issues.append(f"{split_name}:{key}_mismatch")
                first_cutoff_bounds.append(
                    (split_name, actual_anchor_min, actual_anchor_max)
                )
                row_cutoff_bounds.append(
                    (split_name, actual_row_min, actual_row_max)
                )
            else:
                issues.append(f"{split_name}:empty_split")

        for earlier, later in zip(first_cutoff_bounds, first_cutoff_bounds[1:]):
            if earlier[2] >= later[1]:
                issues.append(
                    f"temporal_anchor_not_strict:{earlier[0]}:{later[0]}"
                )
        for earlier, later in zip(
            row_cutoff_bounds,
            row_cutoff_bounds[1:],
        ):
            if earlier[2] >= later[1]:
                issues.append(
                    f"temporal_row_not_strict:{earlier[0]}:{later[0]}"
                )
        if manifest.get("example_count") != total_rows:
            issues.append("manifest_example_count_mismatch")
        if manifest.get("source_example_count") != len(source_examples):
            issues.append("manifest_source_example_count_mismatch")
        source_treated_count = (
            len(source_examples) - len(source_organic_examples)
        )
        if manifest.get("excluded_treated_count") != source_treated_count:
            issues.append("manifest_excluded_treated_count_mismatch")
        excluded_temporal_count = (
            len(source_organic_examples) - total_rows
        )
        if (
            manifest.get("excluded_temporal_boundary_count")
            != excluded_temporal_count
        ):
            issues.append(
                "manifest_excluded_temporal_boundary_count_mismatch"
            )
        if manifest.get("user_count") != len(all_users):
            issues.append("manifest_user_count_mismatch")
        if any(
            count > source_fingerprints.get(fingerprint, 0)
            for fingerprint, count in seen_fingerprints.items()
        ):
            issues.append("split_rows_not_in_organic_source")
        if sum(
            source_fingerprints.values()
        ) - sum(seen_fingerprints.values()) != excluded_temporal_count:
            issues.append("split_exclusion_accounting_mismatch")
        source_users = {
            str(row.get("user_id_hash") or "")
            for row in source_organic_examples
        }
        if all_users != source_users:
            issues.append("split_users_do_not_match_source")
        if (
            source_header
            and expected_key_id
            != source_header.get("pseudonymization_key_id")
        ):
            issues.append("manifest_pseudonymization_key_id_mismatch")
        if (
            source_header
            and manifest.get("feature_columns")
            != source_header.get("feature_columns")
        ):
            issues.append("manifest_feature_columns_mismatch")
        ok = not issues
        natural_return_training_ready = bool(
            ok
            and total_rows > 0
            and len(split_reports) == 3
            and all(
                int(
                    (
                        (manifest.get("splits") or {}).get(name) or {}
                    ).get("example_count", 0)
                )
                > 0
                and int(
                    (
                        (manifest.get("splits") or {}).get(name) or {}
                    ).get("organic_example_count", 0)
                )
                > 0
                for name in ("train", "validation", "test")
            )
        )
        return {
            "ok": ok,
            "training_ready": natural_return_training_ready,
            "training_ready_scope": "natural_return_observational",
            "natural_return_training_ready": (
                natural_return_training_ready
            ),
            "causal_notification_training_ready": False,
            "kind": "returnclock_split",
            "path": str(path),
            "sha256": _sha256_bundle(path),
            "issues": issues,
            "summary": {
                "example_count": total_rows,
                "source_example_count": len(source_examples),
                "excluded_treated_count": source_treated_count,
                "excluded_temporal_boundary_count": (
                    excluded_temporal_count
                ),
                "distinct_users": len(all_users),
                "user_leakage": any(
                    issue.endswith(":user_leakage") for issue in issues
                ),
                "assignment_basis": manifest.get("assignment_basis"),
                "training_filter": manifest.get("training_filter"),
                "post_cutoff_excluded_from_features": manifest.get(
                    "post_cutoff_excluded_from_features"
                ),
                "pseudonymization_key_id": expected_key_id,
            },
        }

    def _validate_nemesis(self, path: Path) -> dict[str, Any]:
        issues: list[str] = []
        try:
            current_catalog_hash, current_card_count, _ = (
                self._require_current_catalog()
            )
        except DatasetToolboxError as exc:
            issues.append(str(exc))
            current_catalog_hash = None
            current_card_count = None
        records = _iter_jsonl(path)
        try:
            _, header = next(records)
        except StopIteration:
            return {
                "ok": False,
                "training_ready": False,
                "kind": "nemesis",
                "path": str(path),
                "issues": ["empty_dataset"],
            }
        if (
            header.get("record_type") != "header"
            or header.get("format") != NEMESIS_EXPORT_FORMAT
            or header.get("format_version") != 1
            or header.get("schema_version") != NEMESIS_SCHEMA
        ):
            issues.append("incompatible_header")
        if set(header) != _NEMESIS_HEADER_KEYS:
            issues.append("header_allowlist_mismatch")
        if _parse_utc_timestamp(header.get("created_at")) is None:
            issues.append("header_created_at_invalid")
        if _contains_raw_user_id(header):
            issues.append("header_raw_user_id")
        include_players = bool(header.get("include_players", False))
        privacy_safe = (
            not include_players
            and header.get("identity_scheme")
            == "side_pseudonyms_p1_1_p2_2"
            and header.get("record_id_scheme")
            == NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
            and header.get("player_group_scheme")
            == NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
        )

        battle_ids: set[str] = set()
        split_groups: set[str] = set()
        lite_split_groups: set[str] = set()
        standard_split_groups: set[str] = set()
        battle_count = lite = standard = eligible_weight = 0
        lite_positive_weight = standard_positive_weight = 0
        domains: dict[str, int] = {}
        for line_number, row in records:
            battle_count += 1
            row_privacy_issues: list[str] = []
            if _contains_raw_user_id(row):
                row_privacy_issues.append(
                    f"line_{line_number}:raw_user_id"
                )
            row_privacy_issues.extend(
                _v5_sensitive_data_issues(
                    row,
                    path=f"line_{line_number}",
                )
            )
            row_privacy = row.get("privacy")
            if (
                not isinstance(row_privacy, Mapping)
                or row_privacy.get("identity_scheme")
                != "side_pseudonyms_p1_1_p2_2"
                or row_privacy.get("include_players") is not False
                or row_privacy.get("record_id_scheme")
                != NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
                or row_privacy.get("player_group_scheme")
                != NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
            ):
                row_privacy_issues.append(
                    f"line_{line_number}:privacy_contract"
                )
            row_privacy_issues.extend(
                _record_id_privacy_issues(
                    row,
                    path=f"line_{line_number}",
                    scheme=_V5_PSEUDONYMIZED_RECORD_ID_SCHEME,
                )
            )
            if row_privacy_issues:
                privacy_safe = False
                issues.extend(row_privacy_issues)
            if row.get("record_type") != "battle":
                issues.append(f"line_{line_number}:record_type")
                continue
            payload = {
                key: value
                for key, value in row.items()
                if key != "record_type"
            }
            try:
                validated = validate_nemesis_record(
                    payload,
                    require_terminal=True,
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(f"line_{line_number}:{exc}")
                if len(issues) >= 100:
                    issues.append("issue_limit_reached")
                    break
                continue
            battle_id = str(validated["battle_id"])
            if battle_id in battle_ids:
                issues.append(f"line_{line_number}:duplicate_battle_id")
            battle_ids.add(battle_id)
            quality = validated["quality"]
            is_lite = bool(quality["eligible_lite"])
            is_standard = bool(quality["eligible_standard"])
            positive_weight = float(quality["sample_weight"]) > 0.0
            lite += is_lite
            standard += is_standard
            eligible_weight += positive_weight
            base = validated["features"]["base"]
            current_card_ids = {
                int(card_id)
                for card_id in (
                    (self._catalog_payload or {}).get("cards") or {}
                )
                if str(card_id).isdigit()
            }
            for seat_name, seat_payload in base["seats"].items():
                for card in seat_payload["initial_deck"]:
                    if card["card_id"] not in current_card_ids:
                        issues.append(
                            f"line_{line_number}:{seat_name}_deck_"
                            f"card_id_not_in_current_catalog"
                        )
            catalog_current = bool(
                current_catalog_hash is not None
                and base.get("catalog_available") is True
                and _catalog_hash_matches(
                    base.get("catalog_hash"),
                    current_catalog_hash,
                )
            )
            if not catalog_current:
                issues.append(
                    f"line_{line_number}:catalog_hash_not_current"
                )
            domain = str(base["domain"])
            domains[domain] = domains.get(domain, 0) + 1
            split_fingerprint = str(
                validated["provenance"]["split_fingerprint"]
            )
            split_groups.add(split_fingerprint)
            if is_lite and positive_weight and catalog_current:
                lite_positive_weight += 1
                lite_split_groups.add(split_fingerprint)
            if is_standard and positive_weight and catalog_current:
                standard_positive_weight += 1
                standard_split_groups.add(split_fingerprint)
            if privacy_safe:
                seats = base["seats"]
                if (
                    seats["p1"]["participant_id"] != 1
                    or seats["p2"]["participant_id"] != 2
                ):
                    issues.append(
                        f"line_{line_number}:participant_ids_not_pseudonymized"
                    )

        if header.get("battle_count") != battle_count:
            issues.append("header_battle_count_mismatch")
        ok = not issues
        # A three-way train/validation/test holdout needs at least one
        # independent matchup group per partition.
        lite_split_ready = len(lite_split_groups) >= 3
        standard_split_ready = len(standard_split_groups) >= 3
        training_ready_lite = bool(
            ok
            and privacy_safe
            and lite_positive_weight > 0
            and lite_split_ready
        )
        standard_readiness_blockers = (
            ["player_disjoint_split_not_materialized"]
            if standard_positive_weight > 0
            else []
        )
        training_ready_standard = bool(
            ok
            and privacy_safe
            and standard_positive_weight > 0
            and standard_split_ready
            and not standard_readiness_blockers
        )
        return {
            "ok": ok,
            "training_ready": bool(
                training_ready_lite or training_ready_standard
            ),
            "training_ready_lite": training_ready_lite,
            "training_ready_standard": training_ready_standard,
            "standard_readiness_blockers": standard_readiness_blockers,
            "kind": "nemesis",
            "path": str(path),
            "sha256": _sha256_file(path),
            "privacy_safe": privacy_safe,
            "issues": issues,
            "summary": {
                "battle_count": battle_count,
                "eligible_lite": lite,
                "eligible_standard": standard,
                "positive_weight_records": eligible_weight,
                "eligible_lite_positive_weight": lite_positive_weight,
                "eligible_standard_positive_weight": (
                    standard_positive_weight
                ),
                "domains": domains,
                "distinct_split_groups": len(split_groups),
                "lite_distinct_split_groups": len(lite_split_groups),
                "standard_distinct_split_groups": len(
                    standard_split_groups
                ),
                "grouped_split_possible": bool(
                    lite_split_ready or standard_split_ready
                ),
                "current_catalog_hash": current_catalog_hash,
                "current_card_count": current_card_count,
            },
        }

    @staticmethod
    def _nemesis_split_algorithms() -> dict[str, dict[str, str]]:
        return {
            "lite_deck_grouped": {
                "population": "eligible_lite_positive_current_catalog",
                "grouped_by": "provenance.split_fingerprint",
                "ordered_by": "sha256_group_key",
                "purpose": "lite_primary_train_validation_test",
            },
            "standard_player_disjoint": {
                "population": "eligible_standard_human_human_positive_current_catalog",
                "assigned_by": (
                    "three_disjoint_battle_anchors_then_balanced_sha256_aliases"
                ),
                "row_inclusion": (
                    "both_player_aliases_assigned_to_same_partition"
                ),
                "cross_partition_edges": (
                    "excluded_and_fingerprinted_in_manifest"
                ),
                "purpose": "standard_primary_player_generalization",
            },
            "standard_chronological": {
                "population": "eligible_standard_human_human_positive_current_catalog",
                "grouped_by": "features.base.feature_cutoff_at",
                "ordered_by": "feature_cutoff_at_utc",
                "purpose": "standard_temporal_drift_evaluation",
            },
            "standard_deck_grouped": {
                "population": "eligible_standard_human_human_positive_current_catalog",
                "grouped_by": "provenance.split_fingerprint",
                "ordered_by": "sha256_group_key",
                "purpose": "standard_deck_generalization_evaluation",
            },
        }

    @staticmethod
    def _nemesis_split_feature_contract() -> dict[str, Any]:
        return {
            "lite_feature_roots": ["features.base"],
            "standard_feature_roots": [
                "features.base",
                "features.extended",
            ],
            "label_root": "label",
            "grouping_metadata_roots": [
                "privacy.player_group_aliases",
                "provenance.split_fingerprint",
            ],
            "forbidden_feature_roots": [
                "battle_id",
                "match_id",
                "privacy",
                "provenance",
                "label",
            ],
            "player_group_aliases_are_features": False,
            "record_ids_are_features": False,
        }

    @staticmethod
    def _bounded_nemesis_exclusions(
        exclusions: Any,
        *,
        fingerprint_limit: int = 20,
    ) -> dict[str, Any]:
        """Return a bounded MCP-safe exclusion summary.

        The artifact manifest retains the complete fingerprint ledger.  MCP
        inspection and tool results expose only a deterministic sample plus
        exact counts so a 100k-battle split cannot produce a multi-megabyte
        response.
        """

        if not isinstance(exclusions, Mapping):
            return {}
        bounded: dict[str, Any] = {}
        for population in ("lite", "standard"):
            entry = exclusions.get(population)
            if isinstance(entry, Mapping):
                bounded[population] = {
                    key: entry.get(key)
                    for key in (
                        "source_count",
                        "included_count",
                        "excluded_count",
                        "by_reason",
                    )
                }
        player_entry = exclusions.get("standard_player_disjoint")
        if isinstance(player_entry, Mapping):
            fingerprints = player_entry.get(
                "excluded_cross_partition_fingerprints"
            )
            fingerprint_rows = (
                [str(value) for value in fingerprints]
                if isinstance(fingerprints, list)
                else []
            )
            bounded["standard_player_disjoint"] = {
                key: player_entry.get(key)
                for key in (
                    "source_eligible_count",
                    "included_count",
                    "excluded_cross_partition_count",
                    "assigned_player_count",
                    "assigned_players_by_split",
                )
            }
            bounded["standard_player_disjoint"].update(
                {
                    "excluded_cross_partition_fingerprints_sample": (
                        fingerprint_rows[:fingerprint_limit]
                    ),
                    "excluded_cross_partition_fingerprints_truncated": (
                        len(fingerprint_rows) > fingerprint_limit
                    ),
                }
            )
        return bounded

    def _nemesis_split_populations(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        current_hash, _, _ = self._require_current_catalog()
        lite: list[dict[str, Any]] = []
        standard: list[dict[str, Any]] = []
        lite_reasons: Counter[str] = Counter()
        standard_reasons: Counter[str] = Counter()
        for row in rows:
            quality = row.get("quality")
            features = row.get("features")
            base = (
                features.get("base")
                if isinstance(features, Mapping)
                else None
            )
            reasons_common: list[str] = []
            if not isinstance(quality, Mapping):
                reasons_common.append("quality_missing")
                positive_weight = False
                eligible_lite = False
                eligible_standard = False
            else:
                positive_weight = (
                    _finite_number(quality.get("sample_weight")) or 0.0
                ) > 0.0
                eligible_lite = quality.get("eligible_lite") is True
                eligible_standard = (
                    quality.get("eligible_standard") is True
                )
            if not positive_weight:
                reasons_common.append("nonpositive_sample_weight")
            catalog_current = bool(
                isinstance(base, Mapping)
                and base.get("catalog_available") is True
                and _catalog_hash_matches(
                    base.get("catalog_hash"),
                    current_hash,
                )
            )
            if not catalog_current:
                reasons_common.append("catalog_not_current")

            lite_exclusions = list(reasons_common)
            if not eligible_lite:
                lite_exclusions.append("not_eligible_lite")
            if not lite_exclusions:
                lite.append(row)
            else:
                lite_reasons.update(set(lite_exclusions))

            standard_exclusions = list(reasons_common)
            if not eligible_standard:
                standard_exclusions.append("not_eligible_standard")
            if not isinstance(base, Mapping) or base.get("domain") != (
                "human-human"
            ):
                standard_exclusions.append("not_human_human")
            extended = (
                features.get("extended")
                if isinstance(features, Mapping)
                else None
            )
            if (
                not isinstance(extended, Mapping)
                or any(extended.get(seat) is None for seat in ("p1", "p2"))
            ):
                standard_exclusions.append(
                    "pre_match_extended_features_missing"
                )
            if not standard_exclusions:
                standard.append(row)
            else:
                standard_reasons.update(set(standard_exclusions))

        exclusions = {
            "lite": {
                "source_count": len(rows),
                "included_count": len(lite),
                "excluded_count": len(rows) - len(lite),
                "by_reason": dict(sorted(lite_reasons.items())),
            },
            "standard": {
                "source_count": len(rows),
                "included_count": len(standard),
                "excluded_count": len(rows) - len(standard),
                "by_reason": dict(sorted(standard_reasons.items())),
            },
        }
        return lite, standard, exclusions

    @staticmethod
    def _partition_nemesis_players(
        rows: list[dict[str, Any]],
        *,
        train_fraction: float,
        validation_fraction: float,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, Any],
    ]:
        """Assign players, then exclude only cross-partition battle edges.

        Matchmaking data normally forms one giant connected graph, so grouping
        whole connected components would make a three-way holdout impossible.
        Instead, seed every partition with a pairwise-disjoint battle, assign
        all remaining aliases deterministically, and retain only rows whose two
        players land in the same partition.
        """

        row_aliases: dict[str, tuple[str, str]] = {}
        row_by_fingerprint: dict[str, dict[str, Any]] = {}
        unique_edges: dict[tuple[str, str], str] = {}
        degree: Counter[str] = Counter()
        for row in rows:
            privacy = row.get("privacy")
            aliases = (
                privacy.get("player_group_aliases")
                if isinstance(privacy, Mapping)
                else None
            )
            if not isinstance(aliases, Mapping):
                raise DatasetToolboxError(
                    "standard Nemesis row has no player group aliases"
                )
            p1 = str(aliases.get("p1") or "")
            p2 = str(aliases.get("p2") or "")
            if (
                re.fullmatch(r"player_[0-9a-f]{32}", p1) is None
                or re.fullmatch(r"player_[0-9a-f]{32}", p2) is None
                or p1 == p2
            ):
                raise DatasetToolboxError(
                    "standard Nemesis row has invalid player group aliases"
                )
            fingerprint = DatasetToolbox._nemesis_split_row_fingerprint(row)
            row_aliases[fingerprint] = (p1, p2)
            row_by_fingerprint[fingerprint] = row
            edge = tuple(sorted((p1, p2)))
            current = unique_edges.get(edge)
            if current is None or fingerprint < current:
                unique_edges[edge] = fingerprint
            degree.update(edge)

        all_aliases = sorted(
            {
                alias
                for aliases in row_aliases.values()
                for alias in aliases
            },
            key=lambda alias: (
                hashlib.sha256(alias.encode("utf-8")).hexdigest(),
                alias,
            ),
        )
        if len(all_aliases) < 6:
            raise DatasetToolboxError(
                "Nemesis player-disjoint split requires at least six "
                "distinct players"
            )
        ordered_edges = sorted(
            unique_edges.items(),
            key=lambda item: (
                degree[item[0][0]] + degree[item[0][1]],
                max(degree[item[0][0]], degree[item[0][1]]),
                hashlib.sha256(
                    ("\n".join(item[0]) + "\n" + item[1]).encode("utf-8")
                ).hexdigest(),
            ),
        )
        anchor_edges: list[tuple[str, str]] = []
        anchored_aliases: set[str] = set()
        for edge, _ in ordered_edges:
            if anchored_aliases.isdisjoint(edge):
                anchor_edges.append(edge)
                anchored_aliases.update(edge)
                if len(anchor_edges) == len(_NEMESIS_SPLIT_NAMES):
                    break
        if len(anchor_edges) != len(_NEMESIS_SPLIT_NAMES):
            raise DatasetToolboxError(
                "Nemesis player-disjoint split requires three pairwise-"
                "disjoint human-human battles"
            )

        alias_count = len(all_aliases)
        train_target = max(
            2,
            min(alias_count - 4, int(alias_count * train_fraction)),
        )
        validation_target = max(
            2,
            min(
                alias_count - train_target - 2,
                int(alias_count * validation_fraction),
            ),
        )
        targets = {
            "train": train_target,
            "validation": validation_target,
            "test": alias_count - train_target - validation_target,
        }
        alias_to_split: dict[str, str] = {}
        assigned_counts = dict.fromkeys(_NEMESIS_SPLIT_NAMES, 0)
        for split_name, edge in zip(_NEMESIS_SPLIT_NAMES, anchor_edges):
            for alias in edge:
                alias_to_split[alias] = split_name
                assigned_counts[split_name] += 1
        for alias in all_aliases:
            if alias in alias_to_split:
                continue
            split_name = min(
                _NEMESIS_SPLIT_NAMES,
                key=lambda candidate: (
                    -(
                        targets[candidate]
                        - assigned_counts[candidate]
                    ),
                    _NEMESIS_SPLIT_NAMES.index(candidate),
                ),
            )
            alias_to_split[alias] = split_name
            assigned_counts[split_name] += 1

        partitions: dict[str, list[dict[str, Any]]] = {
            split_name: [] for split_name in _NEMESIS_SPLIT_NAMES
        }
        cross_partition: list[str] = []
        for fingerprint, aliases in sorted(row_aliases.items()):
            p1_split = alias_to_split[aliases[0]]
            p2_split = alias_to_split[aliases[1]]
            if p1_split != p2_split:
                cross_partition.append(fingerprint)
                continue
            partitions[p1_split].append(row_by_fingerprint[fingerprint])
        for split_name, split_rows in partitions.items():
            split_rows.sort(
                key=lambda row: (
                    _parse_utc_timestamp(
                        ((row.get("features") or {}).get("base") or {}).get(
                            "feature_cutoff_at"
                        )
                    )
                    or datetime.min.replace(tzinfo=timezone.utc),
                    str(row.get("battle_id") or ""),
                )
            )
            if not split_rows:
                raise DatasetToolboxError(
                    "Nemesis player-disjoint assignment produced an empty "
                    f"{split_name} partition"
                )
        metadata = {
            "source_eligible_count": len(rows),
            "included_count": sum(
                len(split_rows) for split_rows in partitions.values()
            ),
            "excluded_cross_partition_count": len(cross_partition),
            "excluded_cross_partition_fingerprints": cross_partition,
            "assigned_player_count": alias_count,
            "assigned_players_by_split": {
                split_name: assigned_counts[split_name]
                for split_name in _NEMESIS_SPLIT_NAMES
            },
        }
        return partitions, metadata

    @staticmethod
    def _partition_nemesis_groups(
        rows: list[dict[str, Any]],
        *,
        group_key: Callable[[dict[str, Any]], str],
        group_order: Callable[[str], Any],
        train_fraction: float,
        validation_fraction: float,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(group_key(row), []).append(row)
        if len(grouped) < 3:
            raise DatasetToolboxError(
                "Nemesis split requires at least three independent groups"
            )
        ordered = sorted(grouped.items(), key=lambda item: group_order(item[0]))
        group_count = len(ordered)
        train_end = max(
            1,
            min(group_count - 2, int(group_count * train_fraction)),
        )
        validation_end = max(
            train_end + 1,
            min(
                group_count - 1,
                int(
                    group_count
                    * (train_fraction + validation_fraction)
                ),
            ),
        )
        group_partitions = {
            "train": ordered[:train_end],
            "validation": ordered[train_end:validation_end],
            "test": ordered[validation_end:],
        }
        result: dict[str, list[dict[str, Any]]] = {}
        for split_name, split_groups in group_partitions.items():
            split_rows = [
                row
                for _, group_rows in split_groups
                for row in group_rows
            ]
            split_rows.sort(
                key=lambda row: (
                    _parse_utc_timestamp(
                        ((row.get("features") or {}).get("base") or {}).get(
                            "feature_cutoff_at"
                        )
                    )
                    or datetime.min.replace(tzinfo=timezone.utc),
                    str(row.get("battle_id") or ""),
                )
            )
            if not split_rows:
                raise DatasetToolboxError(
                    f"Nemesis {split_name} partition is empty"
                )
            result[split_name] = split_rows
        return result

    def _nemesis_partition_plan(
        self,
        *,
        lite_rows: list[dict[str, Any]],
        standard_rows: list[dict[str, Any]],
        train_fraction: float,
        validation_fraction: float,
    ) -> tuple[
        dict[str, dict[str, list[dict[str, Any]]]],
        dict[str, Any] | None,
        list[str],
    ]:
        """Build every currently usable Nemesis assignment.

        Lite remains independently materializable for headless/model-vs-model
        exports.  Standard regimes are published atomically only when all
        three Standard evaluation views satisfy their preconditions.
        """

        def deck_group(row: dict[str, Any]) -> str:
            return str(
                (row.get("provenance") or {}).get("split_fingerprint")
                or ""
            )

        def cutoff_group(row: dict[str, Any]) -> str:
            cutoff = _parse_utc_timestamp(
                ((row.get("features") or {}).get("base") or {}).get(
                    "feature_cutoff_at"
                )
            )
            if cutoff is None:
                raise DatasetToolboxError(
                    "Nemesis Standard record has invalid feature cutoff"
                )
            return cutoff.isoformat()

        def hashed_order(key: str) -> str:
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

        partitions = {
            "lite_deck_grouped": self._partition_nemesis_groups(
                lite_rows,
                group_key=deck_group,
                group_order=hashed_order,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        }
        if not standard_rows:
            return partitions, None, ["no_eligible_standard_records"]

        blockers: list[str] = []
        player_partitions: (
            dict[str, list[dict[str, Any]]] | None
        ) = None
        player_exclusions: dict[str, Any] | None = None
        chronological_partitions: (
            dict[str, list[dict[str, Any]]] | None
        ) = None
        deck_partitions: dict[str, list[dict[str, Any]]] | None = None
        try:
            player_partitions, player_exclusions = (
                self._partition_nemesis_players(
                    standard_rows,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                )
            )
        except DatasetToolboxError:
            blockers.append("player_disjoint_preconditions_not_met")
        try:
            chronological_partitions = self._partition_nemesis_groups(
                standard_rows,
                group_key=cutoff_group,
                group_order=lambda key: _parse_utc_timestamp(key)
                or datetime.min.replace(tzinfo=timezone.utc),
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        except DatasetToolboxError:
            blockers.append(
                "chronological_split_needs_three_cutoff_groups"
            )
        try:
            deck_partitions = self._partition_nemesis_groups(
                standard_rows,
                group_key=deck_group,
                group_order=hashed_order,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        except DatasetToolboxError:
            blockers.append("deck_split_needs_three_matchup_groups")

        if not blockers:
            assert player_partitions is not None
            assert chronological_partitions is not None
            assert deck_partitions is not None
            partitions.update(
                {
                    "standard_player_disjoint": player_partitions,
                    "standard_chronological": chronological_partitions,
                    "standard_deck_grouped": deck_partitions,
                }
            )
        return partitions, player_exclusions, blockers

    @staticmethod
    def _nemesis_split_row_fingerprint(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _validate_nemesis_split(self, path: Path) -> dict[str, Any]:
        issues: list[str] = []
        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink():
            return {
                "ok": False,
                "training_ready": False,
                "training_ready_lite": False,
                "training_ready_standard": False,
                "kind": "nemesis_split",
                "path": str(path),
                "issues": ["manifest_symlink_forbidden"],
            }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "training_ready": False,
                "training_ready_lite": False,
                "training_ready_standard": False,
                "kind": "nemesis_split",
                "path": str(path),
                "issues": [f"manifest:{exc}"],
            }
        if manifest.get("format") != _NEMESIS_SPLIT_FORMAT:
            issues.append("incompatible_split_manifest")
        if manifest.get("format_version") != 1:
            issues.append("incompatible_split_manifest_version")
        if set(manifest) != _NEMESIS_SPLIT_MANIFEST_KEYS:
            issues.append("split_manifest_allowlist_mismatch")
        if _contains_raw_user_id(manifest):
            issues.append("split_manifest_raw_user_id")
        if _parse_utc_timestamp(manifest.get("created_at")) is None:
            issues.append("split_manifest_created_at_invalid")
        if manifest.get("algorithms") != self._nemesis_split_algorithms():
            issues.append("split_algorithms_contract")
        if (
            manifest.get("feature_contract")
            != self._nemesis_split_feature_contract()
        ):
            issues.append("split_feature_contract")
        expected_privacy = {
            "identity_scheme": "side_pseudonyms_p1_1_p2_2",
            "include_players": False,
            "player_group_scheme": (
                NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
            ),
            "record_id_scheme": NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME,
            "grouping_aliases_export_local": True,
            "grouping_aliases_are_training_features": False,
        }
        if manifest.get("privacy") != expected_privacy:
            issues.append("split_privacy_contract")
        requested_fractions = manifest.get("requested_fractions")
        if (
            not isinstance(requested_fractions, Mapping)
            or set(requested_fractions) != set(_NEMESIS_SPLIT_NAMES)
        ):
            issues.append("split_requested_fractions_contract")
            train_fraction = None
            validation_fraction = None
            test_fraction = None
        else:
            train_fraction = _finite_number(
                requested_fractions.get("train")
            )
            validation_fraction = _finite_number(
                requested_fractions.get("validation")
            )
            test_fraction = _finite_number(
                requested_fractions.get("test")
            )
            fractions = (
                train_fraction,
                validation_fraction,
                test_fraction,
            )
            if (
                any(
                    value is None or value <= 0.0 or value >= 1.0
                    for value in fractions
                )
                or abs(sum(value or 0.0 for value in fractions) - 1.0)
                > 1e-9
            ):
                issues.append("split_requested_fractions_values")

        source: Path | None = None
        source_rows: list[dict[str, Any]] = []
        source_header: dict[str, Any] = {}
        source_relative = manifest.get("source")
        if isinstance(source_relative, str):
            try:
                source = self.resolve(
                    source_relative,
                    must_exist=True,
                    expect="file",
                )
                if self._detect_path_kind(source) != "nemesis":
                    issues.append("source_not_nemesis")
                source_validation = self._validate_nemesis(source)
                if not source_validation.get("ok"):
                    issues.extend(
                        f"source:{item}"
                        for item in (
                            source_validation.get("issues") or []
                        )[:30]
                    )
                if source_validation.get("privacy_safe") is not True:
                    issues.append("source_not_privacy_safe")
                if _sha256_file(source) != manifest.get("source_sha256"):
                    issues.append("source_sha256_mismatch")
                raw_source = list(_iter_jsonl(source))
                if raw_source:
                    source_header = raw_source[0][1]
                    source_rows = [row for _, row in raw_source[1:]]
            except DatasetToolboxError as exc:
                issues.append(f"source:{exc}")
        else:
            issues.append("source_missing")

        if source_header:
            if manifest.get("source_format") != source_header.get("format"):
                issues.append("source_format_mismatch")
            if manifest.get("source_schema_version") != source_header.get(
                "schema_version"
            ):
                issues.append("source_schema_version_mismatch")
            if manifest.get("source_battle_count") != len(source_rows):
                issues.append("source_battle_count_mismatch")
        try:
            current_hash, current_count, _ = self._require_current_catalog()
        except DatasetToolboxError as exc:
            issues.append(str(exc))
            current_hash = None
            current_count = None
        if manifest.get("catalog") != {
            "required_current": True,
            "catalog_hash": current_hash,
            "card_count": current_count,
        }:
            issues.append("split_catalog_contract")

        try:
            lite_rows, standard_rows, exclusions = (
                self._nemesis_split_populations(source_rows)
            )
        except DatasetToolboxError as exc:
            issues.append(f"source_population:{exc}")
            lite_rows = []
            standard_rows = []
            exclusions = {}
        expected_partitions: dict[
            str,
            dict[str, list[dict[str, Any]]],
        ] = {}
        standard_blockers: list[str] = []
        if (
            lite_rows
            and train_fraction is not None
            and validation_fraction is not None
        ):
            try:
                (
                    expected_partitions,
                    player_exclusions,
                    standard_blockers,
                ) = self._nemesis_partition_plan(
                    lite_rows=lite_rows,
                    standard_rows=standard_rows,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                )
                if player_exclusions is not None:
                    exclusions = {
                        **exclusions,
                        "standard_player_disjoint": player_exclusions,
                    }
            except DatasetToolboxError as exc:
                issues.append(f"split_partition_plan:{exc}")
        elif not lite_rows:
            issues.append("split_source_has_no_eligible_lite_records")
        if manifest.get("exclusions") != exclusions:
            issues.append("split_exclusions_mismatch")
        source_by_fingerprint = {
            self._nemesis_split_row_fingerprint(row): row
            for row in source_rows
        }
        expected_population = {
            regime: [
                row
                for split_name in _NEMESIS_SPLIT_NAMES
                for row in regime_partitions[split_name]
            ]
            for regime, regime_partitions in expected_partitions.items()
        }

        artifacts = manifest.get("artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != set(expected_partitions)
        ):
            issues.append("split_artifacts_contract")
            artifacts = {}
        regime_reports: dict[str, Any] = {}
        all_expected_paths: set[str] = set()
        for regime in (
            candidate
            for candidate in _NEMESIS_SPLIT_REGIMES
            if candidate in expected_partitions
        ):
            split_entries = artifacts.get(regime)
            if (
                not isinstance(split_entries, Mapping)
                or set(split_entries) != set(_NEMESIS_SPLIT_NAMES)
            ):
                issues.append(f"{regime}:split_entries_contract")
                continue
            expected_rows = expected_population[regime]
            expected_fingerprints = Counter(
                self._nemesis_split_row_fingerprint(row)
                for row in expected_rows
            )
            seen_fingerprints: Counter[str] = Counter()
            seen_group_keys: set[str] = set()
            seen_player_aliases: set[str] = set()
            split_reports: dict[str, Any] = {}
            chronological_bounds: list[tuple[str, datetime, datetime]] = []
            for split_name in _NEMESIS_SPLIT_NAMES:
                entry = split_entries.get(split_name)
                if not isinstance(entry, Mapping):
                    issues.append(f"{regime}:{split_name}:entry_missing")
                    continue
                if set(entry) != _NEMESIS_SPLIT_ENTRY_KEYS:
                    issues.append(
                        f"{regime}:{split_name}:entry_allowlist"
                    )
                expected_file = f"{regime}/{split_name}.jsonl"
                file_name = entry.get("file")
                if file_name != expected_file:
                    issues.append(
                        f"{regime}:{split_name}:unsafe_file_name"
                    )
                    continue
                if file_name in all_expected_paths:
                    issues.append(
                        f"{regime}:{split_name}:duplicate_artifact_path"
                    )
                all_expected_paths.add(str(file_name))
                try:
                    split_path = self.resolve(
                        path.relative_to(self.root) / str(file_name),
                        must_exist=True,
                        expect="file",
                    )
                except DatasetToolboxError as exc:
                    issues.append(f"{regime}:{split_name}:{exc}")
                    continue
                split_validation = self._validate_nemesis(split_path)
                split_reports[split_name] = split_validation
                if not split_validation.get("ok"):
                    issues.extend(
                        f"{regime}:{split_name}:{item}"
                        for item in (
                            split_validation.get("issues") or []
                        )[:30]
                    )
                if _sha256_file(split_path) != entry.get("sha256"):
                    issues.append(
                        f"{regime}:{split_name}:sha256_mismatch"
                    )
                raw_split_rows = list(_iter_jsonl(split_path))
                split_header = (
                    raw_split_rows[0][1] if raw_split_rows else {}
                )
                split_rows = [
                    row for _, row in raw_split_rows[1:]
                ]
                if not split_rows:
                    issues.append(f"{regime}:{split_name}:empty_split")
                    continue
                if (
                    split_header.get("battle_count")
                    != len(split_rows)
                ):
                    issues.append(
                        f"{regime}:{split_name}:header_count_mismatch"
                    )
                if {
                    key: split_header.get(key)
                    for key in _NEMESIS_HEADER_KEYS - {"battle_count"}
                } != {
                    key: source_header.get(key)
                    for key in _NEMESIS_HEADER_KEYS - {"battle_count"}
                }:
                    issues.append(
                        f"{regime}:{split_name}:header_contract_mismatch"
                    )
                split_fingerprints = Counter(
                    self._nemesis_split_row_fingerprint(row)
                    for row in split_rows
                )
                if regime == "standard_player_disjoint":
                    expected_for_split = Counter(
                        self._nemesis_split_row_fingerprint(row)
                        for row in expected_partitions[regime][split_name]
                    )
                    if split_fingerprints != expected_for_split:
                        issues.append(
                            f"{regime}:{split_name}:assignment_mismatch"
                        )
                for fingerprint, count in split_fingerprints.items():
                    if fingerprint not in source_by_fingerprint:
                        issues.append(
                            f"{regime}:{split_name}:row_not_in_source"
                        )
                    if seen_fingerprints[fingerprint]:
                        issues.append(
                            f"{regime}:{split_name}:row_leakage"
                        )
                    seen_fingerprints[fingerprint] += count

                aliases: set[str] = set()
                deck_groups: set[str] = set()
                cutoff_groups: set[str] = set()
                cutoffs: list[datetime] = []
                for row in split_rows:
                    privacy = row.get("privacy") or {}
                    player_aliases = privacy.get(
                        "player_group_aliases"
                    ) or {}
                    aliases.update(
                        str(player_aliases.get(seat) or "")
                        for seat in ("p1", "p2")
                    )
                    deck_group = str(
                        (row.get("provenance") or {}).get(
                            "split_fingerprint"
                        )
                        or ""
                    )
                    deck_groups.add(deck_group)
                    cutoff = _parse_utc_timestamp(
                        ((row.get("features") or {}).get("base") or {}).get(
                            "feature_cutoff_at"
                        )
                    )
                    if cutoff is not None:
                        cutoffs.append(cutoff)
                        cutoff_groups.add(cutoff.isoformat())

                if regime in {
                    "lite_deck_grouped",
                    "standard_deck_grouped",
                }:
                    group_keys = deck_groups
                elif regime == "standard_player_disjoint":
                    group_keys = aliases
                    if aliases & seen_player_aliases:
                        issues.append(
                            f"{regime}:{split_name}:player_leakage"
                        )
                    seen_player_aliases.update(aliases)
                else:
                    group_keys = cutoff_groups
                if group_keys & seen_group_keys:
                    issues.append(f"{regime}:{split_name}:group_leakage")
                seen_group_keys.update(group_keys)
                if entry.get("example_count") != len(split_rows):
                    issues.append(
                        f"{regime}:{split_name}:example_count_mismatch"
                    )
                if entry.get("group_count") != len(group_keys):
                    issues.append(
                        f"{regime}:{split_name}:group_count_mismatch"
                    )
                if entry.get("player_group_count") != len(aliases):
                    issues.append(
                        f"{regime}:{split_name}:player_count_mismatch"
                    )
                if cutoffs:
                    cutoff_min = min(cutoffs)
                    cutoff_max = max(cutoffs)
                    if entry.get(
                        "feature_cutoff_min"
                    ) != cutoff_min.isoformat():
                        issues.append(
                            f"{regime}:{split_name}:cutoff_min_mismatch"
                        )
                    if entry.get(
                        "feature_cutoff_max"
                    ) != cutoff_max.isoformat():
                        issues.append(
                            f"{regime}:{split_name}:cutoff_max_mismatch"
                        )
                    if regime == "standard_chronological":
                        chronological_bounds.append(
                            (split_name, cutoff_min, cutoff_max)
                        )
                else:
                    issues.append(f"{regime}:{split_name}:cutoff_missing")
            if seen_fingerprints != expected_fingerprints:
                issues.append(f"{regime}:population_coverage_mismatch")
            if regime == "standard_chronological":
                for earlier, later in zip(
                    chronological_bounds,
                    chronological_bounds[1:],
                ):
                    if earlier[2] >= later[1]:
                        issues.append(
                            f"{regime}:temporal_order_not_strict:"
                            f"{earlier[0]}:{later[0]}"
                        )
            regime_reports[regime] = split_reports

        expected_bundle_files = {
            "manifest.json",
            *all_expected_paths,
        }
        actual_bundle_files: set[str] = set()
        for item in path.rglob("*"):
            relative_item = str(item.relative_to(path))
            if item.is_symlink():
                issues.append(f"bundle_symlink_forbidden:{relative_item}")
                continue
            if item.is_file():
                actual_bundle_files.add(relative_item)
                if stat.S_IMODE(item.stat().st_mode) != 0o600:
                    issues.append(
                        f"bundle_file_mode_not_private:{relative_item}"
                    )
            elif item.is_dir() and stat.S_IMODE(item.stat().st_mode) != 0o700:
                issues.append(
                    f"bundle_directory_mode_not_private:{relative_item}"
                )
        if actual_bundle_files != expected_bundle_files:
            issues.append("bundle_file_allowlist_mismatch")

        standard_regimes = set(_NEMESIS_SPLIT_REGIMES[1:])
        expected_readiness = {
            "training_ready_lite": (
                "lite_deck_grouped" in expected_partitions
            ),
            "training_ready_standard": (
                standard_regimes.issubset(expected_partitions)
                and not standard_blockers
            ),
            "standard_readiness_blockers": standard_blockers,
            "standard_primary_assignment": "standard_player_disjoint",
            "standard_evaluation_assignments": [
                "standard_chronological",
                "standard_deck_grouped",
            ],
            "one_split_satisfies_all_constraints": False,
        }
        if manifest.get("training_readiness") != expected_readiness:
            issues.append("split_training_readiness_mismatch")
        ok = not issues
        training_ready_lite = bool(
            ok and expected_readiness["training_ready_lite"]
        )
        training_ready_standard = bool(
            ok and expected_readiness["training_ready_standard"]
        )
        return {
            "ok": ok,
            "training_ready": bool(
                training_ready_lite or training_ready_standard
            ),
            "training_ready_lite": training_ready_lite,
            "training_ready_standard": training_ready_standard,
            "kind": "nemesis_split",
            "path": str(path),
            "issues": issues,
            "summary": {
                "source_battle_count": len(source_rows),
                "eligible_lite": len(lite_rows),
                "eligible_standard": len(standard_rows),
                "standard_readiness_blockers": standard_blockers,
                "regimes": regime_reports,
                "player_disjoint_leakage": any(
                    "player_leakage" in issue for issue in issues
                ),
            },
            "sha256": _sha256_bundle(path),
        }

    def _validate_v5_export(self, path: Path) -> dict[str, Any]:
        header = _first_record(path)
        privacy_safe = (
            not bool(header.get("include_players", False))
            and header.get("privacy") == "side_pseudonyms_p1_1_p2_2"
            and header.get("record_id_scheme")
            == _V5_PSEUDONYMIZED_RECORD_ID_SCHEME
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=".validate-v5-",
                dir=self.root,
            ) as temporary:
                target = Path(temporary) / "materialized"
                manifest = materialize_export(
                    path,
                    target,
                    group_id="validation",
                )
                self._attach_current_catalog(target)
                materialized_report = self._validate_v5_materialized(
                    target
                )
        except (MaterializationError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "training_ready": False,
                "kind": "v5_export",
                "path": str(path),
                "sha256": _sha256_file(path),
                "privacy_safe": False,
                "issues": [str(exc)],
            }
        battle_count = len(manifest.get("battle_ids") or [])
        issues = list(materialized_report.get("issues") or [])
        privacy_safe = bool(
            privacy_safe and materialized_report.get("privacy_safe")
        )
        if not privacy_safe:
            issues.append("privacy_header_not_pseudonymized")
        ok = bool(materialized_report.get("ok")) and not issues
        v5_policy_training_ready = bool(
            privacy_safe
            and materialized_report.get("v5_policy_training_ready")
        )
        return {
            "ok": ok,
            "training_ready": v5_policy_training_ready,
            "training_ready_scope": "v5_policy_only",
            "v5_policy_training_ready": v5_policy_training_ready,
            "metronome_training_ready": bool(
                materialized_report.get(
                    "metronome_training_ready"
                )
            ),
            "timestamp_training_ready": bool(
                materialized_report.get(
                    "timestamp_training_ready"
                )
            ),
            "kind": "v5_export",
            "path": str(path),
            "sha256": _sha256_file(path),
            "privacy_safe": privacy_safe,
            "issues": issues,
            "summary": {
                "battle_count": battle_count,
                "results": manifest.get("results"),
                "readiness": {
                    key: materialized_report.get(key)
                    for key in (
                        "v5_policy_training_ready",
                        "metronome_training_ready",
                        "timestamp_training_ready",
                    )
                },
                "materialized": materialized_report.get("summary"),
                "collection_classes": (
                    manifest.get("spec") or {}
                ).get("collection_classes"),
            },
        }

    def _validate_v5_materialized(self, path: Path) -> dict[str, Any]:
        from rlhf_env.components.v5_trace_validate import (
            validate_v5_metronome_contract,
            validate_v5_timestamp_contract,
            validate_v5_trace,
        )

        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink():
            return {
                "ok": False,
                "training_ready": False,
                "kind": "v5_materialized",
                "path": str(path),
                "issues": ["manifest_symlink_forbidden"],
            }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "training_ready": False,
                "kind": "v5_materialized",
                "path": str(path),
                "issues": [f"manifest:{exc}"],
            }
        issues: list[str] = []
        if set(manifest) != _V5_MATERIALIZED_MANIFEST_KEYS:
            issues.append("manifest_allowlist_mismatch")
        if manifest.get("materialized_format") != V5_MATERIALIZED_FORMAT:
            issues.append("incompatible_materialized_format")
        if manifest.get("storage_schema") != "rlhf_v5_storage_v1":
            issues.append("incompatible_storage_schema")
        spec = manifest.get("spec")
        if not isinstance(spec, dict):
            issues.append("manifest_spec_missing")
            spec = {}
        elif set(spec) != _V5_MATERIALIZED_SPEC_KEYS:
            issues.append("manifest_spec_allowlist_mismatch")
        env = manifest.get("env")
        if (
            not isinstance(env, dict)
            or set(env) != _V5_MATERIALIZED_ENV_KEYS
        ):
            issues.append("manifest_env_allowlist_mismatch")
        results_summary = manifest.get("results")
        if (
            not isinstance(results_summary, dict)
            or set(results_summary) != _V5_MATERIALIZED_RESULT_KEYS
        ):
            issues.append("manifest_results_allowlist_mismatch")
        battle_results = manifest.get("battles_results")
        if not isinstance(battle_results, list):
            issues.append("manifest_battle_results_missing")
            battle_results = []
        else:
            for index, result in enumerate(battle_results):
                if (
                    not isinstance(result, dict)
                    or set(result)
                    != _V5_MATERIALIZED_BATTLE_RESULT_KEYS
                ):
                    issues.append(
                        f"manifest_battle_result_{index}_allowlist"
                    )
        privacy_issues = [
            f"manifest_privacy:{issue}"
            for issue in _v5_privacy_issues(
                manifest,
                [],
                [],
                require_meta_seats=False,
            )
        ]
        privacy_issues.extend(
            f"manifest_privacy:{issue}"
            for issue in _v5_sensitive_data_issues(
                manifest,
                path="manifest",
            )
        )
        issues.extend(privacy_issues)
        privacy_contract_safe = bool(
            spec.get("privacy") == "side_pseudonyms_p1_1_p2_2"
            and spec.get("include_players") is False
            and spec.get("record_id_scheme")
            in {
                _V5_PSEUDONYMIZED_RECORD_ID_SCHEME,
                _V5_NATIVE_RECORD_ID_SCHEME,
            }
        )
        if not privacy_contract_safe:
            issues.append("privacy_header_not_pseudonymized")
        try:
            expected_catalog_hash, expected_card_count, _ = (
                self._require_current_catalog()
            )
        except DatasetToolboxError as exc:
            issues.append(str(exc))
            expected_catalog_hash = None
            expected_card_count = None
        catalog_path = path / "catalog.json"
        if catalog_path.is_symlink():
            issues.append("catalog_symlink_forbidden")
        elif self._catalog_payload is not None:
            try:
                attached_catalog = json.loads(
                    catalog_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"catalog_invalid:{exc}")
            else:
                if attached_catalog != self._catalog_payload:
                    issues.append("catalog_payload_not_current")
        if (
            expected_catalog_hash is not None
            and spec.get("current_catalog_hash") != expected_catalog_hash
        ):
            issues.append("manifest_current_catalog_hash_mismatch")
        if (
            expected_card_count is not None
            and spec.get("current_card_count") != expected_card_count
        ):
            issues.append("manifest_current_card_count_mismatch")
        battle_ids = manifest.get("battle_ids")
        if not isinstance(battle_ids, list) or not battle_ids:
            issues.append("manifest_battle_ids_missing")
            battle_ids = []
        elif not all(
            isinstance(battle_id, str) and battle_id
            for battle_id in battle_ids
        ):
            issues.append("manifest_battle_ids_must_be_strings")
            battle_ids = [str(battle_id) for battle_id in battle_ids]
        if len(battle_ids) != len(set(map(str, battle_ids))):
            issues.append("manifest_duplicate_battle_ids")
        record_id_scheme = str(spec.get("record_id_scheme") or "")
        record_id_issues = _record_id_privacy_issues(
            manifest,
            path="manifest",
            scheme=record_id_scheme,
        )
        privacy_issues.extend(record_id_issues)
        issues.extend(record_id_issues)
        battle_result_ids: list[str] = []
        battle_results_by_id: dict[str, dict[str, Any]] = {}
        for index, result in enumerate(battle_results):
            if not isinstance(result, dict):
                continue
            result_battle_id = result.get("battle_id")
            if not isinstance(result_battle_id, str) or not result_battle_id:
                issues.append(
                    f"manifest_battle_result_{index}_battle_id"
                )
                continue
            battle_result_ids.append(result_battle_id)
            if result_battle_id in battle_results_by_id:
                issues.append("manifest_duplicate_battle_result_ids")
            else:
                battle_results_by_id[result_battle_id] = result
        if battle_result_ids != battle_ids:
            issues.append("manifest_battle_result_ids_mismatch")
        validated_battles = 0
        accepted_training_rows = 0
        rejected_rows = 0
        auxiliary_issues: list[str] = []
        human_battles = 0
        timestamp_contract_battles = 0
        metronome_contract_battles = 0
        metronome_observed_labels = 0
        expected_battle_results: list[dict[str, Any]] = []
        for battle_id in battle_ids:
            try:
                v5_dir = self._safe_v5_dir(path, battle_id)
            except DatasetToolboxError as exc:
                issues.append(f"{battle_id}:{exc}")
                continue
            trace_files = (
                v5_dir / "meta.json",
                v5_dir / "turns.jsonl",
                v5_dir / "actions.jsonl",
            )
            if any(trace_file.is_symlink() for trace_file in trace_files):
                issues.append(f"{battle_id}:trace_file_symlink_forbidden")
                continue
            report = validate_v5_trace(
                v5_dir,
                expected_catalog_hash=expected_catalog_hash,
                expected_card_count=expected_card_count,
            )
            if not report.get("ok"):
                issues.extend(
                    f"{battle_id}:{issue}"
                    for issue in (report.get("issues") or [])[:20]
                )
            else:
                validated_battles += 1
            try:
                meta = json.loads(
                    (v5_dir / "meta.json").read_text(encoding="utf-8")
                )
                turns = [
                    row
                    for _, row in _iter_jsonl(v5_dir / "turns.jsonl")
                ]
                actions = [
                    row
                    for _, row in _iter_jsonl(v5_dir / "actions.jsonl")
                ]
            except (OSError, json.JSONDecodeError, DatasetToolboxError) as exc:
                issues.append(f"{battle_id}:read:{exc}")
                continue
            expected_result = _expected_v5_battle_result(
                battle_id,
                meta,
                turns,
            )
            if expected_result is None:
                issues.append(
                    f"{battle_id}:manifest_result_source_fields_invalid"
                )
            else:
                expected_battle_results.append(expected_result)
                actual_result = battle_results_by_id.get(battle_id)
                if actual_result is None:
                    issues.append(
                        f"{battle_id}:manifest_battle_result_missing"
                    )
                else:
                    for field, expected_value in expected_result.items():
                        if not _manifest_values_equal(
                            actual_result.get(field),
                            expected_value,
                        ):
                            issues.append(
                                f"{battle_id}:manifest_battle_result_"
                                f"{field}_mismatch"
                            )
            has_human = bool(
                meta.get("p1_actor_type") == "human"
                or meta.get("p2_actor_type") == "human"
                or any(
                    action.get("decision_source") == "human"
                    for action in actions
                )
            )
            human_battles += int(has_human)
            timestamp_issues = validate_v5_timestamp_contract(meta)
            metronome_issues = validate_v5_metronome_contract(actions)
            if has_human and not timestamp_issues:
                timestamp_contract_battles += 1
            if not metronome_issues:
                metronome_contract_battles += 1
            metronome_observed_labels += sum(
                action.get("decision_source") == "human"
                and action.get("control_source") == "human"
                and action.get("decision_time_censored") is False
                and (
                    action.get("human_decision_time_raw_ms") is not None
                    or action.get("human_decision_time_ms") is not None
                )
                for action in actions
            )
            auxiliary_issues.extend(
                f"{battle_id}:{issue}"
                for issue in timestamp_issues
            )
            auxiliary_issues.extend(
                f"{battle_id}:{issue}"
                for issue in metronome_issues
            )
            bundle_privacy_issues = [
                f"{battle_id}:privacy:{issue}"
                for issue in _v5_privacy_issues(meta, turns, actions)
            ]
            bundle_privacy_issues.extend(
                f"{battle_id}:privacy:{issue}"
                for issue in _v5_sensitive_data_issues(
                    {
                        "meta": meta,
                        "turns": turns,
                        "actions": actions,
                    },
                    path="bundle",
                )
            )
            bundle_privacy_issues.extend(
                f"{battle_id}:privacy:{issue}"
                for issue in _record_id_privacy_issues(
                    {
                        "meta": meta,
                        "turns": turns,
                        "actions": actions,
                    },
                    path="bundle",
                    scheme=record_id_scheme,
                )
            )
            privacy_issues.extend(bundle_privacy_issues)
            issues.extend(bundle_privacy_issues)
            issues.extend(
                f"{battle_id}:policy:{issue}"
                for issue in _v5_policy_issues(meta)
            )
            issues.extend(
                f"{battle_id}:ruleset:{issue}"
                for issue in _v5_current_ruleset_issues(
                    meta,
                    turns,
                    actions,
                )
            )
            accepted_training_rows += sum(
                action.get("accepted") is True
                and action.get("action_type")
                not in {"surrender", "draw", "stalemate"}
                and isinstance(action.get("legal_action_index"), int)
                and not isinstance(
                    action.get("legal_action_index"), bool
                )
                for action in actions
            )
            rejected_rows += sum(
                action.get("accepted") is False for action in actions
            )
            if len(issues) >= 100:
                issues.append("issue_limit_reached")
                break
        if len(expected_battle_results) != len(battle_ids):
            issues.append("manifest_results_trace_count_mismatch")
        else:
            expected_summary = _expected_v5_results_summary(
                expected_battle_results
            )
            if not isinstance(results_summary, dict):
                issues.append("manifest_results_missing")
            else:
                for field, expected_value in expected_summary.items():
                    if not _manifest_values_equal(
                        results_summary.get(field),
                        expected_value,
                    ):
                        issues.append(
                            f"manifest_results_{field}_mismatch"
                        )
        privacy_safe = bool(
            privacy_contract_safe and not privacy_issues
        )
        policy_ok = not issues
        all_issues = [*issues, *auxiliary_issues]
        ok = not all_issues
        v5_policy_training_ready = bool(
            policy_ok
            and privacy_safe
            and battle_ids
            and validated_battles == len(battle_ids)
            and accepted_training_rows > 0
        )
        timestamp_training_ready = bool(
            policy_ok
            and privacy_safe
            and human_battles > 0
            and timestamp_contract_battles == human_battles
        )
        metronome_training_ready = bool(
            policy_ok
            and privacy_safe
            and battle_ids
            and metronome_contract_battles == len(battle_ids)
            and metronome_observed_labels > 0
        )
        return {
            "ok": ok,
            "training_ready": v5_policy_training_ready,
            "training_ready_scope": "v5_policy_only",
            "v5_policy_training_ready": v5_policy_training_ready,
            "metronome_training_ready": metronome_training_ready,
            "timestamp_training_ready": timestamp_training_ready,
            "kind": "v5_materialized",
            "path": str(path),
            "privacy_safe": privacy_safe,
            "issues": all_issues,
            "policy_issues": issues,
            "auxiliary_issues": auxiliary_issues,
            "summary": {
                "battle_count": len(battle_ids),
                "validated_battles": validated_battles,
                "accepted_training_rows": accepted_training_rows,
                "rejected_audit_rows": rejected_rows,
                "human_battles": human_battles,
                "timestamp_contract_battles": (
                    timestamp_contract_battles
                ),
                "metronome_contract_battles": (
                    metronome_contract_battles
                ),
                "metronome_observed_labels": (
                    metronome_observed_labels
                ),
                "current_catalog_hash": expected_catalog_hash,
                "current_card_count": expected_card_count,
                "results": manifest.get("results"),
            },
        }

    # ------------------------------------------------------------------
    # Exports and materialization
    # ------------------------------------------------------------------
    def _require_production_data(self) -> None:
        if not self.production_enabled:
            raise DatasetToolboxError(
                "production_data_disabled: set "
                "RLHF_ENABLE_PRODUCTION_DATASETS=1 when intentionally "
                "exporting read-only production telemetry"
            )

    def _database(self) -> tuple[Any, Any]:
        self._require_production_data()
        if self._database_factory is None or self._settings_factory is None:
            from infrastructure.config import get_settings
            from infrastructure.database import Database

            settings = get_settings()
            return Database(settings.database), settings
        settings = self._settings_factory()
        return self._database_factory(settings.database), settings

    async def export_production_v5(
        self,
        *,
        output: str | Path,
        days: int = 30,
        limit_battles: int = 1000,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        destination = self._prepare_output(
            output,
            suffix=".jsonl",
            overwrite=overwrite,
        )
        normalized_days = max(1, min(int(days), 365))
        bounded_limit = max(1, min(int(limit_battles), 10_000))
        db, _ = self._database()
        await db.connect()
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.candidate.",
            dir=destination.parent,
        ) as candidate_directory:
            candidate_root = Path(candidate_directory)
            candidate_root.chmod(0o700)
            candidate = candidate_root / "dataset.jsonl"
            try:
                selection = await db.list_v5_export_battle_ids(
                    days=normalized_days,
                    limit_battles=bounded_limit,
                )
                battle_ids = [
                    str(value)
                    for value in selection.get("battle_ids", [])
                    if str(value)
                ]
                if not battle_ids:
                    raise DatasetToolboxError("no_valid_v5_battles")
                header = {
                    key: value
                    for key, value in selection.items()
                    if key != "battle_ids"
                }
                header.update(
                    {
                        "record_type": "header",
                        "privacy": "side_pseudonyms_p1_1_p2_2",
                        "include_players": False,
                        "record_id_scheme": (
                            _V5_PSEUDONYMIZED_RECORD_ID_SCHEME
                        ),
                        "battle_count": len(battle_ids),
                        "skipped_invalid": 0,
                        "current_catalog_hash": (
                            self.current_catalog_hash
                        ),
                        "current_card_count": self.current_card_count,
                        "notes": (
                            "Each following line is one complete terminal "
                            "rlhf_v5_storage_v1 battle bundle."
                        ),
                    }
                )
                with candidate.open("x", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(
                        json.dumps(
                            header,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=_json_default,
                        )
                        + "\n"
                    )
                    record_id_namespace = uuid.uuid4()
                    for battle_id in battle_ids:
                        bundle = await db.get_v5_export_battle_bundle(
                            battle_id=battle_id,
                            include_players=False,
                            record_id_namespace=record_id_namespace,
                        )
                        if bundle is None:
                            raise DatasetToolboxError(
                                f"v5_export_bundle_invalid:{battle_id}"
                            )
                        handle.write(
                            json.dumps(
                                {"record_type": "battle", **bundle},
                                ensure_ascii=False,
                                sort_keys=True,
                                default=_json_default,
                            )
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                await db.close()
            validation = self.validate_artifact(candidate)
            if not validation.get("ok"):
                raise DatasetToolboxError(
                    "exported V5 dataset failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )
            os.replace(candidate, destination)
            destination.chmod(0o600)
        return {
            "ok": True,
            "kind": "v5_export",
            "output": str(destination),
            "battle_count": validation["summary"]["battle_count"],
            "sha256": validation["sha256"],
            "privacy": "side_pseudonyms_p1_1_p2_2",
            "training_ready": validation["training_ready"],
            "training_ready_scope": validation.get(
                "training_ready_scope"
            ),
            "v5_policy_training_ready": validation.get(
                "v5_policy_training_ready"
            ),
            "metronome_training_ready": validation.get(
                "metronome_training_ready"
            ),
            "timestamp_training_ready": validation.get(
                "timestamp_training_ready"
            ),
        }

    async def export_returnclock(
        self,
        *,
        output: str | Path,
        start: str | None = None,
        end: str | None = None,
        horizon_hours: int = 7 * 24,
        safety_lag_minutes: int = 10,
        min_analytics_version: int = 2,
        limit: int = 50_000,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self._require_production_data()
        destination = self._prepare_output(
            output,
            suffix=".jsonl",
            overwrite=overwrite,
        )
        from scripts.export_returnclock_dataset import export_returnclock_dataset

        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.candidate.",
            dir=destination.parent,
        ) as candidate_directory:
            candidate_root = Path(candidate_directory)
            candidate_root.chmod(0o700)
            candidate = candidate_root / "dataset.jsonl"
            result = await export_returnclock_dataset(
                output=candidate,
                start=start,
                end=end,
                horizon_hours=horizon_hours,
                safety_lag_minutes=safety_lag_minutes,
                min_analytics_version=min_analytics_version,
                limit=limit,
                salt_env=self.returnclock_salt_env,
                salt_key_id_env=self.returnclock_salt_key_id_env,
                database_factory=self._database_factory,
                settings_factory=self._settings_factory,
            )
            validation = self.validate_artifact(candidate)
            if not validation.get("ok"):
                raise DatasetToolboxError(
                    "exported ReturnClock dataset failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )
            os.replace(candidate, destination)
            destination.chmod(0o600)
        return {
            **result,
            "kind": "returnclock",
            "output": str(destination),
            "sha256": validation["sha256"],
            "training_ready": validation["training_ready"],
            "training_ready_scope": validation.get(
                "training_ready_scope"
            ),
            "natural_return_training_ready": validation.get(
                "natural_return_training_ready"
            ),
            "causal_notification_training_ready": False,
            "split_readiness": validation["summary"],
        }

    def split_returnclock(
        self,
        *,
        source: str | Path,
        output_dir: str | Path,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Create deterministic user-grouped, time-ordered train/val/test sets.

        A user's rows never cross partitions. User groups are ordered by their
        first prediction cutoff and assigned to contiguous partitions, keeping
        evaluation later in cohort time without random user leakage.
        """

        source_path = self.resolve(source, must_exist=True, expect="file")
        if self._detect_path_kind(source_path) != "returnclock":
            raise DatasetToolboxError("source must be a ReturnClock JSONL export")
        source_validation = self._validate_returnclock(source_path)
        if not source_validation.get("ok"):
            raise DatasetToolboxError(
                "source ReturnClock dataset is invalid: "
                + "; ".join(source_validation.get("issues") or [])
            )
        train_fraction = float(train_fraction)
        validation_fraction = float(validation_fraction)
        if not 0.0 < train_fraction < 1.0:
            raise DatasetToolboxError("train_fraction must be in (0, 1)")
        if not 0.0 < validation_fraction < 1.0:
            raise DatasetToolboxError(
                "validation_fraction must be in (0, 1)"
            )
        if train_fraction + validation_fraction >= 1.0:
            raise DatasetToolboxError(
                "train_fraction + validation_fraction must be < 1"
            )

        records = list(_iter_jsonl(source_path))
        header = records[0][1]
        source_examples = [record for _, record in records[1:]]
        examples = [
            row
            for row in source_examples
            if (row.get("post_cutoff") or {}).get(
                "organic_candidate"
            )
            is True
        ]
        excluded_treated_count = len(source_examples) - len(examples)
        by_user: dict[str, list[dict[str, Any]]] = {}
        for row in examples:
            user_hash = str(row["user_id_hash"])
            by_user.setdefault(user_hash, []).append(row)
        if len(by_user) < 3:
            raise DatasetToolboxError(
                "at least three distinct users are required for "
                "train/validation/test grouped split"
            )
        grouped = sorted(
            (
                (
                    min(
                        str(row["prediction_cutoff_at"])
                        for row in user_rows
                    ),
                    user_hash,
                    sorted(
                        user_rows,
                        key=lambda row: (
                            str(row["prediction_cutoff_at"]),
                            str(row["user_id_hash"]),
                        ),
                    ),
                )
                for user_hash, user_rows in by_user.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        group_count = len(grouped)
        strict_boundaries = [
            index
            for index in range(1, group_count)
            if grouped[index - 1][0] < grouped[index][0]
        ]
        if len(strict_boundaries) < 2:
            raise DatasetToolboxError(
                "ReturnClock split needs at least three strictly ordered "
                "user cohorts"
            )
        requested_train_end = max(
            1,
            min(group_count - 2, int(group_count * train_fraction)),
        )
        train_count = min(
            strict_boundaries[:-1],
            key=lambda value: (
                abs(value - requested_train_end),
                value,
            ),
        )
        requested_validation_end = max(
            train_count + 1,
            min(
                group_count - 1,
                int(
                    group_count
                    * (train_fraction + validation_fraction)
                ),
            ),
        )
        validation_end = min(
            [
                boundary
                for boundary in strict_boundaries
                if boundary > train_count
            ],
            key=lambda value: (
                abs(value - requested_validation_end),
                value,
            ),
        )
        partitions = {
            "train": grouped[:train_count],
            "validation": grouped[
                train_count:validation_end
            ],
            "test": grouped[validation_end:],
        }
        validation_boundary = grouped[train_count][0]
        test_boundary = grouped[validation_end][0]

        def _inside_temporal_partition(
            split_name: str,
            row: Mapping[str, Any],
        ) -> bool:
            cutoff = str(row["prediction_cutoff_at"])
            if split_name == "train":
                return cutoff < validation_boundary
            if split_name == "validation":
                return validation_boundary <= cutoff < test_boundary
            return cutoff >= test_boundary

        destination = self._prepare_output(
            output_dir,
            overwrite=overwrite,
        )
        if destination.exists() and not destination.is_dir():
            raise DatasetToolboxError(
                "ReturnClock split output must be a directory"
            )
        if source_path == destination or destination in source_path.parents:
            raise DatasetToolboxError("source and split output must differ")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            )
        )
        temporary.chmod(0o700)
        split_entries: dict[str, dict[str, Any]] = {}
        try:
            for split_name, user_groups in partitions.items():
                split_examples = sorted(
                    [
                        row
                        for _, _, user_rows in user_groups
                        for row in user_rows
                        if _inside_temporal_partition(
                            split_name,
                            row,
                        )
                    ],
                    key=lambda row: (
                        str(row["prediction_cutoff_at"]),
                        str(row["user_id_hash"]),
                    ),
                )
                observed = sum(
                    bool(row["label"]["target_observed"])
                    for row in split_examples
                )
                censored = sum(
                    bool(row["label"]["right_censored"])
                    for row in split_examples
                )
                organic = sum(
                    bool(row["post_cutoff"]["organic_candidate"])
                    for row in split_examples
                )
                treated = 0
                complete_no_return = sum(
                    not bool(row["label"]["target_observed"])
                    and not bool(row["label"]["right_censored"])
                    for row in split_examples
                )
                split_header = {
                    **header,
                    "generated_at": header.get("generated_at"),
                    "split_name": split_name,
                    "split_assignment_basis": (
                        "user_first_prediction_cutoff"
                    ),
                    "source_sha256": _sha256_file(source_path),
                    "source_summary": header.get("summary"),
                    "summary": {
                        "example_count": len(split_examples),
                        "observed_returns": observed,
                        "complete_no_return_horizons": complete_no_return,
                        "right_censored": censored,
                        "treated_intervals": treated,
                        "organic_candidates": organic,
                        "excluded_legacy_sessions": 0,
                        "excluded_unfinished_sessions": 0,
                        "excluded_nonmeaningful_sessions": 0,
                        "inferred_stale_session_ends": 0,
                    },
                }
                split_path = temporary / f"{split_name}.jsonl"
                with split_path.open("x", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    for record in (split_header, *split_examples):
                        handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                split_users = {
                    str(row["user_id_hash"])
                    for row in split_examples
                }
                anchors = [
                    min(
                        str(row["prediction_cutoff_at"])
                        for row in split_examples
                        if str(row["user_id_hash"]) == user_hash
                    )
                    for user_hash in sorted(split_users)
                ]
                split_entries[split_name] = {
                    "file": split_path.name,
                    "sha256": _sha256_file(split_path),
                    "example_count": len(split_examples),
                    "organic_example_count": organic,
                    "treated_example_count": treated,
                    "user_count": len(split_users),
                    "first_cutoff_min": min(anchors),
                    "first_cutoff_max": max(anchors),
                    "row_cutoff_min": min(
                        str(row["prediction_cutoff_at"])
                        for row in split_examples
                    ),
                    "row_cutoff_max": max(
                        str(row["prediction_cutoff_at"])
                        for row in split_examples
                    ),
                }

            manifest = {
                "format": "extraarena_returnclock_split_v1",
                "format_version": 1,
                "source": str(source_path.relative_to(self.root)),
                "source_sha256": _sha256_file(source_path),
                "pseudonymization_key_id": header.get(
                    "pseudonymization_key_id"
                ),
                "assignment_basis": "user_first_prediction_cutoff",
                "strategy": {
                    "grouped_by": "user_id_hash",
                    "ordered_by": "prediction_cutoff_at",
                    "assignment_basis": "user_first_prediction_cutoff",
                },
                "requested_fractions": {
                    "train": train_fraction,
                    "validation": validation_fraction,
                    "test": 1.0 - train_fraction - validation_fraction,
                },
                "training_filter": {
                    "field": "post_cutoff.organic_candidate",
                    "equals": True,
                },
                "source_example_count": len(source_examples),
                "example_count": sum(
                    entry["example_count"]
                    for entry in split_entries.values()
                ),
                "excluded_treated_count": excluded_treated_count,
                "excluded_temporal_boundary_count": (
                    len(examples)
                    - sum(
                        entry["example_count"]
                        for entry in split_entries.values()
                    )
                ),
                "user_count": len(by_user),
                "splits": split_entries,
                "feature_columns": header.get("feature_columns"),
                "post_cutoff_excluded_from_features": True,
            }
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            validation = self._validate_returnclock_split(temporary)
            if not validation.get("training_ready"):
                raise DatasetToolboxError(
                    "generated ReturnClock split failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )

            backup: Path | None = None
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.backup.",
                        dir=destination.parent,
                    )
                )
                backup.rmdir()
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            if backup is not None:
                if backup.is_dir():
                    shutil.rmtree(backup, ignore_errors=True)
                else:
                    backup.unlink(missing_ok=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return {
            "ok": True,
            "kind": "returnclock_split",
            "source": str(source_path),
            "output": str(destination),
            "source_sha256": manifest["source_sha256"],
            "sha256": _sha256_bundle(destination),
            "pseudonymization_key_id": manifest[
                "pseudonymization_key_id"
            ],
            "example_count": manifest["example_count"],
            "source_example_count": len(source_examples),
            "excluded_treated_count": excluded_treated_count,
            "excluded_temporal_boundary_count": manifest[
                "excluded_temporal_boundary_count"
            ],
            "user_count": len(by_user),
            "splits": split_entries,
            "training_ready": True,
            "training_ready_scope": "natural_return_observational",
            "natural_return_training_ready": True,
            "causal_notification_training_ready": False,
        }

    def split_nemesis(
        self,
        *,
        source: str | Path,
        output_dir: str | Path,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Publish separately audited Lite and Standard Nemesis assignments.

        Grouping aliases remain record metadata and are used only to construct
        the player-disjoint Standard assignment.  They are never copied into
        either Lite or Standard feature roots.
        """

        train_fraction = float(train_fraction)
        validation_fraction = float(validation_fraction)
        test_fraction = 1.0 - train_fraction - validation_fraction
        if (
            not all(
                math.isfinite(value)
                for value in (
                    train_fraction,
                    validation_fraction,
                    test_fraction,
                )
            )
            or min(
                train_fraction,
                validation_fraction,
                test_fraction,
            )
            <= 0.0
            or max(
                train_fraction,
                validation_fraction,
                test_fraction,
            )
            >= 1.0
        ):
            raise DatasetToolboxError(
                "Nemesis split fractions must be finite, positive and sum "
                "to one"
            )
        source_path = self.resolve(source, must_exist=True, expect="file")
        if self._detect_path_kind(source_path) != "nemesis":
            raise DatasetToolboxError(
                "Nemesis split source must be a canonical Nemesis export"
            )
        source_validation = self._validate_nemesis(source_path)
        if (
            not source_validation.get("ok")
            or source_validation.get("privacy_safe") is not True
        ):
            raise DatasetToolboxError(
                "Nemesis split source failed canonical privacy/schema "
                "validation: "
                + "; ".join(source_validation.get("issues") or [])
            )
        raw_source = list(_iter_jsonl(source_path))
        if not raw_source:
            raise DatasetToolboxError("Nemesis split source is empty")
        header = raw_source[0][1]
        source_rows = [row for _, row in raw_source[1:]]
        if (
            header.get("include_players") is not False
            or header.get("identity_scheme")
            != "side_pseudonyms_p1_1_p2_2"
            or header.get("player_group_scheme")
            != NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
            or header.get("record_id_scheme")
            != NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
        ):
            raise DatasetToolboxError(
                "Nemesis split requires a pseudonymized export with "
                "export-local player grouping aliases"
            )

        lite_rows, standard_rows, exclusions = (
            self._nemesis_split_populations(source_rows)
        )
        if not lite_rows:
            raise DatasetToolboxError(
                "Nemesis split has no eligible Lite records"
            )
        partitions, player_exclusions, standard_blockers = (
            self._nemesis_partition_plan(
                lite_rows=lite_rows,
                standard_rows=standard_rows,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        )
        if player_exclusions is not None:
            exclusions = {
                **exclusions,
                "standard_player_disjoint": player_exclusions,
            }

        def deck_group(row: dict[str, Any]) -> str:
            return str(
                (row.get("provenance") or {}).get("split_fingerprint")
                or ""
            )

        def cutoff_group(row: dict[str, Any]) -> str:
            cutoff = _parse_utc_timestamp(
                ((row.get("features") or {}).get("base") or {}).get(
                    "feature_cutoff_at"
                )
            )
            if cutoff is None:
                raise DatasetToolboxError(
                    "Nemesis Standard record has invalid feature cutoff"
                )
            return cutoff.isoformat()

        destination = self._prepare_output(
            output_dir,
            overwrite=overwrite,
        )
        if destination.exists() and not destination.is_dir():
            raise DatasetToolboxError(
                "Nemesis split output must be a directory"
            )
        if source_path == destination or destination in source_path.parents:
            raise DatasetToolboxError("source and split output must differ")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            )
        )
        temporary.chmod(0o700)
        split_entries: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            for regime, regime_partitions in partitions.items():
                regime_root = temporary / regime
                regime_root.mkdir(mode=0o700)
                split_entries[regime] = {}
                for split_name, split_rows in regime_partitions.items():
                    split_path = regime_root / f"{split_name}.jsonl"
                    split_header = {
                        **header,
                        "battle_count": len(split_rows),
                    }
                    with split_path.open("x", encoding="utf-8") as handle:
                        os.fchmod(handle.fileno(), 0o600)
                        for row in (split_header, *split_rows):
                            handle.write(
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        handle.flush()
                        os.fsync(handle.fileno())

                    aliases = {
                        str(
                            ((row.get("privacy") or {}).get(
                                "player_group_aliases"
                            ) or {}).get(seat)
                            or ""
                        )
                        for row in split_rows
                        for seat in ("p1", "p2")
                    }
                    if regime in {
                        "lite_deck_grouped",
                        "standard_deck_grouped",
                    }:
                        groups = {deck_group(row) for row in split_rows}
                    elif regime == "standard_player_disjoint":
                        groups = aliases
                    else:
                        groups = {cutoff_group(row) for row in split_rows}
                    cutoffs = [
                        _parse_utc_timestamp(
                            ((row.get("features") or {}).get("base") or {}).get(
                                "feature_cutoff_at"
                            )
                        )
                        for row in split_rows
                    ]
                    if any(cutoff is None for cutoff in cutoffs):
                        raise DatasetToolboxError(
                            f"{regime}/{split_name} has invalid cutoff"
                        )
                    typed_cutoffs = [
                        cutoff for cutoff in cutoffs if cutoff is not None
                    ]
                    split_entries[regime][split_name] = {
                        "file": f"{regime}/{split_path.name}",
                        "sha256": _sha256_file(split_path),
                        "example_count": len(split_rows),
                        "group_count": len(groups),
                        "player_group_count": len(aliases),
                        "feature_cutoff_min": min(
                            typed_cutoffs
                        ).isoformat(),
                        "feature_cutoff_max": max(
                            typed_cutoffs
                        ).isoformat(),
                    }

            current_hash, current_count, _ = self._require_current_catalog()
            manifest = {
                "format": _NEMESIS_SPLIT_FORMAT,
                "format_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source_path.relative_to(self.root)),
                "source_sha256": _sha256_file(source_path),
                "source_format": header.get("format"),
                "source_schema_version": header.get("schema_version"),
                "source_battle_count": len(source_rows),
                "privacy": {
                    "identity_scheme": "side_pseudonyms_p1_1_p2_2",
                    "include_players": False,
                    "player_group_scheme": (
                        NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
                    ),
                    "record_id_scheme": (
                        NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
                    ),
                    "grouping_aliases_export_local": True,
                    "grouping_aliases_are_training_features": False,
                },
                "catalog": {
                    "required_current": True,
                    "catalog_hash": current_hash,
                    "card_count": current_count,
                },
                "requested_fractions": {
                    "train": train_fraction,
                    "validation": validation_fraction,
                    "test": test_fraction,
                },
                "algorithms": self._nemesis_split_algorithms(),
                "exclusions": exclusions,
                "artifacts": split_entries,
                "feature_contract": (
                    self._nemesis_split_feature_contract()
                ),
                "training_readiness": {
                    "training_ready_lite": True,
                    "training_ready_standard": not standard_blockers,
                    "standard_readiness_blockers": standard_blockers,
                    "standard_primary_assignment": (
                        "standard_player_disjoint"
                    ),
                    "standard_evaluation_assignments": [
                        "standard_chronological",
                        "standard_deck_grouped",
                    ],
                    "one_split_satisfies_all_constraints": False,
                },
            }
            self._write_private_json(
                temporary / "manifest.json",
                manifest,
            )
            validation = self._validate_nemesis_split(temporary)
            if (
                not validation.get("training_ready_lite")
                or bool(validation.get("training_ready_standard"))
                != (not standard_blockers)
            ):
                raise DatasetToolboxError(
                    "generated Nemesis split failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )

            backup: Path | None = None
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.backup.",
                        dir=destination.parent,
                    )
                )
                backup.rmdir()
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            if backup is not None:
                if backup.is_dir():
                    shutil.rmtree(backup, ignore_errors=True)
                else:
                    backup.unlink(missing_ok=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return {
            "ok": True,
            "kind": "nemesis_split",
            "source": str(source_path),
            "output": str(destination),
            "source_sha256": manifest["source_sha256"],
            "source_battle_count": len(source_rows),
            "eligible_lite": len(lite_rows),
            "eligible_standard": len(standard_rows),
            "exclusions": self._bounded_nemesis_exclusions(exclusions),
            "artifacts": split_entries,
            "training_ready": True,
            "training_ready_lite": True,
            "training_ready_standard": not standard_blockers,
            "standard_readiness_blockers": standard_blockers,
            "sha256": _sha256_bundle(destination),
        }

    def export_nemesis(
        self,
        *,
        input_path: str | Path,
        output: str | Path,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        source = self.resolve(input_path, must_exist=True, expect="file")
        return self.export_nemesis_records(
            records=load_nemesis_records(source),
            output=output,
            source_label=str(source),
            overwrite=overwrite,
        )

    def export_nemesis_records(
        self,
        *,
        records: Iterable[Mapping[str, Any]],
        output: str | Path,
        source_label: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Atomically publish validated canonical Nemesis records.

        This entry point lets the MCP control plane export an authoritative
        headless group without first manufacturing an intermediate transport
        JSONL. The public artifact remains identical to ``export_nemesis``.
        """

        if not str(source_label).strip():
            raise DatasetToolboxError("Nemesis source_label is required")
        destination = self._prepare_output(
            output,
            suffix=".jsonl",
            overwrite=overwrite,
        )
        if Path(source_label) == destination:
            raise DatasetToolboxError("input and output must differ")
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.candidate.",
            dir=destination.parent,
        ) as candidate_directory:
            candidate_root = Path(candidate_directory)
            candidate_root.chmod(0o700)
            candidate = candidate_root / "dataset.jsonl"
            write_nemesis_export(
                records,
                candidate,
                include_players=False,
            )
            validation = self.validate_artifact(candidate)
            if not validation.get("ok"):
                raise DatasetToolboxError(
                    "exported Nemesis dataset failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )
            os.replace(candidate, destination)
            destination.chmod(0o600)
        return {
            "ok": True,
            "kind": "nemesis",
            "input": str(source_label),
            "output": str(destination),
            "battle_count": validation["summary"]["battle_count"],
            "eligible_lite": validation["summary"]["eligible_lite"],
            "eligible_standard": validation["summary"]["eligible_standard"],
            "sha256": validation["sha256"],
            "privacy": "side_pseudonyms_p1_1_p2_2",
            "training_ready": validation["training_ready"],
            "training_ready_lite": validation["training_ready_lite"],
            "training_ready_standard": validation[
                "training_ready_standard"
            ],
        }

    def materialize_v5(
        self,
        *,
        input_path: str | Path,
        output_dir: str | Path,
        group_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        source = self.resolve(input_path, must_exist=True, expect="file")
        destination = self._prepare_output(
            output_dir,
            overwrite=overwrite,
        )
        if destination.exists() and not destination.is_dir():
            raise DatasetToolboxError(
                "materialized V5 output must be a directory"
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.candidate.",
            dir=destination.parent,
        ) as candidate_directory:
            candidate_root = Path(candidate_directory)
            candidate_root.chmod(0o700)
            candidate = candidate_root / "materialized"
            manifest = materialize_export(
                source,
                candidate,
                group_id=group_id,
                overwrite=False,
            )
            self._attach_current_catalog(candidate)
            manifest = json.loads(
                (candidate / "manifest.json").read_text(encoding="utf-8")
            )
            validation = self.validate_artifact(candidate)
            if not validation.get("ok"):
                raise DatasetToolboxError(
                    "materialized V5 dataset failed validation: "
                    + "; ".join(validation.get("issues") or [])
                )

            backup: Path | None = None
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.backup.",
                        dir=destination.parent,
                    )
                )
                backup.rmdir()
                os.replace(destination, backup)
            try:
                os.replace(candidate, destination)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            destination.chmod(0o700)
        return {
            "ok": True,
            "kind": "v5_materialized",
            "input": str(source),
            "output": str(destination),
            "group_id": manifest["group_id"],
            "battle_count": len(manifest["battle_ids"]),
            "privacy": (manifest.get("spec") or {}).get("privacy"),
            "training_ready": validation["training_ready"],
            "training_ready_scope": validation.get(
                "training_ready_scope"
            ),
            "v5_policy_training_ready": validation.get(
                "v5_policy_training_ready"
            ),
            "metronome_training_ready": validation.get(
                "metronome_training_ready"
            ),
            "timestamp_training_ready": validation.get(
                "timestamp_training_ready"
            ),
        }


__all__ = [
    "DatasetToolbox",
    "DatasetToolboxError",
]
