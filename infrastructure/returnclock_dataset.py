"""Build privacy-preserving ReturnClock training examples.

The online collectors intentionally persist raw, append-only facts.  This
module turns those facts into leakage-safe prediction cutoffs without making a
causal claim about notification timing:

* one example is anchored at the end of a meaningful analytics-v2 session;
* the target is the first following meaningful session inside the horizon;
* incomplete horizons are explicitly right-censored;
* notification decisions/deliveries between cutoff and target are retained as
  treatment metadata instead of being mislabeled as organic returns;
* raw user ids never leave the materializer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_FORMAT = "extraarena_returnclock_dataset_v1"
DATASET_VERSION = 1
DEFAULT_HORIZON_HOURS = 7 * 24
DEFAULT_MIN_ANALYTICS_VERSION = 2
MEANINGFUL_DURATION_SECONDS = 120
MEANINGFUL_SCREEN_COUNT = 2
STALE_OPEN_SESSION_AFTER = timedelta(minutes=30)
SESSIONIZATION_GAP = timedelta(minutes=30)
FEATURE_COLUMNS = (
    "timezone",
    "timezone_known",
    "local_weekday",
    "local_hour",
    "local_hour_sin",
    "local_hour_cos",
    "sessions_1d",
    "sessions_7d",
    "sessions_28d",
    "hours_since_previous_session",
    "median_gap_hours_28d",
    "gap_iqr_hours_28d",
    "recent_local_start_hours",
    "last_session_duration_seconds",
    "last_session_source",
    "last_session_screen_count",
    "last_session_battles",
    "last_session_cases",
    "last_session_entrypoint",
    "last_session_end_inferred",
    "notifications_24h",
    "notifications_7d",
)
_DELIVERY_EXPOSURE_EVENTS = frozenset(
    {
        "sent",
        "delivered",
        "shown",
    }
)
_PROVIDER_ACCEPTED_EVENTS = frozenset({"provider_accepted"})
_DELIVERY_OPEN_EVENTS = frozenset({"opened", "deeplink_opened"})
_HISTORICAL_NOTIFICATION_EVENTS = (
    _DELIVERY_EXPOSURE_EVENTS
    | _PROVIDER_ACCEPTED_EVENTS
    | _DELIVERY_OPEN_EVENTS
)


class ReturnClockDatasetError(ValueError):
    """Raw telemetry is malformed or cannot be exported safely."""


@dataclass(frozen=True)
class ReturnClockDataset:
    """A complete bounded export: one header followed by training examples."""

    header: dict[str, Any]
    examples: list[dict[str, Any]]

    def records(self) -> Iterable[dict[str, Any]]:
        yield self.header
        yield from self.examples


def _as_mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise ReturnClockDatasetError("telemetry row must be mapping-like") from exc


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReturnClockDatasetError(f"{field} must be ISO-8601") from exc
    else:
        raise ReturnClockDatasetError(f"{field} is required")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReturnClockDatasetError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _optional_timestamp(value: Any, *, field: str) -> datetime | None:
    if value is None or value == "":
        return None
    return _parse_timestamp(value, field=field)


def _timestamp_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_screens(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and item:
            normalized.append({"screen": item})
        elif isinstance(item, Mapping):
            screen = str(item.get("screen") or "").strip()
            if screen:
                normalized.append(
                    {
                        "screen": screen,
                        **({"ts": item.get("ts")} if item.get("ts") is not None else {}),
                    }
                )
    return normalized


def _normalize_battle_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value[:1000]:
        if not isinstance(item, str):
            continue
        battle_id = item.strip()
        if not battle_id or len(battle_id) > 128 or battle_id in seen:
            continue
        seen.add(battle_id)
        normalized.append(battle_id)
    return normalized


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _session_attribution_verified(row: Mapping[str, Any]) -> bool:
    return (
        row.get("_attribution_verified") is True
        or _metadata_mapping(row.get("metadata")).get(
            "returnclock_attribution_verified"
        )
        is True
    )


def _session_has_notification_attribution(
    row: Mapping[str, Any],
) -> bool:
    if not _session_attribution_verified(row):
        return False
    return bool(
        row.get("returnclock_decision_id") not in (None, "")
        or row.get("returnclock_delivery_id") not in (None, "")
        or str(row.get("entrypoint") or "").strip().lower()
        == "notification"
    )


def _meaningful_session(row: Mapping[str, Any]) -> bool:
    duration = max(0, _safe_int(row.get("duration_seconds")))
    screens = _normalize_screens(row.get("screens_visited"))
    return bool(
        (
            duration >= MEANINGFUL_DURATION_SECONDS
            and len(screens) >= MEANINGFUL_SCREEN_COUNT
        )
        or _safe_int(row.get("battles_played")) > 0
        or _safe_int(row.get("cases_opened")) > 0
    )


def _analytics_version(row: Mapping[str, Any]) -> int:
    value = row.get("analytics_version")
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return _safe_int(digits, 1)
    return _safe_int(value, 1)


def _resolve_local_timezone(row: Mapping[str, Any]):
    timezone_name = str(row.get("timezone") or row.get("timezone_name") or "").strip()
    if timezone_name:
        try:
            return ZoneInfo(timezone_name), timezone_name, True
        except (ZoneInfoNotFoundError, ValueError):
            pass
    offset = row.get("utc_offset_minutes")
    try:
        offset_minutes = int(offset)
    except (TypeError, ValueError):
        return timezone.utc, "UTC", False
    if not (-14 * 60 <= offset_minutes <= 14 * 60):
        return timezone.utc, "UTC", False
    sign = "+" if offset_minutes >= 0 else "-"
    absolute = abs(offset_minutes)
    label = f"UTC{sign}{absolute // 60:02d}:{absolute % 60:02d}"
    return timezone(timedelta(minutes=offset_minutes)), label, True


def _user_hash(user_id: Any, privacy_salt: str | bytes) -> str:
    salt = privacy_salt.encode("utf-8") if isinstance(privacy_salt, str) else privacy_salt
    if len(salt) < 32:
        raise ReturnClockDatasetError(
            "privacy_salt must contain at least 32 bytes of secret material"
        )
    digest = hmac.new(salt, str(user_id).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:32]


def _pseudonymization_key_id(
    privacy_salt: str | bytes,
    explicit_key_id: str | None,
) -> str:
    """Return a non-secret rotation ID for grouped split compatibility."""
    salt = privacy_salt.encode("utf-8") if isinstance(privacy_salt, str) else privacy_salt
    if len(salt) < 32:
        raise ReturnClockDatasetError(
            "privacy_salt must contain at least 32 bytes of secret material"
        )
    if explicit_key_id is None:
        return f"hmac-sha256:{hashlib.sha256(salt).hexdigest()[:16]}"
    key_id = str(explicit_key_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}", key_id):
        raise ReturnClockDatasetError("privacy_key_id is invalid")
    return key_id


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _event_time(row: Mapping[str, Any]) -> datetime | None:
    for key in ("occurred_at", "event_at", "sent_at", "created_at"):
        value = row.get(key)
        if value is not None:
            return _optional_timestamp(value, field=key)
    return None


def _decision_time(row: Mapping[str, Any]) -> datetime | None:
    for key in ("decided_at", "created_at", "eligible_at", "eligibility_at"):
        value = row.get(key)
        if value is not None:
            return _optional_timestamp(value, field=key)
    return None


def _event_kind(row: Mapping[str, Any]) -> str:
    return str(row.get("event_type") or row.get("status") or "").strip().lower()


def _notification_key(row: Mapping[str, Any]) -> str:
    for key in ("delivery_id", "notification_id", "outbox_id", "decision_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    timestamp = _event_time(row)
    return "|".join(
        [
            str(row.get("channel") or ""),
            str(row.get("event_type") or row.get("status") or ""),
            _timestamp_iso(timestamp) if timestamp else "",
        ]
    )


def _count_notification_history(
    events: Sequence[Mapping[str, Any]],
    *,
    attributed_sessions: Sequence[Mapping[str, Any]] = (),
    start: datetime,
    end: datetime,
) -> int:
    exposed: set[str] = set()
    opened_session_ids: set[str] = set()
    for event in events:
        occurred_at = _event_time(event)
        if occurred_at is None or not (start < occurred_at <= end):
            continue
        if _event_kind(event) in _HISTORICAL_NOTIFICATION_EVENTS:
            exposed.add(_notification_key(event))
            if (
                _event_kind(event) in _DELIVERY_OPEN_EVENTS
                and event.get("session_id") not in (None, "")
            ):
                opened_session_ids.add(str(event["session_id"]))
    for session in attributed_sessions:
        started_at = session.get("_started_at")
        session_id = str(session.get("session_id") or "")
        if (
            _session_has_notification_attribution(session)
            and isinstance(started_at, datetime)
            and start < started_at <= end
            and session_id
            and session_id not in opened_session_ids
        ):
            exposed.add(f"attributed_session:{session_id}")
    return len(exposed)


def _interval_treatment(
    decisions: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    start: datetime,
    end: datetime,
    attributed_decision_ids: Sequence[str],
    attributed_session_ids: Sequence[str],
    notification_attributed: bool,
) -> dict[str, Any]:
    attributed_decision_id_set = {
        str(value) for value in attributed_decision_ids if str(value)
    }
    attributed_session_id_set = {
        str(value) for value in attributed_session_ids if str(value)
    }
    interval_events: list[Mapping[str, Any]] = []
    for event in events:
        occurred_at = _event_time(event)
        session_linked_open = bool(
            attributed_session_id_set
            and str(event.get("session_id") or "")
            in attributed_session_id_set
            and _event_kind(event) in _DELIVERY_OPEN_EVENTS
        )
        if (
            occurred_at is not None
            and start < occurred_at <= end
        ) or session_linked_open:
            interval_events.append(event)

    linked_decision_ids = {
        str(event.get("decision_id"))
        for event in interval_events
        if event.get("decision_id") not in (None, "")
    }
    linked_decision_ids.update(attributed_decision_id_set)
    interval_decisions: list[Mapping[str, Any]] = []
    for decision in decisions:
        decision_id = (
            str(decision.get("decision_id"))
            if decision.get("decision_id") not in (None, "")
            else None
        )
        decided_at = _decision_time(decision)
        if (
            decided_at is not None
            and start < decided_at <= end
        ) or (decision_id is not None and decision_id in linked_decision_ids):
            interval_decisions.append(decision)

    sent_keys = {
        _notification_key(event)
        for event in interval_events
        if _event_kind(event) in _DELIVERY_EXPOSURE_EVENTS
    }
    provider_accepted_keys = {
        _notification_key(event)
        for event in interval_events
        if _event_kind(event) in _PROVIDER_ACCEPTED_EVENTS
    }
    opened_keys = {
        _notification_key(event)
        for event in interval_events
        if _event_kind(event) in _DELIVERY_OPEN_EVENTS
    }
    opened_session_ids = {
        str(event.get("session_id"))
        for event in interval_events
        if (
            _event_kind(event) in _DELIVERY_OPEN_EVENTS
            and event.get("session_id") not in (None, "")
        )
    }
    # A server-verified attributed session is itself an open signal. Add one
    # synthetic key per session only when no corresponding open callback was
    # recorded, so callbacks and sessions cannot double-count the same return.
    for session_id in attributed_session_id_set - opened_session_ids:
        opened_keys.add(f"attributed_session:{session_id}")
    channels = sorted(
        {
            str(event.get("channel"))
            for event in interval_events
            if event.get("channel") not in (None, "")
        }
    )
    variants = sorted(
        {
            str(
                decision.get("treatment_arm")
                or decision.get("experiment_variant")
            )
            for decision in interval_decisions
            if (
                decision.get("treatment_arm")
                or decision.get("experiment_variant")
            )
            not in (None, "")
        }
    )
    assignments = [
        {
            "experiment_id": (
                str(decision.get("experiment_id"))
                if decision.get("experiment_id") not in (None, "")
                else None
            ),
            "treatment_arm": str(
                decision.get("treatment_arm")
                or decision.get("experiment_variant")
                or "observational"
            ),
            "assignment_probability": (
                float(decision.get("assignment_probability"))
                if decision.get("assignment_probability") is not None
                else None
            ),
            "decision": str(decision.get("decision") or "unknown"),
            "decision_source": str(
                decision.get("decision_source") or "unknown"
            ),
            "policy_version": str(
                decision.get("policy_version") or "unknown"
            ),
            "model_version": (
                str(decision.get("model_version"))
                if decision.get("model_version") not in (None, "")
                else None
            ),
        }
        for decision in interval_decisions
    ]
    treatment_assigned = any(
        str(decision.get("decision") or "").strip().lower()
        not in {"", "skip", "no_send", "control"}
        for decision in interval_decisions
    )
    return {
        "notification_decision_count": len(interval_decisions),
        "provider_accepted_count": len(provider_accepted_keys),
        "notification_sent_count": len(sent_keys),
        "notification_opened_count": len(opened_keys),
        "notification_channels": channels,
        "treatment_arms": variants,
        "assignments": assignments,
        "notification_attributed": bool(notification_attributed),
        "treatment_assigned": treatment_assigned,
        "organic_candidate": not (
            provider_accepted_keys
            or sent_keys
            or opened_keys
            or notification_attributed
            or treatment_assigned
        ),
    }


def _normalize_rows(
    rows: Iterable[Mapping[str, Any] | Any],
) -> list[Mapping[str, Any]]:
    return [_as_mapping(row) for row in rows]


def _coalesce_user_sessions(
    sessions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge reloads, BFCache restores and overlapping tabs into return sessions."""
    ordered = sorted(
        sessions,
        key=lambda row: (row["_started_at"], row["_ended_at"]),
    )
    merged: list[dict[str, Any]] = []
    for candidate in ordered:
        if (
            not merged
            or candidate["_started_at"]
            > merged[-1]["_ended_at"] + SESSIONIZATION_GAP
        ):
            merged.append(dict(candidate))
            continue

        current = merged[-1]
        current["_ended_at"] = max(current["_ended_at"], candidate["_ended_at"])
        current["_end_inferred"] = bool(
            current["_end_inferred"] or candidate["_end_inferred"]
        )
        current["_unfinished"] = bool(
            current.get("_unfinished")
            or candidate.get("_unfinished")
        )
        current["_attribution_verified"] = bool(
            current.get("_attribution_verified")
            or candidate.get("_attribution_verified")
        )
        current["_screens"] = (current["_screens"] + candidate["_screens"])[-200:]
        current["screens_visited"] = current["_screens"]
        wall_seconds = max(
            0,
            int(
                (current["_ended_at"] - current["_started_at"]).total_seconds()
            ),
        )
        active_seconds = max(0, _safe_int(current.get("duration_seconds"))) + max(
            0,
            _safe_int(candidate.get("duration_seconds")),
        )
        current["duration_seconds"] = min(wall_seconds, active_seconds)
        merged_battle_ids = list(current.get("_battle_ids") or [])
        seen_battle_ids = set(merged_battle_ids)
        for battle_id in candidate.get("_battle_ids") or []:
            if battle_id not in seen_battle_ids:
                seen_battle_ids.add(battle_id)
                merged_battle_ids.append(battle_id)
        current["_battle_ids"] = merged_battle_ids
        current["battle_ids"] = merged_battle_ids
        current["_legacy_battles_played"] = max(
            0,
            _safe_int(current.get("_legacy_battles_played")),
        ) + max(
            0,
            _safe_int(candidate.get("_legacy_battles_played")),
        )
        current["battles_played"] = (
            len(merged_battle_ids)
            + current["_legacy_battles_played"]
        )
        current["cases_opened"] = max(
            0, _safe_int(current.get("cases_opened"))
        ) + max(0, _safe_int(candidate.get("cases_opened")))
        current["analytics_version"] = max(
            _analytics_version(current),
            _analytics_version(candidate),
        )
        if (
            str(candidate.get("entrypoint") or "").strip().lower()
            == "notification"
        ):
            current["entrypoint"] = "notification"
        for key in ("returnclock_decision_id", "returnclock_delivery_id"):
            if current.get(key) in (None, "") and candidate.get(key) not in (
                None,
                "",
            ):
                current[key] = candidate.get(key)
        for key in ("timezone", "utc_offset_minutes", "source"):
            if candidate.get(key) not in (None, ""):
                current[key] = candidate.get(key)
    return merged


def materialize_returnclock_dataset(
    *,
    sessions: Iterable[Mapping[str, Any] | Any],
    decisions: Iterable[Mapping[str, Any] | Any] = (),
    delivery_events: Iterable[Mapping[str, Any] | Any] = (),
    dataset_end: datetime,
    ingested_before: datetime | None = None,
    privacy_salt: str | bytes,
    privacy_key_id: str | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    min_analytics_version: int = DEFAULT_MIN_ANALYTICS_VERSION,
    cutoff_start: datetime | None = None,
) -> ReturnClockDataset:
    """Materialize a bounded ReturnClock-v1 JSONL-ready dataset.

    ``dataset_end`` is deliberately required: without it the final rows cannot
    distinguish a true no-return horizon from a dataset that simply stopped.
    """

    if horizon_hours < 1 or horizon_hours > 31 * 24:
        raise ReturnClockDatasetError("horizon_hours must be in [1, 744]")
    dataset_end_utc = _parse_timestamp(dataset_end, field="dataset_end")
    ingested_before_utc = (
        _parse_timestamp(ingested_before, field="ingested_before")
        if ingested_before is not None
        else dataset_end_utc
    )
    if ingested_before_utc < dataset_end_utc:
        raise ReturnClockDatasetError(
            "ingested_before must be >= dataset_end"
        )
    pseudonymization_key_id = _pseudonymization_key_id(
        privacy_salt,
        privacy_key_id,
    )
    cutoff_start_utc = (
        _parse_timestamp(cutoff_start, field="cutoff_start")
        if cutoff_start is not None
        else None
    )

    normalized_sessions = _normalize_rows(sessions)
    normalized_decisions = _normalize_rows(decisions)
    normalized_events = _normalize_rows(delivery_events)

    sessions_by_user: dict[Any, list[dict[str, Any]]] = {}
    excluded_legacy = 0
    excluded_unfinished = 0
    excluded_nonmeaningful = 0
    inferred_stale_ends = 0
    for raw in normalized_sessions:
        user_id = raw.get("user_id")
        if user_id is None:
            raise ReturnClockDatasetError("session.user_id is required")
        if _analytics_version(raw) < min_analytics_version:
            excluded_legacy += 1
            continue
        started_at = _optional_timestamp(raw.get("started_at"), field="started_at")
        if started_at is not None and started_at > dataset_end_utc:
            excluded_unfinished += 1
            continue
        ended_at = _optional_timestamp(raw.get("ended_at"), field="ended_at")
        end_inferred = False
        unfinished = False
        normalized_raw = dict(raw)
        if started_at is not None and ended_at is None:
            last_heartbeat_at = _optional_timestamp(
                raw.get("last_heartbeat_at"),
                field="last_heartbeat_at",
            )
            if (
                last_heartbeat_at is not None
                and started_at <= last_heartbeat_at
                and last_heartbeat_at
                <= dataset_end_utc - STALE_OPEN_SESSION_AFTER
            ):
                ended_at = last_heartbeat_at
                end_inferred = True
                inferred_stale_ends += 1
                normalized_raw["duration_seconds"] = max(
                    _safe_int(raw.get("duration_seconds")),
                    int((ended_at - started_at).total_seconds()),
                )
            else:
                unfinished = True
                excluded_unfinished += 1
                marker_end = last_heartbeat_at or dataset_end_utc
                ended_at = max(
                    started_at,
                    min(marker_end, dataset_end_utc),
                )
                normalized_raw["duration_seconds"] = max(
                    _safe_int(raw.get("duration_seconds")),
                    (
                        int(
                            (
                                last_heartbeat_at - started_at
                            ).total_seconds()
                        )
                        if last_heartbeat_at is not None
                        else 0
                    ),
                )
        if started_at is None or ended_at is None or ended_at < started_at:
            excluded_unfinished += 1
            continue
        if ended_at > dataset_end_utc:
            excluded_unfinished += int(not unfinished)
            unfinished = True
            ended_at = dataset_end_utc
        sessions_by_user.setdefault(user_id, []).append(
            {
                **normalized_raw,
                "_started_at": started_at,
                "_ended_at": ended_at,
                "_end_inferred": end_inferred,
                "_unfinished": unfinished,
                "_attribution_verified": (
                    _session_attribution_verified(normalized_raw)
                ),
                "_screens": _normalize_screens(normalized_raw.get("screens_visited")),
                "_battle_ids": _normalize_battle_ids(
                    normalized_raw.get("battle_ids")
                ),
                "_legacy_battles_played": (
                    max(
                        0,
                        _safe_int(normalized_raw.get("battles_played")),
                    )
                    if not _normalize_battle_ids(
                        normalized_raw.get("battle_ids")
                    )
                    else 0
                ),
            }
        )
        normalized_session = sessions_by_user[user_id][-1]
        normalized_session["battle_ids"] = normalized_session[
            "_battle_ids"
        ]
        normalized_session["battles_played"] = (
            len(normalized_session["_battle_ids"])
            + normalized_session["_legacy_battles_played"]
        )

    decisions_by_user: dict[Any, list[Mapping[str, Any]]] = {}
    for decision in normalized_decisions:
        if decision.get("user_id") is not None:
            decisions_by_user.setdefault(decision.get("user_id"), []).append(decision)
    events_by_user: dict[Any, list[Mapping[str, Any]]] = {}
    for event in normalized_events:
        if event.get("user_id") is not None:
            events_by_user.setdefault(event.get("user_id"), []).append(event)

    examples: list[dict[str, Any]] = []
    horizon = timedelta(hours=horizon_hours)
    for user_id, user_sessions in sessions_by_user.items():
        coalesced_sessions = _coalesce_user_sessions(user_sessions)
        meaningful_sessions = [
            session
            for session in coalesced_sessions
            if _meaningful_session(session)
        ]
        finished_sessions = [
            session
            for session in meaningful_sessions
            if not session.get("_unfinished")
        ]
        user_sessions = finished_sessions
        excluded_nonmeaningful += (
            len(coalesced_sessions) - len(meaningful_sessions)
        )
        user_decisions = decisions_by_user.get(user_id, [])
        user_events = events_by_user.get(user_id, [])

        for index, current in enumerate(user_sessions):
            cutoff_at = current["_ended_at"]
            if cutoff_start_utc is not None and cutoff_at < cutoff_start_utc:
                continue
            horizon_end = cutoff_at + horizon
            next_session = next(
                (
                    candidate
                    for candidate in meaningful_sessions
                    if candidate["_started_at"] > cutoff_at
                ),
                None,
            )
            observed_next = (
                next_session
                if next_session is not None and next_session["_started_at"] <= horizon_end
                else None
            )
            if observed_next is not None:
                interval_end = observed_next["_started_at"]
                target_observed = True
                right_censored = False
                time_to_return_minutes: float | None = round(
                    (interval_end - cutoff_at).total_seconds() / 60.0,
                    3,
                )
                observation_window_minutes = time_to_return_minutes
            else:
                interval_end = min(horizon_end, dataset_end_utc)
                target_observed = False
                right_censored = dataset_end_utc < horizon_end
                time_to_return_minutes = None
                observation_window_minutes = round(
                    max(0.0, (interval_end - cutoff_at).total_seconds() / 60.0),
                    3,
                )

            history = [
                row
                for row in user_sessions[: index + 1]
                if row["_started_at"] <= cutoff_at
            ]
            starts = [row["_started_at"] for row in history]
            starts_28d = [
                started_at
                for started_at in starts
                if started_at >= cutoff_at - timedelta(days=28)
            ]
            start_gaps_hours = [
                (later - earlier).total_seconds() / 3600.0
                for earlier, later in zip(starts_28d, starts_28d[1:])
                if later > earlier
            ]
            previous = history[-2] if len(history) >= 2 else None
            hours_since_previous = (
                max(
                    0.0,
                    (
                        current["_started_at"] - previous["_ended_at"]
                    ).total_seconds()
                    / 3600.0,
                )
                if previous is not None
                else None
            )
            local_tz, timezone_label, timezone_known = _resolve_local_timezone(current)
            local_cutoff = cutoff_at.astimezone(local_tz)
            local_start_hours = [
                row["_started_at"].astimezone(local_tz).hour for row in history[-8:]
            ]
            attributed_sessions = [
                session
                for session in coalesced_sessions
                if (
                    cutoff_at < session["_started_at"] <= interval_end
                    and _session_has_notification_attribution(session)
                )
            ]
            attributed_decision_ids = [
                str(session.get("returnclock_decision_id"))
                for session in attributed_sessions
                if session.get("returnclock_decision_id")
                not in (None, "")
            ]
            attributed_session_ids = [
                str(session.get("session_id"))
                for session in attributed_sessions
                if session.get("session_id") not in (None, "")
            ]
            notification_attributed = bool(attributed_sessions)
            treatment = _interval_treatment(
                user_decisions,
                user_events,
                start=cutoff_at,
                end=interval_end,
                attributed_decision_ids=attributed_decision_ids,
                attributed_session_ids=attributed_session_ids,
                notification_attributed=notification_attributed,
            )

            features = {
                "timezone": timezone_label,
                "timezone_known": timezone_known,
                "local_weekday": local_cutoff.weekday(),
                "local_hour": local_cutoff.hour,
                "local_hour_sin": round(
                    math.sin(2.0 * math.pi * local_cutoff.hour / 24.0), 6
                ),
                "local_hour_cos": round(
                    math.cos(2.0 * math.pi * local_cutoff.hour / 24.0), 6
                ),
                "sessions_1d": sum(
                    start >= cutoff_at - timedelta(days=1) for start in starts
                ),
                "sessions_7d": sum(
                    start >= cutoff_at - timedelta(days=7) for start in starts
                ),
                "sessions_28d": sum(
                    start >= cutoff_at - timedelta(days=28) for start in starts
                ),
                "hours_since_previous_session": (
                    round(hours_since_previous, 3)
                    if hours_since_previous is not None
                    else None
                ),
                "median_gap_hours_28d": (
                    round(float(statistics.median(start_gaps_hours)), 3)
                    if start_gaps_hours
                    else None
                ),
                "gap_iqr_hours_28d": (
                    round(
                        float(
                            (_percentile(start_gaps_hours, 0.75) or 0.0)
                            - (_percentile(start_gaps_hours, 0.25) or 0.0)
                        ),
                        3,
                    )
                    if start_gaps_hours
                    else None
                ),
                "recent_local_start_hours": local_start_hours,
                "last_session_duration_seconds": max(
                    0, _safe_int(current.get("duration_seconds"))
                ),
                "last_session_source": str(current.get("source") or "unknown"),
                "last_session_screen_count": len(current["_screens"]),
                "last_session_battles": max(
                    0, _safe_int(current.get("battles_played"))
                ),
                "last_session_cases": max(
                    0, _safe_int(current.get("cases_opened"))
                ),
                "last_session_entrypoint": str(
                    current.get("entrypoint") or "direct"
                ),
                "last_session_end_inferred": bool(current["_end_inferred"]),
                "notifications_24h": _count_notification_history(
                    user_events,
                    attributed_sessions=coalesced_sessions,
                    start=cutoff_at - timedelta(hours=24),
                    end=cutoff_at,
                ),
                "notifications_7d": _count_notification_history(
                    user_events,
                    attributed_sessions=coalesced_sessions,
                    start=cutoff_at - timedelta(days=7),
                    end=cutoff_at,
                ),
            }
            label = {
                "target_observed": target_observed,
                "right_censored": right_censored,
                "time_to_return_minutes": time_to_return_minutes,
                "observation_window_minutes": observation_window_minutes,
            }
            examples.append(
                {
                    "record_type": "example",
                    "dataset_format": DATASET_FORMAT,
                    "dataset_version": DATASET_VERSION,
                    "user_id_hash": _user_hash(user_id, privacy_salt),
                    # Kept only for chronological split/audit; it is not listed
                    # in feature_columns and must never enter the estimator.
                    "prediction_cutoff_at": _timestamp_iso(cutoff_at),
                    "features": features,
                    "label": label,
                    "post_cutoff": treatment,
                }
            )

    examples.sort(
        key=lambda row: (row["prediction_cutoff_at"], row["user_id_hash"])
    )
    summary = {
        "example_count": len(examples),
        "observed_returns": sum(
            row["label"]["target_observed"] for row in examples
        ),
        "complete_no_return_horizons": sum(
            not row["label"]["target_observed"]
            and not row["label"]["right_censored"]
            for row in examples
        ),
        "right_censored": sum(
            row["label"]["right_censored"] for row in examples
        ),
        "treated_intervals": sum(
            not row["post_cutoff"]["organic_candidate"] for row in examples
        ),
        "organic_candidates": sum(
            row["post_cutoff"]["organic_candidate"] for row in examples
        ),
        "excluded_legacy_sessions": excluded_legacy,
        "excluded_unfinished_sessions": excluded_unfinished,
        "excluded_nonmeaningful_sessions": excluded_nonmeaningful,
        "inferred_stale_session_ends": inferred_stale_ends,
    }
    header = {
        "record_type": "header",
        "format": DATASET_FORMAT,
        "format_version": DATASET_VERSION,
        "generated_at": _timestamp_iso(datetime.now(timezone.utc)),
        "dataset_end": _timestamp_iso(dataset_end_utc),
        "ingested_before": _timestamp_iso(ingested_before_utc),
        "cutoff_start": (
            _timestamp_iso(cutoff_start_utc) if cutoff_start_utc is not None else None
        ),
        "horizon_hours": horizon_hours,
        "min_analytics_version": min_analytics_version,
        "sessionization_gap_minutes": int(
            SESSIONIZATION_GAP.total_seconds() / 60
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_columns": [
            "target_observed",
            "right_censored",
            "time_to_return_minutes",
            "observation_window_minutes",
        ],
        "post_cutoff_namespace": "excluded_from_model_features",
        "user_id_scheme": "hmac-sha256-truncated-128",
        "pseudonymization_key_id": pseudonymization_key_id,
        "privacy_note": (
            "user_id_hash is pseudonymous and intended only for grouped "
            "splits inside access-controlled training storage"
        ),
        "meaningful_session": {
            "duration_seconds_gte": MEANINGFUL_DURATION_SECONDS,
            "screen_count_gte": MEANINGFUL_SCREEN_COUNT,
            "or_battle_or_case": True,
        },
        "summary": summary,
    }
    return ReturnClockDataset(header=header, examples=examples)


def write_returnclock_jsonl(
    dataset: ReturnClockDataset,
    output_path: str | Path,
) -> Path:
    """Atomically publish a complete ReturnClock dataset."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            os.fchmod(handle.fileno(), 0o600)
            for record in dataset.records():
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    destination.chmod(0o600)
    return destination
