"""Unified one-record-per-battle contract for Nemesis training.

Nemesis features are frozen before the battle starts. Terminal information is
written only under ``label`` so consumers cannot accidentally train on
post-outcome fields disguised as input features.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional
from uuid import UUID, uuid4, uuid5


NEMESIS_SCHEMA = "extraarena_nemesis_battle_v1"
NEMESIS_EXPORT_FORMAT = "extraarena_nemesis_dataset_export_v1"
NEMESIS_VISIBILITY = "private_server_only"
NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME = "random_per_export_record_ids_v1"
NEMESIS_RAW_RECORD_ID_SCHEME = "raw_record_ids"
NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME = (
    "random_per_export_player_groups_v1"
)
NEMESIS_RAW_PLAYER_GROUP_SCHEME = "raw_player_ids"
TERMINAL_STATUSES = frozenset({"p1_win", "p2_win", "draw", "stalemate"})
ACTOR_TYPES = frozenset({"human", "bot", "llm", "rl"})
DOMAINS = frozenset({"human-human", "human-bot", "model-model"})
SEATS = ("p1", "p2")
_PSEUDONYMIZED_RECORD_ID_RE = re.compile(r"^record_[0-9a-f]{32}$")
_PSEUDONYMIZED_PLAYER_GROUP_RE = re.compile(r"^player_[0-9a-f]{32}$")

MODEL_PROVENANCE_FIELDS = (
    "model_id",
    "model_family",
    "model_version",
    "checkpoint_id",
    "weights_hash",
    "adapter_kind",
)
HISTORY_SUMMARY_FIELDS = (
    "history_total",
    "total",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "trophy_delta",
    "avg_turns",
    "avg_duration_seconds",
    "favorite_mode",
    "current_streak_result",
    "current_streak_count",
    "current_win_streak",
    "max_win_streak",
)


class NemesisContractError(ValueError):
    """A record would violate the leakage, provenance or privacy contract."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NemesisContractError(f"{field} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise NemesisContractError(f"{field} must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NemesisContractError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _required_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise NemesisContractError(f"{field} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as exc:
            raise NemesisContractError(f"{field} must be an integer") from exc
        if str(result) != value.strip():
            raise NemesisContractError(f"{field} must be an integer")
    else:
        raise NemesisContractError(f"{field} must be an integer")
    if result < minimum:
        raise NemesisContractError(f"{field} must be >= {minimum}")
    return result


def _optional_finite(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NemesisContractError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NemesisContractError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise NemesisContractError(f"{field} must be finite")
    return result


def _domain_for(actor_types: tuple[str, str]) -> str:
    human_count = sum(actor == "human" for actor in actor_types)
    if human_count == 2:
        return "human-human"
    if human_count == 1:
        return "human-bot"
    return "model-model"


def _deck_pair_split_fingerprint(
    *,
    seats: Mapping[str, Mapping[str, Any]],
    ruleset: str,
    catalog_hash: str | None,
) -> str:
    """Group repeated/swapped exact matchups into one leakage-safe split."""

    deck_signatures = []
    for seat in SEATS:
        deck = seats[seat]["initial_deck"]
        deck_signatures.append(
            sorted(
                [
                    [int(card["card_id"]), int(card["level"])]
                    for card in deck
                ]
            )
        )
    payload = {
        "ruleset": str(ruleset),
        "catalog_hash": catalog_hash,
        "unordered_decks": sorted(deck_signatures),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_model_provenance(value: Any) -> dict[str, str | None]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, str | None] = {}
    for field in MODEL_PROVENANCE_FIELDS:
        raw = source.get(field)
        result[field] = str(raw) if raw is not None and str(raw).strip() else None
    return result


def _normalize_aux_model_provenance(value: Any) -> dict[str, dict[str, str | None]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, str | None]] = {}
    for raw_name, raw_provenance in sorted(value.items(), key=lambda item: str(item[0])):
        name = str(raw_name).strip()
        if not name or not isinstance(raw_provenance, Mapping):
            continue
        result[name] = _normalize_model_provenance(raw_provenance)
    return result


def _normalize_deck(value: Any, *, field: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise NemesisContractError(f"{field} must be a non-empty list")
    rows: list[dict[str, int]] = []
    seen_slots: set[int] = set()
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise NemesisContractError(f"{field}[{position}] must be an object")
        slot = _required_int(raw.get("slot", position), field=f"{field}[{position}].slot")
        if slot in seen_slots:
            raise NemesisContractError(f"{field} has duplicate slot {slot}")
        seen_slots.add(slot)
        rows.append(
            {
                "slot": slot,
                "card_id": _required_int(
                    raw.get("card_id"),
                    field=f"{field}[{position}].card_id",
                    minimum=1,
                ),
                "level": _required_int(
                    raw.get("level"),
                    field=f"{field}[{position}].level",
                    minimum=1,
                ),
            }
        )
    rows.sort(key=lambda row: row["slot"])
    if [row["slot"] for row in rows] != list(range(len(rows))):
        raise NemesisContractError(f"{field} slots must be contiguous from zero")
    return rows


def _normalize_extended_seat(
    value: Any,
    *,
    seat: str,
    feature_cutoff: datetime,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise NemesisContractError(f"extended.{seat} must be an object or null")
    captured_at_raw = value.get("captured_at")
    captured_at = _parse_timestamp(
        captured_at_raw,
        field=f"extended.{seat}.captured_at",
    )
    if captured_at > feature_cutoff:
        raise NemesisContractError(
            f"extended.{seat}.captured_at is after feature_cutoff_at"
        )

    profile = value.get("profile")
    if not isinstance(profile, Mapping):
        raise NemesisContractError(f"extended.{seat}.profile must be an object")
    normalized_profile = {
        "wins": _required_int(profile.get("wins"), field=f"extended.{seat}.profile.wins"),
        "losses": _required_int(
            profile.get("losses"),
            field=f"extended.{seat}.profile.losses",
        ),
        "trophies": _required_int(
            profile.get("trophies"),
            field=f"extended.{seat}.profile.trophies",
        ),
    }

    summary = value.get("summary")
    if summary is not None and not isinstance(summary, Mapping):
        raise NemesisContractError(f"extended.{seat}.summary must be an object or null")
    normalized_summary: dict[str, Any] | None = None
    if isinstance(summary, Mapping):
        normalized_summary = {}
        for key in HISTORY_SUMMARY_FIELDS:
            raw = summary.get(key)
            if key in {
                "history_total",
                "total",
                "wins",
                "losses",
                "draws",
                "current_streak_count",
                "current_win_streak",
                "max_win_streak",
            }:
                normalized_summary[key] = (
                    _required_int(raw, field=f"extended.{seat}.summary.{key}")
                    if raw is not None
                    else None
                )
            elif key in {
                "win_rate",
                "trophy_delta",
                "avg_turns",
                "avg_duration_seconds",
            }:
                normalized_summary[key] = _optional_finite(
                    raw,
                    field=f"extended.{seat}.summary.{key}",
                )
            else:
                normalized_summary[key] = str(raw) if raw is not None else None
    recent = value.get("recent")
    if not isinstance(recent, list):
        raise NemesisContractError(f"extended.{seat}.recent must be a list")
    normalized_recent: list[dict[str, Any]] = []
    for index, item in enumerate(recent):
        if not isinstance(item, Mapping):
            raise NemesisContractError(f"extended.{seat}.recent[{index}] must be an object")
        completed_raw = item.get("completed_at")
        completed = _parse_timestamp(
            completed_raw,
            field=f"extended.{seat}.recent[{index}].completed_at",
        )
        if completed > captured_at or completed > feature_cutoff:
            raise NemesisContractError(
                f"extended.{seat}.recent[{index}] is not time-causal"
            )
        result = str(item.get("result") or "")
        opponent_type = str(item.get("opponent_actor_type") or "")
        if result not in {"win", "lose", "draw"}:
            raise NemesisContractError(f"extended.{seat}.recent[{index}].result is invalid")
        if opponent_type not in {"human", "bot", "unknown"}:
            raise NemesisContractError(
                f"extended.{seat}.recent[{index}].opponent_actor_type is invalid"
            )
        started_first = item.get("started_first")
        if started_first is not None and not isinstance(started_first, bool):
            raise NemesisContractError(
                f"extended.{seat}.recent[{index}].started_first is invalid"
            )
        normalized_recent.append(
            {
                "result": result,
                "opponent_actor_type": opponent_type,
                "game_mode": str(item.get("game_mode") or ""),
                "completed_at": str(completed_raw),
                "duration_seconds": _required_int(
                    item.get("duration_seconds"),
                    field=f"extended.{seat}.recent[{index}].duration_seconds",
                ),
                "turns_count": _required_int(
                    item.get("turns_count"),
                    field=f"extended.{seat}.recent[{index}].turns_count",
                ),
                "trophy_change": int(item.get("trophy_change") or 0),
                "started_first": started_first,
            }
        )
    return {
        "captured_at": str(captured_at_raw),
        "profile": normalized_profile,
        "summary": normalized_summary,
        "recent": normalized_recent,
    }


def _seat_payload(meta: Mapping[str, Any], seat: str) -> dict[str, Any]:
    actor_type = str(meta.get(f"{seat}_actor_type") or "").lower()
    if actor_type not in ACTOR_TYPES:
        raise NemesisContractError(f"{seat}_actor_type must be one of {sorted(ACTOR_TYPES)}")
    model_map = meta.get("model_provenance")
    model_source = model_map.get(seat) if isinstance(model_map, Mapping) else None
    aux_map = meta.get("aux_model_provenance")
    aux_source = aux_map.get(seat) if isinstance(aux_map, Mapping) else None
    return {
        "participant_id": _required_int(
            meta.get(f"{seat}_user_id"),
            field=f"{seat}_user_id",
            minimum=-2**63,
        ),
        "actor_type": actor_type,
        "model_provenance": _normalize_model_provenance(model_source),
        "aux_model_provenance": _normalize_aux_model_provenance(aux_source),
        "initial_deck": _normalize_deck(
            meta.get(f"{seat}_deck"),
            field=f"{seat}_deck",
        ),
    }


def validate_nemesis_record(
    record: Mapping[str, Any],
    *,
    require_terminal: bool = False,
) -> dict[str, Any]:
    """Validate and return a detached canonical record."""

    if not isinstance(record, Mapping):
        raise NemesisContractError("record must be an object")
    required_top_level = {
        "schema_version",
        "visibility",
        "battle_id",
        "match_id",
        "created_at",
        "features",
        "provenance",
        "quality",
        "label",
    }
    unknown_top_level = set(record) - required_top_level - {"privacy"}
    if unknown_top_level:
        raise NemesisContractError(
            f"unexpected top-level fields: {sorted(unknown_top_level)}"
        )
    missing_top_level = required_top_level - set(record)
    if missing_top_level:
        raise NemesisContractError(
            f"missing top-level fields: {sorted(missing_top_level)}"
        )
    if record.get("schema_version") != NEMESIS_SCHEMA:
        raise NemesisContractError(f"schema_version must be {NEMESIS_SCHEMA}")
    if record.get("visibility") != NEMESIS_VISIBILITY:
        raise NemesisContractError(f"visibility must be {NEMESIS_VISIBILITY}")
    if not str(record.get("battle_id") or "").strip():
        raise NemesisContractError("battle_id is required")
    if not str(record.get("match_id") or "").strip():
        raise NemesisContractError("match_id is required")
    _parse_timestamp(record.get("created_at"), field="created_at")
    features = record.get("features")
    if not isinstance(features, Mapping):
        raise NemesisContractError("features must be an object")
    if set(features) != {"base", "extended"}:
        raise NemesisContractError("features may contain only base and extended")
    base = features.get("base")
    if not isinstance(base, Mapping):
        raise NemesisContractError("features.base must be an object")
    expected_base_fields = {
        "domain",
        "game_mode",
        "ruleset",
        "catalog_hash",
        "catalog_available",
        "card_params_schema",
        "deck_params_schema",
        "starting_player",
        "feature_cutoff_at",
        "battle_started_at",
        "seats",
    }
    if set(base) != expected_base_fields:
        raise NemesisContractError("features.base fields do not match the contract")
    if base.get("domain") not in DOMAINS:
        raise NemesisContractError("features.base.domain is invalid")
    seats = base.get("seats")
    if not isinstance(seats, Mapping) or set(seats) != set(SEATS):
        raise NemesisContractError("features.base.seats must contain p1 and p2")
    if base.get("starting_player") not in SEATS:
        raise NemesisContractError("features.base.starting_player is invalid")
    for field in (
        "ruleset",
        "game_mode",
        "card_params_schema",
        "deck_params_schema",
    ):
        if not str(base.get(field) or "").strip():
            raise NemesisContractError(f"features.base.{field} is required")
    catalog_available = base.get("catalog_available")
    catalog_hash = base.get("catalog_hash")
    if not isinstance(catalog_available, bool):
        raise NemesisContractError("features.base.catalog_available must be bool")
    if catalog_available != bool(str(catalog_hash or "").strip()):
        raise NemesisContractError("catalog availability/hash mismatch")
    feature_cutoff = _parse_timestamp(
        base.get("feature_cutoff_at"),
        field="features.base.feature_cutoff_at",
    )
    battle_started = _parse_timestamp(
        base.get("battle_started_at"),
        field="features.base.battle_started_at",
    )
    if feature_cutoff > battle_started:
        raise NemesisContractError("feature_cutoff_at must not follow battle_started_at")
    actor_types: list[str] = []
    for seat in SEATS:
        seat_payload = seats[seat]
        if not isinstance(seat_payload, Mapping):
            raise NemesisContractError(f"features.base.seats.{seat} must be an object")
        if set(seat_payload) != {
            "participant_id",
            "actor_type",
            "model_provenance",
            "aux_model_provenance",
            "initial_deck",
        }:
            raise NemesisContractError(
                f"features.base.seats.{seat} fields do not match the contract"
            )
        if seat_payload.get("actor_type") not in ACTOR_TYPES:
            raise NemesisContractError(f"features.base.seats.{seat}.actor_type is invalid")
        actor_types.append(str(seat_payload["actor_type"]))
        _required_int(
            seat_payload.get("participant_id"),
            field=f"features.base.seats.{seat}.participant_id",
            minimum=-2**63,
        )
        _normalize_deck(
            seat_payload.get("initial_deck"),
            field=f"features.base.seats.{seat}.initial_deck",
        )
        provenance = seat_payload.get("model_provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != set(
            MODEL_PROVENANCE_FIELDS
        ):
            raise NemesisContractError(
                f"features.base.seats.{seat}.model_provenance is incomplete"
            )
        aux_provenance = seat_payload.get("aux_model_provenance")
        if not isinstance(aux_provenance, Mapping):
            raise NemesisContractError(
                f"features.base.seats.{seat}.aux_model_provenance must be an object"
            )
        for component, component_provenance in aux_provenance.items():
            if (
                not str(component).strip()
                or not isinstance(component_provenance, Mapping)
                or set(component_provenance) != set(MODEL_PROVENANCE_FIELDS)
            ):
                raise NemesisContractError(
                    f"features.base.seats.{seat}.aux_model_provenance is invalid"
                )
    derived_domain = _domain_for((actor_types[0], actor_types[1]))
    if base.get("domain") != derived_domain:
        raise NemesisContractError("features.base.domain mismatches actor types")

    extended = features.get("extended")
    if extended is not None:
        if not isinstance(extended, Mapping) or set(extended) != set(SEATS):
            raise NemesisContractError("features.extended must contain p1 and p2")
        normalized_extended: dict[str, Any] = {}
        for seat in SEATS:
            normalized_extended[seat] = _normalize_extended_seat(
                extended.get(seat),
                seat=seat,
                feature_cutoff=feature_cutoff,
            )
        if dict(extended) != normalized_extended:
            raise NemesisContractError(
                "features.extended contains non-canonical or unexpected fields"
            )

    provenance = record.get("provenance")
    expected_provenance = {
        "source",
        "campaign_id",
        "seed",
        "dataset_generation",
        "checkpoint_mix",
        "split_group",
        "split_fingerprint",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance:
        raise NemesisContractError("provenance fields do not match the contract")
    if not str(provenance.get("source") or "").strip():
        raise NemesisContractError("provenance.source is required")
    if not isinstance(provenance.get("checkpoint_mix"), list):
        raise NemesisContractError("provenance.checkpoint_mix must be a list")
    dataset_generation = _required_int(
        provenance.get("dataset_generation"),
        field="provenance.dataset_generation",
        minimum=1,
    )
    split_fingerprint = str(provenance.get("split_fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", split_fingerprint):
        raise NemesisContractError("provenance.split_fingerprint must be sha256")
    expected_split_fingerprint = _deck_pair_split_fingerprint(
        seats=seats,
        ruleset=str(base["ruleset"]),
        catalog_hash=(
            str(base["catalog_hash"])
            if base.get("catalog_hash") not in {None, ""}
            else None
        ),
    )
    if split_fingerprint != expected_split_fingerprint:
        raise NemesisContractError(
            "provenance.split_fingerprint does not match the deck pair"
        )
    if provenance.get("split_group") != (
        f"deck_pair:{expected_split_fingerprint}"
    ):
        raise NemesisContractError(
            "provenance.split_group does not match split_fingerprint"
        )

    quality = record.get("quality")
    expected_quality = {
        "eligible_lite",
        "eligible_standard",
        "sample_weight",
        "exclusion_reasons",
    }
    if not isinstance(quality, Mapping) or set(quality) != expected_quality:
        raise NemesisContractError("quality fields do not match the contract")
    if not isinstance(quality.get("eligible_lite"), bool) or not isinstance(
        quality.get("eligible_standard"), bool
    ):
        raise NemesisContractError("quality eligibility fields must be bool")
    weight = _optional_finite(quality.get("sample_weight"), field="quality.sample_weight")
    if weight is None or weight < 0:
        raise NemesisContractError("quality.sample_weight must be >= 0")
    if not isinstance(quality.get("exclusion_reasons"), list):
        raise NemesisContractError("quality.exclusion_reasons must be a list")
    exclusion_reasons = quality["exclusion_reasons"]
    if (
        any(not isinstance(reason, str) or not reason.strip() for reason in exclusion_reasons)
        or len(set(exclusion_reasons)) != len(exclusion_reasons)
    ):
        raise NemesisContractError(
            "quality.exclusion_reasons must contain unique non-empty strings"
        )
    if quality["eligible_standard"] and not quality["eligible_lite"]:
        raise NemesisContractError(
            "standard eligibility requires lite eligibility"
        )
    if quality["eligible_lite"] and (weight is None or weight <= 0):
        raise NemesisContractError(
            "lite-eligible records require a positive sample_weight"
        )
    if quality["eligible_standard"]:
        if (
            base["domain"] != "human-human"
            or not base["catalog_available"]
            or not isinstance(extended, Mapping)
            or any(extended.get(seat) is None for seat in SEATS)
        ):
            raise NemesisContractError(
                "standard eligibility requires catalogued human-human "
                "features with both pre-match snapshots"
            )
    expected_exclusions: list[str] = []
    if not catalog_available:
        expected_exclusions.append("catalog_unavailable")
    if base["domain"] == "human-human":
        if not isinstance(extended, Mapping):
            expected_exclusions.extend(
                f"{seat}_snapshot_unavailable" for seat in SEATS
            )
        else:
            expected_exclusions.extend(
                f"{seat}_snapshot_unavailable"
                for seat in SEATS
                if extended.get(seat) is None
            )
    elif base["domain"] == "human-bot":
        expected_exclusions.append("human_bot_standard_auxiliary_only")
    else:
        expected_exclusions.append("model_model_lite_only")
    if dataset_generation != 1:
        expected_exclusions.append("rehydrated_trace_generation")
    expected_exclusions = sorted(set(expected_exclusions))
    expected_lite = dataset_generation == 1
    expected_standard = expected_lite and not expected_exclusions
    expected_weight = (
        0.0
        if not expected_lite
        else 1.0
        if catalog_available
        else 0.5
    )
    if exclusion_reasons != expected_exclusions:
        raise NemesisContractError(
            "quality.exclusion_reasons do not match record provenance "
            "and feature availability"
        )
    if quality["eligible_lite"] is not expected_lite:
        raise NemesisContractError(
            "quality.eligible_lite does not match dataset generation"
        )
    if quality["eligible_standard"] is not expected_standard:
        raise NemesisContractError(
            "quality.eligible_standard does not match the canonical "
            "standard exclusions"
        )
    if weight != expected_weight:
        raise NemesisContractError(
            "quality.sample_weight does not match the canonical policy"
        )

    label = record.get("label")
    if label is None:
        if require_terminal:
            raise NemesisContractError("terminal label is required for export")
    elif isinstance(label, Mapping):
        if set(label) != {"status", "winner_seat", "duration_seconds", "turns_count"}:
            raise NemesisContractError("label fields do not match the contract")
        status = str(label.get("status") or "").lower()
        if status not in TERMINAL_STATUSES:
            raise NemesisContractError("label.status is invalid")
        winner_seat = label.get("winner_seat")
        if status in {"p1_win", "p2_win"}:
            expected = "p1" if status == "p1_win" else "p2"
            if winner_seat != expected:
                raise NemesisContractError("label.winner_seat mismatches status")
        elif winner_seat is not None:
            raise NemesisContractError("draw/stalemate winner_seat must be null")
        for field in ("duration_seconds", "turns_count"):
            if label.get(field) is not None:
                _required_int(label[field], field=f"label.{field}")
    else:
        raise NemesisContractError("label must be an object or null")
    privacy = record.get("privacy")
    if privacy is not None:
        if not isinstance(privacy, Mapping) or set(privacy) != {
            "identity_scheme",
            "include_players",
            "player_group_aliases",
            "player_group_scheme",
            "record_id_scheme",
        }:
            raise NemesisContractError(
                "privacy fields do not match the contract"
            )
        include_players = privacy.get("include_players")
        if not isinstance(include_players, bool):
            raise NemesisContractError("privacy.include_players must be bool")
        expected_scheme = (
            "raw_player_ids"
            if include_players
            else "side_pseudonyms_p1_1_p2_2"
        )
        if privacy.get("identity_scheme") != expected_scheme:
            raise NemesisContractError(
                "privacy.identity_scheme mismatches include_players"
            )
        expected_record_id_scheme = (
            NEMESIS_RAW_RECORD_ID_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
        )
        if privacy.get("record_id_scheme") != expected_record_id_scheme:
            raise NemesisContractError(
                "privacy.record_id_scheme mismatches include_players"
            )
        expected_player_group_scheme = (
            NEMESIS_RAW_PLAYER_GROUP_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
        )
        if (
            privacy.get("player_group_scheme")
            != expected_player_group_scheme
        ):
            raise NemesisContractError(
                "privacy.player_group_scheme mismatches include_players"
            )
        player_group_aliases = privacy.get("player_group_aliases")
        if (
            not isinstance(player_group_aliases, Mapping)
            or set(player_group_aliases) != set(SEATS)
        ):
            raise NemesisContractError(
                "privacy.player_group_aliases must contain p1 and p2"
            )
        if not include_players and (
            any(
                _PSEUDONYMIZED_PLAYER_GROUP_RE.fullmatch(
                    str(player_group_aliases.get(seat) or "")
                )
                is None
                for seat in SEATS
            )
            or player_group_aliases["p1"] == player_group_aliases["p2"]
        ):
            raise NemesisContractError(
                "pseudonymized privacy requires distinct opaque player "
                "group aliases"
            )
        if include_players and (
            str(player_group_aliases.get("p1"))
            != str(seats["p1"]["participant_id"])
            or str(player_group_aliases.get("p2"))
            != str(seats["p2"]["participant_id"])
        ):
            raise NemesisContractError(
                "raw privacy player group aliases must match participant IDs"
            )
        if not include_players and (
            seats["p1"]["participant_id"] != 1
            or seats["p2"]["participant_id"] != 2
        ):
            raise NemesisContractError(
                "pseudonymized privacy requires participant IDs p1=1,p2=2"
            )
        if not include_players and (
            _PSEUDONYMIZED_RECORD_ID_RE.fullmatch(
                str(record.get("battle_id") or "")
            )
            is None
            or _PSEUDONYMIZED_RECORD_ID_RE.fullmatch(
                str(record.get("match_id") or "")
            )
            is None
        ):
            raise NemesisContractError(
                "pseudonymized privacy requires random opaque battle_id "
                "and match_id values"
            )
    return deepcopy(dict(record))


class NemesisBattleCollector:
    """Freeze pre-match features, then attach exactly one terminal label."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = validate_nemesis_record(record)
        self._finalized = self._record.get("label") is not None

    @classmethod
    def from_v5_meta(
        cls,
        meta: Mapping[str, Any],
        *,
        feature_cutoff_at: str,
        extended_by_seat: Optional[Mapping[str, Any]] = None,
        source: str = "production_v5",
        campaign_id: str | None = None,
        seed: int | None = None,
    ) -> "NemesisBattleCollector":
        """Build from an allowlist of pre-match V5 metadata only."""

        if not isinstance(meta, Mapping):
            raise NemesisContractError("meta must be an object")
        feature_cutoff = _parse_timestamp(
            feature_cutoff_at,
            field="feature_cutoff_at",
        )
        battle_started_raw = meta.get("started_at")
        battle_started = _parse_timestamp(
            battle_started_raw,
            field="meta.started_at",
        )
        if feature_cutoff > battle_started:
            raise NemesisContractError("feature_cutoff_at must not follow meta.started_at")
        seats = {seat: _seat_payload(meta, seat) for seat in SEATS}
        actor_types = tuple(seats[seat]["actor_type"] for seat in SEATS)
        extended: dict[str, Any] | None
        if extended_by_seat is None:
            extended = None
        else:
            extended = {
                seat: _normalize_extended_seat(
                    extended_by_seat.get(seat),
                    seat=seat,
                    feature_cutoff=feature_cutoff,
                )
                for seat in SEATS
            }
        match_id = str(meta.get("match_id") or "")
        catalog_hash = str(meta.get("catalog_hash") or "").strip() or None
        domain = _domain_for(actor_types)  # type: ignore[arg-type]
        standard_exclusions: list[str] = []
        if catalog_hash is None:
            standard_exclusions.append("catalog_unavailable")
        if domain == "human-human":
            if not isinstance(extended, Mapping):
                standard_exclusions.extend(
                    [f"{seat}_snapshot_unavailable" for seat in SEATS]
                )
            else:
                standard_exclusions.extend(
                    f"{seat}_snapshot_unavailable"
                    for seat in SEATS
                    if extended.get(seat) is None
                )
        elif domain == "human-bot":
            standard_exclusions.append("human_bot_standard_auxiliary_only")
        else:
            standard_exclusions.append("model_model_lite_only")
        dataset_generation = int(meta.get("dataset_generation") or 1)
        rehydrated_generation = dataset_generation != 1
        if rehydrated_generation:
            standard_exclusions.append("rehydrated_trace_generation")
        standard_exclusions = sorted(set(standard_exclusions))
        split_fingerprint = _deck_pair_split_fingerprint(
            seats=seats,
            ruleset=str(meta.get("ruleset") or ""),
            catalog_hash=catalog_hash,
        )
        checkpoint_mix = sorted(
            {
                str(provenance.get("checkpoint_id") or provenance.get("weights_hash"))
                for seat in SEATS
                for provenance in [
                    seats[seat]["model_provenance"],
                    *seats[seat]["aux_model_provenance"].values(),
                ]
                if provenance.get("checkpoint_id") or provenance.get("weights_hash")
            }
        )
        record = {
            "schema_version": NEMESIS_SCHEMA,
            "visibility": NEMESIS_VISIBILITY,
            "battle_id": str(meta.get("battle_id") or ""),
            "match_id": match_id,
            "created_at": _utc_now_iso(),
            "features": {
                "base": {
                    "domain": domain,
                    "game_mode": str(meta.get("game_mode") or ""),
                    "ruleset": str(meta.get("ruleset") or ""),
                    "catalog_hash": catalog_hash,
                    "catalog_available": catalog_hash is not None,
                    "card_params_schema": str(meta.get("card_params_schema") or ""),
                    "deck_params_schema": str(meta.get("deck_params_schema") or ""),
                    "starting_player": str(meta.get("starting_player") or ""),
                    "feature_cutoff_at": feature_cutoff_at,
                    "battle_started_at": str(battle_started_raw),
                    "seats": seats,
                },
                "extended": extended,
            },
            "provenance": {
                "source": str(source),
                "campaign_id": str(campaign_id) if campaign_id is not None else None,
                "seed": int(seed) if seed is not None else None,
                "dataset_generation": dataset_generation,
                "checkpoint_mix": checkpoint_mix,
                "split_group": f"deck_pair:{split_fingerprint}",
                "split_fingerprint": split_fingerprint,
            },
            "quality": {
                "eligible_lite": not rehydrated_generation,
                "eligible_standard": (
                    not standard_exclusions and not rehydrated_generation
                ),
                "sample_weight": (
                    0.0
                    if rehydrated_generation
                    else 1.0 if catalog_hash is not None else 0.5
                ),
                "exclusion_reasons": standard_exclusions,
            },
            "label": None,
        }
        return cls(record)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._record)

    def finalize(
        self,
        *,
        status: str,
        duration_seconds: int | None = None,
        turns_count: int | None = None,
    ) -> dict[str, Any]:
        normalized = str(status or "").lower()
        if normalized not in TERMINAL_STATUSES:
            raise NemesisContractError(f"status must be one of {sorted(TERMINAL_STATUSES)}")
        label = {
            "status": normalized,
            "winner_seat": (
                "p1"
                if normalized == "p1_win"
                else "p2" if normalized == "p2_win" else None
            ),
            "duration_seconds": duration_seconds,
            "turns_count": turns_count,
        }
        if self._finalized:
            if self._record.get("label") != label:
                raise NemesisContractError("terminal label conflict")
            return self.snapshot()
        self._record["label"] = label
        self._finalized = True
        validate_nemesis_record(self._record, require_terminal=True)
        return self.snapshot()


def _pseudonymized_record_id(value: Any, *, namespace: UUID) -> str:
    """Return an export-local opaque alias without preserving raw substrings."""

    raw = str(value or "")
    if not raw:
        raise NemesisContractError("record identifier is required")
    return f"record_{uuid5(namespace, raw).hex}"


def _privacy_transform(
    record: Mapping[str, Any],
    *,
    include_players: bool,
    record_id_namespace: UUID | None = None,
) -> dict[str, Any]:
    exported = validate_nemesis_record(record, require_terminal=True)
    seats = exported["features"]["base"]["seats"]
    existing_privacy = exported.get("privacy")
    existing_group_aliases = (
        existing_privacy.get("player_group_aliases")
        if isinstance(existing_privacy, Mapping)
        else None
    )
    player_group_sources = {
        seat: (
            existing_group_aliases.get(seat)
            if isinstance(existing_group_aliases, Mapping)
            else seats[seat]["participant_id"]
        )
        for seat in SEATS
    }
    if not include_players:
        namespace = record_id_namespace or uuid4()
        seats["p1"]["participant_id"] = 1
        seats["p2"]["participant_id"] = 2
        exported["battle_id"] = _pseudonymized_record_id(
            exported["battle_id"],
            namespace=namespace,
        )
        exported["match_id"] = _pseudonymized_record_id(
            exported["match_id"],
            namespace=namespace,
        )
        player_group_aliases = {
            seat: (
                "player_"
                + uuid5(
                    namespace,
                    f"player-group:{player_group_sources[seat]}",
                ).hex
            )
            for seat in SEATS
        }
    else:
        player_group_aliases = {
            seat: str(seats[seat]["participant_id"])
            for seat in SEATS
        }
    exported["privacy"] = {
        "identity_scheme": (
            "raw_player_ids"
            if include_players
            else "side_pseudonyms_p1_1_p2_2"
        ),
        "include_players": bool(include_players),
        "player_group_aliases": player_group_aliases,
        "player_group_scheme": (
            NEMESIS_RAW_PLAYER_GROUP_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
        ),
        "record_id_scheme": (
            NEMESIS_RAW_RECORD_ID_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
        ),
    }
    validate_nemesis_record(exported, require_terminal=True)
    return exported


def pseudonymize_nemesis_record(
    record: Mapping[str, Any],
    *,
    record_id_namespace: UUID | None = None,
) -> dict[str, Any]:
    """Prepare one terminal training record under the canonical privacy contract."""

    return _privacy_transform(
        record,
        include_players=False,
        record_id_namespace=record_id_namespace,
    )


def export_nemesis_ndjson(
    records: Iterable[Mapping[str, Any]],
    *,
    include_players: bool = False,
) -> str:
    """Serialize complete battles; never truncate or emit an open collector."""

    exported: list[dict[str, Any]] = []
    seen: set[str] = set()
    record_id_namespace = uuid4()
    for record in records:
        transformed = _privacy_transform(
            record,
            include_players=include_players,
            record_id_namespace=record_id_namespace,
        )
        battle_id = str(transformed["battle_id"])
        if battle_id in seen:
            raise NemesisContractError(f"duplicate battle_id: {battle_id}")
        seen.add(battle_id)
        exported.append(transformed)
    if not exported:
        raise NemesisContractError("at least one terminal battle is required")
    header = {
        "record_type": "header",
        "format": NEMESIS_EXPORT_FORMAT,
        "format_version": 1,
        "schema_version": NEMESIS_SCHEMA,
        "created_at": _utc_now_iso(),
        "battle_count": len(exported),
        "identity_scheme": (
            "raw_player_ids"
            if include_players
            else "side_pseudonyms_p1_1_p2_2"
        ),
        "include_players": bool(include_players),
        "player_group_scheme": (
            NEMESIS_RAW_PLAYER_GROUP_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
        ),
        "record_id_scheme": (
            NEMESIS_RAW_RECORD_ID_SCHEME
            if include_players
            else NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
        ),
    }
    lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    lines.extend(
        json.dumps(
            {"record_type": "battle", **record},
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in exported
    )
    return "\n".join(lines) + "\n"


def write_nemesis_export(
    records: Iterable[Mapping[str, Any]],
    destination: str | Path,
    *,
    include_players: bool = False,
) -> Path:
    """Atomically stream one private whole-battle NDJSON export."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    spool_fd, spool_name = tempfile.mkstemp(
        prefix=f".{target.name}.records.",
        suffix=".tmp",
        dir=target.parent,
    )
    spool = Path(spool_name)
    temporary: str | None = None
    try:
        os.fchmod(spool_fd, 0o600)
        seen: set[str] = set()
        battle_count = 0
        record_id_namespace = uuid4()
        with os.fdopen(spool_fd, "w", encoding="utf-8") as handle:
            for record in records:
                transformed = _privacy_transform(
                    record,
                    include_players=include_players,
                    record_id_namespace=record_id_namespace,
                )
                battle_id = str(transformed["battle_id"])
                if battle_id in seen:
                    raise NemesisContractError(
                        f"duplicate battle_id: {battle_id}"
                    )
                seen.add(battle_id)
                battle_count += 1
                handle.write(
                    json.dumps(
                        {"record_type": "battle", **transformed},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        if battle_count == 0:
            raise NemesisContractError(
                "at least one terminal battle is required"
            )

        header = {
            "record_type": "header",
            "format": NEMESIS_EXPORT_FORMAT,
            "format_version": 1,
            "schema_version": NEMESIS_SCHEMA,
            "created_at": _utc_now_iso(),
            "battle_count": battle_count,
            "identity_scheme": (
                "raw_player_ids"
                if include_players
                else "side_pseudonyms_p1_1_p2_2"
            ),
            "include_players": bool(include_players),
            "player_group_scheme": (
                NEMESIS_RAW_PLAYER_GROUP_SCHEME
                if include_players
                else NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
            ),
            "record_id_scheme": (
                NEMESIS_RAW_RECORD_ID_SCHEME
                if include_players
                else NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
            ),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(header, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            with spool.open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        target.chmod(0o600)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    finally:
        spool.unlink(missing_ok=True)
    return target


__all__ = [
    "ACTOR_TYPES",
    "DOMAINS",
    "NEMESIS_EXPORT_FORMAT",
    "NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME",
    "NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME",
    "NEMESIS_RAW_RECORD_ID_SCHEME",
    "NEMESIS_RAW_PLAYER_GROUP_SCHEME",
    "NEMESIS_SCHEMA",
    "NemesisBattleCollector",
    "NemesisContractError",
    "export_nemesis_ndjson",
    "pseudonymize_nemesis_record",
    "validate_nemesis_record",
    "write_nemesis_export",
]
