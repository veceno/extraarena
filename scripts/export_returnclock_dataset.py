#!/usr/bin/env python3
"""Export bounded ReturnClock telemetry into privacy-preserving JSONL.

The exporter is deliberately offline.  It does not train a model, schedule a
notification, or mutate production state.

Example:

    RETURNCLOCK_DATASET_SALT=... \
      python scripts/export_returnclock_dataset.py \
        --start 2026-08-01T00:00:00Z \
        --end 2026-09-01T00:00:00Z \
        --output datasets/returnclock/2026-08.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infrastructure.config import get_settings
from infrastructure.database import Database, RETURNCLOCK_DATASET_SCHEMA
from infrastructure.returnclock_dataset import (
    DEFAULT_HORIZON_HOURS,
    DEFAULT_MIN_ANALYTICS_VERSION,
    materialize_returnclock_dataset,
    write_returnclock_jsonl,
)


def _parse_timestamp(value: str | None, *, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ReturnClock session/notification telemetry as JSONL.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="First prediction cutoff, ISO-8601.")
    parser.add_argument(
        "--end",
        help=(
            "Exclusive telemetry boundary and censoring time; defaults to "
            "now minus --safety-lag-minutes."
        ),
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=DEFAULT_HORIZON_HOURS,
    )
    parser.add_argument(
        "--safety-lag-minutes",
        type=int,
        default=10,
        help=(
            "When --end is omitted, stop this many minutes before now so "
            "late session/delivery writes cannot cross the dataset boundary."
        ),
    )
    parser.add_argument(
        "--min-analytics-version",
        type=int,
        default=DEFAULT_MIN_ANALYTICS_VERSION,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help=(
            "Maximum rows per raw stream (1..1000000); reads are keyset-"
            "paged inside one repeatable-read snapshot."
        ),
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="Optional diagnostic scope. The export still contains only a hash.",
    )
    parser.add_argument(
        "--salt-env",
        default="RETURNCLOCK_DATASET_SALT",
        help="Environment variable containing the HMAC salt.",
    )
    parser.add_argument(
        "--salt-key-id-env",
        default="RETURNCLOCK_DATASET_SALT_KEY_ID",
        help=(
            "Environment variable containing the non-secret key rotation ID "
            "written to the dataset header."
        ),
    )
    return parser


async def _export(
    args: argparse.Namespace,
    *,
    database_factory: Callable[[Any], Any] | None = None,
    settings_factory: Callable[[], Any] | None = None,
) -> dict:
    if (database_factory is None) != (settings_factory is None):
        raise ValueError(
            "database_factory and settings_factory must be provided together"
        )
    safety_lag_minutes = int(getattr(args, "safety_lag_minutes", 10))
    if safety_lag_minutes < 0 or safety_lag_minutes > 24 * 60:
        raise ValueError("--safety-lag-minutes must be in [0, 1440]")
    ingested_before = datetime.now(timezone.utc)
    end_at = _parse_timestamp(
        args.end,
        default=(
            ingested_before
            - timedelta(minutes=safety_lag_minutes)
        ),
    )
    assert end_at is not None
    if end_at > ingested_before:
        raise ValueError("--end must not be in the future")
    cutoff_start = _parse_timestamp(args.start)
    if cutoff_start is not None and cutoff_start >= end_at:
        raise ValueError("--start must be before --end")
    if args.horizon_hours < 1 or args.horizon_hours > 31 * 24:
        raise ValueError("--horizon-hours must be in [1, 744]")
    if args.min_analytics_version < 1:
        raise ValueError("--min-analytics-version must be >= 1")
    if args.limit < 1 or args.limit > 1_000_000:
        raise ValueError("--limit must be in [1, 1000000]")

    privacy_salt = os.getenv(args.salt_env, "")
    if len(privacy_salt.encode("utf-8")) < 32:
        raise RuntimeError(
            "ReturnClock export HMAC salt must contain at least 32 bytes "
            "of secret material"
        )
    salt_key_id_env = str(
        getattr(args, "salt_key_id_env", "RETURNCLOCK_DATASET_SALT_KEY_ID")
    )
    privacy_key_id = os.getenv(salt_key_id_env, "").strip()
    if not privacy_key_id:
        raise RuntimeError(
            "ReturnClock pseudonymization key rotation ID is not configured"
        )

    # Pull enough pre-cutoff history to construct cadence features.  Targets
    # remain bounded by end_at and are never read from the future.
    history_days = max(28, math.ceil(args.horizon_hours / 24))
    fetch_start = (
        cutoff_start - timedelta(days=history_days)
        if cutoff_start is not None
        else None
    )
    settings = (settings_factory or get_settings)()
    db = (
        database_factory(settings.database)
        if database_factory is not None
        else Database(settings.database)
    )
    await db.connect()
    try:
        raw = await db.fetch_returnclock_dataset_rows(
            start_at=fetch_start,
            end_at=end_at,
            ingested_before=ingested_before,
            user_id=args.user_id,
            limit=args.limit,
        )
    finally:
        await db.close()

    if raw.get("schema") != RETURNCLOCK_DATASET_SCHEMA:
        raise RuntimeError(
            "incompatible raw ReturnClock schema: "
            f"expected {RETURNCLOCK_DATASET_SCHEMA!r}, "
            f"got {raw.get('schema')!r}"
        )

    saturated_streams = [
        stream_name
        for stream_name in ("sessions", "decisions", "delivery_events")
        if len(raw.get(stream_name, [])) >= args.limit
    ]
    if saturated_streams:
        raise RuntimeError(
            "raw export reached --limit for "
            + ", ".join(saturated_streams)
            + "; narrow the time window or increase --limit to avoid false "
            "no-return/censoring labels"
        )

    dataset = materialize_returnclock_dataset(
        sessions=raw.get("sessions", []),
        decisions=raw.get("decisions", []),
        delivery_events=raw.get("delivery_events", []),
        dataset_end=end_at,
        ingested_before=ingested_before,
        cutoff_start=cutoff_start,
        privacy_salt=privacy_salt,
        privacy_key_id=privacy_key_id,
        horizon_hours=args.horizon_hours,
        min_analytics_version=args.min_analytics_version,
    )
    destination = write_returnclock_jsonl(dataset, args.output)
    return {
        "ok": True,
        "output": str(destination),
        "raw_schema": raw.get("schema"),
        "safety_lag_minutes": (
            0 if args.end is not None else safety_lag_minutes
        ),
        "event_time_end_at": end_at.isoformat(),
        "ingested_before": ingested_before.isoformat(),
        "pseudonymization_key_id": privacy_key_id,
        **dataset.header["summary"],
    }


async def export_returnclock_dataset(
    *,
    output: str | Path,
    start: str | None = None,
    end: str | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    safety_lag_minutes: int = 10,
    min_analytics_version: int = DEFAULT_MIN_ANALYTICS_VERSION,
    limit: int = 50_000,
    salt_env: str = "RETURNCLOCK_DATASET_SALT",
    salt_key_id_env: str = "RETURNCLOCK_DATASET_SALT_KEY_ID",
    database_factory: Callable[[Any], Any] | None = None,
    settings_factory: Callable[[], Any] | None = None,
) -> dict:
    """Public async entry point shared by the CLI and ExtraRLHF MCP.

    The HMAC salt is referenced only by environment-variable name.  Its value
    never crosses the Python API or the MCP response boundary.
    """

    return await _export(
        argparse.Namespace(
            output=Path(output),
            start=start,
            end=end,
            horizon_hours=int(horizon_hours),
            safety_lag_minutes=int(safety_lag_minutes),
            min_analytics_version=int(min_analytics_version),
            limit=int(limit),
            user_id=None,
            salt_env=str(salt_env),
            salt_key_id_env=str(salt_key_id_env),
        ),
        database_factory=database_factory,
        settings_factory=settings_factory,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = asyncio.run(_export(args))
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
