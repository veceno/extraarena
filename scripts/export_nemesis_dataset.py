#!/usr/bin/env python3
"""Validate and privacy-transform canonical Nemesis battle records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.nemesis_dataset import write_nemesis_export


def _canonical_record(value: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    if value.get("schema_version") == "extraarena_nemesis_battle_v1":
        return value
    meta = value.get("meta")
    if isinstance(meta, dict):
        nemesis = meta.get("nemesis_record")
        if isinstance(nemesis, dict):
            return nemesis
        raise ValueError(
            f"line {line_number}: V5 battle bundle has no meta.nemesis_record"
        )
    raise ValueError(
        f"line {line_number}: expected canonical Nemesis record or V5 battle bundle"
    )


def _load_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream canonical records without loading a large V5 export into RAM."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"line {line_number}: record must be an object"
                )
            if value.get("record_type") == "header":
                continue
            if value.get("record_type") == "battle":
                value = {
                    key: item
                    for key, item in value.items()
                    if key != "record_type"
                }
            yield _canonical_record(value, line_number=line_number)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Headerless or exported NDJSON records")
    parser.add_argument("output", type=Path, help="Destination private NDJSON export")
    parser.add_argument(
        "--include-players",
        action="store_true",
        help="Authorized diagnostic export with raw participant IDs",
    )
    args = parser.parse_args(argv)
    write_nemesis_export(
        _load_records(args.input),
        args.output,
        include_players=args.include_players,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
