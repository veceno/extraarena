from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ai.train_v2.web_panel import (
    WebPanelConfig,
    collect_panel_data,
    run_web_panel,
)


def write_panel_snapshot(
    *,
    runs_dir: str,
    releases_dir: str | None,
    output_path: str,
) -> dict:
    data = collect_panel_data(runs_dir=runs_dir, releases_dir=releases_dir)
    snapshot = {
        "version": "train_v2_panel_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "path": str(out.resolve()),
        "runs": data.get("lifecycle", {}).get("runs", 0),
        "profiles_ok": data.get("lifecycle", {}).get("profiles_ok", 0),
        "release_bundles": data.get("lifecycle", {}).get("release_bundles", 0),
    }


def run_doctor(*, runs_dir: str, releases_dir: str | None) -> dict:
    data = collect_panel_data(runs_dir=runs_dir, releases_dir=releases_dir)
    issues: list[str] = []

    runs_dir_exists = Path(runs_dir).exists()
    releases_dir_exists = Path(releases_dir).exists() if releases_dir else None

    if not runs_dir_exists:
        issues.append("Runs directory does not exist")
    if releases_dir and not releases_dir_exists:
        issues.append("Releases directory does not exist")

    lifecycle = data.get("lifecycle", {})
    if lifecycle.get("runs", 0) == 0:
        issues.append("No runs found")
    if lifecycle.get("profiles_ok", 0) == 0:
        issues.append("No profiles found")
    if lifecycle.get("profiles_errors", 0) > 0:
        issues.append("Profile registry has error rows")
    if lifecycle.get("release_bundles", 0) == 0:
        issues.append("No release bundles found")
    if (lifecycle.get("acceptance_pass", 0) + lifecycle.get("acceptance_warn", 0) + lifecycle.get("acceptance_fail", 0)) == 0:
        issues.append("No acceptance reports found")

    return {
        "runs_dir_exists": runs_dir_exists,
        "releases_dir_exists": releases_dir_exists,
        "run_count": lifecycle.get("runs", 0),
        "profile_count": lifecycle.get("profiles_ok", 0),
        "release_count": lifecycle.get("release_bundles", 0),
        "issues": issues,
    }


def _cmd_panel(args) -> None:
    config = WebPanelConfig(
        runs_dir=args.runs_dir,
        releases_dir=args.releases_dir,
        host=args.host,
        port=args.port,
    )
    run_web_panel(config)


def _cmd_snapshot(args) -> None:
    result = write_panel_snapshot(
        runs_dir=args.runs_dir,
        releases_dir=args.releases_dir,
        output_path=args.output,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Snapshot written to: {result['path']}")
        print(f"Runs: {result['runs']}, Profiles: {result['profiles_ok']}, Releases: {result['release_bundles']}")


def _cmd_doctor(args) -> None:
    result = run_doctor(
        runs_dir=args.runs_dir,
        releases_dir=args.releases_dir,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"TrainV2 Doctor")
        print(f"  runs_dir_exists: {result['runs_dir_exists']}")
        print(f"  releases_dir_exists: {result['releases_dir_exists']}")
        print(f"  runs: {result['run_count']}")
        print(f"  profiles: {result['profile_count']}")
        print(f"  releases: {result['release_count']}")
        if result["issues"]:
            print("  Issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")
        else:
            print("  No issues found.")


def _main() -> None:
    parser = argparse.ArgumentParser(description="TrainV2 operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # panel
    panel_p = subparsers.add_parser("panel", help="Start local web panel")
    panel_p.add_argument("--runs-dir", default="ai/train_v2/runs")
    panel_p.add_argument("--releases-dir", default=None)
    panel_p.add_argument("--host", default="127.0.0.1")
    panel_p.add_argument("--port", type=int, default=8765)
    panel_p.set_defaults(func=_cmd_panel)

    # snapshot
    snap_p = subparsers.add_parser("snapshot", help="Write panel snapshot JSON")
    snap_p.add_argument("--runs-dir", default="ai/train_v2/runs")
    snap_p.add_argument("--releases-dir", default=None)
    snap_p.add_argument("--output", required=True)
    snap_p.add_argument("--json", action="store_true")
    snap_p.set_defaults(func=_cmd_snapshot)

    # doctor
    doc_p = subparsers.add_parser("doctor", help="Read-only health check")
    doc_p.add_argument("--runs-dir", default="ai/train_v2/runs")
    doc_p.add_argument("--releases-dir", default=None)
    doc_p.add_argument("--json", action="store_true")
    doc_p.set_defaults(func=_cmd_doctor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
