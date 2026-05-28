from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.bot_brain import BerserkInference
from ai.train_v2.shadow import (
    FakeLegacyBrain,
    run_shadow_matchup,
)


def summarize_shadow_result(result: dict) -> dict:
    episodes = result.get("episodes", 0)
    steps = result.get("steps", 0)
    matches = result.get("matches", 0)
    mismatches = result.get("mismatches", 0)
    match_rate = result.get("match_rate", 0.0)
    legacy_invalid_actions = result.get("legacy_invalid_actions", 0)
    overlay_invalid_actions = result.get("overlay_invalid_actions", 0)
    played_invalid_actions = result.get("played_invalid_actions", 0)
    legacy_latency_ms_p50 = result.get("legacy_latency_ms_p50", 0.0)
    legacy_latency_ms_p95 = result.get("legacy_latency_ms_p95", 0.0)
    overlay_latency_ms_p50 = result.get("overlay_latency_ms_p50", 0.0)
    overlay_latency_ms_p95 = result.get("overlay_latency_ms_p95", 0.0)
    mismatch_rate = mismatches / steps if steps > 0 else 0.0

    return {
        "episodes": int(episodes),
        "steps": int(steps),
        "matches": int(matches),
        "mismatches": int(mismatches),
        "match_rate": float(match_rate),
        "legacy_invalid_actions": int(legacy_invalid_actions),
        "overlay_invalid_actions": int(overlay_invalid_actions),
        "played_invalid_actions": int(played_invalid_actions),
        "legacy_latency_ms_p50": float(legacy_latency_ms_p50),
        "legacy_latency_ms_p95": float(legacy_latency_ms_p95),
        "overlay_latency_ms_p50": float(overlay_latency_ms_p50),
        "overlay_latency_ms_p95": float(overlay_latency_ms_p95),
        "mismatch_rate": float(mismatch_rate),
    }


def extract_shadow_mismatches(result: dict, *, limit: int = 20) -> list[dict]:
    if limit <= 0:
        return []

    mismatches: list[dict] = []
    episodes_detail = result.get("episodes_detail", [])
    for episode_index, episode in enumerate(episodes_detail):
        seed = episode.get("summary", {}).get("seed")
        if seed is None:
            seed = episode.get("seed")
        for decision in episode.get("decisions", []):
            if decision.get("match") is False:
                mismatches.append({
                    "episode_index": episode_index,
                    "seed": seed,
                    "step": decision.get("step"),
                    "player_id": decision.get("player_id"),
                    "legacy_action_id": decision.get("legacy_action_id"),
                    "overlay_action_id": decision.get("overlay_action_id"),
                    "played_action_id": decision.get("played_action_id"),
                    "legacy_type": decision.get("legacy", {}).get("type", "unknown"),
                    "overlay_type": decision.get("overlay", {}).get("type", "unknown"),
                    "played_type": decision.get("played", {}).get("type", "unknown"),
                })
                if len(mismatches) >= limit:
                    return mismatches
    return mismatches


def format_shadow_markdown(
    result: dict,
    *,
    title: str = "TrainV2 Shadow Report",
    mismatch_limit: int = 20,
) -> str:
    summary = summarize_shadow_result(result)
    mismatches = extract_shadow_mismatches(result, limit=mismatch_limit)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Episodes: {summary['episodes']}")
    lines.append(f"- Steps: {summary['steps']}")
    lines.append(f"- Match rate: {summary['match_rate']:.3f}")
    lines.append(f"- Mismatches: {summary['mismatches']}")
    lines.append(f"- Legacy invalid actions: {summary['legacy_invalid_actions']}")
    lines.append(f"- Overlay invalid actions: {summary['overlay_invalid_actions']}")
    lines.append(f"- Played invalid actions: {summary['played_invalid_actions']}")
    lines.append("")
    lines.append("## Latency")
    lines.append(
        f"- Legacy p50/p95: {summary['legacy_latency_ms_p50']:.1f} / "
        f"{summary['legacy_latency_ms_p95']:.1f} ms"
    )
    lines.append(
        f"- Overlay p50/p95: {summary['overlay_latency_ms_p50']:.1f} / "
        f"{summary['overlay_latency_ms_p95']:.1f} ms"
    )
    lines.append("")
    lines.append("## Top Mismatches")
    if mismatches:
        lines.append("| Episode | Step | Player | Legacy | Overlay | Played |")
        lines.append("|---:|---:|---:|---|---|---|")
        for m in mismatches:
            lines.append(
                f"| {m['episode_index']} | {m['step']} | {m['player_id']} | "
                f"{m['legacy_type']} | {m['overlay_type']} | {m['played_type']} |"
            )
    else:
        lines.append("No mismatches recorded.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Shadow comparison is read-only and does not modify production profiles.")
    lines.append("")

    return "\n".join(lines)


def write_shadow_evidence_pack(
    result: dict,
    output_dir: str,
    *,
    overlay_path: str | None = None,
    candidate_profile_path: str | None = None,
    candidate_dir: str | None = None,
    title: str = "TrainV2 Shadow Report",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize_shadow_result(result)
    mismatches = extract_shadow_mismatches(result)
    markdown = format_shadow_markdown(result, title=title)

    result_path = out / "shadow_result.json"
    summary_path = out / "shadow_summary.json"
    md_path = out / "shadow_summary.md"
    mismatches_path = out / "shadow_mismatches.json"
    manifest_path = out / "manifest.json"

    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(markdown, encoding="utf-8")
    mismatches_path.write_text(
        json.dumps(mismatches, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "version": "train_v2_shadow_evidence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overlay_path": overlay_path,
        "candidate_profile_path": candidate_profile_path,
        "candidate_dir": candidate_dir,
        "artifacts": {
            "shadow_result": str(result_path.resolve()),
            "shadow_summary": str(summary_path.resolve()),
            "shadow_markdown": str(md_path.resolve()),
            "shadow_mismatches": str(mismatches_path.resolve()),
        },
        "summary": summary,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def attach_shadow_pack_to_candidate(
    candidate_dir: str,
    shadow_pack_dir: str,
    *,
    name: str | None = None,
) -> dict:
    src = Path(shadow_pack_dir)
    base_name = name if name is not None else src.name
    dest = Path(candidate_dir) / "shadow_evidence" / base_name

    if dest.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = Path(candidate_dir) / "shadow_evidence" / f"{base_name}_{suffix}"

    dest.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

    return {
        "candidate_dir": str(Path(candidate_dir).resolve()),
        "shadow_pack_dir": str(src.resolve()),
        "attached_dir": str(dest.resolve()),
    }


def _main():
    parser = argparse.ArgumentParser(description="Build shadow evidence pack from result or overlay")
    parser.add_argument("--input", default=None, help="Existing shadow result JSON")
    parser.add_argument("--overlay", default=None, help="Profile overlay JSON")
    parser.add_argument("--overlay-difficulty", default=None)
    parser.add_argument("--legacy-profile-json", default=None)
    parser.add_argument("--legacy-difficulty", default="easy")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--play-policy", default="legacy", choices=["legacy", "overlay", "random", "greedy_face"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-profile", default=None)
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        if not args.overlay:
            parser.error("--overlay is required when --input is not provided")

        seeds = args.seeds
        if seeds is None:
            seeds = list(range(args.seed, args.seed + args.games))

        legacy_brain = None
        if args.legacy_profile_json:
            with open(args.legacy_profile_json, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            legacy_brain = BerserkInference(profiles=profiles)
        else:
            print(
                "No legacy profile provided; using first-legal fake legacy policy",
                file=sys.stderr,
            )
            legacy_brain = FakeLegacyBrain()

        result = run_shadow_matchup(
            args.overlay,
            legacy_brain=legacy_brain,
            legacy_difficulty=args.legacy_difficulty,
            overlay_difficulty=args.overlay_difficulty,
            seeds=seeds,
            max_steps=args.max_steps,
            play_policy=args.play_policy,
        )

    manifest = write_shadow_evidence_pack(
        result,
        args.output_dir,
        overlay_path=args.overlay,
        candidate_profile_path=args.candidate_profile,
        candidate_dir=args.candidate_dir,
    )

    attached_dir = None
    if args.attach and args.candidate_dir:
        attach_info = attach_shadow_pack_to_candidate(
            args.candidate_dir,
            args.output_dir,
        )
        attached_dir = attach_info["attached_dir"]

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    else:
        summary = manifest["summary"]
        print(f"Shadow evidence pack: {args.output_dir}")
        print(
            f"Summary: steps={summary['steps']} "
            f"match_rate={summary['match_rate']:.3f} "
            f"mismatches={summary['mismatches']}"
        )
        md_path = Path(args.output_dir) / "shadow_summary.md"
        print(f"Markdown: {md_path.resolve()}")
        if attached_dir:
            print(f"Attached: {attached_dir}")


if __name__ == "__main__":
    _main()
