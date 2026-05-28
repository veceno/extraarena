from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.train_v2.profile_registry import load_profile_overlay


@dataclass
class GateThresholds:
    min_winrate_random: float = 0.35
    min_winrate_end_turn: float = 0.80
    max_parity_mismatches: int = 0
    max_overlay_invalid_actions: int = 0
    max_brain_invalid_actions: int = 0
    max_played_invalid_actions: int = 0
    max_legacy_invalid_actions: int = 0
    max_overlay_latency_p95_ms: float = 50.0
    min_shadow_steps: int = 20
    min_required_artifacts: int = 4
    min_match_rate: float | None = None


def _load_json_if_exists(p: Path) -> dict | None:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _discover_shadow_evidence(candidate_dir: Path) -> tuple[Path | None, Path | None]:
    shadow_dir = candidate_dir / "shadow_evidence"
    if not shadow_dir.is_dir():
        return None, None
    packs = [d for d in shadow_dir.iterdir() if d.is_dir()]
    if not packs:
        return None, None
    packs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest = packs[0]
    manifest = latest / "manifest.json"
    summary = latest / "shadow_summary.json"
    return (
        manifest if manifest.exists() else None,
        summary if summary.exists() else None,
    )


def load_gate_inputs(
    *,
    candidate_dir: str | None = None,
    profile_path: str | None = None,
    overlay_path: str | None = None,
    shadow_pack_dir: str | None = None,
    leaderboard_path: str | None = None,
) -> dict:
    candidate: dict | None = None
    profile: dict | None = None
    overlay: dict | None = None
    shadow_manifest: dict | None = None
    shadow_summary: dict | None = None
    leaderboard: dict | None = None
    paths: dict[str, str] = {}
    missing: list[str] = []

    cdir = Path(candidate_dir) if candidate_dir else None

    # 1. Discovery from candidate_dir
    if cdir:
        cand_file = cdir / "candidate.json"
        if cand_file.exists():
            candidate = _load_json_if_exists(cand_file)
            paths["candidate"] = str(cand_file.resolve())
        else:
            missing.append("candidate")

        prof_file = cdir / "candidate_profile.json"
        if prof_file.exists():
            profile = _load_json_if_exists(prof_file)
            paths["profile"] = str(prof_file.resolve())

        ld_file = cdir / "leaderboard.json"
        if ld_file.exists():
            leaderboard = _load_json_if_exists(ld_file)
            paths["leaderboard"] = str(ld_file.resolve())

        manifest_p, summary_p = _discover_shadow_evidence(cdir)
        if manifest_p:
            shadow_manifest = _load_json_if_exists(manifest_p)
            paths["shadow_manifest"] = str(manifest_p.resolve())
        if summary_p:
            shadow_summary = _load_json_if_exists(summary_p)
            paths["shadow_summary"] = str(summary_p.resolve())

    # 2. Explicit overrides / fallbacks
    if profile_path:
        p = Path(profile_path)
        if p.exists():
            profile = _load_json_if_exists(p)
            paths["profile"] = str(p.resolve())
        else:
            profile = None
            missing.append("profile")
    elif cdir and profile is None:
        missing.append("profile")

    if overlay_path:
        p = Path(overlay_path)
        if p.exists():
            try:
                overlay = load_profile_overlay(str(p))
                paths["overlay"] = str(p.resolve())
            except Exception:
                overlay = None
                missing.append("overlay")
        else:
            missing.append("overlay")

    if shadow_pack_dir:
        sp = Path(shadow_pack_dir)
        m = sp / "manifest.json"
        s = sp / "shadow_summary.json"
        if m.exists():
            shadow_manifest = _load_json_if_exists(m)
            paths["shadow_manifest"] = str(m.resolve())
        else:
            missing.append("shadow_manifest")
        if s.exists():
            shadow_summary = _load_json_if_exists(s)
            paths["shadow_summary"] = str(s.resolve())
        else:
            missing.append("shadow_summary")
    elif cdir:
        if shadow_manifest is None:
            missing.append("shadow_manifest")
        if shadow_summary is None:
            missing.append("shadow_summary")

    if leaderboard_path:
        p = Path(leaderboard_path)
        if p.exists():
            leaderboard = _load_json_if_exists(p)
            paths["leaderboard"] = str(p.resolve())
        else:
            missing.append("leaderboard")
    elif cdir and leaderboard is None:
        missing.append("leaderboard")

    # 3. Validate that we have at least something
    if not any([candidate, profile, overlay, shadow_manifest, shadow_summary, leaderboard]):
        raise ValueError("No gate inputs found")

    # deduplicate missing preserving order
    seen = set()
    uniq = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            uniq.append(m)

    return {
        "candidate": candidate,
        "profile": profile,
        "overlay": overlay,
        "shadow_manifest": shadow_manifest,
        "shadow_summary": shadow_summary,
        "leaderboard": leaderboard,
        "paths": paths,
        "missing": uniq,
    }


def _get_leaderboard_row(inputs: dict) -> dict | None:
    leaderboard = inputs.get("leaderboard")
    if not leaderboard:
        return None
    rows = leaderboard.get("rows", [])
    if not rows:
        return None

    model_name = None
    candidate = inputs.get("candidate")
    if candidate:
        model_name = candidate.get("model_name")
    if not model_name:
        profile = inputs.get("profile")
        if profile:
            model_name = profile.get("source", {}).get("model_name") or profile.get("model_name")
    if not model_name:
        shadow_manifest = inputs.get("shadow_manifest")
        if shadow_manifest:
            model_name = shadow_manifest.get("summary", {}).get("model_name")

    if model_name:
        for row in rows:
            if row.get("model_name") == model_name:
                return row

    best = leaderboard.get("best")
    if best:
        return best

    return rows[0]


def evaluate_acceptance_gate(
    inputs: dict,
    thresholds: GateThresholds | None = None,
) -> dict:
    if thresholds is None:
        thresholds = GateThresholds()

    row = _get_leaderboard_row(inputs)
    shadow_summary = inputs.get("shadow_summary")
    missing = inputs.get("missing", [])

    checks: list[dict] = []
    pass_count = 0
    warn_count = 0
    fail_count = 0

    def _add_check(name: str, status: str, value: Any, threshold: Any, message: str) -> None:
        nonlocal pass_count, warn_count, fail_count
        checks.append({
            "name": name,
            "status": status,
            "value": value,
            "threshold": threshold,
            "message": message,
        })
        if status == "pass":
            pass_count += 1
        elif status == "warn":
            warn_count += 1
        elif status == "fail":
            fail_count += 1

    # artifact count
    artifact_count = sum(
        1
        for k in ["candidate", "profile", "overlay", "shadow_manifest", "shadow_summary", "leaderboard"]
        if inputs.get(k) is not None
    )
    if artifact_count >= thresholds.min_required_artifacts:
        _add_check("artifact_count", "pass", artifact_count, thresholds.min_required_artifacts, f"{artifact_count} artifacts found")
    else:
        _add_check("artifact_count", "fail", artifact_count, thresholds.min_required_artifacts, f"Only {artifact_count} artifacts, need {thresholds.min_required_artifacts}")

    # winrate random
    wr_random = row.get("wr_random") if row else None
    if wr_random is not None:
        if wr_random >= thresholds.min_winrate_random:
            _add_check("winrate_random", "pass", wr_random, thresholds.min_winrate_random, "ok")
        else:
            _add_check("winrate_random", "fail", wr_random, thresholds.min_winrate_random, f"winrate {wr_random:.3f} below threshold")
    else:
        _add_check("winrate_random", "warn", None, thresholds.min_winrate_random, "leaderboard data missing")

    # winrate end_turn
    wr_end_turn = row.get("wr_end_turn") if row else None
    if wr_end_turn is not None:
        if wr_end_turn >= thresholds.min_winrate_end_turn:
            _add_check("winrate_end_turn", "pass", wr_end_turn, thresholds.min_winrate_end_turn, "ok")
        else:
            _add_check("winrate_end_turn", "fail", wr_end_turn, thresholds.min_winrate_end_turn, f"winrate {wr_end_turn:.3f} below threshold")
    else:
        _add_check("winrate_end_turn", "warn", None, thresholds.min_winrate_end_turn, "leaderboard data missing")

    # parity mismatches
    parity = row.get("parity_mismatches") if row else None
    if parity is not None:
        if parity <= thresholds.max_parity_mismatches:
            _add_check("parity_mismatches", "pass", parity, thresholds.max_parity_mismatches, "ok")
        else:
            _add_check("parity_mismatches", "fail", parity, thresholds.max_parity_mismatches, f"{parity} mismatches above threshold")
    else:
        _add_check("parity_mismatches", "warn", None, thresholds.max_parity_mismatches, "leaderboard data missing")

    # overlay invalid actions
    overlay_invalid = shadow_summary.get("overlay_invalid_actions") if shadow_summary else None
    if overlay_invalid is not None:
        if overlay_invalid <= thresholds.max_overlay_invalid_actions:
            _add_check("overlay_invalid_actions", "pass", overlay_invalid, thresholds.max_overlay_invalid_actions, "ok")
        else:
            _add_check("overlay_invalid_actions", "fail", overlay_invalid, thresholds.max_overlay_invalid_actions, f"{overlay_invalid} invalid actions above threshold")
    else:
        _add_check("overlay_invalid_actions", "warn", None, thresholds.max_overlay_invalid_actions, "shadow summary missing")

    # brain invalid actions
    brain_invalid = row.get("brain_invalid_total") if row else None
    if brain_invalid is not None:
        if brain_invalid <= thresholds.max_brain_invalid_actions:
            _add_check("brain_invalid_actions", "pass", brain_invalid, thresholds.max_brain_invalid_actions, "ok")
        else:
            _add_check("brain_invalid_actions", "fail", brain_invalid, thresholds.max_brain_invalid_actions, f"{brain_invalid} brain invalid actions above threshold")
    else:
        _add_check("brain_invalid_actions", "warn", None, thresholds.max_brain_invalid_actions, "leaderboard data missing")

    # played invalid actions
    played_invalid = shadow_summary.get("played_invalid_actions") if shadow_summary else None
    if played_invalid is not None:
        if played_invalid <= thresholds.max_played_invalid_actions:
            _add_check("played_invalid_actions", "pass", played_invalid, thresholds.max_played_invalid_actions, "ok")
        else:
            _add_check("played_invalid_actions", "fail", played_invalid, thresholds.max_played_invalid_actions, f"{played_invalid} played invalid actions above threshold")
    else:
        _add_check("played_invalid_actions", "warn", None, thresholds.max_played_invalid_actions, "shadow summary missing")

    # legacy invalid actions
    legacy_invalid = shadow_summary.get("legacy_invalid_actions") if shadow_summary else None
    if legacy_invalid is not None:
        if legacy_invalid <= thresholds.max_legacy_invalid_actions:
            _add_check("legacy_invalid_actions", "pass", legacy_invalid, thresholds.max_legacy_invalid_actions, "ok")
        else:
            _add_check("legacy_invalid_actions", "fail", legacy_invalid, thresholds.max_legacy_invalid_actions, f"{legacy_invalid} legacy invalid actions above threshold")
    else:
        _add_check("legacy_invalid_actions", "warn", None, thresholds.max_legacy_invalid_actions, "shadow summary missing")

    # overlay latency p95
    lat_p95 = shadow_summary.get("overlay_latency_ms_p95") if shadow_summary else None
    if lat_p95 is not None:
        if lat_p95 <= thresholds.max_overlay_latency_p95_ms:
            _add_check("overlay_latency_p95", "pass", lat_p95, thresholds.max_overlay_latency_p95_ms, "ok")
        else:
            _add_check("overlay_latency_p95", "fail", lat_p95, thresholds.max_overlay_latency_p95_ms, f"latency {lat_p95:.1f}ms above threshold")
    else:
        _add_check("overlay_latency_p95", "warn", None, thresholds.max_overlay_latency_p95_ms, "shadow summary missing")

    # shadow steps
    steps = shadow_summary.get("steps") if shadow_summary else None
    if steps is not None:
        if steps >= thresholds.min_shadow_steps:
            _add_check("shadow_steps", "pass", steps, thresholds.min_shadow_steps, "ok")
        else:
            _add_check("shadow_steps", "fail", steps, thresholds.min_shadow_steps, f"only {steps} steps, need {thresholds.min_shadow_steps}")
    else:
        _add_check("shadow_steps", "warn", None, thresholds.min_shadow_steps, "shadow summary missing")

    # match rate
    match_rate = shadow_summary.get("match_rate") if shadow_summary else None
    if match_rate is not None:
        if thresholds.min_match_rate is not None:
            if match_rate >= thresholds.min_match_rate:
                _add_check("match_rate", "pass", match_rate, thresholds.min_match_rate, "ok")
            else:
                _add_check("match_rate", "fail", match_rate, thresholds.min_match_rate, f"match rate {match_rate:.3f} below threshold")
        else:
            _add_check("match_rate", "pass", match_rate, None, "informational only")
    else:
        _add_check("match_rate", "warn", None, thresholds.min_match_rate, "shadow summary missing")

    score = pass_count - fail_count * 2 - warn_count * 0.5

    if fail_count > 0:
        status = "fail"
    elif warn_count > 0:
        status = "warn"
    else:
        status = "pass"

    model_name = None
    candidate_score = None
    if inputs.get("candidate"):
        model_name = inputs["candidate"].get("model_name")
        candidate_score = inputs["candidate"].get("score")
    if not model_name and inputs.get("profile"):
        model_name = inputs["profile"].get("source", {}).get("model_name") or inputs["profile"].get("model_name")
    if candidate_score is None and row:
        candidate_score = row.get("score")

    summary = {
        "model_name": model_name or "unknown",
        "candidate_score": candidate_score if candidate_score is not None else 0.0,
        "shadow_steps": shadow_summary.get("steps") if shadow_summary else None,
        "shadow_match_rate": shadow_summary.get("match_rate") if shadow_summary else None,
        "overlay_latency_p95_ms": shadow_summary.get("overlay_latency_ms_p95") if shadow_summary else None,
        "parity_mismatches": parity if parity is not None else None,
    }

    return {
        "version": "train_v2_acceptance_gate_v1",
        "status": status,
        "score": score,
        "checks": checks,
        "summary": summary,
        "missing": missing,
    }


def format_gate_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("# TrainV2 Candidate Acceptance Gate")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"Status: {result['status'].upper()}")
    lines.append("")
    lines.append("## Summary")
    summary = result["summary"]
    lines.append(f"- Model: {summary.get('model_name', 'unknown')}")
    cs = summary.get("candidate_score")
    lines.append(f"- Candidate score: {cs if cs is not None else 'N/A'}")
    ss = summary.get("shadow_steps")
    lines.append(f"- Shadow steps: {ss if ss is not None else 'N/A'}")
    sm = summary.get("shadow_match_rate")
    lines.append(f"- Shadow match rate: {sm if sm is not None else 'N/A'}")
    lat = summary.get("overlay_latency_p95_ms")
    lines.append(f"- Overlay latency p95: {lat if lat is not None else 'N/A'} ms")
    pm = summary.get("parity_mismatches")
    lines.append(f"- Parity mismatches: {pm if pm is not None else 'N/A'}")
    lines.append("")
    lines.append("## Checks")
    lines.append("| Check | Status | Value | Threshold | Message |")
    lines.append("|---|---|---:|---:|---|")
    for c in result["checks"]:
        val = c["value"]
        val_str = f"{val:.3f}" if isinstance(val, float) else (str(val) if val is not None else "N/A")
        thr = c["threshold"]
        thr_str = f"{thr:.3f}" if isinstance(thr, float) else (str(thr) if thr is not None else "N/A")
        lines.append(f"| {c['name']} | {c['status']} | {val_str} | {thr_str} | {c['message']} |")
    lines.append("")
    lines.append("## Missing Artifacts")
    missing = result.get("missing", [])
    if missing:
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("None")
    lines.append("")
    return "\n".join(lines)


def write_gate_report(
    result: dict,
    output_dir: str,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "acceptance_gate.json"
    md_path = out / "acceptance_gate.md"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path.write_text(format_gate_markdown(result), encoding="utf-8")
    return {
        "result_path": str(json_path.resolve()),
        "markdown_path": str(md_path.resolve()),
        "status": result["status"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="TrainV2 candidate acceptance gate")
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--shadow-pack", default=None)
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-winrate-random", type=float, default=None)
    parser.add_argument("--min-winrate-end-turn", type=float, default=None)
    parser.add_argument("--max-parity-mismatches", type=int, default=None)
    parser.add_argument("--max-overlay-latency-p95-ms", type=float, default=None)
    parser.add_argument("--min-shadow-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    kw: dict[str, Any] = {}
    if args.min_winrate_random is not None:
        kw["min_winrate_random"] = args.min_winrate_random
    if args.min_winrate_end_turn is not None:
        kw["min_winrate_end_turn"] = args.min_winrate_end_turn
    if args.max_parity_mismatches is not None:
        kw["max_parity_mismatches"] = args.max_parity_mismatches
    if args.max_overlay_latency_p95_ms is not None:
        kw["max_overlay_latency_p95_ms"] = args.max_overlay_latency_p95_ms
    if args.min_shadow_steps is not None:
        kw["min_shadow_steps"] = args.min_shadow_steps
    thresholds = GateThresholds(**kw)

    inputs = load_gate_inputs(
        candidate_dir=args.candidate_dir,
        profile_path=args.profile,
        overlay_path=args.overlay,
        shadow_pack_dir=args.shadow_pack,
        leaderboard_path=args.leaderboard,
    )

    result = evaluate_acceptance_gate(inputs, thresholds=thresholds)
    report = write_gate_report(result, args.output_dir)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        pass_count = sum(1 for c in result["checks"] if c["status"] == "pass")
        warn_count = sum(1 for c in result["checks"] if c["status"] == "warn")
        fail_count = sum(1 for c in result["checks"] if c["status"] == "fail")
        print(f"Acceptance: {result['status'].upper()}")
        print(f"Score: {result['score']:.1f}")
        print(f"Checks: {pass_count} pass / {warn_count} warn / {fail_count} fail")
        print(f"Report: {report['markdown_path']}")


if __name__ == "__main__":
    _main()
