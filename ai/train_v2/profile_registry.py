from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    EndTurnPolicy,
    evaluate_berserk_matchup,
    make_train_v2_berserk_brain,
)


PROFILE_SCHEMA_DIMS = {
    "obs_dim": 1456,
    "action_feature_dim": 171,
    "max_candidate_actions": 601,
}


def discover_candidate_profiles(paths: list[str]) -> list[str]:
    found: set[str] = set()

    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            continue

        if p.is_file() and p.name.endswith("candidate_profile.json"):
            found.add(str(p.resolve()))
        elif p.is_dir():
            direct = p / "candidate_profile.json"
            if direct.is_file():
                found.add(str(direct.resolve()))
            for nested in p.rglob("candidate_profile.json"):
                found.add(str(nested.resolve()))

    return sorted(found)


def load_profile_pack(profile_path: str) -> dict:
    p = Path(profile_path)
    pack = json.loads(p.read_text(encoding="utf-8"))
    pack["_profile_path"] = str(p.resolve())

    if "difficulty" not in pack:
        raise ValueError("Missing 'difficulty' field")
    if "profile" not in pack or not isinstance(pack["profile"], dict):
        raise ValueError("Missing or invalid 'profile' field")

    profile = pack["profile"]
    if profile.get("format") != "train_v2_classic_v1":
        raise ValueError(f"Unsupported format: {profile.get('format')}")
    if "model_path" not in profile:
        raise ValueError("Missing profile.model_path")

    for key, expected in PROFILE_SCHEMA_DIMS.items():
        actual = profile.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"Invalid {key}: expected {expected}, got {actual}")

    return pack


def resolve_profile_model_path(profile_pack: dict) -> str:
    profile = profile_pack.get("profile", {})
    model_path = profile.get("model_path", "")
    if not model_path:
        return ""

    p = Path(model_path)
    if p.is_absolute():
        return str(p)

    # Resolve relative to profile file's directory
    profile_path = profile_pack.get("profile_path") or profile_pack.get("_profile_path")
    if profile_path:
        base = Path(profile_path).parent
    else:
        base = Path.cwd()

    return str((base / model_path).resolve())


def build_profile_registry(paths: list[str]) -> dict:
    rows: list[dict] = []
    ok_count = 0
    error_count = 0

    for profile_path in discover_candidate_profiles(paths):
        try:
            pack = load_profile_pack(profile_path)
            onnx_path = resolve_profile_model_path(pack)
            source = pack.get("source", {})
            profile = pack.get("profile", {})

            row = {
                "status": "ok",
                "difficulty": pack.get("difficulty"),
                "model_name": source.get("model_name") or Path(onnx_path).stem,
                "score": source.get("score", -999.0),
                "profile_path": profile_path,
                "onnx_path": onnx_path,
                "onnx_exists": Path(onnx_path).exists() if onnx_path else False,
                "selection": profile.get("selection", "argmax"),
                "temperature_range": profile.get("temperature_range", [1.0, 1.0]),
                "source_run_dir": source.get("source_run_dir"),
                "created_at": pack.get("created_at"),
            }
            ok_count += 1
        except Exception as exc:
            row = {
                "status": "error",
                "profile_path": profile_path,
                "error": str(exc),
            }
            error_count += 1

        rows.append(row)

    def _sort_key(r: dict) -> tuple:
        if r["status"] == "ok":
            score = r.get("score", -999.0)
            name = r.get("model_name", "")
            return (0, -score, name)
        return (1, 0.0, "")

    rows.sort(key=_sort_key)

    best = None
    for r in rows:
        if r["status"] == "ok":
            best = r
            break

    return {
        "version": "train_v2_profile_registry_v1",
        "profiles": len(rows),
        "ok": ok_count,
        "errors": error_count,
        "rows": rows,
        "best": best,
    }


def select_profile(
    registry: dict,
    *,
    selector: str = "best",
    require_onnx: bool = True,
) -> dict | None:
    rows = registry.get("rows", [])

    if selector == "best":
        candidates = [r for r in rows if r["status"] == "ok"]
    else:
        candidates = []
        for r in rows:
            if r["status"] != "ok":
                continue
            if r.get("model_name") == selector:
                candidates.append(r)
            elif r.get("difficulty") == selector:
                candidates.append(r)
            elif r.get("profile_path") == selector:
                candidates.append(r)

    if require_onnx:
        candidates = [r for r in candidates if r.get("onnx_exists")]

    return candidates[0] if candidates else None


def write_profile_overlay(
    profile_pack: dict,
    output_path: str,
    *,
    difficulty: str | None = None,
    relative_to: str | None = None,
) -> dict:
    profile = dict(profile_pack.get("profile", {}))
    overlay_difficulty = difficulty if difficulty is not None else profile_pack.get("difficulty", "train_v2_candidate")

    if relative_to is not None and profile.get("model_path"):
        try:
            profile["model_path"] = os.path.relpath(profile["model_path"], relative_to)
        except ValueError:
            pass

    profile["difficulty"] = overlay_difficulty

    overlay = {
        "version": "train_v2_profile_overlay_v1",
        "created_at": profile_pack.get("created_at"),
        "source_profile_path": profile_pack.get("_profile_path") or profile_pack.get("profile_path"),
        "profiles": {
            overlay_difficulty: profile,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overlay, indent=2, ensure_ascii=False), encoding="utf-8")
    overlay["overlay_path"] = str(out.resolve())
    return overlay


def load_profile_overlay(path: str) -> dict:
    p = Path(path)
    overlay = json.loads(p.read_text(encoding="utf-8"))

    if overlay.get("version") != "train_v2_profile_overlay_v1":
        raise ValueError(f"Invalid overlay version: {overlay.get('version')}")
    if "profiles" not in overlay or not isinstance(overlay["profiles"], dict):
        raise ValueError("Missing or invalid 'profiles' field")

    for difficulty, profile in overlay["profiles"].items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile for {difficulty} is not a dict")
        if profile.get("format") != "train_v2_classic_v1":
            raise ValueError(f"Unsupported format in {difficulty}: {profile.get('format')}")
        for key, expected in PROFILE_SCHEMA_DIMS.items():
            actual = profile.get(key)
            if actual is not None and actual != expected:
                raise ValueError(f"Invalid {key} in {difficulty}: expected {expected}, got {actual}")

    return overlay


def validate_profile_overlay(
    overlay_path: str,
    *,
    seed: int = 42,
    games: int = 1,
    max_steps: int = 80,
) -> dict:
    overlay = load_profile_overlay(overlay_path)

    results: list[dict] = []
    all_ok = True

    for difficulty, profile in overlay["profiles"].items():
        model_path = profile.get("model_path", "")
        if not model_path:
            results.append({"ok": False, "difficulty": difficulty, "error": "model_path missing"})
            all_ok = False
            continue

        if not Path(model_path).is_absolute():
            base = Path(overlay_path).parent
            model_path = str(base / model_path)

        if not Path(model_path).exists():
            results.append({"ok": False, "difficulty": difficulty, "error": f"model_path not found: {model_path}"})
            all_ok = False
            continue

        try:
            brain = make_train_v2_berserk_brain(
                model_path,
                selection=profile.get("selection", "argmax"),
                temperature=tuple(profile.get("temperature_range", [1.0, 1.0])),
            )
            berserk_pol = BerserkBrainPolicy(brain, difficulty="test")
            opp = EndTurnPolicy()
            seeds = list(range(seed, seed + games))
            eval_result = evaluate_berserk_matchup(
                berserk_pol, opp, seeds=seeds, swap_sides=True, max_steps=max_steps,
            )
            results.append({
                "ok": True,
                "difficulty": difficulty,
                "winrate": eval_result["p1_winrate"],
                "brain_invalid_actions": eval_result["p1_brain_invalid_actions"],
                "error": None,
            })
        except Exception as exc:
            results.append({"ok": False, "difficulty": difficulty, "error": str(exc)})
            all_ok = False

    return {
        "ok": all_ok,
        "profiles": results,
    }


def _print_registry_table(registry: dict) -> None:
    ok = registry["ok"]
    errors = registry["errors"]
    rows = registry.get("rows", [])
    print(f"profiles: {len(rows)} ok={ok} errors={errors}")
    if not rows:
        return

    header = f"{'rank':>5} {'score':>7} {'exists':>6} {'selection':>10} {'difficulty':>20} {'model':>20}"
    print(header)
    print("-" * len(header))

    rank = 0
    for row in rows:
        if row["status"] == "ok":
            rank += 1
            exists = "yes" if row.get("onnx_exists") else "no"
            print(
                f"{rank:5d} "
                f"{row['score']:7.3f} "
                f"{exists:>6} "
                f"{row['selection']:>10} "
                f"{row['difficulty']:>20} "
                f"{row['model_name']:>20}"
            )
        else:
            print(f"{'ERR':>5} {'-':>7} {'-':>6} {'-':>10} {'-':>20} {row.get('profile_path', ''):>20}")


def _main():
    parser = argparse.ArgumentParser(description="Build TrainV2 profile registry and overlay")
    parser.add_argument("--paths", nargs="+", required=True, help="Paths to search for candidate_profile.json")
    parser.add_argument("--output-registry", default=None, help="Path to save registry JSON")
    parser.add_argument("--select", default="best", help="Selector: best, model_name, difficulty, or profile_path")
    parser.add_argument("--write-overlay", default=None, help="Path to write overlay JSON")
    parser.add_argument("--difficulty", default=None, help="Overlay difficulty override")
    parser.add_argument("--relative-to", default=None, help="Make model_path relative to this root")
    parser.add_argument("--validate", action="store_true", help="Validate overlay after writing")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of table")
    args = parser.parse_args()

    registry = build_profile_registry(args.paths)

    if args.output_registry:
        p = Path(args.output_registry)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(registry, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Registry saved to {args.output_registry}")

    selected = select_profile(registry, selector=args.select)
    overlay = None
    validation = None

    if selected and args.write_overlay:
        # Load full pack for the selected row
        pack = load_profile_pack(selected["profile_path"])
        overlay = write_profile_overlay(
            pack,
            args.write_overlay,
            difficulty=args.difficulty,
            relative_to=args.relative_to,
        )
        print(f"Overlay written to {overlay['overlay_path']}")

    if overlay and args.validate:
        validation = validate_profile_overlay(overlay["overlay_path"])
        print(f"Validation ok={validation['ok']}")
        for r in validation["profiles"]:
            if r["ok"]:
                print(f"  {r['difficulty']}: winrate={r['winrate']:.3f} invalid={r['brain_invalid_actions']}")
            else:
                print(f"  {r['difficulty']}: error={r['error']}")

    if args.json:
        output = {"registry": registry}
        if overlay:
            output["overlay"] = {k: v for k, v in overlay.items() if k != "overlay_path"}
            output["overlay_path"] = overlay["overlay_path"]
        if validation:
            output["validation"] = validation
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        _print_registry_table(registry)


if __name__ == "__main__":
    _main()
