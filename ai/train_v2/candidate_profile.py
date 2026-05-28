from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    EndTurnPolicy,
    compare_berserk_to_onnx_policy,
    evaluate_berserk_matchup,
    make_train_v2_berserk_brain,
)


def load_candidate(candidate_path: str) -> dict:
    p = Path(candidate_path)
    if p.is_dir():
        candidate_json = p / "candidate.json"
    else:
        candidate_json = p

    if not candidate_json.is_file():
        raise FileNotFoundError(f"candidate.json not found at {candidate_path}")

    return json.loads(candidate_json.read_text(encoding="utf-8"))


def build_train_v2_profile(
    candidate: dict,
    *,
    difficulty: str = "train_v2_candidate",
    selection: str = "argmax",
    temperature: tuple[float, float] = (1.0, 1.0),
    relative_to: str | None = None,
) -> dict:
    model_path = candidate.get("candidate_onnx") or candidate.get("source_onnx")
    if not model_path:
        raise ValueError("Candidate missing candidate_onnx and source_onnx")

    if relative_to is not None:
        try:
            model_path = os.path.relpath(model_path, relative_to)
        except ValueError:
            pass

    return {
        "difficulty": difficulty,
        "profile": {
            "model_path": model_path,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": list(temperature),
            "selection": selection,
        },
        "source": candidate,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Opt-in TrainV2 profile artifact. Production configs are not modified automatically.",
    }


def write_candidate_profile(
    candidate_path: str,
    output_path: str | None = None,
    *,
    difficulty: str = "train_v2_candidate",
    selection: str = "argmax",
    temperature: tuple[float, float] = (1.0, 1.0),
    relative_to: str | None = None,
) -> dict:
    candidate = load_candidate(candidate_path)
    pack = build_train_v2_profile(
        candidate,
        difficulty=difficulty,
        selection=selection,
        temperature=temperature,
        relative_to=relative_to,
    )

    if output_path is None:
        p = Path(candidate_path)
        if p.is_dir():
            out = p / "candidate_profile.json"
        else:
            out = p.parent / "candidate_profile.json"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    pack["profile_path"] = str(out)
    out.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    return pack


def format_profile_snippet(profile_pack: dict) -> str:
    profile = profile_pack.get("profile", {})
    lines = [
        "# Opt-in TrainV2 profile artifact. Production configs are not modified automatically.",
        "TRAIN_V2_CANDIDATE_PROFILE = {",
    ]
    for key, value in profile.items():
        if isinstance(value, str):
            lines.append(f'    "{key}": "{value}",')
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value)
            lines.append(f'    "{key}": [{items}],')
        else:
            lines.append(f'    "{key}": {value},')
    lines.append("}")
    return "\n".join(lines)


def validate_candidate_profile(
    profile_pack: dict,
    *,
    seed: int = 42,
    games: int = 2,
    max_steps: int = 100,
) -> dict:
    profile = profile_pack.get("profile", {})
    model_path = profile.get("model_path")
    selection = profile.get("selection", "argmax")

    if not model_path:
        return {"ok": False, "parity": None, "eval": None, "error": "model_path missing"}

    # Resolve relative model_path
    if not Path(model_path).is_absolute():
        profile_path = profile_pack.get("profile_path")
        if profile_path:
            base = Path(profile_path).parent
        else:
            base = Path.cwd()
        model_path = str(base / model_path)

    if not Path(model_path).exists():
        return {"ok": False, "parity": None, "eval": None, "error": f"model_path not found: {model_path}"}

    try:
        brain = make_train_v2_berserk_brain(
            model_path,
            selection=selection,
            temperature=tuple(profile.get("temperature_range", [1.0, 1.0])),
        )
        berserk_pol = BerserkBrainPolicy(brain, difficulty="test")

        parity = compare_berserk_to_onnx_policy(
            model_path, seed=seed, steps=min(max_steps, 40), selection=selection,
        )

        opp = EndTurnPolicy()
        seeds = list(range(seed, seed + games))
        eval_result = evaluate_berserk_matchup(
            berserk_pol, opp, seeds=seeds, swap_sides=True, max_steps=max_steps,
        )

        return {
            "ok": True,
            "parity": parity,
            "eval": {
                "games": eval_result["games"],
                "winrate": eval_result["p1_winrate"],
                "invalid_actions": eval_result["invalid_actions"],
                "truncations": eval_result["truncations"],
                "avg_steps": eval_result["avg_steps"],
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "parity": None,
            "eval": None,
            "error": str(exc),
        }


def _main():
    parser = argparse.ArgumentParser(description="Build TrainV2 candidate profile pack")
    parser.add_argument("--candidate", required=True, help="Path to candidate.json or candidate directory")
    parser.add_argument("--output", default=None, help="Output profile JSON path")
    parser.add_argument("--difficulty", default="train_v2_candidate")
    parser.add_argument("--selection", default="argmax", choices=["argmax", "softmax", "sample"])
    parser.add_argument("--temperature-min", type=float, default=1.0)
    parser.add_argument("--temperature-max", type=float, default=1.0)
    parser.add_argument("--relative-to", default=None)
    parser.add_argument("--snippet", action="store_true", help="Print Python snippet")
    parser.add_argument("--validate", action="store_true", help="Run smoke validation")
    args = parser.parse_args()

    selection = args.selection
    if selection == "sample":
        selection = "softmax"

    pack = write_candidate_profile(
        args.candidate,
        args.output,
        difficulty=args.difficulty,
        selection=selection,
        temperature=(args.temperature_min, args.temperature_max),
        relative_to=args.relative_to,
    )

    print(f"Profile written to: {pack['profile_path']}")

    if args.snippet:
        print("\n" + format_profile_snippet(pack))

    if args.validate:
        result = validate_candidate_profile(pack)
        print(f"\nValidation ok={result['ok']}")
        if result["parity"]:
            print(f"Parity: {result['parity']['mismatches']} mismatches")
        if result["eval"]:
            print(f"Eval winrate: {result['eval']['winrate']:.3f} | games={result['eval']['games']}")
        if result["error"]:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    _main()
