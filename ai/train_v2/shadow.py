from __future__ import annotations

import argparse
import json
import random as rand_mod
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ai.bot_brain import BerserkInference
from ai.train_v2.classic_actions_v1 import decode_action
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, EndTurnPolicy, GreedyFacePolicy
from ai.train_v2.profile_registry import load_profile_overlay
from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    make_train_v2_berserk_brain,
    _percentile,
)


def describe_stable_action(state, player_id: int, action_id: int) -> dict:
    decoded = decode_action(state, player_id, action_id)
    if decoded is None:
        return {"action_id": action_id, "type": "invalid"}
    d = decoded.to_dict()
    return {
        "action_id": action_id,
        "type": d.get("type"),
        "hand_index": d.get("hand_index"),
        "attacker_id": d.get("attacker_id"),
        "target_id": d.get("target_id"),
        "target_is_hero": d.get("target_is_hero"),
    }


def actions_semantically_equal(state, player_id: int, a: int, b: int) -> bool:
    da = describe_stable_action(state, player_id, a)
    db = describe_stable_action(state, player_id, b)
    if da["type"] == "invalid" and db["type"] == "invalid":
        return a == b
    if da["type"] != db["type"]:
        return False
    if da["type"] == "end_turn":
        return True
    if da["type"] == "play_card":
        return (
            da.get("hand_index") == db.get("hand_index")
            and da.get("target_id") == db.get("target_id")
        )
    if da["type"] == "attack":
        return (
            da.get("attacker_id") == db.get("attacker_id")
            and da.get("target_id") == db.get("target_id")
            and da.get("target_is_hero") == db.get("target_is_hero")
        )
    return False


class OverlayBerserkPolicy:
    def __init__(self, overlay_path: str, *, difficulty: str | None = None):
        overlay = load_profile_overlay(overlay_path)
        profiles = overlay.get("profiles", {})

        if difficulty is not None:
            if difficulty not in profiles:
                raise ValueError(f"Difficulty {difficulty} not found in overlay")
            self._difficulty = difficulty
        else:
            if len(profiles) != 1:
                raise ValueError("Multiple profiles in overlay; specify difficulty")
            self._difficulty = list(profiles.keys())[0]

        profile = profiles[self._difficulty]
        model_path = profile.get("model_path", "")
        if not Path(model_path).is_absolute():
            base = Path(overlay_path).parent
            model_path = str((base / model_path).resolve())

        brain = make_train_v2_berserk_brain(
            model_path,
            selection=profile.get("selection", "argmax"),
            temperature=tuple(profile.get("temperature_range", [1.0, 1.0])),
        )
        self._policy = BerserkBrainPolicy(brain, difficulty="test")
        self.name = f"overlay_{self._difficulty}"

    def reset(self, seed: int) -> None:
        rand_mod.seed(seed)
        np.random.seed(seed)
        if hasattr(self._policy, "reset"):
            self._policy.reset(seed)

    def select_action(self, env: ClassicRLEnv, player_id: int) -> int:
        return self._policy.select_action(env, player_id)


class LegacyBerserkPolicy:
    def __init__(self, brain: BerserkInference | FakeLegacyBrain, *, difficulty: str):
        self._brain = brain
        self._difficulty = difficulty
        self.name = f"legacy_{difficulty}"
        self.latencies_ms: list[float] = []
        self.invalid_actions = 0

    def reset(self, seed: int) -> None:
        rand_mod.seed(seed)
        np.random.seed(seed)
        self.latencies_ms.clear()
        self.invalid_actions = 0

    def select_legal_action_index(self, env: ClassicRLEnv, player_id: int) -> int:
        state = env.clone_state()
        legal = env._env.get_legal_actions(player_id)

        t0 = time.perf_counter()
        legal_idx = self._brain.get_action(
            state, player_id, legal, difficulty=self._difficulty
        )
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if legal_idx < 0 or legal_idx >= len(legal):
            self.invalid_actions += 1
            return 0
        return legal_idx

    def select_action(self, env: ClassicRLEnv, player_id: int) -> int:
        state = env.clone_state()
        legal = env._env.get_legal_actions(player_id)

        legal_idx = self.select_legal_action_index(env, player_id)
        if legal_idx < 0 or legal_idx >= len(legal):
            return 0

        target_action = legal[legal_idx]
        mask = env.action_mask(player_id)

        for aid in range(601):
            if mask[aid] != 1.0:
                continue
            decoded = decode_action(state, player_id, int(aid))
            if decoded is None:
                continue
            if BerserkInference._find_matching_legal_action_index(decoded, [target_action]) == 0:
                return int(aid)

        self.invalid_actions += 1
        return 0


class FakeLegacyBrain:
    """First-legal fallback brain for shadow harness testing without production artifacts."""

    def get_action(self, game_state, player_id, legal_actions, difficulty):
        return 0 if legal_actions else 0


def _get_played_action(env, legacy_aid, overlay_aid, play_policy, rng):
    if play_policy == "legacy":
        return legacy_aid
    if play_policy == "overlay":
        return overlay_aid
    if play_policy == "random":
        cp = env.current_player_id()
        mask = env.action_mask(cp)
        legal_ids = [i for i in range(601) if mask[i] == 1.0]
        return rng.choice(legal_ids) if legal_ids else 0
    if play_policy == "greedy_face":
        p = GreedyFacePolicy()
        return p.select_action(env, env.current_player_id())
    return legacy_aid


def _validate_and_fallback(env, player_id, aid):
    mask = env.action_mask(player_id)
    if 0 <= aid < len(mask) and mask[aid] == 1.0:
        return aid, False
    if mask[0] == 1.0:
        return 0, True
    for i in range(601):
        if mask[i] == 1.0:
            return i, True
    return 0, True


def run_shadow_episode(
    *,
    overlay_policy,
    legacy_policy,
    seed: int,
    max_steps: int = 200,
    play_policy: str = "legacy",
) -> dict:
    env = ClassicRLEnv(seed=seed)
    env.reset(seed=seed)

    rng = rand_mod.Random(seed)
    decisions: list[dict] = []

    overlay_policy.reset(seed)
    legacy_policy.reset(seed)

    terminated = False
    truncated = False

    for step in range(max_steps):
        cp = env.current_player_id()
        snapshot = env.clone_state()

        legacy_aid = legacy_policy.select_action(env, cp)
        overlay_aid = overlay_policy.select_action(env, cp)

        match = actions_semantically_equal(snapshot, cp, legacy_aid, overlay_aid)
        played_aid = _get_played_action(env, legacy_aid, overlay_aid, play_policy, rng)
        played_aid, played_fallback = _validate_and_fallback(env, cp, played_aid)

        _, reward, terminated, truncated, info = env.step(played_aid)

        decisions.append({
            "step": step,
            "player_id": cp,
            "legacy_action_id": legacy_aid,
            "overlay_action_id": overlay_aid,
            "played_action_id": played_aid,
            "match": match,
            "legacy": describe_stable_action(snapshot, cp, legacy_aid),
            "overlay": describe_stable_action(snapshot, cp, overlay_aid),
            "played": describe_stable_action(snapshot, cp, played_aid),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info_invalid_action": bool(info.get("invalid_action", False)),
            "played_fallback": played_fallback,
        })

        if terminated or truncated:
            break

    legacy_lats = legacy_policy.latencies_ms
    overlay_lats = getattr(getattr(overlay_policy, "_policy", None), "latencies_ms", [])
    if overlay_lats is None:
        overlay_lats = []

    played_fallback_count = sum(1 for d in decisions if d["played_fallback"])

    summary = {
        "seed": seed,
        "steps": len(decisions),
        "matches": sum(1 for d in decisions if d["match"]),
        "mismatches": sum(1 for d in decisions if not d["match"]),
        "match_rate": sum(1 for d in decisions if d["match"]) / len(decisions) if decisions else 0.0,
        "terminated": terminated,
        "truncated": truncated,
        "winner_id": env.winner_id(),
        "legacy_invalid_actions": legacy_policy.invalid_actions,
        "overlay_invalid_actions": getattr(getattr(overlay_policy, "_policy", None), "invalid_actions", 0),
        "played_invalid_actions": played_fallback_count,
        "legacy_latency_ms_p50": _percentile(legacy_lats, 50),
        "legacy_latency_ms_p95": _percentile(legacy_lats, 95),
        "overlay_latency_ms_p50": _percentile(overlay_lats, 50),
        "overlay_latency_ms_p95": _percentile(overlay_lats, 95),
    }

    return {"summary": summary, "decisions": decisions}


def run_shadow_matchup(
    overlay_path: str,
    *,
    legacy_brain: BerserkInference | FakeLegacyBrain | None = None,
    legacy_difficulty: str = "easy",
    overlay_difficulty: str | None = None,
    seeds: list[int] | None = None,
    max_steps: int = 200,
    play_policy: str = "legacy",
) -> dict:
    if seeds is None:
        seeds = [42]

    overlay_policy = OverlayBerserkPolicy(overlay_path, difficulty=overlay_difficulty)

    if legacy_brain is None:
        legacy_brain = FakeLegacyBrain()

    legacy_policy = LegacyBerserkPolicy(legacy_brain, difficulty=legacy_difficulty)

    episodes_detail: list[dict] = []
    total_steps = 0
    total_matches = 0
    total_mismatches = 0
    all_legacy_lats: list[float] = []
    all_overlay_lats: list[float] = []
    total_legacy_invalid = 0
    total_overlay_invalid = 0
    total_played_fallback = 0

    for seed in seeds:
        result = run_shadow_episode(
            overlay_policy=overlay_policy,
            legacy_policy=legacy_policy,
            seed=seed,
            max_steps=max_steps,
            play_policy=play_policy,
        )
        episodes_detail.append(result)
        total_steps += result["summary"]["steps"]
        total_matches += result["summary"]["matches"]
        total_mismatches += result["summary"]["mismatches"]
        all_legacy_lats.extend(legacy_policy.latencies_ms)
        all_overlay_lats.extend(getattr(getattr(overlay_policy, "_policy", None), "latencies_ms", []))
        total_legacy_invalid += result["summary"]["legacy_invalid_actions"]
        total_overlay_invalid += result["summary"]["overlay_invalid_actions"]
        total_played_fallback += result["summary"]["played_invalid_actions"]

    n_decisions = total_matches + total_mismatches
    return {
        "episodes": len(seeds),
        "steps": total_steps,
        "matches": total_matches,
        "mismatches": total_mismatches,
        "match_rate": total_matches / n_decisions if n_decisions > 0 else 0.0,
        "legacy_invalid_actions": total_legacy_invalid,
        "overlay_invalid_actions": total_overlay_invalid,
        "played_invalid_actions": total_played_fallback,
        "legacy_latency_ms_p50": _percentile(all_legacy_lats, 50),
        "legacy_latency_ms_p95": _percentile(all_legacy_lats, 95),
        "overlay_latency_ms_p50": _percentile(all_overlay_lats, 50),
        "overlay_latency_ms_p95": _percentile(all_overlay_lats, 95),
        "episodes_detail": episodes_detail,
    }


def _main():
    parser = argparse.ArgumentParser(description="Shadow decision runner: legacy vs TrainV2 overlay")
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--overlay-difficulty", default=None)
    parser.add_argument("--legacy-profile-json", default=None)
    parser.add_argument("--legacy-difficulty", default="easy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--games", type=int, default=None, help="Alternative to --seeds: generate seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed when using --games")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--play-policy", default="legacy", choices=["legacy", "overlay", "random", "greedy_face"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds
    if args.games is not None:
        seeds = list(range(args.seed, args.seed + args.games))

    legacy_brain = None
    if args.legacy_profile_json:
        with open(args.legacy_profile_json, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        legacy_brain = BerserkInference(profiles=profiles)
    else:
        print("No legacy profile provided; using first-legal fake legacy policy", file=sys.stderr)

    result = run_shadow_matchup(
        args.overlay,
        legacy_brain=legacy_brain,
        legacy_difficulty=args.legacy_difficulty,
        overlay_difficulty=args.overlay_difficulty,
        seeds=seeds,
        max_steps=args.max_steps,
        play_policy=args.play_policy,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Output: {args.output}")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        summary = result
        print(f"Shadow episodes: {summary['episodes']} | steps: {summary['steps']}")
        print(
            f"Match rate: {summary['match_rate']:.3f} "
            f"({summary['matches']} / {summary['matches'] + summary['mismatches']})"
        )
        print(
            f"Legacy latency p50/p95: {summary['legacy_latency_ms_p50']:.2f} / "
            f"{summary['legacy_latency_ms_p95']:.2f} ms"
        )
        print(
            f"Overlay latency p50/p95: {summary['overlay_latency_ms_p50']:.2f} / "
            f"{summary['overlay_latency_ms_p95']:.2f} ms"
        )
        print(
            f"Invalids: legacy={summary['legacy_invalid_actions']} "
            f"overlay={summary['overlay_invalid_actions']} "
            f"played_fallback={summary['played_invalid_actions']}"
        )


if __name__ == "__main__":
    _main()
