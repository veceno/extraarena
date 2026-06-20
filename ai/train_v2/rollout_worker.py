"""
Rollout worker process for TrainV2 PPO.

Runs in a separate process (spawn). Owns a ClassicRLEnv.
Receives action ids via Pipe, steps the env, returns observations/masks/features.
Does NOT import MLX or train_ppo.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import signal
from typing import Any, Dict

import numpy as np

# Prevent accidental MLX import in worker
# We can't patch builtins globally safely, so we rely on discipline.
# The key is: worker must not import train_ppo or model_mlx.
try:
    import builtins as _builtins_mod
except ImportError:
    import __builtin__ as _builtins_mod

_original_import = _builtins_mod.__import__

def _import_guard(name, *args, **kwargs):
    if name in ("mlx", "mlx.core", "mlx.nn", "mlx.optimizers"):
        raise ImportError(f"Import of '{name}' is forbidden in rollout worker")
    return _original_import(name, *args, **kwargs)

# Replace import only inside this module context
# We can't patch builtins globally safely, so we rely on discipline.
# The key is: worker must not import train_ppo or model_mlx.

from core.state import GameStatus
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, EndTurnPolicy, GreedyFacePolicy


HISTORICAL_TRAIN_V2_ONNX = {
    "trainv2_0251": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/best_update_0251.onnx",
    "trainv2_0348": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0348.onnx",
    "trainv2_0156": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0156.onnx",
    "trainv2_0408": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0408.onnx",
    "trainv2_0700": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/best_update_0700.onnx",
    "trainv2_0800": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/update_0800.onnx",
    "v4_micro": "ai/models/extra-lr-v4-micro.onnx",
    "v4_lite": "ai/models/extra-lr-v4-lite.onnx",
    "v4_opti": "ai/models/extra-lr-v4-opti.onnx",
    "v4_max": "ai/models/extra-lr-v4-max.onnx",
}


def _make_numpy_array(obj: Any) -> np.ndarray | None:
    """Convert whatever came through pipe back to numpy array."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "__array__"):
        return np.array(obj)
    return np.array(obj)


def _state_payload(env: ClassicRLEnv, *, include_preview: bool, af_dtype) -> dict:
    cp = env.current_player_id()
    return {
        "player_id": cp,
        "obs": env.observe(cp),
        "mask": env.action_mask(cp),
        "af": env.action_features(cp, include_preview=include_preview).astype(af_dtype),
    }


def _all_card_levels(env: ClassicRLEnv, level: int) -> dict[int, int]:
    level = max(1, min(10, int(level)))
    return {int(card_id): level for card_id in env._cards_data.keys()}


def _episode_level_overrides(
    env: ClassicRLEnv,
    *,
    rng: np.random.Generator,
    learner_player_id: int,
    rate: float,
    learner_level: int,
    opponent_level: int,
) -> tuple[dict[int, int] | None, dict[int, int] | None, bool]:
    if learner_player_id not in (1, 2):
        return None, None, False
    if rate <= 0.0 or rng.random() >= rate:
        return None, None, False

    learner_levels = _all_card_levels(env, learner_level)
    opponent_levels = _all_card_levels(env, opponent_level)
    if learner_player_id == 1:
        return learner_levels, opponent_levels, True
    return opponent_levels, learner_levels, True


def _parse_focus_scenarios(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"focus_scenarios_json must be valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("focus_scenarios_json must be a JSON list")

    scenarios: list[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"focus scenario #{idx} must be an object")
        deck = item.get("deck")
        if not isinstance(deck, list) or not deck:
            raise ValueError(f"focus scenario #{idx} must include non-empty deck")
        scenario = dict(item)
        scenario["key"] = str(scenario.get("key") or f"focus_{idx}")
        scenario["deck"] = [int(card_id) for card_id in deck]
        if "level" in scenario:
            scenario["level"] = max(1, min(10, int(scenario["level"])))
        scenarios.append(scenario)
    return scenarios


def _choose_focus_scenario(
    scenarios: list[dict],
    *,
    rng: np.random.Generator,
    rate: float,
) -> dict | None:
    if not scenarios or rate <= 0.0 or rng.random() >= rate:
        return None
    idx = int(rng.integers(0, len(scenarios)))
    return scenarios[idx]


def _focus_levels(
    env: ClassicRLEnv,
    scenario: dict | None,
) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    if not scenario:
        return None, None
    level = scenario.get("level")
    p1_level = scenario.get("p1_level", level)
    p2_level = scenario.get("p2_level", level)
    p1_levels = _all_card_levels(env, int(p1_level)) if p1_level is not None else None
    p2_levels = _all_card_levels(env, int(p2_level)) if p2_level is not None else None
    return p1_levels, p2_levels


def _summary_payload(
    env: ClassicRLEnv,
    *,
    step_count: int,
    truncated: bool,
    invalid_actions: int,
    seed: int,
    starting_player_id: int,
) -> dict:
    st = env._env.state
    return {
        "winner_id": env.winner_id(),
        "status": st.status.value,
        "turns": st.turn_number,
        "steps": step_count,
        "truncated": bool(truncated),
        "p1_hp": st.p1.hero.hp,
        "p2_hp": st.p2.hero.hp,
        "p1_reward": float(env._p1_reward),
        "p2_reward": float(env._p2_reward),
        "invalid_actions": invalid_actions,
        "seed": seed,
        "starting_player_id": starting_player_id,
    }


def _make_legacy_policy(kind: str):
    raise ValueError(f"legacy opponents are unsupported in the v4 bot pipeline: {kind}")


def _make_train_v2_onnx_policy(kind: str):
    from ai.train_v2.berserk_eval import make_train_v2_berserk_brain

    onnx_path = HISTORICAL_TRAIN_V2_ONNX.get(kind)
    if onnx_path is None:
        raise ValueError(f"unknown TrainV2 historical opponent kind: {kind}")
    brain = make_train_v2_berserk_brain(onnx_path, selection="argmax")

    class TrainV2LegalPolicy:
        name = f"historical_{kind}"

        def reset(self, seed: int):
            np.random.seed(seed)

        def select_core_action(self, env: ClassicRLEnv, player_id: int):
            state = env.clone_state()
            legal = env._env.get_legal_actions(player_id)
            if not legal:
                return None
            legal_idx = brain.get_action(state, player_id, legal, difficulty="test")
            if legal_idx < 0 or legal_idx >= len(legal):
                legal_idx = 0
            return legal[legal_idx]

    return TrainV2LegalPolicy()


def _get_opponent_policy(kind: str, cache: dict):
    if kind in cache:
        return cache[kind]
    if kind == "random":
        policy = RandomLegalPolicy()
    elif kind == "end_turn":
        policy = EndTurnPolicy()
    elif kind == "greedy_face":
        policy = GreedyFacePolicy()
    elif kind in ("legacy_max", "legacy_medium", "legacy_random_biggest"):
        policy = _make_legacy_policy(kind)
    elif kind in HISTORICAL_TRAIN_V2_ONNX:
        policy = _make_train_v2_onnx_policy(kind)
    else:
        raise ValueError(f"unknown opponent kind: {kind}")
    cache[kind] = policy
    return policy


def _auto_play_until_learner(
    env: ClassicRLEnv,
    *,
    learner_player_id: int,
    opponent_policy,
    step_count: int,
    max_steps: int,
) -> tuple[int, float, bool, bool]:
    """Play fixed-opponent turns locally until learner is to act or the episode ends."""
    opponent_reward = 0.0
    terminated = env._env.state.status != GameStatus.ONGOING
    truncated = step_count >= max_steps

    while not terminated and not truncated and env.current_player_id() != learner_player_id:
        cp = env.current_player_id()
        if hasattr(opponent_policy, "select_core_action"):
            action = opponent_policy.select_core_action(env, cp)
            if action is None:
                break
            _, reward, terminated, step_truncated, _ = env.step_core_action(action)
        else:
            action_id = opponent_policy.select_action(env, cp)
            _, reward, terminated, step_truncated, _ = env.step(action_id)
        opponent_reward += float(reward)
        step_count += 1
        truncated = bool(step_truncated or step_count >= max_steps)
        terminated = bool(terminated or env._env.state.status != GameStatus.ONGOING)

    return step_count, opponent_reward, terminated, truncated


def _worker_loop(
    config_dict: dict,
    conn: mp.connection.Connection,
    worker_id: int,
):
    """Main worker event loop."""
    try:
        env = ClassicRLEnv(
            seed=config_dict.get("seed", 42),
            verify_mask=config_dict.get("verify_mask", True),
            placement_mode=config_dict.get("placement_mode", "full"),
            include_legal_actions_in_info=False,
        )
    except Exception as exc:
        conn.send({"error": f"env_init_failed: {exc}", "worker_id": worker_id})
        return

    episode_id: int | None = None
    step_count = 0
    p1_reward = 0.0
    p2_reward = 0.0
    current_seed = 0
    current_starting_player_id = 1
    current_max_steps = int(config_dict.get("max_steps_per_episode", 500))
    current_level_handicap = False
    invalid_actions = 0
    opponent_policy_cache: dict = {}
    af_dtype_str = config_dict.get("action_features_dtype", "float32")
    af_dtype = np.float16 if af_dtype_str == "float16" else np.float32
    level_handicap_rate = float(config_dict.get("level_handicap_rate", 0.0) or 0.0)
    learner_level = int(config_dict.get("learner_level", 1) or 1)
    opponent_level = int(config_dict.get("opponent_level", 1) or 1)
    focus_deck_rate = float(config_dict.get("focus_deck_rate", 0.0) or 0.0)
    try:
        focus_scenarios = _parse_focus_scenarios(config_dict.get("focus_scenarios_json"))
    except Exception as exc:
        conn.send({"error": f"focus_scenarios_invalid: {exc}", "worker_id": worker_id})
        return
    current_focus_scenario: dict | None = None

    # Pre-compute include_preview flag
    include_preview = config_dict.get("include_preview_features", False)

    while True:
        try:
            if not conn.poll(timeout=300.0):
                # 5 minute timeout — something is wrong
                conn.send({"error": "poll_timeout", "worker_id": worker_id})
                break
            msg = conn.recv()
        except (EOFError, ConnectionResetError):
            break

        if not isinstance(msg, dict):
            conn.send({"error": "bad_msg_format", "worker_id": worker_id})
            continue

        cmd = msg.get("cmd")

        if cmd == "shutdown":
            break

        if cmd == "reset":
            seed = msg.get("seed", 0)
            episode_id = msg.get("episode_id", 0)
            opponent_kind = msg.get("opponent_kind", "self")
            learner_player_id = int(msg.get("learner_player_id", 1))
            starting_player_id = int(msg.get("starting_player_id", 1))
            current_max_steps = int(msg.get("max_steps", config_dict.get("max_steps_per_episode", 500)))
            step_count = 0
            invalid_actions = 0
            current_seed = seed
            current_starting_player_id = starting_player_id
            try:
                level_rng = np.random.default_rng(seed + worker_id * 7919)
                focus_rng = np.random.default_rng(seed * 104729 + worker_id * 7919)
                focus_scenario = _choose_focus_scenario(
                    focus_scenarios,
                    rng=focus_rng,
                    rate=focus_deck_rate,
                )
                current_focus_scenario = focus_scenario
                focus_p1_levels, focus_p2_levels = _focus_levels(env, focus_scenario)
                p1_levels, p2_levels, used_level_handicap = _episode_level_overrides(
                    env,
                    rng=level_rng,
                    learner_player_id=learner_player_id,
                    rate=level_handicap_rate,
                    learner_level=learner_level,
                    opponent_level=opponent_level,
                )
                if focus_scenario is not None:
                    p1_levels, p2_levels = focus_p1_levels, focus_p2_levels
                    used_level_handicap = False
                current_level_handicap = used_level_handicap
                env.reset(
                    seed=seed,
                    p1_deck_ids=list(focus_scenario["deck"]) if focus_scenario else None,
                    p2_deck_ids=list(focus_scenario["deck"]) if focus_scenario else None,
                    p1_levels=p1_levels,
                    p2_levels=p2_levels,
                    starting_player_id=starting_player_id,
                )
                if opponent_kind != "self":
                    opponent_policy = _get_opponent_policy(opponent_kind, opponent_policy_cache)
                    if hasattr(opponent_policy, "reset"):
                        opponent_policy.reset(seed + worker_id * 1009)
                    step_count, _, terminated, truncated = _auto_play_until_learner(
                        env,
                        learner_player_id=learner_player_id,
                        opponent_policy=opponent_policy,
                        step_count=step_count,
                        max_steps=current_max_steps,
                    )
                    if terminated or truncated:
                        conn.send({
                            "type": "step_result",
                            "worker_id": worker_id,
                            "episode_id": episode_id,
                            "seed": current_seed,
                            "step": step_count,
                            "reward": 0.0,
                            "terminated": terminated,
                            "truncated": truncated,
                            "next_obs": env.observe(env.current_player_id()),
                            "info": {},
                            "summary": _summary_payload(
                                env,
                                step_count=step_count,
                                truncated=bool(truncated),
                                invalid_actions=invalid_actions,
                                seed=current_seed,
                                starting_player_id=current_starting_player_id,
                            ) | {
                                "opponent_kind": opponent_kind,
                                "learner_player_id": learner_player_id,
                                "level_handicap": used_level_handicap,
                                "learner_level": learner_level if used_level_handicap else None,
                                "opponent_level": opponent_level if used_level_handicap else None,
                                "focus_scenario": focus_scenario.get("key") if focus_scenario else None,
                                "focus_level": focus_scenario.get("level") if focus_scenario else None,
                            },
                            "next_state": None,
                        })
                        continue
                payload = _state_payload(env, include_preview=include_preview, af_dtype=af_dtype)
                conn.send({
                    "type": "state",
                    "worker_id": worker_id,
                    "episode_id": episode_id,
                    "seed": current_seed,
                    "step": step_count,
                    "opponent_kind": opponent_kind,
                    "learner_player_id": learner_player_id,
                    "starting_player_id": current_starting_player_id,
                    "level_handicap": used_level_handicap,
                    "focus_scenario": focus_scenario.get("key") if focus_scenario else None,
                    "focus_level": focus_scenario.get("level") if focus_scenario else None,
                    **payload,
                })
            except Exception as exc:
                conn.send({
                    "error": f"reset_failed: {exc}",
                    "episode_id": episode_id,
                    "worker_id": worker_id,
                })
            continue

        if cmd == "step":
            action_id = msg.get("action_id", 0)
            opponent_kind = msg.get("opponent_kind", "self")
            learner_player_id = int(msg.get("learner_player_id", 1))
            try:
                next_obs, reward, terminated, truncated, info = env.step(action_id)
                step_count += 1
                if info.get("invalid_action"):
                    invalid_actions += 1

                if not terminated and step_count >= current_max_steps:
                    truncated = True

                opponent_reward = 0.0
                if opponent_kind != "self" and not terminated and not truncated:
                    opponent_policy = _get_opponent_policy(opponent_kind, opponent_policy_cache)
                    step_count, opponent_reward, terminated, truncated = _auto_play_until_learner(
                        env,
                        learner_player_id=learner_player_id,
                        opponent_policy=opponent_policy,
                        step_count=step_count,
                        max_steps=current_max_steps,
                    )

                done = bool(terminated or truncated or env._env.state.status != GameStatus.ONGOING)
                summary = None
                next_state = None
                if done:
                    summary = _summary_payload(
                        env,
                        step_count=step_count,
                        truncated=bool(truncated),
                        invalid_actions=invalid_actions,
                        seed=current_seed,
                        starting_player_id=current_starting_player_id,
                    )
                    summary["opponent_kind"] = opponent_kind
                    summary["learner_player_id"] = learner_player_id
                    summary["starting_player_id"] = current_starting_player_id
                    summary["level_handicap"] = current_level_handicap
                    summary["learner_level"] = learner_level if current_level_handicap else None
                    summary["opponent_level"] = opponent_level if current_level_handicap else None
                    summary["focus_scenario"] = current_focus_scenario.get("key") if current_focus_scenario else None
                    summary["focus_level"] = current_focus_scenario.get("level") if current_focus_scenario else None
                else:
                    next_state = _state_payload(env, include_preview=include_preview, af_dtype=af_dtype)

                conn.send({
                    "type": "step_result",
                    "worker_id": worker_id,
                    "episode_id": episode_id,
                    "seed": current_seed,
                    "step": step_count,
                    "reward": float(reward) - float(opponent_reward),
                    "terminated": terminated,
                    "truncated": truncated,
                    "next_obs": next_obs,
                    "info": info,
                    "summary": summary,
                    "next_state": next_state,
                })
            except Exception as exc:
                conn.send({
                    "error": f"step_failed: {exc}",
                    "episode_id": episode_id,
                    "worker_id": worker_id,
                })
            continue

        # Unknown command
        conn.send({"error": f"unknown_cmd: {cmd}", "worker_id": worker_id})


def _worker_entrypoint(config_dict: dict, conn: mp.connection.Connection, worker_id: int):
    """Entry point for multiprocessing spawn."""
    # Disable signal handlers inherited from parent
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _worker_loop(config_dict, conn, worker_id)
    except Exception as exc:
        try:
            conn.send({"error": f"worker_crash: {exc}", "worker_id": worker_id})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def spawn_worker(
    config_dict: dict,
    worker_id: int,
    ctx=None,
) -> tuple[mp.Process, mp.connection.Connection, mp.connection.Connection]:
    """Spawn a single worker process and return (process, parent_conn, child_conn)."""
    if ctx is None:
        ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(
        target=_worker_entrypoint,
        args=(config_dict, child_conn, worker_id),
        name=f"rollout_worker_{worker_id}",
    )
    process.start()
    child_conn.close()
    return process, parent_conn, child_conn
