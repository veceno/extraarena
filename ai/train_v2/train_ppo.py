"""
Single-process PPO trainer for TrainV2 action-conditioned policy.
Optimized for Apple M4 Pro with batched inference and action caching.
"""
from __future__ import annotations

import argparse
import json
import random as rand_mod
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.model_mlx import (
    MODEL_VERSION,
    ActionConditionedPolicy,
    masked_logits,
    sample_action,
    save_checkpoint,
    load_checkpoint,
)
from ai.train_v2.policies import RandomLegalPolicy, EndTurnPolicy
from ai.train_v2.rollout_worker import spawn_worker
from core.state import GameStatus

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class PPOConfig:
    total_updates: int = 10
    episodes_per_update: int = 4
    max_steps_per_episode: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float | None = None
    minibatch_size: int = 256
    epochs: int = 3
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    include_action_features: bool = True
    include_preview_features: bool = False
    seed: int = 42
    checkpoint_dir: str = "ai/train_v2/checkpoints"
    metrics_path: str | None = None
    eval_every_updates: int = 0
    eval_games: int = 20
    resume_checkpoint: str | None = None
    start_update: int = 0
    fail_on_non_finite: bool = True
    min_batch_transitions: int = 2
    rollout_workers: int = 1
    verify_mask: bool = True
    placement_mode: str = "append_only"
    profile_actions: bool = False
    action_features_dtype: str = "float32"
    opponent_mix: str = "self:1.0"
    learner_side: str = "random"
    starting_player: str = "random"
    level_handicap_rate: float = 0.0
    learner_level: int = 1
    opponent_level: int = 1


TRAIN_PRESETS = {
    "smoke": {
        "total_updates": 1,
        "episodes_per_update": 1,
        "max_steps_per_episode": 10,
        "hidden_dim": 32,
        "action_hidden_dim": 16,
        "minibatch_size": 8,
        "epochs": 1,
        "include_preview_features": False,
        "verify_mask": True,
        "placement_mode": "append_only",
    },
    "m4_quick": {
        "total_updates": 20,
        "episodes_per_update": 4,
        "max_steps_per_episode": 200,
        "hidden_dim": 128,
        "action_hidden_dim": 64,
        "minibatch_size": 128,
        "epochs": 2,
        "include_preview_features": False,
        "verify_mask": False,
        "placement_mode": "append_only",
        "rollout_workers": 4,
    },
    "m4_night": {
        "total_updates": 200,
        "episodes_per_update": 8,
        "max_steps_per_episode": 300,
        "hidden_dim": 256,
        "action_hidden_dim": 128,
        "minibatch_size": 256,
        "epochs": 3,
        "include_preview_features": False,
        "verify_mask": False,
        "placement_mode": "append_only",
        "rollout_workers": 8,
    },
}


def make_config_from_preset(name: str, **overrides) -> PPOConfig:
    if name not in TRAIN_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(TRAIN_PRESETS)}")
    base = dict(TRAIN_PRESETS[name])
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    base.update(clean_overrides)
    return PPOConfig(**base)


def _config_to_worker_dict(config: PPOConfig) -> dict:
    return {
        "seed": config.seed,
        "verify_mask": config.verify_mask,
        "placement_mode": config.placement_mode,
        "action_features_dtype": config.action_features_dtype,
        "include_preview_features": config.include_preview_features,
        "max_steps_per_episode": config.max_steps_per_episode,
        "level_handicap_rate": config.level_handicap_rate,
        "learner_level": config.learner_level,
        "opponent_level": config.opponent_level,
        "starting_player": config.starting_player,
    }


def _parse_opponent_mix(mix: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for raw_part in (mix or "self:1.0").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight_s = part.split(":", 1)
            weight = float(weight_s)
        else:
            name, weight = part, 1.0
        name = name.strip()
        if weight <= 0:
            continue
        if name not in {
            "self",
            "random",
            "end_turn",
            "greedy_face",
            "legacy_max",
            "legacy_medium",
            "legacy_random_biggest",
            "trainv2_0251",
            "trainv2_0348",
            "trainv2_0156",
            "trainv2_0408",
            "trainv2_0700",
            "trainv2_0800",
            "v4_micro",
            "v4_lite",
            "v4_opti",
            "v4_max",
        }:
            raise ValueError(f"unknown opponent in opponent_mix: {name}")
        out.append((name, weight))
    if not out:
        return [("self", 1.0)]
    return out


def _choose_weighted(rng: rand_mod.Random, weighted: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in weighted)
    r = rng.random() * total
    acc = 0.0
    for name, weight in weighted:
        acc += weight
        if r <= acc:
            return name
    return weighted[-1][0]


def _choose_learner_player(rng: rand_mod.Random, mode: str) -> int:
    if mode == "p1":
        return 1
    if mode == "p2":
        return 2
    if mode != "random":
        raise ValueError(f"learner_side must be random, p1, or p2; got {mode}")
    return 1 if rng.random() < 0.5 else 2


def _choose_starting_player(
    rng: rand_mod.Random,
    mode: str,
    *,
    learner_player_id: int = 0,
) -> int:
    if mode == "p1":
        return 1
    if mode == "p2":
        return 2
    if mode == "learner":
        if learner_player_id not in (1, 2):
            return 1 if rng.random() < 0.5 else 2
        return learner_player_id
    if mode == "opponent":
        if learner_player_id not in (1, 2):
            return 1 if rng.random() < 0.5 else 2
        return 1 if learner_player_id == 2 else 2
    if mode != "random":
        raise ValueError(
            f"starting_player must be random, p1, p2, learner, or opponent; got {mode}"
        )
    return 1 if rng.random() < 0.5 else 2


def collect_policy_episodes_parallel(
    config: PPOConfig,
    model: ActionConditionedPolicy,
    seeds: list[int],
    max_steps: int,
) -> dict:
    if not config.include_action_features:
        raise NotImplementedError("action_features are required for action-conditioned PPO v1")
    if len(seeds) == 0:
        return {"transitions": [], "summaries": [], "inference_ms_p50": 0.0, "inference_ms_p95": 0.0}

    t_rollout0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    n_workers = min(config.rollout_workers, len(seeds))
    config_dict = _config_to_worker_dict(config)
    opponent_mix = _parse_opponent_mix(config.opponent_mix)
    league_rng = rand_mod.Random(config.seed * 17 + len(seeds) * 31 + max_steps)

    # Spawn persistent workers
    workers: list[tuple[mp.Process, mp.connection.Connection]] = []
    conn_to_wid: dict[mp.connection.Connection, int] = {}
    worker_states: list[dict] = []
    for wid in range(n_workers):
        proc, conn, _ = spawn_worker(config_dict, wid, ctx=ctx)
        workers.append((proc, conn))
        conn_to_wid[conn] = wid
        worker_states.append({
            "episode_id": None,
            "seed": None,
            "opponent_kind": "self",
            "learner_player_id": 1,
            "starting_player_id": 1,
            "transitions": [],
            "state": None,
            "pending": None,
        })

    # Queue of seeds
    episode_queue = list(enumerate(seeds))  # (episode_id, seed)
    summaries: list[dict] = []
    all_transitions: list[dict] = []
    inference_ms: list[float] = []

    def _send_to_worker(wid: int, payload: dict) -> None:
        try:
            workers[wid][1].send(payload)
        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            exitcode = workers[wid][0].exitcode
            raise RuntimeError(
                f"Rollout worker {wid} failed during send: {type(exc).__name__}, exitcode={exitcode}"
            ) from exc

    def _assign_next_episode_or_deactivate(wid: int) -> None:
        if episode_queue:
            next_ep_id, next_seed = episode_queue.pop(0)
            opponent_kind = _choose_weighted(league_rng, opponent_mix)
            learner_player_id = _choose_learner_player(league_rng, config.learner_side)
            if opponent_kind == "self":
                learner_player_id = 0
            starting_player_id = _choose_starting_player(
                league_rng,
                config.starting_player,
                learner_player_id=learner_player_id,
            )
            _send_to_worker(wid, {
                "cmd": "reset",
                "seed": next_seed,
                "episode_id": next_ep_id,
                "max_steps": max_steps,
                "opponent_kind": opponent_kind,
                "learner_player_id": learner_player_id,
                "starting_player_id": starting_player_id,
            })
            worker_states[wid] = {
                "episode_id": next_ep_id,
                "seed": next_seed,
                "opponent_kind": opponent_kind,
                "learner_player_id": learner_player_id,
                "starting_player_id": starting_player_id,
                "transitions": [],
                "state": None,
                "pending": None,
            }
        else:
            active_workers.discard(wid)

    # Kick off initial resets
    active_workers: set[int] = set()
    try:
        for wid in range(n_workers):
            if not episode_queue:
                break
            ep_id, seed = episode_queue.pop(0)
            opponent_kind = _choose_weighted(league_rng, opponent_mix)
            learner_player_id = _choose_learner_player(league_rng, config.learner_side)
            if opponent_kind == "self":
                learner_player_id = 0
            starting_player_id = _choose_starting_player(
                league_rng,
                config.starting_player,
                learner_player_id=learner_player_id,
            )
            _send_to_worker(wid, {
                "cmd": "reset",
                "seed": seed,
                "episode_id": ep_id,
                "max_steps": max_steps,
                "opponent_kind": opponent_kind,
                "learner_player_id": learner_player_id,
                "starting_player_id": starting_player_id,
            })
            worker_states[wid] = {
                "episode_id": ep_id,
                "seed": seed,
                "opponent_kind": opponent_kind,
                "learner_player_id": learner_player_id,
                "starting_player_id": starting_player_id,
                "transitions": [],
                "state": None,
                "pending": None,
            }
            active_workers.add(wid)
    except Exception:
        for proc, conn in workers:
            try:
                conn.close()
            except Exception:
                pass
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        raise

    try:
        while active_workers:
            active_conns = [workers[wid][1] for wid in active_workers]
            ready_conns = mp_wait(active_conns, timeout=60.0)
            if not ready_conns:
                dead = [wid for wid in active_workers if not workers[wid][0].is_alive()]
                detail = f"dead workers={dead}" if dead else "no response after 60s"
                raise RuntimeError(f"Rollout worker timeout: {detail}")

            for conn in ready_conns:
                wid = conn_to_wid[conn]
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError) as exc:
                    exitcode = workers[wid][0].exitcode
                    raise RuntimeError(
                        f"Rollout worker {wid} failed before response: {type(exc).__name__}, exitcode={exitcode}"
                    ) from exc

                if "error" in msg:
                    raise RuntimeError(f"Worker {wid} error: {msg['error']} (episode={msg.get('episode_id')})")

                msg_type = msg.get("type")
                ws = worker_states[wid]

                if msg_type == "state":
                    ws["episode_id"] = msg["episode_id"]
                    ws["seed"] = msg.get("seed")
                    ws["state"] = {
                        "obs": msg["obs"],
                        "mask": msg["mask"],
                        "af": msg["af"],
                        "player_id": msg["player_id"],
                        "episode_id": msg["episode_id"],
                        "step": msg["step"],
                        "opponent_kind": msg.get("opponent_kind", ws.get("opponent_kind", "self")),
                        "learner_player_id": msg.get("learner_player_id", ws.get("learner_player_id", 1)),
                        "starting_player_id": msg.get("starting_player_id", ws.get("starting_player_id")),
                    }
                    continue

                if msg_type != "step_result":
                    raise RuntimeError(f"Worker {wid} sent unknown message type: {msg_type}")

                pending = ws.get("pending")
                if pending is None:
                    summary = msg.get("summary")
                    if summary is None:
                        raise RuntimeError(f"Worker {wid} step_result without pending transition")
                    summary = dict(summary)
                    summary["episode_id"] = msg.get("episode_id")
                    summaries.append(summary)
                    ws["transitions"] = []
                    ws["state"] = None
                    _assign_next_episode_or_deactivate(wid)
                    continue

                transition = dict(pending)
                transition.update({
                    "reward": msg["reward"],
                    "done": bool(msg["terminated"]),
                    "truncated": bool(msg["truncated"]),
                    "next_obs": msg["next_obs"],
                })
                ws["transitions"].append(transition)
                ws["pending"] = None

                summary = msg.get("summary")
                if summary is not None:
                    summary = dict(summary)
                    summary["episode_id"] = msg.get("episode_id")
                    summaries.append(summary)
                    all_transitions.extend(ws["transitions"])
                    ws["transitions"] = []
                    ws["state"] = None

                    _assign_next_episode_or_deactivate(wid)
                    continue

                next_state = msg.get("next_state")
                if next_state is None:
                    raise RuntimeError(f"Worker {wid} non-final step_result missing next_state")

                ws["state"] = {
                    "obs": next_state["obs"],
                    "mask": next_state["mask"],
                    "af": next_state["af"],
                    "player_id": next_state["player_id"],
                    "episode_id": msg["episode_id"],
                    "step": msg["step"],
                    "opponent_kind": ws.get("opponent_kind", "self"),
                    "learner_player_id": ws.get("learner_player_id", 1),
                    "starting_player_id": ws.get("starting_player_id"),
                }

            ready_for_inference = [
                wid for wid in active_workers
                if worker_states[wid].get("state") is not None
                and worker_states[wid].get("pending") is None
            ]
            if ready_for_inference:
                B = len(ready_for_inference)
                obs_np = np.stack([worker_states[wid]["state"]["obs"] for wid in ready_for_inference], axis=0)
                af_np = np.stack([worker_states[wid]["state"]["af"] for wid in ready_for_inference], axis=0)
                masks_np = np.stack([worker_states[wid]["state"]["mask"] for wid in ready_for_inference], axis=0)

                obs_mx = mx.array(obs_np)
                af_mx = mx.array(af_np)
                t_inf0 = time.perf_counter()
                logits, values = model(obs_mx, af_mx)
                mx.eval(logits, values)
                inference_ms.append((time.perf_counter() - t_inf0) * 1000.0)
                logits_np = np.array(logits)
                values_np = np.array(values)

                for b, wid in enumerate(ready_for_inference):
                    ws = worker_states[wid]
                    state = ws["state"]
                    action_id, log_prob = sample_action(mx.array(logits_np[b]), masks_np[b])
                    value_scalar = float(values_np[b])

                    ws["pending"] = {
                        "obs": state["obs"],
                        "action_features": state["af"],
                        "mask": state["mask"],
                        "action_id": action_id,
                        "value": value_scalar,
                        "log_prob": log_prob,
                        "player_id": state["player_id"],
                        "episode_id": state["episode_id"],
                        "step": state["step"],
                        "opponent_kind": state.get("opponent_kind", ws.get("opponent_kind", "self")),
                        "learner_player_id": state.get("learner_player_id", ws.get("learner_player_id", 1)),
                        "starting_player_id": state.get("starting_player_id", ws.get("starting_player_id")),
                    }

                    # Send action to worker
                    _send_to_worker(wid, {
                        "cmd": "step",
                        "action_id": action_id,
                        "opponent_kind": ws.get("opponent_kind", "self"),
                        "learner_player_id": ws.get("learner_player_id", 1),
                    })
                    ws["state"] = None

    finally:
        # Cleanup
        for proc, conn in workers:
            try:
                conn.send({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

    summaries.sort(key=lambda s: s.get("episode_id", 0))

    return {
        "transitions": all_transitions,
        "summaries": summaries,
        "inference_ms_p50": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
        "inference_ms_p95": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
        "rollout_time": time.perf_counter() - t_rollout0,
    }

def _init_env(config: PPOConfig, seed: int) -> ClassicRLEnv:
    env = ClassicRLEnv(
        seed=seed,
        verify_mask=config.verify_mask,
        placement_mode=config.placement_mode,
    )
    start_rng = rand_mod.Random(config.seed * 17 + seed)
    env.reset(
        seed=seed,
        starting_player_id=_choose_starting_player(start_rng, config.starting_player),
    )
    return env


def collect_policy_episodes_batched(
    config: PPOConfig,
    model: ActionConditionedPolicy,
    seeds: list[int],
    max_steps: int,
) -> dict:
    if not config.include_action_features:
        raise NotImplementedError("action_features are required for action-conditioned PPO v1")

    envs = [_init_env(config, s) for s in seeds]

    all_transitions: list[dict] = []
    summaries: list[dict] = []
    active = list(range(len(envs)))
    step_counts = [0] * len(envs)
    p1_rewards = [0.0] * len(envs)
    p2_rewards = [0.0] * len(envs)

    while active:
        batch_data: list[dict] = []
        batch_indices: list[int] = []

        for idx in active:
            env = envs[idx]
            st = env._env.state
            if st.status != GameStatus.ONGOING or st.turn_number > env._max_turns or step_counts[idx] >= max_steps:
                continue
            cp = env.current_player_id()
            obs = env.observe(cp)
            mask = env.action_mask(cp)
            af = env.action_features(cp, include_preview=config.include_preview_features)
            batch_data.append({"obs": obs, "mask": mask, "af": af, "cp": cp, "env": env, "env_index": idx})
            batch_indices.append(idx)

        if not batch_data:
            break

        B = len(batch_data)
        obs_np = np.stack([d["obs"] for d in batch_data], axis=0)
        af_np = np.stack([d["af"] for d in batch_data], axis=0)
        masks_np = np.stack([d["mask"] for d in batch_data], axis=0)

        obs_mx = mx.array(obs_np)
        af_mx = mx.array(af_np)
        logits, values = model(obs_mx, af_mx)
        mx.eval(logits, values)
        logits_np = np.array(logits)
        values_np = np.array(values)

        new_active = []
        for b, idx in enumerate(batch_indices):
            env = batch_data[b]["env"]
            cp = batch_data[b]["cp"]
            mask = masks_np[b]
            action_id, log_prob = sample_action(mx.array(logits_np[b]), mask)
            next_obs, reward, terminated, truncated, info = env.step(action_id)
            step_counts[idx] += 1

            if cp == 1:
                p1_rewards[idx] += reward
            else:
                p2_rewards[idx] += reward

            all_transitions.append({
                "obs": batch_data[b]["obs"],
                "action_features": batch_data[b]["af"],
                "mask": mask,
                "action_id": action_id,
                "reward": reward,
                "done": terminated,
                "truncated": truncated,
                "value": float(values_np[b]),
                "log_prob": log_prob,
                "player_id": cp,
                "next_obs": next_obs,
                "env_index": idx,
            })

            st = env._env.state
            if not terminated and not truncated and st.status == GameStatus.ONGOING and step_counts[idx] < max_steps:
                new_active.append(idx)

        active = new_active

    for idx, env in enumerate(envs):
        st = env._env.state
        summaries.append({
            "winner_id": env.winner_id(),
            "status": st.status.value,
            "turns": st.turn_number,
            "steps": step_counts[idx],
            "truncated": st.turn_number > env._max_turns,
            "p1_hp": st.p1.hero.hp,
            "p2_hp": st.p2.hero.hp,
            "p1_reward": p1_rewards[idx],
            "p2_reward": p2_rewards[idx],
            "invalid_actions": sum(1 for t in all_transitions if t.get("env_index") == idx and t["reward"] == -0.05 and not t["done"]),
            "seed": seeds[idx],
        })

    return {"transitions": all_transitions, "summaries": summaries}


def collect_policy_episode(
    env: ClassicRLEnv,
    model: ActionConditionedPolicy,
    *,
    seed: int,
    max_steps: int,
    include_action_features: bool = True,
    include_preview_features: bool = False,
    starting_player_id: int | None = None,
) -> dict:
    if not include_action_features:
        raise NotImplementedError("action_features are required for action-conditioned PPO v1")

    mx.random.seed(seed * 2 + 17)
    env.reset(seed=seed, starting_player_id=starting_player_id)
    transitions: list[dict] = []

    for _step in range(max_steps):
        cp = env.current_player_id()
        obs = env.observe(cp)
        mask = env.action_mask(cp)
        af = env.action_features(cp, include_preview=include_preview_features)

        obs_mx = mx.array(obs[None, :])
        af_mx = mx.array(af[None, :, :])

        logits, value = model(obs_mx, af_mx)
        mx.eval(logits, value)
        logits_1d = logits[0]
        value_scalar = float(value[0].item())

        action_id, log_prob = sample_action(logits_1d, mask)

        next_obs, reward, terminated, truncated, info = env.step(action_id)

        transitions.append({
            "obs": obs,
            "action_features": af,
            "mask": mask,
            "action_id": action_id,
            "reward": reward,
            "done": terminated,
            "truncated": truncated,
            "value": value_scalar,
            "log_prob": log_prob,
            "player_id": cp,
            "next_obs": next_obs,
        })

        if terminated or truncated:
            break

    st = env._env.state
    summary = {
        "winner_id": env.winner_id(),
        "status": st.status.value,
        "turns": st.turn_number,
        "steps": len(transitions),
        "truncated": any(t["truncated"] for t in transitions[-1:]) if transitions else False,
        "p1_hp": st.p1.hero.hp,
        "p2_hp": st.p2.hero.hp,
        "p1_reward": float(env._p1_reward),
        "p2_reward": float(env._p2_reward),
        "invalid_actions": sum(1 for t in transitions if t["reward"] == -0.05 and not t["done"]),
        "seed": seed,
        "starting_player_id": starting_player_id or 1,
    }

    return {"transitions": transitions, "summary": summary}


# ============================================================================
# GAE — PER PLAYER SUBSEQUENCE
# ============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    player_ids: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    returns = np.zeros(N, dtype=np.float32)

    players = np.unique(player_ids)
    for pid in players:
        idx = np.where(player_ids == pid)[0]
        if len(idx) == 0:
            continue
        adv, ret = _gae_one_subsequence(rewards[idx], values[idx], dones[idx], gamma, gae_lambda)
        advantages[idx] = adv
        returns[idx] = ret

    return advantages, returns


def _gae_one_subsequence(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in range(T - 1, -1, -1):
        if dones[t]:
            next_value = 0.0
            next_gae = 0.0
        else:
            next_value = values[t + 1] if t + 1 < T else 0.0
            next_gae = adv[t + 1] if t + 1 < T else 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        adv[t] = delta + gamma * gae_lambda * next_gae

    returns = adv + values
    return adv, returns


# ============================================================================
# PPO UPDATE
# ============================================================================

def ppo_update(
    model: ActionConditionedPolicy,
    optimizer,
    batch: dict,
    config: PPOConfig,
) -> dict:
    obs_np = batch["obs"]
    af_np = batch["action_features"]
    mask_np = batch["mask"]
    act_np = batch["action_ids"]
    olp_np = batch["log_probs"]
    adv_np = batch["advantages"]
    ret_np = batch["returns"]

    N = obs_np.shape[0]
    indices = np.arange(N)

    epoch_metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_fraction": [], "approx_kl": []}

    def loss_fn(model, obs, af, mask, actions, old_lp, advantages, returns):
        logits, values = model(obs, af)
        mlogits = masked_logits(logits, mask)

        probs = nn.softmax(mlogits, axis=-1)

        actions_onehot = mx.zeros_like(probs)
        actions_onehot[mx.arange(actions.shape[0]), actions] = 1.0
        action_probs = mx.sum(probs * actions_onehot, axis=-1)
        new_lp = mx.log(action_probs + 1e-10)

        ratio = mx.exp(new_lp - old_lp)

        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))

        clip_frac = mx.mean(
            mx.where(ratio < 1.0 - config.clip_epsilon, mx.ones_like(ratio),
            mx.where(ratio > 1.0 + config.clip_epsilon, mx.ones_like(ratio),
            mx.zeros_like(ratio)))
        )

        value_loss = config.value_coef * mx.mean((returns - values) ** 2)

        legal_probs = probs * mask
        legal_probs = legal_probs / (mx.sum(legal_probs, axis=-1, keepdims=True) + 1e-10)
        entropy = mx.mean(-mx.sum(legal_probs * mx.log(legal_probs + 1e-10), axis=-1))

        approx_kl = mx.mean(old_lp - new_lp)

        loss = policy_loss + value_loss - config.entropy_coef * entropy

        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "clip_fraction": clip_frac,
            "approx_kl": approx_kl,
        }

    value_and_grad = nn.value_and_grad(model, loss_fn)

    for _epoch in range(config.epochs):
        np.random.shuffle(indices)
        for start in range(0, N, config.minibatch_size):
            idx = indices[start:start + config.minibatch_size]

            ob = mx.array(obs_np[idx])
            af = mx.array(af_np[idx])
            ma = mx.array(mask_np[idx])
            ac = mx.array(act_np[idx], dtype=mx.int32)
            olp = mx.array(olp_np[idx])
            adv = mx.array(adv_np[idx])
            ret = mx.array(ret_np[idx])

            (loss_val, aux), grads = value_and_grad(model, ob, af, ma, ac, olp, adv, ret)

            if config.max_grad_norm is not None:
                grads = _clip_grads(grads, config.max_grad_norm)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            for k in epoch_metrics:
                epoch_metrics[k].append(float(aux[k].item()))

    for k, vals in epoch_metrics.items():
        if not vals:
            raise ValueError(f"ppo_update: no metric values collected for {k}")
        _assert_finite_array(f"ppo_update.{k}", vals)

    pl_mean = float(np.mean(epoch_metrics["policy_loss"]))
    vl_mean = float(np.mean(epoch_metrics["value_loss"]))
    ent_mean = float(np.mean(epoch_metrics["entropy"]))
    kl_mean = float(np.mean(epoch_metrics["approx_kl"]))
    cf_mean = float(np.mean(epoch_metrics["clip_fraction"]))
    loss_mean = pl_mean + vl_mean - config.entropy_coef * ent_mean

    return {
        "loss": loss_mean,
        "policy_loss": pl_mean,
        "value_loss": vl_mean,
        "entropy": ent_mean,
        "approx_kl": kl_mean,
        "clip_fraction": cf_mean,
    }


def _clip_grads(grads, max_norm):
    flat = nn.utils.tree_flatten(grads)
    total_norm = mx.sqrt(sum(mx.sum(v**2) for _, v in flat))
    total_norm = mx.maximum(total_norm, mx.array(1e-6, dtype=mx.float32))
    scale = mx.minimum(mx.array(max_norm, dtype=mx.float32), total_norm) / total_norm
    return nn.utils.tree_unflatten(
        [(k, v * scale) for k, v in flat]
    )


def _assert_finite_array(name: str, arr) -> None:
    a = np.asarray(arr)
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError(f"{name} contains non-finite values")


def _validate_batch(batch: dict, config: PPOConfig) -> None:
    N = len(batch["obs"])
    if N < config.min_batch_transitions:
        raise ValueError(
            f"batch size {N} < min_batch_transitions {config.min_batch_transitions}"
        )

    for field in ["obs", "action_features", "mask", "log_probs", "advantages", "returns"]:
        _assert_finite_array(f"batch.{field}", batch[field])

    masks = batch["mask"]
    actions = batch["action_ids"]
    n_actions = masks.shape[1]
    for i in range(N):
        aid = int(actions[i])
        if aid < 0 or aid >= n_actions:
            raise ValueError(f"action_id out of range at row {i}: {aid} (max {n_actions - 1})")
        if masks[i, aid] != 1.0:
            raise ValueError(f"illegal action at row {i}: action_id={aid} mask={masks[i, aid]}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config: PPOConfig) -> dict:
    rand_mod.seed(config.seed)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    model = ActionConditionedPolicy(
        obs_dim=1456,
        action_feature_dim=171,
        hidden_dim=config.hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
    )
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=config.learning_rate)

    if config.resume_checkpoint:
        loaded = load_checkpoint(config.resume_checkpoint, model, optimizer=optimizer)
        meta = loaded.get("metadata", {})
        resumed_from = int(meta.get("update", 0))
        config.start_update = max(config.start_update, resumed_from)
    else:
        config.start_update = config.start_update or 0

    total_episodes = 0
    total_steps = 0
    last_loss = 0.0
    last_entropy = 0.0
    checkpoint_path = ""
    last_update = config.start_update
    skipped_updates = 0
    rollout_time_total = 0.0
    update_time_total = 0.0

    for update in range(config.start_update + 1, config.start_update + config.total_updates + 1):
        last_update = update
        all_transitions: list[dict] = []

        t_rollout0 = time.perf_counter()
        if config.rollout_workers > 1:
            # Parallel multi-env collection via persistent spawn workers
            seeds = [config.seed * 1000 + update * 100 + ep for ep in range(config.episodes_per_update)]
            result = collect_policy_episodes_parallel(
                config, model, seeds,
                max_steps=config.max_steps_per_episode,
            )
            all_transitions.extend(result["transitions"])
            total_episodes += len(result["summaries"])
            total_steps += len(result["transitions"])
        else:
            if _parse_opponent_mix(config.opponent_mix) != [("self", 1.0)]:
                raise NotImplementedError("league opponent_mix requires rollout_workers > 1")
            serial_rng = rand_mod.Random(config.seed * 19 + update * 97)
            for ep in range(config.episodes_per_update):
                ep_seed = config.seed * 1000 + update * 100 + ep
                starting_player_id = _choose_starting_player(serial_rng, config.starting_player)
                env = ClassicRLEnv(
                    seed=config.seed,
                    verify_mask=config.verify_mask,
                    placement_mode=config.placement_mode,
                )
                result = collect_policy_episode(
                    env=env,
                    model=model,
                    seed=ep_seed,
                    max_steps=config.max_steps_per_episode,
                    include_action_features=config.include_action_features,
                    include_preview_features=config.include_preview_features,
                    starting_player_id=starting_player_id,
                )
                all_transitions.extend(result["transitions"])
                total_episodes += 1
                total_steps += len(result["transitions"])
        rollout_time_total += time.perf_counter() - t_rollout0

        if len(all_transitions) < config.min_batch_transitions:
            skipped_updates += 1
            if config.metrics_path:
                _write_skip_metrics(config, update, total_episodes, total_steps, len(all_transitions))
            print(
                f"[update {update:3d}/{config.start_update + config.total_updates}] "
                f"SKIPPED transitions={len(all_transitions)} "
                f"< min_batch_transitions={config.min_batch_transitions}"
            )
            continue

        batch = _prepare_batch(all_transitions, config)
        _validate_batch(batch, config)
        t_update0 = time.perf_counter()
        metrics = ppo_update(model, optimizer, batch, config)
        update_time_total += time.perf_counter() - t_update0

        if config.fail_on_non_finite:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    raise ValueError(f"metric {k} is non-finite: {v}")

        last_loss = metrics["loss"]
        last_entropy = metrics["entropy"]

        ckpt_path = Path(config.checkpoint_dir) / f"update_{update:04d}.npz"
        save_checkpoint(
            str(ckpt_path),
            model,
            optimizer=optimizer,
            metadata={
                "model_version": MODEL_VERSION,
                "obs_dim": 1456,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
                "config": _config_to_dict(config),
                "update": update,
                "total_steps": total_steps,
                "start_update": config.start_update,
                "resumed_from": config.resume_checkpoint,
            },
        )
        checkpoint_path = str(ckpt_path)

        print(
            f"[update {update:3d}/{config.start_update + config.total_updates}] "
            f"episodes={total_episodes:3d} "
            f"steps={total_steps:5d} "
            f"loss={metrics['loss']:6.4f} "
            f"policy={metrics['policy_loss']:6.4f} "
            f"value={metrics['value_loss']:6.4f} "
            f"ent={metrics['entropy']:6.4f} "
            f"kl={metrics['approx_kl']:6.4f} "
            f"clip={metrics['clip_fraction']:.2f}"
        )

        _write_metrics(config, update, total_episodes, total_steps, metrics,
                        checkpoint_path)

        if (config.eval_every_updates > 0 and
                update % config.eval_every_updates == 0):
            _run_eval(config, update, model, checkpoint_path)

    return {
        "updates": config.total_updates,
        "start_update": config.start_update,
        "last_update": last_update,
        "episodes": total_episodes,
        "steps": total_steps,
        "last_loss": last_loss,
        "last_entropy": last_entropy,
        "checkpoint_path": checkpoint_path,
        "skipped_updates": skipped_updates,
        "rollout_time": rollout_time_total,
        "update_time": update_time_total,
    }


def _prepare_batch(transitions: list[dict], config: PPOConfig) -> dict:
    dones_full = np.array(
        [t["done"] or t["truncated"] for t in transitions], dtype=np.float32
    )

    player_ids = np.array([t["player_id"] for t in transitions])

    advantages, returns = compute_gae(
        rewards=np.array([t["reward"] for t in transitions], dtype=np.float32),
        values=np.array([t["value"] for t in transitions], dtype=np.float32),
        dones=dones_full,
        player_ids=player_ids,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages)
    if adv_std < 1e-8:
        advantages = advantages - adv_mean
    else:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    obs = np.array([t["obs"] for t in transitions], dtype=np.float32)
    af = np.stack([t["action_features"] for t in transitions], axis=0)
    # Worker may already send float16; preserve dtype, convert only if needed
    if config.action_features_dtype == "float16" and af.dtype != np.float16:
        af = af.astype(np.float16)
    elif config.action_features_dtype == "float32" and af.dtype != np.float32:
        af = af.astype(np.float32)

    return {
        "obs": obs,
        "action_features": af,
        "mask": np.array([t["mask"] for t in transitions], dtype=np.float32),
        "action_ids": np.array([t["action_id"] for t in transitions], dtype=np.int32),
        "log_probs": np.array([t["log_prob"] for t in transitions], dtype=np.float32),
        "advantages": advantages.astype(np.float32),
        "returns": returns.astype(np.float32),
    }


def _config_to_dict(config: PPOConfig) -> dict:
    return {
        "total_updates": config.total_updates,
        "episodes_per_update": config.episodes_per_update,
        "max_steps_per_episode": config.max_steps_per_episode,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_epsilon": config.clip_epsilon,
        "learning_rate": config.learning_rate,
        "entropy_coef": config.entropy_coef,
        "value_coef": config.value_coef,
        "hidden_dim": config.hidden_dim,
        "action_hidden_dim": config.action_hidden_dim,
        "include_action_features": config.include_action_features,
        "include_preview_features": config.include_preview_features,
        "seed": config.seed,
        "metrics_path": config.metrics_path,
        "eval_every_updates": config.eval_every_updates,
        "eval_games": config.eval_games,
        "resume_checkpoint": config.resume_checkpoint,
        "start_update": config.start_update,
        "fail_on_non_finite": config.fail_on_non_finite,
        "min_batch_transitions": config.min_batch_transitions,
        "rollout_workers": config.rollout_workers,
        "verify_mask": config.verify_mask,
        "placement_mode": config.placement_mode,
        "profile_actions": config.profile_actions,
        "action_features_dtype": config.action_features_dtype,
        "opponent_mix": config.opponent_mix,
        "learner_side": config.learner_side,
        "starting_player": config.starting_player,
        "level_handicap_rate": config.level_handicap_rate,
        "learner_level": config.learner_level,
        "opponent_level": config.opponent_level,
    }


def _write_metrics(config, update, total_episodes, total_steps, metrics, ckpt_path):
    if config.metrics_path is None:
        return
    p = Path(config.metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "update": update,
        "episodes": total_episodes,
        "steps": total_steps,
        "loss": metrics["loss"],
        "policy_loss": metrics["policy_loss"],
        "value_loss": metrics["value_loss"],
        "entropy": metrics["entropy"],
        "approx_kl": metrics["approx_kl"],
        "clip_fraction": metrics["clip_fraction"],
        "checkpoint_path": ckpt_path,
    }
    with open(str(p), "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_skip_metrics(config, update, total_episodes, total_steps, transitions):
    if config.metrics_path is None:
        return
    p = Path(config.metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type": "skipped_update",
        "update": update,
        "episodes": total_episodes,
        "steps": total_steps,
        "transitions": transitions,
        "reason": "min_batch_transitions",
        "min_batch_transitions": config.min_batch_transitions,
    }
    with open(str(p), "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _run_eval(config, update, model, ckpt_path):
    from ai.train_v2.ppo_eval import load_mlx_policy, evaluate_policy_matchup

    eval_seed = config.seed * 100 + update
    mx.random.seed(eval_seed)

    mlx_pol = load_mlx_policy(ckpt_path, hidden_dim=config.hidden_dim,
                               action_hidden_dim=config.action_hidden_dim, mode="argmax")

    eval_seeds = list(range(eval_seed, eval_seed + config.eval_games))

    for opp_name, opp_cls in [("random", RandomLegalPolicy), ("end_turn", EndTurnPolicy)]:
        result = evaluate_policy_matchup(
            mlx_pol, opp_cls(), seeds=eval_seeds, swap_sides=True, max_steps=config.max_steps_per_episode,
        )
        label = f"eval_vs_{opp_name}_update{update}"
        print(f"  {label}: wr={result['p1_winrate']:.3f} "
              f"turns={result['avg_turns']:.1f} "
              f"fb={result['p1_invalid_fallbacks']}")

        if config.metrics_path:
            p = Path(config.metrics_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            eval_record = {
                "type": "eval",
                "update": update,
                "opponent": opp_name,
                "winrate": result["p1_winrate"],
                "games": result["games"],
                "avg_turns": result["avg_turns"],
                "avg_p1_reward": result["avg_p1_reward"],
                "invalid_actions": result["invalid_actions"],
                "p1_invalid_fallbacks": result["p1_invalid_fallbacks"],
            }
            with open(str(p), "a") as f:
                f.write(json.dumps(eval_record, separators=(",", ":")) + "\n")

    mx.random.seed(config.seed * 1000 + update)


# ============================================================================
# MEMORY ESTIMATE
# ============================================================================

def estimate_update_memory(config: PPOConfig) -> dict:
    episodes = config.episodes_per_update
    steps = config.max_steps_per_episode
    N = episodes * steps

    obs_bytes = N * 1456 * 4
    af_bytes = N * 601 * 171 * (2 if config.action_features_dtype == "float16" else 4)
    mask_bytes = N * 601 * 4
    scalar_bytes = N * 4 * 5  # rewards, values, log_probs, advantages, returns
    action_ids_bytes = N * 4

    total = obs_bytes + af_bytes + mask_bytes + scalar_bytes + action_ids_bytes

    # model parameters rough estimate
    hidden = config.hidden_dim
    ahidden = config.action_hidden_dim
    param_count = (
        1456 * hidden + hidden +
        hidden * hidden + hidden +
        171 * ahidden + ahidden +
        (hidden + ahidden) + 1 +
        hidden + 1
    )
    param_bytes = param_count * 4

    # optimizer state (Adam: 2 copies of params)
    opt_bytes = param_count * 4 * 2

    return {
        "transitions": N,
        "obs_mb": obs_bytes / (1024 * 1024),
        "action_features_mb": af_bytes / (1024 * 1024),
        "mask_mb": mask_bytes / (1024 * 1024),
        "scalar_mb": scalar_bytes / (1024 * 1024),
        "total_mb": total / (1024 * 1024),
        "parameters_mb": param_bytes / (1024 * 1024),
        "optimizer_mb": opt_bytes / (1024 * 1024),
        "rough_peak_mb": (total + param_bytes + opt_bytes) / (1024 * 1024),
    }


# ============================================================================
# CLI
# ============================================================================

def _main():
    parser = argparse.ArgumentParser(description="Train PPO baseline for TrainV2")
    parser.add_argument("--preset", default=None, choices=list(TRAIN_PRESETS), help="Use a named training preset")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--action-hidden-dim", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--eval-every-updates", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--include-preview-features", default=None, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--rollout-workers", type=int, default=None, help="Number of parallel env runners (1 = serial)")
    parser.add_argument("--verify-mask", default=None, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--placement-mode", default=None, choices=["append_only", "full"], help="Action placement mode")
    parser.add_argument("--profile-actions", action="store_true", help="Run action profiling and print results")
    parser.add_argument("--action-features-dtype", default=None, choices=["float32", "float16"])
    parser.add_argument("--opponent-mix", default=None, help="League mix, e.g. self:0.4,random:0.1,greedy_face:0.3,legacy_max:0.2")
    parser.add_argument("--learner-side", default=None, choices=["random", "p1", "p2"])
    parser.add_argument("--starting-player", default=None, choices=["random", "p1", "p2", "learner", "opponent"])
    parser.add_argument("--level-handicap-rate", type=float, default=None, help="Fraction of fixed-opponent episodes where learner uses lower card levels")
    parser.add_argument("--learner-level", type=int, default=None, help="Learner card level in handicap episodes")
    parser.add_argument("--opponent-level", type=int, default=None, help="Opponent card level in handicap episodes")
    args = parser.parse_args()

    if args.preset:
        config = make_config_from_preset(args.preset)
    else:
        config = PPOConfig()

    overrides = {
        "total_updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "max_steps_per_episode": args.max_steps,
        "hidden_dim": args.hidden_dim,
        "action_hidden_dim": args.action_hidden_dim,
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.seed,
        "metrics_path": args.metrics_path,
        "eval_every_updates": args.eval_every_updates,
        "eval_games": args.eval_games,
        "include_preview_features": args.include_preview_features,
        "resume_checkpoint": args.resume_checkpoint,
        "rollout_workers": args.rollout_workers,
        "verify_mask": args.verify_mask,
        "placement_mode": args.placement_mode,
        "profile_actions": args.profile_actions,
        "action_features_dtype": args.action_features_dtype,
        "opponent_mix": args.opponent_mix,
        "learner_side": args.learner_side,
        "starting_player": args.starting_player,
        "level_handicap_rate": args.level_handicap_rate,
        "learner_level": args.learner_level,
        "opponent_level": args.opponent_level,
    }

    for key, val in overrides.items():
        if val is not None:
            setattr(config, key, val)

    mem = estimate_update_memory(config)
    print(f"TrainV2 PPO v2 | MLX | {config.total_updates} updates × {config.episodes_per_update} eps × workers={config.rollout_workers}")
    print(f"Obs={config.hidden_dim}h × {config.action_hidden_dim}ah | lr={config.learning_rate} | seed={config.seed}")
    print(f"Placement={config.placement_mode} | verify_mask={config.verify_mask} | af_dtype={config.action_features_dtype}")
    print(f"Opponent mix={config.opponent_mix} | learner_side={config.learner_side} | starting_player={config.starting_player}")
    print(f"Level handicap rate={config.level_handicap_rate} | learner_level={config.learner_level} | opponent_level={config.opponent_level}")
    print(f"Estimated per-update memory: {mem['total_mb']:.1f} MB  (peak ~{mem['rough_peak_mb']:.1f} MB)")
    if mem["rough_peak_mb"] > 10_000:
        print("WARNING: estimated peak memory > 10 GB — consider reducing workers or steps")
    if config.resume_checkpoint:
        print(f"Resuming from {config.resume_checkpoint}")

    if config.profile_actions:
        from ai.train_v2.profile_actions import profile_action_pipeline
        prof = profile_action_pipeline(
            episodes=3,
            steps_per_episode=20,
            seed=config.seed,
            placement_mode=config.placement_mode,
            verify_mask=config.verify_mask,
        )
        print(f"Action profile: mask_p50={prof['mask_ms_p50']:.2f}ms  fast_feats_p50={prof['features_fast_ms_p50']:.2f}ms")

    t0 = time.perf_counter()

    result = train(config)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s | {result['episodes']} eps | {result['steps']} steps")
    print(f"Checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    _main()
