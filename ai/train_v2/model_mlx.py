"""
Action-conditioned actor-critic in MLX for TrainV2 PPO.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL_VERSION = "classic_action_conditioned_mlx_v1"


def masked_logits(logits: mx.array, mask: mx.array) -> mx.array:
    if not isinstance(mask, mx.array):
        mask = mx.array(mask)
    return mx.where(mask.astype(mx.bool_), logits, mx.array(-1e9, dtype=mx.float32))


def sample_action(logits: mx.array, mask: mx.array) -> tuple[int, float]:
    mlogits = masked_logits(logits, mask)
    probs = nn.softmax(mlogits, axis=-1)
    action = mx.random.categorical(mlogits).item()
    lp = mx.log(probs + 1e-10)
    return action, float(lp[action].item())


def policy_argmax(logits: mx.array, mask: mx.array) -> int:
    mlogits = masked_logits(logits, mask)
    return int(mx.argmax(mlogits, axis=-1).item())


class ActionConditionedPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = 1456,
        action_feature_dim: int = 171,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_feature_dim = action_feature_dim
        self.hidden_dim = hidden_dim
        self.action_hidden_dim = action_hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Linear(action_feature_dim, action_hidden_dim)
        self.candidate_scorer = nn.Linear(hidden_dim + action_hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(
        self,
        obs: mx.array,
        action_features: mx.array,
    ) -> tuple[mx.array, mx.array]:
        B = obs.shape[0]
        state_emb = self.state_encoder(obs)
        state_bc = mx.expand_dims(state_emb, axis=1)
        state_bc = mx.broadcast_to(state_bc, (B, 601, self.hidden_dim))

        action_emb = mx.reshape(action_features, (B * 601, self.action_feature_dim))
        action_emb = self.action_encoder(action_emb)
        action_emb = mx.reshape(action_emb, (B, 601, self.action_hidden_dim))

        joint = mx.concatenate([state_bc, action_emb], axis=-1)
        joint = mx.reshape(joint, (B * 601, self.hidden_dim + self.action_hidden_dim))
        raw_logits = self.candidate_scorer(joint)
        logits = mx.reshape(raw_logits, (B, 601))

        value = self.value_head(state_emb).squeeze(-1)
        return logits, value


def flatten_params(model) -> dict[str, np.ndarray]:
    flat = nn.utils.tree_flatten(model.trainable_parameters())
    return {k: np.array(v) for k, v in flat}


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np_dict = flatten_params(model)

    if optimizer is not None:
        flat_opt = nn.utils.tree_flatten(optimizer.state)
        for key, val in flat_opt:
            np_dict[f"_opt_{key}"] = np.array(val)

    meta = (metadata or {}).copy()
    meta.setdefault("model_version", MODEL_VERSION)
    meta.setdefault("obs_dim", 1456)
    meta.setdefault("action_feature_dim", 171)
    meta.setdefault("max_candidate_actions", 601)
    interaction_kind = getattr(model, "state_action_interaction_kind", None)
    if interaction_kind is not None:
        meta.setdefault("state_action_interaction", str(interaction_kind))
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta_json = json.dumps(meta)
    np_dict["__meta__"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

    np.savez(str(path), **np_dict)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
) -> dict:
    loaded = dict(np.load(path, allow_pickle=True))

    meta_raw = loaded.pop("__meta__", None)
    metadata = {}
    if meta_raw is not None:
        if hasattr(meta_raw, 'tobytes'):
            metadata = json.loads(meta_raw.tobytes().decode("utf-8"))
        elif hasattr(meta_raw, 'item'):
            v = meta_raw.item()
            if hasattr(v, 'decode'):
                metadata = json.loads(v.decode("utf-8"))
            else:
                metadata = json.loads(str(v))
        else:
            metadata = json.loads(str(meta_raw))

    weight_pairs: list[tuple[str, mx.array]] = []
    opt_pairs: list[tuple[str, mx.array]] = []

    for key, val in loaded.items():
        if key.startswith("_opt_"):
            opt_pairs.append((key[5:], mx.array(val)))
        else:
            weight_pairs.append((key, mx.array(val)))

    params = nn.utils.tree_unflatten(weight_pairs)
    model.update(params)
    mx.eval(model.parameters())

    result = {"metadata": metadata}

    if opt_pairs and optimizer is not None:
        opt_tree = nn.utils.tree_unflatten(opt_pairs)
        optimizer.state = opt_tree
        mx.eval(optimizer.state)
        result["optimizer_restored"] = True
    elif opt_pairs:
        result["optimizer_restored"] = False
        result["optimizer_note"] = "optimizer state present but no optimizer object provided"

    return result
