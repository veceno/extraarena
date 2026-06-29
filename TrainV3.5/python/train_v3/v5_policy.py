"""V5-specific MLX policies for action-conditioned PPO training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.nn as nn

from .contracts import (
    ACTION_FEATURE_DIM,
    HISTORY_DIM,
    OBS_V1_DIM,
    OBS_V5_DIM,
    PRIVATE_INFO_DIM,
    V5_GLOBAL_DIM,
)


@dataclass(frozen=True)
class V5PolicyShape:
    obs_dim: int = OBS_V5_DIM
    action_feature_dim: int = ACTION_FEATURE_DIM
    global_dim: int = V5_GLOBAL_DIM
    base_dim: int = OBS_V1_DIM
    private_dim: int = PRIVATE_INFO_DIM
    history_dim: int = HISTORY_DIM


class V5ActionConditionedPolicy(nn.Module):
    """Split-encoder action-conditioned actor critic for V5 observations."""

    policy_kind = "v5_split_encoder"

    def __init__(
        self,
        *,
        obs_dim: int = OBS_V5_DIM,
        action_feature_dim: int = ACTION_FEATURE_DIM,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        base_hidden_dim: int | None = None,
        private_hidden_dim: int | None = None,
        history_hidden_dim: int | None = None,
        global_hidden_dim: int | None = None,
    ):
        super().__init__()
        if int(obs_dim) != OBS_V5_DIM:
            raise ValueError(f"V5ActionConditionedPolicy requires obs_dim={OBS_V5_DIM}")
        self.obs_dim = int(obs_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.encoder_dims = {
            "base": OBS_V1_DIM,
            "global": V5_GLOBAL_DIM,
            "private": PRIVATE_INFO_DIM,
            "history": HISTORY_DIM,
        }

        base_h = int(base_hidden_dim or hidden_dim)
        global_h = int(global_hidden_dim or max(16, hidden_dim // 8))
        private_h = int(private_hidden_dim or max(64, hidden_dim // 2))
        history_h = int(history_hidden_dim or max(64, hidden_dim // 2))

        self.base_encoder = nn.Sequential(nn.Linear(OBS_V1_DIM, base_h), nn.SiLU())
        self.global_encoder = nn.Sequential(nn.Linear(V5_GLOBAL_DIM, global_h), nn.SiLU())
        self.private_encoder = nn.Sequential(nn.Linear(PRIVATE_INFO_DIM, private_h), nn.SiLU())
        self.history_encoder = nn.Sequential(nn.Linear(HISTORY_DIM, history_h), nn.SiLU())
        fused_dim = base_h + global_h + private_h + history_h
        self.state_fuser = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Linear(action_feature_dim, action_hidden_dim)
        self.candidate_scorer = nn.Linear(hidden_dim + action_hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode_state(self, obs):
        import mlx.core as mx

        base_end = OBS_V1_DIM
        global_end = base_end + V5_GLOBAL_DIM
        private_end = global_end + PRIVATE_INFO_DIM
        base = obs[:, :base_end]
        globals_v5 = obs[:, base_end:global_end]
        private = obs[:, global_end:private_end]
        history = obs[:, private_end:]

        state_parts = [
            self.base_encoder(base),
            self.global_encoder(globals_v5),
            self.private_encoder(private),
            self.history_encoder(history),
        ]
        return self.state_fuser(mx.concatenate(state_parts, axis=-1))

    def __call__(self, obs, action_features):
        import mlx.core as mx

        state_emb = self.encode_state(obs)
        batch = obs.shape[0]
        state_bc = mx.expand_dims(state_emb, axis=1)
        state_bc = mx.broadcast_to(state_bc, (batch, 601, self.hidden_dim))

        action_emb = mx.reshape(action_features, (batch * 601, self.action_feature_dim))
        action_emb = self.action_encoder(action_emb)
        action_emb = mx.reshape(action_emb, (batch, 601, self.action_hidden_dim))

        joint = mx.concatenate([state_bc, action_emb], axis=-1)
        joint = mx.reshape(joint, (batch * 601, self.hidden_dim + self.action_hidden_dim))
        logits = mx.reshape(self.candidate_scorer(joint), (batch, 601))
        value = self.value_head(state_emb).squeeze(-1)
        return logits, value


def create_v5_policy(
    *,
    policy_kind: str = "v5_split_encoder",
    obs_dim: int = OBS_V5_DIM,
    action_feature_dim: int = ACTION_FEATURE_DIM,
    hidden_dim: int = 256,
    action_hidden_dim: int = 128,
    **kwargs: Any,
):
    if policy_kind == "v5_split_encoder":
        return V5ActionConditionedPolicy(
            obs_dim=obs_dim,
            action_feature_dim=action_feature_dim,
            hidden_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            **kwargs,
        )
    if policy_kind == "baseline_mlp":
        from ai.train_v2.model_mlx import ActionConditionedPolicy

        return ActionConditionedPolicy(
            obs_dim=obs_dim,
            action_feature_dim=action_feature_dim,
            hidden_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
        )
    raise ValueError("policy_kind must be v5_split_encoder or baseline_mlp")


__all__ = ["V5ActionConditionedPolicy", "V5PolicyShape", "create_v5_policy"]
