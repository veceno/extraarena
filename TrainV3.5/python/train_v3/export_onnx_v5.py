"""V5 ONNX export toolchain (Block E1 / E-E1, E-E10, E-E13).

This is the V5 ONNX exporter -- a NEW file distinct from the V4-frozen
``ai/train_v2/export_onnx.py``. The V4 exporter is LIVE-tracked and is NOT
edited here; this module mirrors its structure (Torch mirror of the MLX
policy + WEIGHT_MAP + npz loader + torch.onnx.export + sidecar JSON + CLI)
but targets the V5 split-encoder architecture.

What it does
-------------

* ``TorchV5ActionConditionedPolicy`` -- a pure-torch ``torch.nn.Module`` mirror
  of ``v5_policy.V5ActionConditionedPolicy``. Mirroring in torch (rather than
  exporting the MLX module directly) keeps the exporter MLX-independent: the
  loader reads the ``.npz`` checkpoint produced by
  ``ai.train_v2.model_mlx.save_checkpoint`` directly via ``np.load`` and copies
  the weights into the torch mirror, exactly as the V4 exporter does.

* The forward returns a 3-tuple ``(logits, value, mana_draw_logit)`` -- the V5
  contract. ``logits`` is ``[B, 601]`` over the FROZEN 601 candidate space
  (``MAX_CANDIDATE_ACTIONS=601`` stays frozen; ``mana_draw`` is NEVER a 602nd
  candidate). ``value`` and ``mana_draw_logit`` are each ``[B, 1]`` -- the torch
  mirror deliberately does NOT squeeze (unlike the MLX policy which squeezes
  both to ``[B]`` at ``v5_policy.py:137,140``) so the ONNX output rank matches
  the E1 test ``[1, 1]`` assertions. The E2 parity test aligns shapes (squeeze
  onnx / unsqueeze mlx) before comparing; shape alignment is an E2 concern,
  not E1. The exported ONNX emits the RAW ``mana_draw_logit``; the legal mask
  is applied at inference time by the prod ``_get_action_v5`` path /
  ``V5RlhfAdapter.select_action``.

LIFO V5-detector dependency (E5)
--------------------------------

The V5 sidecar ``model_version`` is ``"v5_split_encoder_onnx_v1"`` -- DISTINCT
from the V4 string ``"classic_action_conditioned_onnx_v1"``. BUT a distinct
``model_version`` ALONE does NOT prevent the V4 ``_sidecar_kind_detector``
(``policy_adapters.py:219-248``) from matching the V5 sidecar via the
``inputs``/``action_feature_dim`` OR-branches (``:240-241``) -- the V4
detector would return ``"action_onnx"`` for a V5 sidecar if it ran first.
V5 routing relies on the LIFO V5 detector (``register_detector``, ``:289``)
returning ``"v5"`` (never ``None``). E1 does NOT register the detector -- E5
does. A missing/unregistered V5 detector is therefore a regression that E2
catches (the V4 detector matching V5 is the hard-LIFO-dependency canary).

Frozen-classic / V4-unchanged guard
------------------------------------

``ai/train_v2/export_onnx.py`` (V4 exporter), ``ai/train_v2/model_mlx.py``
(npz format), ``ai/train_v2/onnx_policy.py`` (V4 ONNX runtime adapter),
``v5_policy.py``, ``contracts.py``, ``v5_card_shape_v1.py`` and
``warm_start_v5.py`` are all READ-ONLY with respect to this component. E1
adds a NEW file under ``train_v3`` and imports only the READ-ONLY
``contracts`` constants and the ``CARD_SHAPE_VERSION`` re-export from
``ai.train_v2.v5_card_shape_v1``. It does NOT import ``v5_policy`` (the torch
mirror stands alone -- importing MLX would make the module MLX-dependent) and
does NOT import ``model_mlx`` (the loader reads the npz directly via
``np.load``, mirroring ``export_onnx.py:83``).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .contracts import (
    ACTION_FEATURE_DIM,
    HISTORY_DIM,
    MAX_CANDIDATE_ACTIONS,
    OBS_V1_DIM,
    OBS_V5_DIM,
    PRIVATE_INFO_DIM,
    V5_GLOBAL_DIM,
)
from ai.train_v2.v5_card_shape_v1 import CARD_SHAPE_VERSION

MODEL_VERSION_V5_ONNX = "v5_split_encoder_onnx_v1"


class TorchV5ActionConditionedPolicy(torch.nn.Module):
    """Pure-torch mirror of ``v5_policy.V5ActionConditionedPolicy`` for ONNX export.

    The architecture mirrors ``v5_policy.py:65-85`` (split encoders + fuser +
    parallel mana_draw head). The forward returns a 3-tuple and deliberately
    keeps ``value`` and ``mana_draw_logit`` as ``[B, 1]`` (no squeeze) so the
    ONNX output rank matches the E1 ``[1, 1]`` assertions; the MLX policy
    squeezes both to ``[B]`` (``v5_policy.py:137,140``), but the torch mirror
    must NOT -- shape alignment is delegated to E2.
    """

    def __init__(
        self,
        *,
        obs_dim: int = OBS_V5_DIM,
        action_feature_dim: int = ACTION_FEATURE_DIM,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        base_hidden_dim: Optional[int] = None,
        private_hidden_dim: Optional[int] = None,
        history_hidden_dim: Optional[int] = None,
        global_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if int(obs_dim) != OBS_V5_DIM:
            raise ValueError(
                f"TorchV5ActionConditionedPolicy requires obs_dim={OBS_V5_DIM}, got {obs_dim}"
            )
        self.obs_dim = int(obs_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim)

        # Per-encoder hidden dims -- mirror v5_policy.py:60-63 EXACTLY.
        base_h = int(base_hidden_dim or hidden_dim)
        global_h = int(global_hidden_dim or max(16, hidden_dim // 8))
        private_h = int(private_hidden_dim or max(64, hidden_dim // 2))
        history_h = int(history_hidden_dim or max(64, hidden_dim // 2))
        self.base_h = base_h
        self.global_h = global_h
        self.private_h = private_h
        self.history_h = history_h
        fused_dim = base_h + global_h + private_h + history_h

        self.base_encoder = torch.nn.Sequential(
            torch.nn.Linear(OBS_V1_DIM, base_h),
            torch.nn.SiLU(),
        )
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Linear(V5_GLOBAL_DIM, global_h),
            torch.nn.SiLU(),
        )
        self.private_encoder = torch.nn.Sequential(
            torch.nn.Linear(PRIVATE_INFO_DIM, private_h),
            torch.nn.SiLU(),
        )
        self.history_encoder = torch.nn.Sequential(
            torch.nn.Linear(HISTORY_DIM, history_h),
            torch.nn.SiLU(),
        )
        self.state_fuser = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.action_encoder = torch.nn.Linear(action_feature_dim, action_hidden_dim)
        self.candidate_scorer = torch.nn.Linear(hidden_dim + action_hidden_dim, 1)
        self.state_action_query = torch.nn.Linear(hidden_dim, action_hidden_dim)
        self.state_action_gate = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.state_action_gate.weight)
        self.state_action_interaction_scale = float(action_hidden_dim) ** -0.5
        self.state_action_gate_cap = 0.1
        self.value_head = torch.nn.Linear(hidden_dim, 1)
        # Parallel binary mana_draw head (spec section 0.89 gamma). NOT a 602nd candidate --
        # MAX_CANDIDATE_ACTIONS=601 stays frozen. Reads the same fused state_emb
        # as value_head; the legal mask is applied at inference time.
        self.mana_draw_head = torch.nn.Linear(hidden_dim, 1)

    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        # Split obs along the LAST dim EXACTLY as v5_policy.py:87-104.
        base_end = OBS_V1_DIM                       # 1456
        global_end = base_end + V5_GLOBAL_DIM        # 1488
        private_end = global_end + PRIVATE_INFO_DIM  # 3888
        base = obs[:, :base_end]
        globals_v5 = obs[:, base_end:global_end]
        private = obs[:, global_end:private_end]
        history = obs[:, private_end:]

        # Encode each split, then concat in the v5_policy.py:98-103 order
        # [base, globals_v5, private, history], then fuse. The MLX policy
        # concatenates the *encoded* parts (each encoder applied before the
        # cat); we mirror that exactly -- the slices above are the RAW splits,
        # the cat below is over the ENCODED embeddings.
        state_parts = [
            self.base_encoder(base),
            self.global_encoder(globals_v5),
            self.private_encoder(private),
            self.history_encoder(history),
        ]
        fused = torch.cat(state_parts, dim=-1)
        return self.state_fuser(fused)

    def forward(
        self,
        observation: torch.Tensor,
        action_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 3-tuple contract: (logits[B,601], value[B,1], mana_draw_logit[B,1]).
        # NO mana_draw_legal mask arg -- the ONNX emits the RAW mana_draw_logit;
        # the legal mask is applied at inference by the prod _get_action_v5 path
        # / V5RlhfAdapter.select_action.
        state_emb = self.encode_state(observation)
        B = state_emb.shape[0]

        # Broadcast state over the 601 candidate axis.
        state_bc = state_emb.unsqueeze(1).expand(-1, MAX_CANDIDATE_ACTIONS, -1)

        af_flat = action_features.reshape(B * MAX_CANDIDATE_ACTIONS, self.action_feature_dim)
        action_emb = self.action_encoder(af_flat).reshape(
            B, MAX_CANDIDATE_ACTIONS, self.action_hidden_dim
        )

        joint = torch.cat([state_bc, action_emb], dim=-1)
        joint_flat = joint.reshape(B * MAX_CANDIDATE_ACTIONS, -1)
        raw_logits = self.candidate_scorer(joint_flat)
        legacy_logits = raw_logits.reshape(B, MAX_CANDIDATE_ACTIONS)
        state_query = torch.tanh(self.state_action_query(state_emb)).unsqueeze(1)
        bounded_action_emb = torch.tanh(action_emb)
        interaction_gate = self.state_action_gate_cap * torch.tanh(
            self.state_action_gate.weight[0, 0]
        )
        interaction_logits = torch.sum(state_query * bounded_action_emb, dim=-1)
        logits = (
            legacy_logits
            + interaction_logits * self.state_action_interaction_scale * interaction_gate
        )

        # value and mana_draw_logit are PARALLEL scalar heads on state_emb.
        # Kept as [B, 1] (do NOT squeeze) so the ONNX output rank matches the
        # E1 test [1, 1] assertions. E2 aligns shapes (squeeze onnx / unsqueeze
        # mlx) before the parity compare.
        value = self.value_head(state_emb)
        mana_draw_logit = self.mana_draw_head(state_emb)
        return logits, value, mana_draw_logit


# V5 param-level weight map. Each entry is (mlx_npz_key_prefix, torch_attr_path).
# The mlx npz keys are the MLX nn.Sequential/Linear param dotted names produced
# by model_mlx.flatten_params (nn.Sequential uses ``.layers.N`` in MLX). The
# torch_attr_path uses BRACKET indexing for Sequential layers (resolved by
# _resolve_torch_param). The .weight/.bias suffixes are appended in the loader.
# torch.nn.Sequential param names are ``base_encoder.0.weight`` (index, NOT
# ``.layers.0``); _resolve_torch_param(model, "base_encoder[0]") resolves the
# bracket index to model.base_encoder[0]. The mlx npz key IS
# ``base_encoder.layers.0`` (MLX nn.Sequential uses ``.layers.N``). This
# asymmetry is correct -- the map bridges the two naming schemes.
V5_WEIGHT_MAP = [
    ("base_encoder.layers.0", "base_encoder[0]"),
    ("global_encoder.layers.0", "global_encoder[0]"),
    ("private_encoder.layers.0", "private_encoder[0]"),
    ("history_encoder.layers.0", "history_encoder[0]"),
    ("state_fuser.layers.0", "state_fuser[0]"),
    ("state_fuser.layers.2", "state_fuser[2]"),
    ("action_encoder", "action_encoder"),
    ("candidate_scorer", "candidate_scorer"),
    ("state_action_query", "state_action_query"),
    ("state_action_gate", "state_action_gate"),
    ("value_head", "value_head"),
    ("mana_draw_head", "mana_draw_head"),
]


def _resolve_torch_param(model, key: str):
    """Resolve a bracket-indexed attr path, e.g. ``state_fuser[2]`` -> model.state_fuser[2].

    Copied from export_onnx.py:127-136 (kept self-contained under train_v3 so
    this module does not import the V4-frozen V4 exporter).
    """
    parts = key.split("[")
    obj = model
    for part in parts:
        part = part.rstrip("]")
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def load_v5_torch_from_mlx_checkpoint(
    checkpoint_path: str,
    *,
    hidden_dim: Optional[int] = None,
    action_hidden_dim: Optional[int] = None,
    base_hidden_dim: Optional[int] = None,
    private_hidden_dim: Optional[int] = None,
    history_hidden_dim: Optional[int] = None,
    global_hidden_dim: Optional[int] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Load a V5 MLX ``.npz`` checkpoint into the torch mirror.

    Mirrors ``export_onnx.py:77-124`` (np.load + __meta__ JSON parse + WEIGHT_MAP
    copy with the .T transpose fallback). Rejects a V4 npz misrouted here via
    the obs_dim assertion (primary) plus a belt-and-braces key check.
    """
    loaded = dict(np.load(checkpoint_path, allow_pickle=True))

    # Parse __meta__ -- mirror the export_onnx.py:85-97 fallback chain.
    meta_raw = loaded.pop("__meta__", None)
    metadata: Dict[str, Any] = {}
    if meta_raw is not None:
        if hasattr(meta_raw, "tobytes"):
            metadata = json.loads(meta_raw.tobytes().decode("utf-8"))
        elif hasattr(meta_raw, "item"):
            v = meta_raw.item()
            if hasattr(v, "decode"):
                metadata = json.loads(v.decode("utf-8"))
            else:
                metadata = json.loads(str(v))
        else:
            metadata = json.loads(str(meta_raw))

    # V4-REJECTION (load-bearing, E-E1 test (e)). Assert the checkpoint is V5,
    # NOT V4. The V4 npz has obs_dim 1456 in metadata + keys state_encoder.layers.0
    # (NOT base_encoder.layers.0).
    meta_obs_dim = metadata.get("obs_dim")
    if meta_obs_dim is not None and int(meta_obs_dim) != OBS_V5_DIM:
        raise ValueError(
            f"load_v5_torch_from_mlx_checkpoint: expected V5 obs_dim={OBS_V5_DIM}, "
            f"got {meta_obs_dim} (this looks like a V4 checkpoint -- use the V4 exporter)"
        )
    # Belt-and-braces: a V4 npz that omits obs_dim but carries the V4 state_encoder key.
    if "base_encoder.layers.0.weight" not in loaded and "state_encoder.layers.0.weight" in loaded:
        raise ValueError(
            "load_v5_torch_from_mlx_checkpoint: checkpoint has state_encoder.layers.0 "
            "but no base_encoder.layers.0 -- this is a V4 checkpoint, not V5"
        )

    cfg = metadata.get("config", {})
    hd = hidden_dim or cfg.get("hidden_dim", 256)
    ahd = action_hidden_dim or cfg.get("action_hidden_dim", 128)
    base_hd = base_hidden_dim or cfg.get("base_hidden_dim", None)
    private_hd = private_hidden_dim or cfg.get("private_hidden_dim", None)
    history_hd = history_hidden_dim or cfg.get("history_hidden_dim", None)
    global_hd = global_hidden_dim or cfg.get("global_hidden_dim", None)

    model = TorchV5ActionConditionedPolicy(
        obs_dim=OBS_V5_DIM,
        action_feature_dim=ACTION_FEATURE_DIM,
        hidden_dim=hd,
        action_hidden_dim=ahd,
        base_hidden_dim=base_hd,
        private_hidden_dim=private_hd,
        history_hidden_dim=history_hd,
        global_hidden_dim=global_hd,
    )

    for mlx_key, torch_key in V5_WEIGHT_MAP:
        w_mlx = loaded.get(f"{mlx_key}.weight")
        b_mlx = loaded.get(f"{mlx_key}.bias")
        torch_module = _resolve_torch_param(model, torch_key)

        if w_mlx is not None and hasattr(torch_module, "weight"):
            t = torch.from_numpy(w_mlx).float()
            if t.shape != torch_module.weight.shape:
                # .T transpose fallback (mirror export_onnx.py:115-119): MLX
                # Linear stores weight as [out, in] matching torch, but a shape
                # mismatch triggers the transpose for safety.
                t = t.T
            torch_module.weight.data.copy_(t)

        if (
            b_mlx is not None
            and hasattr(torch_module, "bias")
            and torch_module.bias is not None
        ):
            torch_module.bias.data.copy_(torch.from_numpy(b_mlx).float())

    return model, metadata


def export_v5_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    opset: int = 17,
    placement_mode: Optional[str] = None,
) -> str:
    """Export a V5 MLX checkpoint to ONNX with the V5 3-output contract + sidecar.

    Mirrors ``export_onnx.py:139-172`` (torch.onnx.export under
    warnings.catch_warnings) but with the V5 3-tuple forward, the 7128/601/171
    dummy shapes, and the V5 sidecar fields per E-E13.
    """
    model, metadata = load_v5_torch_from_mlx_checkpoint(checkpoint_path)
    model.eval()
    effective_placement_mode = placement_mode or metadata.get("config", {}).get(
        "placement_mode", "append_only"
    )

    dummy_obs = torch.randn(1, OBS_V5_DIM)
    dummy_af = torch.randn(1, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (dummy_obs, dummy_af),
            str(out),
            opset_version=opset,
            input_names=["observation", "action_features"],
            output_names=["logits", "value", "mana_draw_logit"],
            dynamic_axes={
                "observation": {0: "batch"},
                "action_features": {0: "batch"},
                "logits": {0: "batch"},
                "value": {0: "batch"},
                "mana_draw_logit": {0: "batch"},
            },
            dynamo=False,
        )

    # V5 SIDECAR (E-E13). ADAPTED from export_onnx.py:174-188 with NEW fields:
    # model_version "v5_split_encoder_onnx_v1" (distinct from the V4 string),
    # the 3-output list, mana_draw_head flag, format "v5", and card_shape_version.
    sidecar = {
        "model_version": MODEL_VERSION_V5_ONNX,
        "source_checkpoint": str(Path(checkpoint_path).absolute()),
        "obs_dim": OBS_V5_DIM,
        "action_feature_dim": ACTION_FEATURE_DIM,
        "max_candidate_actions": MAX_CANDIDATE_ACTIONS,
        "placement_mode": effective_placement_mode,
        "inputs": ["observation", "action_features"],
        "outputs": ["logits", "value", "mana_draw_logit"],
        "mana_draw_head": True,
        "state_action_interaction": "gated_bilinear_query_cap01_v1",
        "format": "v5",
        "card_shape_version": CARD_SHAPE_VERSION,
        "config": metadata.get("config", {}),
    }

    json_path = str(out) + ".json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    return str(out)


def _main():
    parser = argparse.ArgumentParser(
        description="Export a V5 MLX PPO checkpoint to ONNX (3-output contract)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to V5 .npz checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--placement-mode",
        default=None,
        choices=["append_only", "full"],
    )
    args = parser.parse_args()

    result = export_v5_checkpoint_to_onnx(
        args.checkpoint,
        args.output,
        opset=args.opset,
        placement_mode=args.placement_mode,
    )
    print(f"Exported: {result}")
    print(f"Sidecar:  {result}.json")


if __name__ == "__main__":
    _main()


__all__ = [
    "TorchV5ActionConditionedPolicy",
    "V5_WEIGHT_MAP",
    "load_v5_torch_from_mlx_checkpoint",
    "export_v5_checkpoint_to_onnx",
]