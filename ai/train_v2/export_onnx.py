"""
ONNX export toolchain for TrainV2 PPO checkpoints.
Converts MLX model weights → PyTorch mirror → ONNX.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

MODEL_VERSION_ONNX = "classic_action_conditioned_onnx_v1"


class TorchActionConditionedPolicy(torch.nn.Module):
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

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.action_encoder = torch.nn.Linear(action_feature_dim, action_hidden_dim)
        self.candidate_scorer = torch.nn.Linear(
            hidden_dim + action_hidden_dim, 1
        )
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observation: torch.Tensor,
        action_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder(observation)

        B = state_emb.shape[0]
        state_bc = state_emb.unsqueeze(1).expand(-1, 601, -1)

        af_flat = action_features.reshape(B * 601, self.action_feature_dim)
        action_emb = self.action_encoder(af_flat)
        action_emb = action_emb.reshape(B, 601, self.action_hidden_dim)

        joint = torch.cat([state_bc, action_emb], dim=-1)
        joint_flat = joint.reshape(B * 601, -1)
        raw_logits = self.candidate_scorer(joint_flat)
        logits = raw_logits.reshape(B, 601)

        value = self.value_head(state_emb).squeeze(-1)
        return logits, value


_WEIGHT_MAP = [
    ("state_encoder.layers.0", "state_encoder[0]"),
    ("state_encoder.layers.2", "state_encoder[2]"),
    ("action_encoder", "action_encoder"),
    ("candidate_scorer", "candidate_scorer"),
    ("value_head", "value_head"),
]


def load_torch_from_mlx_checkpoint(
    checkpoint_path: str,
    *,
    hidden_dim: int | None = None,
    action_hidden_dim: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    loaded = dict(np.load(checkpoint_path, allow_pickle=True))

    meta_raw = loaded.pop("__meta__", None)
    metadata = {}
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

    cfg = metadata.get("config", {})
    hd = hidden_dim or cfg.get("hidden_dim", 256)
    ahd = action_hidden_dim or cfg.get("action_hidden_dim", 128)

    model = TorchActionConditionedPolicy(
        obs_dim=1456,
        action_feature_dim=171,
        hidden_dim=hd,
        action_hidden_dim=ahd,
    )

    for mlx_key, torch_key in _WEIGHT_MAP:
        w_mlx = loaded.get(f"{mlx_key}.weight")
        b_mlx = loaded.get(f"{mlx_key}.bias")
        torch_module = _resolve_torch_param(model, torch_key)

        if w_mlx is not None and hasattr(torch_module, "weight"):
            t = torch.from_numpy(w_mlx).float()
            if t.shape != torch_module.weight.shape:
                t = t.T
            torch_module.weight.data.copy_(t)

        if b_mlx is not None and hasattr(torch_module, "bias") and torch_module.bias is not None:
            torch_module.bias.data.copy_(torch.from_numpy(b_mlx).float())

    return model, metadata


def _resolve_torch_param(model, key: str):
    parts = key.split("[")
    obj = model
    for part in parts:
        part = part.rstrip("]")
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    opset: int = 17,
    placement_mode: str | None = None,
) -> str:
    model, metadata = load_torch_from_mlx_checkpoint(checkpoint_path)
    model.eval()
    effective_placement_mode = placement_mode or metadata.get("config", {}).get("placement_mode", "append_only")

    dummy_obs = torch.randn(1, 1456)
    dummy_af = torch.randn(1, 601, 171)

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
            output_names=["logits", "value"],
            dynamic_axes={
                "observation": {0: "batch"},
                "action_features": {0: "batch"},
                "logits": {0: "batch"},
                "value": {0: "batch"},
            },
            dynamo=False,
        )

    sidecar = {
        "model_version": MODEL_VERSION_ONNX,
        "source_checkpoint": str(Path(checkpoint_path).absolute()),
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "placement_mode": effective_placement_mode,
        "inputs": ["observation", "action_features"],
        "outputs": ["logits", "value"],
        "config": metadata.get("config", {}),
    }

    json_path = str(out) + ".json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    return str(out)


def _main():
    parser = argparse.ArgumentParser(description="Export MLX PPO checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--placement-mode", default=None, choices=["append_only", "full"])
    args = parser.parse_args()

    result = export_checkpoint_to_onnx(
        args.checkpoint, args.output, opset=args.opset, placement_mode=args.placement_mode
    )
    print(f"Exported: {result}")
    print(f"Sidecar:  {result}.json")


if __name__ == "__main__":
    _main()
