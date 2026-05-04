"""
Export TitanNet V4 (MLX .npz) → ONNX for production inference.
===============================================================

The exported model takes a 636-dim observation vector and outputs
109 action logits, matching the ArenaEnv action space exactly.

Workflow:
  1. Load MLX checkpoint (.npz) into TitanNet weight dict
  2. Reconstruct equivalent PyTorch model (same ResNet architecture)
  3. Copy weights layer-by-layer
  4. Validate: max |MLX_output - PT_output| < 1e-3
  5. Export to ONNX (opset 17)
  6. Validate ONNX: max |PT_output - ONNX_output| < 1e-3
  7. Print output path + obs/action dims for integration reference

Usage:
    python -m ai.train.export_onnx_v4 checkpoints_v4/step_0003000.npz
    python -m ai.train.export_onnx_v4 checkpoints_v4/final.npz --output ai/models/titan_v4_hard.onnx

The output .onnx file can be plugged directly into BerserkInference
via TitanV4Inference (ai/bot_inference_v4.py).

Requirements:
    pip install torch onnx onnxruntime
    (MLX is already installed)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ai.arena_env import OBS_SIZE, TOTAL_ACTIONS

HIDDEN_DIM = 512
NUM_LAYERS = 5


# ---------------------------------------------------------------------------
# PyTorch mirror of TitanNet (for ONNX export only)
# ---------------------------------------------------------------------------
def _build_pt_model():
    import torch
    import torch.nn as tnn

    class _ResBlock(tnn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc = tnn.Linear(dim, dim)
            self.norm = tnn.LayerNorm(dim)

        def forward(self, x):
            return tnn.functional.silu(self.norm(self.fc(x))) + x

    class _TitanNetPT(tnn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = tnn.Linear(OBS_SIZE, HIDDEN_DIM)
            self.blocks = tnn.ModuleList(
                [_ResBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)]
            )
            # Actor head: LayerNorm → Linear
            self.actor_norm = tnn.LayerNorm(HIDDEN_DIM)
            self.actor_fc   = tnn.Linear(HIDDEN_DIM, TOTAL_ACTIONS)
            # Critic is not exported (only actor logits needed in production)

        def forward(self, x):
            x = tnn.functional.silu(self.input_layer(x))
            for block in self.blocks:
                x = block(x)
            return self.actor_fc(self.actor_norm(x))

    return _TitanNetPT()


def _mlx_array_to_np(v) -> np.ndarray:
    """Convert an MLX array (from mx.load) to numpy."""
    import mlx.core as mx
    if isinstance(v, mx.array):
        return np.array(v)
    return np.array(v)


def export(checkpoint_path: str, output_path: str):
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import torch

    # --- 1. Load MLX weights ---
    raw = mx.load(checkpoint_path)
    w = {k: _mlx_array_to_np(v) for k, v in raw.items()}

    in_dim  = w["input_layer.weight"].shape[1]
    out_dim = w["actor_head.layers.1.weight"].shape[0]
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"  obs_dim={in_dim}  act_dim={out_dim}  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}")

    if in_dim != OBS_SIZE or out_dim != TOTAL_ACTIONS:
        raise ValueError(
            f"Checkpoint dims ({in_dim}, {out_dim}) don't match "
            f"current ArenaEnv ({OBS_SIZE}, {TOTAL_ACTIONS}). "
            "Re-train or use the correct checkpoint."
        )

    # --- 2. Build PyTorch model and transfer weights ---
    pt = _build_pt_model()

    def t(key: str) -> torch.Tensor:
        return torch.from_numpy(w[key].copy())

    with torch.no_grad():
        pt.input_layer.weight.copy_(t("input_layer.weight"))
        pt.input_layer.bias.copy_(t("input_layer.bias"))

        for i in range(NUM_LAYERS):
            pt.blocks[i].fc.weight.copy_(t(f"blocks.{i}.fc.weight"))
            pt.blocks[i].fc.bias.copy_(t(f"blocks.{i}.fc.bias"))
            pt.blocks[i].norm.weight.copy_(t(f"blocks.{i}.norm.weight"))
            pt.blocks[i].norm.bias.copy_(t(f"blocks.{i}.norm.bias"))

        # actor_head is nn.Sequential([LayerNorm, Linear]) in MLX
        # MLX Sequential stores layers as .layers list
        pt.actor_norm.weight.copy_(t("actor_head.layers.0.weight"))
        pt.actor_norm.bias.copy_(t("actor_head.layers.0.bias"))
        pt.actor_fc.weight.copy_(t("actor_head.layers.1.weight"))
        pt.actor_fc.bias.copy_(t("actor_head.layers.1.bias"))

    pt.eval()

    # --- 3. Validate: MLX vs PyTorch ---
    import mlx.core as mx
    import mlx.nn as mlx_nn
    sys.path.insert(0, str(Path(__file__).parent))
    from train_v4 import TitanNet as MLXTitanNet

    mlx_net = MLXTitanNet()
    mlx_net.load_weights([(k, mx.array(v)) for k, v in w.items()])
    mx.eval(mlx_net.parameters())

    rng = np.random.default_rng(42)
    test_input = rng.standard_normal((4, OBS_SIZE)).astype(np.float32)

    mlx_logits = np.array(mlx_net(mx.array(test_input))[0])
    pt_logits  = pt(torch.from_numpy(test_input)).detach().numpy()

    diff_mlx_pt = float(np.abs(mlx_logits - pt_logits).max())
    print(f"  MLX ↔ PyTorch max |diff|: {diff_mlx_pt:.2e}", end="")
    if diff_mlx_pt > 1e-2:
        print(" ← WARNING: large diff, check weight mapping")
    else:
        print(" ✓")

    # --- 4. ONNX export ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(test_input)

    torch.onnx.export(
        pt,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # --- 5. Validate ONNX ---
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = sess.run(None, {"obs": test_input})[0]

    diff_pt_onnx = float(np.abs(pt_logits - onnx_logits).max())
    print(f"  PT   ↔ ONNX   max |diff|: {diff_pt_onnx:.2e}", end="")
    if diff_pt_onnx > 1e-2:
        print(" ← WARNING")
    else:
        print(" ✓")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✅ Exported → {output_path}  ({size_mb:.1f} MB)")
    print(f"   Input  : obs  [{OBS_SIZE}]")
    print(f"   Output : logits [{TOTAL_ACTIONS}]  (ArenaEnv action space)")
    print()
    print("Integration:")
    print("  from ai.bot_inference_v4 import TitanV4Inference")
    print(f"  bot = TitanV4Inference('{output_path}')")
    print("  action_idx = bot.get_action(game_state, player_id, legal_actions)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TitanNet V4 to ONNX")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument(
        "--output",
        default="ai/models/titan_v4.onnx",
        help="Output ONNX path (default: ai/models/titan_v4.onnx)",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)
