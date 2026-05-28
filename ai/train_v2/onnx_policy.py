"""
ONNX inference policy for TrainV2 — same interface as MlxPolicy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai.train_v2.policies import Policy
from ai.train_v2.classic_actions_v1 import build_action_mask, encode_action_features


class OnnxActionPolicy(Policy):
    def __init__(
        self,
        onnx_path: str,
        *,
        mode: str = "argmax",
        temperature: float = 1.0,
        seed: int = 0,
        verify_mask: bool = True,
    ):
        import onnxruntime as ort

        if mode not in ("argmax", "sample"):
            raise ValueError(f"mode must be 'argmax' or 'sample', got '{mode}'")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._mode = mode
        self._temperature = temperature
        self._seed = seed
        self._invalid_fallbacks = 0
        sidecar_path = Path(str(onnx_path) + ".json")
        sidecar = {}
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self._placement_mode = sidecar.get("placement_mode", "append_only")
        sidecar_config = sidecar.get("config") if isinstance(sidecar.get("config"), dict) else {}
        self._include_preview = bool(
            sidecar.get(
                "include_preview_features",
                sidecar_config.get("include_preview_features", True),
            )
        )
        self._verify_mask = verify_mask

        stem = Path(onnx_path).stem
        self.name = f"onnx_{mode}_{stem}"

    def reset(self, seed: int):
        self._invalid_fallbacks = 0
        if self._mode == "sample":
            np.random.seed(seed + self._seed)

    def select_action(self, env, player_id: int) -> int:
        obs = env.observe(player_id)
        state = env.clone_state()
        mask = build_action_mask(
            state,
            player_id,
            verify_mask=self._verify_mask,
            placement_mode=self._placement_mode,
        )
        af = encode_action_features(
            state,
            player_id,
            include_preview=self._include_preview,
            verify_mask=self._verify_mask,
            placement_mode=self._placement_mode,
            mask=mask,
        )

        obs_batch = obs[np.newaxis, :].astype(np.float32)
        af_batch = af[np.newaxis, :, :].astype(np.float32)

        outputs = self._session.run(
            ["logits", "value"],
            {"observation": obs_batch, "action_features": af_batch},
        )

        logits = outputs[0][0]

        mlogits = np.where(mask.astype(bool), logits, -1e9).astype(np.float32)

        if self._mode == "sample":
            scaled = mlogits / self._temperature
            shifted = scaled - np.max(scaled)
            exps = np.exp(shifted)
            probs = exps / (np.sum(exps) + 1e-10)
            probs *= mask
            probs /= probs.sum() + 1e-10
            aid = int(np.random.choice(len(probs), p=probs))
        else:
            aid = int(np.argmax(mlogits))

        if mask[aid] != 1.0:
            self._invalid_fallbacks += 1
            legal = [int(i) for i, v in enumerate(mask) if v == 1.0]
            aid = legal[0] if legal else 0

        return aid
