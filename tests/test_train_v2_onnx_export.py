"""
Tests for ONNX export and ONNX inference policy (Task 07).
"""
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pytest

from ai.train_v2.classic_rl_env import ClassicRLEnv

from ai.train_v2.export_onnx import (
    TorchActionConditionedPolicy,
    load_torch_from_mlx_checkpoint,
    export_checkpoint_to_onnx,
)
from ai.train_v2.train_ppo import PPOConfig, train


def _train_tiny_ckpt(ckpt_dir, seed=42):
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    config = PPOConfig(
        total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
        hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
        seed=seed, checkpoint_dir=ckpt_dir,
    )
    return train(config)["checkpoint_path"]


@pytest.fixture
def tiny_ckpt():
    ckpt_dir = "/tmp/_t07_export_fixture"
    path = _train_tiny_ckpt(ckpt_dir)
    yield path
    shutil.rmtree(ckpt_dir, ignore_errors=True)


class TestTorchMirror:
    def test_torch_mirror_forward_shapes(self):
        import torch
        m = TorchActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        m.eval()

        obs = torch.randn(1, 1456)
        af = torch.randn(1, 601, 171)

        with torch.no_grad():
            logits, value = m(obs, af)

        assert logits.shape == (1, 601), f"logits: {logits.shape}"
        assert value.shape == (1,), f"value: {value.shape}"

    def test_torch_matches_mlx_outputs(self, tiny_ckpt):
        import torch

        torch_model, _ = load_torch_from_mlx_checkpoint(tiny_ckpt)
        torch_model.eval()

        from ai.train_v2.model_mlx import ActionConditionedPolicy, load_checkpoint
        mlx_model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(mlx_model.parameters())
        load_checkpoint(tiny_ckpt, mlx_model)

        obs_np = np.random.randn(2, 1456).astype(np.float32)
        af_np = np.random.randn(2, 601, 171).astype(np.float32)

        obs_mx = mx.array(obs_np)
        af_mx = mx.array(af_np)
        logits_mlx, value_mlx = mlx_model(obs_mx, af_mx)
        mx.eval(logits_mlx, value_mlx)

        with torch.no_grad():
            logits_torch, value_torch = torch_model(
                torch.from_numpy(obs_np), torch.from_numpy(af_np)
            )

        assert torch.allclose(
            logits_torch, torch.from_numpy(np.array(logits_mlx)), atol=1e-4
        ), "logits mismatch"
        assert torch.allclose(
            value_torch, torch.from_numpy(np.array(value_mlx)), atol=1e-4
        ), "value mismatch"


class TestONNXExport:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_export_writes_sidecar_metadata(self, tiny_ckpt):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            export_checkpoint_to_onnx(tiny_ckpt, onnx_path, opset=17)

            assert Path(onnx_path).exists()
            sidecar_path = Path(onnx_path + ".json")
            assert sidecar_path.exists()

            meta = json.loads(sidecar_path.read_text())
            assert meta["obs_dim"] == 1456
            assert meta["action_feature_dim"] == 171
            assert meta["max_candidate_actions"] == 601
            assert "observation" in meta["inputs"]
            assert "action_features" in meta["inputs"]
            assert "logits" in meta["outputs"]
            assert "value" in meta["outputs"]
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_export_cli_smoke(self, tiny_ckpt):
        import subprocess, sys

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ai.train_v2.export_onnx",
                 "--checkpoint", tiny_ckpt, "--output", onnx_path, "--opset", "17"],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
            assert Path(onnx_path).exists()
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnxruntime_matches_mlx(self, tiny_ckpt):
        import onnxruntime as ort

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            export_checkpoint_to_onnx(tiny_ckpt, onnx_path, opset=17)

            from ai.train_v2.model_mlx import ActionConditionedPolicy, load_checkpoint
            mlx_model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
            mx.eval(mlx_model.parameters())
            load_checkpoint(tiny_ckpt, mlx_model)

            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            obs_np = np.random.randn(2, 1456).astype(np.float32)
            af_np = np.random.randn(2, 601, 171).astype(np.float32)

            logits_mlx, value_mlx = mlx_model(mx.array(obs_np), mx.array(af_np))
            mx.eval(logits_mlx, value_mlx)

            ort_out = sess.run(
                ["logits", "value"],
                {"observation": obs_np, "action_features": af_np},
            )
            logits_ort = ort_out[0]
            value_ort = np.array(ort_out[1]).squeeze()

            assert np.allclose(logits_ort, np.array(logits_mlx), atol=1e-4), "logits mismatch"
            assert np.allclose(value_ort, np.array(value_mlx), atol=1e-4), "value mismatch"
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)


class TestOnnxPolicy:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_selects_legal_action(self, tiny_ckpt):
        from ai.train_v2.onnx_policy import OnnxActionPolicy

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            export_checkpoint_to_onnx(tiny_ckpt, onnx_path, opset=17)
            p = OnnxActionPolicy(onnx_path, mode="argmax")

            env = ClassicRLEnv(seed=42)
            env.reset(seed=100)

            for _ in range(10):
                cp = env.current_player_id()
                mask = env.action_mask(cp)
                aid = p.select_action(env, cp)
                assert mask[aid] == 1.0, f"action {aid} illegal"
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_mlx_argmax_onnx_argmax_match(self, tiny_ckpt):
        from ai.train_v2.onnx_policy import OnnxActionPolicy
        from ai.train_v2.ppo_eval import MlxPolicy, load_mlx_policy

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            export_checkpoint_to_onnx(tiny_ckpt, onnx_path, opset=17)

            mlx_pol = load_mlx_policy(tiny_ckpt, hidden_dim=32, action_hidden_dim=16, mode="argmax")
            onnx_pol = OnnxActionPolicy(onnx_path, mode="argmax")

            env = ClassicRLEnv(seed=42)
            env.reset(seed=777)

            # MLX (Metal float32) и ONNX-runtime (CPU float32) численно расходятся
            # на ~1e-8. Если в каком-то шаге две легальные атаки имеют логиты,
            # различающиеся меньше этого шума, argmax у MLX и ONNX может
            # разминуться — и траектории за 5 шагов полностью разойдутся. Это
            # НЕ баг экспорта: расхождение в пределах float-шума на неразрешимом
            # ничьем логите. Поэтому проверяем экспорт-фиделити численно-корректно:
            #   (1) логиты MLX и ONNX близки (LOGIT_TOL много больше float-шума
            #       ~1e-8 и много меньше расхождения при реальном баге экспорта
            #       O(0.1–1)) — ловит любой реальный расхождение, включая
            #       вырожденный (все-нули) ONNX, у которого внутренний margin
            #       ничтожен, но |mlx - onnx| огромен;
            #   (2) когда top-2 margin у MLX больше LOGIT_TOL (ясный победитель),
            #       argmax MLX и ONNX обязаны совпадать — поведенческая проверка.
            LOGIT_TOL = 1e-3

            def _masked_argmax_with_margin(logits, mask):
                arr = np.asarray(logits, dtype=np.float32)
                m = np.asarray(mask, dtype=bool)
                legal = np.where(m)[0]
                if len(legal) == 0:
                    return 0, np.inf
                legal_logits = arr[legal]
                order = np.argsort(legal_logits)[::-1]
                top_idx = int(legal[order[0]])
                second = legal_logits[order[1]] if len(legal) > 1 else -np.inf
                margin = float(legal_logits[order[0]] - second)
                return top_idx, margin

            for step in range(5):
                cp = env.current_player_id()
                obs = env.observe(cp)
                mask = env.action_mask(cp)
                af = env.action_features(cp)

                logits_mlx, _ = mlx_pol._model(
                    mx.array(obs[None, :]), mx.array(af[None, :, :])
                )
                mx.eval(logits_mlx)
                logits_mlx_np = np.asarray(logits_mlx[0], dtype=np.float32)
                a_mlx, margin_mlx = _masked_argmax_with_margin(logits_mlx_np, mask)

                outs = onnx_pol._session.run(
                    ["logits", "value"],
                    {
                        "observation": np.asarray(obs, dtype=np.float32)[None, :],
                        "action_features": np.asarray(af, dtype=np.float32)[None, :, :],
                    },
                )
                logits_onnx_np = np.asarray(outs[0][0], dtype=np.float32)
                a_onnx, _ = _masked_argmax_with_margin(logits_onnx_np, mask)

                # (1) Численная фиделити экспорта.
                max_logit_diff = float(np.max(np.abs(logits_mlx_np - logits_onnx_np)))
                assert max_logit_diff < LOGIT_TOL, (
                    f"step {step}: |MLX-ONNX| max logit diff = {max_logit_diff:.3e} "
                    f"превышает LOGIT_TOL={LOGIT_TOL:.0e} — реальное расхождение экспорта"
                )
                # (2) Поведенческая проверка при ясном победителе (margin > tol).
                if margin_mlx > LOGIT_TOL and a_mlx != a_onnx:
                    raise AssertionError(
                        f"step {step}: ясный победитель (margin={margin_mlx:.3e}) "
                        f"но argmax расходится: MLX={a_mlx}, ONNX={a_onnx}"
                    )

                _, _, _, _, _ = env.step(a_mlx)
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)

    def test_temperature_guard(self):
        with pytest.raises(ValueError):
            from ai.train_v2.onnx_policy import OnnxActionPolicy
            OnnxActionPolicy("/tmp/nonexistent.onnx", mode="sample", temperature=0)

        with pytest.raises(ValueError):
            from ai.train_v2.onnx_policy import OnnxActionPolicy
            OnnxActionPolicy("/tmp/nonexistent.onnx", mode="sample", temperature=-1)


class TestOnnxSampleSeed:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnx_sample_policy_reset_deterministic(self, tiny_ckpt):
        from ai.train_v2.onnx_policy import OnnxActionPolicy

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            export_checkpoint_to_onnx(tiny_ckpt, onnx_path, opset=17)
            p = OnnxActionPolicy(onnx_path, mode="sample", seed=10)

            env = ClassicRLEnv(seed=42)

            env.reset(seed=100)
            p.reset(100)
            a1 = p.select_action(env, env.current_player_id())

            env.reset(seed=100)
            p.reset(100)
            a2 = p.select_action(env, env.current_player_id())

            assert a1 == a2, f"sample with same seed={100} must be deterministic: {a1} vs {a2}"
        finally:
            Path(onnx_path).unlink(missing_ok=True)
            Path(onnx_path + ".json").unlink(missing_ok=True)
