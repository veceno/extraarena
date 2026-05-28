"""
Tests for resume training and presets (Task 13).
"""
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from ai.train_v2.train_ppo import (
    PPOConfig,
    make_config_from_preset,
    train,
    TRAIN_PRESETS,
)
from ai.train_v2.model_mlx import load_checkpoint, ActionConditionedPolicy
from ai.train_v2.experiment import ExperimentConfig, run_experiment


class TestMakeConfigFromPreset:
    def test_make_config_from_preset_smoke(self):
        cfg = make_config_from_preset("smoke")
        assert cfg.total_updates == 1
        assert cfg.hidden_dim == 32
        assert cfg.action_hidden_dim == 16
        assert cfg.minibatch_size == 8
        assert cfg.epochs == 1

    def test_make_config_from_preset_with_overrides(self):
        cfg = make_config_from_preset("smoke", total_updates=3)
        assert cfg.total_updates == 3
        assert cfg.hidden_dim == 32

    def test_make_config_from_preset_none_overrides_not_applied(self):
        cfg = make_config_from_preset("smoke", total_updates=None, hidden_dim=None)
        assert cfg.total_updates == 1
        assert cfg.hidden_dim == 32


class TestTrainResume:
    def test_train_resume_continues_update_numbers(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t13_ppo_ckpts_")
        try:
            cfg1 = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
            )
            result1 = train(cfg1)
            ckpt1 = result1["checkpoint_path"]
            assert Path(ckpt1).exists()
            assert Path(ckpt1).name == "update_0001.npz"
            assert result1["start_update"] == 0
            assert result1["last_update"] == 1

            cfg2 = PPOConfig(
                total_updates=2,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                resume_checkpoint=ckpt1,
            )
            result2 = train(cfg2)
            assert Path(result2["checkpoint_path"]).name == "update_0003.npz"
            assert result2["start_update"] == 1
            assert result2["last_update"] == 3
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_resume_metadata_contains_resumed_from(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t13_ppo_meta_")
        try:
            cfg1 = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
            )
            result1 = train(cfg1)
            ckpt1 = result1["checkpoint_path"]

            cfg2 = PPOConfig(
                total_updates=2,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                resume_checkpoint=ckpt1,
            )
            train(cfg2)

            ckpt2_path = Path(ckpt_dir) / "update_0002.npz"
            assert ckpt2_path.exists()

            model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
            mx.eval(model.parameters())
            loaded = load_checkpoint(str(ckpt2_path), model)
            meta = loaded["metadata"]
            assert meta["resumed_from"] == ckpt1
            assert meta["start_update"] == 1
            assert meta["update"] == 2
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_resume_restores_model_outputs_smoke(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t13_ppo_resm_smoke_")
        try:
            cfg1 = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
            )
            result1 = train(cfg1)
            ckpt1 = result1["checkpoint_path"]
            assert Path(ckpt1).exists()

            cfg2 = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                resume_checkpoint=ckpt1,
            )
            result2 = train(cfg2)
            assert Path(result2["checkpoint_path"]).exists()
            assert result2["last_update"] == 2
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


class TestExperimentResume:
    def test_experiment_accepts_preset(self):
        output_dir = tempfile.mkdtemp(prefix="_t13_exp_preset_")
        try:
            cfg = ExperimentConfig(
                name="smoke_preset",
                output_dir=output_dir,
                seed=42,
                preset="smoke",
                export_onnx=False,
            )
            summary = run_experiment(cfg)
            assert Path(summary["run_dir"]).exists()
            assert Path(summary["run_dir"]) / "summary.json"

            ckpt_path = summary.get("checkpoint_path")
            if ckpt_path:
                model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
                mx.eval(model.parameters())
                loaded = load_checkpoint(ckpt_path, model)
                ckpt_cfg = loaded["metadata"]["config"]
                assert ckpt_cfg["hidden_dim"] == 32
                assert ckpt_cfg["action_hidden_dim"] == 16
                assert ckpt_cfg["total_updates"] == 1

            assert summary["train"]["updates"] == 1
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_experiment_resume_smoke(self):
        output_dir = tempfile.mkdtemp(prefix="_t13_exp_resume_")
        try:
            cfg1 = ExperimentConfig(
                name="first_run",
                output_dir=output_dir,
                seed=42,
                updates=1,
                episodes_per_update=1,
                max_steps=5,
                hidden_dim=32,
                action_hidden_dim=16,
                export_onnx=False,
            )
            summary1 = run_experiment(cfg1)
            ckpt1 = summary1.get("checkpoint_path")
            assert ckpt1 is not None, "first run must produce a checkpoint"
            assert Path(ckpt1).name == "update_0001.npz"

            cfg2 = ExperimentConfig(
                name="second_run",
                output_dir=output_dir,
                seed=42,
                updates=1,
                episodes_per_update=1,
                max_steps=5,
                hidden_dim=32,
                action_hidden_dim=16,
                export_onnx=False,
                resume_checkpoint=ckpt1,
            )
            summary2 = run_experiment(cfg2)
            ckpt2 = summary2.get("checkpoint_path")
            assert ckpt2 is not None
            assert Path(ckpt2).name == "update_0002.npz"
            assert summary2["train"]["start_update"] == 1
            assert summary2["train"]["last_update"] == 2
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
