"""
Experiment runner for PPO baseline training + ONNX export + BerserkInference eval.

CLI:
    python3 -m ai.train_v2.experiment --name smoke --updates 3
    python3 -m ai.train_v2.experiment --ablation --name preview_ablation --updates 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict

from ai.train_v2.train_ppo import PPOConfig, train, make_config_from_preset, TRAIN_PRESETS
from ai.train_v2.export_onnx import export_checkpoint_to_onnx
from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    make_train_v2_berserk_brain,
    evaluate_berserk_matchup,
    compare_berserk_to_onnx_policy,
    benchmark_feature_modes,
    OPPONENT_REGISTRY,
)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class ExperimentConfig:
    name: str = "debug"
    output_dir: str = "ai/train_v2/runs"
    seed: int = 42
    updates: int = 3
    episodes_per_update: int = 2
    max_steps: int = 100
    hidden_dim: int = 64
    action_hidden_dim: int = 32
    learning_rate: float | None = None
    entropy_coef: float | None = None
    include_preview_features: bool = False
    eval_games: int = 4
    eval_max_steps: int = 200
    export_onnx: bool = True
    preset: str | None = None
    resume_checkpoint: str | None = None
    rollout_workers: int | None = None
    verify_mask: bool | None = None
    placement_mode: str | None = None
    action_features_dtype: str | None = None
    profile_actions: bool = False
    opponent_mix: str | None = None
    learner_side: str | None = None
    starting_player: str | None = None
    level_handicap_rate: float | None = None
    learner_level: int | None = None
    opponent_level: int | None = None
    focus_scenarios_json: str | None = None
    focus_deck_rate: float | None = None
    explicit_overrides: tuple[str, ...] = field(default_factory=tuple)


# ============================================================================
# RUN EXPERIMENT
# ============================================================================


def _experiment_overrides(config: ExperimentConfig, run_dir: Path, metrics_path: str, ckpt_dir: Path) -> dict:
    defaults = ExperimentConfig()
    overrides = {
        "seed": config.seed,
        "checkpoint_dir": str(ckpt_dir),
        "metrics_path": metrics_path,
        "resume_checkpoint": config.resume_checkpoint,
    }

    mapping = {
        "updates": "total_updates",
        "episodes_per_update": "episodes_per_update",
        "max_steps": "max_steps_per_episode",
        "hidden_dim": "hidden_dim",
        "action_hidden_dim": "action_hidden_dim",
        "learning_rate": "learning_rate",
        "entropy_coef": "entropy_coef",
        "include_preview_features": "include_preview_features",
        "rollout_workers": "rollout_workers",
        "verify_mask": "verify_mask",
        "placement_mode": "placement_mode",
        "action_features_dtype": "action_features_dtype",
        "profile_actions": "profile_actions",
        "opponent_mix": "opponent_mix",
        "learner_side": "learner_side",
        "starting_player": "starting_player",
        "level_handicap_rate": "level_handicap_rate",
        "learner_level": "learner_level",
        "opponent_level": "opponent_level",
        "focus_scenarios_json": "focus_scenarios_json",
        "focus_deck_rate": "focus_deck_rate",
    }

    for exp_field, ppo_field in mapping.items():
        value = getattr(config, exp_field)
        is_explicit = exp_field in config.explicit_overrides
        is_non_preset_default = not config.preset and value is not None
        is_changed = value is not None and value != getattr(defaults, exp_field)
        if is_explicit or is_non_preset_default or is_changed:
            overrides[ppo_field] = value

    return overrides


def run_experiment(config: ExperimentConfig) -> dict:
    base = Path(config.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{config.name}_{timestamp}"
    if run_dir.exists():
        run_dir = base / f"{config.name}_{timestamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config.json"
    cfg_path.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False)
    )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(run_dir / "metrics.jsonl")

    overrides = _experiment_overrides(config, run_dir, metrics_path, ckpt_dir)

    if config.preset:
        ppo_cfg = make_config_from_preset(config.preset, **overrides)
    else:
        ppo_cfg = PPOConfig(**overrides)

    train_result = train(ppo_cfg)

    checkpoint_path = train_result.get("checkpoint_path", "")
    onnx_path: str | None = None
    eval_result: dict | None = None
    parity_result: dict | None = None
    feature_bench: dict | None = None

    if checkpoint_path and config.export_onnx:
        exported_dir = run_dir / "exported"
        exported_dir.mkdir(parents=True, exist_ok=True)
        ckpt_stem = Path(checkpoint_path).stem
        onnx_out = str(exported_dir / f"{ckpt_stem}.onnx")
        onnx_path = export_checkpoint_to_onnx(checkpoint_path, onnx_out, opset=17, placement_mode=ppo_cfg.placement_mode)

        brain = make_train_v2_berserk_brain(onnx_path, selection="argmax")
        berserk_pol = BerserkBrainPolicy(brain, difficulty="test")

        eval_seeds = list(
            range(config.seed * 100, config.seed * 100 + config.eval_games)
        )

        eval_result = {}
        for opp_name in ["random", "end_turn", "greedy_face"]:
            opp_cls = OPPONENT_REGISTRY[opp_name]
            opp_pol = opp_cls()
            er = evaluate_berserk_matchup(
                berserk_pol,
                opp_pol,
                seeds=eval_seeds,
                swap_sides=True,
                max_steps=config.eval_max_steps,
            )
            eval_result[opp_name] = {
                "winrate": er["p1_winrate"],
                "games": er["games"],
                "avg_turns": er["avg_turns"],
                "avg_steps": er["avg_steps"],
                "invalid_actions": er["invalid_actions"],
                "brain_invalid_actions": er["p1_brain_invalid_actions"],
                "latency_ms_p50": er["p1_latency_ms_p50"],
                "latency_ms_p95": er["p1_latency_ms_p95"],
            }

        parity_result = compare_berserk_to_onnx_policy(
            onnx_path,
            seed=config.seed,
            steps=min(config.eval_max_steps, 40),
            selection="argmax",
        )

        feature_bench = benchmark_feature_modes(
            onnx_path, steps=min(config.eval_max_steps, 30)
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config": asdict(config),
        "train": {
            "updates": train_result["updates"],
            "start_update": train_result.get("start_update", 0),
            "last_update": train_result.get("last_update", train_result["updates"]),
            "episodes": train_result["episodes"],
            "steps": train_result["steps"],
            "last_loss": train_result["last_loss"],
            "last_entropy": train_result["last_entropy"],
        },
        "checkpoint_path": checkpoint_path or None,
        "onnx_path": onnx_path,
        "eval": eval_result,
        "parity": parity_result,
        "feature_benchmark": feature_bench,
    }

    sum_path = run_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


# ============================================================================
# PREVIEW ABLATION
# ============================================================================

def run_preview_ablation(
    *,
    base_name: str = "preview_ablation",
    output_dir: str = "ai/train_v2/runs",
    seed: int = 42,
    updates: int = 2,
    episodes_per_update: int = 1,
    max_steps: int = 50,
    eval_games: int = 4,
    eval_max_steps: int = 200,
    hidden_dim: int = 64,
    action_hidden_dim: int = 32,
) -> dict:
    cfg_fast = ExperimentConfig(
        name=f"{base_name}_fast",
        output_dir=output_dir,
        seed=seed,
        updates=updates,
        episodes_per_update=episodes_per_update,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
        include_preview_features=False,
        eval_games=eval_games,
        eval_max_steps=eval_max_steps,
        export_onnx=True,
    )

    cfg_preview = ExperimentConfig(
        name=f"{base_name}_preview",
        output_dir=output_dir,
        seed=seed,
        updates=updates,
        episodes_per_update=episodes_per_update,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
        include_preview_features=True,
        eval_games=eval_games,
        eval_max_steps=eval_max_steps,
        export_onnx=True,
    )

    result_fast = run_experiment(cfg_fast)
    result_preview = run_experiment(cfg_preview)

    def _random_wr(er):
        if er and er.get("eval"):
            return er["eval"].get("random", {}).get("winrate")
        return None

    def _speedup(er):
        if er and er.get("feature_benchmark"):
            return er["feature_benchmark"].get("fast_vs_full_speedup")
        return None

    def _steps(er):
        return er.get("train", {}).get("steps", 0)

    comparison = {
        "fast_vs_random_wr": _random_wr(result_fast),
        "preview_vs_random_wr": _random_wr(result_preview),
        "fast_steps": _steps(result_fast),
        "preview_steps": _steps(result_preview),
        "fast_feature_speedup": _speedup(result_fast),
        "preview_feature_speedup": _speedup(result_preview),
    }

    return {
        "fast": result_fast,
        "preview": result_preview,
        "comparison": comparison,
    }


# ============================================================================
# CLI
# ============================================================================

def _main():
    parser = argparse.ArgumentParser(description="Run a TrainV2 PPO experiment")
    parser.add_argument("--name", default="debug", help="Experiment name")
    parser.add_argument("--output-dir", default="ai/train_v2/runs", help="Output root directory")
    parser.add_argument("--preset", default=None, choices=list(TRAIN_PRESETS), help="Named training preset")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--action-hidden-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--include-preview-features", action="store_true")
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run preview ablation (fast vs preview)")
    parser.add_argument("--rollout-workers", type=int, default=None, help="Parallel env runners")
    parser.add_argument("--verify-mask", default=None, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--placement-mode", default=None, choices=["append_only", "full"], help="Action placement mode")
    parser.add_argument("--action-features-dtype", default=None, choices=["float32", "float16"])
    parser.add_argument("--profile-actions", action="store_true")
    parser.add_argument("--opponent-mix", default=None, help="League mix, e.g. self:0.5,random:0.1,greedy_face:0.2,trainv2_0700:0.2")
    parser.add_argument("--learner-side", default=None, choices=["random", "p1", "p2"])
    parser.add_argument("--starting-player", default=None, choices=["random", "p1", "p2", "learner", "opponent"])
    parser.add_argument("--level-handicap-rate", type=float, default=None)
    parser.add_argument("--learner-level", type=int, default=None)
    parser.add_argument("--opponent-level", type=int, default=None)
    parser.add_argument("--focus-scenarios-json", default=None)
    parser.add_argument("--focus-deck-rate", type=float, default=None)
    args = parser.parse_args()

    if args.ablation:
        abl_kwargs: dict = {
            "base_name": args.name,
            "output_dir": args.output_dir,
        }
        abl_optional = {
            "seed": args.seed,
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "max_steps": args.max_steps,
            "eval_games": args.eval_games,
            "eval_max_steps": args.eval_max_steps,
            "hidden_dim": args.hidden_dim,
            "action_hidden_dim": args.action_hidden_dim,
        }
        for k, v in abl_optional.items():
            if v is not None:
                abl_kwargs[k] = v
        result = run_preview_ablation(**abl_kwargs)
        comp = result["comparison"]
        print(f"Ablation: {args.name}")
        print(f"  fast     steps={comp['fast_steps']}  vs_random_wr={comp['fast_vs_random_wr']}  speedup={comp['fast_feature_speedup']}")
        print(f"  preview  steps={comp['preview_steps']}  vs_random_wr={comp['preview_vs_random_wr']}  speedup={comp['preview_feature_speedup']}")
        print(f"  fast run_dir: {result['fast']['run_dir']}")
        print(f"  preview run_dir: {result['preview']['run_dir']}")
    else:
        cfg_kwargs: dict = {
            "name": args.name,
            "output_dir": args.output_dir,
            "preset": args.preset,
            "resume_checkpoint": args.resume_checkpoint,
            "include_preview_features": bool(args.include_preview_features),
            "export_onnx": not args.no_export_onnx,
        }

        optional_map = {
            "seed": args.seed,
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "max_steps": args.max_steps,
            "hidden_dim": args.hidden_dim,
            "action_hidden_dim": args.action_hidden_dim,
            "learning_rate": args.learning_rate,
            "entropy_coef": args.entropy_coef,
            "eval_games": args.eval_games,
            "eval_max_steps": args.eval_max_steps,
            "rollout_workers": args.rollout_workers,
            "verify_mask": args.verify_mask,
            "placement_mode": args.placement_mode,
            "action_features_dtype": args.action_features_dtype,
            "profile_actions": args.profile_actions,
            "opponent_mix": args.opponent_mix,
            "learner_side": args.learner_side,
            "starting_player": args.starting_player,
            "level_handicap_rate": args.level_handicap_rate,
            "learner_level": args.learner_level,
            "opponent_level": args.opponent_level,
            "focus_scenarios_json": args.focus_scenarios_json,
            "focus_deck_rate": args.focus_deck_rate,
        }

        for k, v in optional_map.items():
            if v is not None:
                cfg_kwargs[k] = v
                if k != "profile_actions" or v:
                    cfg_kwargs.setdefault("explicit_overrides", []).append(k)

        if "explicit_overrides" in cfg_kwargs:
            cfg_kwargs["explicit_overrides"] = tuple(cfg_kwargs["explicit_overrides"])

        cfg = ExperimentConfig(**cfg_kwargs)
        result = run_experiment(cfg)
        print(f"Experiment: {args.name}")
        print(f"  run_dir:    {result['run_dir']}")
        print(f"  updates:    {result['train']['updates']}")
        print(f"  episodes:   {result['train']['episodes']}")
        print(f"  steps:      {result['train']['steps']}")
        print(f"  last_loss:  {result['train']['last_loss']:.4f}")
        if result["onnx_path"]:
            print(f"  onnx:       {result['onnx_path']}")
        if result["eval"]:
            for opp, er in result["eval"].items():
                print(f"  vs_{opp}: wr={er['winrate']:.3f}, steps={er['avg_steps']:.1f}, invalid={er['invalid_actions']}, latency_p50={er['latency_ms_p50']:.1f}ms")
        if result["parity"]:
            print(f"  parity: {result['parity']['checked']} checked, {result['parity']['matches']} matches, {result['parity']['mismatches']} mismatches")


if __name__ == "__main__":
    _main()
