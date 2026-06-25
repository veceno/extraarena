# TrainV2 Night Run

Overnight training launcher with preflight checks, estimates, and a morning checklist.

## Quick Start

### Dry run (recommended first time)

```bash
python3 -m ai.train_v2.night_run \
  --name night_fast_v1 \
  --preset m4_night \
  --seed 42 \
  --output-dir ai/train_v2/runs \
  --dry-run
```

This prints preflight status, estimated transitions, and memory without starting training.

### Real overnight run

```bash
python3 -m ai.train_v2.night_run \
  --name night_fast_v1 \
  --preset m4_night \
  --seed 42 \
  --output-dir ai/train_v2/runs \
  --export-onnx
```

## Safer Warmup

Use a quick preset first to verify everything works:

```bash
python3 -m ai.train_v2.night_run \
  --name warmup_v1 \
  --preset m4_quick \
  --dry-run
```

Then run without `--dry-run`:

```bash
python3 -m ai.train_v2.night_run \
  --name warmup_v1 \
  --preset m4_quick \
  --export-onnx
```

## Presets

| Preset | Updates | Episodes/Update | Max Steps | Hidden | Notes |
|---|---|---|---|---|---|
| `smoke` | 1 | 1 | 10 | 32 | CI smoke |
| `m4_quick` | 20 | 4 | 200 | 128 | Short warmup |
| `m4_night` | 200 | 8 | 300 | 256 | Overnight run |

## Morning Checklist

After a run completes, use these commands:

```bash
# Review training metrics
python3 -m ai.train_v2.monitor --run <run_dir>

# Open the operator panel
python3 -m ai.train_v2.operator panel \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases

# Resume from the latest checkpoint
python3 -m ai.train_v2.experiment \
  --preset m4_night \
  --resume-checkpoint <CHECKPOINT> \
  --name night_fast_v1_resume \
  --output-dir ai/train_v2/runs

# Export ONNX from a checkpoint
python3 -m ai.train_v2.export_onnx \
  --checkpoint <CHECKPOINT> \
  --output <ONNX_PATH>
```

The run directory also contains `night_run_summary.json` with these commands pre-filled.

## Preview Features

```bash
python3 -m ai.train_v2.night_run \
  --name preview_test \
  --preset m4_quick \
  --include-preview-features
```

> **Warning:** `include-preview-features` increases memory usage and compute. Only use if your model is designed to consume preview features.

## Skipped Updates

If `monitor` reports skipped updates, it usually means `min_batch_transitions` was not reached. Common causes:
- Very short episodes (reduce `max_steps_per_episode` or increase `episodes_per_update`)
- Truncation before enough transitions accumulate
- Try a larger preset or resume with adjusted hyperparameters

## Safety

- Preflight checks MLX, ClassicRLEnv, and output dir before starting.
- `--dry-run` never trains or creates run directories.
- Real runs write `night_run_summary.json` in the run directory for the morning checklist.
- No production configs are modified.

## Audit (2026-06-25)

Scope: `docs/TRAIN_V2_NIGHT_RUN.md` (TrainV2 overnight launcher). Project has moved most active training work to `TrainV3/` (Rust-backed `trainv3_core`, `train_v3.rust_ppo`) and `web/server.py` exposes TrainV3 schema endpoints, but `ai/train_v2/night_run.py` and its CLI surface still exist unchanged and remain the documented entry point for the train_v2 module — so historical framing is correct.

Checked:
- `ai/train_v2/night_run.py` (CLI flags: `--name`, `--preset`, `--seed`, `--output-dir`, `--include-preview-features`, `--export-onnx`, `--dry-run`, plus undocumented `--eval-games`, `--eval-max-steps`, `--max-expected-hours`, `--rollout-workers`, `--verify-mask`, `--placement-mode`, `--json`)
- `ai/train_v2/train_ppo.py:82-121` (`TRAIN_PRESETS` table — `smoke`/`m4_quick`/`m4_night` matches doc Updates/Episodes/Max Steps/Hidden columns exactly)
- `ai/train_v2/operator.py:122-131` (`panel` subcommand + `--runs-dir`/`--releases-dir`)
- `ai/train_v2/monitor.py:122-126` (`--run` flag)
- `ai/train_v2/experiment.py:305-333` (`--preset`, `--resume-checkpoint`, `--name`, `--output-dir`)
- `ai/train_v2/export_onnx.py:194-198` (`--checkpoint`, `--output`)
- `ai/train_v2/classic_rl_env.py` (ClassicRLEnv class — used by night_run preflight, line 49)
- `ai/train_v2/train_ppo.py:66, 984-985, 1089-1096` (`min_batch_transitions` default = 2, used by Skipped Updates section)
- `ai/train_v2/night_run.py:195` (`night_run_summary.json` path)
- `TrainV3/python/train_v3/` and `TrainV3/scripts/` (no `night_run` analog — separate phase-based scripts like `run_v5_adaptive.py`, `run_phase*_*.py`)

What was fixed: no code-level edits required — all CLI invocations, script paths, preset values, flag names, and behavior claims match the current `ai/train_v2/` source. Doc retains historical train_v2 framing as intended.

What I could not verify: throughput numbers ("~100 transitions/sec" ballpark in `night_run.py:101`) are aspirational; the doc does not quote them. `scripts/` at project root contains only `graphify_project.sh`, `precompile_webapp_index.py`, `static_for_playwright.py` — no train_v2 wrapper — so doc's direct `python3 -m ai.train_v2.*` invocations remain the canonical path.
