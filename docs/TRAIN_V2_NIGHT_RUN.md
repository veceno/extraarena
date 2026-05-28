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
