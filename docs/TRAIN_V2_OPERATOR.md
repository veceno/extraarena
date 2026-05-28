# TrainV2 Operator Tool

A read-only local dashboard and CLI for viewing TrainV2 artifacts: runs, profiles, release bundles, shadow evidence, and acceptance gates.

## What It Is

- **Web panel** — local HTTP server with a lightweight HTML/JS/CSS dashboard.
- **Snapshot** — export panel state to JSON for sharing or archiving.
- **Doctor** — read-only health check for common missing artifacts.

All tools are **read-only**. Nothing modifies production configs or game state.

For a detailed artifact reference and version strings, see [TRAIN_V2_ARTIFACTS.md](TRAIN_V2_ARTIFACTS.md).

## Smoke Check

Quickly verify the operator pipeline with a synthetic artifact tree:

```bash
python3 -m ai.train_v2.operator_smoke --root /tmp/trainv2_operator_smoke
```

With JSON output:

```bash
python3 -m ai.train_v2.operator_smoke --root /tmp/trainv2_operator_smoke --json --keep
```

> **Note:** Smoke artifacts are synthetic and not suitable for inference. This only checks artifact contracts, discovery, and UI/operator layers.

## Standard Local Workflow

1. Run suite to produce training runs.
2. Promote candidate from run.
3. Generate candidate profile (`candidate_profile.json`).
4. Build registry/overlay (`profile_registry`, `profile_overlay.json`).
5. Run shadow report (`shadow_evidence/`).
6. Run acceptance gate (`acceptance_gate/`).
7. Build release bundle (`releases/`).
8. Open panel to review everything.

Compact commands:

```bash
python3 -m ai.train_v2.candidate_profile --candidate run_dir --output candidate_profile.json
python3 -m ai.train_v2.profile_registry --paths ai/train_v2/runs --write-overlay overlay.json
python3 -m ai.train_v2.shadow_report --overlay overlay.json --output-dir shadow_pack --games 4
python3 -m ai.train_v2.acceptance_gate --candidate-dir candidate_dir --shadow-pack shadow_pack --output-dir gate
python3 -m ai.train_v2.release_bundle --candidate-dir candidate_dir --output-dir releases --name update_0003
python3 -m ai.train_v2.operator panel --runs-dir ai/train_v2/runs --releases-dir ai/train_v2/releases
```

## Commands

### Start web panel

```bash
python3 -m ai.train_v2.web_panel \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases \
  --port 8765
```

Or via operator CLI:

```bash
python3 -m ai.train_v2.operator panel \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases \
  --port 8765
```

Open: http://127.0.0.1:8765

### Write snapshot JSON

```bash
python3 -m ai.train_v2.operator snapshot \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases \
  --output trainv2_snapshot.json
```

### Run health check

```bash
python3 -m ai.train_v2.operator doctor \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases
```

With JSON output:

```bash
python3 -m ai.train_v2.operator doctor \
  --runs-dir ai/train_v2/runs \
  --releases-dir ai/train_v2/releases \
  --json
```

## Safety

- The panel is **read-only**. No buttons trigger training, eval, shadow runs, or production updates.
- File access is restricted to `--runs-dir` and `--releases-dir` roots. Path traversal is blocked.
- Only safe file types (`.json`, `.md`, `.txt`, `.log`) are served via `/api/file` and `/api/artifact`.

## Troubleshooting

### Empty panel

- Check that `--runs-dir` exists and contains run directories with `config.json`, `summary.json`, or `metrics.jsonl`.
- Check that `--releases-dir` exists if you expect release bundles.

### Path denied

- `/api/file` only allows paths inside the configured roots.
- Symlinks or paths outside roots are rejected.

### Bad JSON skipped

- Corrupted JSON artifacts are silently skipped in the panel. They may appear as error rows in registry or be absent entirely.

### No profiles found

- Ensure run directories contain `candidate_profile.json` or that you ran the profile registry builder.

### Browser issues

- The panel uses vanilla JS and no external assets. Any modern browser should work.
- If `localStorage` is disabled, tab persistence simply falls back to the default first tab.
