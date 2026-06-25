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
- Only safe file types (`.json`, `.md`, `.txt`, `.log`, `.js`, `.css`, `.html`) are served via `/api/file` and `/api/artifact`.

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

## Audit (2026-06-25)

Checked: doc describes TrainV2 operator tool (`ai.train_v2.operator*`, `ai.train_v2.web_panel`, `ai.train_v2.candidate_profile`, `ai.train_v2.profile_registry`, `ai.train_v2.shadow_report`, `ai.train_v2.acceptance_gate`, `ai.train_v2.release_bundle`, `ai.train_v2.operator_smoke`) and the relationship to TrainV3 (`/TrainV3/`).

Verified against source:
- Smoke: `python3 -m ai.train_v2.operator_smoke --root <path> [--json] [--keep]` — `operator_smoke.py` L355–358.
- Compact commands: `candidate_profile.py` L184–193 (matches `--candidate`, `--output`), `profile_registry.py` L320–328 (matches `--paths`, `--write-overlay`), `shadow_report.py` L220–235 (matches `--overlay`, `--output-dir`, `--games`), `acceptance_gate.py` L455–467 (matches `--candidate-dir`, `--shadow-pack`, `--output-dir`), `release_bundle.py` L405–417 (matches `--candidate-dir`, `--output-dir`, `--name`), `operator.py` L121–146 (matches `panel` / `snapshot` / `doctor` subcommands with `--runs-dir`, `--releases-dir`, `--port 8765`).
- Web panel default port 8765 (`web_panel.py` L25, L426).
- Empty-panel troubleshoot references `config.json` / `summary.json` / `metrics.jsonl` (`run_index.py` L15–17).
- "No profiles found" troubleshoot references `candidate_profile.json` (`profile_registry.py` L32–38).
- Path-traversal safety and `_is_path_allowed` / `_resolve_roots` (`web_panel.py` L28–46).

Fixed:
- `docs/TRAIN_V2_OPERATOR.md:105` — extended served file types from `(.json, .md, .txt, .log)` to `(.json, .md, .txt, .log, .js, .css, .html)` to match `web_panel.read_safe_text_file` allowlist (line 65).

Not changed:
- Historical "TrainV2" framing kept; `ai/train_v2/` remains the codec/operator source of truth and TrainV3 (`/TrainV3/`) imports it (e.g. `model_mlx`, `classic_actions_v1`, `onnx_policy`). Operator tool name in code and on disk is still `ai.train_v2.*`, so module paths and CLI names stay as written.
- Snapshot JSON default filename `trainv2_snapshot.json` is just an example (operator passes through `--output`), unchanged.
