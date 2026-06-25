# Graphify

Graphify builds a local knowledge graph for this repository under `graphify-out/`.

This project is large and includes generated media, model outputs, local worktrees,
and training artifacts. The root `.graphifyignore` keeps extraction focused on
source code, tests, docs, configs, and small SQL/data files.

## Setup

Install the official package:

```bash
uv tool install graphifyy
```

Install project-scoped Codex rules:

```bash
graphify install --project --platform codex
```

## Build

Use the project wrapper:

```bash
scripts/graphify_project.sh
```

By default this builds an AST/code graph and does not require an LLM API key.
It uses one worker by default because this repository has large local artifacts;
override it with `GRAPHIFY_MAX_WORKERS=4` if your machine has enough memory.
The wrapper performs a full rebuild each time and clears only generated
`graphify-out/graph.json` plus `graphify-out/manifest.json`; the cache remains.
Outputs:

- `graphify-out/graph.json`
- `graphify-out/manifest.json`

The wrapper uses `--no-cluster` by default so it stays practical on this
repository. For a fuller report after the graph exists, run:

```bash
graphify cluster-only . --no-viz --no-label
```

To include markdown/html/csv documents in semantic extraction, configure a valid
Graphify-supported API key and run. The wrapper defaults to DeepSeek plus
`--mode deep` for semantic mode; override with `GRAPHIFY_BACKEND=openai`,
`GRAPHIFY_MODE=deep`, `GRAPHIFY_TOKEN_BUDGET=...`, or
`GRAPHIFY_MAX_CONCURRENCY=...`.

```bash
export DEEPSEEK_API_KEY=...
scripts/graphify_project.sh --semantic
```

## Query

```bash
graphify query "How does admin analytics flow through the backend?"
graphify explain "Database.get_admin_analytics_overview"
graphify path "web.server" "Database.get_admin_analytics_overview"
```

## Audit (2026-06-25)

Checked against:
- `/Users/laveqox/Documents/ExtraArenaRaS/scripts/graphify_project.sh`
- `/Users/laveqox/Documents/ExtraArenaRaS/.codex/skills/graphify/SKILL.md`
- `/Users/laveqox/Documents/ExtraArenaRaS/.opencode/commands/graphify.md`
- `/Users/laveqox/Documents/ExtraArenaRaS/AGENTS.md`
- `/Users/laveqox/Documents/ExtraArenaRaS/.graphifyignore`
- `/Users/laveqox/Documents/ExtraArenaRaS/graphify-out/` (graph.json, GRAPH_REPORT.md, manifest.json)
- Live `graphify --help` and `graphify cluster-only . --no-viz --no-label` execution.

Verified (no changes needed):
- Package name `graphifyy` and `uv tool install` form match SKILL.md step 1.
- `graphify install --project --platform codex` matches `graphify install --help` and SKILL.md platforms list (codex included).
- `scripts/graphify_project.sh` exists and uses the env vars / defaults the doc describes (`GRAPHIFY_BACKEND=deepseek`, `GRAPHIFY_MODE=deep`, `--max-concurrency`, `--token-budget`, `GRAPHIFY_MAX_WORKERS`).
- `--no-cluster` default and clear-only-`graph.json`+`manifest.json` behavior match `scripts/graphify_project.sh` (lines 44-52).
- `graphify cluster-only . --no-viz --no-label` runs successfully and regenerates GRAPH_REPORT.md (verified live: 282 communities).
- `graphify query / explain / path` commands match `graphify --help` output.
- `.graphifyignore` exists at repo root and `graphify-out/` is gitignored; graph artifacts (graph.json 11.2MB, GRAPH_REPORT.md 94KB, manifest.json 55KB) all present.
- GRAPH_REPORT.md header `Built from commit: 86e18c87` is one commit behind current `HEAD` (367d4344 "chore: extend .gitignore..."); graph is mildly stale but structure descriptions remain accurate.

No changes needed — content matches current code.
