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
