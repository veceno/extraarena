---
name: extra-rlhf
description: "Use for anything about the ExtraArena RLHF data-collection, private dataset administration, and training-orchestration environment (rlhf_env, port 8090, MCP stdio): run battles, manage/validate/export V5, Nemesis, ReturnClock, Metronome and TimeStamp data, orchestrate the Extra-LR pipeline, or play as a sub-agent. Routes to three sub-skills."
---

# ExtraRLHF

Autonomous RLHF **data collection + private dataset administration + training
orchestration** for ExtraArena. A deterministic arena engine (`core.engine`) is
driven headless through an MCP stdio server, producing per-turn,
full-information traces for imitation learning / RLHF. Headless battles remain
files-only and isolated from production. A separate, fail-closed dataset plane
can inspect/validate local artifacts and, only after explicit opt-in, perform
read-only pseudonymized V5 and ReturnClock exports from production.

This is the **umbrella** skill. It routes to one of three levels depending on
your job. Pick your level, then open that sub-skill.

## Three orchestration levels

| Level | Sub-skill | You are… | Scope | Primary tools |
|---|---|---|---|---|
| 0 | `extrarlhf-pipeline-orchestration` | the **pipeline model-manager** | collect → export → validate/split → train → eval → promote | `get_training_data_status`, `list_training_exports`, `validate_training_export`, export/materialize/split tools, model registry + delegates to L1/L2 |
| 1 | `extrarlhf-gen-orchestration` | the **data-generation orchestrator** | plan + dispatch a fleet of series, monitor, validate, ship dataset | `start_series`, `next_battle`, `finish_series`, `list_active_series`, `get_agent_status`, `get_v5_dataset_summary`, `validate_v5_traces` |
| 2 | `extrarlhf-player` | the **player sub-agent** | play **one** battle as p1 (human/llm) | `get_match_status`, `get_state`, `get_legal_actions`, `submit_action`, `advance_bot`, `surrender`, `get_action_history` |

**Composition:** L0 directs L1; L1 runs bounded L2 workers; L2 owns one complete
start→play→finish lifecycle. A live `match_id` is process-local: never create it
in one MCP process and hand it to another. L1 can also run model-vs-model series
(`p1_actor_type="rl"`) that auto-play without an L2 worker.

## When to use which

- "Run a data-collection campaign / generate N battles vs model X" → **L1**.
- "Play this one battle / take my turn / what are my legal actions" → **L2**.
- "Run the whole pipeline: collect, train, benchmark new vs old, promote" → **L0**.
- "Register a checkpoint / export or inspect V5, Nemesis or ReturnClock / is a
  dataset training-ready?" → **L0**.
- Unsure → start at L0; it tells you when to drop to L1/L2.

## Setup (do this once)

Register the MCP server in your client and install the skills — see
[`INSTALL.md`](./INSTALL.md). Quick check it's alive:

```bash
./rlhf_env/start_rlhf_env.sh setup --python /path/to/python3.13
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | ./rlhf_env/.venv/bin/python -m rlhf_env.mcp_server
```

Pin both the interpreter and exact checkout in client config. Use the
checkout-local interpreter created by `start_rlhf_env.sh setup --python ...`
(`<REPO_ROOT>/rlhf_env/.venv/bin/python`) and verify it with
`import numpy, onnxruntime, asyncpg`; do not infer dependency readiness from the
Python minor version. For curated checkpoints, pass the adapter kind explicitly
even when auto-detection is available.

MCP responses use the standard wire shape: JSON text in `content[]`, the same
object in `structuredContent`, and `isError`. Consume `structuredContent` when
the client exposes it.

## Private dataset plane

Local inspect/list/validate/materialize/split operations are available by
default and confined to `--datasets-dir` (default `datasets/`). Production
reads are **disabled by default**. Enable them only in a trusted process with:

```bash
export RLHF_ENABLE_PRODUCTION_DATASETS=1
export RETURNCLOCK_DATASET_SALT='<export-specific secret, at least 32 bytes>'
export RETURNCLOCK_DATASET_SALT_KEY_ID='<non-secret rotation id>'
```

Never put the salt value, DSN, raw user IDs, or `include_players=true` in MCP
arguments. ReturnClock exports are pseudonymized—not anonymous—and stay in
private training storage. `RETURNCLOCK_DATASET_SALT_KEY_ID` identifies a
rotation without exposing the secret.

Read V5 readiness per training contour. For a headless group,
`training_ready` is a backward-compatible alias for
`v5_policy_training_ready` and `training_ready_scope` is
`v5_policy_only`. This can gate V5 policy targets and a separately eligible
Nemesis Lite export. It does **not** make Metronome or TimeStamp ready:
require their explicit readiness fields and observed production labels. Never
treat headless wall-clock/CPU timings or synthetic actor latency as human
decision-time labels.

`split_nemesis_training_dataset` always supports a separately eligible
Lite deck-grouped handoff. Standard views are conditional: require
`training_ready_standard=true` and inspect `standard_readiness_blockers`.
Human-bot/model-model rows train Lite only in the current canonical pipeline;
do not silently feed masked human-bot extensions into Standard.

TimeStamp is fail-closed against target leakage. Its model inputs are only the
prebattle deck or deck pair, `starting_player`, and explicitly approved
features that already existed before the battle started. `duration_seconds`,
`turns`, `finished_at`, and every value derived from them are labels or audit
metadata only. Never pass the complete `timestamp_features`/`meta` object to a
trainer or serving adapter.

ReturnClock natural-return training consumes only the organic-only files
published by `split_returnclock_training_dataset`; the raw export may retain
treated rows for audit, but it is not a trainer input. Production extraction is
keyset-paged, up to 1,000,000 rows per raw stream inside one repeatable-read
snapshot. Exclusive `end_at` bounds event time/censoring; a later
`ingested_before` independently bounds row creation, so late status updates do
not erase existing assignments. Hitting the ceiling is a stop condition, not
evidence of a complete export. Raw export and split currently materialize the
selected bounded window in memory; size large windows against available RAM.

For durable handoff, export to a new versioned path with `overwrite=false`,
validate it, then promote an external pointer. Fresh-path publication uses a
same-filesystem temp rename; overwrite has rollback for ordinary errors but is
not guaranteed crash-atomic across process or power failure.

## Key concepts (read once)

- **Actor types** `p1_actor_type ∈ {human, llm, rl}` decide who plays p1 and how
  (`submit_action` for human/llm, `advance_bot` auto-play for rl). → `concepts.md`
- **`battle_tag`** (`{p1}-vs-{bot|rl}`) slices the dataset by match kind. → `concepts.md`
- **Agent codenames** pin a series to a named sub-agent; auto-released on
  completion (self-healing reap). → `concepts.md`
- **`degraded`/`policy_warnings`** signal a silent fallback — always check before
  trusting a trace; `weights_hash` verifies the real checkpoint played. → `concepts.md`
- **On-disk layout**: `sessions/<group>/{manifest, summary, catalog, battles/...,
  battles/<bid>/v5/{meta,turns,actions}.jsonl}`. → `data-format.md`
- **All MCP tools**, including cross-contour export/validate/split. →
  `mcp-tools.md`

## Universal, not version-locked

Nothing here is hardcoded to a specific model version. "V5" is both the
**storage layout** name for the omniscient trace
(`v5/{meta,turns,actions}.jsonl`) and the implemented V5 adapter kind. The same
orchestration works for legacy
(`legacy_onnx`), action-conditioned (`action_onnx`/`v4`), future adapters, and
baselines — register a new kind via `register_custom_model` or
`default_registry().register(...)`.

## References

- [`references/mcp-tools.md`](./references/mcp-tools.md) — complete tool reference
- [`references/data-format.md`](./references/data-format.md) — on-disk manifest / trace / agents_index
- [`references/concepts.md`](./references/concepts.md) — actor types, battle_tag, codenames, adapter registry, determinism
- [`INSTALL.md`](./INSTALL.md) — register MCP + install skills in Claude Code / Codex / OpenCode / generic
- [`mcp/extra-rlhf.mcp.json`](./mcp/extra-rlhf.mcp.json) — MCP server config snippet
- Source: `rlhf_env/` (`mcp_server.py`, `components/`, `DOCS.md`)
