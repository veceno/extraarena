---
name: extra-rlhf
description: "Use for anything about the ExtraArena RLHF data-collection & training-orchestration environment (rlhf_env, port 8090, MCP stdio): running semi-synthetic battles, generating training traces, orchestrating the Extra-LR training pipeline, or playing battles as a sub-agent. Routes to three sub-skills — pipeline orchestration, data-generation orchestration, and player."
---

# ExtraRLHF

Autonomous RLHF **data-collection + training-orchestration** environment for the
ExtraArena card game. A deterministic arena engine (`core.engine`) is driven
**headless** through an MCP stdio server (25 tools), producing per-turn,
full-information traces for imitation learning / RLHF. No prod stack, no DB —
files only, separate process (web `127.0.0.1:8090`, MCP stdio).

This is the **umbrella** skill. It routes to one of three levels depending on
your job. Pick your level, then open that sub-skill.

## Three orchestration levels

| Level | Sub-skill | You are… | Scope | Primary tools |
|---|---|---|---|---|
| 0 | `extrarlhf-pipeline-orchestration` | the **pipeline model-manager** | full train loop: collect → train → eval → promote | `list_active_series`, `list_v5_groups`, `validate_v5_traces`, `register_custom_model`, `list_models` + delegates to L1/L2 |
| 1 | `extrarlhf-gen-orchestration` | the **data-generation orchestrator** | plan + dispatch a fleet of series, monitor, validate, ship dataset | `start_series`, `next_battle`, `finish_series`, `list_active_series`, `get_agent_status`, `get_v5_dataset_summary`, `validate_v5_traces` |
| 2 | `extrarlhf-player` | the **player sub-agent** | play **one** battle as p1 (human/llm) | `get_match_status`, `get_state`, `get_legal_actions`, `submit_action`, `advance_bot`, `surrender`, `get_action_history` |

**Composition:** L0 directs L1; L1 spawns many L2 agents in parallel; L2 plays
one battle. L1 can also run model-vs-model series (`p1_actor_type="rl"`) that
auto-play with no L2 agent.

## When to use which

- "Run a data-collection campaign / generate N battles vs model X" → **L1**.
- "Play this one battle / take my turn / what are my legal actions" → **L2**.
- "Run the whole pipeline: collect, train, benchmark new vs old, promote" → **L0**.
- "Register a new checkpoint / what models are available / is the dataset ready" → **L0** (or L1 for the dataset check).
- Unsure → start at L0; it tells you when to drop to L1/L2.

## Setup (do this once)

Register the MCP server in your client and install the skills — see
[`INSTALL.md`](./INSTALL.md). Quick check it's alive:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | python3 -m rlhf_env.mcp_server   # expect 25 tools
```

If `ai.model_benchmark` (layer A) is absent in your checkout, onnx auto-detect
is unavailable — pass `p2_model_path` + `p2_model_kind` explicitly to play onnx
models (see `concepts.md`).

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
- **All 25 tools**, args, returns. → `mcp-tools.md`

## Universal, not version-locked

Nothing here is hardcoded to a specific model version. "V5" appears only as the
**storage layout** name for the omniscient trace (`v5/{meta,turns,actions}.jsonl`)
and as one reserved adapter kind. The same orchestration works for legacy
(`legacy_onnx`), action-conditioned (`action_onnx`/`v4`), future adapters, and
baselines — register a new kind via `register_custom_model` or
`default_registry().register(...)`.

## References

- [`references/mcp-tools.md`](./references/mcp-tools.md) — full 25-tool reference
- [`references/data-format.md`](./references/data-format.md) — on-disk manifest / trace / agents_index
- [`references/concepts.md`](./references/concepts.md) — actor types, battle_tag, codenames, adapter registry, determinism
- [`INSTALL.md`](./INSTALL.md) — register MCP + install skills in Claude Code / Codex / OpenCode / generic
- [`mcp/extra-rlhf.mcp.json`](./mcp/extra-rlhf.mcp.json) — MCP server config snippet
- Source: `rlhf_env/` (`mcp_server.py`, `components/`, `DOCS.md`)