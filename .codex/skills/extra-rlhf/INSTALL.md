# Installing ExtraRLHF (MCP + skills)

Two pieces: (1) register the **MCP server** in your client, (2) install the
**skills** so the model knows the playbooks. Both are optional-but-recommended:
the MCP gives tools; the skills give the workflows.

The MCP server is **stdio JSON-RPC 2.0**, launched from the **repo root**:

```bash
python3 -m rlhf_env.mcp_server \
  --models-dir ai/models \
  --sessions-dir rlhf_env/sessions \
  --cards-path ai/cards.json
```

CLI flags can also be env vars: `RLHF_MODELS_DIR`, `RLHF_SESSIONS_DIR`,
`RLHF_CARDS_PATH`, `RLHF_LOG_LEVEL`. A bash wrapper exists:
`./rlhf_env/start_rlhf_env.sh mcp`.

> Replace `<REPO_ROOT>` below with the absolute path to your ExtraArenaRaS
> checkout. The `cwd` must be the repo root so the `rlhf_env` package and the
> relative `ai/models`, `ai/cards.json` paths resolve.

---

## 1. Claude Code

### MCP (project-scoped, recommended for this repo)
Create `.mcp.json` in the repo root (or merge into an existing one):

```json
{
  "mcpServers": {
    "extra-rlhf": {
      "command": "python3",
      "args": ["-m", "rlhf_env.mcp_server",
               "--models-dir", "ai/models",
               "--sessions-dir", "rlhf_env/sessions",
               "--cards-path", "ai/cards.json"],
      "cwd": "<REPO_ROOT>"
    }
  }
}
```

Or via CLI (run from repo root, so `cwd` is implicit):

```bash
claude mcp add extra-rlhf -- python3 -m rlhf_env.mcp_server \
  --models-dir ai/models --sessions-dir rlhf_env/sessions --cards-path ai/cards.json
```

Verify: `claude mcp list` → `extra-rlhf`. In a session, `/mcp` shows it connected.

### Skills
Claude Code discovers skills from `~/.claude/skills/` (user) or `.claude/skills/`
(project). This bundle lives at `.codex/skills/` (repo-tracked). Symlink each
skill so it's discovered without duplicating files:

```bash
mkdir -p ~/.claude/skills
for s in extra-rlhf extrarlhf-pipeline-orchestration extrarlhf-gen-orchestration extrarlhf-player; do
  ln -sfn "<REPO_ROOT>/.codex/skills/$s" "$HOME/.claude/skills/$s"
done
```

(Per repo `.gitignore`, `.claude/` is local-only — the symlink keeps the
source-of-truth tracked in `.codex/skills/` while exposing it to Claude Code.)

---

## 2. Codex (OpenAI Codex CLI)

### MCP
Add to `~/.codex/config.toml`:

```toml
[mcp_servers.extra-rlhf]
command = "python3"
args = ["-m", "rlhf_env.mcp_server",
        "--models-dir", "ai/models",
        "--sessions-dir", "rlhf_env/sessions",
        "--cards-path", "ai/cards.json"]
cwd = "<REPO_ROOT>"
```

### Skills
Codex discovers skills from `.codex/skills/` — this bundle is already there, so
no extra step. (It's repo-tracked via `!.codex/**` in `.gitignore`.)

---

## 3. OpenCode

### MCP
Add to `opencode.json` (or `~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "extra-rlhf": {
      "type": "local",
      "command": ["python3", "-m", "rlhf_env.mcp_server",
                   "--models-dir", "ai/models",
                   "--sessions-dir", "rlhf_env/sessions",
                   "--cards-path", "ai/cards.json"],
      "cwd": "<REPO_ROOT>"
    }
  }
}
```

### Skills
Symlink into OpenCode's skills dir (e.g. `~/.config/opencode/skills/` or the
project `.opencode/skills/`):

```bash
mkdir -p ~/.config/opencode/skills
for s in extra-rlhf extrarlhf-pipeline-orchestration extrarlhf-gen-orchestration extrarlhf-player; do
  ln -sfn "<REPO_ROOT>/.codex/skills/$s" "$HOME/.config/opencode/skills/$s"
done
```

---

## 4. Generic MCP client (any stdio client)

Point your client at the command above. Config snippet template:
[`mcp/extra-rlhf.mcp.json`](./mcp/extra-rlhf.mcp.json). The server speaks
standard MCP: `initialize` → `tools/list` (25 tools) → `tools/call`.

---

## Verify it works

```bash
# 1. server alive + tools enumerate
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | python3 -m rlhf_env.mcp_server   # expect 25 tools

# 2. start a quick series (llm-vs-bot, auto-plays)
echo '{
  "jsonrpc":"2.0","id":2,"method":"tools/call",
  "params":{"name":"start_series",
            "arguments":{"spec":{"p2_model":"random","battles_planned":1,"seed":42}}}
}' | python3 -m rlhf_env.mcp_server
```

In your client, ask: *"Use the extra-rlhf skill to run 3 llm-vs-bot battles vs
random."* It should route to `extrarlhf-gen-orchestration` (L1) and call
`start_series`.

---

## Notes & troubleshooting

- **Python**: needs Python 3.10+ with `aiohttp`, `numpy`, `onnxruntime`, `mcp`
  (`rlhf_env/requirements.txt`). Use `./rlhf_env/start_rlhf_env.sh setup` to make
  a venv, or run with system python (`--no-venv`).
- **Layer A (`ai.model_benchmark`) absent**: onnx auto-detect is unavailable →
  onnx models won't resolve by name. Pass `p2_model_path` + `p2_model_kind`
  explicitly (see `references/concepts.md`). Baselines (`random`/`greedy_face`/
  `end_turn`) always work.
- **Port**: the **MCP server is stdio** (no port). The **web arena** is
  `127.0.0.1:8090` (`./rlhf_env/start_rlhf_env.sh` / `--port`). They're separate
  processes sharing the same engine + sessions dir.
- **Do not touch prod** (port 8081). `rlhf_env` is autonomous; `config.json`
  switches the dev/prod/local backend the rlhf-proxy reaches for `/api/rlhf/*`
  (default `dev` → 8082).
- **Logs**: `--log-level DEBUG` or `RLHF_LOG_LEVEL=DEBUG` (stderr; doesn't break
  stdio since only stdout is JSON-RPC).