# Installing ExtraRLHF (MCP + skills)

Two pieces: (1) register the **MCP server** in your client, (2) install the
**skills** so the model knows the playbooks. Both are optional-but-recommended:
the MCP gives tools; the skills give the workflows.

The MCP server is **stdio JSON-RPC 2.0**, launched from the **repo root**:

```bash
<PYTHON_WITH_RLHF_DEPS> -m rlhf_env.mcp_server \
  --models-dir ai/models \
  --sessions-dir rlhf_env/sessions \
  --datasets-dir datasets \
  --cards-path ai/cards.json
```

CLI flags can also be env vars: `RLHF_MODELS_DIR`, `RLHF_SESSIONS_DIR`,
`RLHF_DATASETS_DIR`, `RLHF_CARDS_PATH`, `RLHF_LOG_LEVEL`. A bash wrapper exists:
`./rlhf_env/start_rlhf_env.sh mcp`.

> Replace `<REPO_ROOT>` below with the absolute path to the exact checkout or
> worktree you are curating, and `<PYTHON_WITH_RLHF_DEPS>` with an absolute
> interpreter path. Run
> `./rlhf_env/start_rlhf_env.sh setup --python <PYTHON_WITH_RLHF_DEPS_BASE>`,
> then use
> `<REPO_ROOT>/rlhf_env/.venv/bin/python`. Verify that exact interpreter with
> `-c 'import numpy, onnxruntime, asyncpg'`; a Python minor version alone does
> not prove the dependencies are installed. The `cwd` must be the repo root so
> packages and relative paths resolve.

---

## 1. Claude Code

### MCP (project-scoped, recommended for this repo)
Create `.mcp.json` in the repo root (or merge into an existing one):

```json
{
  "mcpServers": {
    "extra-rlhf": {
      "command": "<PYTHON_WITH_RLHF_DEPS>",
      "args": ["-m", "rlhf_env.mcp_server",
               "--models-dir", "ai/models",
               "--sessions-dir", "rlhf_env/sessions",
               "--datasets-dir", "datasets",
               "--cards-path", "ai/cards.json"],
      "cwd": "<REPO_ROOT>"
    }
  }
}
```

Or via CLI (run from repo root, so `cwd` is implicit):

```bash
claude mcp add extra-rlhf -- <PYTHON_WITH_RLHF_DEPS> -m rlhf_env.mcp_server \
  --models-dir ai/models --sessions-dir rlhf_env/sessions \
  --datasets-dir datasets --cards-path ai/cards.json
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
command = "<PYTHON_WITH_RLHF_DEPS>"
args = ["-m", "rlhf_env.mcp_server",
        "--models-dir", "ai/models",
        "--sessions-dir", "rlhf_env/sessions",
        "--datasets-dir", "datasets",
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
      "command": ["<PYTHON_WITH_RLHF_DEPS>", "-m", "rlhf_env.mcp_server",
                   "--models-dir", "ai/models",
                   "--sessions-dir", "rlhf_env/sessions",
                   "--datasets-dir", "datasets",
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
standard MCP: `initialize` → `tools/list` → `tools/call`. A successful tool
result is available as JSON text in `content[0].text` and as the same object in
`structuredContent`; errors set `isError=true`.

---

## Verify it works

```bash
# 1. server alive + tools enumerate
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | <PYTHON_WITH_RLHF_DEPS> -m rlhf_env.mcp_server

# 2. start a quick series (llm-vs-bot; the caller plays p1)
echo '{
  "jsonrpc":"2.0","id":2,"method":"tools/call",
  "params":{"name":"start_series",
            "arguments":{"spec":{"p2_model":"random","battles_planned":1,"seed":42}}}
}' | <PYTHON_WITH_RLHF_DEPS> -m rlhf_env.mcp_server
```

In your client, ask: *"Use the extra-rlhf skill to run 3 llm-vs-bot battles vs
random."* It should route to `extrarlhf-gen-orchestration` (L1) and call
`start_series`.

---

## Notes & troubleshooting

- **Python**: use Python 3.13 with dependencies from
  `rlhf_env/requirements.txt` (`aiohttp`, `numpy`, `onnxruntime`, `mcp`,
  `asyncpg`, `python-dotenv`). Use
  `./rlhf_env/start_rlhf_env.sh setup --python /path/to/python3.13` to make a
  venv from an explicit interpreter (`RLHF_PYTHON` is equivalent), or pin the
  known dependency-bearing interpreter above.
- **Layer A (`ai.model_benchmark`) absent**: onnx auto-detect is unavailable →
  onnx models won't resolve by name. Pass `p2_model_path` + `p2_model_kind`
  explicitly (see `references/concepts.md`). Baselines (`random`/`greedy_face`/
  `end_turn`) always work.
- **Port**: the **MCP server is stdio** (no port). The **web arena** is
  `127.0.0.1:8090` (`./rlhf_env/start_rlhf_env.sh` / `--port`). They're separate
  processes sharing the same engine + sessions dir.
- **Headless collection does not touch prod** (port 8081). The only exception is
  the explicit read-only dataset plane described below.
- **Logs**: `--log-level DEBUG` or `RLHF_LOG_LEVEL=DEBUG` (stderr; doesn't break
  stdio since only stdout is JSON-RPC).

## Optional production dataset exports

Local listing, inspection, validation, V5 materialization and ReturnClock
splitting work with the default configuration. Production PostgreSQL reads are
fail-closed until one of these equivalent opt-ins is present:

```bash
# Prefer environment variables in the trusted launcher/service:
export RLHF_ENABLE_PRODUCTION_DATASETS=1
export RETURNCLOCK_DATASET_SALT='<export-specific HMAC secret, at least 32 bytes>'
export RETURNCLOCK_DATASET_SALT_KEY_ID='<non-secret rotation id>'

# Or add --enable-production-datasets to the MCP command.
```

`RLHF_RETURNCLOCK_SALT_ENV` / `--returnclock-salt-env` selects the **name** of
the environment variable containing the secret; the default is
`RETURNCLOCK_DATASET_SALT`.
`RLHF_RETURNCLOCK_SALT_KEY_ID_ENV` /
`--returnclock-salt-key-id-env` similarly selects the env containing the
non-secret rotation id. Never add the salt value or a production DSN to a
checked-in MCP config. `RETURNCLOCK_DATASET_SALT_KEY_ID` is safe to log and
must change when the secret rotates.

The datasets directory is a private boundary: tool paths must resolve beneath
it, symlinks/path traversal are rejected, private files use mode `0600`, and
raw user IDs are not an export option. Files and fresh versioned directories
are promoted only after validation; directory `overwrite=true` is rollback-safe
for handled errors but not crash-atomic, so immutable training handoffs should
use `overwrite=false`. ReturnClock output is pseudonymized, not anonymous; keep
it in closed training storage.
