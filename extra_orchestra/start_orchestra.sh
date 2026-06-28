#!/usr/bin/env bash
# ExtraOrchestra launcher — mirrors rlhf_env/start_rlhf_env.sh.
# Ensures venv + deps, checks the port, execs the aiohttp server.
#
#   ./extra_orchestra/start_orchestra.sh             # HTTP-сервер (редактор + арена, порт 8095)
#   ./extra_orchestra/start_orchestra.sh mcp         # MCP stdio-сервер (для агентов)
#   ORCH_AUTO_START=0 ./...start_orchestra.sh mcp    # MCP без auto-start HTTP-сервера
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

HOST="${ORCHESTRA_HOST:-127.0.0.1}"
PORT="${ORCHESTRA_PORT:-8095}"
BASE_URL="${ORCH_BASE_URL:-http://${HOST}:${PORT}}"

COMMAND="server"
if [ $# -gt 0 ] && [ "$1" = "mcp" ]; then
  COMMAND="mcp"; shift
fi

VENV="$ROOT/.venv"
if [ ! -d "$VENV" ]; then
  echo "[orchestra] creating venv at $VENV"
  python3 -m venv "$VENV"
fi

# Выбираем python: venv, если в нём есть aiohttp; иначе системный python3.
PY="python3"
if [ -f "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import aiohttp' 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PY="python"
else
  if ! python3 -c 'import aiohttp' 2>/dev/null; then
    echo "[orchestra] missing aiohttp — pip install -r extra_orchestra/requirements.txt" >&2
  fi
fi

if [ "$COMMAND" = "server" ]; then
  if command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[orchestra] port $PORT already in use" >&2
      exit 1
    fi
  fi
  exec "$PY" -m extra_orchestra.server --host "$HOST" --port "$PORT" "$@"
fi

# mcp: auto-start HTTP-сервера по умолчанию (ORCH_AUTO_START=0 чтобы выключить)
exec "$PY" -m extra_orchestra.mcp_server --base-url "$BASE_URL" "$@"