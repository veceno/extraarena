#!/usr/bin/env bash
# RLHF-среда ExtraArena — лаунчер
# Использование:
#   ./rlhf_env/start_rlhf_env.sh                       # web @ 127.0.0.1:8090
#   ./rlhf_env/start_rlhf_env.sh --port 9000           # кастомный порт
#   ./rlhf_env/start_rlhf_env.sh --host 0.0.0.0        # публичный хост
#   ./rlhf_env/start_rlhf_env.sh --models-dir /path    # сторонние модели (V5-чекпоинты и т.п.)
#   ./rlhf_env/start_rlhf_env.sh mcp                   # запуск MCP-сервера (stdio)
#   ./rlhf_env/start_rlhf_env.sh setup                 # только создать venv + поставить deps
#   ./rlhf_env/start_rlhf_env.sh help                  # эта справка

set -euo pipefail

# ----- Defaults -----------------------------------------------------------
HOST="${RLHF_HOST:-127.0.0.1}"
PORT="${RLHF_PORT:-8090}"
MODELS_DIR="${RLHF_MODELS_DIR:-ai/models}"
SESSIONS_DIR="${RLHF_SESSIONS_DIR:-rlhf_env/sessions}"
CARDS_PATH="${RLHF_CARDS_PATH:-ai/cards.json}"
VENV_DIR="${RLHF_VENV:-rlhf_env/.venv}"
REQUIREMENTS="rlhf_env/requirements.txt"

# ----- Helpers ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[1;34m[rlhf-env]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[rlhf-env]\033[0m %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
RLHF-среда ExtraArena — лаунчер.

Команды:
  web             запуск web-сервера (по умолчанию)
  mcp             запуск MCP stdio-сервера
  setup           только создать venv и поставить зависимости
  help            эта справка

Опции:
  --host HOST         хост (default: 127.0.0.1)
  --port PORT         порт (default: 8090)
  --models-dir DIR    директория с .onnx + sidecar (default: ai/models)
  --sessions-dir DIR  куда писать сессии (default: rlhf_env/sessions)
  --cards-path FILE   путь к cards.json (default: ai/cards.json)
  --venv DIR          путь к venv (default: rlhf_env/.venv)
  --no-venv           не создавать/использовать venv (запускать в системном python)

Переменные окружения (override дефолтов):
  RLHF_HOST, RLHF_PORT, RLHF_MODELS_DIR, RLHF_SESSIONS_DIR, RLHF_CARDS_PATH,
  RLHF_VENV, RLHF_LOG_LEVEL
EOF
}

# ----- venv setup ---------------------------------------------------------
ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    err "python3 не найден. Установите Python 3.10+ и попробуйте снова."
    exit 1
  fi
  PY_VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  log "Python: $PY ($PY_VERSION)"
}

ensure_venv() {
  if [ "${NO_VENV:-0}" = "1" ]; then
    PY=python3
    return
  fi
  if [ ! -d "$VENV_DIR" ]; then
    log "Создаю venv в $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY=python
  log "venv активен: $VENV_DIR"
}

install_deps() {
  log "Проверяю зависимости…"
  if [ ! -f "$REQUIREMENTS" ]; then
    err "Файл зависимостей не найден: $REQUIREMENTS"
    exit 1
  fi
  # Установить только если нужно
  if ! "$PY" -c "import aiohttp, numpy, onnxruntime" >/dev/null 2>&1; then
    log "Ставлю пакеты из $REQUIREMENTS…"
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r "$REQUIREMENTS"
  else
    log "Зависимости уже установлены (aiohttp, numpy, onnxruntime)"
  fi
}

check_port() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:"$port" -sTCP:LISTEN -P -n >/dev/null 2>&1; then
      err "Порт $port уже занят. Используйте --port или освободите порт."
      return 1
    fi
  fi
  return 0
}

check_paths() {
  if [ ! -d "$MODELS_DIR" ]; then
    err "Директория моделей не найдена: $MODELS_DIR"
    err "  Укажите --models-dir или создайте ai/models/."
    exit 1
  fi
  if [ ! -f "$CARDS_PATH" ]; then
    err "Каталог карт не найден: $CARDS_PATH"
    err "  Укажите --cards-path или создайте ai/cards.json."
    exit 1
  fi
  # Создадим sessions_dir, если нужно
  mkdir -p "$SESSIONS_DIR"
}

# ----- Parse args ---------------------------------------------------------
COMMAND="web"
NO_VENV=0

while [ $# -gt 0 ]; do
  case "$1" in
    web|mcp|setup|help) COMMAND="$1"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --sessions-dir) SESSIONS_DIR="$2"; shift 2 ;;
    --cards-path) CARDS_PATH="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --no-venv) NO_VENV=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Неизвестный аргумент: $1"; usage; exit 1 ;;
  esac
done

# ----- Run ----------------------------------------------------------------
ensure_python

case "$COMMAND" in
  help)
    usage
    exit 0
    ;;
  setup)
    if [ "$NO_VENV" = "0" ]; then ensure_venv; fi
    install_deps
    log "Setup OK."
    exit 0
    ;;
  web)
    if [ "$NO_VENV" = "0" ]; then ensure_venv; fi
    install_deps
    check_paths
    if ! check_port "$PORT"; then exit 1; fi
    log "Запускаю web @ http://$HOST:$PORT"
    exec "$PY" -m rlhf_env.server \
      --host "$HOST" \
      --port "$PORT" \
      --models-dir "$MODELS_DIR" \
      --sessions-dir "$SESSIONS_DIR" \
      --cards-path "$CARDS_PATH"
    ;;
  mcp)
    if [ "$NO_VENV" = "0" ]; then ensure_venv; fi
    install_deps
    check_paths
    log "Запускаю MCP-сервер (stdio)"
    exec "$PY" -m rlhf_env.mcp_server \
      --models-dir "$MODELS_DIR" \
      --sessions-dir "$SESSIONS_DIR" \
      --cards-path "$CARDS_PATH"
    ;;
  *)
    err "Неизвестная команда: $COMMAND"
    usage
    exit 1
    ;;
esac