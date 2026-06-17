#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "$DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$DIR/.env"
    set +a
fi

if [ "${DATABASE_URL:-}" = "postgresql://user:password@localhost:5432/extraarena" ]; then
    echo "⚠️  DATABASE_URL в .env похож на пример из .env.example; использую локальную БД проекта."
    export DATABASE_URL=""
    export DB_HOST="${LOCAL_DB_HOST:-localhost}"
    export DB_PORT="${LOCAL_DB_PORT:-5432}"
    export DB_USER="${LOCAL_DB_USER:-postgres}"
    export DB_PASSWORD="${LOCAL_DB_PASSWORD:-}"
    export DB_NAME="${LOCAL_DB_NAME:-extraarena}"
fi

LOG_FILE="$DIR/extraarena.log"
PID_FILE="$DIR/extraarena.pid"
CLOUDPUB_PID_FILE="$DIR/cloudpub.pid"
PORT="${WEBAPP_PORT:-${WEB_PORT:-8081}}"
CLOUDPUB_BIN="${CLOUDPUB_BIN:-/Applications/cloudpub.app/Contents/MacOS/cloudpub}"
CLOUDPUB_SERVICE_GUID="${CLOUDPUB_SERVICE_GUID:-}"
CLOUDPUB_PUBLIC_URL="${CLOUDPUB_PUBLIC_URL:-${WEBAPP_URL:-}}"
export MATCH_STATE_BACKEND="${MATCH_STATE_BACKEND:-memory}"
export WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"

if [ "$(printf '%s' "$MATCH_STATE_BACKEND" | tr '[:upper:]' '[:lower:]')" = "memory" ]; then
    for WORKER_ENV in WEB_CONCURRENCY GUNICORN_WORKERS UVICORN_WORKERS; do
        WORKER_COUNT="${!WORKER_ENV:-}"
        if [ -z "$WORKER_COUNT" ]; then
            continue
        fi
        case "$WORKER_COUNT" in
            *[!0-9]*)
                echo "❌ $WORKER_ENV должен быть целым числом для MATCH_STATE_BACKEND=memory."
                exit 1
                ;;
        esac
        if [ "$WORKER_COUNT" -gt 1 ]; then
            echo "❌ MATCH_STATE_BACKEND=memory поддерживает только один web worker."
            echo "   Установите WEB_CONCURRENCY=1 и не задавайте GUNICORN_WORKERS/UVICORN_WORKERS больше 1."
            echo "   Для multi-worker сначала нужен shared match-state backend."
            exit 1
        fi
    done
fi

health_ready() {
    local url="$1"
    python3 - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        if response.status < 200 or response.status >= 300:
            sys.exit(1)
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

if payload.get("status") == "ok" and payload.get("service") == "extraarena-webapp":
    sys.exit(0)
sys.exit(1)
PY
}

pid_matches_project() {
    local pid="$1"
    local cmd cwd

    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
    case "$cmd" in
        *"$DIR/main.py"*|*"python"*"$DIR"*|*"Python"*"$DIR"*)
            return 0
            ;;
        *" main.py"*)
            [ "$cwd" = "$DIR" ]
            return $?
            ;;
        *)
            return 1
            ;;
    esac
}

stop_pid() {
    local pid="$1"
    local label="${2:-процесс}"

    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    echo "🔻 Останавливаю $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true

    for _ in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done

    echo "⚠️  PID $pid не завершился, принудительно останавливаю..."
    kill -9 "$pid" 2>/dev/null || true
}

# Убить старый процесс из pid-файла, если висит
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ]; then
        if pid_matches_project "$OLD_PID"; then
            stop_pid "$OLD_PID" "старый процесс из $PID_FILE"
        else
            echo "⚠️  $PID_FILE указывает на чужой или уже завершенный PID $OLD_PID; не останавливаю его."
        fi
    fi
fi

if [ -f "$CLOUDPUB_PID_FILE" ]; then
    OLD_CLOUDPUB_PID=$(cat "$CLOUDPUB_PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_CLOUDPUB_PID" ]; then
        stop_pid "$OLD_CLOUDPUB_PID" "старый CloudPub из $CLOUDPUB_PID_FILE"
    fi
    rm -f "$CLOUDPUB_PID_FILE"
fi

# Убить orphan-процессы этого проекта, которые реально держат веб-порт.
PORT_PIDS=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
for PORT_PID in $PORT_PIDS; do
    CMD=$(ps -p "$PORT_PID" -o command= 2>/dev/null || true)
    CWD=$(lsof -a -p "$PORT_PID" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
    case "$CMD" in
        *"$DIR/main.py"*|*"python"*"$DIR"*|*"Python"*"$DIR"*)
            stop_pid "$PORT_PID" "процесс, слушающий порт $PORT"
            ;;
        *" main.py"*)
            if [ "$CWD" = "$DIR" ]; then
                stop_pid "$PORT_PID" "процесс, слушающий порт $PORT"
            else
                echo "❌ Порт $PORT занят процессом main.py из другой директории (PID $PORT_PID, cwd=$CWD): $CMD"
                exit 1
            fi
            ;;
        *)
            echo "❌ Порт $PORT занят чужим процессом (PID $PORT_PID): $CMD"
            echo "   Остановите его вручную или измените WEB_PORT."
            exit 1
            ;;
    esac
done

rm -f "$PID_FILE"

if [ "${EXTRAARENA_TRUNCATE_LOG:-false}" = "true" ]; then
    : > "$LOG_FILE"
else
    touch "$LOG_FILE"
fi

echo "🚀 Запускаю ExtraArena..."
python3 - "$DIR" "$LOG_FILE" "$PID_FILE" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
log_path = Path(sys.argv[2])
pid_path = Path(sys.argv[3])

pid = os.fork()
if pid > 0:
    raise SystemExit

os.setsid()

pid = os.fork()
if pid > 0:
    raise SystemExit

os.chdir(root)
os.umask(0)

flags = os.O_WRONLY | os.O_CREAT
if os.environ.get("EXTRAARENA_TRUNCATE_LOG", "").lower() in {"1", "true", "yes", "on"}:
    flags |= os.O_TRUNC
else:
    flags |= os.O_APPEND
fd = os.open(log_path, flags, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.close(fd)

with open("/dev/null", "rb", buffering=0) as devnull:
    os.dup2(devnull.fileno(), 0)

with open(pid_path, "w") as f:
    f.write(str(os.getpid()))

os.execvp("python3", ["python3", "main.py"])
PY

for _ in $(seq 1 20); do
    if [ -s "$PID_FILE" ]; then
        break
    fi
    sleep 0.1
done
PID=$(cat "$PID_FILE")

echo "📋 PID: $PID"
echo "📄 Лог: $LOG_FILE"

# Ждём готовности веб-сервера
echo "⏳ Ожидаю веб-сервер..."
READY=0
for i in $(seq 1 30); do
    if health_ready "http://127.0.0.1:$PORT/ready"; then
        echo "✅ Веб-сервер готов (http://127.0.0.1:$PORT/ready)"
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "❌ Веб-сервер не прошёл readiness-check: http://127.0.0.1:$PORT/ready"
    echo "📄 Последние строки лога:"
    tail -n 80 "$LOG_FILE" 2>/dev/null || true
    stop_pid "$PID" "неуспешно стартовавший веб-сервер"
    exit 1
fi

if [ -n "$CLOUDPUB_SERVICE_GUID" ] && [ -x "$CLOUDPUB_BIN" ]; then
    echo "⏳ Запускаю CloudPub-туннель..."
    nohup "$CLOUDPUB_BIN" start "$CLOUDPUB_SERVICE_GUID" > /dev/null 2>&1 &
    CLOUDPUB_PID=$!
    disown "$CLOUDPUB_PID" 2>/dev/null || true
    echo "$CLOUDPUB_PID" > "$CLOUDPUB_PID_FILE"
    echo "📋 CloudPub PID: $CLOUDPUB_PID"
    if [ -n "$CLOUDPUB_PUBLIC_URL" ]; then
        CLOUDPUB_READY=0
        for _ in $(seq 1 15); do
            if health_ready "$CLOUDPUB_PUBLIC_URL/ready"; then
                echo "✅ CloudPub готов ($CLOUDPUB_PUBLIC_URL)"
                CLOUDPUB_READY=1
                break
            fi
            sleep 2
        done
        if [ "$CLOUDPUB_READY" -eq 1 ]; then
            sleep 20
            if ! health_ready "$CLOUDPUB_PUBLIC_URL/ready"; then
                CLOUDPUB_READY=0
                echo "⚠️  CloudPub ответил при старте, но затем стал недоступен: $CLOUDPUB_PUBLIC_URL/ready"
            fi
        fi
        if [ "$CLOUDPUB_READY" -ne 1 ]; then
            echo "⚠️  CloudPub не ответил на $CLOUDPUB_PUBLIC_URL/ready"
        fi
    fi
    if ! kill -0 "$CLOUDPUB_PID" 2>/dev/null; then
        echo "⚠️  CloudPub-процесс завершился сразу после запуска; проверьте настройки CloudPub."
        rm -f "$CLOUDPUB_PID_FILE"
    fi
elif [ -n "$CLOUDPUB_SERVICE_GUID" ]; then
    echo "⚠️  CloudPub CLI не найден: $CLOUDPUB_BIN"
fi

# Ждём старта бота
echo "⏳ Ожидаю Telegram бота..."
for i in $(seq 1 30); do
    if grep -q "Run polling" "$LOG_FILE" 2>/dev/null; then
        BOT_LINE=$(grep "Run polling" "$LOG_FILE" | tail -1)
        echo "✅ $BOT_LINE"
        break
    fi
    sleep 2
done

echo ""
echo "🎮 ExtraArena запущен."
echo "   Веб-интерфейс: http://127.0.0.1:$PORT"
echo "   Просмотр логов: tail -f $LOG_FILE"
echo "   Остановка:      kill $PID"
