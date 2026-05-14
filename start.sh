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

LOG_FILE="$DIR/extraarena.log"
PID_FILE="$DIR/extraarena.pid"
PORT="${WEBAPP_PORT:-${WEB_PORT:-8081}}"

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
        stop_pid "$OLD_PID" "старый процесс из $PID_FILE"
    fi
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

# Очищаем старый лог-файл
: > "$LOG_FILE"

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

fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
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
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Веб-сервер готов (http://127.0.0.1:$PORT/health)"
        break
    fi
    sleep 2
done

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
