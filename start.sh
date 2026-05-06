#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

LOG_FILE="$DIR/extraarena.log"
PID_FILE="$DIR/extraarena.pid"

# Убить старый процесс, если висит
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "🔻 Останавливаю старый процесс (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
fi

# Очищаем старый лог-файл
: > "$LOG_FILE"

echo "🚀 Запускаю ExtraArena..."
nohup python3 main.py >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "📋 PID: $PID"
echo "📄 Лог: $LOG_FILE"

# Ждём готовности веб-сервера
echo "⏳ Ожидаю веб-сервер..."
for i in $(seq 1 30); do
    if curl -sf http://0.0.0.0:8081/health > /dev/null 2>&1; then
        echo "✅ Веб-сервер готов (http://0.0.0.0:8081/health)"
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
echo "   Веб-интерфейс: http://0.0.0.0:8081"
echo "   Просмотр логов: tail -f $LOG_FILE"
echo "   Остановка:      kill $PID"
