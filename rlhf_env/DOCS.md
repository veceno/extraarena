# RLHF-среда ExtraArena — Полная документация

> **Версия:** 0.1.0
> **Дата:** 2026-06-24
> **Зачем:** автономная среда для сбора обучающих траекторий (human-vs-model,
> model-vs-model) на детерминированном движке ExtraArena. Не зависит от
> прод-стека, запускается отдельно, хранит данные в файлах.

---

## Содержание

1. [Контекст и мотивация](#1-контекст-и-мотивация)
2. [Архитектура](#2-архитектура)
3. [Установка и запуск](#3-установка-и-запуск)
4. [Web-интерфейс](#4-web-интерфейс)
5. [REST API](#5-rest-api)
6. [WebSocket-протокол](#6-websocket-протокол)
7. [MCP-сервер](#7-mcp-сервер)
8. [Формат данных на диске](#8-формат-данных-на-диске)
9. [Добавление собственных моделей](#9-добавление-собственных-моделей)
10. [CLI-лаунчер start_rlhf_env.sh](#10-cli-лаунчер-start_rlhf_envsh)
11. [Программный API (Python)](#11-программный-api-python)
12. [Тестирование](#12-тестирование)
13. [Troubleshooting](#13-troubleshooting)
14. [Расширение среды](#14-расширение-среды)

---

## 1. Контекст и мотивация

### 1.1. Проблема

Боты Extra-LR-V4 / V4-Max обучены на синтетике / self-play. Чтобы поднять
V5 и далее, нужны **реальные игровые траектории**: человек играет против
модели, и каждый ход + состояние + метаданные записываются для последующего
imitation learning или RLHF.

### 1.2. Почему отдельная среда

- **Не задеваем прод.** БД, бот, прод-веб остаются нетронутыми.
- **Отдельный порт** (8090) и процесс — можно запустить/остановить когда угодно.
- **Файлы вместо БД** — проще инспектировать, версионировать, переносить.
- **MCP-агентам** удобно: запустить N боёв, забрать логи, поменять модель «на лету».
- **Web 1:1 как прод** — те же CSS-классы `.arena-*`, `.board-slot`, `.hand-card`, `.hp-*`.

### 1.3. Что внутри

| Компонент | Назначение |
|-----------|-----------|
| `server.py` | aiohttp app: HTTP + WebSocket |
| `mcp_server.py` | MCP stdio (JSON-RPC 2.0) |
| `components/battle_runner.py` | Один матч с engine.step + лог |
| `components/session_manager.py` | asyncio-словарь активных групп |
| `components/policy_registry.py` | Сканирует `ai/models/*.onnx` + sidecar |
| `components/policy_factory.py` | `build_policy(spec)` → адаптер |
| `components/deck_builder.py` | Случайная арена-дека |
| `components/manifest.py` | Manifest + summary JSON |
| `components/log_schema.py` | Схема battle_log v1.0 |
| `components/inference_params.py` | Defaults от sidecar |
| `index.html` / `battle.html` | UI |

---

## 2. Архитектура

### 2.1. Слои

```
┌──────────────────────────────────────────────────────────────┐
│                       Browser @ 8090                          │
│   (1:1 CSS, WebSocket к /ws/groups/.../battles/...)            │
└──────────────────────────────────────────────────────────────┘
                            │ HTTP / WS
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  aiohttp server.py                            │
│   /health /api/registry/* /api/groups/* /ws/groups/...        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                SessionManager                                 │
│   asyncio.create_task(_run_group(state)) per group             │
│   tracks GroupState (status, task, manifest_writer)            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│            BattleRunner per battle                             │
│   engine.step(legal[policy.get_action(state, pid, legal)])     │
│   writes battle_log.json + append_battle_result(manifest)      │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│      core.engine.ArenaEnvironment                             │
│   deterministic, pure (state_in, action) → (state_out, ok)    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│   ai.bot_brain.BerserkInference  или  baselines               │
│   V3 (legacy ONNX)  /  V4 (action-conditioned) / random …    │
└──────────────────────────────────────────────────────────────┘
```

### 2.2. Поток данных (один бой)

```
loop:
  1. legal = engine.get_legal_actions(pid)
  2. if pid == human: ждём WS-сообщение {index: N}
     else: action_idx = policy.get_action(state, pid, legal, params)
  3. ok, err = engine.step(pid, legal[action_idx])
  4. log.append({turn, actor, action, state_before, state_after, ts_ms})
  5. if pid == human: broadcast state via WS
  6. check victory / max_turns → break
end loop
write battle_log.json
manifest.append_battle_result(...)
```

### 2.3. Поток данных (группа)

```
Browser "Запустить N боёв" → POST /api/groups {p1, p2, count, deck, seed, …}
   ↓
session_manager.start(spec)
   ├─ group_id = uuid4().hex[:12]
   ├─ ManifestWriter(group_id, spec, group_dir)  → manifest.json сразу
   └─ for i in 1..N:
         BattleRunner(...).arun()
         manifest.append_battle_result(...)
   └─ manifest.finalize()  → summary.json + finished_at
```

### 2.4. State на диске

```
rlhf_env/sessions/<group_id>/
├── manifest.json        ← группа: spec, env, results, battle_ids, battles_results
├── summary.json         ← финальный winrate + агрегаты
└── battles/
    ├── <battle_id_1>.json  ← полный battle_log (1.0)
    ├── <battle_id_2>.json
    └── ...
```

---

## 3. Установка и запуск

### 3.1. Зависимости

`rlhf_env/requirements.txt`:
```
aiohttp>=3.9.0
numpy>=1.24.0
onnxruntime>=1.17.0
mcp>=1.0.0
```

### 3.2. Запуск через `start_rlhf_env.sh`

```bash
./rlhf_env/start_rlhf_env.sh                       # web @ 127.0.0.1:8090
./rlhf_env/start_rlhf_env.sh --port 9000           # другой порт
./rlhf_env/start_rlhf_env.sh --host 0.0.0.0        # доступ снаружи
./rlhf_env/start_rlhf_env.sh --models-dir /path    # другая папка моделей
./rlhf_env/start_rlhf_env.sh --sessions-dir /path  # другая папка сессий
./rlhf_env/start_rlhf_env.sh mcp                   # MCP stdio
./rlhf_env/start_rlhf_env.sh setup                 # только venv + deps
./rlhf_env/start_rlhf_env.sh --no-venv             # использовать системный Python
```

Что делает скрипт:
1. Создаёт `rlhf_env/.venv` если нет.
2. Ставит deps из `requirements.txt`.
3. Проверяет, что порт свободен.
4. Стартует `python -m rlhf_env.server` (или `mcp_server`).

### 3.3. Запуск вручную

```bash
python3 -m rlhf_env.server --port 8090
python3 -m rlhf_env.mcp_server    # для MCP
```

Env-vars (читаются `server.py`):
- `RLHF_HOST` (default `127.0.0.1`)
- `RLHF_PORT` (default `8090`)
- `RLHF_MODELS_DIR` (default `ai/models`)
- `RLHF_SESSIONS_DIR` (default `rlhf_env/sessions`)
- `RLHF_CARDS_PATH` (default `ai/cards.json`)

### 3.4. Что НЕ требуется

- Не нужен запущенный прод-сервер.
- Не нужна БД.
- Не нужны права админа.

---

## 4. Web-интерфейс

### 4.1. Главная страница (`/`)

Форма «Новая группа боёв»:
- **p1_model** / **p2_model** — выбор модели (dropdown из registry).
- **deck_strategy** — `random_arenaenv` или `custom_json`.
- **custom_deck_p1** / **custom_deck_p2** — JSON-список `[hero, *warriors, *potions]`.
- **battles_planned** — количество боёв.
- **max_turns** — лимит ходов на бой (default 60).
- **seed** — для воспроизводимости.
- **starting_player** — `random` / `p1` / `p2`.

После нажатия «Запустить» — редирект на `/groups/<gid>`.

### 4.2. Страница группы (`/groups/<gid>`)

- Статус (running / completed / error).
- Прогресс-бар по боям.
- Winrate p1 / p2.
- Таблица battle_id → ссылка на `/groups/<gid>/battles/<bid>`.

### 4.3. Страница боя (`/groups/<gid>/battles/<bid>`)

- 1:1 рендер арены (использует `arena_styles.css`).
- Кнопки «Закончить ход», «Сыграть карту», «Атаковать».
- WebSocket для human-vs-model.

---

## 5. REST API

### 5.1. `GET /health`

```json
{"status": "ok", "models_loaded": 7, "rlhf_env_version": "0.1.0"}
```

### 5.2. `GET /api/registry/models`

Возвращает список всех моделей, найденных в `ai/models/*.onnx` + sidecar.

```json
{
  "models": [
    {
      "name": "extra-lr-v4-max",
      "kind": "action_onnx",
      "policy_file": "ai/models/extra-lr-v4-max.onnx",
      "sidecar_file": "ai/models/extra-lr-v4-max.onnx.json",
      "inference": {"mode": "argmax", "temperature": 0.0, ...}
    },
    {"name": "random", "kind": "random", ...},
    {"name": "end_turn", "kind": "end_turn", ...}
  ]
}
```

### 5.3. `GET /api/registry/sample-deck`

```json
{
  "deck": [1, 42, 14, 39, 27, 14, 44, 24, 44, 33, 24, 33, 39, 42, 27, 8, 8, 13, 13, 10, 10],
  "size": 21
}
```

### 5.4. `POST /api/groups`

Запустить группу.

```json
{
  "p1_model": "extra-lr-v4-max",
  "p2_model": "end_turn",
  "deck_strategy": "random_arenaenv",
  "battles_planned": 5,
  "seed": 42,
  "starting_player": "random",
  "max_turns": 40
}
```

Ответ:
```json
{"group_id": "a3b8c1d2e4f5", "status": "running"}
```

### 5.5. `GET /api/groups`

Список всех групп (running + completed/loaded с диска).

### 5.6. `GET /api/groups/{gid}`

```json
{
  "group_id": "a3b8c1d2e4f5",
  "status": "completed",
  "started_at": "2026-06-24T12:00:00Z",
  "finished_at": "2026-06-24T12:05:00Z",
  "battles_planned": 5,
  "battles_finished": 5,
  "winrate_p1": 0.6,
  "winrate_p2": 0.4,
  "last_error": null
}
```

### 5.7. `GET /api/groups/{gid}/manifest`

Полный manifest.json (включая spec, env, results, battles_results, battle_ids).

### 5.8. `GET /api/groups/{gid}/battles`

```json
{"battle_ids": ["...", "...", "..."]}
```

### 5.9. `GET /api/groups/{gid}/battles/{bid}`

Полный battle_log.json.

### 5.10. `POST /api/groups/{gid}/stop`

Остановить running группу. Если группа уже completed — возвращает 404.

---

## 6. WebSocket-протокол

### 6.1. Endpoint

`ws://127.0.0.1:8090/ws/groups/{gid}/battles/{bid}`

### 6.2. Сообщения клиент → сервер

```json
{"type": "action", "index": 3}        // выбрать legal[3]
{"type": "ready"}                     // готов начать / продолжить
```

### 6.3. Сообщения сервер → клиент

```json
{"type": "state", "view": {...state_view...}}
{"type": "legal_actions", "actions": [{...}, ...]}
{"type": "ended", "result": {"status": "P1_WIN", "winner_user_id": 1000}}
{"type": "error", "message": "..."}
```

### 6.4. Пример

```js
const ws = new WebSocket("ws://127.0.0.1:8090/ws/groups/.../battles/...");
ws.onopen = () => ws.send(JSON.stringify({type: "ready"}));
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "legal_actions") renderActions(msg.actions);
  if (msg.type === "state") renderBoard(msg.view);
  if (msg.type === "ended") showResult(msg.result);
};
```

---

## 7. MCP-сервер

### 7.1. Запуск

```bash
./rlhf_env/start_rlhf_env.sh mcp
# или
python3 -m rlhf_env.mcp_server
```

stdio JSON-RPC 2.0.

### 7.2. Инструменты

| Tool | Аргументы | Возврат |
|------|-----------|---------|
| `start_battle_group` | `spec: dict` | `{group_id, manifest_path}` |
| `stop_battle_group` | `group_id: str` | `{stopped: bool, finished_battles: int}` |
| `list_battle_groups` | — | `[{group_id, status, winrate}, ...]` |
| `get_battle_group_status` | `group_id: str` | `{status, current_battle, winrate}` |
| `get_battle_group_manifest` | `group_id: str` | `{manifest: dict, path: str}` |
| `download_battle_logs` | `group_id, format="json"|"zip"` | `{path, size}` |

### 7.3. Пример (curl-style через stdin)

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 -m rlhf_env.mcp_server
```

Ожидаем 6 инструментов в ответе.

```bash
echo '{
  "jsonrpc":"2.0","id":2,
  "method":"tools/call",
  "params":{
    "name":"start_battle_group",
    "arguments":{"spec":{"p1_model":"extra-lr-v4-max","p2_model":"random","battles_planned":3}}
  }
}' | python3 -m rlhf_env.mcp_server
```

---

## 8. Формат данных на диске

### 8.1. manifest.json

```json
{
  "manifest_version": "1.0",
  "group_id": "a3b8c1d2e4f5",
  "created_at": "2026-06-24T12:00:00Z",
  "finished_at": "2026-06-24T12:05:00Z",
  "spec": {
    "p1_model": "extra-lr-v4-max",
    "p2_model": "random",
    "deck_strategy": "random_arenaenv",
    "custom_decks": null,
    "battles_planned": 5,
    "seed": 42,
    "starting_player": "random",
    "max_turns": 60
  },
  "env": {
    "rlhf_env_version": "0.1.0",
    "core_engine_commit": "a012cac1",
    "python_version": "3.11.5",
    "platform": "macOS-26.5-arm64",
    "onnxruntime_version": "1.22.1",
    "numpy_version": "2.3.2",
    "aiohttp_version": "3.9.1"
  },
  "results": {
    "battles_finished": 5,
    "battles_planned": 5,
    "p1_wins": 3, "p2_wins": 1, "draws": 1,
    "winrate_p1": 0.6, "winrate_p2": 0.2,
    "avg_turns": 12.3, "avg_duration_seconds": 0.42
  },
  "battle_ids": ["b1", "b2", "b3", "b4", "b5"],
  "battles_results": [
    {
      "battle_id": "b1",
      "battle_log_path": "rlhf_env/sessions/.../battles/b1.json",
      "winner_user_id": 1000, "loser_user_id": 2000,
      "status": "P1_WIN", "turns": 12, "duration_seconds": 0.4
    }
  ]
}
```

### 8.2. summary.json

Краткий финальный snapshot (аналог `results` из manifest).

### 8.3. battles/&lt;bid&gt;.json

```json
{
  "log_version": "1.0",
  "battle_id": "b1",
  "group_id": "a3b8c1d2e4f5",
  "started_at": "2026-06-24T12:00:00Z",
  "finished_at": "2026-06-24T12:00:42Z",
  "duration_seconds": 0.4,
  "result": {
    "winner_user_id": 1000,
    "loser_user_id": 2000,
    "status": "P1_WIN"
  },
  "models": {
    "p1": {"name": "extra-lr-v4-max", "kind": "action_onnx", "policy_file": "..."},
    "p2": {"name": "random", "kind": "random", "policy_file": null}
  },
  "decks": {"p1": [1, 14, ...], "p2": [2, 23, ...]},
  "actions": [
    {
      "turn": 1,
      "actor": 1,
      "kind": "play_card",
      "action_dict": {"type": "play_card", "hand_index": 0, "target_id": "..."},
      "timestamp_ms": 1234,
      "state_before_summary": {"p1_hp": 30, "p2_hp": 30, "p1_mana": 1, ...},
      "state_after_summary": {"p1_hp": 30, "p2_hp": 28, "p1_mana": 0, ...},
      "action_history_new_lines": ["P1: Игрок 1 кладёт карту 'X'"]
    }
  ],
  "final_state_summary": {"turn_number": 14, "p1_hp": 0, "p2_hp": 12, ...}
}
```

---

## 9. Добавление собственных моделей

### 9.1. V4 (action-conditioned ONNX)

Положите в `ai/models/`:
- `my-v5.onnx`
- `my-v5.onnx.json` (sidecar)

Sidecar (`my-v5.onnx.json`):
```json
{
  "format": "train_v2_classic_v1",
  "obs_dim": 256,
  "num_actions": 128,
  "temperature": 0.0,
  "selection": "argmax"
}
```

Поле `format` обязательно `"train_v2_classic_v1"` — тогда `BerserkInference`
из `ai.bot_brain` подхватит модель через `policy_factory._load_berserk()`.

### 9.2. V3 (legacy ONNX)

Один файл `my-v3.onnx` без sidecar. `policy_registry.inspect_model()`
автоматически определит формат через probe input shape.

Используется `LegacyOnnxPolicy` из `ai.model_benchmark.policies` +
`ai.model_benchmark.legacy_codec` для кодирования/декодирования.

### 9.3. Baselines

`policy_factory.build_policy({"name": "random"})` — встроенные политики,
не требуют ONNX:
- `random` — `_RLHFRandomPolicy`
- `greedy_face` — `_RLHFGreedyFacePolicy`
- `end_turn` — `_RLHFEndTurnPolicy`

### 9.4. Сторонние модели

`policy_registry.PolicyRegistry.scan(models_dir)` принимает любой путь:

```bash
./rlhf_env/start_rlhf_env.sh --models-dir /path/to/external_models
```

Поддерживаются оба формата (V4 sidecar + V3 legacy auto-detect).

---

## 10. CLI-лаунчер start_rlhf_env.sh

### 10.1. Subcommands

| Subcommand | Что делает |
|------------|-----------|
| `web` (default) | Запустить web-сервер |
| `mcp` | Запустить MCP stdio-сервер |
| `setup` | Только создать venv + установить deps |
| `help` | Показать help |

### 10.2. Флаги web

```
--host <ip>           (default 127.0.0.1)
--port <int>          (default 8090)
--models-dir <path>   (default ai/models)
--sessions-dir <path> (default rlhf_env/sessions)
--cards-path <path>   (default ai/cards.json)
--venv <path>         (default rlhf_env/.venv)
--no-venv             (использовать системный python)
```

### 10.3. Env vars

Все флаги имеют env-аналоги (см. `RLHF_*`).

### 10.4. Graceful shutdown

`SIGTERM` → aiohttp завершает текущие бои, сохраняет manifest.

---

## 11. Программный API (Python)

### 11.1. Минимальный пример

```python
import asyncio
from pathlib import Path
from rlhf_env.components.session_manager import SessionManager
from rlhf_env.components.policy_registry import PolicyRegistry

async def main():
    sm = SessionManager(
        sessions_dir=Path("/tmp/rlhf_runs"),
        models_dir=Path("ai/models"),
        registry=PolicyRegistry.scan("ai/models"),
    )
    spec = {
        "p1_model": "extra-lr-v4-max",
        "p2_model": "end_turn",
        "battles_planned": 10,
        "seed": 42,
        "max_turns": 60,
    }
    gid = sm.start(spec)
    while True:
        s = sm.status(gid)
        if s["status"] in ("completed", "error"):
            break
        await asyncio.sleep(0.5)
    print(sm.get_manifest(gid))

asyncio.run(main())
```

### 11.2. Без event loop

Если у вас уже синхронный код и нет loop — используйте `astart`:

```python
gid = asyncio.run(sm.astart(spec))
```

### 11.3. Кастомная политика

```python
from rlhf_env.components.policy_factory import _RLHFRandomPolicy
pol = _RLHFRandomPolicy(seed=0)

from rlhf_env.components.battle_runner import BattleRunner
from core.engine import ArenaEnvironment

runner = BattleRunner(
    group_id="my", battle_id="b1",
    policy_a=pol, policy_b=pol,
    engine=ArenaEnvironment(state),
    battle_log_path=Path("/tmp/battle.json"),
    max_turns=20,
)
log = asyncio.run(runner.arun())
```

---

## 12. Тестирование

### 12.1. Unit-тесты

```bash
python3 -m pytest rlhf_env/tests/ -v
# 44 теста, ~3 сек
```

Покрытие:
- `test_log_schema.py` — battle_log v1.0, manifest v1.0
- `test_deck_builder.py` — каталог, случайные деки, парсинг, валидация
- `test_policy_factory.py` — baselines, V4 sidecar, V3 legacy, ошибки
- `test_battle_runner.py` — 1 бой end-to-end (random / V4-Max)
- `test_manifest.py` — writer, append, finalize, resume
- `test_session_manager.py` — start/complete/stop/manifest/list

### 12.2. Smoke E2E

```bash
python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 2 --models random
python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 1 --models v4-max
```

Запускает свой сервер, создаёт группу, валидирует manifest + battle_log,
проверяет файлы на диске. Коды возврата: `0` = OK, `1` = fail.

### 12.3. Регрессии

Перед коммитом:
```bash
python3 -m pytest rlhf_env/tests/ -q        # наши новые
python3 -m pytest tests/ -q --ignore=...    # прод-тесты (если меняли engine)
```

---

## 13. Troubleshooting

### 13.1. Сервер не стартует: `OSError: [Errno 48] Address already in use`

Порт занят. `--port 8091` или убейте процесс:
```bash
lsof -i :8090
kill <PID>
```

### 13.2. `model not found in registry`

`ai/models/<name>.onnx` не найден. Проверьте:
- Файл существует?
- Имя указано БЕЗ расширения `.onnx`?
- `PolicyRegistry.scan(models_dir)` — какой путь сканируется?

### 13.3. `BerserkInference` ругается на shape

V4 sidecar должен иметь `"format": "train_v2_classic_v1"`.
V3 legacy — без sidecar, автоопределение.

### 13.4. Бой не завершается (max_turns)

`max_turns` слишком мал. Увеличьте до 60+. Либо `end_turn` политика
зацикливается — это нормально, она завершает ход пока противник не победит.

### 13.5. WebSocket сразу закрывается

`/ws/groups/{gid}/battles/{bid}` — gid и bid должны существовать.
Проверьте `GET /api/groups/{gid}` — статус должен быть `running` или `loaded`.

### 13.6. Тесты pytest падают с `no current event loop`

Python 3.10+: `asyncio.get_event_loop()` deprecated. Запускайте через
`asyncio.run(main())` или используйте `sm.astart(spec)` из async-контекста.

### 13.7. macOS: `python` not found

Используйте `python3` или `pyenv`. В скрипте `start_rlhf_env.sh` уже
есть авто-детект.

### 13.8. ONNX runtime warning

Подавить: `export ORT_LOGGING_LEVEL=3` перед запуском.

---

## 14. Расширение среды

### 14.1. Добавить новую политику

`policy_factory.py`:
```python
class _MyNewPolicy:
    kind = "my_new"
    def __init__(self, **kwargs): ...
    def get_action(self, state, pid, legal_actions, params=None): ...

def build_policy(spec):
    if spec.get("name") == "my_new":
        return _MyNewPolicy(**spec)
    ...
```

Зарегистрировать в `policy_registry.py::resolve_spec`.

### 14.2. Добавить новый формат deck_strategy

`deck_builder.py`:
```python
def build_my_deck(catalog, rng, **kwargs):
    ...
```

И в `session_manager._build_game_state` — выбор по `spec["deck_strategy"]`.

### 14.3. Добавить метрику в manifest

`manifest.py::append_battle_result` — добавить новое поле в `result` dict
и обновлять в aggregate-логике.

### 14.4. Добавить новый MCP tool

`mcp_server.py`:
```python
self.tools["my_tool"] = {
    "description": "...",
    "inputSchema": {...},
    "handler": self._tool_my_tool,
}
```

### 14.5. Human-in-the-loop

WebSocket endpoint поддерживает `human_player` в `BattleRunner`. Передайте
`human_player=1000` и отправляйте `{type: "action", index: N}` для ходов
человека, остальные ходы за модель.

---

## Приложение A: минимальный sidecar V4

```json
{
  "format": "train_v2_classic_v1",
  "obs_dim": 256,
  "num_actions": 128,
  "temperature": 0.0,
  "selection": "argmax",
  "notes": "Моя V5-Max после self-play 50k шагов"
}
```

## Приложение B: полный spec группы

```json
{
  "p1_model": "extra-lr-v4-max",
  "p2_model": "extra-lr-v3-max",
  "deck_strategy": "random_arenaenv",
  "custom_deck_p1": null,
  "custom_deck_p2": null,
  "battles_planned": 10,
  "seed": 42,
  "starting_player": "random",
  "max_turns": 60,
  "inference": {
    "temperature": 0.1,
    "selection": "sample",
    "obs_dim": 256
  },
  "human_player": null
}
```

## Приложение C: используемые core API

```python
from core.engine import ArenaEnvironment
from core.state import GameState, GameStatus
from core.classic_setup import create_classic_game_state
from core.converter import card_from_db, deck_from_card_ids
```

`engine.get_legal_actions(player_id)` → list[int]
`engine.step(player_id, action_id)` → (ok: bool, error: str | None)
`engine.get_state_copy()` → GameState (для сериализации)
`engine.state` → текущий GameState (для политик)
