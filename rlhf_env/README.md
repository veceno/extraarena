# RLHF-среда ExtraArena

Автономная среда для сбора обучающих данных (человек-vs-модель или модель-vs-модель)
на основе `core.engine.ArenaEnvironment`. Не зависит от прод-стека (БД, бот, прод-веб),
запускается отдельным процессом, порт и пути настраиваются.

## Что это

- Web-интерфейс 1:1 как прод-арена (те же CSS-классы).
- Любые ONNX из `ai/models/` + baselines (random / greedy_face / end_turn).
- Группы боёв (batch'и) с manifest + battle_log в файловой системе.
- MCP-сервер для программного управления группами.
- Никаких изменений в прод-коде, никакой БД.

## Быстрый старт

```bash
# из корня репозитория
./rlhf_env/start_rlhf_env.sh                # web @ 127.0.0.1:8090
./rlhf_env/start_rlhf_env.sh --port 9000    # другой порт
./rlhf_env/start_rlhf_env.sh mcp            # MCP-сервер (stdio)
./rlhf_env/start_rlhf_env.sh setup          # только создать venv + deps
```

После запуска web: открыть `http://127.0.0.1:8090` в браузере, выбрать модель/колоду/количество боёв, нажать «Запустить».

## Архитектура (5 слоёв)

```
[Browser @ 8090]
    ↓ HTTP / WebSocket
[server.py: aiohttp app]
    ↓ start(spec)
[SessionManager] ── manage ──> [asyncio task per group]
    ↓
[BattleRunner per battle]
    ↓ step(legal_action)
[core.engine.ArenaEnvironment]   ← чистое ядро, без обвязки RL
    ↓
[ai/models/*.onnx] (через BerserkInference) или baselines
```

Все бои пишутся в `rlhf_env/sessions/<group_id>/{manifest.json, summary.json, battles/<battle_id>.json}`.

## Структура каталога

```
rlhf_env/
├── README.md              ← этот файл (quick start)
├── DOCS.md                ← полная документация (архитектура, API, расширение)
├── start_rlhf_env.sh      ← bash-лаунчер (venv, deps, env vars)
├── requirements.txt       ← aiohttp, numpy, onnxruntime, mcp
├── server.py              ← aiohttp app
├── mcp_server.py          ← MCP stdio-server
├── index.html             ← форма старта серии (POST /api/groups → redirect_url → /arena)
├── static/                ← rlhf.js / rlhf.css (клиент формы)
├── webapp_borrow/         ← verbatim копия webapp/: arena.html, arena.js, safe-area.js, arena-styles.css (1:1 арена)
├── components/
│   ├── policy_registry.py
│   ├── policy_factory.py
│   ├── deck_builder.py
│   ├── inference_params.py
│   ├── log_schema.py
│   ├── manifest.py
│   ├── battle_runner.py
│   └── session_manager.py
├── sessions/              ← файлы боёв (НЕ в git)
├── static/                ← JS/CSS
└── tests/
    ├── test_log_schema.py
    ├── test_deck_builder.py
    ├── test_policy_factory.py
    ├── test_battle_runner.py
    ├── test_manifest.py
    ├── test_session_manager.py
    └── smoke_e2e.py       ← автоматический E2E против реального сервера
```

## Использование

### Web-интерфейс

1. Откройте `http://127.0.0.1:8090/`.
2. Выберите p1_model / p2_model, deck_strategy, battles_planned.
3. Нажмите «Запустить группу».
4. Играйте против модели в браузере (1:1 как в проде).

### API (для интеграций)

| Метод | URL | Назначение |
|-------|-----|-----------|
| `GET`  | `/health` | Статус сервера + количество загруженных моделей |
| `GET`  | `/api/registry/models` | Список всех доступных моделей с sidecar |
| `GET`  | `/api/registry/sample-deck` | Пример случайной арены-деки |
| `POST` | `/api/groups` | Старт группы боёв (spec: p1, p2, count, deck, seed, …) |
| `GET`  | `/api/groups` | Список всех групп (running + completed с диска) |
| `GET`  | `/api/groups/{gid}` | Статус группы |
| `GET`  | `/api/groups/{gid}/manifest` | Полный manifest.json |
| `GET`  | `/api/groups/{gid}/battles` | Список battle_id |
| `GET`  | `/api/groups/{gid}/battles/{bid}` | Battle log |
| `POST` | `/api/groups/{gid}/stop` | Остановить running группу |
| `WS`   | `/ws/groups/{gid}/battles/{bid}` | WebSocket для human-vs-model |

### MCP

```bash
./rlhf_env/start_rlhf_env.sh mcp
```

Доступно 6 инструментов: `start_battle_group`, `stop_battle_group`,
`list_battle_groups`, `get_battle_group_status`, `get_battle_group_manifest`,
`download_battle_logs`. Подробнее — `DOCS.md#mcp`.

### CLI / скрипты

```bash
# Smoke E2E (запускает свой сервер на alt-порту и валидирует весь pipeline)
python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 2 --models random
python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 1 --models v4-max
```

## Тестирование

```bash
python3 -m pytest rlhf_env/tests/ -v          # 44 unit-теста (~3 сек)
python3 rlhf_env/tests/smoke_e2e.py --port 8096 --models random
```

## Что НЕ делаем

- Не трогаем прод-код (`web/`, `bot/`, `infrastructure/`, `core/`).
- Не используем БД — только файлы.
- Не валидируем Android Java (см. примечание в MEMORY).

## Подробная документация

`DOCS.md` — расширенная версия (≥ 500 строк): архитектура с диаграммами,
полный API, формат battle_log и manifest, рецепты sidecar-файлов,
добавление собственных моделей, troubleshooting.
