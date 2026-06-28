# ExtraOrchestra

Утилита для разыгрывания игровых сцен по сценариям, предпросмотра на реальной
арене и экспорта в mp4. Назначение — демонстрационные ролики карт для новичков.

Наследует внешний вид, механики, модификаторы и звуки арены из основной игры
по образцу `rlhf_env/`: frozen-снапшот `webapp/arena.*` в `extra_orchestra/webapp_borrow/`
( snapshot `1282fcb8` / `worktree-NewCards2606` ), aiohttp-сервер реплицирует
HTTP/Socket.IO контракт, сериализаторы скопированы verbatim из `battle_engine.py`.

## Запуск

```bash
./extra_orchestra/start_orchestra.sh
# → http://127.0.0.1:8095/            (редактор сценариев)
# → http://127.0.0.1:8095/player       (предпросмотр арены)
```

Зависимости: `pip install -r extra_orchestra/requirements.txt`; системные
`playwright install chromium` и `ffmpeg`.

## Что внутри

- `components/scenario_engine.py` — строит `GameState` руками из init-сцены и
  прогоняет графы ходов через `core.engine.ArenaEnvironment` → список кадров.
- `components/arena_engine.py` — `OrchestraBattleEngine`, шим над ядром с
  verbatim-сериализаторами и портом `sound_events`.
- `components/recorder.py` — Playwright + ffmpeg → mp4.
- `static/editor.*` — визуальный node-редактор (Phase 2; Phase 1 — form-форма).
- `static/orchestra-bridge.js` — мост: тянет кадры и зовёт `handleStateChanged`.
- `scenarios/soldatik_demo.json` — демо «Солдатик vs 3 механики».

Полная спецификация DSL, pipeline записи и заметки о детерминизме — в `DOCS.md`.