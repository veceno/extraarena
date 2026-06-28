# ExtraOrchestra — документация

## Сценарный DSL (v1)

Сценарий — JSON со схемой `extra_orchestra.scenario.v1`. См. пример
`scenarios/soldatik_demo.json` и спецификацию в плане
(`/Users/laveqox/.claude/plans/quizzical-soaring-rain.md`).

Ключевые поля:
- `seed` — детерминизм рандома.
- `viewer_side` — `"p1"` / `"p2"`, перспектива камеры.
- `classic_params` — `ClassicParams` (модификаторы арены: `spells_free`,
  `summon_ready_on_play`, `sudden_death_*`, `overdraw_to_discard`,
  `mana_per_turn`, `hero_health_multiplier`, `card_level_mode`, …).
- `init_scene` — первичная расстановка: `turn_number` (произвольный, напр. 15),
  `starting_side`, `p1`/`p2` с персоной (nickname, title, avatar_url,
  background_url, rarity, trophies, extra_pass, nickname_glow_disabled),
  `mana`/`max_mana`, `hero`, `hand[]`, `board[]`, `deck[]`.
  Карта: `{card_id, level, hp_override, attack_override, mechanics_override,
  is_ready, is_frozen}`.
- `turns[]` — графы ходов: `{id, side, duration_ms, end_with_end_turn, nodes[]}`.
  `nodes` — упорядоченный список действий с `delay_ms`:
  `play_card` / `attack` / `end_turn` / `mana_draw` / `wait`.

Модель выполнения: init-сцена → граф хода 1 (side A) → post-act сцена (снимок
состояния) → граф хода 2 (side B) → … Состояние между ходами наследуется
(один `ArenaEnvironment`, шагаем `engine.step`).

## Pipeline записи mp4

1. `scenario_engine.run_scenario` → `List[Frame]` (`{snapshot, sound_events,
   display_ms, action_kind, turn_id}`).
2. `recorder.py` (Playwright sync_api): headless Chromium грузит `player.html`,
   мост `orchestra-bridge.js` проигрывает кадры через `handleStateChanged({state,
   sound_events})` с `await sleep(display_ms)`. `record_video_dir` пишет webm.
3. `ffmpeg -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -r 30 -movflags
   +faststart` → mp4.
4. Phase 3: серверный аудио-микс из timeline + `sound_events` +
   `card_sfx_config.json` + зацикленная фоновая музыка арены
   (`arena_theme.wav`, громкость 0.3) → mux в mp4 (видео + SFX + музыка).

## Детерминизм (RISK A)

`core/effects.py` использует module-`random` (не `ArenaEnvironment._rng`) в
`cast_random_spell`/`cleave_X_Y`/`battlecry_damage_X_random`/`armor_X_Y`.
`scenario_engine` monkeypatch'ит `core.effects.random = random.Random(seed)`
на время прогона, чтобы реплеи были воспроизводимы.

## Borrowed arena (minimal-port constraint)

`webapp_borrow/` — FROZEN-снапшот `webapp/` @ `1282fcb8` (NewCards2606). НЕ
автосинхронизируется. Ручное обновление — `scripts/sync_borrowed.py` (копирует
4 файла, добавляет NOTE-заголовок, вшивает baked `window.__orchestraInit` hook
в arena.js — обход prebattle-гейта). Audio-remap НЕ нужен: сервер монтирует
`/DesignAssets/` целиком, поэтому оригинальные пути
`/DesignAssets/Sounds/arena/*` разрешаются напрямую (arena.js принимает оба
префикса). Никогда полный auto-refresh из `webapp/`.

## Что доставлено (Phase 1 + audio)

- `components/`: `cards_catalog.py`, `scenario_store.py`, `arena_engine.py`
  (OrchestraBattleEngine — verbatim-сериализаторы из `battle_engine.py` +
  порт sound_events с детерминированным event_id), `scenario_engine.py`
  (build GameState руками + run turn-graphs → List[Frame], RISK A
  monkeypatch `core.effects.random`, детерминированные instance_id через
  uuid5), `arena_io.py` (fake-JWT + audio_query), `audio_mix.py` (серверный
  аудио-микс из timeline+sound_events+card_sfx_config → ffmpeg amix),
  `recorder.py` (Playwright sync_api + ffmpeg → mp4).
- `server.py` (aiohttp, порт 8095): editor/player/borrowed-arena +
  `/DesignAssets/` + контракт-стабы (`/api/runtime/status`, `/api/settings`,
  `/api/cards`, `/api/battle/state` — отдаёт init-кадр по run_id) +
  `/api/orchestra/*` (cards, cosmetics, scenarios CRUD, validate,
  compute-frames, frames, record-задачи).
- `static/`: `editor.{html,css,js}` (**визуальный graph-редактор v2**:
  один SVG-канвас, pan/zoom через `viewBox` (drag-пустое=pan, wheel=zoom);
  ноды со свободным позиционированием (drag), порты слева/справа; рёбра
  рисует пользователь — **port-drag** от выходного порта одного узла к
  другому (geometric hit-test: drop в радиусе порта ИЛИ внутри коробки
  узла); палитра (сцена/hold, ход-маркер, play_card/attack/mana_draw/
  end_turn) — новый узел цепочкой вправо от выбранного; панель свойств
  справа (per-kind: display_ms / side+intro_ms / action-поля); computed-side
  chip на action-узлах (static-walk от start, flip на каждом end_turn) +
  warning если side узла ≠ позиции в пути (забыт end_turn); init-форма в
  широком overlay (persona + zone-chips, клик по карте каталога → в зону);
  авто-раскладка, удалить выбранное (init нельзя), debounced two-way JSON
  sync; **v1-сценарии при загрузке авто-мигрируются в v2** через
  `/api/orchestra/migrate-v1`; validate/preview/export), `orchestra-bridge.js`
  (path B: `__orchestraInit` + скрытие `#prebattle-screen` + stub `window.io`
  + `userId=<viewer_uid>` + итерация кадров через `handleStateChanged` с
  `sleep(display_ms)`).
- `components/scenario_graph_runner.py` — **v2 graph DSL + раннер**:
  `graph={start, nodes[], edges[]}`; один путь (max-1 исходящее И max-1
  входящее на узел, ровно один init=start, все узлы достижимы, без циклов);
  `validate_graph_structure` (без catalog) → dry-run `run_scenario_graph`;
  каждый action-узел несёт ОБЯЗАТЕЛЬНЫЙ `side ∈ {p1,p2}` — раннер валидирует
  `side_uids[side]==current_turn_owner_id` ПЕРЕД apply_action (забыт end_turn
  → явный ScenarioError, не silent wrong-side); fail-fast на `ok:False` +
  `game_over`-кадр; monkeypatch `core.effects.random` на весь обход; Frame
  dict byte-identical с v1 → recorder/bridge не трогаются. `migrate_v1_to_v2`
  (init→scene/init, wait→scene/hold, пустой ход→hold, end_with_end_turn без
  явного → добавляет end_turn, детерминированная auto-layout). Dispatch по
  `schema` (v1→v1 раннер, v2→graph).
- `scenarios/soldatik-demo.json` (v1, kept for back-compat + migration test)
  + `tests/` (engine/serializers/sound_events/audio_mix/smoke +
  `test_scenario_graph.py` — 50 test: structure/runner/migration/dispatch/
  store back-compat; Playwright no-visual editor e2e `test_editor_graph_e2e.py`
  под `ORCH_E2E=1`; mp4 e2e под `ORCH_E2E=1` — h264 мобильный портрет
  828×1792 (414×896 CSS × device_scale_factor 2) 30fps + aac (SFX + музыка);
  `test_mcp_server.py` — 47 test MCP-сервера: pure-хелперы, live tool calls,
  export_mp4 через stub-client (без Playwright), stdio JSON-RPC end-to-end
  (unknown-method → top-level error, notifications без ответа, parse-error id=null)).

## MCP-сервер (для агентов) — `extra_orchestra/mcp_server.py`

Hand-rolled JSON-RPC 2.0 over stdio (как `rlhf_env/mcp_server.py`), тонкий
async HTTP-клиент к серверу оркестра (порт 8095). 16 инструментов в 4 группах:

- **Загрузка:** `list_scenarios`, `get_scenario(name, as_v2=true)` (v1
  авто-мигрируется в v2), `migrate_v1_to_v2(scenario)`.
- **Создание (графами):** `create_blank_scenario(meta)` — пустой v2-граф с
  init-узлом (шаблон, форматы совпадают с editor.js `blankV2`); `build_graph(spec)`
  — собирает v2 из декларативных `nodes`/`edges` (kind ∈ scene/turn/action;
  edges необязательны → авто-цепочка `s0→n1→…`; `base_scenario_name` унаследует
  init+classic_params+meta из сохранённого сценария), валидирует через HTTP,
  возвращает `{scenario, validation}`; `save_scenario`, `delete_scenario`,
  `validate_scenario` — structure+dry-run.
- **Предпросмотр:** `preview_frames(scenario)` — прогон → покадровые
  «итоговые сцены»: на каждом шаге графа structured-снапшот
  (p1/p2 board с card_id/name/attack/hp/mechanics/is_ready, hero hp, mana,
  hand_count+hand_card_ids, turn, current_player_id, action_kind, display_ms) +
  `run_id`/`frame_count`/`total_ms` + `preview_arena_url` (с `_auth=`).
  **НЕ изображения** — данные для рассуждения; vision/omni-хост рендерит сам.
  p1/p2 маппятся по `side_uids` + `snapshot.player.user_id` (player/opponent
  viewer-relative → swap если player≠p1). `get_frames(run_id, summary=true)`
  — пере-выбрать кадры ранее созданного прогона (summary=false — raw run).
- **Экспорт:** `export_mp4(scenario, wait=true, timeout=180, inline=false)` —
  записывает mp4 (мобильный портрет 828×1792 @ device_scale_factor 2, видео +
  SFX + фоновая музыка арены, crf 10 / preset slow) через Playwright+ffmpeg
  сервера; `export_gif(scenario, wait=true, timeout=180, inline=false)` — то же
  в GIF (двухпроходный palettegen+paletteuse; без звука — формат не
  поддерживает; ширина 540 / fps 15 по умолч., см. config.json
  `gif_width`/`gif_fps`); `wait=true` блокирует до готовности. **`inline=true`
  — отдать БАЙТЫ файла инлайн как MCP content-item** (gif → `ImageContent`,
  omni/vision-клиенты рендерят инлайн; mp4 → `ResourceContent` с base64-`blob`,
  агент декодирует/сохраняет) — иначе только `download_url` (localhost-ссылка,
  которую MCP-клиент не может fetch'ить). `get_record_status(job_id)` — только
  статус/метаданные; **`get_record_file(job_id)` — достать готовый файл
  инлайн** (байты mp4/gif как content-item + `{size_bytes, mime_type,
  file_name, download_url}`; требует `status=done`). `list_cards(filter)`,
  `list_cosmetics()`. dispatch выносит `_content` из text-метаданных в
  отдельный content-item (base64 не дублируется в text → не раздувается).

Запуск: `./extra_orchestra/start_orchestra.sh mcp` (auto-start HTTP-сервера по
умолчанию, `--no-auto-start` / `ORCH_AUTO_START=0` чтобы выключить) ИЛИ
`python3 -m extra_orchestra.mcp_server --base-url http://127.0.0.1:8095`.
Session/client/server создаются ВНУТРИ event-loop'а
(`asyncio.run(_amain(base_url))`) — aiohttp 3.9 требует этого; stdin читается
через `asyncio.StreamReader` (cancellable, не блокирует executor-поток на
shutdown). Content items: `{"type":"text","text":json.dumps(result)}`
(универсально-совместимо; MCP 2024-11-05 spec — rlhf-паттерн `type:"json"`
non-standard). JSON-RPC 2.0 strictly: unknown tool →
`{content,isError:true}`; unknown method → top-level `{error:{code:-32601}}`
(НЕ вложенный в `result`); missing/invalid tool params → `-32602`;
non-object/batch-не-dict элемент → `-32600`; notifications (нет `id`) → без ответа;
batch (массив запросов) → массив ответов; parse-error → `id:null`; `ping` → `{}`.
`isError:true` поднимается когда `_tool` вернул execution-error (`{error:...}` без
`ok`) — валидационный результат `{ok:false,error:...}` остаётся `isError:false`
(агент читает `ok=false`). `--auto-start` использует `BooleanOptionalAction`
(`--no-auto-start`); auto-start передаёт `--host`/`--port` из `--base-url` серверу,
fail-fast (exit 2) если бэкенд не поднялся (не крутит stdio-loop против мёртвого
порта); логирует ошибку если порт занят чужим процессом.
`start_orchestra.sh` выбирает venv-python только если в нём есть aiohttp, иначе
системный python3 (venv мог быть не populated).

**Adversarial verify (2026-06-28, 47 агентов, 42 confirmed findings → фиксы):**
majors: build_graph не принимал канонические v2 action-узлы
(`{kind:action, side, action:{type,...}}` из editor.js/сохранённых сценариев) →
silent `bad action.type 'None'`; теперь читает и плоскую, и nested-форму.
Unknown JSON-RPC method возвращал error вложенный в `result` → top-level
`error`. Notifications (без `id`) получали ответ с `id:0` → теперь без ответа.
minors: `viewer_side` в summarize_frames был числовым uid → derives side string
из side_uids; export_mp4 wait=true висел до timeout на lost/unknown job → break
на error/no-status; build_graph base_scenario_name травил name/classic_params в
None → `or`-fallbacks + meta наследуется всегда (даже с init_scene);
auto-gen node id мог коллизировать с явным → dedup + raise; `get_frames` tool
добавлен (re-fetch прогонов по run_id); export_mp4 wait=false → `pending` (не
`started`); `_spawn_server` 40 throwaway loops → `time.sleep`; logf fd leak →
close после Popen; `--no-auto-start` CLI; `get_event_loop` → `get_running_loop`.
nits: export_mp4 description без `mp4_path?`; action `side` без silent default
(None → validate ловит на authoring); edge без from/to → ValueError; hidden
opponent hand → `hand_hidden` + `hand_card_ids` без None; create_blank
`display_ms`/`match_id` выровнены с editor.js blankV2 (2000 / "new-scenario").
Тесты: 47 passed (вкл. export_mp4 stub-client, p2-viewer mapping, base
inheritance, stdio protocol edges).

Phase 3 — опц. Socket.IO live-preview, slow-mo/frame-step, демо на каждую
новую карту 47–52.

## Известные правки ввода в эксплуатацию

- **Preview «Сессия истекла»:** `compute-frames` отдаёт `auth` (fake-JWT),
  editor.js дописывает `&_auth=<auth>` в URL предпросмотра (arena.js boot
  иначе показывает modal — authToken пуст).
- **mp4 = 5с preBattleScreen:** bridge обходил `startPrebattleSequence()`,
  который скрывает `#prebattle-screen`; теперь bridge зовёт
  `hidePrebattleScreen()` + CSS-фолбэк `display:none`.
- **Мобильное соотношение + звук (preview & export):** рекордер пишет в
  мобильном портретном viewport (414×896 CSS, `device_scale_factor=2` → mp4
  828×1792); ширина ≤420px → срабатывает `@media (max-width:420px)` из
  arena-styles.css → мобильный лейаут арены (а не «экран компьютера»).
  Предпросмотр: editor.js открывает popup 414×896, bridge зовёт
  `window.resizeTo(414,896)` если открыт как popup; URL предпросмотра несёт
  `music=1&sfx=1` (раньше `music=0` → музыка не играла). Экспорт: серверный
  аудио-микс теперь включает зацикленную `arena_theme.wav` (volume 0.3)
  поверх SFX-timeline (`-stream_loop -1` + `-shortest`); `with_audio=true`
  по умолчанию (раньше `false` → SFX отсутствовали).
- **Предпросмотр в мобильной рамке (iframe):** `/preview` отдаёт
  `static/preview.html` — «телефон»-обёртку с `<iframe src="/player?...">`
  фиксированного мобильного портретного размера (414×896, `aspect-ratio`,
  `width: min(414px,96vw,44.36vh)`). У iframe свой viewport →
  `@media (max-width:420px)` срабатывает даже в широком десктоп-окне →
  область предпросмотра искусственно сужена до мобильного соотношения (раньше
  /player в широком окне = десктоп-версия арены). `preview.html` пробрасывает
  query в `/player` и сам добавляет `music=1/sfx=1/ea_platform`, если их нет.
  Editor «Предпросмотр» и MCP `preview_arena_url` открывают `/preview`.
- **Качество mp4 + экспорт GIF:** `config.json` `recording.crf` (по умолч. 10 ≈
  визуально lossless) и `recording.preset` (`slow`) пробрасываются в ffmpeg
  `mix_audio_into_mp4` (раньше хардкод `crf 18 / medium` → низкое качество).
  Новый `components/gif_export.py` `webm_to_gif` — двухпроходный
  `palettegen`+`paletteuse` (Bayer-dither, `stats_mode=diff`), `gif_fps`/`gif_width`
  из config (по умолч. 15/540; `width=0`=native). `recorder.py` рефактор:
  общий `_capture_webm` + `record_run_to_gif` (GIF без звука — формат не
  поддерживает; webm переиспользуется, дубль-запись арены не делается).
  Сервер `/api/orchestra/record?format=gif|mp4`, редактор «Экспорт GIF»,
  MCP-инструмент `export_gif`.