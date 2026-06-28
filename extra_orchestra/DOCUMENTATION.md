# ExtraOrchestra — исчерпывающая документация (общий справочник)

ExtraOrchestra — отдельная утилита для **детерминированного разыгрывания
игровых сцен по сценариям** и их воспроизведения на реальной арене
ExtraArenaRaS с последующим предпросмотром и экспортом в видео (mp4 с звуком
арены / GIF). Цель — демонстрационные и учебные ролики карт/механик для
новичков (например, демо Солдатика: ход 15, у оппонента 3 юнита с
механиками → игрок выставляет Солдатика → `aoe_silence` снимает все три).

Утилита полностью наследует внешний вид арены, механики карт, модификаторы,
визуальные/звуковые эффекты из основной игры — по образцу `rlhf_env/`
(frozen-снапшот `webapp/arena.*` + aiohttp-сервер, реплицирующий
Socket.IO/HTTP-контракт; сериализаторы скопированы verbatim; `arena.js`
работает без правок).

> См. также: `SEQUENCE_REPLAY.md` — углублённо про воспроизведение действий в
> заданной последовательности (граф → раннер → кадры → арена) и интеграцию с
> онбординг-туториалом. `DOCS.md` — короткая справка + история аудитов.

---

## 1. Расположение и запуск

Пакет живёт в `extra_orchestra/` корня репозитория (в dev-сетапе — в worktree
`.claude/worktrees/ExtraOrchestra/`). Запуск:

```bash
# HTTP-сервер (редактор + арена + API), порт 8095
./extra_orchestra/start_orchestra.sh
# → http://127.0.0.1:8095/         визуальный graph-редактор
# → http://127.0.0.1:8095/player   арена (предпросмотр прогона)
# → http://127.0.0.1:8095/preview  мобильная iframe-рамка предпросмотра

# MCP stdio-сервер (для агентов; auto-start HTTP по умолчанию)
./extra_orchestra/start_orchestra.sh mcp
ORCH_AUTO_START=0 ./extra_orchestra/start_orchestra.sh mcp   # без auto-start
python3 -m extra_orchestra.mcp_server --base-url http://127.0.0.1:8095
```

`start_orchestra.sh` создаёт `.venv`, выбирает python с `aiohttp` (venv ИЛИ
системный), для `mcp` — `exec python -m extra_orchestra.mcp_server`.
Зависимости: `aiohttp`, `playwright` (+ `playwright install chromium`),
системный `ffmpeg`/`ffprobe`. См. `requirements.txt`.

> **MCP stdio-чистота:** для регистрации MCP-клиента используйте прямой
> `python3 -m extra_orchestra.mcp_server` (через `bash -c "cd <root> && exec …"`),
> а НЕ `start_orchestra.sh mcp` — скрипт при пересоздании venv пишет diagnostic
> в **stdout**, что ломает JSON-RPC-стрим. В steady-state скрипт тоже чист, но
> прямой `python -m` безопаснее. `cwd` обязан быть корнем, содержащим
> `extra_orchestra/` (и `core/`, `infrastructure/` — это полный checkout репо).

---

## 2. Архитектура (верхний уровень)

```
extra_orchestra/
├── server.py                 # aiohttp, порт 8095: editor/player/preview + borrowed-arena + /api/orchestra/* + контракт-стабы
├── config.json               # порт, пути, дефолты записи (fps/wh/dsf/crf/preset/gif_*)
├── start_orchestra.sh        # venv + python-selection + exec server|mcp
├── requirements.txt          # aiohttp, playwright (ffmpeg — системный)
├── DOCS.md / DOCUMENTATION.md / SEQUENCE_REPLAY.md
├── webapp_borrow/            # FROZEN-снапшот webapp/ @ 1282fcb8 (NewCards2606, 50 карт)
│   ├── arena.html            # + baked 1-line: window.__orchestraInit hook
│   ├── arena.js              # + baked __orchestraInit (обход prebattle-гейта)
│   ├── arena-styles.css      # verbatim
│   └── safe-area.js          # verbatim
├── static/                   # offline, без npm/CDN
│   ├── editor.{html,css,js}  # визуальный graph-редактор v2 (SVG-канвас)
│   ├── player.html           # обёртка арены + bridge
│   ├── preview.html          # мобильная iframe-рамка (414×896)
│   ├── orchestra-bridge.js   # path B: __orchestraInit + stub io + итерация кадров
│   └── orchestra.css
├── components/
│   ├── cards_catalog.py      # cards.json + card_scaling + asset-URL + deterministic_instance_id
│   ├── scenario_store.py     # save/load сценариев в scenarios/*.json
│   ├── arena_engine.py       # OrchestraBattleEngine: shim над core.engine + verbatim-сериализаторы + порт sound_events
│   ├── scenario_engine.py    # v1: build GameState руками + run turn-graphs → List[Frame] (RISK A monkeypatch)
│   ├── scenario_graph_runner.py  # v2 graph DSL + раннер + validate_graph_structure + migrate_v1_to_v2 + dispatch
│   ├── arena_io.py           # make_fake_jwt + audio_query
│   ├── audio_mix.py          # серверный аудио-микс timeline+sound_events+card_sfx_config+arena_theme → ffmpeg amix → mux в mp4
│   ├── recorder.py           # Playwright sync_api (_capture_webm) + record_run_to_mp4 / record_run_to_gif
│   └── gif_export.py         # webm_to_gif (двухпроходный palettegen+paletteuse)
├── scenarios/                # *.json сценарии (soldatik-demo.json, …)
├── recordings/               # выходные mp4/gif (gitignore)
└── tests/                    # test_scenario_engine / _graph / _serializers / _sound_events / _audio_mix / _gif_export / _smoke_e2e / _mcp_server / _editor_graph_e2e
```

### Поток данных (end-to-end)

```
сценарий (JSON, v1 или v2 graph)
  └─ run_scenario_dispatch ─────────────────────┐
       v1: scenario_engine.run_scenario          │  build_initial_state → ArenaEnvironment
       v2: scenario_graph_runner.run_scenario_graph  │  (apply_start_effects=False, monkeypatch core.effects.random)
       → List[Frame] {snapshot, sound_events, display_ms, action_kind, turn_id, node_id, error?}
  └─ server хранит run в _runs[run_id]
       ├─ /api/orchestra/frames/<run_id>          → frames (для bridge/preview/MCP)
       ├─ /player?id=<run_id>&autoplay=1&_auth=…  → arena.html + orchestra-bridge.js проигрывает кадры
       └─ /api/orchestra/record?format=mp4|gif    → фоновый thread → recorder → ffmpeg → recordings/<name>-<jobid>.<ext>
```

Каждый **Frame** — это `{snapshot, sound_events, display_ms, action_kind,
turn_id, node_id, error?}` где `snapshot` — viewer-perspective
`get_full_state(viewer_uid)` (тот же JSON, что арена получает по Socket.IO в
prod-игре). `display_ms` — сколько миллисекунд показывать кадр (cadence
предпросмотра/записи). `sound_events` — детерминированные SFX-события для
серверного аудио-микса.

---

## 3. Сценарный DSL

Поддерживаются две схемы; dispatch — по полю `schema`:

- `extra_orchestra.scenario.v1` — линейные `turns[]` с `nodes[]` (hand-edited,
  kept for back-compat). `soldatik-demo.json` — v1.
- `extra_orchestra.scenario.v2` — **граф выполнения** `graph={start, nodes[],
  edges[]}` (то, что рисует визуальный редактор). v1 авто-мигрируется в v2
  через `migrate_v1_to_v2` / `POST /api/orchestra/migrate-v1`.

### 3.1. Общие top-level поля

| Поле | Тип | Назначение |
|---|---|---|
| `schema` | str | `…v1` или `…v2` (dispatch) |
| `name` | str | Имя сценария → имя файла записи (санитизируется, см. §8) |
| `seed` | int | Детерминизм `core.effects.random` + `uuid5` instance_id |
| `viewer_side` | `"p1"`/`"p2"` | Чья перспектива камеры (какой uid = viewer) |
| `match_id` | str | Идентификатор боя (в prod-контракт; по умолч. `"orchestra"`) |
| `classic_params` | obj | Модификаторы арены (см. §3.3) |
| `game_mode` | str | По умолч. `"classic"` → `resolve_mode_config` |
| `init_scene` (v1) / init-узел (v2) | obj | Первичная расстановка (см. §3.2) |
| `turns[]` (v1) | list | Линейные ходы с nodes |
| `graph` (v2) | obj | `{start, nodes[], edges[]}` |
| `layout`, `editor` (v2) | obj | Только для редактора; раннер игнорирует |

### 3.2. init-сцена

`init_scene` (v1) или `scene` type=`init` (v2, это `graph.start`):

```jsonc
{
  "turn_number": 15,          // произвольный (ставится напрямую на GameState.turn_number)
  "starting_side": "p1",      // кто current_turn_owner_id
  "display_ms": 2200,         // кадр init-сцены показывается столько (v2; v1 → init_intro_ms)
  "p1": { "user_id": 1001, "nickname": "Демо", "title": "Новичок",
          "avatar_url": "/DesignAssets/PlayerCosmetics/Avatars/1.png",
          "background_url": "/DesignAssets/PlayerCosmetics/Background/7.png",
          "rarity": "common", "trophies": 0,
          "mana": 10, "max_mana": 10,
          "hero": { "card_id": 1, "level": 1 },
          "hand":   [ { "card_id": 47, "level": 1 } ],
          "board":  [],
          "deck":   [ { "card_id": 5, "level": 1 }, { "card_id": 6, "level": 1 } ] },
  "p2": { "user_id": 2002, "is_bot": true, "nickname": "Режиссёр", "title": "Босс",
          "avatar_url": "/DesignAssets/PlayerCosmetics/Avatars/2.png",
          "background_url": "/DesignAssets/PlayerCosmetics/Background/3.png",
          "rarity": "epic", "mana": 7, "max_mana": 7,
          "hero": { "card_id": 1, "level": 1 },
          "hand":  [],
          "board": [ { "card_id": 24, "level": 1, "mechanics_override": ["shield","shield_refresh"], "is_ready": true },
                     { "card_id": 30, "level": 1, "mechanics_override": ["taunt"], "is_ready": true },
                     { "card_id": 34, "level": 1, "mechanics_override": ["deathrattle_aoe_damage_2"], "is_ready": true } ],
          "deck": [] }
}
```

**Карта** в зоне: `{card_id, level, hp_override?, attack_override?,
mechanics_override?, is_ready?, is_frozen?}`. Стат берётся через
`core.card_scaling.scale_card_by_level`, override'ы применяются после.
`instance_id` генерируется **детерминированно** через
`uuid5(seed:side:zone:index:card_id:level)` (не `uuid4`!) — это фиксирует
снапшоты для byte-идентичного replay.

`build_initial_state` строит `GameState` **напрямую** (НЕ
`create_classic_game_state` — тот хардкодит `turn=1` и авто-раздаёт руку) и
`ArenaEnvironment(..., apply_start_effects=False, rng=Random(seed))`.
`apply_start_effects=False` — init-сцена остаётся ровно как автор описал (без
авто-раздачи/mana/wake); start-of-turn эффекты следующих ходов честно
применятся через `end_turn`.

### 3.3. classic_params (модификаторы арены)

Только поля, входящие в `ClassicParams` (лишние игнорируются): `spells_free`,
`summon_ready_on_play`, `sudden_death_enabled`, `sudden_death_turn`,
`sudden_death_hp_pct`, `overdraw_to_discard`, `mana_per_turn`,
`hero_health_multiplier`, `card_level_mode`, `turn_duration_seconds`, и др.
Базовая конфигурация берётся через `resolve_mode_config(game_mode)`, затем
classic-поля переопределяются из сценария (`dataclasses.replace`).

### 3.4. v1 — turns[] + nodes[]

`turns[i] = {id, side, duration_ms, end_with_end_turn, nodes[]}`. `nodes[]` —
упорядоченный список действий с `delay_ms` (пауза ПЕРЕД действием → cadence):

| `type` | Поля | Соответствие core action |
|---|---|---|
| `play_card` | `hand_index`, `target_id?`/`target_index?`/`target_is_hero?`, `position?`, `delay_ms?` | `PlayCardAction` |
| `attack` | `attacker_id?`/`attacker_index`, `target_id?`/`target_index?`/`target_is_hero`, `delay_ms?` | `AttackAction` |
| `end_turn` | `delay_ms?` | `EndTurnAction` |
| `mana_draw` | `delay_ms?` | `ManaDrawAction` |
| `wait` | `delay_ms` | удержание текущего снимка (не шаг ядра) |

`end_with_end_turn=true` без явного `end_turn`-узла → неявный `EndTurnAction`
в конце хода. **Side-guard:** если `current_turn_owner_id != side_uids[side]` —
`ScenarioError` («Author must end_turn explicitly to pass the turn»), не
silent wrong-side action.

### 3.5. v2 — graph

`graph = {start, nodes[], edges[]}`. Node kinds:

- `scene` type=`init` — строит GameState (ровно один, `=graph.start`, без
  входящих рёбер). `display_ms` — длительность init-кадра.
- `scene` type=`hold` — удерживает текущий снимок на `display_ms` (замена v1
  `wait`/post-act). `display_ms` default 600.
- `action` — шаг ядра. **ОБЯЗАТЕЛЬНО** `side ∈ {p1,p2}` + `action={type, delay_ms, …}`
  (те же поля, что в v1 node, но вложены в `action`). `type ∈ {play_card,
  attack, mana_draw, end_turn}`.
- `turn` — опц. sanity-маркер (`turn.side`, `turn.intro_ms` default 0 → нет
  кадра). `intro_ms>0` → отдельный `turn_intro`-кадр. Задаёт `turn_id` для
  последующих action-кадров.

**Граф = один путь:** max-1 исходящее И max-1 входящее ребро на узел, ровно
один init, все узлы достижимы из start, без циклов, init не имеет входящих.
`validate_graph_structure` проверяет всё это без catalog (быстрая
структурная валидация). `layout`/`editor` — top-level, раннер игнорирует.

```jsonc
{ "schema": "extra_orchestra.scenario.v2", "name": "demo", "seed": 42,
  "viewer_side": "p1", "match_id": "demo",
  "classic_params": { "sudden_death_enabled": false, "mana_per_turn": 1, "turn_duration_seconds": 25 },
  "graph": {
    "start": "s0",
    "nodes": [
      { "id": "s0", "kind": "scene", "scene": { "type": "init", "turn_number": 15, "starting_side": "p1", "display_ms": 2000, "p1": { /* … */ }, "p2": { /* … */ } } },
      { "id": "n1", "kind": "turn",  "turn":  { "side": "p1", "intro_ms": 400 } },
      { "id": "n2", "kind": "action", "side": "p1", "action": { "type": "play_card", "hand_index": 0, "delay_ms": 800 } },
      { "id": "n3", "kind": "action", "side": "p1", "action": { "type": "end_turn", "delay_ms": 500 } },
      { "id": "n4", "kind": "scene", "scene": { "type": "hold", "display_ms": 1500 } }
    ],
    "edges": [ { "id": "e1", "from": "s0", "to": "n1" }, { "id": "e2", "from": "n1", "to": "n2" },
               { "id": "e3", "from": "n2", "to": "n3" }, { "id": "e4", "from": "n3", "to": "n4" } ]
  },
  "layout": { "s0": {"x":60,"y":200}, "n1": {"x":290,"y":200}, /* … */ },
  "editor": { "zoom": 1 } }
```

`turn_id` кадра = id ближайшего предшествующего `turn`-узла, `"init"` для
init-кадра, или `"__no_turn__"`. Frame dict **byte-identical** с v1 →
recorder/bridge не различают схемы.

### 3.6. v1 → v2 миграция (`migrate_v1_to_v2`)

Детерминированная auto-layout: `init_scene` → `scene/init` (id `s0`);
`turns[]` разворачиваются в цепочку узлов (`wait`→`scene/hold`, action→
`action` с `side` хода); `end_with_end_turn=true` без явного `end_turn` →
добавляется явный `end_turn`-узел (**включая пустые ходы** — фикс: пустой ход
с `end_with_end_turn=true` больше не теряет переход хода). Пустой ход без
`end_with_end_turn` → `scene/hold` на `duration_ms` (turn-маркер не эмитится).
`NODE_GAP=220`, координаты `x=40,90,…`.

---

## 4. Компоненты

### 4.1. `cards_catalog.py`

`CardsCatalog` грузит `cards.json` (50 карт NewCards2606), строит
`CardInstance` через `scale_card_by_level`. `deterministic_instance_id(seed,
side, zone, index, card_id, level)` = `uuid5` — детерминизм instance_id.
`build_instance(it, instance_id)` применяет `hp_override`/`attack_override`/
`mechanics_override`/`is_ready`/`is_frozen`. `list_cosmetics()` — аватары/фоны.

### 4.2. `arena_engine.py` — OrchestraBattleEngine

Shim над `core.engine.ArenaEnvironment` (как `rlhf_env/components/arena_engine.py`):
- **Verbatim из `battle_engine.py`:** `get_full_state` (1001),
  `_serialize_player_state` (1155), `_serialize_card` (1203),
  `_serialize_action_history` (1242), `_consume_card_feedback_events` (576),
  `_is_play_sound_mechanic` (511), `_make_sound_event` (545),
  `_find_card_feedback_event` (585). Включая `mana_draw_count_this_turn`
  (отсутствует в rlhf-шиме).
- **Порт:** `_sound_events_for_action` (475–642) — deploy/mechanic/attack;
  `event_id` детерминированный `"orchestra:<turn>:<kind>:<card>:<event>"`
  (arena.js дедуплит SFX по event_id).
- **Новое:** `apply_action(uid, action) → {ok, error?, action_kind, snapshot,
  sound_events, game_over?, winner_id?}` — `step` → снимок → feedback →
  sound_events одним вызовом; on failure — rollback `state` из deepcopy.
- `aoe_silence` НЕ входит в `_is_play_sound_mechanic` → только deploy-событие
  (как в prod).

### 4.3. `scenario_engine.py` (v1)

`build_initial_state(scenario, catalog) → (env, engine, viewer_uid,
side_uids)`. `_node_to_action` маппит node → `BaseAction` (ссылки по индексам:
`hand_index`/`attacker_index`/`target_index`; `target_is_hero`; продвинуто —
`*_id` instance_id). `run_scenario` → `List[Frame]` с side-guard и fail-fast
на `ok:False`. **RISK A:** `core.effects.random = Random(seed)` на время
прогона (module-random в `cast_random_spell`/`cleave`/`armor_X_Y`).

### 4.4. `scenario_graph_runner.py` (v2)

`validate_graph_structure` (без catalog, быстрая) → `run_scenario_graph`
(обход по пути от start, тот же Frame shape). Каждый `action`-узел: side-guard
ПЕРЕД `apply_action` (забыт `end_turn` → явный `ScenarioError`). `game_over` →
досрочный стоп traversal. Dispatch `run_scenario_dispatch` /
`validate_scenario_dispatch` по `schema`. `migrate_v1_to_v2`.

### 4.5. `scenario_store.py`

save/load в `scenarios/*.json`. **Path-traversal guard:** `load(name)` →
`Path(name).name` (basename-confined, `../secret` не читается вне `scenarios/`).

### 4.6. `recorder.py`

`_capture_webm(run, cfg, viewer_uid, *, base_url, speed, extra_wait_ms) →
(webm_path, tmpdir)` — общая стадия захвата (переиспользуется mp4 и gif).
Playwright sync_api: headless Chromium, мобильный портретный viewport
(414×896 CSS × `device_scale_factor` → 828×1792 webm), `record_video_dir`,
`add_init_script` (stub `window.io`, `ExtraArenaApp=true`), `page.route`
нейтрализует external CDN (telegram-web-app.js, socket.io.min.js) для
offline-детерминизма, ждёт `window.__orchestraDone`, `context.close()` → webm.
- `record_run_to_mp4` → `mix_audio_into_mp4` (видео + серверный аудио-микс).
- `record_run_to_gif` → `webm_to_gif` (GIF без звука).
- finally: `webm_path.unlink` + `shutil.rmtree(tmpdir)` (нет утечки tmpdir).

### 4.7. `audio_mix.py`

`build_audio_timeline(frames)` — из `sound_events`+`card_sfx_config.json`
строит `[(offset_ms, wav_path, volume)]` (deploy сразу, mechanic ~+200ms,
attack сразу). `mix_audio_into_mp4(webm, timeline, out, *, fps, crf, preset,
music_path?, music_volume)` → ffmpeg: видео `libx264 -preset {preset} -crf
{crf} -pix_fmt yuv420p -r {fps} -movflags +faststart`; аудио — `adelay`+`volume`
на каждый клип + `amix` (чанкование ≤32 входов, RISK G) + зацикленная
`arena_theme.wav` (`-stream_loop -1` + `-shortest`, volume 0.3) → mux `aac`.
`crf` (default 10 ≈ lossless) и `preset` (default `slow`) — из config.

### 4.8. `gif_export.py`

`webm_to_gif(webm, out, *, fps=15, width=0)` — двухпроходный:
pass1 `palettegen=stats_mode=diff`, pass2 `paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`;
`scale={width}:-2:flags=lanczos` (чётная высота; `width=0`=native); fps
clamp ≤30; palette-файл unlinks в `finally`. GIF **без звука** (формат не
поддерживает).

### 4.9. `arena_io.py`

`make_fake_jwt(uid, seed)` — JWT-подобный токен для `?_auth=` (arena.js boot
иначе показывает «Сессия истекла»). `audio_query` — хелпер.

---

## 5. Сервер (`server.py`, aiohttp, порт 8095)

### 5.1. Страницы и статики

| Маршрут | Назначение |
|---|---|
| `GET /` , `/editor` | визуальный graph-редактор (`editor.html`) |
| `GET /player` | арена + bridge (`player.html`), query `?id=<run_id>&autoplay=1&_auth=…&ea_platform=android_app&music=1&sfx=1&speed=…` |
| `GET /preview` | мобильная iframe-рамка (`preview.html`), сама добавляет `music=1/sfx=1/ea_platform` |
| `GET /arena` , `/arena.js` , `/safe-area.js` , `/arena-styles.css` | borrowed-снапшот (FROZEN @ 1282fcb8) |
| `/static/` , `/DesignAssets/` , `/assets/audio/` | статики (DesignAssets — целиком, поэтому оригинальные пути `/DesignAssets/Sounds/arena/*` разрешаются напрямую; audio-remap не нужен) |

### 5.2. Контракт-стабы (критичны для arena.js boot)

`GET /api/runtime/status` → `{maintenance_mode:{enabled:false}, is_admin:false}`
(404 здесь → «Соединение разорвано»). `GET /api/settings` →
`{sound_sfx:true, sound_music:false, …}`. `GET /api/cards`. `GET /health` →
`{status:"ok"}`. `GET /api/battle/state?match_id=<run_id>` — отдаёт init-кадр.

### 5.3. Orchestra API (`/api/orchestra/*`)

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/cards` | каталог 50 карт (stаты + mechanics + image) |
| GET | `/cosmetics` | аватары/фоны |
| GET | `/scenarios` | список сохранённых |
| GET | `/scenarios/{name}` | загрузить (v1/v2 as-is) |
| POST | `/scenarios` | сохранить |
| DELETE | `/scenarios/{name}` | удалить |
| POST | `/validate` | structure + dry-run → `{ok, error, frame_count, total_ms}` |
| POST | `/compute-frames` | прогон → `{run_id, frame_count, total_ms, viewer_uid, auth, error?}` (auth = fake-JWT) |
| POST | `/migrate-v1` | v1 → v2 |
| GET | `/frames/{run_id}` | full run-документ `{frames, viewer_uid, side_uids, …}` |
| GET | `/frames/{run_id}/manifest` | lightweight manifest |
| POST | `/record?format=mp4\|gif` | стартовать запись (body=сценарий) → `{job_id, run_id, status:"pending", format, mp4_name, file_name}` |
| GET | `/record/{job_id}` | статус → `{status, format, error?, mp4_name, file_name}` |
| GET | `/record/{job_id}/download` | скачать готовый файл (404 пока не `done`) |

`orch_record` санитизирует `scenario.name` через `_safe_name_slug` (см. §8) и
guard'ит `out_path.resolve().relative_to(RECORDINGS_DIR)`. Запись идёт в фоновом
`threading.Thread` (`_run_record_job` → `record_run_to_gif`/`record_run_to_mp4`),
по завершении `job["mp4_path"] = out` (каноническое поле пути к файлу — mp4 ИЛИ
gif — используется download'ом).

### 5.4. Запись (`config.json` `recording`)

```jsonc
{ "fps": 30, "width": 414, "height": 896, "device_scale_factor": 2,
  "with_audio": true, "headless": true, "crf": 10, "preset": "slow",
  "gif_fps": 15, "gif_width": 540 }
```

---

## 6. Предпросмотр и запись

### 6.1. Bridge (`orchestra-bridge.js`, path B)

Грузится ПОСЛЕ `arena.js` (classic-script → общее lexical-окружение, top-level
`let` и `handleStateChanged` доступны по bare-имени). Обязанности:
1. `window.__orchestraInit()` (baked hook → prebattle-гейт снят, RISK C);
2. скрытие `#prebattle-screen` (`hidePrebattleScreen()` + CSS-фолбэк
   `display:none` — иначе mp4 = 5с preBattle-оверлея);
3. stub `window.io` (Socket.IO не нужен — мы гоним `handleStateChanged` сами);
4. `userId = <viewer_uid>` (top-level let, RISK E — viewer_id gate `arena.js:3589`);
5. `fetch('/api/orchestra/frames/<run_id>')` → итерация кадров через
   `handleStateChanged({state, sound_events, data:{actor_user_id, sound_events}})`
   с `await sleep(display_ms/speed)`;
6. controls (`window.__orchestraController`: play/pause/step/seek/speed) и
   `window.__orchestraDone=true` по завершении (ждёт recorder).

URL `/player?…&ea_platform=android_app` обязателен (внешний-browser gate).
`music=1&sfx=1` для предпросмотра со звуком; рекордер ставит `music=0`
(in-page музыку всё равно не записать — её добавляет серверный микс).

### 6.2. Мобильное соотношение

Запись и предпросмотр — мобильный портрет. Ширина ≤420px CSS → срабатывает
`@media (max-width:420px)` из `arena-styles.css` → мобильный лейаут арены (а не
десктоп-версия). webm/mp4 = 414×896 CSS × `device_scale_factor=2` = **828×1792**.
Предпросмотр: `/preview` отдаёт `preview.html` — «телефон»-обёртку с
`<iframe src="/player?…">` фиксированного размера 414×896 (`aspect-ratio`,
`width:min(414px,96vw,44.36vh)`). У iframe свой viewport → media-query
срабатывает даже в широком десктоп-окне. Spinner до загрузки `/player`.

### 6.3. Звук

`with_audio=true` по умолч. (раньше `false` → SFX отсутствовали). Серверный
микс включает SFX-timeline + зацикленную `arena_theme.wav` (volume 0.3).
Headless Chromium не имеет audio-output device → в-page звук был бы тишиным
(RISK B); поэтому микс серверный, из известного timeline+sound_events+config.
GIF звука не содержит (формат не поддерживает).

---

## 7. MCP-сервер (`mcp_server.py`)

Hand-rolled JSON-RPC 2.0 over stdio (pattern `rlhf_env/mcp_server.py`), тонкий
async HTTP-клиент к серверу оркестра (8095). **16 инструментов** в 4 группах:

- **Загрузка:** `list_scenarios`, `get_scenario(name, as_v2=true)` (v1
  авто-мигрируется), `migrate_v1_to_v2(scenario)`.
- **Создание (графами):** `create_blank_scenario(meta)`, `build_graph(spec)`
  (собирает v2 из декларативных `nodes`/`edges`; `base_scenario_name`
  наследует init+classic_params+meta; валидация через HTTP), `save_scenario`,
  `delete_scenario`, `validate_scenario`.
- **Предпросмотр:** `preview_frames(scenario)` (покадровые structured «итоговые
  сцены» p1/p2 board+hero+mana+mechanics, НЕ изображения; viewer_side derives
  из side_uids) + `preview_arena_url`; `get_frames(run_id, summary=true)`.
- **Экспорт:** `export_mp4(scenario, wait, timeout, inline)`,
  `export_gif(scenario, wait, timeout, inline)`,
  `get_record_status(job_id)`, **`get_record_file(job_id)`** (достать готовый
  файл инлайн — байты как MCP content-item: gif→`ImageContent`, mp4→
  `ResourceContent` с base64-`blob`; `inline=true` на export прикрепляет байты
  к ответу), `list_cards(filter)`, `list_cosmetics()`.

`dispatch` tools/call выносит `result["_content"]` из text-метаданных в
отдельные content-items → `content=[text, image|resource]`, base64 **не
дублируется** в text. `isError=true` когда `_tool` вернул execution-error
(`{error:…}` без `ok`); валидационный `{ok:false,error}` остаётся
`isError:false`. JSON-RPC strict: unknown method → top-level `{error:{-32601}}`
(НЕ в `result`); missing/invalid params → `-32602`; non-object/batch-not-dict →
`-32600`; notifications (без `id`) → без ответа; parse-error → `id:null`;
`ping` → `{}`.

Запуск: `start_orchestra.sh mcp` (auto-start HTTP по умолч.,
`--no-auto-start`/`ORCH_AUTO_START=0`) или
`python3 -m extra_orchestra.mcp_server`. GOTCHA: `aiohttp.ClientSession`
создаётся ВНУТРИ loop'а (`asyncio.run(_amain(base_url))`); stdin через
`asyncio.StreamReader` (cancellable). `--auto-start` передаёт `--host`/`--port`
из `--base-url` серверу; fail-fast (exit 2) если бэкенд не поднялся.

> Skill `extra-orchestra` (см. `~/.claude/skills/extra-orchestra/SKILL.md`) —
> емкий саммари для агентов с примерами.

---

## 8. Безопасность и gotchas

- **Path-traversal в `orch_record`:** `scenario.name` санитизируется
  `_safe_name_slug` — `re.sub(r"[^A-Za-z0-9_-]+","_",…).strip("_")` (точки
  запрещены → `..` невозможен; пустой → `orchestra`); плюс
  `out_path.resolve().relative_to(RECORDINGS_DIR.resolve())` guard.
- **Path-traversal в `scenario_store.load`:** `Path(name).name` (basename).
- **Детерминизм (RISK A):** monkeypatch `core.effects.random = Random(seed)` +
  `uuid5` instance_id. Два прогона с тем же seed → идентичный frame-list.
- **Prebattle gate (RISK C):** baked `window.__orchestraInit` + скрытие
  `#prebattle-screen` (top-level `let`, НЕ на `window`).
- **viewer_id gate (RISK E):** bridge ставит `window.userId=<viewer_uid>`.
- **Wire drift (RISK F):** `scripts/sync_borrowed.py` (ручной) +
  `test_serializers.py`; snapshot `1282fcb8` в заголовках borrowed-файлов.
  **Никогда полный auto-refresh из `webapp/`.**
- **Запись mp4 = 5с preBattleScreen:** bridge обходит `startPrebattleSequence`,
  поэтому зовёт `hidePrebattleScreen()` + CSS-фолбэк.
- **Preview «Сессия истекла»:** `compute-frames` отдаёт `auth`; editor.js /
  bridge дописывают `&_auth=<auth>` в URL.
- **Сервер НЕ hot-reload'ится:** long-running 8095-процесс не подхватывает
  правки кода — перезапускать перед live-верификацией MCP/экспорта
  (`kill <pid>; python3 -m extra_orchestra.server --port 8095`).
- **Cache-bust editor:** `editor.html` ссылается на `editor.js?v=v2graphN` /
  `editor.css?v=v2graphN` — bumped при каждом изменении static-файлов
  (headless Chrome иначе кэширует старую версию).
- **`[hidden]{display:none!important}` reset** в editor.css — иначе
  `.orch-overlay{display:flex}` перекрывает `hidden` и блокирует клики.

---

## 9. Тесты и верификация

`python3 -m pytest extra_orchestra/tests/ -q` → 147 passed, 3 skipped.
Gated Playwright e2e (`ORCH_E2E=1`): `test_e2e_record_mp4` (h264 828×1792
portrait + aac), `test_e2e_record_gif` (gif portrait, no-audio),
`test_editor_graph_e2e`.

- `test_scenario_engine.py` — v1soldatik (aoe_silence снимает все 3 механики;
  turn_number==15 на init); детерминизм (два прогона == frame-list).
- `test_scenario_graph.py` — v2 structure/runner/migration/dispatch/store
  back-compat (50 test).
- `test_serializers.py` — `OrchestraBattleEngine.get_full_state` ==
  `BattleEngine.get_full_state` byte-for-byte (wire-drift детект).
- `test_sound_events.py` — play Солдатика → `[{deploy},{mechanic:aoe_silence}]`.
- `test_audio_mix.py` — синтетический timeline → корректный ffmpeg
  filter_complex (mock subprocess).
- `test_gif_export.py` — двухпроходная palette-цепочка (mock).
- `test_smoke_e2e.py` — HTTP-контракт (50 карт, soldatik, validate,
  compute-frames, frames) + gated mp4/gif e2e + path-traversal regression.
- `test_mcp_server.py` — 70 test MCP-сервера (pure-хелперы, live tool calls,
  export stub-client, stdio JSON-RPC edges, **inline-файл** через
  `_StubClient.record_download`).