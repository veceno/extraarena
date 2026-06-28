# ExtraOrchestra — воспроизведение действий в заданной последовательности

Этот документ — исчерпывающе про то, **как ExtraOrchestra воспроизводит
игровые действия в строго заданной заранее последовательности**: как
последовательность описывается (граф), как раннер идёт по ней шаг за шагом,
как каждое действие применяется к реальному игровому ядру, как получается
покадровая лента, и как эта лента проигрывается на настоящей арене. Во второй
половине — карта текущего онбординг-туториала и точки, где наработки
оркестры естественно ложатся в переработку обучения новичков (сам туториал
не меняется — только контекст для будущей интеграции).

> См. также: `DOCUMENTATION.md` — общий справочник (API, компоненты, MCP).
> Код: `components/scenario_graph_runner.py` (v2 раннер),
> `components/scenario_engine.py` (v1 + общие хелперы),
> `components/arena_engine.py` ( shim ядра), `static/orchestra-bridge.js`
> (плейер кадров).

---

## 1. Модель одним абзацем

Автор описывает бой как **направленный граф действий**: одна init-сцена
(начальная расстановка) → цепочка узлов (`play_card` / `attack` / `mana_draw`
/ `end_turn` / `hold`), соединённых рёбрами в один путь. Раннер строит
`GameState` один раз из init-сцены, обходит путь от `start`, и на каждом
`action`-узле вызывает настоящее игровое ядро (`ArenaEnvironment.step` через
`OrchestraBattleEngine.apply_action`), после чего делает снимок состояния
(viewer-perspective `get_full_state`). Так получается лента кадров
`List[Frame]`, где каждый кадр = «состояние после действия + звуковые события +
сколько мс его показывать». Эту ленту bridge проигрывает на реальной арене
через `handleStateChanged` — тот же入口, что Socket.IO в prod-игре. Всё
детерминировано (seed + `uuid5` instance_id) — два прогона дают идентичные
кадры.

---

## 2. Как описывается последовательность (v2 graph)

### 2.1. Узлы

Граф `graph = {start, nodes[], edges[]}`. Узлы четырёх видов:

| kind | type/поля | Что делает раннер |
|---|---|---|
| `scene` | `type:"init"`, `turn_number`, `starting_side`, `display_ms`, `p1`, `p2` | Строит `GameState` (один, =`start`, без входящих). Эмитит init-кадр (`action_kind:"init"`, `turn_id:"init"`). |
| `scene` | `type:"hold"`, `display_ms` | Удерживает текущий снимок `display_ms` мс (замена `wait`/post-act). `action_kind:"hold"`. |
| `turn` | `turn.side`, `turn.intro_ms` (default 0) | Sanity-маркер чьего хода. `intro_ms>0` → отдельный `turn_intro`-кадр. Задаёт `turn_id` для последующих action-кадров. |
| `action` | `side ∈ {p1,p2}` **обязательно**, `action:{type, delay_ms, …}` | Шаг ядра. `type ∈ {play_card, attack, mana_draw, end_turn}`. |

### 2.2. action-узел → BaseAction

Поля action-узла (вложены в `action`, либо плоско на узле — редактор
сохраняет каноническую nested-форму `{kind:action, side, action:{…}}`):

| `action.type` | Поля | `core.actions` | Разрешение ссылок |
|---|---|---|---|
| `play_card` | `hand_index` (0-based), `target_id?`/`target_index?`/`target_is_hero?`, `position?`, `delay_ms?` | `PlayCardAction` | target — по `target_index` в board оппонента, `target_is_hero` → hero оппонента, иначе `target_id` |
| `attack` | `attacker_id?`/`attacker_index`, `target_id?`/`target_index?`/`target_is_hero`, `delay_ms?` | `AttackAction` | attacker — по `attacker_index` в своей board; target — по `target_index` в board оппонента, `target_is_hero` → `target_id=None` |
| `end_turn` | `delay_ms?` | `EndTurnAction` | — |
| `mana_draw` | `delay_ms?` | `ManaDrawAction` | — |

**Автор-френдли:** ссылки по индексам (`hand_index`, `attacker_index`,
`target_index`) — автор не знает `instance_id`. Продвинутый случай — прямые
`*_id` (instance_id). Разрешение (`_resolve_attacker_id` /
`_resolve_target_for_play` / `_resolve_target_for_attack`) проверяет границы
board и кидает `ScenarioError` при выходе за индекс.

`delay_ms` — пауза ПЕРЕД действием (точнее, длительность показа
результирующего кадра) → cadence предпросмотра/записи. При ошибке действия
`display_ms = max(delay, 800)` (минимум 0.8с, чтобы ошибка была видна).

### 2.3. Структура графа = один путь

`validate_graph_structure` (без catalog, быстрая):
- ровно один `scene/init`, и он = `graph.start`;
- max-1 исходящее И max-1 входящее ребро на узел (→ один путь);
- init не имеет входящих; ни одно ребро не ведёт в init;
- все узлы достижимы из `start`; нет циклов, нет self-loop, нет dangling рёбер;
- уникальные `id` узлов и рёбер;
- per-node: `action.side ∈ {p1,p2}`, `action.type` валиден, `scene.type ∈
  {init,hold}`, `turn.side` (если есть) валиден.

`layout`/`editor` — top-level, раннер **игнорирует** (чистая executable-семантика;
координаты только для визуального редактора).

### 2.4. v1 (turns[]) как частный случай

v1 — линейные `turns[i] = {side, duration_ms, end_with_end_turn, nodes[]}`.
Семантически то же: `nodes[]` упорядочены, `wait`-узел = `hold`,
`end_with_end_turn=true` без явного `end_turn` → неявный `EndTurnAction`.
`migrate_v1_to_v2` разворачивает это в граф (детерминированная auto-layout,
`NODE_GAP=220`). Frame dict **byte-identical** v1/v2 → recorder/bridge не
различают схемы. Dispatch — по `schema`.

---

## 3. Как раннер обходит последовательность

`run_scenario_graph(scenario, catalog)` (`scenario_graph_runner.py:176`):

1. `validate_graph_structure` → при ошибке `_result(…, error)` (frames=[]).
2. `_compat_scenario_for_init` собирает v1-совместимый dict из top-level +
   init-узла (чтобы переиспользовать `build_initial_state`).
3. **RISK A:** `_effects.random = Random(seed)` на время
   `build_initial_state` (строит `env, engine, viewer_uid, side_uids`).
   `apply_start_effects=False` → init-сцена ровно как автор описал.
4. Восстанавливает monkeypatch `Random(seed)` на весь обход.
5. `cur = graph["start"]`; цикл `while cur is not None`:
   - **scene/init** → init-кадр (`display_ms` из `scene.display_ms` или 1200);
     `last_turn_id = None`.
   - **scene/hold** → кадр-удержание (`display_ms` или 600); `turn_id =
     last_turn_id or "__no_turn__"`.
   - **turn** → side-guard (см. §4); если `intro_ms>0` → `turn_intro`-кадр;
     `last_turn_id = nid`.
   - **action** → side-guard → `_node_to_action` → `engine.apply_action` →
     кадр с `snapshot`, `sound_events`, `action_kind`, `delay_ms`. При
     `ok:False` — error-кадр + **break** (fail-fast). При `game_over` —
     `game_over`-кадр + break.
   - `cur = outgoing[cur]` (переход к следующему узлу по единственному ребру).
6. `finally: _effects.random = orig_random` (восстановление module-random).
7. `_result(frames, viewer_uid, side_uids, match_id, err)` —
   `total_ms = sum(display_ms)`.

**Ключевое:** состояние **наследуется** между узлами — один `ArenaEnvironment`,
шагаем `engine.step`. Никакого «телепорта состояния» между шагами (в отличие
от текущего онбординг-туториала — см. §7). init строится один раз.

---

## 4. Side/turn guards — почему «забытый end_turn» не тихо ломает бой

Каждый `action`-узел несёт **обязательный** `side ∈ {p1,p2}`. ПЕРЕД
`apply_action` раннер проверяет `side_uids[side] ==
env.state.current_turn_owner_id`:

```
if side_uid != current_owner:
    err = f"node {nid}: side '{side}' (uid {side_uid}) is not current turn owner
           (uid {current_owner}). Forgot end_turn?"
    → error-кадр (display_ms=max(delay,800)) + break
```

Это восстанавливает v1 safety-guard (`scenario_engine.py:318-324`): если автор
хочет, чтобы ходил p2, он обязан явно поставить `end_turn` от p1 — иначе
**явный ScenarioError**, а не silent wrong-side action (игрок «ходит» в чужой
ход). `turn`-узел делает ту же проверку для маркера хода.

`end_turn` шагает ядро → start-of-turn эффекты следующего игрока
(mana/draw/wake) честно и видимо применяются — это и есть «переход хода» в
последовательности.

---

## 5. Кадр и звук

`make_frame(snapshot, sound_events, display_ms, action_kind, turn_id, node_id,
error?)`:

- `snapshot` — `engine.get_full_state(viewer_uid)`, viewer-perspective (тот же
  JSON, что арена получает в prod). p1/p2 в нём — viewer-relative
  (`player`/`opponent`), маппинг обратно по `side_uids` делает
  `summarize_frames` в MCP.
- `sound_events` — порт `_sound_events_for_action` (`battle_engine.py:475-642`):
  deploy/mechanic/attack с детерминированным `event_id`
  `"orchestra:<turn>:<kind>:<card>:<event>"` (arena.js дедуплит SFX по event_id).
  `aoe_silence` НЕ входит в `_is_play_sound_mechanic` → только deploy-событие.
- `action_kind` — `"init"`/`"hold"`/`"turn_intro"`/`"play_card"`/`"attack"`/
  `"mana_draw"`/`"end_turn"`/`"game_over"` (или `result.action_kind`).
- `display_ms` — cadence (сколько мс показывать кадр).
- `turn_id` — ближайший предшествующий `turn`-узел / `"init"` / `"__no_turn__"`.

`total_ms = sum(display_ms)` — бюджет реального времени проигрывания.

---

## 6. Как кадры проигрываются на арене (bridge)

`orchestra-bridge.js` (path B):

1. `window.__orchestraInit()` (baked hook → prebattle-гейт снят) +
   `hidePrebattleScreen()` (иначе оверлей висит поверх поля).
2. stub `window.io` (Socket.IO не нужен).
3. `userId = <viewer_uid>` (top-level `let` arena.js — общее lexical-окружение
   classic-scripts; viewer_id gate `arena.js:3589`).
4. `fetch('/api/orchestra/frames/<run_id>')` → массив кадров.
5. Цикл: `handleStateChanged({state: f.snapshot, sound_events: f.sound_events,
   data:{actor_user_id: viewerUid, sound_events: …}})` → `await
   sleep(display_ms / speed)` → следующий кадр. `handleStateChanged` —
   **тот же入口**, что Socket.IO `state_changed` в prod-игре → арена
   рендерит каждый кадр как настоящий ход.
6. controls (`window.__orchestraController`: pause/resume/step/seek/speed);
   по завершении `window.__orchestraDone = true` (ждёт recorder).

Запись (recorder) — то же, но в headless Chromium с `record_video_dir` →
webm; затем ffmpeg: mp4 (видео + серверный аудио-микс из `sound_events` +
`arena_theme.wav`) или GIF (palettegen+paletteuse, без звука). Серверный микс
нужен потому, что headless Chromium не имеет audio-output device (RISK B) —
но мы знаем полный timeline+sound_events+`card_sfx_config.json`.

---

## 7. Онбординг-туториал: текущая модель и где ложится оркестра

> Сам туториал **не меняется**. Этот раздел — контекст для будущей переработки
> обучения новичков с использованием наработок ExtraOrchestra.

### 7.1. Как туториал устроен сейчас

- **Ядро:** `onboarding_tutorial.py` — `TUTORIAL_STEPS` (80-150, 9 шагов 0..8),
  `WRONG_ACTION_FEEDBACK`, фабрики карт, `TutorialBattleEngine(BattleEngine)`
  (252-602). `_build_state_for_step(step)` (273-392) на каждом шаге **заново
  строит `GameState` с нуля** из заготовок `_hero`/`_warrior` (детерминированные
  `uuid5` instance_id), выставляя нужные HP/board/hand под конкретный шаг.
  `apply_tutorial_action` (535-601) вызывает реальный `BattleEngine.execute_action`
  только для `play_card`/`attack`/`end_turn`, и сразу после успеха
  `set_tutorial_step(step+1)` перебирает состояние под следующий шаг.
  Противник «ходит» через `auto_continue` (549-566) — переключение шагов с
  `scripted_event` метками, не реальный AI.
- **Сервер:** `web/server.py` — `/api/onboarding/*` (status, welcome/complete,
  tutorial/start, tutorial/action, menu-tour, complete, newbie-path).
  `_handle_onboarding_tutorial_action` (1083-1230),
  `_ensure_tutorial_engine_for_user` (20167-20187) кладёт
  `TutorialBattleEngine` в `active_matches[match_id]`,
  `match_game_modes[match_id]="tutorial"`. `match_id = tutorial-{user_id}`,
  редирект `/arena?id={match_id}&onboarding=1`.
- **БД:** `infrastructure/database.py` — `user_onboarding` (status, current_step,
  tutorial_step, tutorial_match_id, newbie_path_progress JSONB, …),
  `onboarding_events` (step, completed, time_spent, metadata JSONB).
- **Фронтенд:** `webapp/arena.js` — `onboarding*` globals, `isOnboardingTutorialState`
  (2453-2469), `playOnboardingStepCue`, auto-advance; `arena.html` —
  `arena-onboarding-layer`; `main.js` — welcome-флоу.
- **Контракт туториала** (`tutorial_payload`, 407-434): `step_index`, `message`,
  `hint`, `target` (селектор подсветки), `allowed` (единственное разрешённое
  действие), `legal_actions`, `attacker_instance_id`. `allowed` ≈
  `{type, card_id, attacker_card_id, target_is_hero}`.

### 7.2. Семантическое соответствие

`onboarding_tutorial.py` и `scenario_graph_runner.py` — **две реализации одной
идеи** (детерминированный scripted battle):

| Туториал сейчас | ExtraOrchestra |
|---|---|
| `TUTORIAL_STEPS` (Python-словарь шагов) | v2 graph `nodes[]` (JSON, редактируется визуально) |
| `_build_state_for_step` (per-step телепорт состояния) | `scene/init` строит состояние **один раз**, раннер идёт по пути (состояние наследуется) |
| `apply_tutorial_action` (один action → step+1) | `action`-узел → `apply_action` → следующий узел по ребру |
| `allowed` (`{type, card_id, attacker_card_id, target_is_hero}`) | action-узел `{type, side, hand_index/attacker_index/target_index/target_is_hero}` — изоморфны |
| `validate_tutorial_action` (506-533) | `validate_graph_structure` + side-guard |
| `scripted_event` (оппонент «ходит») | `action`-узлы с `side:"p2"` в графе (реальный ход через ядро, не заглушка) |
| `tutorial_payload()` → `tutorial` в `get_full_state` | `summarize_frames` / frames-контракт `orch_get_frames` |

### 7.3. Естественные точки интеграции (по убыванию)

1. **`TUTORIAL_STEPS` + `_build_state_for_step` (80-392)** → заменить
   императивный реконструктор на загрузку extraOrchestra-сценария (init scene
   + graph ходов). Состояние строится один раз, `apply_tutorial_action`
   проигрывает следующий node графа через `scenario_graph_runner`. Убирает
   дублирование между `TUTORIAL_STEPS` и `_build_state_for_step`; сценарий
   редактируется в визуальном редакторе оркестры, а не в Python.
2. **`apply_tutorial_action` / `set_tutorial_step` (535-601, 393-395)** —
   место, где graph-runner подключается как «движок шагов». `allowed` уже
   изоморфен action-node; `validate_tutorial_action` maps 1:1 на
   `validate_graph_structure` + `_node_to_action`.
3. **`_handle_onboarding_tutorial_action` (web/server.py:1083-1230) +
   `_ensure_tutorial_engine_for_user` (20167-20187)** — слой адаптации:
   выбирать сценарий по `user_id`/локали и инстанцировать движок поверх
   graph-runner'а. `match_id = tutorial-{user_id}` и
   `is_onboarding_tutorial=True` контракт сохранить.
4. **`scenarios/` как источник учебных сцен** — onboarding-сценарии (бывший
   `TUTORIAL_STEPS` → `scenarios/onboarding_basic.json`) рядом с
   `soldatik-demo.json`. Product-команда редактирует туториал без правки
   Python; recorder генерирует превью/видео онбординга.
5. **Фронтенд `arena.js` (2453-2469, onboarding overlay)** — почти не требует
   изменений: `tutorial_payload()` уже отдаёт тот же контракт, что
   `orch_get_frames` при покадровом воспроизведении. Достаточно выровнять
   поле `tutorial` в `get_full_state` (397-405) с frames-контрактом.
6. **БД не требует изменений** — `tutorial_step INT` ≈ индекс node в графе;
   `onboarding_events.metadata JSONB` уже хранит `step_id`/`tutorial_step`.

### 7.4. Что даёт переход на graph-модель (контекст для переработки)

- **Состояние наследуется** вместо per-step телепорта → шаги честно видят
  последствия предыдущих (board/hp/mana меняются по-настоящему), проще
  авторить многошаговые сценарии.
- **Реальный ход оппонента** через `action side:"p2"` (а не `auto_continue`
  переключение шагов) → демонстрация механик врага честная.
- **Тот же движок, что в prod** (`ArenaEnvironment.step` через
  `OrchestraBattleEngine`) — нет расхождения между туториалом и реальным боем.
- **Визуальный graph-редактор** (`static/editor.*`) + MCP-инструменты
  (`build_graph`, `preview_frames`, `export_mp4`) → scenario-команда и агенты
  авторят/проверяют/экспортят учебные ролики без правки кода.
- **Детерминизм** (seed + `uuid5` + RISK A monkeypatch) — воспроизводимо для
  QA и регрессий.
- **Покадровая лента** → переиспользуется и для live-предпросмотра в туториале,
  и для mp4/GIF-демо (один сценарий — три выхода: игра, превью, видео).

### 7.5. Что важно сохранить при интеграции

- **Side-guard**: graph-модель требует явного `end_turn` для перехода хода.
  В туториале с одним «активным» игроком это естественно, но сценарий должен
  корректно закрывать ходы (иначе `ScenarioError` вместо тихого перехода).
- **`apply_start_effects=False`**: init-сцена = ровно расстановка автора. Если
  туториалу нужны start-of-turn эффекты на каком-то шаге — их даёт `end_turn`,
  а не авто-применение на старте.
- **`allowed` ↔ action-node**: при замене `validate_tutorial_action` на graph
  —一对一 map «разрешённое действие шага» = текущий action-узел; подсветка
  `target` берётся из полей узла (`hand_index`/`attacker_index`/`target_index`
  → селектор).
- **viewer-perspective snapshot**: `get_full_state(viewer_uid)` отдаёт
  viewer-relative `player`/`opponent`; фронтенд туториала уже работает в этой
  модели — менять не нужно.
- **Никаких правок `webapp/`**: оркестра работает на frozen `webapp_borrow/`
  (minimal-port constraint); интеграция в tuториал идёт через серверный
  адаптер + сценарии, не через правки `arena.js`.

---

## 8. Практический пример (Солдатик demo)

`scenarios/soldatik-demo.json` (v1, ход 15): p1 hand = Солдатик (card 47,
`aoe_silence`), p2 board = Годжо Сатору (24, shield+shield_refresh), Наофуми
(30, taunt), Крипер (34, deathrattle_aoe_damage_2). Один ход p1: `wait` 1500мс
→ `play_card hand_index=0` (Солдатик) → `wait` 2600мс.

Раннер: init-кадр (ход 15, оппонент с 3 механиками) → `wait`-кадр (удержание)
→ `play_card`: `apply_action` выставляет Солдатика → `aoe_silence` снимает все
3 механики (`_silence_units(candidates, limit=3)`, `core/effects.py:~1020`) →
кадр с пустыми `mechanics=[]` у всех вражеских юнитов + deploy-sound_event →
`wait`-кадр. Bridge проигрывает это на арене: виден init → Солдатик
выставляется → визуал `aoe_silence` → 3 врага теряют mechanic-иконки.
Recorder → `828×1792` mp4 (h264 + aac, SFX + музыка) или `540×1168` GIF.

Тот же сценарий через MCP: `get_scenario("soldatik-demo", as_v2=true)` →
`preview_frames` (покадровые structured-сцены для рассуждения без vision) →
`export_gif(inline=true)` (байты GIF как image-content) / `export_mp4(inline=true)`
(байты mp4 как resource-blob). См. `~/.claude/skills/extra-orchestra/SKILL.md`.