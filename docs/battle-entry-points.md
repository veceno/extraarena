# Точки входа в бой (Главное меню → Арена)

## Обзор

Игрок попадает на арену одним из следующих путей:

| # | Точка входа | game_mode | Противник | Награды |
|---|-------------|-----------|-----------|---------|
| 1 | «В БОЙ» → Классика | `classic` | PvP / бот | трофеи + кейсы |
| 2 | «В БОЙ» → ExtraArena: Blitz | `extra_arena:blitz` | PvP / бот | трофеи + кейсы |
| 3 | «В БОЙ» → ExtraArena: Draft | `extra_arena:draft` | PvP / бот | трофеи + кейсы (placeholder) |
| 4 | Круг → Тренировка | `training` | Бот (выбор сложности) | нет |
| 5 | Круг → Дружеский бой | `friendly` | Друг | нет |

---

## Карта последовательностей

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         ПОЛНАЯ КАРТА: ГЛАВНОЕ МЕНЮ → АРЕНА                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

  ИГРОК                      FRONTEND (index.html)                BACKEND (server.py)          ЯДРО
  ─────                      ─────────────────────                ───────────────────          ────
    │                              │                                      │                      │
    │  тап «В БОЙ»                 │                                      │                      │
    │ ──────────────────────────> │                                      │                      │
    │                              │                                      │                      │
    │                    ┌─────────┴──────────┐                           │                      │
    │                    │  ArenaScreen        │                           │                      │
    │                    │  onClick →          │                           │                      │
    │                    │  onStartBattle()    │                           │                      │
    │                    └─────────┬──────────┘                           │                      │
    │                              │                                      │                      │
    │                    ┌─────────▼──────────┐                           │                      │
    │                    │  GameModeSheet      │                           │                      │
    │                    │  ┌───────────────┐  │                           │                      │
    │                    │  │ classic       │  │                           │                      │
    │                    │  │ extra_arena   │──┤─ blitz ▸ "extra_arena:blitz"
    │                    │  │               │  │─ draft ▸ "extra_arena:draft"
    │                    │  ├───────────────┤  │                           │                      │
    │                    │  │ выбор колоды  │  │                           │                      │
    │                    │  └───────────────┘  │                           │                      │
    │                    └─────────┬──────────┘                           │                      │
    │                              │                                      │                      │
    │                    onConfirm(fullMode, deckId)                       │                      │
    │                              │                                      │                      │
    │                    ┌─────────▼──────────┐                           │                      │
    │                    │  window.startBattle │                           │                      │
    │                    │  (profile,deckId,  │                           │                      │
    │                    │   mode)            │                           │                      │
    │                    └─────────┬──────────┘                           │                      │
    │                              │                                      │                      │
    │                              │  POST /api/match/find                │                      │
    │                              │  ┌─────────────────────────────┐    │                      │
    │                              │  │ user_id, trophies,           │    │                      │
    │                              │  │ game_mode, selected_deck_id  │    │                      │
    │                              │  └─────────────────────────────┘    │                      │
    │                              │ ─────────────────────────────────────>                      │
    │                              │                                      │                      │
    │                              │                    ┌─────────────────▼──────────────────┐   │
    │                              │                    │  match_find_handler                │   │
    │                              │                    │  → matchmaker.find_match()         │   │
    │                              │                    │    ├─ PvP: _search_loop()          │   │
    │                              │                    │    │    → _pair_players()          │   │
    │                              │                    │    └─ No opponent:                 │   │
    │                              │                    │         _create_bot_match()        │   │
    │                              │                    └─────────────────┬──────────────────┘   │
    │                              │                                      │                      │
    │                              │                    ┌─────────────────▼──────────────────┐   │
    │                              │                    │  _prepare_and_cache_engine()        │   │
    │                              │                    │  ┌─────────────────────────────┐   │   │
    │                              │                    │  │ загрузка профилей из БД     │   │   │
    │                              │                    │  │ загрузка колод (deck_ids)   │   │   │
    │                              │                    │  │ загрузка card_cache         │   │   │
    │                              │                    │  └─────────────────────────────┘   │   │
    │                              │                    │                                      │   │
    │                              │                    │  new BattleEngine(                    │   │
    │                              │                    │    game_mode=...)────────────────────>│   │
    │                              │                    │                                      │   │
    │                              │                    │  engine.create_match(            │   │
    │                              │                    │    p1_data, p2_data)─────────────────>│   │
    │                              │                    │                                      │   │
    │                              │                    │                              ┌───────▼──────┐
    │                              │                    │                              │ ArenaEnviron-│
    │                              │                    │                              │ ment(game_   │
    │                              │                    │                              │ state,       │
    │                              │                    │                              │ mana_per_turn│
    │                              │                    │                              │ =2 if blitz  │
    │                              │                    │                              │ else 1)      │
    │                              │                    │                              └──────────────┘
    │                              │                    │                                      │
    │                              │                    │  active_matches[match_id] = engine   │
    │                              │                    │  match_game_modes[match_id] = mode   │
    │                              │                    └─────────────────┬──────────────────┘   │
    │                              │                                      │                      │
    │                              │  ◄──── JSON ────────────────────── │                      │
    │                              │  {status:"found", match_id,           │                      │
    │                              │   opponent_name, opponent_avatar,     │                      │
    │                              │   opponent_trophies, game_mode}       │                      │
    │                              │                                      │                      │
    │                              │  ┌─────────────────────────┐        │                      │
    │                              │  │ PreBattleScreen (3 сек)  │        │                      │
    │                              │  │ VS-экран с отсчётом     │        │                      │
    │                              │  └────────────┬────────────┘        │                      │
    │                              │               │                     │                      │
    │                              │  window.location.href =              │                      │
    │                              │  "/arena?id={matchId}                │                      │
    │                              │   &user_id={userId}"                │                      │
    │                              │               │                     │                      │
    │                              ▼               ▼                     │                      │
    │                    ┌─────────────────────────────┐                 │                      │
    │                    │  arena.html + arena.js       │                 │                      │
    │                    │                              │                 │                      │
    │                    │  matchId = urlParams.get('id')│                │                      │
    │                    │  userId = urlParams.get(     │                 │                      │
    │                    │           'user_id')         │                 │                      │
    │                    └──────────────┬──────────────┘                 │                      │
    │                                   │                                │                      │
    │                                   │  socket.emit('join_match',     │                      │
    │                                   │    {match_id, user_id})        │                      │
    │                                   │ ──────────────────────────────>│                      │
    │                                   │                                │                      │
    │                                   │                    join_match handler                 │
    │                                   │                    → sio.enter_room(sid, match_id)    │
    │                                   │                    → emit('joined_match')             │
    │                                   │                                │                      │
    │                                   │  socket.emit('client_ready',   │                      │
    │                                   │    {match_id, user_id})        │                      │
    │                                   │ ──────────────────────────────>│                      │
    │                                   │                                │                      │
    │                                   │                    client_ready handler               │
    │                                   │                    → engine.mark_client_ready()       │
    │                                   │                    → check_and_run_bot() (если бот)   │
    │                                   │                                │                      │
  ───                                 ───                              ───                    ───
  БОЙ НАЧАЛСЯ — обмен ходами через Socket.IO: play_card / attack / end_turn / surrender
  ───                                 ───                              ───                    ───
```

---

## Детали режимов

### 1. Классика (`classic`)

```
ПОЛЬЗОВАТЕЛЬ                    FRONTEND                        BACKEND
───────────                    ────────                        ───────
  │                               │                               │
  │ ArenaScreen → В БОЙ           │                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │ GameModeSheet → Классика      │                               │
  │ выбор колоды                  │                               │
  │ «В БОЙ — Классика»            │                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │                               │  POST /api/match/find          │
  │                               │  game_mode = "classic"         │
  │                               │ ────────────────────────────> │
  │                               │                               │
  │                               │                    Matchmaker.find_match()
  │                               │                    ├─ ��сть соперник: _pair_players()
  │                               │                    └─ нет соперника: _create_bot_match()
  │                               │                               │
  │                               │                    _prepare_and_cache_engine()
  │                               │                    → BattleEngine(game_mode="classic")
  │                               │                       turn_duration = 25
  │                               │                       mana_per_turn = 1
  │                               │                    → engine.create_match()
  │                               │                       ArenaEnvironment(state, mana_per_turn=1)
  │                               │                               │
  │                               │  ◄─── {status:"found", ...}  │
  │                               │                               │
  │ PreBattleScreen (3 сек)       │                               │
  │ → /arena?id=...&user_id=...  │                               │
```

**Параметры:** стандартные правила — 25 сек на ход, +1 мана/ход, трофеи начисляются.

---

### 2. ExtraArena: Blitz (`extra_arena:blitz`)

```
ПОЛЬЗОВАТЕЛЬ                    FRONTEND                        BACKEND
───────────                    ────────                        ───────
  │                               │                               │
  │ GameModeSheet                 │                               │
  │ ├─ Выбор: extra_arena         │                               │
  │ └─ Саб-модификатор: blitz ⚡  │                               │
  │                               │                               │
  │ fullMode = "extra_arena:blitz"│                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │                               │  POST /api/match/find          │
  │                               │  game_mode = "extra_arena:blitz"
  │                               │ ────────────────────────────> │
  │                               │                               │
  │                               │                    _prepare_and_cache_engine()
  │                               │                    → BattleEngine(game_mode="extra_arena:blitz")
  │                               │                       _is_blitz = True
  │                               │                       turn_duration = 5
  │                               │                    → engine.create_match()
  │                               │                       ArenaEnvironment(state, mana_per_turn=2)
  │                               │                               │
  │ PreBattleScreen               │                               │
  │ → /arena?id=...&user_id=...  │                               │
```

**Параметры Blitz:** 5 сек на ход, +2 маны/ход, ускоренный темп.

**Разбор строки `game_mode`:**

| Уровень | Файл | Строка | Что делает |
|---------|------|--------|-----------|
| Frontend | `index.html:14276` | `const fullMode = mode === 'extra_arena' ? 'extra_arena' : mode;` | `mode` уже приходит каноничным id из `GameModeSheet` |
| Server | `server.py:10531` | `raw_game_mode = data.get("game_mode") or data.get("mode") or "classic"` → `_resolve_db_aware_mode(...)` | Канонизация через `infrastructure/match_modes.resolve_mode_config` |
| BattleEngine | `battle_engine.py:114-117` | `self.mode_config = resolve_mode_config(game_mode); self.turn_duration = self.mode_config.classic.turn_duration_seconds` | Blitz-параметры читаются из `ModeConfig` |
| Engine | `battle_engine.py:346-349` | `ArenaEnvironment(game_state, classic_params=self.mode_config.classic)` | `mana_per_turn` берётся из `classic.mana_per_turn` |
| Config | `infrastructure/match_modes.py:78-91` | `extra_arena:blitz → ClassicParams(turn_duration_seconds=5, mana_per_turn=2, hero_health_multiplier=0.5, …)` | Источник истины для blitz-параметров |

---

### 3. ExtraArena: Draft (`extra_arena:draft`)

```
ПОЛЬЗОВАТЕЛЬ                    FRONTEND                        BACKEND
───────────                    ────────                        ───────
  │                               │                               │
  │ GameModeSheet                 │                               │
  │ ├─ Выбор: extra_arena         │                               │
  │ └─ Саб-модификатор: draft 🃏  │                               │
  │                               │                               │
  │ fullMode = "extra_arena:draft"│                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │                               │  POST /api/match/find          │
  │                               │  game_mode = "extra_arena:draft"
  │                               │ ────────────────────────────> │
  │                               │                               │
  │                               │                    BattleEngine(game_mode="extra_arena:draft")
  │                               │                       _is_blitz = False
  │                               │                       turn_duration = 25
  │                               │                       mana_per_turn = 1
  │                               │                               │
  │                               │  ⚠️ PLACEHOLDER                 │
  │                               │  Бой проходит как классика     │
  │                               │  Draft-логика не реализована   │
```

**Draft — placeholder.** Строка `"extra_arena:draft"` принимается и передаётся, но `_is_blitz = (game_mode == "extra_arena:blitz")` → только blitz даёт `True`. Draft идёт как обычный классический бой (25 сек, +1 мана). Подтверждено в `SYSTEMS_AUDIT.md:39`: «Draft-режим — есть в UI как опция, но backend-логики не имеет».

---

### 4. Тренировка (`training`)

```
ПОЛЬЗОВАТЕЛЬ                    FRONTEND                        BACKEND
───────────                    ────────                        ───────
  │                               │                               │
  │ ArenaScreen → круг (спасат.) │                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │ BattlePickSheet               │                               │
  │ ├─ Вкладка: «🤖 Тренировка»   │                               │
  │ ├─ Сложность: lite/easy/      │                               │
  │ │   medium/hard/max           │                               │
  │ ├─ Выбор колоды               │                               │
  │ └─ «В БОЙ»                    │                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │                               │  window.startVsBot(diff,       │
  │                               │    profile, deckId, 'training')│
  │                               │                               │
  │                               │  POST /api/match/vs-bot        │
  │                               │  ┌─────────────────────────┐  │
  │                               │  │ user_id                  │  │
  │                               │  │ difficulty: "medium"     │  │
  │                               │  │ game_mode: "training"    │  │
  │                               │  │ deck_id: 1               │  │
  │                               │  └─────────────────────────┘  │
  │                               │ ────────────────────────────> │
  │                               │                               │
  │                               │                    match_vs_bot_handler
  │                               │                    → matchmaker._create_bot_match()
  │                               │                    → bot_info["difficulty"] = запрос.difficulty
  │                               │                    → _prepare_and_cache_engine()
  │                               │                       BattleEngine(game_mode="training")
  │                               │                               │
  │                               │                    if game_mode == "training":
  │                               │                      opponent_name = "🤖 Тренер"
  │                               │                               │
  │                               │  ◄─── {status:"found", ...}  │
  │                               │                               │
  │ PreBattleScreen → /arena      │                               │
```

**Особенности тренировки:**

- **Нет трофеев** — `_process_battle_end()` (`server.py:2801-2803`): использует `mode_config = resolve_mode_config(game_mode); rewards = mode_config.rewards`; режимы `training`/`friendly` определены в `infrastructure/match_modes.py:93-104` с `RewardParams(enabled=False, ...)` (`NO_REWARDS`).
- **Нет штрафов за surrender** — `server.py:3058-3059`: `eligible_mode = str(game_mode or "").lower() not in ("training", "friendly")` — пропуск штрафов/наград для training/friendly; `rewards.trophies` тоже False для этих режимов (`server.py:2857, 2864`).
- **Сложность бота** передаётся через `bot_info["difficulty"]` (`server.py:10299`) → `bot_difficulty` в `BattleEngine`.
- **5 уровней сложности**: `tier_lite_0000`, `tier_easy_0100`, `tier_medium_1200`, `tier_hard_4500`, `tier_max_9000` (`index.html:6763-6769` — константа `DIFFICULTIES`).
- **Имя противника**: константа `TRAINING_BOT_NAME = "🤖 Тренер"` (`server.py:108`), подставляется через профиль бота (`server.py:827, 10804`).

---

### 5. Дружеский бой (`friendly`)

```
ПОЛЬЗОВАТЕЛЬ                    FRONTEND                        BACKEND
───────────                    ────────                        ───────
  │                               │                               │
  │ ArenaScreen → круг (спасат.) │                               │
  │ ────────────────────────────> │                               │
  │                               │                               │
  │ BattlePickSheet               │                               │
  │ ├─ Вкладка: «🤝 Дружеская     │                               │
  │ │   игра»                     │                               │
  │ ├─ Список друзей: пуст        │                               │
  │ ├─ Ввод Telegram ID           │                               │
  │ └─ Кнопка: disabled           │                               │
  │    «Отправить вызов»          │                               │
  │                               │                               │
  │  ⚠️ ЗАГЛУШКА                  │                               │
  │  «Дружеские игры в            │                               │
  │   разработке»                 │                               │
  │                               │                               │
  │                               │  НЕТ ЗАПРОСА                   │
  │                               │                               │
  │                               │           НО ЭНДПОИНТ ЕСТЬ:    │
  │                               │           server.py:3900      │
  │                               │           POST /api/match/    │
  │                               │           friendly            │
  │                               │           ┌────────────────┐  │
  │                               │           │ user_id         │  │
  │                               │           │ from_user_id    │  │
  │                               │           └────────────────┘  │
  │                               │           → BattleEngine(      │
  │                               │             game_mode=        │
  │                               │             "friendly")       │
  │                               │           → нет трофеев       │
```

**Текущий статус (2026-06-25):**

- Фронтенд: вкладка «🤝 Дружеская игра» функциональна (`index.html:7034-7099`): список друзей онлайн (`/api/friends/list`), кнопка «В бой» (`index.html:7076-7079`), ввод ExtraID друга через `InviteInput` (`index.html:7094`). Может быть неактивна, если выключен feature-flag `friendly` в runtime-статусе или `matchModesMeta.friendly.enabled === false` (`index.html:6854`).
- Бэкенд: дружеский бой реализован через `/api/friends/invite*` (`server.py:17041-17045`): `invite`, `invite/status`, `invite/respond`, `invite/pending`, `invite/cancel`. **Отдельного `/api/match/friendly` POST-эндпоинта не существует** — в `app.router.add_post` зарегистрированы только `/api/match/find` и `/api/match/vs-bot` (`server.py:12429-12430`).
- `game_mode="friendly"` помещается в `match_game_modes` (`server.py:8821`); движок собирается через `_prepare_and_cache_engine` после `claim_friend_invite_accept`.
- **Нет трофеев** — режим `friendly` определён в `infrastructure/match_modes.py:99-104` с `NO_REWARDS`; `_process_battle_end` использует `mode_config.rewards.trophies` (`server.py:2857, 2864`).
- **Нет штрафов за surrender** — `server.py:3058`: `eligible_mode = str(game_mode or "").lower() not in ("training", "friendly")`.
- `IncomingInviteModal` (`index.html:13690`) — модалка входящего вызова; поллинг `/api/friends/invite/pending` каждые 3 сек (`index.html:17455`).

---

## Сводная таблица параметров BattleEngine

| Параметр | classic | extra_arena:blitz | extra_arena:draft | training | friendly |
|----------|---------|-------------------|-------------------|----------|----------|
| `turn_duration` | 25 сек | 5 сек | 25 сек | 25 сек | 25 сек |
| `mana_per_turn` | +1 | +2 | +1 | +1 | +1 |
| `hero_health_multiplier` | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 |
| `ruleset` | classic | classic | draft (не реализован) | classic | classic |
| Трофеи | ✅ | ✅ | ❌ (mode unavailable) | ❌ | ❌ |
| Штраф за surrender | ✅ | ✅ | ❌ | ❌ | ❌ |
| Противник | PvP / бот | PvP / бот | (бой не стартует) | Бот | Друг |
| Статус | ✅ | ✅ | ⚠️ `mode_unavailable` | ✅ | ✅ |

---

## Ключевые файлы

| Файл | Роль |
|------|------|
| `webapp/index.html:8074-8093` | Кнопка «В БОЙ» в ArenaScreen |
| `webapp/index.html:8094-8096` | Кнопка «спасательный круг» → BattlePickSheet |
| `webapp/index.html:14155-14428` | `GameModeSheet` — выбор режима/модификатора/колоды |
| `webapp/index.html:6771-7128` | `BattlePickSheet` — дружеский бой / тренировка |
| `webapp/index.html:6763-6769` | Константа `DIFFICULTIES` (5 уровней бота) |
| `webapp/index.html:17565` | Колбэк `onConfirm` → `window.startBattle()` |
| `webapp/index.html:2284-2383` | `window.startBattle()` — PvP поиск |
| `webapp/index.html:2386-2455` | `window.startVsBot()` — тренировка vs бот |
| `webapp/index.html:14799-14990` | `PreBattleScreen` — VS-экран перед боем |
| `webapp/index.html:13690` | `IncomingInviteModal` — входящий дружеский вызов |
| `webapp/arena.js:3191-3199` | arena.js — извлечение `?id`/`?_auth` из URL (инициализация матча) |
| `web/server.py:10522-10669` | `match_find_handler` — обработчик поиска PvP (`POST /api/match/find`) |
| `web/server.py:10671-10810` | `match_vs_bot_handler` — обработчик тренировки (`POST /api/match/vs-bot`) |
| `web/server.py:9827-10390` | `_prepare_and_cache_engine` — создание BattleEngine |
| `web/server.py:17041-17045` | `/api/friends/invite*` — эндпоинты дружеского боя (invite/respond/pending/cancel) |
| `web/server.py:12429-12430` | Регистрация `match_find` и `match_vs_bot` маршрутов |
| `web/server.py:2741-3054` | `_process_battle_end` — начисление трофеев (проверяет `mode_config.rewards`) |
| `web/server.py:2348-2411` | `join_match` — Socket.IO вход в комнату матча |
| `web/server.py:2447-2511` | `client_ready` — сигнал готовности, запуск бота |
| `battle_engine.py:74-122` | `BattleEngine.__init__` — `mode_config`, `turn_duration`, `game_mode` |
| `battle_engine.py:174-372` | `BattleEngine.create_match` — создание ArenaEnvironment |
| `battle_engine.py:346-349` | `ArenaEnvironment(game_state, classic_params=self.mode_config.classic)` |
| `core/engine.py:224-256` | `ArenaEnvironment.__init__` — `mana_per_turn` через `classic_params` |
| `infrastructure/match_modes.py:46-58` | `ModeConfig` — dataclass c `classic` + `rewards` |
| `infrastructure/match_modes.py:71-151` | `MODE_CONFIGS` — словарь всех режимов (blitz/draft/training/friendly/…) |
| `infrastructure/matchmaking.py` | `Matchmaker.find_match` / `_create_bot_match` |

---

## Audit (2026-06-25)

**Checked against current code:** `web/server.py`, `battle_engine.py`, `webapp/index.html`, `webapp/arena.js`, `core/engine.py`, `infrastructure/match_modes.py`, `infrastructure/matchmaking.py`.

**Fixes applied:**

- `docs/battle-entry-points.md:7-14` (overview table) — убрал «(в разработке)» для `friendly`; реализован через `/api/friends/invite*`.
- `docs/battle-entry-points.md:219-223` (blitz pipeline) — `_is_blitz` флаг больше не существует; параметры живут в `infrastructure/match_modes.py:ModeConfig`, а движок читает их через `self.mode_config.classic.turn_duration_seconds` и `ArenaEnvironment(..., classic_params=self.mode_config.classic)`. Обновил строки на актуальные.
- `docs/battle-entry-points.md:303-307` (training details) — строки `server.py:390/792/4632/4857` устарели. Reward-skip и surrender-skip теперь через `mode_config.rewards` (`match_modes.py:NO_REWARDS`, `server.py:2801-2803, 3058`) и константу `TRAINING_BOT_NAME` (`server.py:108`); сложности — `DIFFICULTIES` массив (`index.html:6763-6769`).
- `docs/battle-entry-points.md:348-354` (friendly status) — эндпоинта `POST /api/match/friendly` не существует; дружеский бой идёт через `/api/friends/invite*` (`server.py:17041-17045`). Фронтенд не «disabled», а функционален (`index.html:7034-7099`). Поллинг — каждые 3 сек (`index.html:17455`), не 5.
- `docs/battle-entry-points.md:358-368` (BattleEngine table) — убрал строку `_is_blitz`; добавил `hero_health_multiplier` и `ruleset` (откуда видно, что `extra_arena:draft` имеет ruleset `draft`, который не реализован → `ruleset_not_implemented`).
- `docs/battle-entry-points.md:374-403` (Ключевые файлы) — все номера строк обновлены под актуальные: `match_find_handler` (10522-10669), `match_vs_bot_handler` (10671-10810), `_prepare_and_cache_engine` (9827-10390), `_process_battle_end` (2741-3054), `join_match` (2348-2411), `client_ready` (2447-2511), `BattleEngine.__init__` (74-122), `create_match` (174-372), `ArenaEnvironment` (battle_engine.py:346-349, core/engine.py:224-256), `startBattle` (2284-2383), `startVsBot` (2386-2455), `PreBattleScreen` (14799-14990), `GameModeSheet` (14155-14428), `BattlePickSheet` (6771-7128), «В БОЙ» кнопка (8074-8093), «спасательный круг» (8094-8096), `arena.js` URL-парсинг (3191-3199), `IncomingInviteModal` (13690), `onConfirm` (17565). Добавил ссылки на `infrastructure/match_modes.py:ModeConfig` и `MODE_CONFIGS`.

**Unverified / оставлено как было:**

- Диаграмма ASCII-последовательностей в разделе «Карта последовательностей» (lines 19-140) и в детальных секциях — концептуально верна по шагам, но точные строки в коде могли сдвинуться; полная перерисовка диаграммы вне scope этого аудита (только текстовые правки).
- Имя файла `webapp/index.html` vs возможные бандлы (`webapp/index.compiled.js`) — диаграммы ссылаются на исходник `index.html`; бандл собирается через `precompile_webapp_index.py` и не влияет на структуру.
- Конкретные координаты пакетов, отправляемых в `openPreBattle` (lines 2300-2313 актуальны на момент аудита, но могут дрейфовать).
