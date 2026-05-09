# Точки входа в бой (Главное меню → Арена)

## Обзор

Игрок попадает на арену одним из следующих путей:

| # | Точка входа | game_mode | Противник | Награды |
|---|-------------|-----------|-----------|---------|
| 1 | «В БОЙ» → Классика | `classic` | PvP / бот | трофеи + кейсы |
| 2 | «В БОЙ» → ExtraArena: Blitz | `extra_arena:blitz` | PvP / бот | трофеи + кейсы |
| 3 | «В БОЙ» → ExtraArena: Draft | `extra_arena:draft` | PvP / бот | трофеи + кейсы (placeholder) |
| 4 | Круг → Тренировка | `training` | Бот (выбор сложности) | нет |
| 5 | Круг → Дружеский бой | `friendly` | Друг | нет (в разработке) |

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
| Frontend | `index.html:3553` | `fullMode = 'extra_arena:' + subMod` | Формирует строку |
| Server | `server.py:4693` | `game_mode = data.get("game_mode")` | Принимает как есть |
| BattleEngine | `battle_engine.py:85-87` | `self._is_blitz = (game_mode == "extra_arena:blitz")` | Только blitz активирует флаг |
| Engine | `battle_engine.py:293` | `mana_per_turn=2 if self._is_blitz else 1` | blitz → +2 маны |
| Engine | `battle_engine.py:87` | `self.turn_duration = 5 if self._is_blitz else 25` | blitz → 5 сек |

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

- **Нет трофеев** — `_process_battle_end()` (`server.py:390`): `if game_mode in ("training", "friendly"): return` (без начисления)
- **Нет штрафов за surrender** — `server.py:792`: `if game_mode in ("training", "friendly"):` — пропуск штрафных санкций
- **Сложность бота** передаётся через `bot_info["difficulty"]` (`server.py:4632`) → `bot_difficulty` в `BattleEngine`
- **5 уровней сложности**: `lite`, `easy`, `medium`, `hard`, `max` (`index.html:1740-1746`)
- **Имя противника**: `"🤖 Тренер"` (`server.py:4857`)

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

**Текущий статус:**

- Фронтенд: кнопка отправки вызова **disabled** (`index.html:1844`), предупреждение о разработке (`index.html:1842`)
- Бэкенд: эндпоинт `/api/match/friendly` существует и полностью готов (`server.py:3900-3972`)
- **Нет трофеев** — как и training, `_process_battle_end()` пропускает начисление
- **Нет штрафов за surrender** — как и training
- `IncomingInviteModal` (`index.html:4281`) — модалка входящего вызова, поллинг `/api/friends/invite/pending` каждые 5 сек (`index.html:4199-4213`)

---

## Сводная таблица параметров BattleEngine

| Параметр | classic | extra_arena:blitz | extra_arena:draft | training | friendly |
|----------|---------|-------------------|-------------------|----------|----------|
| `_is_blitz` | `False` | `True` | `False` | `False` | `False` |
| `turn_duration` | 25 сек | 5 сек | 25 сек | 25 сек | 25 сек |
| `mana_per_turn` | +1 | +2 | +1 | +1 | +1 |
| Трофеи | ✅ | ✅ | ✅ | ❌ | ❌ |
| Штраф за surrender | ✅ | ✅ | ✅ | ❌ | ❌ |
| Противник | PvP / бот | PvP / бот | PvP / бот | Бот | Друг |
| Статус | ✅ | ✅ | ⚠️ placeholder | ✅ | ⚠️ фронтенд заблокирован |

---

## Ключевые файлы

| Файл | Роль |
|------|------|
| `webapp/index.html:2175-2185` | Кнопка «В БОЙ» в ArenaScreen |
| `webapp/index.html:2186-2188` | Кнопка «спасательный круг» → BattlePickSheet |
| `webapp/index.html:3525-3672` | GameModeSheet — выбор режима/саб-модификатора/колоды |
| `webapp/index.html:1748-1876` | BattlePickSheet — тренировка / дружеский бой |
| `webapp/index.html:4271` | Колбэк onConfirm → `window.startBattle()` |
| `webapp/index.html:331-401` | `window.startBattle()` — PvP поиск |
| `webapp/index.html:404-444` | `window.startVsBot()` — тренировка vs бот |
| `webapp/index.html:3867-3999` | PreBattleScreen — VS-экран перед боем |
| `webapp/arena.js:111-137` | arena.js — инициализация, чтение `?id` и `?user_id` из URL |
| `web/server.py:4682-4793` | `match_find_handler` — обработчик поиска PvP |
| `web/server.py:4795-4874` | `match_vs_bot_handler` — обработчик тренировки |
| `web/server.py:4379-4678` | `_prepare_and_cache_engine` — создание BattleEngine |
| `web/server.py:3900-3972` | `/api/match/friendly` — эндпоинт дружеского боя |
| `web/server.py:349-415` | `_process_battle_end` — начисление трофеев (пропуск для training/friendly) |
| `web/server.py:144-183` | `join_match` — Socket.IO вход в комнату матча |
| `web/server.py:206-259` | `client_ready` — сигнал готовности, запуск бота |
| `battle_engine.py:50-113` | `BattleEngine.__init__` — `_is_blitz`, `turn_duration`, `game_mode` |
| `battle_engine.py:119-313` | `BattleEngine.create_match` — создание ArenaEnvironment |
| `battle_engine.py:293` | `ArenaEnvironment(game_state, mana_per_turn=2 if blitz else 1)` |
| `core/engine.py:207-216` | `ArenaEnvironment.__init__` — `mana_per_turn` |
| `infrastructure/matchmaking.py:50-135` | `Matchmaker.find_match` — поиск соперника |
| `infrastructure/matchmaking.py:270-393` | `Matchmaker._create_bot_match` — создание бота |
