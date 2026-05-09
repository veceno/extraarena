# Аудит готовности систем ExtraArenaRaS

---

## Боевое ядро

### Classic Battle Mode — ✅ READY

| Компонент | Файл | Готовность |
|-----------|------|-----------|
| Игровой цикл `step()` | `core/engine.py:220` | ✅ Полный |
| Розыгрыш карт (воины + зелья) | `core/engine.py:323-421` | ✅ |
| Атака (taunt, aura, обмен ударами) | `core/engine.py:423-525` | ✅ |
| Завершение хода (мана, добор, реген, решафл) | `core/engine.py:527-609` | ✅ |
| Чистка мёртвых + deathrattle | `core/engine.py:611-651` | ✅ |
| Проверка победы | `core/engine.py:653-658` | ✅ |
| Legal actions (для UI/RL) | `core/engine.py:839-986` | ✅ |
| Preview delta (предпросмотр HP) | `core/engine.py:785-837` | ✅ |
| Создание матча (колоды, руки, герои) | `battle_engine.py:118-306` | ✅ |
| Сдача / AFK-детекция (2 таймаута) | `battle_engine.py:809-881` | ✅ |
| Награды за бой (трофеи, монеты) | `web/server.py:259-561` | ✅ |
| Тесты (щит, броня, фриз, таунт, батлкрай, ...) | `tests/test_mechanics.py` (1076 строк) | ✅ Все проходят |
| **Единственный TODO** | `core/effects.py:599` — получение summon-карты из БД | ⚠️ Минорно |

Известные нюансы (не критичные): burn при 4 картах в руке, P2 побеждает при одновременной смерти героев, нет валидации колоды при входе в бой (есть fallback).

---

### Blitz Mode — ✅ READY

| Фича | Classic | Blitz | Файл |
|-------|---------|-------|------|
| Таймер хода | 25 сек | 5 сек | `battle_engine.py:86` |
| Прирост маны | +1/ход | +2/ход | `battle_engine.py:291` |
| HP героев | 100% | 50% | `battle_engine.py:191-195` |
| Задержка ботов | 4-6 сек | 0.3-0.8 сек | `web/server.py:1108-1113` |
| Предбоевой экран | ✅ | ✅ с бейджем ⚡ | `webapp/index.html:3142` |

**Draft-режим** — есть в UI как опция `extra_arena:draft`, но backend-логики не имеет. Это чистый placeholder.

---

## Матчмейкинг — ⚠️ PARTIAL

### Что работает
- Поиск по трофеям с расширяющимся окном: 50 → 200 → 500
- Soft-start: игроки с <300 трофеев получают ботов мгновенно
- Таймаут 15 сек → fallback на бота
- `asyncio.Lock()` защищает от race condition
- Полный пайплайн: find_match → pair_players → create_match → BattleEngine
- Отмена старой очереди при повторном входе (`_drop_existing`)

### Критическая проблема
**Очередь и кэш матчей — полностью in-memory** (`matchmaking.py:44-48`). При рестарте сервера:
- Все игроки в очереди теряют своё место
- Все активные матчи теряются
- Нет персистентности в БД
- Нет cron-очистки orphan-записей

> `matchmaking.py:29-48`, `battle_engine.py:18`, `web/server.py:41`

---

## Редактор колод — ✅ READY

| Endpoint | Файл |
|----------|------|
| `GET /api/deck/presets` | `web/server.py:2442` |
| `POST /api/deck/presets/save` | `web/server.py:2490` |
| `POST /api/deck/presets/delete` | `web/server.py:2630` |
| `POST /api/deck/presets/rename` | `web/server.py:2693` |
| `POST /api/deck/presets/set-primary` | `web/server.py:2758` |

- Валидация: ровно 9 слотов при сохранении
- 5 пресетов (3 бесплатных, 2 по ExtraPass)
- Random Fill, выбор героя, drag-and-drop слотов
- Интеграция с матчмейкингом: колода загружается в `create_match()`

---

## Glory Path (Трофейная дорога) — ⚠️ PARTIAL

### Что есть
- 19 рубежей от 150 до 10 000 трофеев (`webapp/index.html:526-546`)
- Награды: монеты, гемы, кейсы T1-T5, гарантированные карты, частицы
- 10 лиг (Bronze → Grandmaster) с диапазонами трофеев
- Визуальный компонент `TrophyRoadSheet` с прогресс-баром, статусами (completed/next/locked), анимацией
- `LeagueInfoSheet` — таблица всех лиг

### Чего нет
- **Нет бэкенда для claiming наград** — рубежи только показывают что будет, не выдают
- Нет таблицы `claimed_milestones` в БД
- Нет `POST /api/glory/claim` эндпоинта

---

## Профиль игрока — ✅ READY

### API: `GET /api/profile` → 33 поля (`web/server.py:1528-1659`)

```json
{
  "user_id", "username", "first_name", "photo_url",
  "extra_pass", "trophies", "max_trophies", "league",
  "keys", "gems", "coins", "squad_id",
  "status", "reg_date", "stars", "energy", "energy_cd",
  "season", "title", "img", "selected_hero_id",
  "custom_nickname", "nickname_changed",
  "settings": { /* 9 флагов уведомлений + welcome_shown + ... */ },
  "should_show_welcome"
}
```

### Схема БД
- **users**: 19 полей (`database.py:952-974`)
- **profiles**: img, title, custom_nickname, nickname_changed (`database.py:1088-1097`)
- **user_settings**: 13 полей (`database.py:1134-1150`)

### Чего не хватает в UI
- Статистика боёв (всего побед/поражений/winrate)
- Любимая карта
- Количество открытых кейсов

---

## Кейсы + Tap Upgrade — ✅ READY (BE) + ✅ READY (анимация)

| Слой | Что | Файл |
|------|-----|------|
| Конфиг дроп-рейтов (T1-T5) | `case_config.py:35-73` | ✅ |
| Tap Upgrade (4 тапа, 25/20/15/10%) | `case_config.py:136-141` | ✅ |
| Генерация наград (карты + монеты) | `case_system.py:236-334` | ✅ |
| Частицы за дубликаты (база × множитель) | `case_system.py:171-193` | ✅ |
| Открытие кейса (4 тапа) | `case_system.py:337-408` | ✅ |
| Анимация кейса (React, 4 фазы) | `webapp/index.html:1090-1264` | ✅ |
| Звуки тапов и открытия | `webapp/index.html:13-24` | ✅ |
| T5 jackpot (500 частиц за common) | `case_config.py:133` | ✅ |

Косметическая анимация тапов (фронтенд) использует фикс. шанс 28% для визуала, реальный апгрейд — на сервере.

---

## OpenAI (AI Боты) — ✅ READY

### Две системы

| Система | Файл | Как работает |
|---------|------|-------------|
| **BotAI (rule-based)** | `ai/bot_ai.py` | Приоритет: атака героя → атака юнита → игра карты → конец хода |
| **BerserkInference (ONNX)** | `ai/bot_brain.py` | Нейросеть «Мидория v3»: 997 признаков, 200 действий, temperature sampling |

### Уровни сложности: 5

`lite`, `easy`, `medium`, `hard`, `max` — с разными temperature (0.1–1.8) и задержками

### Генерация колод (`ai/bot_factory.py`)
1. Копирует колоду реального игрока-донора
2. Fallback: случайный герой + 8 случайных карт

Авто-активация ONNX для ботов с ID `810416*` (`web/server.py:1130`).

---

## Магазин — ⚠️ PARTIAL

### Бэкенд — ✅

| Endpoint | Файл |
|----------|------|
| `POST /api/shop/buy` — покупка за гемы | `web/server.py:6221` |
| `GET /api/shop/sets` — наборы из БД | `web/server.py:6911` |
| Админские CRUD для наборов (create/update/delete) | `web/server.py:6912-6916` |
| Telegram Stars-платежи (`create_stars_invoice`) | `web/server.py:6026` |
| Обработка успешного платежа | `bot/handlers.py:546-634` |

Поддерживаемые покупки: `case`, `case_tier_N`, `coins_N`, `keys_N`, `shop_set_N`.

### Фронтенд — ⚠️ Placeholder

`ShopScreen` (`webapp/index.html:1036-1080`) показывает 4 захардкоженных товара:

| Товар | Цена |
|-------|------|
| 100 гемов | 49 ⭐ |
| 500 монет | 29 ⭐ |
| 3 ключа | 99 ⭐ |
| ExtraPass | 199 ⭐ |

Не подключён к `/api/shop/sets` — товары не загружаются из БД динамически.

---

## Уведомления (Telegram-бот) — ⚠️ PARTIAL

### Что работает

| Тип | Триггер | Файл |
|-----|---------|------|
| Dice ready | Cooldown истёк → сообщение в Telegram | `bot/handlers.py:354-378` |
| Платёж прошёл | `SuccessfulPayment` → награды | `bot/handlers.py:546-634` |

### Инфраструктура готова, но не активирована

В БД есть 7 флагов уведомлений (`user_settings`):
`notif_cases`, `notif_daily_rewards`, `notif_game_invites`, `notif_friend_requests`, `notif_events`, `notif_news`, `notif_dice`

Из них реально работает только **`notif_dice`**. Остальные 6 — только переключатели в UI без логики отправки.

### In-app уведомления
- Toast: `showNotification()` (`main.js:5036`) — 4 типа, авто-скрытие 5 сек
- Alert: `showGameAlert()` (`main.js:5071`)
- Confirm: `showGameConfirm()` (`main.js:5118`)

---

## Community — ⚠️ PARTIAL

### Бэкенд — ✅

| Endpoint | Файл |
|----------|------|
| `GET /api/community/posts` | `web/server.py:3089` |
| `POST /api/community/posts/create` (admin) | `web/server.py:3233` |
| `POST /api/community/posts/like` | `web/server.py:6014` |
| `GET /api/community/chat/messages` | `web/server.py:3303` |
| `POST /api/community/chat/send` | `web/server.py:3420` |

Таблицы: `global_chat`, `community_posts`, `post_likes`.

### Фронтенд — ❌ Placeholder

Вкладка «Коммьюнити» рендерит `PlaceholderScreen`:

> «Скоро здесь появятся возможности для общения и клубы»

Чат и посты не отображаются, несмотря на готовый бэкенд. Друзья: «0 онлайн» (заглушка).

---

## Системы, отсутствующие полностью

| Система | В коде | Что есть |
|---------|--------|----------|
| **Клановая система** | ❌ | Только `users.squad_id = 0`; нет таблицы `squads`, нет API |
| **Battle Pass** | ❌ | UI-заглушки (`main.js:731-749`); ExtraPass — отдельный премиум, не пропуск |
| **Friendly matches** | ❌ | UI placeholder с disabled-кнопкой («в разработке») |
| **Daily rewards** | ❌ | Только переключатель `notif_daily_rewards` в настройках |
| **Power Race** | ❌ | Ноль упоминаний в коде |
| **Рефералы** | ❌ | Ноль упоминаний в коде |

---

## Системы, частично реализованные

| Система | Готовность | Комментарий |
|---------|-----------|-------------|
| **Косметика / предметы** | ⚠️ Partial | Админские API для создания items (`server.py:2320-2423`); нет магазина/инвентаря для игроков |
| **ExtraID** | ⚠️ Partial | Telegram ID как аккаунт; кнопка «Копировать» в меню; мокап `extraid-mockup.html` не интегрирован |
| **Support** | ⚠️ Partial | Глобальный чат работает; отдельной системы тикетов/FAQ/поддержки нет |

---

## Полностью готовые системы

| Система | Файлы |
|---------|-------|
| **Mail/Inbox** | `database.py:1735-1791`, `server.py:5841-5999` — таблица, 3 API, покупки → письма, welcome-бонусы |
| **Promocode** | `database.py:1652-2355`, `server.py:1969-2122` — создание (admin), хранение, валидация, погашение, UI |
| **Tutorial/Onboarding** | `server.py:7026-7216`, `main.js:3219` — 3 шага (карта → обучение → подарок), создание пользователя с бонусами |
| **Dice** | `server.py:6918-7102`, `database.py:1610-1650` — cooldown, бросок, Telegram-уведомление о готовности |
| **Admin Panel** | 16+ админских эндпоинтов (`server.py:1776-5367`): игроки, карты, предметы, промокоды, shop sets, статистика |

---

## Итоговая сводка

| # | Система | Статус |
|---|---------|--------|
| 1 | Classic Battle | ✅ READY |
| 2 | Blitz Mode | ✅ READY |
| 3 | Draft Mode | ❌ UI placeholder |
| 4 | Matchmaking | ⚠️ In-memory (теряется при рестарте) |
| 5 | Deck Editor | ✅ READY |
| 6 | Glory Path | ⚠️ Нет бэкенда для claiming |
| 7 | Player Profile | ✅ READY |
| 8 | Cases + Tap Upgrade | ✅ READY |
| 9 | AI Bots (5 сложностей) | ✅ READY |
| 10 | Shop | ⚠️ Бэкенд ✅ / Фронтенд placeholder |
| 11 | Notifications | ⚠️ Только dice; 6 каналов dormant |
| 12 | Community | ⚠️ Бэкенд ✅ / Фронтенд placeholder |
| 13 | Clan/Squad | ❌ |
| 14 | Battle Pass | ❌ |
| 15 | Friendly Matches | ❌ |
| 16 | Daily Rewards | ❌ |
| 17 | Power Race | ❌ |
| 18 | Referrals | ❌ |
| 19 | Cosmetics/Items | ⚠️ Админский API без магазина |
| 20 | Mail/Inbox | ✅ READY |
| 21 | Promocode | ✅ READY |
| 22 | Tutorial/Onboarding | ✅ READY |
| 23 | Dice | ✅ READY (BE) |
| 24 | ExtraID | ⚠️ Мокап не интегрирован |
| 25 | Support | ⚠️ Только чат |
| 26 | Admin Panel | ✅ READY (BE) / ⚠️ Partial (FE) |

**Готовых к бете:** 10 систем. **Частично готовых:** 10 систем. **Отсутствуют:** 6 систем.
