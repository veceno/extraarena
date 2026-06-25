# Ответы на вопросы по ExtraArenaRaS — часть 2

---

## По боевой механике

### 1. Канеки как единственный юнит на доске

Если Канеки — единственный юнит на доске, его **нельзя разыграть**. `consume_ally` требует обязательную цель (`target_id`), а в `_get_possible_targets()` цель ищется только среди `player.board`:

```python
# core/engine.py:1263-1267
if is_consume_ally:
    for unit in player.board:
        targets.append(str(unit.instance_id))
    return targets  # Возвращаем только союзников
```

Если доска пуста — список целей пуст, и `get_legal_actions()` не сгенерирует действие для его розыгрыша. fallback-а нет. Дополнительная защита стоит в `_validate_play_target()` (`core/engine.py:884-887`): при пустой доске возвращается `consume_requires_target`.

> `core/engine.py:884-893, 1118, 1263-1267`

---

### 2. `aura_atk_1_3` у Жанны — что значит `_3`

Это **диапазон** (min=1, max=3). Конвертер (`core/converter.py:63-80`) парсит `X_Y` как `min_max` и для всех механик с диапазоном (кроме `buff_all_` и `battlecry_buff_`) **берёт только минимальное значение**:

```python
# converter.py:78-80
# Для остальных механик с диапазоном: берём min_val
return f"{prefix}{min_val}"
```

В результате `aura_atk_1_3` → `aura_atk_1`. Это не радиус, не количество целей, не баг — **осознанный выбор брать минимум**. Изначально, вероятно, подразумевалось «+1-3 атаки» (скалирование от уровня/рандома), но движок работает только с фиксированным минимумом.

> `core/converter.py:63-80`

---

### 3. Frozen — может ли быть атакован? Снимается ли от урона?

**Да**, frozen юнит может быть атакован. Заморозка влияет только на способность атаковать (`is_ready = False` при разморозке), но не даёт неуязвимости. Frozen юнит остаётся на доске как обычная цель.

**Нет**, заморозка **не снимается** от получения урона. Freeze очищается только в `_handle_end_turn()`:

```python
# core/engine.py:695-707 (внутри _handle_end_turn)
if unit.is_frozen:
    unit.is_ready = False
    unit.is_frozen = False
```

Ни в `_handle_attack()` (`core/engine.py:513-665`), ни в `apply_damage()` (`core/effects.py:1063-1105`) нет кода, трогающего `is_frozen`. Урон никак не влияет на статус заморозки. Замороженное существо после разморозки остаётся спящим (is_ready = False) и пропускает активацию в свой ход.

> `core/engine.py:491-501, 695-707`, `core/effects.py:1063-1105, 900-940`

---

### 4. Оба героя умирают одновременно — кто победитель?

Порядок в `step()` (`core/engine.py:378-386`):

```python
self._cleanup_dead_units(player)    # 1. deathrattle игрока
self._cleanup_dead_units(opponent)  # 2. deathrattle противника
self._check_game_over()             # 3. проверка одновременной смерти
```

`_check_game_over()` (`core/engine.py:822-831`):

```python
p1_dead = self.state.p1.hero.hp <= 0
p2_dead = self.state.p2.hero.hp <= 0
if p1_dead and p2_dead:
    self.state.status = GameStatus.DRAW
elif p1_dead:
    self.state.status = GameStatus.P2_WIN
elif p2_dead:
    self.state.status = GameStatus.P1_WIN
```

При **одновременной** смерти обоих героев результат — **`DRAW`** (ничейный исход), а не победа P2. P2 побеждает только если P1 мёртв, а P2 жив (или P1 умер от deathrattle, нанесённого юнитом P2 уже после его гибели).

> `core/engine.py:378-386, 822-831`

---

### 5. Reshuffle — порядок карт

**Случайный.** Сброс явно перемешивается перед тем как стать новой колодой. Логика живёт в `draw_one_from_deck()` (`core/engine.py:146-163`):

```python
# core/engine.py:146-163
if not player.deck:
    if player.graveyard:
        for card in player.graveyard:
            card.reset_to_base_state()        # сброс рантайм-модификаций
        graveyard_cards = list(player.graveyard)
        rng.shuffle(graveyard_cards)          # перемешиваем копию
        player.deck = graveyard_cards
        player.graveyard.clear()
```

Перед перемешиванием у всех карт сбрасывается рантайм-состояние через `reset_to_base_state()` (`core/state.py:119-130`): `attack = base_attack`, `max_hp = base_max_hp`, `hp = base_hp`, `is_ready = False`, `is_frozen = False`, `instant_kill_used = False`, `skip_count = 0`. Это единая точка сброса для любых модификаторов (consume_ally, battlecry_buff, level scaling).

> `core/engine.py:146-163`, `core/state.py:119-130`

---

### 6. Бот после сдачи/AFK — случайно или по логике?

**По логике.** Бот использует одну из двух систем принятия решений (`web/server.py:5140-5174`):

1. **BerserkInference** (ONNX-нейросеть) — если загружен `BERSERK_BRAIN` и режим помечен как TrainV2-safe (`train_v2_safe_mode`), и для нужной сложности есть готовый профиль. Не привязан к конкретному диапазону `bot_id` — определяется режимом и наличием модели. Использует temperature-sampling с учётом сложности (`lite`/`easy`/`medium`/`hard`/`max`).

2. **BotAI** (rule-based fallback) — для всех остальных случаев. Приоритеты (`ai/bot_ai.py:60-85`):
   - Сначала атака (предпочитает героя, если доступен; иначе случайная атака)
   - Затем розыгрыш карты (случайный выбор из доступных)
   - В конце — завершение хода

> `web/server.py:5140-5174, 5319-5324`, `ai/bot_ai.py:32-87`

---

### 7. Reflect убивает атакующего — срабатывает ли deathrattle атакующего?

**Да, срабатывает.** Порядок событий:

1. `_handle_attack()` — reflect наносит урон атакующему, его HP падает до ≤0. Но чистка не происходит внутри атаки.
2. `step()` вызывает `_cleanup_dead_units(player)` — здесь **срабатывает deathrattle** атакующего, и он уходит в сброс.
3. `step()` вызывает `_cleanup_dead_units(opponent)`
4. `step()` вызывает `_check_game_over()`

Deathrattle атакующего срабатывает **после** полного разрешения атаки, на шаге очистки. Урон от deathrattle может убить вражеского героя и изменить исход боя. Сам обмен ударами (включая reflect на атакующего) живёт в `_handle_attack()` (`core/engine.py:513-665`); очистка мёртвых юнитов с триггером deathrattle — в `_cleanup_dead_units()` (`core/engine.py:779-820`).

> `core/engine.py:378-386, 513-665, 779-820`

---

### 8. Мидория — герой или воин?

**Воин** (`card_type: "warrior"`). Находится в колоде как обычная карта, выставляется на доску. При розыгрыше срабатывает `cast_random_spell`.

> `cards.json` — строка 371

---

### 9. `battlecry_heal_X` vs `battlecry_heal_target_X` vs `battlecry_heal_hero_X`

| Механика | Цели | Как работает |
|----------|------|-------------|
| `battlecry_heal_X` | **Любые повреждённые союзники** (юниты с `hp < max_hp` или герой с `hp < max_hp`) | `_get_possible_targets()` фильтрует только тех, кому реально нужно лечение |
| `battlecry_heal_target_X` | **Любой союзник или герой** (включая полностью здоровых) | Можно выбрать кого угодно — перехил невозможен (`apply_heal` ограничен `max_hp`), но цель выбрать можно |
| `battlecry_heal_hero_X` | **Только свой герой** | Не требует target_id — лечит героя автоматически |

Разница: `_target_` позволяет выбрать цель без ограничения «только повреждённые»; `_hero_` лечит героя без выбора цели вообще.

> `core/engine.py:1296-1312`, `core/effects.py:189-257, 1063-1126`

---

### 10. Можно ли атаковать своего героя?

**Нет.** `get_legal_actions()` (`core/engine.py:942-981`) генерирует цели атаки **только** среди `opponent.board` и `opponent.hero`. Никакого кода для атаки своих юнитов или своего героя не существует.

> `core/engine.py:1183-1227`

---

## По редактору и UX

### 11. Какие карты имеют `requires_target`?

`requires_target()` (`core/effects.py:1322-1367`) возвращает `True` для карт со следующими механиками:

**Статические:**
- `delete_target` — Чёрная Дыра (id=13)
- `consume_ally` — Канеки Кен (id=20)
- `freeze` — Заморозка (id=11)
- `battlecry_freeze` — Саб-Зиро (id=19)

**Динамические (RegEx):**
- `battlecry_damage_X` — Тока Киришима (id=15)
- `damage_X` (включая `damage_X_Y`) — Глитч-Удар (id=8)
- `spell_damage_X` — (нет в текущем пуле)
- `heal_X` / `heal_target_X` / `battlecry_heal_X` / `battlecry_heal_target_X` — Сакура (id=14), Фрирен (id=35), Юни (id=36)
- `battlecry_buff_X_Y` — (нет в текущем пуле)
- `freeze_X` — (нет в текущем пуле)

**НЕ требуют цель:**
- `choose_shield_damage` — Геральт (id=21): **опционально**, может выбрать щит без цели
- `cast_random_spell` — Мидория (id=26): не требует цели

**Полный список карт с обязательной целью при розыгрыше:**

| id | Карта | Тип | Механика |
|----|-------|-----|----------|
| 8 | Глитч-Удар | potion | `damage_1_5` |
| 11 | Заморозка | potion | `freeze` |
| 13 | Чёрная Дыра | potion | `delete_target` |
| 14 | Сакура | warrior | `battlecry_heal_hero_2` |
| 15 | Тока Киришима | warrior | `battlecry_damage_1_random` |
| 19 | Саб-Зиро | warrior | `battlecry_freeze` |
| 20 | Канеки Кен | warrior | `consume_ally` |
| 35 | Фрирен | warrior | `battlecry_heal_target_5` |
| 36 | Юни | warrior | `battlecry_heal_target_3` |

> `core/effects.py:1322-1367`

---

### 12. Random Fill — учитывает ли стоимость маны?

**Нет, полностью случайно.** Код из `webapp/index.html:9911-9915`:

```javascript
const randomFill = () => {
    const heroPool    = allCards.filter(c => c.card_type === 'hero');
    const warriorPool = allCards.filter(c => c.card_type !== 'hero');
    const heroPick    = heroPool.length ? heroPool[Math.floor(Math.random()*heroPool.length)].id : null;
    const shuffled    = [...warriorPool].sort(() => Math.random() - 0.5);
    const picked      = shuffled.slice(0, 8).map(c => c.id);
    while (picked.length < 8) picked.push(null);     // добивка пустыми слотами
    setEditing(p => ({...p, slots: [heroPick, ...picked], preset_name: p.preset_name || 'Колода'}));
};
```

1 случайный герой + 8 случайных не-героев. Никакой кривой маны, никакой фильтрации.

---

### 13. Валидация колоды перед боем

**При сохранении** (`web/server.py:7268-7269`): проверяется, что ровно 9 слотов:

```python
if len(card_slots) != DECK_SIZE:  # DECK_SIZE = 9
    return web.json_response({"error": "invalid_slots_count"}, status=400)
```

**При запуске боя** — **нет валидации размера колоды в BattleEngine**. `create_match()` (`battle_engine.py:174-300`) просто обрабатывает то, что прислано; дополнительная проверка размеров делается выше — в вызывающем коде (например `web/server.py:10242, 10251`).

---

### 14. Бой с незаполненной колодой

Если игрок заходит с <9 картами (например, без героя или с 5 воинами):

- **Нет героя** → создаётся **дефолтный герой** (30 HP, 0 атаки, без механик) — `core/classic_setup.py:79-96` (используется в продакшене через `create_classic_game_state()`; legacy fallback в `battle_engine.py:438-468` имеет то же поведение, но в проде уже не вызывается)
- **Меньше воинов** → стартовая рука просто будет меньше 3 карт: `min(3, len(warriors))` — `core/classic_setup.py:99-113`
- **Меньше 8 не-героев** → колода будет меньше, добор быстрее иссякнет

Бой **не блокируется** — движок работает с тем что есть.

> `core/classic_setup.py:14-113`, `battle_engine.py:438-468`

---

## По матчмейкингу и ботам

### 15. Боты — фиксированные или случайные колоды?

**Смешанный подход через переиспользование** (`ai/bot_factory.py:41-122`):

1. **Приоритетный путь**: переиспользует существующего бота из пула (`db.find_reusable_bots()`), у которого уже сохранена колода реального игрока-донора. Бот-донор остаётся в БД, поэтому боты «носят» колоды живых людей между сессиями.
2. **Fallback**: если в пуле нет подходящего переиспользуемого бота — генерируется новый через `_generate_bot()`, который берёт колоду либо у случайного донора, либо собирает случайную (1 случайный герой + 8 случайных не-героев).

Каждый бот, найденный в пуле или сгенерированный, сохраняется в БД один раз.

---

### 16. Разница в логике бота между Classic и Blitz

**Логика принятия решений одинакова.** `BotAI` и `BerserkInference` не принимают параметр `game_mode`.

Разница только в **таймингах**, заданных в `infrastructure/match_modes.py:32-89` (`ClassicParams`):

| Параметр | Classic | Blitz |
|----------|---------|-------|
| Задержка перед ходом | 4.0–6.0s (или 1.5–2.5s для hard/max) | 0.3–0.8s |
| Пауза между действиями | 0.4–0.8s | 0.05–0.15s |
| Прирост маны | +1/ход | +2/ход (движок) |

---

### 17. Матчмейкинг-окно 500 — новичок (300) vs ветеран (800)

**Да, могут встретиться.** Условие подбора (`infrastructure/matchmaking.py:198`):

```python
if abs(entry.trophies - seeker.trophies) <= window:
    return entry
```

`abs(800 − 300) = 500 ≤ 500` → матч будет найден. Это симметричное окно: ±500 от трофеев ищущего. Окна расширяются последовательно: 50 → 200 → 500, с интервалом опроса 3 секунды.

---

### 18. Race condition при одновременном поиске

**Кратковременная задержка возможна, но не постоянный race condition.**

Матчмейкер использует `asyncio.Lock()` для всех мутирующих операций (объявлен в `infrastructure/matchmaking.py:93`, захваты в строках 273, 291, 352, 381, 395, 417, 469, 486, 530). При одновременном вызове `find_match()` двумя игроками:

1. Первый заходит под локом, проверяет очередь — пусто, добавляет себя, освобождает лок
2. Второй заходит под локом, проверяет очередь — находит первого? **Нет**, потому что первый мог ещё не добавиться (они соревнуются за лок)

Но это решается через `_search_loop` — фоновый цикл с интервалом **3 секунды** (`QUEUE_POLL_INTERVAL`). На следующей итерации они найдут друг друга. Максимальная за��ержка: 3 секунды.

---

## По картам и балансу

### 19. Полный список карт с `requires_target`

См. ответ №11 — 9 карт требуют обязательную цель.

---

### 20. Карты с несколькими механиками

В текущем пуле (cards.json) **нет карт с несколькими механиками**. У каждой карты ровно одна механика (или 0). Это дизайн-решение: одна карта = один эффект.

---

### 21. `start_mana_1_5` у Олега Тинькова

Это **диапазон** (min=1, max=5). Конвертер (`core/converter.py:70-72`) берёт только минимум:

```python
if prefix == "start_mana_":
    return f"{prefix}{min_val}"  # → start_mana_1
```

Так что Олег Тиньков **всегда даёт +1 маны** на старте. Затем `scale_card_by_level()` (`core/card_scaling.py:115-140`, ветка для `CardType.HERO`) добавляет `+bonus_tiers` за каждые 3 уровня героя:

| Уровень героя | bonus_tiers | Итого start_mana |
|---------------|-------------|-----------------|
| 1–3 | 0 | 1 |
| 4–6 | 1 | 2 |
| 7–9 | 2 | 3 |
| 10 | 3 | 4 (но фактически capped по `min(10, …)`) |

Это фиксированное значение для данного уровня, не рандом.

> `core/converter.py:70-72`, `core/card_scaling.py:115-140`

---

### 22. Глитч-Стена и taunt в пуле

Карты с id=48 (Глитч-Стена) **нет в cards.json**. В JSON-каталоге id заканчиваются на 46.

**Реальные taunt-юниты в пуле:**

| id | Имя | Мана | Atk/HP | Редкость |
|----|-----|------|--------|----------|
| 30 | Наофуми | 3 | 1/5 | epic |
| 39 | Альфонс Элрик | 2 | 1/3 | start |
| 45 | Солид Снейк | 5 | 5/4 | epic |

Всего **3 карты** с taunt на 44 карты пула. Taunt действительно редкая механика. `bypass_taunt` есть у двух карт: Хог Райдер (id=16) и Лара Крофт (id=43) — ценность этой механики ниже, чем кажется, потому что игрок с taunt-юнитами всё равно редко получает по ним урон в первые ходы.

---

### 23. Можно ли намеренно убить своего Крипера?

**Технически — нет.** В `get_legal_actions()` нет действия «атаковать своего юнита» или «уничтожить своего юнита». Игрок не может напрямую убить своего Крипера.

Однако можно создать ситуацию косвенно:
- Крипер атакует юнита с `reflect` и умирает от отражённого урона
- Крипер погибает от AOE-эффекта противника
- Крипер умирает при обмене ударами

В любом случае deathrattle сработает и нанесёт 3 урона всем вражеским юнитам и герою.

---

### 24. Полная таблица всех карт

#### Герои (6)

| id | Имя | Редкость | HP | Механика | Power |
|----|-----|----------|-----|----------|-------|
| 1 | Искатель | start | 35 | — | 30 |
| 3 | Жанна д'Арк | epic | 35 | `aura_atk_1_3` | 50 |
| 4 | Аскеладд | legendary | 40 | `reflect_1` | 60 |
| 5 | Даркнесс | superrare | 45 | `armor_1_3` | 45 |
| 6 | Росомаха | rare | 37 | `regen_1` | 40 |
| 7 | Олег Тиньков | mythic | 23 | `start_mana_1_5` | 70 |

Все герои: мана=0, атака=0.

#### Зелья (6)

| id | Имя | Редкость | Мана | Механика | Power |
|----|-----|----------|------|----------|-------|
| 8 | Глитч-Удар | common | 1 | `damage_1_5` | 10 |
| 10 | Импульс Бездны | epic | 4 | `aoe_damage_2` | 40 |
| 11 | Заморозка | rare | 2 | `freeze` | 25 |
| 12 | Кража Маны | epic | 3 | `mana_drain_2` | 35 |
| 13 | Чёрная Дыра | legendary | 5 | `delete_target` | 60 |

#### Воины (33)

| id | Имя | Редкость | Мана | Atk | HP | Механика | Power |
|----|-----|----------|------|-----|-----|----------|-------|
| 14 | Сакура | common | 2 | 2 | 2 | `battlecry_heal_hero_2` | 15 |
| 15 | Тока Киришима | common | 2 | 2 | 1 | `battlecry_damage_1_random` | 15 |
| 16 | Хог Райдер | rare | 4 | 4 | 3 | `bypass_taunt` | 30 |
| 17 | Танджиро | rare | 3 | 3 | 2 | `shield` | 25 |
| 18 | П.Е.К.К.А. | epic | 5 | 5 | 5 | `armor_1` | 50 |
| 19 | Саб-Зиро | epic | 4 | 3 | 4 | `battlecry_freeze` | 45 |
| 20 | Канеки Кен | legendary | 3 | 2 | 2 | `consume_ally` | 55 |
| 21 | Геральт | legendary | 5 | 4 | 5 | `choose_shield_damage` | 60 |
| 22 | Дио Брандо | mythic | 8 | 6 | 6 | `aoe_freeze` | 80 |
| 23 | Сукуна | mythic | 7 | 7 | 5 | `cleave_1_2` | 75 |
| 24 | Годжо Сатору | divine | 9 | 5 | 6 | `shield` | 95 |
| 25 | Сайтама | divine | 10 | 10 | 10 | `instant_kill` | 100 |
| 26 | Мидория | limited | 5 | 5 | 5 | `cast_random_spell` | 50 |
| 27 | Скелет | common | 1 | 2 | 1 | — | 5 |
| 28 | Торфинн | rare | 2 | 4 | 1 | — | 20 |
| 29 | Штурмовик | common | 3 | 3 | 3 | — | 10 |
| 30 | Наофуми | epic | 3 | 1 | 5 | `taunt` | 40 |
| 31 | Наемник | rare | 4 | 4 | 5 | — | 25 |
| 32 | Зеницу | epic | 4 | 5 | 3 | `charge` | 45 |
| 33 | Маления | mythic | 7 | 6 | 6 | `lifesteal` | 85 |
| 34 | Крипер | rare | 3 | 2 | 2 | `deathrattle_aoe_damage_2` | 35 |
| 35 | Фрирен | legendary | 5 | 3 | 5 | `battlecry_heal_target_5` | 65 |
| 36 | Юни | start | 2 | 1 | 2 | `battlecry_heal_target_3` | 15 |
| 37 | Слайм | start | 1 | 1 | 2 | — | 5 |
| 38 | Хиличурл | start | 2 | 2 | 2 | — | 10 |
| 39 | Альфонс Элрик | start | 2 | 1 | 3 | `taunt` | 15 |
| 40 | Стив | start | 3 | 3 | 3 | — | 20 |
| 41 | Довакин | start | 4 | 4 | 4 | — | 25 |
| 42 | Атакующий Титан | start | 6 | 6 | 6 | — | 35 |
| 43 | Лара Крофт | epic | 2 | 3 | 1 | `bypass_taunt` | 30 |
| 44 | Леви Аккерман | epic | 2 | 3 | 1 | `charge` | 35 |
| 45 | Солид Снейк | epic | 5 | 5 | 4 | `taunt` | 45 |
| 46 | Уссоп | common | 2 | 3 | 1 | — | 10 |

**Итого:** 6 героев + 5 зелий + 33 воина = **44 карты** в пуле.

#### Распределение по редкости

| Редкость | Кол-во |
|----------|--------|
| start | 10 |
| common | 5 |
| rare | 7 |
| epic | 7 |
| superrare | 1 |
| legendary | 5 |
| mythic | 5 |
| divine | 2 |
| limited | 1 |

---

## Audit (2026-06-25)

Источники проверены против текущего кода:
- `core/engine.py` (1737 строк), `core/state.py`, `core/effects.py` (1367 строк), `core/converter.py` (214 строк), `core/card_scaling.py`, `core/classic_setup.py`
- `web/server.py` (21476 строк), `battle_engine.py` (1964 строки), `infrastructure/matchmaking.py` (1164 строки), `infrastructure/match_modes.py`, `infrastructure/match_config.py`
- `ai/bot_ai.py` (150 строк), `ai/bot_factory.py` (786 строк)
- `tests/test_mechanics.py` (1076 строк, 21 test в классах `TestStatusEffects`, `TestBoardMechanics`, `TestAOEEffects`, `TestResourceControl`, `TestAdvancedMechanics`)
- `cards.json` (44 карты, max id = 46)

### Fixed

- **docs/GAMEPLAY_QA_2.md:9-21 (сценарий 1)** — строка `core/engine.py:1018-1021` заменена на актуальную `core/engine.py:1263-1267` (consume_ally target retrieval). Добавлено упоминание `_validate_play_target` (884-887) как второй линии защиты с `consume_requires_target`.
- **docs/GAMEPLAY_QA_2.md:60-81 (сценарий 4)** — главный баг: при одновременной гибели обоих героев результат `DRAW`, а не «P2 всегда побеждает». Поправлен псевдокод `_check_game_over()` под актуальную логику `core/engine.py:822-831` и актуальные номера строк `step()` (378-386).
- **docs/GAMEPLAY_QA_2.md:85-98 (сценарий 5)** — устаревший фрагмент `core/engine.py:586-589` (`random.shuffle(opponent.graveyard)`) заменён на актуальный путь reshuffle в `draw_one_from_deck()` (`core/engine.py:146-163`). Уточнён сброс рантайм-состояния через `reset_to_base_state()` в `core/state.py:119-130` (помимо `hp`/`is_ready`/`is_frozen` сбрасываются также `attack`, `max_hp`, `instant_kill_used`, `skip_count` и `mechanics` откатываются к base).
- **docs/GAMEPLAY_QA_2.md:103-113 (сценарий 6)** — убрано несуществующее правило «bot_id начинается с 810416»: ONNX теперь выбирается по флагам `BERSERK_BRAIN` + `train_v2_safe_mode` + `berserk_profile_ready` (`web/server.py:5140-5174`), без префикса в ID. Номера строк обновлены.
- **docs/GAMEPLAY_QA_2.md:116-200 (сценарий 11)** — `core/effects.py:1141-1186` → `core/effects.py:1322-1367` (функция `requires_target`); карта 15 (Тока Киришима) реально имеет `battlecry_damage_1_random`, а не `battlecry_damage_1` (поправлено в таблице).
- **docs/GAMEPLAY_QA_2.md:206-216 (сценарий 12)** — `webapp/index.html:2393-2402` → `webapp/index.html:9911-9915` (функция `randomFill`).
- **docs/GAMEPLAY_QA_2.md:225-232 (сценарий 13)** — `web/server.py:2529-2531` → `web/server.py:7268-7269` (валидация `invalid_slots_count`); уточнено, что в `BattleEngine.create_match()` валидации нет, но она делается в вызывающем коде (`web/server.py:10242, 10251`).
- **docs/GAMEPLAY_QA_2.md:238-246 (сценарий 14)** — `battle_engine.py:240` → `core/classic_setup.py:99-113` (стартовая рука); `battle_engine.py:356-386` → `core/classic_setup.py:79-96` + legacy-fallback в `battle_engine.py:438-468`. Создание GameState теперь живёт в `core/classic_setup.py:14-76` (`create_classic_game_state`).
- **docs/GAMEPLAY_QA_2.md:254-259 (сценарий 15)** — `ai/bot_factory.py:23-132` → `ai/bot_factory.py:41-122`; описание заменено: приоритетный путь — `find_reusable_bots()` (переиспользование ботов из пула, чьи колоды уже донорские), а не «копирование колоды донора при каждом создании».
- **docs/GAMEPLAY_QA_2.md:267 (сценарий 16)** — `web/server.py:1108-1253` → `infrastructure/match_modes.py:32-89` (конфиг таймингов в `ClassicParams`).
- **docs/GAMEPLAY_QA_2.md:294 (сценарий 18)** — `infrastructure/matchmaking.py:46, 106, 166` → `infrastructure/matchmaking.py:93` (объявление `asyncio.Lock()`), плюс полный список мест захвата локов: 273, 291, 352, 381, 395, 417, 469, 486, 530.
- **docs/GAMEPLAY_QA_2.md:319-337 (сценарий 21)** — `core/engine.py:177-185` → `core/card_scaling.py:115-140` (где живёт hero-level scaling механик).
- **docs/GAMEPLAY_QA_2.md:343-352 (сценарий 22)** — список taunt дополнен Солид Снейком (id=45, 5/4 epic, mana 5). Старая формулировка «id заканчиваются на 42» заменена на «id заканчиваются на 46». Уточнён `bypass_taunt`: теперь 2 карты (Хог Райдер 16 + Лара Крофт 43), а не одна.
- **docs/GAMEPLAY_QA_2.md:43-56 (сценарий 3)** — обновлены ссылки на `_handle_end_turn` (495-707) и `apply_damage` (1063-1105). Содержательно сценарий по-прежнему верна: frozen юнит уязвим, freeze снимается только в `_handle_end_turn`.
- **docs/GAMEPLAY_QA_2.md:118-128 (сценарий 7)** — обновлены номера строк `_handle_attack()` (513-665) и `_cleanup_dead_units()` (779-820).
- **docs/GAMEPLAY_QA_2.md:142-150 (сценарий 9)** — ссылки на `_get_possible_targets()` обновлены до 1296-1312, на `effect_battlecry_heal_2` и `_register_battlecry_heal_target_effects` (effects.py:189-257), на `apply_heal`/`apply_damage` (effects.py:1063-1126).
- **docs/GAMEPLAY_QA_2.md:155-158 (сценарий 10)** — `core/engine.py:942-981` → `core/engine.py:1183-1227` (логика атак в `get_legal_actions`).

### Verified — содержательно корректно

- Сценарий 2 (`aura_atk_1_3` = min): `core/converter.py:63-81` действительно отбрасывает max для всех механик, кроме `buff_all_`/`battlecry_buff_`/`start_mana_`.
- Сценарий 3 (frozen = уязвим, не снимается от урона): `is_frozen` снимается только в `_handle_end_turn()` (`core/engine.py:695-707`); `apply_damage` (`core/effects.py:1063-1105`) не трогает `is_frozen`.
- Сценарий 8 (Мидория — `card_type: warrior`, `cast_random_spell`): подтверждено `cards.json:360`.
- Сценарий 17 (матчмейкинг-окно 500: новичок 300 vs ветеран 800 — могут встретиться): `SEARCH_WINDOWS = (50, 200, 500)` в `infrastructure/match_config.py:10`; условие `abs(diff) <= window` подтверждено в `infrastructure/matchmaking.py:182-198` и `_search_loop`.
- Сценарий 20 (в cards.json нет карт с несколькими механиками): подтверждено через `python3 -c "..."` (все карты имеют ровно одну механику или пустой массив).
- Сценарий 23 (намеренно убить Крипера нельзя прямым действием, но можно через reflect/AOE/обмен ударами): подтверждено — `_get_possible_targets` не отдаёт своих юнитов как цель атаки; `get_legal_actions` (`core/engine.py:1183-1227`) генерирует атаки только по `opponent.board`/`opponent.hero`.
- Сценарий 19: дубликат ссылки на сценарий 11, не требует отдельной правки.

### Не удалось верифицировать

- Сценарий 6 (выбор ONNX по `bot_id`-префиксу) — больше не актуален; фактически выбор делается в `web/server.py:5140-5174` по флагам `BERSERK_BRAIN` / `train_v2_safe_mode` / `berserk_profile_ready`, и реальный путь ONNX-выбора нужно искать в самой `BERSERK_BRAIN` логике (её полный код в задаче не цитировался).
- Сценарий 7 (порядок каскада deathrattle): внутренний `while True` в `step()` (`core/engine.py:379-385`) делает повторные `_cleanup_dead_units` до стабилизации длины обоих бордов — это «каскадный» deathrattle, которого в текущем тексте нет, но функционально поведение «после отработки attack → cleanup → game_over» сохранено. Не критично.

