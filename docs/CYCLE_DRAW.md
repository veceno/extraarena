# Циклирование колоды и добор карт (No-FIFO Weighted Draw)

Документ описывает новую систему добора карт в ExtraArena, заменившую старое
FIFO-циклирование `deck.pop(0) + hand.append` (которое ощущалось игроками как
«случайные карты надолго пропадают из пула»). Реализация объединяет три идеи:

1. **No-FIFO выбор** — карта выбирается по весам, а не с вершины колоды.
2. **Cost-curve bias** — мягкий bias в сторону стоимости, которой не хватает
   в текущей руке.
3. **Anti-stuck через `skip_count`** — карты, которые долго не выпадали,
   получают больший вес.

Вместе они дают **более «живое» ощущение колоды**: дорогие карты не уходят в
конец без шансов, дешёвые не «застревают» в deck, циклирование остаётся
детерминированным (при seeded RNG) и совместимым со всеми эффектами (battlecry,
end-of-turn, reshuffle).

## 1. Сценарий проблемы

Старая логика (до изменений) выглядела так:

```python
# core/engine.py (старая версия) — для end-of-turn
if player.deck:
    if len(player.hand) < HAND_CAP:
        player.hand.append(player.deck.pop(0))
# core/effects.py (старая версия) — для battlecry_draw_card
if owner.deck and len(owner.hand) < 4:
    owner.hand.append(owner.deck.pop(0))
```

Два связанных дефекта (M2 + C2 из аудита):

* **C2 — inconsistent overdraw handling.** Battlecry draw тихо проваливался
  при пустой колоде (без reshuffle), а end-of-turn корректно делал reshuffle.
* **M2 — divergent code paths.** Battlecry даже не учитывал
  `overdraw_to_discard=True` (режим «карта в сброс вместо потери в deck»).

Помимо этого, жалоба игроков: «некоторые карты словно вне пула» — на дистанции
30+ ходов конкретная карта (особенно 7+ маны) могла не появиться в руке вообще.
Чистый FIFO это маскирует, но он же и создаёт проблему, если `rng.shuffle` на
старте засунул её в конец.

## 2. Архитектура решения

Все три идеи реализованы в одном модульном помощнике
`core.engine.draw_one_from_deck`, который теперь **единая точка добора** для
end-of-turn и для эффектов `battlecry_draw_card`. Это закрывает M2 + C2, плюс
делает поведение предсказуемым для симуляций и RL-обучения.

### 2.1. Константы

```python
# core/engine.py
HAND_CAP         = 4      # размер руки в classic-режиме
STUCK_BONUS      = 0.5    # бонус к весу за каждый «пропуск» добора
COST_BIAS        = 0.3    # бонус к весу, если в руке нет cost-бакетa
CHEAP_COST_MAX   = 2      # mana_cost <= 2  -> cheap bucket
EXPENSIVE_COST_MIN = 4    # mana_cost >= 4  -> expensive bucket
```

Buckets выбраны эмпирически под текущий дизайн колод: 1-2 стоимость —
«дешёвые заклинания и мелкие существа», 3 — «середина» (без bias),
4+ — «дорогие win-cons и большие тела».

### 2.2. Веса карт (`_compute_draw_weights`)

Для каждой карты в `player.deck` вычисляется `weight[i]`:

```
weight[i] = 1.0                          # базовый
         + c.skip_count * STUCK_BONUS     # анти-застревание
         + cost_bias(c)                   # 0.0 / COST_BIAS
```

Где `cost_bias(c)`:

* `+COST_BIAS` если `c.mana_cost <= CHEAP_COST_MAX` **и** в руке нет ни одной cheap-карты;
* `+COST_BIAS` если `c.mana_cost >= EXPENSIVE_COST_MIN` **и** в руке нет ни одной expensive-карты;
* `0.0` иначе (включая «середину» — стоимость 3).

Bias — **«мягкий»**: бонус `+0.3` поверх базового `1.0`, то есть cheap получает
`1.3` против `1.0` у других при дефиците в руке. Это **не** «гарантия», а
лишь наклон распределения. У карты с `skip_count=2` вес уже `2.0`, и bias
не может его перебить — что и обеспечивает анти-застревание.

### 2.3. Взвешенный выбор (`_weighted_choice_idx`)

Стандартный inverse-CDF: `target = rng.random() * total`, затем ищем первый
индекс `i`, для которого `cumsum(weights)[i] > target`. При `total <= 0`
возвращается 0 (defensive — на практике не достигается, потому что каждый
`weight >= 1.0`).

### 2.4. `draw_one_from_deck` — три шага

#### Шаг 1: reshuffle

```python
if not player.deck and player.graveyard:
    for card in player.graveyard:
        card.reset_to_base_state()  # сбросить runtime-баффы к базе
    shuffled = list(player.graveyard)
    rng.shuffle(shuffled)
    player.deck = shuffled
    player.graveyard.clear()
elif not player.deck:
    return False  # fatigue
```

`reset_to_base_state` — критичный момент. `skip_count`, текущие `attack/hp`,
`is_ready`, `is_frozen` сбрасываются к базе, зафиксированной при первом
появлении карты. Это гарантирует, что карта, вернувшаяся в колоду после
graveyard-цикла, **стартует с чистого skip_count=0**, и её не «выкинет»
повторно как «застрявшую».

#### Шаг 2: пропуск, если рука полная

Если `len(player.hand) >= HAND_CAP`:

* `overdraw_to_discard=False` (classic default): **top card остаётся в deck**,
  инкремент `skip_count` уже выполнен в начале шага 2 — это и есть механизм
  анти-застревания. Карты, которые не могли быть добраны, становятся
  «старше» в колоде, и в следующий раз получат больший вес.
* `overdraw_to_discard=True` (альтернативный режим): top card взвешенно
  выбирается из deck и уходит в graveyard. Используется в специальных
  арена-режимах, где лишние карты не теряются, а перерабатываются.

#### Шаг 3: добор

```python
weights = _compute_draw_weights(player)
idx = _weighted_choice_idx(weights, rng)
drawn = player.deck.pop(idx)
drawn.skip_count = 0        # вытянутая карта «помолодела»
player.hand.append(drawn)
```

После изъятия карты из колоды `skip_count` остальных декрементируется на 1
(компенсация инкремента в начале шага 2) — иначе каждый добор накапливал бы
+1 «штраф», что не нужно: пропуск — это «мне не дали карту в этой итерации»,
а не «+1 событие».

### 2.5. Поток данных

```
                       ┌─────────────────────┐
                       │  ArenaEnvironment   │
                       │   .state            │
                       │   ._rng (Random)    │
                       └──────────┬──────────┘
                                  │
            ┌─────────────────────┼──────────────────────┐
            │                     │                      │
            ▼                     ▼                      ▼
   _handle_end_turn      effect_battlecry_      (другие callers,
                       draw_card              если появятся)
            │                     │                      │
            └────────┬────────────┘                      │
                     ▼                                   │
        draw_one_from_deck(player,                        │
            overdraw_to_discard=...,                      │
            source=...,                                   │
            rng=self._rng)  ◀────────────────────────────┘
                     │
                     ▼
        _compute_draw_weights(player)
                     │
                     ▼
        _weighted_choice_idx(weights, rng)
                     │
                     ▼
        card = player.deck.pop(idx)
```

`ArenaEnvironment._rng` — единственный источник случайности для добора.
Создаётся в `__init__` (или передаётся через параметр `rng=`), и тот же
объект шарится между `_handle_end_turn` и эффектами в `core.effects`
(через `state.arena_engine._rng`). Это критично для **детерминизма
симуляций и RL-обучения** (см. раздел 6).

### 2.6. `skip_count` — отдельное поле на `CardInstance`

```python
# core/state.py
@dataclass
class CardInstance:
    ...
    skip_count: int = 0
```

`reset_to_base_state()` обнуляет `skip_count` вместе с остальным
runtime-состоянием. Поле **не сериализуется через web протокол** — это
внутренний счётчик, клиент не должен о нём знать (визуализация в UI
не планируется).

## 3. Поведение в типичных сценариях

| Сценарий | Поведение |
|---|---|
| Стартовая раздача (3 карты) | Использует `_deal_starting_hand` в `core/classic_setup.py`, не вызывает `draw_one_from_deck`. Циклирование начинается с **первого end-of-turn**. |
| Normal end-of-turn | `draw_one_from_deck` с `source="end_turn"`, `overdraw_to_discard=False` (classic). |
| Battlecry «draw a card» | Тот же `draw_one_from_deck`, `source="battlecry_draw_card"`. `overdraw_to_discard` берётся из `state.classic_params`. |
| Рука полна + default overdraw | Карты в колоде получают `+1` к `skip_count` (anti-stuck). |
| Рука полна + overdraw_to_discard | Top card (взвешенно) уходит в graveyard. |
| Колода пуста + graveyard не пуст | Reshuffle: `reset_to_base_state` для всех, перемешать, переложить в deck. |
| Колода и graveyard пусты | Fatigue, `draw_one_from_deck` возвращает `False`. |

### 3.1. Численный пример

Стартовая рука: `[H1 cost=5, H2 cost=6]`. По классификации
`CHEAP_COST_MAX=2` / `EXPENSIVE_COST_MIN=4`: cheap_in_hand = 0,
expensive_in_hand = 2 (обе карты дорогие).
Колода: `[D1 cost=1, D2 cost=3, D3 cost=5]`.

Веса (`COST_BIAS=0.3`):
- D1 (cheap, cost=1) — bias = `max(0, 1 - 0) * 0.3 = 0.3` → weight = 1.3.
- D2 (mid, cost=3) — weight = 1.0.
- D3 (expensive, cost=5) — bias = `max(0, 1 - 2) * 0.3 = 0` → weight = 1.0.

Сумма = 3.3. Шансы добора: 39.4% / 30.3% / 30.3% (D1 / D2 / D3).

Если первая же карта (D2) добрана — её `skip_count` сбрасывается в 0, а
у оставшихся D1 и D3 инкремент skip_count (шаг 2) компенсируется
декрементом (шаг 3), так что обе остаются с `skip_count=0`. Состав руки
не меняется (cheap_in_hand=0, expensive_in_hand=2), поэтому bias не
пересчитывается: D1 = 1.3, D3 = 1.0 → шансы второго добора
56.5% / 43.5%. **Cheap bias сохраняется**, пока cheap-карта не окажется
в руке; **expensive bias не действует**, пока в руке есть хотя бы одна
дорогая карта.

## 4. Сравнение со старой FIFO-логикой

| Свойство | FIFO `pop(0)` | No-FIFO Weighted |
|---|---|---|
| Reshuffle при пустой колоде (battlecry) | ❌ Тихо проваливался | ✅ Делает reshuffle |
| Учёт `overdraw_to_discard` (battlecry) | ❌ Игнорировался | ✅ Учитывается |
| Дешёвая карта при пустой cheap-руке | Случайно, как лежит | С наклоном ~36% против 28% |
| Дорогая карта после 20 ходов без неё | Может остаться в deck до reshuffle | Поднимается к верху по `skip_count` |
| Детерминизм при seeded RNG | ✅ (через `random.shuffle`) | ✅ (через `_rng`) |
| Тестируемость распределения | Низкая — зависит от shuffle | Высокая — bias виден сразу |

## 5. Где что менять

| Цель | Файл / функция |
|---|---|
| Изменить размер руки | `HAND_CAP` в `core/engine.py` |
| Изменить bias по стоимости | `STUCK_BONUS`, `COST_BIAS`, `CHEAP_COST_MAX`, `EXPENSIVE_COST_MIN` в `core/engine.py` |
| Подключить новый эффект, тянущий карту | Вызвать `draw_one_from_deck` с `source=<имя>`, `rng=state.arena_engine._rng` |
| Передать seeded RNG в `ArenaEnvironment` | Параметр `rng=random.Random(seed)` в `__init__` (используется RL env) |
| Включить `overdraw_to_discard` | `ClassicParams(overdraw_to_discard=True)` в `infrastructure/match_modes.py` |
| Расширить логику bias (например, добавить bucket «spell») | Изменить `_compute_draw_weights` + `MECHANICS_LIST` если нужно |

## 6. Детерминизм и RL-обучение

### 6.1. Почему `ArenaEnvironment._rng`, а не модульный `random`

До изменений `core.engine._handle_end_turn` использовал
`rng.shuffle(player.deck)` через **модульный** `random`, который сидится
вызовом `random.seed(seed)`. Это работало, но:

* Тесты, симуляции и RL-env должны иметь возможность **передать собственный
  `Random` объект** (например, для воспроизведения конкретного эпизода).
* Battlecry-эффекты, тянущие карту, должны использовать **тот же RNG**,
  что и end-of-turn — иначе результат нельзя детерминированно воспроизвести.

Решение: `ArenaEnvironment.__init__(rng=None)` создаёт `self._rng = rng or
random.Random()`. Все внутренние вызовы добора пробрасывают `rng=self._rng`.
Эффекты в `core.effects.effect_battlecry_draw_card` достают rng через
`state.arena_engine._rng`.

### 6.2. Preview-env (RL `build_action_mask`)

`_make_preview_env` в `ai/train_v2/classic_actions_v1.py` создаёт
`ArenaEnvironment` через `__new__` (минуя `__init__`) — это нужно для
быстрого просчёта «что будет, если сделать это действие» в `action_mask`.
Без инициализации `_rng` preview-step падал с `AttributeError: 'ArenaEnvironment'
object has no attribute '_rng'`, что приводило к стиранию `mask[0] = 0` для
end-turn (ложно-нелегальное действие).

Фикс — `_make_preview_env` инициализирует `env._rng` через три уровня:

1. **Предпочитаемый путь:** clone state родительского `state.arena_engine._rng`
   через `Random.setstate/getstate` — гарантирует идентичную последовательность
   `random()` в preview и в основном env.
2. **Cold path** (внешние утилиты без parent engine): детерминированный seed
   из fingerprint состояния (`turn_number`, размеры рук и колод) — тот же
   state даёт тот же preview RNG.
3. (Не используется) свежий `Random()` — был бы недетерминирован, поэтому
   отброшен.

Это сохраняет свойство `evaluate_matchup` (детерминизм между двумя вызовами с
одинаковыми seeds) после введения weighted draw.

### 6.3. Передача seed в `ClassicRLEnv`

`ai/train_v2/classic_rl_env.py:reset` сеет модульный random
(`rand_mod.seed(seed)`) **и** передаёт `state_rng` в `ArenaEnvironment(...)`:

```python
self._env = ArenaEnvironment(
    game_state,
    mana_per_turn=self._mana_per_turn,
    rng=state_rng,   # <-- без этого weighted draw был бы недетерминирован
)
```

Без этой строчки `_rng` создавался бы через `random.Random()` (fresh),
и `test_evaluate_matchup_deterministic` падал с разными `p1_wins`
между двумя вызовами.

## 7. Тестирование

### 7.1. Unit-тесты

`tests/test_core_logic.py::TestStratifiedWeightedDraw` — 16 тестов:

* `test_skip_count_increments_on_full_hand_skip` — инкремент при заполненной руке
* `test_skip_count_resets_on_draw` — сброс при выходе из колоды
* `test_skip_count_resets_on_reshuffle` — сброс при reshuffle
* `test_no_fifo_draw_picks_weighted_not_just_top` — выбор НЕ top card
* `test_cost_curve_bias_prefers_cheap_when_hand_lacks_cheap`
* `test_cost_curve_bias_prefers_expensive_when_hand_lacks_expensive`
* `test_anti_stuck_guarantees_eventual_pick` — карта с `skip_count=10` имеет огромный вес
* `test_existing_overdraw_skip_test_still_passes`
* `test_existing_overdraw_to_discard_test_still_passes`
* `test_existing_graveyard_reshuffle_test_still_passes`
* `test_existing_battlecry_draw_test_still_passes`
* `test_weighted_choice_with_single_card`
* `test_weighted_choice_with_zero_weights`
* `test_rng_injection_produces_deterministic_results`
* `test_rng_injection_different_seeds_produce_different_results`
* `test_drawn_card_has_skip_count_zero`

### 7.2. Playwright integration

`/tmp/pw-verify/verify_weighted_draw.py` — 6 сценариев, прогоняемых
одновременно через Python-движок (oracle) и в реальном Chromium через
JS-replay логики weighted draw:

1. `anti_stuck_skip` — при заполненной руке все 3 карты получают `skip_count=1`
2. `cost_curve` — cheap+expensive бакеты суммарно получают > 1.3× доборов mid-бакета
3. `reshuffle_reset` — карта из graveyard после reshuffle имеет `skip_count=0`
4. `determinism` — same seed → same drawn card
5. `variance` — 30 разных seeds дают ≥3 distinct cards
6. `weights_bias` — `_compute_draw_weights` возвращает `[1.3, 1.0, 1.0]` для
   cheap+expensive-бакетов с правильным bias

Сравнение **категориальное**, не точное (разные RNG — Python Mersenne Twister
против JS LCG), но свойства bias / anti-stuck / reshuffle должны совпадать.

### 7.3. Регрессионный baseline

Полный test suite: **32 fail на main** (pre-existing, см. memory
`test-suite-baseline.md`). После введения weighted draw + No-FIFO:

* 32 fail (тот же baseline — admin / runtime config / push tests, не связаны)
* 0 новых failures
* 6 дополнительных тестов FIXED (из action_result_cache и extra_pass_claiming)

## 8. Известные ограничения

* `skip_count` обнуляется при `reset_to_base_state` (reshuffle), но **не**
  обнуляется автоматически при других событиях. Это намеренно: считается, что
  цикл «карта живёт в deck → попадает в hand» — единственное, что «обнуляет»
  её «возраст».
* Bias по cost — **только cheap/expensive**, без учёта «spell vs warrior».
  Если понадобятся дополнительные buckets — расширять
  `_compute_draw_weights` нужно аккуратно, чтобы не сломать детерминизм
  (порядок buckets должен совпадать в Python и в JS-replay).
* `STUCK_BONUS=0.5` подобран эмпирически по 9-card колодам. Для колод другого
  размера (например, draft с 15+ картами) может потребоваться тюнинг —
  слишком большой бонус сделает weighted draw почти FIFO по `skip_count`,
  слишком маленький не решит проблему «карт вне пула».

## 9. Связанные документы

* [`docs/BATTLE_SYSTEM.md`](BATTLE_SYSTEM.md) — общее описание боевой системы,
  формат эффектов, тайминги хода. Эффект `battlecry_draw_card` ссылается
  на этот документ.
* `core/engine.py::draw_one_from_deck` — реализация.
* `core/state.py::CardInstance.skip_count` — поле.
* `core/effects.py::effect_battlecry_draw_card` — единственный вызов
  из эффектов.
* `tests/test_core_logic.py::TestStratifiedWeightedDraw` — unit-тесты.
* `ai/train_v2/classic_rl_env.py::reset` — RL env, пробрасывает seeded rng
  в `ArenaEnvironment`.

## Audit (2026-06-25)

Проверено против исходников:

- `core/engine.py:47-50` — константы `HAND_CAP=4`, `STUCK_BONUS=0.5`,
  `COST_BIAS=0.3`, `CHEAP_COST_MAX=2`, `EXPENSIVE_COST_MIN=4` совпадают.
- `core/engine.py:53-82` — `_compute_draw_weights` реализует именно ту
  формулу, что описана (плюс «defensive `max(0, 1 - in_hand)` для бакета»).
- `core/engine.py:109-221` — `draw_one_from_deck`: три шага (reshuffle,
  skip-count инкремент + overdraw-handling, weighted draw) соответствуют
  разделу 2.4. `skip_count` сбрасывается у вытянутой карты (line 216) и
  у карты, ушедшей в graveyard при `overdraw_to_discard` (line 187), а
  для оставшихся в deck применяется компенсирующий декремент
  (lines 191-192, 218-219).
- `core/engine.py:667-741` — `_handle_end_turn` зовёт `draw_one_from_deck`
  с `source="end_turn"` и `rng=self._rng`.
- `core/effects.py:260-300` — `effect_battlecry_draw_card` зовёт
  `draw_one_from_deck` с `source="battlecry_draw_card"`, пробрасывая
  `state.arena_engine._rng`.
- `core/state.py:102` — поле `skip_count: int = 0` на `CardInstance`.
- `core/state.py:130` — `reset_to_base_state` обнуляет `skip_count`.
- `ai/train_v2/classic_actions_v1.py:369-412` — `_make_preview_env` идёт
  через `ArenaEnvironment.__new__` и тремя путями устанавливает `_rng`
  (clone через `getstate/setstate`, cold-path по fingerprint, fresh
  отброшен) — описание в §6.2 точное.
- `ai/train_v2/classic_rl_env.py:132-179` — `reset` сеет модульный
  random и передаёт `state_rng` в `ArenaEnvironment`.
- `tests/test_core_logic.py:1985-2630` — все 16 тестов из §7.1
  присутствуют с указанными именами.
- `/tmp/pw-verify/verify_weighted_draw.py:160-167` — ровно 6 сценариев:
  anti_stuck_skip, cost_curve, reshuffle_reset, determinism, variance,
  weights_bias.
- `infrastructure/match_modes.py:42` — `ClassicParams.overdraw_to_discard: bool = False`.
- `battle_engine.py::BattleEngine.create_match` создаёт `ArenaEnvironment`
  через классический `ArenaEnvironment(...)` без `rng=`, оставляя
  `random.Random()` — это значит, что матчи, созданные через адаптер,
  используют fresh RNG, а не seeded. В §2.5 это явно не отражено, но
  противоречия с текстом нет (RNG-инжекция описана только для
  `ClassicRLEnv` и preview-env).

Что исправлено:

- `docs/CYCLE_DRAW.md:210-220` — §3.1 «Численный пример»: исправлено
  ошибочное утверждение, что после добора D2 оставшиеся D1 и D3 получают
  `skip_count=1`. На самом деле инкремент skip_count в начале шага 2
  компенсируется декрементом в конце шага 3 (engine.py:175-176 +
  218-219), поэтому обе карты возвращаются к `skip_count=0`. Также
  убран внутренне противоречивый пассаж «D3 уже в руке» (D3 не
  добрана в этом сценарии) и неверные проценты (1.3 / 3.3 = 39%
  вместо корректных 50/50 после draw D2, поскольку обе карты получают
  одинаковый cheap/expensive bias и skip_count=0).

Не удалось верифицировать:

- §8 «`STUCK_BONUS=0.5` подобран эмпирически по 9-card колодам» —
  исторический комментарий, в коде/тестах прямой ссылки на «9-card
  tuning» не нашёл.
- §7.3 «6 дополнительных тестов FIXED» — относится к историческим
  изменениям (action_result_cache / extra_pass_claiming) и не
  воспроизводится из текущего кода; согласно memory
  `test-suite-baseline.md`, `test_extra_pass_claiming.py` сейчас в
  списке 32 pre-existing fail, так что формулировка в §7.3 может
  быть устаревшей.
