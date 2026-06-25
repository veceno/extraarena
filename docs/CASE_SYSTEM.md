# Система кейсов (Case System)

## Тиры кейсов

| Тир | Название | Монеты | Карты | Макс. редкость | Цена в гемах |
|-----|----------|--------|-------|----------------|--------------|
| T1 | Обычный | 40–120 | 2–4 | superrare | 5 |
| T2 | Улучшенный | 120–240 | 2–5 | epic | 15 |
| T3 | Элитный | 240–560 | 3–5 | legendary | 40 |
| T4 | Мифический | 560–1200 | 4–6 | mythic | 100 |
| T5 | Божественный | 1200–2400 | 4–7 | divine | 250 |

### T5 бонусы (дополнительно)
- **10%** — 10–50 гемов
- (других бонусных типов в коде нет)

---

## Шансы выпадения редкостей

| Редкость | T1 | T2 | T3 | T4 | T5 |
|-----------|----|----|----|----|----|
| common | **82%** | **64%** | **50.5%** | **37%** | **28%** |
| rare | **16.2%** | **27%** | **27%** | **27%** | **22.5%** |
| superrare | **1.8%** | **7.2%** | **13.5%** | **18%** | **22.5%** |
| epic | — | **1.8%** | **7.2%** | **10.8%** | **13.5%** |
| legendary | — | — | **1.8%** | **5.4%** | **7.2%** |
| mythic | — | — | — | **1.8%** | **4.5%** |
| divine | — | — | — | — | **1.8%** |
| limited | — | — | — | — | 0% (0.15% при ивенте) |

> Все строки в сумме дают ровно 1.0; `select_rarity` нормализует на случай дрейфа конфигурации.

### Замена на стартовую редкость (Юни)
При выпадении rare или superrare есть шанс заменить на `start`:

| Исходная редкость | Шанс замены |
|--------------------|-------------|
| rare | **5%** |
| superrare | **2%** |

---

## Апгрейд тира (4 тапа)

При открытии кейса игрок делает 4 тапа. Каждый тап имеет шанс повысить тир на 1 (максимум T5).

| Тап | Шанс апгрейда |
|-----|--------------|
| 1 | **25%** |
| 2 | **20%** |
| 3 | **15%** |
| 4 | **10%** |

- Тир может только увеличиваться
- Максимальный тир — T5
- Ultra-бонус к шансу тапа **отключён** (см. ниже «ExtraPass и Ultra»)

---

## Частицы за дубликаты

### База по редкости

| common | rare | start | superrare | epic | legendary | mythic | divine | limited |
|--------|------|-------|-----------|------|-----------|--------|--------|---------|
| 2 | 3 | 4 | 5 | 10 | 20 | 40 | 100 | 0 |

### Множитель тира кейса

| T1 | T2 | T3 | T4 | T5 |
|----|----|----|----|----|
| ×1.30 | ×0.65 | ×0.81 | ×1.14 | ×1.95 |

T1-множитель удержан на 1.0 намеренно, чтобы базовый common-дубль не схлопнулся в 0.

### Джекпот T5
Если из T5 выпадает common-карта-дубликат — фиксированные **125 частиц** (вместо 2 × 1.25 = 2.5).

---

## ExtraPass и Ultra

`extra_pass` (`inactive` / `active` / `ultra`) определяет только наличие **ручного Ultra-реролла** после открытия (`ULTRA_REROLL_ATTEMPTS = 1`, `get_case_reroll_attempts`, `score_case_rewards`).

С 2026-06-25 **все бонусы шансов за Ultra вырезаны**:
- Ultra больше **не** повышает шансы редкостей (нет `ULTRA_RARITY_MULTIPLIERS`);
- Ultra больше **не** добавляет +10% к шансу тап-апгрейда (нет `ULTRA_TAP_CHANCE_BONUS`);
- Ultra больше **не** умножает шанс T5-гемов на 1.8 (нет `ULTRA_T5_GEMS_CHANCE_MULTIPLIER`).

Параметр `extra_pass` сохранён в сигнатурах `select_rarity`, `_generate_single_case_rewards`, `roll_tier_upgrade` для совместимости и для проброса уровня в `process_case_opening`/`generate_case_rewards` (где проверяется флаг реролла). На шансы награды он больше не влияет.

---

## Как заработать кейсы

1. **Победы:** каждые N побед — 1 ключ (T1 кейс):
   - Без подписки: 5 побед
   - ExtraPass: 4 победы
   - ExtraPass Ultra: 3 победы
2. **Покупка за гемы:** наборы кейсов в магазине (`case_pack_1/3/5/10`); прямая покупка по тирам (`case_tier_X`) отключена (`legacy_case_tier_disabled`)
3. **Battle Pass / Glory Path:**
   - Glory Path: кейсы не выдаёт (только keys/coins/gems/particles на позициях 300/1000/1500/3000/4000/7000/8000)
   - BP Free: T2 (10), T3 (20), T3 (30), T4 (40)
   - BP Premium: T3 (10), T4 (20), T4 (30), T5 (40)
   - BP Ultra: T5 (42)

---

## Файлы

| Файл | Назначение |
|------|-----------|
| `infrastructure/case_config.py` | Все конфиги (шансы, награды, частицы, апгрейд) |
| `infrastructure/case_system.py` | Логика: апгрейд, выбор редкости, генерация наград, Ultra-реролл |
| `infrastructure/shop_config.py` | Цены кейсов в магазине |
| `infrastructure/database.py` | Таблица `user_cases`, `keys`, сидирование reward_tracks |
| `web/server.py` | API-эндпоинты: открытие, пропуск, open-from-keys, счётчик побед |
| `cards.json` | Все карты (пул наград) |

## Audit (2026-06-25, правки шансов и Ultra)

Правки (в два прохода):
- **Проход 1 (баланс и Ultra):**
  - `infrastructure/case_config.py`: переписаны `TIER_RARITY_PROBABILITIES`, `TIER_REWARDS_COUNT` (coins −20%, cards min −35% / max −25%), `BASE_PARTICLES_BY_RARITY` (≈ −75% через `math.ceil(base × 0.25)`), `TIER_PARTICLES_MULTIPLIER` (T2–T5 ×0.25, T1 удержан на 1.0), `T5_COMMON_JACKPOT_PARTICLES` = 125.
  - `infrastructure/case_system.py`: вырезаны `ULTRA_TAP_CHANCE_BONUS`, `ULTRA_RARITY_MULTIPLIERS`, `ULTRA_T5_GEMS_CHANCE_MULTIPLIER` и их применения в `roll_tier_upgrade`, `select_rarity`, `_generate_single_case_rewards`. Реролл-логика (`ULTRA_REROLL_ATTEMPTS`, `get_case_reroll_attempts`, `score_case_rewards`, `extra_pass_bonus.reroll_*`) сохранена.
  - `tests/test_case_system.py::test_simulate_case_tap_results_uses_server_rolls`: тест расширен — проверяет, что после вырезки Ultra-бонуса `simulate_case_tap_results("ultra")` и `simulate_case_tap_results("inactive")` возвращают одинаковую последовательность (откат `roll_tier_upgrade` мокируется итератором `[1,2,2,3]` на каждый вызов).
- **Проход 2 (фикс «+0 частиц» и +7% к max карт):**
  - `TIER_REWARDS_COUNT["cards"]`: max +1 во всех тирах (min не тронут). T1 (2,3)→(2,4), T2 (2,4)→(2,5), T3 (3,4)→(3,5), T4 (4,5)→(4,6), T5 (4,6)→(4,7). Средний прирост числа карт ≈ +13%.
  - `TIER_PARTICLES_MULTIPLIER`: T2–T5 подняты на +15–20% (0.38→0.50, 0.50→0.62, 0.75→0.88, 1.25→1.50) — фикс того, что `int(2 × 0.38) = 0` для T2 common, и аналогичные нули для других тиров; теперь `int(base × mult) ≥ 1` для всех не-limited клеток.
- **Проход 3 (+30% к множителю частиц на всех тирах):**
  - `TIER_PARTICLES_MULTIPLIER`: 1.00→1.30, 0.50→0.65, 0.62→0.81, 0.88→1.14, 1.50→1.95. Все клетки остаются ≥1.

Проверены все конкретные цифры (шансы, цены, множители, диапазоны) против:
- `infrastructure/case_config.py` (`TIER_RARITY_PROBABILITIES`, `TIER_REWARDS_COUNT`, `BASE_PARTICLES_BY_RARITY`, `TIER_PARTICLES_MULTIPLIER`, `T5_COMMON_JACKPOT_PARTICLES`, `TIER_UPGRADE_CHANCES`, `START_RARITY_REPLACEMENT`, `MAX_RARITY_BY_TIER`, `LIMITED_EVENT_PROBABILITY`).
- `infrastructure/case_system.py` (`_generate_single_case_rewards`, `select_rarity`, `roll_tier_upgrade`, `process_case_opening`).
- `infrastructure/database.py` (seed `reward_tracks`: `glory`, `bp_free`, `bp_premium`, `bp_ultra`).
- `web/server.py` (`wins_for_case` на строке 2908 = 3/4/5, роуты `/api/cases/*`, `legacy_case_tier_disabled` на строке 17616–17621).

Исправлено в этом аудите:
- Все таблицы в этом документе синхронизированы с новыми значениями в `case_config.py`.
- Добавлен раздел «ExtraPass и Ultra» с описанием того, что именно осталось от Ultra и что вырезано.

Что НЕ удалось проверить без живого рантайма:
- Точные значения, которые могли бы быть переопределены в БД через `admin_rewards_tracks_update` после seed (на момент аудита `reward_tracks` соответствует seed-данным; возможные ручные правки в проде не верифицированы).

## Audit (2026-06-25)

Проверены все конкретные цифры (шансы, цены, множители, диапазоны) против:
- `infrastructure/case_config.py` (`TIER_RARITY_PROBABILITIES`, `TIER_REWARDS_COUNT`, `BASE_PARTICLES_BY_RARITY`, `TIER_PARTICLES_MULTIPLIER`, `T5_COMMON_JACKPOT_PARTICLES`, `TIER_UPGRADE_CHANCES`, `START_RARITY_REPLACEMENT`, `MAX_RARITY_BY_TIER`, `LIMITED_EVENT_PROBABILITY`).
- `infrastructure/case_system.py` (`_generate_single_case_rewards`, `select_rarity`, `roll_tier_upgrade`, `process_case_opening`).
- `infrastructure/database.py` (seed `reward_tracks`: `glory`, `bp_free`, `bp_premium`, `bp_ultra`).
- `web/server.py` (`wins_for_case` на строке 2908 = 3/4/5, роуты `/api/cases/*`, `legacy_case_tier_disabled` на строке 17616–17621).

Исправлено:
- `docs/CASE_SYSTEM.md:15` — удалена строка про «5% — 1–5 осколков лимитированных карт»: в `case_config.py` нет ни осколков, ни бонусной выдачи limited-карт, поле `BASE_PARTICLES_BY_RARITY['limited']=0`.
- `docs/CASE_SYSTEM.md:84` — уточнено, что прямая покупка кейсов по тирам (`case_tier_X`) отключена (`legacy_case_tier_disabled`); в магазине продаются только наборы `case_pack_1/3/5/10` (`shop_config.py:33`).
- `docs/CASE_SYSTEM.md:85` — убрана строка про Glory Path T2/T3/T3/T4: в seed `reward_tracks` (database.py:14818–14839) у `glory` нет ни одной записи `reward_type='case'`, только keys/coins/gems/particles.

Что НЕ удалось проверить без живого рантайма:
- Точные значения, которые могли бы быть переопределены в БД через `admin_rewards_tracks_update` после seed (на момент аудита `reward_tracks` соответствует seed-данным; возможные ручные правки в проде не верифицированы).

