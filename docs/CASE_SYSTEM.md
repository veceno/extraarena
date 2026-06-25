# Система кейсов (Case System)

## Тиры кейсов

| Тир | Название | Монеты | Карты | Макс. редкость | Цена в гемах |
|-----|----------|--------|-------|----------------|--------------|
| T1 | Обычный | 50–150 | 3–4 | superrare | 5 |
| T2 | Улучшенный | 150–300 | 4–5 | epic | 15 |
| T3 | Элитный | 300–700 | 5–6 | legendary | 40 |
| T4 | Мифический | 700–1500 | 6–7 | mythic | 100 |
| T5 | Божественный | 1500–3000 | 7–8 | divine | 250 |

### T5 бонусы (дополнительно)
- **10%** — 10–50 гемов
- (других бонусных типов в коде нет)

---

## Шансы выпадения редкостей

| Редкость | T1 | T2 | T3 | T4 | T5 |
|-----------|----|----|----|----|----|
| common | **80%** | **60%** | **45%** | **30%** | **20%** |
| rare | **18%** | **30%** | **30%** | **30%** | **25%** |
| superrare | **2%** | **8%** | **15%** | **20%** | **25%** |
| epic | — | **2%** | **8%** | **12%** | **15%** |
| legendary | — | — | **2%** | **6%** | **8%** |
| mythic | — | — | — | **2%** | **5%** |
| divine | — | — | — | — | **2%** |
| limited | — | — | — | — | 0% (0.15% при ивенте) |

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

---

## Частицы за дубликаты

### База по редкости

| common | rare | start | superrare | epic | legendary | mythic | divine | limited |
|--------|------|-------|-----------|------|-----------|--------|--------|---------|
| 5 | 10 | 15 | 20 | 40 | 80 | 160 | 400 | 0 |

### Множитель тира кейса

| T1 | T2 | T3 | T4 | T5 |
|----|----|----|----|----|
| ×1.0 | ×1.5 | ×2.0 | ×3.0 | ×5.0 |

### Джекпот T5
Если из T5 выпадает common-карта-дубликат — фиксированные **500 частиц** (вместо 5 × 5 = 25).

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
| `infrastructure/case_system.py` | Логика: апгрейд, выбор редкости, генерация наград |
| `infrastructure/shop_config.py` | Цены кейсов в магазине |
| `infrastructure/database.py` | Таблица `user_cases`, `keys`, сидирование reward_tracks |
| `web/server.py` | API-эндпоинты: открытие, пропуск, open-from-keys, счётчик побед |
| `cards.json` | Все карты (пул наград) |

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

