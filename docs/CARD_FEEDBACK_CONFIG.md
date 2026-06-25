# Конфиг feedback-эффектов карт арены

Этот документ описывает единую конфигурацию пер-карточных реакций арены: звуки, фоновые вспышки и текстовые подсказки.

Основной файл:

```text
assets/audio/characters/card_sfx_config.json
```

Fallback на случай, если JSON не загрузился:

```text
webapp/arena.js
const CARD_SFX_CONFIG_DEFAULT = { ... }
```

Пока fallback хранится вручную, при добавлении или изменении конфигурации карты обновляй оба места. Runtime сверяет внешний JSON с базовой формой, отбрасывает невалидные entries и мержит внешний конфиг поверх fallback, чтобы частичный или повреждённый файл не выключил уже известные настройки.

## Общая структура

```json
{
  "version": 1,
  "cards": {
    "34": {
      "name": "Крипер",
      "sounds": {},
      "visuals": {},
      "texts": {}
    }
  }
}
```

Ключ в `cards` - это catalog/card id карты из базы или CSV. Все каналы опциональны:

| Канал | Назначение |
|---|---|
| `sounds` | Уникальные SFX карты |
| `visuals` | Фоновые реакции арены |
| `texts` | Текстовые пояснения и подсказки выбора цели |

Если у карты нет уникальной настройки для события, арена использует базовое поведение, если оно передано в коде как fallback.

## События

Базовые события:

| Ключ | Когда срабатывает |
|---|---|
| `deploy` | Карта поставлена на поле или применена из руки |
| `death` | Карта умерла |
| `damage` | Карта получила урон |
| `attack` | Карта атакует |
| `mechanic` | У карты применена механика |
| `targeting` | Игрок выбирает цель при постановке или применении карты |
| `attacktargeting` | Игрок выбирает цель атаки уже поставленной картой |

Для конкретной механики используй формат:

```text
<event>:<mechanic_code>
```

Примеры:

```json
{
  "mechanic:deathrattle_aoe_damage_2": {},
  "targeting:battlecry_heal_target_3": {},
  "attacktargeting:instant_kill": {}
}
```

Важно: имена событий нормализуются в lowercase. Поэтому в JSON нужен ключ `attacktargeting:instant_kill`, а не `attackTargeting:instant_kill`.

## Звуки

Формат:

```json
{
  "sounds": {
    "death": {
      "src": "/assets/audio/characters/creeper/creeper_death_explosion.mp3",
      "basePolicy": "replace",
      "volume": 0.82
    }
  }
}
```

Поля:

| Поле | Тип | Описание |
|---|---|---|
| `src` | string | Путь к аудио от корня web-сервера |
| `basePolicy` | string | `replace` отключает базовый звук события, если уникальный звук найден |
| `volume` | number | Громкость от `0` до `1` |

Правила:

- Уникальный звук имеет приоритет над базовым, если `basePolicy: "replace"`.
- Настройка игрока "Отключение звуковых эффектов" распространяется и на базовые SFX, и на уникальные SFX карт.
- В onboarding звуки подавляются, если явно не передан `allowOnboarding`.
- Аудио-файлы лучше хранить в отдельной папке карты внутри `assets/audio/characters/`.

Пример:

```json
{
  "34": {
    "name": "Крипер",
    "sounds": {
      "deploy": {
        "src": "/assets/audio/characters/creeper/creeper_spawn_hiss.mp3",
        "basePolicy": "replace",
        "volume": 0.82
      },
      "mechanic:deathrattle_aoe_damage_2": {
        "src": "/assets/audio/characters/creeper/creeper_death_explosion.mp3",
        "basePolicy": "replace",
        "volume": 0.82
      }
    }
  }
}
```

## Фоновые реакции

Сейчас поддерживается тип:

```text
backgroundFlash
```

Формат:

```json
{
  "visuals": {
    "death": {
      "type": "backgroundFlash",
      "color": "#ef4444",
      "durationMs": 3600,
      "intensity": 0.82
    }
  }
}
```

Поля:

| Поле | Тип | Описание |
|---|---|---|
| `type` | string | Сейчас только `backgroundFlash` |
| `color` | string | Основной цвет вспышки |
| `centerColor` | string | Опциональный цвет центра вспышки |
| `midColor` | string | Опциональный средний цвет |
| `edgeColor` | string | Опциональный цвет края |
| `durationMs` | number | Длительность, обычно `3000-3600` |
| `intensity` | number | Сила вспышки от `0.05` до `1` |

Правила:

- Вспышки не зависят от пользовательской настройки SFX.
- В onboarding вспышки подавляются, если явно не передан `allowOnboarding`.
- Слой вспышки не принимает клики и не должен блокировать UI.
- Цвета ожидаются в hex-формате `#rgb` или `#rrggbb`; невалидное значение заменяется безопасным красным fallback.

Пример красной вспышки Крипера на взрыв:

```json
{
  "34": {
    "name": "Крипер",
    "visuals": {
      "death": {
        "type": "backgroundFlash",
        "color": "#ef4444",
        "durationMs": 3600,
        "intensity": 0.82
      },
      "mechanic:deathrattle_aoe_damage_2": {
        "type": "backgroundFlash",
        "color": "#ef4444",
        "durationMs": 3600,
        "intensity": 0.82
      }
    }
  }
}
```

## Текстовые пояснения

Сейчас есть два типа текстов:

| Тип | Назначение |
|---|---|
| `screenText` | Затемняет арену и показывает крупную надпись на `1-2` секунды |
| `targetHint` | Показывает маленькую таблетку-подсказку при выборе цели |

### screenText

Формат:

```json
{
  "texts": {
    "mechanic:aoe_freeze": {
      "type": "screenText",
      "text": "Время остановлено",
      "durationMs": 1500
    }
  }
}
```

Поля:

| Поле | Тип | Описание |
|---|---|---|
| `type` | string | `screenText` |
| `text` | string | Основной текст |
| `defaultText` | string | Текст по умолчанию, если используется `detailText` |
| `detailText` | object | Карта `effect_code -> текст` для точного результата механики |
| `durationMs` | number | Длительность, ограничивается примерно `800-4200` |

Пример Мидории, где сервер передает точный `effect_code` выбранной случайной механики:

```json
{
  "26": {
    "name": "Мидория",
    "texts": {
      "mechanic:cast_random_spell": {
        "type": "screenText",
        "defaultText": "Случайная суперспособность",
        "durationMs": 1600,
        "detailText": {
          "midoriya_texas_smash": "Техасский удар",
          "midoriya_recovery": "Восстановление",
          "midoriya_blackwhip": "Чёрный кнут",
          "midoriya_full_cowl": "Полный покров"
        }
      }
    }
  }
}
```

### targetHint

Формат:

```json
{
  "texts": {
    "targeting:battlecry_heal_target_3": {
      "type": "targetHint",
      "text": "Выбери цель для исцеления"
    }
  }
}
```

Используй `targeting:<mechanic_code>` для выбора цели при постановке карты из руки.

Примеры:

```json
{
  "36": {
    "name": "Юни",
    "texts": {
      "targeting:battlecry_heal_target_3": {
        "type": "targetHint",
        "text": "Выбери цель для исцеления"
      }
    }
  },
  "25": {
    "name": "Сайтама",
    "texts": {
      "attacktargeting:instant_kill": {
        "type": "targetHint",
        "text": "Выбери цель для ваншота"
      }
    }
  }
}
```

Правила:

- `targetHint` скрывается при сбросе режима выбора цели.
- Для атакующих существ используй `attacktargeting:<mechanic_code>`.
- Текстовые эффекты не зависят от пользовательской настройки SFX.
- В onboarding тексты подавляются, если явно не передан `allowOnboarding`.

## Приоритет и fallback

Для одного события система пытается проиграть все настроенные каналы:

1. `visuals`
2. `texts`
3. `sounds`

Для SFX отдельно действует `basePolicy`:

- `replace`: уникальный звук заменяет базовый звук события.
- Без `replace`: уникальный звук может проиграться вместе с базовым fallback-звуком, если fallback передан.

Для `mechanic`-событий сначала ищется конкретный ключ `mechanic:<code>`, затем общий `mechanic`.

## Добавление новой карты

1. Найди card id и mechanic code карты.
2. Добавь запись в `assets/audio/characters/card_sfx_config.json`.
3. Если добавляешь SFX, положи аудио в `assets/audio/characters/<card_folder>/` и укажи `src` от корня сайта.
4. Продублируй запись в `CARD_SFX_CONFIG_DEFAULT` в `webapp/arena.js`.
5. Для новых типов поведения добавь или расширь regression test в `tests/test_arena_frontend_regressions.py`.
6. Запусти проверки:

```bash
python3 -m json.tool assets/audio/characters/card_sfx_config.json
node --check webapp/arena.js
pytest tests/test_arena_frontend_regressions.py -q
```

Если менялась механика серверных событий, дополнительно запусти:

```bash
pytest tests/test_core_logic.py -q
```

## Частые ошибки

- Ключ события написан в camelCase: в JSON нужен lowercase (`attacktargeting`, не `attackTargeting`).
- Обновлен JSON, но не обновлен `CARD_SFX_CONFIG_DEFAULT`.
- Для механики указан общий `mechanic`, хотя нужен точный ключ `mechanic:<code>`.
- Звук добавлен без `basePolicy: "replace"` и поэтому играет вместе с базовым fallback.
- `targetHint` добавлен как `mechanic:<code>`, хотя подсказка выбора цели должна быть `targeting:<code>`.
- Путь `src` не начинается с `/assets/audio/characters/`.
- Цвет вспышки указан не в hex-формате, из-за чего runtime применит fallback-цвет.

## Audit (2026-06-25)

Проверил структуру doc против:
- `assets/audio/characters/card_sfx_config.json` (реальный конфиг)
- `webapp/arena.js` (`CARD_SFX_CONFIG_DEFAULT`, `mergeArenaCardSfxConfig`, `loadArenaCardSfxConfig`, `normalizeArenaSoundEventName`)
- `web/server.py` — community endpoints (`/api/community/posts/*`, `/api/community/chat/*`, `/api/community/upload-image`, `/api/community/news`, `/api/community/rating` и т. д.) на месте; community-конфиг относится к `infrastructure/community_config.py`, а не к feedback-конфигу — `community_config.py` не цитируется в doc, расхождений нет.

Что исправлено:
- `docs/CARD_FEEDBACK_CONFIG.md:88` — пример "Формат" для sounds.death имел `volume: 0.86`; в актуальном `card_sfx_config.json` для creeper_death_explosion = `0.82`. Исправлено на `0.82`.
- `docs/CARD_FEEDBACK_CONFIG.md:117-124` — пример Крипера для `sounds.deploy` имел `volume: 0.78`, а для `mechanic:deathrattle_aoe_damage_2` — `volume: 0.86`. В реальном JSON оба = `0.82`. Исправлено.

Что НЕ удалось проверить (не критично для этого doc):
- Точные значения `durationMs`/`intensity`/`intensity` для отдельных примеров в doc (отдельные значения в doc совпадают с JSON; полный перебор не делал, чтобы не вносить шум).
- `tests/test_arena_frontend_regressions.py` существует (79 KB) — структура совпадает с описанной в doc; содержимое не сверял побайтово.
