"""
Конфигурация системы кейсов для ChibiArena.

Все параметры (количество наград, шансы редкостей, множители частиц, шанс апгрейда тира)
должны быть конфигурируемыми, а не захардкоженными в коде.
"""

import copy

from dataclasses import dataclass
from typing import Dict, List


# Порядок редкостей от самой частой к самой редкой
RARITY_ORDER = [
    "common",           # Обычная
    "rare",             # Редкая
    "start",            # Стартовая (Юни)
    "superrare",        # Сверхредкая
    "epic",             # Эпическая
    "legendary",        # Легендарная
    "mythic",           # Мифическая
    "divine",           # Божественная
    "limited",          # Лимитированная
]


def rarity_ordinal(rarity: str) -> int:
    """Вернуть 1-based порядковый номер редкости (от менее редких к более редким). Неизвестная → 1."""
    try:
        return RARITY_ORDER.index(rarity) + 1
    except ValueError:
        return 1


def fallback_coins_for_rarity(rarity: str) -> int:
    """Компенсация монетами за повторную карту: 100 * n, где n — порядковый номер редкости."""
    return 100 * rarity_ordinal(rarity)


# Множитель для расчёта компенсации за частицы, когда целевая карта не в коллекции игрока.
# 1.0 для common/start, +0.1 за каждую следующую редкость по RARITY_ORDER.
PARTICLES_FALLBACK_RARITY_MULTIPLIER = {
    "common": 1.0,
    "start": 1.0,
    "rare": 1.1,
    "superrare": 1.2,
    "epic": 1.3,
    "legendary": 1.4,
    "mythic": 1.5,
    "divine": 1.6,
    "limited": 1.7,
}

# Нижняя граница для limited-карт (BASE_PARTICLES_BY_RARITY['limited'] == 0, формула даёт 0).
PARTICLES_FALLBACK_COIN_CAP = 3200


def fallback_coins_for_particles(reward_amount: int, rarity: str) -> int:
    """Вернуть компенсацию в монетах, когда нельзя зачислить частицы целевой карте.

    Формула: 10 × reward_amount × множитель_редкости. Неизвестная редкость → 1.0.
    limited: ограничено снизу значением PARTICLES_FALLBACK_COIN_CAP.
    """
    amount = max(0, int(reward_amount or 0))
    multiplier = PARTICLES_FALLBACK_RARITY_MULTIPLIER.get(rarity, 1.0)
    result = int(round(10 * amount * multiplier))
    if rarity == "limited" and result < PARTICLES_FALLBACK_COIN_CAP:
        return PARTICLES_FALLBACK_COIN_CAP
    return result


# Максимальные редкости по тиру кейса
MAX_RARITY_BY_TIER = {
    1: "superrare",      # T1: до Сверхредкой включительно
    2: "epic",           # T2: до Эпической включительно
    3: "legendary",      # T3: до Легендарной включительно
    4: "mythic",         # T4: до Мифической включительно
    5: "divine",         # T5: все, включая Божественную
}

# Таблицы вероятностей редкостей по тиру кейса
# Правки 2026-06-25: шансы редких редкостей (≥rare) снижены на 10%, common забирает
# освободившуюся массу. Сумма по каждой строке строго равна 1.0.
TIER_RARITY_PROBABILITIES = {
    1: {  # T1 (максимум Сверхредкая)
        "common": 0.820,
        "rare": 0.162,
        "superrare": 0.018,
    },
    2: {  # T2 (максимум Эпическая)
        "common": 0.640,
        "rare": 0.270,
        "superrare": 0.072,
        "epic": 0.018,
    },
    3: {  # T3 (максимум Легендарная)
        "common": 0.505,
        "rare": 0.270,
        "superrare": 0.135,
        "epic": 0.072,
        "legendary": 0.018,
    },
    4: {  # T4 (максимум Мифическая)
        "common": 0.370,
        "rare": 0.270,
        "superrare": 0.180,
        "epic": 0.108,
        "legendary": 0.054,
        "mythic": 0.018,
    },
    5: {  # T5 (все редкости)
        "common": 0.280,
        "rare": 0.225,
        "superrare": 0.225,
        "epic": 0.135,
        "legendary": 0.072,
        "mythic": 0.045,
        "divine": 0.018,
        # Лимитированная: по умолчанию 0%; при активном событии - 0.1–0.2%
        "limited": 0.00,  # Конфигурируется отдельно через флаг события
    },
}

# Шанс замены на Стартовую (Юни)
START_RARITY_REPLACEMENT = {
    "rare": 0.05,        # 5% шанс заменить Редкую на Стартовую
    "superrare": 0.02,   # 2% шанс заменить Сверхредкую на Стартовую
}

# Количество наград по тиру кейса
# Правки 2026-06-25: монеты −20% (×0.8), количество карт min −35% / max −25%
# (см. заметки в docs/CASE_SYSTEM.md).
# Правки 2026-06-25 (повтор): max карт +1 во всех тирах; монеты не трогаю.
TIER_REWARDS_COUNT = {
    1: {  # T1 (Обычный)
        "coins": (40, 120),      # Монеты: 40–120
        "cards": (2, 4),         # Карт: 2–4
    },
    2: {  # T2 (Улучшенный)
        "coins": (120, 240),     # Монеты: 120–240
        "cards": (2, 5),         # Карт: 2–5
    },
    3: {  # T3 (Элитный)
        "coins": (240, 560),     # Монеты: 240–560
        "cards": (3, 5),         # Карт: 3–5
    },
    4: {  # T4 (Мифический)
        "coins": (560, 1200),    # Монеты: 560–1200
        "cards": (4, 6),         # Карт: 4–6
    },
    5: {  # T5 (Божественный)
        "coins": (1200, 2400),   # Монеты: 1200–2400
        "cards": (4, 7),         # Карт: 4–7
        # Дополнительно: низкий шанс бонусных гемов
        "gems_chance": 0.10,     # 10% шанс на бонусные гемы
        "gems_amount": (10, 50), # Количество гемов
    },
}

# Базовое количество частиц по редкости
# Правки 2026-06-25: значения снижены ~на 75% (math.ceil(base × 0.25)) для
# сохранения иерархии common < rare < superrare < … < divine.
BASE_PARTICLES_BY_RARITY = {
    "common": 2,      # было 5
    "rare": 3,        # было 10
    "start": 4,       # было 15
    "superrare": 5,   # было 20
    "epic": 10,       # было 40
    "legendary": 20,  # было 80
    "mythic": 40,     # было 160
    "divine": 100,    # было 400
    "limited": 150,   # Лимитированные дают частицы (выше divine): дубликаты конвертируются в частицы
}

# Множитель частиц от тира кейса
# Правки 2026-06-25: T2–T5 снижены ~на 75% (×0.25, округлено вверх до сотых).
# T1 оставлен 1.0, чтобы базовый common-дубль не схлопнулся в ноль
# (1.25 → ceil → 2 частицы × 1.0 = 2, раньше было 5 × 1.0 = 5, итого −60% по
# T1 common; редкие редкости на T1 теперь дают меньше, но всё ещё видимо).
# Правки 2026-06-25 (повтор №1): T2–T5 подняты на +15–20% от прежних значений,
# чтобы int(base × mult) ≥ 1 для common на всех тирах (фикс «+0 частиц»).
# Правки 2026-06-25 (повтор №2): +30% поверх повтора №1 (T1–T5 включительно).
TIER_PARTICLES_MULTIPLIER = {
    1: 1.30,   # было 1.00 → 1.30 (+30%)
    2: 0.65,   # было 0.38 → 0.50 → 0.65 (+30%)
    3: 0.81,   # было 0.50 → 0.62 → 0.81 (+31%)
    4: 1.14,   # было 0.75 → 0.88 → 1.14 (+30%)
    5: 1.95,   # было 1.25 → 1.50 → 1.95 (+30%)
}

# Джекпот частиц для Обычной карты из T5
T5_COMMON_JACKPOT_PARTICLES = 125  # было 500 (×0.25)

# Шанс апгрейда тира на каждом тапе
TIER_UPGRADE_CHANCES = {
    1: 0.25,  # Тап 1: 25%
    2: 0.20,  # Тап 2: 20%
    3: 0.15,  # Тап 3: 15%
    4: 0.10,  # Тап 4: 10%
}

# ID карты Юни (стартовая редкость)
UNI_CARD_ID = 36

# Флаг активности события для лимитированных карт
LIMITED_EVENT_ACTIVE = False
LIMITED_EVENT_PROBABILITY = 0.0015  # 0.15% при активном событии (между 0.1% и 0.2%)


# ----------------------------------------------------------------------------
# Конфигурация кейсов в реальном времени (game_settings row "case_config")
# ----------------------------------------------------------------------------
# Все константы выше остаются значениями по умолчанию. Администратор может
# переопределять их в реальном времени (без рестарта сервера) через MCP
# (admin.case_config.read / admin.case_config.patch) или через админ-панель
# (/api/admin/case-config). Roll-функции case_system принимают необязательный
# параметр case_config; при None используются LIVE module-global объекты
# (важно для тестов, которые monkeypatch-ят эти словари на месте).

# Ключ в таблице game_settings
CASE_CONFIG_KEY = "case_config"

# Категории полей конфигурации кейсов
TIER_KEYED_FIELDS = {
    "tier_rarity_probabilities",
    "tier_particles_multiplier",
    "tier_rewards_count",
    "max_rarity_by_tier",
    "tier_upgrade_chances",
}
RARITY_KEYED_FIELDS = {
    "base_particles_by_rarity",
    "start_rarity_replacement",
}
SCALAR_FIELDS = {
    "t5_common_jackpot_particles",
    "limited_event_active",
    "limited_event_probability",
}
CASE_CONFIG_FIELDS = TIER_KEYED_FIELDS | RARITY_KEYED_FIELDS | SCALAR_FIELDS

# Соответствие имён полей blob -> имена module-level констант (для build_default)
_CASE_CONFIG_FIELD_TO_CONST = {
    "tier_rarity_probabilities": "TIER_RARITY_PROBABILITIES",
    "tier_particles_multiplier": "TIER_PARTICLES_MULTIPLIER",
    "base_particles_by_rarity": "BASE_PARTICLES_BY_RARITY",
    "tier_rewards_count": "TIER_REWARDS_COUNT",
    "start_rarity_replacement": "START_RARITY_REPLACEMENT",
    "max_rarity_by_tier": "MAX_RARITY_BY_TIER",
    "t5_common_jackpot_particles": "T5_COMMON_JACKPOT_PARTICLES",
    "tier_upgrade_chances": "TIER_UPGRADE_CHANCES",
    "limited_event_active": "LIMITED_EVENT_ACTIVE",
    "limited_event_probability": "LIMITED_EVENT_PROBABILITY",
}


def _coerce_tier_keys(value, default_keys):
    """Привести строковые ключи тиров ('1'..'5') из JSONB к int, оставить только известные тиры.

    default_keys — множество int-ключей тиров, разрешённых для данного поля
    (берётся из дефолта: 1..5 для большинства, 1..4 для tier_upgrade_chances).
    """
    if not isinstance(value, dict):
        return {}
    result = {}
    for k, v in value.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            continue
        if ik in default_keys:
            result[ik] = v
    return result


def build_default_case_config() -> dict:
    """Вернуть deepcopy конфигурации кейсов, построенный из текущих констант модуля.

    Используется как значение по умолчанию для game_settings и как база для
    merge/fill. Deepcopy гарантирует, что слияние патча не мутирует module-globals.
    """
    blob = {}
    for field, const_name in _CASE_CONFIG_FIELD_TO_CONST.items():
        # deepcopy: для вложенных dict нужна полная независимая копия,
        # чтобы слияние патча не мутировало module-globals.
        blob[field] = copy.deepcopy(globals()[const_name])
    return blob


def merge_case_config_patch(current: dict, patch: dict) -> dict:
    """Структурный deep-merge патча в текущую конфигурацию (write-side).

    - tier-keyed поля: merge по тирам — патч заменяет только указанные тиры,
      остальные сохраняются (иначе partial-патч одного тира удалил бы остальные).
    - rarity-keyed поля: merge по редкостям — патч заменяет только указанные
      редкости (иначе {'base_particles_by_rarity':{'limited':150}} обнулил бы
      common/rare/divine и вновь занулил бы частицы — ровно тот баг, что чиним).
    - скаляры: прямая замена.
    Не мутирует current; возвращает новый словарь.
    """
    result = dict(current)
    for field, pval in patch.items():
        if field not in CASE_CONFIG_FIELDS:
            # Отвергаем неизвестные поля (MCP-путь делает это в нормализаторе;
            # HTTP-путь /tests попадают сюда — единая точка отказа, ValueError→400).
            raise ValueError("unsupported_case_config_field")
        if field in TIER_KEYED_FIELDS:
            if not isinstance(pval, dict):
                raise ValueError(f"invalid_{field}")
            cur = current.get(field) or {}
            default_keys = set((cur.keys())) if cur else set()
            coerced = _coerce_tier_keys(pval, default_keys)
            result[field] = {**cur, **coerced}
        elif field in RARITY_KEYED_FIELDS:
            if not isinstance(pval, dict):
                raise ValueError(f"invalid_{field}")
            cur = current.get(field) or {}
            result[field] = {**cur, **pval}
        else:  # SCALAR_FIELDS
            result[field] = pval
    return result


def fill_case_config_defaults(stored: dict) -> dict:
    """Заполнить пропущенные ключи и тиры из значений по умолчанию (read-side deep-fill).

    Коэрсит строковые ключи тиров из JSONB к int. Не затирает административные
    правки: для tier-keyed полей отсутствующие тиры берутся из дефолта, для
    rarity-keyed — отсутствующие редкости из дефолта.
    """
    defaults = build_default_case_config()
    result = dict(defaults)
    for field in CASE_CONFIG_FIELDS:
        if field not in stored:
            continue
        sval = stored[field]
        if field in TIER_KEYED_FIELDS:
            default_keys = set(defaults[field].keys())
            coerced = _coerce_tier_keys(sval, default_keys)
            result[field] = {**defaults[field], **coerced}
        elif field in RARITY_KEYED_FIELDS:
            result[field] = {**defaults[field], **(sval or {})}
        else:  # SCALAR_FIELDS
            result[field] = sval
    return result


def resolve_case_config(case_config):
    """Центральный резолвер конфигурации для roll-функций case_system.

    - None => LIVE-ссылки на module-global объекты (НЕ копии). Это load-bearing
      для тестов, которые делают monkeypatch.setitem(case_system.TIER_REWARDS_COUNT, ...).
    - dict => fill_case_config_defaults (deep-fill + coerce tier keys).
    """
    if case_config is None:
        # MUST be live references, not copies — tests monkeypatch these module dicts in place.
        return {
            "tier_rarity_probabilities": TIER_RARITY_PROBABILITIES,
            "tier_particles_multiplier": TIER_PARTICLES_MULTIPLIER,
            "base_particles_by_rarity": BASE_PARTICLES_BY_RARITY,
            "tier_rewards_count": TIER_REWARDS_COUNT,
            "start_rarity_replacement": START_RARITY_REPLACEMENT,
            "max_rarity_by_tier": MAX_RARITY_BY_TIER,
            "t5_common_jackpot_particles": T5_COMMON_JACKPOT_PARTICLES,
            "tier_upgrade_chances": TIER_UPGRADE_CHANCES,
            "limited_event_active": LIMITED_EVENT_ACTIVE,
            "limited_event_probability": LIMITED_EVENT_PROBABILITY,
        }
    return fill_case_config_defaults(case_config)


def validate_case_config(blob: dict) -> None:
    """Структурная валидация полного blob'а конфигурации кейсов. Raises ValueError при ошибке.

    Ожидает ПОЛНЫЙ blob (все поля) — merge/fill всегда дают полный набор.
    """
    if not isinstance(blob, dict):
        raise ValueError("invalid_case_config_shape")

    def _check_tier_keys(d, field, lo, hi):
        if not isinstance(d, dict) or not d:
            raise ValueError(f"invalid_{field}")
        for k in d:
            if not isinstance(k, int) or not (lo <= k <= hi):
                raise ValueError(f"invalid_{field}")

    # tier_rarity_probabilities (tiers 1..5)
    trp = blob.get("tier_rarity_probabilities")
    _check_tier_keys(trp, "tier_rarity_probabilities", 1, 5)
    for tier, probs in trp.items():
        if not isinstance(probs, dict) or not probs:
            raise ValueError("invalid_tier_rarity_probabilities")
        s = 0.0
        for rarity, p in probs.items():
            if rarity not in RARITY_ORDER:
                raise ValueError("invalid_tier_rarity_rarity")
            if not isinstance(p, (int, float)) or not (0 <= p <= 1):
                raise ValueError("invalid_tier_rarity_prob_value")
            s += p
        if abs(s - 1.0) > 0.02:
            raise ValueError("invalid_tier_rarity_sum")

    # tier_particles_multiplier (tiers 1..5)
    tpm = blob.get("tier_particles_multiplier")
    _check_tier_keys(tpm, "tier_particles_multiplier", 1, 5)
    for tier, m in tpm.items():
        if not isinstance(m, (int, float)) or m < 0:
            raise ValueError("invalid_tier_particles_multiplier")

    # base_particles_by_rarity (rarity -> number >= 0)
    bpb = blob.get("base_particles_by_rarity")
    if not isinstance(bpb, dict) or not bpb:
        raise ValueError("invalid_base_particles_by_rarity")
    for rarity, v in bpb.items():
        if rarity not in RARITY_ORDER:
            raise ValueError("invalid_base_particles_rarity")
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("invalid_base_particles_value")

    # tier_rewards_count (tiers 1..5)
    trc = blob.get("tier_rewards_count")
    _check_tier_keys(trc, "tier_rewards_count", 1, 5)
    for tier, cfg in trc.items():
        if not isinstance(cfg, dict):
            raise ValueError("invalid_tier_rewards_count")
        coins = cfg.get("coins")
        cards = cfg.get("cards")
        if not (isinstance(coins, (list, tuple)) and len(coins) == 2
                and isinstance(coins[0], int) and isinstance(coins[1], int)
                and coins[0] <= coins[1]):
            raise ValueError("invalid_tier_rewards_coins")
        if not (isinstance(cards, (list, tuple)) and len(cards) == 2
                and isinstance(cards[0], int) and isinstance(cards[1], int)
                and cards[0] <= cards[1]):
            raise ValueError("invalid_tier_rewards_cards")
        if "gems_chance" in cfg:
            gc = cfg["gems_chance"]
            if not isinstance(gc, (int, float)) or not (0 <= gc <= 1):
                raise ValueError("invalid_tier_rewards_gems_chance")
        if "gems_amount" in cfg:
            ga = cfg["gems_amount"]
            if not (isinstance(ga, (list, tuple)) and len(ga) == 2
                    and isinstance(ga[0], int) and isinstance(ga[1], int)
                    and ga[0] <= ga[1]):
                raise ValueError("invalid_tier_rewards_gems_amount")

    # start_rarity_replacement (rarity -> float 0..1)
    srr = blob.get("start_rarity_replacement")
    if not isinstance(srr, dict):
        raise ValueError("invalid_start_rarity_replacement")
    for rarity, p in srr.items():
        if rarity not in RARITY_ORDER:
            raise ValueError("invalid_start_rarity_replacement_rarity")
        if not isinstance(p, (int, float)) or not (0 <= p <= 1):
            raise ValueError("invalid_start_rarity_replacement_value")

    # max_rarity_by_tier (tiers 1..5 -> rarity)
    mrb = blob.get("max_rarity_by_tier")
    _check_tier_keys(mrb, "max_rarity_by_tier", 1, 5)
    for tier, rarity in mrb.items():
        if rarity not in RARITY_ORDER:
            raise ValueError("invalid_max_rarity_by_tier")

    # t5_common_jackpot_particles (int >= 0)
    jc = blob.get("t5_common_jackpot_particles")
    if not isinstance(jc, int) or jc < 0:
        raise ValueError("invalid_t5_jackpot")

    # tier_upgrade_chances (tiers 1..4 -> float 0..1)
    tuc = blob.get("tier_upgrade_chances")
    _check_tier_keys(tuc, "tier_upgrade_chances", 1, 4)
    for tier, p in tuc.items():
        if not isinstance(p, (int, float)) or not (0 <= p <= 1):
            raise ValueError("invalid_tier_upgrade_chances")

    # limited_event_active (bool)
    if not isinstance(blob.get("limited_event_active"), bool):
        raise ValueError("invalid_limited_event_active")

    # limited_event_probability (float 0..1)
    lep = blob.get("limited_event_probability")
    if not isinstance(lep, (int, float)) or not (0 <= lep <= 1):
        raise ValueError("invalid_limited_event_probability")
