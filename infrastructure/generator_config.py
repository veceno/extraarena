"""Конфигурация генератора ключей.

Интервалы в часах, caps в количестве ключей.
Каждый цикл Генератор даёт ровно +1 ключ (yield).
"""

GENERATOR_LEVELS = {
    1: {
        "f2p":    {"interval_hours": 12, "cap": 2},
        "active": {"interval_hours": 8,  "cap": 3},
        "ultra":  {"interval_hours": 6,  "cap": 4},
    },
    2: {
        "f2p":    {"interval_hours": 10, "cap": 3},
        "active": {"interval_hours": 7,  "cap": 4},
        "ultra":  {"interval_hours": 5,  "cap": 5},
    },
}

GENERATOR_UPGRADE_COST = {
    2: {"coins": 2000, "gems": 100},
}

GENERATOR_MAX_LEVEL = max(GENERATOR_LEVELS.keys())
