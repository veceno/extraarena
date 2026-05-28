"""Конфигурация генератора ключей.

Интервалы в часах, caps в количестве ключей.
Каждый цикл Генератор даёт ровно +1 ключ (yield).
"""

GENERATOR_LEVELS = {
    1: {
        "f2p":    {"interval_hours": 8, "cap": 2},
        "active": {"interval_hours": 6, "cap": 3},
        "ultra":  {"interval_hours": 4, "cap": 4},
    },
    2: {
        "f2p":    {"interval_hours": 7, "cap": 3},
        "active": {"interval_hours": 5, "cap": 4},
        "ultra":  {"interval_hours": 3, "cap": 5},
    },
    3: {
        "f2p":    {"interval_hours": 6, "cap": 4},
        "active": {"interval_hours": 4, "cap": 5},
        "ultra":  {"interval_hours": 2, "cap": 6},
    },
    4: {
        "f2p":    {"interval_hours": 5, "cap": 5},
        "active": {"interval_hours": 3, "cap": 6},
        "ultra":  {"interval_hours": 2, "cap": 7},
    },
    5: {
        "f2p":    {"interval_hours": 4, "cap": 6},
        "active": {"interval_hours": 2, "cap": 8},
        "ultra":  {"interval_hours": 1, "cap": 9},
    },
    6: {
        "f2p":    {"interval_hours": 3, "cap": 8},
        "active": {"interval_hours": 2, "cap": 10},
        "ultra":  {"interval_hours": 1, "cap": 12},
    },
}

GENERATOR_UPGRADE_COST = {
    2: {"gems": 100},
    3: {"gems": 250},
    4: {"gems": 500},
    5: {"gems": 900},
    6: {"gems": 1500},
}

GENERATOR_MAX_LEVEL = max(GENERATOR_LEVELS.keys())
