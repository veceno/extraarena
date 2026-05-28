"""Конфигурация клановой системы ExtraArena."""

import re

CLAN_NAME_MIN = 2
CLAN_NAME_MAX = 24
CLAN_TAG_MIN = 3
CLAN_TAG_MAX = 5
CLAN_DESC_MAX = 200
CLAN_BASE_SLOTS = 15
CLAN_MAX_OFFICERS = 5

TAG_PATTERN = re.compile(r'^[A-Z0-9]{3,5}$')

SLOT_UPGRADES = {
    1: (500, 5),
    2: (1000, 5),
    3: (2000, 5),
}
MAX_SLOT_LEVEL = 3

CLAN_ACTIVITY_TYPES = {
    "created": "создал клан",
    "join": "вступил в клан",
    "leave": "покинул клан",
    "kick": "исключён из клана",
    "promote": "повышен до Офицера",
    "demote": "понижен до Участника",
    "upgrade": "улучшил клан",
    "transfer": "передал владение",
}

MOCK_SHOP_REWARDS = [
    {"id": "r1", "name": "Рамка «Toxic Glow»", "rarity": "common", "cost": 100},
    {"id": "r2", "name": "Титул «Fox Hunter»", "rarity": "rare", "cost": 250},
    {"id": "r3", "name": "Рамка «Violet Flames»", "rarity": "epic", "cost": 600},
    {"id": "r4", "name": "Граница «Thunder»", "rarity": "epic", "cost": 800},
    {"id": "r5", "name": "Титул «War Commander»", "rarity": "epic", "cost": 400},
    {"id": "r6", "name": "Аватар «Chrome Fox»", "rarity": "rare", "cost": 300},
]
