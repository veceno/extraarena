"""Генератор и парсер колод для RLHF-среды.

Два режима:
1. **Random ArenaENV deck** — случайная колода по правилам ArenaENV:
   ровно 1 герой, ≤ N воинов (по умолчанию 8 уникальных, по 2 копии = 16),
   ≤ M зелий (по умолчанию 2 уникальных, по 2 копии = 4).
2. **Custom JSON deck** — пользовательский JSON со списком card_id.

Каталог берётся из `ai/cards.json` (источник правды для ExtraArena).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_CARDS_PATH = Path("ai/cards.json")

# Размер колоды дословно повторяет прод: infrastructure/config.py::DECK_SIZE = 9
# (1 герой + 8 non-hero). В проде дубли card_id ЗАПРЕЩЕНЫ — см.
# infrastructure/database.py::get_user_deck_presets (is_playable требует
# `not duplicate_cards`) и ai/bot_factory.py::_sanitize_deck (dedup через
# `used: set`). Поэтому RLHF-арена тоже строит колоду из УНИКАЛЬНЫХ card_id:
# warrior_copies=1, potion_copies=1. Раньше было copies=2 — это создавало
# 2 одинаковые карты в колоде, и они могли обе попасть в руку (не баг
# циклирования, а drift построения колоды). Несоответствие исправлено.
PROD_DECK_SIZE = 9
TARGET_NON_HERO = PROD_DECK_SIZE - 1  # 8 non-hero карт

DEFAULT_DECK_SIZE = {
    "warrior_unique_min": 5,
    "warrior_unique_max": 7,
    "warrior_copies": 1,
    "potion_unique_min": 1,
    "potion_unique_max": 3,
    "potion_copies": 1,
}


@dataclass(frozen=True)
class CardCatalog:
    """Каталог карт, загруженный из cards.json."""

    cards: Dict[int, Dict[str, Any]]
    heroes: List[int] = field(default_factory=list)
    warriors: List[int] = field(default_factory=list)
    potions: List[int] = field(default_factory=list)

    @property
    def card_ids(self) -> List[int]:
        return list(self.cards.keys())

    def card(self, card_id: int) -> Dict[str, Any]:
        return self.cards[card_id]


def load_catalog(path: Path | str = DEFAULT_CARDS_PATH) -> CardCatalog:
    """Загружает cards.json и категоризирует по card_type."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cards_dict: Dict[int, Dict[str, Any]] = {}
    for entry in raw:
        cid = int(entry.get("id"))
        cards_dict[cid] = entry
    heroes = [cid for cid, c in cards_dict.items() if c.get("card_type") == "hero"]
    warriors = [cid for cid, c in cards_dict.items() if c.get("card_type") == "warrior"]
    potions = [cid for cid, c in cards_dict.items() if c.get("card_type") == "potion"]
    return CardCatalog(cards=cards_dict, heroes=heroes, warriors=warriors, potions=potions)


def build_random_arena_deck(
    catalog: CardCatalog,
    *,
    rng: Optional[random.Random] = None,
    hero_id: Optional[int] = None,
    config: Dict[str, int] | None = None,
) -> List[int]:
    """Генерирует случайную ArenaENV-колоду.

    Возвращает список card_id в формате [hero, warrior, warrior, ..., potion, ...].
    Дека не сериализуется в готовые CardInstance — это работа deck_from_card_ids.

    Повторяет прод-формат (infrastructure/config.py::DECK_SIZE = 9): ровно 1
    герой + 8 non-hero, ВСЕ card_id УНИКАЛЬНЫ. Дубли card_id в проде запрещены
    (deck_presets.is_playable требует `not duplicate_cards`; bot_factory
    _sanitize_deck дедуплит) — поэтому copies всегда 1, а число воинов/зелий
    подбирается так, чтобы суммарно non-hero == TARGET_NON_HERO (8).
    """
    rng = rng or random.Random()
    cfg = {**DEFAULT_DECK_SIZE, **(config or {})}

    if not catalog.heroes:
        raise ValueError("cards catalog has no heroes")
    if not catalog.warriors:
        raise ValueError("cards catalog has no warriors")

    hero = hero_id if hero_id is not None else rng.choice(catalog.heroes)

    # copies всегда 1 — дубли card_id запрещены (см. комментарий в DEFAULT_DECK_SIZE).
    # Конфиг с copies>1 игнорируется осознанно: это drift от прод-поведения.
    warrior_copies = 1
    potion_copies = 1

    # Подбираем число зелий в допустимом диапазоне, остальное добираем воинами
    # так, чтобы warrior_count + potion_count == TARGET_NON_HERO (8).
    potion_min = int(cfg["potion_unique_min"])
    potion_max_cfg = int(cfg["potion_unique_max"])
    warrior_min = int(cfg["warrior_unique_min"])
    warrior_max_cfg = int(cfg["warrior_unique_max"])

    available_potions = len(catalog.potions)
    available_warriors = len(catalog.warriors)

    # Диапазон числа зелий с учётом доступности и лимитов по воинам/размеру колоды.
    max_potions = min(potion_max_cfg, available_potions, TARGET_NON_HERO - warrior_min)
    if available_potions == 0 or max_potions <= 0:
        potion_count = 0
    else:
        min_potions = min(potion_min, max_potions)
        potion_count = rng.randint(min_potions, max_potions)

    warrior_count = TARGET_NON_HERO - potion_count

    # Корректируем, если воинов не хватает или выходим за уникальный максимум.
    if warrior_count > available_warriors:
        warrior_count = available_warriors
        potion_count = TARGET_NON_HERO - warrior_count
    warrior_count = min(warrior_count, warrior_max_cfg)
    warrior_count = max(warrior_count, warrior_min)
    # Финальная подстройка зелий под фактическое число воинов.
    potion_count = max(0, TARGET_NON_HERO - warrior_count)

    chosen_warriors = rng.sample(catalog.warriors, warrior_count)
    warriors: List[int] = []
    for w in chosen_warriors:
        warriors.extend([w] * warrior_copies)
    rng.shuffle(warriors)

    potions: List[int] = []
    if catalog.potions and potion_count > 0:
        chosen_potions = rng.sample(catalog.potions, potion_count)
        for p in chosen_potions:
            potions.extend([p] * potion_copies)
        rng.shuffle(potions)

    return [hero, *warriors, *potions]


def parse_custom_deck(payload: Any) -> List[int]:
    """Парсит JSON колоды от пользователя.

    Поддерживает форматы:
      - список int:  [1, 14, 14, 15, ...]
      - dict:        {"hero": 1, "cards": [14, 14, 15, 8], "name": "..."}
    Возвращает плоский список card_id.
    """
    if isinstance(payload, list):
        ids = [int(x) for x in payload]
    elif isinstance(payload, dict):
        ids = []
        if "hero" in payload:
            ids.append(int(payload["hero"]))
        if "cards" in payload:
            ids.extend(int(x) for x in payload["cards"])
        if not ids and "deck" in payload:
            ids.extend(int(x) for x in payload["deck"])
    else:
        raise ValueError(f"unsupported deck payload type: {type(payload).__name__}")

    if not ids:
        raise ValueError("deck is empty")
    return ids


def validate_deck(
    deck_ids: Iterable[int],
    catalog: CardCatalog,
    *,
    require_hero: bool = True,
    min_size: int = 6,
    max_size: int = 30,
) -> Tuple[bool, str]:
    """Проверяет базовую валидность колоды против каталога."""
    ids = list(deck_ids)
    if len(ids) < min_size:
        return False, f"deck too small: {len(ids)} < {min_size}"
    if len(ids) > max_size:
        return False, f"deck too large: {len(ids)} > {max_size}"

    for cid in ids:
        if cid not in catalog.cards:
            return False, f"unknown card_id: {cid}"

    heroes = [cid for cid in ids if catalog.card(cid).get("card_type") == "hero"]
    if require_hero and len(heroes) != 1:
        return False, f"deck must contain exactly 1 hero, got {len(heroes)}"
    if not require_hero and len(heroes) > 1:
        return False, f"deck must contain at most 1 hero, got {len(heroes)}"

    return True, "ok"


def deck_summary(deck_ids: List[int], catalog: CardCatalog) -> Dict[str, Any]:
    """Краткое текстовое описание колоды для UI и манифеста."""
    counts: Dict[str, int] = {"hero": 0, "warrior": 0, "potion": 0}
    name_counts: Dict[str, int] = {}
    for cid in deck_ids:
        ctype = catalog.card(cid).get("card_type", "warrior")
        counts[ctype] = counts.get(ctype, 0) + 1
        cname = catalog.card(cid).get("name", f"card_{cid}")
        name_counts[cname] = name_counts.get(cname, 0) + 1
    return {"counts": counts, "cards": name_counts, "size": len(deck_ids)}


__all__ = [
    "DEFAULT_CARDS_PATH",
    "DEFAULT_DECK_SIZE",
    "CardCatalog",
    "load_catalog",
    "build_random_arena_deck",
    "parse_custom_deck",
    "validate_deck",
    "deck_summary",
]
