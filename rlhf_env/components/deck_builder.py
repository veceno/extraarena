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

# Дефолтные размеры колоды (подобраны так, чтобы деки помещались в лимит руки 4-10)
DEFAULT_DECK_SIZE = {
    "warrior_unique_min": 5,
    "warrior_unique_max": 8,
    "warrior_copies": 2,
    "potion_unique_min": 1,
    "potion_unique_max": 3,
    "potion_copies": 2,
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
    """
    rng = rng or random.Random()
    cfg = {**DEFAULT_DECK_SIZE, **(config or {})}

    if not catalog.heroes:
        raise ValueError("cards catalog has no heroes")
    if not catalog.warriors:
        raise ValueError("cards catalog has no warriors")

    hero = hero_id if hero_id is not None else rng.choice(catalog.heroes)

    warrior_count = rng.randint(
        int(cfg["warrior_unique_min"]),
        min(int(cfg["warrior_unique_max"]), len(catalog.warriors)),
    )
    chosen_warriors = rng.sample(catalog.warriors, warrior_count)
    warriors: List[int] = []
    for w in chosen_warriors:
        warriors.extend([w] * int(cfg["warrior_copies"]))
    rng.shuffle(warriors)

    potions: List[int] = []
    if catalog.potions:
        max_potions = min(int(cfg["potion_unique_max"]), len(catalog.potions))
        if max_potions > 0:
            potion_count = rng.randint(int(cfg["potion_unique_min"]), max_potions)
            chosen_potions = rng.sample(catalog.potions, potion_count)
            for p in chosen_potions:
                potions.extend([p] * int(cfg["potion_copies"]))
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
