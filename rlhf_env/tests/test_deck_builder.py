"""Тесты deck_builder."""
from __future__ import annotations

import random

import pytest

from rlhf_env.components.deck_builder import (
    build_random_arena_deck,
    deck_summary,
    load_catalog,
    parse_custom_deck,
    validate_deck,
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_load_catalog(catalog):
    assert len(catalog.heroes) >= 1
    assert len(catalog.warriors) >= 5
    assert isinstance(catalog.cards, dict)


def test_build_random_arena_deck(catalog):
    rng = random.Random(42)
    deck = build_random_arena_deck(catalog, rng=rng)
    assert isinstance(deck, list)
    assert len(deck) >= 6
    # первая карта — герой
    hero = catalog.card(deck[0])
    assert hero["card_type"] == "hero"
    # остальные — воины или зелья
    for cid in deck[1:]:
        ctype = catalog.card(cid)["card_type"]
        assert ctype in {"warrior", "potion"}


def test_build_random_arena_deck_deterministic(catalog):
    deck1 = build_random_arena_deck(catalog, rng=random.Random(7))
    deck2 = build_random_arena_deck(catalog, rng=random.Random(7))
    assert deck1 == deck2


def test_parse_custom_deck_list():
    deck = parse_custom_deck([1, 14, 14, 15, 8])
    assert deck == [1, 14, 14, 15, 8]


def test_parse_custom_deck_dict():
    deck = parse_custom_deck({"hero": 1, "cards": [14, 15, 8]})
    assert deck == [1, 14, 15, 8]


def test_parse_custom_deck_invalid():
    with pytest.raises(ValueError):
        parse_custom_deck([])


def test_validate_deck_ok(catalog):
    rng = random.Random(1)
    deck = build_random_arena_deck(catalog, rng=rng)
    ok, msg = validate_deck(deck, catalog)
    assert ok, msg


def test_validate_deck_too_small(catalog):
    ok, msg = validate_deck([1, 14, 14], catalog)
    assert not ok
    assert "too small" in msg


def test_validate_deck_unknown_card(catalog):
    ok, msg = validate_deck([1, 14, 14, 99999, 8, 14], catalog)
    assert not ok
    assert "unknown card_id" in msg


def test_validate_deck_no_hero(catalog):
    # Все воины — нет hero
    warriors = [w for w in catalog.warriors[:8]]
    deck = [warriors[0]] + [c for c in warriors[1:] for _ in range(2)]
    ok, msg = validate_deck(deck, catalog)
    assert not ok
    assert "hero" in msg


def test_deck_summary(catalog):
    rng = random.Random(0)
    deck = build_random_arena_deck(catalog, rng=rng)
    s = deck_summary(deck, catalog)
    assert "counts" in s
    assert s["counts"]["hero"] == 1
    assert s["size"] == len(deck)
    assert s["counts"]["hero"] + s["counts"]["warrior"] + s["counts"].get("potion", 0) == len(deck)