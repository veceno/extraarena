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
    assert len(deck) == 9
    assert len(set(deck)) == 9
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
    ok, msg = validate_deck([1, 14, 15, 16, 17, 18, 19, 20, 99999], catalog)
    assert not ok
    assert "unknown card_id" in msg


def test_validate_deck_no_hero(catalog):
    # Все воины — нет hero
    deck = catalog.warriors[:9]
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


def test_current_ruleset_catalog_is_exactly_50_cards_with_new_cards(catalog):
    assert len(catalog.cards) == 50
    assert set(range(47, 53)).issubset(catalog.cards)


def test_random_arena_decks_cover_every_new_card(catalog):
    seen = set()
    for seed in range(512):
        seen.update(build_random_arena_deck(catalog, rng=random.Random(seed)))
    assert set(range(47, 53)).issubset(seen)


def test_validate_deck_rejects_duplicates_and_non_prod_size(catalog):
    duplicate = [catalog.heroes[0], *catalog.warriors[:7], catalog.warriors[0]]
    ok, msg = validate_deck(duplicate, catalog)
    assert not ok and "duplicate" in msg
    ok, msg = validate_deck([catalog.heroes[0], *catalog.warriors[:5]], catalog)
    assert not ok and "too small" in msg
