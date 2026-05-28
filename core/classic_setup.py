"""
Pure helper for classic match initialization.
Reusable by production (BattleEngine) and training (RL envs).
"""
from __future__ import annotations

import random as rand_mod
from typing import List
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState


def create_classic_game_state(
    p1_user_id: int,
    p2_user_id: int,
    p1_deck: List[CardInstance],
    p2_deck: List[CardInstance],
    *,
    p1_is_bot: bool = False,
    p2_is_bot: bool = False,
    p1_trophies: int = 0,
    p2_trophies: int = 0,
    starting_player_id: int | None = None,
    rng: rand_mod.Random | None = None,
) -> GameState:
    """
    Create a classic GameState from two decks.

    Returns a GameState WITHOUT having applied start-game hero effects.
    ArenaEnvironment(game_state) will apply them.
    """
    rng = rng or rand_mod.Random()

    _p1_deck = list(p1_deck)
    _p2_deck = list(p2_deck)

    p1_hero = _extract_hero(_p1_deck)
    p2_hero = _extract_hero(_p2_deck)
    if starting_player_id not in (p1_user_id, p2_user_id):
        starting_player_id = p1_user_id
    p1_starts = starting_player_id == p1_user_id

    p1 = PlayerState(
        user_id=p1_user_id,
        is_bot=p1_is_bot,
        hero=p1_hero,
        mana=1 if p1_starts else 0,
        max_mana=1 if p1_starts else 0,
        hand=[],
        board=[],
        deck=[],
        trophies=p1_trophies,
    )
    p2 = PlayerState(
        user_id=p2_user_id,
        is_bot=p2_is_bot,
        hero=p2_hero,
        mana=1 if not p1_starts else 0,
        max_mana=1 if not p1_starts else 0,
        hand=[],
        board=[],
        deck=[],
        trophies=p2_trophies,
    )

    _deal_starting_hand(p1, _p1_deck, rng)
    _deal_starting_hand(p2, _p2_deck, rng)

    return GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=starting_player_id,
        turn_number=1,
        status=GameStatus.ONGOING,
    )


def _extract_hero(deck: List[CardInstance]) -> CardInstance:
    for i, card in enumerate(deck):
        if card.card_type == CardType.HERO:
            return deck.pop(i)
    return CardInstance(
        instance_id=uuid4(),
        card_id=0,
        name="Герой",
        card_type=CardType.HERO,
        rarity="common",
        mana_cost=0,
        attack=0,
        hp=30,
        max_hp=30,
        mechanics=[],
        is_ready=True,
        description="Дефолтный герой",
    )


def _deal_starting_hand(
    player: PlayerState,
    deck: List[CardInstance],
    rng: rand_mod.Random,
) -> None:
    playable = [c for c in deck if c.card_type != CardType.HERO]
    warriors = [c for c in playable if c.card_type == CardType.WARRIOR]
    potions = [c for c in playable if c.card_type == CardType.POTION]

    warriors.sort(key=lambda c: c.mana_cost)

    hand_count = min(3, len(warriors))
    player.hand = warriors[:hand_count]
    player.deck = warriors[hand_count:] + potions
    rng.shuffle(player.deck)
