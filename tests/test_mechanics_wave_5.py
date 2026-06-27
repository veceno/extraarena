"""
Тесты для пятой волны механик (карты 47-52, 2026-06-26):
  - aoe_silence / aoe_silence_all          (Солдатик)
  - team_wide_shield / team_wide_shield_all (Соул Гудман)
  - target_ally_max_hp_plus[_universal]_N  (Криста Ленц)
  - rebirth_N                               (Бан)
  - crime_and_punishment_N                  (Достоевский)
Плюс проверки масштабирования по уровню и краевые случаи.
"""
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.card_scaling import scale_card_by_level


def _hero(name="Hero", hp=30, mechanics=()):
    return CardInstance(
        instance_id=uuid4(),
        card_id=1,
        name=name,
        card_type=CardType.HERO,
        hp=hp,
        max_hp=hp,
        attack=0,
        mana_cost=0,
        mechanics=list(mechanics),
    )


def _unit(name, hp, attack=0, mechanics=(), is_ready=False, card_id=100, max_hp=None):
    return CardInstance(
        instance_id=uuid4(),
        card_id=card_id,
        name=name,
        card_type=CardType.WARRIOR,
        hp=hp,
        max_hp=hp if max_hp is None else max_hp,
        attack=attack,
        mana_cost=2,
        mechanics=list(mechanics),
        is_ready=is_ready,
    )


def make_state(
    p1_hero_mech=(),
    p2_hero_mech=(),
    p1_hero_hp=30,
    p2_hero_hp=30,
):
    p1 = PlayerState(
        user_id=1,
        is_bot=False,
        hero=_hero("Hero P1", p1_hero_hp, p1_hero_mech),
        mana=10,
        max_mana=10,
        hand=[],
        board=[],
        deck=[],
    )
    p2 = PlayerState(
        user_id=2,
        is_bot=False,
        hero=_hero("Hero P2", p2_hero_hp, p2_hero_mech),
        mana=10,
        max_mana=10,
        hand=[],
        board=[],
        deck=[],
    )
    return GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=1,
        turn_number=1,
        status=GameStatus.ONGOING,
    )


def _warrior(name, mechanics, mana=7, attack=2, hp=4):
    return CardInstance(
        instance_id=uuid4(),
        card_id=200,
        name=name,
        card_type=CardType.WARRIOR,
        hp=hp,
        max_hp=hp,
        attack=attack,
        mana_cost=mana,
        mechanics=list(mechanics),
        is_ready=False,
    )


def _potion(name, mechanics, mana=4):
    """Зелье-носитель механики — не занимает слот доски, обходит board_full.
    Используется в тестах, где нужно ≥5 союзников на доске до розыгрыша."""
    return CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name=name,
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=mana,
        mechanics=list(mechanics),
        is_ready=False,
    )


# ===========================================================================
# aoe_silence / aoe_silence_all
# ===========================================================================


def test_aoe_silence_strips_enemy_mechanics():
    """Солдатик: при постановке вражеские существа теряют механики (taunt/regen)."""
    state = make_state()
    env = ArenaEnvironment(state)

    taunt_unit = _unit("Taunt", hp=5, mechanics=["taunt"], card_id=101)
    regen_unit = _unit("Regen", hp=5, mechanics=["regen_2"], card_id=102)
    state.p2.board = [taunt_unit, regen_unit]

    state.p1.hand = [_warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, f"play should succeed: {err}"

    assert taunt_unit.mechanics == [], f"taunt unit must be silenced, got {taunt_unit.mechanics}"
    assert regen_unit.mechanics == [], f"regen unit must be silenced, got {regen_unit.mechanics}"
    # Урон не наносится — только механики снимаются.
    assert taunt_unit.hp == 5 and regen_unit.hp == 5


def test_aoe_silence_respects_limit_of_three():
    """aoe_silence снимает механики максимум с 3 врагов — 4-й сохраняет."""
    state = make_state()
    env = ArenaEnvironment(state)

    state.p2.board = [
        _unit("E1", hp=5, mechanics=["taunt"], card_id=101),
        _unit("E2", hp=5, mechanics=["regen_1"], card_id=102),
        _unit("E3", hp=5, mechanics=["armor_1"], card_id=103),
        _unit("E4", hp=5, mechanics=["reflect_1"], card_id=104),
    ]
    state.p1.hand = [_warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    silenced = [u for u in state.p2.board if u.mechanics == []]
    assert len(silenced) == 3, f"exactly 3 units silenced, got {len(silenced)}"
    # Порядок доски сохранён — первые 3 silenced, 4-й сохраняет механику.
    assert state.p2.board[3].mechanics == ["reflect_1"]


def test_aoe_silence_skips_units_without_mechanics():
    """Юниты без механик пропускаются и не расходуют лимит."""
    state = make_state()
    env = ArenaEnvironment(state)

    plain = _unit("Plain", hp=5, mechanics=[], card_id=101)
    buffed = _unit("Buffed", hp=5, mechanics=["taunt"], card_id=102)
    state.p2.board = [plain, buffed]
    state.p1.hand = [_warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    assert buffed.mechanics == []
    # plain остался без механик (как и был), hp не изменился.
    assert plain.mechanics == [] and plain.hp == 5


def test_aoe_silence_skip_does_not_consume_limit_slot():
    """Юнит без механик не расходует лимит: 1 plain + 4 с механиками →
    ровно 3 механических юнита silenced, 4-й сохраняет механику. Если бы plain
    занимал слот лимита, было бы silenced только 2 из 4 — тест ловит регрессию."""
    state = make_state()
    env = ArenaEnvironment(state)

    plain = _unit("Plain", hp=5, mechanics=[], card_id=101)
    mechs = [
        _unit("M1", hp=5, mechanics=["taunt"], card_id=102),
        _unit("M2", hp=5, mechanics=["regen_1"], card_id=103),
        _unit("M3", hp=5, mechanics=["armor_1"], card_id=104),
        _unit("M4", hp=5, mechanics=["reflect_1"], card_id=105),
    ]
    state.p2.board = [plain] + mechs
    state.p1.hand = [_warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    silenced = [u for u in mechs if u.mechanics == []]
    retained = [u for u in mechs if u.mechanics != []]
    assert len(silenced) == 3, f"exactly 3 mechanical units silenced (limit), got {len(silenced)}"
    assert len(retained) == 1, f"1 mechanical unit keeps its mechanic, got {len(retained)}"
    assert plain.mechanics == [] and plain.hp == 5, "plain unit untouched (no mechanic, no damage)"


def test_aoe_silence_strips_shield_and_keeps_frozen_flag():
    """Silence снимает щит (щит не спасает) но НЕ сбрасывает статус-флаги (is_frozen)."""
    state = make_state()
    env = ArenaEnvironment(state)

    enemy = _unit("Shielded", hp=5, mechanics=["shield", "taunt"], card_id=101)
    enemy.is_frozen = True
    state.p2.board = [enemy]
    state.p1.hand = [_warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    assert enemy.mechanics == [], "shield must be stripped by silence (shield does not save)"
    assert enemy.is_frozen is True, "status flags applied TO the card must persist"
    assert enemy.hp == 5, "silence deals no damage"


def test_aoe_silence_all_strips_every_enemy():
    """aoe_silence_all снимает механики со всех врагов без лимита."""
    state = make_state()
    env = ArenaEnvironment(state)

    state.p2.board = [_unit(f"E{i}", hp=5, mechanics=["taunt"], card_id=100 + i) for i in range(5)]
    state.p1.hand = [_warrior("Silencer All", ["aoe_silence_all"], mana=7, attack=4, hp=5)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    assert all(u.mechanics == [] for u in state.p2.board), "all 5 enemies must be silenced"


# ===========================================================================
# team_wide_shield / team_wide_shield_all
# ===========================================================================


def test_team_wide_shield_grants_shields_to_allies():
    """Соул Гудман: союзные юниты на поле получают одноразовый щит."""
    state = make_state()
    env = ArenaEnvironment(state)

    a = _unit("Ally A", hp=4, mechanics=[], card_id=101)
    b = _unit("Ally B", hp=4, mechanics=[], card_id=102)
    state.p1.board = [a, b]
    soul = _warrior("Соул Гудман", ["team_wide_shield"], mana=7, attack=2, hp=4)
    state.p1.hand = [soul]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=2))
    assert ok, err

    assert "shield" in a.mechanics and "shield" in b.mechanics
    # КАРТА-НОСИТЕЛЬ — защитник, не защищаемый: сама щит не получает.
    assert "shield" not in soul.mechanics, "source card must not shield itself"


def test_team_wide_shield_does_not_steal_slot_from_allies():
    """Регрессия self-exclusion: 4 союзника + Соул Гудман (warrior) на позиции 0.
    До фикса Соул вставал в board перед process_effects и забирал один из 3
    лимитных слотов — союзникам доставалось только 2 щита. После фикса ровно 3
    союзника защищены, Соул без щита, 1 союзник остался без щита."""
    state = make_state()
    env = ArenaEnvironment(state)

    allies = [_unit(f"A{i}", hp=4, mechanics=[], card_id=101 + i) for i in range(4)]
    state.p1.board = allies
    soul = _warrior("Соул Гудман", ["team_wide_shield"], mana=7, attack=2, hp=4)
    state.p1.hand = [soul]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert ok, err

    shielded_allies = [u for u in allies if "shield" in u.mechanics]
    assert len(shielded_allies) == 3, f"exactly 3 allies shielded (limit), got {len(shielded_allies)}"
    assert "shield" not in soul.mechanics, "source must not shield itself or steal a slot"
    assert any("shield" not in u.mechanics for u in allies), "one ally must remain unshielded"


def test_team_wide_shield_respects_limit_and_skips_shielded():
    """Уже защищённый юнит пропускается (не расходует лимит); лимит — 3 выдачи."""
    state = make_state()
    env = ArenaEnvironment(state)

    already = _unit("Already", hp=4, mechanics=["shield"], card_id=101)
    fresh = [_unit(f"F{i}", hp=4, mechanics=[], card_id=102 + i) for i in range(4)]
    state.p1.board = [already] + fresh
    # 5 союзников на доске + носитель-зелье (не занимает слот) — обходим board_full.
    state.p1.hand = [_potion("Shield Potion", ["team_wide_shield"], mana=7)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert ok, err

    shielded = [u for u in fresh if "shield" in u.mechanics]
    assert len(shielded) == 3, f"exactly 3 fresh allies shielded (limit), got {len(shielded)}"
    assert "shield" in already.mechanics, "already-shielded keeps its single shield"
    # Один из fresh остался без щита.
    assert any("shield" not in u.mechanics for u in fresh)


def test_team_wide_shield_all_covers_every_ally():
    """team_wide_shield_all даёт щит всем союзным юнитам без лимита."""
    state = make_state()
    env = ArenaEnvironment(state)

    state.p1.board = [_unit(f"A{i}", hp=4, mechanics=[], card_id=100 + i) for i in range(5)]
    state.p1.hand = [_potion("Shield All Potion", ["team_wide_shield_all"], mana=7)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert ok, err

    assert all("shield" in u.mechanics for u in state.p1.board)


def test_team_wide_shield_does_not_cover_hero():
    """Герой щитом не покрывается — механика про карты на поле."""
    state = make_state()
    env = ArenaEnvironment(state)

    state.p1.board = [_unit("Ally", hp=4, mechanics=[], card_id=101)]
    soul = _warrior("Соул Гудман", ["team_wide_shield"], mana=7, attack=2, hp=4)
    state.p1.hand = [soul]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=1))
    assert ok, err

    assert "shield" not in state.p1.hero.mechanics, "hero must not receive team-wide shield"
    assert "shield" not in soul.mechanics, "source card must not shield itself"


# ===========================================================================
# target_ally_max_hp_plus[_universal]_N
# ===========================================================================


def test_max_hp_plus_universal_buffs_ally_unit_without_healing():
    """Криста: +N к max_hp союзника, текущее hp НЕ меняется (нет лечения)."""
    state = make_state()
    env = ArenaEnvironment(state)

    ally = _unit("Ally", hp=2, max_hp=2, mechanics=[], card_id=101)
    state.p1.board = [ally]
    state.p1.hand = [_warrior("Криста Ленц", ["target_ally_max_hp_plus_universal_1"], mana=2, attack=1, hp=2)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=str(ally.instance_id), position=1))
    assert ok, err

    assert ally.max_hp == 3, f"max_hp must be 2+1=3, got {ally.max_hp}"
    assert ally.hp == 2, f"hp must NOT change (no heal), got {ally.hp}"


def test_max_hp_plus_universal_can_target_hero():
    """Универсальный вариант может нацеливаться на героя."""
    state = make_state(p1_hero_hp=30)
    env = ArenaEnvironment(state)

    state.p1.hand = [_warrior("Криста Ленц", ["target_ally_max_hp_plus_universal_2"], mana=2, attack=1, hp=2)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=str(state.p1.hero.instance_id), position=0))
    assert ok, err

    assert state.p1.hero.max_hp == 32, f"hero max_hp 30+2=32, got {state.p1.hero.max_hp}"
    assert state.p1.hero.hp == 30, "hero hp must not change (no heal)"


def test_max_hp_plus_non_universal_excludes_hero():
    """Обычный target_ally_max_hp_plus_N не предлагает героя как цель."""
    state = make_state()
    env = ArenaEnvironment(state)

    ally = _unit("Ally", hp=2, max_hp=2, mechanics=[], card_id=101)
    state.p1.board = [ally]

    targets = env._get_possible_targets(state.p1, state.p2, ["target_ally_max_hp_plus_1"])
    assert str(ally.instance_id) in targets
    assert str(state.p1.hero.instance_id) not in targets, "non-universal must exclude hero"


def test_max_hp_plus_non_universal_rejects_hero_target():
    """Попытка нацелиться на героя обычной версией — розыгрыш отклоняется."""
    state = make_state()
    env = ArenaEnvironment(state)

    state.p1.board = [_unit("Ally", hp=2, max_hp=2, mechanics=[], card_id=101)]
    state.p1.hand = [_warrior("Non-universal", ["target_ally_max_hp_plus_1"], mana=2, attack=1, hp=2)]

    ok, err = env.step(1, PlayCardAction(hand_index=0, target_id=str(state.p1.hero.instance_id), position=1))
    assert not ok, "play targeting hero with non-universal variant must fail"
    assert err == "target_not_found"


# ===========================================================================
# rebirth_N
# ===========================================================================


def test_rebirth_survives_first_lethal_and_is_consumed():
    """Бан: первый летальный удар — выживает с N HP, механика тратится одноразово."""
    state = make_state()
    env = ArenaEnvironment(state)

    ban = _unit("Бан", hp=7, max_hp=7, attack=3, mechanics=["rebirth_1"], card_id=101, is_ready=False)
    state.p1.board = [ban]
    atk1 = _unit("Atk1", hp=20, max_hp=20, attack=7, mechanics=[], card_id=110, is_ready=True)
    atk2 = _unit("Atk2", hp=20, max_hp=20, attack=7, mechanics=[], card_id=111, is_ready=True)
    state.p2.board = [atk1, atk2]

    # Передаём ход p2.
    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    # Первый летальный — rebirth спасает.
    ok, err = env.step(2, AttackAction(attacker_id=str(atk1.instance_id), target_id=str(ban.instance_id), target_is_hero=False))
    assert ok, err
    assert ban.hp == 1, f"rebirth must set hp=1, got {ban.hp}"
    assert "rebirth_1" not in ban.mechanics, "rebirth must be consumed"
    assert ban in state.p1.board, "saved unit stays on board"

    # Второй летальный — способности больше нет, юнит погибает.
    ok, err = env.step(2, AttackAction(attacker_id=str(atk2.instance_id), target_id=str(ban.instance_id), target_is_hero=False))
    assert ok, err
    assert ban not in state.p1.board, "unit must die on second lethal (no rebirth left)"


def test_rebirth_does_not_trigger_deathrattle_when_saving():
    """Спасённый rebirth юнит не считается умершим — deathrattle не активируется."""
    state = make_state()
    env = ArenaEnvironment(state)

    # Юнит с rebirth И deathrattle одновременно.
    victim = _unit("Victim", hp=1, max_hp=1, attack=0, mechanics=["rebirth_1", "deathrattle_aoe_damage_5"], card_id=101)
    state.p1.board = [victim]
    atk1 = _unit("Atk1", hp=20, max_hp=20, attack=1, mechanics=[], card_id=110, is_ready=True)
    atk2 = _unit("Atk2", hp=20, max_hp=20, attack=1, mechanics=[], card_id=111, is_ready=True)
    state.p2.board = [atk1, atk2]

    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    p2_hero_before = state.p2.hero.hp

    # Первый летальный — rebirth спасает, deathrattle НЕ срабатывает.
    ok, err = env.step(2, AttackAction(attacker_id=str(atk1.instance_id), target_id=str(victim.instance_id), target_is_hero=False))
    assert ok, err
    assert victim.hp == 1 and victim in state.p1.board
    assert "deathrattle_aoe_damage_5" in victim.mechanics, "deathrattle must remain (not triggered)"
    assert state.p2.hero.hp == p2_hero_before, "deathrattle must NOT fire on rebirth save"

    # Второй летальный — гибель, deathrattle срабатывает (5 урона герою p2).
    ok, err = env.step(2, AttackAction(attacker_id=str(atk2.instance_id), target_id=str(victim.instance_id), target_is_hero=False))
    assert ok, err
    assert victim not in state.p1.board, "unit must die on second lethal"
    assert state.p2.hero.hp == p2_hero_before - 5, f"deathrattle must deal 5 to p2 hero, got {state.p2.hero.hp}"


# ===========================================================================
# crime_and_punishment_N
# ===========================================================================


def test_crime_and_punishment_damages_opponent_hero_on_ally_death():
    """Достоевский: когда противник убивает вашу карту — его герой получает N урона."""
    state = make_state(p1_hero_mech=["crime_and_punishment_2"], p1_hero_hp=30, p2_hero_hp=30)
    env = ArenaEnvironment(state)

    victim = _unit("Victim", hp=1, max_hp=1, attack=0, mechanics=[], card_id=101)
    state.p1.board = [victim]
    attacker = _unit("Atk", hp=20, max_hp=20, attack=5, mechanics=[], card_id=110, is_ready=True)
    state.p2.board = [attacker]

    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    p2_hero_before = state.p2.hero.hp
    ok, err = env.step(2, AttackAction(attacker_id=str(attacker.instance_id), target_id=str(victim.instance_id), target_is_hero=False))
    assert ok, err

    assert victim not in state.p1.board, "victim must die"
    assert state.p2.hero.hp == p2_hero_before - 2, f"p2 hero must take 2 (cap), got {state.p2.hero.hp}"


def test_crime_and_punishment_fires_per_dead_unit():
    """Карание срабатывает за КАЖДУЮ погибшую карту — две смерти = 2*N урона."""
    state = make_state(p1_hero_mech=["crime_and_punishment_2"], p1_hero_hp=30, p2_hero_hp=30)
    env = ArenaEnvironment(state)

    state.p1.board = [
        _unit("V1", hp=1, max_hp=1, mechanics=[], card_id=101),
        _unit("V2", hp=1, max_hp=1, mechanics=[], card_id=102),
    ]
    # p2 разыгрывает зелье AOE, убивающее обоих.
    aoe_potion = CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name="AOE Potion",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=4,
        mechanics=["aoe_damage_2"],
        is_ready=False,
    )
    state.p2.hand = [aoe_potion]

    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    p2_hero_before = state.p2.hero.hp
    ok, err = env.step(2, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert ok, err

    assert state.p2.hero.hp == p2_hero_before - 4, f"two deaths => 4 dmg to p2 hero, got {state.p2.hero.hp}"


def test_crime_and_punishment_pierces_armor():
    """Урон кары игнорирует броню героя-убийцы (прямое снятие HP)."""
    state = make_state(
        p1_hero_mech=["crime_and_punishment_2"],
        p2_hero_mech=["armor_5"],
        p1_hero_hp=30,
        p2_hero_hp=30,
    )
    env = ArenaEnvironment(state)

    victim = _unit("Victim", hp=1, max_hp=1, attack=0, mechanics=[], card_id=101)
    state.p1.board = [victim]
    attacker = _unit("Atk", hp=20, max_hp=20, attack=5, mechanics=[], card_id=110, is_ready=True)
    state.p2.board = [attacker]

    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    ok, err = env.step(2, AttackAction(attacker_id=str(attacker.instance_id), target_id=str(victim.instance_id), target_is_hero=False))
    assert ok, err

    # armor_5 не должен reduzir урон кары: 30 - 2 = 28 (а не 30).
    assert state.p2.hero.hp == 28, f"cap must pierce armor (28), got {state.p2.hero.hp}"


def test_crime_and_punishment_no_friendly_fire_self_damage():
    """Своя карта гибнет без противника-убийцы — кара не должна бить собственного героя.
    В этом движке friendly fire отсутствует: карта владельца гибнет только от урона
    противника, поэтому кара всегда направлена на героя-врага. Проверяем, что гибель
    своей карты НЕ карает собственного героя (cap цель = opponent.hero)."""
    state = make_state(p1_hero_mech=["crime_and_punishment_2"], p1_hero_hp=30, p2_hero_hp=30)
    env = ArenaEnvironment(state)

    victim = _unit("Victim", hp=1, max_hp=1, attack=0, mechanics=[], card_id=101)
    state.p1.board = [victim]
    attacker = _unit("Atk", hp=20, max_hp=20, attack=5, mechanics=[], card_id=110, is_ready=True)
    state.p2.board = [attacker]

    ok, err = env.step(1, EndTurnAction())
    assert ok, err

    p1_hero_before = state.p1.hero.hp
    ok, err = env.step(2, AttackAction(attacker_id=str(attacker.instance_id), target_id=str(victim.instance_id), target_is_hero=False))
    assert ok, err

    # Достоевский (p1 hero) — владелец жертвы; его собственный hp не страдает от своей кары.
    assert state.p1.hero.hp == p1_hero_before, "owner hero must not be punished by own cap"


# ===========================================================================
# Масштабирование по уровню
# ===========================================================================


def test_rebirth_scales_with_level():
    """rebirth_N растёт на +1 каждые 2 уровня: lvl5 rebirth_1 -> rebirth_3."""
    ban = _unit("Бан", hp=7, max_hp=7, attack=3, mechanics=["rebirth_1"], card_id=50)
    ban.rarity = "mythic"
    scale_card_by_level(ban, 5)
    assert "rebirth_3" in ban.mechanics, f"lvl5 rebirth should be 3, got {ban.mechanics}"


def test_crime_and_punishment_scales_with_level():
    """crime_and_punishment_N (hero) растёт на +1 каждые 3 уровня: lvl7 cap_2 -> cap_4."""
    dost = _hero("Достоевский", hp=32, mechanics=["crime_and_punishment_2"])
    dost.rarity = "mythic"
    scale_card_by_level(dost, 7)
    assert "crime_and_punishment_4" in dost.mechanics, f"lvl7 cap should be 4, got {dost.mechanics}"
    assert dost.max_hp == 32 + (7 - 1) * 2, f"hero +2 HP/level, got {dost.max_hp}"


def test_max_hp_plus_scales_with_level():
    """target_ally_max_hp_plus_universal_N растёт на +1 каждые 2 уровня: lvl5 1 -> 3."""
    krista = _warrior("Криста Ленц", ["target_ally_max_hp_plus_universal_1"], mana=2, attack=1, hp=2)
    krista.rarity = "common"
    scale_card_by_level(krista, 5)
    assert "target_ally_max_hp_plus_universal_3" in krista.mechanics, f"lvl5 max_hp buff should be 3, got {krista.mechanics}"


def test_aoe_silence_and_team_shield_do_not_scale():
    """Бинарные механики (silence / team_shield) не масштабируются по уровню."""
    sol = _warrior("Солдатик", ["aoe_silence"], mana=7, attack=4, hp=5)
    sol.rarity = "legendary"
    scale_card_by_level(sol, 9)
    assert sol.mechanics == ["aoe_silence"], "aoe_silence must not scale"

    soul = _warrior("Соул Гудман", ["team_wide_shield"], mana=7, attack=2, hp=4)
    soul.rarity = "legendary"
    scale_card_by_level(soul, 9)
    assert soul.mechanics == ["team_wide_shield"], "team_wide_shield must not scale"