from web import server


class _Hero:
    def __init__(self, health: int, max_health: int = 100):
        self.health = health
        self.max_health = max_health


class _State:
    def __init__(self, health: int = 100, max_health: int = 100):
        self.hero = _Hero(health, max_health)
        self.surrender_processed = False


def test_battle_star_awards_loss_normal_win_and_pvp_bonus():
    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=True,
        is_bot_match=False,
        turns_count=16,
        did_surrender=False,
        did_afk=False,
        winner_state=_State(60),
    )

    assert awards["winner"] == 4
    assert awards["loser"] == 1
    assert awards["special"] is False


def test_battle_star_awards_fast_win_caps_at_five():
    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=True,
        is_bot_match=False,
        turns_count=15,
        did_surrender=False,
        did_afk=False,
        winner_state=_State(60),
    )

    assert awards["winner"] == 5
    assert awards["loser"] == 1
    assert "fast_win" in awards["winner_reasons"]


def test_battle_star_awards_special_ten_for_clean_dominant_pvp_win():
    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=True,
        is_bot_match=False,
        turns_count=12,
        did_surrender=False,
        did_afk=False,
        winner_state=_State(80),
    )

    assert awards["winner"] == 10
    assert awards["loser"] == 1
    assert awards["special"] is True


def test_battle_star_awards_reads_legacy_wrapper_hero_hp():
    class _CoreHero:
        hp = 24
        max_hp = 30

    class _CoreState:
        hero = _CoreHero()

    class _Wrapper:
        _core = _CoreState()
        hero_hp = 24
        surrender_processed = False

    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=True,
        is_bot_match=False,
        turns_count=12,
        did_surrender=False,
        did_afk=False,
        winner_state=_Wrapper(),
    )

    assert awards["winner"] == 10
    assert awards["special"] is True


def test_battle_star_awards_no_fast_or_special_for_surrender_farming():
    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=True,
        is_bot_match=False,
        turns_count=8,
        did_surrender=True,
        did_afk=False,
        winner_state=_State(100),
    )

    assert awards["winner"] == 4
    assert awards["loser"] == 1
    assert awards["special"] is False
    assert "fast_win" not in awards["winner_reasons"]


def test_battle_star_awards_bot_fast_win_has_no_pvp_bonus_or_special():
    awards = server._calculate_battle_star_awards(
        winner_is_active_human=True,
        loser_is_active_human=False,
        is_bot_match=True,
        turns_count=8,
        did_surrender=False,
        did_afk=False,
        winner_state=_State(100),
    )

    assert awards["winner"] == 4
    assert awards["loser"] == 0
    assert awards["special"] is False
