from datetime import datetime, timezone
from pathlib import Path

from infrastructure.database import Database


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "webapp" / "index.html"
ARENA_JS = ROOT / "webapp" / "arena.js"
ARENA_CSS = ROOT / "webapp" / "arena-styles.css"
SERVER = ROOT / "web" / "server.py"
DATABASE = ROOT / "infrastructure" / "database.py"


def _fake_db(fetch_returns=None):
    db = Database.__new__(Database)
    db._pool = True
    if fetch_returns is not None:
        db.fetch = fetch_returns
    return db


def _row(**kw):
    class F:
        def __init__(self, d):
            self._d = d
            for k, v in d.items():
                setattr(self, k, v)

        def get(self, key, default=None):
            return self._d.get(key, default)

        def __getitem__(self, key):
            return self._d[key]

    return F(kw)


def _run(coro):
    import asyncio

    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)


def test_extra_pass_personalization_settings_are_server_backed():
    db_source = DATABASE.read_text(encoding="utf-8")
    server_source = SERVER.read_text(encoding="utf-8")

    assert "nickname_glow_disabled BOOLEAN NOT NULL DEFAULT false" in db_source
    assert "hide_player_id_public BOOLEAN NOT NULL DEFAULT false" in db_source
    assert '"nickname_glow_disabled"' in db_source
    assert '"hide_player_id_public"' in db_source
    assert '"nickname_glow_disabled": settings_record.get("nickname_glow_disabled", False)' in server_source
    assert '"hide_player_id_public": settings_record.get("hide_player_id_public", False)' in server_source


def test_battle_history_exposes_opponent_pass_and_public_id_flags():
    calls = 0

    async def mock_fetch(query, *args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                _row(
                    match_id="m-premium",
                    p1_user_id=101,
                    p2_user_id=202,
                    winner_user_id=101,
                    loser_user_id=202,
                    p1_trophy_change=25,
                    p2_trophy_change=-10,
                    game_mode="classic",
                    match_type=None,
                    duration_seconds=120,
                    turns_count=5,
                    created_at=datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
                )
            ]
        if calls == 2:
            return []
        if calls == 3:
            return [
                _row(
                    user_id=202,
                    first_name="Ultra",
                    username="ultra_user",
                    is_bot=False,
                    trophies=900,
                    custom_nickname="UltraName",
                    img=None,
                    equipped_avatar_url="/avatar.png",
                    extra_pass="ultra",
                    nickname_glow_disabled=True,
                    hide_player_id_public=True,
                )
            ]
        return []

    history = _run(_fake_db(fetch_returns=mock_fetch).get_battle_history(101))

    assert history[0]["opponent_extra_pass"] == "ultra"
    assert history[0]["opponent_nickname_glow_disabled"] is True
    assert history[0]["opponent_hide_player_id_public"] is True


def test_squads_payload_includes_member_pass_and_public_id_flags():
    db_source = DATABASE.read_text(encoding="utf-8")
    member_query = db_source.split("async def get_clan_members", 1)[1].split(
        "async def get_member_role",
        1,
    )[0]

    assert "COALESCE(u.extra_pass, 'inactive') AS extra_pass" in member_query
    assert "COALESCE(us.nickname_glow_disabled, FALSE) AS nickname_glow_disabled" in member_query
    assert "COALESCE(us.hide_player_id_public, FALSE) AS hide_player_id_public" in member_query
    assert "LEFT JOIN user_settings us ON us.user_id = cm.user_id" in member_query


def test_main_ui_uses_shared_premium_nickname_and_public_id_helpers():
    source = INDEX.read_text(encoding="utf-8")

    assert "const premiumNicknameClassName" in source
    assert "const PremiumNickname" in source
    assert "const PublicPlayerId" in source
    assert "nickname_glow_disabled" in source
    assert "hide_player_id_public" in source

    for marker in [
        "const TopBar =",
        "const RatingPlayerCard =",
        "const RatingRankRow =",
        "const PlayerRow =",
        "const SquadsScreen =",
        "const NewsCard =",
        "const IdeaCard =",
        "const PreBattleScreen =",
        "const BattleHistorySheet =",
        "const ProfileScreen =",
    ]:
        start = source.index(marker)
        block = source[start:start + 30000]
        assert "PremiumNickname" in block

    assert "PublicPlayerId player={friend}" in source
    assert "PublicPlayerId player={player}" in source


def test_arena_uses_tiered_premium_nickname_classes_for_both_players():
    js_source = ARENA_JS.read_text(encoding="utf-8")
    css_source = ARENA_CSS.read_text(encoding="utf-8")

    assert "function applyPremiumNicknameVisual" in js_source
    assert "applyPremiumNicknameVisual(nameText, playerState?.extra_pass" in js_source
    assert "applyPremiumNicknameVisual(nameText, opponentState?.extra_pass" in js_source
    assert "applyPremiumNicknameVisual(nameEl, profile?.extra_pass" in js_source
    assert ".premium-nickname.pass" in css_source
    assert ".premium-nickname.ultra" in css_source
