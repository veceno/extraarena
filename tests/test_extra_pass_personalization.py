from datetime import datetime, timezone
from pathlib import Path

from infrastructure.database import Database
from web.admin_capabilities import ADMIN_CAPABILITIES


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "webapp" / "index.html"
ARENA_JS = ROOT / "webapp" / "arena.js"
ARENA_CSS = ROOT / "webapp" / "arena-styles.css"
SERVER = ROOT / "web" / "server.py"
DATABASE = ROOT / "infrastructure" / "database.py"


def _admin_tool_enum(tool_id, field):
    capability = next(cap for cap in ADMIN_CAPABILITIES if cap.id == tool_id)
    return tuple(capability.input_schema["properties"][field]["enum"])


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


def test_profile_cosmetic_class_meta_covers_catalog_rarities_and_unknown_fallback():
    source = INDEX.read_text(encoding="utf-8")
    meta_block = source.split("const COSMETIC_CLASS_META = {", 1)[1].split("};", 1)[0]
    picker_block = source.split("const CosmeticPickerSheet", 1)[1].split(
        "const ProfileScreen",
        1,
    )[0]
    profile_block = source.split("const ProfileScreen = ", 1)[1].split(
        "const ConnectionHealthOverlay",
        1,
    )[0]
    admin_rarities = set(_admin_tool_enum("admin.catalog.cards.create", "rarity"))
    admin_rarities.update(_admin_tool_enum("admin.catalog.items.create", "rarity"))

    for class_name in admin_rarities:
        assert f"{class_name}:" in meta_block
        assert f"{class_name}:" in picker_block

    assert "const getCosmeticClassMeta = (className) =>" in source
    assert "COSMETIC_CLASS_META[key] || COSMETIC_CLASS_META.starter" in source
    assert "const meta = getCosmeticClassMeta(sel?.class);" in profile_block
    assert "const bgClass = getCosmeticClassMeta(bgItem?.class);" in profile_block
    assert "const titleMeta = getCosmeticClassMeta(titleItem?.class);" in profile_block
    assert "const titleColor = getCosmeticClassMeta(titleRarity).color || accentColor;" in source

    assert "COSMETIC_CLASS_META[sel.class]" not in profile_block
    assert "COSMETIC_CLASS_META[bgItem.class]" not in profile_block
    assert "COSMETIC_CLASS_META[titleItem.class]" not in profile_block
    assert "titleRarity === 'epic'" not in source
    assert "titleRarity === 'limited'" not in source


def test_reward_and_arena_rarity_surfaces_cover_admin_catalog_rarities():
    source = INDEX.read_text(encoding="utf-8")
    arena_js = ARENA_JS.read_text(encoding="utf-8")
    arena_css = ARENA_CSS.read_text(encoding="utf-8")
    admin_rarities = set(_admin_tool_enum("admin.catalog.cards.create", "rarity"))
    admin_rarities.update(_admin_tool_enum("admin.catalog.items.create", "rarity"))

    case_block = source.split("const RARITY_COLORS = {", 1)[1].split(
        "const caseRewardSfxId",
        1,
    )[0]
    collection_block = source.split("const RARITY_COLOR = {", 1)[1].split(
        "const CARD_TYPE_LABEL",
        1,
    )[0]
    league_reward_block = source.split("const rewardRarityLabel = (rarity) =>", 1)[1].split(
        "const leagueRewardTitle",
        1,
    )[0]

    for rarity in admin_rarities:
        assert f"{rarity}:" in case_block
        assert f"{rarity}:" in collection_block
        assert f"{rarity}:" in league_reward_block
        assert f"'{rarity}'" in arena_js
        assert f"title-{rarity}" in arena_css
        assert f"avatar-class-{rarity}" in arena_css

    assert "function normalizeArenaRarity" in arena_js
    assert "function applyArenaTitleRarityClass" in arena_js
    assert "if (rarity === 'mythic'" not in arena_js
    assert "else if (rarity === 'limited')" not in arena_js


def test_arena_uses_tiered_premium_nickname_classes_for_both_players():
    js_source = ARENA_JS.read_text(encoding="utf-8")
    css_source = ARENA_CSS.read_text(encoding="utf-8")

    assert "function applyPremiumNicknameVisual" in js_source
    assert "applyPremiumNicknameVisual(nameText, playerState?.extra_pass" in js_source
    assert "applyPremiumNicknameVisual(nameText, opponentState?.extra_pass" in js_source
    assert "applyPremiumNicknameVisual(nameEl, profile?.extra_pass" in js_source
    assert ".premium-nickname.pass" in css_source
    assert ".premium-nickname.ultra" in css_source
