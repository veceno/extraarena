import time
import uuid

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import get_settings
from web import server


def _season(season_id, number, start, end, prefix):
    return {
        "id": season_id,
        "season_number": number,
        "name": f"Season {number}",
        "status": "scheduled",
        "start_date": start,
        "end_date": end,
        "free_track_type": f"{prefix}_free",
        "pass_track_type": f"{prefix}_premium",
        "ultra_track_type": f"{prefix}_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    }


def test_season_schedule_overview_flags_exact_handoff_and_reward_presence():
    seasons = [
        _season(1, 1, "2026-06-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00", "s1"),
        _season(2, 2, "2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", "s2"),
    ]

    overview = server._build_season_schedule_overview(
        seasons,
        {
            "s1_free": 45,
            "s1_premium": 40,
            "s1_ultra": 5,
            "s2_free": 0,
            "s2_premium": 40,
            "s2_ultra": 5,
        },
    )

    assert overview[0]["relation_to_next"]["status"] == "aligned"
    assert overview[0]["has_reward_tracks"] is True
    assert overview[1]["has_reward_tracks"] is False


def test_season_schedule_overview_flags_gap_between_seasons():
    seasons = [
        _season(1, 1, "2026-06-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00", "s1"),
        _season(2, 2, "2026-07-03T00:00:00+00:00", "2026-08-01T00:00:00+00:00", "s2"),
    ]

    overview = server._build_season_schedule_overview(seasons, {})

    assert overview[0]["relation_to_next"]["status"] == "gap"
    assert overview[0]["relation_to_next"]["days"] == 2


def test_extra_pass_json_import_maps_lanes_to_season_track_types():
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_7_free",
        "pass_track_type": "season_7_pass",
        "ultra_track_type": "season_7_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    })

    rows = server._normalize_reward_track_import_payload(
        {
            "free": [{"position": 1, "reward_type": "coins", "reward_amount": 150}],
            "premium": [{"position": 40, "type": "gems", "amount": 300}],
            "ultra": [{"position": 45, "reward_type": "card", "reward_amount": 1, "meta": {"rarity": ["epic"]}}],
        },
        season,
    )

    assert [row["track_type"] for row in rows] == [
        "season_7_free",
        "season_7_pass",
        "season_7_ultra",
    ]
    assert [row["extra_pass_required"] for row in rows] == [False, True, True]
    assert rows[2]["reward_meta"] == {"rarity": ["epic"]}


def test_extra_pass_json_import_accepts_case_rewards():
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_7_free",
        "pass_track_type": "season_7_pass",
        "ultra_track_type": "season_7_ultra",
        "max_stars": 45,
    })

    rows = server._normalize_reward_track_import_payload(
        {"free": [{"position": 10, "reward_type": "case", "reward_amount": 3}]},
        season,
    )

    assert rows == [
        {
            "track_type": "season_7_free",
            "position": 10,
            "reward_type": "case",
            "reward_amount": 3,
            "reward_meta": None,
            "extra_pass_required": False,
        }
    ]


def test_extra_pass_json_import_splits_random_and_specific_card_rewards():
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_7_free",
        "pass_track_type": "season_7_pass",
        "ultra_track_type": "season_7_ultra",
        "max_stars": 45,
    })

    rows = server._normalize_reward_track_import_payload(
        {
            "premium": [{"position": 2, "reward_type": "card", "reward_amount": 1, "meta": {"rarity": ["rare"]}}],
            "ultra": [{"position": 41, "reward_type": "specific_card", "reward_amount": 1, "meta": {"card_id": 46}}],
        },
        season,
    )

    assert rows[0]["reward_type"] == "card"
    assert rows[0]["reward_meta"] == {"rarity": ["rare"]}
    assert rows[1]["reward_type"] == "specific_card"
    assert rows[1]["reward_meta"] == {"card_id": 46}


@pytest.mark.parametrize(
    "row,error",
    [
        ({"position": 1, "reward_type": "coins", "reward_amount": 0}, "invalid_reward_amount"),
        ({"position": 1, "reward_type": "card", "reward_amount": 2, "meta": {"rarity": ["epic"]}}, "invalid_card_reward_amount"),
        ({"position": 1, "reward_type": "specific_card", "reward_amount": 1}, "specific_card_id_required"),
    ],
)
def test_extra_pass_json_import_rejects_invalid_reward_configs(row, error):
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_7_free",
        "pass_track_type": "season_7_pass",
        "ultra_track_type": "season_7_ultra",
        "max_stars": 45,
    })

    with pytest.raises(ValueError, match=error):
        server._normalize_reward_track_import_payload({"free": [row]}, season)


class _CardValidationDB:
    def __init__(self, card):
        self.card = card

    async def get_card_info(self, card_id):
        return self.card if int(card_id) == 46 else None


@pytest.mark.asyncio
async def test_admin_reward_validation_requires_existing_warrior_specific_card():
    assert await server._admin_validate_reward_track_config(
        _CardValidationDB({"id": 46, "card_type": "warrior"}),
        {"reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}},
    ) is None
    assert await server._admin_validate_reward_track_config(
        _CardValidationDB(None),
        {"reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}},
    ) == "specific_card_not_found"
    assert await server._admin_validate_reward_track_config(
        _CardValidationDB({"id": 46, "card_type": "hero"}),
        {"reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}},
    ) == "specific_card_must_be_warrior"


def test_extra_pass_json_import_rejects_row_level_stage_costs():
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_7_free",
        "pass_track_type": "season_7_pass",
        "ultra_track_type": "season_7_ultra",
        "max_stars": 45,
    })

    with pytest.raises(ValueError, match="stage_cost_formula_owned"):
        server._normalize_reward_track_import_payload(
            {
                "free": [
                    {
                        "position": 1,
                        "reward_type": "coins",
                        "reward_amount": 150,
                        "required_stars": 25,
                    }
                ]
            },
            season,
        )


class _SeasonImportExtraIDDB:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": 101}


class _SeasonImportDB:
    def __init__(self):
        self.replaced_reward_tracks = []
        self.season = _season(
            1,
            1,
            "2026-06-01T00:00:00+00:00",
            "2026-07-01T00:00:00+00:00",
            "s1",
        )

    async def is_admin(self, user_id):
        return int(user_id) == 101

    async def get_season_by_id(self, season_id):
        return self.season if int(season_id) == 1 else None

    async def get_seasons(self):
        return [self.season]

    async def get_all_reward_tracks(self):
        return []

    async def get_season_reset_summaries(self):
        return {}

    async def replace_reward_tracks(self, track_types, rows):
        self.replaced_reward_tracks.append((list(track_types), list(rows)))
        return []


@pytest.mark.asyncio
async def test_admin_season_rewards_import_rejects_empty_replace(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", "season-import-jwt-secret-that-is-long-enough-2026")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "season-import-admin-session-secret-2026")
    get_settings.cache_clear()
    session_id = str(uuid.uuid4())
    db = _SeasonImportDB()
    app = server.create_web_app(
        db,
        bot_token="bot-token",
        extraid_db=_SeasonImportExtraIDDB(session_id),
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    token = jwt.encode(
        {"user_id": 101, "session_id": session_id, "iat": int(time.time()), "exp": int(time.time()) + 600},
        get_settings().jwt_secret,
        algorithm="HS256",
    )
    try:
        response = await client.post(
            "/api/admin/seasons/1/rewards/import",
            headers={"Authorization": f"Bearer {token}"},
            json={"tracks": {}, "replace": True},
        )
        payload = await response.json()

        assert response.status == 400
        assert payload["error"] == "empty_reward_tracks"
        assert db.replaced_reward_tracks == []
    finally:
        await client.close()
        get_settings.cache_clear()
