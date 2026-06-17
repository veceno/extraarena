import pytest
from datetime import datetime, timedelta, timezone

from infrastructure.database import Database
from web import server


def _reward(track_type, position, reward_type="coins", amount=100, ep_required=False):
    return {
        "id": position,
        "track_type": track_type,
        "position": position,
        "reward_type": reward_type,
        "reward_amount": amount,
        "reward_meta": None,
        "extra_pass_required": ep_required,
        "is_active": True,
    }


def test_extra_pass_payload_uses_season_track_config_without_filling_fake_positions():
    season = {
        "id": 7,
        "slug": "neon-rift",
        "name": "Неоновый разлом",
        "subtitle": "точный конфиг",
        "description": "Сезон из БД",
        "start_date": None,
        "end_date": None,
        "is_active": True,
        "max_stars": 99,
        "free_track_type": "bp_free",
        "pass_track_type": "custom_pass",
        "ultra_track_type": "custom_ultra",
        "pass_end_position": 12,
        "ultra_start_position": 13,
        "theme": {"accent": "#00ffee"},
    }
    profile = {"stars": 5, "extra_pass": "inactive"}
    tracks = {
        "bp_free": [_reward("bp_free", 2)],
        "custom_pass": [_reward("custom_pass", 7, ep_required=True)],
        "custom_ultra": [_reward("custom_ultra", 14, ep_required=True)],
    }

    payload = server._build_extra_pass_payload(
        profile=profile,
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"bp_free": set(), "custom_pass": set(), "custom_ultra": set()},
    )

    assert payload["season"]["name"] == "Неоновый разлом"
    assert payload["season"]["max_stars"] == 99
    assert [track["track_type"] for track in payload["tracks"]] == [
        "bp_free",
        "custom_pass",
        "custom_ultra",
    ]
    assert [tier["position"] for tier in payload["tiers"]] == [2, 7, 14]


def test_extra_pass_formula_progression_thresholds_and_stage_progress():
    season = server._normalize_extra_pass_season({
        "max_stars": 3,
        "stage_cost_min": 2,
        "stage_cost_growth": 1,
        "stage_cost_exponent": 1,
        "stage_cost_cap": 99,
    })

    progression = server._extra_pass_progression(season)
    progress = server._extra_pass_progress_for_stars(8, season)

    assert progression["stage_count"] == 3
    assert progression["stage_costs"] == [3, 4, 5]
    assert progression["stage_thresholds"] == [3, 7, 12]
    assert progression["total_required_stars"] == 12
    assert progress["unlocked_stage"] == 2
    assert progress["next_stage"] == 3
    assert progress["stars_to_next"] == 4
    assert progress["stage_percent"] == 20
    assert progress["season_percent"] == 67


def test_extra_pass_payload_uses_required_stars_instead_of_position():
    season = server._normalize_extra_pass_season({
        "max_stars": 3,
        "stage_cost_min": 2,
        "stage_cost_growth": 1,
        "stage_cost_exponent": 1,
        "stage_cost_cap": 99,
    })
    profile = {"stars": 8, "extra_pass": "active"}
    tracks = {
        "bp_free": [
            _reward("bp_free", 2),
            _reward("bp_free", 3),
        ],
        "bp_premium": [
            _reward("bp_premium", 3, amount=400, ep_required=True),
        ],
        "bp_ultra": [],
    }

    payload = server._build_extra_pass_payload(
        profile=profile,
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"bp_free": set(), "bp_premium": set(), "bp_ultra": set()},
    )

    stage2 = next(tier for tier in payload["tiers"] if tier["position"] == 2)
    stage3 = next(tier for tier in payload["tiers"] if tier["position"] == 3)

    assert payload["progress"]["unlocked_stage"] == 2
    assert payload["progress"]["stars_to_next"] == 4
    assert stage2["required_stars"] == 7
    assert stage2["tracks"]["free"]["available"] is True
    assert stage3["required_stars"] == 12
    assert stage3["tracks"]["free"]["progress_locked"] is True
    assert stage3["tracks"]["premium"]["progress_locked"] is True


def test_extra_pass_summary_counts_rewards_unlocked_by_buying_pass():
    season = server._normalize_extra_pass_season(None)
    profile = {"stars": 13, "extra_pass": "inactive"}
    tracks = {
        "bp_free": [
            _reward("bp_free", 1),
            _reward("bp_free", 2),
        ],
        "bp_premium": [
            _reward("bp_premium", 1, amount=400, ep_required=True),
            _reward("bp_premium", 5, amount=500, ep_required=True),
            _reward("bp_premium", 12, amount=600, ep_required=True),
        ],
        "bp_ultra": [
            _reward("bp_ultra", 41, amount=120, ep_required=True),
        ],
    }

    payload = server._build_extra_pass_payload(
        profile=profile,
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"bp_free": {1}, "bp_premium": set(), "bp_ultra": set()},
    )

    assert payload["summary"]["available_now"] == 1
    assert payload["summary"]["claimable_with_extra_pass"] == 1
    assert payload["summary"]["claimable_with_ultra"] == 0


def test_ultra_profile_unlocks_premium_and_ultra_custom_tracks():
    season = server._normalize_extra_pass_season({
        "free_track_type": "s2_free",
        "pass_track_type": "s2_pass",
        "ultra_track_type": "s2_ultra",
        "max_stars": 5,
        "pass_end_position": 3,
        "ultra_start_position": 4,
        "stage_cost_min": 1,
        "stage_cost_growth": 0,
        "stage_cost_exponent": 1,
        "stage_cost_cap": 1,
    })
    profile = {
        "stars": 10,
        "extra_pass": "ultra",
        "extra_pass_expires_at": datetime.now(timezone.utc) + timedelta(days=1),
    }
    tracks = {
        "s2_free": [_reward("s2_free", 1)],
        "s2_pass": [_reward("s2_pass", 2, ep_required=True)],
        "s2_ultra": [_reward("s2_ultra", 4, ep_required=True)],
    }

    payload = server._build_extra_pass_payload(
        profile=profile,
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"s2_free": set(), "s2_pass": set(), "s2_ultra": set()},
    )

    stage2 = next(tier for tier in payload["tiers"] if tier["position"] == 2)
    stage4 = next(tier for tier in payload["tiers"] if tier["position"] == 4)

    assert payload["access"]["mode"] == "ultra"
    assert stage2["tracks"]["premium"]["available"] is True
    assert stage2["tracks"]["premium"]["access_locked"] is False
    assert stage4["tracks"]["ultra"]["available"] is True
    assert stage4["tracks"]["ultra"]["access_locked"] is False
    assert payload["summary"]["claimable_with_extra_pass"] == 0
    assert payload["summary"]["claimable_with_ultra"] == 0


def test_expired_ultra_and_low_stars_report_distinct_lock_reasons():
    season = server._normalize_extra_pass_season({
        "max_stars": 5,
        "pass_end_position": 3,
        "ultra_start_position": 4,
        "stage_cost_min": 1,
        "stage_cost_growth": 0,
        "stage_cost_exponent": 1,
        "stage_cost_cap": 1,
    })
    tracks = {
        "bp_free": [],
        "bp_premium": [_reward("bp_premium", 2, ep_required=True)],
        "bp_ultra": [_reward("bp_ultra", 4, ep_required=True)],
    }

    expired_payload = server._build_extra_pass_payload(
        profile={
            "stars": 10,
            "extra_pass": "ultra",
            "extra_pass_expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"bp_free": set(), "bp_premium": set(), "bp_ultra": set()},
    )
    low_stars_payload = server._build_extra_pass_payload(
        profile={
            "stars": 1,
            "extra_pass": "ultra",
            "extra_pass_expires_at": datetime.now(timezone.utc) + timedelta(days=1),
        },
        season=season,
        tracks_by_type=tracks,
        claimed_by_type={"bp_free": set(), "bp_premium": set(), "bp_ultra": set()},
    )

    expired_premium = next(tier for tier in expired_payload["tiers"] if tier["position"] == 2)["tracks"]["premium"]
    low_star_ultra = next(tier for tier in low_stars_payload["tiers"] if tier["position"] == 4)["tracks"]["ultra"]

    assert expired_payload["access"]["mode"] == "inactive"
    assert expired_premium["access_locked"] is True
    assert expired_premium["lock_reason"] == "extra_pass_required"
    assert low_star_ultra["access_locked"] is False
    assert low_star_ultra["progress_locked"] is True
    assert low_star_ultra["lock_reason"] == "progress_required"


def test_configured_extra_pass_track_types_are_valid_reward_tracks():
    season = server._normalize_extra_pass_season({
        "free_track_type": "season_free",
        "pass_track_type": "season_pass",
        "ultra_track_type": "season_ultra",
    })

    assert server._reward_track_allowed("season_free", season) is True
    assert server._reward_track_allowed("season_pass", season) is True
    assert server._reward_track_allowed("season_ultra", season) is True
    assert server._reward_track_allowed("totally_unknown", season) is False


class _SeasonSeedDB(Database):
    def __init__(self):
        self._pool = object()
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return "INSERT 0 1"


@pytest.mark.asyncio
async def test_default_extra_pass_season_seed_supplies_all_insert_columns():
    db = _SeasonSeedDB()

    await db._seed_default_season()

    query, args = db.calls[0]
    assert "pass_end_position, ultra_start_position, theme" in query
    assert "$17, $18, $19::jsonb" in query
    assert len(args) == 19
