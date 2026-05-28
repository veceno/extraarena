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


def test_extra_pass_summary_counts_rewards_unlocked_by_buying_pass():
    season = server._normalize_extra_pass_season(None)
    profile = {"stars": 10, "extra_pass": "inactive"}
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
    assert payload["summary"]["claimable_with_extra_pass"] == 2
    assert payload["summary"]["claimable_with_ultra"] == 0


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
