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
