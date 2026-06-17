from web import server as web_server


def test_recent_opponents_query_excludes_bot_accounts():
    source = open("infrastructure/database.py", encoding="utf-8").read()
    start = source.index("async def get_recent_opponents")
    block = source[start:source.index("async def create_payment", start)]

    assert "JOIN users u ON u.user_id = l.opponent_id" in block
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in block


def test_public_battle_history_strips_bot_identity_markers():
    row = web_server._sanitize_public_battle_history_row({
        "battle_id": "battle-1",
        "opponent_id": 900000001,
        "opponent_name": "Shadow",
        "opponent_is_bot": True,
        "opponent_avatar_url": "/bot.png",
        "mode": "classic",
    })

    assert row["battle_id"] == "battle-1"
    assert row["opponent_name"] == "Shadow"
    assert row["opponent_id"] is None
    assert "opponent_is_bot" not in row


def test_public_battle_history_keeps_human_opponent_id_without_bot_marker():
    row = web_server._sanitize_public_battle_history_row({
        "battle_id": "battle-2",
        "opponent_id": 12345,
        "opponent_name": "Human",
        "opponent_is_bot": False,
        "mode": "classic",
    })

    assert row["opponent_id"] == 12345
    assert "opponent_is_bot" not in row


def test_recent_opponents_copy_describes_socially_available_players():
    source = open("webapp/index.html", encoding="utf-8").read()

    assert "Доступные соперники" in source
    assert "Здесь появятся игроки, которым можно отправить заявку." in source
    assert "После матчей здесь появятся игроки, которым можно отправить заявку." not in source
