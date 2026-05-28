from pathlib import Path

from infrastructure.database import _normalize_poll_options, _parse_poll_expires_at


def test_normalize_poll_options_keeps_valid_choices_with_stable_ids():
    options = _normalize_poll_options([
        {"id": "10", "text": "Да"},
        {"text": "Нет"},
        "  Может быть  ",
        {"id": 99, "text": ""},
    ])

    assert options == [
        {"id": 10, "text": "Да"},
        {"id": 2, "text": "Нет"},
        {"id": 3, "text": "Может быть"},
    ]


def test_parse_poll_expires_at_accepts_iso_string():
    parsed = _parse_poll_expires_at("2026-05-26T23:24:24.425964+00:00")

    assert parsed.year == 2026
    assert parsed.tzinfo is not None


def test_news_feed_renders_poll_card_only_for_real_poll_attachment():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "hasPollAttachment" in source
    assert "hasPollAttachment(p)" in source
