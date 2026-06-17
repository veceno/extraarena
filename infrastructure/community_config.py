from __future__ import annotations

import os

# ── PolzaAI moderation ──────────────────────────────────────────────────────
MODERATION_API_BASE_URL: str = "https://polza.ai/api/v1"
MODERATION_MODEL: str = "google/gemini-2.5-flash-lite"
MODERATION_API_KEY: str = os.environ.get("POLZA_AI_KEY", "")

SUBMISSION_RATE_LIMIT: int = 3
SUBMISSION_RATE_WINDOW_MINUTES: int = 10

MODERATION_PROMPTS: dict[str, str] = {
    "IDEA": (
        "Ты модератор контента для игрового сообщества карточной игры ExtraArena. "
        "Проверь предложенную идею на уместность: отсутствие мата, спама, оскорблений, "
        "пропаганды, призывов к насилию. Идея должна быть связана с игровыми улучшениями. "
        "Если идея содержит нецензурные слова, оскорбления, спам или явно нерелевантный контент — "
        "отклони. Допускаются идеи любого рода об игровых механиках, балансе, фичах. "
        "Ответь строго JSON без лишнего текста: "
        '{\"decision\": \"approve\" | \"reject\", \"reason\": \"краткая причина на русском (только если reject)\"}'
    ),
    "BUG": (
        "Ты модератор контента для игрового сообщества ExtraArena. "
        "Проверь баг-репорт: отсутствие мата, спама, оскорблений. "
        "Баг-репорт должен описывать техническую проблему в игре. "
        "Если это явный спам, оскорбления или нерелевантный контент — отклони. "
        "Ответь строго JSON без лишнего текста: "
        '{\"decision\": \"approve\" | \"reject\", \"reason\": \"краткая причина на русском (только если reject)\"}'
    ),
    "ANNOUNCEMENT": (
        "Ты модератор объявлений сквадов в карточной игре ExtraArena. "
        "Проверь объявление сквада: отсутствие мата, спама, мошенничества, оскорблений. "
        "Объявление должно быть о скваде в игре: набор участников, турниры, требования. "
        "Запрещено: ссылки на сторонние ресурсы, личные данные, мошеннические схемы. "
        "Ответь строго JSON без лишнего текста: "
        '{\"decision\": \"approve\" | \"reject\", \"reason\": \"краткая причина на русском (только если reject)\"}'
    ),
    "SQUAD": (
        "Ты модератор сквадов в карточной игре ExtraArena. "
        "Проверь название, тег, описание и визуальную персонализацию сквада: отсутствие мата, "
        "оскорблений, дискриминации, сексуального контента, политической агитации, спама, "
        "мошенничества, ссылок на сторонние ресурсы и личных данных. "
        "Тег должен выглядеть как нейтральный игровой тег. Аватарка или фон, если есть, "
        "не должны содержать NSFW, насилие, hate symbols, рекламу или QR/контактные данные. "
        "Ответь строго JSON без лишнего текста: "
        '{\"decision\": \"approve\" | \"reject\", \"reason\": \"краткая причина на русском (только если reject)\"}'
    ),
}

# ── Announcement pricing ────────────────────────────────────────────────────
ANNOUNCE_BASE_COST: int = 500
ANNOUNCE_WORD_EXTRA_COST: int = 50        # per 100 words after first 100
ANNOUNCE_IMAGE_COST: int = 100
ANNOUNCE_DURATION_COSTS: dict[str, int] = {
    "1d": 0,
    "3d": 100,
    "7d": 300,
    "forever": 700,
}
ANNOUNCE_PIN_BASE_COST: int = 1500
ANNOUNCE_PIN_OVERBID_STEP: int = 500
ANNOUNCE_BOOST_DISCOUNT: float = 0.10

# ── Mock clan data (replace with real clans table query when ready) ─────────
MOCK_CLANS: list[dict] = [
    {
        "id": 1,
        "name": "Абсолютики",
        "owner_id": 1001,
        "trophies": 12500,
        "rank": 2,
        "members_count": 14,
        "max_members": 88,
        "has_boost": True,
    },
    {
        "id": 2,
        "name": "Phantom Legion",
        "owner_id": 1002,
        "trophies": 9800,
        "rank": 5,
        "members_count": 22,
        "max_members": 50,
        "has_boost": False,
    },
    {
        "id": 3,
        "name": "Dark Knights",
        "owner_id": 1003,
        "trophies": 7200,
        "rank": 12,
        "members_count": 35,
        "max_members": 50,
        "has_boost": True,
    },
    {
        "id": 4,
        "name": "Стражи Бездны",
        "owner_id": 1004,
        "trophies": 5100,
        "rank": 24,
        "members_count": 18,
        "max_members": 30,
        "has_boost": False,
    },
    {
        "id": 5,
        "name": "Neon Wolves",
        "owner_id": 1005,
        "trophies": 3400,
        "rank": 41,
        "members_count": 9,
        "max_members": 30,
        "has_boost": False,
    },
]


def calc_announcement_price(
    text: str,
    has_image: bool,
    duration_key: str,
    is_pinned: bool,
    has_boost: bool,
    pin_price: int | None = None,
) -> dict:
    """Calculate gem cost for an announcement. Returns full breakdown."""
    word_count = len(text.strip().split()) if text.strip() else 0
    word_extra = max(0, ((word_count - 100 + 99) // 100)) * ANNOUNCE_WORD_EXTRA_COST if word_count > 100 else 0
    image_extra = ANNOUNCE_IMAGE_COST if has_image else 0
    duration_extra = ANNOUNCE_DURATION_COSTS.get(duration_key, 0)
    pin_base = ANNOUNCE_PIN_BASE_COST if is_pinned else 0
    actual_pin_price = max(0, int(pin_price or pin_base)) if is_pinned else 0
    pin_extra = actual_pin_price if is_pinned else 0
    extra_pin = max(0, actual_pin_price - ANNOUNCE_PIN_BASE_COST) if is_pinned else 0
    subtotal = ANNOUNCE_BASE_COST + word_extra + image_extra + duration_extra + pin_base
    discount = int(subtotal * ANNOUNCE_BOOST_DISCOUNT) if has_boost else 0
    total = subtotal - discount + extra_pin
    return {
        "base": ANNOUNCE_BASE_COST,
        "word_extra": word_extra,
        "word_count": word_count,
        "image_extra": image_extra,
        "duration_extra": duration_extra,
        "pin_extra": pin_extra,
        "pin_base": pin_base,
        "pin_overbid_extra": extra_pin,
        "subtotal": subtotal,
        "discount": discount,
        "total": total,
    }
