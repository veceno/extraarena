from __future__ import annotations

from hashlib import md5
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.types import InlineQuery, InlineQueryResultPhoto

from bot.profile_card_renderer import render_profile_card, render_thumbnail
from infrastructure.database import Database

router = Router(name="inline")

GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated" / "inline" / "profile"


def _card_cache_key(dto: dict) -> str:
    stable = (
        f"{dto['user_id']}_{dto['trophies']}_{dto['max_trophies']}"
        f"_{dto['battle_count']}_{dto['win_count']}"
        f"_{dto['title']}_{dto['display_name']}"
    )
    digest = md5(stable.encode()).hexdigest()[:12]
    return f"profile_{digest}"


def register_inline_handlers(dp: Dispatcher, webapp_url: str, db: Database | None = None) -> None:

    base_url = webapp_url.rstrip("/")

    @router.inline_query()
    async def handle_inline_query(inline_query: InlineQuery, bot: Bot) -> None:
        query = (inline_query.query or "").strip()
        from_user = inline_query.from_user

        if query.lower() == "me":
            is_me = True
            target_id = from_user.id
        else:
            is_me = False
            try:
                target_id = int(query)
            except ValueError:
                await inline_query.answer([], cache_time=60, is_personal=True)
                return

        if not db or not from_user:
            await inline_query.answer([], cache_time=60)
            return

        dto = await db.get_public_player_card(target_id)
        if not dto:
            await inline_query.answer([], cache_time=60, is_personal=True)
            return

        cache_key = _card_cache_key(dto)
        card_path = GENERATED_DIR / f"{cache_key}.png"
        thumb_path = GENERATED_DIR / f"{cache_key}_thumb.png"

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        if not card_path.exists():
            png_bytes = await render_profile_card(bot.token, dto)
            card_path.write_bytes(png_bytes)
            thumb_bytes = render_thumbnail(png_bytes)
            thumb_path.write_bytes(thumb_bytes)

        photo_url = f"{base_url}/generated/inline/profile/{cache_key}.png"
        thumb_url = f"{base_url}/generated/inline/profile/{cache_key}_thumb.png"

        description = (
            f"{dto['league_emoji']} {dto['league_name']}"
            f"  |  \U0001f3c6 {dto['trophies']}"
            f"  |  \u2694 {dto['battle_count']} \u0431\u043e\u0451\u0432"
        )

        await inline_query.answer(
            [
                InlineQueryResultPhoto(
                    id=cache_key,
                    photo_url=photo_url,
                    thumbnail_url=thumb_url,
                    title=dto["display_name"],
                    description=description,
                    photo_width=600,
                    photo_height=340,
                )
            ],
            cache_time=60,
            is_personal=is_me,
        )

    dp.include_router(router)
