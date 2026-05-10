from __future__ import annotations

from io import BytesIO
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw, ImageFont

from infrastructure.config import LEAGUE_CONFIG

FONT_DIR = Path(__file__).resolve().parents[1] / "DesignAssets" / "Font"

CARD_WIDTH = 600
CARD_HEIGHT = 340
THUMB_WIDTH = 300
THUMB_HEIGHT = 170
CORNER_RADIUS = 16
AVATAR_SIZE = 80
AVATAR_X = 50
AVATAR_Y = 70

T = {
    "bg":       (15, 10, 26),
    "bgCard":   (26, 16, 48),
    "purple1":  (45, 31, 82),
    "purple2":  (61, 42, 112),
    "purple3":  (91, 63, 160),
    "purpleHL": (124, 92, 191),
    "purpleLt": (156, 125, 224),
    "orange":   (245, 146, 30),
    "orangeD":  (217, 117, 16),
    "white":    (240, 236, 255),
    "grey1":    (196, 184, 232),
    "grey2":    (122, 111, 160),
    "grey3":    (74, 61, 106),
}


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"FuturaPT-{name}.ttf"), size)


async def _fetch_avatar_bytes(bot_token: str, user_id: int) -> bytes | None:
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
            async with session.get(url, params={"user_id": user_id, "limit": 1},
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not data.get("ok") or not data.get("result", {}).get("total_count", 0):
                    return None
                file_id = data["result"]["photos"][0][0]["file_id"]

            file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
            async with session.get(file_url, params={"file_id": file_id},
                                   timeout=aiohttp.ClientTimeout(total=5)) as file_resp:
                if file_resp.status != 200:
                    return None
                file_data = await file_resp.json()
                if not file_data.get("ok"):
                    return None
                file_path = file_data["result"]["file_path"]

            download_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
            async with session.get(download_url,
                                   timeout=aiohttp.ClientTimeout(total=5)) as dl_resp:
                if dl_resp.status == 200:
                    return await dl_resp.read()
    except Exception:
        pass
    return None


async def _fetch_avatar_from_url(img_url: str) -> bytes | None:
    if not img_url:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(img_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.read()
    except Exception:
        pass
    return None


def _make_circle_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    return mask


def _make_rounded_mask(w: int, h: int, radius: int) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    return mask


def _draw_avatar(draw: ImageDraw.Draw, img: Image.Image, avatar_bytes: bytes | None,
                 display_name: str) -> None:
    avatar_img: Image.Image
    if avatar_bytes:
        try:
            avatar_img = Image.open(BytesIO(avatar_bytes)).convert("RGBA")
        except Exception:
            avatar_img = _letter_avatar(display_name)
    else:
        avatar_img = _letter_avatar(display_name)

    avatar_img = avatar_img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)

    circle_mask = _make_circle_mask(AVATAR_SIZE)
    bordered = Image.new("RGBA", (AVATAR_SIZE + 6, AVATAR_SIZE + 6), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(bordered)
    border_draw.ellipse((0, 0, AVATAR_SIZE + 6, AVATAR_SIZE + 6),
                        fill=T["purpleHL"])
    bordered.paste(avatar_img, (3, 3), circle_mask)

    final_mask = _make_circle_mask(AVATAR_SIZE + 6)
    bordered.putalpha(final_mask)
    img.paste(bordered, (AVATAR_X - 3, AVATAR_Y - 3), bordered)


def _letter_avatar(display_name: str) -> Image.Image:
    letter = display_name.strip()[0].upper() if display_name else "E"
    av = Image.new("RGBA", (AVATAR_SIZE, AVATAR_SIZE), (0, 0, 0, 0))
    av_draw = ImageDraw.Draw(av)
    av_draw.ellipse((0, 0, AVATAR_SIZE, AVATAR_SIZE), fill=T["purple2"])
    try:
        font = _load_font("Bold", 34)
    except OSError:
        font = ImageFont.load_default()
    bbox = font.getbbox(letter)
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]
    av_draw.text(
        ((AVATAR_SIZE - lw) // 2, (AVATAR_SIZE - lh) // 2 - 2),
        letter, fill=T["white"], font=font,
    )
    return av


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _format_num(n: int) -> str:
    if n >= 1000:
        return f"{n:,}".replace(",", " ")
    return str(n)


async def render_profile_card(bot_token: str, dto: dict) -> bytes:
    img = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # gradient background: lighter top → darker bottom
    for y in range(CARD_HEIGHT):
        t = y / max(CARD_HEIGHT - 1, 1)
        r = int(T["bgCard"][0] + (T["bg"][0] - T["bgCard"][0]) * t)
        g = int(T["bgCard"][1] + (T["bg"][1] - T["bgCard"][1]) * t)
        b = int(T["bgCard"][2] + (T["bg"][2] - T["bgCard"][2]) * t)
        draw.line((0, y, CARD_WIDTH, y), fill=(r, g, b, 255))

    # top accent line
    draw.line((0, 3, CARD_WIDTH, 3), fill=T["purpleHL"])

    # avatar: Telegram Bot API → profiles.img → letter fallback
    avatar_bytes = await _fetch_avatar_bytes(bot_token, dto["user_id"])
    if not avatar_bytes and dto.get("img"):
        avatar_bytes = await _fetch_avatar_from_url(dto["img"])
    _draw_avatar(draw, img, avatar_bytes, dto["display_name"])

    text_x = AVATAR_X + AVATAR_SIZE + 25

    # display name
    try:
        name_font = _load_font("Bold", 26)
    except OSError:
        name_font = ImageFont.load_default()
    name = dto["display_name"]
    if len(name) > 24:
        name = name[:22] + "…"
    draw.text((text_x, 72), name, fill=T["white"], font=name_font)

    # title
    try:
        title_font = _load_font("Book", 16)
    except OSError:
        title_font = ImageFont.load_default()
    draw.text((text_x, 106), f"\U0001f4cb {dto['title']}", fill=T["grey1"], font=title_font)

    # league badge
    league = LEAGUE_CONFIG.get(dto["league"], LEAGUE_CONFIG[1])
    badge_rgb = _hex_to_rgb(league["color"])
    badge_text = f"{league['emoji']}  {league['name']}"
    try:
        badge_font = _load_font("Demi", 14)
    except OSError:
        try:
            badge_font = _load_font("Bold", 14)
        except OSError:
            badge_font = ImageFont.load_default()
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_w = badge_bbox[2] - badge_bbox[0] + 24
    badge_h = badge_bbox[3] - badge_bbox[1] + 12
    badge_x = text_x
    badge_y = 132

    # semi-transparent badge background
    badge_bg = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    badge_bg_draw = ImageDraw.Draw(badge_bg)
    badge_bg_draw.rounded_rectangle(
        (0, 0, badge_w - 1, badge_h - 1), radius=8,
        fill=(*badge_rgb, 40), outline=(*badge_rgb, 80),
    )
    img.paste(badge_bg, (badge_x, badge_y), badge_bg)

    draw.text((badge_x + 12, badge_y + 4), badge_text, fill=badge_rgb, font=badge_font)

    # separator line
    sep_y = 195
    draw.line((35, sep_y, CARD_WIDTH - 35, sep_y), fill=T["purple1"])

    # trophy section (left)
    try:
        big_font = _load_font("Bold", 42)
    except OSError:
        big_font = ImageFont.load_default()
    try:
        stat_label_font = _load_font("Book", 13)
    except OSError:
        stat_label_font = ImageFont.load_default()

    trophies_str = _format_num(dto["trophies"])
    draw.text((50, sep_y + 18), f"\U0001f3c6 {trophies_str}", fill=T["orange"], font=big_font)
    draw.text((50, sep_y + 68), "\u0442\u0440\u043e\u0444\u0435\u0435\u0432", fill=T["grey2"], font=stat_label_font)

    # max trophies
    try:
        small_font = _load_font("Book", 15)
    except OSError:
        small_font = ImageFont.load_default()
    draw.text(
        (50, sep_y + 92),
        f"\u2b06 {_format_num(dto['max_trophies'])} \u043c\u0430\u043a\u0441.",
        fill=T["grey1"], font=small_font,
    )

    # battle stats (right)
    col2_x = 310
    try:
        stats_val_font = _load_font("Bold", 22)
    except OSError:
        stats_val_font = ImageFont.load_default()

    # Battle count
    battle_icon = "\u2694\ufe0f"
    draw.text((col2_x, sep_y + 18), f"{battle_icon}  {_format_num(dto['battle_count'])}",
              fill=T["grey1"], font=stats_val_font)
    draw.text((col2_x + 5, sep_y + 48), "\u0431\u043e\u0451\u0432", fill=T["grey2"],
              font=stat_label_font)

    # Wins
    win_icon = "\u2705"
    draw.text((col2_x + 170, sep_y + 18), f"{win_icon}  {_format_num(dto['win_count'])}",
              fill=T["grey1"], font=stats_val_font)
    draw.text((col2_x + 175, sep_y + 48), "\u043f\u043e\u0431\u0435\u0434", fill=T["grey2"],
              font=stat_label_font)

    # Win rate
    if dto["battle_count"] > 0:
        winrate = round(dto["win_count"] / dto["battle_count"] * 100)
    else:
        winrate = 0
    draw.text(
        (col2_x + 170, sep_y + 78),
        f"\U0001f4ca  {winrate}%",
        fill=T["purpleLt"], font=_load_font("Bold", 20),
    )

    # footer branding
    try:
        footer_font = _load_font("Book", 11)
    except OSError:
        footer_font = ImageFont.load_default()
    footer_text = "ExtraCards  \u2022  Profile Card"
    fbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    fw = fbox[2] - fbox[0]
    draw.text(((CARD_WIDTH - fw) // 2, CARD_HEIGHT - 26), footer_text,
              fill=T["grey3"], font=footer_font)

    # small league dot
    dot_r = 4
    draw.ellipse(
        (CARD_WIDTH - 35 - dot_r * 2, 12, CARD_WIDTH - 35, 12 + dot_r * 2),
        fill=badge_rgb,
    )

    # rounded corners mask
    rounded_mask = _make_rounded_mask(CARD_WIDTH, CARD_HEIGHT, CORNER_RADIUS)
    img.putalpha(rounded_mask)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_thumbnail(main_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(main_bytes))
    thumb = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="PNG")
    return buf.getvalue()
