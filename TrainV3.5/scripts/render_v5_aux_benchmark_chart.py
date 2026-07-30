#!/usr/bin/env python3
"""Render the Phase-C all-model auxiliary A/B benchmark as a static PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DISPLAY_NAMES = {
    "extra-lr-v5-postB-preV5-u29250": "ExtraLR V5 postB-preV5 (u29250)",
    "extra-lr-v4-opti": "ExtraLR V4-opti",
    "extra-lr-v4-max": "ExtraLR V4-max",
    "extra-lr-v4-micro": "ExtraLR V4-micro",
    "extra-lr-v4-lite": "ExtraLR V4-lite",
    "extra-lr-v3-max": "ExtraLR V3-max",
    "extra-lr-v3-medium": "ExtraLR V3-medium",
    "OnlyVersusRandomBiggest": "OnlyVersusRandomBiggest",
    "greedy_face": "Greedy Face",
    "random": "Random",
    "end_turn": "End Turn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(path), size=size)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = sorted(
        payload["opponents"],
        key=lambda row: (row["with_aux"]["score_rate"], row["no_aux"]["score_rate"]),
    )

    labels = [DISPLAY_NAMES.get(row["opponent"], row["opponent"]) for row in rows]
    no_aux = [row["no_aux"]["score_rate"] * 100 for row in rows]
    with_aux = [row["with_aux"]["score_rate"] * 100 for row in rows]

    width, height = 2400, 1620
    image = Image.new("RGB", (width, height), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    title_font = _font(50, bold=True)
    subtitle_font = _font(25)
    label_font = _font(25)
    value_font = _font(22, bold=True)
    small_font = _font(22)
    small_bold = _font(22, bold=True)

    plot_left, plot_right = 680, 2260
    plot_top, plot_bottom = 260, 1315
    plot_width = plot_right - plot_left
    row_height = (plot_bottom - plot_top) / len(rows)
    bar_height = 27
    max_value = 106.0

    draw.rounded_rectangle(
        (70, 45, width - 70, height - 60),
        radius=28,
        fill="#ffffff",
        outline="#e2e6ef",
        width=2,
    )
    draw.text(
        (120, 95),
        "ExtraLR V5 Phase C: результат против всех моделей",
        font=title_font,
        fill="#172033",
    )
    draw.text(
        (120, 165),
        "256 боёв на соперника в каждом режиме · полная история, рука и колода игрока доступны",
        font=subtitle_font,
        fill="#667085",
    )

    for tick in range(0, 101, 10):
        x = plot_left + plot_width * tick / max_value
        draw.line((x, plot_top, x, plot_bottom), fill="#e1e5ed", width=2)
        text = str(tick)
        box = draw.textbbox((0, 0), text, font=small_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, plot_bottom + 20),
            text,
            font=small_font,
            fill="#667085",
        )

    base_index = next(
        i for i, row in enumerate(rows) if row["opponent"] == payload["base_v5"]
    )
    base_y = plot_top + row_height * base_index
    draw.rounded_rectangle(
        (100, base_y + 3, width - 105, base_y + row_height - 3),
        radius=14,
        fill="#fff7da",
    )

    for index, (label, no_value, with_value) in enumerate(
        zip(labels, no_aux, with_aux, strict=True)
    ):
        center_y = plot_top + row_height * (index + 0.5)
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (plot_left - 28 - (label_box[2] - label_box[0]), center_y - 16),
            label,
            font=label_font,
            fill="#273142",
        )

        for value, offset, color in (
            (no_value, -19, "#34495e"),
            (with_value, 19, "#3a86ff"),
        ):
            x_end = plot_left + plot_width * value / max_value
            y_center = center_y + offset
            draw.rounded_rectangle(
                (
                    plot_left,
                    y_center - bar_height / 2,
                    x_end,
                    y_center + bar_height / 2,
                ),
                radius=8,
                fill=color,
            )
            draw.text(
                (min(x_end + 12, plot_right - 46), y_center - 14),
                f"{value:.1f}",
                font=value_font,
                fill="#273142",
            )

    draw.text(
        (plot_left + plot_width / 2 - 120, plot_bottom + 65),
        "Score rate кандидата, %",
        font=subtitle_font,
        fill="#39445a",
    )

    legend_y = 1430
    draw.rounded_rectangle((125, legend_y, 167, legend_y + 28), radius=7, fill="#34495e")
    draw.text((182, legend_y - 2), "Phase C без суб-моделей", font=small_font, fill="#39445a")
    draw.rounded_rectangle((590, legend_y, 632, legend_y + 28), radius=7, fill="#3a86ff")
    draw.text((647, legend_y - 2), "Phase C + суб-модели", font=small_font, fill="#39445a")

    delta = rows[base_index]["delta_percentage_points"]
    draw.text(
        (1460, legend_y - 2),
        f"Главный H2H против u29250: +{delta:.1f} п.п.",
        font=small_bold,
        fill="#8a5a00",
    )
    draw.text(
        (125, 1505),
        "Всего: 5 632/5 632 терминальных боя; ошибок, таймаутов и невалидных действий — 0.",
        font=small_font,
        fill="#667085",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
