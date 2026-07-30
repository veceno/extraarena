#!/usr/bin/env python3
"""Render final ExtraLR V5 release benchmark charts as high-resolution PNGs.

The benchmark emits four correlated battles per seed (both policy seats and
both starting-player directions). Confidence intervals therefore resample
whole seed clusters rather than treating individual battles as independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

BACKGROUND = "#F7F8FA"
PANEL = "#FFFFFF"
INK = "#17212B"
MUTED = "#64748B"
GRID = "#D8DEE8"
BLUE = "#2F63B8"
BLUE_DARK = "#173B72"
GOLD = "#B78513"

DISPLAY_NAMES = {
    "extra-lr-v5-postB-preV5-u29250": "ExtraLR V5 post-B · u29250",
    "extra-lr-v5-postB-lite-preV5-u18500": "ExtraLR V5 Lite · u18500",
    "ExtraLR V5 (h299 no-assist)": "ExtraLR V5 · h299 no-assist",
    "extra-lr-v4-max": "ExtraLR V4 Max",
    "extra-lr-v4-opti": "ExtraLR V4 Opti",
    "extra-lr-v4-lite": "ExtraLR V4 Lite",
    "extra-lr-v4-micro": "ExtraLR V4 Micro",
    "extra-lr-v3-max": "ExtraLR V3 Max",
    "extra-lr-v3-medium": "ExtraLR V3 Medium",
    "OnlyVersusRandomBiggest": "OnlyVersusRandomBiggest",
    "greedy_face": "Greedy Face",
    "random": "Legal Random",
    "end_turn": "End Turn",
}

OPPONENT_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "ExtraLR V5 (h299 no-assist)",
            "extra-lr-v5-postB-preV5-u29250",
            "extra-lr-v5-postB-lite-preV5-u18500",
            "extra-lr-v4-max",
            "extra-lr-v4-opti",
            "extra-lr-v4-lite",
            "extra-lr-v4-micro",
            "extra-lr-v3-max",
            "extra-lr-v3-medium",
            "OnlyVersusRandomBiggest",
            "greedy_face",
            "random",
            "end_turn",
        )
    )
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def _candidate_name(payload: dict[str, Any]) -> str:
    ranked = [row["name"] for row in payload["models"] if row.get("ranked")]
    if len(ranked) != 1:
        raise ValueError(f"expected one ranked candidate, got {ranked}")
    return str(ranked[0])


def _score(row: dict[str, Any], candidate: str) -> float:
    if bool(row.get("draw")):
        return 0.5
    return 1.0 if row.get("winner_name") == candidate else 0.0


def _cluster_bootstrap_ci(
    cluster_scores: list[float],
    *,
    seed_key: str,
    repeats: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(cluster_scores, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    remaining = repeats
    while remaining:
        batch = min(250, remaining)
        indices = rng.integers(0, values.size, size=(batch, values.size))
        samples.append(values[indices].mean(axis=1))
        remaining -= batch
    distribution = np.concatenate(samples)
    return (
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def summarize_raw(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = _candidate_name(payload)
    results = list(payload["results"])
    errors = [row for row in results if row.get("error")]
    nonterminal = [
        row
        for row in results
        if row.get("timed_out")
        or row.get("truncated")
        or row.get("status") not in {"p1_win", "p2_win", "draw"}
    ]
    invalid = sum(
        int(row.get("invalid_actions", {}).get(candidate, 0))
        for row in results
    )
    if errors or nonterminal or invalid:
        raise ValueError(
            f"invalid benchmark artifact {path}: errors={len(errors)}, "
            f"nonterminal={len(nonterminal)}, invalid={invalid}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row["opponent_model"])].append(row)

    summaries: list[dict[str, Any]] = []
    for opponent, rows in groups.items():
        wins = sum(row.get("winner_name") == candidate for row in rows)
        draws = sum(bool(row.get("draw")) for row in rows)
        losses = len(rows) - wins - draws
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_seed[int(row["seed"])].append(_score(row, candidate))
        cluster_scores = [
            float(np.mean(scores)) for scores in by_seed.values()
        ]
        ci_low, ci_high = _cluster_bootstrap_ci(
            cluster_scores,
            seed_key=f"{path}:{candidate}:{opponent}",
        )
        summaries.append(
            {
                "opponent": opponent,
                "label": DISPLAY_NAMES.get(opponent, opponent),
                "games": len(rows),
                "seed_clusters": len(by_seed),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": (wins + 0.5 * draws) / len(rows),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summaries.sort(
        key=lambda row: (
            OPPONENT_ORDER.get(row["opponent"], 10_000),
            row["label"],
        )
    )
    return payload, summaries


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    size: int,
    color: str = INK,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(
        xy,
        value,
        fill=color,
        font=_font(size, bold=bold),
        anchor=anchor,
    )


def render_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    output: Path,
    footer: str,
) -> None:
    width = 2400
    row_height = 112
    top = 300
    bottom = 190
    height = top + len(rows) * row_height + bottom
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 42, width - 42, height - 42),
        radius=32,
        fill=PANEL,
        outline="#E4E8EF",
        width=2,
    )

    _text(draw, (100, 90), title, size=54, bold=True)
    _text(draw, (100, 166), subtitle, size=28, color=MUTED)

    label_right = 670
    plot_left = 700
    plot_right = 1780
    stats_left = 1825
    axis_y = 255
    for tick in (0, 25, 50, 75, 100):
        x = int(plot_left + (plot_right - plot_left) * tick / 100)
        color = GOLD if tick == 50 else GRID
        if tick == 50:
            for dash_y in range(axis_y, height - bottom + 10, 24):
                draw.line(
                    (x, dash_y, x, min(dash_y + 13, height - bottom + 10)),
                    fill=color,
                    width=4,
                )
        else:
            draw.line(
                (x, axis_y, x, height - bottom + 10),
                fill=color,
                width=2,
            )
        _text(
            draw,
            (x, axis_y - 18),
            f"{tick}%",
            size=22,
            color=GOLD if tick == 50 else MUTED,
            anchor="ms",
        )
    _text(
        draw,
        (int((plot_left + plot_right) / 2), axis_y - 62),
        "Score rate",
        size=23,
        color=MUTED,
        anchor="ms",
    )
    _text(
        draw,
        (stats_left, axis_y - 62),
        "Score [95% CI] · W–L–D · Battles",
        size=23,
        color=MUTED,
        anchor="ls",
    )

    for index, row in enumerate(rows):
        y = top + index * row_height
        center = y + 40
        _text(
            draw,
            (label_right, center),
            str(row["label"]),
            size=26,
            bold=True,
            anchor="rm",
        )
        score = float(row["score_rate"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        score_x = int(plot_left + (plot_right - plot_left) * score)
        low_x = int(plot_left + (plot_right - plot_left) * low)
        high_x = int(plot_left + (plot_right - plot_left) * high)
        draw.rounded_rectangle(
            (plot_left, center - 21, max(plot_left + 2, score_x), center + 21),
            radius=10,
            fill=BLUE,
        )
        draw.line((low_x, center, high_x, center), fill=BLUE_DARK, width=5)
        draw.line((low_x, center - 11, low_x, center + 11), fill=BLUE_DARK, width=4)
        draw.line((high_x, center - 11, high_x, center + 11), fill=BLUE_DARK, width=4)
        draw.ellipse(
            (score_x - 8, center - 8, score_x + 8, center + 8),
            fill=PANEL,
            outline=BLUE_DARK,
            width=3,
        )
        stats = (
            f"{score * 100:5.1f}% "
            f"[{low * 100:4.1f}–{high * 100:4.1f}]  ·  "
            f"{row['wins']}–{row['losses']}–{row['draws']}  ·  n={row['games']}"
        )
        _text(draw, (stats_left, center), stats, size=23, anchor="lm")

    footer_y = height - bottom + 58
    _text(draw, (100, footer_y), footer, size=23, color=MUTED)
    _text(
        draw,
        (100, footer_y + 45),
        "Whiskers: 95% bootstrap CI over whole seed clusters. Dashed gold reference: 50% parity.",
        size=22,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def parse_pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("pair must use LABEL=/path/to/raw.json")
    label, path = value.split("=", 1)
    return label, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    roster = sub.add_parser("roster")
    roster.add_argument("--raw", type=Path, required=True)
    roster.add_argument("--title", required=True)
    roster.add_argument("--subtitle", required=True)
    roster.add_argument("--output", type=Path, required=True)
    h2h = sub.add_parser("h2h")
    h2h.add_argument("--pair", action="append", type=parse_pair, required=True)
    h2h.add_argument("--title", required=True)
    h2h.add_argument("--subtitle", required=True)
    h2h.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "roster":
        payload, rows = summarize_raw(args.raw.resolve())
        seed_counts = sorted({int(row["seed_clusters"]) for row in rows})
        seed_count_label = (
            str(seed_counts[0])
            if len(seed_counts) == 1
            else f"{seed_counts[0]}–{seed_counts[-1]}"
        )
        render_chart(
            rows,
            title=args.title,
            subtitle=args.subtitle,
            output=args.output.resolve(),
            footer=(
                f"{seed_count_label} seed clusters per opponent × 4 symmetric "
                "seat/start cells · 50 cards · mana draw · level 4 · full information"
            ),
        )
        return

    pair_rows: list[dict[str, Any]] = []
    seed_counts: list[int] = []
    for label, path in args.pair:
        _payload, rows = summarize_raw(path)
        if len(rows) != 1:
            raise ValueError(f"H2H pair artifact must contain one opponent: {path}")
        row = dict(rows[0])
        row["label"] = label
        pair_rows.append(row)
        seed_counts.append(int(row["seed_clusters"]))
    render_chart(
        pair_rows,
        title=args.title,
        subtitle=args.subtitle,
        output=args.output.resolve(),
        footer=(
            f"{min(seed_counts)} seed clusters per pair × 4 symmetric "
            "seat/start cells · bar is score rate of the left-hand model"
        ),
    )


if __name__ == "__main__":
    main()
