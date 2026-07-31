#!/usr/bin/env python3
"""Build and verify the Android-only card artwork pack.

Canonical game artwork remains in ``DesignAssets``.  Android excludes the
original card directory and overlays this deterministic WebP pack instead.
``--check`` is intentionally read-only and fail-closed so a release cannot use
stale, incomplete, corrupt, oversized, or unexpectedly expanded generated
assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageOps, ImageStat


REPORT_SCHEMA = "extraarena_android_assets_v2"
REPORT_NAME = "asset-optimization-report.json"
CARD_WEBP_QUALITY = 92
MIN_CARD_PSNR_DB = 34.0
MAX_CARD_OUTPUT_BYTES = 30_000_000
MAX_CARD_OUTPUT_RATIO = 0.25
EXCLUDED_RELATIVE_PATHS = {
    Path("Cards.zip"),
    Path("Arena/Sounds/arena_theme_legacy_20260616.wav"),
    Path("Arena/Sounds/arena_theme_v2_loop.wav"),
}
EXCLUDED_DIRECTORY_NAMES = {"Cards copy"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IGNORED_CARD_FILES = {".DS_Store"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated pack without changing it",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_path(output: Path) -> Path:
    resolved = output.resolve()
    generated_output = (
        resolved.name == "optimizedExtraArenaAssets" and "generated" in resolved.parts
    )
    repository_output = (
        resolved.name == "optimized-assets" and resolved.parent.name == "android-app"
    )
    if not generated_output and not repository_output:
        raise ValueError(f"refusing unsafe generated output: {resolved}")
    return resolved


def reset_generated_output(output: Path) -> Path:
    resolved = validate_output_path(output)
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def normalized_image(path: Path) -> Image.Image:
    """Decode one image, apply EXIF orientation, and return owned RGB(A) pixels."""

    try:
        with Image.open(path) as opened:
            opened.load()
            transposed = ImageOps.exif_transpose(opened)
            has_alpha = "A" in transposed.getbands() or "transparency" in opened.info
            return transposed.convert("RGBA" if has_alpha else "RGB")
    except Exception as error:  # Pillow exposes several decoder-specific exceptions.
        raise ValueError(f"cannot decode image {path}: {error}") from error


def discover_source_cards(source: Path) -> dict[str, Path]:
    card_source = source / "Cards"
    if not card_source.is_dir():
        raise ValueError(f"missing source card directory: {card_source}")

    cards: dict[str, Path] = {}
    for path in sorted(card_source.iterdir(), key=lambda item: item.name):
        if path.name in IGNORED_CARD_FILES:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected source card entry: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported source card file: {path}")
        if not path.stem:
            raise ValueError(f"source card has an empty stem: {path}")
        if path.stem in cards:
            raise ValueError(
                f"duplicate source card stem {path.stem!r}: "
                f"{cards[path.stem].name}, {path.name}"
            )
        normalized_image(path).close()
        cards[path.stem] = path

    if not cards:
        raise ValueError(f"no source cards found in {card_source}")
    return cards


def psnr_db(source: Image.Image, encoded: Image.Image) -> float:
    if source.size != encoded.size:
        raise ValueError(
            f"image dimensions changed from {source.size} to {encoded.size}"
        )
    mode = "RGBA" if "A" in source.getbands() else "RGB"
    source_pixels = source.convert(mode)
    encoded_pixels = encoded.convert(mode)
    difference = ImageChops.difference(source_pixels, encoded_pixels)
    rms = ImageStat.Stat(difference).rms
    rmse = math.sqrt(sum(channel * channel for channel in rms) / len(rms))
    if rmse == 0:
        return 99.0
    return 20.0 * math.log10(255.0 / rmse)


def expected_exclusions() -> list[str]:
    return sorted(str(path) for path in EXCLUDED_RELATIVE_PATHS) + sorted(
        EXCLUDED_DIRECTORY_NAMES
    )


def build_assets(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    # Decode and validate every source before deleting the last known-good pack.
    source_cards = discover_source_cards(source)
    output = reset_generated_output(output)
    card_destination = output / "DesignAssets" / "Cards"
    card_destination.mkdir(parents=True, exist_ok=True)

    items: dict[str, dict[str, Any]] = {}
    source_bytes = 0
    output_bytes = 0
    minimum_psnr = 99.0
    for stem, path in sorted(source_cards.items()):
        target = card_destination / f"{stem}.webp"
        source_image = normalized_image(path)
        try:
            source_image.save(
                target,
                "WEBP",
                quality=CARD_WEBP_QUALITY,
                method=6,
                exact=True,
            )
            encoded_image = normalized_image(target)
            try:
                quality = round(psnr_db(source_image, encoded_image), 4)
                dimensions = encoded_image.size
            finally:
                encoded_image.close()
        finally:
            source_image.close()

        if quality < MIN_CARD_PSNR_DB:
            raise ValueError(
                f"card {path.name} PSNR {quality:.4f} dB is below "
                f"{MIN_CARD_PSNR_DB:.4f} dB"
            )

        source_size = path.stat().st_size
        target_size = target.stat().st_size
        source_bytes += source_size
        output_bytes += target_size
        minimum_psnr = min(minimum_psnr, quality)
        items[stem] = {
            "source_file": f"Cards/{path.name}",
            "source_sha256": sha256_file(path),
            "source_bytes": source_size,
            "output_file": f"DesignAssets/Cards/{target.name}",
            "output_sha256": sha256_file(target),
            "output_bytes": target_size,
            "width": dimensions[0],
            "height": dimensions[1],
            "psnr_db": quality,
        }

    output_ratio = round(output_bytes / source_bytes, 6)
    if output_bytes > MAX_CARD_OUTPUT_BYTES:
        raise ValueError(
            f"optimized cards use {output_bytes} bytes, budget is {MAX_CARD_OUTPUT_BYTES}"
        )
    if output_ratio > MAX_CARD_OUTPUT_RATIO:
        raise ValueError(
            f"optimized card ratio {output_ratio:.6f}, budget is "
            f"{MAX_CARD_OUTPUT_RATIO:.6f}"
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "card_webp_quality": CARD_WEBP_QUALITY,
        "card_dimensions_preserved": True,
        "only_card_assets": True,
        "cards": {
            "count": len(items),
            "source_bytes": source_bytes,
            "output_bytes": output_bytes,
            "saved_bytes": source_bytes - output_bytes,
            "output_ratio": output_ratio,
            "minimum_psnr_db": minimum_psnr,
            "items": items,
        },
        "quality_guard": {
            "minimum_psnr_db": MIN_CARD_PSNR_DB,
            "maximum_output_bytes": MAX_CARD_OUTPUT_BYTES,
            "maximum_output_ratio": MAX_CARD_OUTPUT_RATIO,
        },
        "output_bytes": output_bytes,
        "excluded": expected_exclusions(),
    }
    (output / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def discover_output_cards(output: Path, source_stems: set[str]) -> dict[str, Path]:
    report_path = output / REPORT_NAME
    cards_directory = output / "DesignAssets" / "Cards"
    if not report_path.is_file():
        raise ValueError(f"missing optimization report: {report_path}")
    if not cards_directory.is_dir():
        raise ValueError(f"missing optimized card directory: {cards_directory}")

    expected_directories = {Path("DesignAssets"), Path("DesignAssets/Cards")}
    output_cards: dict[str, Path] = {}
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if path.is_symlink():
            raise ValueError(f"unexpected output symlink: {relative}")
        if path.is_dir():
            if relative not in expected_directories:
                raise ValueError(f"unexpected output directory: {relative}")
            continue
        if relative == Path(REPORT_NAME):
            continue
        if path.parent != cards_directory or path.suffix.lower() != ".webp":
            raise ValueError(f"unexpected output file: {relative}")
        if path.name != f"{path.stem}.webp":
            raise ValueError(f"non-canonical optimized card name: {relative}")
        if path.stem in output_cards:
            raise ValueError(f"duplicate optimized card stem: {path.stem}")
        output_cards[path.stem] = path

    require_equal(set(output_cards), source_stems, "optimized card stems")
    return output_cards


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read optimization report {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("optimization report root must be an object")
    return report


def check_assets(source: Path, output: Path) -> dict[str, Any]:
    """Read-only validation of source freshness, output integrity, and budgets."""

    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output = validate_output_path(output)
    if not output.is_dir():
        raise ValueError(f"missing generated asset directory: {output}")

    source_cards = discover_source_cards(source)
    output_cards = discover_output_cards(output, set(source_cards))
    report = load_report(output / REPORT_NAME)

    require_equal(
        set(report),
        {
            "schema",
            "card_webp_quality",
            "card_dimensions_preserved",
            "only_card_assets",
            "cards",
            "quality_guard",
            "output_bytes",
            "excluded",
        },
        "report fields",
    )
    require_equal(report["schema"], REPORT_SCHEMA, "report schema")
    require_equal(report["card_webp_quality"], CARD_WEBP_QUALITY, "WebP quality")
    require_equal(report["card_dimensions_preserved"], True, "dimension policy")
    require_equal(report["only_card_assets"], True, "asset scope")
    require_equal(report["excluded"], expected_exclusions(), "excluded assets")
    require_equal(
        report["quality_guard"],
        {
            "minimum_psnr_db": MIN_CARD_PSNR_DB,
            "maximum_output_bytes": MAX_CARD_OUTPUT_BYTES,
            "maximum_output_ratio": MAX_CARD_OUTPUT_RATIO,
        },
        "quality guard",
    )

    cards_report = report["cards"]
    if not isinstance(cards_report, dict):
        raise ValueError("cards report must be an object")
    require_equal(
        set(cards_report),
        {
            "count",
            "source_bytes",
            "output_bytes",
            "saved_bytes",
            "output_ratio",
            "minimum_psnr_db",
            "items",
        },
        "cards report fields",
    )
    items = cards_report["items"]
    if not isinstance(items, dict):
        raise ValueError("card report items must be an object")
    require_equal(set(items), set(source_cards), "reported card stems")

    total_source_bytes = 0
    total_output_bytes = 0
    measured_psnr: list[float] = []
    expected_item_fields = {
        "source_file",
        "source_sha256",
        "source_bytes",
        "output_file",
        "output_sha256",
        "output_bytes",
        "width",
        "height",
        "psnr_db",
    }
    for stem, source_path in sorted(source_cards.items()):
        target = output_cards[stem]
        metadata = items[stem]
        if not isinstance(metadata, dict):
            raise ValueError(f"report entry for card {stem} must be an object")
        require_equal(set(metadata), expected_item_fields, f"card {stem} report fields")

        require_equal(
            metadata["source_file"], f"Cards/{source_path.name}", f"card {stem} source file"
        )
        require_equal(
            metadata["output_file"],
            f"DesignAssets/Cards/{target.name}",
            f"card {stem} output file",
        )
        require_equal(
            metadata["source_sha256"], sha256_file(source_path), f"card {stem} source SHA-256"
        )
        require_equal(
            metadata["output_sha256"], sha256_file(target), f"card {stem} output SHA-256"
        )
        require_equal(
            metadata["source_bytes"], source_path.stat().st_size, f"card {stem} source size"
        )
        require_equal(
            metadata["output_bytes"], target.stat().st_size, f"card {stem} output size"
        )

        try:
            with Image.open(target) as opened:
                require_equal(opened.format, "WEBP", f"card {stem} encoded format")
                opened.verify()
        except Exception as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(f"cannot decode optimized card {target}: {error}") from error

        source_image = normalized_image(source_path)
        encoded_image = normalized_image(target)
        try:
            require_equal(encoded_image.size, source_image.size, f"card {stem} dimensions")
            require_equal(metadata["width"], source_image.width, f"card {stem} width")
            require_equal(metadata["height"], source_image.height, f"card {stem} height")
            quality = round(psnr_db(source_image, encoded_image), 4)
        finally:
            source_image.close()
            encoded_image.close()
        require_equal(metadata["psnr_db"], quality, f"card {stem} PSNR")
        if quality < MIN_CARD_PSNR_DB:
            raise ValueError(
                f"card {stem} PSNR {quality:.4f} dB is below {MIN_CARD_PSNR_DB:.4f} dB"
            )

        total_source_bytes += source_path.stat().st_size
        total_output_bytes += target.stat().st_size
        measured_psnr.append(quality)

    output_ratio = round(total_output_bytes / total_source_bytes, 6)
    if total_output_bytes > MAX_CARD_OUTPUT_BYTES:
        raise ValueError(
            f"optimized cards use {total_output_bytes} bytes, budget is {MAX_CARD_OUTPUT_BYTES}"
        )
    if output_ratio > MAX_CARD_OUTPUT_RATIO:
        raise ValueError(
            f"optimized card ratio {output_ratio:.6f}, budget is "
            f"{MAX_CARD_OUTPUT_RATIO:.6f}"
        )

    require_equal(cards_report["count"], len(source_cards), "card count")
    require_equal(cards_report["source_bytes"], total_source_bytes, "source bytes")
    require_equal(cards_report["output_bytes"], total_output_bytes, "card output bytes")
    require_equal(
        cards_report["saved_bytes"], total_source_bytes - total_output_bytes, "saved bytes"
    )
    require_equal(cards_report["output_ratio"], output_ratio, "card output ratio")
    require_equal(cards_report["minimum_psnr_db"], min(measured_psnr), "minimum PSNR")
    require_equal(report["output_bytes"], total_output_bytes, "total output bytes")
    return report


def main() -> None:
    args = parse_args()
    if args.check:
        report = check_assets(args.source, args.output)
        print(
            f"asset pack OK: {report['cards']['count']} cards, "
            f"{report['cards']['output_bytes']} bytes"
        )
    else:
        report = build_assets(args.source, args.output)
        print(
            f"asset pack built: {report['cards']['count']} cards, "
            f"{report['cards']['output_bytes']} bytes"
        )


if __name__ == "__main__":
    main()
