from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "android-app" / "scripts" / "optimize_assets.py"
SOURCE = ROOT / "DesignAssets"
OUTPUT = ROOT / "android-app" / "optimized-assets"
REPORT = OUTPUT / "asset-optimization-report.json"

SPEC = importlib.util.spec_from_file_location("android_asset_optimizer", SCRIPT)
assert SPEC and SPEC.loader
optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer)


def test_committed_android_card_pack_is_fresh_and_decodable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(SOURCE),
            "--output",
            str(OUTPUT),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "asset pack OK:" in result.stdout


def test_android_card_pack_scope_exclusions_and_size_budget() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    source_cards = {
        path.stem
        for path in (SOURCE / "Cards").iterdir()
        if path.is_file() and path.suffix.lower() in optimizer.IMAGE_SUFFIXES
    }
    output_cards = {path.stem for path in (OUTPUT / "DesignAssets" / "Cards").glob("*.webp")}
    output_files = {path.relative_to(OUTPUT) for path in OUTPUT.rglob("*") if path.is_file()}
    expected_files = {Path("asset-optimization-report.json")} | {
        Path("DesignAssets/Cards") / f"{stem}.webp" for stem in source_cards
    }

    assert report["schema"] == optimizer.REPORT_SCHEMA
    assert report["only_card_assets"] is True
    assert report["excluded"] == optimizer.expected_exclusions()
    assert output_files == expected_files
    assert source_cards == output_cards == set(report["cards"]["items"])
    assert report["cards"]["count"] == len(source_cards)
    assert report["cards"]["output_bytes"] <= optimizer.MAX_CARD_OUTPUT_BYTES
    assert report["cards"]["output_ratio"] <= optimizer.MAX_CARD_OUTPUT_RATIO
    assert report["cards"]["minimum_psnr_db"] >= optimizer.MIN_CARD_PSNR_DB


def test_gradle_excludes_replaced_and_dead_design_assets() -> None:
    gradle = (ROOT / "android-app" / "app" / "build.gradle.kts").read_text(
        encoding="utf-8"
    )

    for exclusion in (
        'exclude("Cards/**")',
        'exclude("Cards copy/**")',
        'exclude("Cards.zip")',
        'exclude("ea_vendor/babel.min.js")',
        'exclude("Arena/Sounds/arena_theme_legacy_20260616.wav")',
        'exclude("Arena/Sounds/arena_theme_v2_loop.wav")',
    ):
        assert exclusion in gradle
    assert 'from(optimizedExtraArenaAssetsDir.resolve("DesignAssets/Cards"))' in gradle
    assert 'into("DesignAssets/Cards")' in gradle


def test_optimizer_check_fails_closed_for_stale_extra_and_corrupt_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tiny fixtures have packaging ratios unlike production artwork; this test
    # exercises integrity behavior while the committed pack exercises budgets.
    monkeypatch.setattr(optimizer, "MIN_CARD_PSNR_DB", 0.0)
    monkeypatch.setattr(optimizer, "MAX_CARD_OUTPUT_BYTES", 10_000_000)
    monkeypatch.setattr(optimizer, "MAX_CARD_OUTPUT_RATIO", 10.0)

    source = tmp_path / "source"
    cards = source / "Cards"
    cards.mkdir(parents=True)
    Image.new("RGB", (96, 96), (35, 70, 140)).save(cards / "alpha.png")
    Image.new("RGB", (96, 96), (180, 90, 35)).save(cards / "beta.jpg", quality=100)
    output = tmp_path / "android-app" / "optimized-assets"

    optimizer.build_assets(source, output)
    optimizer.check_assets(source, output)

    Image.new("RGB", (96, 96), (10, 20, 30)).save(cards / "alpha.png")
    with pytest.raises(ValueError, match="source SHA-256"):
        optimizer.check_assets(source, output)

    optimizer.build_assets(source, output)
    unexpected = output / "DesignAssets" / "unexpected.txt"
    unexpected.write_text("must not ship", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected output file"):
        optimizer.check_assets(source, output)
    unexpected.unlink()

    target = output / "DesignAssets" / "Cards" / "alpha.webp"
    target.write_bytes(b"not a decodable WebP")
    report = json.loads((output / optimizer.REPORT_NAME).read_text(encoding="utf-8"))
    metadata = report["cards"]["items"]["alpha"]
    metadata["output_sha256"] = optimizer.sha256_file(target)
    metadata["output_bytes"] = target.stat().st_size
    (output / optimizer.REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot decode optimized card"):
        optimizer.check_assets(source, output)
