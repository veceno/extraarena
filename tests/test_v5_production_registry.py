from pathlib import Path

from infrastructure.config import (
    BOT_DIFFICULTY_ALIASES,
    BOT_MODEL_PROFILES,
    BOT_STRENGTH_TIERS,
)


def test_production_registry_contains_only_v4_micro_and_v5_family():
    assert tuple(BOT_MODEL_PROFILES) == (
        "extra-lr-v4-micro",
        "extra-lr-v5-lite",
        "extra-lr-v5",
        "extra-lr-v5-ultra",
    )

    assert BOT_MODEL_PROFILES["extra-lr-v4-micro"]["format"] == "train_v2_classic_v1"
    for name in ("extra-lr-v5-lite", "extra-lr-v5", "extra-lr-v5-ultra"):
        profile = BOT_MODEL_PROFILES[name]
        assert profile["format"] == "v5"
        assert profile["obs_dim"] == 7128
        assert profile["enemy_hand_known"] is True
        assert profile["enemy_deck_known"] is True
        assert profile["enemy_deck_order_known"] is True


def test_v5_ultra_shares_policy_and_enables_both_assists():
    v5 = BOT_MODEL_PROFILES["extra-lr-v5"]
    ultra = BOT_MODEL_PROFILES["extra-lr-v5-ultra"]

    assert v5["model_path"] == ultra["model_path"] == "ai/models/extra-lr-v5.onnx"
    assert v5["assembler_enabled"] is False
    assert v5["cardoptimum_enabled"] is False
    assert ultra["assembler_enabled"] is True
    assert ultra["assembler_model_path"].endswith("extra_lr_assembler_v1.onnx")
    assert ultra["cardoptimum_enabled"] is True
    assert ultra["cardoptimum_model_path"].endswith("extra_lr_cardoptimum_v1.onnx")


def test_metronome_is_enabled_for_every_production_model():
    for profile in BOT_MODEL_PROFILES.values():
        assert profile["metronome_enabled"] is True
        assert profile["metronome_model_path"].endswith("extra_lr_metronome_v1.onnx")


def test_trophy_road_keeps_tiers_but_moves_through_four_model_bands():
    assert tuple((tier["min_trophies"], tier["max_trophies"]) for tier in BOT_STRENGTH_TIERS) == (
        (0, 99),
        (100, 299),
        (300, 599),
        (600, 999),
        (1000, 1199),
        (1200, 1999),
        (2000, 2999),
        (3000, 4499),
        (4500, 5999),
        (6000, 7499),
        (7500, 8999),
        (9000, 1_000_000_000),
    )

    for tier in BOT_STRENGTH_TIERS:
        lo = tier["min_trophies"]
        expected = (
            "extra-lr-v4-micro"
            if lo < 300
            else "extra-lr-v5-lite"
            if lo < 1200
            else "extra-lr-v5"
            if lo < 4500
            else "extra-lr-v5-ultra"
        )
        assert tier["brain_profile"] == expected


def test_model_aliases_target_the_first_tier_of_each_band():
    assert BOT_DIFFICULTY_ALIASES["v4-micro"] == "tier_lite_0000"
    assert BOT_DIFFICULTY_ALIASES["v5-lite"] == "tier_easy_plus_0300"
    assert BOT_DIFFICULTY_ALIASES["v5"] == "tier_medium_1200"
    assert BOT_DIFFICULTY_ALIASES["v5-ultra"] == "tier_hard_4500"


def test_training_ui_exposes_four_models_and_defaults_to_v4_micro():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    difficulty_block = source.split("const DIFFICULTIES = [", 1)[1].split("];", 1)[0]

    assert difficulty_block.count("{id:") == 4
    assert "label:'V4 Micro'" in difficulty_block
    assert "label:'V5 Lite'" in difficulty_block
    assert "label:'V5'" in difficulty_block
    assert "label:'V5 Ultra'" in difficulty_block
    assert "React.useState('tier_lite_0000')" in source
