"""
Tests for catalog audit and profile harness (Task 03).
"""
import pytest

from ai.train_v2.catalog_audit import (
    load_current_catalog,
    audit_catalog,
    collect_catalog_mechanics,
)
from ai.train_v2.profile_env import benchmark_env


class TestCatalogAudit:
    def test_catalog_loads_and_normalizes_mechanics(self):
        catalog = load_current_catalog()
        assert isinstance(catalog, dict)
        assert len(catalog) > 0

        for cid, item in catalog.items():
            mechs = item.get("mechanics", [])
            assert isinstance(mechs, list), f"mechanics for id={cid} must be list, got {type(mechs)}"

    def test_audit_catalog_shape_and_no_errors(self):
        result = audit_catalog()
        assert "summary" in result
        assert "mechanics" in result
        assert "encoding" in result
        assert "warnings" in result
        assert "errors" in result

        enc = result["encoding"]
        assert enc["card_shape_dim"] == 64
        assert enc["obs_dim"] == 1456
        assert enc["action_feature_dim"] == 171
        assert enc["all_cards_encode"] is True, f"NaN/Inf cards: {enc['nan_or_inf_cards']}"

        assert result["errors"] == [], f"unexpected errors: {result['errors']}"

    def test_unknown_mechanics_are_reported_not_crashing(self):
        synthetic = {
            1: {"id": 1, "card_type": "hero", "mechanics": [], "name": "Hero", "base_attack": 0, "base_hp": 30, "mana_cost": 0, "rarity": "common"},
            2: {"id": 2, "card_type": "warrior", "mechanics": ["weird_new_mech_3"], "name": "Weirdo", "base_attack": 3, "base_hp": 4, "mana_cost": 3, "rarity": "common"},
        }
        result = audit_catalog(synthetic)
        mech = result["mechanics"]
        assert "weird_new_mech_3" in mech["unknown_families"], (
            f"weird_new_mech_3 should be in unknown_families, got {mech['unknown_families']}"
        )
        assert result["errors"] == [], f"unexpected errors: {result['errors']}"

    def test_unparsed_numeric_scalar_detection(self):
        synthetic = {
            1: {"id": 1, "card_type": "hero", "mechanics": [], "name": "Hero", "base_attack": 0, "base_hp": 30, "mana_cost": 0, "rarity": "common"},
            2: {"id": 2, "card_type": "warrior", "mechanics": ["heal_all_5"], "name": "HealAll", "base_attack": 0, "base_hp": 4, "mana_cost": 3, "rarity": "common"},
        }
        result = audit_catalog(synthetic)
        mech = result["mechanics"]
        assert "heal_all_5" in mech["unparsed_scalars"], (
            f"heal_all_5 should be in unparsed_scalars (numeric suffix, no scalar parser), "
            f"got {mech['unparsed_scalars']}"
        )

    def test_identity_leakage_smoke_check(self):
        result = audit_catalog()
        assert result["encoding"]["identity_leakage_checked"] is True


class TestProfileEnv:
    def test_benchmark_env_smoke(self):
        result = benchmark_env(episodes=3, seed=1)
        assert result["episodes"] == 3
        assert result["steps"] > 0
        assert result["turns"] > 0
        assert result["seconds"] > 0
        assert result["steps_per_sec"] > 0
        assert result["episodes_per_sec"] > 0
        assert result["avg_steps_per_episode"] > 0
        assert result["avg_turns_per_episode"] > 0
        assert result["reset_seconds"] >= 0
        assert result["mask_seconds"] > 0
        assert result["step_seconds"] > 0
        assert result["include_action_features"] is False

    def test_benchmark_env_with_action_features_smoke(self):
        result = benchmark_env(episodes=1, seed=1, include_action_features=True)
        assert result["episodes"] == 1
        assert result["features_seconds"] >= 0
        assert result["steps"] > 0
        assert result["include_action_features"] is True
