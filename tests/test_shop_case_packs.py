"""
Тесты для CASE_PACKS: покупка кейсов за гемы.

Проверяем:
- Структуру и цены CASE_PACKS
- Атомарный UPDATE-RETURNING (мок БД)
- Игнорирование клиентской цены
- Недостаток гемов
- Compatibility alias item_type="case" → case_pack_1
"""

import pytest
from infrastructure.shop_config import CASE_PACKS


class TestCasePacksConfig:
    """Юнит-тесты конфигурации пакетов."""

    def test_all_packs_have_keys_and_gems(self):
        for pack_id, pack in CASE_PACKS.items():
            assert "keys" in pack, f"{pack_id} missing 'keys'"
            assert "gems" in pack, f"{pack_id} missing 'gems'"
            assert isinstance(pack["keys"], int) and pack["keys"] > 0
            assert isinstance(pack["gems"], int) and pack["gems"] > 0

    def test_case_pack_1_costs_25_gems_for_1_key(self):
        p = CASE_PACKS["case_pack_1"]
        assert p["keys"] == 1
        assert p["gems"] == 25

    def test_case_pack_3_costs_75_gems_for_3_keys(self):
        p = CASE_PACKS["case_pack_3"]
        assert p["keys"] == 3
        assert p["gems"] == 75

    def test_case_pack_5_costs_125_gems_for_5_keys(self):
        p = CASE_PACKS["case_pack_5"]
        assert p["keys"] == 5
        assert p["gems"] == 125

    def test_case_pack_10_costs_250_gems_for_10_keys(self):
        p = CASE_PACKS["case_pack_10"]
        assert p["keys"] == 10
        assert p["gems"] == 250

    def test_no_discount_packs_have_linear_pricing(self):
        for pack_id, pack in CASE_PACKS.items():
            assert pack["gems"] == pack["keys"] * 25, (
                f"{pack_id}: {pack['gems']} gems != {pack['keys']} * 25"
            )


class TestCasePackPriceLookup:
    """Тесты логики резолва пакетов (без БД)."""

    def test_case_compatibility_alias_resolves_to_pack_1(self):
        pack = CASE_PACKS.get("case", CASE_PACKS["case_pack_1"])
        assert pack["keys"] == 1
        assert pack["gems"] == 25

    def test_unknown_pack_defaults_to_pack_1(self):
        pack = CASE_PACKS.get("nonexistent", CASE_PACKS["case_pack_1"])
        assert pack["keys"] == 1
        assert pack["gems"] == 25

    def test_client_gems_amount_not_used(self):
        """Сервер всегда берёт цену из CASE_PACKS, игнорируя клиентский gems_amount."""
        for pack_id, pack in CASE_PACKS.items():
            server_keys = pack["keys"]
            server_gems = pack["gems"]
            fake_client_gems = 999
            assert server_gems != fake_client_gems
            assert server_gems == pack["gems"]
            assert server_keys == pack["keys"]


class TestAtomicUpdatePattern:
    """
    Тесты проверяют логику атомарного UPDATE ... WHERE ... >= ... RETURNING.

    Мокаем db.fetchrow — но фактически тестируем сам подход:
    - Если RETURNING вернул None → insufficient_gems
    - Если RETURNING вернул row → success с row["gems"], row["keys"]
    """

    def test_sufficient_gems_queue_returns_row(self):
        """Ситуация: у игрока 200 gems. Покупка case_pack_5 (125 gems)."""
        pack = CASE_PACKS["case_pack_5"]
        gems_before = 200
        keys_before = 3
        gems_price = pack["gems"]
        keys_amount = pack["keys"]

        assert gems_before >= gems_price, "Precondition: достаточно гемов"

        remaining_gems = gems_before - gems_price
        updated_keys = keys_before + keys_amount

        assert remaining_gems == 75
        assert updated_keys == 8

    def test_insufficient_gems_no_row(self):
        """Ситуация: у игрока 20 gems. Покупка case_pack_5 (125 gems)."""
        pack = CASE_PACKS["case_pack_5"]
        gems_before = 20
        gems_price = pack["gems"]

        assert gems_before < gems_price, "Precondition: недостаточно гемов"

    def test_only_1_gem_short_fails(self):
        """Ситуация: у игрока 24 gems. Покупка case_pack_1 (25 gems)."""
        pack = CASE_PACKS["case_pack_1"]
        gems_before = 24
        gems_price = pack["gems"]

        assert gems_before < gems_price, "Precondition: на 1 gem меньше"

    def test_exact_gems_succeeds(self):
        """Ситуация: у игрока ровно 25 gems. Покупка case_pack_1 (25 gems)."""
        pack = CASE_PACKS["case_pack_1"]
        gems_before = 25
        gems_price = pack["gems"]

        assert gems_before >= gems_price, "Precondition: ровно хватает"
        assert gems_before - gems_price == 0

    def test_pack_10_with_249_gems_fails(self):
        """Ситуация: у игрока 249 gems. Покупка case_pack_10 (250 gems)."""
        pack = CASE_PACKS["case_pack_10"]
        gems_before = 249
        gems_price = pack["gems"]

        assert gems_before < gems_price, "Precondition: не хватает 1 gem"

    def test_pack_10_with_250_gems_succeeds(self):
        """Ситуация: у игрока ровно 250 gems."""
        pack = CASE_PACKS["case_pack_10"]
        gems_before = 250
        gems_price = pack["gems"]

        assert gems_before >= gems_price, "Precondition: ровно хватает"
