import json
from pathlib import Path

from core.converter import card_from_db
from infrastructure import database as database_module
from infrastructure.case_config import UNI_CARD_ID


ROOT = Path(__file__).resolve().parents[1]


def _load_cards(path: str) -> dict[int, dict]:
    cards = json.loads((ROOT / path).read_text(encoding="utf-8"))
    ids = [int(card["id"]) for card in cards]
    assert len(ids) == len(set(ids)), f"{path} must not contain duplicate card ids"
    return {int(card["id"]): card for card in cards}


def test_balance_patch_catalog_entries_are_present_and_consistent():
    expected = {
        1: {"base_hp": 35},
        4: {"mechanics": '["reflect_1"]'},
        6: {"base_hp": 37},
        7: {"base_hp": 23},
        18: {"mana_cost": 5},
        20: {"mana_cost": 3},
        23: {"mechanics": '["cleave_1_2"]'},
        24: {"mechanics": '["shield", "shield_refresh"]', "base_hp": 6},
        25: {"description": "Один удар. На первом своем ходу уничтожает выбранного противника один раз."},
        27: {"base_attack": 2, "base_hp": 1},
        29: {"base_attack": 3, "base_hp": 4},
        30: {"base_hp": 5},
        31: {"base_attack": 4, "base_hp": 5},
        34: {"mechanics": '["deathrattle_aoe_damage_2"]'},
        40: {"base_attack": 3, "base_hp": 3},
        41: {"base_attack": 4, "base_hp": 4},
        42: {"base_attack": 6, "base_hp": 6},
        43: {"name": "Лара Крофт", "rarity": "epic", "mana_cost": 2, "base_attack": 3, "base_hp": 1, "mechanics": '["bypass_taunt"]'},
        44: {"name": "Леви Аккерман", "rarity": "epic", "mana_cost": 2, "base_attack": 3, "base_hp": 1, "mechanics": '["charge"]'},
        45: {"name": "Солид Снейк", "rarity": "epic", "mana_cost": 5, "base_attack": 5, "base_hp": 4, "mechanics": '["taunt"]'},
        46: {"name": "Уссоп", "rarity": "common", "mana_cost": 2, "base_attack": 3, "base_hp": 1, "mechanics": "[]"},
    }

    for catalog_path in ("cards.json", "ai/cards.json"):
        cards = _load_cards(catalog_path)
        for card_id, fields in expected.items():
            assert card_id in cards, f"{catalog_path} missing card {card_id}"
            for key, value in fields.items():
                assert cards[card_id][key] == value


def test_gojo_catalog_converts_to_level_one_six_hp_refreshing_shield():
    cards = _load_cards("cards.json")

    gojo = card_from_db(cards[24], level=1)

    assert gojo.hp == 6
    assert gojo.max_hp == 6
    assert gojo.mechanics == ["shield", "shield_refresh"]


def test_sukuna_catalog_uses_two_target_cleave_shape():
    cards = _load_cards("cards.json")

    sukuna = card_from_db(cards[23], level=1)

    assert sukuna.mechanics == ["cleave_1_2"]


def test_balance_patch_sql_contains_all_v01_card_updates():
    sql = (ROOT / "infrastructure/sql/2026_05_30_balance_cards.sql").read_text(encoding="utf-8")
    expected_fragments = [
        "(4,  'Аскеладд'",
        "'[\"reflect_1\"]'::jsonb",
        "(6,  'Росомаха'",
        "(18, 'П.Е.К.К.А.'",
        "(20, 'Канеки Кен'",
        "(23, 'Сукуна'",
        "'[\"cleave_1_2\"]'::jsonb",
        "(24, 'Годжо Сатору'",
        "'[\"shield\", \"shield_refresh\"]'::jsonb",
        "(27, 'Скелет'",
        "(29, 'Штурмовик'",
        "(31, 'Наемник'",
        "(34, 'Крипер'",
        "'[\"deathrattle_aoe_damage_2\"]'::jsonb",
        "(36, 'Юни'",
        "(37, 'Слайм'",
        "(38, 'Хиличурл'",
        "(39, 'Альфонс Элрик'",
        "(40, 'Стив'",
        "(41, 'Довакин'",
        "(42, 'Атакующий Титан'",
        "(43, 'Лара Крофт'",
        "(46, 'Уссоп'",
        "При первом ходе Сайтамы уничтожает выбранного противника. Срабатывает один раз; щит блокирует удар",
    ]

    for fragment in expected_fragments:
        assert fragment in sql


def test_startup_schema_runs_repeatable_balance_seed():
    source = (ROOT / "infrastructure/database.py").read_text(encoding="utf-8")

    assert "async def _seed_balance_cards" in source
    assert "await self._seed_balance_cards()" in source
    assert source.index("cards_changed = await self._ensure_cards_table()") < source.index(
        "await self._seed_balance_cards()"
    )


def test_startup_schema_forces_touka_random_battlecry():
    source = (ROOT / "infrastructure/database.py").read_text(encoding="utf-8")

    assert "battlecry_damage_1_random" in source
    assert "WHERE id = 15" in source
    assert "UPDATE cards" in source
    assert "WHERE id = 25" in source


def test_starter_deck_template_is_legal_and_catalog_backed():
    starter_ids = getattr(database_module, "STARTER_DECK_CARD_IDS", None)

    assert starter_ids == [1, 36, 37, 38, 39, 40, 41, 42, 46]
    assert len(starter_ids) == len(set(starter_ids)) == database_module.DECK_SIZE

    for catalog_path in ("cards.json", "ai/cards.json"):
        cards = _load_cards(catalog_path)
        missing = [card_id for card_id in starter_ids if card_id not in cards]
        assert missing == [], f"{catalog_path} missing starter deck ids {missing}"
        assert cards[starter_ids[0]]["card_type"] == "hero"
        for card_id in starter_ids[1:]:
            assert cards[card_id]["card_type"] != "hero"


def test_uni_case_lookup_uses_explicit_yuni_card_id():
    assert UNI_CARD_ID == 36
    for catalog_path in ("cards.json", "ai/cards.json"):
        cards = _load_cards(catalog_path)
        assert cards[UNI_CARD_ID]["name"] == "Юни"


def test_seeded_mechanics_descriptions_match_actual_mechanic_values():
    descriptions = database_module._CARD_MECHANICS_DESC

    assert "1 урон" in descriptions[4]
    assert "2 урона" not in descriptions[4]
    assert "2 урон" in descriptions[34] or "2 урона" in descriptions[34]
    assert "3 урон" not in descriptions[34] and "3 урона" not in descriptions[34]
    assert descriptions[25] == (
        "При первом ходе Сайтамы уничтожает выбранного противника. "
        "Срабатывает один раз; щит блокирует удар"
    )
    assert descriptions[22] == "При выходе замораживает до 3 вражеских существ на доске"
    assert descriptions[23] == "При атаке наносит дополнительный урон до 2 соседним существам противника"
