import json
from pathlib import Path
from uuid import uuid4

from battle_engine import BattleEngine
from core.state import CardInstance, CardType
from infrastructure import card_assets
from infrastructure.card_assets import card_asset_url, resolve_card_asset_path


ROOT = Path(__file__).resolve().parents[1]


def _catalog_ids() -> list[int]:
    cards = json.loads((ROOT / "cards.json").read_text(encoding="utf-8"))
    return [int(card["id"]) for card in cards]


def test_every_beta_catalog_card_resolves_to_existing_asset():
    missing = [card_id for card_id in _catalog_ids() if resolve_card_asset_path(card_id) is None]

    assert missing == []


def test_new_wave_card_assets_resolve_to_jpg_files():
    for card_id in (43, 44, 45, 46):
        path = resolve_card_asset_path(card_id)
        assert path is not None
        assert path.suffix == ".jpg"
        assert card_asset_url(card_id) == f"/DesignAssets/Cards/{card_id}.jpg"


def test_preview_asset_resolution_prefers_webp_preview(monkeypatch, tmp_path):
    cards_dir = tmp_path / "Cards"
    previews_dir = tmp_path / "CardsPreview" / "w384"
    cards_dir.mkdir()
    previews_dir.mkdir(parents=True)
    (cards_dir / "7.png").write_bytes(b"original")
    (previews_dir / "7.webp").write_bytes(b"preview")
    monkeypatch.setattr(card_assets, "CARD_ASSETS_DIR", cards_dir)
    monkeypatch.setattr(card_assets, "CARD_PREVIEW_ASSETS_DIR", previews_dir)

    assert resolve_card_asset_path(7, variant="preview") == previews_dir / "7.webp"
    assert card_asset_url(7, variant="preview") == "/DesignAssets/CardsPreview/w384/7.webp"
    assert resolve_card_asset_path(7) == cards_dir / "7.png"
    assert card_asset_url(7) == "/DesignAssets/Cards/7.png"


def test_preview_asset_resolution_falls_back_to_original(monkeypatch, tmp_path):
    cards_dir = tmp_path / "Cards"
    previews_dir = tmp_path / "CardsPreview" / "w384"
    cards_dir.mkdir()
    previews_dir.mkdir(parents=True)
    (cards_dir / "44.jpg").write_bytes(b"original")
    monkeypatch.setattr(card_assets, "CARD_ASSETS_DIR", cards_dir)
    monkeypatch.setattr(card_assets, "CARD_PREVIEW_ASSETS_DIR", previews_dir)

    assert resolve_card_asset_path(44, variant="preview") == cards_dir / "44.jpg"
    assert card_asset_url(44, variant="preview") == "/DesignAssets/Cards/44.jpg"
    assert resolve_card_asset_path(44, variant="full") == cards_dir / "44.jpg"


def test_battle_serialized_card_image_uses_resolved_extension():
    engine = BattleEngine(db=None, match_id="asset-test")
    card = CardInstance(
        instance_id=uuid4(),
        card_id=44,
        name="Леви Аккерман",
        card_type=CardType.WARRIOR,
    )

    assert engine._serialize_card(card)["image"] == "/DesignAssets/Cards/44.jpg"
