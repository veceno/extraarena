"""Каталог карт и косметики для сценарного движка и редактора.

Источник правды — ``cards.json`` (50 карт, актуальный граф игры на
worktree-NewCards2606). ``CardInstance`` строится через ``core.converter.
card_from_db`` (нормализация механик + ``scale_card_by_level``), поверх
накладываются overrides из сценария.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_DNS, UUID, uuid5

from core.converter import card_from_db
from core.state import CardInstance
from infrastructure.card_assets import card_asset_url

logger = logging.getLogger(__name__)


def deterministic_instance_id(*, seed: int, side: str, zone: str, index: int,
                              card_id: int, level: int) -> UUID:
    """Стабильный instance_id из позиции карты в сценарии (детерминизм реплеев).

    ``uuid4`` по умолчанию делал бы прогоны одного сценария разными; вместо него
    uuid5 от ключа ``seed:side:zone:index:card_id:level``.
    """
    key = f"{int(seed)}:{side}:{zone}:{int(index)}:{int(card_id)}:{int(level)}"
    return uuid5(NAMESPACE_DNS, "extra-orchestra:" + key)


def _default_cards_path() -> Path:
    return Path(__file__).resolve().parents[2] / "cards.json"


class CardsCatalog:
    """Леницо загружаемый каталог карт."""

    def __init__(self, cards_path: Optional[Path] = None) -> None:
        self.cards_path = Path(cards_path) if cards_path else _default_cards_path()
        self._by_id: Optional[Dict[int, Dict[str, Any]]] = None

    def _load(self) -> Dict[int, Dict[str, Any]]:
        if self._by_id is not None:
            return self._by_id
        data = json.loads(self.cards_path.read_text(encoding="utf-8"))
        by_id: Dict[int, Dict[str, Any]] = {}
        for row in data:
            try:
                cid = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            by_id[cid] = row
        self._by_id = by_id
        return by_id

    @property
    def by_id(self) -> Dict[int, Dict[str, Any]]:
        return self._load()

    def has(self, card_id: int) -> bool:
        return int(card_id) in self.by_id

    def row(self, card_id: int) -> Dict[str, Any]:
        return self.by_id[int(card_id)]

    def build_instance(
        self,
        spec: Dict[str, Any],
        instance_id: Optional[UUID] = None,
    ) -> CardInstance:
        """Построить CardInstance из сценарной spec карты.

        spec: ``{card_id, level, hp_override?, attack_override?,
        mechanics_override?, is_ready?, is_frozen?}``. ``instance_id`` —
        опциональный стабильный UUID (для детерминизма реплеев).
        """
        card_id = int(spec["card_id"])
        level = int(spec.get("level", 1) or 1)
        row = self.by_id[card_id]
        card = card_from_db(row, level)
        if instance_id is not None:
            card.instance_id = instance_id

        if "mechanics_override" in spec and spec["mechanics_override"] is not None:
            card.mechanics = list(spec["mechanics_override"])
        if spec.get("attack_override") is not None:
            card.attack = int(spec["attack_override"])
        if spec.get("hp_override") is not None:
            card.hp = int(spec["hp_override"])
        if spec.get("is_ready") is not None:
            card.is_ready = bool(spec["is_ready"])
        if spec.get("is_frozen") is not None:
            card.is_frozen = bool(spec["is_frozen"])
        return card

    def list_cards(self) -> List[Dict[str, Any]]:
        """Список карт для пикеров редактора (с базовыми статами + image url)."""
        out: List[Dict[str, Any]] = []
        for cid in sorted(self.by_id):
            row = self.by_id[cid]
            mechanics_raw = row.get("mechanics", "[]")
            if isinstance(mechanics_raw, str):
                try:
                    mechanics = json.loads(mechanics_raw)
                except json.JSONDecodeError:
                    mechanics = []
            else:
                mechanics = mechanics_raw or []
            out.append({
                "id": cid,
                "name": row.get("name", ""),
                "card_type": row.get("card_type", "warrior"),
                "rarity": row.get("rarity", "common"),
                "mana_cost": row.get("mana_cost", 0),
                "base_attack": row.get("base_attack", 0),
                "base_hp": row.get("base_hp", 0),
                "mechanics": mechanics,
                "simplified_levelup": bool(row.get("simplified_levelup", False)),
                "image": card_asset_url(cid),
                "description": row.get("description", ""),
            })
        return out


def list_cosmetics(base_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Список аватаров и фонов профиля из /DesignAssets/PlayerCosmetics/.

    Возвращает ``{"avatars": [...], "backgrounds": [...]}`` с url'ами.
    """
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
    cos = root / "DesignAssets" / "PlayerCosmetics"

    def _scan(sub: str, url_prefix: str) -> List[Dict[str, Any]]:
        d = cos / sub
        items: List[Dict[str, Any]] = []
        if not d.is_dir():
            return items
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            items.append({
                "name": p.stem,
                "url": f"/DesignAssets/PlayerCosmetics/{sub}/{p.name}",
            })
        return items

    return {
        "avatars": _scan("Avatars", "/DesignAssets/PlayerCosmetics/Avatars"),
        "backgrounds": _scan("Background", "/DesignAssets/PlayerCosmetics/Background"),
    }