from __future__ import annotations

from pathlib import Path

from infrastructure.config import BASE_DIR


CARD_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
CARD_ASSET_URL_PREFIX = "/DesignAssets/Cards"
CARD_PREVIEW_ASSET_URL_PREFIX = "/DesignAssets/CardsPreview/w384"
CARD_ASSETS_DIR = BASE_DIR / "DesignAssets" / "Cards"
CARD_PREVIEW_ASSETS_DIR = BASE_DIR / "DesignAssets" / "CardsPreview" / "w384"


def _coerce_card_id(card_id: int | str | None) -> int | None:
    if card_id is None:
        return None
    try:
        return int(card_id)
    except (TypeError, ValueError):
        return None


def _resolve_original_card_asset_path(safe_id: int) -> Path | None:
    for suffix in CARD_ASSET_EXTENSIONS:
        candidate = CARD_ASSETS_DIR / f"{safe_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _resolve_preview_card_asset_path(safe_id: int) -> Path | None:
    candidate = CARD_PREVIEW_ASSETS_DIR / f"{safe_id}.webp"
    if candidate.is_file():
        return candidate
    return None


def resolve_card_asset_path(card_id: int | str | None, *, variant: str | None = None) -> Path | None:
    """Return the first existing card asset path for the requested variant."""
    safe_id = _coerce_card_id(card_id)
    if safe_id is None:
        return None

    if variant == "preview":
        preview_path = _resolve_preview_card_asset_path(safe_id)
        if preview_path is not None:
            return preview_path

    return _resolve_original_card_asset_path(safe_id)


def _url_for_card_asset_path(path: Path) -> str:
    if path.parent == CARD_PREVIEW_ASSETS_DIR:
        return f"{CARD_PREVIEW_ASSET_URL_PREFIX}/{path.name}"
    return f"{CARD_ASSET_URL_PREFIX}/{path.name}"


def card_asset_url(
    card_id: int | str | None,
    *,
    fallback_id: int = 9,
    variant: str | None = None,
) -> str:
    path = resolve_card_asset_path(card_id, variant=variant)
    if path is None:
        fallback_path = resolve_card_asset_path(fallback_id, variant=variant)
        if fallback_path is None:
            return f"{CARD_ASSET_URL_PREFIX}/{fallback_id}.png"
        path = fallback_path
    return _url_for_card_asset_path(path)
