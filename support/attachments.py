from __future__ import annotations

import hashlib
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SUPPORT_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "uploads" / "support"
MAX_SUPPORT_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_SUPPORT_IMAGE_PIXELS = 12_000_000
MAX_SUPPORT_IMAGE_DIMENSION = 6000
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SupportAttachmentMetadata:
    path: Path
    original_filename: str
    content_type: str
    source_content_type: str
    sha256: str
    size_bytes: int
    width: int | None = None
    height: int | None = None

    def as_record(self, *, upload_root: Path = SUPPORT_UPLOAD_ROOT) -> dict[str, Any]:
        return {
            "storage_path": support_attachment_static_path(self.path, upload_root=upload_root),
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "source_content_type": self.source_content_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
        }


def detect_image_signature(data: bytes) -> str:
    head = bytes(data[:16])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("unsupported_image_type")


def safe_support_filename(filename: str | None) -> str:
    name = Path(str(filename or "")).name.strip()
    if not name:
        return "attachment"
    name = SAFE_FILENAME_RE.sub("_", name).strip("._")
    return name or "attachment"


def compress_support_attachment(
    data: bytes,
    *,
    original_filename: str | None = None,
    upload_root: Path = SUPPORT_UPLOAD_ROOT,
    now: datetime | None = None,
    quality: int = 82,
) -> SupportAttachmentMetadata:
    if not data:
        raise ValueError("empty_attachment")
    if len(data) > MAX_SUPPORT_ATTACHMENT_BYTES:
        raise ValueError("attachment_too_large")

    detected = detect_image_signature(data)
    source_content_type = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }[detected]

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    bucket = upload_root / f"{current.year:04d}" / f"{current.month:02d}"
    bucket.mkdir(parents=True, exist_ok=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(PathLikeBytes(data)) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("invalid_image_dimensions")
                if (
                    width > MAX_SUPPORT_IMAGE_DIMENSION
                    or height > MAX_SUPPORT_IMAGE_DIMENSION
                    or width * height > MAX_SUPPORT_IMAGE_PIXELS
                ):
                    raise ValueError("image_too_large")
                converted = image.convert("RGB")
                path = bucket / f"{uuid.uuid4().hex}.webp"
                converted.save(path, format="WEBP", quality=quality, method=6)
    except Image.DecompressionBombError as exc:
        raise ValueError("image_too_large") from exc
    except Image.DecompressionBombWarning as exc:
        raise ValueError("image_too_large") from exc

    written = path.read_bytes()
    return SupportAttachmentMetadata(
        path=path,
        original_filename=safe_support_filename(original_filename),
        content_type="image/webp",
        source_content_type=source_content_type,
        sha256=hashlib.sha256(written).hexdigest(),
        size_bytes=len(written),
        width=width,
        height=height,
    )


def support_attachment_static_path(path: Path, *, upload_root: Path = SUPPORT_UPLOAD_ROOT) -> str:
    resolved_root = upload_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("attachment_outside_upload_root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("invalid_attachment_path")
    return "/uploads/support/" + "/".join(relative.parts)


def resolve_support_attachment_path(filename: str, *, upload_root: Path = SUPPORT_UPLOAD_ROOT) -> Path:
    cleaned = str(filename or "").lstrip("/")
    if "\\" in cleaned or ".." in cleaned:
        raise ValueError("invalid_attachment_path")
    parts = tuple(part for part in cleaned.split("/") if part)
    if len(parts) != 3:
        raise ValueError("invalid_attachment_path")
    year, month, name = parts
    if not year.isdigit() or not month.isdigit() or safe_support_filename(name) != name:
        raise ValueError("invalid_attachment_path")
    path = (upload_root / year / month / name).resolve()
    root = upload_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("attachment_outside_upload_root") from exc
    return path


class PathLikeBytes:
    """Small file-like adapter for Pillow without importing BytesIO globally."""

    def __init__(self, data: bytes) -> None:
        from io import BytesIO

        self._buffer = BytesIO(data)

    def read(self, *args: Any) -> bytes:
        return self._buffer.read(*args)

    def seek(self, *args: Any) -> int:
        return self._buffer.seek(*args)

    def tell(self) -> int:
        return self._buffer.tell()
