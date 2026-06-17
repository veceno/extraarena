from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from support.attachments import (
    compress_support_attachment,
    detect_image_signature,
    safe_support_filename,
    support_attachment_static_path,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (96, 64), color=(32, 120, 220))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_detect_image_signature_accepts_png_jpeg_webp_and_rejects_unknown():
    png = _png_bytes()
    jpeg_buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(240, 80, 70)).save(jpeg_buffer, format="JPEG")
    webp_buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(80, 220, 110)).save(webp_buffer, format="WEBP")

    assert detect_image_signature(png) == "png"
    assert detect_image_signature(jpeg_buffer.getvalue()) == "jpeg"
    assert detect_image_signature(webp_buffer.getvalue()) == "webp"
    with pytest.raises(ValueError, match="unsupported_image_type"):
        detect_image_signature(b"not an image")


def test_compress_support_attachment_writes_webp_metadata_under_month_bucket(tmp_path):
    upload_root = tmp_path / "uploads" / "support"
    source = _png_bytes()

    metadata = compress_support_attachment(
        source,
        original_filename="../screenshots/arena bug.png",
        upload_root=upload_root,
        now=datetime(2026, 6, 16, tzinfo=timezone.utc),
    )

    written = metadata.path
    written_bytes = written.read_bytes()

    assert written.parent == upload_root / "2026" / "06"
    assert written.suffix == ".webp"
    assert written_bytes[:4] == b"RIFF"
    assert written_bytes[8:12] == b"WEBP"
    assert metadata.sha256 == hashlib.sha256(written_bytes).hexdigest()
    assert metadata.content_type == "image/webp"
    assert metadata.source_content_type == "image/png"
    assert metadata.width == 96
    assert metadata.height == 64
    assert metadata.original_filename == "arena_bug.png"
    assert metadata.size_bytes == len(written_bytes)
    assert support_attachment_static_path(written, upload_root=upload_root) == (
        f"/uploads/support/2026/06/{written.name}"
    )


def test_safe_support_filename_removes_paths_and_unsafe_characters():
    assert safe_support_filename("../../secret token.png") == "secret_token.png"
    assert safe_support_filename("   ") == "attachment"


def test_compress_support_attachment_rejects_images_over_pixel_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("support.attachments.MAX_SUPPORT_IMAGE_PIXELS", 100)

    with pytest.raises(ValueError, match="image_too_large"):
        compress_support_attachment(_png_bytes(), upload_root=tmp_path)
