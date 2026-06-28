"""Серверный экспорт webm → GIF (двухпроходный с палитрой).

GIF не поддерживает звук — поэтому экспорт в GIF чисто визуальный (без SFX/
музыки арены). Для качества цвета используется двухпроходный pipeline
``palettegen`` + ``paletteuse`` (libavfilter): сначала по всему клипу
строится оптимальная 256-цветная палитра (``stats_mode=diff`` — учитывает
только меняющиеся области), затем кадры квантуются в неё с дизерингом Bayer
(``bayer_scale=5`` — мягкий, без сильного «мозаичного» шума).

Размер: по умолчанию GIF масштабируется до ``width`` (540px) с сохранением
мобильного портретного аспекта; ``width=0`` — native-разрешение webm. Меньше
``fps``/``width`` → заметно меньший файл.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _even(n: int) -> int:
    # libswscale требует чётные размеры по ширине/высоте.
    return n if n % 2 == 0 else max(2, n - 1)


def webm_to_gif(
    video_webm: Path,
    out_gif: Path,
    *,
    fps: int = 15,
    width: int = 0,
) -> Path:
    """Конвертировать webm → GIF (двухпроходный palette).

    ``width`` — целевая ширина GIF (чётная); ``0`` = не масштабировать (native).
    """
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    fps = max(1, min(int(fps) or 15, 30))
    scale_filter = ""
    if width and int(width) > 0:
        w = _even(int(width))
        # scale=W:-2 → высота авто, кратно 2 (чётная).
        scale_filter = f",scale={w}:-2:flags=lanczos"

    palette = out_gif.with_suffix(".palette.png")

    # Pass 1: палитра по всему клипу.
    pal_cmd = [
        "ffmpeg", "-y", "-i", str(video_webm),
        "-vf", f"fps={fps}{scale_filter},palettegen=stats_mode=diff",
        "-update", "1", str(palette),
    ]
    logger.info("gif pass 1 (palette): %s", " ".join(pal_cmd[:6]) + " ...")
    try:
        subprocess.run(pal_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"gif palettegen failed: {err}") from exc

    # Pass 2: квантование кадров в палитру.
    use_cmd = [
        "ffmpeg", "-y", "-i", str(video_webm), "-i", str(palette),
        "-filter_complex",
        f"[0:v]fps={fps}{scale_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        str(out_gif),
    ]
    logger.info("gif pass 2 (paletteuse): %s", " ".join(use_cmd[:6]) + " ...")
    try:
        subprocess.run(use_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"gif paletteuse failed: {err}") from exc
    finally:
        try:
            palette.unlink(missing_ok=True)
        except Exception:
            pass

    return out_gif


__all__ = ["webm_to_gif"]