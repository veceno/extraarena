"""Playwright-рекордер: проигрывает кадры в borrowed-арене → webm → mp4.

Поток (path B из плана):
  1. headless Chromium (мобильный портретный viewport ≤420px → срабатывает
     ``@media (max-width: 420px)`` из arena-styles.css → мобильный лейаут)
     грузит ``/player?id=<run_id>&autoplay=1&_auth=<jwt>
     &ea_platform=android_app&music=0&sfx=1``;
  2. ``arena-bridge.js`` (в ``player.html``) вызывает ``window.__orchestraInit()``
     (baked hook → prebattle-гейт снят), stub'ит ``window.io``, ставит
     ``userId=<viewer_uid>`` и итерирует кадры через ``handleStateChanged`` с
     ``await sleep(display_ms)``;
  3. ``/api/battle/state?match_id=<run_id>`` отдаёт первый кадр → arena.js
     сразу рисует init-сцену;
  4. ждём ``window.__orchestraDone === true``, закрываем контекст → webm
     финализируется;
  5. ``audio_mix.mix_audio_into_mp4`` → mp4 (видео + серверный аудио-микс
     SFX + зацикленная фоновая музыка арены).

Headless Chromium не имеет audio-output device, поэтому звук микшируется на
сервере из timeline+sound_events+arena_theme (см. ``audio_mix.py``); в-page
WebAudio был бы тишиным (RISK B). ``music=0`` в URL рекордера сознателен —
in-page музыку всё равно не записать; музыку добавляет серверный микс.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from .arena_io import make_fake_jwt
from .audio_mix import ARENA_MUSIC_PATH, build_audio_timeline, mix_audio_into_mp4
from .gif_export import webm_to_gif

logger = logging.getLogger(__name__)

INIT_SCRIPT = """
window.io = function () {
  var stub = {
    on: function () { return this; },
    off: function () { return this; },
    once: function () { return this; },
    emit: function () { return this; },
    close: function () { return this; },
    connect: function () { return this; },
    disconnect: function () { return this; },
    connected: true,
    id: 'orch-stub',
    io: { on: function () { return this; }, off: function () { return this; } }
  };
  return stub;
};
window.ExtraArenaApp = true;
"""


def _capture_webm(
    run: Dict[str, Any],
    cfg: Dict[str, Any],
    viewer_uid: int,
    *,
    base_url: str = "http://127.0.0.1:8095",
    speed: float = 1.0,
    extra_wait_ms: int = 1500,
) -> tuple:
    """Проиграть прогон в headless Chromium → вернуть ``(webm_path, tmpdir)``.

    Общая стадия захвата для mp4- и GIF-экспорта (GIF не пере-записывает
    арену, а кодирует уже захваченный webm). Мобильный портретный viewport
    (≤420px) + ``device_scale_factor`` для чёткости. ``tmpdir`` возвращает
    созданный ``mkdtemp``-каталог (Playwright ``record_video_dir``) — вызывающий
    обязан подчистить его (``shutil.rmtree``).
    """
    from playwright.sync_api import sync_playwright

    run_id = run["run_id"]
    total_ms = int(run.get("total_ms", 0) or 0)
    if not (run.get("frames", []) or []):
        raise RuntimeError("run has no frames")

    jwt = make_fake_jwt(uid=int(viewer_uid), seed=str(run_id))
    url = (
        f"{base_url}/player?id={quote(run_id)}&autoplay=1"
        f"&_auth={quote(jwt)}&ea_platform=android_app&music=0&sfx=1"
        f"&speed={speed}"
    )

    # бюджет: реальное время проигрывания + буфер на boot/рендер.
    playback_ms = int(total_ms / max(speed, 0.01))
    done_timeout_ms = max(45000, playback_ms + 30000)

    width = int(cfg.get("width", 414))
    height = int(cfg.get("height", 896))
    headless = bool(cfg.get("headless", True))
    # CSS-viewport обязан остаться мобильным (≤420px). Playwright записывает
    # video в CSS-пикселях; если запросить здесь удвоенный record_video_size,
    # Chromium кладёт маленький кадр в левый верхний угол большого canvas.
    # Поэтому захватываем ровно viewport, а upscale до dsf применяем ниже в
    # ffmpeg при сборке финального MP4.
    dsf = max(1, int(cfg.get("device_scale_factor", 2)))

    tmpdir = Path(tempfile.mkdtemp(prefix="orch-rec-"))
    webm_path: Optional[Path] = None

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-web-security",
                f"--window-size={width},{height}",
            ],
        )
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=dsf,
            record_video_dir=str(tmpdir),
            record_video_size={"width": width, "height": height},
        )
        context.add_init_script(INIT_SCRIPT)
        page = context.new_page()

        # Нейтрализуем внешние CDN-скрипты arena.html (telegram-web-app.js,
        # socket.io.min.js): Offline-детерминизм + реальный socket.io не лезет
        # на наш origin (иначе connect_error → «Соединение разорвано»).
        # window.io остаётся нашим stub'ом (init_script + bridge).
        def _neuter(route):
            try:
                route.fulfill(status=200, content_type="application/javascript", body="/* neutered by orchestra */")
            except Exception:
                pass
        page.route("**/telegram.org/js/telegram-web-app.js", _neuter)
        page.route("**/socket.io.min.js", _neuter)

        # Соберём консольные ошибки для диагностики (не блокируем).
        page.on("pageerror", lambda err: logger.warning("[player] pageerror: %s", err))

        logger.info("recording run=%s url=%s (total_ms=%d, timeout=%dms)",
                    run_id, url.replace(jwt, "***"), total_ms, done_timeout_ms)
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Ждём, пока bridge проиграет все кадры.
        try:
            page.wait_for_function(
                "window.__orchestraDone === true",
                timeout=done_timeout_ms,
            )
        except Exception as exc:
            # соберём диагностический стейт перед падением
            try:
                state = page.evaluate(
                    "() => ({done: window.__orchestraDone, err: window.__orchestraError,"
                    " frame: window.__orchestraController && window.__orchestraController.frame,"
                    " total: window.__orchestraController && window.__orchestraController.total,"
                    " playing: window.__orchestraController && window.__orchestraController.playing})"
                )
            except Exception:
                state = {}
            logger.error("bridge did not finish: %s | state=%s", exc, state)
            raise RuntimeError(f"bridge did not finish: {exc} | state={state}") from exc

        page.wait_for_timeout(extra_wait_ms)
        video = page.video
        context.close()
        if video is not None:
            webm_path = Path(video.path())
        browser.close()

    if webm_path is None or not webm_path.exists():
        raise RuntimeError(f"video file not produced (webm={webm_path})")
    return webm_path, tmpdir


def record_run_to_mp4(
    run: Dict[str, Any],
    out_path: str,
    cfg: Dict[str, Any],
    viewer_uid: int,
    *,
    base_url: str = "http://127.0.0.1:8095",
    speed: float = 1.0,
    extra_wait_ms: int = 1500,
) -> str:
    """Записать прогон ``run`` в mp4 (видео + SFX + музыка) по пути ``out_path``."""
    frames = run.get("frames", []) or []
    fps = int(cfg.get("fps", 30))
    crf = int(cfg.get("crf", 10))
    preset = str(cfg.get("preset", "slow"))

    webm_path, tmpdir = _capture_webm(run, cfg, viewer_uid, base_url=base_url,
                                      speed=speed, extra_wait_ms=extra_wait_ms)
    try:
        timeline = build_audio_timeline(frames) if cfg.get("with_audio", False) else []
        logger.info("audio timeline: %d clips (with_audio=%s, crf=%d, preset=%s)",
                    len(timeline), cfg.get("with_audio"), crf, preset)
        output_scale = max(1, int(cfg.get("device_scale_factor", 2)))
        mp4 = mix_audio_into_mp4(
            webm_path, timeline, Path(out_path),
            fps=fps, crf=crf, preset=preset,
            music_path=ARENA_MUSIC_PATH if cfg.get("with_audio", False) else None,
            output_width=int(cfg.get("width", 414)) * output_scale,
            output_height=int(cfg.get("height", 896)) * output_scale,
        )
        return str(mp4)
    finally:
        try:
            webm_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def record_run_to_gif(
    run: Dict[str, Any],
    out_path: str,
    cfg: Dict[str, Any],
    viewer_uid: int,
    *,
    base_url: str = "http://127.0.0.1:8095",
    speed: float = 1.0,
    extra_wait_ms: int = 1500,
) -> str:
    """Записать прогон ``run`` в GIF (двухпроходный palette) по пути ``out_path``.

    GIF не поддерживает звук → экспорт чисто визуальный (без SFX/музыки).
    """
    gif_fps = int(cfg.get("gif_fps", 15))
    gif_width = int(cfg.get("gif_width", 540))

    webm_path, tmpdir = _capture_webm(run, cfg, viewer_uid, base_url=base_url,
                                      speed=speed, extra_wait_ms=extra_wait_ms)
    try:
        gif = webm_to_gif(webm_path, Path(out_path), fps=gif_fps, width=gif_width)
        return str(gif)
    finally:
        try:
            webm_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = ["record_run_to_mp4", "record_run_to_gif"]
