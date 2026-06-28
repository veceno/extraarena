"""Серверный аудио-микс: timeline+sound_events+card_sfx_config → ffmpeg mix.

Headless Chromium не имеет audio-output device → ``page.video()`` пишет webm
без звука, а in-page WebAudio может дать тишину (RISK B). Поэтому микшируем
звук арены детерминированно на сервере: мы знаем полный timeline (``List[
Frame]`` с ``display_ms``) и ``sound_events`` каждого кадра, а также
``card_sfx_config.json`` (per-card deploy/attack/mechanic src+volume) и
``ARENA_SFX``-таблицу (имя → wav-файл, извлечена из ``arena.html``).

Логика разрешения звука (упрощённый порт ``processArenaSoundEvents`` /
``resolveArenaCardSfx`` из arena.js):
  - ``deploy``  → ``cards[card_id].sounds.deploy.src`` (иначе тишина);
  - ``attack``  → ``cards[card_id].sounds.attack|.hit.src`` иначе
                  ``ARENA_SFX.cardAttacked`` → ``card_attacked.wav``;
  - ``mechanic``→ ``cards[card_id].sounds[mechanic].src`` (иначе тишина).
Объём — из config (``volume``), дефолт 0.8.

Смещение звука = кумулятивная сумма ``display_ms`` предыдущих кадров (событие
звучит в момент показа кадра). Чанкование по ≤32 входов (RISK G).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOUNDS_DIR = REPO_ROOT / "DesignAssets" / "Sounds" / "arena"
CARD_SFX_CONFIG_PATH = SOUNDS_DIR / "card_sfx_config.json"

# Фоновая музыка арены (из arena.html <audio id="arena-bg-music">). В arena.js
# играется зацикленно на громкости 0.3; headless Chromium не имеет audio-output
# device, поэтому для экспорта музыку микшируем на сервере тем же ffmpeg.
ARENA_MUSIC_PATH = REPO_ROOT / "DesignAssets" / "Arena" / "Sounds" / "arena_theme.wav"
ARENA_MUSIC_VOLUME = 0.3

# ARENA_SFX name → audio element id → wav-файл (из arena.html <audio> source src).
# Путь в arena.html — `../DesignAssets/Sounds/...`; на диске — `DesignAssets/...`.
_ARENA_SFX_FILES = {
    "battleStart": "DesignAssets/Sounds/arena/battle_start.wav",
    "cardAttacked": "DesignAssets/Sounds/arena/card_attacked.wav",
    "cardDeath": "DesignAssets/Sounds/arena/card_death.wav",
    "cardFrozen": "DesignAssets/Sounds/arena/card_frozen.wav",
    "cardHeal": "DesignAssets/Sounds/arena/card_heal.wav",
    "cardSelected": "DesignAssets/Sounds/arena/card_selected.wav",
    "manaDraw": "DesignAssets/Sounds/arena/AddCard.wav",
    "heroDamage": "DesignAssets/Sounds/arena/hero_damage.wav",
    "heroDeath": "DesignAssets/Sounds/arena/hero_death.wav",
    "nextMove": "DesignAssets/Sounds/arena/next_move_btn_pressed.wav",
    "playerTurnStart": "DesignAssets/Sounds/arena/player_turn_start.wav",
    "victory": "DesignAssets/Sounds/arena/victory.wav",
    "defeat": "DesignAssets/Sounds/arena/defeat.wav",
    "surrender": "DesignAssets/Sounds/arena/surrender.wav",
    "lowTimeTick": "DesignAssets/Sounds/arena/low_time_tick.wav",
}

DEFAULT_VOLUME = 0.8
MECHANIC_SOUND_KEYS = ("deploy", "attack", "hit", "death", "hurt", "ambient")


def _load_card_sfx_config() -> Dict[str, Any]:
    try:
        return json.loads(CARD_SFX_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("card_sfx_config.json not readable: %s", CARD_SFX_CONFIG_PATH)
        return {"cards": {}}


def _card_sounds(cfg: Dict[str, Any], card_id: Any) -> Dict[str, Any]:
    cards = cfg.get("cards", {}) or {}
    entry = cards.get(str(card_id)) if card_id is not None else None
    if entry is None and card_id is not None:
        entry = cards.get(int(card_id) if str(card_id).isdigit() else card_id)
    return (entry or {}).get("sounds", {}) or {}


def _src_to_disk_path(src: str) -> Optional[Path]:
    """``/DesignAssets/...`` или ``../DesignAssets/...`` → Path на диске."""
    if not src:
        return None
    s = src.lstrip()
    # абсолютный от корня сайта
    m = re.match(r"^/DesignAssets/(.+)$", s)
    if m:
        p = REPO_ROOT / "DesignAssets" / m.group(1)
        return p if p.exists() else None
    # относительный `../DesignAssets/...`
    m = re.match(r"^\.\./DesignAssets/(.+)$", s)
    if m:
        p = REPO_ROOT / "DesignAssets" / m.group(1)
        return p if p.exists() else None
    # fallback: другие относительные — попробуем от DesignAssets
    p = REPO_ROOT / s.lstrip("/")
    return p if p.exists() else None


def _resolve_event_file(cfg: Dict[str, Any], event: Dict[str, Any]) -> Tuple[Optional[Path], float]:
    """Вернуть ``(wav_path, volume)`` для одного sound_event."""
    name = str(event.get("event") or "")
    card_id = event.get("card_id")
    mechanic = event.get("mechanic")
    sounds = _card_sounds(cfg, card_id)

    def _pick(key: str) -> Tuple[Optional[Path], float]:
        slot = sounds.get(key)
        if isinstance(slot, dict):
            p = _src_to_disk_path(slot.get("src", ""))
            if p:
                return p, float(slot.get("volume", DEFAULT_VOLUME) or DEFAULT_VOLUME)
        elif isinstance(slot, str):
            p = _src_to_disk_path(slot)
            if p:
                return p, DEFAULT_VOLUME
        return None, DEFAULT_VOLUME

    if name == "deploy":
        p, v = _pick("deploy")
        if p:
            return p, v
        return None, DEFAULT_VOLUME
    if name == "attack":
        for k in ("attack", "hit", "hurt"):
            p, v = _pick(k)
            if p:
                return p, v
        # generic
        p = _src_to_disk_path(_ARENA_SFX_FILES["cardAttacked"])
        return p, 0.74
    if name == "mechanic" and mechanic:
        p, v = _pick(str(mechanic))
        if p:
            return p, v
        return None, DEFAULT_VOLUME
    return None, DEFAULT_VOLUME


def build_audio_timeline(frames: List[Dict[str, Any]]) -> List[Tuple[int, Path, float]]:
    """``[(offset_ms, wav_path, volume), ...]`` из кадров со звуком."""
    cfg = _load_card_sfx_config()
    timeline: List[Tuple[int, Path, float]] = []
    offset_ms = 0
    for frame in frames:
        events = frame.get("sound_events") or []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            p, vol = _resolve_event_file(cfg, ev)
            if p is not None:
                timeline.append((offset_ms, p, vol))
        offset_ms += int(frame.get("display_ms", 0) or 0)
    return timeline


def _chunk(seq: List, n: int) -> List[List]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def mix_audio_into_mp4(
    video_webm: Path,
    timeline: List[Tuple[int, Path, float]],
    out_mp4: Path,
    *,
    fps: int = 30,
    chunk_size: int = 32,
    crf: int = 10,
    preset: str = "slow",
    music_path: Optional[Path] = None,
    music_volume: float = ARENA_MUSIC_VOLUME,
) -> Path:
    """Смиксовать видео + SFX-timeline (+ зацикленная музыка) → mp4 (libx264 + aac).

    ``music_path`` — wav-файл фоновой музыки арены; проигрывается зацикленно
    (``-stream_loop -1``) на ``music_volume`` и обрезается по длине видео
    (``-shortest``). ``crf`` (ниже = выше качество, по умолч. 10 ≈ визуально
    lossless) и ``preset`` (по умолч. ``slow`` — лучшее сжатие) управляют
    качеством видео. Если ``timeline`` пуст И музыки нет — кодируем только видео
    (без аудиодорожки).
    """
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    has_music = bool(music_path) and Path(music_path).exists()

    if not timeline and not has_music:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_webm),
            "-c:v", "libx264", "-preset", str(preset), "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-movflags", "+faststart", str(out_mp4),
        ]
        logger.info("ffmpeg (video-only, crf=%s preset=%s): %s", crf, preset, " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        return out_mp4

    # Чанкование: ffmpeg filter_complex с большим числом входов тяжёлый.
    # Соберём все входы и amix за один проход (для ≤chunk_size), иначе
    # микшируем по частям во временные wav и собираем финальный amix.
    inputs: List[str] = ["-i", str(video_webm)]
    for (_off, p, _v) in timeline:
        inputs += ["-i", str(p)]
    music_idx: Optional[int] = None
    if has_music:
        # -stream_loop -1 перед входом → бесконечный луп музыки; -shortest
        # обрежет выход по длине видео (конечного) — амикс не зависнет.
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]
        music_idx = len(timeline) + 1  # 0=video, 1..N=sfx, N+1=music

    filter_parts: List[str] = []
    amix_labels: List[str] = []
    for idx, (off, _p, vol) in enumerate(timeline, start=1):
        a = f"a{idx}"
        filter_parts.append(
            f"[{idx}:a]adelay={off}|{off},volume={vol:.3f}[{a}]"
        )
        amix_labels.append(f"[{a}]")
    if has_music:
        filter_parts.append(f"[{music_idx}:a]volume={music_volume:.3f}[amusic]")
        amix_labels.append("[amusic]")
    amix_in = "".join(amix_labels)
    n = len(amix_labels)
    filter_complex = ";".join(filter_parts) + f";{amix_in}amix=inputs={n}:duration=longest:normalize=0[aout]"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", str(preset), "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", str(out_mp4),
    ]
    logger.info("ffmpeg (video+audio, %d sfx + music=%s, crf=%s preset=%s): %s",
                len(timeline), has_music, crf, preset, " ".join(cmd[:8]) + " ...")
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


__all__ = ["build_audio_timeline", "mix_audio_into_mp4", "ARENA_MUSIC_PATH", "ARENA_MUSIC_VOLUME"]