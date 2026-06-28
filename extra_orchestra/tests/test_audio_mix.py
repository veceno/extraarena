"""Тесты audio_mix: timeline-смещения и структура ffmpeg filter_complex (mock subprocess)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from extra_orchestra.components import audio_mix
from extra_orchestra.components.audio_mix import build_audio_timeline, mix_audio_into_mp4


def _frame(snapshot_id, display_ms, sound_events):
    return {
        "snapshot": {"id": snapshot_id},
        "sound_events": sound_events,
        "display_ms": display_ms,
        "action_kind": "play_card",
        "turn_id": "t1",
        "node_id": "n",
        "error": None,
    }


def test_build_audio_timeline_offsets(monkeypatch):
    # замокаем card_sfx_config: card 47 deploy → фиктивный путь (создадим temp)
    tmp = Path(__file__).resolve().parent / "_tmp_wav.wav"
    tmp.write_bytes(b"RIFFxxxxWAVEfmt ")
    cfg = {"cards": {"47": {"sounds": {"deploy": {"src": "/DesignAssets/Sounds/arena/characters/047_soldier/soldier_deploy.mp3", "volume": 0.82}}}}}
    monkeypatch.setattr(audio_mix, "_load_card_sfx_config", lambda: cfg)
    # пусть _src_to_disk_path находит реальный soldier_deploy.mp3
    frames = [
        _frame(0, 2200, []),
        _frame(1, 1500, []),
        _frame(2, 1200, [{"event": "deploy", "card_id": 47, "instance_id": "i1"}]),
        _frame(3, 2600, []),
    ]
    tl = build_audio_timeline(frames)
    assert len(tl) == 1
    offset_ms, path, vol = tl[0]
    # offset = 2200 + 1500 = 3700 (старт 3-го кадра)
    assert offset_ms == 3700
    assert path.name == "soldier_deploy.mp3"
    assert abs(vol - 0.82) < 1e-6
    tmp.unlink(missing_ok=True)


def test_mix_audio_into_mp4_command_structure(monkeypatch, tmp_path):
    webm = tmp_path / "v.webm"
    webm.write_bytes(b"x")
    out = tmp_path / "o.mp4"
    # два клипа
    w1 = tmp_path / "a.wav"; w1.write_bytes(b"x")
    w2 = tmp_path / "b.wav"; w2.write_bytes(b"x")
    timeline = [(0, w1, 0.8), (500, w2, 0.6)]

    captured = {}

    def fake_run(cmd, check, capture_output):
        captured["cmd"] = cmd
        # эмуляция записи выхода
        out.write_bytes(b"mp4data")
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(audio_mix.subprocess, "run", fake_run)
    mix_audio_into_mp4(webm, timeline, out, fps=30)
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "adelay=0|0" in fc
    assert "adelay=500|500" in fc
    assert "volume=0.800" in fc
    assert "amix=inputs=2:duration=longest:normalize=0[aout]" in fc
    assert "-map" in cmd and "0:v" in cmd and "[aout]" in cmd
    assert "-r" in cmd and "30" in cmd
    assert "-shortest" in cmd
    assert str(out) in cmd


def test_mix_audio_video_only_when_empty_timeline(monkeypatch, tmp_path):
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "o.mp4"
    captured = {}
    def fake_run(cmd, check, capture_output):
        captured["cmd"] = cmd; out.write_bytes(b"mp4"); return mock.MagicMock(returncode=0)
    monkeypatch.setattr(audio_mix.subprocess, "run", fake_run)
    mix_audio_into_mp4(webm, [], out, fps=30)
    cmd = captured["cmd"]
    fc_idx = cmd.index("-filter_complex") if "-filter_complex" in cmd else -1
    assert fc_idx == -1  # без аудио — без filter_complex
    assert "-map" not in cmd or "[aout]" not in cmd
    assert "-c:a" not in cmd


def test_mix_audio_music_added_to_filter(monkeypatch, tmp_path):
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "o.mp4"
    w1 = tmp_path / "a.wav"; w1.write_bytes(b"x")
    music = tmp_path / "theme.wav"; music.write_bytes(b"x")
    timeline = [(0, w1, 0.8)]
    captured = {}
    def fake_run(cmd, check, capture_output):
        captured["cmd"] = cmd; out.write_bytes(b"mp4"); return mock.MagicMock(returncode=0)
    monkeypatch.setattr(audio_mix.subprocess, "run", fake_run)
    mix_audio_into_mp4(webm, timeline, out, fps=30, music_path=music, music_volume=0.3)
    cmd = captured["cmd"]
    fc = cmd[cmd.index("-filter_complex") + 1]
    # music — бесконечный луп (-stream_loop -1 перед входом)
    assert "-stream_loop" in cmd and "-1" in cmd
    assert str(music) in cmd
    # music-вход идёт ПОСЛЕ sfx: 0=video, 1=sfx, 2=music
    assert "[2:a]volume=0.300[amusic]" in fc
    # амикс включает sfx + music → inputs=2
    assert "amix=inputs=2:duration=longest:normalize=0[aout]" in fc
    assert "-shortest" in cmd  # обрезать по длине видео


def test_mix_audio_music_only_no_sfx(monkeypatch, tmp_path):
    """Пустой SFX-timeline + музыка → всё равно аудиодорожка (только музыка)."""
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "o.mp4"
    music = tmp_path / "theme.wav"; music.write_bytes(b"x")
    captured = {}
    def fake_run(cmd, check, capture_output):
        captured["cmd"] = cmd; out.write_bytes(b"mp4"); return mock.MagicMock(returncode=0)
    monkeypatch.setattr(audio_mix.subprocess, "run", fake_run)
    mix_audio_into_mp4(webm, [], out, fps=30, music_path=music)
    cmd = captured["cmd"]
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]volume=0.300[amusic]" in fc  # 0=video, 1=music
    assert "amix=inputs=1:duration=longest:normalize=0[aout]" in fc
    assert "-shortest" in cmd