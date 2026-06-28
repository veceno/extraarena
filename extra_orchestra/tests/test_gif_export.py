"""Тесты gif_export: структура двухпроходной ffmpeg-цепочки (mock subprocess)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from extra_orchestra.components import gif_export
from extra_orchestra.components.gif_export import webm_to_gif


def _fake_run_factory(captured, out_gif):
    def fake_run(cmd, check, capture_output):
        captured.append(cmd)
        # palettegen пишет .palette.png; paletteuse пишет итоговый gif
        if "palettegen" in " ".join(cmd):
            pal = Path(cmd[-1])
            pal.write_bytes(b"PNG")
        else:
            out_gif.write_bytes(b"GIF89a")
        return mock.MagicMock(returncode=0)
    return fake_run


def _vf(cmd):
    return cmd[cmd.index("-vf") + 1]


def test_webm_to_gif_two_pass_palette(monkeypatch, tmp_path):
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "o.gif"
    captured = []
    monkeypatch.setattr(gif_export.subprocess, "run", _fake_run_factory(captured, out))
    webm_to_gif(webm, out, fps=15, width=540)
    assert out.exists()
    assert len(captured) == 2  # два прохода
    pal_cmd, use_cmd = captured
    # pass 1: palettegen
    vf = _vf(pal_cmd)
    assert "palettegen=stats_mode=diff" in vf
    assert "fps=15" in vf
    assert "scale=540:-2" in vf  # чётная высота (-2)
    assert "lanczos" in vf
    # pass 2: paletteuse
    fc = use_cmd[use_cmd.index("-filter_complex") + 1]
    assert "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" in fc
    assert "fps=15" in fc and "scale=540:-2" in fc
    assert str(out) in use_cmd
    # palette-файл удалён после проходов
    assert not (tmp_path / "o.palette.png").exists()


def test_webm_to_gif_native_width_no_scale(monkeypatch, tmp_path):
    """width=0 → native (без scale-фильтра), только fps."""
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "n.gif"
    captured = []
    monkeypatch.setattr(gif_export.subprocess, "run", _fake_run_factory(captured, out))
    webm_to_gif(webm, out, fps=12, width=0)
    vf = _vf(captured[0])
    assert "scale=" not in vf
    assert "fps=12" in vf
    assert "palettegen=stats_mode=diff" in vf


def test_webm_to_gif_even_width(monkeypatch, tmp_path):
    """Нечётная ширина приводится к чётной (libswscale)."""
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "odd.gif"
    captured = []
    monkeypatch.setattr(gif_export.subprocess, "run", _fake_run_factory(captured, out))
    webm_to_gif(webm, out, fps=15, width=541)  # 541 → 540
    assert "scale=540:-2" in _vf(captured[0])


def test_webm_to_gif_fps_clamped(monkeypatch, tmp_path):
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "f.gif"
    captured = []
    monkeypatch.setattr(gif_export.subprocess, "run", _fake_run_factory(captured, out))
    webm_to_gif(webm, out, fps=99, width=400)  # clamps to 30
    assert "fps=30" in _vf(captured[0])


def test_webm_to_gif_palettegen_failure_raises(monkeypatch, tmp_path):
    """Ошибка ffmpeg на pass 1 → RuntimeError с stderr, gif не создан."""
    import subprocess as sp
    webm = tmp_path / "v.webm"; webm.write_bytes(b"x")
    out = tmp_path / "bad.gif"

    def fake_run(cmd, check, capture_output):
        raise sp.CalledProcessError(1, cmd, stderr=b"boom stderr")

    monkeypatch.setattr(gif_export.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="gif palettegen failed"):
        webm_to_gif(webm, out, fps=15, width=400)