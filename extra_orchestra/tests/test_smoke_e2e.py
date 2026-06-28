"""End-to-end smoke.

Часть 1 (fast, всегда): HTTP-контракт оркестра — каталог карт (50),
сценарии, validate, compute-frames, fetch frames — через aiohttp TestClient.

Часть 2 (slow, gated ``ORCH_E2E=1``): полный Playwright-pipeline против
живого сервера — record_run_to_mp4(soldatik) → mp4 (h264, мобильный портрет
828x1792 @ device_scale_factor=2, 30fps, audio-дорожка с SFX + музыкой).
Запуск: ``ORCH_E2E=1 python -m pytest extra_orchestra/tests/
test_smoke_e2e.py -k e2e -s``.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from extra_orchestra.server import OrchestraServer

HERE = Path(__file__).resolve().parent
SOLDATIK = HERE.parent / "scenarios" / "soldatik-demo.json"


def _server_port(server: OrchestraServer, test_server: TestServer) -> int:
    # TestServer/TestClient поднимаются на ephemeral-порту после start_server.
    port = getattr(test_server, "port", None)
    if port:
        return port
    raise RuntimeError("could not determine test server port")


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    import json as _json
    v = (_json.loads(out).get("streams") or [{}])[0]
    # audio?
    cmd_a = ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "json", str(path)]
    out_a = subprocess.run(cmd_a, capture_output=True, text=True).stdout
    has_audio = bool(_json.loads(out_a).get("streams"))
    return {"vcodec": v.get("codec_name", ""), "width": v.get("width"), "height": v.get("height"),
            "has_audio": has_audio}


@pytest_asyncio.fixture
async def client():
    server = OrchestraServer("127.0.0.1", 0)
    test_server = TestServer(server.app)
    cli = TestClient(test_server)
    await cli.start_server()
    yield cli, server, test_server
    await cli.close()


# ----------------------------- HTTP smoke --------------------------------


@pytest.mark.asyncio
async def test_cards_count_50(client):
    cli, _server, _ts = client
    r = await cli.get("/api/orchestra/cards")
    data = await r.json()
    assert len(data["cards"]) == 50
    ids = {c["id"] for c in data["cards"]}
    assert {47, 24, 30, 34} <= ids  # Солдатик + 3 врага демо


@pytest.mark.asyncio
async def test_scenarios_list_has_soldatik(client):
    cli, _server, _ts = client
    r = await cli.get("/api/orchestra/scenarios")
    data = await r.json()
    names = [s["name"] for s in data["scenarios"]]
    assert "Солдатик demo" in names


@pytest.mark.asyncio
async def test_validate_and_compute_frames(client):
    cli, _server, _ts = client
    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    r = await cli.post("/api/orchestra/validate", json=sc)
    v = await r.json()
    assert v["ok"], v["error"]
    assert v["frame_count"] >= 3
    r = await cli.post("/api/orchestra/compute-frames", json=sc)
    d = await r.json()
    assert d["error"] is None, d["error"]
    run_id = d["run_id"]
    r = await cli.get(f"/api/orchestra/frames/{run_id}")
    run = await r.json()
    assert run["frame_count"] == d["frame_count"]
    assert run["frames"][0]["snapshot"]["turn"] == 15
    r = await cli.get(f"/api/battle/state?match_id={run_id}")
    st = await r.json()
    assert st["turn"] == 15


@pytest.mark.asyncio
async def test_contract_stubs_present(client):
    cli, _server, _ts = client
    for path in ("/api/runtime/status", "/api/settings", "/api/cards", "/health"):
        r = await cli.get(path)
        assert r.status == 200, (path, r.status)
    st = await (await cli.get("/api/runtime/status")).json()
    assert st["maintenance_mode"]["enabled"] is False


# --------------------------- Playwright e2e -------------------------------


@pytest.mark.asyncio
async def test_e2e_record_mp4(client, tmp_path):
    if os.environ.get("ORCH_E2E") != "1":
        pytest.skip("set ORCH_E2E=1 to run Playwright mp4 e2e")
    cli, server, test_server = client
    port = _server_port(server, test_server)
    base_url = f"http://127.0.0.1:{port}"

    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    r = await cli.post("/api/orchestra/compute-frames", json=sc)
    d = await r.json()
    assert d["error"] is None, d["error"]
    run = server._runs[d["run_id"]]

    from extra_orchestra.components.recorder import record_run_to_mp4
    out = tmp_path / "soldatik.mp4"
    cfg = {"fps": 30, "width": 414, "height": 896, "device_scale_factor": 2,
           "headless": True, "with_audio": True}
    mp4 = await asyncio.to_thread(
        record_run_to_mp4, run, str(out), cfg, d["viewer_uid"], base_url=base_url, speed=1.5
    )
    assert Path(mp4).exists() and Path(mp4).stat().st_size > 1000
    info = ffprobe(Path(mp4))
    assert "h264" in info.get("vcodec", "")
    # мобильный портрет: 414×896 CSS × device_scale_factor 2 = 828×1792
    assert info.get("width") == 828 and info.get("height") == 1792
    assert info.get("height") > info.get("width")  # portrait (мобильное соотношение)
    assert info.get("has_audio") is True  # deploy Солдатика → аудио-дорожка (SFX + музыка)


@pytest.mark.asyncio
async def test_e2e_record_gif(client, tmp_path):
    if os.environ.get("ORCH_E2E") != "1":
        pytest.skip("set ORCH_E2E=1 to run Playwright gif e2e")
    cli, server, test_server = client
    port = _server_port(server, test_server)
    base_url = f"http://127.0.0.1:{port}"

    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    r = await cli.post("/api/orchestra/compute-frames", json=sc)
    d = await r.json()
    assert d["error"] is None, d["error"]
    run = server._runs[d["run_id"]]

    from extra_orchestra.components.recorder import record_run_to_gif
    out = tmp_path / "soldatik.gif"
    cfg = {"fps": 30, "width": 414, "height": 896, "device_scale_factor": 2,
           "headless": True, "with_audio": True, "gif_fps": 15, "gif_width": 540}
    gif = await asyncio.to_thread(
        record_run_to_gif, run, str(out), cfg, d["viewer_uid"], base_url=base_url, speed=1.5
    )
    assert Path(gif).exists() and Path(gif).stat().st_size > 1000
    # ffprobe: GIF — image/gif, портрет (высота > ширины при width=540).
    info = ffprobe(Path(gif))
    assert info.get("vcodec") in ("gif", "animatedgif", "giflib") or info.get("vcodec", "").endswith("gif")
    assert info.get("height", 0) > info.get("width", 0)  # portrait
    # GIF не имеет аудио-дорожки (формат не поддерживает звук)
    assert info.get("has_audio") is False


# ----------------------------- path-traversal ------------------------------


@pytest.mark.asyncio
async def test_record_name_traversal_sanitized(client, monkeypatch):
    """scenario.name с path-traversal санитизуется — файл записи не выходит из RECORDINGS_DIR.

    Регрессия на finding verification-воркфлоу: unsanitized ``name`` позволял
    ``../`` выйти за пределы ``recordings/``. Рекордер подменён на no-op, чтобы
    не запускать Playwright — проверяется только синхронно вычисленное имя.
    """
    from extra_orchestra.server import RECORDINGS_DIR, _safe_name_slug

    # unit: чистая функция — точки/слеши/``..`` схлопываются, пустое → orchestra
    assert _safe_name_slug("../../etc/evil") == "etc_evil"
    assert _safe_name_slug("../../../tmp/x") == "tmp_x"
    assert _safe_name_slug("") == "orchestra"
    assert _safe_name_slug("....//y") == "y"
    assert _safe_name_slug("demo.v2") == "demo_v2"  # точка → разделитель
    assert ".." not in _safe_name_slug("../..")
    assert "/" not in _safe_name_slug("/etc/passwd")

    # integration: эндпоинт отдаёт чистое имя, путь остаётся внутри RECORDINGS_DIR
    cli, server, _ts = client
    monkeypatch.setattr(server, "_run_record_job", lambda *a, **k: None)
    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    sc["name"] = "../../etc/evil"
    r = await cli.post("/api/orchestra/record", json=sc)
    assert r.status == 200, await r.text()
    d = await r.json()
    fname = d["file_name"]
    assert "/" not in fname and ".." not in fname
    assert fname.endswith(".mp4")
    (RECORDINGS_DIR / fname).resolve().relative_to(RECORDINGS_DIR.resolve())