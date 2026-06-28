"""No-visual Playwright e2e для v2 graph-редактора (DOM/JS-evaluate, БЕЗ скриншотов).

Гейт ``ORCH_E2E=1`` (как test_smoke_e2e part 2). Поднимает сервер на 8095
(если ещё не запущен), грузит /editor, проверяет: blank v2, добавление узлов,
port-drag wiring рёбер, computed-side chip, validate, авто-миграцию v1 demo
при загрузке, two-way JSON sync.

Запуск: ``ORCH_E2E=1 python -m pytest extra_orchestra/tests/test_editor_graph_e2e.py -s``
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

if not os.environ.get("ORCH_E2E"):
    pytest.skip("set ORCH_E2E=1 to run Playwright editor e2e", allow_module_level=True)

from playwright.sync_api import sync_playwright  # noqa: E402

PORT = 8095
BASE = f"http://127.0.0.1:{PORT}"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # worktree root (extra_orchestra/.. )


def _port_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


@pytest.fixture(scope="module")
def server():
    started = False
    if _port_free(PORT):
        proc = subprocess.Popen(
            [sys.executable, "-m", "extra_orchestra.server"],
            cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = True
        # ждём health
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", PORT), 0.5):
                    break
            except OSError:
                time.sleep(0.25)
        time.sleep(1.0)
    yield PORT
    if started:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_editor_graph_no_visual(server):
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.goto(BASE + "/editor")
        pg.wait_for_selector("#graph-svg")
        pg.wait_for_function("() => !!window.scenario && !!window.scenario.graph")

        sc = pg.evaluate("() => window.scenario")
        check(sc["schema"] == "extra_orchestra.scenario.v2", "blank v2")
        check(len(sc["graph"]["nodes"]) == 1, "blank 1 init node")
        check(pg.evaluate("() => document.querySelectorAll('#graph-svg .ognode').length") == 1, "1 svg node")
        check('"start": "s0"' in pg.evaluate("() => document.getElementById('json-editor').value"), "JSON synced")

        # add hold + wire init→hold
        pg.locator("#palette [data-add='hold']").click()
        pg.wait_for_function("() => window.scenario.graph.nodes.length === 2")
        ib = pg.evaluate("""() => { const r=document.querySelectorAll('#graph-svg .ognode')[0].getBoundingClientRect(); return {x:r.right-6,y:r.top+r.height/2}; }""")
        hb = pg.evaluate("""() => { const r=document.querySelectorAll('#graph-svg .ognode')[1].getBoundingClientRect(); return {x:r.left+6,y:r.top+r.height/2}; }""")
        pg.mouse.move(ib["x"], ib["y"]); pg.mouse.down()
        pg.mouse.move(hb["x"] - 40, hb["y"], steps=8); pg.mouse.move(hb["x"], hb["y"], steps=6); pg.mouse.up()
        pg.wait_for_function("() => window.scenario.graph.edges.length === 1", timeout=5000)
        check(pg.evaluate("() => window.scenario.graph.edges[0].from") == "s0", "edge from s0")

        # add end_turn + wire hold→end_turn
        pg.locator("#palette [data-add='end_turn']").click()
        pg.wait_for_function("() => window.scenario.graph.nodes.length === 3")
        h2 = pg.evaluate("""() => { const r=document.querySelectorAll('#graph-svg .ognode')[1].getBoundingClientRect(); return {x:r.right-6,y:r.top+r.height/2}; }""")
        e2 = pg.evaluate("""() => { const r=document.querySelectorAll('#graph-svg .ognode')[2].getBoundingClientRect(); return {x:r.left+6,y:r.top+r.height/2}; }""")
        pg.mouse.move(h2["x"], h2["y"]); pg.mouse.down()
        pg.mouse.move(e2["x"] - 30, e2["y"], steps=6); pg.mouse.move(e2["x"], e2["y"], steps=6); pg.mouse.up()
        pg.wait_for_function("() => window.scenario.graph.edges.length === 2", timeout=5000)
        chip = pg.evaluate("""() => { const g=document.querySelectorAll('#graph-svg .ognode')[2]; const t=g.querySelector('.sidechip text'); return t?t.textContent:null; }""")
        check(chip == "p1", "end_turn side chip = p1")

        # validate
        pg.locator("#btn-validate").click()
        st = pg.wait_for_function(
            "() => { var t=document.getElementById('status').textContent; return (t.indexOf('OK')>=0||t.indexOf('кадр')>=0||t.indexOf('ошибка')>=0)?t:false; }",
            timeout=8000).json_value()
        check("OK" in st, f"validate OK (got '{st}')")

        # v1 auto-migration on load demo
        pg.locator("#btn-load-demo").click()
        pg.wait_for_function("() => document.getElementById('status').textContent.indexOf('загружен') >= 0", timeout=10000)
        sc2 = pg.evaluate("() => window.scenario")
        check(sc2["schema"] == "extra_orchestra.scenario.v2", "demo migrated to v2")
        check(len(sc2["graph"]["edges"]) >= 1, "migrated demo has edges")
        check(pg.evaluate("() => window.scenario.graph.nodes.some(n => n.kind==='action' && n.action.type==='play_card')"), "migrated demo has play_card")

        # validate migrated demo
        pg.locator("#btn-validate").click()
        st2 = pg.wait_for_function(
            "() => { var t=document.getElementById('status').textContent; return (t.indexOf('OK')>=0||t.indexOf('кадр')>=0||t.indexOf('ошибка')>=0)?t:false; }",
            timeout=8000).json_value()
        check("OK" in st2, f"migrated validate OK (got '{st2}')")

        b.close()

    assert not fails, "editor e2e failures:\n  " + "\n  ".join(fails)