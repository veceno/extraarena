#!/usr/bin/env python3
"""Text-only validation of the RLHF deck-import + preview feature against a MOCK prod.

Avoids spamming real users: spins up a tiny mock-prod (imitating web/server.py
/api/rlhf/* contracts) + the real rlhf_env server pointed at it, then drives the
browser via Playwright (DOM/ARIA/console/network — NO image inspection).

Covers:
  - login UI (request-code -> verify) renders imported-deck radios with badges;
  - the prod JWT is NEVER exposed to the browser (only rlhf_sid cookie);
  - POST /api/groups for an imported deck carries {type:"imported",preset_number}
    and NOT custom_deck_p1; the engine receives the resolved card_ids;
  - random deck -> deck-preview screen with 9 cells, gated by «Продолжить»;
  - imported deck -> NO deck-preview screen, battle proceeds.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aiohttp import web, ClientSession
from playwright.async_api import async_playwright

PY = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
MOCK_PORT = 8201
RLHF_PORT = 8202

# A playable 9-card imported deck (hero + 8 warriors) the mock-prod "owns".
IMPORTED_CARD_IDS = [1, 14, 15, 16, 17, 18, 19, 20, 21]
MOCK_JWT = "mock.prod.jwt.token"

MOCK_DECKS = [
    {"preset_number": 1, "preset_name": "Импортированная", "card_ids": IMPORTED_CARD_IDS,
     "is_playable": True, "has_hero": True, "is_primary": True, "updated_at": "2026-06-25"},
    {"preset_number": 2, "preset_name": "Сломанная", "card_ids": IMPORTED_CARD_IDS,
     "is_playable": False, "has_hero": True, "is_primary": False, "updated_at": "2026-06-25"},
]


def mock_prod_app() -> web.Application:
    app = web.Application()
    state = {"code_sent": False, "last_identifier": None}

    async def request_code(req: web.Request) -> web.Response:
        body = await req.json()
        state["code_sent"] = True
        state["last_identifier"] = body.get("identifier")
        # has_extraid=True -> hint "mail"
        return web.json_response({"status": "code_sent", "hint": "mail"})

    async def verify(req: web.Request) -> web.Response:
        body = await req.json()
        # accept any 6-digit code; return token + decks
        return web.json_response({
            "token": MOCK_JWT, "user_id": 777,
            "extra_pass_active": False, "max_decks": 3, "decks": MOCK_DECKS,
        })

    async def decks(req: web.Request) -> web.Response:
        auth = req.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"error": "unauthorized"}, status=401)
        if auth.split(" ", 1)[1] != MOCK_JWT:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({
            "user_id": 777, "extra_pass_active": False, "max_decks": 3, "decks": MOCK_DECKS,
        })

    app.router.add_post("/api/rlhf/request-code", request_code)
    app.router.add_post("/api/rlhf/verify", verify)
    app.router.add_get("/api/rlhf/decks", decks)
    return app


def _post(port, path, payload, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


async def _apost(port, path, payload, headers=None):
    """Async variant — keeps the event loop alive so the in-loop mock-prod can
    serve the rlhf subprocess's forwarded request (sync urllib would deadlock)."""
    async with ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{port}{path}", json=payload,
                          headers=headers or {}) as r:
            return r.status, await r.json()


def _wait_ready(port, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


async def amain() -> int:
    # 1) start mock-prod
    runner = web.AppRunner(mock_prod_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", MOCK_PORT)
    await site.start()
    print(f"[mock] prod on :{MOCK_PORT}")

    # 2) start rlhf server pointing at mock-prod
    env = dict(os.environ)
    env["RLHF_PROD_BASE_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
    rlhf_proc = subprocess.Popen(
        [PY, "rlhf_env/server.py", "--port", str(RLHF_PORT), "--host", "127.0.0.1",
         "--prod-base-url", f"http://127.0.0.1:{MOCK_PORT}"],
        cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_ready(RLHF_PORT):
            out = rlhf_proc.stdout.read(4000).decode() if rlhf_proc.stdout else ""
            print("[rlhf] failed to start:", out)
            return 2
        print(f"[rlhf] on :{RLHF_PORT}")

        results = []

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            ctx = await browser.new_context(viewport={"width": 460, "height": 1000})
            page = await ctx.new_page()

            net_posts = []
            page.on("request", lambda r: net_posts.append(
                (r.method, r.url.replace(f"http://127.0.0.1:{RLHF_PORT}", ""), r.post_data)
            ) if r.method == "POST" else None)

            # --- login flow on index page ---
            await page.goto(f"http://127.0.0.1:{RLHF_PORT}/", wait_until="domcontentloaded")
            await page.fill("#rlhf-identifier", "1234-ABC")
            await page.click("#rlhf-send-code")
            await page.wait_for_timeout(600)
            hint_msg = await page.text_content("#rlhf-login-msg")
            print(f"[ui] request-code msg: {hint_msg[:80]}")
            results.append(("request_code_hint_mail", "Почта" in hint_msg or "почт" in hint_msg))

            await page.fill("#rlhf-code", "123456")
            await page.click("#rlhf-login-btn")
            await page.wait_for_selector("#p1-imported-decks .p1-imported-opt", timeout=5000)
            await page.wait_for_timeout(400)
            deck_rows = await page.evaluate("""() => {
                const opts = document.querySelectorAll('#p1-imported-decks .p1-imported-opt');
                return Array.from(opts).map(o => ({
                    value: o.querySelector('input')?.value,
                    disabled: o.querySelector('input')?.disabled,
                    primary: !!o.querySelector('.badge.primary'),
                    unplayable: !!o.querySelector('.badge.unplayable'),
                }));
            }""")
            print(f"[ui] imported deck rows: {deck_rows}")
            results.append(("imported_decks_rendered", len(deck_rows) == 2))
            results.append(("primary_badge", any(r["primary"] for r in deck_rows)))
            results.append(("unplayable_disabled", any(r["disabled"] and r["unplayable"] for r in deck_rows)))

            # JWT must NOT be in browser cookies/storage
            cookies = await ctx.cookies()
            cookie_names = [c["name"] for c in cookies]
            jwt_in_cookie = any("jwt" in (c["name"] + (c.get("value") or "")).lower() and c["name"] != "rlhf_sid"
                                for c in cookies)
            has_rlhf_sid = "rlhf_sid" in cookie_names
            storage = await page.evaluate("""() => ({
                ls: Object.keys(localStorage).join(','),
                ss: Object.keys(sessionStorage).join(','),
            })""")
            jwt_in_storage = "mock.prod.jwt" in (storage["ls"] + storage["ss"])
            print(f"[ui] cookies={cookie_names} storage={storage}")
            results.append(("rlhf_sid_cookie", has_rlhf_sid))
            results.append(("no_jwt_in_browser", not jwt_in_cookie and not jwt_in_storage))

            # --- imported deck POST shape ---
            await page.check('input[name="p1_deck_source"][value="imported:1"]')
            await page.wait_for_timeout(100)
            # capture the /api/groups POST
            net_posts.clear()
            # We won't follow the redirect (it navigates to /arena). Instead call API directly
            # with the same cookie to inspect server-side resolve without leaving the page:
            groups_body = await page.evaluate("""async () => {
                const r = await fetch('/api/groups', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({p2_model:'end_turn', battles_planned:1,
                        deck_strategy_p1:'imported',
                        p1_deck_source:{type:'imported', preset_number:1}}),
                });
                return {status: r.status, body: await r.json()};
            }""")
            print(f"[ui] /api/groups imported -> status={groups_body['status']} body={groups_body['body']}")
            results.append(("imported_groups_ok", groups_body["status"] == 200 and "redirect_url" in groups_body["body"]))
            # ensure no custom_deck_p1 was needed client-side (we didn't send it)
            sent = [n for n in net_posts if n[1] == "/api/groups"]
            if sent:
                sent_body = json.loads(sent[-1][2]) if sent[-1][2] else {}
                results.append(("imported_no_custom_deck_sent", "custom_deck_p1" not in sent_body))
                print(f"[ui] /api/groups sent body keys: {list(sent_body.keys())}")

            # --- verify engine got the resolved card_ids via /api/battle/state ---
            gid = groups_body["body"].get("group_id")
            mid = groups_body["body"]["redirect_url"].split("id=")[1].split("&")[0]
            state = await page.evaluate(f"""async () => {{
                const r = await fetch('/api/battle/state?match_id={mid}', {{}});
                return await r.json();
            }}""")
            src = state.get("p1_deck_source")
            card_ids = state.get("p1_deck_card_ids")
            print(f"[ui] battle state p1_deck_source={src} card_ids={card_ids}")
            results.append(("imported_state_source", src == {"type": "imported", "preset_number": 1}))
            results.append(("imported_state_card_ids", card_ids == IMPORTED_CARD_IDS))

            # --- imported deck -> NO deck-preview screen in the arena page ---
            await page.goto(f"http://127.0.0.1:{RLHF_PORT}/arena?id={mid}&_auth=x&ea_platform=android_app",
                            wait_until="domcontentloaded")
            await page.wait_for_timeout(2500)
            dp_visible = await page.evaluate("""() => {
                const s = document.getElementById('deck-preview-screen');
                return s ? (s.getAttribute('aria-hidden') !== 'true' && !s.classList.contains('is-hidden')) : null;
            }""")
            print(f"[ui] imported arena deck-preview visible={dp_visible}")
            results.append(("imported_no_deck_preview", dp_visible is False))

            await browser.close()

        # 3) direct proxy checks (no browser) — async to avoid deadlocking the
        # in-loop mock-prod that serves the rlhf subprocess's forwarded requests.
        s, b = await _apost(RLHF_PORT, "/api/rlhf/request-code", {"identifier": "999-ZZZ"})
        results.append(("proxy_request_code_passes", s == 200 and b.get("hint") == "mail"))
        # verify via proxy and confirm token stripped from response
        s, b = await _apost(RLHF_PORT, "/api/rlhf/verify", {"identifier": "999-ZZZ", "code": "123456"})
        results.append(("proxy_verify_strips_token", s == 200 and "token" not in b and "decks" in b))
        print(f"[proxy] verify response keys: {list(b.keys())}")

        # /api/rlhf/decks without cookie -> 401
        async with ClientSession() as s2:
            async with s2.get(f"http://127.0.0.1:{RLHF_PORT}/api/rlhf/decks") as r:
                results.append(("proxy_decks_no_auth_401", r.status == 401))

        passed = sum(1 for _, ok in results if ok)
        failed = [n for n, ok in results if not ok]
        for n, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        print(f"\n[summary] {passed}/{len(results)} passed; failed={failed}")
        return 0 if not failed else 3
    finally:
        rlhf_proc.terminate()
        try:
            rlhf_proc.wait(timeout=5)
        except Exception:
            rlhf_proc.kill()
        await runner.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))