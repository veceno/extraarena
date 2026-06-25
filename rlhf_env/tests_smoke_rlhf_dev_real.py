#!/usr/bin/env python3
"""Текст-only (без image input) проверка реального RLHF-флоу на ДЕВ-окружении.

В отличие от tests_smoke_rlhf_decks.py (mock-prod), тут идём через живой rlhf-прокси
(порт 8090 → дев-инстанс игры 8082) с реальной БД и синтетическим аккаунтом:
  1. UI: вводим display_id → «Отправить код» → достаём код из user_mail (psql);
  2. UI: вводим код → «Войти» → рендер импортированных колод с бейджами;
  3. выбираем импортированную колоду → старт серии → /arena?id=…;
  4. для импортированной колоды экран превью НЕ появляется (сразу в бой);
  5. прод-JWT НЕ попадает в браузер (только cookie rlhf_sid).

Синтетический user_id >= 9_100_000_000_000 → telegram_linked=False → код идёт в
почту, Telegram не дёргается (нет спама реальным юзерам).

Запуск (rlhf 8090 + дев-инстанс 8082 должны быть подняты):
    python3 rlhf_env/tests_smoke_rlhf_dev_real.py
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from playwright.async_api import async_playwright

RLHF_PORT = int(os.environ.get("RLHF_PORT", "8090"))
BASE = f"http://127.0.0.1:{RLHF_PORT}"
# Синтетический аккаунт, созданный rlhf_env/dev_seed_synthetic.py.
DISPLAY_ID = os.environ.get("RLHF_DEV_DISPLAY_ID", "2371-UPT")
USER_ID = os.environ.get("RLHF_DEV_USER_ID", "9100000000009")

PSQL = "/Applications/Postgres.app/Contents/Versions/18/bin/psql"
PSQL_DSN_ARGS = ["-h", "localhost", "-p", "5434", "-U", "postgres", "-d", "laveqox", "-tAc"]


def fetch_code_from_mail() -> str:
    sql = (f"SELECT text FROM user_mail WHERE user_id={USER_ID} "
           f"AND category='rlhf_login' ORDER BY created_at DESC LIMIT 1;")
    out = subprocess.check_output([PSQL, *PSQL_DSN_ARGS, sql], text=True, timeout=5)
    m = re.search(r"\b(\d{6})\b", out)
    return m.group(1) if m else ""


async def main() -> int:
    results: list[tuple[str, bool]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 460, "height": 1000})
        page = await ctx.new_page()

        net_posts: list[tuple[str, str]] = []
        page.on("request", lambda r: net_posts.append((r.method, r.url.replace(BASE, "")))
               if r.method == "POST" else None)

        # --- env badge present ---
        await page.goto(BASE + "/", wait_until="domcontentloaded")
        await page.wait_for_timeout(300)
        badge = await page.text_content("#rlhf-env-badge")
        print(f"[ui] env badge: {badge}")
        results.append(("env_badge_shown", badge is not None and "Среда подключения" in (badge or "")))

        # --- step 1: send code ---
        await page.fill("#rlhf-identifier", DISPLAY_ID)
        await page.click("#rlhf-send-code")
        await page.wait_for_timeout(800)
        hint = await page.text_content("#rlhf-login-msg")
        print(f"[ui] request-code hint: {(hint or '')[:80]}")
        results.append(("request_code_hint_mail", "почт" in (hint or "").lower()))

        # --- fetch code from in-game mail (real DB) ---
        code = fetch_code_from_mail()
        print(f"[ui] code from mail: {code}")
        results.append(("code_fetched_from_mail", bool(code) and len(code) == 6))

        # --- step 2: verify -> imported decks render ---
        await page.fill("#rlhf-code", code)
        await page.click("#rlhf-login-btn")
        await page.wait_for_selector("#p1-imported-decks .p1-imported-opt", timeout=8000)
        await page.wait_for_timeout(400)
        deck_rows = await page.evaluate("""() => Array.from(
            document.querySelectorAll('#p1-imported-decks .p1-imported-opt')).map(o => ({
                value: o.querySelector('input')?.value,
                disabled: o.querySelector('input')?.disabled,
                primary: !!o.querySelector('.badge.primary'),
                unplayable: !!o.querySelector('.badge.unplayable'),
            }))""")
        print(f"[ui] imported deck rows: {deck_rows}")
        results.append(("imported_decks_rendered", len(deck_rows) >= 1))
        results.append(("primary_badge", any(r["primary"] for r in deck_rows)))

        # JWT must NOT be in browser cookies/storage (only rlhf_sid)
        cookies = await ctx.cookies()
        cookie_names = [c["name"] for c in cookies]
        storage = await page.evaluate("""() => ({ls: Object.keys(localStorage).join(','),
            ss: Object.keys(sessionStorage).join(',')})""")
        jwt_leak = any("jwt" in (c["name"] + (c.get("value") or "")).lower() and c["name"] != "rlhf_sid"
                       for c in cookies)
        print(f"[ui] cookies={cookie_names} storage={storage}")
        results.append(("rlhf_sid_cookie", "rlhf_sid" in cookie_names))
        results.append(("no_jwt_in_browser", not jwt_leak and "jwt" not in (storage["ls"] + storage["ss"]).lower()))

        # --- step 3: select imported deck, start series, go to arena ---
        playable = next((r for r in deck_rows if not r["disabled"]), None)
        if playable:
            await page.check(f'input[name="p1_deck_source"][value="{playable["value"]}"]')
            await page.wait_for_timeout(150)
        # start series via the form submit (creates /arena?id= redirect)
        net_posts.clear()
        groups = await page.evaluate("""async () => {
            const v = document.querySelector('input[name=p1_deck_source]:checked')?.value || 'random';
            const body = {p2_model:'end_turn', battles_planned:1};
            if (v === 'random') { body.p1_deck_source = {type:'random'}; body.deck_strategy_p1='random_arenaenv'; }
            else { body.p1_deck_source = {type:'imported', preset_number: parseInt(v.split(':')[1],10)}; }
            const r = await fetch('/api/groups', {method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify(body)});
            return {status: r.status, body: await r.json()};
        }""")
        print(f"[ui] /api/groups imported -> status={groups['status']} body_keys={list(groups['body'].keys())}")
        results.append(("imported_groups_ok", groups["status"] == 200 and "redirect_url" in groups["body"]))

        mid = groups["body"]["redirect_url"].split("id=")[1].split("&")[0]
        # server-side resolve: engine got the imported card_ids
        state = await page.evaluate(f"""async () => {{
            const r = await fetch('/api/battle/state?match_id={mid}', {{}});
            return await r.json();
        }}""")
        src = state.get("p1_deck_source")
        card_ids = state.get("p1_deck_card_ids")
        print(f"[ui] battle state p1_deck_source={src} card_ids={card_ids}")
        results.append(("imported_state_source", src == {"type": "imported", "preset_number": 1}))
        results.append(("imported_state_card_ids", isinstance(card_ids, list) and len(card_ids) == 9))

        # --- step 4: imported deck -> NO deck-preview screen ---
        await page.goto(f"{BASE}/arena?id={mid}&_auth=x&ea_platform=android_app",
                        wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)
        dp_visible = await page.evaluate("""() => {
            const s = document.getElementById('deck-preview-screen');
            return s ? (s.getAttribute('aria-hidden') !== 'true' && !s.classList.contains('is-hidden')) : false;
        }""")
        print(f"[ui] imported arena deck-preview visible={dp_visible}")
        results.append(("imported_no_deck_preview", dp_visible is False))

        await browser.close()

    passed = sum(1 for _, ok in results if ok)
    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"\n[summary] {passed}/{len(results)} passed; failed={failed}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))