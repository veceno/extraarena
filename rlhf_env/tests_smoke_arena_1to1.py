"""Playwright text-only smoke-тест арены 1:1 (без image input).

Верификация рендера через DOM/ARIA/console/network, а не скриншоты.
Снимок сохраняется на диск для отдельного vision-review, но НЕ анализируется здесь.

Запуск (сервер на RLHF_SMOKE_PORT, по умолчанию 8101):
    python3 rlhf_env/tests_smoke_arena_1to1.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from playwright.async_api import async_playwright

PORT = int(os.environ.get("RLHF_SMOKE_PORT", "8101"))
BASE = f"http://127.0.0.1:{PORT}"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


async def main() -> int:
    body = _post("/api/match/find", {
        "p2_model": "random",
        "battles_planned": 1, "seed": 31337,
    })
    redirect_url = body["redirect_url"]
    match_id = body["match_id"]
    print(f"[smoke] match_id={match_id}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 420, "height": 900})
        page = await ctx.new_page()

        console_logs: list[str] = []
        pageerrors: list[str] = []
        page.on("console", lambda m: console_logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: pageerrors.append(str(e)))

        net_status: dict[str, int] = {}
        page.on("response", lambda r: net_status.__setitem__(r.url.replace(BASE, ""), r.status))

        await page.goto(BASE + redirect_url, wait_until="domcontentloaded")

        # 1) Socket.IO: ждём joined_match + match_ready (поллинг console-логов).
        async def _wait_console(pred, timeout_ms=15000):
            elapsed = 0
            while elapsed < timeout_ms:
                if pred(console_logs):
                    return True
                await page.wait_for_timeout(250)
                elapsed += 250
            return pred(console_logs)

        joined = await _wait_console(lambda L: any("Вступили в матч" in l for l in L))
        print(f"[smoke] socket: joined_match={joined}")

        # RLHF deck-preview gate: для случайной колоды arena.js показывает экран
        # превью и ждёт клик «Продолжить» перед запуском отсчёта (по спеке фичи).
        # Кликаем, чтобы бой пошёл; для импортированной/нестучайной колоды экрана нет.
        clicked_preview = False
        try:
            await page.wait_for_selector("#deck-preview-continue:visible", timeout=8000)
            dp_cells = await page.evaluate(
                "() => document.querySelectorAll('#deck-preview-grid .deck-preview-cell').length"
            )
            dp_visible = await page.evaluate("""() => {
                const s = document.getElementById('deck-preview-screen');
                return s ? s.getAttribute('aria-hidden') !== 'true' && !s.classList.contains('is-hidden') : false;
            }""")
            print(f"[smoke] deck-preview visible={dp_visible} cells={dp_cells}")
            await page.click("#deck-preview-continue")
            clicked_preview = True
            print("[smoke] clicked deck-preview «Продолжить»")
        except Exception:
            print("[smoke] no deck-preview gate (non-random deck) — skip")
        print(f"[smoke] deck-preview clicked={clicked_preview}")

        ready = await _wait_console(lambda L: any("match_ready" in l for l in L))
        print(f"[smoke] socket: match_ready={ready}")

        # 2) нет fatal abort
        body_text = await page.evaluate("document.body ? document.body.innerText.slice(0,500) : ''")
        assert "Браузер не поддерживается" not in body_text, "arena.js aborted (ea_platform missing)"
        print("[smoke] no 'Браузер не поддерживается' abort")

        # 3) root container + computed dimensions
        root_info = await page.evaluate("""() => {
            const c = document.querySelector('#arena-root, #game-container, #arena-app, #app, body > div');
            if (!c) return null;
            const r = c.getBoundingClientRect();
            const cs = getComputedStyle(c);
            return {tag: c.tagName, id: c.id, cls: c.className,
                    w: Math.round(r.width), h: Math.round(r.height),
                    display: cs.display, vis: cs.visibility};
        }""")
        print(f"[smoke] root container: {root_info}")

        # 4) battle screen DOM markers (hand/board) — ждём, что что-то отрендерилось
        await page.wait_for_timeout(2500)
        dom_counts = await page.evaluate("""() => {
            const q = (s) => document.querySelectorAll(s).length;
            return {
                hand_cards: q('.hand-card, .player-hand .card, [data-hand-card]'),
                board_units: q('.board-unit, .board .card, .unit, .board-card, [data-board-unit]'),
                hero_p1: q('.player-hero, .hero.player, [data-hero-player]'),
                hero_p2: q('.opponent-hero, .hero.opponent, [data-hero-opponent]'),
                turn_timer: q('#turn-timer-text, .turn-timer, [data-turn-timer]'),
            };
        }""")
        print(f"[smoke] DOM counts: {dom_counts}")

        # 5) ARIA snapshot (text-only) — структура арены через locator.aria_snapshot()
        try:
            snap_text = await page.locator("body").aria_snapshot()
            lines = [ln for ln in snap_text.splitlines() if ln.strip()]
            arena_texts = [ln for ln in lines if any(k in ln for k in
                          ["HP", "мана", "ход", "Сдаться", "сда", "Берсек", "Вы", "бой", "HP "])]
            print(f"[smoke] ARIA lines={len(lines)} | arena-relevant={len(arena_texts)}")
            for n in arena_texts[:10]:
                print("   ARIA:", n.strip()[:80])
        except Exception as exc:
            print(f"[smoke] ARIA snapshot failed: {exc}")

        # 6) network: критичные 200, никаких 5xx на JS/CSS
        crit = {k.split("?")[0]: v for k, v in net_status.items()}
        for path in ["/arena.js", "/arena-styles.css", "/safe-area.js", "/arena"]:
            st = crit.get(path)
            print(f"[smoke] net {path}: {st}")
        asset_200 = sum(1 for k, v in net_status.items() if k.startswith("/DesignAssets/") and v == 200)
        asset_404 = sum(1 for k, v in net_status.items() if k.startswith("/DesignAssets/") and v == 404)
        print(f"[smoke] DesignAssets: {asset_200} ok / {asset_404} 404")
        js5xx = [k for k, v in net_status.items() if k.endswith(".js") and v >= 500]
        print(f"[smoke] JS 5xx: {js5xx}")

        # 7) pageerrors
        fatal_errors = [e for e in pageerrors if "Браузер не поддерживается" not in e]
        print(f"[smoke] pageerrors: {len(pageerrors)} (fatal={len(fatal_errors)})")
        for e in pageerrors[:5]:
            print("   ERR:", e[:160])

        # 8) legal_actions из currentState (если ещё активен)
        la = await page.evaluate("window.currentState ? (window.currentState.legal_actions||[]).length : -1")
        cur_mid = await page.evaluate("window.currentState ? window.currentState.match_id : null")
        print(f"[smoke] currentState.legal_actions length = {la} (match_id={cur_mid})")

        # screenshot (для отдельного vision-review, не анализируется)
        shot = REPO / "rlhf_env" / "smoke_arena.png"
        await page.screenshot(path=str(shot), full_page=True)
        print(f"[smoke] screenshot saved (requires vision review): {shot}")

        await browser.close()

        ok = (
            joined and ready
            and root_info is not None
            and root_info.get("h", 0) > 100
            and "Браузер не поддерживается" not in body_text
            and not fatal_errors
            and crit.get("/arena.js") == 200
            and crit.get("/arena-styles.css") == 200
        )
        print("[smoke] RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))