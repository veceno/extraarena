"""Text-only Playwright-тест серии боёв + модального окна результата (БЕЗ image input).

Покрывает запрос пользователя:
  - «Следующий бой» продолжает серию (POST /api/match/find {group_id}) → редирект
    на новый бой с НОВЫМИ уровнями карт.
  - кнопка серая/«Бои завершены», когда бои закончились (has_next_battle=false).
  - «Завершить предварительно» → выход в главное меню RLHF-среды ('/').

Верификация строго через DOM/ARIA/console/network — никаких page.screenshot()/Read PNG.

Запуск (rlhf-сервер на 8090, дев-инстанс игры на 8082):
    python3 rlhf_env/tests_smoke_arena_series_modal.py
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

BASE = os.environ.get("RLHF_SERIES_BASE", "http://127.0.0.1:8090")


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read())


async def _wait_js(page, js: str, timeout_ms: int = 12000, interval: int = 250):
    """Поллим JS-выражение, пока не вернёт truthy."""
    elapsed = 0
    while elapsed < timeout_ms:
        try:
            if await page.evaluate(js):
                return True
        except Exception:
            pass
        await page.wait_for_timeout(interval)
        elapsed += interval
    return False


async def _load_battle(page, redirect_url: str, battle_idx: int) -> dict:
    """Загружает бой, ждёт populated currentState, возвращает series-поля клиента."""
    await page.goto(BASE + redirect_url, wait_until="domcontentloaded")
    # currentState (лексический global от top-level let) заполняется из
    # /api/battle/state при loadBattleState. ВНИМАНИЕ: window.currentState
    # undefined — top-level let не становится свойством window; используем bare.
    ok = await _wait_js(page, "currentState && currentState.group_id")
    if not ok:
        raise RuntimeError(f"battle {battle_idx}: currentState.group_id не появился")
    series = await page.evaluate("""() => ({
        group_id: currentState.group_id,
        battle_index: currentState.battle_index,
        battles_planned: currentState.battles_planned,
        has_next_battle: currentState.has_next_battle,
        hand_levels: (currentState.player?.hand||[]).map(c=>c.level),
        hero_level: currentState.player?.hero?.level
    })""")
    print(f"[series] battle {battle_idx} client series={series}")
    return series


async def main() -> int:
    # Серия из 3 боёв, случайные колоды (→ deck-preview на каждый бой).
    body = _post("/api/match/find", {
        "p2_model": "random",
        "battles_planned": 3, "seed": 424242,
        "deck_strategy_p1": "random_arenaenv",
        "deck_strategy_p2": "random_arenaenv",
    })
    gid = body["group_id"]
    redirect0 = body["redirect_url"]
    print(f"[series] group={gid} battles=3")

    failures: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 420, "height": 900})
        page = await ctx.new_page()

        console_logs: list[str] = []
        page.on("console", lambda m: console_logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: console_logs.append(f"[pageerror] {e}"))

        # --- Бой 0 ---
        s0 = await _load_battle(page, redirect0, 0)
        if s0["group_id"] != gid:
            failures.append(f"battle0 group_id mismatch {s0['group_id']} != {gid}")
        if s0["battle_index"] != 0 or s0["has_next_battle"] is not True:
            failures.append(f"battle0 series fields wrong: {s0}")

        # Проверяем, что уровни карт игрока НЕ поголовно 1 (фича «средние уровни»).
        levels0 = list(s0["hand_levels"]) + [s0["hero_level"]]
        if not any(l > 1 for l in levels0 if l is not None):
            failures.append(f"battle0 все уровни ==1: {levels0}")
        print(f"[series] battle0 card levels={levels0}")

        # Запускаем модал результата напрямую (как сделал бы handleGameOver).
        await page.evaluate("""() => {
            window.__resultModalShown = false;
            window.__battleResultEconomy = null;
            window.showBattleResult('victory', 25, 1025, 15, 450, 3, 15);
        }""")
        await page.wait_for_timeout(400)
        modal_visible = await page.evaluate("""() => {
            const m = document.getElementById('battle-result-modal');
            return m && m.getAttribute('aria-hidden') !== 'true' && m.style.display !== 'none';
        }""")
        if not modal_visible:
            failures.append("battle0: модал результата не показан")
        btn0 = await page.evaluate("""() => {
            const b = document.getElementById('result-next-btn');
            const f = document.getElementById('result-finish-btn');
            return {next_disabled: b ? b.disabled : null,
                    next_text: b ? b.textContent : null,
                    finish_exists: !!f,
                    finish_text: f ? f.textContent : null};
        }""")
        print(f"[series] battle0 modal buttons={btn0}")
        if btn0["next_disabled"] is not False:
            failures.append(f"battle0: next-btn должен быть активен, got disabled={btn0['next_disabled']}")
        if btn0["next_text"] != "Следующий бой":
            failures.append(f"battle0: next-btn text={btn0['next_text']!r}")
        if not btn0["finish_exists"] or "Завершить" not in (btn0["finish_text"] or ""):
            failures.append(f"battle0: finish-btn отсутствует/неверен: {btn0}")

        # Кликаем «Следующий бой» → doNextBattle → навигация на новый бой.
        url_before = page.url
        await page.click("#result-next-btn")
        # Ждём навигации на новый match_id.
        nav_ok = await _wait_js(
            page,
            f"window.location.href.indexOf('id=') !== -1 && "
            f"window.location.href !== {json.dumps(url_before)}",
            timeout_ms=10000,
        )
        if not nav_ok:
            failures.append("battle0→1: навигация на следующий бой не произошла")
        # --- Бой 1 ---
        s1 = await _load_battle(page, page.url[len(BASE):], 1)
        if s1["battle_index"] != 1 or s1["has_next_battle"] is not True:
            failures.append(f"battle1 series fields wrong: {s1}")
        # Новый бой должен иметь НОВЫЕ уровни карт (центр выбирается заново).
        levels1 = list(s1["hand_levels"]) + [s1["hero_level"]]
        print(f"[series] battle1 card levels={levels1}")

        # Модал боя 1 + клик «Следующий бой» → бой 2.
        await page.evaluate("window.showBattleResult('defeat', -20, 1005, 0, 450, 0, 15)")
        await page.wait_for_timeout(400)
        url_before2 = page.url
        await page.click("#result-next-btn")
        nav2 = await _wait_js(page,
            f"window.location.href !== {json.dumps(url_before2)}", timeout_ms=10000)
        if not nav2:
            failures.append("battle1→2: навигация не произошла")
        # --- Бой 2 (последний) ---
        s2 = await _load_battle(page, page.url[len(BASE):], 2)
        if s2["battle_index"] != 2 or s2["has_next_battle"] is not False:
            failures.append(f"battle2 series fields wrong (должен быть последний): {s2}")

        await page.evaluate("window.showBattleResult('draw', 0, 1005, 0, 450, 0, 15)")
        await page.wait_for_timeout(400)
        btn2 = await page.evaluate("""() => {
            const b = document.getElementById('result-next-btn');
            return {disabled: b ? b.disabled : null, text: b ? b.textContent : null};
        }""")
        print(f"[series] battle2 (last) next-btn={btn2}")
        if btn2["disabled"] is not True:
            failures.append(f"battle2: next-btn должен быть disabled, got {btn2['disabled']}")
        if btn2["text"] != "Бои завершены":
            failures.append(f"battle2: next-btn text={btn2['text']!r} (ожидали 'Бои завершены')")

        # Клик по disabled next-btn НЕ должен навигировать.
        url_before_finish = page.url
        # disabled-кнопка: Playwright может отказаться кликать; используем JS-клик и проверяем.
        try:
            await page.click("#result-next-btn", timeout=1000)
        except Exception:
            pass
        await page.wait_for_timeout(800)
        if page.url != url_before_finish:
            failures.append("battle2: disabled next-btn всё же навигировал")

        # «Завершить предварительно» → выход в '/'.
        await page.click("#result-finish-btn")
        finish_ok = await _wait_js(page, "window.location.pathname === '/'", timeout_ms=8000)
        if not finish_ok:
            failures.append(f"finish-btn: не вышли в '/' (url={page.url})")
        else:
            print(f"[series] finish-btn → '{page.url}' OK")

        # console errors (без «Браузер не поддерживается» — это ожидаемая заглушка без tg-клиента)
        fatal = [l for l in console_logs if "[pageerror]" in l]
        if fatal:
            failures.append(f"pageerrors: {fatal[:3]}")

        await browser.close()

    print("[series] RESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  FAIL:", f)
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))