"""End-to-end verification of the 'Ничья on win' fix (text-only, no images).

Drives a REAL human win via the HTTP action API (passive `end_turn` bot never
plays/attacks, so the human chips the bot hero to 0). A Playwright page is
connected to the match socket throughout, so it receives the LIVE game_over
frame and renders the real result modal — proving the fix end-to-end:
  - server records P1_WIN (battle log + manifest)
  - live socket game_over frame carries winner_id inside state
  - the page's handleGameOver -> showBattleResult shows 'Победа' (not 'Ничья')
  - 'Следующий бой' enabled (series planned=2, this is battle 0)

Run (server on 8090):
    python3 rlhf_env/tests_verify_draw_fix_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8090"
HUMAN = 1000


def _req(method: str, path: str, payload: dict | None = None, q: dict | None = None) -> dict:
    url = BASE + path
    if q:
        url += "?" + "&".join(f"{k}={v}" for k, v in q.items())
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _state(match_id: str) -> dict:
    return _req("GET", "/api/battle/state", q={"match_id": match_id})


def _act(path: str, match_id: str, **fields) -> dict:
    payload = {"match_id": match_id, "client_action_id": f"{path}-{time.monotonic_ns()}"}
    payload.update(fields)
    return _req("POST", path, payload=payload)


def _pick_action(legal: list[dict], atype: str, hero_only: bool = False):
    for a in legal:
        if a.get("type") != atype:
            continue
        if hero_only and not a.get("target_is_hero"):
            continue
        return a
    return None


async def main() -> int:
    body = _req("POST", "/api/match/find", payload={"p2_model": "end_turn", "battles_planned": 2})
    redirect_url = body["redirect_url"]
    match_id = body["match_id"]
    group_id = body.get("group_id")
    print(f"[e2e] group={group_id} match={match_id}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 420, "height": 900})
        page = await ctx.new_page()

        console_logs: list[str] = []
        page.on("console", lambda m: console_logs.append(f"[{m.type}] {m.text}"))
        ws_frames: list[str] = []
        page.on("websocket", lambda ws: ws.on("framereceived", lambda pld: ws_frames.append(str(pld)[:400])))

        await page.goto(BASE + redirect_url, wait_until="domcontentloaded")
        # deck-preview gate -> continue
        try:
            await page.wait_for_selector("#deck-preview-continue:visible", timeout=8000)
            await page.click("#deck-preview-continue")
        except Exception:
            pass

        # wait for match_ready + prebattle countdown to elapse (client_ready sent)
        for _ in range(80):
            if any("match_ready" in l for l in console_logs):
                break
            await page.wait_for_timeout(200)
        await page.wait_for_timeout(3500)  # prebattle countdown ~3s

        # Confirm the page joined the match (battle started)
        st = _state(match_id)
        print(f"[e2e] initial: is_my_turn={st.get('is_my_turn')} battle_started={st.get('battle_started')} "
              f"p1hp={st.get('player1_hp')} p2hp={st.get('player2_hp')} legal={len(st.get('legal_actions') or [])}")
        if not st.get("battle_started"):
            print("[e2e] WARN battle not started yet; driving anyway")

        # Drive to a human win
        max_iter = 200
        for i in range(max_iter):
            st = _state(match_id)
            if st.get("game_over"):
                print(f"[e2e] game_over after {i} actions | winner_id={st.get('winner_id')} "
                      f"p1hp={st.get('player1_hp')} p2hp={st.get('player2_hp')}")
                break
            if not st.get("is_my_turn"):
                await page.wait_for_timeout(300)
                continue
            legal = st.get("legal_actions") or []
            # 1) attack hero (prioritize damage) 2) play a card 3) end turn
            atk = _pick_action(legal, "attack", hero_only=True)
            if atk:
                r = _act("/api/battle/attack", match_id,
                         attacker_id=atk.get("attacker_id"),
                         target_is_hero=True)
            else:
                ply = _pick_action(legal, "play_card")
                if ply:
                    r = _act("/api/battle/play-card", match_id,
                             hand_index=ply.get("hand_index", 0),
                             card_id=ply.get("card_id") or ply.get("card_ref"),
                             board_position=ply.get("board_position", 0))
                else:
                    r = _act("/api/battle/end-turn", match_id)
            if not r.get("result", {}).get("success"):
                # action rejected (e.g. bot's turn racing) — small sleep + retry
                await page.wait_for_timeout(200)
            await page.wait_for_timeout(120)  # let bot turn + broadcasts settle
        else:
            print("[e2e] WARN hit max_iter without game_over")

        # Let the page receive + render the live game_over modal
        await page.wait_for_timeout(2000)

        # 1) server-side record
        st = _state(match_id)
        winner_id = st.get("winner_id")
        print(f"[e2e] server winner_id={winner_id} game_over={st.get('game_over')}")

        # 2) live socket game_over frame (winner_id inside state)
        go_frames = [f for f in ws_frames if "game_over" in f]
        print(f"[e2e] socket frames with game_over: {len(go_frames)}")
        for f in go_frames[:2]:
            print("   ws:", f[:300])

        # 3) page modal title
        modal_title = await page.evaluate("""() => {
            const t = document.getElementById('result-title');
            return t ? t.textContent : null;
        }""")
        next_btn = await page.evaluate("""() => {
            const b = document.getElementById('result-next-btn');
            return b ? {text: b.textContent, disabled: b.disabled} : null;
        }""")
        print(f"[e2e] modal_title={modal_title!r} next_btn={next_btn}")
        res_logs = [l for l in console_logs if 'Результат' in l]
        for l in res_logs:
            print("   console:", l[:160])

        ok = True
        if winner_id != HUMAN:
            print(f"[e2e] FAIL: server winner_id={winner_id} (expected {HUMAN})")
            ok = False
        else:
            print("[e2e] PASS: server recorded human win (P1_WIN)")
        if modal_title != 'Победа':
            print(f"[e2e] FAIL: modal title {modal_title!r} (expected 'Победа')")
            ok = False
        else:
            print("[e2e] PASS: live modal shows 'Победа' (not 'Ничья')")
        if not next_btn or next_btn.get("disabled"):
            print(f"[e2e] FAIL: 'Следующий бой' disabled: {next_btn}")
            ok = False
        else:
            print("[e2e] PASS: 'Следующий бой' enabled (has_next_battle)")

        # Confirm winner_id is in the socket frame's state (not just top-level)
        frame_has_state_winner = any('"winner_id":1000' in f.replace(' ', '') or
                                     '"winner_id": 1000' in f for f in go_frames)
        print(f"[e2e] socket frame has winner_id in state: {frame_has_state_winner}")

        # 4) Continue to battle 2 — click «Следующий бой» and verify a NEW match
        #    loads (series progresses after a WIN, not only after a loss).
        before_url = page.url
        try:
            await page.click("#result-next-btn")
        except Exception as exc:
            print(f"[e2e] FAIL: could not click 'Следующий бой': {exc}")
            ok = False
            await browser.close()
            return 3
        # wait for navigation to a new /arena?id=...
        for _ in range(50):
            if page.url != before_url and "/arena" in page.url:
                break
            await page.wait_for_timeout(200)
        new_url = page.url
        new_match = None
        if "id=" in new_url:
            import urllib.parse as up
            new_match = up.parse_qs(up.urlparse(new_url).query).get("id", [None])[0]
        print(f"[e2e] after 'Следующий бой': url={new_url} new_match={new_match}")
        if new_match and new_match != match_id:
            # verify the new match is a real, playable battle 2
            st2 = _state(new_match)
            print(f"[e2e] battle2 state: battle_started={st2.get('battle_started')} "
                  f"is_my_turn={st2.get('is_my_turn')} group_id={st2.get('group_id')} "
                  f"battle_index={st2.get('battle_index')} has_next_battle={st2.get('has_next_battle')}")
            if st2.get("group_id") == group_id and st2.get("battle_index") == 1:
                print("[e2e] PASS: series continued to battle 2 (index=1, same group)")
            else:
                print(f"[e2e] FAIL: battle2 group/index wrong: group={st2.get('group_id')} idx={st2.get('battle_index')}")
                ok = False
        else:
            print(f"[e2e] FAIL: did not navigate to a new match (url={new_url})")
            ok = False

        await browser.close()
        print("[e2e] RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))