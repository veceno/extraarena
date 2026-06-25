"""Deterministic verification of the series-flow fix (pure HTTP, no images).

Reproduces the user's exact bug: a 3-battle series where the HUMAN wins each
battle via hero-kill. Before the fix, the HTTP action response returned a state
WITHOUT series fields (has_next_battle/group_id/battle_index/battles_planned);
the client did `currentState = result.state` (arena.js:5798) wiping
has_next_battle, and renderBattleState's game_over branch (arena.js:3934) called
showBattleResult directly — racing the socket game_over frame's
handleGameOver→showBattleResult. When the HTTP path won, the modal read
has_next_battle=undefined → «Бои завершены» mid-series (user saw 2/3 running,
blocked from battle 3).

Fix (match_runner._state_with_series + arena_io sibling paths): every HTTP
action/surrender response now carries the series fields, so both modal paths
agree — race eliminated.

This test drives a real human win per battle (custom cheap charge-minion deck
vs a passive `end_turn` bot that never plays/attacks) and asserts the killing
action's HTTP response state carries `has_next_battle` with the correct value
for battles 0,1,2 of a 3-battle series — plus that all 3 battles are played and
the manifest finalizes 3/3. Pure HTTP → fast, deterministic, no Playwright race.

Run (server on 8090):
    python3 rlhf_env/tests_verify_series_3_battles.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BASE = "http://127.0.0.1:8090"
HUMAN = 1000
SESSIONS = "rlhf_env/sessions"

# Hero 1 (Искатель) + 8 distinct cheap warriors, including 2 CHARGE minions
# (44 Леви cost2 atk3 charge, 32 Зеницу cost4 atk5 charge) so the human can deal
# reliable face damage the turn a minion enters play. The `end_turn` bot never
# plays/attacks, so human minions survive and the human wins deterministically.
DECK = [1, 27, 37, 28, 38, 46, 43, 44, 32]
# P1 minions at MAX level (10) → high attack, so the human kills the bot hero in
# a few turns regardless of opening-hand shuffle. Bot kept at level 1 (low HP).
LEVELS_P1 = {c: 10 for c in DECK}
LEVELS_P2 = {c: 1 for c in DECK}


def _req(method: str, path: str, payload: dict | None = None, q: dict | None = None) -> dict:
    url = BASE + path
    if q:
        url += "?" + "&".join(f"{k}={v}" for k, v in q.items())
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        out = json.loads(e.read())
        out["__status"] = e.code
        return out


def _state(match_id: str) -> dict:
    return _req("GET", "/api/battle/state", q={"match_id": match_id})


def _act(path: str, match_id: str, **fields) -> dict:
    payload = {"match_id": match_id, "client_action_id": f"{path}-{time.monotonic_ns()}"}
    payload.update(fields)
    return _req("POST", path, payload=payload)


def _pick(legal: list[dict], atype: str, hero_only: bool = False, no_target: bool = False):
    for a in legal:
        if a.get("type") != atype:
            continue
        if hero_only and not a.get("target_is_hero"):
            continue
        if no_target and a.get("target_id") is not None:
            continue
        return a
    return None


def _start_series(battles_planned: int) -> dict:
    return _req("POST", "/api/match/find", payload={
        "p2_model": "end_turn",
        "battles_planned": battles_planned,
        "starting_player": "p1",  # human always starts → bot turns triggered by human end_turn
        "custom_deck_p1": DECK,
        "custom_deck_p2": DECK,
        "card_levels_p1": LEVELS_P1,
        "card_levels_p2": LEVELS_P2,
    })


def _wait_started(match_id: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = _state(match_id)
        if st.get("battle_started"):
            return True
        time.sleep(0.3)
    return False


def _drive_to_human_win(match_id: str) -> dict | None:
    """Play until the human wins (hero-kill). Returns the killing action's HTTP
    response (the one whose result.game_over=true), or None if no win in budget."""
    last_kill = None
    for _ in range(800):
        st = _state(match_id)
        if st.get("game_over"):
            return last_kill
        if not st.get("is_my_turn"):
            time.sleep(0.15)
            continue
        legal = st.get("legal_actions") or []
        # 1) attack hero (forward the action's own target_id)
        atk = _pick(legal, "attack", hero_only=True)
        if atk:
            r = _act("/api/battle/attack", match_id,
                     attacker_id=atk.get("attacker_id"),
                     target_id=atk.get("target_id"), target_is_hero=True)
        else:
            aan = _pick(legal, "attack")
            if aan:
                r = _act("/api/battle/attack", match_id,
                         attacker_id=aan.get("attacker_id"),
                         target_id=aan.get("target_id"),
                         target_is_hero=bool(aan.get("target_is_hero")))
            else:
                # 2) play a minion (no target) — safe
                ply = _pick(legal, "play_card", no_target=True)
                if ply:
                    r = _act("/api/battle/play-card", match_id,
                             hand_index=ply.get("hand_index", 0),
                             card_id=ply.get("card_id") or ply.get("card_ref"),
                             board_position=ply.get("board_position", 0))
                else:
                    # 3) play a targeted card, forwarding its target
                    plt = _pick(legal, "play_card")
                    if plt:
                        r = _act("/api/battle/play-card", match_id,
                                 hand_index=plt.get("hand_index", 0),
                                 card_id=plt.get("card_id") or plt.get("card_ref"),
                                 board_position=plt.get("board_position", 0),
                                 target_id=plt.get("target_id"),
                                 target_is_hero=bool(plt.get("target_is_hero")))
                    else:
                        r = _act("/api/battle/end-turn", match_id)
        if r.get("result", {}).get("game_over"):
            last_kill = r
        time.sleep(0.05)
    return last_kill


def main() -> int:
    body = _start_series(battles_planned=3)
    if body.get("error"):
        print(f"[series3] FAIL: could not start series: {body}")
        return 3
    group_id = body.get("group_id")
    match_id = body["match_id"]
    print(f"[series3] group={group_id} first_match={match_id} planned=3")

    ok = True
    current_match = match_id
    for battle_index in range(3):
        expected_has_next = (battle_index + 1) < 3
        # the 2nd+ battle is started via next_match (POST /api/match/find {group_id})
        if battle_index > 0:
            nb = _req("POST", "/api/match/find", payload={"group_id": group_id})
            if nb.get("error") == "series_complete":
                print(f"[series3] FAIL battle {battle_index}: series_complete early (should have a next battle)")
                ok = False
                break
            current_match = nb["match_id"]
            print(f"[series3] advanced to battle {battle_index} match={current_match}")

        time.sleep(1.0)  # let the match settle (bot ready / start)
        if not _wait_started(current_match):
            st = _state(current_match)
            print(f"[series3] FAIL battle {battle_index}: match never started "
                  f"(battle_started={st.get('battle_started')} match_status={st.get('match_status')})")
            ok = False
            break
        kill = _drive_to_human_win(current_match)

        # --- server-side winner
        st = _state(current_match)
        winner = st.get("winner_id")
        if winner != HUMAN:
            print(f"[series3] FAIL battle {battle_index}: server winner_id={winner} (expected human {HUMAN})")
            ok = False
        else:
            print(f"[series3] PASS battle {battle_index}: server recorded human win (p2hp={st.get('player2_hp')})")

        # --- deterministic fix check: the killing HTTP response carries series fields
        ks = kill.get("state") if kill else None
        if ks is None:
            print(f"[series3] FAIL battle {battle_index}: no killing HTTP response captured")
            ok = False
        else:
            got_next = ks.get("has_next_battle")
            got_group = ks.get("group_id")
            got_idx = ks.get("battle_index")
            got_planned = ks.get("battles_planned")
            got_game_over = ks.get("game_over")
            print(f"[series3]   kill state: game_over={got_game_over} has_next_battle={got_next} "
                  f"group_id={got_group} battle_index={got_idx} battles_planned={got_planned}")
            if got_game_over is not True:
                print(f"[series3] FAIL battle {battle_index}: kill state game_over!=True")
                ok = False
            if got_next != expected_has_next:
                print(f"[series3] FAIL battle {battle_index}: kill state has_next_battle={got_next} "
                      f"(expected {expected_has_next}) — THIS IS THE USER'S BUG if False mid-series")
                ok = False
            else:
                print(f"[series3] PASS battle {battle_index}: kill state has_next_battle={got_next}")
            if got_group != group_id or got_idx != battle_index or got_planned != 3:
                print(f"[series3] FAIL battle {battle_index}: series fields wrong "
                      f"(group={got_group} idx={got_idx} planned={got_planned})")
                ok = False

    # --- manifest 3/3 finalized
    time.sleep(0.5)
    mpath = os.path.join(SESSIONS, group_id, "manifest.json")
    if not os.path.exists(mpath):
        print(f"[series3] FAIL: manifest not found {mpath}")
        return 3
    manifest = json.load(open(mpath))
    res = manifest.get("results", {})
    finished = res.get("battles_finished")
    planned = res.get("battles_planned")
    finished_at = manifest.get("finished_at")
    print(f"[series3] manifest: finished={finished}/{planned} finished_at={finished_at}")
    if finished != 3 or planned != 3:
        print(f"[series3] FAIL: manifest 3/3 expected, got {finished}/{planned}")
        ok = False
    else:
        print("[series3] PASS: manifest 3/3")
    if not finished_at:
        print("[series3] FAIL: manifest not finalized (finished_at=None)")
        ok = False
    else:
        print("[series3] PASS: manifest finalized")

    print("[series3] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())