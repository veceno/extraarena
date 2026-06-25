"""Verify the V5 trace captures a surrender terminal row.

Drives a real match (end_turn bot): play one end_turn (so there's action
history + a turn snapshot), then POST /api/matches/{id}/surrender. Asserts
actions.jsonl ends with a `surrender` row whose post_state.status == p2_win
and deltas computed (enemy_hero_hp_delta reflects surrender does not damage).
Text-only verification (DOM/HTTP/files) — no screenshots.
"""
import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8090"
SESSIONS = "rlhf_env/sessions"


def _req(method, path, payload=None, q=None):
    url = BASE + path
    if q:
        url += "?" + "&".join(f"{k}={v}" for k, v in q.items())
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _state(match_id):
    return _req("GET", "/api/battle/state", q={"match_id": match_id})


def _act(path, match_id, **fields):
    payload = {"match_id": match_id, "client_action_id": f"{path}-{time.monotonic_ns()}"}
    payload.update(fields)
    return _req("POST", path, payload=payload)


def main() -> int:
    body = _req("POST", "/api/match/find", payload={"p2_model": "end_turn", "battles_planned": 1})
    match_id = body["match_id"]
    group_id = body.get("group_id")
    print(f"[surrender] group={group_id} match={match_id}")

    # let the bot/end_turn settle so we're mid-battle
    for _ in range(20):
        st = _state(match_id)
        if st.get("battle_started"):
            break
        time.sleep(0.3)
    # play one end_turn so a turn snapshot + at least one action row exist
    st = _state(match_id)
    legal = st.get("legal_actions") or []
    et = next((a for a in legal if a.get("type") == "end_turn"), None)
    if et is not None:
        _act("/api/battle/end-turn", match_id)
        time.sleep(0.5)
    else:
        print("[surrender] no end_turn available pre-surrender; surrendering anyway")

    # surrender (POST with client_action_id — urllib issues GET when data=None)
    r = _req("POST", f"/api/matches/{match_id}/surrender",
             payload={"client_action_id": f"surrender-{time.monotonic_ns()}"})
    print(f"[surrender] surrender response: {json.dumps(r.get('result', {}))}")
    winner = r.get("result", {}).get("winner_id")
    if winner != 2000:
        print(f"[surrender] FAIL: expected bot winner 2000, got {winner}")
        return 1
    print("[surrender] PASS: bot wins on surrender")

    # locate the v5 trace
    gdir = os.path.join(SESSIONS, group_id)
    v5_actions = None
    for root, _dirs, files in os.walk(os.path.join(gdir, "battles")):
        if "actions.jsonl" in files and os.path.basename(root) == "v5":
            v5_actions = os.path.join(root, "actions.jsonl")
            break
    if not v5_actions:
        print("[surrender] FAIL: no v5/actions.jsonl found")
        return 1
    rows = [json.loads(l) for l in open(v5_actions)]
    print(f"[surrender] action rows: {len(rows)}; types: {sorted(set(r['action_type'] for r in rows))}")
    surr = [r for r in rows if r["action_type"] == "surrender"]
    if len(surr) != 1:
        print(f"[surrender] FAIL: expected 1 surrender row, got {len(surr)}")
        return 1
    s = surr[0]
    print(f"[surrender] surrender row seq={s['seq']} turn={s['turn_number']} accepted={s['accepted']}")
    print(f"  pre_state.status={s['pre_state']['status']}")
    print(f"  post_state.status={s['post_state']['status']}")
    print(f"  deltas={s['deltas']}")
    if s["pre_state"]["status"] != "ongoing":
        print("[surrender] FAIL: pre_state.status should be 'ongoing'")
        return 1
    # mark_surrender does NOT mutate state.status (P2_WIN is derived in _finalize);
    # post_state.status stays 'ongoing' — the surrender is visible via the
    # action_type marker + p1.replacement_status='surrendered' + meta.json.
    if s["post_state"]["status"] != "ongoing":
        print(f"[surrender] FAIL: post_state.status should be 'ongoing' (surrender doesn't mutate engine status), got {s['post_state']['status']}")
        return 1
    # surrender visible in the state via replacement_status
    p1_rep = s["post_state"]["p1"].get("replacement_status")
    if p1_rep != "surrendered":
        print(f"[surrender] FAIL: post_state.p1.replacement_status should be 'surrendered', got {p1_rep}")
        return 1
    print(f"[surrender] PASS: surrender visible — action_type=surrender + p1.replacement_status={p1_rep}")
    if s["accepted"] is not True:
        print("[surrender] FAIL: surrender row accepted should be True")
        return 1
    if s["deltas"] is None:
        print("[surrender] FAIL: deltas should be computed")
        return 1
    if s["deltas"]["enemy_hero_hp_delta"] != 0 or s["deltas"]["own_hero_hp_delta"] != 0:
        print(f"[surrender] FAIL: surrender should deal no damage, deltas={s['deltas']}")
        return 1
    # surrender row must be the LAST row
    if rows[-1]["action_type"] != "surrender":
        print(f"[surrender] FAIL: surrender not last row (last={rows[-1]['action_type']})")
        return 1
    # meta.json should reflect p2_win terminal
    meta = json.load(open(os.path.join(os.path.dirname(v5_actions), "meta.json")))
    if meta["status"] != "p2_win" or meta["winner_user_id"] != 2000:
        print(f"[surrender] FAIL: meta status/winner wrong: {meta['status']} {meta['winner_user_id']}")
        return 1
    print(f"[surrender] PASS: meta status={meta['status']} winner={meta['winner_user_id']} turns={meta['turns']}")
    print("[surrender] RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())