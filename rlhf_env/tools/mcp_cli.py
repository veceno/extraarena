#!/usr/bin/env python3
"""Compact CLI client for the rlhf MCP bridge. One call per move.

Subcommands:
  start [--seed N] [--first p1|p2|random]   -> start_series + initial compact state + numbered legal
  state <mid>                                -> compact state + numbered legal (auto-advance if bot's turn)
  act <mid> <index>                          -> pick legal[index], submit, print next compact state + legal
  next <gid>                                  -> next_battle + compact state + legal (or series_complete)
  manifest <gid>                             -> group manifest

Compact format keeps the LLM agent's per-move context tiny.
"""
import json, socket, sys, os

SOCK = "/tmp/rlhf_mcp.sock"

def call(tool, args=None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK); s.setblocking(True)
    s.sendall((json.dumps({"tool": tool, "args": args or {}, "id": 1}) + "\n").encode())
    buf = b""
    while True:
        c = s.recv(65536)
        if not c: break
        buf += c
    s.close()
    resp = json.loads(buf.decode().strip())
    if "error" in resp:
        return {"__error": resp["error"]}
    return resp.get("result", {})

def _atk(o): return o.get("attack") if o.get("attack") is not None else o.get("atk", 0)
def _hp(o):  return o.get("hp_current") if o.get("hp_current") is not None else o.get("hp", 0)
def _short(uuid): return str(uuid)[-6:] if uuid else "?"
def _can_atk(o): return o.get("can_attack", True) and not o.get("exhausted", False) and not o.get("summoning_sickness", False)

def build_action(legal):
    t = legal.get("type")
    if t == "play_card":
        return {"type":"play_card","hand_index":legal.get("hand_index"),
                "target_position":legal.get("position") or 0,"target_id":legal.get("target_id")}
    if t == "attack":
        return {"type":"attack","attacker_id":legal.get("attacker_id"),
                "target_id":legal.get("target_id"),"target_is_hero":legal.get("target_is_hero", False)}
    return {"type":"end_turn"}

def fmt_board(board, mine):
    out = []
    for i, b in enumerate(board or []):
        nm = b.get("name") or b.get("card_id") or "?"
        tag = f"b{i}" if mine else f"B{i}"
        ready = "" if mine else ""
        can = "ready" if (not mine or _can_atk(b)) else "zzz"
        out.append(f"{tag}:{nm} a{_atk(b)} h{_hp(b)}{(' '+can) if mine else ''}")
    return " ".join(out) if out else "(empty)"

def fmt_hand(hand):
    out = []
    for i, c in enumerate(hand or []):
        if c.get("hidden"): out.append(f"[{i}]??"); continue
        nm = c.get("name") or c.get("card_id") or "?"
        chg = " charge" if c.get("charge") or c.get("has_charge") else ""
        kw = c.get("keywords") or []
        kws = f"({','.join(kw)})" if kw else ""
        out.append(f"[{i}]{nm} c{c.get('mana_cost', c.get('mana','?'))} a{_atk(c)} h{_hp(c)}{chg}{kws}")
    return " ".join(out) if out else "(empty)"

def fmt_legal(legal, st):
    me_board = (st.get("player",{}) or {}).get("board") or []
    opp_board = (st.get("opponent",{}) or {}).get("board") or []
    me_hand = (st.get("player",{}) or {}).get("hand") or []
    out = []
    for i, a in enumerate(legal or []):
        t = a.get("type")
        if t == "play_card":
            hi = a.get("hand_index")
            c = me_hand[hi] if hi is not None and hi < len(me_hand) else {}
            nm = c.get("name") or "?"
            tgt = f" -> tgt{_short(a.get('target_id'))}" if a.get("target_id") else ""
            out.append(f"[{i}] play h{hi} {nm}(c{c.get('mana_cost',c.get('mana','?'))},a{_atk(c)},h{_hp(c)}) pos{a.get('position',0)}{tgt}")
        elif t == "attack":
            atk = a.get("attacker_id")
            # find attacker name in my board
            aname = next((f"b{j}:{(b.get('name') or '?')}" for j,b in enumerate(me_board) if b.get("instance_id")==atk), _short(atk))
            if a.get("target_is_hero"):
                out.append(f"[{i}] atk {aname} -> HERO (face)")
            else:
                tid = a.get("target_id")
                tname = next((f"B{j}:{(b.get('name') or '?')}(a{_atk(b)},h{_hp(b)})" for j,b in enumerate(opp_board) if b.get("instance_id")==tid), _short(tid))
                out.append(f"[{i}] atk {aname} -> {tname}")
        else:
            out.append(f"[{i}] end_turn")
    return "  ".join(out) if out else "(none)"

def show(st, mid=None, gid=None):
    if not isinstance(st, dict) or st.get("__error"):
        print("ERROR:", st); return st
    me = st.get("player",{}) or {}; opp = st.get("opponent",{}) or {}
    over = st.get("is_ended") or st.get("game_over")
    winner = st.get("winner_id")
    wtag = "" if not over else f"  WINNER={'me' if winner==1000 else ('bot' if winner else '?')}"
    print(f"--- turn={st.get('turn')} me_p1 hp={st.get('player1_hp')} mana={me.get('mana')}/{me.get('max_mana')} | bot hp={st.get('player2_hp')} | is_my={st.get('is_my_turn')} over={over}{wtag}")
    if mid: print(f"    match={mid} group={gid or ''}")
    print(f"    my_board:  {fmt_board(me.get('board'), True)}")
    print(f"    bot_board: {fmt_board(opp.get('board'), False)}")
    print(f"    hand:      {fmt_hand(me.get('hand'))}")
    legal = st.get("legal_actions") or []
    print(f"    legal:     {fmt_legal(legal, st)}")
    return st

def cmd_start(args):
    spec = {"p2_model":"extra-lr-v4-max","battles_planned":1,
            "starting_player": args.get("--first","random"),
            "seed": int(args.get("--seed","0") or 0),
            "deck_strategy_p1":"random_arenaenv","deck_strategy_p2":"random_arenaenv",
            "p1_name":"Claude","p2_name":"extra-lr-v4-max"}
    r = call("start_series", {"spec": spec})
    if r.get("__error"): print("start error:", r); return
    mid = r.get("match_id"); gid = r.get("group_id")
    st = call("get_state", {"match_id": mid})
    st = _to_my_turn(mid, st)
    print(f"START group={gid} match={mid} opponent={r.get('opponent')}")
    show(st, mid, gid)

def _to_my_turn(mid, st):
    """Advance bot until it's my turn or game over. Returns final state."""
    guard = 0
    while not (st.get("is_ended") or st.get("game_over")) and not st.get("is_my_turn") and guard < 40:
        r = call("advance_bot", {"match_id": mid})
        st = r if isinstance(r, dict) and ("is_my_turn" in r or "is_ended" in r) else call("get_state", {"match_id": mid})
        guard += 1
    return st

def cmd_state(mid):
    st = call("get_state", {"match_id": mid})
    st = _to_my_turn(mid, st)
    show(st, mid)

def cmd_act(mid, idx):
    st = call("get_state", {"match_id": mid})
    if st.get("__error"): print("act get_state error:", st); return
    if st.get("is_ended") or st.get("game_over"):
        print("ALREADY OVER"); show(st, mid); return
    st = _to_my_turn(mid, st)
    if st.get("is_ended") or st.get("game_over"):
        print("ENDED during bot turn"); show(st, mid); return
    legal = st.get("legal_actions") or []
    if idx < 0 or idx >= len(legal):
        print(f"BAD index {idx}; legal len={len(legal)}"); show(st, mid); return
    action = build_action(legal[idx])
    resp = call("submit_action", {"match_id": mid, "action": action})
    if resp.get("__error"): print("submit error:", resp); return
    ns = resp.get("state") if isinstance(resp, dict) else None
    if not isinstance(ns, dict) or "legal_actions" not in ns:
        ns = call("get_state", {"match_id": mid})
    res = (resp.get("result") or {}) if isinstance(resp, dict) else {}
    err = res.get("error") or resp.get("error")
    if err: print(f"  (action rejected: {err})")
    print(f"ACT[{idx}] {json.dumps(action, ensure_ascii=False)}")
    # always re-fetch full state (submit response state may be stripped of winner_id/hp)
    ns = call("get_state", {"match_id": mid})
    ns = _to_my_turn(mid, ns)
    show(ns, mid)

def cmd_next(gid):
    r = call("next_battle", {"group_id": gid})
    if r.get("status") == "series_complete" or r.get("__error"):
        print("series_complete:", r); return
    mid = r.get("match_id")
    st = call("get_state", {"match_id": mid})
    st = _to_my_turn(mid, st)
    print(f"NEXT group={gid} match={mid}")
    show(st, mid, gid)

def cmd_manifest(gid):
    m = call("get_battle_group_manifest", {"group_id": gid})
    print(json.dumps(m, ensure_ascii=False, indent=2))

def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return
    cmd = a[0]
    if cmd == "start":
        args = {}
        i = 1
        while i < len(a):
            if a[i] == "--seed": args["--seed"] = a[i+1]; i += 2
            elif a[i] == "--first": args["--first"] = a[i+1]; i += 2
            else: i += 1
        cmd_start(args)
    elif cmd == "state" and len(a) >= 2: cmd_state(a[1])
    elif cmd == "act" and len(a) >= 3: cmd_act(a[1], int(a[2]))
    elif cmd == "next" and len(a) >= 2: cmd_next(a[1])
    elif cmd == "manifest" and len(a) >= 2: cmd_manifest(a[1])
    else: print(__doc__)

if __name__ == "__main__":
    main()