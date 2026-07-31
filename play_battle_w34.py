#!/usr/bin/env python3
"""L2 player sub-agent — one battle as p1 (LLM) vs V4-Max bot."""
import json, subprocess, sys, time, select, os

REPO = "/Users/laveqox/Documents/ExtraArenaRaS"

# === Persistent MCP subprocess for the whole battle ===
mcp = subprocess.Popen(
    [sys.executable, "-u", "-m", "rlhf_env.mcp_server",
     "--models-dir", "ai/models",
     "--sessions-dir", "rlhf_env/sessions",
     "--cards-path", "ai/cards.json",
     "--log-level", "ERROR"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    cwd=REPO
)

def mcp_call(method, params, timeout=120):
    req_id = int(time.time() * 1_000_000) & 0x7fffffff
    msg = {"jsonrpc":"2.0","id":req_id,"method":method,"params":params}
    try:
        mcp.stdin.write((json.dumps(msg)+"\n").encode())
        mcp.stdin.flush()
    except Exception as e:
        return {"error": f"stdin_write: {e}"}
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        if b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("id") == req_id:
                if "error" in d:
                    return {"error": d["error"]}
                r = d.get("result", {})
                if isinstance(r, dict) and "content" in r and r["content"]:
                    return r["content"][0].get("data", r["content"][0])
                return r
        ready, _, _ = select.select([mcp.stdout], [], [], 0.1)
        if ready:
            chunk = mcp.stdout.read(4096)
            if not chunk:
                return {"error": "EOF"}
            buf += chunk
    return {"error": "timeout"}

def llm_choose_action(state, legal, strategic_notes, battle_history):
    if not legal:
        return {"type": "end_turn"}
    p = state.get("player", {})
    o = state.get("opponent", {})
    my_hp = p.get("hero", {}).get("hp_current", 0)
    my_max_hp = p.get("hero", {}).get("max_hp", 30)
    opp_hp = o.get("hero", {}).get("hp_current", 0)
    opp_max_hp = o.get("hero", {}).get("max_hp", 30)
    my_mana = p.get("mana", 0)
    my_max_mana = p.get("max_mana", 0)
    opp_mana = o.get("mana", 0)
    my_hand = p.get("hand", [])
    my_board = p.get("board", [])
    opp_board = o.get("board", [])
    opp_hand_count = o.get("hand_count", 0)
    turn = state.get("turn", 0)

    def fmt_board(cards):
        out = []
        for c in cards:
            mech = c.get("mechanics", []) or []
            tags = []
            if 'taunt' in mech: tags.append("TAUNT")
            if 'shield' in mech: tags.append("SHIELD")
            if 'charge' in mech: tags.append("CHARGE")
            if 'divine_shield' in mech: tags.append("DIVINE")
            if c.get("can_attack") and c.get("is_ready"): tags.append("READY")
            else: tags.append("SLEEP")
            out.append(f"id={str(c.get('instance_id',''))[:8]} atk={c.get('attack',0)} hp={c.get('hp_current',0)}/{c.get('hp',0)} {','.join(tags)}")
        return "; ".join(out) or "(empty)"

    def fmt_hand(cards):
        out = []
        for c in cards:
            out.append(f"cost={c.get('mana_cost',0)} atk={c.get('attack',0)} hp={c.get('hp',0)} name={c.get('name','?')}")
        return "; ".join(out) or "(empty)"

    my_board_s = fmt_board(my_board)
    opp_board_s = fmt_board(opp_board)
    my_hand_s = fmt_hand(my_hand)

    la_summary = {"play_card": 0, "attack": 0, "end_turn": 0, "mana_draw": 0}
    la_examples = {"play_card": [], "attack": [], "mana_draw": []}
    for a in legal:
        t = a.get("type", "?")
        la_summary[t] = la_summary.get(t, 0) + 1
        if t in la_examples and len(la_examples[t]) < 3:
            la_examples[t].append(a)
    la_count = ", ".join(f"{k}={v}" for k, v in la_summary.items() if v)

    notes = "\n".join(strategic_notes[-5:]) if strategic_notes else "(start of battle)"
    history = "\n".join(battle_history[-8:]) if battle_history else "(start)"

    prompt = f"""ExtraArena, turn {turn}. You=p1 LLM, opponent=V4-Max bot.

STATE: my_hp={my_hp}/{my_max_hp}, opp_hp={opp_hp}/{opp_max_hp}
MANA: me={my_mana}/{my_max_mana}, opp_mana={opp_mana}
MY HAND ({len(my_hand)}): {my_hand_s}
MY BOARD ({len(my_board)}): {my_board_s}
OPP BOARD ({len(opp_board)}): {opp_board_s}
OPP HAND: {opp_hand_count} cards hidden
LEGAL: {la_count}
Examples: play_card={json.dumps(la_examples.get('play_card',[])[:2])}, attack={json.dumps(la_examples.get('attack',[])[:2])}, mana_draw={json.dumps(la_examples.get('mana_draw',[])[:1])}

STRATEGIC_NOTES (this battle so far):
{notes}

RECENT MOVES:
{history}

TASK: Choose ONE legal action. Think step-by-step:
1. Can I win this turn (lethal)? My board atk vs opp_hp
2. Can opponent win next turn? Threat to my_hp
3. Best trades? What opponent units are most dangerous?
4. Should I play_card from hand (which one gives best value)?
5. mana_draw if desperate for answers (cost grows, use when hand is empty or low)
6. end_turn only if no useful action

Return ONLY this JSON (no other text):
{{"action": <one legal_action>, "reasoning": "<one sentence why>"}}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=60,
            cwd=REPO
        )
        if result.returncode != 0 or not result.stdout:
            return {"type": "end_turn"}
        response = result.stdout.strip()
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            if "action" in parsed and isinstance(parsed["action"], dict):
                return parsed["action"]
    except Exception as e:
        print(f"llm_choose_action error: {e}", flush=True)
    return {"type": "end_turn"}


battle = {"seed":8034,"starting_player":"llm","agent_name":"mix-w34-llmfirst"}
strategic_notes = []
battle_history = []

spec = {
    "p2_model": "extra-lr-v4-max",
    "battles_planned": 1,
    "seed": battle["seed"],
    "starting_player": battle["starting_player"],
    "deck_strategy_p1": "random_arenaenv",
    "deck_strategy_p2": "random_arenaenv",
    "p1_name": "Claude-LLM",
    "p2_name": "v4-max",
    "p1_actor_type": "llm",
    "agent_name": battle["agent_name"],
}
r = mcp_call("tools/call", {"name": "start_series", "arguments": {"spec": spec}}, timeout=180)
if not isinstance(r, dict) or "error" in r or "match_id" not in r:
    print(f"FATAL start_series: {r}", flush=True)
    mcp.terminate()
    sys.exit(1)
mid = r["match_id"]
gid = r["group_id"]
print(f"start: match={mid} group={gid} seed={battle['seed']} sp={battle['starting_player']}", flush=True)

consec_reject = 0
winner = None
turns = 0
my_actions = 0
plays = 0
attacks_made = 0
face_attacks = 0
end_turns_made = 0
mana_draws_made = 0
last_turn = 0
my_hp_end = 30
opp_hp_end = 30

for step in range(500):
    if mcp.poll() is not None:
        print(f"FATAL mcp_died_rc={mcp.returncode}", flush=True)
        break
    s = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
    if not isinstance(s, dict):
        time.sleep(0.1)
        continue
    if s.get("is_ended") or s.get("winner_id") is not None:
        winner = s.get("winner_id")
        break
    if not s.get("is_my_turn"):
        mcp_call("tools/call", {"name": "advance_bot", "arguments": {"match_id": mid}}, timeout=15)
        time.sleep(0.05)
        continue
    state = mcp_call("tools/call", {"name": "get_state", "arguments": {"match_id": mid}}, timeout=10)
    if not isinstance(state, dict):
        continue
    legal_r = mcp_call("tools/call", {"name": "get_legal_actions", "arguments": {"match_id": mid}}, timeout=10)
    legal = []
    if isinstance(legal_r, dict):
        legal = legal_r.get("legal_actions", []) or []
    if not legal:
        mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": {"type": "end_turn"}}}, timeout=10)
        continue
    last_turn = state.get("turn", last_turn)
    player = state.get("player", {})
    opp = state.get("opponent", {})
    my_hp_end = player.get("hero", {}).get("hp_current", my_hp_end)
    opp_hp_end = opp.get("hero", {}).get("hp_current", opp_hp_end)
    action = llm_choose_action(state, legal, strategic_notes, battle_history)
    r = mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": action}}, timeout=10)
    my_actions += 1
    atype = action.get("type")
    if atype == "play_card": plays += 1
    elif atype == "attack":
        attacks_made += 1
        if action.get("target_is_hero"): face_attacks += 1
    elif atype == "end_turn": end_turns_made += 1
    elif atype == "mana_draw": mana_draws_made += 1
    battle_history.append(f"s{step} t{last_turn}: {atype} {json.dumps(action)[:80]}")
    if len(battle_history) > 30:
        battle_history = battle_history[-30:]
    rejected = False
    if isinstance(r, dict):
        if r.get("success") is False: rejected = True
        elif r.get("result", {}).get("success") is False: rejected = True
        elif r.get("error"): rejected = True
        elif r.get("is_ended") or r.get("winner_id") is not None or r.get("game_over"):
            winner = r.get("winner_id") or r.get("result", {}).get("winner_id")
            break
    if rejected:
        consec_reject += 1
        if consec_reject >= 3:
            mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": {"type": "end_turn"}}}, timeout=10)
            consec_reject = 0
    else:
        consec_reject = 0

turns = last_turn
final = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
if isinstance(final, dict) and final.get("is_ended"):
    winner = final.get("winner_id", winner)
mcp_call("tools/call", {"name": "finish_series", "arguments": {"group_id": gid}}, timeout=10)
mcp.terminate()
try:
    mcp.wait(timeout=5)
except Exception:
    mcp.kill()

result = {
    "agent_idx": 34,
    "seed": battle["seed"],
    "starting_player": battle["starting_player"],
    "match_id": mid,
    "group_id": gid,
    "winner_id": winner,
    "winner_side": "p1" if winner == 1000 else "p2" if winner == 2000 else "draw_or_unknown",
    "turns_played": turns,
    "my_actions": my_actions,
    "plays": plays,
    "attacks": attacks_made,
    "face_attacks": face_attacks,
    "end_turns": end_turns_made,
    "mana_draws": mana_draws_made,
    "my_hp_end": my_hp_end,
    "opp_hp_end": opp_hp_end,
    "transport": "raw_pipe + claude_subprocess_per_move",
}
print(f"RESULT: {json.dumps(result)}", flush=True)
