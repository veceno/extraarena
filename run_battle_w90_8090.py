#!/usr/bin/env python3
"""L2 player sub-agent: play 1 battle as p1 vs V4-Max.
seed=8090, starting_player=llm, agent_name=mix-w90-llmfirst.
Uses raw MCP subprocess + claude -p for each move.
"""
import json
import subprocess
import sys
import time
import select
import os

REPO = "/Users/laveqox/Documents/ExtraArenaRaS"

# === Persistent MCP subprocess — held for ENTIRE battle (not per call) ===
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
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    try:
        mcp.stdin.write((json.dumps(msg) + "\n").encode())
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
    my_board_s = ", ".join(
        f"id={c.get('instance_id', '')[:8]} atk={c.get('attack', 0)} hp={c.get('hp_current', 0)} "
        f"{'READY' if c.get('can_attack') and c.get('is_ready') else 'SLEEP'}"
        f"{' TAUNT' if 'taunt' in c.get('mechanics', []) else ''}"
        f"{' SHIELD' if 'shield' in c.get('mechanics', []) else ''}"
        f"{' CHARGE' if 'charge' in c.get('mechanics', []) else ''}"
        f"{' RUSH' if 'rush' in c.get('mechanics', []) else ''}"
        for c in my_board
    ) or "(empty)"
    opp_board_s = ", ".join(
        f"atk={c.get('attack', 0)} hp={c.get('hp_current', 0)} "
        f"{'TAUNT' if 'taunt' in c.get('mechanics', []) else ''}"
        f"{' SHIELD' if 'shield' in c.get('mechanics', []) else ''}"
        f"{' CHARGE' if 'charge' in c.get('mechanics', []) else ''}"
        f"{' RUSH' if 'rush' in c.get('mechanics', []) else ''}"
        for c in opp_board
    ) or "(empty)"
    my_hand_s = ", ".join(
        f"cost={c.get('mana_cost', 0)} atk={c.get('attack', 0)} hp={c.get('hp', 0)} "
        f"{'[' + ','.join(c.get('mechanics', [])) + ']' if c.get('mechanics') else ''}"
        for c in my_hand
    ) or "(empty)"
    la_summary = {"play_card": 0, "attack": 0, "end_turn": 0, "mana_draw": 0}
    la_examples = {"play_card": [], "attack": []}
    for a in legal:
        t = a.get("type", "?")
        la_summary[t] = la_summary.get(t, 0) + 1
        if t in la_examples and len(la_examples[t]) < 3:
            la_examples[t].append(a)
    la_count = ", ".join(f"{k}={v}" for k, v in la_summary.items() if v)

    notes_block = "\n".join(strategic_notes[-5:]) if strategic_notes else "(start of battle)"
    history_block = "\n".join(battle_history[-6:]) if battle_history else "(start)"

    prompt = f"""ExtraArena RLHF battle, turn {turn}. You=p1 (LLM), opponent=V4-Max bot.

== STATE SNAPSHOT ==
- My HP: {my_hp}/{my_max_hp}
- Opp HP: {opp_hp}/{opp_max_hp}
- My mana: {my_mana}/{my_max_mana}
- Opp mana: {opp_mana}
- My hand ({len(my_hand)}): {my_hand_s}
- My board ({len(my_board)}): {my_board_s}
- Opp board ({len(opp_board)}): {opp_board_s}
- Opp hand: {opp_hand_count} cards (hidden)

== LEGAL ACTIONS ==
Counts: {la_count}
Sample play_card: {json.dumps(la_examples.get('play_card', [])[:2], ensure_ascii=False)}
Sample attack: {json.dumps(la_examples.get('attack', [])[:2], ensure_ascii=False)}

== STRATEGIC NOTES (this battle so far) ==
{notes_block}

== RECENT MOVES ==
{history_block}

== THINK DEEPLY, STEP BY STEP ==
1. LETHAL CHECK: Can I kill the opponent this turn? Total my-board attack (only READY/Charge/Rush) vs opp_hp.
2. SURVIVAL: Can opponent kill me next turn? Sum of opp-board attack vs my_hp. Any taunts I can play to block?
3. THREAT ANALYSIS: Which opponent units are most dangerous? Big atk? Taunt? Lethal next turn?
4. TRADES: If I attack, can I kill a high-value opp unit without dying? Or do I attack face for damage?
5. HAND OPTIONS: For each playable card, what does it give me? Taunt = blocker, Charge/Rush = immediate pressure, big-stat = tempo. Don't waste mana on low-impact plays.
6. MANA_DRAW: Use only if my hand is empty/desperately small AND I have mana AND I need answers. mana_draw costs 1 mana (I think — check), gives a card. If hand is fine and board needs work, prefer playing cards.
7. END_TURN: Only if no useful action. Sometimes passing with mana is correct (e.g., save for next turn, no good targets, want to keep reactive hand).

== OUTPUT FORMAT ==
Return ONLY this JSON (no prose, no markdown fences):
{{"action": <one full legal_action from the list above>, "reasoning": "<one concise sentence why>"}}

The action object must be COPIED VERBATIM from the legal actions list (same keys, same values). Do not invent new actions."""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=60,
            cwd=REPO
        )
        if result.returncode != 0 or not result.stdout:
            return {"type": "end_turn"}
        response = result.stdout.strip()
        # Strip code fences if any
        if response.startswith("```"):
            lines = response.splitlines()
            response = "\n".join(l for l in lines if not l.strip().startswith("```"))
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(response[start:end])
            except Exception:
                return {"type": "end_turn"}
            if "action" in parsed and isinstance(parsed["action"], dict):
                return parsed["action"]
    except subprocess.TimeoutExpired:
        return {"type": "end_turn"}
    except Exception:
        pass
    return {"type": "end_turn"}


# === MAIN BATTLE ===
battle = {"seed": 8090, "starting_player": "llm", "agent_name": "mix-w90-llmfirst"}
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
    raise SystemExit(1)
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
move_count = 0

for step in range(500):
    if mcp.poll() is not None:
        print(f"FATAL mcp_died_rc={mcp.returncode}", flush=True)
        break
    s = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
    if not isinstance(s, dict):
        continue
    if s.get("is_ended") or s.get("winner_id") is not None:
        winner = s.get("winner_id")
        break
    if not s.get("is_my_turn"):
        # Opponent's turn — just advance
        mcp_call("tools/call", {"name": "advance_bot", "arguments": {"match_id": mid}}, timeout=15)
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
    move_count += 1
    action = llm_choose_action(state, legal, strategic_notes, battle_history)
    r = mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": action}}, timeout=10)
    my_actions += 1
    atype = action.get("type")
    if atype == "play_card":
        plays += 1
    elif atype == "attack":
        attacks_made += 1
        if action.get("target_is_hero"):
            face_attacks += 1
    elif atype == "end_turn":
        end_turns_made += 1
    elif atype == "mana_draw":
        mana_draws_made += 1
    battle_history.append(f"step={step} turn={last_turn} move#{move_count}: {atype} {json.dumps(action, ensure_ascii=False)[:120]}")
    # Adaptive strategic notes
    if move_count % 3 == 0 or my_hp_end < 10 or opp_hp_end < 10:
        strategic_notes.append(
            f"turn={last_turn}: my_hp={my_hp_end} opp_hp={opp_hp_end} "
            f"my_mana={player.get('mana', 0)}/{player.get('max_mana', 0)} "
            f"hand={len(player.get('hand', []))} board={len(player.get('board', []))} "
            f"opp_board={len(opp.get('board', []))} opp_mana={opp.get('mana', 0)} opp_hand={opp.get('hand_count', 0)}"
        )
    rejected = False
    if isinstance(r, dict):
        if r.get("success") is False:
            rejected = True
        elif r.get("result", {}).get("success") is False:
            rejected = True
        elif r.get("error"):
            rejected = True
        elif r.get("is_ended") or r.get("winner_id") is not None or r.get("game_over"):
            winner = r.get("winner_id") or r.get("result", {}).get("winner_id")
            break
    if rejected:
        consec_reject += 1
        strategic_notes.append(f"turn={last_turn}: REJECTED action {atype} — try different")
        if consec_reject >= 3:
            mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": {"type": "end_turn"}}}, timeout=10)
            consec_reject = 0
    else:
        consec_reject = 0
    if move_count > 200:
        # Safety: extremely long battle, end it
        print(f"SAFETY: too many moves ({move_count}), ending", flush=True)
        break

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

winner_side = "p1" if winner == 1000 else ("p2" if winner == 2000 else "draw_or_unknown")
result = {
    "agent_idx": 90,
    "seed": battle["seed"],
    "starting_player": battle["starting_player"],
    "match_id": mid,
    "group_id": gid,
    "winner_id": winner,
    "winner_side": winner_side,
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
print(f"RESULT: {json.dumps(result, ensure_ascii=False)}", flush=True)
