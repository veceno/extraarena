#!/usr/bin/env python3
"""L2 player sub-agent: plays 1 battle as p1 vs V4-Max model.
Agent index 137, seed 8137, starting_player=v4_max.
"""
import json
import subprocess
import sys
import time
import select
import os

REPO = "/Users/laveqox/Documents/ExtraArenaRaS"

# === Persistent MCP subprocess — один процесс на ВЕСЬ бой ===
mcp = None


def mcp_start():
    global mcp
    if mcp is not None:
        try:
            mcp.terminate()
            mcp.wait(timeout=2)
        except Exception:
            try:
                mcp.kill()
            except Exception:
                pass
    mcp = subprocess.Popen(
        [sys.executable, "-u", "-m", "rlhf_env.mcp_server",
         "--models-dir", "ai/models",
         "--sessions-dir", "rlhf_env/sessions",
         "--cards-path", "ai/cards.json",
         "--log-level", "ERROR"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        cwd=REPO
    )


mcp_start()


def mcp_call(method, params, timeout=120):
    global mcp
    req_id = int(time.time() * 1_000_000) & 0x7fffffff
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    for attempt in range(2):
        if mcp is None or mcp.poll() is not None:
            print(f"[mcp] restarting (attempt {attempt})", flush=True)
            mcp_start()
        try:
            mcp.stdin.write((json.dumps(msg) + "\n").encode())
            mcp.stdin.flush()
        except Exception as e:
            print(f"[mcp] stdin_write: {e}, restarting", flush=True)
            mcp_start()
            continue
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
    return {"error": "max_retries"}


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

    def fmt_card(c, on_board):
        cid = c.get("instance_id", "")[:8] if on_board else ""
        atk = c.get("attack", 0)
        hp = c.get("hp_current", c.get("hp", 0)) if on_board else c.get("hp", 0)
        ready = c.get("can_attack", False) and c.get("is_ready", False)
        mech = c.get("mechanics", []) or []
        tags = ""
        if "taunt" in mech: tags += " TAUNT"
        if "shield" in mech: tags += " SHIELD"
        if "charge" in mech: tags += " CHARGE"
        if "divine_shield" in mech: tags += " DS"
        if on_board:
            return f"[{cid} atk={atk} hp={hp} {'READY' if ready else 'SLEEP'}{tags}]"
        return f"(cost={c.get('mana_cost',0)} atk={atk} hp={hp})"

    my_board_s = " ".join(fmt_card(c, True) for c in my_board) or "(empty)"
    opp_board_s = " ".join(fmt_card(c, True) for c in opp_board) or "(empty)"
    my_hand_s = " ".join(fmt_card(c, False) for c in my_hand) or "(empty)"

    la_summary = {}
    la_examples = {"play_card": [], "attack": []}
    for a in legal:
        t = a.get("type", "?")
        la_summary[t] = la_summary.get(t, 0) + 1
        if t in la_examples and len(la_examples[t]) < 3:
            la_examples[t].append(a)
    la_count = ", ".join(f"{k}={v}" for k, v in la_summary.items() if v)
    notes_text = "\n".join(strategic_notes[-3:]) if strategic_notes else "(start of battle)"
    history_text = "\n".join(battle_history[-6:]) if battle_history else "(start)"

    # Compute immediate damage potential for self
    my_atk_total = sum(c.get("attack", 0) for c in my_board if c.get("can_attack") and c.get("is_ready"))
    opp_atk_total = sum(c.get("attack", 0) for c in opp_board)

    prompt = f"""You are playing ExtraArena, a Hearthstone-like card game. Turn {turn}. You are p1 (left side) playing against V4-Max bot (V4 ONNX policy).

=== STATE SNAPSHOT ===
my_hp={my_hp}/{my_max_hp}    opp_hp={opp_hp}/{opp_max_hp}
my_mana={my_mana}/{my_max_mana}    opp_mana={opp_mana}
MY HAND ({len(my_hand)} cards): {my_hand_s}
MY BOARD ({len(my_board)} minions): {my_board_s}
OPP BOARD ({len(opp_board)} minions): {opp_board_s}
opp_hand_size={opp_hand_count} (hidden)
immediate_threat: I can deal {my_atk_total} dmg face now, opponent can deal {opp_atk_total} dmg face next turn
LEGAL ACTIONS: {la_count}
play_card examples: {json.dumps(la_examples.get('play_card',[])[:2])}
attack examples:  {json.dumps(la_examples.get('attack',[])[:2])}
end_turn / mana_draw: also available if listed

=== STRATEGIC NOTES (this battle) ===
{notes_text}

=== RECENT MOVES ===
{history_text}

=== DECISION FRAMEWORK ===
Think step by step before answering:
1. LETHAL CHECK: can I kill opponent this turn? Total my board atk vs opp_hp. If yes, attack face.
2. THREAT CHECK: how much dmg can opp deal next turn? Any taunts? If lethal from opp, must clear/shield.
3. BOARD TRADES: which enemy minion is most dangerous (high atk, will kill my stuff)? Use my atks efficiently.
4. CARD PLAYS: which card from hand gives best tempo? Consider cost vs board impact (taunts, big stats, removal, draw).
5. MANA USAGE: any unspent mana I should use? mana_draw spends 2 mana to draw 1 card (hand was empty). Use when hand empty or you need answers and opp will overextend.
6. PASSING: only end_turn if no action improves your position.

The V4-Max bot is a strong policy. It values board control, face damage, and trades efficiently. Try to outmaneuver it by:
- Reading the board state carefully (sleep vs ready, taunts, shields)
- Not over-committing into aoe
- Preserving health when ahead
- Going face when you can race (opponent can't kill you back)
- Using mana_draw to refill hand when you have mana but no plays

Return ONLY this JSON (no markdown, no extra text):
{{"action": <one legal_action object exactly as in legal list>, "reasoning": "<one sentence why>"}}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=120,
            cwd=REPO
        )
        if result.returncode != 0 or not result.stdout:
            print(f"[llm] claude failed: rc={result.returncode} stderr={result.stderr[:200]}", flush=True)
            return {"type": "end_turn"}
        response = result.stdout.strip()
        # Find JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            if "action" in parsed and isinstance(parsed["action"], dict):
                reasoning = parsed.get("reasoning", "")
                print(f"[llm] chose: {parsed['action'].get('type')} | {reasoning[:100]}", flush=True)
                return parsed["action"]
    except Exception as e:
        print(f"[llm] exception: {e}", flush=True)
    return {"type": "end_turn"}


def main():
    battle = {"seed": 8137, "starting_player": "v4_max", "agent_name": "mix-w137-v4first"}
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
    print(f"START: match={mid} group={gid} seed={battle['seed']} sp={battle['starting_player']}", flush=True)

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
    rejected_actions = []

    for step in range(500):
        if mcp is not None and mcp.poll() is not None:
            print(f"[mcp] dead rc={mcp.returncode}, restarting", flush=True)
            mcp_start()
        time.sleep(0.15)  # gentle polling
        s = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
        if not isinstance(s, dict):
            continue
        if s.get("is_ended") or s.get("winner_id") is not None:
            winner = s.get("winner_id")
            break
        if not s.get("is_my_turn"):
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
            rejected_actions.append((step, last_turn, action, r))
            print(f"[REJECT] step={step} t={last_turn} action={action} resp={r}", flush=True)
            if consec_reject >= 3:
                mcp_call("tools/call", {"name": "submit_action", "arguments": {"match_id": mid, "action": {"type": "end_turn"}}}, timeout=10)
                consec_reject = 0
        else:
            consec_reject = 0
            # Add strategic note on key events
            if atype == "mana_draw":
                strategic_notes.append(f"t{last_turn}: I used mana_draw to refill hand")
            if atype == "play_card":
                hand_before = state.get("player", {}).get("hand", [])
                # see what got played
                target_card = action.get("card_id") or action.get("hand_index")
                strategic_notes.append(f"t{last_turn}: I played a card (hand_idx/action={target_card})")

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
        "agent_idx": 137,
        "seed": battle["seed"],
        "starting_player": battle["starting_player"],
        "match_id": mid,
        "group_id": gid,
        "winner_id": winner,
        "winner_side": "p1" if winner == 1000 else ("p2" if winner == 2000 else "draw_or_unknown"),
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
    # Also write to file
    with open("/Users/laveqox/Documents/ExtraArenaRaS/l2_battle_137_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
