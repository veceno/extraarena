import json, subprocess, sys, time, select, os

REPO = "/Users/laveqox/Documents/ExtraArenaRaS"

mcp = subprocess.Popen(
    [sys.executable, "-u", "-m", "rlhf_env.mcp_server",
     "--models-dir", "ai/models",
     "--sessions-dir", "rlhf_env/sessions",
     "--cards-path", "ai/cards.json",
     "--log-level", "ERROR"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    cwd=REPO
)

def mcp_call(method, params, timeout=180):
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

battle = {"seed": 10126, "agent_name": "v4max-selfplay-w126"}

spec = {
    "p1_actor_type": "rl",
    "p1_model": "extra-lr-v4-max",
    "p2_model": "extra-lr-v4-max",
    "battles_planned": 1,
    "seed": battle["seed"],
    "starting_player": "random",
    "deck_strategy_p1": "random_arenaenv",
    "deck_strategy_p2": "random_arenaenv",
    "p1_name": "v4max-p1",
    "p2_name": "v4max-p2",
    "agent_name": battle["agent_name"],
}
r = mcp_call("tools/call", {"name": "start_series", "arguments": {"spec": spec}}, timeout=180)
if not isinstance(r, dict) or "error" in r or "match_id" not in r:
    print(f"FATAL start_series: {r}", flush=True)
    mcp.terminate()
    raise SystemExit(1)
mid = r["match_id"]
gid = r["group_id"]
winner = r.get("winner_id")
is_ended = r.get("is_ended", False)
turns = 0
print(f"start: match={mid} group={gid} seed={battle['seed']} winner={winner} ended={is_ended}", flush=True)

if not is_ended:
    step = 0
    for step in range(50):
        s = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
        if not isinstance(s, dict): continue
        if s.get("is_ended") or s.get("winner_id") is not None:
            winner = s.get("winner_id", winner)
            is_ended = True
            break
        mcp_call("tools/call", {"name": "advance_bot", "arguments": {"match_id": mid}}, timeout=30)
        time.sleep(0.1)
    turns = step + 1

final = mcp_call("tools/call", {"name": "get_match_status", "arguments": {"match_id": mid}}, timeout=10)
if isinstance(final, dict) and final.get("is_ended"):
    winner = final.get("winner_id", winner)

mcp_call("tools/call", {"name": "finish_series", "arguments": {"group_id": gid}}, timeout=30)
mcp.terminate()
try:
    mcp.wait(timeout=5)
except Exception:
    mcp.kill()

result = {
    "agent_idx": 126,
    "seed": battle["seed"],
    "match_id": mid,
    "group_id": gid,
    "winner_id": winner,
    "winner_side": "p1" if winner == 1000 else "p2" if winner == 2000 else "draw_or_unknown",
    "is_ended": is_ended,
    "turns_played": turns,
    "transport": "rl-vs-rl run_auto",
}
print(f"RESULT: {json.dumps(result)}", flush=True)
