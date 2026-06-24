#!/usr/bin/env python3
"""Интерактивный WS-smoke: открыть бой, дождаться state, поиграть end_turn до конца."""
import asyncio
import json
import sys

import websockets


async def play_against_bot(host: str, port: int, gid: str, bid: str) -> dict:
    """Играем: на каждом своём ходу шлём end_turn (или play_card)."""
    url = f"ws://{host}:{port}/ws/groups/{gid}/battles/{bid}"
    print(f"[ws-smoke] connecting to {url}")
    final_result = None
    states_seen = 0
    async with websockets.connect(url, open_timeout=10) as ws:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[ws-smoke] TIMEOUT — no message for 30s")
                return {"error": "timeout", "states_seen": states_seen}
            except websockets.ConnectionClosed as e:
                print(f"[ws-smoke] connection closed: {e}")
                break
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "state":
                states_seen += 1
                s = msg.get("state", {})
                legal = msg.get("legal_actions", [])
                your_turn = msg.get("your_turn")
                turn = s.get("turn_number")
                p1_hp = s.get("p1", {}).get("hp")
                p2_hp = s.get("p2", {}).get("hp")
                status = s.get("status")
                print(
                    f"[ws-smoke] state #{states_seen}: turn={turn} "
                    f"p1_hp={p1_hp} p2_hp={p2_hp} status={status} "
                    f"your_turn={your_turn} legal={len(legal)}"
                )
                if not your_turn:
                    # Ждём ход бота (state придёт снова с your_turn=True)
                    continue
                # Наш ход — ищем end_turn в legal_actions
                end_turn_idx = None
                for la in legal:
                    if la.get("action", {}).get("type") == "end_turn":
                        end_turn_idx = la["index"]
                        break
                if end_turn_idx is None:
                    # Может быть только play_card — попробуем
                    play_card_idx = None
                    for la in legal:
                        if la.get("action", {}).get("type") == "play_card":
                            play_card_idx = la["index"]
                            break
                    if play_card_idx is not None:
                        await ws.send(json.dumps({"type": "action", "index": play_card_idx}))
                    else:
                        print(f"[ws-smoke] no usable action in {legal!r}")
                        break
                else:
                    await ws.send(json.dumps({"type": "action", "index": end_turn_idx}))
            elif t == "result":
                final_result = msg.get("battle_log", {}).get("result", {})
                print(f"[ws-smoke] RESULT: {final_result}")
                break
            elif t == "error":
                print(f"[ws-smoke] ERROR: {msg.get('message')}")
                return {"error": msg.get("message"), "states_seen": states_seen}
    return {"final_result": final_result, "states_seen": states_seen}


async def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()

    # 1) Создаём interactive группу
    import urllib.request
    spec = {
        "interactive": True,
        "p1_model": "random",
        "p2_model": "end_turn",
        "battles_planned": 1,
        "seed": 42,
        "starting_player": "random",
        "max_turns": 30,
        "human_player": 1000,
        "deck_strategy": "random_arenaenv",
    }
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/api/groups",
        data=json.dumps(spec).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    gid = data["group_id"]
    bid = data["battle_id"]
    print(f"[ws-smoke] group={gid} battle={bid}")

    # 2) Играем
    res = await play_against_bot(args.host, args.port, gid, bid)
    print(f"[ws-smoke] DONE: {res}")

    # 3) Проверяем, что battle_log.json на диске
    import os
    log_path = f"/tmp/rlhf_sessions/{gid}/battles/{bid}.json"
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        actions = log.get("actions", [])
        human_actions = [a for a in actions if a.get("actor") == 1000]
        bot_actions = [a for a in actions if a.get("actor") == 2000]
        print(f"[ws-smoke] battle_log.json: total_actions={len(actions)} "
              f"(human={len(human_actions)}, bot={len(bot_actions)})")
        print(f"[ws-smoke] first human action: {human_actions[0] if human_actions else 'NONE'}")
        print(f"[ws-smoke] ✅ battle_log written")
        return 0
    else:
        print(f"[ws-smoke] ❌ battle_log.json NOT found at {log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))