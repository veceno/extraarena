#!/usr/bin/env python3
"""Smoke-тест серии из N боёв human-vs-model через WS.

Создаёт группу с battles_planned=N, поочерёдно играет каждый бой end_turn'ом
и проверяет, что:
  - все N battle_log.json записаны на диск;
  - manifest.json содержит N battle_ids;
  - после последнего боя next_battle_id == None.
"""
import asyncio
import json
import os
import sys
import urllib.request

import websockets


async def play_one_battle(host: str, port: int, gid: str, bid: str) -> dict:
    url = f"ws://{host}:{port}/ws/groups/{gid}/battles/{bid}"
    final_result = None
    states_seen = 0
    series_index = None
    series_total = None
    async with websockets.connect(url, open_timeout=10) as ws:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
            except asyncio.TimeoutError:
                return {"error": "timeout", "states_seen": states_seen}
            except websockets.ConnectionClosed as e:
                return {"error": f"closed: {e}", "states_seen": states_seen}
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "state":
                states_seen += 1
                if series_index is None:
                    series_index = msg.get("series_index")
                    series_total = msg.get("series_total")
                legal = msg.get("legal_actions", [])
                your_turn = msg.get("your_turn")
                if not your_turn:
                    continue
                # Ищем end_turn или play_card
                idx = None
                for la in legal:
                    if la.get("action", {}).get("type") == "end_turn":
                        idx = la["index"]; break
                if idx is None:
                    for la in legal:
                        if la.get("action", {}).get("type") == "play_card":
                            idx = la["index"]; break
                if idx is None:
                    return {"error": "no usable action", "states_seen": states_seen}
                await ws.send(json.dumps({"type": "action", "index": idx}))
            elif t == "result":
                final_result = msg
                return {
                    "result": msg.get("battle_log", {}).get("result", {}),
                    "states_seen": states_seen,
                    "series_index": series_index,
                    "series_total": series_total,
                    "next_battle_id": msg.get("next_battle_id"),
                }
            elif t == "error":
                return {"error": msg.get("message"), "states_seen": states_seen}
    return {"error": "no result", "states_seen": states_seen}


async def main_async(args) -> int:
    spec = {
        "interactive": True,
        "p1_model": "human",
        "p2_model": "end_turn",
        "battles_planned": args.n,
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
    print(f"[series] group={gid}, battles_planned={data['battles_planned']}")
    manifest_bids = data.get("battle_ids") or [data["battle_id"]]
    # Создать manifest_path → проверим
    print(f"[series] returned battle_ids={manifest_bids}")

    cur_bid = data["battle_id"]
    i = 1
    while cur_bid is not None:
        print(f"[series] === playing battle {i}/{args.n} (bid={cur_bid}) ===")
        res = await play_one_battle(args.host, args.port, gid, cur_bid)
        if "error" in res:
            print(f"[series] ❌ error: {res['error']} (states_seen={res.get('states_seen')})")
            return 1
        print(f"[series] battle {i}: result={res['result']} "
              f"series_index={res['series_index']}/{res['series_total']} "
              f"states={res['states_seen']} next={res['next_battle_id']}")
        cur_bid = res["next_battle_id"]
        i += 1
        if i > args.n + 1:
            print(f"[series] ❌ runaway loop (more than {args.n} battles)")
            return 1

    # Проверка: N battle_log.json на диске
    log_dir = f"/tmp/rlhf_sessions/{gid}/battles"
    files = sorted(os.listdir(log_dir)) if os.path.isdir(log_dir) else []
    json_files = [f for f in files if f.endswith(".json")]
    print(f"[series] disk: {len(json_files)} battle_log.json in {log_dir}")
    if len(json_files) != args.n:
        print(f"[series] ❌ expected {args.n} logs, got {len(json_files)}")
        return 1

    # Проверка: manifest.json содержит N battle_ids
    mp = f"/tmp/rlhf_sessions/{gid}/manifest.json"
    with open(mp) as f:
        m = json.load(f)
    if len(m.get("battle_ids", [])) != args.n:
        print(f"[series] ❌ manifest.battle_ids has {len(m.get('battle_ids', []))} items, expected {args.n}")
        return 1

    # Проверка: summary.json
    sp = f"/tmp/rlhf_sessions/{gid}/summary.json"
    if not os.path.exists(sp):
        print(f"[series] ❌ summary.json missing")
        return 1
    with open(sp) as f:
        s = json.load(f)
    print(f"[series] summary: {s}")

    print(f"[series] ✅ {args.n} battles played, all logs written, manifest complete")
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())