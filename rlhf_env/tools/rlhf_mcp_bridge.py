#!/usr/bin/env python3
"""Persistent unix-socket bridge to rlhf MCPServer._tool (one bot-brain, shared).

Simple line protocol: client sends one JSON object {tool, args, id} per connection;
server returns one JSON line {id, result} (or {id, error}) and closes.
Lets many short-lived clients (one per move) drive a SINGLE persistent HeadlessHub
+ MCPServer = genuine MCP tools + one ONNX bot-brain across 10 concurrent battles.
"""
from __future__ import annotations
import asyncio, json, os, socket, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
logging.basicConfig(level=logging.WARNING)
from rlhf_env.mcp_server import HeadlessHub, MCPServer
from rlhf_env.components.policy_registry import PolicyRegistry

SOCK = "/tmp/rlhf_mcp.sock"

async def handle(conn, srv):
    req = None
    try:
        buf = b""
        conn.setblocking(False)
        while True:
            try:
                chunk = await asyncio.to_thread(lambda: conn.recv(65536))
            except BlockingIOError:
                await asyncio.sleep(0.005); continue
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf or len(buf) > 1_000_000:
                break
        req = json.loads(buf.decode("utf-8").strip() or "{}")
        if "tool" in req:
            result = await srv._tool(req["tool"], req.get("args", {}) or {})
        else:
            result = await srv.dispatch(req.get("method", "tools/call"), req.get("params", {}))
        conn.setblocking(True)
        conn.sendall((json.dumps({"id": req.get("id"), "result": result}, ensure_ascii=False) + "\n").encode())
    except Exception as e:
        try:
            conn.setblocking(True)
            conn.sendall((json.dumps({"id": (req or {}).get("id"), "error": str(e)}, ensure_ascii=False) + "\n").encode())
        except Exception:
            pass
    finally:
        try: conn.close()
        except Exception: pass

async def main():
    reg = PolicyRegistry.scan("ai/models")
    hub = HeadlessHub(sessions_dir="rlhf_env/sessions", models_dir="ai/models", cards_path="ai/cards.json")
    srv = MCPServer(hub, reg)
    if os.path.exists(SOCK): os.remove(SOCK)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK); s.listen(64)
    print(f"[bridge] listening on {SOCK}", flush=True)
    while True:
        conn, _ = await asyncio.to_thread(s.accept)
        asyncio.create_task(handle(conn, srv))

if __name__ == "__main__":
    asyncio.run(main())