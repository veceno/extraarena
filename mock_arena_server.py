#!/usr/bin/env python3
"""Mock-сервер ExtraArena: статика + REST + socket.io (asyncio + aiohttp).

Использует aiohttp + python-socketio (ASGIApp) на одном порту.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from urllib.parse import urlparse, parse_qs

import socketio
from aiohttp import web

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(ROOT, "webapp")

USER_ID = 6803854304

INITIAL_STATE = {
    "match_id": "battle-mock-1",
    "battle_id": "battle-mock-1",
    "turn": 3,
    "is_my_turn": True,
    "time_remaining": 25,
    "current_player_id": USER_ID,
    "viewer_id": USER_ID,
    "player_ids": [USER_ID, 99999],
    "player": {
        "user_id": USER_ID,
        "name": "Игрок",
        "title": "",
        "hp": 25,
        "max_hp": 30,
        "hero": {"hp": 25, "max_hp": 30},
        "hand": [],
        "mana": 5,
        "max_mana": 10,
        "board": [
            {"instance_id": "pb-1", "card_id": 40, "name": "Стив", "attack": 3, "hp": 3,
             "max_hp": 3, "mechanics": [], "can_attack": True, "sleeping": False},
            {"instance_id": "pb-2", "card_id": 7, "name": "Олег Тиньков", "attack": 2, "hp": 4,
             "max_hp": 4, "mechanics": [], "can_attack": True, "sleeping": False},
        ],
    },
    "opponent": {
        "user_id": 99999,
        "name": "Оппонент",
        "title": "",
        "hp": 25,
        "max_hp": 30,
        "hero": {"hp": 25, "max_hp": 30},
        "hand": [],
        "mana": 4,
        "max_mana": 9,
        "board": [
            {"instance_id": "ob-1", "card_id": 22, "name": "Наофуми", "attack": 4, "hp": 6,
             "max_hp": 6, "mechanics": ["taunt"], "can_attack": False, "sleeping": True},
            {"instance_id": "ob-2", "card_id": 29, "name": "Штурмовик", "attack": 2, "hp": 3,
             "max_hp": 3, "mechanics": [], "can_attack": False, "sleeping": True},
        ],
    },
    "legal_actions": [
        {"type": "attack", "attacker_id": "pb-1", "target_id": "ob-1", "target_is_hero": False,
         "reason": "valid target"},
    ],
}

game_state = dict(INITIAL_STATE)
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")


@sio.event
async def connect(sid, environ):
    print(f"[sio] client connected: {sid}")
    await sio.emit("state_changed", {"state": game_state}, to=sid)


@sio.event
async def join_battle(sid, data):
    print(f"[sio] join_battle: {data}")
    await sio.emit("state_changed", {"state": game_state}, to=sid)


@sio.event
async def client_ready(sid, data):
    print(f"[sio] client_ready: {data}")
    await sio.emit("state_changed", {"state": game_state}, to=sid)


@sio.event
async def disconnect(sid):
    print(f"[sio] client disconnected: {sid}")


# --- HTTP handlers ---

async def static_handler(request: web.Request) -> web.Response:
    rel = request.match_info.get("path", "")
    if rel in ("", "arena", "battle"):
        rel = "arena.html"
    full = os.path.normpath(os.path.join(STATIC_ROOT, rel))
    if not full.startswith(STATIC_ROOT) or not os.path.isfile(full):
        return web.Response(status=404, text="not found")
    ext = os.path.splitext(full)[1].lower()
    ctype = {
        ".html": "text/html", ".js": "text/javascript",
        ".css": "text/css", ".json": "application/json",
        ".svg": "image/svg+xml", ".png": "image/png",
        ".jpg": "image/jpeg", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".ico": "image/x-icon",
    }.get(ext, "application/octet-stream")
    with open(full, "rb") as f:
        data = f.read()
    return web.Response(
        body=data, content_type=ctype,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


async def api_battle_state(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "state": game_state})


async def api_battle_action(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "state": game_state})


async def api_battle_connect(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "redirect_url": f"/arena?id={game_state['match_id']}"})


async def api_whoami(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "user_id": USER_ID, "auth": "mock"})


async def api_default(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "mocked": True, "endpoint": request.path})


async def healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def make_app() -> web.Application:
    app = web.Application()
    sio.attach(app)

    # API (register BEFORE catch-all static, otherwise the regex swallows them)
    app.router.add_get("/api/battle/state", api_battle_state)
    app.router.add_post("/api/battle/state", api_battle_state)
    app.router.add_post("/api/battle/action", api_battle_action)
    app.router.add_post("/api/battle/attack", api_battle_action)
    app.router.add_post("/api/battle/play-card", api_battle_action)
    app.router.add_post("/api/battle/end-turn", api_battle_action)
    app.router.add_post("/api/battle/mana-draw", api_battle_action)
    app.router.add_get("/api/battle/connect", api_battle_connect)
    app.router.add_get("/api/rlhf/battle/connect", api_battle_connect)
    app.router.add_get("/api/rlhf/battle/state", api_battle_state)
    app.router.add_post("/api/rlhf/battle/state", api_battle_state)
    app.router.add_post("/api/rlhf/battle/action", api_battle_action)
    app.router.add_get("/api/rlhf/auth/whoami", api_whoami)
    app.router.add_post("/api/rlhf/auth/whoami", api_whoami)
    app.router.add_get("/healthz", healthz)

    # Static (catch-all last)
    app.router.add_get("/", static_handler)
    app.router.add_get("/arena", static_handler)
    app.router.add_get("/battle", static_handler)
    app.router.add_get("/arena.html", static_handler)
    app.router.add_get(r"/{path:.*}", static_handler)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    if not os.path.isdir(STATIC_ROOT):
        sys.exit(f"webapp/ not found at {STATIC_ROOT}")
    app = make_app()
    print(f"[mock] http+socket.io on http://{args.host}:{args.port} (static={STATIC_ROOT})", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=lambda *a, **kw: None)


if __name__ == "__main__":
    main()