"""Шаг 1: HeadlessHub + MCPServer._tool прогон боя без subprocess."""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import logging
logging.basicConfig(level=logging.WARNING)

from rlhf_env.mcp_server import HeadlessHub, MCPServer
from rlhf_env.components.policy_registry import PolicyRegistry

SPEC = {
    "p2_model": "random",
    "difficulty": "default",
    "battles_planned": 1,
    "seed": 7,
    "starting_player": "p1",
}

MAX_STEPS = 150


def build_action(legal: dict) -> dict:
    """Преобразовать сериализованное легальное действие в action для submit_action."""
    t = legal.get("type")
    if t == "play_card":
        return {
            "type": "play_card",
            "hand_index": legal.get("hand_index"),
            "target_position": legal.get("position") or 0,
            "target_id": legal.get("target_id"),
        }
    if t == "attack":
        return {
            "type": "attack",
            "attacker_id": legal.get("attacker_id"),
            "target_id": legal.get("target_id"),
            "target_is_hero": legal.get("target_is_hero", False),
        }
    return {"type": "end_turn"}


async def run() -> int:
    registry = PolicyRegistry.scan("ai/models")
    hub = HeadlessHub(sessions_dir="rlhf_env/sessions", models_dir="ai/models", cards_path="ai/cards.json")
    server = MCPServer(hub, registry)

    log = []

    try:
        start = await server._tool("start_series", {"spec": SPEC})
    except Exception:
        log.append({"event": "start_series_exception", "tb": traceback.format_exc()})
        print(json.dumps(log, ensure_ascii=False, indent=2))
        return 1
    log.append({"event": "start_series", "data": start})
    match_id = start.get("match_id")
    group_id = start.get("group_id")
    if not match_id:
        log.append({"event": "no_match_id"})
        print(json.dumps(log, ensure_ascii=False, indent=2))
        return 1

    step = 0
    game_over = False
    while step < MAX_STEPS and not game_over:
        step += 1
        try:
            state = await server._tool("get_state", {"match_id": match_id})
        except Exception:
            log.append({"event": "get_state_exception", "step": step, "tb": traceback.format_exc()})
            break
        is_ended = state.get("is_ended") or state.get("game_over")
        if is_ended:
            log.append({"event": "game_over_at_state", "step": step, "winner_id": state.get("winner_id")})
            game_over = True
            break

        try:
            legal_resp = await server._tool("get_legal_actions", {"match_id": match_id})
        except Exception:
            log.append({"event": "get_legal_actions_exception", "step": step, "tb": traceback.format_exc()})
            break
        legal = legal_resp.get("legal_actions", [])
        is_my = legal_resp.get("is_my_turn", False)

        if not is_my or not legal:
            # ход бота — прокручиваем
            try:
                r = await server._tool("advance_bot", {"match_id": match_id})
                log.append({"event": "advance_bot", "step": step, "is_ended": r.get("is_ended")})
                if r.get("is_ended"):
                    game_over = True
                    break
            except Exception:
                log.append({"event": "advance_bot_exception", "step": step, "tb": traceback.format_exc()})
                break
            continue

        # выбираем первое легальное
        chosen = legal[0]
        action = build_action(chosen)
        try:
            resp = await server._tool("submit_action", {"match_id": match_id, "action": action})
        except Exception:
            log.append({"event": "submit_action_exception", "step": step, "action": action, "tb": traceback.format_exc()})
            break
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        is_ended = (resp.get("state", {}) or {}).get("is_ended") or result.get("game_over")
        log.append({
            "event": "submit_action",
            "step": step,
            "action": action,
            "success": result.get("success"),
            "error": result.get("error") or resp.get("error"),
            "is_ended": is_ended,
            "winner_id": (resp.get("state", {}) or {}).get("winner_id"),
        })
        if is_ended:
            game_over = True
            break

    if not game_over:
        # surrender
        try:
            r = await server._tool("surrender", {"match_id": match_id})
            log.append({"event": "surrender", "data": r})
        except Exception:
            log.append({"event": "surrender_exception", "tb": traceback.format_exc()})

    # get_dataset
    try:
        ds = await server._tool("get_dataset", {"group_id": group_id})
        log.append({"event": "get_dataset", "data": ds})
    except Exception:
        log.append({"event": "get_dataset_exception", "tb": traceback.format_exc()})

    # read NDJSON rows
    ndjson_summary = {}
    try:
        ds_path = Path(ds["dataset_jsonl"])
        if ds_path.exists():
            rows = [json.loads(line) for line in ds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            ndjson_summary = {
                "rows": len(rows),
                "decision_sources": sorted({r.get("decision_source") for r in rows}),
                "accepted_values": sorted({str(r.get("accepted")) for r in rows}),
                "won_values": sorted({str(r.get("won")) for r in rows}),
                "winner_user_ids": sorted({str(r.get("winner_user_id")) for r in rows}),
                "is_bot_values": sorted({str(r.get("is_bot")) for r in rows}),
                "visibility_values": sorted({r.get("visibility") for r in rows}),
            }
            # F01: opponent hand hidden check in state_json
            opp_hand_hidden_count = 0
            for r in rows:
                sj = r.get("state_json", {})
                # acting_user_id
                acting = r.get("acting_user_id")
                # если актор — p1 (human_user_id 1000), то p2 — оппонент
                p2 = sj.get("p2", {})
                hand = p2.get("hand", [])
                if hand and all(isinstance(h, dict) and h.get("hidden") is True for h in hand):
                    opp_hand_hidden_count += 1
            ndjson_summary["opp_hand_hidden_rows"] = opp_hand_hidden_count
            log.append({"event": "ndjson_summary", "data": ndjson_summary})
    except Exception:
        log.append({"event": "ndjson_read_exception", "tb": traceback.format_exc()})

    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))