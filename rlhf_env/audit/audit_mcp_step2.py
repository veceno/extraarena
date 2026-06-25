"""Шаг 2: реальный stdio MCP subprocess харнесс."""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SPEC = {
    "p2_model": "random",
    "difficulty": "default",
    "battles_planned": 1,
    "seed": 7,
    "starting_player": "p1",
}
MAX_STEPS = 150


def build_action(legal: dict) -> dict:
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


class MCPClient:
    def __init__(self, proc: asyncio.subprocess.Process):
        self.proc = proc
        self._req_id = 0
        self._stdout_buf = b""
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self.stderr_lines: list[str] = []
        self.initial: dict | None = None

    async def _drain_stderr(self) -> None:
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            self.stderr_lines.append(line.decode("utf-8", "replace").rstrip())

    async def _read_loop(self) -> None:
        # первое сообщение — ready line (id=0)
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                return
            try:
                msg = json.loads(line)
            except Exception:
                self.stderr_lines.append("UNPARSEABLE_STDOUT: " + line.decode("utf-8", "replace")[:200])
                continue
            # ready line has result.ready
            if self.initial is None and isinstance(msg.get("result"), dict) and "ready" in msg["result"]:
                self.initial = msg
                continue
            rid = msg.get("id")
            fut = self._pending.get(rid)
            if fut is not None:
                fut.set_result(msg)
            else:
                self.stderr_lines.append(f"ORPHAN_RESPONSE id={rid}: {str(msg)[:200]}")

    async def call(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self._req_id += 1
        rid = self._req_id
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        line = json.dumps(req) + "\n"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        self.proc.stdin.write(line.encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            resp = await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(rid, None)
        return resp

    async def call_tool(self, name: str, args: dict, timeout: float = 60.0) -> dict:
        resp = await self.call("tools/call", {"name": name, "arguments": args}, timeout=timeout)
        if "error" in resp:
            return {"_rpc_error": resp["error"]}
        result = resp.get("result", {})
        content = result.get("content", [])
        if content and isinstance(content, list) and content[0].get("type") == "json":
            return content[0].get("data", {})
        return result

    async def aclose(self) -> None:
        self._reader_task.cancel()
        self._stderr_task.cancel()


async def run() -> int:
    log: list[dict] = []
    env = {"RLHF_LOG_LEVEL": "WARNING", "PYTHONPATH": str(REPO)}
    import os
    env = {**os.environ, **env}
    proc = await asyncio.create_subprocess_exec(
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "-m", "rlhf_env.mcp_server",
        "--models-dir", "ai/models",
        "--sessions-dir", "rlhf_env/sessions",
        "--cards-path", "ai/cards.json",
        cwd=str(REPO),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = MCPClient(proc)

    try:
        # ждём initial ready line (с таймаутом)
        try:
            await asyncio.wait_for(_wait_initial(client), timeout=15.0)
            log.append({"event": "initial_ready", "data": client.initial})
        except asyncio.TimeoutError:
            log.append({"event": "initial_ready_timeout", "stderr": list(client.stderr_lines)})
            print(json.dumps(log, ensure_ascii=False, indent=2))
            return 1

        # initialize
        try:
            init = await client.call("initialize", {}, timeout=15.0)
            log.append({"event": "initialize", "data": init.get("result")})
        except Exception:
            log.append({"event": "initialize_exception", "tb": traceback.format_exc()})

        # tools/list
        try:
            tl = await client.call("tools/list", {}, timeout=15.0)
            tools = tl.get("result", {}).get("tools", [])
            log.append({"event": "tools_list", "count": len(tools), "names": [t.get("name") for t in tools]})
        except Exception:
            log.append({"event": "tools_list_exception", "tb": traceback.format_exc()})

        # start_series
        try:
            start = await client.call_tool("start_series", {"spec": SPEC}, timeout=30.0)
            log.append({"event": "start_series", "data": start})
        except Exception:
            log.append({"event": "start_series_exception", "tb": traceback.format_exc()})
            print(json.dumps(log, ensure_ascii=False, indent=2))
            return 1
        if "_rpc_error" in start or start.get("error"):
            log.append({"event": "start_series_failed", "data": start})
            print(json.dumps(log, ensure_ascii=False, indent=2))
            return 1

        match_id = start.get("match_id")
        group_id = start.get("group_id")

        step = 0
        game_over = False
        while step < MAX_STEPS and not game_over:
            step += 1
            try:
                state = await client.call_tool("get_state", {"match_id": match_id}, timeout=30.0)
            except Exception:
                log.append({"event": "get_state_exception", "step": step, "tb": traceback.format_exc()})
                break
            if state.get("is_ended") or state.get("game_over"):
                log.append({"event": "game_over_at_state", "step": step, "winner_id": state.get("winner_id")})
                game_over = True
                break
            try:
                legal_resp = await client.call_tool("get_legal_actions", {"match_id": match_id}, timeout=30.0)
            except Exception:
                log.append({"event": "get_legal_actions_exception", "step": step, "tb": traceback.format_exc()})
                break
            legal = legal_resp.get("legal_actions", [])
            is_my = legal_resp.get("is_my_turn", False)
            if not is_my or not legal:
                try:
                    r = await client.call_tool("advance_bot", {"match_id": match_id}, timeout=45.0)
                    log.append({"event": "advance_bot", "step": step, "is_ended": r.get("is_ended")})
                    if r.get("is_ended"):
                        game_over = True
                        break
                except Exception:
                    log.append({"event": "advance_bot_exception", "step": step, "tb": traceback.format_exc()})
                    break
                continue
            chosen = legal[0]
            action = build_action(chosen)
            try:
                resp = await client.call_tool("submit_action", {"match_id": match_id, "action": action}, timeout=45.0)
            except Exception:
                log.append({"event": "submit_action_exception", "step": step, "tb": traceback.format_exc()})
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
            try:
                r = await client.call_tool("surrender", {"match_id": match_id}, timeout=30.0)
                log.append({"event": "surrender", "data": r})
            except Exception:
                log.append({"event": "surrender_exception", "tb": traceback.format_exc()})

        # get_dataset
        try:
            ds = await client.call_tool("get_dataset", {"group_id": group_id}, timeout=15.0)
            log.append({"event": "get_dataset", "data": ds})
        except Exception:
            log.append({"event": "get_dataset_exception", "tb": traceback.format_exc()})

        # NDJSON summary
        try:
            ds_path = Path(ds["dataset_jsonl"])
            if ds_path.exists():
                rows = [json.loads(line) for line in ds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                summary = {
                    "rows": len(rows),
                    "decision_sources": sorted({r.get("decision_source") for r in rows}),
                    "accepted_values": sorted({str(r.get("accepted")) for r in rows}),
                    "won_values": sorted({str(r.get("won")) for r in rows}),
                    "winner_user_ids": sorted({str(r.get("winner_user_id")) for r in rows}),
                    "is_bot_values": sorted({str(r.get("is_bot")) for r in rows}),
                    "visibility_values": sorted({r.get("visibility") for r in rows}),
                }
                # F01 check
                opp_hidden = 0
                opp_leak = 0
                for r in rows:
                    sj = r.get("state_json", {})
                    p2 = sj.get("p2", {})
                    hand = p2.get("hand", [])
                    if hand and all(isinstance(h, dict) and h.get("hidden") is True for h in hand):
                        opp_hidden += 1
                    # leak: any hand card with actual fields (name/card_id) on opponent
                    if hand and any(isinstance(h, dict) and not h.get("hidden") for h in hand):
                        opp_leak += 1
                summary["opp_hand_hidden_rows"] = opp_hidden
                summary["opp_hand_leak_rows"] = opp_leak
                log.append({"event": "ndjson_summary", "data": summary})
        except Exception:
            log.append({"event": "ndjson_read_exception", "tb": traceback.format_exc()})

        if client.stderr_lines:
            log.append({"event": "stderr_lines", "lines": client.stderr_lines})

    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
        await client.aclose()

    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


async def _wait_initial(client: MCPClient):
    while client.initial is None:
        await asyncio.sleep(0.05)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))