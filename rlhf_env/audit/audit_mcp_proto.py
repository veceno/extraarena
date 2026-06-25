"""Проверка JSON-RPC framing: unknown method, notification, content-type."""
from __future__ import annotations
import asyncio, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


async def read_line(proc):
    return await proc.stdout.readline()


async def main():
    env = {**os.environ, "RLHF_LOG_LEVEL": "WARNING", "PYTHONPATH": str(REPO)}
    proc = await asyncio.create_subprocess_exec(
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "-m", "rlhf_env.mcp_server",
        "--models-dir", "ai/models", "--sessions-dir", "rlhf_env/sessions",
        "--cards-path", "ai/cards.json",
        cwd=str(REPO), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out = []
    # ready line
    ready = json.loads(await read_line(proc))
    out.append({"ready_line": ready})

    async def send(req):
        proc.stdin.write((json.dumps(req) + "\n").encode())
        await proc.stdin.drain()
        return json.loads(await read_line(proc))

    # 1) unknown method
    r1 = await send({"jsonrpc": "2.0", "id": 10, "method": "totally/unknown"})
    out.append({"unknown_method_response": r1})

    # 2) notification (no id) — notifications/initialized
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
    await proc.stdin.drain()
    # does server respond? try read with timeout
    try:
        line = await asyncio.wait_for(read_line(proc), timeout=2.0)
        out.append({"notification_got_response": json.loads(line) if line else None})
    except asyncio.TimeoutError:
        out.append({"notification_got_response": None, "note": "no response (correct for notification)"})

    # 3) tools/call unknown tool — should be protocol error per spec
    r3 = await send({"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    out.append({"unknown_tool_response": r3})

    # 4) content type check — call list_models, inspect content item type
    r4 = await send({"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}})
    content = r4.get("result", {}).get("content", [])
    out.append({"list_models_content_types": [c.get("type") for c in content], "raw": r4})

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())