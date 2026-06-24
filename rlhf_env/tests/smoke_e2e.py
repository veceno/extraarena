#!/usr/bin/env python3
"""RLHF-среда — smoke E2E тест.

Запускает локальный web-сервер (на alt-порту), создаёт группу боёв через API,
поллит статус до завершения, проверяет manifest + battle_log.

Использование:
    python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 2 --models v4-max
    python3 rlhf_env/tests/smoke_e2e.py --port 8096 --battles 1 --models random --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _http_get(url: str, timeout: float = 5.0) -> dict:
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_post(url: str, payload: dict, timeout: float = 5.0) -> dict:
    req = Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _http_get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            return True
        except (URLError, Exception):
            time.sleep(0.3)
    return False


def wait_for_group_done(port: int, gid: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = _http_get(f"http://127.0.0.1:{port}/api/groups/{gid}", timeout=2.0)
        except URLError:
            time.sleep(0.3)
            continue
        if s.get("status") in ("completed", "error", "loaded"):
            return s
        time.sleep(0.3)
    raise TimeoutError(f"group {gid} did not complete in {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="RLHF smoke E2E")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--battles", type=int, default=2)
    parser.add_argument("--models", type=str, default="v4-max",
                        help="p1 model name: v4-max | v4-lite | v3-max | random")
    parser.add_argument("--p2", type=str, default="end_turn")
    parser.add_argument("--sessions-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Перевод имен в реальные модели
    model_map = {
        "v4-max": "extra-lr-v4-max",
        "v4-lite": "extra-lr-v4-lite",
        "v4-micro": "extra-lr-v4-micro",
        "v4-opti": "extra-lr-v4-opti",
        "v3-max": "extra-lr-v3-max",
        "v3-medium": "extra-lr-v3-medium",
        "random": "random",
    }
    p1_model = model_map.get(args.models, args.models)

    # Подготовка sessions_dir
    if args.sessions_dir:
        sessions_dir = Path(args.sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
    else:
        sessions_dir = Path(tempfile.mkdtemp(prefix="rlhf_smoke_"))

    if not args.quiet:
        print(f"[smoke] sessions_dir: {sessions_dir}")

    env = os.environ.copy()
    env["RLHF_PORT"] = str(args.port)
    env["RLHF_SESSIONS_DIR"] = str(sessions_dir)

    # Запускаем web-сервер в фоне
    log_file = sessions_dir / "server.log"
    log_handle = open(log_file, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rlhf_env.server",
         "--port", str(args.port),
         "--sessions-dir", str(sessions_dir)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        if not wait_for_server(args.port, timeout=30):
            print(f"[smoke] FAIL: server did not start within 30s. See {log_file}")
            return 1
        if not args.quiet:
            print(f"[smoke] server up @ :{args.port}")

        # 1. Health
        h = _http_get(f"http://127.0.0.1:{args.port}/health")
        assert h["status"] == "ok", f"health failed: {h}"
        if not args.quiet:
            print(f"[smoke] health OK: models_loaded={h['models_loaded']}")

        # 2. List models
        models = _http_get(f"http://127.0.0.1:{args.port}/api/registry/models")
        model_names = [m["name"] for m in models["models"]]
        if not args.quiet:
            print(f"[smoke] models in registry: {len(model_names)}")

        # 3. Sample deck
        sample = _http_get(f"http://127.0.0.1:{args.port}/api/registry/sample-deck")
        assert isinstance(sample["deck"], list) and len(sample["deck"]) >= 6
        if not args.quiet:
            print(f"[smoke] sample deck: {sample['deck']}")

        # 4. Start group
        spec = {
            "p1_model": p1_model,
            "p2_model": args.p2,
            "deck_strategy": "random_arenaenv",
            "battles_planned": args.battles,
            "seed": 42,
            "starting_player": "random",
            "max_turns": 40,
        }
        start_resp = _http_post(f"http://127.0.0.1:{args.port}/api/groups", spec)
        gid = start_resp["group_id"]
        if not args.quiet:
            print(f"[smoke] started group: {gid}")

        # 5. Wait for completion
        s = wait_for_group_done(args.port, gid, timeout=120.0)
        assert s["status"] in ("completed", "loaded"), f"unexpected status: {s}"
        if not args.quiet:
            print(f"[smoke] group done: status={s['status']}, "
                  f"battles={s['battles_finished']}, winrate_p1={s.get('winrate_p1')}")

        # 6. Manifest
        manifest = _http_get(f"http://127.0.0.1:{args.port}/api/groups/{gid}/manifest")
        assert manifest["results"]["battles_finished"] == args.battles
        assert "env" in manifest
        assert "rlhf_env_version" in manifest["env"]
        if not args.quiet:
            print(f"[smoke] manifest OK: results={manifest['results']}")

        # 7. Battle log
        battles = _http_get(f"http://127.0.0.1:{args.port}/api/groups/{gid}/battles")
        assert len(battles["battle_ids"]) == args.battles
        bid = battles["battle_ids"][0]
        blog = _http_get(f"http://127.0.0.1:{args.port}/api/groups/{gid}/battles/{bid}")
        assert blog["log_version"] == "1.0"
        assert blog["battle_id"] == bid
        assert len(blog["actions"]) >= 1
        if not args.quiet:
            print(f"[smoke] battle[{bid}] OK: status={blog['result']['status']}, "
                  f"actions={len(blog['actions'])}")

        # 8. Files on disk
        group_dir = sessions_dir / gid
        assert (group_dir / "manifest.json").exists()
        assert (group_dir / "summary.json").exists()
        assert len(list((group_dir / "battles").glob("*.json"))) == args.battles
        if not args.quiet:
            print(f"[smoke] files on disk OK: {group_dir}")

        # 9. List groups
        lst = _http_get(f"http://127.0.0.1:{args.port}/api/groups")
        assert any(g["group_id"] == gid for g in lst["groups"])
        if not args.quiet:
            print(f"[smoke] list groups OK: total={len(lst['groups'])}")

        # 10. Stop неактивной группы → должен вернуть ошибку
        stop = _http_post(f"http://127.0.0.1:{args.port}/api/groups/{gid}/stop", {})
        # OK если stopped=False (уже completed) или True — это нормально
        if not args.quiet:
            print(f"[smoke] stop response: {stop}")

        print(f"\n[smoke] ✅ ALL CHECKS PASSED")
        print(f"[smoke] group_id: {gid}")
        print(f"[smoke] files: {group_dir}")
        return 0

    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_handle.close()


if __name__ == "__main__":
    sys.exit(main())