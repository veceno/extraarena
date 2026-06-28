"""ExtraOrchestra aiohttp-сервер (порт 8095).

Раздаёт:
  - editor (форм-редактор Phase 1) и player (предпросмотр арены) из ``static/``;
  - borrowed-арену из ``webapp_borrow/`` (arena.html/js/css + safe-area.js);
  - ``/DesignAssets/`` и ``/assets/audio/`` из репо (картинки/звуки карт);
  - контракт-стабы, которые arena.js фетчит напрямую (``/api/runtime/status``,
    ``/api/settings``, ``/api/cards``) — КРИТИЧНО: 404 на runtime/status →
    модалка «Соединение разорвано» (arena.js:3141);
  - Orchestra API: каталог карт/косметики, сценарии, validate, compute-frames,
    record-задачи.

Запуск: ``python3 -m extra_orchestra.server --host 127.0.0.1 --port 8095``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE
REPO_ROOT = HERE.parent
WEBAPP_BORROW = HERE / "webapp_borrow"
STATIC_DIR = HERE / "static"
SCENARIOS_DIR = HERE / "scenarios"
RECORDINGS_DIR = HERE / "recordings"


def _safe_name_slug(raw: str) -> str:
    """Санитизировать имя сценария для использования в имени файла записи.

    Запрещает path-traversal (``../`` / абсолютные пути / ``..``-компоненты):
    всё кроме букв, цифр, дефиса и подчёркивания схлопывается в ``_``, ведущие/
    конечные ``_`` срезаются. Точки НЕ разрешены (расширение ``.{ext}``
    добавляется отдельно) → ``..`` физически не может попасть в slug. Пустой
    результат → ``orchestra``. Двойная защита — ``relative_to(RECORDINGS_DIR)``
    в ``orch_record``.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "")).strip("_")
    return slug or "orchestra"

# ``core`` / ``infrastructure`` лежат в корне репо — добавим в sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra_orchestra.components.cards_catalog import CardsCatalog, list_cosmetics  # noqa: E402
from extra_orchestra.components.scenario_store import ScenarioStore  # noqa: E402
from extra_orchestra.components.scenario_graph_runner import (  # noqa: E402
    run_scenario_dispatch as run_scenario,
    validate_scenario_dispatch as validate_scenario,
    migrate_v1_to_v2,
)
from extra_orchestra.components.arena_io import make_fake_jwt, audio_query  # noqa: E402

logger = logging.getLogger("extra_orchestra.server")


class OrchestraServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.catalog = CardsCatalog()
        self.store = ScenarioStore(SCENARIOS_DIR)
        self.app = web.Application()
        self.app["orchestra"] = self
        # in-memory хранилища прогонов и записей
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._setup_routes()

    # ------------------------------------------------------------------
    def _setup_routes(self) -> None:
        r = self.app.router
        # HTML
        r.add_get("/", self.handle_editor)
        r.add_get("/editor", self.handle_editor)
        r.add_get("/player", self.handle_player)
        r.add_get("/preview", self.handle_preview)
        r.add_get("/arena", self.handle_arena_page)
        # borrowed arena assets
        r.add_get("/arena.js", self.handle_arena_js)
        r.add_get("/safe-area.js", self.handle_safe_area_js)
        r.add_get("/arena-styles.css", self.handle_arena_css)
        # local static
        if STATIC_DIR.exists():
            r.add_static("/static/", path=str(STATIC_DIR), show_index=False)
        # repo assets
        design_assets = REPO_ROOT / "DesignAssets"
        if design_assets.exists():
            r.add_static("/DesignAssets/", path=str(design_assets), show_index=False)
        assets_audio = REPO_ROOT / "assets" / "audio"
        if assets_audio.exists():
            r.add_static("/assets/audio/", path=str(assets_audio), show_index=False)
        # contract stubs (arena.js fetches directly)
        r.add_get("/api/runtime/status", self.api_runtime_status)
        r.add_get("/api/settings", self.api_settings)
        r.add_get("/api/cards", self.api_cards)
        r.add_get("/api/battle/state", self.api_battle_state)
        r.add_get("/health", self.api_health)
        # Orchestra API
        r.add_get("/api/orchestra/cards", self.orch_cards)
        r.add_get("/api/orchestra/cosmetics", self.orch_cosmetics)
        r.add_get("/api/orchestra/scenarios", self.orch_list_scenarios)
        r.add_get("/api/orchestra/scenarios/{name}", self.orch_get_scenario)
        r.add_post("/api/orchestra/scenarios", self.orch_save_scenario)
        r.add_delete("/api/orchestra/scenarios/{name}", self.orch_delete_scenario)
        r.add_post("/api/orchestra/validate", self.orch_validate)
        r.add_post("/api/orchestra/compute-frames", self.orch_compute_frames)
        r.add_post("/api/orchestra/migrate-v1", self.orch_migrate_v1)
        r.add_get("/api/orchestra/frames/{run_id}", self.orch_get_frames)
        r.add_get("/api/orchestra/frames/{run_id}/manifest", self.orch_frames_manifest)
        r.add_post("/api/orchestra/record", self.orch_record)
        r.add_get("/api/orchestra/record/{job_id}", self.orch_record_status)
        r.add_get("/api/orchestra/record/{job_id}/download", self.orch_record_download)

    # ------------------------------------------------------------------
    # HTML / borrowed
    # ------------------------------------------------------------------
    async def handle_editor(self, _req: web.Request) -> web.Response:
        return await self._serve_static("editor.html", "text/html")

    async def handle_player(self, _req: web.Request) -> web.Response:
        # arena.html (borrowed) + инъекция bridge-скрипта перед </body>.
        # Bridge — отдельный classic-script: разделяет с arena.js глобальное
        # lexical-окружение (top-level `let` userId/prebattle* и function
        # handleStateChanged доступны по bare-имени). /player грузится внутри
        # <iframe> страницы /preview → мобильный viewport (см. handle_preview).
        path = WEBAPP_BORROW / "arena.html"
        if not path.exists():
            return web.Response(text="arena.html not found in webapp_borrow/", status=500)
        html = path.read_text(encoding="utf-8")
        inject = (
            '<script src="/static/orchestra-bridge.js"></script>'
        )
        if "</body>" in html:
            html = html.replace("</body>", inject + "</body>", 1)
        else:
            html = html + inject
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    async def handle_preview(self, _req: web.Request) -> web.Response:
        # Обёртка-«телефон»: <iframe src="/player?..."> фиксированного мобильного
        # портретного размера. У iframe свой viewport → @media (max-width:420px)
        # из arena-styles.css срабатывает → мобильный лейаут арены даже в широком
        # десктоп-окне (область предпросмотра искусственно сужена до моб. аспекта).
        return await self._serve_static("preview.html", "text/html")

    async def handle_arena_page(self, req: web.Request) -> web.Response:
        path = WEBAPP_BORROW / "arena.html"
        if not path.exists():
            return web.Response(text="arena.html not found in webapp_borrow/", status=500)
        return web.Response(text=path.read_text(encoding="utf-8"), content_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    async def handle_arena_js(self, _req: web.Request) -> web.Response:
        return await self._serve_borrowed("arena.js", "application/javascript")

    async def handle_safe_area_js(self, _req: web.Request) -> web.Response:
        return await self._serve_borrowed("safe-area.js", "application/javascript")

    async def handle_arena_css(self, _req: web.Request) -> web.Response:
        return await self._serve_borrowed("arena-styles.css", "text/css")

    async def _serve_borrowed(self, name: str, ctype: str) -> web.Response:
        path = WEBAPP_BORROW / name
        if not path.exists():
            return web.Response(text=f"{name} not found", status=404)
        return web.Response(body=path.read_bytes(), content_type=ctype,
                            headers={"Cache-Control": "no-cache"})

    async def _serve_static(self, name: str, ctype: str) -> web.Response:
        path = STATIC_DIR / name
        if not path.exists():
            return web.Response(text=f"{name} not found in static/", status=404)
        return web.Response(text=path.read_text(encoding="utf-8"), content_type=ctype,
                            headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------
    # Contract stubs
    # ------------------------------------------------------------------
    async def api_health(self, _req: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "extra_orchestra"})

    async def api_runtime_status(self, _req: web.Request) -> web.Response:
        # 404 здесь → markArenaConnectionFailure() → модалка «Соединение разорвано».
        return web.json_response({
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
            "is_admin": False,
        })

    async def api_settings(self, _req: web.Request) -> web.Response:
        return web.json_response({
            "sound_music": False,
            "sound_sfx": True,
            "social_disable_talkies": True,
            "ads_enabled": False,
            "notif_cases": False,
            "notif_daily_rewards": False,
            "notif_game_invites": False,
            "notif_friend_requests": False,
            "notif_events": False,
            "notif_news": False,
            "notification_delivery_mode": "app_then_telegram",
            "social_block_friend_requests": False,
            "social_block_friendly_invites_from_friends": False,
            "social_block_friendly_invites_from_non_friends": True,
            "nickname_glow_disabled": False,
            "hide_player_id_public": False,
        })

    async def api_cards(self, _req: web.Request) -> web.Response:
        return web.json_response({"cards": self.catalog.list_cards()})

    async def api_battle_state(self, req: web.Request) -> web.Response:
        # arena.js loadBattleState вызывает /api/battle/state?match_id=<id>.
        # Если <id> — это run_id оркестра, отдаём первый кадр (init-сцена),
        # чтобы arena.js сразу отрисовал начальную расстановку (prebattle
        # уже снят bridge-хуком). Иначе — минимальный стаб.
        mid = req.rel_url.query.get("match_id") or req.rel_url.query.get("id")
        run = self._runs.get(mid) if mid else None
        if run and run.get("frames"):
            return web.json_response(run["frames"][0]["snapshot"])
        return web.json_response({"match_id": mid or "orchestra", "status": "active"})

    # ------------------------------------------------------------------
    # Orchestra API
    # ------------------------------------------------------------------
    async def orch_cards(self, _req: web.Request) -> web.Response:
        return web.json_response({"cards": self.catalog.list_cards()})

    async def orch_cosmetics(self, _req: web.Request) -> web.Response:
        return web.json_response(list_cosmetics(base_dir=REPO_ROOT))

    async def orch_list_scenarios(self, _req: web.Request) -> web.Response:
        return web.json_response({"scenarios": self.store.list()})

    async def orch_get_scenario(self, req: web.Request) -> web.Response:
        name = req.match_info["name"]
        sc = self.store.load(name)
        if sc is None:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(sc)

    async def orch_save_scenario(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "scenario must be object"}, status=400)
        path = self.store.save(body)
        return web.json_response({"ok": True, "file": path.name})

    async def orch_delete_scenario(self, req: web.Request) -> web.Response:
        name = req.match_info["name"]
        ok = self.store.delete(name)
        return web.json_response({"ok": ok})

    async def orch_validate(self, req: web.Request) -> web.Response:
        scenario = await self._read_scenario(req)
        if scenario is None:
            return web.json_response({"error": "scenario required (name or body)"}, status=400)
        try:
            return web.json_response(validate_scenario(scenario, self.catalog))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                      "frame_count": 0, "total_ms": 0}, status=400)

    async def orch_migrate_v1(self, req: web.Request) -> web.Response:
        scenario = await self._read_scenario(req)
        if scenario is None:
            return web.json_response({"error": "scenario required (name or body)"}, status=400)
        try:
            v2 = migrate_v1_to_v2(scenario)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        return web.json_response(v2)

    async def orch_compute_frames(self, req: web.Request) -> web.Response:
        scenario = await self._read_scenario(req)
        if scenario is None:
            return web.json_response({"error": "scenario required (name or body)"}, status=400)
        try:
            result = run_scenario(scenario, self.catalog)
        except Exception as exc:  # noqa: BLE001 — движок может бросить из build_initial_state
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}", "frame_count": 0, "total_ms": 0},
                status=400)
        run_id = uuid.uuid4().hex
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "scenario_name": scenario.get("name"),
                "viewer_uid": result["viewer_uid"],
                "side_uids": result["side_uids"],
                "match_id": result["match_id"],
                "frames": result["frames"],
                "frame_count": result["frame_count"],
                "total_ms": result["total_ms"],
                "error": result["error"],
                "created_at": time.time(),
            }
        return web.json_response({
            "run_id": run_id,
            "frame_count": result["frame_count"],
            "total_ms": result["total_ms"],
            "error": result["error"],
            "viewer_uid": result["viewer_uid"],
            # полный manifest отдаём сразу (bridge может работать без доп. запроса)
            "side_uids": result["side_uids"],
            "match_id": result["match_id"],
            # fake-JWT для ?_auth= в URL предпросмотра (иначе arena.js boot →
            # «Сессия истекла», т.к. authToken пуст → modal line 3360).
            "auth": make_fake_jwt(uid=result["viewer_uid"], seed=run_id),
        })

    async def orch_get_frames(self, req: web.Request) -> web.Response:
        run_id = req.match_info["run_id"]
        run = self._runs.get(run_id)
        if run is None:
            return web.json_response({"error": "unknown_run"}, status=404)
        return web.json_response(run)

    async def orch_frames_manifest(self, req: web.Request) -> web.Response:
        run_id = req.match_info["run_id"]
        run = self._runs.get(run_id)
        if run is None:
            return web.json_response({"error": "unknown_run"}, status=404)
        return web.json_response({
            "run_id": run_id,
            "frame_count": run["frame_count"],
            "total_ms": run["total_ms"],
            "error": run["error"],
            "viewer_uid": run["viewer_uid"],
            "side_uids": run["side_uids"],
            "match_id": run["match_id"],
        })

    async def orch_record(self, req: web.Request) -> web.Response:
        scenario = await self._read_scenario(req)
        if scenario is None:
            return web.json_response({"error": "scenario required (name or body)"}, status=400)
        # прогон
        try:
            result = run_scenario(scenario, self.catalog)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        if result.get("error"):
            return web.json_response({"error": result["error"]}, status=400)
        run_id = uuid.uuid4().hex
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id, "scenario_name": scenario.get("name"),
                "viewer_uid": result["viewer_uid"], "side_uids": result["side_uids"],
                "match_id": result["match_id"], "frames": result["frames"],
                "frame_count": result["frame_count"], "total_ms": result["total_ms"],
                "error": None, "created_at": time.time(),
            }
        job_id = uuid.uuid4().hex
        name_slug = _safe_name_slug(scenario.get("name", "orchestra"))
        fmt = (req.rel_url.query.get("format") or "mp4").strip().lower()
        if fmt not in ("mp4", "gif"):
            fmt = "mp4"
        ext = "gif" if fmt == "gif" else "mp4"
        out_name = f"{name_slug}-{job_id[:8]}.{ext}"
        out_path = (RECORDINGS_DIR / out_name).resolve()
        # двойная защита от path-traversal: имя файла обязано остаться внутри
        # RECORDINGS_DIR даже если _safe_name_slug что-то упустило.
        try:
            out_path.relative_to(RECORDINGS_DIR.resolve())
        except ValueError:
            return web.json_response({"error": "invalid scenario name"}, status=400)
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id, "run_id": run_id, "status": "pending",
                "format": fmt, "mp4_path": None, "out_name": out_name,
                "error": None, "created_at": time.time(),
            }
        # запустим рекордер в фоне (thread + asyncio loop)
        cfg = _record_config()
        thread = threading.Thread(
            target=self._run_record_job,
            args=(job_id, run_id, str(out_path), cfg, result["viewer_uid"], fmt),
            daemon=True,
        )
        thread.start()
        return web.json_response({"job_id": job_id, "run_id": run_id, "status": "pending",
                                  "format": fmt, "mp4_name": out_name, "file_name": out_name})

    def _run_record_job(self, job_id: str, run_id: str, out_path: str,
                        cfg: Dict[str, Any], viewer_uid: int, fmt: str = "mp4") -> None:
        try:
            from extra_orchestra.components.recorder import record_run_to_gif, record_run_to_mp4
            base_url = f"http://{self.host}:{self.port}"
            if fmt == "gif":
                out = record_run_to_gif(self._runs[run_id], out_path, cfg, viewer_uid,
                                        base_url=base_url)
            else:
                out = record_run_to_mp4(self._runs[run_id], out_path, cfg, viewer_uid,
                                        base_url=base_url)
            with self._lock:
                # mp4_path — каноническое поле пути к готовому файлу (mp4 ИЛИ gif),
                # используется orch_record_download. Доп. fields — для статуса.
                self._jobs[job_id].update({"status": "done", "mp4_path": out})
        except Exception as exc:  # noqa: BLE001
            logger.error("record job %s failed: %s", job_id, exc, exc_info=True)
            with self._lock:
                self._jobs[job_id].update({"status": "failed", "error": str(exc)})

    async def orch_record_status(self, req: web.Request) -> web.Response:
        job_id = req.match_info["job_id"]
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "unknown_job"}, status=404)
        fname = job.get("out_name") or (Path(job["mp4_path"]).name if job.get("mp4_path") else None)
        return web.json_response({
            "job_id": job_id, "status": job["status"],
            "format": job.get("format", "mp4"),
            "error": job["error"], "mp4_name": fname, "file_name": fname,
        })

    async def orch_record_download(self, req: web.Request) -> web.Response:
        job_id = req.match_info["job_id"]
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.get("status") != "done" or not job.get("mp4_path"):
            return web.json_response({"error": "not_ready"}, status=404)
        p = Path(job["mp4_path"])
        if not p.exists():
            return web.json_response({"error": "file_missing"}, status=404)
        return web.FileResponse(p, headers={"Content-Disposition": f'attachment; filename="{p.name}"'})

    # ------------------------------------------------------------------
    async def _read_scenario(self, req: web.Request) -> Optional[Dict[str, Any]]:
        """Принять сценарий из JSON-тела (POST) либо ?name= (load from store)."""
        if req.method == "POST":
            try:
                body = await req.json()
            except Exception:
                return None
            if isinstance(body, dict) and "scenario" in body and isinstance(body["scenario"], dict):
                return body["scenario"]
            # целый сценарий (v1 init_scene/turns или v2 graph, даже пустой) —
            # отдаём как есть, НЕ интерпретируя поле name как ссылку на store
            if isinstance(body, dict) and (
                body.get("graph") is not None or body.get("init_scene") is not None
                or body.get("turns") is not None
                or str(body.get("schema", "")).startswith("extra_orchestra.scenario")
            ):
                return body
            # только {name: ...} — загрузить из store
            if isinstance(body, dict) and "name" in body:
                return self.store.load(body["name"])
            if isinstance(body, dict):
                return body
            return None
        name = req.rel_url.query.get("name")
        if name:
            return self.store.load(name)
        return None


def _record_config() -> Dict[str, Any]:
    cfg_path = PKG_ROOT / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    rec = cfg.get("recording", {}) or {}
    return {
        "fps": int(rec.get("fps", 30)),
        # мобильный портретный viewport: ширина ≤420px → срабатывает
        # @media (max-width:420px) из arena-styles.css → мобильный лейаут арены.
        "width": int(rec.get("width", 414)),
        "height": int(rec.get("height", 896)),
        "device_scale_factor": int(rec.get("device_scale_factor", 2)),
        "headless": bool(rec.get("headless", True)),
        "with_audio": bool(rec.get("with_audio", True)),
        # качество mp4: crf ниже → выше качество (10 ≈ визуально lossless);
        # preset slow — лучшее сжатие при том же качестве.
        "crf": int(rec.get("crf", 10)),
        "preset": str(rec.get("preset", "slow")),
        # GIF-экспорт: fps и ширина (0 = native mobile-разрешение). Меньше fps/
        # ширины — меньше файл; palettegen+paletteuse для качества цвета.
        "gif_fps": int(rec.get("gif_fps", 15)),
        "gif_width": int(rec.get("gif_width", 540)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ExtraOrchestra server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    server = OrchestraServer(args.host, args.port)
    logger.info("ExtraOrchestra → http://%s:%d/  (editor)  /player  (arena preview)",
                args.host, args.port)
    web.run_app(server.app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())