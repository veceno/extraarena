"""aiohttp-сервер RLHF-среды ExtraArena (арена 1:1 с игрой).

Запуск:
    python -m rlhf_env.server                  # 127.0.0.1:8090
    python -m rlhf_env.server --port 9000
    ./rlhf_env/start_rlhf_env.sh                # лаунчер (venv + deps)

Арена 1:1: отдаём верbatim arena.html/arena.js/safe-area.js/arena-styles.css
из rlhf_env/webapp_borrow/ и реплицируем сервервер-контракт игры (Socket.IO +
HTTP /api/battle/*, /api/match/find) через ArenaTransport, чтобы реальный
arena.js работал БЕЗ правок. Боевку ведёт RlhfBattleEngine (шим над core.engine),
бота — match_runner через policy_factory. Сами файлы игры не трогаем.

API:
    GET  /                        → index.html (форма «Новая серия»)
    GET  /arena?id=...&_auth=...&ea_platform=android_app → arena.html (1:1)
    GET  /groups, /groups/{gid}    → HTML-список/статус групп
    GET  /api/registry/models     → реестр моделей
    GET  /api/registry/deck-strategies
    GET  /api/registry/sample-deck
    GET  /api/cards               → каталог карт
    GET  /api/groups, /api/groups/{gid}, .../manifest, .../battles, .../battles/{bid}
    POST /api/groups              → алиас создания серии (см. /api/match/find)
    POST /api/match/find          → старт/продолжение серии → редирект на /arena
    POST /api/battle/{state,play-card,attack,end-turn,preview}
    POST /api/matches/{id}/surrender
    GET  /api/runtime/status        → health-monitor арены (200, нет maintenance)
    GET  /api/settings               → пользовательские настройки (200, дефолты)
    GET  /health

Socket.IO (default namespace /socket.io/):
    join_match / client_ready / surrender / state_changed / match_ready / game_over
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict

import aiohttp
from aiohttp import web, ClientTimeout

# корень репо в sys.path для import core.*
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.arena_io import ArenaTransport, make_fake_jwt, _audio_query  # noqa: E402
from rlhf_env.components.arena_match_manager import ArenaMatchManager  # noqa: E402
from rlhf_env.components.deck_builder import (  # noqa: E402
    build_random_arena_deck,
    deck_summary,
    load_catalog,
)
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_MODELS_DIR = "ai/models"
DEFAULT_SESSIONS_DIR = "rlhf_env/sessions"
DEFAULT_CARDS_PATH = "ai/cards.json"
DEFAULT_AUTH_SEED = "rlhf"
# База прода (web/server.py, порт 8081) для /api/rlhf/* — rlhf работает тонким
# прокси: БД/Telegram/JWT остаются в игре, браузер видит только same-origin rlhf
# и cookie rlhf_sid (никогда не получает прод-JWT).
DEFAULT_PROD_BASE_URL = os.environ.get("RLHF_PROD_BASE_URL", "http://127.0.0.1:8081")
RLHF_SID_COOKIE = "rlhf_sid"

# Файл лёгкого переключения среды подключения (rlhf_env/config.json):
#   { "environment": "<имя>", "environments": { "<имя>": {"base_url","label"} } }
# Поменяй `environment`, чтобы rlhf-прокси пошёл в другую копию игры за /api/rlhf/*.
RLHF_CONFIG_PATH = _HERE / "config.json"

# Директория с verbatim-копиями файлов арены.
WEBAPP_BORROW = _HERE / "webapp_borrow"


def load_rlhf_config() -> Dict[str, Any]:
    """Читает rlhf_env/config.json. Возвращает {} при отсутствии/ошибке (молчит)."""
    import json
    if not RLHF_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(RLHF_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 — конфиг опционален, не роняем запуск
        logger.warning("RLHF config.json unreadable (%s); using defaults", exc)
        return {}


def resolve_target_environment(
    *,
    cli_base_url: str | None,
    cli_env: str | None,
    env_base_url: str | None,
    env_env: str | None,
    config: Dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Разрешает, куда rlhf-прокси стучится за /api/rlhf/*.

    Возвращает (base_url, env_name, label). Прецедентность (старшее — выше):
      1) явный URL: --prod-base-url / RLHF_PROD_BASE_URL
      2) имя среды: --env / RLHF_ENV
      3) config.json: поле `environment` → environments[<имя>].base_url
      4) DEFAULT_PROD_BASE_URL (http://127.0.0.1:8081)
    """
    cfg = config if config is not None else load_rlhf_config()
    environments = cfg.get("environments", {}) if isinstance(cfg, dict) else {}

    def _lookup(name: str) -> tuple[str, str] | None:
        entry = environments.get(name)
        if not isinstance(entry, dict) or not entry.get("base_url"):
            return None
        return str(entry["base_url"]).rstrip("/"), str(entry.get("label") or name)

    # 1) явный URL — высший приоритет, имя среды определяем по совпадению, иначе "custom"
    if cli_base_url:
        normalized = cli_base_url.rstrip("/")
        for name, entry in environments.items():
            if isinstance(entry, dict) and str(entry.get("base_url", "")).rstrip("/") == normalized:
                return normalized, name, str(entry.get("label") or name)
        return normalized, "custom", normalized

    if env_base_url:
        normalized = env_base_url.rstrip("/")
        for name, entry in environments.items():
            if isinstance(entry, dict) and str(entry.get("base_url", "")).rstrip("/") == normalized:
                return normalized, name, str(entry.get("label") or name)
        return normalized, "custom", normalized

    # 2) имя среды из CLI/env
    for name in (cli_env, env_env):
        if name:
            hit = _lookup(name)
            if hit:
                return hit[0], name, hit[1]
            logger.warning("RLHF env '%s' not in config.json environments; falling back", name)

    # 3) config.json: поле `environment`
    cfg_env = cfg.get("environment") if isinstance(cfg, dict) else None
    if cfg_env:
        hit = _lookup(str(cfg_env))
        if hit:
            return hit[0], str(cfg_env), hit[1]
        logger.warning("RLHF config.json environment='%s' not found in `environments`", cfg_env)

    # 4) дефолт
    return DEFAULT_PROD_BASE_URL.rstrip("/"), "prod", "Прод (8081)"


class WebApp:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        models_dir: str = DEFAULT_MODELS_DIR,
        sessions_dir: str = DEFAULT_SESSIONS_DIR,
        cards_path: str = DEFAULT_CARDS_PATH,
        auth_seed: str = DEFAULT_AUTH_SEED,
        prod_base_url: str = DEFAULT_PROD_BASE_URL,
        rlhf_env_name: str = "prod",
        rlhf_env_label: str = "Прод (8081)",
    ) -> None:
        self.host = host
        self.port = port
        self.models_dir = Path(models_dir)
        self.sessions_dir = Path(sessions_dir)
        self.cards_path = Path(cards_path)
        self.prod_base_url = (prod_base_url or DEFAULT_PROD_BASE_URL).rstrip("/")
        self.rlhf_env_name = rlhf_env_name
        self.rlhf_env_label = rlhf_env_label

        self.catalog = load_catalog(cards_path)
        self.registry = PolicyRegistry.scan(models_dir)
        self.manager = ArenaMatchManager(
            sessions_dir=sessions_dir,
            models_dir=models_dir,
            catalog=self.catalog,
            registry=self.registry,
            cards_path=cards_path,
        )
        self.auth_token = make_fake_jwt(auth_seed)

        # HTTP-клиент к проду (создаётся в event-loop на startup) + серверный
        # session-store: cookie rlhf_sid -> {jwt, user_id, decks, max_decks,
        # extra_pass_active}. Браузер НИКОГДА не видит прод-JWT.
        self._http: aiohttp.ClientSession | None = None
        self._rlhf_sessions: Dict[str, Dict[str, Any]] = {}

        self.app = web.Application()
        self.app["webapp"] = self
        self.transport = ArenaTransport(self.manager, auth_token=self.auth_token)
        self.transport.attach(self.app)
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)
        self._setup_routes()

    async def _on_startup(self, _app: web.Application) -> None:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=ClientTimeout(total=10))

    async def _on_cleanup(self, _app: web.Application) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()

    # ------------------------------------------------------------------
    def _setup_routes(self) -> None:
        r = self.app.router
        # HTML
        r.add_get("/", self.handle_index)
        r.add_get("/arena", self.handle_arena_page)
        r.add_get("/groups", self.handle_groups_page)
        r.add_get("/groups/{gid}", self.handle_group_page)
        # Статика арены (verbatim из webapp_borrow/)
        r.add_get("/arena.js", self.handle_arena_js)
        r.add_get("/safe-area.js", self.handle_safe_area_js)
        r.add_get("/arena-styles.css", self.handle_arena_css)
        # Локальная статика RLHF-формы
        r.add_static("/static/", path=str(_HERE / "static"), show_index=False)
        # Ассеты карт/персонажей из репо (не меняет игру — только отдаёт существующие
        # бинарники, чтобы arena.js рендерил картинки карт вместо 404).
        design_assets = _REPO_ROOT / "DesignAssets"
        if design_assets.exists():
            r.add_static("/DesignAssets/", path=str(design_assets), show_index=False)
        assets_audio = _REPO_ROOT / "assets" / "audio"
        if assets_audio.exists():
            r.add_static("/assets/audio/", path=str(assets_audio), show_index=False)
        # API реестра/каталога
        r.add_get("/api/registry/models", self.api_list_models)
        r.add_get("/api/registry/deck-strategies", self.api_deck_strategies)
        r.add_get("/api/registry/sample-deck", self.api_sample_deck)
        r.add_get("/api/cards", self.api_cards)
        # API групп/манифестов
        r.add_get("/api/groups", self.api_list_groups)
        r.add_post("/api/groups", self.api_start_group)
        r.add_get("/api/groups/{gid}", self.api_group_status)
        r.add_get("/api/groups/{gid}/manifest", self.api_group_manifest)
        r.add_get("/api/groups/{gid}/battles", self.api_group_battles)
        r.add_get("/api/groups/{gid}/battles/{bid}", self.api_battle_log)
        # Досрочно закрыть серию («Завершить» в арене) — финализирует манифест.
        r.add_post("/api/groups/{gid}/finish", self.api_group_finish)
        r.add_get("/health", self.api_health)
        # HTTP-контракт арены, который arena.js фетчит напрямую (health-monitor +
        # user settings). Auth игнорируется (как в prod: require_user_id под
        # try/except → user_id=None), поэтому _auth в query просто не читается.
        # /api/runtime/status — КРИТИЧЕН: health-monitor (arena.js:3141) при 404
        # вызывает markArenaConnectionFailure() → модалка «Соединение разорвано».
        r.add_get("/api/runtime/status", self.api_runtime_status)
        r.add_get("/api/settings", self.api_user_settings)
        # RLHF-логин + импорт колод (тонкий прокси на прод /api/rlhf/*).
        # БД/Telegram/JWT — в игре; rlhf хранит только cookie-сессию rlhf_sid.
        r.add_post("/api/rlhf/request-code", self.rlhf_proxy_request_code)
        r.add_post("/api/rlhf/verify", self.rlhf_proxy_verify)
        r.add_get("/api/rlhf/decks", self.rlhf_proxy_decks)
        r.add_get("/api/rlhf/me", self.rlhf_proxy_me)
        r.add_post("/api/rlhf/logout", self.rlhf_proxy_logout)
        # /api/match/find и /api/battle/* / /api/matches/{id}/surrender
        # регистрируются в ArenaTransport.attach() — там же Socket.IO.

    # ------------------------------------------------------------------
    # HTML / static
    # ------------------------------------------------------------------
    async def handle_index(self, _request: web.Request) -> web.Response:
        html = (_HERE / "index.html").read_text(encoding="utf-8")
        badge = (
            f'Среда подключения: <span class="env-tag">{self.rlhf_env_label}</span>'
            f' &middot; {self.prod_base_url}'
        )
        html = html.replace("__RLHF_ENV_BADGE__", badge)
        return web.Response(text=html, content_type="text/html")

    async def handle_arena_page(self, request: web.Request) -> web.Response:
        # arena.js требует ?id=...&_auth=...&ea_platform=android_app.
        # Если запрос пришёл без ea_platform — добавим, чтобы избежать abort'а.
        match_id = request.rel_url.query.get("id") or request.rel_url.query.get("match_id")
        if not match_id:
            return web.Response(text="match_id required", status=400)
        html_path = WEBAPP_BORROW / "arena.html"
        if not html_path.exists():
            return web.Response(text="arena.html not found in webapp_borrow/", status=500)
        html = html_path.read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")

    async def handle_arena_js(self, request: web.Request) -> web.Response:
        return await self._serve_borrowed("arena.js", "application/javascript",
                                           query=request.rel_url.query)

    async def handle_safe_area_js(self, _request: web.Request) -> web.Response:
        return await self._serve_borrowed("safe-area.js", "application/javascript")

    async def handle_arena_css(self, _request: web.Request) -> web.Response:
        return await self._serve_borrowed("arena-styles.css", "text/css")

    async def _serve_borrowed(self, name: str, ctype: str, query=None) -> web.Response:
        path = WEBAPP_BORROW / name
        if not path.exists():
            return web.Response(text=f"{name} not found", status=404)
        body = path.read_bytes()
        headers = {"Cache-Control": "no-cache"}
        return web.Response(body=body, content_type=ctype, headers=headers)

    async def handle_groups_page(self, _request: web.Request) -> web.Response:
        groups = self.manager.list_groups()
        rows = "".join(
            f"<tr><td><a href='/groups/{g['group_id']}'>{g['group_id']}</a></td>"
            f"<td>{g['status']}</td><td>{g.get('battles_finished',0)}/{g['battles_planned']}</td>"
            f"<td>{g.get('current_battle',0)}</td></tr>"
            for g in groups
        ) or "<tr><td colspan='4'>нет групп</td></tr>"
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>RLHF-Groups</title>"
            "<link rel='stylesheet' href='/static/rlhf.css'>"
            "</head><body><div class='rlhf-container'>"
            "<h1>Battle Groups</h1><p><a href='/'>← Назад</a></p>"
            "<table class='rlhf-table'><thead><tr>"
            "<th>ID</th><th>Status</th><th>Battles</th><th>Current</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            "</div></body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    async def handle_group_page(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        manifest = self._read_manifest(gid)
        if manifest is None:
            return web.Response(text="Group not found", status=404)
        spec = manifest.get("spec", {})
        results = manifest.get("results", {})
        body = (
            f"<h1>Group {gid}</h1>"
            f"<p>Status: {('completed' if manifest.get('finished_at') else 'running')}</p>"
            f"<h2>Spec</h2><pre>{_json_pretty(spec)}</pre>"
            f"<h2>Results</h2><pre>{_json_pretty(results)}</pre>"
            f"<h2>Battles</h2><ul>"
            + "".join(f"<li><a href='/api/groups/{gid}/battles/{bid}'>{bid}</a></li>"
                      for bid in manifest.get("battle_ids", []))
            + "</ul><p><a href='/'>← Назад</a></p>"
        )
        return web.Response(text=f"<!doctype html><html><body>{body}</body></html>",
                            content_type="text/html")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def _read_manifest(self, gid: str) -> Dict[str, Any] | None:
        path = self.manager.sessions_dir / gid / "manifest.json"
        if not path.exists():
            return None
        import json
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    async def api_health(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "rlhf_env_version": __version__,
            "models_loaded": len(self.registry.specs),
            "sessions_dir": str(self.manager.sessions_dir),
            "arena_1to1": (WEBAPP_BORROW / "arena.js").exists(),
            "rlhf_environment": {
                "name": self.rlhf_env_name,
                "label": self.rlhf_env_label,
                "prod_base_url": self.prod_base_url,
            },
        })

    async def api_runtime_status(self, _request: web.Request) -> web.Response:
        """Контракт /api/runtime/status (prod web/server.py:6221).

        Health-monitor arena.js (arena.js:3141) фетчит этот эндпоинт каждые
        ARENA_HEALTH_PING_INTERVAL_MS=15000мс. При response.ok==true вызывается
        clearArenaConnectionIssue(), а arenaMaintenanceBlocks(data) решает,
        показывать ли модалку «Технические работы»:
            !!(data && data.maintenance_mode && data.maintenance_mode.enabled)
            && (!data.is_admin || isArenaAndroidShell())
        Возвращаем maintenance_mode.enabled=false, is_admin=false → блок=false
        → модалка не показывается. _auth не валидируется (как в prod: HTTPException
        ловится → user_id=None → 200 безусловно).
        """
        return web.json_response({
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},  # пусто = все фичи доступны (prod RUNTIME_FEATURE_DEFAULTS)
            "disabled_card_ids": [],
            "is_admin": False,  # RLHF-пользователь не админ
        })

    async def api_user_settings(self, _request: web.Request) -> web.Response:
        """Контракт /api/settings (prod web/server.py:6479).

        arena.js loadTalkieStartupSettings (arena.js:1508) — GRACEFUL
        (`if (!response.ok) return;`), поэтому 404 не вызывает disconnect, но
        устраняет 404-шум в консоли и включает звук/talkie-настройки по умолчанию.
        Минимальный набор, который читают applyArenaSoundSettingsFromUserSettings
        и applyTalkieDisableByDefault. POST здесь не нужен — arena.js шлёт только GET.
        """
        return web.json_response({
            "sound_music": True,
            "sound_sfx": True,
            "social_disable_talkies": False,
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

    # ------------------------------------------------------------------
    # RLHF: тонкий прокси на прод /api/rlhf/* + серверный session-store
    # ------------------------------------------------------------------
    async def _prod_request(
        self, method: str, path: str, *, json_body: Any | None = None, jwt: str | None = None,
    ) -> tuple[int, Any]:
        """Форвард на прод. Возвращает (status, parsed_json|text)."""
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=ClientTimeout(total=10))
        url = f"{self.prod_base_url}{path}"
        headers: Dict[str, str] = {}
        if jwt:
            headers["Authorization"] = f"Bearer {jwt}"
        try:
            async with self._http.request(method, url, json=json_body, headers=headers) as resp:
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = await resp.text()
                return resp.status, body
        except Exception as exc:
            logger.warning("[rlhf-proxy] prod %s %s failed: %s", method, path, exc)
            return 502, {"error": "prod_unreachable", "detail": str(exc)}

    def _session_get(self, request: web.Request) -> Dict[str, Any] | None:
        sid = request.cookies.get(RLHF_SID_COOKIE)
        return self._rlhf_sessions.get(sid) if sid else None

    def _session_set_cookie(self, response: web.Response, sid: str) -> web.Response:
        response.set_cookie(RLHF_SID_COOKIE, sid, httponly=True, samesite="Lax", path="/")
        return response

    def _session_clear_cookie(self, response: web.Response) -> web.Response:
        response.del_cookie(RLHF_SID_COOKIE, path="/")
        return response

    @staticmethod
    def _public_session_payload(sess: Dict[str, Any]) -> Dict[str, Any]:
        """То, что безопасно отдать браузеру (без прод-JWT)."""
        return {
            "user_id": sess.get("user_id"),
            "extra_pass_active": sess.get("extra_pass_active", False),
            "max_decks": sess.get("max_decks", 3),
            "decks": sess.get("decks", []),
        }

    async def rlhf_proxy_request_code(self, request: web.Request) -> web.Response:
        """POST /api/rlhf/request-code {identifier} → пасsthrough на прод."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        status, payload = await self._prod_request(
            "POST", "/api/rlhf/request-code", json_body=body
        )
        return web.json_response(payload, status=status)

    async def rlhf_proxy_verify(self, request: web.Request) -> web.Response:
        """POST /api/rlhf/verify {identifier, code} → создаёт rlhf-сессию.

        Получает от прода {token, user_id, decks, max_decks, extra_pass_active},
        mintит rlhf_sid, хранит JWT серверно, отдаёт браузеру колоды БЕЗ token.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        status, payload = await self._prod_request(
            "POST", "/api/rlhf/verify", json_body=body
        )
        if status != 200 or not isinstance(payload, dict) or "token" not in payload:
            return web.json_response(payload, status=status)
        token = payload["token"]
        sess = {
            "jwt": token,
            "user_id": payload.get("user_id"),
            "extra_pass_active": payload.get("extra_pass_active", False),
            "max_decks": payload.get("max_decks", 3),
            "decks": payload.get("decks", []),
        }
        sid = secrets.token_hex(16)
        self._rlhf_sessions[sid] = sess
        resp = web.json_response(self._public_session_payload(sess), status=200)
        return self._session_set_cookie(resp, sid)

    async def rlhf_proxy_decks(self, request: web.Request) -> web.Response:
        """GET /api/rlhf/decks → ре-фетч колод с прода по JWT из сессии."""
        sess = self._session_get(request)
        if sess is None:
            return web.json_response({"error": "authentication_required"}, status=401)
        status, payload = await self._prod_request(
            "GET", "/api/rlhf/decks", jwt=sess.get("jwt")
        )
        if status == 200 and isinstance(payload, dict):
            # обновим кеш колод в сессии
            sess["decks"] = payload.get("decks", sess.get("decks", []))
            sess["extra_pass_active"] = payload.get("extra_pass_active", sess.get("extra_pass_active", False))
            sess["max_decks"] = payload.get("max_decks", sess.get("max_decks", 3))
            return web.json_response(self._public_session_payload(sess), status=200)
        if status == 401:
            # прод-JWT истёк — чистим сессию
            sid = request.cookies.get(RLHF_SID_COOKIE)
            if sid:
                self._rlhf_sessions.pop(sid, None)
            resp = web.json_response({"error": "authentication_required"}, status=401)
            return self._session_clear_cookie(resp)
        return web.json_response(payload, status=status)

    async def rlhf_proxy_me(self, request: web.Request) -> web.Response:
        """GET /api/rlhf/me → состояние сессии для UI (без JWT)."""
        sess = self._session_get(request)
        if sess is None:
            return web.json_response({"authenticated": False}, status=200)
        return web.json_response(
            {"authenticated": True, **self._public_session_payload(sess)}, status=200
        )

    async def rlhf_proxy_logout(self, request: web.Request) -> web.Response:
        """POST /api/rlhf/logout → чистит rlhf-сессию."""
        sid = request.cookies.get(RLHF_SID_COOKIE)
        if sid:
            self._rlhf_sessions.pop(sid, None)
        resp = web.json_response({"ok": True}, status=200)
        return self._session_clear_cookie(resp)

    async def api_list_models(self, _request: web.Request) -> web.Response:
        return web.json_response({"models": self.registry.list_specs()})

    async def api_deck_strategies(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "strategies": [
                {"id": "random_arenaenv", "label": "Случайные ArenaENV колоды",
                 "description": "1 hero + warriors (5-8 × 2) + potions (1-3 × 2)"},
                {"id": "custom", "label": "Загрузить JSON-колоду",
                 "description": "custom_deck_p1 / custom_deck_p2 из spec"},
            ],
        })

    async def api_sample_deck(self, _request: web.Request) -> web.Response:
        import random as rnd
        deck = build_random_arena_deck(self.catalog, rng=rnd.Random())
        return web.json_response({"deck": deck, "summary": deck_summary(deck, self.catalog)})

    async def api_cards(self, _request: web.Request) -> web.Response:
        # Массив карт (id/name/mana_cost/base_attack/base_hp/rarity/card_type) —
        # удобнее для итерации в браузере (rlhf.js + arena.js превью колод).
        return web.json_response({"cards": list(self.catalog.cards.values())})

    async def api_list_groups(self, _request: web.Request) -> web.Response:
        return web.json_response({"groups": self.manager.list_groups()})

    async def _resolve_imported_deck(
        self, request: web.Request, preset_number: int
    ) -> tuple[list[int] | None, str | None]:
        """Серверный резолв импортированной колоды из rlhf-сессии (с ре-фетчем прода).

        Возвращает (card_ids, error). card_ids — flat [int] (slot0 = hero), как в
        `Database.get_user_deck_presets`. Браузер шлёт только preset_number; реальные
        card_ids берутся из сессии/прода и валидируются (is_playable + принадлежность
        пользователю — прод отдаёт только его колоды). Любой браузерный custom_deck_p1
        на imported-пути игнорируется.
        """
        sess = self._session_get(request)
        if sess is None:
            return None, "deck_not_owned_or_unplayable"
        decks = sess.get("decks") or []
        # Если в кеше сессии колод нет / устарели — ре-фетч с прода по JWT.
        if not decks:
            status, payload = await self._prod_request(
                "GET", "/api/rlhf/decks", jwt=sess.get("jwt")
            )
            if status == 200 and isinstance(payload, dict):
                decks = payload.get("decks", []) or []
                sess["decks"] = decks
            elif status == 401:
                sid = request.cookies.get(RLHF_SID_COOKIE)
                if sid:
                    self._rlhf_sessions.pop(sid, None)
                return None, "authentication_required"
            else:
                return None, "deck_not_owned_or_unplayable"
        for d in decks:
            if int(d.get("preset_number", -1)) == preset_number:
                if not d.get("is_playable"):
                    return None, "deck_not_owned_or_unplayable"
                card_ids = d.get("card_ids") or []
                if not isinstance(card_ids, list) or len(card_ids) != 9:
                    return None, "deck_not_owned_or_unplayable"
                return [int(c) for c in card_ids], None
        return None, "deck_not_owned_or_unplayable"

    async def api_start_group(self, request: web.Request) -> web.Response:
        """Алиас создания серии (возвращает group_id + первый match_id)."""
        try:
            spec = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        # P1: резолв импортированной колоды серверно (см. _resolve_imported_deck).
        src = spec.get("p1_deck_source") or {"type": "random"}
        if isinstance(src, dict) and src.get("type") == "imported":
            preset_number = int(src.get("preset_number", -1))
            card_ids, err = await self._resolve_imported_deck(request, preset_number)
            if err is not None:
                return web.json_response({"error": err}, status=400)
            # Импортированная колода втекает в движок через существующий канал custom_deck_p1
            # (parse_custom_deck принимает [int]); браузерный custom_deck_p1 игнорируется.
            spec["custom_deck_p1"] = card_ids
            spec["deck_strategy_p1"] = "custom"
        try:
            match = self.manager.create_series(spec)
        except Exception as exc:
            logger.exception("[server] create_series failed")
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({
            "group_id": match.group_id,
            "match_id": match.engine.match_id,
            "battles_planned": match.battles_planned,
            "redirect_url": f"/arena?id={match.engine.match_id}&_auth={self.auth_token}&ea_platform=android_app{_audio_query(match.spec)}",
        })

    async def api_group_status(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        m = self._read_manifest(gid)
        if m is None:
            return web.json_response({"error": "group not found"}, status=404)
        return web.json_response({"group_id": gid, "manifest": m})

    async def api_group_finish(self, request: web.Request) -> web.Response:
        """POST /api/groups/{gid}/finish — досрочно закрыть серию («Завершить»).

        Тело не нужно (arena.js шлёт sendBeacon без тела). Идемпотентно
        финализирует манифест с уже записанными боями — иначе серия, которую
        человек завершил кнопкой/уходом в меню, висела в статусе «running».
        """
        gid = request.match_info.get("gid")
        if not gid:
            return web.json_response({"error": "gid_required"}, status=400)
        try:
            self.manager.finish_series(gid)
        except KeyError:
            return web.json_response({"error": "group not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            logger.warning("api_group_finish failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"status": "ok", "group_id": gid})

    async def api_group_manifest(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        m = self._read_manifest(gid)
        if m is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(m)

    async def api_group_battles(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        m = self._read_manifest(gid)
        if m is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"battle_ids": m.get("battle_ids", [])})

    async def api_battle_log(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        bid = request.match_info["bid"]
        path = self.manager.sessions_dir / gid / "battles" / f"{bid}.json"
        if not path.exists():
            return web.json_response({"error": "battle not found"}, status=404)
        import json
        return web.json_response(json.loads(path.read_text(encoding="utf-8")))


def _json_pretty(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLHF-среда ExtraArena (arena 1:1)")
    p.add_argument("--host", default=os.environ.get("RLHF_HOST", DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.environ.get("RLHF_PORT", DEFAULT_PORT)))
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", DEFAULT_MODELS_DIR))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", DEFAULT_SESSIONS_DIR))
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", DEFAULT_CARDS_PATH))
    p.add_argument("--auth-seed", default=os.environ.get("RLHF_AUTH_SEED", DEFAULT_AUTH_SEED))
    # --prod-base-url / RLHF_PROD_BASE_URL — явный URL (высший приоритет).
    # None по умолчанию, чтобы отличить «не задан» от заданного значения.
    p.add_argument("--prod-base-url", default=os.environ.get("RLHF_PROD_BASE_URL"))
    # --env / RLHF_ENV — имя среды из config.json (prod/dev/local/...).
    p.add_argument("--env", default=os.environ.get("RLHF_ENV"))
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Разрешаем среду подключения: явный URL > имя среды > config.json > дефолт.
    base_url, env_name, env_label = resolve_target_environment(
        cli_base_url=args.prod_base_url,
        cli_env=args.env,
        env_base_url=os.environ.get("RLHF_PROD_BASE_URL"),
        env_env=os.environ.get("RLHF_ENV"),
    )

    app = WebApp(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        sessions_dir=args.sessions_dir,
        cards_path=args.cards_path,
        auth_seed=args.auth_seed,
        prod_base_url=base_url,
        rlhf_env_name=env_name,
        rlhf_env_label=env_label,
    )

    logger.info("=" * 60)
    logger.info("RLHF-среда ExtraArena v%s (arena 1:1)", __version__)
    logger.info("  http://%s:%d", app.host, app.port)
    logger.info("  models_dir:    %s", app.models_dir)
    logger.info("  sessions_dir:  %s", app.sessions_dir)
    logger.info("  cards:         %s", app.cards_path)
    logger.info("  ONNX models:   %d", len(app.registry.specs))
    logger.info("  webapp_borrow: %s (arena.js=%s)",
                WEBAPP_BORROW, (WEBAPP_BORROW / "arena.js").exists())
    logger.info("  rlhf_environment: %s (%s) -> %s", env_name, env_label, base_url)
    logger.info("=" * 60)

    web.run_app(app.app, host=app.host, port=app.port, print=lambda *a, **k: None)


if __name__ == "__main__":
    main()