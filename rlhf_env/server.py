"""aiohttp-сервер RLHF-среды ExtraArena.

Запуск:
    python -m rlhf_env.server                  # 127.0.0.1:8090
    python -m rlhf_env.server --port 9000      # кастомный порт
    python -m rlhf_env.server --models-dir /path/to/v5-checkpoints
    ./rlhf_env/start_rlhf_env.sh               # через лаунчер (venv + deps)

API:
    GET  /                                → index.html (форма «Новая группа»)
    GET  /battle                          → battle.html (страница боя)
    GET  /groups                          → HTML-список групп
    GET  /groups/{gid}                    → HTML-статус группы
    GET  /api/registry/models             → JSON реестра моделей
    GET  /api/registry/deck-strategies    → JSON доступных стратегий колод
    GET  /api/registry/sample-deck        → JSON сэмпл случайной колоды
    POST /api/groups                      → старт группы (JSON spec) → {group_id}
    GET  /api/groups                      → список всех групп
    GET  /api/groups/{gid}                → статус группы
    GET  /api/groups/{gid}/manifest       → manifest.json
    GET  /api/groups/{gid}/battles        → список battle_id
    GET  /api/groups/{gid}/battles/{bid}  → battle_log.json
    POST /api/groups/{gid}/stop           → остановить группу
    GET  /sessions/{gid}/...              → алиасы для прямого доступа к файлам

WS:
    WS /ws/groups/{gid}                   → live state_changed / turn_end / game_over
                                            (для interactive human-vs-model)

Все файлы пишутся в SESSIONS_DIR (по умолчанию rlhf_env/sessions/).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aiohttp import WSCloseCode, WSMsgType, web

# Добавляем корень репо в sys.path, чтобы import core.* работал
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.deck_builder import (  # noqa: E402
    DEFAULT_CARDS_PATH,
    build_random_arena_deck,
    deck_summary,
    load_catalog,
)
from rlhf_env.components.manifest import _utc_now_iso  # noqa: E402
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402
from rlhf_env.components.session_manager import SessionManager  # noqa: E402

logger = logging.getLogger(__name__)

# Дефолты
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_MODELS_DIR = "ai/models"
DEFAULT_SESSIONS_DIR = "rlhf_env/sessions"
DEFAULT_CARDS_PATH = "ai/cards.json"


# ============================================================================
# State-shape для WebSocket (RLHF-friendly, упрощённый по сравнению с прод)
# ============================================================================

def _state_to_view(state) -> Dict[str, Any]:
    """GameState → JSON-словарь для WS-клиента."""
    from core.state import GameStatus

    s = state
    p1 = s.p1
    p2 = s.p2
    return {
        "turn_number": s.turn_number,
        "current_turn_owner_id": s.current_turn_owner_id,
        "status": s.status.name if isinstance(s.status, GameStatus) else str(s.status),
        "is_my_turn_p1": s.current_turn_owner_id == p1.user_id,
        "is_over": s.status != GameStatus.ONGOING,
        "p1": _player_to_view(p1),
        "p2": _player_to_view(p2),
    }


def _player_to_view(p) -> Dict[str, Any]:
    return {
        "user_id": p.user_id,
        "is_bot": p.is_bot,
        "hero": _card_to_view(p.hero),
        "hand": [_card_to_view(c) for c in p.hand],
        "board": [_card_to_view(c) for c in p.board],
        "mana": p.mana,
        "max_mana": p.max_mana,
        "deck_count": len(p.deck),
        "graveyard_count": len(p.graveyard),
    }


def _card_to_view(c) -> Dict[str, Any]:
    return {
        "card_id": c.card_id,
        "instance_id": str(c.instance_id),
        "name": c.name,
        "rarity": c.rarity,
        "card_type": c.card_type.value if hasattr(c.card_type, "value") else str(c.card_type),
        "mana_cost": c.mana_cost,
        "attack": c.attack,
        "hp": c.hp,
        "max_hp": c.max_hp,
        "mechanics": list(c.mechanics) if c.mechanics else [],
        "is_ready": bool(c.is_ready),
        "is_frozen": bool(c.is_frozen),
    }


def _legal_actions_to_view(engine, player_id: int) -> List[Dict[str, Any]]:
    """Возвращает список legal actions в viewer-friendly формате."""
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(engine.get_legal_actions(player_id)):
        try:
            d = a.to_dict()
        except Exception:
            d = {"type": "unknown"}
        out.append({"index": i, "action": d})
    return out


# ============================================================================
# WS-эндпоинт для human-vs-model
# ============================================================================

class InteractiveBattle:
    """Один интерактивный бой: ждёт action от человека через WS, бот играет сам."""

    def __init__(
        self,
        *,
        group_id: str,
        engine,
        human_player: int,
        bot_policy: Any,
        battle_id: str,
        manifest,
        registry=None,
    ):
        self.group_id = group_id
        self.engine = engine
        self.human_player = human_player
        self.bot_policy = bot_policy
        self.battle_id = battle_id
        self.manifest = manifest
        self.registry = registry
        self.action_queue: asyncio.Queue = asyncio.Queue()
        self.ws: Optional[web.WebSocketResponse] = None
        self.task: Optional[asyncio.Task] = None
        self.battle_log: Dict[str, Any] = {}
        self.cancelled = False

    async def run(self) -> Dict[str, Any]:
        from core.state import GameStatus
        from rlhf_env.components.battle_runner import BattleRunner

        # bot_policy = "человек играет за p1/p2, бот за второго"
        # В нашей арене: bot — это P2 (opponent).
        p1_pol = _HumanProxyPolicy(self) if self.human_player == 1 else self.bot_policy
        p2_pol = _HumanProxyPolicy(self) if self.human_player == 2 else self.bot_policy

        # Runner не блокирует: используем BattleRunner в интерактивном режиме
        battle_log_path = self.manifest.group_dir / "battles" / f"{self.battle_id}.json"
        runner = BattleRunner(
            group_id=self.group_id,
            battle_id=self.battle_id,
            policy_a=p1_pol,
            policy_b=p2_pol,
            engine=self.engine,
            battle_log_path=battle_log_path,
            human_player=self.human_player,
            max_turns=60,
        )
        self.battle_log = await runner.arun(action_queue=self.action_queue)
        return self.battle_log

    async def push_state(self) -> None:
        if self.ws is None or self.ws.closed:
            return
        s = self._safe_state()
        legal = _legal_actions_to_view(self.engine, self.human_player)
        await self.ws.send_json({
            "type": "state",
            "state": s,
            "legal_actions": legal,
            "your_turn": self.engine.state.current_turn_owner_id == self.human_player,
        })

    def _safe_state(self) -> Dict[str, Any]:
        try:
            return _state_to_view(self.engine.state)
        except Exception as exc:
            logger.warning("[InteractiveBattle] state snapshot failed: %s", exc)
            return {"error": str(exc)}


class _HumanProxyPolicy:
    """Политика-«заглушка»: достаёт action_idx из очереди BattleRunner'а."""

    name = "human"

    def __init__(self, ib: InteractiveBattle):
        self.ib = ib

    def select_action(self, engine, player_id: int) -> int:
        # Не вызывается в интерактивном режиме — BattleRunner берёт action
        # напрямую из action_queue, минуя select_action.
        # Этот метод нужен только для совместимости сигнатуры.
        raise RuntimeError("HumanProxyPolicy.select_action should not be called in interactive mode")


# ============================================================================
# Web-приложение
# ============================================================================

class WebApp:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        models_dir: str = DEFAULT_MODELS_DIR,
        sessions_dir: str = DEFAULT_SESSIONS_DIR,
        cards_path: str = DEFAULT_CARDS_PATH,
    ):
        self.host = host
        self.port = port
        self.models_dir = Path(models_dir)
        self.sessions_dir = Path(sessions_dir)
        self.cards_path = Path(cards_path)

        self.catalog = load_catalog(cards_path)
        self.registry = PolicyRegistry.scan(models_dir)
        self.session_manager = SessionManager(
            sessions_dir=sessions_dir,
            models_dir=models_dir,
            catalog=self.catalog,
            registry=self.registry,
        )

        self.app = web.Application()
        self._setup_routes()

    # ------------------------------------------------------------------
    def _setup_routes(self) -> None:
        r = self.app.router
        # HTML
        r.add_get("/", self.handle_index)
        r.add_get("/battle", self.handle_battle_page)
        r.add_get("/groups", self.handle_groups_page)
        r.add_get("/groups/{gid}", self.handle_group_page)
        # Static (CSS, JS, images)
        r.add_static("/static/", path=str(_HERE / "static"), show_index=False)
        r.add_static("/css/", path=str(_HERE), show_index=False)
        r.add_static("/js/", path=str(_HERE), show_index=False)
        # API
        r.add_get("/api/registry/models", self.api_list_models)
        r.add_get("/api/registry/deck-strategies", self.api_deck_strategies)
        r.add_get("/api/registry/sample-deck", self.api_sample_deck)
        r.add_get("/api/cards", self.api_cards)
        r.add_get("/api/groups", self.api_list_groups)
        r.add_post("/api/groups", self.api_start_group)
        r.add_get("/api/groups/{gid}", self.api_group_status)
        r.add_get("/api/groups/{gid}/manifest", self.api_group_manifest)
        r.add_get("/api/groups/{gid}/battles", self.api_group_battles)
        r.add_get("/api/groups/{gid}/battles/{bid}", self.api_battle_log)
        r.add_post("/api/groups/{gid}/stop", self.api_group_stop)
        r.add_get("/health", self.api_health)
        # WS (интерактив)
        r.add_get("/ws/groups/{gid}/battles/{bid}", self.ws_battle)

    # ------------------------------------------------------------------
    # HTML handlers
    # ------------------------------------------------------------------
    async def handle_index(self, _request: web.Request) -> web.Response:
        html = (_HERE / "index.html").read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")

    async def handle_battle_page(self, _request: web.Request) -> web.Response:
        html = (_HERE / "battle.html").read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")

    async def handle_groups_page(self, _request: web.Request) -> web.Response:
        groups = self.session_manager.list()
        rows = "".join(
            f"<tr><td><a href='/groups/{g['group_id']}'>{g['group_id']}</a></td>"
            f"<td>{g['status']}</td><td>{g['battles_finished']}/{g['battles_planned']}</td>"
            f"<td>{g['winrate_p1']:.2f}</td><td>{g['started_at']}</td></tr>"
            for g in groups
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>RLHF-Groups</title>"
            "<link rel='stylesheet' href='/static/rlhf.css'>"
            "</head><body><div class='rlhf-container'>"
            "<h1>Battle Groups</h1><p><a href='/'>← Назад</a></p>"
            "<table class='rlhf-table'><thead><tr>"
            "<th>ID</th><th>Status</th><th>Battles</th><th>WR(p1)</th><th>Started</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            "</div></body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    async def handle_group_page(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        manifest = self.session_manager.get_manifest(gid)
        if manifest is None:
            return web.Response(text="Group not found", status=404)
        spec = manifest.get("spec", {})
        results = manifest.get("results", {})
        body = (
            f"<h1>Group {gid}</h1>"
            f"<p>Status: {('completed' if manifest.get('finished_at') else 'running')}</p>"
            f"<h2>Spec</h2><pre>{json.dumps(spec, indent=2, ensure_ascii=False)}</pre>"
            f"<h2>Results</h2><pre>{json.dumps(results, indent=2, ensure_ascii=False)}</pre>"
            f"<h2>Battles</h2><ul>"
            + "".join(
                f"<li><a href='/api/groups/{gid}/battles/{bid}'>{bid}</a></li>"
                for bid in manifest.get("battle_ids", [])
            )
            + "</ul><p><a href='/'>← Назад</a></p>"
        )
        return web.Response(text=f"<!doctype html><html><body>{body}</body></html>", content_type="text/html")

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------
    async def api_health(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "rlhf_env_version": __version__,
            "models_loaded": len(self.registry.specs),
            "sessions_dir": str(self.sessions_dir),
        })

    async def api_list_models(self, _request: web.Request) -> web.Response:
        return web.json_response({"models": self.registry.list_specs()})

    async def api_deck_strategies(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "strategies": [
                {
                    "id": "random_arenaenv",
                    "label": "Случайные ArenaENV колоды",
                    "description": "1 hero + случайные warriors (5-8 × 2 копии) + случайные potions (1-3 × 2 копии)",
                },
                {
                    "id": "custom",
                    "label": "Загрузить JSON-колоду",
                    "description": "Использовать custom_deck_p1 / custom_deck_p2 из spec",
                },
            ],
        })

    async def api_sample_deck(self, _request: web.Request) -> web.Response:
        import random as rnd
        deck = build_random_arena_deck(self.catalog, rng=rnd.Random())
        return web.json_response({
            "deck": deck,
            "summary": deck_summary(deck, self.catalog),
        })

    async def api_cards(self, _request: web.Request) -> web.Response:
        return web.json_response({"cards": self.catalog.cards})

    async def api_list_groups(self, _request: web.Request) -> web.Response:
        return web.json_response({"groups": self.session_manager.list()})

    async def api_start_group(self, request: web.Request) -> web.Response:
        try:
            spec = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        try:
            group_id = self.session_manager.start(spec)
        except Exception as exc:
            logger.exception("[server] start_group failed")
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({
            "group_id": group_id,
            "status": "running",
            "manifest_path": f"/api/groups/{group_id}/manifest",
        })

    async def api_group_status(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        s = self.session_manager.status(gid)
        if s is None:
            # может быть на диске
            manifest = self.session_manager.get_manifest(gid)
            if manifest is None:
                return web.json_response({"error": "group not found"}, status=404)
            return web.json_response({"group_id": gid, "status": "loaded", "manifest": manifest})
        return web.json_response(s)

    async def api_group_manifest(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        m = self.session_manager.get_manifest(gid)
        if m is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(m)

    async def api_group_battles(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        m = self.session_manager.get_manifest(gid)
        if m is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"battle_ids": m.get("battle_ids", [])})

    async def api_battle_log(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        bid = request.match_info["bid"]
        b = self.session_manager.battle_log(gid, bid)
        if b is None:
            return web.json_response({"error": "battle not found"}, status=404)
        return web.json_response(b)

    async def api_group_stop(self, request: web.Request) -> web.Response:
        gid = request.match_info["gid"]
        ok = self.session_manager.stop(gid)
        if not ok:
            return web.json_response({"error": "not running or not found"}, status=404)
        return web.json_response({"stopped": True, "group_id": gid})

    # ------------------------------------------------------------------
    # WS handler — интерактивный бой (human vs model)
    # ------------------------------------------------------------------
    async def ws_battle(self, request: web.Request) -> web.WebSocketResponse:
        """WS для live-обновлений состояния боя.

        Протокол:
          Server → Client:
            {"type": "state", "state": {...}, "legal_actions": [...], "your_turn": bool}
            {"type": "result", "battle_log": {...}}
            {"type": "error", "message": "..."}
          Client → Server:
            {"type": "action", "index": int}
            {"type": "ping"}
        """
        gid = request.match_info["gid"]
        bid = request.match_info["bid"]
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        try:
            await self._ws_loop(ws, gid, bid)
        except Exception as exc:
            logger.exception("[ws_battle] loop failed")
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            await ws.close(code=WSCloseCode.GOING_AWAY)
        return ws

    async def _ws_loop(self, ws: web.WebSocketResponse, gid: str, bid: str) -> None:
        """Интерактивный цикл: человек играет за P1, бот — P2.

        Использует уже существующий BattleRunner в режиме arun(action_queue=...).
        """
        manifest = self.session_manager.get_manifest(gid)
        if manifest is None:
            await ws.send_json({"type": "error", "message": "group not found"})
            return

        # Спека для конкретного боя — обычно в spec['interactive_battle']
        spec = manifest.get("spec", {})
        p2_model_name = spec.get("p2_model", "end_turn")
        difficulty = str(spec.get("difficulty", "default"))

        from rlhf_env.components.policy_factory import build_policy
        bot = build_policy({"name": p2_model_name, "difficulty": difficulty})

        # Создаём GameState (random deck, иначе из spec)
        import random as rnd
        from core.engine import ArenaEnvironment
        from rlhf_env.components.deck_builder import build_random_arena_deck
        from rlhf_env.components.session_manager import _build_game_state

        rng = rnd.Random(int(spec.get("seed", 0)) + int(bid[-4:], 16))
        catalog = self.catalog
        p1_ids = build_random_arena_deck(catalog, rng=rng)
        p2_ids = build_random_arena_deck(catalog, rng=rng)
        gs = _build_game_state(p1_ids, p2_ids, catalog, starting_player="random", rng=rng)
        engine = ArenaEnvironment(gs)

        from rlhf_env.components.battle_runner import BattleRunner

        battle_log_path = self.session_manager.sessions_dir / gid / "battles" / f"{bid}.json"

        # human = p1 (1000)
        # bot = p2 (2000)
        # HumanProxyPolicy затыкает select_action для P1
        from rlhf_env.components.policy_factory import _RLHFEndTurnPolicy

        # HumanProxy: забирает idx из ws_input_queue
        human_queue: asyncio.Queue = asyncio.Queue()
        ib = _InteractiveBattleShim(ws, engine, human_queue, gid, bid)

        class _HumanProxyPol:
            name = "human_proxy"
            def select_action(self, engine_, pid):
                # не вызывается в interactive mode
                return 0

        class _BotPol:
            name = f"bot:{p2_model_name}"
            def select_action(self, engine_, pid):
                # bot — p2
                return int(bot.select_action(engine_, pid))

        runner = BattleRunner(
            group_id=gid,
            battle_id=bid,
            policy_a=_HumanProxyPol(),  # для P1 (заглушка)
            policy_b=_BotPol(),         # для P2 (бот)
            engine=engine,
            battle_log_path=battle_log_path,
            human_player=1000,
            max_turns=int(spec.get("max_turns", 60)),
        )

        async def _reader():
            """Читает WS-сообщения от клиента и кладёт action в human_queue."""
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    if data.get("type") == "action":
                        idx = int(data.get("index", -1))
                        if idx < 0:
                            continue
                        await human_queue.put(idx)
                    elif data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                elif msg.type == WSMsgType.ERROR:
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

        # Запускаем параллельно: reader + battle-runner с периодическим push_state
        reader_task = asyncio.create_task(_reader())
        try:
            await runner.arun(action_queue=human_queue)
        finally:
            reader_task.cancel()

        await ws.send_json({"type": "result", "battle_log": runner.battle_log})
        # Обновим manifest: добавим battle_id в completed battles
        manifest_obj = self.session_manager._groups.get(gid)
        if manifest_obj is not None:
            manifest_obj.manifest.append_battle_result(
                battle_id=bid,
                battle_log_path=str(battle_log_path),
                winner_user_id=runner.battle_log["result"]["winner_user_id"],
                loser_user_id=runner.battle_log["result"]["loser_user_id"],
                status=runner.battle_log["result"]["status"],
                turns=runner.battle_log["final_state_summary"]["turn_number"],
                duration_seconds=runner.battle_log["duration_seconds"],
            )


class _InteractiveBattleShim:
    """Тонкий shim для future live-push (пока не используется, задел)."""
    def __init__(self, ws, engine, queue, gid, bid):
        self.ws = ws
        self.engine = engine
        self.queue = queue
        self.gid = gid
        self.bid = bid


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLHF-среда ExtraArena (web @ port)")
    p.add_argument("--host", default=os.environ.get("RLHF_HOST", DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.environ.get("RLHF_PORT", DEFAULT_PORT)))
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", DEFAULT_MODELS_DIR))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", DEFAULT_SESSIONS_DIR))
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", DEFAULT_CARDS_PATH))
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = WebApp(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        sessions_dir=args.sessions_dir,
        cards_path=args.cards_path,
    )

    logger.info("=" * 60)
    logger.info("RLHF-среда ExtraArena v%s", __version__)
    logger.info("  http://%s:%d", app.host, app.port)
    logger.info("  models_dir:    %s", app.models_dir)
    logger.info("  sessions_dir:  %s", app.sessions_dir)
    logger.info("  cards:         %s", app.cards_path)
    logger.info("  ONNX loaded:   %d", len(app.registry.specs))
    logger.info("=" * 60)

    web.run_app(app.app, host=app.host, port=app.port, print=lambda *a, **k: None)


if __name__ == "__main__":
    main()
