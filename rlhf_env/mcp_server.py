"""MCP-сервер RLHF-среды ExtraArena (stdio) — игра агентом headless + метаданные.

Инструменты разделены на две группы:

  Управление серией:
    start_series(spec)                — создать серию, вернуть первый match_id
    next_battle(group_id)             — следующий бой серии (или series_complete)
    list_battle_groups()              — список групп
    get_battle_group_status(gid)      — статус группы
    get_battle_group_manifest(gid)    — manifest.json
    get_dataset(group_id)             — путь к каноничному NDJSON (dataset.jsonl)
    list_battle_manifests(gid)        — список battle_log + .jsonl путей
    download_battle_logs(gid, fmt)    — zip/json папка группы

  Игра (агент играет за человека, headless, без браузера/WS):
    get_state(match_id)               — полный actor-perspective state (как /api/battle/state)
    get_legal_actions(match_id)      — список легальных действий
    submit_action(match_id, action)  — выполнить действие человека; авто-advance бота
    advance_bot(match_id)            — прокрутить один ход бота (если сейчас ход бота)
    surrender(match_id)              — сдаться → финализация + NDJSON flush

Контракт совпадает с браузерной ареной (тот же RlhfBattleEngine + MatchRunner),
поэтому данные из MCP-игры и из браузера идентичны по форме.

Запуск:
    ./rlhf_env/start_rlhf_env.sh mcp
    python -m rlhf_env.mcp_server
    python -m rlhf_env.mcp_server --models-dir /path/to/v5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.arena_match_manager import ArenaMatchManager  # noqa: E402
from rlhf_env.components.match_runner import MatchRunner  # noqa: E402
from rlhf_env.components.policy_factory import BOT_MAX_DIFFICULTY  # noqa: E402
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402

logger = logging.getLogger(__name__)


def _policy_info(policy: Any) -> Optional[Dict[str, Any]]:
    """Лёгкое provenance-описание политики: {name,kind,model_path,weights_hash,weights_version}.

    Используется в ответах start_series/register_custom_model, чтобы оркестратор
    видел resolved adapter (V2/V3/V4/V5/baseline) без полного state.
    """
    if policy is None:
        return None
    info: Dict[str, Any] = {
        "name": getattr(policy, "name", None),
        "kind": getattr(policy, "kind", None),
    }
    for attr in ("model_path", "weights_hash", "weights_version"):
        v = getattr(policy, attr, None)
        if v is not None:
            info[attr] = v
    return info


def _runner_warnings(runner: Optional[MatchRunner]) -> List[str]:
    """BUG4: проброс policy-fallback предупреждений (напр. v5-stub → end_turn)
    в ответ start_series/next_battle, чтобы MCP-клиент видел деградацию модели,
    а не молчаливый «нормальный» бой."""
    if runner is None:
        return []
    return list(getattr(runner, "policy_fallbacks", []) or [])


# ============================================================================
# HeadlessHub — реестр матчей + ленивые MatchRunner'ы (без WS/broadcaster)
# ============================================================================

class HeadlessHub:
    """Связывает ArenaMatchManager с MatchRunner'ами для headless-игры."""

    def __init__(self, *, sessions_dir: str, models_dir: str, cards_path: str,
                 registry: Optional["PolicyRegistry"] = None):
        # ЕДИНЫЙ реестр моделей на hub: MCPServer.list_models/register_custom_model
        # и ArenaMatchManager.start_series (_build_match → build_policy) должны делить
        # один объект, иначе register_custom_model добавляет модель в реестр A, а
        # start_series резолвит из реестра B → KeyError (BUG1: кастомные модели unusable).
        self.manager = ArenaMatchManager(
            sessions_dir=sessions_dir, models_dir=models_dir, cards_path=cards_path,
            registry=registry,
        )
        self._runners: Dict[str, MatchRunner] = {}

    def _match(self, match_id: str):
        return self.manager.get_match(match_id)

    async def get_runner(self, match_id: str) -> Optional[MatchRunner]:
        r = self._runners.get(match_id)
        if r is not None:
            return r
        match = self.manager.get_match(match_id)
        if match is None:
            return None
        r = MatchRunner(match)
        # broadcaster=None → WS-бродкасты не нужны (headless).
        self._runners[match_id] = r
        return r


# ============================================================================
# MCP-сервер (JSON-RPC 2.0 over stdio)
# ============================================================================

class MCPServer:
    def __init__(self, hub: HeadlessHub, registry: Optional[PolicyRegistry] = None):
        self.hub = hub
        # BUG1 фикс: list_models/register_custom_model и start_series должны делить
        # один реестр. Авторитативный — hub.manager.registry (его использует
        # _build_match). Дополнительно вливаем спеки из переданного registry, если
        # они были созданы отдельным scan (main()/тесты), чтобы ничего не потерять.
        self.registry = hub.manager.registry
        if registry is not None and registry is not self.registry:
            for spec in registry.specs:
                if spec.name not in self.registry._name_index:
                    self.registry.add_spec(spec)
        self.tools = self._build_tools()

    # ------------------------------------------------------------------
    # tool schemas
    # ------------------------------------------------------------------
    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "start_series",
                "description": (
                    "Создать серию боёв (человек vs модель) и вернуть первый match_id. "
                    "spec: {p2_model, battles_planned, seed?, starting_player?, "
                    "deck_strategy_p1?, deck_strategy_p2?, custom_deck_p1?, custom_deck_p2?, "
                    "p1_name?, p2_name?, ...}. Модель всегда играет на максимум (argmax); "
                    "сложность не выбирается. Агент играет за человека (p1) через submit_action."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": {
                                "p2_model": {
                                    "type": ["string", "object"],
                                    "description": "имя модели-оппонента ('random'/имя из registry) ИЛИ объект {name,path,kind} для custom model by path+adapter.",
                                },
                                "p2_model_path": {"type": "string", "description": "путь к .onnx оппонента (custom by path; под models_dir/repo root)"},
                                "p2_model_kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"], "description": "адаптер оппонента (V2/V3=legacy_onnx, V4=action_onnx, V5=v5, baseline)"},
                                "p1_model": {"type": ["string", "object"], "description": "имя/объект RL-модели для p1 (при p1_actor_type='rl', model-vs-model)"},
                                "p1_model_path": {"type": "string"},
                                "p1_model_kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"]},
                                "battles_planned": {"type": "integer", "default": 1, "minimum": 1, "maximum": 1000},
                                "seed": {"type": "integer", "default": 0},
                                "starting_player": {"type": "string", "default": "random", "enum": ["random", "p1", "p2"]},
                                "deck_strategy_p1": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom", "preset"]},
                                "deck_strategy_p2": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom", "preset"]},
                                "custom_deck_p1": {"type": "array", "items": {"type": "integer"}},
                                "custom_deck_p2": {"type": "array", "items": {"type": "integer"}},
                                "preset_number_p1": {"type": "integer", "description": "номер preset-колоды (deck_strategy_p1='preset')"},
                                "preset_number_p2": {"type": "integer"},
                                "p1_deck_source": {"type": "object", "description": "явная форма {type:'imported',preset_number:N} (движок поддерживает)"},
                                "agent_name": {"type": "string", "description": "кодовое имя суб-агента (опц.); auto-assign из пула если не задано"},
                                "p1_name": {"type": "string"}, "p2_name": {"type": "string"},
                                "p1_actor_type": {
                                    "type": "string", "enum": ["human", "llm", "rl"], "default": "llm",
                                    "description": "тип актора p1: 'llm' (MCP-модель, default), 'human' (браузер) или 'rl' (наша RL-модель auto-play, model-vs-model). Определяет battle_tag и decision_source в V5-трейсах.",
                                },
                            },
                            "required": [],
                        },
                    },
                    "required": ["spec"],
                },
            },
            {
                "name": "next_battle",
                "description": "Перейти к следующему бою серии. Возвращает match_id или series_complete.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_state",
                "description": "Полный actor-perspective state боя (тот же формат, что /api/battle/state в браузере).",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "get_legal_actions",
                "description": "Список легальных действий для текущего (человека) игрока.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "submit_action",
                "description": (
                    "Выполнить действие игрока (p1). action: {type:'play_card'|'attack'|'end_turn'|'mana_draw', ...}. "
                    "play_card: {type:'play_card', card_id_from_hand|hand_index, target_position?, target_id?, target_is_hero?}. "
                    "attack: {type:'attack', attacker_id, target_id, target_is_hero?}. "
                    "end_turn: {type:'end_turn'}. "
                    "mana_draw: {type:'mana_draw'} — добор карты за ману (стоимость 2*(count+1)/ход, не передаёт ход). "
                    "После play_card/attack/end_turn ход бота прокручивается автоматически; после mana_draw — нет."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "match_id": {"type": "string"},
                        "action": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["play_card", "attack", "end_turn", "mana_draw"]},
                                "card_id_from_hand": {"type": "integer"},
                                "hand_index": {"type": "integer"},
                                "target_position": {"type": "integer"},
                                "target_id": {"type": ["integer", "string"]},
                                "target_is_hero": {"type": "boolean"},
                                "attacker_id": {"type": ["integer", "string"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["match_id", "action"],
                },
            },
            {
                "name": "advance_bot",
                "description": "Прокрутить один ход бота (если сейчас ход бота). Иначе no-op.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "surrender",
                "description": "Сдаться (человек) → финализация боя + flush NDJSON.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "list_battle_groups",
                "description": "Список всех групп боёв.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_battle_group_status",
                "description": "Статус группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_battle_group_manifest",
                "description": "Полное содержимое manifest.json группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_dataset",
                "description": "Путь к каноничному NDJSON (dataset.jsonl) и per-battle .jsonl файлам группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_battle_manifests",
                "description": "Список battle_log.json + analytics .jsonl путей по группе.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "download_battle_logs",
                "description": "Собрать логи группы в zip или вернуть список json-файлов.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["json", "zip"], "default": "json"},
                    },
                    "required": ["group_id"],
                },
            },
            {
                "name": "list_models",
                "description": "Список доступных моделей-оппонентов (из registry + зарегистрированные через register_custom_model).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "list_active_series",
                "description": (
                    "«Общая картинка» активных серий: (a) число активных игр по агенту и (b) по модели-оппоненту. "
                    "Возвращает агентов с прогрессом боёв N/M + wins/losses/draws/decks и группировку by_model."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_agent_status",
                "description": "Статус «играющего» суб-агента по кодовому имени: бои N/M, победы, поражения, ничьи, колоды, оппонент, p1_actor_type.",
                "inputSchema": {"type": "object", "properties": {"agent_name": {"type": "string"}}, "required": ["agent_name"]},
            },
            {
                "name": "get_match_status",
                "description": "Лёгкий статус боя (polling без полного state): turn, is_ended, winner, is_my_turn, current_player_id, action_count.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "get_action_history",
                "description": "История ходов боя (replay длинных боёв без re-fetch fullstate).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"match_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 200}},
                    "required": ["match_id"],
                },
            },
            {
                "name": "finish_series",
                "description": "Завершить серию досрочно: закрыть manifest (finished_at + summary) и освободить кодовое имя агента.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_preset_decks",
                "description": "Список preset-колод (ArenaENV Random / JSON-imported) для deck_strategy='preset'.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "register_custom_model",
                "description": (
                    "Зарегистрировать кастомную модель by path+adapter (V2/V3/V4/V5/baseline) in-memory: "
                    "позволяет оркестратору выбирать early-snapshot V5 и т.п. через {name,path,kind} в start_series. "
                    "path должен лежать под models_dir или корнем репо (защита от path-traversal)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string", "description": "путь к .onnx под models_dir/repo root"},
                        "kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"]},
                    },
                    "required": ["name", "path"],
                },
            },
            {
                "name": "get_v5_dataset_summary",
                "description": "V5 training orchestrator: сводка по группе — строки, v5_trace_ok, распределение battle_tag, turns/actions total.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_v5_groups",
                "description": "V5 orchestrator: список групп с v5-трейсами (фильтр по battle_tag).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"battle_tag": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 100}},
                },
            },
            {
                "name": "get_v5_trace",
                "description": "V5 orchestrator: содержимое v5/trace боя (meta|turns|actions).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "battle_id": {"type": "string"},
                        "what": {"type": "string", "enum": ["meta", "turns", "actions"]},
                    },
                    "required": ["group_id", "battle_id", "what"],
                },
            },
            {
                "name": "validate_v5_traces",
                "description": "V5 orchestrator: проверить целостность v5/trace всех боёв группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
        ]

    # ------------------------------------------------------------------
    # tool handlers (async)
    # ------------------------------------------------------------------
    async def _tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        hub = self.hub

        if name == "start_series":
            spec = args.get("spec", {})
            if not isinstance(spec, dict):
                raise ValueError("spec must be an object")
            spec.setdefault("battles_planned", 1)
            # MCP = LLM-актор по умолчанию (полу-синтетические бои тегируются
            # llm-vs-bot, decision_source='llm' в V5-трейсах).
            spec.setdefault("p1_actor_type", "llm")
            match = hub.manager.create_series(spec)
            # headless: прокручиваем все бот-ходы до хода human/llm ИЛИ до game_over.
            # run_auto единообразно покрывает три режима: human/llm-vs-bot (ходит
            # p2 если он стартует, стоп на p1), rl-vs-bot и rl-vs-rl (model-vs-model,
            # бой до game_over без submit_action). Прежний «if is_current_player_bot:
            # run_bot_turn()» ломал p1-as-RL (по умолчанию ходил только p2).
            runner = await hub.get_runner(match.engine.match_id)
            if runner is not None:
                await self._run_auto_bounded(runner)
            p1_actor = match.engine.p1_actor_type
            return {
                "group_id": match.group_id,
                "match_id": match.engine.match_id,
                "battle_id": match.battle_id,
                "battles_planned": match.battles_planned,
                "player_ids": [match.engine.human_user_id, match.engine.bot_user_id],
                "opponent": {"model": spec.get("p2_model"), "difficulty": BOT_MAX_DIFFICULTY},
                "p1_actor_type": p1_actor,
                "battle_tag": match.engine.battle_tag,
                "agent_name": getattr(match, "agent_name", None),
                "p1_model": _policy_info(match.p1_policy),
                "p2_model": _policy_info(match.bot_policy),
                "is_ended": match.engine.is_ended,
                "winner_id": (match.engine._get_winner_id() if match.engine.is_ended else None),
                "policy_warnings": _runner_warnings(runner),
                "degraded": bool(_runner_warnings(runner)),
            }

        if name == "next_battle":
            gid = args["group_id"]
            match = hub.manager.next_match(gid)
            if match is None:
                return {"status": "series_complete", "group_id": gid}
            runner = await hub.get_runner(match.engine.match_id)
            if runner is not None:
                await self._run_auto_bounded(runner)
            return {"match_id": match.engine.match_id, "battle_id": match.battle_id, "group_id": gid,
                    "agent_name": getattr(match, "agent_name", None),
                    "p1_actor_type": match.engine.p1_actor_type,
                    "battle_tag": match.engine.battle_tag,
                    "is_ended": match.engine.is_ended,
                    "winner_id": (match.engine._get_winner_id() if match.engine.is_ended else None),
                    "policy_warnings": _runner_warnings(runner)}

        if name == "get_state":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            return match.engine.get_full_state(viewer_id=match.engine.human_user_id)

        if name == "get_legal_actions":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            uid = match.engine.human_user_id
            legal = match.engine.get_legal_actions(uid) if not match.engine.is_current_player_bot() else []
            return {"legal_actions": legal, "is_my_turn": match.engine.get_current_player_id() == uid}

        if name == "submit_action":
            match_id = args["match_id"]
            action = args.get("action", {})
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            # F4(audit): p1-RL (model-vs-model) не управляется через submit_action —
            # иначе внешний клиент вливает p1-action мимо match.p1_policy auto-play,
            # и он мис-тегируется decision_source='rl' в v5/actions.jsonl → портит
            # V5 training-данные. p1-RL водится только advance_bot/run_auto
            # (run_bot_turn с match.p1_policy). Симметрично surrender/get_legal_actions.
            if getattr(runner.match.engine, "p1_actor_type", "human") == "rl":
                return {"error": "submit_action_unavailable_for_rl_p1"}
            action = dict(action)
            action.setdefault("client_action_id", f"mcp_{match_id}_{int(asyncio.get_event_loop().time()*1000)&0xffff}")
            resp = await runner.execute_human_action(action)
            # авто-advance бота уже выполнен в execute_human_action (create_task run_bot_turn),
            # но в headless-режоте create_task мог быть запланирован — дождёмся его.
            await self._drain_bot(runner)
            return resp

        if name == "advance_bot":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            engine = runner.match.engine
            if not engine.is_current_player_bot() and getattr(runner.match, "p1_policy", None) is None:
                return {"status": "not_bot_turn"}
            cur = engine.get_current_player_id()
            if cur == engine.bot_user_id:
                await runner.run_bot_turn()
            elif cur == engine.human_user_id and getattr(runner.match, "p1_policy", None) is not None:
                # p1-as-RL (model-vs-model): один ход RL-модели.
                await runner.run_bot_turn(player_id=cur, policy=runner.match.p1_policy)
            else:
                return {"status": "not_bot_turn"}
            await self._drain_bot(runner)
            return {"status": "ok", "is_ended": engine.is_ended,
                    "winner_id": (engine._get_winner_id() if engine.is_ended else None)}

        if name == "surrender":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            resp = await runner.surrender()
            return resp

        if name == "list_battle_groups":
            return {"groups": hub.manager.list_groups()}

        if name == "get_battle_group_status":
            gid = args["group_id"]
            hub.manager.reap_completed(gid)
            m = hub.manager.list_groups()
            for g in m:
                if g["group_id"] == gid:
                    return g
            return {"error": "group not found"}

        if name == "get_battle_group_manifest":
            gid = args["group_id"]
            path = hub.manager.sessions_dir / gid / "manifest.json"
            if not path.exists():
                return {"error": "group not found"}
            return json.loads(path.read_text(encoding="utf-8"))

        if name == "get_dataset":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            dataset = gdir / "dataset.jsonl"
            battles_dir = gdir / "battles"
            per_battle = sorted(str(p) for p in battles_dir.glob("*.jsonl")) if battles_dir.exists() else []
            return {
                "group_id": gid,
                "dataset_jsonl": str(dataset),
                "dataset_exists": dataset.exists(),
                "dataset_rows": sum(1 for _ in dataset.open()) if dataset.exists() else 0,
                "per_battle_jsonl": per_battle,
            }

        if name == "list_battle_manifests":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid / "battles"
            if not gdir.exists():
                return {"error": "group not found"}
            return {"battles": sorted(str(p) for p in gdir.glob("*.json"))}

        if name == "download_battle_logs":
            gid = args["group_id"]
            fmt = args.get("format", "json")
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            if fmt == "zip":
                zip_path = hub.manager.sessions_dir / f"{gid}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in gdir.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=f.relative_to(gdir))
                return {"path": str(zip_path), "size": zip_path.stat().st_size, "format": "zip"}
            return {"path": str(gdir), "format": "json",
                    "files": sorted(str(p.relative_to(hub.manager.sessions_dir)) for p in gdir.rglob("*") if p.is_file())}

        if name == "list_models":
            return {"models": self.registry.list_specs()}

        if name == "list_active_series":
            # Self-heal: освободить агентов завершённых серий, до которых клиент
            # не дошёл next_battle/finish_series (иначе утекают в agents_index).
            hub.manager.reap_all_completed()
            groups = hub.manager.list_groups()
            # (a) по агенту + (d) «общая картинка» серии; (b) по модели-оппоненту.
            agents: List[Dict[str, Any]] = []
            by_model: Dict[str, Dict[str, Any]] = {}
            for g in groups:
                ag = g.get("agent_name")
                agents.append({
                    "agent_name": ag,
                    "group_id": g["group_id"],
                    "status": g.get("status"),
                    "battles": f"{g.get('battles_finished',0)}/{g.get('battles_planned',0)}",
                    "wins": g.get("wins", 0),
                    "losses": g.get("losses", 0),
                    "draws": g.get("draws", 0),
                    "p1_actor_type": g.get("p1_actor_type"),
                    "opponent_model": g.get("p2_model"),
                    "current_match_id": g.get("current_match_id"),
                })
                mkey = str(g.get("p2_model") or "random")
                bm = by_model.setdefault(mkey, {"model": mkey, "groups": 0, "wins": 0, "losses": 0, "draws": 0})
                bm["groups"] += 1
                bm["wins"] += g.get("wins", 0)
                bm["losses"] += g.get("losses", 0)
                bm["draws"] += g.get("draws", 0)
            running = [g for g in groups if g.get("status") == "running"]
            return {
                "count": len(running),
                "agents": agents,
                "by_model": list(by_model.values()),
            }

        if name == "get_agent_status":
            st = hub.manager.agent_registry.status(args["agent_name"])
            # Self-heal: если серия агента доиграна, а клиент не звал next_battle/
            # finish_series — освободим имя через manager.reap_completed (in-process).
            gid = st.get("group_id")
            if gid:
                hub.manager.reap_completed(gid)
                st = hub.manager.agent_registry.status(args["agent_name"])
            return st

        if name == "get_match_status":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            # Self-heal: бой завершён и серия доиграна → освободить агента сейчас,
            # не ждать next_battle/finish_series (фикс утечки codename в agents_index).
            hub.manager.reap_completed(match.group_id)
            engine = match.engine
            uid = engine.human_user_id
            return {
                "match_id": engine.match_id,
                "group_id": match.group_id,
                "battle_id": match.battle_id,
                "agent_name": getattr(match, "agent_name", None),
                "turn": getattr(engine, "turn", None),
                "is_ended": engine.is_ended,
                "winner_id": (engine._get_winner_id() if engine.is_ended else None),
                "current_player_id": engine.get_current_player_id(),
                # Workflow-B Issue #2: на завершённой игре ничей ход — не тянем
                # current_player_id (после surrender он мог остаться human).
                "is_my_turn": (not engine.is_ended and engine.get_current_player_id() == uid),
                "p1_actor_type": engine.p1_actor_type,
                "battle_tag": engine.battle_tag,
            }

        if name == "get_action_history":
            runner = await hub.get_runner(args["match_id"])
            if runner is None:
                return {"error": "match_not_found"}
            actions = runner.battle_log.get("actions", [])
            limit = int(args.get("limit", 200) or 200)
            return {"actions": actions[-limit:], "count": len(actions)}

        if name == "finish_series":
            gid = args["group_id"]
            live = hub.manager._groups.get(gid)
            if live is None:
                return {"error": "group not found"}
            return hub.manager.finish_series(gid)

        if name == "list_preset_decks":
            # Preset-колоды (ArenaENV Random / JSON-imported) хранятся в прод-БД;
            # headless-среда без БД их не имеет. deck_strategy='random_arenaenv'
            # использует рандом-генератор движка, 'custom' — custom_deck_* из spec.
            return {"presets": [], "note": "preset decks require the prod DB; use deck_strategy='random_arenaenv' or 'custom' with custom_deck_* in headless env"}

        if name == "register_custom_model":
            from rlhf_env.components.policy_adapters import default_registry
            from rlhf_env.components.policy_registry import ModelSpec
            mname = str(args["name"]).strip()
            path = str(args["path"])
            kind = args.get("kind", "auto")
            # path-traversal защита: путь должен лежать под models_dir или корнем репо.
            safe = hub.manager._safe_model_path(path)
            reg = default_registry()
            # F7: грузим sidecar (.onnx.json) перед detect_kind — как scan_directory.
            # Иначе пользовательские V5-детекторы, читающие sidecar (obs_dim/inputs),
            # получают None и не могут определить kind для auto-registered модели.
            detected = None
            if kind == "auto":
                try:
                    from rlhf_env.components.policy_factory import _load_sidecar
                    sidecar = _load_sidecar(safe)
                    detected = reg.detect_kind(str(safe), sidecar, name=mname)
                except Exception:  # noqa: BLE001
                    detected = None
            # 'auto' (а не 'unknown') если детект не сработал — пусть build's own
            # auto-detect branch (он тоже грузит sidecar) дожмёт на start_series.
            final_kind = kind if kind != "auto" else (detected or "auto")
            self.registry.add_spec(ModelSpec(
                name=mname, path=str(safe), sidecar_path=None, kind=final_kind,
                description="custom (registered via MCP)",
            ))
            return {"registered": True, "name": mname, "path": str(safe),
                    "kind": final_kind, "detected_kind": detected}

        # --- V5 training orchestrator --------------------------------------

        if name == "get_v5_dataset_summary":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            man = hub.manager.sessions_dir / gid / "manifest.json"
            if not man.exists():
                return {"error": "manifest not found"}
            m = json.loads(man.read_text(encoding="utf-8"))
            results = m.get("battles_results", []) or []
            tag_dist: Dict[str, int] = {}
            v5_ok = 0
            turns_total = 0
            actions_total = 0
            for r in results:
                t = r.get("battle_tag")
                if t:
                    tag_dist[t] = tag_dist.get(t, 0) + 1
                if r.get("v5_trace_ok"):
                    v5_ok += 1
                v5dir = gdir / "battles" / r["battle_id"] / "v5"
                tpath = v5dir / "turns.jsonl"
                apath = v5dir / "actions.jsonl"
                if tpath.exists():
                    turns_total += sum(1 for _ in tpath.open())
                if apath.exists():
                    actions_total += sum(1 for _ in apath.open())
            return {
                "group_id": gid,
                "battles_finished": len(results),
                "v5_trace_ok_count": v5_ok,
                "battle_tag_distribution": tag_dist,
                "turns_total": turns_total,
                "actions_total": actions_total,
                "rows": actions_total,
            }

        if name == "list_v5_groups":
            tag_filter = args.get("battle_tag")
            limit = int(args.get("limit", 100) or 100)
            out: List[Dict[str, Any]] = []
            sdir = hub.manager.sessions_dir
            if sdir.exists():
                for gdir in sorted(sdir.iterdir()):
                    man = gdir / "manifest.json"
                    if not man.exists():
                        continue
                    try:
                        m = json.loads(man.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        continue
                    results = m.get("battles_results", []) or []
                    tags = {r.get("battle_tag") for r in results if r.get("battle_tag")}
                    if tag_filter and tag_filter not in tags:
                        continue
                    out.append({
                        "group_id": gdir.name,
                        "agent_name": m.get("agent_name"),
                        "battles_finished": len(results),
                        "battle_tags": sorted(t for t in tags if t),
                        "v5_trace_ok_count": sum(1 for r in results if r.get("v5_trace_ok")),
                        "finished_at": m.get("finished_at"),
                    })
                    if len(out) >= limit:
                        break
            return {"groups": out}

        if name == "get_v5_trace":
            gid = args["group_id"]
            bid = args["battle_id"]
            what = args["what"]
            v5dir = hub.manager.sessions_dir / gid / "battles" / bid / "v5"
            if not v5dir.exists():
                return {"error": "v5 trace not found", "path": str(v5dir)}
            fname = {"meta": "meta.json", "turns": "turns.jsonl", "actions": "actions.jsonl"}[what]
            fpath = v5dir / fname
            if not fpath.exists():
                return {"error": f"{fname} not found", "path": str(fpath)}
            if fname.endswith(".json"):
                return {"data": json.loads(fpath.read_text(encoding="utf-8")), "rows_count": 1}
            rows = [json.loads(l) for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip()]
            return {"data": rows, "rows_count": len(rows)}

        if name == "validate_v5_traces":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            man = hub.manager.sessions_dir / gid / "manifest.json"
            m = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {"battles_results": []}
            results = m.get("battles_results", []) or []
            checked = 0
            ok = 0
            broken: List[Dict[str, Any]] = []
            for r in results:
                checked += 1
                v5dir = gdir / "battles" / r["battle_id"] / "v5"
                issues: List[str] = []
                for need in ("meta.json", "turns.jsonl", "actions.jsonl"):
                    fp = v5dir / need
                    if not fp.exists():
                        issues.append(f"missing {need}")
                    elif need.endswith(".jsonl") and fp.stat().st_size == 0:
                        issues.append(f"empty {need}")
                if issues:
                    broken.append({"battle_id": r["battle_id"], "issues": issues})
                else:
                    ok += 1
            return {"checked": checked, "ok": ok, "broken": broken}

        raise ValueError(f"unknown tool: {name}")

    async def _run_auto_bounded(self, runner: MatchRunner) -> bool:
        """run_auto с защитой от зависания (F4): кроме turn-cap внутри run_auto,
        оборачиваем в asyncio.wait_for — если auto-play не укладывается в таймаут
        (напр. обе политики сломаны, длинный rl-vs-rl с реальными delay'ами),
        stdio-вызов не виснет. Возвращает True если auto-play завершился штатно,
        False по таймауту (бой остался незавершённым — оркестратор может
        finish_series)."""
        try:
            await asyncio.wait_for(asyncio.shield(runner.run_auto()), timeout=120.0)
            return True
        except asyncio.TimeoutError:
            logger.warning("run_auto timed out match=%s — leaving battle unfinished", runner.match.engine.match_id)
            runner.policy_fallbacks.append("run_auto timed out (120s) — battle left unfinished; use finish_series to close")
            return False
        except Exception:  # noqa: BLE001
            logger.warning("run_auto crashed: %s", exc_info=True)
            return False

    async def _drain_bot(self, runner: MatchRunner) -> None:
        """Дождаться завершения запланированной бот-рутины (если была)."""
        task = runner._bot_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("bot task timed out while draining")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------
    async def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rlhf-env", "version": __version__},
                "capabilities": {"tools": {}},
            }
        if method == "tools/list":
            return {"tools": self.tools}
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            try:
                result = await self._tool(name, args)
                return {"content": [{"type": "json", "data": result}], "isError": False}
            except Exception as exc:  # noqa: BLE001
                logger.exception("[mcp] tool %s failed", name)
                return {"content": [{"type": "json", "data": {"error": str(exc)}}], "isError": True}
        return {"error": {"code": -32601, "message": f"unknown method: {method}"}}


# ============================================================================
# Async stdio loop (единый event-loop — MatchRunner lock/tasks живы между вызовами)
# ============================================================================

async def _amain(server: MCPServer) -> None:
    """Stdio JSON-RPC loop. Один запрос — один ответ (строго 1:1, без unsolicited-
    banner'ов, чтобы не ломать MCP-клиенты и синхронные stdio-харнессы)."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            req_id = msg.get("id", 0)
            method = msg.get("method", "")
            params = msg.get("params", {}) or {}
            result = await server.dispatch(method, params)
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": 0,
                                         "error": {"code": -32700, "message": str(exc)}}) + "\n")
        sys.stdout.flush()


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-сервер RLHF-среды (stdio)")
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", "ai/models"))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", "rlhf_env/sessions"))
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", "ai/cards.json"))
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "WARNING"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    registry = PolicyRegistry.scan(args.models_dir)
    hub = HeadlessHub(sessions_dir=args.sessions_dir, models_dir=args.models_dir, cards_path=args.cards_path)
    server = MCPServer(hub, registry)
    logger.info("MCP server starting (stdio). tools=%d, models=%d", len(server.tools), len(registry.specs))
    try:
        asyncio.run(_amain(server))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()