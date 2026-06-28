"""Сценарный движок: строит ``GameState`` руками и прогоняет графы ходов.

Модель выполнения (см. план/DOCS):
  init-сцена → граф хода 1 (side A) → граф хода 2 (side B) → …
Состояние между ходами наследуется (один ``ArenaEnvironment``, шагаем
``engine.step`` через ``OrchestraBattleEngine.apply_action``).

Каждое действие узла → один ``Frame`` (снимок после действия + sound_events
+ ``display_ms`` = сколько кадров показывается результат). ``wait``-узел =
удержание текущего состояния на ``delay_ms``. ``end_turn`` (явный или
неявный при ``end_with_end_turn``) шагает ядро и порождает start-of-turn
эффекты следующего игрока (mana/draw/wake) — честно и видимо.

Ссылки на карты в узлах — по индексам (автор-френдли): ``hand_index`` для
play_card, ``attacker_index``/``target_index`` для атаки (0-based позиции
на доске), ``target_is_hero`` для удара в героя. ``instance_id``-ссылки
тоже поддерживаются (продвинутый случай).

Детерминизм (RISK A): ``core/effects.py`` использует module-``random`` (не
``ArenaEnvironment._rng``) в ``cast_random_spell``/``cleave``/``armor_X_Y``.
На время прогона monkeypatch'им ``core.effects.random = Random(seed)``.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import core.effects as _effects
from core.actions import AttackAction, EndTurnAction, ManaDrawAction, PlayCardAction
from core.engine import ArenaEnvironment
from core.state import GameStatus, GameState, PlayerState, ReplacementStatus
from infrastructure.match_modes import ClassicParams, ModeConfig, resolve_mode_config

from .arena_engine import OrchestraBattleEngine
from .cards_catalog import CardsCatalog, deterministic_instance_id

logger = logging.getLogger(__name__)

INIT_INTRO_MS = 1200
IMPLICIT_END_TURN_MS = 500
WAIT_FRAME_DEFAULT_MS = 600


class ScenarioError(ValueError):
    """Нарушение структуры сценария (нечего играть / не та сторона / etc)."""


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def make_frame(*, snapshot: Dict[str, Any], sound_events: List[Dict[str, Any]],
               display_ms: int, action_kind: str, turn_id: str,
               node_id: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "snapshot": snapshot,
        "sound_events": sound_events,
        "display_ms": int(display_ms),
        "action_kind": action_kind,
        "turn_id": turn_id,
        "node_id": node_id,
        "error": error,
    }


# ---------------------------------------------------------------------------
# ClassicParams
# ---------------------------------------------------------------------------

def _classic_params_from(spec: Optional[Dict[str, Any]]) -> ClassicParams:
    spec = spec or {}
    fields = {f.name for f in ClassicParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in spec.items() if k in fields and v is not None}
    return ClassicParams(**kwargs)


def _mode_config_from(spec: Optional[Dict[str, Any]], game_mode: str) -> ModeConfig:
    base = resolve_mode_config(game_mode)
    classic = _classic_params_from(spec)
    # frozen dataclass — пересоберём с новым classic
    from dataclasses import replace
    return replace(base, classic=classic)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _build_card_list(catalog: CardsCatalog, items: List[Dict[str, Any]], *,
                     seed: int, side: str, zone: str) -> List:
    out = []
    for i, it in enumerate(items or []):
        cid = int(it.get("card_id", 0) or 0)
        lvl = int(it.get("level", 1) or 1)
        iid = deterministic_instance_id(seed=seed, side=side, zone=zone, index=i,
                                        card_id=cid, level=lvl)
        out.append(catalog.build_instance(it, instance_id=iid))
    return out


def _build_hero(catalog: CardsCatalog, hero_spec: Dict[str, Any], *,
                seed: int, side: str):
    cid = int(hero_spec.get("card_id", 1) or 1)
    lvl = int(hero_spec.get("level", 1) or 1)
    iid = deterministic_instance_id(seed=seed, side=side, zone="hero", index=0,
                                    card_id=cid, level=lvl)
    return catalog.build_instance(hero_spec, instance_id=iid)


def _build_player(catalog: CardsCatalog, side_spec: Dict[str, Any], user_id: int,
                  *, seed: int, side: str) -> PlayerState:
    hero = _build_hero(catalog, side_spec.get("hero") or {"card_id": 1, "level": 1},
                       seed=seed, side=side)
    return PlayerState(
        user_id=int(user_id),
        is_bot=bool(side_spec.get("is_bot", False)),
        replacement_status=ReplacementStatus.ACTIVE,
        hero=hero,
        mana=int(side_spec.get("mana", 0) or 0),
        max_mana=int(side_spec.get("max_mana", side_spec.get("mana", 0)) or 0),
        mana_draw_count_this_turn=int(side_spec.get("mana_draw_count_this_turn", 0) or 0),
        hand=_build_card_list(catalog, side_spec.get("hand", []), seed=seed, side=side, zone="hand"),
        board=_build_card_list(catalog, side_spec.get("board", []), seed=seed, side=side, zone="board"),
        deck=_build_card_list(catalog, side_spec.get("deck", []), seed=seed, side=side, zone="deck"),
        trophies=int(side_spec.get("trophies", 0) or 0),
    )


def build_initial_state(
    scenario: Dict[str, Any],
    catalog: CardsCatalog,
) -> Tuple[ArenaEnvironment, OrchestraBattleEngine, int, Dict[str, int]]:
    """Построить ``ArenaEnvironment`` + шим из init-сцены сценария.

    Returns ``(env, engine, viewer_uid, side_uids)`` где
    ``side_uids = {"p1": ..., "p2": ...}``.
    """
    init = scenario.get("init_scene") or {}
    if not init:
        raise ScenarioError("init_scene is required")

    p1_spec = init.get("p1") or {}
    p2_spec = init.get("p2") or {}
    p1_uid = int(p1_spec.get("user_id", 1001))
    p2_uid = int(p2_spec.get("user_id", 2002))

    seed = int(scenario.get("seed", 0) or 0)
    p1 = _build_player(catalog, p1_spec, p1_uid, seed=seed, side="p1")
    p2 = _build_player(catalog, p2_spec, p2_uid, seed=seed, side="p2")

    starting_side = init.get("starting_side", "p1")
    if starting_side not in ("p1", "p2"):
        raise ScenarioError(f"bad starting_side: {starting_side}")
    current_turn_owner_id = p1_uid if starting_side == "p1" else p2_uid

    turn_number = int(init.get("turn_number", 1) or 1)

    state = GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=current_turn_owner_id,
        turn_number=turn_number,
        status=GameStatus.ONGOING,
    )

    game_mode = scenario.get("game_mode", "classic")
    mode_config = _mode_config_from(scenario.get("classic_params"), game_mode)
    classic = mode_config.classic

    rng = random.Random(seed)

    # apply_start_effects=False: init-сцена должна остаться ровно как автор
    # описал (без авто-раздачи/mana/wake). start-of-turn эффекты следующих
    # ходов честно применятся через end_turn.
    env = ArenaEnvironment(state, classic_params=classic, apply_start_effects=False, rng=rng)

    viewer_side = scenario.get("viewer_side", "p1")
    viewer_uid = p1_uid if viewer_side == "p1" else p2_uid

    match_id = str(scenario.get("match_id", "orchestra"))
    engine = OrchestraBattleEngine(
        match_id=match_id,
        arena=env,
        mode_config=mode_config,
        p1_user_id=p1_uid,
        p2_user_id=p2_uid,
        p1_profile=p1_spec,
        p2_profile=p2_spec,
        game_mode=game_mode,
        event_id_prefix=str(scenario.get("event_id_prefix", "orchestra")),
    )
    return env, engine, viewer_uid, {"p1": p1_uid, "p2": p2_uid}


# ---------------------------------------------------------------------------
# Node → action resolution
# ---------------------------------------------------------------------------

def _resolve_attacker_id(engine: OrchestraBattleEngine, side_uid: int, node: Dict[str, Any]):
    state = engine._arena.state
    player, _opp = engine._arena._resolve_player_pair(side_uid)
    if player is None:
        raise ScenarioError(f"cannot resolve player for uid {side_uid}")
    if node.get("attacker_id"):
        return str(node["attacker_id"])
    idx = node.get("attacker_index")
    if idx is None:
        raise ScenarioError("attack node needs attacker_id or attacker_index")
    if not (0 <= int(idx) < len(player.board)):
        raise ScenarioError(f"attacker_index {idx} out of board (len={len(player.board)})")
    return str(player.board[int(idx)].instance_id)


def _resolve_target_for_play(engine: OrchestraBattleEngine, side_uid: int, node: Dict[str, Any]):
    """Цель для play_card (заклинание/юнит с эффектом по цели)."""
    if node.get("target_is_hero"):
        opponent = engine._arena.state.p2 if engine._arena.state.p1.user_id == side_uid else engine._arena.state.p1
        return str(opponent.hero.instance_id)
    if node.get("target_id"):
        return str(node["target_id"])
    idx = node.get("target_index")
    if idx is None:
        return None
    opponent = engine._arena.state.p2 if engine._arena.state.p1.user_id == side_uid else engine._arena.state.p1
    if not (0 <= int(idx) < len(opponent.board)):
        raise ScenarioError(f"target_index {idx} out of opponent board (len={len(opponent.board)})")
    return str(opponent.board[int(idx)].instance_id)


def _resolve_target_for_attack(engine: OrchestraBattleEngine, side_uid: int, node: Dict[str, Any]):
    if node.get("target_is_hero"):
        return None  # AttackAction с target_is_hero=True, target_id=None
    if node.get("target_id"):
        return str(node["target_id"])
    idx = node.get("target_index")
    if idx is None:
        raise ScenarioError("attack node needs target_id/target_index/target_is_hero")
    opponent = engine._arena.state.p2 if engine._arena.state.p1.user_id == side_uid else engine._arena.state.p1
    if not (0 <= int(idx) < len(opponent.board)):
        raise ScenarioError(f"target_index {idx} out of opponent board (len={len(opponent.board)})")
    return str(opponent.board[int(idx)].instance_id)


def _node_to_action(engine: OrchestraBattleEngine, side_uid: int, node: Dict[str, Any]):
    ntype = node.get("type")
    if ntype == "play_card":
        return PlayCardAction(
            hand_index=int(node.get("hand_index", 0)),
            target_id=_resolve_target_for_play(engine, side_uid, node),
            position=node.get("position"),
        )
    if ntype == "attack":
        return AttackAction(
            attacker_id=_resolve_attacker_id(engine, side_uid, node),
            target_id=_resolve_target_for_attack(engine, side_uid, node),
            target_is_hero=bool(node.get("target_is_hero", False)),
        )
    if ntype == "end_turn":
        return EndTurnAction()
    if ntype == "mana_draw":
        return ManaDrawAction()
    raise ScenarioError(f"unknown node type: {ntype}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Dict[str, Any],
    catalog: Optional[CardsCatalog] = None,
) -> Dict[str, Any]:
    """Прогнать сценарий → ``{frames, viewer_uid, side_uids, match_id, error?}``.

    Каждый frame: ``{snapshot, sound_events, display_ms, action_kind, turn_id,
    node_id, error?}``. ``snapshot`` — viewer-perspective ``get_full_state``.
    """
    catalog = catalog or CardsCatalog()
    frames: List[Dict[str, Any]] = []

    # --- RISK A: детерминизм module-random ---
    orig_random = _effects.random
    seed = int(scenario.get("seed", 0) or 0)
    _effects.random = random.Random(seed)
    try:
        env, engine, viewer_uid, side_uids = build_initial_state(scenario, catalog)
    finally:
        _effects.random = orig_random

    match_id = engine.match_id
    turns = scenario.get("turns", []) or []

    def _snapshot() -> Dict[str, Any]:
        return engine.get_full_state(viewer_uid)

    # Frame 0 — init-сцена.
    init_display = int((scenario.get("init_scene") or {}).get("display_ms")
                       or scenario.get("init_intro_ms") or INIT_INTRO_MS)
    frames.append(make_frame(
        snapshot=_snapshot(), sound_events=[],
        display_ms=init_display,
        action_kind="init", turn_id="init",
    ))

    err: Optional[str] = None
    try:
        # восстановим monkeypatch на время прогона ходов
        _effects.random = random.Random(seed)
        try:
            for turn in turns:
                turn_id = turn.get("id") or f"turn-{len(frames)}"
                side = turn.get("side")
                if side not in side_uids:
                    raise ScenarioError(f"turn {turn_id}: bad side '{side}'")
                side_uid = side_uids[side]

                current_owner = env.state.current_turn_owner_id
                if current_owner != side_uid:
                    raise ScenarioError(
                        f"turn {turn_id}: side '{side}' (uid {side_uid}) is not the "
                        f"current turn owner (uid {current_owner}). Author must end_turn "
                        f"explicitly to pass the turn."
                    )

                nodes = turn.get("nodes", []) or []
                if not nodes:
                    # пустой ход — удерживаем состояние на duration_ms
                    frames.append(make_frame(
                        snapshot=_snapshot(), sound_events=[],
                        display_ms=int(turn.get("duration_ms", WAIT_FRAME_DEFAULT_MS)),
                        action_kind="wait", turn_id=turn_id,
                    ))

                explicit_end_turn = False
                for node in nodes:
                    ntype = node.get("type")
                    delay = int(node.get("delay_ms", 0) or 0)

                    if ntype == "wait":
                        frames.append(make_frame(
                            snapshot=_snapshot(), sound_events=[],
                            display_ms=delay or WAIT_FRAME_DEFAULT_MS,
                            action_kind="wait", turn_id=turn_id, node_id=node.get("id"),
                        ))
                        continue

                    action = _node_to_action(engine, side_uid, node)
                    result = engine.apply_action(side_uid, action)
                    if not result.get("ok"):
                        err = f"turn {turn_id} node {node.get('id')}: action '{ntype}' failed: {result.get('error')}"
                        frames.append(make_frame(
                            snapshot=result.get("snapshot") or _snapshot(),
                            sound_events=[],
                            display_ms=max(delay, 800),
                            action_kind=str(result.get("action_kind") or ntype),
                            turn_id=turn_id, node_id=node.get("id"), error=err,
                        ))
                        raise ScenarioError(err)

                    frames.append(make_frame(
                        snapshot=result["snapshot"],
                        sound_events=result.get("sound_events", []),
                        display_ms=delay,
                        action_kind=str(result.get("action_kind") or ntype),
                        turn_id=turn_id, node_id=node.get("id"),
                    ))
                    if ntype == "end_turn":
                        explicit_end_turn = True
                    if result.get("game_over"):
                        # бой окончен — досрочный стоп
                        return _result(frames, viewer_uid, side_uids, match_id)

                if turn.get("end_with_end_turn") and not explicit_end_turn:
                    result = engine.apply_action(side_uid, EndTurnAction())
                    frames.append(make_frame(
                        snapshot=result.get("snapshot") or _snapshot(),
                        sound_events=result.get("sound_events", []),
                        display_ms=int(turn.get("end_turn_ms", IMPLICIT_END_TURN_MS)),
                        action_kind="end_turn", turn_id=turn_id, node_id="__implicit__",
                    ))
        finally:
            _effects.random = orig_random
    except ScenarioError as exc:
        err = str(exc)

    return _result(frames, viewer_uid, side_uids, match_id, err)


def _result(frames, viewer_uid, side_uids, match_id, error=None) -> Dict[str, Any]:
    total_ms = sum(int(f.get("display_ms", 0)) for f in frames)
    return {
        "match_id": match_id,
        "viewer_uid": viewer_uid,
        "side_uids": side_uids,
        "frames": frames,
        "frame_count": len(frames),
        "total_ms": total_ms,
        "error": error,
    }


def validate_scenario(scenario: Dict[str, Any], catalog: Optional[CardsCatalog] = None) -> Dict[str, Any]:
    """Сухой прогон: возвращает ``{ok, error, frame_count, total_ms}`` без кадров-данных."""
    catalog = catalog or CardsCatalog()
    try:
        res = run_scenario(scenario, catalog)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "frame_count": 0, "total_ms": 0}
    return {
        "ok": not res.get("error"),
        "error": res.get("error"),
        "frame_count": res.get("frame_count", 0),
        "total_ms": res.get("total_ms", 0),
    }


__all__ = [
    "ScenarioError",
    "build_initial_state",
    "run_scenario",
    "validate_scenario",
    "make_frame",
]