"""Граф-сценарный онбординг-туториал (prod-safe).

Раньше обучение было захардкоженным пошаговым автоматом: ``TUTORIAL_STEPS``
(словарь) + ``_build_state_for_step``, который на каждом шаге **телепортировал**
свежий ``GameState`` в нужную расстановку, подделывая последствия (летал фейком
hero.hp=0; «удар по Альфонсу» оставлял труп hp=0 на доске). Противник по-настоящему
не ходил — ``auto_continue`` просто переключал шаги.

Теперь обучение описано **одним граф-сценарием** ``scenarios/onboarding_basic.json``
(формат ``extra_orchestra.scenario.v2`` — тот же, что в ExtraOrchestra): init-сцена
+ один путь узлов (action/scene/turn). Состояние строится **один раз** из init-сцены
и обходится по графу через **настоящий** движок (``BattleEngine.execute_action`` →
``ArenaEnvironment.step``): p1-шаги (узлы action side=p1 с ``tutorial``) ждут новичка,
прочие action-узлы (ходы оппонента, end_turn-ы) auto-применяются. Состояние наследуется
по пути, последствия настоящие (летал — реальный ``P1_WIN``, смерть Альфонса — реальное
удаление с доски).

**Prod-независимость:** модуль НЕ импортирует ``extra_orchestra`` — это dev-утилита
(MCP :8095, граф-редактор, preview/export), её нет на проде. Здесь используются только
prod-модули (``battle_engine``, ``core.*``, ``infrastructure.*``) + репозиторный
``cards.json`` + детерминированные ``uuid5`` instance_id по той же формуле/seed, что в
``cards_catalog.deterministic_instance_id``. Поэтому dev-``preview_frames`` оркестры
предсказывает prod-состояние туториала **точно** — тот же сценарий валидируется/предпросматривается/экспортируется
оркестрой в dev без правок prod-кода. Внешний контракт (``tutorial_payload``, имена
констант/класса, форма результатов ``apply_tutorial_action``) сохранён побайтово →
``web/server.py`` правок не требует.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from random import Random
from typing import Any, Optional
from uuid import NAMESPACE_DNS, uuid5

from battle_engine import BattleEngine
from core.actions import AttackAction, EndTurnAction, PlayCardAction
from core.engine import ArenaEnvironment
from core.state import GameState, GameStatus, PlayerState

logger = logging.getLogger(__name__)


ONBOARDING_STATUS_NOT_STARTED = "not_started"
ONBOARDING_STATUS_WELCOME = "welcome"
ONBOARDING_STATUS_TUTORIAL_BATTLE = "tutorial_battle"
ONBOARDING_STATUS_MENU_TOUR = "menu_tour"
ONBOARDING_STATUS_COMPLETED = "completed"

ONBOARDING_STATUSES = {
    ONBOARDING_STATUS_NOT_STARTED,
    ONBOARDING_STATUS_WELCOME,
    ONBOARDING_STATUS_TUTORIAL_BATTLE,
    ONBOARDING_STATUS_MENU_TOUR,
    ONBOARDING_STATUS_COMPLETED,
}

ONBOARDING_MENU_STEPS = ("arena", "collection", "decks")
ONBOARDING_MIDORIA_ASSET = "/DesignAssets/MidoriaOnboardingGuide.png"

NEWBIE_PATH_TASKS = [
    {
        "id": "open_starter_case",
        "title": "Открой стартовый кейс",
        "completion_text": "Есть. Кейс открыт.",
        "reward": {"type": "coins", "amount": 50},
    },
    {
        "id": "view_new_card",
        "title": "Посмотри новую карту",
        "completion_text": "Карта в коллекции. Уже можно думать, кого взять в отряд.",
        "reward": {"type": "coins", "amount": 50},
    },
    {
        "id": "save_first_deck",
        "title": "Сохрани первую колоду",
        "completion_text": "Колода сохранена.",
        "reward": {"type": "coins", "amount": 75},
    },
    {
        "id": "play_regular_battle",
        "title": "Сыграй обычный бой",
        "completion_text": "Первый настоящий бой принят.",
        "reward": {"type": "coins", "amount": 100},
    },
    {
        "id": "join_telegram_channel",
        "title": "Вступи в Telegram-канал ExtraArena",
        "completion_text": "Ты в канале. Теперь важные новости точно не пройдут мимо.",
        "action_url": "https://t.me/extraarena",
        "reward": {"type": "coins", "amount": 100},
    },
    {
        "id": "claim_newbie_reward",
        "title": "Забери награду новичка",
        "completion_text": "Маршрут новичка закрыт. Дальше уже по-взрослому.",
        "reward": {"type": "coins", "amount": 150},
    },
]


TUTORIAL_BOT_ID_OFFSET = 9_000_000_000_000
TUTORIAL_TURN_DURATION_SECONDS = 99
TUTORIAL_SCRIPTED_AUTO_DELAY_MS = 5600


WRONG_ACTION_FEEDBACK = {
    "generic": "Сейчас не туда. Следуй подсветке.",
    "wrong_target": "Эту цель пока не трогаем. Нам нужен герой.",
    "sleeping_unit": "Рано. Эта карта еще спит.",
    "wrong_alphonse": "Сейчас нужен Альфонс. Он примет удар на себя.",
    "tutorial_lock": "Этот бой учебный. Действуем по плану.",
}


def tutorial_bot_id_for_user(user_id: int) -> int:
    return TUTORIAL_BOT_ID_OFFSET + int(user_id)


def tutorial_match_id_for_user(user_id: int) -> str:
    return f"tutorial-{int(user_id)}"


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Конфиг обучения: сценарий-граф (prod-asset, лежит в репо рядом с модулем).
# ---------------------------------------------------------------------------

_TUTORIAL_SEED = 42
_REPO_ROOT = Path(__file__).resolve().parent
_TUTORIAL_SCENARIO_PATH = _REPO_ROOT / "scenarios" / "onboarding_basic.json"
_CARDS_JSON_PATH = _REPO_ROOT / "cards.json"


def _load_cards_by_id() -> dict[int, dict[str, Any]]:
    data = json.loads(_CARDS_JSON_PATH.read_text(encoding="utf-8"))
    by_id: dict[int, dict[str, Any]] = {}
    for row in data:
        try:
            by_id[int(row["id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return by_id


_CARDS_BY_ID = _load_cards_by_id()


def _deterministic_instance_id(*, seed: int, side: str, zone: str, index: int, card_id: int, level: int):
    """Стабильный instance_id по позиции карты в сценарии.

    Та же формула/seed, что ``cards_catalog.deterministic_instance_id`` в
    extra_orchestra → реплеи (вкл. dev preview_frames) дают идентичные id.
    """
    key = f"{int(seed)}:{side}:{zone}:{int(index)}:{int(card_id)}:{int(level)}"
    return uuid5(NAMESPACE_DNS, "extra-orchestra:" + key)


def _build_card_from_spec(spec: dict[str, Any], *, seed: int, side: str, zone: str, index: int):
    """CardInstance из сценарной spec карты: база из cards.json + overrides + id.

    prod-локальный аналог ``CardsCatalog.build_instance`` (без импорта
    extra_orchestra). overrides: mechanics/attack/hp/max_hp/is_ready/is_frozen.
    """
    card_id = int(spec.get("card_id", 0) or 0)
    level = int(spec.get("level", 1) or 1)
    row = _CARDS_BY_ID[card_id]
    from core.converter import card_from_db

    card = card_from_db(row, level)
    card.instance_id = _deterministic_instance_id(
        seed=seed, side=side, zone=zone, index=index, card_id=card_id, level=level
    )
    if spec.get("mechanics_override") is not None:
        card.mechanics = list(spec["mechanics_override"])
    if spec.get("attack_override") is not None:
        card.attack = int(spec["attack_override"])
    if spec.get("hp_override") is not None:
        card.hp = int(spec["hp_override"])
    if spec.get("max_hp_override") is not None:
        card.max_hp = int(spec["max_hp_override"])
    if spec.get("is_ready") is not None:
        card.is_ready = bool(spec["is_ready"])
    if spec.get("is_frozen") is not None:
        card.is_frozen = bool(spec["is_frozen"])
    return card


def _build_card_list(items, *, seed: int, side: str, zone: str):
    return [
        _build_card_from_spec(it, seed=seed, side=side, zone=zone, index=i)
        for i, it in enumerate(items or [])
    ]


def _build_player(side_spec: dict[str, Any], user_id: int, *, seed: int, side: str) -> PlayerState:
    hero_spec = side_spec.get("hero") or {"card_id": 1, "level": 1}
    hero = _build_card_from_spec(hero_spec, seed=seed, side=side, zone="hero", index=0)
    mana = int(side_spec.get("mana", 0) or 0)
    return PlayerState(
        user_id=int(user_id),
        is_bot=bool(side_spec.get("is_bot", False)),
        hero=hero,
        mana=mana,
        max_mana=int(side_spec.get("max_mana", mana) or 0),
        hand=_build_card_list(side_spec.get("hand", []), seed=seed, side=side, zone="hand"),
        board=_build_card_list(side_spec.get("board", []), seed=seed, side=side, zone="board"),
        deck=_build_card_list(side_spec.get("deck", []), seed=seed, side=side, zone="deck"),
        trophies=int(side_spec.get("trophies", 0) or 0),
    )


class TutorialArenaEnvironment(ArenaEnvironment):
    """ArenaEnvironment для онбординг-туториала.

    Сценарий онбординга — скриптовый: колоды пустые, добор карт не предусмотрен
    (руки автор задаёт явно в init/узлах). Базовый ``_handle_end_turn`` честно
    добирает 1 карту в начале хода; при пустой колоде это resurrect'ит сброс
    (reshuffle graveyard → deck → hand) — убитый Альфонс возвращается в руку
    новичка на следующем конце хода. В туториале «мёртв = мёртв», поэтому перед
    добором чистим graveyard того, кто добирает (opponent): тогда
    ``draw_one_from_deck`` идёт по пути fatigue (пустая колода + пустой сброс →
    ``return False``), и убитые бойцы не воскресают. Остальная логика
    end-of-turn (смена хода, мана, пробуждение, реген) остаётся как в базе.
    """

    def _handle_end_turn(self, player: PlayerState, opponent: PlayerState) -> None:
        opponent.graveyard.clear()
        super()._handle_end_turn(player, opponent)


def _build_initial_arena(scenario: dict[str, Any], p1_uid: int, p2_uid: int, *, mode_config) -> ArenaEnvironment:
    """Построить ArenaEnvironment один раз из init-сцены (apply_start_effects=False).

    Фактические user_id берутся снаружи (p1_uid=новичок, p2_uid=бот), НЕ из
    сценарных плейсхолдеров 1001/2002 — instance_id всё равно side-based (uuid5
    по side/zone/index/card_id), поэтому совпадает с dev preview_frames оркестры.
    """
    graph = scenario["graph"]
    init_node = next(n for n in graph["nodes"] if n["id"] == graph["start"])
    scene = init_node["scene"]
    seed = int(scenario.get("seed", 0) or 0)
    p1 = _build_player(scene.get("p1") or {}, p1_uid, seed=seed, side="p1")
    p2 = _build_player(scene.get("p2") or {}, p2_uid, seed=seed, side="p2")
    starting_side = scene.get("starting_side", "p1")
    current_turn_owner_id = int(p1_uid) if starting_side == "p1" else int(p2_uid)
    state = GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=current_turn_owner_id,
        turn_number=int(scene.get("turn_number", 1) or 1),
        status=GameStatus.ONGOING,
    )
    # apply_start_effects=False: init-сцена остаётся ровно как автор описал;
    # start-of-turn эффекты следующих ходов честно применятся через end_turn.
    # TutorialArenaEnvironment: подавляет resurrect убитых бойцов из сброса
    # при конце хода (см. класс) — иначе Альфонс возвращается в руку после смерти.
    return TutorialArenaEnvironment(
        state,
        classic_params=mode_config.classic,
        apply_start_effects=False,
        rng=Random(seed),
    )


# ---------------------------------------------------------------------------
# Граф → линейный путь + step-узлы (узлы с tutorial). TUTORIAL_STEPS выводится
# из графа, чтобы web/server.py (TUTORIAL_STEPS.get(step-1).get("id")) не менять.
# ---------------------------------------------------------------------------


def _load_tutorial_scenario() -> dict[str, Any]:
    return json.loads(_TUTORIAL_SCENARIO_PATH.read_text(encoding="utf-8"))


def _build_linear_path(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    graph = scenario["graph"]
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    outgoing: dict[str, str | None] = {}
    for edge in graph.get("edges", []):
        outgoing.setdefault(edge["from"], edge["to"])
    path: list[dict[str, Any]] = []
    cur = graph["start"]
    seen: set[str] = set()
    while cur is not None and cur in nodes_by_id and cur not in seen:
        seen.add(cur)
        node = nodes_by_id[cur]
        path.append(node)
        cur = outgoing.get(cur)
    return path


_TUTORIAL_SCENARIO = _load_tutorial_scenario()
_TUTORIAL_PATH = _build_linear_path(_TUTORIAL_SCENARIO)
_TUTORIAL_STEP_NODES = [n for n in _TUTORIAL_PATH if n.get("tutorial")]
_TUTORIAL_STEP_INDEX_OF = {n["id"]: i for i, n in enumerate(_TUTORIAL_STEP_NODES)}
_TUTORIAL_PATH_INDEX_OF = {n["id"]: i for i, n in enumerate(_TUTORIAL_PATH)}

TUTORIAL_STEPS: dict[int, dict[str, Any]] = {
    i: dict(node["tutorial"]) for i, node in enumerate(_TUTORIAL_STEP_NODES)
}
TUTORIAL_FINAL_STEP = len(_TUTORIAL_STEP_NODES) - 1
TUTORIAL_DISPLAY_STEPS_TOTAL = TUTORIAL_FINAL_STEP


# ---------------------------------------------------------------------------
# Хелперы payload (читают наследованное состояние real-движка).
# ---------------------------------------------------------------------------


def _snapshot_id_for_card(engine: "TutorialBattleEngine", card_id: int, owner: str, zone: str) -> str | None:
    if not engine._arena:
        return None
    state = engine._arena.state
    player = state.p1 if owner == "player" else state.p2
    cards: list
    if zone == "hand":
        cards = player.hand
    elif zone == "board":
        cards = player.board
    else:
        cards = [player.hero]
    for card in cards:
        if int(card.card_id) == int(card_id):
            return str(card.instance_id)
    return None


def _previous_tutorial_message(step_index: int) -> str | None:
    if step_index <= 0:
        return None
    previous = TUTORIAL_STEPS.get(step_index - 1)
    if not previous:
        return None
    return str(previous.get("after") or previous.get("message") or "").strip() or None


class TutorialBattleEngine(BattleEngine):
    """Deterministic onboarding battle driven by the onboarding graph-scenario.

    Subclass prod ``BattleEngine`` (NOT OrchestraBattleEngine): ходы применяются
    через настоящий ``execute_action`` → ``ArenaEnvironment.step``, состояние
    наследуется по пути графа. p1-шаги ждут новичка, остальные узлы auto-применяются.
    """

    def __init__(self, *, user_id: int, tutorial_step: int = 0, db: Any = None, active_matches: Optional[dict[str, BattleEngine]] = None) -> None:
        bot_id = tutorial_bot_id_for_user(user_id)
        super().__init__(
            db=db,
            match_id=tutorial_match_id_for_user(user_id),
            player_ids=[int(user_id), bot_id],
            is_bot_match=True,
            active_matches=active_matches,
            p1_name="Ты",
            p2_name="Кто-то злой",
            game_mode="training",
        )
        # Параметры режима — из сценария (99с/mana_per_turn=1/no sudden death),
        # чтобы и арена, и сериализатор были согласованы с onboarding-конфигом.
        from dataclasses import replace
        from infrastructure.match_modes import ClassicParams

        classic = self._classic_params_from(_TUTORIAL_SCENARIO.get("classic_params"))
        self.mode_config = replace(self.mode_config, classic=classic)
        self.turn_duration = self.mode_config.classic.turn_duration_seconds

        self.tutorial_step = max(0, min(int(tutorial_step or 0), TUTORIAL_FINAL_STEP))
        self.is_onboarding_tutorial = True
        self.bot_id = bot_id
        self._rng_seed = _TUTORIAL_SEED
        self._replay_to_step(self.tutorial_step)

    @staticmethod
    def _classic_params_from(spec: Optional[dict[str, Any]]):
        from infrastructure.match_modes import ClassicParams

        spec = spec or {}
        fields = {f.name for f in ClassicParams.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in spec.items() if k in fields and v is not None}
        return ClassicParams(**kwargs)

    # -- deterministic state build + replay ---------------------------------

    def _build_fresh_arena(self) -> ArenaEnvironment:
        return _build_initial_arena(
            _TUTORIAL_SCENARIO,
            int(self.player_ids[0]),
            int(self.player_ids[1]),
            mode_config=self.mode_config,
        )

    def _sync_engine_state(self) -> None:
        state = self._arena.state
        self.current_player_id = state.current_turn_owner_id
        self.turn = state.turn_number
        self.is_ended = state.status != GameStatus.ONGOING
        self.client_ready = True
        self.client_ready_users = {int(self.player_ids[0]), int(self.player_ids[1])}

    def _side_uid(self, side: str) -> int:
        return int(self.player_ids[0]) if side == "p1" else int(self.player_ids[1])

    def _action_obj_from_node(self, node: dict[str, Any], side_uid: int):
        """BaseAction из узла графа (auto-применение silent/p1-replay ходов).

        Индексы разрешаются по board/hand действующей стороны (автор-френдли).
        """
        spec = node.get("action") or {}
        atype = spec.get("type")
        player, opponent = self._arena._resolve_player_pair(side_uid)
        if atype == "play_card":
            hand_index = int(spec.get("hand_index", 0) or 0)
            target_id = None
            if spec.get("target_is_hero"):
                target_id = str(opponent.hero.instance_id)
            elif spec.get("target_index") is not None:
                ti = int(spec["target_index"])
                if 0 <= ti < len(opponent.board):
                    target_id = str(opponent.board[ti].instance_id)
            elif spec.get("target_id") is not None:
                target_id = spec["target_id"]
            return PlayCardAction(hand_index=hand_index, target_id=target_id, position=spec.get("position"))
        if atype == "attack":
            attacker_id = spec.get("attacker_id")
            if attacker_id is None and spec.get("attacker_index") is not None:
                ai = int(spec["attacker_index"])
                if 0 <= ai < len(player.board):
                    attacker_id = str(player.board[ai].instance_id)
            target_id = spec.get("target_id")
            target_is_hero = bool(spec.get("target_is_hero", False))
            if not target_is_hero and target_id is None and spec.get("target_index") is not None:
                ti = int(spec["target_index"])
                if 0 <= ti < len(opponent.board):
                    target_id = str(opponent.board[ti].instance_id)
            return AttackAction(
                attacker_id=str(attacker_id) if attacker_id is not None else "",
                target_id=target_id,
                target_is_hero=target_is_hero,
            )
        if atype == "end_turn":
            return EndTurnAction()
        raise ValueError(f"unknown tutorial action type: {atype}")

    def _apply_node(self, node: dict[str, Any]) -> dict[str, Any]:
        """Применить action-узел графа через настоящий execute_action (silent/replay)."""
        side_uid = self._side_uid(node.get("side", "p1"))
        action_obj = self._action_obj_from_node(node, side_uid)
        result = self.execute_action(side_uid, action_obj)
        if result.get("success") is False:
            # Сценарий провалидирован в dev (preview_frames); failure здесь =
            # баг авторства сценария (нелегальный ход/side-guard). Не глотаем тихо.
            raise RuntimeError(
                f"tutorial scenario action failed at node {node.get('id')!r}: {result.get('error')}"
            )
        return result

    def _replay_to_step(self, step: int) -> None:
        """Детерминированный реплей из init до step-узла N (auto-проигрывая все action-узлы до него)."""
        import core.effects as _effects

        self._rng_seed = _TUTORIAL_SEED
        orig_random = _effects.random
        _effects.random = Random(_TUTORIAL_SEED)
        try:
            self._arena = self._build_fresh_arena()
            self._rng = Random(_TUTORIAL_SEED)
            target_path_index = _TUTORIAL_PATH_INDEX_OF[_TUTORIAL_STEP_NODES[step]["id"]]
            # применяем все action-узлы ПЕРЕД step-узлом N (init s0 = path[0] уже построен)
            for node in _TUTORIAL_PATH[1:target_path_index]:
                if node.get("kind") == "action":
                    self._apply_node(node)
            self.tutorial_step = step
        finally:
            _effects.random = orig_random
        self._sync_engine_state()

    def set_tutorial_step(self, step: int) -> None:
        self._replay_to_step(max(0, min(int(step), TUTORIAL_FINAL_STEP)))

    # -- advance along the graph -------------------------------------------

    def _advance_to_next_step(self, from_path_index: int) -> dict[str, Any] | None:
        """Применить silent action-узлы после from_path_index до ближайшего step-узла.

        Возвращает результат последнего применённого действия (нужно для game_over
        при летале) или None, если silent-узлов не было.
        """
        import core.effects as _effects

        orig_random = _effects.random
        _effects.random = getattr(self, "_rng", Random(_TUTORIAL_SEED))
        last_result: dict[str, Any] | None = None
        try:
            cursor = from_path_index
            while cursor < len(_TUTORIAL_PATH):
                node = _TUTORIAL_PATH[cursor]
                if node.get("tutorial"):
                    break  # дошли до следующего step-узла
                if node.get("kind") == "action":
                    last_result = self._apply_node(node)
                cursor += 1
            if cursor < len(_TUTORIAL_PATH):
                self.tutorial_step = _TUTORIAL_STEP_INDEX_OF[_TUTORIAL_PATH[cursor]["id"]]
            else:
                self.tutorial_step = TUTORIAL_FINAL_STEP
        finally:
            _effects.random = orig_random
        self._sync_engine_state()
        return last_result

    # -- public API (contract preserved) ------------------------------------

    def get_full_state(self, viewer_id: Optional[int] = None) -> dict[str, Any]:
        state = super().get_full_state(viewer_id=viewer_id)
        state["game_mode"] = "tutorial"
        state["is_onboarding_tutorial"] = True
        state["tutorial"] = self.tutorial_payload()
        state["match_status"] = "active"
        state["battle_started"] = True
        state["legal_actions"] = self._tutorial_legal_actions()
        return state

    def tutorial_payload(self) -> dict[str, Any]:
        step = TUTORIAL_STEPS.get(self.tutorial_step, TUTORIAL_STEPS[TUTORIAL_FINAL_STEP])
        allowed = dict(step.get("allowed") or {})
        display_step = self.tutorial_step if self.tutorial_step > 0 else None
        payload = {
            "step_index": self.tutorial_step,
            "step_id": step["id"],
            "message": step["message"],
            "hint": step.get("hint"),
            "target": step.get("target"),
            "allowed": allowed,
            "is_auto_step": bool(step.get("is_auto_step") or allowed.get("type") == "auto_continue"),
            "player_step": display_step,
            "player_steps_total": TUTORIAL_DISPLAY_STEPS_TOTAL,
            "display_step": display_step,
            "display_steps_total": TUTORIAL_DISPLAY_STEPS_TOTAL,
            "previous_message": _previous_tutorial_message(self.tutorial_step),
            "final_step": TUTORIAL_FINAL_STEP,
            "wrong_action_feedback": WRONG_ACTION_FEEDBACK,
            "midoria_asset": ONBOARDING_MIDORIA_ASSET,
        }
        if step.get("auto_advance_delay_ms"):
            payload["auto_advance_delay_ms"] = int(step["auto_advance_delay_ms"])
        payload["attacker_instance_id"] = _snapshot_id_for_card(self, 37, "player", "board")
        payload["alphonse_instance_id"] = _snapshot_id_for_card(self, 39, "player", "board")
        payload["hand_attacker_instance_id"] = _snapshot_id_for_card(self, 37, "player", "hand")
        payload["hand_alphonse_instance_id"] = _snapshot_id_for_card(self, 39, "player", "hand")
        return payload

    def _tutorial_legal_actions(self) -> list[dict[str, Any]]:
        if not self._arena:
            return []

        step = TUTORIAL_STEPS.get(self.tutorial_step)
        allowed = dict(step.get("allowed") or {}) if step else {}
        action_type = allowed.get("type")
        state = self._arena.state

        if action_type == "play_card":
            expected_card_id = _safe_int(allowed.get("card_id"))
            hand_index = next(
                (
                    index
                    for index, card in enumerate(state.p1.hand)
                    if _safe_int(card.card_id) == expected_card_id
                ),
                None,
            )
            if hand_index is None:
                return []
            return [
                {
                    "type": "play_card",
                    "hand_index": hand_index,
                    "card_id": expected_card_id,
                    "target_id": None,
                    "position": len(state.p1.board),
                }
            ]

        if action_type == "end_turn":
            return [{"type": "end_turn"}]

        if action_type == "auto_continue":
            return []

        if action_type == "attack":
            expected_attacker = _safe_int(allowed.get("attacker_card_id"))
            attacker = next(
                (
                    card
                    for card in state.p1.board
                    if _safe_int(card.card_id) == expected_attacker
                ),
                None,
            )
            if not attacker:
                return []
            actions: list[dict[str, Any]] = [
                {
                    "type": "attack",
                    "attacker_id": str(attacker.instance_id),
                    "target_id": None,
                    "target_is_hero": bool(allowed.get("target_is_hero")),
                }
            ]
            # also_allow_minion_targets: advertise every opponent board minion as an
            # extra attackable target (with target_is_hero=False) so the arena can offer
            # a "choose your target" step where tapping a minion is a wrong, redirectable
            # action. validate_tutorial_action still rejects minion taps via the
            # target_is_hero mismatch — this flag only widens what's highlighted/tappable.
            if allowed.get("also_allow_minion_targets"):
                for minion in state.p2.board:
                    actions.append({
                        "type": "attack",
                        "attacker_id": str(attacker.instance_id),
                        "target_id": str(minion.instance_id),
                        "target_is_hero": False,
                    })
            return actions

        return []

    def _wrong_feedback_for(self, action: dict[str, Any]) -> str:
        step = TUTORIAL_STEPS.get(self.tutorial_step, {})
        allowed = step.get("allowed", {}) or {}
        # Per-step override: a tutorial node may carry a `wrong_feedback` dict keyed by
        # reason (wrong_target / wrong_card / sleeping_unit / generic) to replace the
        # generic WRONG_ACTION_FEEDBACK text for that step. Falls back to the shared
        # constants when the step doesn't define one for the matched reason.
        wf = step.get("wrong_feedback", {}) or {}
        atype = action.get("type")
        if atype == "attack" and not action.get("target_is_hero") and allowed.get("target_is_hero"):
            return wf.get("wrong_target") or WRONG_ACTION_FEEDBACK["wrong_target"]
        if atype == "play_card":
            return wf.get("wrong_card") or WRONG_ACTION_FEEDBACK["wrong_alphonse"]
        if atype == "attack":
            return wf.get("sleeping_unit") or WRONG_ACTION_FEEDBACK["sleeping_unit"]
        return wf.get("generic") or WRONG_ACTION_FEEDBACK["generic"]

    def validate_tutorial_action(self, action: dict[str, Any]) -> tuple[bool, str]:
        step = TUTORIAL_STEPS.get(self.tutorial_step)
        if not step:
            return False, WRONG_ACTION_FEEDBACK["tutorial_lock"]
        allowed = dict(step.get("allowed") or {})
        action_type = str(action.get("type") or "")
        if action_type != allowed.get("type"):
            return False, self._wrong_feedback_for(action)
        if action_type == "play_card":
            expected_card_id = allowed.get("card_id")
            actual_card_id = action.get("card_id")
            if actual_card_id is None and self._arena:
                hand_index = _safe_int(action.get("hand_index"))
                hand = self._arena.state.p1.hand
                if hand_index is not None and 0 <= hand_index < len(hand):
                    actual_card_id = hand[hand_index].card_id
            if _safe_int(actual_card_id, 0) != _safe_int(expected_card_id, 0):
                return False, self._wrong_feedback_for(action)
        if action_type == "attack":
            if bool(action.get("target_is_hero")) != bool(allowed.get("target_is_hero")):
                return False, self._wrong_feedback_for(action)
            expected_attacker = allowed.get("attacker_card_id")
            if expected_attacker and self._arena:
                attacker_id = str(action.get("attacker_id") or "")
                attacker = next((card for card in self._arena.state.p1.board if str(card.instance_id) == attacker_id), None)
                if not attacker or int(attacker.card_id) != int(expected_attacker):
                    return False, self._wrong_feedback_for(action)
        return True, ""

    def apply_tutorial_action(self, action: dict[str, Any]) -> dict[str, Any]:
        ok, feedback = self.validate_tutorial_action(action)
        if not ok:
            return {
                "success": False,
                "error": "tutorial_wrong_action",
                "feedback": feedback,
                "tutorial_step": self.tutorial_step,
            }

        left_step = self.tutorial_step
        left_node = _TUTORIAL_STEP_NODES[left_step]
        left_path_index = _TUTORIAL_PATH_INDEX_OF[left_node["id"]]

        # continue / auto_continue / complete — beat-шаги (без действия пользователя).
        if action["type"] == "continue":
            self._advance_to_next_step(left_path_index + 1)
            return {"success": True, "tutorial_step": self.tutorial_step}

        if action["type"] == "auto_continue":
            previous_message = TUTORIAL_STEPS.get(left_step, {}).get("message")
            self._advance_to_next_step(left_path_index + 1)
            leaving_step_id = (left_node.get("tutorial") or {}).get("id")
            if leaving_step_id == "taunt_demo":
                scripted_event = "opponent_attack_taunt"
            elif leaving_step_id == "taunt_hit":
                scripted_event = "opponent_attacked_taunt"
            else:
                scripted_event = "tutorial_auto_continue"
            return {
                "success": True,
                "tutorial_step": self.tutorial_step,
                "after_message": previous_message,
                "scripted_event": scripted_event,
            }

        if action["type"] == "complete":
            self.tutorial_step = TUTORIAL_FINAL_STEP
            return {
                "success": True,
                "tutorial_step": self.tutorial_step,
                "game_over": True,
                "winner_id": int(self.player_ids[0]),
            }

        # play_card / attack / end_turn — action-шаг: действие новичка через real-движок.
        if action["type"] == "play_card":
            position = len(self._arena.state.p1.board) if self._arena else 0
            action_obj = PlayCardAction(
                hand_index=_safe_int(action.get("hand_index"), 0) or 0,
                target_id=action.get("target_id"),
                position=position,
            )
        elif action["type"] == "attack":
            action_obj = AttackAction(
                attacker_id=str(action.get("attacker_id") or ""),
                target_id=action.get("target_id"),
                target_is_hero=bool(action.get("target_is_hero")),
            )
        else:
            action_obj = EndTurnAction()

        result = self.execute_action(int(self.player_ids[0]), action_obj)
        if result.get("success") is False:
            return result

        self._advance_to_next_step(left_path_index + 1)
        after_message = TUTORIAL_STEPS.get(left_step, {}).get("after")
        payload: dict[str, Any] = {
            "success": True,
            "tutorial_step": self.tutorial_step,
            "after_message": after_message,
        }
        if result.get("game_over") or self.tutorial_step >= TUTORIAL_FINAL_STEP:
            payload.update({"game_over": True, "winner_id": int(result.get("winner_id") or self.player_ids[0])})
        return payload