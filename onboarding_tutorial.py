from __future__ import annotations

from typing import Any, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from battle_engine import BattleEngine
from core.actions import AttackAction, EndTurnAction, PlayCardAction
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState


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
        "id": "claim_newbie_reward",
        "title": "Забери награду новичка",
        "completion_text": "Маршрут новичка закрыт. Дальше уже по-взрослому.",
        "reward": {"type": "coins", "amount": 150},
    },
]


TUTORIAL_BOT_ID_OFFSET = 9_000_000_000_000
TUTORIAL_TURN_DURATION_SECONDS = 99
TUTORIAL_TAUNT_DEMO_STEP = 5
TUTORIAL_TAUNT_HIT_STEP = 6
TUTORIAL_LETHAL_STEP = 7
TUTORIAL_FINAL_STEP = 8
TUTORIAL_DISPLAY_STEPS_TOTAL = TUTORIAL_FINAL_STEP
TUTORIAL_SCRIPTED_AUTO_DELAY_MS = 5600


TUTORIAL_STEPS = {
    0: {
        "id": "battle_start",
        "message": "Вот твой герой. Вот герой врага. Его HP — наша цель.",
        "hint": None,
        "target": "opponent_hero",
        "allowed": {"type": "continue"},
    },
    1: {
        "id": "play_attacker",
        "message": "Ставь бойца на поле. Ему нужно занять позицию, прежде чем бить.",
        "hint": "Выставь эту карту",
        "target": "hand_card:37",
        "allowed": {"type": "play_card", "card_id": 37},
        "after": "Он спит до следующего хода. Нормально: только вышел на поле.",
    },
    2: {
        "id": "end_turn",
        "message": "Жми конец хода. Пусть враг сделает ход.",
        "hint": "Завершить ход",
        "target": "end_turn",
        "allowed": {"type": "end_turn"},
        "after": "Враг выставил сильного бойца. Выглядит неприятно, но у нас есть броня потяжелее.",
    },
    3: {
        "id": "attack_hero",
        "message": "Наш боец готов. Бей героя, не разменивайся. HP вниз — победа ближе.",
        "hint": "Выбери бойца и нажми героя врага",
        "target": "opponent_hero",
        "allowed": {"type": "attack", "attacker_card_id": 37, "target_is_hero": True},
    },
    4: {
        "id": "play_alphonse",
        "message": "Теперь Альфонс. Он закрывает проход к герою.",
        "hint": "Выставь Альфонса",
        "target": "hand_card:39",
        "allowed": {"type": "play_card", "card_id": 39},
    },
    5: {
        "id": "taunt_demo",
        "message": "Видишь? Это Провокация. Враг обязан сначала ударить Альфонса.",
        "hint": None,
        "target": "board_card:39",
        "allowed": {"type": "auto_continue"},
        "is_auto_step": True,
        "auto_advance_delay_ms": TUTORIAL_SCRIPTED_AUTO_DELAY_MS,
    },
    6: {
        "id": "taunt_hit",
        "message": "Стив ударил Альфонса. Провокация сработала — путь к герою открыт.",
        "hint": None,
        "target": "board_card:39",
        "allowed": {"type": "auto_continue"},
        "is_auto_step": True,
        "auto_advance_delay_ms": TUTORIAL_SCRIPTED_AUTO_DELAY_MS,
    },
    7: {
        "id": "lethal",
        "message": "Путь открыт. Добивай героя.",
        "hint": "Атакуй героя врага",
        "target": "opponent_hero",
        "allowed": {"type": "attack", "attacker_card_id": 37, "target_is_hero": True},
    },
    8: {
        "id": "victory",
        "message": "Готово. Главное правило поймано: победа — это 0 HP у героя врага.\nТеперь покажу, где собирать отряд и искать новые бои.",
        "hint": "В меню",
        "target": None,
        "allowed": {"type": "complete"},
    },
}


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


def _tutorial_instance_id(user_id: int, key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"extraarena:onboarding-tutorial:{int(user_id)}:{key}")


def _hero(name: str, hp: int, *, instance_id: UUID | None = None) -> CardInstance:
    return CardInstance(
        instance_id=instance_id or uuid4(),
        card_id=1,
        name=name,
        card_type=CardType.HERO,
        rarity="start",
        mana_cost=0,
        attack=0,
        hp=hp,
        max_hp=hp,
        mechanics=[],
        is_ready=True,
        description="Герой",
    )


def _warrior(
    card_id: int,
    name: str,
    attack: int,
    hp: int,
    mana: int,
    mechanics: Optional[list[str]] = None,
    *,
    instance_id: UUID | None = None,
) -> CardInstance:
    return CardInstance(
        instance_id=instance_id or uuid4(),
        card_id=card_id,
        name=name,
        card_type=CardType.WARRIOR,
        rarity="start",
        mana_cost=mana,
        attack=attack,
        hp=hp,
        max_hp=hp,
        mechanics=list(mechanics or []),
        is_ready=False,
        description="Учебная карта" if card_id != 39 else "Душа в доспехах. Обладает Провокацией и защищает слабых.",
        mechanics_desc="Провокация: враг обязан атаковать эту карту первой." if card_id == 39 else "",
    )


def _snapshot_id_for_card(engine: "TutorialBattleEngine", card_id: int, owner: str, zone: str) -> str | None:
    if not engine._arena:
        return None
    state = engine._arena.state
    player = state.p1 if owner == "player" else state.p2
    cards: list[CardInstance]
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
    """Deterministic onboarding battle controlled by a server-side step machine."""

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
        self.turn_duration = TUTORIAL_TURN_DURATION_SECONDS
        self.tutorial_step = max(0, min(int(tutorial_step or 0), TUTORIAL_FINAL_STEP))
        self.is_onboarding_tutorial = True
        self.bot_id = bot_id
        self._build_state_for_step(self.tutorial_step)

    def _build_state_for_step(self, step: int) -> None:
        user_id = int(self.player_ids[0])
        bot_id = int(self.player_ids[1])
        player = PlayerState(
            user_id=user_id,
            is_bot=False,
            hero=_hero("Твой герой", 18, instance_id=_tutorial_instance_id(user_id, "player-hero")),
            mana=2,
            max_mana=2,
            hand=[],
            board=[],
            deck=[],
            trophies=0,
        )
        opponent = PlayerState(
            user_id=bot_id,
            is_bot=True,
            hero=_hero("Герой врага", 7, instance_id=_tutorial_instance_id(user_id, "opponent-hero")),
            mana=2,
            max_mana=2,
            hand=[],
            board=[],
            deck=[],
            trophies=0,
        )

        attacker = _warrior(37, "Слайм", 3, 2, 1, instance_id=_tutorial_instance_id(user_id, "player-attacker-37"))
        threat = _warrior(40, "Стив", 3, 3, 3, instance_id=_tutorial_instance_id(user_id, "opponent-threat-40"))
        alphonse = _warrior(
            39,
            "Альфонс Элрик",
            1,
            3,
            2,
            ["taunt"],
            instance_id=_tutorial_instance_id(user_id, "player-alphonse-39"),
        )

        current_owner = user_id
        turn = 1

        if step <= 1:
            player.hand = [attacker]
        elif step == 2:
            attacker.is_ready = False
            player.board = [attacker]
            player.hand = [alphonse]
        elif step == 3:
            attacker.is_ready = True
            player.board = [attacker]
            player.hand = [alphonse]
            opponent.board = [threat]
            opponent.hero.hp = 7
            turn = 3
        elif step == 4:
            attacker.is_ready = False
            player.board = [attacker]
            player.hand = [alphonse]
            opponent.board = [threat]
            opponent.hero.hp = 4
            turn = 3
        elif step == TUTORIAL_TAUNT_DEMO_STEP:
            attacker.is_ready = False
            threat.is_ready = True
            player.board = [attacker, alphonse]
            player.hand = []
            opponent.board = [threat]
            opponent.hero.hp = 4
            current_owner = bot_id
            turn = 4
        elif step == TUTORIAL_TAUNT_HIT_STEP:
            attacker.is_ready = False
            threat.is_ready = False
            threat.hp = 2
            alphonse.hp = 0
            alphonse.is_ready = False
            player.board = [attacker, alphonse]
            player.hand = []
            opponent.board = [threat]
            opponent.hero.hp = 4
            current_owner = bot_id
            turn = 4
        elif step == TUTORIAL_LETHAL_STEP:
            attacker.is_ready = True
            threat.is_ready = False
            threat.hp = 2
            player.board = [attacker]
            player.hand = []
            opponent.board = [threat]
            opponent.hero.hp = 4
            turn = 5
        else:
            attacker.is_ready = False
            player.board = [attacker]
            player.hand = []
            opponent.board = [threat]
            opponent.hero.hp = 0
            current_owner = user_id
            turn = 5

        game_state = GameState(
            p1=player,
            p2=opponent,
            current_turn_owner_id=current_owner,
            turn_number=turn,
            status=GameStatus.P1_WIN if step >= TUTORIAL_FINAL_STEP else GameStatus.ONGOING,
        )
        if step == TUTORIAL_TAUNT_DEMO_STEP:
            game_state.action_history.append(("opponent", "Кто-то злой готовит удар, но Провокация Альфонса закрывает героя."))
        elif step == TUTORIAL_TAUNT_HIT_STEP:
            game_state.action_history.append(("opponent", "Стив ударил Альфонса на 3 HP. Провокация защитила героя."))
        elif step == TUTORIAL_LETHAL_STEP:
            game_state.action_history.append(("opponent", "Стив ударил Альфонса. Провокация сработала и открыла путь к герою."))
        self._arena = ArenaEnvironment(game_state, classic_params=self.mode_config.classic)
        self.current_player_id = current_owner
        self.turn = turn
        self.client_ready = True
        self.client_ready_users = {user_id}
        self.is_ended = step >= TUTORIAL_FINAL_STEP

    def set_tutorial_step(self, step: int) -> None:
        self.tutorial_step = max(0, min(int(step), TUTORIAL_FINAL_STEP))
        self._build_state_for_step(self.tutorial_step)

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
            return [
                {
                    "type": "attack",
                    "attacker_id": str(attacker.instance_id),
                    "target_id": None,
                    "target_is_hero": bool(allowed.get("target_is_hero")),
                }
            ]

        return []

    def _wrong_feedback_for(self, action: dict[str, Any]) -> str:
        allowed = TUTORIAL_STEPS.get(self.tutorial_step, {}).get("allowed", {})
        if self.tutorial_step == 4 and action.get("type") == "play_card":
            return WRONG_ACTION_FEEDBACK["wrong_alphonse"]
        if action.get("type") == "attack" and not action.get("target_is_hero") and allowed.get("target_is_hero"):
            return WRONG_ACTION_FEEDBACK["wrong_target"]
        if action.get("type") == "attack" and self.tutorial_step <= 2:
            return WRONG_ACTION_FEEDBACK["sleeping_unit"]
        return WRONG_ACTION_FEEDBACK["generic"]

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

        if action["type"] == "continue":
            self.set_tutorial_step(1)
            return {"success": True, "tutorial_step": self.tutorial_step}

        if action["type"] == "auto_continue":
            previous_message = TUTORIAL_STEPS.get(self.tutorial_step, {}).get("message")
            if self.tutorial_step == TUTORIAL_TAUNT_DEMO_STEP:
                next_step = TUTORIAL_TAUNT_HIT_STEP
                scripted_event = "opponent_attack_taunt"
            elif self.tutorial_step == TUTORIAL_TAUNT_HIT_STEP:
                next_step = TUTORIAL_LETHAL_STEP
                scripted_event = "opponent_attacked_taunt"
            else:
                next_step = min(self.tutorial_step + 1, TUTORIAL_FINAL_STEP)
                scripted_event = "tutorial_auto_continue"
            self.set_tutorial_step(next_step)
            return {
                "success": True,
                "tutorial_step": self.tutorial_step,
                "after_message": previous_message,
                "scripted_event": scripted_event,
            }

        if action["type"] == "complete":
            self.set_tutorial_step(TUTORIAL_FINAL_STEP)
            return {"success": True, "tutorial_step": self.tutorial_step, "game_over": True, "winner_id": int(self.player_ids[0])}

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

        next_step = min(self.tutorial_step + 1, TUTORIAL_FINAL_STEP)
        self.set_tutorial_step(next_step)
        payload: dict[str, Any] = {
            "success": True,
            "tutorial_step": self.tutorial_step,
            "after_message": TUTORIAL_STEPS.get(self.tutorial_step - 1, {}).get("after"),
        }
        if self.tutorial_step >= TUTORIAL_FINAL_STEP:
            payload.update({"game_over": True, "winner_id": int(self.player_ids[0])})
        return payload
