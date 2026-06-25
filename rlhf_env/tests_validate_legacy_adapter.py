"""Валидатор корректности маппинга действий legacy-adapter'а.

Для каждого legacy ONNX-модели и seed 1..5 строит матч через ArenaMatchManager
(HeadlessHub-совместимый путь), и на первых бот-ходах проверяет:
  - decoded BaseAction (из adapter) ∈ get_legal_actions_raw(bot_id);
  - engine.execute_action(bot_id, chosen).success == True;
  - бот НЕ всегда играет end_turn;
  - read-only shim: живая арена НЕ мутируется после inference
    (state.turn_number / hand / board до и после select_action идентичны).
Сравнивает с тем, что вернул LegacyOnnxPolicy.select_action(shim) (TrainV2 id)
и какие legal-actions были доступны.
"""
from __future__ import annotations

import copy
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, List

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlhf_env.components.arena_match_manager import ArenaMatchManager
from rlhf_env.components.policy_factory import _LegacyOnnxBotPolicy, _LiveArenaShim
from ai.train_v2.classic_actions_v1 import decode_action
from core.actions import EndTurnAction, PlayCardAction, AttackAction

BOT_ID = 2000
HUMAN_ID = 1000
MAX_BOT_TURNS = 5
MAX_ACTIONS_PER_TURN = 10


def _state_signature(engine) -> tuple:
    """Полная сводка играбельных полей state (БЕЗ back-references arena_engine).

    GameState содержит ссылку arena_engine -> ArenaEnvironment (без __eq__),
    поэтому прямое `deepcopy(state) != state` ложно-положительно. Сравниваем
    только значимые поля: turn/owner/status + оба игрока (hand/board/mana/hero).
    """
    s = engine._arena.state
    p1, p2 = s.p1, s.p2

    def _card_key(c):
        return (
            c.instance_id,
            c.card_id,
            c.mana_cost,
            c.attack,
            c.hp,
            c.max_hp,
            c.is_ready,
            c.is_frozen,
            tuple(c.mechanics),
        )

    def _player_key(ps):
        return (
            ps.user_id,
            ps.mana,
            ps.max_mana,
            ps.hero.hp,
            ps.hero.max_hp,
            tuple(_card_key(c) for c in ps.hand),
            tuple(_card_key(c) for c in ps.board),
            tuple(_card_key(c) for c in ps.deck),
        )

    return (
        s.turn_number,
        s.current_turn_owner_id,
        s.status,
        _player_key(p1),
        _player_key(p2),
        list(s.action_history),
    )


def _action_label(a: Any) -> str:
    if isinstance(a, EndTurnAction):
        return "end_turn"
    if isinstance(a, PlayCardAction):
        return "play_card"
    if isinstance(a, AttackAction):
        return "attack"
    return "other"


def _find_exact(legal: List[Any], base: Any) -> int:
    for i, a in enumerate(legal):
        if a == base:
            return i
    return -1


def run_one(model_name: str, seed: int, sessions_dir: str) -> dict:
    manager = ArenaMatchManager(
        sessions_dir=sessions_dir,
        models_dir="ai/models",
        cards_path="ai/cards.json",
    )
    spec = {
        "p2_model": model_name,
        "battles_planned": 1,
        "seed": seed,
        "starting_player": "p2",
        "deck_strategy_p1": "random_arenaenv",
        "deck_strategy_p2": "random_arenaenv",
    }
    match = manager.create_series(spec)
    engine = match.engine
    bot_policy = match.bot_policy

    result = {
        "model": model_name,
        "seed": seed,
        "is_legacy_adapter": isinstance(bot_policy, _LegacyOnnxBotPolicy),
        "total_actions": 0,
        "end_turn": 0,
        "play_card": 0,
        "attack": 0,
        "execute_failed": 0,
        "decoded_not_in_legal": 0,
        "shim_mutated": 0,
        "train_v2_id_not_in_shim_legal": 0,
        "adapter_used_fallback": 0,
        "turns_seen": 0,
        "failures": [],
    }

    if not result["is_legacy_adapter"]:
        result["failures"].append(f"bot_policy is {type(bot_policy).__name__}, not _LegacyOnnxBotPolicy")
        return result

    inner = bot_policy._inner  # LegacyOnnxPolicy

    for _turn in range(MAX_BOT_TURNS):
        if engine.is_ended:
            break
        # Передать ход боту, если сейчас ход человека.
        cur = engine.get_current_player_id()
        if cur == HUMAN_ID:
            r = engine.end_turn(HUMAN_ID)
            if not r.get("success"):
                result["failures"].append(f"human end_turn failed: {r.get('error')}")
                break
        if engine.is_ended:
            break
        if engine.get_current_player_id() != BOT_ID:
            break
        result["turns_seen"] += 1

        for _step in range(MAX_ACTIONS_PER_TURN):
            if engine.is_ended or engine.get_current_player_id() != BOT_ID:
                break
            legal = engine.get_legal_actions_raw(BOT_ID)
            if not legal:
                result["failures"].append("bot turn but no legal actions")
                break

            # Снимок живой арены ДО inference.
            pre_sig = _state_signature(engine)

            # 1) Что вернёт raw LegacyOnnxPolicy (TrainV2 id) и legal-ids shim'а.
            shim = _LiveArenaShim(engine._arena)
            shim_legal_ids = shim.legal_action_ids(BOT_ID)
            try:
                train_v2_id = int(inner.select_action(shim, BOT_ID))
            except Exception as exc:  # noqa: BLE001
                result["failures"].append(f"inner.select_action raised: {exc!r}")
                break

            # Снимок живой арены ПОСЛЕ inference (read-only shim check).
            post_sig = _state_signature(engine)
            if pre_sig != post_sig:
                result["shim_mutated"] += 1
                result["failures"].append(
                    f"shim mutated live arena: pre={pre_sig} post={post_sig}"
                )

            # TrainV2 id должен быть в legal-ids shim'а (иначе adapter выбрал
            # id, не соответствующий ни одному легальному действию).
            if train_v2_id not in shim_legal_ids:
                result["train_v2_id_not_in_shim_legal"] += 1
                result["failures"].append(
                    f"train_v2_id={train_v2_id} not in shim legal_action_ids "
                    f"(len={len(shim_legal_ids)})"
                )

            # 2) Декодировать TrainV2 id -> BaseAction.
            decoded = decode_action(shim.clone_state(), BOT_ID, train_v2_id)

            # 3) Adapter: idx в engine.get_legal_actions_raw.
            idx = int(bot_policy.select_action(engine._arena, BOT_ID))
            if idx < 0 or idx >= len(legal):
                result["failures"].append(
                    f"adapter returned out-of-range idx={idx} (legal len={len(legal)})"
                )
                break
            chosen = legal[idx]

            # 4) decoded ∈ legal (точное value-equality совпадение)?
            exact_idx = _find_exact(legal, decoded) if decoded is not None else -1
            if decoded is None or exact_idx < 0:
                result["decoded_not_in_legal"] += 1
                result["failures"].append(
                    f"decoded={decoded} (train_v2_id={train_v2_id}) NOT in legal; "
                    f"chosen={chosen.to_dict()}; legal_types={[_action_label(a) for a in legal]}"
                )

            # 4b) Adapter должен вернуть именно то действие, которое модель
            # выбрала (idx == exact_idx). Если idx != exact_idx при exact_idx>=0
            # — сработал fallback-путь адаптера (loosen/type-fallback), и бот
            # играет НЕ то, что хотела модель.
            if exact_idx >= 0 and idx != exact_idx:
                result["adapter_used_fallback"] = result.get("adapter_used_fallback", 0) + 1
                result["failures"].append(
                    f"adapter idx={idx} != exact_idx={exact_idx} for decoded={decoded}; "
                    f"chosen={chosen.to_dict()} (fallback path triggered)"
                )

            # 5) execute_action -> success.
            res = engine.execute_action(BOT_ID, chosen)
            if not res.get("success"):
                result["execute_failed"] += 1
                result["failures"].append(
                    f"execute_action failed: {res.get('error')} chosen={chosen.to_dict()}"
                )

            # Статистика по типам.
            label = _action_label(chosen)
            result["total_actions"] += 1
            if label == "end_turn":
                result["end_turn"] += 1
            elif label == "play_card":
                result["play_card"] += 1
            elif label == "attack":
                result["attack"] += 1

            # Если бот завершил ход — выходим из внутреннего цикла.
            if isinstance(chosen, EndTurnAction) and res.get("success"):
                break
            if res.get("game_over"):
                break

    # Итоговая проверка «не всегда end_turn».
    if result["total_actions"] > 0 and result["end_turn"] == result["total_actions"]:
        result["failures"].append(
            f"adapter played ONLY end_turn ({result['end_turn']}/{result['total_actions']})"
        )
    if result["total_actions"] == 0:
        result["failures"].append("no bot actions were executed")
    return result


def main() -> int:
    models = [
        "OnlyVersusRandomBiggest",
        "extra-lr-v3-medium",
        "extra-lr-v3-max",
    ]
    all_failures: list[str] = []
    print("=" * 78)
    print("LEGACY ADAPTER MAPPING VALIDATION")
    print("=" * 78)
    with tempfile.TemporaryDirectory(prefix="rlhf_val_") as tmp:
        for model in models:
            for seed in range(1, 6):
                r = run_one(model, seed, tmp)
                ok = not r["failures"]
                tag = "OK " if ok else "FAIL"
                print(
                    f"[{tag}] model={r['model']:<26} seed={r['seed']} "
                    f"is_legacy={r['is_legacy_adapter']} turns={r['turns_seen']} "
                    f"actions={r['total_actions']} (end={r['end_turn']} "
                    f"play={r['play_card']} atk={r['attack']}) "
                    f"exec_fail={r['execute_failed']} decoded_not_legal={r['decoded_not_in_legal']} "
                    f"shim_mut={r['shim_mutated']} tvid_bad={r['train_v2_id_not_in_shim_legal']}"
                )
                if r["failures"]:
                    for f in r["failures"][:6]:
                        print(f"        - {f}")
                    all_failures.extend(
                        f"{r['model']}/seed{r['seed']}: {x}" for x in r["failures"]
                    )
    print("=" * 78)
    if all_failures:
        print(f"TOTAL FAILURES: {len(all_failures)}")
        for f in all_failures[:40]:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())