"""Dump real onboarding-tutorial engine output as Playwright e2e fixtures.

Drives the real ``TutorialBattleEngine`` (graph-driven, prod code path — no
extra_orchestra import) through the full happy-path flow AND the choose-target
wrong-path (tap the opponent minion → tutorial_wrong_action → retry the hero),
and writes the exact server-shaped JSON the arena receives for
``/api/battle/state`` and every ``/api/onboarding/tutorial/action`` request. The
node Playwright script mocks those endpoints with this real data and drives the
real ``webapp/arena.js`` frontend, so the e2e validates: real engine state → real
Midoria dialogs + board render + wrong-target redirect + victory/menu-tour
handoff. No ``visual_input``.

Run from the worktree root:
    python3 tests/e2e/_dump_onboarding_fixtures.py
writes ``tests/e2e/onboarding_fixtures.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Worktree root is where onboarding_tutorial.py + cards.json live.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboarding_tutorial import (
    TUTORIAL_FINAL_STEP,
    TutorialBattleEngine,
    tutorial_match_id_for_user,
)

USER_ID = 12345
MATCH_ID = tutorial_match_id_for_user(USER_ID)
OUT = Path(__file__).resolve().parent / "onboarding_fixtures.json"


def _state(engine: TutorialBattleEngine) -> dict:
    return engine.get_full_state(viewer_id=USER_ID)


def _onboarding_payload(tutorial_step: int, status: str = "tutorial_battle") -> dict:
    """Minimal onboarding envelope the arena actually reads.

    The arena only inspects ``onboarding.status`` on the ``complete`` response
    (``finishOnboardingTutorialForMenu``); non-complete responses ignore it.
    """
    return {"status": status, "tutorial_step": int(tutorial_step)}


def _wrap(result: dict, engine: TutorialBattleEngine, *, status: str = "tutorial_battle", http_status: int = 200) -> dict:
    payload = {
        "match_id": MATCH_ID,
        "result": result,
        "state": _state(engine),
        "onboarding": _onboarding_payload(engine.tutorial_step, status=status),
    }
    if result.get("feedback"):
        payload["feedback"] = result["feedback"]
    # The e2e HTTP mock reads _status to set the response code (the real server
    # returns 409 for tutorial_wrong_action). The client ignores this field.
    if http_status != 200:
        payload["_status"] = http_status
    return payload


def _attacker_id(engine: TutorialBattleEngine) -> str:
    return engine.tutorial_payload().get("attacker_instance_id") or ""


def _opponent_minion_id(engine: TutorialBattleEngine) -> str:
    """Стив's instance id on the opponent board (for the choose-target wrong path)."""
    board = engine._arena.state.p2.board if engine._arena else []
    return str(board[0].instance_id) if board else ""


def main() -> None:
    engine = TutorialBattleEngine(user_id=USER_ID)
    responses: list[dict] = []

    # step0 — initial /api/battle/state response.
    step0_state = _state(engine)

    # 1) continue (step0 goal -> step1 play_attacker)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "continue"}), engine))

    # 2) play Слайм (step1 play_attacker -> step2 sleep)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "play_card", "card_id": 37, "hand_index": 0}), engine))

    # 3) end turn (step2 sleep -> p2 plays Стив -> step3 threat)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "end_turn"}), engine))

    # 4) continue (step3 threat -> p2 ends -> step4 choose_target)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "continue"}), engine))

    # 5) WRONG PATH: tap Стив at choose_target -> tutorial_wrong_action (409), no state change
    wrong = engine.apply_tutorial_action(
        {"type": "attack", "attacker_id": _attacker_id(engine), "target_id": _opponent_minion_id(engine), "target_is_hero": False}
    )
    responses.append(_wrap(wrong, engine, http_status=409))

    # 6) attack hero (step4 choose_target -> step5 tempo, 8->4)
    responses.append(_wrap(
        engine.apply_tutorial_action({"type": "attack", "attacker_id": _attacker_id(engine), "target_is_hero": True}),
        engine,
    ))

    # 7) continue (step5 tempo -> step6 danger)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "continue"}), engine))

    # 8) continue (step6 danger -> step7 taunt_intro)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "continue"}), engine))

    # 9) play Альфонс (step7 taunt_intro -> p1 ends -> step8 taunt_demo)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "play_card", "card_id": 39, "hand_index": 0}), engine))

    # 10) auto_continue (step8 taunt_demo -> Стив attacks Альфонс/dies -> p2 ends -> step9 lethal)
    responses.append(_wrap(engine.apply_tutorial_action({"type": "auto_continue"}), engine))

    # 11) attack hero lethal (step9 lethal -> step10 victory, real P1_WIN 4->0)
    responses.append(_wrap(
        engine.apply_tutorial_action({"type": "attack", "attacker_id": _attacker_id(engine), "target_is_hero": True}),
        engine,
    ))

    # 12) complete -> menu_tour (no `state` in the server response)
    complete_result = engine.apply_tutorial_action({"type": "complete"})
    responses.append({
        "match_id": MATCH_ID,
        "result": complete_result,
        "redirect_url": "/?onboarding_menu=1",
        "onboarding": _onboarding_payload(TUTORIAL_FINAL_STEP, status="menu_tour"),
    })

    fixtures = {
        "user_id": USER_ID,
        "match_id": MATCH_ID,
        "final_step": TUTORIAL_FINAL_STEP,
        "step0_state": step0_state,
        "responses": responses,
    }
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(responses)} action responses, final_step={TUTORIAL_FINAL_STEP})")


if __name__ == "__main__":
    main()