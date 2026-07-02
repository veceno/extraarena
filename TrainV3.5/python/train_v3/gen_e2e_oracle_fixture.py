"""Generate a true end-to-end Python-oracle golden fixture
``golden_trace_e2e_oracle.json``.

Unlike the legacy fixtures (whose state trajectory was frozen against the
DEFAULT deck pool and can therefore only be RE-ENODED, never re-simulated,
because Phase 5 added card 51 and shifted the default-deck draw outcomes),
this fixture is generated ENTIRELY from the Python oracle via
``golden_trace.build_golden_trace`` with an EXPLICIT deck override
(``--p1-deck-ids`` / ``--p2-deck-ids``), so re-simulation is legal and
deterministic.

The scenario is a multi-mechanic state that INCLUDES a minion attack
(att_pos >= 0) plus two ported mechanics:
  * ``taunt``                      — p1 board (card 30 Наофуми)
  * ``deathrattle_aoe_damage_2``   — p2 board (card 34 Крипер)
The minion attack is step t6: action_id 545 = _ATTACK_BASE(545) +
att_idx(0)*_NUM_ATTACK_TARGETS(8) + tcode(0)  →  p1 board[0] (taunt) attacks
p2 board[0] (deathrattle). att_pos = 0 (>= 0).

Single-card-type decks (p1 all 30, p2 all 34) make the end-of-turn draw
deterministic across RNG modalities: the drawn CARD is always card 30 / 34
regardless of which deck index the weighted draw picks, and the deck zone is
encoded in obs only as a SUMMARY (count + mean/max stats + mechanic tallies —
classic_obs_v1._encode_one_zone), so the obs / mask / action_features are
identical under both Python MT19937 and the Rust ``WorkerRng::Deterministic``
zero-RNG that ``assert_trace_transitions_match`` uses (golden_kernel.rs:1524).
The full matcher compares obs_v1 + obs_v5 + mask + action_features +
reward_components byte-level — locking in att_pos (/6.0) + the modality closure.

Usage (from the worktree TrainV3.5 dir):
  PYTHONPATH=<worktree-root>:<worktree-root>/TrainV3.5/python \
      python3 -m train_v3.gen_e2e_oracle_fixture
"""
from __future__ import annotations

import json
from pathlib import Path

from train_v3.contracts import InfoModeV5
from train_v3.golden_trace import build_golden_trace

OUT = Path("rust/trainv3_core/tests/fixtures/golden_trace_e2e_oracle.json")

SEED = 11
P1_DECK = [30] * 8  # Наофуми — taunt (cost 3)
P2_DECK = [34] * 8  # Крипер — deathrattle_aoe_damage_2 (cost 3)
MANA_PER_TURN = 4
STEPS = 8
# t0 p1 end_turn(0)            — mana=1, cannot afford cost-3
# t1 p2 play Крипер hand[0] pos0 (aid=1)
# t2 p2 end_turn(0)
# t3 p1 play Наофуми hand[0] pos0 (aid=1)
# t4 p1 end_turn(0)
# t5 p2 end_turn(0)            — p2 declines to attack, keeps Крипер ready
# t6 p1 attack board[0] -> p2 board[0] (aid=545)   <-- minion attack, att_pos=0
# t7 p1 end_turn(0)
ACTION_IDS = [0, 1, 0, 1, 0, 0, 545, 0]


def main() -> None:
    info_mode = InfoModeV5(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=False,
    )
    trace = build_golden_trace(
        seed=SEED,
        steps=STEPS,
        placement_mode="append_only",
        verify_mask=False,
        include_preview=False,
        include_v5=True,
        info_mode=info_mode,
        v5_weighted_reward=True,
        choose="first",
        p1_deck_ids=P1_DECK,
        p2_deck_ids=P2_DECK,
        action_ids=ACTION_IDS,
        mana_per_turn=MANA_PER_TURN,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"[e2e-oracle] wrote {OUT} ({len(trace['steps'])} steps)")
    # Sanity: confirm the attack step is present and att_pos >= 0.
    attack_steps = [s for s in trace["steps"] if 545 <= s["action_id"] <= 600]
    assert attack_steps, "no attack action_id in trace"
    print(f"[e2e-oracle] attack step(s): {[(s['t'], s['action_id']) for s in attack_steps]}")
    # Confirm both mechanics appear in some post-state board.
    mechs_seen = set()
    for s in trace["steps"]:
        for p in ("p1", "p2"):
            for c in s["post"]["state"][p]["board"]:
                mechs_seen.update(c["mechanics"])
    print(f"[e2e-oracle] mechanics present in post-states: {sorted(mechs_seen)}")
    assert "taunt" in mechs_seen and any(m.startswith("deathrattle") for m in mechs_seen), \
        f"expected taunt + deathrattle mechanics, got {sorted(mechs_seen)}"


if __name__ == "__main__":
    main()