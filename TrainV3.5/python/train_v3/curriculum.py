"""B4 — per-lane learner loss tracker + adaptive curriculum reweight.

Design reference: ``design.md:119`` "oversample the lane the learner is losing to".
A4 ``sample_opponent_identities`` (``rust_live_self_play.py:476``) samples from a
STATIC parsed mix each update; B4 makes that mix DYNAMIC by aggregating per-identity
learner win/loss from a real A4 ``LiveRolloutBatch`` (``rust_live_self_play.py:419``)
and reweighting the mix toward lanes the learner is losing to, proportional to the
loss margin, capped at ~25%/update (D-B8, ``BLOCK_B_PLAN.md:156``).

SCOPE: B4 is per-lane-loss curriculum ONLY. The mana_draw-collapse monitor is NOT
here (B2 exposes the hook, B3 ``collapse_reweight_boost`` consumes it; the
mana_draw band is A5 ``check_mana_draw_band`` consumed by B6; monitor wiring belongs
to the B8 driver).

B4 PRODUCES a reweighted mix (``CurriculumReweighter.reweight``) that the B8 driver
Wires into A4 ``sample_opponent_identities``'s ``opponent_mix`` arg. B4 does NOT edit
A4's sampler nor B3's mix builder — both are consumed read-only.

Outcome derivation (the adapter): for each env ``i`` in the batch, the opponent
identity is ``rollout.opponent_identities[i]`` (``rust_live_self_play.py:435``, one
canonical identity per env, CONSTANT per env across the batch) and the learner's net
attributed reward is ``rollout.transitions.rewards[:, i].sum()`` — A4's
``reward_attribution`` (``rust_live_self_play.py:783``) attributes rewards
LEARNER-ONLY (opponent-actor envs get ZERO reward, ``rust_live_self_play.py:786``),
so per-env reward-sum is the learner's net outcome for that env across its episodes
in the batch. ``dispatch_log`` (``rust_live_self_play.py:765``) carries NO win/loss
field, so outcomes are derived from ``transitions.rewards`` paired with
``opponent_identities``, NOT from dispatch_log entries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

__all__ = [
    "LaneOutcome",
    "extract_lane_outcomes",
    "CurriculumReweighter",
]


# --- Lane outcome -------------------------------------------------------------

@dataclass(frozen=True)
class LaneOutcome:
    """One env's learner outcome against a single opponent identity.

    ``identity`` is the canonical opponent identity for that env
    (``LiveRolloutBatch.opponent_identities[i]``). ``outcome`` is the learner's net
    result for env ``i`` across its episodes in the batch, derived from the sign of
    ``transitions.rewards[:, i].sum()``: ``"win"`` if the net attributed reward is
    positive, ``"loss"`` if negative, ``"draw"`` if exactly zero.
    """

    identity: str
    outcome: str  # one of {"win", "loss", "draw"}


def extract_lane_outcomes(rollout: Any) -> List[LaneOutcome]:
    """Adapter: derive per-env learner outcomes from a real ``LiveRolloutBatch``.

    For each env ``i``: ``identity = rollout.opponent_identities[i]`` and
    ``r = float(rollout.transitions.rewards[:, i].sum())`` — A4 attributes rewards
    learner-only (``rust_live_self_play.py:783-787``), so the per-env reward-sum is
    the learner's net outcome for that env. The outcome label is the sign of ``r``.

    Returns one ``LaneOutcome`` per env, in env order. Pure: does not mutate
    ``rollout``.
    """
    identities: Sequence[str] = rollout.opponent_identities
    rewards = rollout.transitions.rewards  # (steps, env_count) float32
    out: List[LaneOutcome] = []
    env_count = len(identities)
    for i in range(env_count):
        r = float(rewards[:, i].sum())
        if r > 0.0:
            outcome = "win"
        elif r < 0.0:
            outcome = "loss"
        else:
            outcome = "draw"
        out.append(LaneOutcome(identity=str(identities[i]), outcome=outcome))
    return out


# --- Curriculum reweighter ----------------------------------------------------

# Outcome label set (kept as a module constant for clarity / future callers).
_WIN = "win"
_LOSS = "loss"
_DRAW = "draw"
_OUTCOMES = frozenset({_WIN, _LOSS, _DRAW})


class CurriculumReweighter:
    """Per-lane learner-loss curriculum: oversample lanes the learner is losing to.

    Holds a rolling window of recent per-update lane-outcome aggregates. Each
    ``update(outcomes)`` call appends one update's aggregate (per-identity
    win/loss/draw counts). ``per_lane_loss_rate`` aggregates those counts over the
    window and returns ``losses / (wins + losses)`` per identity (draws excluded —
    they count neither as a win nor a loss; identities with zero decided envs default
    to ``0.5`` neutral, no boost). ``reweight(mix)`` boosts ONLY lanes the learner is
    losing to (``loss_rate > 0.5``) by ``1.0 + min(loss_rate - 0.5, cap)``
    (D-B8 adaptive, ``BLOCK_B_PLAN.md:156``), capped so the max multiplicative factor
    is ``1 + cap`` (default ``1.25x``), then renormalizes the mix to sum to exactly
    ``1.0``. Lanes the learner beats (``loss_rate <= 0.5``) and lanes with no window
    data get factor ``1.0`` (NO boost). ``reweight`` is pure (no input mutation).
    """

    def __init__(self, window_n: int) -> None:
        if int(window_n) <= 0:
            raise ValueError("window_n must be positive")
        self.window_n = int(window_n)
        # Each entry: dict[identity, {"win": int, "loss": int, "draw": int}].
        self._window: "deque[dict[str, dict[str, int]]]" = deque(maxlen=self.window_n)

    def update(self, outcomes: Sequence[LaneOutcome]) -> None:
        """Append one update's per-identity win/loss/draw aggregate to the window.

        Invalid outcome labels raise ``ValueError`` (defensive — the adapter only
        emits the three canonical labels).
        """
        agg: dict[str, dict[str, int]] = {}
        for lo in outcomes:
            if lo.outcome not in _OUTCOMES:
                raise ValueError(f"invalid outcome label: {lo.outcome!r}")
            counts = agg.setdefault(lo.identity, {_WIN: 0, _LOSS: 0, _DRAW: 0})
            counts[lo.outcome] += 1
        self._window.append(agg)

    def per_lane_loss_rate(self) -> dict[str, float]:
        """Aggregate over the rolling window: ``losses / (wins + losses)`` per identity.

        Draws are EXCLUDED from the denominator (they neither win nor lose).
        Identities with zero decided envs (``wins + losses == 0``) default to ``0.5``
        (neutral — no boost). Returns a dict mapping every identity seen in the
        window to its loss rate.
        """
        totals: dict[str, dict[str, int]] = {}
        for agg in self._window:
            for ident, counts in agg.items():
                t = totals.setdefault(ident, {_WIN: 0, _LOSS: 0, _DRAW: 0})
                t[_WIN] += counts[_WIN]
                t[_LOSS] += counts[_LOSS]
                t[_DRAW] += counts[_DRAW]
        rates: dict[str, float] = {}
        for ident, t in totals.items():
            decided = t[_WIN] + t[_LOSS]
            if decided == 0:
                rates[ident] = 0.5
            else:
                rates[ident] = t[_LOSS] / decided
        return rates

    def reweight(
        self,
        mix: List[Tuple[str, float]],
        *,
        cap: float = 0.25,
    ) -> List[Tuple[str, float]]:
        """Reweight ``mix`` toward lanes the learner is losing to.

        For each lane ``(name, weight)`` in ``mix``:
        - if ``loss_rate[name] > 0.5`` (learner is losing to it): boost factor =
          ``1.0 + min(loss_rate - 0.5, cap)`` (proportional to the loss margin,
          capped at ``cap`` so the max factor is ``1 + cap`` = ``1.25x`` by default);
        - else (learner beats it, ties it at 0.5, or no window data): factor ``1.0``
          (NO boost).

        The boosted weights are renormalized so the returned mix sums to exactly
        ``1.0``. PURE: returns a new list, does not mutate ``mix``. The cap prevents
        the mix collapsing to a single lane in one update (D-B8).
        """
        if not mix:
            raise ValueError("mix must contain at least one (name, weight) entry")
        rates = self.per_lane_loss_rate()
        boosted: List[Tuple[str, float]] = []
        for name, weight in mix:
            w = float(weight)
            lr = rates.get(name)
            if lr is None or lr <= 0.5:
                factor = 1.0
            else:
                factor = 1.0 + min(lr - 0.5, float(cap))
            boosted.append((name, w * factor))
        total = sum(w for _, w in boosted)
        if total <= 0.0:
            # Degenerate: all-zero weights — fall back to uniform over the mix.
            n = len(boosted)
            return [(name, 1.0 / n) for name, _ in boosted]
        return [(name, w / total) for name, w in boosted]