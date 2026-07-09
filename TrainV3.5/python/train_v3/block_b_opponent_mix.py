"""Block B component B3 -- ``block_b_opponent_mix.py`` -- the full Block-B opponent
mix composition + the v4-orig alias layer + the ``punish_empty_board`` dispatch
enable + the D-B5 hybrid self-snapshot prevalence + the mana_draw-collapse reweight
entry point (NEW).

V5-Max pipeline position: Block A in-worktree COMPLETE -> Block B; B1
(``snapshot_pool.py``) DONE + fixed (frozen_non_self_share 0.75 -> 0.95, the
spec-literal frozen non-self = V4-orig 0.75 + exploit 0.15 + tail 0.05) -> B2
(``v4_orig_temp_spectrum.py``) DONE -> this file is B3.

PURPOSE (``BLOCK_B_PLAN.md:316-392``): the full Block-B opponent mix =
  * self-snapshots (from B1 pool, D-B5 hybrid dynamic weight, grown 0 -> 0.05 as the
    pool fills, cap 0.05 spec-literal after the B1 fix);
  * V4-orig spectrum (B2, 0.40 / 0.20 / 0.15 frozen RATIOS);
  * exploit-lanes continuous (stall / anti_draw_greed / punish_empty_board, 0.05
    each);
  * tail (greedy_face 0.03, random 0.01, end_turn 0.01) -- reweighted from the
    Phase-A 0.10 / 0.05 / 0.10.

Two DISTINCT gaps are closed here (do NOT conflate them, ``BLOCK_B_PLAN.md:326-346``):

  (a) Parse-side canonical set (the v4-orig-t07 / v4-orig-t12 gap):
      ``league_v5.parse_v5_opponent_mix`` (``league_v5.py:43-60``) rejects names not
      in ``V5_OPPONENT_KINDS`` (``league_v5.py:12-21``). ``v4-orig-t07`` /
      ``v4-orig-t12`` are genuinely ABSENT from ``V5_OPPONENT_KINDS`` (only ``v4max``
      is present, ``league_v5.py:15``). ``league_v5.py`` is frozen-classic READ-ONLY
      (A3/A5 pattern; do NOT edit ``V5_OPPONENT_KINDS`` -- ``BLOCK_B_PLAN.md:875-
      880``). So B3 builds its OWN canonical set + validator
      (``parse_block_b_opponent_mix``) mirroring ``parse_v5_opponent_mix``'s logic
      but with the EXTENDED ``BLOCK_B_IDENTITIES`` superset. ``v4-orig-*`` names
      resolve to B2 identities via ``V4_ORIG_TEMP_ALIASES`` (``v4_orig_temp_spectrum
      .py:145``). B3 does NOT route ``v4-orig-*`` through ``parse_v5_opponent_mix``
      (would raise). For the ``V5_OPPONENT_KINDS`` names, B3's validator accepts
      them directly (its own set is a superset).

  (b) Dispatch-enable for ``punish_empty_board`` (the A4 uncomment):
      ``punish_empty_board`` PARSES (it is in ``V5_OPPONENT_KINDS`` via
      ``*EXPLOIT_AGENT_KINDS``, ``gauntlet_v5.py:13`` / ``league_v5.py:20``) but did
      NOT DISPATCH -- A4 ``resolve_opponent_dispatch`` (``rust_live_self_play.py:
      166-182``) raised ``ValueError`` because Rust rule code 5 was commented out in
      ``RULE_AGENT_CODES`` (``rust_live_self_play.py:143``). The A4 uncomment (this
      workflow EDIT 2) additively enables it: ``resolve_opponent_dispatch(
      "punish_empty_board")`` now returns ``(RULE_DISPATCH, 5)``. Zero Rust change
      (``worker.rs:1258 ExploitAgentKind::PunishEmptyBoard`` already exists). This
      is an A4-file edit (NOT frozen-classic; A4 is Block-A code, editable per
      ``BLOCK_B_PLAN.md:884-888``).

D-B5 HYBRID (CONFIRMED, user 2026-07-01 -- ``BLOCK_B_PLAN.md:151-154`` + §2): the
spec (``design.md:118``) FREEZES absolute weights -- V4-orig 0.40+0.20+0.15 = 0.75,
exploit 0.05*3 = 0.15, tail 0.03+0.01+0.01 = 0.05; FROZEN non-self TOTAL = 0.95.
    self-snapshot share = residual 1.0 - 0.75 = 0.25, GROWN as the pool fills
(prevalence rises with pool size, B1 ``self_snapshot_prevalence_weight``), with a
mana_draw-collapse monitor (B2 hook ``v4_orig_temp_spectrum.py:158-176``, B4 logic)
that can BOOST self-snapshot above 0.05 (compressing frozen non-self
proportionally) on out-of-band vs V4-orig lanes. B3 EXPOSES the boost entry point
(``collapse_reweight_boost`` + ``build_block_b_opponent_mix(collapse_boost=...)``);
the monitor LOGIC is B4 (deferred).

CONSTRAINTS (frozen-classic guard, ``BLOCK_B_PLAN.md:860-901``): B3 is a NEW file.
NO edit to ``classic_*/reward_v5/v5_trace/warm_start_v5/run_phase26*/run_v5_accept
ance/league_v5.py/gauntlet_v5.py/opponents_v5.py`` (all read-only). EDITABLE in this
workflow: B1 ``snapshot_pool.py`` (the 0.75 -> 0.95 fix + test, EDIT 1), A4
``rust_live_self_play.py`` (the :143 uncomment, EDIT 2), B3 NEW file (EDIT 3). NO
Rust edit (``worker.rs:1258 PunishEmptyBoard`` already exists; NO cargo re-run). NO
TrainV3.5-into-prod. Source-vs-source: A4 ``resolve_opponent_dispatch`` +
``league_v5.parse_v5_opponent_mix`` logic = oracle, B3 validator = UUT (mirror the
parse logic, do NOT re-invent); avoid self-referential fixture regen. Synthetic
tests only (fake pool / fake SnapshotPool with ``self_snapshot_prevalence_weight``;
no real MLX/Rust/ONNX).
"""

from __future__ import annotations

from typing import Any, Protocol

# B2 V4-orig temperature-spectrum identities + the canonical-name alias map
# (``v4_orig_temp_spectrum.py:118-147``). Re-exported so the v4-orig-* names resolve
# to B2 identities through ONE alias map (the alias layer lives in B3, NOT in
# ``league_v5.V5_OPPONENT_KINDS`` -- ``BLOCK_B_PLAN.md:879-880``).
from .v4_orig_temp_spectrum import (
    V4_ORIG_ARGMAX,
    V4_ORIG_T07,
    V4_ORIG_T12,
    V4_ORIG_TEMP_ALIASES,
    V4_ORIG_TEMP_IDENTITIES,
    V4_ORIG_TEMP_WEIGHTS,
)

# =============================================================================
# Canonical identity set (the B3 superset -- a STRICT superset of
# ``league_v5.V5_OPPONENT_KINDS`` for the Block-B names + the v4-orig-* names).
# =============================================================================
# ``league_v5.V5_OPPONENT_KINDS`` (``league_v5.py:12-21``) = {self, v5_snapshot,
# v4max, random, greedy_face, end_turn, llm_teacher, *EXPLOIT_AGENT_KINDS} where
# EXPLOIT_AGENT_KINDS (``gauntlet_v5.py:8-16``) = {face_rush, board_control,
# greedy_trade, stall, punish_empty_board, anti_draw_greed, anti_hand_leak_overfit}.
# So stall / anti_draw_greed / punish_empty_board / greedy_face / random / end_turn
# / self / v5_snapshot ALL parse via ``parse_v5_opponent_mix``. BUT v4-orig-argmax /
# v4-orig-t07 / v4-orig-t12 are NOT in V5_OPPONENT_KINDS (only 'v4max' is). B3's own
# canonical set adds the three v4-orig-* names (resolved to B2 identities) and drops
# the Block-B-irrelevant names (v4max is superseded by the v4-orig-* spectrum;
# llm_teacher / face_rush / board_control / greedy_trade / anti_hand_leak_overfit
# are not in the Block-B mix). B3's validator accepts the V5_OPPONENT_KINDS names
# that appear in the Block-B mix directly (its set is a superset for those names).
BLOCK_B_IDENTITIES: frozenset[str] = frozenset({
    "self",
    "v5_snapshot",
    "v4-orig-argmax",
    "v4-orig-t07",
    "v4-orig-t12",
    "stall",
    "anti_draw_greed",
    "punish_empty_board",
    "greedy_face",
    "random",
    "end_turn",
})

# -----------------------------------------------------------------------------
# Frozen D-B5 weights (``design.md:118``, ``BLOCK_B_PLAN.md:142-154``). The D-B5
# hybrid collapse monitor (B4) may reweight at runtime via ``collapse_boost``; these
# are the frozen spec-literal absolute weights (the within-group RATIOS are
# load-bearing -- ``test_frozen_ratios_preserved``).
# -----------------------------------------------------------------------------
# V4-orig spectrum: keep the B2 0.40:0.20:0.15 ratio, but reduce the frozen
# absolute share from 0.75 to 0.55 so the blind V4 lane stays a pressure lane
# rather than the dominant training distribution.
BLOCK_B_V4_ORIG_WEIGHTS: dict[str, float] = {
    "v4-orig-argmax": 0.55 * (0.40 / 0.75),
    "v4-orig-t07": 0.55 * (0.20 / 0.75),
    "v4-orig-t12": 0.55 * (0.15 / 0.75),
}
BLOCK_B_V4_ORIG_TOTAL = sum(BLOCK_B_V4_ORIG_WEIGHTS.values())  # 0.55

# Exploit-lanes continuous: stall / anti_draw_greed / punish_empty_board, 0.05 each.
BLOCK_B_EXPLOIT_WEIGHTS: dict[str, float] = {
    "stall": 0.05,
    "anti_draw_greed": 0.05,
    "punish_empty_board": 0.05,
}
BLOCK_B_EXPLOIT_TOTAL = 0.15  # 0.05 * 3

# Tail: greedy_face 0.03 / random 0.01 / end_turn 0.01 (reweighted from Phase-A
# 0.10 / 0.05 / 0.10 -- ``BLOCK_B_PLAN.md:348-349``).
BLOCK_B_TAIL_WEIGHTS: dict[str, float] = {
    "greedy_face": 0.03,
    "random": 0.01,
    "end_turn": 0.01,
}
BLOCK_B_TAIL_TOTAL = 0.05  # 0.03 + 0.01 + 0.01

#: Frozen non-self TOTAL (V4-orig 0.55 + exploit 0.15 + tail 0.05). The self-snapshot
#: share is the RESIDUAL ``1 - FROZEN_NON_SELF_TOTAL`` = 0.25, grown as the pool
#: fills (B1 ``self_snapshot_prevalence_weight``). This intentionally follows the
#: Q5 mitigation from the design/handoff notes: keep the blind V4-orig lane modest
#: and self-snapshot prevalence high.
FROZEN_NON_SELF_TOTAL: float = 0.75

#: The residual self-snapshot share when the pool is full.
SELF_SNAPSHOT_SHARE_CAP: float = 1.0 - FROZEN_NON_SELF_TOTAL  # 0.25

#: Upper bound for the collapse-boosted self-snapshot share (the boost may compress
#: frozen non-self but never to zero -- a 0.05 non-self floor is kept so the V4-orig
#: + exploit + tail lanes always remain in the mix). B4 wires the monitor signal;
#: B3 only enforces the floor.
_MAX_SELF_SHARE: float = 0.95

# The frozen within-group RATIOS (group_weight / FROZEN_NON_SELF_TOTAL of the
# non-self budget). These are the proportions used to distribute ``non_self_budget``
# across the three groups; within each group the identity ratios are preserved.
_V4_ORIG_GROUP_RATIO = BLOCK_B_V4_ORIG_TOTAL / FROZEN_NON_SELF_TOTAL    # 0.75/0.95
_EXPLOIT_GROUP_RATIO = BLOCK_B_EXPLOIT_TOTAL / FROZEN_NON_SELF_TOTAL    # 0.15/0.95
_TAIL_GROUP_RATIO = BLOCK_B_TAIL_TOTAL / FROZEN_NON_SELF_TOTAL          # 0.05/0.95

# -----------------------------------------------------------------------------
# Alias map for the v4-orig-t07 / v4-orig-t12 parse gap (``BLOCK_B_PLAN.md:326-332,
# 390-392``). Covers ONLY the v4-orig-* names (resolves them to B2 spectrum
# identities). ``punish_empty_board`` does NOT need an alias -- it parses natively
# via ``V5_OPPONENT_KINDS`` / ``*EXPLOIT_AGENT_KINDS``. The alias map lives in B3,
# NOT in ``league_v5.V5_OPPONENT_KINDS`` (``BLOCK_B_PLAN.md:879-880``).
# -----------------------------------------------------------------------------
BLOCK_B_V4_ORIG_ALIASES: dict[str, str] = dict(V4_ORIG_TEMP_ALIASES)


def resolve_v4_orig_identity(name: str) -> str:
    """Resolve a ``v4-orig-*`` canonical name to its B2 spectrum identity name.

    ``V4_ORIG_TEMP_ALIASES`` (``v4_orig_temp_spectrum.py:145``) is the identity-to-
    canonical map (self-describing -- each identity's canonical name IS its name).
    This helper is the B3 alias-resolution entry point: given ``v4-orig-argmax`` /
    ``v4-orig-t07`` / ``v4-orig-t12``, return the B2 identity name. Raises
    ``KeyError`` for a name NOT in the alias map (so callers do not silently alias
    ``punish_empty_board`` -- it parses natively, no alias needed,
    ``test_alias_map_covers_only_t07_t12``).
    """
    if name not in BLOCK_B_V4_ORIG_ALIASES:
        raise KeyError(
            f"not a v4-orig alias: {name!r} (the B3 alias map covers ONLY "
            f"v4-orig-argmax / v4-orig-t07 / v4-orig-t12; punish_empty_board parses "
            f"natively via V5_OPPONENT_KINDS)"
        )
    return BLOCK_B_V4_ORIG_ALIASES[name]


# =============================================================================
# Pool protocol (the B3 surface B3 reads from B1; synthetic tests use a fake pool).
# =============================================================================
class _BlockBPool(Protocol):
    """The B3 read-surface on B1 ``SnapshotPool``: just the D-B5 prevalence weight.

    B3 does NOT require the full ``SnapshotPool`` -- only
    ``self_snapshot_prevalence_weight()`` (B1, grown 0 -> 0.05 as the pool fills).
    Synthetic tests pass a tiny fake object exposing this one method (no real
    SnapshotPool / MLX / Rust required).
    """

    def self_snapshot_prevalence_weight(self) -> float: ...


# =============================================================================
# Validator -- mirrors ``league_v5.parse_v5_opponent_mix`` (``league_v5.py:43-60``)
# logic but accepts ``BLOCK_B_IDENTITIES`` (the superset). Source-vs-source UUT:
# ``parse_v5_opponent_mix`` = oracle, this validator = UUT (mirror the parse logic,
# do NOT re-invent).
# =============================================================================
def parse_block_b_opponent_mix(raw: str) -> list[tuple[str, float]]:
    """Parse a Block-B opponent-mix string into a ``[(identity, weight)]`` list.

    Mirrors ``league_v5.parse_v5_opponent_mix`` (``league_v5.py:43-60``) exactly:
      * split on ``,``, strip each part, skip empty;
      * ``name:weight`` (weight = ``float``) or bare ``name`` (weight = 1.0);
      * skip ``weight <= 0.0``;
      * raise ``ValueError`` on an unknown identity (NOT in ``BLOCK_B_IDENTITIES``);
      * default ``"self:1.0"`` when the raw string is empty / all-skipped.

    The ONLY difference from ``parse_v5_opponent_mix`` is the canonical set: B3
    accepts the EXTENDED ``BLOCK_B_IDENTITIES`` (superset with v4-orig-argmax /
    v4-orig-t07 / v4-orig-t12) instead of ``V5_OPPONENT_KINDS``. The v4-orig-* names
    do NOT route through ``parse_v5_opponent_mix`` (would raise -- they are absent
    from ``V5_OPPONENT_KINDS``); B3's own validator accepts them directly.
    """
    out: list[tuple[str, float]] = []
    for raw_part in (raw or "self:1.0").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight_s = part.split(":", 1)
            weight = float(weight_s)
        else:
            name, weight = part, 1.0
        name = name.strip()
        if weight <= 0.0:
            continue
        if name not in BLOCK_B_IDENTITIES:
            raise ValueError(f"unknown Block-B opponent kind: {name}")
        out.append((name, weight))
    return out or [("self", 1.0)]


# =============================================================================
# Mix builder (D-B5 hybrid, always sums to 1.0)
# =============================================================================
def _self_snapshot_split(self_snapshot_weight: float) -> list[tuple[str, float]]:
    """Split the self-snapshot weight across the self-play identities.

    When ``self_snapshot_weight > 0`` the share is split equally between ``self``
    (the live learner playing itself) and ``v5_snapshot`` (a prior pool snapshot
    loaded back via B1 ``load_as_self_prev_opponent_select_fn``). When the pool is
    empty (``self_snapshot_weight == 0``) both get zero -- the non-self budget is
    the full 1.0. Both identities are always emitted (with zero weight when the
    pool is empty) so the mix identity set is stable; zero-weight entries are
    skipped by ``parse_block_b_opponent_mix``.
    """
    half = self_snapshot_weight / 2.0
    return [("self", half), ("v5_snapshot", half)]


def build_block_b_opponent_mix(
    pool: _BlockBPool, *, collapse_boost: float = 1.0
) -> list[tuple[str, float]]:
    """Build the full Block-B opponent mix (D-B5 hybrid, always sums to 1.0).

    Composition:
      * ``self_snapshot_weight = pool.self_snapshot_prevalence_weight()`` (B1,
        grown 0 -> 0.25 as the pool fills). With
        ``collapse_boost > 1.0`` (the mana_draw-collapse monitor, B4) the
        self-snapshot share is RAISED above 0.25 (compressing frozen non-self
        proportionally), capped at ``_MAX_SELF_SHARE`` (0.95 -- a 0.05 non-self
        floor is kept so the V4-orig + exploit + tail lanes always remain).
      * ``non_self_budget = 1 - self_snapshot_weight``, distributed in FROZEN
        RATIOS: V4-orig ``0.55/0.75``, exploit ``0.15/0.75``, tail ``0.05/0.75`` of
        ``non_self_budget``.
      * WITHIN V4-orig the 0.40 : 0.20 : 0.15 ratio is preserved; within exploit
        0.05 : 0.05 : 0.05 (equal); within tail 0.03 : 0.01 : 0.01.

    The mix ALWAYS sums to 1.0 (``test_weights_account_to_one``). The frozen
    within-group RATIOS are preserved at any pool size
    (``test_frozen_ratios_preserved``).
    """
    if collapse_boost <= 0.0:
        raise ValueError("collapse_boost must be > 0")
    base = float(pool.self_snapshot_prevalence_weight())
    self_snapshot_weight = min(base * float(collapse_boost), _MAX_SELF_SHARE)
    # Clamp non-negative (a misbehaving pool could return a tiny negative).
    if self_snapshot_weight < 0.0:
        self_snapshot_weight = 0.0
    non_self_budget = 1.0 - self_snapshot_weight

    mix: list[tuple[str, float]] = []
    mix.extend(_self_snapshot_split(self_snapshot_weight))

    # V4-orig group (within-group 0.40 : 0.20 : 0.15 ratio preserved).
    v4_orig_budget = non_self_budget * _V4_ORIG_GROUP_RATIO
    for name, frozen_w in BLOCK_B_V4_ORIG_WEIGHTS.items():
        mix.append((name, v4_orig_budget * (frozen_w / BLOCK_B_V4_ORIG_TOTAL)))

    # Exploit group (within-group 0.05 : 0.05 : 0.05 equal).
    exploit_budget = non_self_budget * _EXPLOIT_GROUP_RATIO
    for name, frozen_w in BLOCK_B_EXPLOIT_WEIGHTS.items():
        mix.append((name, exploit_budget * (frozen_w / BLOCK_B_EXPLOIT_TOTAL)))

    # Tail group (within-group 0.03 : 0.01 : 0.01).
    tail_budget = non_self_budget * _TAIL_GROUP_RATIO
    for name, frozen_w in BLOCK_B_TAIL_WEIGHTS.items():
        mix.append((name, tail_budget * (frozen_w / BLOCK_B_TAIL_TOTAL)))

    return mix


def collapse_reweight_boost(factor: float) -> dict[str, float]:
    """The mana_draw-collapse reweight ENTRY POINT (D-B5 hybrid, ``BLOCK_B_PLAN.md:
    373-375``).

    Given a boost ``factor`` (> 0), returns a reweight config that RAISES the
    self-snapshot share above the 0.05 spec-literal cap (compressing frozen non-self
    proportionally) when passed to ``build_block_b_opponent_mix`` as
    ``collapse_boost``. A ``factor`` of 1.0 is the identity (no boost -- the
    spec-literal frozen mix); > 1.0 boosts self-snapshot prevalence.

    B3 ONLY EXPOSES this entry point. The mana_draw-collapse monitor LOGIC that
    decides WHEN to boost (the learner's mana_draw usage vs V4-orig lanes drops out
    of the A5 band) lives in B4 (deferred -- ``BLOCK_B_PLAN.md:281-284``). B4 wires
    its monitor signal to this entry point: ``build_block_b_opponent_mix(pool,
    **collapse_reweight_boost(factor))``.
    """
    if float(factor) <= 0.0:
        raise ValueError("collapse_boost factor must be > 0")
    return {"collapse_boost": float(factor)}


def build_block_b_mix_string(
    pool: _BlockBPool, *, collapse_boost: float = 1.0
) -> str:
    """Render the Block-B mix as a parseable mix string (for validation via
    ``parse_block_b_opponent_mix`` + for A4 ``sample_opponent_identities``
    ``rust_live_self_play.py:471``).

    The string is ``"name:weight,name:weight,..."`` with high-precision weights so
    the round-trip through ``parse_block_b_opponent_mix`` preserves the accounting
    to 1.0 (within float tolerance). Zero-weight entries are emitted (the parser
    skips them) so the identity set is stable.
    """
    mix = build_block_b_opponent_mix(pool, collapse_boost=collapse_boost)
    return ",".join(f"{name}:{weight:.10f}" for name, weight in mix)


__all__ = [
    "BLOCK_B_EXPLOIT_TOTAL",
    "BLOCK_B_EXPLOIT_WEIGHTS",
    "BLOCK_B_IDENTITIES",
    "BLOCK_B_TAIL_TOTAL",
    "BLOCK_B_TAIL_WEIGHTS",
    "BLOCK_B_V4_ORIG_ALIASES",
    "BLOCK_B_V4_ORIG_TOTAL",
    "BLOCK_B_V4_ORIG_WEIGHTS",
    "FROZEN_NON_SELF_TOTAL",
    "SELF_SNAPSHOT_SHARE_CAP",
    "V4_ORIG_ARGMAX",
    "V4_ORIG_T07",
    "V4_ORIG_T12",
    "V4_ORIG_TEMP_ALIASES",
    "V4_ORIG_TEMP_IDENTITIES",
    "V4_ORIG_TEMP_WEIGHTS",
    "build_block_b_mix_string",
    "build_block_b_opponent_mix",
    "collapse_reweight_boost",
    "parse_block_b_opponent_mix",
    "resolve_v4_orig_identity",
]
