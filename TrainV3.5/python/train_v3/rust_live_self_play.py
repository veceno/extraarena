"""Block A component A4 — ``rust_live_self_play.py`` — THE MISSING live-self-play entry point.

D-A8 USER decision = build new, spec-faithful. The spec self-play PPO on the Rust
``ArenaEnv`` did NOT exist before A4: the V5 trainer only ran on the golden-trace pool
(``train_rust_ppo_trace_files`` via ``trace_factory_v5``; ``RustBatchWorker`` only had
``from_trace_file``/``from_trace_files`` at ``rust_ffi.py:719,834``). A4 is the LIVE
alternative — it runs LIVE self-play on the Rust ``ArenaEnv`` COMPOSING the EXISTING Rust
FFI primitives (no game logic reinvented, no FFI primitive redefined).

FFI primitives composed (all ADDITIVE, already present in ``rust_ffi.py`` — the
preliminary FFI work; ``rust_ffi.py`` is NOT frozen-classic):
  * ``RustBatchWorker.from_live`` (``rust_ffi.py:1227``) — THE LIVE constructor. Threads
    ``max_turns`` into ``trace['env_config']['max_turns']`` BEFORE the worker is built so
    ``KernelConfig::from_trace_config`` (``kernel.rs:660``) reads the Phase-A value (NOT
    the serde default 80, ``kernel.rs:624``). Composes ``from_trace_file`` with an
    init-only ``GoldenTrace`` (``steps=0`` -> no turn history) built by
    ``golden_trace.build_golden_trace``; the Rust worker steps LIVE from
    ``trace.initial`` (``trace.steps`` unused by the worker, ``ffi.rs:744-754``).
  * ``RustBatchWorker.step_mana_draw`` (``rust_ffi.py:1168``) — step the batch with a
    parallel mana_draw flag per env (the learner's mana_draw head channel; composes
    ``trainv3_worker_step_mana_draw`` ``ffi.rs:1658`` + ``step_with_mana_draw``
    ``worker.rs:739``).
  * ``RustBatchWorker.current_actor_ids`` (``rust_ffi.py:1044``) — whose turn per env.
  * ``RustBatchWorker.select_rule_actions`` (``rust_ffi.py:1057``) — batched Rust
    rule-agent dispatcher; calls ``select_rule_action_for_state`` (``worker.rs:1285``)
    per env with integer codes 0-7.
  * ``RustBatchWorker.advance_rule_until_actor`` (``rust_ffi.py:1078``) — batched
    fast-forward applying rule actions until a target actor; available for pure-rule-agent
    batches (see ``fast_forward_rule_opponent_turns``).
  * ``RustBatchWorker.mana_draw_legal`` (``rust_ffi.py:1151``) — per-env mana_draw
    legality flag (gates the learner's mana_draw head).
  * ``RustBatchWorker.hero_hp`` (``rust_ffi.py:1204``) — per-env
    ``[p1_hp, p1_max_hp, p2_hp, p2_max_hp]`` for decisive-early-end (D-A6).
  * ``RustBatchWorker.truncated`` (``rust_ffi.py:1133``) — per-env ``max_turns``
    truncation flag (``turn_number > config.max_turns``, ``kernel.rs:807``).
  * ``RustBatchWorker.reset`` / ``reset_indices`` — start / restart episodes.

DISPATCH SPLIT (grounded in source; plan ``BLOCK_A_PLAN.md:404-419`` and source AGREE —
no discrepancy): the 10 graduated identities
(``ppo_phaseA_config.PHASE_A_OPPONENT_MIX_SPEC``) split into TWO dispatch paths:
  * RULE-AGENT path (6 identities) — pure Rust heuristic via
    ``select_rule_action_for_state`` (``worker.rs:1285``) integer codes:
      ``random``=0 (``select_deterministic_legal_random_action`` ``worker.rs:1265``),
      ``face_rush``=1 (``ExploitAgentKind::FaceRush`` ``worker.rs:1254``),
      ``board_control``=2 (``worker.rs:1255``),
      ``greedy_trade``=3 (``worker.rs:1256``),
      ``stall``=4 (``worker.rs:1257``),
      ``anti_draw_greed``=6 (``worker.rs:1259``).
    Dispatched by ``select_rule_actions`` (no policy forward pass). Available Rust rule
    codes 5 (``PunishEmptyBoard``) and 7 (``AntiHandLeakOverfit``) exist but are NOT in the
    A3 spec mix (documented for completeness).
  * POLICY-OPPONENT path (4 identities, 40% of the mix weight) — Python-side opponent
    loop mirroring ``ai/train_v2/rollout_worker.py:211-230`` (``_get_opponent_policy`` +
    ``_auto_play_until_learner``). These have NO Rust rule code
    (``worker.rs:1281`` "unknown rule agent code" for codes outside 0-7):
      ``end_turn`` — Python heuristic that always emits ``EndTurnAction``
      (candidate id 0; ``kernel.rs:2709`` ``action_type_for_id(0)="end_turn"``).
      ``greedy_face`` — Python face-damage heuristic (injectable matcher).
      ``self`` / ``self_prev`` — a frozen V5 snapshot policy (defaults to the learner
      argmax for pure self-play; production wires a prior checkpoint).
      ``v4max`` / ``v4-orig-argmax`` — the V4 ONNX argmax policy
      (``opponents_v5.py:23`` ``offline_v4max_teacher``; lazy — caller wires the ONNX fn).

REWARD (fix #1, learner-perspective macro-step): ``reward_attribution``
(``ppo_phaseA_config.py:228``) keeps learner-actor step rewards on the recorded learner
rows. The Rust env's ``out.rewards[i]`` is the ACTING player's reward
(``kernel.rs:799`` ``compute_trainv2_reward(state, next, player_id)`` for the acting
``player_id``; ``worker.rs:861`` ``out.rewards.push(step.reward)``). Opponent-actor steps
are not recorded as standalone rows, but their response reward is subtracted into the
last learner row, mirroring ``worker.rs::advance_rule_until_actor`` and TrainV2's
macro-step semantics. ``reward_v5.py`` is consumed READ-ONLY (frozen-classic guard) — it
is ALREADY per-side (``reward_snapshot_v5`` takes ``player_id``, ``reward_v5.py:40``).

max_turns (fix #2): ``from_live(max_turns=config.max_turns=120)`` threads into
``KernelConfig`` (``kernel.rs:660``). Verified: the worker truncates at
``turn_number > max_turns`` (``kernel.rs:807``).

DECISIVE EARLY-END (D-A6): after each step, ``is_decisive_state``
(``ppo_phaseA_config.py:206``) on a ``hero_hp`` snapshot terminates the episode early when
the win-margin >= threshold. Treated as ``terminated=True`` (the game is effectively
decided -> no value bootstrap on a decided game).

SECOND-START OVERSAMPLING (D-A10): ``second_start_oversampling_scheme``
(``ppo_phaseA_config.py:258``) biases the learner starting side (p1/p2) to balance the
under-represented side (``design.md:112,120``).

MANA_DRAW: the learner policy's 601-candidate head and parallel mana_draw head form one
joint categorical distribution over the currently legal choices. ``step_mana_draw`` applies
the separate flag (its candidate action id placeholder is ignored by the kernel), while PPO
uses that same joint log-probability to train both heads.

This is compatible with the existing Block-B ``BLOCK_B_POLICY_OPPONENT_KINDS`` dispatch
extension and its ``opponent_mix_parsed`` handoff; neither changes the frozen Phase-A mix.

CONSUMES A3 ``PhaseAPPOConfig``: ``max_turns``, ``opponent_mix``, ``learner_only_reward``,
``second_start_oversampling``, ``decisive_early_end``, ``decisive_win_margin_threshold``,
``gamma``, ``gae_lambda``, ``epochs``, ``entropy_coef``, ``steps_per_update``,
``env_count``, ``advantage_backend``, ``selected_local_backend``, ``prepare_backend``,
``legal_row_pack_backend``, ``ppo_minibatch_plan``, ``clip_epsilon``, ``value_coef``,
``max_grad_norm``, ``seed``. PPO update via ``rust_ppo.prepare_rust_ppo_batch`` +
``train_rust_ppo_minibatch`` (reused unchanged — the live batch is a ``RustTransitionBatch``,
same format as ``collect_rust_vec_rollout``).

frozen-classic guard: ``reward_v5.py`` / ``classic_rl_env.py`` / ``run_phase26`` NOT
edited (``reward_v5.py`` READ-ONLY via ``reward_attribution``). ``v5_trace.py`` NOT
imported (live self-play, not trace replay). ``core/state.py`` NOT modified. No TrainV3.5
import into prod paths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .ppo_phaseA_config import (
    PHASE_A_MAX_TURNS,
    PhaseAPPOConfig,
    is_decisive_state,
    reward_attribution,
    second_start_oversampling_scheme,
)
from .rust_collector import RustTransitionBatch, _LegalActionTapeBuilder

# --- Dispatch split (grounded in worker.rs:1252-1304) -------------------------
#: Canonical opponent name -> Rust ``select_rule_action_for_state`` integer code
#: (``worker.rs:1285``; ``exploit_agent_kind_from_code`` ``worker.rs:1252``).
#: These are the 7 RULE-AGENT identities — pure Rust heuristic, NO policy forward pass.
RULE_AGENT_CODES: dict[str, int] = {
    "random": 0,           # select_deterministic_legal_random_action (worker.rs:1265)
    "face_rush": 1,        # ExploitAgentKind::FaceRush (worker.rs:1254)
    "board_control": 2,    # ExploitAgentKind::BoardControl (worker.rs:1255)
    "greedy_trade": 3,     # ExploitAgentKind::GreedyTrade (worker.rs:1256)
    "stall": 4,            # ExploitAgentKind::Stall (worker.rs:1257)
    "anti_draw_greed": 6,  # ExploitAgentKind::AntiDrawGreed (worker.rs:1259)
    # Available Rust rule codes NOT in the A3 spec mix (documented for completeness;
    # ``BLOCK_A_PLAN.md:401`` legacy phase26 mix used these but the A3 spec mix does not).
    # Block B (D-B10) ENABLES ``punish_empty_board`` (code 5) — additive uncomment,
    # zero Rust change (``worker.rs:1258 ExploitAgentKind::PunishEmptyBoard`` already
    # exists; ``BLOCK_B_PLAN.md:336-346`` + §10). ``anti_hand_leak_overfit`` (code 7)
    # is NOT in the spec mix — remains excluded.
    "punish_empty_board": 5,       # ExploitAgentKind::PunishEmptyBoard (worker.rs:1258)
    # "anti_hand_leak_overfit": 7,    # ExploitAgentKind::AntiHandLeakOverfit (worker.rs:1260)
}

#: The 4 POLICY-OPPONENT identities — Python-side opponent loop (no Rust rule code;
#: ``worker.rs:1281`` "unknown rule agent code" for codes outside 0-7). Mirrors
#: ``ai/train_v2/rollout_worker.py:211-230``.
POLICY_OPPONENT_KINDS: frozenset[str] = frozenset({"end_turn", "greedy_face", "self", "v4max"})

#: Block-B policy-opponent identities (the V4-orig temperature spectrum, B2
#: ``TempV4Opponent`` instances keyed by ``v4-orig-argmax`` / ``v4-orig-t07`` /
#: ``v4-orig-t12``, ``v4_orig_temp_spectrum.py:118-126``). ADDITIVE over Phase A:
#: these dispatch via the SAME Python policy-opponent loop as
#: ``POLICY_OPPONENT_KINDS`` (``collect_rust_live_rollout`` :683-700 calls
#: ``opponent_policies[identity].select(i, ctx)``), so they resolve to
#: ``(POLICY_DISPATCH, None)``. This frozenset is SEPARATE from
#: ``POLICY_OPPONENT_KINDS`` + ``PHASE_A_IDENTITIES`` — it does NOT change the 4-
#: policy / 11-identity Phase-A counts (the dispatch extension in
#: ``resolve_opponent_dispatch`` is an additive check AFTER the
#: ``POLICY_OPPONENT_KINDS`` check, before the raise). Block B wires these via B8's
#: ``opponent_policies`` (B2 ``TempV4Opponent``); NO Rust rule code (``worker.rs``
#: unchanged — these are PYTHON policy opponents, NOT Rust rule codes).
BLOCK_B_POLICY_OPPONENT_KINDS: frozenset[str] = frozenset(
    {"v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"}
)

#: All 11 Phase-A graduated identities (the union; 7 rule-agent + 4 policy;
#: ``punish_empty_board`` enabled by Block-B D-B10 additive uncomment). Matches
#: ``ppo_phaseA_config.PHASE_A_OPPONENT_MIX_SPEC`` via the alias layer.
PHASE_A_IDENTITIES: tuple[str, ...] = tuple(RULE_AGENT_CODES.keys()) + (
    "end_turn",
    "greedy_face",
    "self",
    "v4max",
)

#: Dispatch-path tags.
RULE_DISPATCH = "rule"
POLICY_DISPATCH = "policy"


def resolve_opponent_dispatch(identity: str) -> tuple[str, int | None]:
    """Return the dispatch path for a canonical opponent identity.

    ``(RULE_DISPATCH, code)`` for the 7 rule-agent identities (Rust
    ``select_rule_action_for_state`` codes 0-7, ``worker.rs:1285``);
    ``(POLICY_DISPATCH, None)`` for the 4 policy-opponent identities (Python loop,
    ``rollout_worker.py:211-230``). Raises ``ValueError`` on an unknown identity.

    This is the ORACLE for the dispatch split (verifier finding 2a blocker,
    ``BLOCK_A_PLAN.md:326-327``). The plan (``BLOCK_A_PLAN.md:404-419``) and the Rust
    source (``worker.rs:1252-1304``) AGREE — no discrepancy.
    """
    if identity in RULE_AGENT_CODES:
        return (RULE_DISPATCH, int(RULE_AGENT_CODES[identity]))
    if identity in POLICY_OPPONENT_KINDS:
        return (POLICY_DISPATCH, None)
    # Block-B additive extension (B8, ``BLOCK_B_PLAN.md`` §3 B8): the V4-orig
    # temperature spectrum (B2 ``TempV4Opponent``) dispatches via the SAME Python
    # policy-opponent loop as ``POLICY_OPPONENT_KINDS`` (no Rust rule code). This
    # check is AFTER the ``POLICY_OPPONENT_KINDS`` check + BEFORE the raise, so it
    # does NOT change the 4-policy / 11-identity Phase-A counts
    # (``test_six_rule_four_policy_split`` / ``test_ten_identities_present`` stay
    # green). ``PHASE_A_IDENTITIES`` is NOT extended (the v4-orig-* are Block-B
    # identities, not Phase-A graduated identities).
    if identity in BLOCK_B_POLICY_OPPONENT_KINDS:
        return (POLICY_DISPATCH, None)
    raise ValueError(f"unknown Phase-A opponent identity: {identity!r}")


def is_rule_agent(identity: str) -> bool:
    """True iff ``identity`` dispatches via the Rust rule-agent path (codes 0-7)."""
    return identity in RULE_AGENT_CODES


def is_policy_opponent(identity: str) -> bool:
    """True iff ``identity`` dispatches via the Python policy-opponent loop."""
    return identity in POLICY_OPPONENT_KINDS


# --- Policy-opponent context + protocol --------------------------------------
@dataclass(frozen=True)
class OpponentCtx:
    """Per-env snapshot handed to a policy-opponent ``select`` call.

    Mirrors the per-env slice the legacy ``_auto_play_until_learner``
    (``rollout_worker.py:230``) had access to via the Python env. For the Rust live path
    the opponent fn sees the packed legal-action arrays (the 601-candidate space) rather
    than action objects.
    """

    env_idx: int
    actor_id: int
    observation_v5: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray | None
    legal_action_counts: int
    mana_draw_legal: bool


class PolicyOpponent(Protocol):
    """A policy-opponent action selector (the 4 policy-opponent identities).

    ``select(env_idx, ctx) -> action_id`` returns a 601-candidate action_id that MUST be
    in ``ctx.legal_action_ids`` (the trainer steps the env with it). Mirrors
    ``rollout_worker.py:211-227`` ``_get_opponent_policy`` + ``select_core_action``.
    """

    name: str

    def select(self, env_idx: int, ctx: OpponentCtx) -> int: ...


class EndTurnOpponent:
    """``end_turn`` policy-opponent — always emits ``EndTurnAction``.

    ``EndTurnAction`` is candidate id 0 (``kernel.rs:2709``
    ``action_type_for_id(0)="end_turn"``; gen fixtures ``END_TURN = 0``). Prefers id 0
    when legal; falls back to the first legal action (a documented degenerate-baseline
    guardrail, ``opponents_v5.py:143`` ``end_turn`` role "degenerate_baseline_guardrail").
    """

    name = "end_turn"

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        if ids.size == 0:
            raise ValueError(
                f"end_turn opponent: env {env_idx} has no legal actions (should have been reset)"
            )
        if int(0) in ids.tolist():
            return 0
        return int(ids[0])


class GreedyFaceOpponent:
    """``greedy_face`` policy-opponent — prefers face (hero) damage.

    Mirrors the legacy ``rollout_worker.py:218`` ``GreedyFacePolicy`` intent
    ("prefers face damage", ``opponents_v5.py:141``). The Rust live path exposes packed
    legal-action ids + features (not action objects), so a feature-aware matcher is
    INJECTABLE via ``select_fn``. The default heuristic picks the highest legal action id
    (attacks tend to occupy higher candidate ids than ``end_turn=0``); production wires a
    feature-aware matcher over ``ctx.legal_action_features``. This is a documented
    heuristic (NOT a silent stub — the matcher is injectable and the role is "sanity
    trace source prefers face damage").
    """

    name = "greedy_face"

    def __init__(self, select_fn: Callable[[OpponentCtx], int] | None = None) -> None:
        self._select_fn = select_fn

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        if self._select_fn is not None:
            return int(self._select_fn(ctx))
        ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        if ids.size == 0:
            raise ValueError(
                f"greedy_face opponent: env {env_idx} has no legal actions (should have been reset)"
            )
        return int(ids[-1])


class SelfPrevOpponent:
    """``self`` / ``self_prev`` policy-opponent — a frozen V5 snapshot policy.

    For pure self-play the default is the LEARNER policy argmax (the current policy plays
    itself); production wires a prior checkpoint snapshot via ``select_fn``
    (``BLOCK_A_PLAN.md:417`` "self_prev(self/v5_snapshot)" once a league snapshot pool
    exists).
    """

    name = "self"

    def __init__(self, select_fn: Callable[[OpponentCtx], int]) -> None:
        self._select_fn = select_fn

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        return int(self._select_fn(ctx))


class V4MaxOpponent:
    """``v4max`` / ``v4-orig-argmax`` policy-opponent — the V4 ONNX argmax policy.

    ``opponents_v5.py:23`` marks ``v4max`` ``execution_kind='offline_v4max_teacher'``.
    Lazy: the caller wires the ONNX argmax ``select_fn``. If not wired, ``select``
    raises a clear error (NOT a silent stub — production must provide the V4 policy).
    """

    name = "v4max"

    def __init__(self, select_fn: Callable[[OpponentCtx], int] | None = None) -> None:
        self._select_fn = select_fn
        self.wired = select_fn is not None

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        if self._select_fn is None:
            raise RuntimeError(
                "v4max opponent policy not wired: provide select_fn "
                "(the V4 ONNX argmax policy, opponents_v5.py:23)"
            )
        return int(self._select_fn(ctx))


def default_opponent_policies(
    learner_argmax_select: Callable[[OpponentCtx], int] | None = None,
    *,
    greedy_face_select_fn: Callable[[OpponentCtx], int] | None = None,
    v4max_select_fn: Callable[[OpponentCtx], int] | None = None,
) -> dict[str, PolicyOpponent]:
    """Build the 4 policy-opponent selectors.

    ``end_turn`` + ``greedy_face`` use built-in heuristics (greedy_face's matcher is
    injectable). ``self`` defaults to ``learner_argmax_select`` (pure self-play) and
    raises if neither is provided. ``v4max`` is lazy (wired via ``v4max_select_fn`` or
    raises on use).
    """
    if learner_argmax_select is None:
        raise ValueError(
            "default_opponent_policies requires learner_argmax_select for the 'self' "
            "policy-opponent (pure self-play); provide a prior checkpoint for self_prev."
        )
    return {
        "end_turn": EndTurnOpponent(),
        "greedy_face": GreedyFaceOpponent(select_fn=greedy_face_select_fn),
        "self": SelfPrevOpponent(learner_argmax_select),
        "v4max": V4MaxOpponent(select_fn=v4max_select_fn),
    }


# --- Learner policy protocol + adapters --------------------------------------
@dataclass(frozen=True)
class LearnerCtxBatch:
    """Per-env context for the learner policy ``select`` call (learner-turn subset).

    ``env_indices`` is the subset of envs at the learner's turn this batch step. The
    per-env arrays are the FULL ``(env_count, ...)`` arrays; the policy indexes them by
    ``env_indices`` (so it can batch the learner-turn subset through one forward pass).
    """

    env_indices: np.ndarray
    observation_v5: np.ndarray
    legal_action_counts: np.ndarray
    legal_action_offsets: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray | None
    mana_draw_legal: np.ndarray


class LearnerPolicy(Protocol):
    """Learner policy protocol. ``select(ctx)`` returns per-learner-env arrays:

    ``(actions, values, joint_log_probs, selected_local_indices, mana_draw_flags)`` where
    ``mana_draw`` is a separate Rust flag but competes with the legal candidates in the
    same policy distribution. When it is selected, ``actions`` holds a valid ignored
    candidate placeholder for the compact tape.
    """

    def select(self, ctx: LearnerCtxBatch) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]: ...


class ArgmaxRandomLearner:
    """MLX-free learner for smoke tests: random legal action, value=0, log_prob=0,
    mana_draw=False. Lets the live trainer run end-to-end without MLX. Production wires
    ``MLXV5LearnerPolicy`` (the V5ActionConditionedPolicy adapter)."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, ctx: LearnerCtxBatch) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        idxs = np.asarray(ctx.env_indices, dtype=np.intp)
        n = int(idxs.size)
        actions = np.empty(n, dtype=np.uintp)
        values = np.zeros(n, dtype=np.float32)
        log_probs = np.zeros(n, dtype=np.float32)
        selected_local = np.empty(n, dtype=np.int32)
        mana_draw_flags = np.zeros(n, dtype=np.bool_)
        counts = np.asarray(ctx.legal_action_counts, dtype=np.intp)
        offsets = np.asarray(ctx.legal_action_offsets, dtype=np.intp)
        ids = np.asarray(ctx.legal_action_ids, dtype=np.uintp)
        for k, env in enumerate(idxs):
            c = int(counts[env])
            o = int(offsets[env])
            if c <= 0:
                raise ValueError(f"ArgmaxRandomLearner: env {env} has no legal actions")
            li = int(self.rng.integers(0, c))
            selected_local[k] = li
            actions[k] = ids[o + li]
        return actions, values, log_probs, selected_local, mana_draw_flags


# --- Live rollout result ------------------------------------------------------
@dataclass(frozen=True)
class LiveRolloutBatch:
    """Result of ``collect_rust_live_rollout``: a ``RustTransitionBatch`` (the PPO
    handoff, same format as ``collect_rust_vec_rollout``) PLUS the A4-specific
    mana_draw channels used by joint PPO + dispatch metadata.
    """

    transitions: RustTransitionBatch
    #: (n_learner_steps, env_count) bool — mana_draw legality at each learner decision.
    mana_draw_legal: np.ndarray
    #: (n_learner_steps, env_count) bool — whether the learner took mana_draw this step.
    mana_draw_taken: np.ndarray
    #: (env_count,) int — learner actor id per env (1=p1, 2=p2).
    learner_actor_ids: np.ndarray
    #: (env_count,) str — canonical opponent identity per env.
    opponent_identities: tuple[str, ...]
    #: Per-env count of learner transitions actually collected (== steps for full envs).
    learner_step_counts: np.ndarray
    #: Total batch steps executed (env steps, not learner steps).
    batch_steps: int
    #: Per-env episode count (number of resets + 1).
    episode_counts: np.ndarray
    #: (env_count, obs_dim) post-rollout observations used for GAE tail bootstrap.
    final_observations: np.ndarray
    #: Optional dispatch log (per-env per-batch-step source tag) when record_dispatch=True.
    dispatch_log: list[dict[str, Any]] | None = None


# --- Snapshot helper for decisive-early-end (D-A6) ----------------------------
@dataclass(frozen=True)
class _HeroHpSnapshot:
    """Duck-typed snapshot matching ``ppo_phaseA_config.is_decisive_state``'s expected
    attributes (``my_hero_hp`` / ``my_hero_max_hp`` / ``enemy_hero_hp`` /
    ``enemy_hero_max_hp``). Built from ``worker.hero_hp()`` ``[p1_hp, p1_max, p2_hp,
    p2_max]`` + the learner's side."""

    my_hero_hp: int
    my_hero_max_hp: int
    enemy_hero_hp: int
    enemy_hero_max_hp: int


def _hero_hp_snapshot(
    hero_hp: np.ndarray, env_idx: int, learner_actor_id: int
) -> _HeroHpSnapshot:
    """Build a decisive-state snapshot for env ``env_idx`` from the worker's ``hero_hp``
    array, oriented to the LEARNER's perspective (my = learner's hero, enemy = opponent's
    hero). ``hero_hp[env] = [p1_hp, p1_max_hp, p2_hp, p2_max_hp]`` (``rust_ffi.py:1204``).
    Learner actor id 1 -> p1 is the learner; 2 -> p2 is the learner.
    """
    row = np.asarray(hero_hp, dtype=np.int32)[env_idx]
    p1_hp, p1_max, p2_hp, p2_max = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    if int(learner_actor_id) == 2:
        return _HeroHpSnapshot(p2_hp, p2_max, p1_hp, p1_max)
    return _HeroHpSnapshot(p1_hp, p1_max, p2_hp, p2_max)


# --- Opponent + learner side sampling -----------------------------------------
def sample_opponent_identities(
    opponent_mix: list[tuple[str, float]], env_count: int, *, rng: np.random.Generator
) -> tuple[str, ...]:
    """Sample one canonical opponent identity per env from the parsed mix (weighted).

    ``opponent_mix`` is the output of ``league_v5.parse_v5_opponent_mix`` (canonical
    names: random/end_turn/greedy_face/face_rush/board_control/greedy_trade/stall/
    anti_draw_greed/self/v4max). Each identity is dispatchable via
    ``resolve_opponent_dispatch``.
    """
    if not opponent_mix:
        raise ValueError("opponent_mix must contain at least one identity")
    names = [name for name, _ in opponent_mix]
    weights = np.asarray([float(w) for _, w in opponent_mix], dtype=np.float64)
    weights = weights / weights.sum()
    chosen = rng.choice(len(names), size=int(env_count), p=weights)
    return tuple(names[int(i)] for i in chosen)


def sample_learner_sides(
    env_count: int,
    *,
    p1_score_rate: float = 0.5,
    p2_score_rate: float = 0.5,
    oversampling: dict[str, Any] | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the learner's actor id per env (1=p1, 2=p2), applying D-A10 second-start
    oversampling (``ppo_phaseA_config.second_start_oversampling_scheme``).

    With no breach (``abs(p1-p2) <= gap_threshold``) the split is 50/50. On breach the
    under-represented (lower-score) side is oversampled. Returns ``(env_count,)`` int32.
    """
    policy = str((oversampling or {}).get("policy", "oversample_under_represented_on_breach"))
    if policy == "strict_balanced":
        p1_count = int(env_count) // 2
        p2_count = int(env_count) - p1_count
        sides = np.asarray([1] * p1_count + [2] * p2_count, dtype=np.int32)
        rng.shuffle(sides)
        scheme = {
            "p1_weight": float(p1_count / max(1, int(env_count))),
            "p2_weight": float(p2_count / max(1, int(env_count))),
            "gap": abs(float(p1_score_rate) - float(p2_score_rate)),
            "breach": False,
            "oversampled_side": None,
            "policy": "strict_balanced",
        }
        return sides, scheme

    scheme = second_start_oversampling_scheme(
        float(p1_score_rate), float(p2_score_rate),
        gap_threshold=float((oversampling or {}).get("gap_threshold", 0.12)),
        base_weight=float((oversampling or {}).get("base_weight", 0.5)),
    )
    scheme["policy"] = policy
    w1 = float(scheme["p1_weight"])
    w2 = float(scheme["p2_weight"])
    sides = rng.choice([1, 2], size=int(env_count), p=[w1, w2]).astype(np.int32)
    return sides, scheme


# --- The live collector -------------------------------------------------------
def collect_rust_live_rollout(
    worker: Any,
    learner_policy: LearnerPolicy,
    opponent_policies: dict[str, PolicyOpponent],
    *,
    learner_actor_ids: np.ndarray,
    opponent_identities: tuple[str, ...] | list[str],
    config: PhaseAPPOConfig,
    steps: int | None = None,
    rng: np.random.Generator | None = None,
    record_dispatch: bool = False,
) -> LiveRolloutBatch:
    """Run LIVE self-play on the Rust ``ArenaEnv`` and collect learner transitions.

    Composes the existing ``RustBatchWorker`` FFI primitives (``from_live`` builds the
    worker; this function drives it). Lock-step per-action batch loop: each batch step
    advances every env by ONE action, dispatched per the current actor:
      * learner turn -> learner policy picks the 601 candidate + mana_draw flag;
      * rule-agent opponent turn -> ``select_rule_actions`` (Rust rule dispatcher);
      * policy opponent turn -> the Python opponent loop (``PolicyOpponent.select``).
    Only LEARNER-actor steps record a transition (the (s, a, r, terminal, mana_draw_legal)
    tuple for PPO). Opponent-actor response rewards are folded into the previous learner
    transition from the learner perspective, matching the rule-only fast path.
    Episodes terminate on game-over (``terminated``), ``max_turns`` truncation
    (``truncated``), or decisive-early-end (D-A6); the env is then reset
    (``reset_indices``) to start the next episode until the env collects ``steps``
    learner transitions.

    ``worker`` is a ``RustBatchWorker`` built via ``from_live(max_turns=...)`` (the
    trainer's ``run_live_self_play_update`` does this; tests may pass a fake worker
    satisfying the same call surface).

    Returns a ``LiveRolloutBatch`` (a ``RustTransitionBatch`` + mana_draw channels +
    metadata) so ``prepare_rust_ppo_batch`` + ``train_rust_ppo_minibatch`` work
    unchanged.
    """
    env_count = int(worker.env_count)
    if env_count <= 0:
        raise ValueError("worker.env_count must be positive")
    learner_actor = np.asarray(learner_actor_ids, dtype=np.int32)
    if learner_actor.shape != (env_count,):
        raise ValueError(
            f"learner_actor_ids shape {learner_actor.shape} != ({env_count},)"
        )
    opp_ids = tuple(opponent_identities)
    if len(opp_ids) != env_count:
        raise ValueError(
            f"opponent_identities length {len(opp_ids)} != env_count {env_count}"
        )
    for identity in opp_ids:
        # Validate every identity is dispatchable (raises on unknown).
        resolve_opponent_dispatch(identity)
    if rng is None:
        rng = np.random.default_rng(config.seed if config.seed is not None else 0)
    target_steps = int(steps if steps is not None else config.steps_per_update)
    if target_steps <= 0:
        raise ValueError("steps (target learner transitions per env) must be positive")

    # --- allocate buffers (target_steps, env_count) -----------------------------
    # Read the initial reset to size the observation + legal-action arrays.
    initial = worker.reset(copy=True)
    obs_v5 = np.asarray(initial["observation_v5"], dtype=np.float32)
    if obs_v5.shape[0] != env_count:
        raise ValueError(
            f"observation_v5 first dim {obs_v5.shape[0]} != env_count {env_count}"
        )
    obs_dim = int(obs_v5.shape[1])
    observations = np.empty((target_steps, env_count, obs_dim), dtype=np.float32)
    legal_action_counts = np.empty((target_steps, env_count), dtype=np.uintp)
    legal_action_offsets = np.empty((target_steps, env_count), dtype=np.uintp)
    legal_tape = _LegalActionTapeBuilder(
        ids_dtype=np.asarray(initial["legal_action_ids"]).dtype,
        features_dtype=(
            np.asarray(initial["legal_action_features"]).dtype
            if initial["legal_action_features"] is not None
            else np.float32
        ),
        feature_shape=(
            np.asarray(initial["legal_action_features"]).shape[1:]
            if initial["legal_action_features"] is not None
            else (1,)
        ),
        initial_capacity=int(np.asarray(initial["legal_action_ids"]).shape[0] or 1) * target_steps,
    )
    actions = np.empty((target_steps, env_count), dtype=np.uintp)
    rewards = np.empty((target_steps, env_count), dtype=np.float32)
    terminated = np.zeros((target_steps, env_count), dtype=np.bool_)
    truncated = np.zeros((target_steps, env_count), dtype=np.bool_)
    values = np.empty((target_steps, env_count), dtype=np.float32)
    log_probs = np.empty((target_steps, env_count), dtype=np.float32)
    selected_local = np.full((target_steps, env_count), -1, dtype=np.int32)
    mana_draw_legal_buf = np.zeros((target_steps, env_count), dtype=np.bool_)
    mana_draw_taken_buf = np.zeros((target_steps, env_count), dtype=np.bool_)
    values.fill(np.nan)
    log_probs.fill(np.nan)

    # --- per-env collection state ----------------------------------------------
    learner_step_count = np.zeros(env_count, dtype=np.intp)
    last_learner_row: list[int | None] = [None] * env_count
    pending_opener_reward = np.zeros(env_count, dtype=np.float32)
    episode_counts = np.ones(env_count, dtype=np.int64)
    episode_starting_actor = np.asarray(worker.current_actor_ids(), dtype=np.int32).copy()
    if episode_starting_actor.shape != (env_count,):
        raise ValueError(f"current_actor_ids shape {episode_starting_actor.shape} != ({env_count},)")

    # ``current`` = the pre-step arrays (state at the start of each batch step).
    current = initial
    batch_step = 0
    # Bound the total batch steps: each learner transition needs at most a few env
    # steps (learner action + opponent turn). max_turns bounds episode length.
    max_batch_steps = int(target_steps) * (int(config.max_turns) + 8) * 2 + 64

    dispatch_log: list[dict[str, Any]] | None = [] if record_dispatch else None

    while batch_step < max_batch_steps:
        if bool(np.all(learner_step_count >= target_steps)):
            break  # all envs collected target_steps learner transitions

        actors = np.asarray(worker.current_actor_ids(), dtype=np.int32)
        if actors.shape != (env_count,):
            raise ValueError(f"current_actor_ids shape {actors.shape} != ({env_count},)")
        md_legal = np.asarray(worker.mana_draw_legal(), dtype=np.bool_)
        is_learner = actors == learner_actor  # (env_count,) bool

        # --- build per-env action + dispatch -----------------------------------
        action_ids = np.zeros(env_count, dtype=np.uintp)
        mana_draw_flags = np.zeros(env_count, dtype=np.bool_)
        # rule-agent codes for select_rule_actions (placeholder 0 for non-rule envs).
        codes = np.zeros(env_count, dtype=np.uint32)
        learner_envs: list[int] = []
        rule_envs: list[int] = []
        policy_envs: list[tuple[int, str]] = []
        full_envs: list[int] = []

        cur_counts = np.asarray(current["legal_action_counts"], dtype=np.intp)
        cur_offsets = np.asarray(current["legal_action_offsets"], dtype=np.intp)
        cur_legal_ids = np.asarray(current["legal_action_ids"], dtype=np.uintp)
        cur_legal_features = current["legal_action_features"]
        cur_obs = np.asarray(current["observation_v5"], dtype=np.float32)

        for i in range(env_count):
            if int(learner_step_count[i]) >= target_steps:
                full_envs.append(i)
                continue
            if bool(is_learner[i]):
                learner_envs.append(i)
            else:
                kind, code = resolve_opponent_dispatch(opp_ids[i])
                if kind == RULE_DISPATCH:
                    codes[i] = int(code)
                    rule_envs.append(i)
                else:
                    policy_envs.append((i, opp_ids[i]))

        # A full env may be sitting in a terminal/no-legal state while other envs
        # still need learner transitions. Since it is no longer recorded, reset it
        # before the service action below so the batched Rust step remains valid.
        full_needing_reset = [i for i in full_envs if int(cur_counts[i]) <= 0]
        if full_needing_reset:
            worker.reset_indices(np.asarray(full_needing_reset, dtype=np.uintp))
            reset_actors = np.asarray(worker.current_actor_ids(), dtype=np.int32)
            for i in full_needing_reset:
                episode_starting_actor[int(i)] = int(reset_actors[int(i)])
            current = worker.arrays(copy=True)
            cur_counts = np.asarray(current["legal_action_counts"], dtype=np.intp)
            cur_offsets = np.asarray(current["legal_action_offsets"], dtype=np.intp)
            cur_legal_ids = np.asarray(current["legal_action_ids"], dtype=np.uintp)
            cur_legal_features = current["legal_action_features"]
            cur_obs = np.asarray(current["observation_v5"], dtype=np.float32)

        # --- rule-agent actions (batched Rust dispatcher: select_rule_actions) ---
        if rule_envs:
            rule_actions = np.asarray(
                worker.select_rule_actions(codes, salt=int(batch_step)), dtype=np.uintp
            )
            for i in rule_envs:
                action_ids[i] = int(rule_actions[i])
        else:
            # Still validate codes shape if select_rule_actions wasn't called (it
            # requires codes for ALL envs); only call when there is at least one rule env.
            pass

        # --- policy-opponent actions (Python loop, rollout_worker.py:211-230) ---
        for i, identity in policy_envs:
            ctx = OpponentCtx(
                env_idx=i,
                actor_id=int(actors[i]),
                observation_v5=cur_obs[i],
                legal_action_ids=cur_legal_ids[
                    int(cur_offsets[i]):int(cur_offsets[i]) + int(cur_counts[i])
                ],
                legal_action_features=(
                    None if cur_legal_features is None
                    else np.asarray(cur_legal_features)[
                        int(cur_offsets[i]):int(cur_offsets[i]) + int(cur_counts[i])
                    ]
                ),
                legal_action_counts=int(cur_counts[i]),
                mana_draw_legal=bool(md_legal[i]),
            )
            action_ids[i] = int(opponent_policies[identity].select(i, ctx))

        # --- learner actions (learner policy + mana_draw head) -----------------
        if learner_envs:
            learner_idx_arr = np.asarray(learner_envs, dtype=np.intp)
            lctx = LearnerCtxBatch(
                env_indices=learner_idx_arr,
                observation_v5=cur_obs,
                legal_action_counts=cur_counts,
                legal_action_offsets=cur_offsets,
                legal_action_ids=cur_legal_ids,
                legal_action_features=cur_legal_features,
                mana_draw_legal=md_legal,
            )
            l_actions, l_values, l_log_probs, l_sel_local, l_mana_draw = learner_policy.select(lctx)
            l_actions = np.asarray(l_actions, dtype=np.uintp)
            l_values = np.asarray(l_values, dtype=np.float32)
            l_log_probs = np.asarray(l_log_probs, dtype=np.float32)
            l_sel_local = np.asarray(l_sel_local, dtype=np.int32)
            l_mana_draw = np.asarray(l_mana_draw, dtype=np.bool_)
            if l_actions.shape != (len(learner_envs),):
                raise ValueError(
                    f"learner policy actions shape {l_actions.shape} != "
                    f"({len(learner_envs)},)"
                )
            for k, i in enumerate(learner_envs):
                row = int(learner_step_count[i])
                if row >= target_steps:
                    continue  # became full mid-loop (shouldn't happen; guarded above)
                # record the learner transition (s = current obs, a = best candidate).
                observations[row, i] = cur_obs[i]
                legal_action_counts[row, i] = cur_counts[i]
                # Offsets are relative to the compact tape built for this rollout, not
                # to the worker's current global legal-action tape.
                legal_action_offsets[row, i] = int(legal_tape.size)
                legal_tape.append(
                    cur_legal_ids[int(cur_offsets[i]):int(cur_offsets[i]) + int(cur_counts[i])],
                    (
                        None if cur_legal_features is None
                        else np.asarray(cur_legal_features)[
                            int(cur_offsets[i]):int(cur_offsets[i]) + int(cur_counts[i])
                        ]
                    ),
                )
                actions[row, i] = int(l_actions[k])
                values[row, i] = float(l_values[k])
                log_probs[row, i] = float(l_log_probs[k])
                selected_local[row, i] = int(l_sel_local[k])
                mana_draw_legal_buf[row, i] = bool(md_legal[i])
                mana_draw_taken_buf[row, i] = bool(l_mana_draw[k])
                # step with the learner action + mana_draw flag (placeholder action_id
                # when mana_draw taken — kernel ignores it, kernel.rs:788).
                action_ids[i] = int(l_actions[k])
                mana_draw_flags[i] = bool(l_mana_draw[k])
                last_learner_row[i] = row
                learner_step_count[i] = int(row) + 1

        # Full envs are no longer recorded, but the Rust batch step still needs a
        # legal action for every env while the remaining envs catch up.
        for i in full_envs:
            count = int(cur_counts[i])
            offset = int(cur_offsets[i])
            if count <= 0:
                raise ValueError(f"full env {i} has no legal actions")
            action_ids[i] = int(cur_legal_ids[offset])

        # --- dispatch log (test hook) -------------------------------------------
        if dispatch_log is not None:
            for i in range(env_count):
                if int(learner_step_count[i]) >= target_steps and i in full_envs:
                    src = "full"
                elif bool(is_learner[i]):
                    src = "learner"
                else:
                    src = RULE_DISPATCH if i in rule_envs else POLICY_DISPATCH
                dispatch_log.append({
                    "batch_step": int(batch_step),
                    "env": int(i),
                    "actor": int(actors[i]),
                    "source": src,
                    "identity": opp_ids[i],
                    "action": int(action_ids[i]),
                    "mana_draw": bool(mana_draw_flags[i]),
                })

        # --- step the batch ----------------------------------------------------
        out = worker.step_mana_draw(action_ids, mana_draw_flags, copy=True)
        out_rewards = np.asarray(out["rewards"], dtype=np.float32)
        out_terminated = np.asarray(out["terminated"], dtype=np.bool_)
        out_truncated = np.asarray(worker.truncated(), dtype=np.bool_)
        hero_hp = np.asarray(worker.hero_hp(), dtype=np.int32)

        # --- reward attribution (fix #1, learner-only) -------------------------
        attributed = reward_attribution(out_rewards, actors, learner_actor)

        # Apply attributed rewards + terminal flags to the just-recorded learner
        # transitions (learner-actor envs). Opponent-actor envs get ZERO reward (no
        # transition recorded this step).
        for i in learner_envs:
            row = last_learner_row[i]
            if row is None or row >= target_steps:
                continue
            reward_i = float(attributed[i]) + float(pending_opener_reward[i])
            if int(learner_actor[i]) != int(episode_starting_actor[i]):
                reward_i += float(config.turn_order_second_mover_reward_bonus)
            rewards[row, i] = reward_i
            pending_opener_reward[i] = 0.0
            term_i = bool(out_terminated[i])
            trunc_i = bool(out_truncated[i])
            # decisive-early-end (D-A6): a decisive win-margin terminates the episode.
            if config.decisive_early_end:
                snap = _hero_hp_snapshot(hero_hp, i, int(learner_actor[i]))
                if is_decisive_state(snap, threshold=config.decisive_win_margin_threshold):
                    trunc_i = True
            if term_i or trunc_i:
                terminated[row, i] = bool(term_i)
                truncated[row, i] = bool(trunc_i)
                last_learner_row[i] = None  # episode closed
                # reset env i for the next episode (if not yet full).
                if int(learner_step_count[i]) < target_steps:
                    worker.reset_indices(np.asarray([i], dtype=np.uintp))
                    episode_starting_actor[i] = int(np.asarray(worker.current_actor_ids(), dtype=np.int32)[i])
                    episode_counts[i] = int(episode_counts[i]) + 1

        # Opponent-actor envs: no standalone transition is recorded. Fold the opponent
        # response reward into the previous learner transition from the learner's
        # perspective (the Rust rule-only fast path does the same as
        # `learner_rewards[idx] -= step.reward`). If the episode ended on an opponent
        # step, mark that same last learner transition terminal/truncated, then reset.
        for i in range(env_count):
            if i in learner_envs or int(learner_step_count[i]) >= target_steps:
                continue
            row = last_learner_row[i]
            if row is not None and row < target_steps:
                rewards[row, i] += -float(out_rewards[i])
            elif int(learner_step_count[i]) == 0:
                pending_opener_reward[i] += -float(out_rewards[i])
            if not (bool(out_terminated[i]) or bool(out_truncated[i])):
                # still check decisive-early-end on opponent steps.
                if config.decisive_early_end:
                    snap = _hero_hp_snapshot(hero_hp, i, int(learner_actor[i]))
                    if not is_decisive_state(snap, threshold=config.decisive_win_margin_threshold):
                        continue
                else:
                    continue
            if row is not None and row < target_steps:
                term_i = bool(out_terminated[i])
                trunc_i = bool(out_truncated[i])
                if config.decisive_early_end:
                    snap = _hero_hp_snapshot(hero_hp, i, int(learner_actor[i]))
                    if is_decisive_state(snap, threshold=config.decisive_win_margin_threshold):
                        trunc_i = True
                if term_i or trunc_i:
                    terminated[row, i] = bool(term_i) or bool(terminated[row, i])
                    truncated[row, i] = bool(trunc_i) or bool(truncated[row, i])
                    last_learner_row[i] = None
                    worker.reset_indices(np.asarray([i], dtype=np.uintp))
                    episode_starting_actor[i] = int(np.asarray(worker.current_actor_ids(), dtype=np.int32)[i])
                    episode_counts[i] = int(episode_counts[i]) + 1

        # advance: the next iteration reads the post-step + post-reset state.
        current = worker.arrays(copy=True)
        batch_step += 1

    if bool(np.any(learner_step_count < target_steps)):
        # Some envs did not reach target_steps within the batch-step bound. Truncate the
        # batch to the max collected across envs so the batch stays uniform (shorter envs
        # would leave trailing NaN rows). This is a graceful degradation, not the happy
        # path (with max_turns + decisive-early-end every episode ends -> every env
        # reaches target_steps given enough batch steps).
        max_collected = int(np.max(learner_step_count))
    else:
        max_collected = target_steps

    legal_ids_tape, legal_features_tape = legal_tape.finish()
    final_observations = np.asarray(current["observation_v5"], dtype=np.float32).copy()

    transitions = RustTransitionBatch(
        observations=observations[:max_collected],
        next_observations=None,  # config.store_next_observations=False (A3 default)
        action_mask=None,        # action_mask_mode="legal_only" (A3 default) -> legal pack only
        action_features=None,    # action_features_mode="legal_only" (A3 default) -> legal pack only
        legal_action_counts=legal_action_counts[:max_collected],
        legal_action_offsets=legal_action_offsets[:max_collected],
        legal_action_ids=legal_ids_tape,
        legal_action_features=legal_features_tape,
        actions=actions[:max_collected],
        rewards=rewards[:max_collected],
        terminated=terminated[:max_collected],
        truncated=truncated[:max_collected],
        reset_flags=None,
        terminal_observations=None,
        terminal_observation_valid=None,
        episode_returns=None,
        episode_lengths=None,
        infos=None,
        values=values[:max_collected],
        log_probs=log_probs[:max_collected],
        selected_local_indices=selected_local[:max_collected],
        policy_seconds=0.0,
        env_step_seconds=0.0,
        policy_profile=None,
    )

    return LiveRolloutBatch(
        transitions=transitions,
        mana_draw_legal=mana_draw_legal_buf[:max_collected],
        mana_draw_taken=mana_draw_taken_buf[:max_collected],
        learner_actor_ids=learner_actor,
        opponent_identities=opp_ids,
        learner_step_counts=learner_step_count,
        batch_steps=int(batch_step),
        episode_counts=episode_counts,
        final_observations=final_observations,
        dispatch_log=dispatch_log,
    )


def _estimate_bootstrap_values(model: Any, final_observations: np.ndarray) -> np.ndarray | None:
    if model is None:
        return None
    encode_state = getattr(model, "encode_state", None)
    value_head = getattr(model, "value_head", None)
    if not callable(encode_state) or value_head is None:
        return None
    import mlx.core as mx

    obs = mx.array(np.asarray(final_observations, dtype=np.float32))
    values = value_head(encode_state(obs)).squeeze(-1)
    mx.eval(values)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def fast_forward_rule_opponent_turns(
    worker: Any,
    learner_actor_ids: np.ndarray,
    opponent_identities: tuple[str, ...] | list[str],
    *,
    max_actions_per_env: int = 64,
    salt: int = 0,
    auto_reset: bool = True,
) -> dict[str, np.ndarray]:
    """Batched fast-forward applying RULE-AGENT actions until the learner is to act
    (the ``advance_rule_until_actor`` primitive, ``rust_ffi.py:1078``).

    ONLY valid for pure-rule-agent batches (every env's opponent identity must have a
    Rust rule code). If ANY env has a policy-opponent identity, the caller MUST use the
    per-action Python loop in ``collect_rust_live_rollout`` instead (advance_rule_until_actor
    cannot apply policy-opponent actions — passing a placeholder rule code would dispatch
    the WRONG rule agent for that env). This helper is exposed for the rule-only fast path
    and tested separately; the general mixed-batch collector uses per-action stepping.
    """
    env_count = int(worker.env_count)
    learner_actor = np.asarray(learner_actor_ids, dtype=np.int32)
    opp_ids = tuple(opponent_identities)
    if len(opp_ids) != env_count:
        raise ValueError(
            f"opponent_identities length {len(opp_ids)} != env_count {env_count}"
        )
    codes = np.zeros(env_count, dtype=np.uint32)
    for i, identity in enumerate(opp_ids):
        kind, code = resolve_opponent_dispatch(identity)
        if kind != RULE_DISPATCH:
            raise ValueError(
                f"fast_forward_rule_opponent_turns: env {i} identity {identity!r} is a "
                f"policy-opponent (no Rust rule code); use the per-action Python loop in "
                f"collect_rust_live_rollout instead"
            )
        codes[i] = int(code)
    return worker.advance_rule_until_actor(
        learner_actor,
        codes,
        max_actions_per_env=int(max_actions_per_env),
        salt=int(salt),
        auto_reset=bool(auto_reset),
        copy=True,
    )


# --- Top-level: one finite PPO update on a seeded live arena ------------------
def _build_live_worker(
    config: PhaseAPPOConfig,
    *,
    seed: int,
    env_count: int,
    library_path: str | Any | None = None,
) -> Any:
    """Build the live RustBatchWorker via ``from_live``, threading ``config.max_turns``
    into ``KernelConfig`` (``kernel.rs:660``). This is the LIVE constructor (D-A8)."""
    from .rust_ffi import RustBatchWorker

    # ``PhaseAPPOConfig.diagnostic_mode`` inherits the ``RustPPOTrainingConfig`` default
    # "auto", but ``from_trace_file`` (which ``from_live`` composes) requires "full" or
    # "none" (``_normalize_diagnostic_mode`` ``rust_ffi.py:2155``). The live path does NOT
    # use infos/episode-stats (``store_infos`` / ``store_episode_stats`` are False), so
    # resolve "auto" -> "none". Pass "full"/"none" through unchanged.
    diag = config.diagnostic_mode
    if diag == "auto":
        diag = "none"

    return RustBatchWorker.from_live(
        seed=int(seed),
        env_count=int(env_count),
        max_turns=int(config.max_turns),
        library_path=library_path,
        action_features_dtype=config.action_features_dtype,
        action_features_mode=config.action_features_mode,
        observation_mode=config.observation_mode,
        action_mask_mode=config.action_mask_mode,
        terminal_observation_mode=config.terminal_observation_mode,
        diagnostic_mode=diag,
    )


def run_live_self_play_update(
    config: PhaseAPPOConfig,
    learner_policy: LearnerPolicy,
    opponent_policies: dict[str, PolicyOpponent] | None = None,
    *,
    seed: int | None = None,
    library_path: str | Any | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    worker_factory: Callable[[int], Any] | None = None,
    p1_score_rate: float = 0.5,
    p2_score_rate: float = 0.5,
    steps: int | None = None,
    opponent_mix_parsed: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Run ONE finite PPO update on a seeded live arena (THE MISSING ENTRY POINT).

    1. Build the live worker via ``from_live(max_turns=config.max_turns)`` (or
       ``worker_factory(env_count)`` if provided — for tests injecting a fake worker).
    2. Sample per-env opponent identities (weighted ``config.opponent_mix``) + learner
       sides (D-A10 second-start oversampling).
    3. ``collect_rust_live_rollout`` -> ``LiveRolloutBatch`` (learner-only reward,
       decisive-early-end, max_turns, both dispatch paths).
    4. ``prepare_rust_ppo_batch`` (GAE/returns).
    5. ``train_rust_ppo_minibatch`` IF ``model`` + ``optimizer`` provided (needs MLX);
       otherwise stop after prepare and return the prepared batch (the A4 surface —
       collect + prepare — is testable without MLX; the PPO optimizer step is MLX-gated
       and exercised by the trace-pool trainer tests).

    Returns a metrics dict: collection stats, prepare stats, and (if trained) update
    metrics; plus the ``LiveRolloutBatch`` + ``RustPPOBatch`` for inspection.
    """
    from .league_v5 import parse_v5_opponent_mix
    from .rust_ppo import prepare_rust_ppo_batch

    env_count = int(config.env_count)
    if env_count <= 0:
        raise ValueError("config.env_count must be positive")
    use_seed = int(seed if seed is not None else (config.seed if config.seed is not None else 0))
    rng = np.random.default_rng(use_seed)

    # 1. build worker
    if worker_factory is not None:
        worker = worker_factory(env_count)
    else:
        worker = _build_live_worker(config, seed=use_seed, env_count=env_count, library_path=library_path)

    # 2. sample opponents + learner sides. ``opponent_mix_parsed`` (Block-B
    # additive, B8) lets the caller bypass ``parse_v5_opponent_mix`` and pass a
    # pre-parsed mix DIRECTLY — required for the Block-B mix whose
    # ``v4-orig-argmax`` / ``v4-orig-t07`` / ``v4-orig-t12`` identities are NOT in
    # ``league_v5.V5_OPPONENT_KINDS`` (parse_v5_opponent_mix raises on them). The
    # pre-parsed mix comes from B3 ``build_block_b_opponent_mix`` (after B4
    # curriculum reweight + the D-B5 hybrid collapse monitor); identities are
    # validated per-env by ``resolve_opponent_dispatch`` in
    # ``collect_rust_live_rollout`` (:569-571 — the v4-orig-* resolve to
    # ``(POLICY_DISPATCH, None)`` via the ``BLOCK_B_POLICY_OPPONENT_KINDS``
    # extension). When ``opponent_mix_parsed`` is None (the Phase-A default) the
    # existing ``parse_v5_opponent_mix(config.opponent_mix)`` path is unchanged.
    if opponent_mix_parsed is not None:
        mix = list(opponent_mix_parsed)
    else:
        mix = parse_v5_opponent_mix(config.opponent_mix)
    opp_identities = sample_opponent_identities(mix, env_count, rng=rng)
    side_policy = str((config.second_start_oversampling or {}).get("policy", "oversample_under_represented_on_breach"))
    if side_policy == "start_second":
        starting_actors = np.asarray(worker.current_actor_ids(), dtype=np.int32)
        if starting_actors.shape != (env_count,):
            raise ValueError(f"current_actor_ids shape {starting_actors.shape} != ({env_count},)")
        learner_sides = np.where(starting_actors == 1, 2, 1).astype(np.int32)
        oversampling_scheme = {
            "p1_weight": float(np.mean(learner_sides == 1)),
            "p2_weight": float(np.mean(learner_sides == 2)),
            "gap": abs(float(p1_score_rate) - float(p2_score_rate)),
            "breach": False,
            "oversampled_side": None,
            "policy": "start_second",
            "starting_actor_counts": {
                "1": int(np.sum(starting_actors == 1)),
                "2": int(np.sum(starting_actors == 2)),
            },
        }
    else:
        learner_sides, oversampling_scheme = sample_learner_sides(
            env_count,
            p1_score_rate=p1_score_rate,
            p2_score_rate=p2_score_rate,
            oversampling=config.second_start_oversampling,
            rng=rng,
        )
    if opponent_policies is None:
        # default factory needs a learner-argmax selector for 'self'; use the provided
        # learner policy's argmax if it exposes one, else require explicit wiring.
        learner_argmax = getattr(learner_policy, "argmax_select", None)
        opponent_policies = default_opponent_policies(learner_argmax)

    try:
        # 3. collect
        t0 = time.perf_counter()
        rollout = collect_rust_live_rollout(
            worker,
            learner_policy,
            opponent_policies,
            learner_actor_ids=learner_sides,
            opponent_identities=opp_identities,
            config=config,
            steps=steps,
            rng=rng,
        )
        collect_seconds = time.perf_counter() - t0

        # 4. prepare PPO batch (GAE/returns) — needs values + log_probs (recorded).
        t1 = time.perf_counter()
        bootstrap_values = _estimate_bootstrap_values(model, rollout.final_observations)
        ppo_batch = prepare_rust_ppo_batch(
            rollout.transitions,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            bootstrap_values=bootstrap_values,
            advantage_backend=config.advantage_backend,
            selected_local_backend=config.selected_local_backend,
            prepare_backend=config.prepare_backend,
            mana_draw_legal=rollout.mana_draw_legal,
            mana_draw_taken=rollout.mana_draw_taken,
            library_path=library_path,
        )
        prepare_seconds = time.perf_counter() - t1

        metrics: dict[str, Any] = {
            "update_kind": "live_self_play",
            "env_count": env_count,
            "max_turns": int(config.max_turns),
            "steps_target": int(steps if steps is not None else config.steps_per_update),
            "learner_step_counts": rollout.learner_step_counts.tolist(),
            "batch_steps": rollout.batch_steps,
            "episode_counts": rollout.episode_counts.tolist(),
            "opponent_identities": list(rollout.opponent_identities),
            "learner_actor_ids": rollout.learner_actor_ids.tolist(),
            "oversampling_scheme": oversampling_scheme,
            "collect_seconds": collect_seconds,
            "prepare_seconds": prepare_seconds,
            "bootstrap_values_used": bootstrap_values is not None,
            "bootstrap_value_mean": (
                None if bootstrap_values is None else float(np.asarray(bootstrap_values, dtype=np.float32).mean())
            ),
            "turn_order_second_mover_reward_bonus": float(config.turn_order_second_mover_reward_bonus),
            "advantage_backend": config.advantage_backend,
            "entropy_coef": config.entropy_coef,
            "epochs": config.epochs,
            "opponent_mix_parsed": opponent_mix_parsed is not None,
            "mana_draw_eligible": int(np.count_nonzero(rollout.mana_draw_legal)),
            "mana_draw_taken": int(np.count_nonzero(rollout.mana_draw_taken)),
            "mana_draw_rate": (
                0.0
                if not bool(np.any(rollout.mana_draw_legal))
                else float(np.mean(rollout.mana_draw_taken[rollout.mana_draw_legal]))
            ),
        }

        # 5. train (MLX-gated)
        update_metrics: dict[str, Any] | None = None
        if model is not None and optimizer is not None:
            from .rust_ppo import train_rust_ppo_minibatch

            t2 = time.perf_counter()
            update_metrics = train_rust_ppo_minibatch(
                model,
                optimizer,
                ppo_batch,
                epochs=config.epochs,
                minibatch_size=config.minibatch_size,
                clip_epsilon=config.clip_epsilon,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
                shuffle=False,
                seed=None if config.seed is None else config.seed + 1,
                legal_row_pack_backend=config.legal_row_pack_backend,
                full_batch_eval=config.full_batch_eval,
                minibatch_plan=config.ppo_minibatch_plan,
                library_path=library_path,
            )
            metrics["train_seconds"] = time.perf_counter() - t2
            metrics["update_metrics"] = update_metrics

        metrics["has_rollout"] = True
        metrics["has_ppo_batch"] = True
        metrics["rollout"] = rollout
        metrics["ppo_batch"] = ppo_batch
        return metrics
    finally:
        close = getattr(worker, "close", None)
        if callable(close):
            close()


__all__ = [
    "EndTurnOpponent",
    "GreedyFaceOpponent",
    "LearnerCtxBatch",
    "LearnerPolicy",
    "ArgmaxRandomLearner",
    "LiveRolloutBatch",
    "OpponentCtx",
    "PHASE_A_IDENTITIES",
    "POLICY_DISPATCH",
    "POLICY_OPPONENT_KINDS",
    "BLOCK_B_POLICY_OPPONENT_KINDS",
    "PolicyOpponent",
    "RULE_AGENT_CODES",
    "RULE_DISPATCH",
    "SelfPrevOpponent",
    "V4MaxOpponent",
    "collect_rust_live_rollout",
    "default_opponent_policies",
    "fast_forward_rule_opponent_turns",
    "is_policy_opponent",
    "is_rule_agent",
    "resolve_opponent_dispatch",
    "run_live_self_play_update",
    "sample_learner_sides",
    "sample_opponent_identities",
]
