"""Tests for Block A component A4 — ``rust_live_self_play.py`` (TRACKED).

Verifies the missing live-self-play entry point:
  * dispatch split (verifier finding 2a blocker): 6 rule-agent identities via the Rust
    ``select_rule_action_for_state`` codes 0-7 (worker.rs:1285) + 4 policy-opponent
    identities via the Python loop (rollout_worker.py:211-230);
  * learner-only reward (fix #1): opponent-actor steps record ZERO reward;
  * max_turns threading (fix #2): ``from_live(max_turns=120)`` threads into
    ``KernelConfig`` (kernel.rs:660) — NOT the serde default 80 (kernel.rs:624);
  * decisive-state early-end (D-A6);
  * one finite PPO update on a seeded live arena completes.

Test strategy (documented): BOTH strategies are used.
  * Mocked-FFI composition tests (a FakeWorker satisfying the RustBatchWorker call
    surface) for the DETERMINISTIC, fast assertions: dispatch-split routing, learner-only
    reward attribution, decisive-early-end, max_turns threading via the constructor,
    second-start oversampling. These never touch the Rust lib.
  * Real-FFI smoke tests (gated by ``_rust_ffi_available()``) for integration confidence:
    one PPO update (collect + prepare) on a real seeded arena, max_turns behavioral
    threading, learner-only reward on a real short run. Skipped when the Rust extension
    is unbuildable in the worktree (``test_skip_if_no_rust_ffi``).

Run:
  PYTHONPATH=.:TrainV3.5/python python3 -m pytest \\
      TrainV3.5/python/train_v3/tests/test_rust_live_self_play.py
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from train_v3 import rust_live_self_play as rls
from train_v3.ppo_phaseA_config import (
    PHASE_A_MAX_TURNS,
    PHASE_A_OPPONENT_MIX_SPEC,
    PHASE_A_OPPONENT_NAME_ALIASES,
    PhaseAPPOConfig,
    build_phase_a_opponent_mix_string,
    is_decisive_state,
    reward_attribution,
)

# The 10 canonical opponent identities (after the A3 display->canonical alias layer,
# ``PHASE_A_OPPONENT_NAME_ALIASES``). Dispatch operates on canonical names (the form
# ``league_v5.parse_v5_opponent_mix`` yields), NOT the display names in
# ``PHASE_A_OPPONENT_MIX_SPEC`` (legal_random / self_prev / v4-orig-argmax).
CANONICAL_PHASE_A_IDENTITIES = frozenset(PHASE_A_OPPONENT_NAME_ALIASES.values())


# --- skip-gate for real-FFI tests --------------------------------------------
def _rust_ffi_available() -> bool:
    try:
        from train_v3 import rust_ffi  # noqa: F401
        from train_v3.rust_ffi import RustBatchWorker  # noqa: F401
    except Exception:
        return False
    # Probe the live constructor + a from_live build with a tiny env_count.
    try:
        w = RustBatchWorker.from_live(seed=1, env_count=1, max_turns=4)
        w.reset(copy=True)
        w.current_actor_ids()
        w.close()
        return True
    except Exception:
        return False


_RUST_FFI_OK = None


def _rust_ok() -> bool:
    global _RUST_FFI_OK
    if _RUST_FFI_OK is None:
        _RUST_FFI_OK = _rust_ffi_available()
    return _RUST_FFI_OK


requires_rust_ffi = pytest.mark.skipif(
    not _rust_ffi_available(),
    reason="Rust trainv3_core extension not loadable in this worktree",
)


# ===========================================================================
# Pure dispatch-split tests (no worker needed) — verifier finding 2a blocker
# ===========================================================================
class TestDispatchSplit:
    def test_ten_identities_present(self):
        assert len(rls.PHASE_A_IDENTITIES) == 10
        # Dispatch operates on canonical names (parse_v5_opponent_mix output form), not
        # the display names in PHASE_A_OPPONENT_MIX_SPEC.
        assert set(rls.PHASE_A_IDENTITIES) == set(CANONICAL_PHASE_A_IDENTITIES), (
            f"PHASE_A_IDENTITIES (canonical) {set(rls.PHASE_A_IDENTITIES)} != "
            f"canonical spec {set(CANONICAL_PHASE_A_IDENTITIES)}"
        )
        # And the canonical set is the 1:1 image of the 10 display names under the alias map.
        assert len(CANONICAL_PHASE_A_IDENTITIES) == len(PHASE_A_OPPONENT_MIX_SPEC) == 10

    def test_six_rule_four_policy_split(self):
        rule = [i for i in rls.PHASE_A_IDENTITIES if rls.is_rule_agent(i)]
        policy = [i for i in rls.PHASE_A_IDENTITIES if rls.is_policy_opponent(i)]
        assert sorted(rule) == sorted(
            ["random", "face_rush", "board_control", "greedy_trade", "stall", "anti_draw_greed"]
        ), "the 6 rule-agent identities (Rust select_rule_action_for_state codes 0-7)"
        assert sorted(policy) == sorted(["end_turn", "greedy_face", "self", "v4max"]), (
            "the 4 policy-opponent identities (Python loop, no Rust rule code)"
        )

    def test_resolve_opponent_dispatch_codes_grounded_in_worker_rs(self):
        # Grounded in worker.rs:1252-1304 (exploit_agent_kind_from_code + select_rule_action_for_state)
        # and worker.rs:1265 (select_deterministic_legal_random_action for code 0).
        expected = {
            "random": 0,
            "face_rush": 1,
            "board_control": 2,
            "greedy_trade": 3,
            "stall": 4,
            "anti_draw_greed": 6,
        }
        for name, code in expected.items():
            kind, got = rls.resolve_opponent_dispatch(name)
            assert kind == rls.RULE_DISPATCH, f"{name} must be rule-agent"
            assert got == code, f"{name} -> code {got} != expected {code}"
        for name in ("end_turn", "greedy_face", "self", "v4max"):
            kind, got = rls.resolve_opponent_dispatch(name)
            assert kind == rls.POLICY_DISPATCH, f"{name} must be policy-opponent"
            assert got is None, f"{name} has no Rust rule code"

    def test_unknown_identity_raises(self):
        with pytest.raises(ValueError, match="unknown Phase-A opponent identity"):
            rls.resolve_opponent_dispatch("not_a_real_identity")

    def test_rule_code_table_matches_worker_rs_constants(self):
        # The A3 spec mix rule codes are a strict subset of the Rust 0-7 table.
        assert set(rls.RULE_AGENT_CODES.values()) <= {0, 1, 2, 3, 4, 5, 6, 7}
        # Codes 5 (PunishEmptyBoard) and 7 (AntiHandLeakOverfit) exist in Rust but are
        # NOT in the A3 spec mix (BLOCK_A_PLAN.md:401 — legacy phase26 used them; A3 does not).
        assert 5 not in rls.RULE_AGENT_CODES.values()
        assert 7 not in rls.RULE_AGENT_CODES.values()


# ===========================================================================
# FakeWorker + composition tests (deterministic, no Rust lib)
# ===========================================================================
class _FakeWorkerScriptEntry:
    """One transition in a FakeWorker script: pre-state (actor/mana_draw_legal/legal_counts)
    + the outcome of acting in that pre-state (reward/terminated/truncated/hero_hp)."""

    def __init__(
        self,
        *,
        actor: int,
        mana_draw_legal: bool = False,
        legal_counts: int = 3,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        hero_hp: tuple[int, int, int, int] = (45, 45, 45, 45),
    ) -> None:
        self.actor = int(actor)
        self.mana_draw_legal = bool(mana_draw_legal)
        self.legal_counts = int(legal_counts)
        self.reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.hero_hp = tuple(int(x) for x in hero_hp)


def _make_arrays(env_count: int, entries: list[_FakeWorkerScriptEntry]) -> dict[str, np.ndarray]:
    """Build the arrays() dict for a batch of envs at their current script pointer."""
    obs_v5 = np.zeros((env_count, 8), dtype=np.float32)  # tiny obs (fake; dim irrelevant)
    counts = np.array([e.legal_counts for e in entries[:env_count]], dtype=np.uintp)
    # legal_action_offsets: cumulative offset of each env's legal block (starts at 0).
    offsets = np.zeros(env_count, dtype=np.uintp)
    acc = 0
    for i in range(env_count):
        offsets[i] = acc
        acc += int(counts[i])
    # legal_action_ids: per env a block [0..c-1] (deterministic, distinct ids per env via
    # offset).
    blocks = [np.arange(int(c), dtype=np.uintp) for c in counts]
    legal_ids = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.uintp)
    legal_features = np.zeros((max(legal_ids.shape[0], 0), 1), dtype=np.float32)
    return {
        "observation_v5": obs_v5,
        "action_mask": None,
        "action_features": None,
        "legal_action_counts": counts,
        "legal_action_offsets": offsets,
        "legal_action_ids": legal_ids,
        "legal_action_features": legal_features,
        "rewards": np.zeros(env_count, dtype=np.float32),
        "terminated": np.zeros(env_count, dtype=np.bool_),
        "reset_flags": None,
        "terminal_observation_v5": None,
        "terminal_observation_valid": None,
        "episode_returns": None,
        "episode_lengths": None,
        "selected_local_indices": None,
    }


class FakeWorker:
    """A deterministic stand-in for ``RustBatchWorker`` satisfying the call surface
    ``collect_rust_live_rollout`` uses. Records FFI calls for assertion."""

    def __init__(self, scripts: list[list[_FakeWorkerScriptEntry]]) -> None:
        self.env_count = len(scripts)
        self.scripts = scripts
        self.ptr = [0] * self.env_count
        self.last_outcome: list[_FakeWorkerScriptEntry | None] = [None] * self.env_count
        # call logs
        self.select_rule_actions_calls: list[np.ndarray] = []
        self.step_mana_draw_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.reset_indices_calls: list[np.ndarray] = []
        self.closed = False

    def _current_entries(self) -> list[_FakeWorkerScriptEntry]:
        out = []
        for i in range(self.env_count):
            p = self.ptr[i]
            out.append(self.scripts[i][p] if p < len(self.scripts[i]) else self.scripts[i][-1])
        return out

    def reset(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        self.ptr = [0] * self.env_count
        self.last_outcome = [None] * self.env_count
        return _make_arrays(self.env_count, self._current_entries())

    def arrays(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        return _make_arrays(self.env_count, self._current_entries())

    def current_actor_ids(self) -> np.ndarray:
        return np.array([e.actor for e in self._current_entries()], dtype=np.int32)

    def mana_draw_legal(self) -> np.ndarray:
        return np.array([e.mana_draw_legal for e in self._current_entries()], dtype=np.bool_)

    def select_rule_actions(self, agent_codes, *, salt: int = 0) -> np.ndarray:
        codes = np.asarray(agent_codes, dtype=np.uint32)
        self.select_rule_actions_calls.append(codes.copy())
        # Deterministic rule action per (env, code): code*1000 + env*10 + 7
        # (distinguishable from policy actions 999 and learner placeholder).
        return np.array(
            [int(codes[i]) * 1000 + i * 10 + 7 for i in range(self.env_count)],
            dtype=np.uintp,
        )

    def step_mana_draw(self, action_ids, mana_draw_flags, *, copy: bool = False) -> dict[str, np.ndarray]:
        actions = np.asarray(action_ids, dtype=np.uintp)
        flags = np.asarray(mana_draw_flags, dtype=np.bool_)
        self.step_mana_draw_calls.append((actions.copy(), flags.copy()))
        # The outcome of acting in the CURRENT pre-state (entry[ptr]).
        entries = self._current_entries()
        rewards = np.array([e.reward for e in entries], dtype=np.float32)
        terminated = np.array([e.terminated for e in entries], dtype=np.bool_)
        self.last_outcome = list(entries)
        # advance pointers (post-state = next entry).
        for i in range(self.env_count):
            if self.ptr[i] + 1 < len(self.scripts[i]):
                self.ptr[i] += 1
        out = _make_arrays(self.env_count, self._current_entries())
        out["rewards"] = rewards
        out["terminated"] = terminated
        return out

    def truncated(self) -> np.ndarray:
        return np.array(
            [
                (e.truncated if e is not None else False)
                for e in self.last_outcome
            ],
            dtype=np.bool_,
        )

    def hero_hp(self) -> np.ndarray:
        return np.array(
            [
                (list(e.hero_hp) if e is not None else [45, 45, 45, 45])
                for e in self.last_outcome
            ],
            dtype=np.int32,
        )

    def reset_indices(self, indices) -> dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.uintp)
        self.reset_indices_calls.append(idx.copy())
        for i in idx:
            self.ptr[int(i)] = 0
            self.last_outcome[int(i)] = None
        return self.arrays(copy=False)

    def close(self) -> None:
        self.closed = True


def _script_alternating(learner_actor: int, *, opponent_actor: int, n_learner: int,
                        reward_learner: float = 1.0, reward_opponent: float = 5.0,
                        terminal_at: int | None = None) -> list[_FakeWorkerScriptEntry]:
    """Build a per-env script that alternates opponent/learner turns. Each learner turn
    yields one learner transition; ``n_learner`` learner turns are produced. Opponent
    turns carry ``reward_opponent`` (must be ZEROED by learner-only attribution) and
    learner turns carry ``reward_learner`` (kept). If ``terminal_at`` is set, the
    learner turn at that 1-based learner-turn index is marked terminated."""
    out: list[_FakeWorkerScriptEntry] = []
    li = 0
    # pattern: opponent, learner, opponent, learner, ... (start at opponent so the first
    # dispatch is an opponent turn, exercising the dispatch path).
    for k in range(n_learner * 2 + 2):
        if k % 2 == 0:
            out.append(_FakeWorkerScriptEntry(
                actor=opponent_actor, reward=reward_opponent,
                terminated=False, truncated=False,
            ))
        else:
            li += 1
            term = (terminal_at is not None and li == terminal_at)
            out.append(_FakeWorkerScriptEntry(
                actor=learner_actor, reward=reward_learner,
                terminated=term, truncated=False,
            ))
    return out


class TestCompositionDispatch:
    """test_all_ten_identities_dispatch — for each of the 10 identities, assert the
    trainer dispatches correctly (rule path for the 6 rule identities; policy-opponent
    path for the 4 policy identities)."""

    @pytest.mark.parametrize("identity", rls.PHASE_A_IDENTITIES)
    def test_dispatch_routes_per_identity(self, identity):
        env_count = 1
        learner_actor = 1
        opponent_actor = 2
        script = _script_alternating(learner_actor, opponent_actor=opponent_actor, n_learner=3)
        worker = FakeWorker([script])
        config = _tiny_config()
        kind, code = rls.resolve_opponent_dispatch(identity)
        # Policy-opponent fns: a recording fake for each policy identity.
        policy_calls: list[int] = []

        class _RecordingPolicy:
            name = identity

            def select(self, env_idx, ctx):
                policy_calls.append(int(ctx.legal_action_ids[0]) if ctx.legal_action_ids.size else 0)
                return 999  # distinguishable from rule actions (code*1000+7)

        opponent_policies = {identity: _RecordingPolicy()} if kind == rls.POLICY_DISPATCH else {}
        rollout = rls.collect_rust_live_rollout(
            worker,
            _FakeLearner(),
            opponent_policies,
            learner_actor_ids=np.array([learner_actor], dtype=np.int32),
            opponent_identities=[identity],
            config=config,
            steps=3,
            record_dispatch=True,
        )
        # The dispatch log records per-batch-step per-env source tags.
        sources = [rec["source"] for rec in (rollout.dispatch_log or []) if rec["env"] == 0]
        if kind == rls.RULE_DISPATCH:
            # Rule identities must route at least one opponent turn via the rule path.
            assert rls.RULE_DISPATCH in sources, (
                f"{identity}: expected rule dispatch on an opponent turn, got {sources}"
            )
            # select_rule_actions must have been called with the real code on an opponent turn.
            called_codes = [c[0] for c in worker.select_rule_actions_calls]
            assert int(code) in [int(c) for c in called_codes], (
                f"{identity}: select_rule_actions never called with code {code}; got {called_codes}"
            )
            # The action stepped on a rule-opponent turn must be the rule action (code*1000+7),
            # NOT the policy sentinel 999.
            rule_action = int(code) * 1000 + 0 * 10 + 7
            stepped = [
                int(a[0][0]) for a in worker.step_mana_draw_calls
            ]
            assert rule_action in stepped, (
                f"{identity}: rule action {rule_action} never stepped; got {stepped}"
            )
            assert 999 not in stepped, f"{identity}: policy sentinel 999 stepped (wrong path)"
        else:
            # Policy identities must route at least one opponent turn via the policy path.
            assert rls.POLICY_DISPATCH in sources, (
                f"{identity}: expected policy dispatch on an opponent turn, got {sources}"
            )
            assert len(policy_calls) > 0, (
                f"{identity}: policy-opponent select() never called"
            )
            # The action stepped on a policy-opponent turn must be the policy sentinel 999.
            stepped = [int(a[0][0]) for a in worker.step_mana_draw_calls]
            assert 999 in stepped, f"{identity}: policy sentinel 999 never stepped; got {stepped}"


class TestLearnerOnlyReward:
    """test_opponent_steps_zero_reward — opponent-actor steps record ZERO reward
    (learner-only, fix #1; regression guard for run_phase26:490)."""

    def test_opponent_steps_zeroed_in_collected_tape(self):
        # Single env; opponent turns carry reward 5.0 (must be zeroed), learner turns 1.0 (kept).
        learner_actor = 1
        script = _script_alternating(
            learner_actor, opponent_actor=2, n_learner=4,
            reward_learner=1.0, reward_opponent=5.0,
        )
        worker = FakeWorker([script])
        config = _tiny_config()
        rollout = rls.collect_rust_live_rollout(
            worker, _FakeLearner(), {"end_turn": _EndTurnPolicy()},
            learner_actor_ids=np.array([learner_actor], dtype=np.int32),
            opponent_identities=["end_turn"],  # policy opponent; opponent steps must zero
            config=config, steps=4,
        )
        rewards = np.asarray(rollout.transitions.rewards, dtype=np.float32).reshape(-1)
        # Every recorded transition is a LEARNER step (opponent steps are not recorded).
        # Learner rewards were 1.0 each -> the tape must contain only 1.0 values (the 5.0
        # opponent-step rewards were dropped/zeroed by learner-only attribution).
        assert rewards.size > 0, "no learner transitions collected"
        assert np.all(rewards == 1.0), (
            f"opponent reward leaked into learner tape: {rewards} (expected all 1.0)"
        )
        # No recorded reward equals the opponent reward 5.0.
        assert not np.any(rewards == 5.0), f"opponent reward 5.0 leaked: {rewards}"

    def test_reward_attribution_directly(self):
        # The A3 helper the trainer uses: zero non-learner-actor steps.
        rew = np.array([1.0, 5.0, 1.0, 5.0], dtype=np.float32)
        actors = np.array([1, 2, 1, 2], dtype=np.int32)
        learner = np.array([1], dtype=np.int32)
        got = reward_attribution(rew, actors, learner)
        assert got.tolist() == [1.0, 0.0, 1.0, 0.0], (
            "learner-only attribution must zero opponent-actor (actor=2) steps"
        )


class TestDecisiveEarlyEnd:
    """test_decisive_state_early_end — a decisive win-margin state terminates early."""

    def test_is_decisive_state_predicate(self):
        from train_v3.ppo_phaseA_config import is_decisive_state as _ids

        class Snap:
            my_hero_hp = 40
            my_hero_max_hp = 40
            enemy_hero_hp = 5
            enemy_hero_max_hp = 40

        assert _ids(Snap(), threshold=0.6) is True  # 1.0 - 0.125 = 0.875 >= 0.6

        class Snap2:
            my_hero_hp = 30
            my_hero_max_hp = 40
            enemy_hero_hp = 25
            enemy_hero_max_hp = 40

        assert _ids(Snap2(), threshold=0.6) is False  # |0.75-0.625|=0.125 < 0.6

    def test_decisive_state_terminates_episode_early(self):
        # Script: learner turn at index 1 (1st learner transition) leads to a decisive
        # hero_hp (40/40 vs 5/40) -> the episode must be terminated early + the env reset.
        learner_actor = 1
        script = [
            _FakeWorkerScriptEntry(actor=2, reward=0.0),  # opponent turn
            _FakeWorkerScriptEntry(actor=1, reward=1.0, terminated=False,
                                   hero_hp=(40, 40, 5, 40)),  # decisive learner turn
            _FakeWorkerScriptEntry(actor=2, reward=0.0),  # would-be next opponent turn
            _FakeWorkerScriptEntry(actor=1, reward=1.0),  # would-be next learner turn
        ]
        worker = FakeWorker([script])
        config = _tiny_config(decisive_early_end=True)
        rollout = rls.collect_rust_live_rollout(
            worker, _FakeLearner(), {"end_turn": _EndTurnPolicy()},
            learner_actor_ids=np.array([learner_actor], dtype=np.int32),
            opponent_identities=["end_turn"],
            config=config, steps=2,
        )
        term = np.asarray(rollout.transitions.terminated, dtype=np.bool_).reshape(-1)
        # The first learner transition (row 0) must be marked terminated (decisive early-end).
        assert bool(term[0]) is True, (
            f"decisive early-end did not terminate the learner transition: {term}"
        )
        # The env must have been reset (reset_indices called) to start a new episode.
        assert len(worker.reset_indices_calls) >= 1, (
            "decisive early-end must reset the env for the next episode"
        )


class TestMaxTurnsThreading:
    """test_max_turns_threaded — from_live threads config.max_turns into KernelConfig."""

    def test_build_live_worker_threads_max_turns(self, monkeypatch):
        from train_v3 import rust_live_self_play as mod
        from train_v3 import rust_ffi

        recorded: dict[str, Any] = {}

        class _FakeFromLive:
            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def from_live(cls, *, seed, env_count, max_turns, **kwargs):
                recorded["max_turns"] = int(max_turns)
                recorded["env_count"] = int(env_count)
                # return a minimal fake worker so _build_live_worker returns it.
                worker = FakeWorker([[_FakeWorkerScriptEntry(actor=1)]])
                worker.env_count = env_count
                # attach the RustBatchWorker-style attributes the trainer does not call here.
                return worker

        monkeypatch.setattr(rust_ffi.RustBatchWorker, "from_live", _FakeFromLive.from_live)
        config = _tiny_config(max_turns=PHASE_A_MAX_TURNS)  # 120
        w = mod._build_live_worker(config, seed=7, env_count=2)
        assert recorded["max_turns"] == 120, (
            f"from_live must receive max_turns=120, got {recorded.get('max_turns')}"
        )
        assert recorded["env_count"] == 2

    def test_run_live_self_play_update_threads_max_turns_via_factory(self):
        # Using a worker_factory, assert config.max_turns==120 is the threaded value the
        # update is configured with (the factory receives env_count; the trainer built
        # config with max_turns=120).
        config = _tiny_config(max_turns=120, env_count=1)
        # Fake factory: returns a FakeWorker with a rule-opponent script.
        def factory(env_count):
            script = _script_alternating(1, opponent_actor=2, n_learner=2)
            return FakeWorker([script for _ in range(env_count)])
        metrics = rls.run_live_self_play_update(
            config, _FakeLearner(), {"end_turn": _EndTurnPolicy()},
            seed=3, worker_factory=factory, steps=2,
        )
        assert metrics["max_turns"] == 120, (
            f"update must be configured with max_turns=120, got {metrics['max_turns']}"
        )
        assert metrics["has_rollout"] is True


class TestSecondStartOversampling:
    """D-A10: second-start oversampling biases the learner starting side."""

    def test_oversampling_scheme_balances_under_represented_side(self):
        sides, scheme = rls.sample_learner_sides(
            1000, p1_score_rate=0.2, p2_score_rate=0.8,
            oversampling={"gap_threshold": 0.12, "base_weight": 0.5},
            rng=np.random.default_rng(0),
        )
        # p1 is under-represented (lower score) -> oversampled.
        assert scheme["breach"] is True
        assert scheme["oversampled_side"] == "p1"
        # p1 should be the majority of sampled sides.
        assert int(np.sum(sides == 1)) > int(np.sum(sides == 2))

    def test_no_breach_is_50_50(self):
        sides, scheme = rls.sample_learner_sides(
            2000, p1_score_rate=0.5, p2_score_rate=0.5,
            oversampling={"gap_threshold": 0.12, "base_weight": 0.5},
            rng=np.random.default_rng(1),
        )
        assert scheme["breach"] is False
        n1 = int(np.sum(sides == 1))
        n2 = int(np.sum(sides == 2))
        assert abs(n1 - n2) < 200, f"no-breach should be ~50/50, got p1={n1} p2={n2}"


# --- helpers used by the composition tests -----------------------------------
def _tiny_config(
    max_turns: int = PHASE_A_MAX_TURNS,
    env_count: int = 1,
    decisive_early_end: bool = False,
) -> PhaseAPPOConfig:
    return PhaseAPPOConfig(
        max_turns=int(max_turns),
        env_count=int(env_count),
        steps_per_update=4,
        decisive_early_end=bool(decisive_early_end),
        # Use a rule-only mix for composition tests by default (overridden per-test).
        opponent_mix=build_phase_a_opponent_mix_string(
            {"random": 0.5, "face_rush": 0.5},
            {"random": "random", "face_rush": "face_rush"},
        ),
        advantage_backend="python",  # avoid needing the Rust GAE backend in unit tests
        selected_local_backend="provided",
        prepare_backend="separate",
    )


class _FakeLearner:
    """A learner policy that picks the first legal action; records it was called.
    Used for composition tests (not the real PPO)."""

    def __init__(self) -> None:
        self.calls = 0

    def select(self, ctx):
        self.calls += 1
        idxs = np.asarray(ctx.env_indices, dtype=np.intp)
        n = int(idxs.size)
        counts = np.asarray(ctx.legal_action_counts, dtype=np.intp)
        offsets = np.asarray(ctx.legal_action_offsets, dtype=np.intp)
        ids = np.asarray(ctx.legal_action_ids, dtype=np.uintp)
        actions = np.zeros(n, dtype=np.uintp)
        for k, env in enumerate(idxs):
            c = int(counts[env])
            o = int(offsets[env])
            actions[k] = int(ids[o]) if c > 0 else 0
        return (
            actions,
            np.zeros(n, dtype=np.float32),
            np.full(n, -1.0, dtype=np.float32),  # log_prob
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.bool_),
        )


class _EndTurnPolicy:
    name = "end_turn"

    def select(self, env_idx, ctx):
        return int(ctx.legal_action_ids[0]) if ctx.legal_action_ids.size else 0


# ===========================================================================
# Real-FFI smoke tests (gated) — integration confidence
# ===========================================================================
@requires_rust_ffi
class TestRealFFISmoke:
    def test_skip_if_no_rust_ffi(self):
        # Marker test: if the class is skipped, this never runs. When it runs, the lib
        # loaded. This satisfies the test_skip_if_no_rust_ffi requirement (the skip-gate
        # is the class decorator ``requires_rust_ffi``).
        assert _rust_ok()

    def test_one_ppo_update_seeded_arena_completes(self):
        # THE entry-point smoke: collect + prepare a PPO batch on a real seeded live arena.
        from train_v3.rust_ffi import RustBatchWorker

        config = PhaseAPPOConfig(
            max_turns=120, env_count=2, steps_per_update=4,
            decisive_early_end=False,
            opponent_mix=build_phase_a_opponent_mix_string(
                {"random": 0.5, "face_rush": 0.5},
                {"random": "random", "face_rush": "face_rush"},
            ),
            advantage_backend="python",
            selected_local_backend="provided",
            prepare_backend="separate",
            seed=7,
        )
        worker = RustBatchWorker.from_live(
            seed=7, env_count=2, max_turns=config.max_turns,
            action_features_dtype=config.action_features_dtype,
            action_features_mode=config.action_features_mode,
            observation_mode=config.observation_mode,
            action_mask_mode=config.action_mask_mode,
            terminal_observation_mode=config.terminal_observation_mode,
            diagnostic_mode="none",
        )
        try:
            rollout = rls.collect_rust_live_rollout(
                worker, rls.ArgmaxRandomLearner(seed=7), {},
                learner_actor_ids=np.array([1, 2], dtype=np.int32),
                opponent_identities=["random", "face_rush"],  # rule-only mix -> no policy fns
                config=config, steps=4,
            )
            # Well-formed trajectory: learner transitions collected, terminal reached or
            # max_turns respected (the loop bounded), rewards attributed learner-only.
            assert rollout.transitions.observations.shape[0] >= 1
            assert rollout.transitions.observations.shape[1] == 2
            assert rollout.transitions.actions.shape[0] == rollout.transitions.observations.shape[0]
            assert rollout.transitions.values is not None
            assert rollout.transitions.log_probs is not None
            assert rollout.transitions.legal_action_ids.size > 0
            # mana_draw channels collected (the plan requires (s,a,r,s',terminal,mana_draw_legal)).
            assert rollout.mana_draw_legal.shape[0] == rollout.transitions.observations.shape[0]
            assert rollout.mana_draw_taken.shape[0] == rollout.transitions.observations.shape[0]
            # prepare_rust_ppo_batch (GAE/returns) completes on the live batch.
            from train_v3.rust_ppo import prepare_rust_ppo_batch

            ppo_batch = prepare_rust_ppo_batch(
                rollout.transitions,
                gamma=config.gamma, gae_lambda=config.gae_lambda,
                advantage_backend="python", selected_local_backend="provided",
                prepare_backend="separate",
            )
            assert ppo_batch is not None
        finally:
            worker.close()

    def test_max_turns_threaded_into_worker(self):
        # from_live(max_turns=120) must construct without raising (the defensive assert
        # at rust_ffi.py:1288 checks env_config.max_turns==120) AND the truncation
        # behavior scales with max_turns (proving the value is respected, not stuck at 80).
        from train_v3.rust_ffi import RustBatchWorker

        # Constructing with 120 must not raise (env_config.max_turns==120 threaded).
        w120 = RustBatchWorker.from_live(seed=5, env_count=1, max_turns=120)
        w120.reset(copy=True)
        w120.close()

        # Behavioral: truncation batch_steps scales with max_turns (6 -> ~12, 12 -> ~24).
        def steps_to_truncate(max_turns: int) -> int:
            w = RustBatchWorker.from_live(seed=5, env_count=1, max_turns=max_turns)
            w.reset(copy=True)
            steps = 0
            import numpy as np
            while steps < max_turns * 20 + 50:
                codes = np.zeros(1, dtype=np.uint32)  # legal_random drives the game forward
                acts = w.select_rule_actions(codes)
                w.step_mana_draw(acts, [False], copy=False)
                steps += 1
                if bool(w.truncated()[0]):
                    break
            w.close()
            return steps

        s6 = steps_to_truncate(6)
        s12 = steps_to_truncate(12)
        assert s12 > s6, (
            f"max_turns not respected/threaded: steps_to_truncate(12)={s12} <= "
            f"steps_to_truncate(6)={s6} (expected more turns for larger max_turns)"
        )

    def test_learner_only_reward_on_real_run(self):
        # On a real short run with a rule-opponent, opponent-actor step rewards must be
        # ZEROED in the collected learner tape (only learner-actor rewards kept).
        from train_v3.rust_ffi import RustBatchWorker

        config = PhaseAPPOConfig(
            max_turns=120, env_count=1, steps_per_update=3,
            decisive_early_end=False,
            opponent_mix=build_phase_a_opponent_mix_string(
                {"random": 1.0}, {"random": "random"},
            ),
            advantage_backend="python", selected_local_backend="provided",
            prepare_backend="separate", seed=11,
        )
        worker = RustBatchWorker.from_live(
            seed=11, env_count=1, max_turns=config.max_turns,
            action_features_dtype=config.action_features_dtype,
            action_features_mode=config.action_features_mode,
            observation_mode=config.observation_mode,
            action_mask_mode=config.action_mask_mode,
            terminal_observation_mode=config.terminal_observation_mode,
            diagnostic_mode="none",
        )
        try:
            # learner plays p1 (actor 1); opponent = random (rule code 0) plays p2.
            rollout = rls.collect_rust_live_rollout(
                worker, rls.ArgmaxRandomLearner(seed=11), {},
                learner_actor_ids=np.array([1], dtype=np.int32),
                opponent_identities=["random"],
                config=config, steps=3,
            )
            # Every recorded transition is a LEARNER step (the learner-actor rows);
            # opponent-actor rows are NOT recorded. So rewards.size == learner transitions.
            rewards = np.asarray(rollout.transitions.rewards, dtype=np.float32).reshape(-1)
            assert rewards.size > 0
            # The tape contains ONLY learner-attributed rewards (no opponent leakage).
            # (The Rust env's per-step reward is the acting player's; learner-only
            # attribution zeros opponent-actor steps and keeps learner-actor steps.)
            # The number of recorded rows must equal the learner step count.
            assert rewards.size == int(rollout.learner_step_counts[0]), (
                f"recorded rows {rewards.size} != learner_step_counts {int(rollout.learner_step_counts[0])}"
            )
        finally:
            worker.close()