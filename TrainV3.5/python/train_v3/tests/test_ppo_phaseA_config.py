"""Block A component A3 — ppo_phaseA_config tests (TRACKED, PURE-PYTHON).

Five items (six functions; item 3 splits into two) for
``TrainV3.5/python/train_v3/ppo_phaseA_config.py`` — the Phase-A PPO config dataclass
+ helpers encoding the 5 root-cause fixes (``design.md:103-109``).

PURE-PYTHON: NO MLX import, NO Rust FFI call. The max_turns threading is tested via
the pure-python ``build_trace_env_config()`` dict + the documented
``LIVE_MAX_TURNS_THREADING_NOTE`` (which references ``kernel.rs:660``), NOT a real
``KernelConfig`` construction (that is A4's Rust-side job). ``ppo_phaseA_config``
imports ``league_v5`` + ``gauntlet_v5`` (pure-python, no mlx; ``rust_ffi`` is
lazy-load so no ``.dylib`` is opened at import). ``rust_trainer`` is NOT imported by
the config module at top level (only lazily inside ``to_rust_ppo_config()``, which
these tests never call), so this file never triggers the Rust import chain.

  (1) ``test_learner_only_reward_zeroes_opponent`` — ``reward_attribution`` zeroes
      opponent-actor rewards, keeps learner-actor rewards; sum of attributed ==
      learner raw rewards (regression guard for ``run_phase26:490
      step_rewards = learner_rewards + opponent_rewards``).
  (2) ``test_opponent_mix_parses_to_spec_weights`` — graduated mix parses via
      ``league_v5.parse_v5_opponent_mix`` (read-only) to exactly the 10 spec weights
      summing to 1.0; display names are NOT accepted by the parser (validation !=
      dispatch — A4 implements runtime dispatch).
  (3a) ``test_max_turns_and_decisive_early_end`` — ``config.max_turns == 120`` +
       ``decisive_early_end`` encoded as a pure predicate of a state snapshot.
  (3b) ``test_max_turns_threaded_to_rust`` — ``build_trace_env_config()`` carries
       ``max_turns == 120`` (trace-pool plumbing) + the live-constructor threading is
       documented (``LIVE_MAX_TURNS_THREADING_NOTE`` references ``kernel.rs:660`` +
       serde default 80). (verifier finding 2b)
  (4) ``test_entropy_and_epochs_pinned`` — ``entropy_coef == 0.01`` (NOT 0.035),
       ``epochs == 6`` (NOT 1 / 3) — regression guards.
  (5) ``test_second_start_oversampling_scheme`` — p1/p2 oversampling scheme present
      + well-formed (weights sum to 1.0, both in [0,1]; breach oversamples the
      under-represented side; balanced when gap <= 0.12).

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest \
TrainV3.5/python/train_v3/tests/test_ppo_phaseA_config.py``
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from train_v3.ppo_phaseA_config import (
    LIVE_MAX_TURNS_THREADING_NOTE,
    PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD,
    PHASE_A_EPOCHS,
    PHASE_A_ENTROPY_COEF,
    PHASE_A_MAX_TURNS,
    PHASE_A_OPPONENT_MIX_SPEC,
    PHASE_A_OPPONENT_NAME_ALIASES,
    PHASE_A_P1_P2_GAP_THRESHOLD,
    PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC,
    PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE,
    PhaseAPPOConfig,
    build_phase_a_opponent_mix_string,
    build_phase_a_random_bootstrap_config,
    build_phase_a_random_bootstrap_opponent_mix_string,
    build_trace_env_config,
    is_decisive_state,
    reward_attribution,
    second_start_oversampling_scheme,
    validate_phase_a_opponent_mix,
    validate_phase_a_random_bootstrap_opponent_mix,
)


def _snap(my_hp, my_max, en_hp, en_max):
    """Duck-typed V5RewardSnapshot (reward_v5.py:12-52) for the decisive predicate."""
    return SimpleNamespace(
        my_hero_hp=my_hp,
        my_hero_max_hp=my_max,
        enemy_hero_hp=en_hp,
        enemy_hero_max_hp=en_max,
    )


def test_learner_only_reward_zeroes_opponent():
    # Fix #1: trainer-side learner-only attribution (regression guard for
    # run_phase26:490 `step_rewards = learner_rewards + opponent_rewards`).
    # step_rewards tape: learner, opponent, learner, opponent
    step_rewards = np.array([1.0, -0.5, 2.0, 0.3], dtype=np.float32)
    actor_ids = np.array([10, 20, 10, 20])
    learner_actor = 10

    attributed = reward_attribution(step_rewards, actor_ids, learner_actor)

    # opponent-actor steps zeroed; learner-actor steps kept
    assert attributed.tolist() == [1.0, 0.0, 2.0, 0.0]
    assert float(attributed[1]) == 0.0  # opponent step zeroed
    assert float(attributed[3]) == 0.0  # opponent step zeroed
    # the sum of attributed rewards == the learner's raw rewards (opponent credit gone)
    learner_raw = float(step_rewards[actor_ids == learner_actor].sum())
    assert float(attributed.sum()) == pytest.approx(learner_raw)
    assert float(attributed.sum()) == pytest.approx(3.0)  # 1.0 + 2.0

    # also works with plain python lists + scalar learner id
    attributed_list = reward_attribution([1.0, -0.5, 2.0, 0.3], [10, 20, 10, 20], 10)
    assert attributed_list.tolist() == [1.0, 0.0, 2.0, 0.0]

    # a tape with NO learner steps -> all zero (opponent-only tape yields no credit)
    all_opp = reward_attribution([0.5, 0.7, -0.2], [20, 20, 20], 10)
    assert all_opp.tolist() == [0.0, 0.0, 0.0]

    # config carries the Fix #1 flag (default ON)
    assert PhaseAPPOConfig().learner_only_reward is True


def test_opponent_mix_parses_to_spec_weights():
    # Fix #5: the graduated mix VALIDATES via the EXISTING league_v5.parse_v5_opponent_mix
    # (read-only call; parser NOT edited) to exactly the 10 spec weights summing to 1.0.
    parsed = validate_phase_a_opponent_mix()
    assert len(parsed) == len(PHASE_A_OPPONENT_MIX_SPEC) == 10

    # exact spec weights (design.md:111)
    expected_spec = {
        "legal_random": 0.10,
        "end_turn": 0.05,
        "greedy_face": 0.10,
        "face_rush": 0.10,
        "board_control": 0.10,
        "greedy_trade": 0.10,
        "stall": 0.10,
        "anti_draw_greed": 0.10,
        "self_prev": 0.10,
        "v4-orig-argmax": 0.15,
    }
    assert PHASE_A_OPPONENT_MIX_SPEC == expected_spec

    # each parsed (canonical, weight) maps back through aliases to the spec weight
    canonical_to_display = {
        PHASE_A_OPPONENT_NAME_ALIASES[d]: d for d in PHASE_A_OPPONENT_NAME_ALIASES
    }
    parsed_dict = {canonical: weight for canonical, weight in parsed}
    for display, spec_weight in PHASE_A_OPPONENT_MIX_SPEC.items():
        canonical = PHASE_A_OPPONENT_NAME_ALIASES[display]
        assert canonical in parsed_dict, f"{display}->{canonical} missing from parsed mix"
        assert parsed_dict[canonical] == pytest.approx(spec_weight), (
            f"{display} ({canonical}): parsed {parsed_dict[canonical]} != spec {spec_weight}"
        )

    # weights sum to 1.0
    assert sum(PHASE_A_OPPONENT_MIX_SPEC.values()) == pytest.approx(1.0)
    assert sum(weight for _, weight in parsed) == pytest.approx(1.0)

    # the config default opponent_mix string == the built canonical string
    cfg = PhaseAPPOConfig()
    assert cfg.opponent_mix == build_phase_a_opponent_mix_string()
    # every canonical name in the string is a known V5 opponent kind (parse succeeds)
    from train_v3.league_v5 import parse_v5_opponent_mix

    parse_v5_opponent_mix(cfg.opponent_mix)  # no raise

    # validation != dispatch: the DISPLAY names (legal_random / self_prev /
    # v4-orig-argmax) are NOT accepted by the parser — A3's alias layer translates them,
    # and the parser stays strict (unedited). A4 implements runtime dispatch.
    with pytest.raises(ValueError):
        parse_v5_opponent_mix("legal_random:0.5")  # display name -> rejected
    with pytest.raises(ValueError):
        parse_v5_opponent_mix("self_prev:0.5")  # display name -> rejected
    with pytest.raises(ValueError):
        parse_v5_opponent_mix("v4-orig-argmax:0.5")  # display name -> rejected


def test_random_bootstrap_mix_is_teacher_free_and_random_heavy():
    parsed = validate_phase_a_random_bootstrap_opponent_mix()
    parsed_dict = {name: weight for name, weight in parsed}

    assert PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE == pytest.approx(0.98)
    assert PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC == {
        "legal_random": 0.70,
        "end_turn": 0.05,
        "greedy_face": 0.10,
        "face_rush": 0.05,
        "board_control": 0.05,
        "greedy_trade": 0.05,
    }
    assert sum(PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC.values()) == pytest.approx(1.0)
    assert parsed_dict["random"] == pytest.approx(0.70)

    # Phase-A bootstrap is direct ArenaEnv PPO, not teacher-data distillation.
    assert "self_prev" not in PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC
    assert "v4-orig-argmax" not in PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC
    assert "llm_teacher" not in PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC
    bootstrap_mix = build_phase_a_random_bootstrap_opponent_mix_string()
    assert "v4max" not in bootstrap_mix
    assert "self" not in bootstrap_mix

    cfg = build_phase_a_random_bootstrap_config(run_name="phaseA_random_bootstrap_test")
    assert cfg.run_name == "phaseA_random_bootstrap_test"
    assert cfg.opponent_mix == bootstrap_mix
    assert cfg.opponent_mix_spec == PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC
    assert cfg.curriculum_metadata["phase_a_profile"] == "random_bootstrap"
    assert cfg.curriculum_metadata["distillation"] == "disabled"
    assert cfg.curriculum_metadata["teacher_source"] == "none"
    assert cfg.curriculum_metadata["target_random_score"] == pytest.approx(0.98)


def test_random_bootstrap_config_accepts_validated_custom_mix():
    cfg = build_phase_a_random_bootstrap_config(
        opponent_mix="random:1.0",
        opponent_mix_spec={"random": 1.0},
        curriculum_metadata={"opponent_mix_override": "random:1.0"},
    )

    assert cfg.opponent_mix == "random:1.0"
    assert cfg.opponent_mix_spec == {"random": 1.0}
    assert cfg.curriculum_metadata["phase_a_profile"] == "random_bootstrap"
    assert cfg.curriculum_metadata["opponent_mix_override"] == "random:1.0"


def test_max_turns_and_decisive_early_end():
    cfg = PhaseAPPOConfig()
    # Fix #2: max_turns == 120 (>= 120 per design.md:106)
    assert cfg.max_turns == PHASE_A_MAX_TURNS == 120
    assert cfg.max_turns >= 120
    # decisive_early_end encoded (flag + threshold + pure predicate)
    assert cfg.decisive_early_end is True
    assert cfg.decisive_win_margin_threshold == PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD

    # decisive predicate is a PURE function of a state snapshot (deterministic)
    assert is_decisive_state(_snap(30, 30, 30, 30)) is False  # balanced
    assert is_decisive_state(_snap(20, 30, 22, 30)) is False  # lead 0.067 < 0.60
    # overwhelming lead -> decisive (1.0 - 0.3 = 0.7 >= 0.60)
    assert is_decisive_state(_snap(30, 30, 9, 30)) is True
    # symmetric: enemy dominating learner -> still decisive (absolute margin)
    assert is_decisive_state(_snap(9, 30, 30, 30)) is True
    # determinism: same snapshot -> same result (pure function)
    s = _snap(30, 30, 5, 30)
    assert is_decisive_state(s) == is_decisive_state(s) is True
    # threshold boundary: lead == 0.60 (1.0 - 12/30 = 0.60)
    s_boundary = _snap(30, 30, 12, 30)
    assert is_decisive_state(s_boundary, threshold=0.60) is True  # >= boundary
    assert is_decisive_state(s_boundary, threshold=0.61) is False  # 0.60 < 0.61
    # just below default threshold: lead 0.5667 < 0.60
    s_below = _snap(30, 30, 13, 30)  # 1.0 - 0.4333 = 0.5667
    assert is_decisive_state(s_below) is False  # default 0.60
    assert is_decisive_state(s_below, threshold=0.55) is True  # 0.5667 >= 0.55


def test_max_turns_threaded_to_rust():
    # Fix #2 trace-pool plumbing: build_trace_env_config returns a dict carrying
    # max_turns==120 (pure-python, no FFI). trace_factory_v5.py MUST write this into
    # trace['env_config'] so kernel.rs:660 from_trace_config reads 120, NOT the serde
    # default 80 (kernel.rs:624).
    env_config = build_trace_env_config()
    assert isinstance(env_config, dict)
    assert env_config["max_turns"] == 120

    # alongside level_handicap (as trace_factory_v5.py:101 writes level_handicap)
    env_config2 = build_trace_env_config(level_handicap={"p1_level": 2})
    assert env_config2["max_turns"] == 120
    assert env_config2["level_handicap"] == {"p1_level": 2}

    # extra env_config fields pass through (adaptive_strength etc.)
    env_config3 = build_trace_env_config(adaptive_strength=0.5)
    assert env_config3["max_turns"] == 120
    assert env_config3["adaptive_strength"] == 0.5

    # custom max_turns honored (still >= 120 when caller uses the constant)
    env_config4 = build_trace_env_config(max_turns=PHASE_A_MAX_TURNS)
    assert env_config4["max_turns"] == 120

    # Fix #2 live-constructor threading is DOCUMENTED (A4 implements; A3 specifies).
    # The note MUST reference kernel.rs:660 (from_trace_config) + the serde default 80
    # (kernel.rs:624) + max_turns, so the live path cannot silently fall back to a no-op.
    assert "kernel.rs:660" in LIVE_MAX_TURNS_THREADING_NOTE
    assert "kernel.rs:624" in LIVE_MAX_TURNS_THREADING_NOTE
    assert "max_turns" in LIVE_MAX_TURNS_THREADING_NOTE

    # the config carries max_turns for the A4 live constructor to thread into KernelConfig
    cfg = PhaseAPPOConfig()
    assert cfg.max_turns == 120


def test_entropy_and_epochs_pinned():
    cfg = PhaseAPPOConfig()
    # Fix #3: entropy_coef == 0.01 (NOT 0.035 — the phase26 override, BLOCK_A_PLAN.md:314)
    assert cfg.entropy_coef == PHASE_A_ENTROPY_COEF == 0.01
    assert cfg.entropy_coef != 0.035  # regression guard: the phase26 bug must NOT return
    # Fix #4: epochs == 6 (NOT 1 [phase26] or 3 [RustPPOTrainingConfig default],
    # BLOCK_A_PLAN.md:315-316)
    assert cfg.epochs == PHASE_A_EPOCHS == 6
    assert cfg.epochs != 1
    assert cfg.epochs != 3
    # pinned at module level too (imported constants are the source of truth)
    assert PHASE_A_ENTROPY_COEF == 0.01
    assert PHASE_A_EPOCHS == 6

    # a config constructed with an explicit override still pins (frozen dataclass):
    # epochs/entropy_coef are NOT overridable to the buggy values without an explicit
    # kwarg, and the defaults are the pinned values.
    assert PhaseAPPOConfig().entropy_coef == 0.01
    assert PhaseAPPOConfig().epochs == 6
    assert PhaseAPPOConfig().max_grad_norm == pytest.approx(0.5)
    assert PhaseAPPOConfig().target_kl == pytest.approx(0.03)


def test_second_start_oversampling_scheme():
    cfg = PhaseAPPOConfig()
    # the scheme is present on the config + well-formed
    scheme_params = cfg.second_start_oversampling
    assert scheme_params["gap_threshold"] == PHASE_A_P1_P2_GAP_THRESHOLD == 0.12
    assert scheme_params["base_weight"] == 0.5
    assert scheme_params["policy"] == "oversample_under_represented_on_breach"

    # balanced (gap <= 0.12): no oversampling, 0.5/0.5
    balanced = second_start_oversampling_scheme(0.50, 0.55)  # gap 0.05 <= 0.12
    assert balanced["breach"] is False
    assert balanced["oversampled_side"] is None
    assert balanced["p1_weight"] == pytest.approx(0.5)
    assert balanced["p2_weight"] == pytest.approx(0.5)

    # breach: p1 under-represented (lower rate) -> oversample p1
    p1_under = second_start_oversampling_scheme(0.40, 0.70)  # gap 0.30 > 0.12
    assert p1_under["breach"] is True
    assert p1_under["oversampled_side"] == "p1"
    assert p1_under["p1_weight"] > p1_under["p2_weight"]  # under side oversampled
    assert p1_under["p1_weight"] > 0.5

    # breach: p2 under-represented -> oversample p2
    p2_under = second_start_oversampling_scheme(0.80, 0.45)  # gap 0.35 > 0.12
    assert p2_under["breach"] is True
    assert p2_under["oversampled_side"] == "p2"
    assert p2_under["p2_weight"] > p2_under["p1_weight"]
    assert p2_under["p2_weight"] > 0.5

    # well-formed for a sweep of rate pairs: weights sum to 1.0, both in [0,1], gap exact
    for r1, r2 in [(0.5, 0.5), (0.4, 0.7), (0.8, 0.45), (0.1, 0.9), (0.55, 0.50), (0.62, 0.50)]:
        s = second_start_oversampling_scheme(r1, r2)
        assert 0.0 <= s["p1_weight"] <= 1.0, (r1, r2, s)
        assert 0.0 <= s["p2_weight"] <= 1.0, (r1, r2, s)
        assert s["p1_weight"] + s["p2_weight"] == pytest.approx(1.0), (r1, r2, s)
        assert s["gap"] == pytest.approx(abs(r1 - r2)), (r1, r2, s)
        assert s["breach"] == (s["gap"] > 0.12), (r1, r2, s)
        # if breach, the oversampled side is the lower-rate side
        if s["breach"]:
            if r1 < r2:
                assert s["oversampled_side"] == "p1"
                assert s["p1_weight"] >= s["p2_weight"]
            else:
                assert s["oversampled_side"] == "p2"
                assert s["p2_weight"] >= s["p1_weight"]
