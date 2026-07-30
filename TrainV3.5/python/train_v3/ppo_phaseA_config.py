"""Block A component A3 — Phase-A PPO config + 5 root-cause fix helpers (PURE-PYTHON).

Encodes the 5 root-cause fixes (``design.md:103-109``) as a SUPERSET of
``RustPPOTrainingConfig`` (``rust_trainer.py:23-71``) consumed by BOTH the
trace-pool trainer AND the new live-self-play trainer (A4). This module is plain
data + helper functions: NO MLX import, NO Rust FFI call. The Rust-threading
(max_turns -> ``KernelConfig``) happens in A4; A3 only SPECIFIES the plumbing.

Superset relationship (``BLOCK_A_PLAN.md:267-336``): ``PhaseAPPOConfig`` carries
every field of ``RustPPOTrainingConfig`` (with Phase-A pinned values where the spec
mandates) PLUS the new A3 fix fields. It is a STANDALONE frozen dataclass (does NOT
inherit ``RustPPOTrainingConfig``) so this module imports WITHOUT the
``rust_trainer`` -> ``rust_ffi`` chain — keeping the config + its tests pure-python
(no MLX, no Rust FFI call). ``to_rust_ppo_config()`` performs the LAZY conversion for
A4 (the only place that legitimately touches the Rust import chain).

Frozen-classic guard: ``reward_v5.py`` / ``classic_rl_env.py`` / the legacy
``run_phase26_noassist_easy_gate.py`` are NOT edited here. ``reward_v5.py`` is
ALREADY per-side (``reward_snapshot_v5`` takes a ``player_id``, ``reward_v5.py:40``);
the learner+opponent summation bug lives ONLY in the legacy phase26 script
(``run_phase26:490 step_rewards = learner_rewards + opponent_rewards``). Fix #1 is a
TRAINER-SIDE attribution change mirrored here, NOT a ``reward_v5.py`` edit
(``reward_v5.py`` consumed READ-ONLY).

Fix mapping (``design.md:103-109``, ``BLOCK_A_PLAN.md:278-331``):
  #1 LEARNER-ONLY reward        -> ``learner_only_reward=True`` + ``reward_attribution()``
  #2 max_turns>=120 + early-end  -> ``max_turns=120`` + ``decisive_early_end`` + plumbing
  #3 entropy_coef=0.01 (PINNED)  -> ``entropy_coef=0.01`` (regression guard vs 0.035)
  #4 epochs=6 (PINNED)          -> ``epochs=6`` (regression guard vs 1 / 3)
  #5 graduated opponent_mix      -> ``opponent_mix`` + ``validate_phase_a_opponent_mix()``
  +  second-start oversampling  -> ``second_start_oversampling`` + scheme fn (D-A10)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .league_v5 import V5LeagueConfig, parse_v5_opponent_mix


# --- Fix #2: max_turns plumbing ------------------------------------------------
#: Spec mandates >= 120 (``design.md:106``); the Rust ``KernelConfig`` serde default
#: is 80 (``kernel.rs:624``). Without explicit plumbing, generated traces fall back
#: to the serde default 80 (verifier finding 2b, ``BLOCK_A_PLAN.md:300-312``).
PHASE_A_MAX_TURNS = 120

#: Documentation of the LIVE-constructor threading (A4 implements; A3 specifies).
#: A4's live ``from_live``/``from_scenario`` constructor MUST accept a ``max_turns``
#: argument and set ``config.max_turns`` BEFORE ``KernelConfig`` construction so that
#: ``kernel.rs:660 from_trace_config`` reads the Phase-A value (not the fallback
#: 80). The fallback 80 has two sources: the ``KernelConfig`` Rust ``Default`` impl
#: (``kernel.rs:624``) and the ``GoldenTraceConfig.max_turns`` serde default
#: ``default_max_turns()`` (``kernel.rs:396-398``). Captured here as a constant the
#: test asserts on, so the live path cannot silently regress to a no-op.
LIVE_MAX_TURNS_THREADING_NOTE = (
    "A4 live constructor MUST accept max_turns and set config.max_turns BEFORE "
    "KernelConfig construction (kernel.rs:660 from_trace_config reads "
    "GoldenTraceConfig.max_turns). Without this the live path falls back to 80 "
    "(KernelConfig Default impl at kernel.rs:624; GoldenTraceConfig serde default "
    "default_max_turns() at kernel.rs:396-398)."
)

# --- Fix #3 / #4: pinned PPO hyperparams (regression guards) -------------------
#: PINNED; phase26 overrode to 0.035 (``BLOCK_A_PLAN.md:314``). Already the
#: ``RustPPOTrainingConfig`` default (``rust_trainer.py:38``); pinned here as a guard.
PHASE_A_ENTROPY_COEF = 0.01
#: PINNED; phase26 used 1 (``run_phase26:292``), ``RustPPOTrainingConfig`` default 3
#: (``rust_trainer.py:34``); D-A7 mid-band = 6 (``BLOCK_A_PLAN.md:315-316``).
PHASE_A_EPOCHS = 6

# --- Fix #2: decisive-state early-end (D-A6) ----------------------------------
#: Win-margin threshold (absolute hero-hp FRACTION lead) above which the game is
#: effectively decided -> early-end. Design default (``BLOCK_A_PLAN.md:689-690``);
#: user-tunable. decisive_early_end is a PURE function of a state snapshot.
PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD = 0.60

# --- Fix #5: graduated opponent_mix (design.md:111) ---------------------------
#: Display name -> spec weight (the human-readable mix from ``design.md:111``).
#: These are the EXACT spec weights the test asserts against.
PHASE_A_OPPONENT_MIX_SPEC: dict[str, float] = {
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

#: Current Phase-A bootstrap profile (2026-07-05): direct Rust ArenaEnv PPO against
#: random-heavy rule opponents. This replaces the attempted LLM/V4Max
#: semi-synthetic distillation lane for the FIRST phase only. Later Block-B league,
#: Block-C human-vs-preV5, and repair lanes keep their own opponent mixes.
PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE = 0.98
PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC: dict[str, float] = {
    "legal_random": 0.70,
    "end_turn": 0.05,
    "greedy_face": 0.10,
    "face_rush": 0.05,
    "board_control": 0.05,
    "greedy_trade": 0.05,
}

#: Display name -> canonical name accepted by ``league_v5.parse_v5_opponent_mix``
#: (``V5_OPPONENT_KINDS``, ``league_v5.py:12-21`` + ``gauntlet_v5.EXPLOIT_AGENT_KINDS``,
#: ``gauntlet_v5.py:8-16``). The display names in ``design.md:111`` (legal_random /
#: self_prev / v4-orig-argmax) are NOT directly parseable; this alias layer
#: translates them. A3 VALIDATES via the existing parser (read-only); the parser is
#: NOT edited (additive validation, not a replacement).
PHASE_A_OPPONENT_NAME_ALIASES: dict[str, str] = {
    "legal_random": "random",
    "end_turn": "end_turn",
    "greedy_face": "greedy_face",
    "face_rush": "face_rush",
    "board_control": "board_control",
    "greedy_trade": "greedy_trade",
    "stall": "stall",
    "anti_draw_greed": "anti_draw_greed",
    # self_prev = a previous-self identity; "self" is the Phase-A primary (current
    # policy self-play). A4 may dispatch self_prev to "v5_snapshot" once a league
    # snapshot pool exists (BLOCK_A_PLAN.md:319 "self_prev(self/v5_snapshot)").
    "self_prev": "self",
    "v4-orig-argmax": "v4max",
}

# --- D-A10: second-start oversampling (design.md:112,120) ---------------------
#: ``design.md:120`` p1_p2_score_gap <= 0.12 acceptance; oversample p2-init on breach.
PHASE_A_P1_P2_GAP_THRESHOLD = 0.12


def build_phase_a_opponent_mix_string(
    spec: dict[str, float] | None = None,
    aliases: dict[str, str] | None = None,
) -> str:
    """Build the canonical mix string parseable by ``league_v5.parse_v5_opponent_mix``.

    Translates the display-name spec (``design.md:111``) to the canonical names in
    ``V5_OPPONENT_KINDS`` and joins as ``"name:weight,..."`` (the parser's format).
    """
    spec = spec if spec is not None else PHASE_A_OPPONENT_MIX_SPEC
    aliases = aliases if aliases is not None else PHASE_A_OPPONENT_NAME_ALIASES
    return ",".join(f"{aliases[name]}:{weight}" for name, weight in spec.items())


def build_phase_a_random_bootstrap_opponent_mix_string(
    spec: dict[str, float] | None = None,
    aliases: dict[str, str] | None = None,
) -> str:
    """Build the current Phase-A random-bootstrap opponent mix.

    This is intentionally teacher-free: no LLM, no V4Max, no previous-self snapshot.
    The mix is random-heavy enough to optimize directly for the 98% vs-random target,
    while keeping a small amount of simple deterministic pressure so the bootstrap
    does not learn only to exploit one legal-random quirk.
    """
    return build_phase_a_opponent_mix_string(
        spec if spec is not None else PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC,
        aliases if aliases is not None else PHASE_A_OPPONENT_NAME_ALIASES,
    )


def validate_phase_a_opponent_mix(
    spec: dict[str, float] | None = None,
    aliases: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Validate the graduated mix parses to the spec weights (Fix #5).

    VALIDATION ONLY (NOT runtime dispatch): the actual rule-agent (codes 0-7) vs
    policy-opponent dispatch for the 10 identities is A4's job
    (``BLOCK_A_PLAN.md:326-327`` — the verify gap). A3 only checks the mix is
    well-formed + matches the spec weights via the EXISTING
    ``league_v5.parse_v5_opponent_mix`` (read-only call; parser NOT edited).

    Raises ``AssertionError`` if a parsed weight diverges from spec or the mix does
    not sum to 1.0; raises ``ValueError`` (from the parser) on an unknown name.
    """
    spec = spec if spec is not None else PHASE_A_OPPONENT_MIX_SPEC
    aliases = aliases if aliases is not None else PHASE_A_OPPONENT_NAME_ALIASES
    mix_str = build_phase_a_opponent_mix_string(spec, aliases)
    parsed = parse_v5_opponent_mix(mix_str)  # read-only; raises ValueError on unknown
    canonical_to_display = {aliases[d]: d for d in aliases}
    if len(parsed) != len(spec):
        raise AssertionError(
            f"opponent_mix parse length {len(parsed)} != spec length {len(spec)}"
        )
    total = 0.0
    for canonical, weight in parsed:
        display = canonical_to_display.get(canonical)
        if display is None or display not in spec:
            raise AssertionError(
                f"parsed canonical {canonical!r} has no spec display mapping"
            )
        if abs(weight - spec[display]) > 1e-9:
            raise AssertionError(
                f"opponent_mix weight mismatch for {display}->{canonical}: "
                f"parsed {weight} != spec {spec[display]}"
            )
        total += weight
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"opponent_mix weights sum to {total}, expected 1.0")
    return parsed


def validate_phase_a_random_bootstrap_opponent_mix(
    spec: dict[str, float] | None = None,
    aliases: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Validate the current Phase-A random-bootstrap mix."""
    return validate_phase_a_opponent_mix(
        spec if spec is not None else PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC,
        aliases if aliases is not None else PHASE_A_OPPONENT_NAME_ALIASES,
    )


def build_trace_env_config(
    max_turns: int = PHASE_A_MAX_TURNS,
    level_handicap: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a trace ``env_config`` dict carrying ``max_turns`` (Fix #2 trace-pool path).

    ``trace_factory_v5.py`` MUST write ``env_config['max_turns'] = 120`` into generated
    traces (alongside ``level_handicap``, ``trace_factory_v5.py:101``) so the Rust
    ``KernelConfig.from_trace_config`` (``kernel.rs:660``) reads 120 instead of the
    serde default 80 (``kernel.rs:624``). Pure-python (no FFI): the returned dict is the
    exact shape the trace factory writes into ``trace['env_config']``.

    The current ``trace_factory_v5.py`` writes only ``level_handicap`` into
    ``env_config`` (``trace_factory_v5.py:101``) — ``max_turns`` is absent, so generated
    traces fall back to the Rust serde default 80. A4 wires this dict (or the constant)
    into the trace factory; A3 SPECIFIES the shape + provides the testable dict.
    """
    env_config: dict[str, Any] = {}
    if level_handicap:
        env_config["level_handicap"] = level_handicap
    env_config["max_turns"] = int(max_turns)
    for key, value in extra.items():
        env_config[key] = value
    return env_config


def is_decisive_state(
    snapshot: Any,
    *,
    threshold: float = PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD,
) -> bool:
    """Fix #2 decisive-state early-end predicate (D-A6).

    Pure function of a state snapshot: win-margin (absolute hero-hp FRACTION lead) >=
    threshold -> the game is effectively decided -> early-end. ``snapshot`` is
    duck-typed to ``V5RewardSnapshot`` (``reward_v5.py:12-52``): attributes
    ``my_hero_hp`` / ``my_hero_max_hp`` / ``enemy_hero_hp`` / ``enemy_hero_max_hp``.
    ``reward_v5.py`` is consumed READ-ONLY (frozen-classic guard) — this predicate does
    NOT import it, only matches its snapshot shape (so the early-end decision can be
    evaluated wherever a V5RewardSnapshot is available without coupling to reward_v5).
    """
    my_max = max(1, int(getattr(snapshot, "my_hero_max_hp", 1)))
    en_max = max(1, int(getattr(snapshot, "enemy_hero_max_hp", 1)))
    my_frac = max(0.0, float(getattr(snapshot, "my_hero_hp", 0))) / my_max
    en_frac = max(0.0, float(getattr(snapshot, "enemy_hero_hp", 0))) / en_max
    return abs(my_frac - en_frac) >= float(threshold)


def reward_attribution(
    step_rewards: Any,
    actor_ids: Any,
    learner_actor: Any,
) -> np.ndarray:
    """Fix #1 learner-only reward attribution (TRAINER-SIDE).

    Zero rewards for steps NOT taken by the learner actor; keep learner-actor step
    rewards. Mirrors the fix for the legacy
    ``run_phase26_noassist_easy_gate.py:490 step_rewards = learner_rewards +
    opponent_rewards`` bug (which summed opponent credit into the learner's tape).
    ``reward_v5.py`` is ALREADY per-side (``reward_snapshot_v5`` takes ``player_id``,
    ``reward_v5.py:40``) — the bug is trainer-side, so the attribution lives here in
    the config, NOT in ``reward_v5.py`` (frozen-classic guard; ``reward_v5.py``
    READ-ONLY).

    For the trace-pool path this is a NO-OP: ``collect_rust_vec_rollout``
    (``rust_collector.py:218``) records ``step.rewards`` already learner-attributed per
    the Rust env. The live-self-play path (A4) calls this to attribute the tape before
    PPO, so opponent-actor steps carry ZERO credit (regression guard for
    ``run_phase26:490``).
    """
    rewards = np.asarray(step_rewards, dtype=np.float32)
    actors = np.asarray(actor_ids)
    learner = np.asarray(learner_actor)  # scalar actor id -> 0-d array; broadcasts
    is_learner = actors == learner
    attributed = np.where(is_learner, rewards, np.float32(0.0))
    return attributed.astype(np.float32, copy=False)


def second_start_oversampling_scheme(
    p1_score_rate: float,
    p2_score_rate: float,
    *,
    gap_threshold: float = PHASE_A_P1_P2_GAP_THRESHOLD,
    base_weight: float = 0.5,
) -> dict[str, Any]:
    """D-A10 second-start oversampling (``design.md:112,120``): gap-weighted p1/p2 init split.

    If ``abs(p1_rate - p2_rate) > gap_threshold`` (``design.md:120``
    ``p1_p2_score_gap <= 0.12`` acceptance), oversample the UNDER-represented
    (lower-score) starting side to balance p1/p2. Pure function of the two observed
    side score rates. Returns a well-formed scheme dict
    ``{p1_weight, p2_weight, gap, breach, oversampled_side}`` with weights summing to
    1.0 (both in [0, 1]).

    Mirrors Block B (``design.md:119-120``); A4 applies the scheme when sampling the
    starting side for live-self-play episodes.
    """
    gap = abs(float(p1_score_rate) - float(p2_score_rate))
    breach = gap > float(gap_threshold)
    if not breach:
        return {
            "p1_weight": float(base_weight),
            "p2_weight": float(base_weight),
            "gap": float(gap),
            "breach": False,
            "oversampled_side": None,
        }
    # Oversample the under-represented (lower-rate) side; shift proportional to how
    # far beyond the threshold the gap is, capped at base_weight so weights stay in
    # [0, 1] (the under side gets at most all the weight).
    shift = min(float(base_weight), gap - float(gap_threshold))
    if p1_score_rate < p2_score_rate:
        w1 = float(base_weight) + shift
        w2 = max(0.0, 1.0 - w1)
        side = "p1"
    else:
        w2 = float(base_weight) + shift
        w1 = max(0.0, 1.0 - w2)
        side = "p2"
    return {
        "p1_weight": float(w1),
        "p2_weight": float(w2),
        "gap": float(gap),
        "breach": True,
        "oversampled_side": side,
    }


@dataclass(frozen=True)
class PhaseAPPOConfig:
    """Phase-A PPO config — SUPERSET of ``RustPPOTrainingConfig``
    (``rust_trainer.py:23-71``).

    Carries every ``RustPPOTrainingConfig`` field (with Phase-A pinned values where the
    spec mandates) PLUS the 5 root-cause fix fields. STANDALONE (no inheritance) so this
    module imports WITHOUT the ``rust_trainer`` -> ``rust_ffi`` chain (pure-python: no
    MLX, no Rust FFI call). Use ``to_rust_ppo_config()`` to obtain the base
    ``RustPPOTrainingConfig`` for A4 (lazy import — the only place the Rust chain is
    touched, and A4 has FFI anyway).

    Field superset is verified by the field-name parity test in the A4 suite; this
    module's own tests stay pure-python (no ``rust_trainer`` import).
    """

    # --- base RustPPOTrainingConfig fields (rust_trainer.py:25-71) ---
    run_name: str | None = None
    model_name: str = "extra-lr-v5-adaptive"
    v5_league_config: V5LeagueConfig | None = None
    curriculum_metadata: dict[str, Any] = field(default_factory=dict)
    updates: int = 1
    env_count: int = 16
    steps_per_update: int = 30
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = PHASE_A_EPOCHS  # Fix #4 PINNED 6 (NOT 1 [phase26] / 3 [RustPPOTrainingConfig default])
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = PHASE_A_ENTROPY_COEF  # Fix #3 PINNED 0.01 (NOT 0.035 [phase26 override])
    max_grad_norm: float | None = 0.5
    target_kl: float | None = 0.03
    observation_key: str = "observation_v5"
    action_features_dtype: str = "float32"
    observation_mode: str = "v5_only"
    action_features_mode: str = "legal_only"
    action_mask_mode: str = "legal_only"
    terminal_observation_mode: str = "none"
    store_next_observations: bool = False
    store_infos: bool = False
    store_truncated: bool = False
    store_reset_flags: bool = False
    store_episode_stats: bool = False
    diagnostic_mode: str = "auto"
    advantage_backend: str = "rust"
    selected_local_backend: str = "provided"
    prepare_backend: str = "separate"
    legal_row_pack_backend: str = "auto"
    full_batch_eval: bool = False
    policy_scoring_backend: str = "padded"
    policy_selection_backend: str = "rust"
    policy_padding_mode: str = "single"
    policy_bucket_max_padding_ratio: float = 1.35
    policy_bucket_min_rows: int = 2048
    ppo_minibatch_plan: str = "contiguous"
    log_selected_trace_paths: bool = True
    trace_pool_reset_mode: str = "cycle"
    v5_runtime_mode_source: str = "manifest_cycle"
    trace_manifest_path: str | Path | None = None
    league_manifest_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 1
    metrics_path: str | Path | None = None
    seed: int | None = None

    # --- A3 root-cause fix fields ---
    # Fix #1: learner-only reward (TRAINER-SIDE attribution; reward_v5.py untouched).
    learner_only_reward: bool = True
    # Fix #2: max_turns >= 120 (PHASE_A_MAX_TURNS) + decisive-state early-end (D-A6).
    # max_turns is NOT a RustPPOTrainingConfig field — it threads via the trace
    # env_config (build_trace_env_config) / the A4 live constructor (see
    # LIVE_MAX_TURNS_THREADING_NOTE), NOT via the PPO config -> KernelConfig path.
    max_turns: int = PHASE_A_MAX_TURNS
    decisive_early_end: bool = True
    decisive_win_margin_threshold: float = PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD
    turn_order_second_mover_reward_bonus: float = 0.0
    # Fix #5: graduated opponent_mix (design.md:111). Canonical-string form parseable
    # by league_v5.parse_v5_opponent_mix; validate via validate_phase_a_opponent_mix().
    # Runtime dispatch (rule-agent codes 0-7 vs policy-opponent path) is A4's job.
    opponent_mix: str = field(default_factory=build_phase_a_opponent_mix_string)
    opponent_mix_spec: dict[str, float] = field(
        default_factory=lambda: dict(PHASE_A_OPPONENT_MIX_SPEC)
    )
    opponent_mix_aliases: dict[str, str] = field(
        default_factory=lambda: dict(PHASE_A_OPPONENT_NAME_ALIASES)
    )
    # D-A10: second-start oversampling (design.md:112,120). Parameter dict consumed by
    # second_start_oversampling_scheme(); A4 applies it when sampling the start side.
    second_start_oversampling: dict[str, Any] = field(
        default_factory=lambda: {
            "gap_threshold": PHASE_A_P1_P2_GAP_THRESHOLD,
            "base_weight": 0.5,
            "policy": "oversample_under_represented_on_breach",
        }
    )

    def to_rust_ppo_config(self):
        """Convert to a base ``RustPPOTrainingConfig`` for A4 (LAZY rust_trainer import).

        Only the ``RustPPOTrainingConfig`` fields are forwarded; the A3 fix fields
        (``learner_only_reward`` / ``max_turns`` / ``decisive_*`` / ``opponent_mix`` /
        ``second_start_oversampling``) are NOT ``RustPPOTrainingConfig`` fields —
        ``max_turns`` threads via the trace env_config / live constructor, and the
        opponent_mix dispatch is A4-side. Triggers the ``rust_trainer`` import chain
        (``rust_ffi`` is lazy-load; NO FFI call happens here). Kept lazy so this module
        stays pure-python at import time.
        """
        from .rust_trainer import RustPPOTrainingConfig  # lazy: keeps module pure-python

        base_fields = (
            "run_name", "model_name", "v5_league_config", "curriculum_metadata",
            "updates", "env_count", "steps_per_update", "gamma", "gae_lambda",
            "epochs", "minibatch_size", "clip_epsilon", "value_coef", "entropy_coef",
            "max_grad_norm", "observation_key", "action_features_dtype",
            "observation_mode", "action_features_mode", "action_mask_mode",
            "terminal_observation_mode", "store_next_observations", "store_infos",
            "store_truncated", "store_reset_flags", "store_episode_stats",
            "diagnostic_mode", "advantage_backend", "selected_local_backend",
            "prepare_backend", "legal_row_pack_backend", "full_batch_eval",
            "policy_scoring_backend", "policy_selection_backend",
            "policy_padding_mode", "policy_bucket_max_padding_ratio",
            "policy_bucket_min_rows", "ppo_minibatch_plan",
            "log_selected_trace_paths", "trace_pool_reset_mode",
            "v5_runtime_mode_source", "trace_manifest_path", "league_manifest_path",
            "checkpoint_dir", "checkpoint_every", "metrics_path", "seed",
        )
        kwargs = {name: getattr(self, name) for name in base_fields}
        return RustPPOTrainingConfig(**kwargs)


def build_phase_a_random_bootstrap_config(**overrides: Any) -> PhaseAPPOConfig:
    """Build the current Phase-A config: random-heavy Rust ArenaEnv PPO bootstrap.

    Use this for the FIRST V5 ExtraLR phase instead of semi-synthetic
    LLM/V4Max distillation. The returned config is still a normal
    ``PhaseAPPOConfig`` so A4 live self-play, Block-B handoff, and tests can reuse
    the existing machinery.
    """
    metadata = dict(overrides.pop("curriculum_metadata", {}) or {})
    metadata.setdefault("phase_a_profile", "random_bootstrap")
    metadata.setdefault("distillation", "disabled")
    metadata.setdefault("teacher_source", "none")
    metadata.setdefault(
        "bootstrap_note",
        "Rust ArenaEnv PPO against random-heavy rule opponents; no ExtraRLHF LLM/V4Max teacher data.",
    )
    metadata.setdefault(
        "target_random_score",
        PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE,
    )
    opponent_mix = overrides.pop(
        "opponent_mix",
        build_phase_a_random_bootstrap_opponent_mix_string(),
    )
    opponent_mix_spec = overrides.pop(
        "opponent_mix_spec",
        dict(PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC),
    )
    opponent_mix_aliases = overrides.pop(
        "opponent_mix_aliases",
        dict(PHASE_A_OPPONENT_NAME_ALIASES),
    )
    return PhaseAPPOConfig(
        opponent_mix=opponent_mix,
        opponent_mix_spec=opponent_mix_spec,
        opponent_mix_aliases=opponent_mix_aliases,
        curriculum_metadata=metadata,
        **overrides,
    )


__all__ = [
    "LIVE_MAX_TURNS_THREADING_NOTE",
    "PHASE_A_DECISIVE_WIN_MARGIN_THRESHOLD",
    "PHASE_A_EPOCHS",
    "PHASE_A_ENTROPY_COEF",
    "PHASE_A_MAX_TURNS",
    "PHASE_A_OPPONENT_MIX_SPEC",
    "PHASE_A_OPPONENT_NAME_ALIASES",
    "PHASE_A_P1_P2_GAP_THRESHOLD",
    "PHASE_A_RANDOM_BOOTSTRAP_OPPONENT_MIX_SPEC",
    "PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE",
    "PhaseAPPOConfig",
    "build_phase_a_opponent_mix_string",
    "build_phase_a_random_bootstrap_config",
    "build_phase_a_random_bootstrap_opponent_mix_string",
    "build_trace_env_config",
    "is_decisive_state",
    "reward_attribution",
    "second_start_oversampling_scheme",
    "validate_phase_a_opponent_mix",
    "validate_phase_a_random_bootstrap_opponent_mix",
]
