"""Block B component B2 -- ``v4_orig_temp_spectrum.py`` -- V4-orig temperature
spectrum: three policy-opponents from ONE frozen V4 ONNX model (NEW).

V5-Max pipeline position: Block A in-worktree COMPLETE -> Block B; B1
(``snapshot_pool.py``) DONE -> this file is B2, a SIBLING of B1 (no pool dep).

PURPOSE (``BLOCK_B_PLAN.md:253-304``): three policy-opponents from one frozen V4
ONNX -- ``v4-orig-argmax`` (temp=0 / argmax, weight 0.40), ``v4-orig-t07``
(temp=0.7, weight 0.20), ``v4-orig-t12`` (temp=1.2, weight 0.15) per D-B6. The
underlying V4 ONNX policy ``ai/train_v2/onnx_policy.py:16 OnnxActionPolicy``
ALREADY exposes a ``temperature`` param (validated >0 at ``:30-31``;
``scaled = mlogits / self._temperature`` at ``:93``; ``mode='argmax'|'sample'``
at ``:21``; argmax IGNORES temperature at ``:101``). A4's ``V4MaxOpponent``
(``rust_live_self_play.py:297``) is argmax-only and does NOT expose temperature.
B2 WRAPS the ONNX policy with a temperature param, NOT a port -- it does NOT edit
``onnx_policy.py`` (consumed READ-ONLY) or ``V4MaxOpponent`` (A4-built, wrapped
additively via a NEW ``TempV4Opponent`` class).

ADAPTER (the load-bearing piece, ``BLOCK_B_PLAN.md:264-271``):
``OnnxActionPolicy.select_action(env, player_id)`` is ENV-based (reads legal
actions from a Python env via ``build_action_mask`` + ``encode_action_features``,
``onnx_policy.py:62-78``), whereas A4's policy-opponent loop calls
``PolicyOpponent.select(env_idx, ctx: OpponentCtx)`` with PACKED arrays (no env,
``rust_live_self_play.py:215-225`` / ``:678-695``). B2 builds a thin adapter that,
given an ``OpponentCtx``, reproduces ``OnnxActionPolicy``'s forward math WITHOUT
the env (source-vs-source: ``onnx_policy.py`` math = oracle, this adapter = UUT):
  (1) reconstruct the full 601-mask from ``ctx.legal_action_ids`` (True at legal
      ids, False elsewhere);
  (2) reconstruct the full (601,171) ``action_features`` from
      ``ctx.legal_action_features`` placed at the legal ids (zeros elsewhere) --
      if ``legal_action_features`` is None, the identity cannot run (raises);
  (3) run the ONNX session with ``observation=ctx.observation_v5`` (single batch)
      + the reconstructed af -> ``logits`` (``onnx_policy.py:83-86``);
  (4) ``mlogits = np.where(mask, logits, -1e9)`` (``onnx_policy.py:90``);
  (5) argmax identity: ``aid = argmax(mlogits)`` (``:101``, temperature IGNORED);
      t07/t12: ``scaled = mlogits / temperature`` (``:93``), softmax + ``np.random
      .choice`` (``:94-99``) -- MUST stay within ``ctx.legal_action_ids``;
  (6) legal-fallback (``onnx_policy.py:103-106``): if the picked aid is not legal,
      fall back to the first legal id (``legal[0] if legal else 0``). This is
      UNREACHABLE when the mask is applied correctly (argmax of masked logits is
      always legal; the sample distribution is zeroed outside the mask and
      renormalized) -- it exists purely to mirror the oracle's defensive guard.

DEPENDENCY INJECTION: the factory accepts an injected ``session`` (an object with
a ``run(output_names, feeds)`` method matching ``onnxruntime.InferenceSession``)
so unit tests use a ``_FakeOnnxSession`` with a known logits vector -- no real
ONNX / onnxruntime / npz required. A real path string is loaded lazily via
``onnxruntime`` (skip-gate when onnxruntime or the V4 ONNX is absent).

Q5 BIAS (``design.md:198``, ``BLOCK_B_PLAN.md:281-284``): V4-orig is blind to
mana_draw + new cards; the learner could over-fit "opponent never draws." B2
EXPOSES a mana_draw-usage-vs-V4-orig-lanes monitor HOOK -- a placeholder callback
the D-B5 hybrid collapse monitor in B3/B4 wires. B2 does NOT implement the
monitor itself (deferred to B3/B4 per plan); it only exposes the hook so the
spectrum identities can be queried/flagged at the select call site.

CONSTRAINTS (frozen-classic guard, ``BLOCK_B_PLAN.md:718-728``): B2 is a NEW file.
NO edit to ``classic_obs_v1``/``classic_actions_v1``/``classic_card_shape_v1``/
``classic_rl_env.py``/``reward_v5.py``/``v5_trace.py``/``warm_start_v5.py``/
``run_phase26*``/``run_v5_acceptance``/``league_v5``/``gauntlet_v5``/
``opponents_v5``. ``onnx_policy.py`` consumed READ-ONLY (reuse its math
constants / forward math, do NOT import the class -- it forces ``onnxruntime`` at
import; the adapter reproduces ``:90``/``:93``/``:101`` exactly). ``V4MaxOpponent``
(A4-built, NOT frozen-classic) WRAPPED not edited. NO TrainV3.5 import into prod.
Source-vs-source: ``OnnxActionPolicy`` math = oracle, B2 adapter = UUT. Synthetic
tests only (``_FakeOnnxSession`` with known logits; skip-gate if onnxruntime or
V4 ONNX absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Union

import numpy as np

# A4 OpponentCtx + PolicyOpponent protocol (``rust_live_self_play.py:196-225``).
# Importing ``rust_live_self_play`` does NOT pull MLX/Rust at import time (verified
# in-worktree: only numpy + ppo_phaseA_config + rust_collector -> rust_vec_env ->
# rust_ffi lazy-load; the ``.dylib`` is not opened at import). Kept top-level so
# ``TempV4Opponent`` can be declared as implementing ``PolicyOpponent``.
from .rust_live_self_play import OpponentCtx

# -----------------------------------------------------------------------------
# Read-only math constants (mirrors the oracle ``ai/train_v2/onnx_policy.py`` +
# ``ai/train_v2/classic_actions_v1.py:46-47``). Reproduced locally so this module
# does NOT import ``onnx_policy`` (which forces ``onnxruntime`` at import) nor
# ``classic_actions_v1`` (frozen-classic; read-only reuse of the constants only).
# -----------------------------------------------------------------------------
MAX_CANDIDATE_ACTIONS = 601  # classic_actions_v1.py:46 MAX_CANDIDATE_ACTIONS
ACTION_FEATURE_DIM = 171     # classic_actions_v1.py:47 ACTION_FEATURE_DIM
_MLOGIT_MASK_VALUE = -1e9    # onnx_policy.py:90 np.where(mask, logits, -1e9)
_SOFTMAX_EPS = 1e-10         # onnx_policy.py:96,98 denom epsilon


# =============================================================================
# Identity dataclass + the three frozen identities (D-B6: t07=0.7, t12=1.2)
# =============================================================================
@dataclass(frozen=True)
class V4TempSpectrumIdentity:
    """One V4-orig temperature-spectrum identity.

    ``mode='argmax'`` mirrors ``OnnxActionPolicy`` argmax mode (``onnx_policy.py:
    101``) -- temperature is IGNORED (the spec's "temp=0" for ``v4-orig-argmax``
    is a label for argmax determinism, not a division). ``mode='sample'`` mirrors
    the sample mode (``onnx_policy.py:93-99``) with ``scaled = mlogits /
    temperature``. ``weight`` is the frozen D-B5 spec weight (0.40 / 0.20 / 0.15)
    carried unless the D-B5 hybrid reweight triggers (B3/B4 monitor).
    """

    name: str
    mode: str  # 'argmax' | 'sample'
    temperature: float
    weight: float


V4_ORIG_ARGMAX = V4TempSpectrumIdentity(
    name="v4-orig-argmax", mode="argmax", temperature=0.0, weight=0.40
)
V4_ORIG_T07 = V4TempSpectrumIdentity(
    name="v4-orig-t07", mode="sample", temperature=0.7, weight=0.20
)
V4_ORIG_T12 = V4TempSpectrumIdentity(
    name="v4-orig-t12", mode="sample", temperature=1.2, weight=0.15
)

V4_ORIG_TEMP_IDENTITIES: tuple[V4TempSpectrumIdentity, ...] = (
    V4_ORIG_ARGMAX,
    V4_ORIG_T07,
    V4_ORIG_T12,
)

# Frozen D-B5 weights (``BLOCK_B_PLAN.md:299``). The D-B5 hybrid collapse monitor
# (B3/B4) may reweight at runtime; these are the frozen spec-literal values.
V4_ORIG_TEMP_WEIGHTS: dict[str, float] = {
    ident.name: ident.weight for ident in V4_ORIG_TEMP_IDENTITIES
}

# Canonical-name alias map (``BLOCK_B_PLAN.md:272-274, 326-332``). ``v4-orig-t07``
# / ``v4-orig-t12`` are genuinely ABSENT from ``league_v5.V5_OPPONENT_KINDS``; the
# B3 alias layer resolves them. B2 exposes the identity-to-canonical map so the
# spectrum identities are self-describing (the alias resolution itself lives in
# B3, NOT here -- B2 only registers the canonical names).
V4_ORIG_TEMP_ALIASES: dict[str, str] = {
    ident.name: ident.name for ident in V4_ORIG_TEMP_IDENTITIES
}


# =============================================================================
# Q5 mana_draw-collapse monitor HOOK (placeholder; B3/B4 wires, B2 does NOT
# implement the monitor -- ``BLOCK_B_PLAN.md:281-284``).
# =============================================================================
# The hook is invoked at each select call site with (identity_name, ctx, aid) so a
# B3/B4 monitor can observe mana_draw-usage-vs-V4-orig-lanes and trigger the D-B5
# hybrid reweight when the learner's mana_draw usage drops out of band. B2 only
# EXPOSES the hook; it does NOT define the collapse logic.
ManaDrawCollapseMonitor = Callable[[str, "OpponentCtx", int], None]
_mana_draw_collapse_monitor: ManaDrawCollapseMonitor | None = None


def register_mana_draw_collapse_monitor(
    fn: ManaDrawCollapseMonitor | None,
) -> None:
    """Register (or clear with ``None``) the Q5 mana_draw-collapse monitor hook.

    B2 does NOT implement the monitor; this only exposes the registration point so
    B3/B4 can wire the D-B5 hybrid collapse observer (``BLOCK_B_PLAN.md:281-284``).
    """
    global _mana_draw_collapse_monitor
    _mana_draw_collapse_monitor = fn


def mana_draw_collapse_monitor_hook() -> ManaDrawCollapseMonitor | None:
    """Return the currently-registered Q5 monitor hook (or ``None``)."""
    return _mana_draw_collapse_monitor


# =============================================================================
# Adapter select_fn -- reproduces ``OnnxActionPolicy.select_action`` forward math
# from a packed ``OpponentCtx`` (no env). Source-vs-source UUT.
# =============================================================================
def _validate_identity(identity: V4TempSpectrumIdentity) -> None:
    """Mirror ``OnnxActionPolicy.__init__`` validation (``onnx_policy.py:28-31``)."""
    if identity.mode not in ("argmax", "sample"):
        raise ValueError(
            f"mode must be 'argmax' or 'sample', got '{identity.mode}'"
        )
    if identity.mode == "sample" and identity.temperature <= 0:
        # onnx_policy.py:30-31 validates temperature > 0 (argmax ignores temp).
        raise ValueError(
            f"temperature must be > 0 for sample mode, got {identity.temperature}"
        )


def make_v4_temp_select_fn(
    session: Any, identity: V4TempSpectrumIdentity
) -> Callable[[OpponentCtx], int]:
    """Build a ``select_fn(ctx: OpponentCtx) -> int`` for one spectrum identity.

    The adapter reproduces ``OnnxActionPolicy.select_action`` forward math
    (``onnx_policy.py:62-108``) from the packed ``OpponentCtx`` arrays -- no env:
      (1) reconstruct the 601-mask from ``ctx.legal_action_ids``;
      (2) reconstruct the (601,171) ``action_features`` from
          ``ctx.legal_action_features`` (raises if None -- the identity cannot
          run without packed features);
      (3) run the injected session -> ``logits`` (``onnx_policy.py:83-86``);
      (4) ``mlogits = np.where(mask, logits, -1e9)`` (``:90``);
      (5) argmax (``:101``) OR ``scaled = mlogits / temperature`` + softmax +
          ``np.random.choice`` (``:93-99``);
      (6) legal-fallback (``:103-106``) -- unreachable when the mask is applied.

    The returned aid is always in ``ctx.legal_action_ids``. The Q5 monitor hook is
    invoked at the call site if registered (B3/B4 wires it; B2 does NOT implement
    the monitor).
    """
    _validate_identity(identity)

    def select_fn(ctx: OpponentCtx) -> int:
        legal_ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        if legal_ids.size == 0:
            raise ValueError(
                f"v4-orig temp spectrum '{identity.name}': env {ctx.env_idx} "
                "has no legal actions (should have been reset)"
            )
        # (1) reconstruct the full 601 mask (``onnx_policy.py:65-70`` build_action_mask
        # equivalent -- True at legal ids, False elsewhere).
        mask_bool = np.zeros(MAX_CANDIDATE_ACTIONS, dtype=bool)
        mask_bool[legal_ids] = True
        mask_float = mask_bool.astype(np.float32)

        # (2) reconstruct the full (601,171) action_features from the packed legal
        # slice (``onnx_policy.py:71-78`` encode_action_features equivalent).
        if ctx.legal_action_features is None:
            raise ValueError(
                f"v4-orig temp spectrum '{identity.name}': ctx.legal_action_features "
                "is None -- the adapter needs the packed (k,171) legal-candidate "
                "features to reproduce OnnxActionPolicy's forward"
            )
        laf = np.asarray(ctx.legal_action_features, dtype=np.float32)
        if laf.shape[0] != legal_ids.size:
            raise ValueError(
                f"v4-orig temp spectrum '{identity.name}': legal_action_features "
                f"row count {laf.shape[0]} != legal_action_ids count {legal_ids.size}"
            )
        if laf.ndim != 2 or laf.shape[1] != ACTION_FEATURE_DIM:
            raise ValueError(
                f"v4-orig temp spectrum '{identity.name}': legal_action_features "
                f"must be (k, {ACTION_FEATURE_DIM}), got shape {laf.shape}"
            )
        af = np.zeros(
            (MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32
        )
        af[legal_ids] = laf

        # (3) run the ONNX session (single batch) -- ``onnx_policy.py:80-86``.
        obs = np.asarray(ctx.observation_v5, dtype=np.float32)
        obs_batch = obs[np.newaxis, :]
        af_batch = af[np.newaxis, :, :]
        outputs = session.run(
            ["logits", "value"],
            {"observation": obs_batch, "action_features": af_batch},
        )
        logits = np.asarray(outputs[0][0], dtype=np.float32)

        # (4) mlogits = np.where(mask, logits, -1e9) -- ``onnx_policy.py:90``.
        mlogits = np.where(mask_bool, logits, _MLOGIT_MASK_VALUE).astype(np.float32)

        # (5) argmax (``:101``, temperature IGNORED) OR sample (``:93-99``).
        if identity.mode == "sample":
            scaled = mlogits / identity.temperature  # :93
            shifted = scaled - np.max(scaled)        # :94
            exps = np.exp(shifted)                    # :95
            probs = exps / (np.sum(exps) + _SOFTMAX_EPS)  # :96
            probs = probs * mask_float                # :97
            probs = probs / (probs.sum() + _SOFTMAX_EPS)  # :98
            aid = int(np.random.choice(len(probs), p=probs))  # :99
        else:
            aid = int(np.argmax(mlogits))  # :101

        # (6) legal-fallback -- ``onnx_policy.py:103-106``. Unreachable when the
        # mask is applied (argmax of masked logits is legal; the sample
        # distribution is zeroed outside the mask + renormalized). Kept to mirror
        # the oracle's defensive guard exactly.
        if not bool(mask_bool[aid]):
            legal = [int(i) for i, v in enumerate(mask_bool) if v]
            aid = legal[0] if legal else 0

        # Q5 monitor hook (placeholder; B3/B4 wires the D-B5 collapse observer).
        # B2 does NOT implement the monitor -- it only exposes the call site.
        hook = _mana_draw_collapse_monitor
        if hook is not None:
            try:
                hook(identity.name, ctx, aid)
            except Exception:
                # A monitor hook failure MUST NOT break the rollout. The monitor
                # is an observer; B2's contract is to return a legal aid.
                pass

        return aid

    return select_fn


# =============================================================================
# TempV4Opponent -- a PolicyOpponent wrapping a spectrum select_fn (NEW class;
# does NOT edit A4 ``V4MaxOpponent``).
# =============================================================================
class TempV4Opponent:
    """A V4-orig temperature-spectrum policy-opponent (wraps a ``select_fn``).

    Implements the A4 ``PolicyOpponent`` protocol (``rust_live_self_play.py:215-
    225``): ``select(env_idx, ctx) -> int`` returns a 601-candidate action_id that
    is in ``ctx.legal_action_ids``. This is a NEW class -- it does NOT edit A4
    ``V4MaxOpponent`` (``rust_live_self_play.py:297``, argmax-only); it wraps the
    spectrum ``select_fn`` produced by ``make_v4_temp_select_fn`` and carries the
    frozen ``V4TempSpectrumIdentity`` for introspection (name / mode / temperature
    / weight).
    """

    name: str

    def __init__(
        self,
        name: str,
        select_fn: Callable[[OpponentCtx], int] | None,
        identity: V4TempSpectrumIdentity | None = None,
    ) -> None:
        self.name = name
        self._select_fn = select_fn
        self.identity = identity
        self.wired = select_fn is not None

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        if self._select_fn is None:
            raise RuntimeError(
                f"v4-orig temp spectrum opponent '{self.name}' not wired: "
                "provide a select_fn (make_v4_temp_select_fn(session, identity))"
            )
        return int(self._select_fn(ctx))


# =============================================================================
# Session resolution + skip-gate
# =============================================================================
class V4OnnxUnavailableError(RuntimeError):
    """Raised when the V4 ONNX model or ``onnxruntime`` is unavailable.

    The real-path factory test converts this to ``pytest.skip`` (worktree has no
    V4 ONNX npz / no onnxruntime). Synthetic tests inject a ``_FakeOnnxSession``
    directly and never hit this path.
    """


def _resolve_session(session_or_path: Any) -> Any:
    """Resolve a session-like object or an ONNX path to a runnable session.

    If ``session_or_path`` is a ``str`` / ``Path``, load it lazily via
    ``onnxruntime`` (mirrors ``OnnxActionPolicy.__init__`` ``:33-35``,
    CPUExecutionProvider). Otherwise return it as-is (dependency injection -- the
    caller passes a real ``onnxruntime.InferenceSession`` or a test
    ``_FakeOnnxSession``).
    """
    if isinstance(session_or_path, (str, Path)):
        path = Path(session_or_path)
        if not path.exists():
            raise V4OnnxUnavailableError(
                f"V4 ONNX model not found at {path} (worktree skip-gate)"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise V4OnnxUnavailableError(
                f"onnxruntime not importable: {exc} (worktree skip-gate)"
            ) from exc
        return ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    return session_or_path


def build_v4_temp_spectrum_opponents(
    session_or_path: Any,
) -> dict[str, TempV4Opponent]:
    """Build the three V4-orig temperature-spectrum ``PolicyOpponent``s.

    ``session_or_path`` is either an injected session (a ``.run(output_names,
    feeds)`` object -- real ``onnxruntime.InferenceSession`` or a test
    ``_FakeOnnxSession``) or an ONNX path string (loaded lazily via
    ``onnxruntime``; raises ``V4OnnxUnavailableError`` if the path / onnxruntime is
    absent -- the real-path test skips on this).

    Returns ``{name: TempV4Opponent}`` for the three identities
    (``v4-orig-argmax`` / ``v4-orig-t07`` / ``v4-orig-t12``), each wrapping a
    ``select_fn`` built by ``make_v4_temp_select_fn`` over the SAME session (one
    frozen V4 ONNX model -> three identities, ``test_three_identities_from_one_
    model``).
    """
    session = _resolve_session(session_or_path)
    opponents: dict[str, TempV4Opponent] = {}
    for identity in V4_ORIG_TEMP_IDENTITIES:
        select_fn = make_v4_temp_select_fn(session, identity)
        opponents[identity.name] = TempV4Opponent(
            name=identity.name, select_fn=select_fn, identity=identity
        )
    return opponents


__all__ = [
    "ACTION_FEATURE_DIM",
    "MAX_CANDIDATE_ACTIONS",
    "ManaDrawCollapseMonitor",
    "TempV4Opponent",
    "V4OnnxUnavailableError",
    "V4_ORIG_ARGMAX",
    "V4_ORIG_T07",
    "V4_ORIG_T12",
    "V4_ORIG_TEMP_ALIASES",
    "V4_ORIG_TEMP_IDENTITIES",
    "V4_ORIG_TEMP_WEIGHTS",
    "V4TempSpectrumIdentity",
    "build_v4_temp_spectrum_opponents",
    "make_v4_temp_select_fn",
    "mana_draw_collapse_monitor_hook",
    "register_mana_draw_collapse_monitor",
]