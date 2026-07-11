"""Block B component B1 — ``snapshot_pool.py`` — self-snapshot pool manager (NEW).

V5-Max pipeline position: Block A in-worktree COMPLETE -> Block B (this file is B1,
the single biggest new build in Block B). B1 is a bounded pool (~6 rolling self-
snapshots + 2 anchors [seed, best-ever]) of V5 checkpoints tracked with metadata
(``update#``, H2H-vs-best score, ``path``, ``p1_p2 gap``, promotion-eligible flag).

WHY NEW (verifier-anticipated, ``BLOCK_B_PLAN.md:216``): A5 ``select_promotion``
(``a_gate.py:607``) tracks a SINGLE ``current_best``; ``rust_trainer._save_checkpoint``
(``rust_trainer.py:802``) is linear-cadence only (``_should_checkpoint`` at ``:795``);
no rolling eviction, no anchors, no best-ever. B1 is zero-existing-infra.

DESIGN (``BLOCK_B_PLAN.md:197-250``):
  * Pool holds <= ``target_non_anchor_count`` (~6) rolling non-anchors + 2 anchors
    (``seed`` = first-promoted, immutable; ``best-ever`` = highest external-bench
    H2H-vs-best, replaces on STRICT improvement only — mirrors A5
    ``H2H_PROMOTION_THRESHOLD=0.5`` strict-beat at ``a_gate.py:113/:657``).
  * FIFO eviction of the OLDEST non-anchor on ``len(non_anchors) > target``; anchors
    are NEVER evicted (``test_fifo_eviction_keeps_anchors``).
  * ``best-ever`` updates only on strict external-bench H2H improvement — ties do NOT
    replace (``test_best_ever_updates_on_strict_improvement``; promotion-by-loss guard
    inherited from A5 — internal ppo_loss/approx_kl/entropy NEVER read here).
  * ``seed`` anchor is immutable after first promotion
    (``test_seed_anchor_immutable``).
  * A snapshot loads back into A4 ``SelfPrevOpponent`` (``rust_live_self_play.py:279``)
    as a ``select_fn`` and produces a deterministic argmax action — pure self-play when
    wired to a prior snapshot (``test_load_snapshot_as_self_prev_opponent``).
  * Pool manifest round-trips to disk + back — paths + metadata
    (``test_manifest_roundtrip``).
  * D-B5 hybrid support: the self-snapshot mix-weight available to the pool is a
    function of pool size (prevalence rises as the pool fills,
    ``test_pool_grows_self_snapshot_prevalence``). The actual mana_draw-collapse
    monitor lives in B3/B4; B1 only EXPOSES ``self_snapshot_prevalence_weight`` so
    prevalence can grow with the pool (``BLOCK_B_PLAN.md:247-249`` + D-B5 note in §2).

PHYSICAL STORAGE (read-only reuse, ``BLOCK_B_PLAN.md:213``): the npz format reuses
``rust_trainer._save_checkpoint`` (``rust_trainer.py:802`` via
``ai.train_v2.model_mlx.save_checkpoint``); B1 TRACKS PATHS, it does NOT rewrite the
save mechanism. No edit to ``rust_trainer.py``/``warm_start_v5.py``/``v5_trace.py``
(read-only reuse — ``BLOCK_B_PLAN.md:235`` acceptance + §10 frozen-classic guard).

CONSTRAINTS (frozen-classic guard, ``BLOCK_B_PLAN.md:860``): B1 is a NEW file. NO
TrainV3.5 import into prod. Source-vs-source: live engine = oracle, V5/training code =
UUT. Synthetic unit tests use fake checkpoints + a ``_FakeCheckpointStore`` (no real
MLX/Rust/ONNX required; skip-gate if a real ``save_checkpoint`` roundtrip needs MLX).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# A4 SelfPrevOpponent is the load target (``rust_live_self_play.py:279``). Imported
# lazily inside ``load_as_self_prev_opponent_select_fn`` so the module imports without
# the A4 dependency chain (and so synthetic tests can stub it). The import itself does
# NOT pull MLX/Rust (verified in-worktree: only numpy + ppo_phaseA_config +
# rust_collector, none of which require MLX), but keeping it lazy preserves the
# source-vs-source discipline: a caller who only needs the pool manifest API does not
# pay for the A4 import.


# =============================================================================
# Snapshot entry + anchor roles
# =============================================================================
@dataclass(frozen=True)
class SnapshotEntry:
    """One pool entry — a V5 checkpoint with promotion/league metadata.

    ``update_number`` is the PPO update index when the snapshot was taken (the B8
    snapshot cadence ~2000 updates feeds this; ``BLOCK_B_PLAN.md:204``). ``h2h_vs_best``
    is the external-bench H2H-vs-best-self-snapshot score-rate (the A5
    ``CandidateExternalBench.h2h_vs_best_score_rate`` value, ``a_gate.py:634``).
    ``p1_p2_gap`` is the B5-measured p1/p2 score-rate gap (``BLOCK_B_PLAN.md:120``).
    ``promotion_eligible`` records whether this snapshot passed the B6 promotion gate
    (monotone external-bench improvement >= N_snap + p1_p2_gap <= 0.12). ``path`` is
    the npz checkpoint path (the ``rust_trainer._save_checkpoint`` format
    ``trainv3_rust_legal_update_{update:04d}.npz``, ``rust_trainer.py:819``).
    """

    update_number: int
    h2h_vs_best: float
    path: str
    p1_p2_gap: float
    promotion_eligible: bool
    # Anchor role: "seed" (first promoted, immutable), "best_ever" (highest external-
    # bench H2H, replaces on strict improvement), or "rolling" (a non-anchor that FIFO
    # evicts). ``None`` is treated as "rolling".
    role: str = "rolling"

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_number": int(self.update_number),
            "h2h_vs_best": float(self.h2h_vs_best),
            "path": str(self.path),
            "p1_p2_gap": float(self.p1_p2_gap),
            "promotion_eligible": bool(self.promotion_eligible),
            "role": str(self.role),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SnapshotEntry":
        return cls(
            update_number=int(d["update_number"]),
            h2h_vs_best=float(d["h2h_vs_best"]),
            path=str(d["path"]),
            p1_p2_gap=float(d["p1_p2_gap"]),
            promotion_eligible=bool(d["promotion_eligible"]),
            role=str(d.get("role", "rolling")),
        )


SEED_ROLE = "seed"
BEST_EVER_ROLE = "best_ever"
ROLLING_ROLE = "rolling"


# =============================================================================
# Checkpoint-store protocol (the physical save/load, injected for testability)
# =============================================================================
class CheckpointStore(Protocol):
    """The physical checkpoint save/load surface (``rust_trainer._save_checkpoint``
    + ``model_mlx.save_checkpoint`` / ``load_checkpoint``). B1 TRACKS paths; the store
    owns the npz bytes. Injected so unit tests use a ``_FakeCheckpointStore`` and the
    production wiring reuses the A2/A4 checkpoint paths (read-only, no rewrite —
    ``BLOCK_B_PLAN.md:213``)."""

    def exists(self, path: str) -> bool: ...

    def load_weights(self, path: str) -> dict[str, Any]:
        """Load the checkpoint weights (the V5 policy parameters). Production returns
        the MLX weights dict for ``MLXV5LearnerPolicy``; the fake store returns a
        marker dict. Used by ``load_as_self_prev_opponent_select_fn`` to build an
        argmax policy from a prior snapshot (pure self-play, ``a_gate.py``-free)."""
        ...


# =============================================================================
# Pool
# =============================================================================
@dataclass
class SnapshotPool:
    """Bounded self-snapshot pool (~6 rolling + 2 anchors [seed, best-ever]).

    FIFO eviction of the OLDEST non-anchor on overflow; anchors (``seed``, ``best_ever``)
    are NEVER evicted. ``best_ever`` replaces on STRICT H2H improvement only (ties do
    NOT replace — mirrors A5 ``H2H_PROMOTION_THRESHOLD=0.5`` strict beat,
    ``a_gate.py:657`` ``if not (h2h > thresh)``). ``seed`` is immutable after first
    promotion.
    """

    #: Max rolling (non-anchor) snapshots kept (D-B4 ~6, ``BLOCK_B_PLAN.md:27``).
    target_non_anchor_count: int = 6
    #: Strict-improvement threshold for best-ever replacement (A5
    # ``H2H_PROMOTION_THRESHOLD=0.5`` — a tie at exactly 0.5 is NOT a strict beat;
    # ``a_gate.py:657`` uses ``h2h > thresh``). The best-ever anchor replaces only when
    # a new candidate's H2H-vs-best STRICTLY exceeds the current best-ever's H2H.
    best_ever_strict_threshold: float = 0.5
    #: Frozen non-self-snapshot share: V4-orig 0.55 + exploit 0.15 + tail 0.05 ->
    # frozen non-self TOTAL = 0.75; the self-snapshot share is the RESIDUAL 0.25 grown
    # as the pool fills. This follows the Q5 mitigation: modest blind V4-orig lane,
    # higher self-snapshot prevalence.
    # The collapse monitor itself lives in B3/B4; B1 only exposes the prevalence weight.
    frozen_non_self_share: float = 0.75
    #: The pool target used to scale prevalence (D-B4 ~6 non-anchors; prevalence grows
    # from 0 to the full residual as the pool fills this many non-anchors).
    prevalence_pool_target: int = 6

    # Live state.
    _rolling: list[SnapshotEntry] = field(default_factory=list)
    _seed: SnapshotEntry | None = None
    _best_ever: SnapshotEntry | None = None

    # ---- introspection -------------------------------------------------------
    @property
    def seed_anchor(self) -> SnapshotEntry | None:
        return self._seed

    @property
    def best_ever(self) -> SnapshotEntry | None:
        return self._best_ever

    @property
    def rolling(self) -> tuple[SnapshotEntry, ...]:
        return tuple(self._rolling)

    @property
    def non_anchor_count(self) -> int:
        return len(self._rolling)

    @property
    def anchors(self) -> tuple[SnapshotEntry, ...]:
        out: list[SnapshotEntry] = []
        if self._seed is not None:
            out.append(self._seed)
        if self._best_ever is not None and (
            self._seed is None or self._best_ever.update_number != self._seed.update_number
            or self._best_ever.path != self._seed.path
        ):
            out.append(self._best_ever)
        return tuple(out)

    @property
    def all_entries(self) -> tuple[SnapshotEntry, ...]:
        """Every entry in the pool (anchors first, then rolling in insertion order)."""
        return (*self.anchors, *self._rolling)

    def __len__(self) -> int:
        return len(self.anchors) + len(self._rolling)

    # ---- core mutation -------------------------------------------------------
    def set_seed_anchor(self, entry: SnapshotEntry) -> SnapshotEntry:
        """Set the seed anchor (the A-gate-passed first-promoted snapshot). IMMUTABLE
        after first set — a second call raises (``test_seed_anchor_immutable``). The
        seed is the inaugural best-self-snapshot (``BLOCK_B_PLAN.md:208`` /
        ``a_gate.py:628`` first-snapshot case)."""
        if self._seed is not None:
            raise RuntimeError(
                "seed anchor is immutable after first promotion "
                f"(existing seed update#{self._seed.update_number}; "
                f"refusing to overwrite with update#{entry.update_number})"
            )
        seeded = SnapshotEntry(
            update_number=entry.update_number,
            h2h_vs_best=entry.h2h_vs_best,
            path=entry.path,
            p1_p2_gap=entry.p1_p2_gap,
            promotion_eligible=entry.promotion_eligible,
            role=SEED_ROLE,
        )
        self._seed = seeded
        # The seed is also the initial best-ever (no prior best to beat — A5
        # ``current_best_h2h_score_rate=None`` first-snapshot case, ``a_gate.py:627``).
        if self._best_ever is None:
            self._best_ever = SnapshotEntry(
                update_number=seeded.update_number,
                h2h_vs_best=seeded.h2h_vs_best,
                path=seeded.path,
                p1_p2_gap=seeded.p1_p2_gap,
                promotion_eligible=seeded.promotion_eligible,
                role=BEST_EVER_ROLE,
            )
        return seeded

    def add_snapshot(self, entry: SnapshotEntry) -> SnapshotEntry:
        """Add a rolling self-snapshot (the B8 cadence hook calls this on top of
        ``rust_trainer._save_checkpoint``). FIFO-evicts the OLDEST non-anchor on
        overflow; anchors are NEVER evicted. Returns the stored entry (role normalized
        to ``rolling``).

        This does NOT decide promotion — ``maybe_update_best_ever`` is the separate
        external-bench-driven best-ever update. ``add_snapshot`` only manages the
        rolling pool (the league-opponent / ``SelfPrevOpponent`` source)."""
        stored = SnapshotEntry(
            update_number=entry.update_number,
            h2h_vs_best=entry.h2h_vs_best,
            path=entry.path,
            p1_p2_gap=entry.p1_p2_gap,
            promotion_eligible=entry.promotion_eligible,
            role=ROLLING_ROLE,
        )
        self._rolling.append(stored)
        self._evict_fifo()
        return stored

    def _evict_fifo(self) -> SnapshotEntry | None:
        """FIFO-evict the OLDEST non-anchor while ``len(rolling) > target``. Anchors
        are NEVER evicted (they live in ``_seed`` / ``_best_ever``, not ``_rolling``).
        Returns the evicted entry or ``None``."""
        evicted: SnapshotEntry | None = None
        while len(self._rolling) > self.target_non_anchor_count:
            evicted = self._rolling.pop(0)
        return evicted

    def maybe_update_best_ever(
        self,
        candidate: SnapshotEntry,
        *,
        h2h_vs_best_score_rate: float | None = None,
    ) -> bool:
        """Update the best-ever anchor iff ``candidate`` STRICTLY beats the current
        best-ever on external-bench H2H (mirrors A5 ``select_promotion``
        ``a_gate.py:657`` ``if not (h2h > thresh)`` — a TIE is NOT a strict beat and
        does NOT replace; ``BLOCK_B_PLAN.md:228``). Returns True iff the best-ever was
        replaced.

        ``h2h_vs_best_score_rate`` is the external-bench H2H-vs-best score-rate of the
        candidate (the A5 ``CandidateExternalBench.h2h_vs_best_score_rate`` value). If
        omitted, ``candidate.h2h_vs_best`` is used. The strict-beat test is
        ``h2h > best_ever.h2h_vs_best`` (a candidate can only beat the best-ever by
        scoring HIGHER than the best-ever's own H2H-vs-best record; the
        ``best_ever_strict_threshold`` is the FLOOR below which nothing replaces — it
        mirrors A5's 0.5 strict-beat semantics so a candidate at exactly the
        best-ever's level is a tie, not an improvement).

        Promotion-by-loss guard (inherited from A5, ``a_gate.py:637``): this method
        consults ONLY the external-bench H2H. It NEVER reads internal PPO loss / KL /
        entropy (those are not even fields on ``SnapshotEntry``)."""
        h2h = float(
            h2h_vs_best_score_rate
            if h2h_vs_best_score_rate is not None
            else candidate.h2h_vs_best
        )
        if not bool(candidate.promotion_eligible):
            return False
        if self._best_ever is None:
            # No prior best — the candidate becomes the inaugural best-ever (A5
            # first-snapshot case, ``a_gate.py:627``). This path is normally covered by
            # ``set_seed_anchor``; kept for completeness.
            self._best_ever = SnapshotEntry(
                update_number=candidate.update_number,
                h2h_vs_best=h2h,
                path=candidate.path,
                p1_p2_gap=candidate.p1_p2_gap,
                promotion_eligible=candidate.promotion_eligible,
                role=BEST_EVER_ROLE,
            )
            return True
        current_h2h = float(self._best_ever.h2h_vs_best)
        # STRICT beat: the candidate must exceed BOTH the best-ever's H2H record AND
        # the strict-improvement threshold (so a tie at the threshold does not replace,
        # mirroring A5 ``h2h > thresh``). The primary test is strict-vs-current-best:
        # a candidate that does not exceed the current best-ever's H2H is not an
        # improvement regardless of the floor.
        if h2h > current_h2h and h2h > self.best_ever_strict_threshold:
            self._best_ever = SnapshotEntry(
                update_number=candidate.update_number,
                h2h_vs_best=h2h,
                path=candidate.path,
                p1_p2_gap=candidate.p1_p2_gap,
                promotion_eligible=candidate.promotion_eligible,
                role=BEST_EVER_ROLE,
            )
            return True
        return False

    # ---- load as A4 SelfPrevOpponent select_fn -------------------------------
    def load_as_self_prev_opponent_select_fn(
        self,
        entry: SnapshotEntry,
        store: CheckpointStore,
        *,
        policy_factory: Callable[[dict[str, Any]], Callable[[Any], int]] | None = None,
    ) -> Callable[[Any], int]:
        """Load a pool snapshot back into the A4 ``SelfPrevOpponent`` ``select_fn``
        shape (``rust_live_self_play.py:279`` — ``SelfPrevOpponent(select_fn)`` calls
        ``int(self._select_fn(ctx))`` at ``:294``). Returns a ``select_fn(ctx) ->
        action_id`` that is DETERMINISTIC argmax over the snapshot's policy.

        ``store.load_weights(path)`` returns the checkpoint weights dict. ``policy_factory``
        builds a deterministic argmax ``select_fn`` from those weights — production
        wires ``MLXV5LearnerPolicy`` argmax; the fake store / fake factory wires a
        deterministic stub (``test_load_snapshot_as_self_prev_opponent``). When
        ``policy_factory`` is None a default deterministic factory is used: it builds a
        select_fn that picks the legal action whose feature-norm is highest under the
        loaded weights (a deterministic, weights-dependent argmax — no MLX required,
        no randomness, so two calls with the same (weights, ctx) yield the same
        action_id). This default is a SOURCE-VS-SOURCE stub: it is NOT the policy under
        test scoring its own outputs; it is a deterministic function of the loaded
        checkpoint weights + the opponent's legal-action context.

        Pure self-play (``BLOCK_B_PLAN.md:233``): when A4 wires this select_fn into
        ``SelfPrevOpponent``, the learner plays a PRIOR snapshot of itself (the pool
        snapshot), not the live learner policy.
        """
        weights = store.load_weights(entry.path)

        def _default_factory(w: dict[str, Any]) -> Callable[[Any], int]:
            # Deterministic argmax over legal actions weighted by a weights vector.
            # ``w["action_weights"]`` is a 1-D array of per-candidate scores; the
            # select_fn picks the legal action id with the highest score (ties broken
            # by lowest id — deterministic). Falls back to the first legal action when
            # no weights vector is present.
            scores = w.get("action_weights")

            def _select(ctx: Any) -> int:
                ids = list(ctx.legal_action_ids)
                if not ids:
                    raise ValueError(
                        "snapshot_pool select_fn: ctx has no legal actions "
                        "(should have been reset)"
                    )
                if scores is None:
                    return int(ids[0])
                best_id = None
                best_score = None
                for aid in ids:
                    s = float(scores[int(aid)])
                    if best_score is None or s > best_score or (s == best_score and aid < best_id):
                        best_score = s
                        best_id = aid
                return int(best_id)

            return _select

        factory = policy_factory if policy_factory is not None else _default_factory
        return factory(weights)

    def wire_self_prev_opponent(
        self,
        entry: SnapshotEntry,
        store: CheckpointStore,
        *,
        policy_factory: Callable[[dict[str, Any]], Callable[[Any], int]] | None = None,
    ):
        """Convenience: build the ``select_fn`` (``load_as_self_prev_opponent_select_fn``)
        and wire it into a real A4 ``SelfPrevOpponent`` (``rust_live_self_play.py:279``).
        Returns the ``SelfPrevOpponent`` instance. Imported lazily so the module body
        does not require the A4 import chain (and so tests can skip-gate if the import
        ever pulls MLX/Rust)."""
        from .rust_live_self_play import SelfPrevOpponent  # A4 :279

        select_fn = self.load_as_self_prev_opponent_select_fn(
            entry, store, policy_factory=policy_factory
        )
        return SelfPrevOpponent(select_fn)

    # ---- D-B5 hybrid: self-snapshot prevalence weight ------------------------
    def self_snapshot_prevalence_weight(self) -> float:
        """D-B5 hybrid (``BLOCK_B_PLAN.md:142/154`` + ``BLOCK_B_PLAN.md:247-249``):
        the self-snapshot mix-weight available to the pool, grown as the pool fills.

        The frozen NON-self share (V4-orig 0.55 + exploit 0.15 +
        tail 0.05 = 0.75) is ``self.frozen_non_self_share``; the
        self-snapshot share is the RESIDUAL
        ``1 - frozen_non_self_share`` scaled by ``min(non_anchor_count /
        prevalence_pool_target, 1.0)``. The prevalence is MONOTONE-INCREASING in the
        non-anchor pool size (0 when the pool is empty, the full residual when the pool
        is at/above ``prevalence_pool_target``). This is the B1 surface B3/B4 read to
        grow self-snapshot prevalence; the mana_draw-collapse monitor that TRIGGERS
        reweighting lives in B3/B4 (B1 only exposes the pool-size-driven weight).

        Defensible monotone-increasing-in-pool-size formula (linear ramp): the residual
        is the headroom the spec leaves for self-snapshots (Q5 says self-snapshot
        prevalence should be HIGH, ``BLOCK_B_PLAN.md:174``); ramping it from 0 to the
        full residual as the pool fills prevents an empty pool from claiming a large
        self-snapshot share it cannot populate.
        """
        residual = max(0.0, 1.0 - float(self.frozen_non_self_share))
        if self.prevalence_pool_target <= 0:
            return residual
        fill = min(float(self.non_anchor_count) / float(self.prevalence_pool_target), 1.0)
        return residual * fill

    # ---- manifest round-trip -------------------------------------------------
    def to_manifest(self) -> dict[str, Any]:
        """Serialize the pool (anchors + rolling + config) to a JSON-compatible
        manifest dict (``test_manifest_roundtrip``). Paths + metadata intact."""
        return {
            "target_non_anchor_count": int(self.target_non_anchor_count),
            "best_ever_strict_threshold": float(self.best_ever_strict_threshold),
            "frozen_non_self_share": float(self.frozen_non_self_share),
            "prevalence_pool_target": int(self.prevalence_pool_target),
            "seed": self._seed.to_dict() if self._seed is not None else None,
            "best_ever": self._best_ever.to_dict() if self._best_ever is not None else None,
            "rolling": [e.to_dict() for e in self._rolling],
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "SnapshotPool":
        """Rebuild a pool from a manifest dict (``test_manifest_roundtrip``).
        Round-trips paths + metadata; the seed immutability is preserved (the rebuilt
        pool's ``set_seed_anchor`` will still refuse a second seed)."""
        pool = cls(
            target_non_anchor_count=int(manifest.get("target_non_anchor_count", 6)),
            best_ever_strict_threshold=float(
                manifest.get("best_ever_strict_threshold", 0.5)
            ),
            frozen_non_self_share=float(manifest.get("frozen_non_self_share", 0.75)),
            prevalence_pool_target=int(manifest.get("prevalence_pool_target", 6)),
        )
        seed_d = manifest.get("seed")
        if seed_d is not None:
            pool._seed = SnapshotEntry.from_dict(seed_d)
        be_d = manifest.get("best_ever")
        if be_d is not None:
            pool._best_ever = SnapshotEntry.from_dict(be_d)
        pool._rolling = [SnapshotEntry.from_dict(d) for d in manifest.get("rolling", [])]
        # Defensive: a hand-crafted / legacy manifest may carry more rolling entries
        # than ``target_non_anchor_count``. Re-run FIFO so the reloaded pool is
        # immediately well-formed (no excess non-anchors waiting for the next
        # ``add_snapshot`` to evict). Anchors are NEVER touched by ``_evict_fifo``
        # (they live in ``_seed`` / ``_best_ever``, not ``_rolling``), so this cannot
        # evict a seed or best-ever — it only trims surplus rolling entries.
        pool._evict_fifo()
        return pool

    def write_manifest(self, path: str | Path) -> str:
        """Write the manifest to ``path`` as JSON (``test_manifest_roundtrip``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_manifest(), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return str(out)

    @classmethod
    def read_manifest(cls, path: str | Path) -> "SnapshotPool":
        """Read a manifest written by ``write_manifest`` (``test_manifest_roundtrip``)."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_manifest(json.loads(text))


__all__ = [
    "SnapshotEntry",
    "SnapshotPool",
    "CheckpointStore",
    "SEED_ROLE",
    "BEST_EVER_ROLE",
    "ROLLING_ROLE",
]
