"""Private, in-memory production recorder for ExtraLR V5 training data.

The browser-facing battle state deliberately hides the opponent's hand and
deck.  This recorder is a separate, server-only surface which keeps the full
state required by the ``rlhf_v5_storage_v1`` offline bridge.

The implementation lives in ``core`` and intentionally does not import
``rlhf_env``.  Production and the headless RLHF harness may therefore evolve
independently while emitting the same storage contract.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from threading import RLock
import time
from typing import Any, Callable, Deque, Mapping, Optional

from core.actions import AttackAction, BaseAction, EndTurnAction, ManaDrawAction, PlayCardAction


V5_STORAGE_SCHEMA = "rlhf_v5_storage_v1"
V5_VISIBILITY = "omniscient_offline_only"
V5_ACTIONS_VERSION = "classic_actions_v1"
V5_OBS_VERSION = "classic_obs_v1"
V5_CARD_SHAPE_VERSION = "classic_card_shape_v1"
V5_CARD_PARAMS_SCHEMA = "train_v3_card_params_v1"
V5_DECK_PARAMS_SCHEMA = "train_v3_deck_params_v1"

CONTROL_SOURCES = frozenset({"human", "bot", "replacement_bot", "timeout"})
DECISION_SOURCES = frozenset({"human", "llm", "bot", "rl"})
# Backward-compatible public name used by BattleEngine source classification.
ACTION_SOURCES = CONTROL_SOURCES | frozenset({"llm", "rl"})
ACTION_SOURCE_ACTOR_TYPES: dict[str, str] = {
    "human": "human",
    "bot": "bot",
    "replacement_bot": "bot",
    "timeout": "bot",
    "llm": "llm",
    "rl": "rl",
}
TRAINING_ACTION_TYPES = frozenset({"play_card", "attack", "end_turn", "mana_draw"})

V5_POLICY_FAILURE_PREFIX = "v5_policy_failure:"
V5_POLICY_FAILURE_CODES = frozenset(
    {
        "decode_failed",
        "empty_legal_actions",
        "invalid_action_index",
        "invalid_io_contract",
        "invalid_output_contract",
        "legal_mapping_failed",
        "mana_surface_mismatch",
        "no_legal_candidate",
        "non_finite_logits",
        "non_finite_mana_logit",
        "profile_unavailable",
        "unexpected_failure",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def v5_policy_failure_warning(code: Any) -> str:
    """Return a bounded, secret-free warning code for trace metadata."""

    normalized = str(code or "").strip().lower()
    if normalized not in V5_POLICY_FAILURE_CODES:
        normalized = "unexpected_failure"
    return f"{V5_POLICY_FAILURE_PREFIX}{normalized}"


def v5_policy_failure_error(code: Any) -> RuntimeError:
    """Build the stable RuntimeError used by every fail-closed V5 adapter."""

    return RuntimeError(v5_policy_failure_warning(code))


def v5_policy_failure_code(error: BaseException) -> str:
    """Extract a known code without copying arbitrary exception text."""

    message = str(error or "")
    if message.startswith(V5_POLICY_FAILURE_PREFIX):
        code = message[len(V5_POLICY_FAILURE_PREFIX) :]
        if code in V5_POLICY_FAILURE_CODES:
            return code
    return "unexpected_failure"


def canonical_actor_type(decision_source: str) -> str:
    """Map detailed control source to the stable actor taxonomy."""

    source = str(decision_source or "human").lower()
    return ACTION_SOURCE_ACTOR_TYPES.get(source, "bot")


def _engine_dataset_ids(engine: Any) -> tuple[str, str]:
    """Return immutable storage trace ID and reusable gameplay match ID."""

    match_id = str(getattr(engine, "match_id", "") or "")
    battle_id = str(getattr(engine, "v5_dataset_trace_id", "") or match_id)
    return battle_id, match_id


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _json_safe(value: Any) -> Any:
    """Best-effort immutable JSON-compatible copy for DB/outbox consumers."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:  # noqa: BLE001 - recorder must never break gameplay.
            pass
    return str(value)


def _board_power(board: Any) -> float:
    return float(
        sum(
            max(0, int(getattr(card, "attack", 0)))
            * max(0, int(getattr(card, "hp", 0)))
            for card in (board or ())
        )
    )


@dataclass(frozen=True)
class _DecisionClock:
    started_monotonic: float
    censored: bool = False
    censor_reason: Optional[str] = None


class InMemoryV5DatasetRecorder:
    """Thread-safe, match-owned ``rlhf_v5_storage_v1`` recorder.

    All writes are in memory.  ``snapshot()``, ``checkpoint()``, ``finalize()``
    and ``abort()`` return detached objects suitable for a DB transaction or a
    durable outbox; mutating a returned object cannot corrupt live recording.
    """

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = RLock()
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._started = False
        self._finalized = False
        self._aborted = False
        self._recording_started_monotonic: Optional[float] = None
        self._battle_started_monotonic: Optional[float] = None
        self._last_turn_snapshot: Optional[int] = None
        self._next_seq = 1
        self._next_control_seq = 1
        self._meta: dict[str, Any] = {}
        self._turns: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._control_events: list[dict[str, Any]] = []
        self._pending_actions: dict[int, dict[str, Any]] = {}
        self._next_contexts: dict[int, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=8)
        )
        self._decision_clocks: dict[int, _DecisionClock] = {}

    # ------------------------------------------------------------------
    # Match lifecycle and metadata
    # ------------------------------------------------------------------
    def start_match(
        self,
        engine: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Bind the recorder to an initialized battle and snapshot turn one."""

        with self._lock:
            self._ensure_started_unlocked(engine, metadata=metadata)
            self._append_turn_if_needed_unlocked(engine)

    def _ensure_started_unlocked(
        self,
        engine: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._started:
            if metadata:
                self._merge_metadata_unlocked(metadata)
            return

        arena = getattr(engine, "_arena", None)
        if arena is None:
            return
        state = arena.state
        now_monotonic = float(self._monotonic_clock())
        now_wall = float(self._wall_clock())
        self._recording_started_monotonic = now_monotonic
        self._battle_started_monotonic = now_monotonic
        battle_id, match_id = _engine_dataset_ids(engine)
        p1_actor = "bot" if bool(getattr(state.p1, "is_bot", False)) else "human"
        p2_actor = "bot" if bool(getattr(state.p2, "is_bot", False)) else "human"
        self._meta = {
            "schema_version": V5_STORAGE_SCHEMA,
            "visibility": V5_VISIBILITY,
            "private_server_only": True,
            "v5_trace_present": True,
            "battle_id": battle_id,
            "match_id": match_id,
            "dataset_generation": int(
                getattr(engine, "v5_dataset_generation", 1) or 1
            ),
            "dataset_generation_reason": str(
                getattr(engine, "v5_dataset_generation_reason", "initial")
                or "initial"
            ),
            "created_at": _utc_now_iso(),
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "status": "ongoing",
            "terminal_reason": None,
            "aborted": False,
            "abort_reason": None,
            "winner_user_id": None,
            "duration_seconds": None,
            "turns": None,
            "actions_version": V5_ACTIONS_VERSION,
            "obs_version": V5_OBS_VERSION,
            "card_shape_version": V5_CARD_SHAPE_VERSION,
            "card_params_schema": V5_CARD_PARAMS_SCHEMA,
            "deck_params_schema": V5_DECK_PARAMS_SCHEMA,
            "catalog_hash": None,
            "engine_version": {"component": "production_battle_engine"},
            "game_mode": str(getattr(engine, "game_mode", "classic") or "classic"),
            "ruleset": str(getattr(engine, "ruleset", "classic") or "classic"),
            "p1_user_id": int(state.p1.user_id),
            "p2_user_id": int(state.p2.user_id),
            "p1_is_bot": bool(getattr(state.p1, "is_bot", False)),
            "p2_is_bot": bool(getattr(state.p2, "is_bot", False)),
            "p1_actor_type": p1_actor,
            "p2_actor_type": p2_actor,
            "battle_tag": f"{p1_actor}-vs-{p2_actor}",
            "starting_player_id": int(state.current_turn_owner_id),
            "starting_player": (
                "p1" if int(state.current_turn_owner_id) == int(state.p1.user_id) else "p2"
            ),
            "match_start_unix_ms": int(now_wall * 1000),
            "start_metadata": {
                "turn_number": int(state.turn_number),
                "starting_player_id": int(state.current_turn_owner_id),
                "client_ready_anchored": False,
            },
            "p1_deck": self._fallback_initial_deck(state.p1),
            "p2_deck": self._fallback_initial_deck(state.p2),
            "model_provenance": {},
            "aux_model_provenance": {},
            "degraded": False,
            "policy_warnings": [],
            "timestamp_features": {
                "p1_deck_size": self._full_deck_size(state.p1),
                "p2_deck_size": self._full_deck_size(state.p2),
                "starting_player": (
                    "p1"
                    if int(state.current_turn_owner_id) == int(state.p1.user_id)
                    else "p2"
                ),
            },
        }
        self._started = True
        if metadata:
            self._merge_metadata_unlocked(metadata)

    @staticmethod
    def _full_deck_size(player: Any) -> int:
        return int(
            len(getattr(player, "hand", ()) or ())
            + len(getattr(player, "deck", ()) or ())
            + len(getattr(player, "board", ()) or ())
            + len(getattr(player, "graveyard", ()) or ())
            + (1 if getattr(player, "hero", None) is not None else 0)
        )

    @staticmethod
    def _fallback_initial_deck(player: Any) -> list[dict[str, Any]]:
        cards = [
            getattr(player, "hero", None),
            *(getattr(player, "hand", ()) or ()),
            *(getattr(player, "deck", ()) or ()),
            *(getattr(player, "board", ()) or ()),
            *(getattr(player, "graveyard", ()) or ()),
        ]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            if card is None:
                continue
            instance_key = str(getattr(card, "instance_id", ""))
            if instance_key in seen:
                continue
            seen.add(instance_key)
            rows.append(
                {
                    "card_id": int(getattr(card, "card_id", 0)),
                    "level": int(getattr(card, "level", 1) or 1),
                    "instance_id": None,
                }
            )
        return rows

    def merge_metadata(self, engine: Any, metadata: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_started_unlocked(engine)
            self._merge_metadata_unlocked(metadata)

    def mark_policy_degraded(self, engine: Any, code: Any) -> str:
        """Permanently exclude a policy-fallback trace from training readiness."""

        warning = v5_policy_failure_warning(code)
        with self._lock:
            self._ensure_started_unlocked(engine)
            self._meta["degraded"] = True
            warnings = self._meta.setdefault("policy_warnings", [])
            if not isinstance(warnings, list):
                warnings = []
                self._meta["policy_warnings"] = warnings
            if warning not in warnings:
                warnings.append(warning)
        return warning

    def _merge_metadata_unlocked(self, metadata: Mapping[str, Any]) -> None:
        for key, value in _json_safe(metadata).items():
            if key == "degraded":
                # A policy failure is a permanent property of the trace.
                # Later lifecycle metadata must never make it trainable again.
                self._meta["degraded"] = bool(
                    self._meta.get("degraded") is True or value is True
                )
                continue
            if key == "policy_warnings":
                warnings = self._meta.setdefault("policy_warnings", [])
                if not isinstance(warnings, list):
                    warnings = []
                    self._meta["policy_warnings"] = warnings
                incoming = value if isinstance(value, list) else [value]
                for item in incoming:
                    warning = v5_policy_failure_warning(
                        v5_policy_failure_code(RuntimeError(str(item)))
                    )
                    if warning not in warnings:
                        warnings.append(warning)
                if warnings:
                    self._meta["degraded"] = True
                continue
            if (
                key in self._meta
                and isinstance(self._meta[key], dict)
                and isinstance(value, dict)
            ):
                self._meta[key].update(value)
            else:
                self._meta[key] = value
        for player in (1, 2):
            actor_type = self._meta.get(f"p{player}_actor_type")
            if actor_type in DECISION_SOURCES:
                self._meta[f"p{player}_is_bot"] = actor_type in {
                    "bot",
                    "rl",
                    "llm",
                }
        p1_actor = self._meta.get("p1_actor_type")
        p2_actor = self._meta.get("p2_actor_type")
        if p1_actor in DECISION_SOURCES and p2_actor in DECISION_SOURCES:
            self._meta["battle_tag"] = f"{p1_actor}-vs-{p2_actor}"

    def mark_battle_started(
        self,
        engine: Any,
        *,
        reason: str = "client_ready",
        now_monotonic: Optional[float] = None,
        now_wall: Optional[float] = None,
    ) -> None:
        """Anchor TimeStamp duration to the moment gameplay becomes observable."""

        with self._lock:
            self._ensure_started_unlocked(engine)
            if not self._started:
                return
            start_metadata = self._meta.setdefault("start_metadata", {})
            if start_metadata.get("client_ready_anchored") is True:
                start_metadata["duplicate_start_anchor_count"] = int(
                    start_metadata.get("duplicate_start_anchor_count") or 0
                ) + 1
                return
            # Re-anchoring after an action would create a negative/incomplete
            # duration.  Keep the original anchor and surface the late signal.
            if self._actions or self._pending_actions:
                start_metadata["late_start_anchor_ignored"] = str(reason)
                return
            mono = (
                float(now_monotonic)
                if now_monotonic is not None
                else float(self._monotonic_clock())
            )
            wall = (
                float(now_wall) if now_wall is not None else float(self._wall_clock())
            )
            self._battle_started_monotonic = mono
            self._meta["started_at"] = datetime.fromtimestamp(
                wall, timezone.utc
            ).isoformat().replace("+00:00", "Z")
            self._meta["match_start_unix_ms"] = int(wall * 1000)
            start_metadata.update(
                {
                    "client_ready_anchored": True,
                    "anchor_reason": str(reason),
                }
            )

    # ------------------------------------------------------------------
    # Human decision clocks and one-shot action context
    # ------------------------------------------------------------------
    def arm_human_decision_clock(
        self,
        user_id: int,
        *,
        now_monotonic: Optional[float] = None,
        censored: bool = False,
        censor_reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._decision_clocks[int(user_id)] = _DecisionClock(
                started_monotonic=(
                    float(now_monotonic)
                    if now_monotonic is not None
                    else float(self._monotonic_clock())
                ),
                censored=bool(censored),
                censor_reason=(str(censor_reason) if censor_reason else None),
            )

    def censor_human_decision_clock(self, user_id: int, reason: str) -> None:
        self.arm_human_decision_clock(
            user_id,
            censored=True,
            censor_reason=str(reason or "unspecified"),
        )

    def capture_action_context(
        self,
        *,
        user_id: int,
        action_json: Mapping[str, Any],
        decision_source: str,
        control_source: Optional[str] = None,
        actor_type: Optional[str] = None,
        request_monotonic: Optional[float] = None,
        client_action_id: Optional[str] = None,
        human_decision_time_ms: Optional[float] = None,
        decision_time_censored: Optional[bool] = None,
        decision_censor_reason: Optional[str] = None,
        metronome_prediction_ms: Optional[float] = None,
        metronome_applied_ms: Optional[float] = None,
        metronome_fallback_used: Optional[bool] = None,
        quality_score: Optional[float] = None,
        legacy_action_index: Optional[int] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        queue: bool = True,
    ) -> dict[str, Any]:
        """Capture request-entry timing and optionally queue it for one action."""

        with self._lock:
            raw_source = str(decision_source or "human").lower()
            if raw_source not in ACTION_SOURCES:
                raw_source = "human"
            if control_source is not None:
                control = str(control_source).lower()
            elif raw_source in CONTROL_SOURCES:
                control = raw_source
            else:
                control = "bot"
            if control not in CONTROL_SOURCES:
                control = "human" if raw_source == "human" else "bot"
            if (
                control == "timeout"
                and str(action_json.get("type") or "") != "end_turn"
            ):
                control = "bot"

            if raw_source in DECISION_SOURCES:
                source = raw_source
            else:
                source = "human" if control == "human" else "bot"
            if control in {"replacement_bot", "timeout"}:
                source = "bot"
            now = (
                float(request_monotonic)
                if request_monotonic is not None
                else float(self._monotonic_clock())
            )

            measured_ms: Optional[float] = None
            censored = False
            censor_reason: Optional[str] = None
            raw_measured_ms: Optional[float] = None
            if source == "human" and control == "human":
                decision_clock = self._decision_clocks.get(int(user_id))
                if human_decision_time_ms is not None:
                    measured_ms = max(0.0, float(human_decision_time_ms))
                elif decision_clock is None:
                    censored = True
                    censor_reason = "not_observed"
                elif decision_clock.censored:
                    censored = True
                    censor_reason = decision_clock.censor_reason or "clock_censored"
                else:
                    measured_ms = max(
                        0.0, (now - decision_clock.started_monotonic) * 1000.0
                    )
                raw_measured_ms = measured_ms
                if measured_ms is not None and not (100.0 <= measured_ms <= 25_000.0):
                    measured_ms = None
                    censored = True
                    censor_reason = "outside_training_window"

            if decision_time_censored is not None:
                censored = bool(decision_time_censored)
            if decision_censor_reason:
                censor_reason = str(decision_censor_reason)
            if censored:
                measured_ms = None
                censor_reason = censor_reason or "unspecified"
            elif source == "human" and control == "human":
                censor_reason = None
            if source != "human" or control != "human":
                measured_ms = None
                raw_measured_ms = None
                censored = False
                censor_reason = None

            def _finite_optional(value: Optional[float]) -> Optional[float]:
                if value is None:
                    return None
                number = float(value)
                return number if math.isfinite(number) else None

            context = {
                "action_json": _json_safe(action_json),
                "decision_source": source,
                "control_source": control,
                "actor_type": str(actor_type or canonical_actor_type(raw_source)),
                "request_monotonic_ms": int(now * 1000),
                "client_action_id": (
                    str(client_action_id) if client_action_id is not None else None
                ),
                "human_decision_time_ms": (
                    int(round(measured_ms)) if measured_ms is not None else None
                ),
                "human_decision_time_raw_ms": _finite_optional(raw_measured_ms),
                "decision_time_censored": bool(censored),
                "decision_censor_reason": censor_reason,
                "metronome_prediction_ms": _finite_optional(
                    metronome_prediction_ms
                ),
                "metronome_applied_ms": _finite_optional(metronome_applied_ms),
                "metronome_fallback_used": (
                    bool(metronome_fallback_used)
                    if metronome_fallback_used is not None
                    else None
                ),
                "quality_score": (
                    float(quality_score) if quality_score is not None else None
                ),
                "legacy_action_index": legacy_action_index,
                "extra_context": _json_safe(extra_context or {}),
            }
            if queue:
                self._next_contexts[int(user_id)].append(deepcopy(context))
            return deepcopy(context)

    def consume_action_context(
        self,
        user_id: int,
        *,
        fallback_action_json: Mapping[str, Any],
        decision_source: str,
        control_source: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            queue = self._next_contexts.get(int(user_id))
            if queue:
                context = queue.popleft()
                if not queue:
                    self._next_contexts.pop(int(user_id), None)
                return context
        return self.capture_action_context(
            user_id=int(user_id),
            action_json=fallback_action_json,
            decision_source=decision_source,
            control_source=control_source,
            actor_type=actor_type,
            queue=False,
        )

    # ------------------------------------------------------------------
    # Action transitions
    # ------------------------------------------------------------------
    def before_action(
        self,
        engine: Any,
        *,
        user_id: int,
        action: Optional[BaseAction],
        context: Mapping[str, Any],
    ) -> int:
        with self._lock:
            self._ensure_started_unlocked(engine)
            if not self._started or self._finalized:
                return -1
            self._append_turn_if_needed_unlocked(engine)
            state = engine._arena.state
            raw_action = _json_safe(
                context.get("action_json")
                or (action.to_dict() if action is not None else {"type": "unknown"})
            )
            legal = self._legal_actions(engine, int(user_id))
            legal_index = self._resolve_legal_index(action, raw_action, legal)
            action_native = None
            if legal_index is not None and 0 <= legal_index < len(legal):
                action_native = _json_safe(legal[legal_index].to_dict())
            elif action is not None:
                action_native = _json_safe(action.to_dict())

            source_card, target_card = self._resolve_source_target(
                engine, int(user_id), action, raw_action
            )
            actor_player = (
                1 if int(user_id) == int(state.p1.user_id) else 2
            )
            pre_state = self._snapshot_state(engine)
            seq = self._next_seq
            self._next_seq += 1
            battle_id, match_id = _engine_dataset_ids(engine)
            row = {
                "seq": seq,
                "battle_id": battle_id,
                "match_id": match_id,
                "turn_number": int(state.turn_number),
                "actor_user_id": int(user_id),
                "actor_player": actor_player,
                "acting_user_id": int(user_id),  # legacy admin field
                "acting_player": actor_player,  # legacy admin field
                "decision_source": context.get("decision_source", "human"),
                "control_source": context.get(
                    "control_source", context.get("decision_source", "human")
                ),
                "actor_type": context.get(
                    "actor_type",
                    canonical_actor_type(str(context.get("decision_source", "human"))),
                ),
                "is_bot": bool(
                    getattr(
                        state.p1 if actor_player == 1 else state.p2,
                        "is_bot",
                        False,
                    )
                ),
                "controlled_by_bot": context.get("control_source")
                in {"bot", "replacement_bot", "timeout"}
                or context.get("decision_source") in {"bot", "llm", "rl"},
                "legal_action_index": legal_index,
                "action_type": str(raw_action.get("type") or "unknown"),
                "action_json": raw_action,
                "action_native": action_native,
                "training_action_native": None,
                "source_card": source_card,
                "target_card": target_card,
                "legal_actions": [_json_safe(item.to_dict()) for item in legal],
                "legal_action_count": len(legal),
                "pre_state": pre_state,
                "post_state": None,
                "state_json": deepcopy(pre_state),  # legacy admin field
                "deltas": None,
                "accepted": None,
                "error": None,
                "is_training_label": False,
                "timestamp_ms": int(self._monotonic_clock() * 1000),
                "recorded_at": _utc_now_iso(),
                "visibility": V5_VISIBILITY,
                "human_decision_time_ms": context.get("human_decision_time_ms"),
                "human_decision_time_raw_ms": context.get(
                    "human_decision_time_raw_ms"
                ),
                "decision_time_censored": bool(
                    context.get("decision_time_censored", False)
                ),
                "decision_censor_reason": context.get("decision_censor_reason"),
                "metronome_prediction_ms": context.get(
                    "metronome_prediction_ms"
                ),
                "metronome_applied_ms": context.get("metronome_applied_ms"),
                "metronome_fallback_used": context.get(
                    "metronome_fallback_used"
                ),
                "client_action_id": context.get("client_action_id"),
                "quality_score": context.get("quality_score"),
                "context_json": _json_safe(context.get("extra_context") or {}),
            }
            self._pending_actions[seq] = {
                "row": row,
                "pre_reward": self._reward_snapshot(state, int(user_id)),
                "legacy_action_index": context.get("legacy_action_index"),
            }
            return seq

    def record_rejected_action(
        self,
        engine: Any,
        *,
        user_id: int,
        action_json: Mapping[str, Any],
        context: Mapping[str, Any],
        error: str,
    ) -> int:
        token = self.before_action(
            engine,
            user_id=int(user_id),
            action=None,
            context={**dict(context), "action_json": _json_safe(action_json)},
        )
        self.after_action(
            engine,
            token=token,
            accepted=False,
            error=str(error),
        )
        return token

    def after_action(
        self,
        engine: Any,
        *,
        token: int,
        accepted: bool,
        error: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if token is None or int(token) < 0:
            return None
        with self._lock:
            pending = self._pending_actions.pop(int(token), None)
            if pending is None:
                return None
            row = pending["row"]
            state = engine._arena.state
            post_reward = self._reward_snapshot(
                state, int(row["actor_user_id"])
            )
            row["post_state"] = self._snapshot_state(engine)
            row["deltas"] = self._deltas(pending["pre_reward"], post_reward)
            row["accepted"] = bool(accepted)
            row["error"] = (
                str(error)
                if error
                else None
                if accepted
                else "rejected_without_error"
            )
            row["is_training_label"] = bool(
                accepted
                and row["action_type"] in TRAINING_ACTION_TYPES
                and row["legal_action_index"] is not None
            )
            if row["is_training_label"]:
                row["training_action_native"] = deepcopy(row["action_native"])
            row["context_json"] = {
                **dict(row.get("context_json") or {}),
                "accepted": bool(accepted),
                "error": row["error"],
                "decision_source": row["decision_source"],
                "actor_type": row["actor_type"],
                "human_decision_time_ms": row["human_decision_time_ms"],
                "human_decision_time_raw_ms": row[
                    "human_decision_time_raw_ms"
                ],
                "decision_time_censored": row["decision_time_censored"],
                "decision_censor_reason": row["decision_censor_reason"],
                "metronome_prediction_ms": row["metronome_prediction_ms"],
                "metronome_applied_ms": row["metronome_applied_ms"],
            }
            self._actions.append(row)
            if accepted and row["control_source"] in {"human", "timeout"}:
                self._decision_clocks.pop(int(row["actor_user_id"]), None)
            return {
                "row": deepcopy(row),
                "legacy_action_index": pending.get("legacy_action_index"),
            }

    @staticmethod
    def _legal_actions(engine: Any, user_id: int) -> list[BaseAction]:
        try:
            return list(engine._arena.get_legal_actions(int(user_id)))
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _resolve_legal_index(
        action: Optional[BaseAction],
        raw_action: Mapping[str, Any],
        legal: list[BaseAction],
    ) -> Optional[int]:
        action_type = str(raw_action.get("type") or "")
        if isinstance(action, EndTurnAction) or action_type == "end_turn":
            return next(
                (index for index, item in enumerate(legal) if isinstance(item, EndTurnAction)),
                None,
            )
        if isinstance(action, ManaDrawAction) or action_type == "mana_draw":
            return next(
                (index for index, item in enumerate(legal) if isinstance(item, ManaDrawAction)),
                None,
            )
        if isinstance(action, PlayCardAction):
            for index, item in enumerate(legal):
                if not isinstance(item, PlayCardAction):
                    continue
                # Web placement and engine append-only placement historically
                # use different position values; position is not a discriminator.
                if int(item.hand_index) != int(action.hand_index):
                    continue
                if (str(item.target_id) if item.target_id else None) != (
                    str(action.target_id) if action.target_id else None
                ):
                    continue
                return index
            return None
        if isinstance(action, AttackAction):
            for index, item in enumerate(legal):
                if not isinstance(item, AttackAction):
                    continue
                if str(item.attacker_id) != str(action.attacker_id):
                    continue
                if bool(item.target_is_hero) != bool(action.target_is_hero):
                    continue
                if not action.target_is_hero and (
                    str(item.target_id) if item.target_id else None
                ) != (str(action.target_id) if action.target_id else None):
                    continue
                return index
        return None

    def _resolve_source_target(
        self,
        engine: Any,
        user_id: int,
        action: Optional[BaseAction],
        raw_action: Mapping[str, Any],
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        state = engine._arena.state
        if int(state.p1.user_id) == int(user_id):
            me, enemy = state.p1, state.p2
        elif int(state.p2.user_id) == int(user_id):
            me, enemy = state.p2, state.p1
        else:
            return None, None

        source = None
        target = None
        if isinstance(action, PlayCardAction):
            if 0 <= int(action.hand_index) < len(me.hand):
                source = me.hand[int(action.hand_index)]
            if action.target_id:
                target = self._find_card_by_instance(
                    [me.hero, enemy.hero, *me.board, *enemy.board],
                    action.target_id,
                )
        elif isinstance(action, AttackAction):
            source = self._find_card_by_instance(me.board, action.attacker_id)
            target = (
                enemy.hero
                if action.target_is_hero
                else self._find_card_by_instance(enemy.board, action.target_id)
            )
        elif raw_action.get("type") == "play_card":
            card_ref = raw_action.get(
                "card_ref", raw_action.get("hand_index")
            )
            try:
                hand_index = int(card_ref)
            except (TypeError, ValueError):
                hand_index = -1
            if 0 <= hand_index < len(me.hand):
                source = me.hand[hand_index]
        return self._snapshot_card(engine, source), self._snapshot_card(engine, target)

    @staticmethod
    def _find_card_by_instance(cards: Any, instance_id: Any) -> Any:
        if instance_id is None:
            return None
        wanted = str(instance_id)
        return next(
            (
                card
                for card in (cards or ())
                if str(getattr(card, "instance_id", "")) == wanted
            ),
            None,
        )

    # ------------------------------------------------------------------
    # Omniscient state and reward snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def _snapshot_card(engine: Any, card: Any) -> Optional[dict[str, Any]]:
        if card is None:
            return None
        try:
            return _json_safe(engine._snapshot_card(card))
        except Exception:  # noqa: BLE001
            return {
                "instance_id": str(getattr(card, "instance_id", "")),
                "card_id": int(getattr(card, "card_id", 0)),
                "level": int(getattr(card, "level", 1) or 1),
                "card_type": str(
                    _enum_value(getattr(card, "card_type", "warrior"))
                ),
                "mana_cost": int(getattr(card, "mana_cost", 0)),
                "attack": int(getattr(card, "attack", 0)),
                "hp": int(getattr(card, "hp", 0)),
                "max_hp": int(getattr(card, "max_hp", 0)),
                "mechanics": list(getattr(card, "mechanics", ()) or ()),
                "is_ready": bool(getattr(card, "is_ready", False)),
                "is_frozen": bool(getattr(card, "is_frozen", False)),
            }

    def _snapshot_player(self, engine: Any, player: Any) -> dict[str, Any]:
        status = getattr(player, "replacement_status", "active")
        return {
            "user_id": int(player.user_id),
            "is_bot": bool(getattr(player, "is_bot", False)),
            "replacement_status": str(_enum_value(status) or "active"),
            "hero": self._snapshot_card(engine, player.hero),
            "mana": int(getattr(player, "mana", 0)),
            "max_mana": int(getattr(player, "max_mana", 0)),
            "trophies": int(getattr(player, "trophies", 0)),
            "hand": [
                self._snapshot_card(engine, card)
                for card in (getattr(player, "hand", ()) or ())
            ],
            "deck": [
                self._snapshot_card(engine, card)
                for card in (getattr(player, "deck", ()) or ())
            ],
            "board": [
                self._snapshot_card(engine, card)
                for card in (getattr(player, "board", ()) or ())
            ],
            "graveyard": [
                self._snapshot_card(engine, card)
                for card in (getattr(player, "graveyard", ()) or ())
            ],
            "mana_draw_count_this_turn": int(
                getattr(player, "mana_draw_count_this_turn", 0)
            ),
        }

    def _snapshot_state(self, engine: Any) -> dict[str, Any]:
        state = engine._arena.state
        return {
            "turn_number": int(state.turn_number),
            "current_turn_owner_id": int(state.current_turn_owner_id),
            "status": str(_enum_value(state.status)),
            "p1": self._snapshot_player(engine, state.p1),
            "p2": self._snapshot_player(engine, state.p2),
            "action_history": _json_safe(
                list(getattr(state, "action_history", ()) or ())
            ),
            "history": _json_safe(list(getattr(state, "history", ()) or ())),
            "v5_history_events": _json_safe(
                list(getattr(state, "v5_history_events", ()) or ())
            ),
            "pending_card_feedback_events": _json_safe(
                list(getattr(state, "pending_card_feedback_events", ()) or ())
            ),
            "visibility": V5_VISIBILITY,
        }

    @staticmethod
    def _reward_snapshot(state: Any, user_id: int) -> dict[str, Any]:
        if int(state.p1.user_id) == int(user_id):
            me, enemy = state.p1, state.p2
        else:
            me, enemy = state.p2, state.p1
        return {
            "my_hero_hp": int(me.hero.hp),
            "enemy_hero_hp": int(enemy.hero.hp),
            "my_board_count": len(me.board),
            "enemy_board_count": len(enemy.board),
            "my_board_power": _board_power(me.board),
            "enemy_board_power": _board_power(enemy.board),
        }

    @staticmethod
    def _deltas(pre: Mapping[str, Any], post: Mapping[str, Any]) -> dict[str, Any]:
        pre_board_delta = float(pre["my_board_power"]) - float(
            pre["enemy_board_power"]
        )
        post_board_delta = float(post["my_board_power"]) - float(
            post["enemy_board_power"]
        )
        return {
            "enemy_hero_hp_delta": int(
                pre["enemy_hero_hp"] - post["enemy_hero_hp"]
            ),
            "own_hero_hp_delta": int(
                pre["my_hero_hp"] - post["my_hero_hp"]
            ),
            "my_board_count_delta": int(
                post["my_board_count"] - pre["my_board_count"]
            ),
            "enemy_board_count_delta": int(
                post["enemy_board_count"] - pre["enemy_board_count"]
            ),
            "board_power_delta": float(post_board_delta - pre_board_delta),
        }

    def _append_turn_if_needed_unlocked(self, engine: Any) -> None:
        if not self._started or getattr(engine, "_arena", None) is None:
            return
        turn_number = int(engine._arena.state.turn_number)
        if turn_number == self._last_turn_snapshot:
            return
        row = self._snapshot_state(engine)
        self._turns.append(row)
        self._last_turn_snapshot = turn_number

    # ------------------------------------------------------------------
    # Control changes
    # ------------------------------------------------------------------
    def record_control_change(
        self,
        engine: Any,
        *,
        user_id: int,
        previous_status: Any,
        new_status: Any,
        reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._ensure_started_unlocked(engine)
            if not self._started or self._finalized:
                return
            previous = str(_enum_value(previous_status) or "active")
            new = str(_enum_value(new_status) or "active")
            if previous == new:
                return
            battle_id, _match_id = _engine_dataset_ids(engine)
            self._control_events.append(
                {
                    "seq": self._next_control_seq,
                    "battle_id": battle_id,
                    # The mutation happened after this many completed action
                    # transitions. Validators can therefore distinguish one
                    # audited control-plane change from a corrupted state gap.
                    "after_action_seq": len(self._actions),
                    "turn_number": int(engine._arena.state.turn_number),
                    "user_id": int(user_id),
                    "previous_status": previous,
                    "new_status": new,
                    "reason": str(reason) if reason else None,
                    "timestamp_ms": int(self._monotonic_clock() * 1000),
                    "recorded_at": _utc_now_iso(),
                }
            )
            self._next_control_seq += 1

    # ------------------------------------------------------------------
    # Checkpoint/finalization API
    # ------------------------------------------------------------------
    def snapshot(self, engine: Optional[Any] = None) -> dict[str, Any]:
        with self._lock:
            if engine is not None:
                self._ensure_started_unlocked(engine)
            return self._snapshot_unlocked()

    def checkpoint(
        self,
        engine: Any,
        *,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started_unlocked(engine)
            self._append_turn_if_needed_unlocked(engine)
            payload = self._snapshot_unlocked()
            payload["checkpoint"] = {
                "created_at": _utc_now_iso(),
                "reason": str(reason) if reason else None,
                "pending_action_count": len(self._pending_actions),
            }
            return payload

    def finalize(
        self,
        engine: Any,
        *,
        winner_user_id: Optional[int] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started_unlocked(engine)
            if not self._started:
                return self._snapshot_unlocked()
            if metadata:
                self._merge_metadata_unlocked(metadata)
            if not self._finalized:
                self._append_turn_if_needed_unlocked(engine)
                state = engine._arena.state
                resolved_status = status or str(_enum_value(state.status))
                resolved_winner = winner_user_id
                if resolved_winner is None:
                    status_value = str(_enum_value(state.status))
                    if status_value == "p1_win":
                        resolved_winner = int(state.p1.user_id)
                    elif status_value == "p2_win":
                        resolved_winner = int(state.p2.user_id)
                now = float(self._monotonic_clock())
                start = self._battle_started_monotonic
                duration = max(0.0, now - start) if start is not None else None
                self._meta.update(
                    {
                        "finished_at": _utc_now_iso(),
                        "status": str(resolved_status),
                        "terminal_reason": str(reason) if reason else None,
                        "winner_user_id": resolved_winner,
                        "duration_seconds": (
                            round(float(duration), 3)
                            if duration is not None
                            else None
                        ),
                        "turns": int(state.turn_number),
                        "final_state": self._snapshot_state(engine),
                    }
                )
                self._meta.setdefault("timestamp_features", {}).update(
                    {
                        "duration_seconds": self._meta["duration_seconds"],
                        "turns": int(state.turn_number),
                        "winner_player": (
                            1
                            if resolved_winner == int(state.p1.user_id)
                            else 2
                            if resolved_winner == int(state.p2.user_id)
                            else None
                        ),
                    }
                )
                self._finalized = True
            elif reason and not self._meta.get("terminal_reason"):
                self._meta["terminal_reason"] = str(reason)
            return self._snapshot_unlocked()

    def abort(
        self,
        engine: Any,
        *,
        reason: str,
        status: str = "aborted",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started_unlocked(engine)
            if self._finalized:
                return self._snapshot_unlocked()
            if metadata:
                self._merge_metadata_unlocked(metadata)
            self._aborted = True
            payload = self.finalize(
                engine,
                winner_user_id=None,
                status=str(status),
                reason=str(reason),
            )
            self._meta["aborted"] = True
            self._meta["abort_reason"] = str(reason)
            payload = self._snapshot_unlocked()
            return payload

    def _snapshot_unlocked(self) -> dict[str, Any]:
        meta = dict(self._meta)
        # Canonical storage has three files. Sparse control-plane transitions
        # live in meta so they survive DB persistence/materialization and can
        # explain otherwise intentional post_state -> pre_state differences.
        meta["control_events"] = self._control_events
        return deepcopy(
            {
                "schema_version": V5_STORAGE_SCHEMA,
                "visibility": V5_VISIBILITY,
                "meta": meta,
                "turns": self._turns,
                "actions": self._actions,
                "control_events": self._control_events,
                "counts": {
                    "turns": len(self._turns),
                    "actions": len(self._actions),
                    "accepted_actions": sum(
                        1 for row in self._actions if row.get("accepted") is True
                    ),
                    "rejected_actions": sum(
                        1 for row in self._actions if row.get("accepted") is False
                    ),
                    "training_labels": sum(
                        1
                        for row in self._actions
                        if row.get("is_training_label") is True
                    ),
                    "control_events": len(self._control_events),
                    "pending_actions": len(self._pending_actions),
                },
                "finalized": bool(self._finalized),
                "aborted": bool(self._aborted),
            }
        )


# Compact public name used by BattleEngine and production tests.
V5DatasetRecorder = InMemoryV5DatasetRecorder


__all__ = [
    "ACTION_SOURCES",
    "ACTION_SOURCE_ACTOR_TYPES",
    "CONTROL_SOURCES",
    "DECISION_SOURCES",
    "InMemoryV5DatasetRecorder",
    "TRAINING_ACTION_TYPES",
    "V5DatasetRecorder",
    "V5_POLICY_FAILURE_CODES",
    "V5_POLICY_FAILURE_PREFIX",
    "V5_STORAGE_SCHEMA",
    "V5_VISIBILITY",
    "canonical_actor_type",
    "v5_policy_failure_code",
    "v5_policy_failure_error",
    "v5_policy_failure_warning",
]
