"""Training-only auxiliary model interfaces for Extra-LR V5."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .v5_artifacts import AuxDatasetManifest, write_manifest_json

ASSEMBLER_LABEL = "extra-sublr-assembler-v1"
DESIRERER_LABEL = "extra-sublr-desirerer-v1"


@dataclass(frozen=True)
class AssemblerCandidate:
    deck_ids: list[int]
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass(frozen=True)
class DrawScore:
    card_id: int
    score: float


@dataclass(frozen=True)
class AssemblerDatasetRow:
    opponent_deck_ids: list[int]
    candidate_deck_ids: list[int]
    target_winrate: float
    source_run: str
    games: int
    confidence: float = 1.0

    def validate(self) -> "AssemblerDatasetRow":
        if not self.opponent_deck_ids:
            raise ValueError("opponent_deck_ids must not be empty")
        if not self.candidate_deck_ids:
            raise ValueError("candidate_deck_ids must not be empty")
        if not 0.0 <= float(self.target_winrate) <= 1.0:
            raise ValueError("target_winrate must be in [0, 1]")
        if int(self.games) <= 0:
            raise ValueError("games must be positive")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not str(self.source_run):
            raise ValueError("source_run must not be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblerDatasetRow":
        return cls(
            opponent_deck_ids=[int(item) for item in data["opponent_deck_ids"]],
            candidate_deck_ids=[int(item) for item in data["candidate_deck_ids"]],
            target_winrate=float(data["target_winrate"]),
            source_run=str(data["source_run"]),
            games=int(data["games"]),
            confidence=float(data.get("confidence", 1.0)),
        ).validate()


@dataclass(frozen=True)
class DesirererDatasetRow:
    state_summary: dict[str, Any]
    deck_ids: list[int]
    hand_ids: list[int]
    candidate_card_id: int
    target_next_turn_delta: float
    draw_assist_strength: float

    def validate(self) -> "DesirererDatasetRow":
        if not isinstance(self.state_summary, dict):
            raise ValueError("state_summary must be an object")
        if not self.deck_ids:
            raise ValueError("deck_ids must not be empty")
        if int(self.candidate_card_id) not in {int(card_id) for card_id in self.deck_ids}:
            raise ValueError("candidate_card_id must be present in deck_ids")
        if not 0.0 <= float(self.draw_assist_strength) <= 1.0:
            raise ValueError("draw_assist_strength must be in [0, 1]")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesirererDatasetRow":
        return cls(
            state_summary=dict(data["state_summary"]),
            deck_ids=[int(item) for item in data["deck_ids"]],
            hand_ids=[int(item) for item in data.get("hand_ids", [])],
            candidate_card_id=int(data["candidate_card_id"]),
            target_next_turn_delta=float(data["target_next_turn_delta"]),
            draw_assist_strength=float(data["draw_assist_strength"]),
        ).validate()


class DeckMatchupEvaluator:
    """Baseline evaluator for `extra-sublr-assembler-v1`.

    The public contract is model-like: score opponent deck + candidate bot deck.
    The current implementation is deterministic and lightweight so search and
    data plumbing can be tested before replacing the scorer with a learned model.
    """

    def score_candidate(self, opponent_deck_ids: Iterable[int], candidate_deck_ids: Iterable[int]) -> float:
        opponent = [int(card_id) for card_id in opponent_deck_ids]
        candidate = [int(card_id) for card_id in candidate_deck_ids]
        if not candidate:
            return 0.0
        opponent_mean = sum(opponent) / max(len(opponent), 1)
        candidate_mean = sum(candidate) / len(candidate)
        diversity = len(set(candidate)) / len(candidate)
        counter_pressure = 0.5 + (candidate_mean - opponent_mean) / 200.0
        return _clip01(0.75 * counter_pressure + 0.25 * diversity)

    def search_best(
        self,
        opponent_deck_ids: Iterable[int],
        candidates: Iterable[AssemblerCandidate],
    ) -> AssemblerCandidate:
        best: AssemblerCandidate | None = None
        best_score = -1.0
        for candidate in candidates:
            score = self.score_candidate(opponent_deck_ids, candidate.deck_ids)
            if score > best_score:
                best_score = score
                best = candidate
        if best is None:
            raise ValueError("candidates must not be empty")
        return AssemblerCandidate(
            deck_ids=list(best.deck_ids),
            metadata=dict(best.metadata),
            score=float(best_score),
        )


class DrawDesirerer:
    """Baseline scorer for `extra-sublr-desirerer-v1` draw assistance."""

    def score_draw_options(
        self,
        *,
        deck_ids: Iterable[int],
        hand_ids: Iterable[int],
        board_power_ratio: float,
        draw_assist_strength: float,
    ) -> list[DrawScore]:
        hand_set = {int(card_id) for card_id in hand_ids}
        pressure = 1.0 - _clip01(board_power_ratio)
        assist = _clip01(draw_assist_strength)
        scores: list[DrawScore] = []
        for card_id in deck_ids:
            cid = int(card_id)
            duplicate_penalty = 0.08 if cid in hand_set else 0.0
            card_pressure_value = min(cid / 50.0, 1.0)
            score = 0.35 + 0.45 * assist * card_pressure_value + 0.20 * pressure - duplicate_penalty
            scores.append(DrawScore(card_id=cid, score=_clip01(score)))
        return sorted(scores, key=lambda item: (-item.score, item.card_id))


class DrawAssistController:
    """Runtime draw-assist gate over the baseline desirerer scorer."""

    def __init__(self, desirerer: DrawDesirerer | None = None) -> None:
        self.desirerer = desirerer or DrawDesirerer()

    def choose_draw(
        self,
        *,
        deck_ids: Iterable[int],
        hand_ids: Iterable[int],
        board_power_ratio: float,
        draw_assist_enabled: bool,
        draw_assist_strength: float,
    ) -> dict[str, Any]:
        assist = _clip01(draw_assist_strength)
        scores = self.desirerer.score_draw_options(
            deck_ids=deck_ids,
            hand_ids=hand_ids,
            board_power_ratio=board_power_ratio,
            draw_assist_strength=assist,
        )
        ranked_options = [
            {
                "rank": idx,
                "card_id": int(item.card_id),
                "score": float(item.score),
            }
            for idx, item in enumerate(scores, start=1)
        ]
        selected_card_id = (
            int(ranked_options[0]["card_id"]) if bool(draw_assist_enabled) and assist > 0.0 and ranked_options else None
        )
        return {
            "selected_card_id": selected_card_id,
            "draw_assist_enabled": bool(draw_assist_enabled),
            "draw_assist_strength": assist,
            "board_power_ratio": float(board_power_ratio),
            "ranked_options": ranked_options,
        }


def build_assembler_rows_from_matchup_summaries(summaries: Iterable[dict[str, Any]]) -> list[AssemblerDatasetRow]:
    rows: list[AssemblerDatasetRow] = []
    for idx, summary in enumerate(summaries):
        if not isinstance(summary, dict):
            raise ValueError(f"matchup summary {idx} must be an object")
        games = _positive_int(_required(summary, "games"), f"matchup summary {idx} games")
        target_winrate = _matchup_winrate(summary, games, idx)
        source_run = str(
            summary.get("source_run")
            or summary.get("run_name")
            or summary.get("matchup_id")
            or f"matchup-summary-{idx:06d}"
        )
        row = AssemblerDatasetRow(
            opponent_deck_ids=_first_int_list(summary, ("opponent_deck_ids", "p2_deck_ids", "enemy_deck_ids")),
            candidate_deck_ids=_first_int_list(summary, ("candidate_deck_ids", "p1_deck_ids", "learner_deck_ids", "deck_ids")),
            target_winrate=target_winrate,
            source_run=source_run,
            games=games,
            confidence=float(summary.get("confidence", 1.0)),
        ).validate()
        rows.append(row)
    return rows


def build_assembler_rows_from_v5_trace(trace: dict[str, Any], *, source_run: str = "") -> list[AssemblerDatasetRow]:
    if not isinstance(trace, dict):
        raise ValueError("trace must be an object")
    initial = trace.get("initial")
    if not isinstance(initial, dict):
        raise ValueError("trace initial must be an object")
    state = initial.get("state")
    if not isinstance(state, dict):
        raise ValueError("trace initial.state must be an object")
    p1 = _player_state(state, "p1", 0)
    p2 = _player_state(state, "p2", 0)
    candidate_deck_ids = _player_card_pool_ids(p1, "trace initial p1")
    opponent_deck_ids = _player_card_pool_ids(p2, "trace initial p2")
    if not candidate_deck_ids or not opponent_deck_ids:
        return []
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trace steps must be a list")
    target_winrate = _trace_candidate_winrate(trace)
    return [
        AssemblerDatasetRow(
            opponent_deck_ids=opponent_deck_ids,
            candidate_deck_ids=candidate_deck_ids,
            target_winrate=target_winrate,
            source_run=str(source_run or initial.get("state_sha256", "v5-trace")),
            games=max(1, len(steps)),
            confidence=min(1.0, 0.35 + 0.02 * max(1, len(steps))),
        ).validate()
    ]


def build_desirerer_rows_from_v5_trace(trace: dict[str, Any], *, source_run: str = "") -> list[DesirererDatasetRow]:
    if not isinstance(trace, dict):
        raise ValueError("trace must be an object")
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trace steps must be a list")
    env_config = trace.get("env_config") if isinstance(trace.get("env_config"), dict) else {}
    draw_assist_strength = float(env_config.get("draw_assist_strength", 0.0) or 0.0)
    rows: list[DesirererDatasetRow] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"trace step {idx} must be an object")
        row = _desirerer_row_from_step(step, idx, source_run=source_run, draw_assist_strength=draw_assist_strength)
        if row is not None:
            rows.append(row)
    return rows


def save_assembler_dataset(rows: Iterable[AssemblerDatasetRow], path: str | Path) -> Path:
    return _save_jsonl((row.to_dict() for row in rows), path)


def save_assembler_dataset_with_manifest(
    rows: Iterable[AssemblerDatasetRow],
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    source_manifest_ids: Iterable[str] = (),
) -> tuple[Path, Path]:
    materialized = list(rows)
    out = save_assembler_dataset(materialized, dataset_path)
    manifest = AuxDatasetManifest(
        dataset_kind="assembler",
        dataset_path=out,
        rows=len(materialized),
        source_manifest_ids=tuple(str(item) for item in source_manifest_ids),
        label=ASSEMBLER_LABEL,
    )
    return out, write_manifest_json(manifest, manifest_path)


def load_assembler_dataset(path: str | Path) -> list[AssemblerDatasetRow]:
    return [AssemblerDatasetRow.from_dict(row) for row in _load_jsonl(path)]


def save_desirerer_dataset(rows: Iterable[DesirererDatasetRow], path: str | Path) -> Path:
    return _save_jsonl((row.to_dict() for row in rows), path)


def save_desirerer_dataset_with_manifest(
    rows: Iterable[DesirererDatasetRow],
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    source_manifest_ids: Iterable[str] = (),
) -> tuple[Path, Path]:
    materialized = list(rows)
    out = save_desirerer_dataset(materialized, dataset_path)
    manifest = AuxDatasetManifest(
        dataset_kind="desirerer",
        dataset_path=out,
        rows=len(materialized),
        source_manifest_ids=tuple(str(item) for item in source_manifest_ids),
        label=DESIRERER_LABEL,
    )
    return out, write_manifest_json(manifest, manifest_path)


def load_desirerer_dataset(path: str | Path) -> list[DesirererDatasetRow]:
    return [DesirererDatasetRow.from_dict(row) for row in _load_jsonl(path)]


def evaluate_assembler_baseline(rows: Iterable[AssemblerDatasetRow], evaluator: DeckMatchupEvaluator | None = None) -> dict[str, Any]:
    evaluator = evaluator or DeckMatchupEvaluator()
    abs_errors: list[float] = []
    for row in rows:
        row.validate()
        predicted = evaluator.score_candidate(row.opponent_deck_ids, row.candidate_deck_ids)
        abs_errors.append(abs(predicted - float(row.target_winrate)))
    return {
        "rows": len(abs_errors),
        "mae": 0.0 if not abs_errors else sum(abs_errors) / len(abs_errors),
    }


def evaluate_desirerer_baseline(rows: Iterable[DesirererDatasetRow], desirerer: DrawDesirerer | None = None) -> dict[str, Any]:
    desirerer = desirerer or DrawDesirerer()
    total = 0
    hits = 0
    for row in rows:
        row.validate()
        board_power_ratio = float(row.state_summary.get("board_power_ratio", 1.0) or 1.0)
        scores = desirerer.score_draw_options(
            deck_ids=row.deck_ids,
            hand_ids=row.hand_ids,
            board_power_ratio=board_power_ratio,
            draw_assist_strength=row.draw_assist_strength,
        )
        total += 1
        hits += int(bool(scores) and scores[0].card_id == int(row.candidate_card_id))
    return {
        "rows": total,
        "top_card_match_rate": 0.0 if total == 0 else hits / total,
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"{key} is required")
    return data[key]


def _positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _matchup_winrate(summary: dict[str, Any], games: int, idx: int) -> float:
    if "target_winrate" in summary:
        value = float(summary["target_winrate"])
    elif "target_value" in summary:
        value = float(summary["target_value"])
    elif "p1_winrate" in summary:
        value = float(summary["p1_winrate"])
    elif "winrate" in summary:
        value = float(summary["winrate"])
    elif "p1_wins" in summary:
        value = int(summary["p1_wins"]) / games
    else:
        raise ValueError(f"matchup summary {idx} winrate is required")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"matchup summary {idx} winrate must be in [0, 1]")
    return value


def _first_int_list(data: dict[str, Any], keys: tuple[str, ...]) -> list[int]:
    for key in keys:
        if key in data:
            values = data[key]
            if not isinstance(values, list | tuple):
                raise ValueError(f"{key} must be a list")
            return [int(item) for item in values]
    raise ValueError(f"{keys[0]} is required")


def _trace_candidate_winrate(trace: dict[str, Any]) -> float:
    initial_state = trace["initial"]["state"]
    final_state = _trace_final_state(trace) or initial_state
    initial_p1 = _player_state(initial_state, "p1", 0)
    initial_p2 = _player_state(initial_state, "p2", 0)
    final_p1 = _player_state(final_state, "p1", 0)
    final_p2 = _player_state(final_state, "p2", 0)
    initial_hp_delta = _hero_hp(initial_p1) - _hero_hp(initial_p2)
    final_hp_delta = _hero_hp(final_p1) - _hero_hp(final_p2)
    hp_delta = final_hp_delta - initial_hp_delta
    board_delta = _cards_power(final_p1.get("board", []), "trace final p1.board") - _cards_power(
        final_p2.get("board", []),
        "trace final p2.board",
    )
    reward_sum = 0.0
    for step in trace.get("steps", []):
        if isinstance(step, dict):
            reward_sum += float(step.get("reward", step.get("base_reward", 0.0)) or 0.0)
    return _clip01(0.5 + hp_delta / 80.0 + board_delta / 400.0 + reward_sum / 20.0)


def _trace_final_state(trace: dict[str, Any]) -> dict[str, Any] | None:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trace steps must be a list")
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        post = step.get("post")
        if isinstance(post, dict) and isinstance(post.get("state"), dict):
            return post["state"]
    return None


def _player_card_pool_ids(player: dict[str, Any], name: str) -> list[int]:
    ids: list[int] = []
    for zone in ("deck", "hand", "board", "graveyard"):
        cards = player.get(zone, [])
        ids.extend(_card_ids(cards, f"{name}.{zone}"))
    return ids


def _desirerer_row_from_step(
    step: dict[str, Any],
    idx: int,
    *,
    source_run: str,
    draw_assist_strength: float,
) -> DesirererDatasetRow | None:
    actor_id = int(step.get("acting_player_id", 0) or 0)
    if actor_id not in {1, 2}:
        raise ValueError(f"trace step {idx} acting_player_id must be 1 or 2")
    pre = step.get("pre")
    if not isinstance(pre, dict):
        raise ValueError(f"trace step {idx} pre must be an object")
    state = pre.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"trace step {idx} pre.state must be an object")
    me_key = "p1" if actor_id == 1 else "p2"
    enemy_key = "p2" if actor_id == 1 else "p1"
    me = _player_state(state, me_key, idx)
    enemy = _player_state(state, enemy_key, idx)
    deck_ids = _card_ids(me.get("deck", []), f"trace step {idx} {me_key}.deck")
    if not deck_ids:
        return None
    hand_ids = _card_ids(me.get("hand", []), f"trace step {idx} {me_key}.hand")
    my_board_power = _cards_power(me.get("board", []), f"trace step {idx} {me_key}.board")
    enemy_board_power = _cards_power(enemy.get("board", []), f"trace step {idx} {enemy_key}.board")
    components = step.get("reward_components_v5") if isinstance(step.get("reward_components_v5"), dict) else {}
    board_power_ratio = float(components.get("board_power_ratio", my_board_power / max(enemy_board_power, 1.0)) or 0.0)
    state_summary = {
        "actor_id": actor_id,
        "board_power_ratio": board_power_ratio,
        "enemy_board_power": enemy_board_power,
        "enemy_hand_count": len(_card_ids(enemy.get("hand", []), f"trace step {idx} {enemy_key}.hand")),
        "enemy_hero_hp": _hero_hp(enemy),
        "legal_action_count": len(pre.get("legal_ids", []) if isinstance(pre.get("legal_ids"), list) else []),
        "my_board_power": my_board_power,
        "my_hand_count": len(hand_ids),
        "my_hero_hp": _hero_hp(me),
        "source_run": str(source_run),
        "state_sha256": str(pre.get("state_sha256", "")),
        "step_t": int(step.get("t", idx)),
        "turn_number": int(state.get("turn_number", 0) or 0),
    }
    return DesirererDatasetRow(
        state_summary=state_summary,
        deck_ids=deck_ids,
        hand_ids=hand_ids,
        candidate_card_id=deck_ids[0],
        target_next_turn_delta=float(step.get("reward", step.get("base_reward", 0.0)) or 0.0),
        draw_assist_strength=draw_assist_strength,
    ).validate()


def _player_state(state: dict[str, Any], key: str, idx: int) -> dict[str, Any]:
    player = state.get(key)
    if not isinstance(player, dict):
        raise ValueError(f"trace step {idx} pre.state.{key} must be an object")
    return player


def _card_ids(cards: Any, name: str) -> list[int]:
    if not isinstance(cards, list):
        raise ValueError(f"{name} must be a list")
    out: list[int] = []
    for card in cards:
        if not isinstance(card, dict) or "card_id" not in card:
            raise ValueError(f"{name} entries must contain card_id")
        out.append(int(card["card_id"]))
    return out


def _cards_power(cards: Any, name: str) -> float:
    if not isinstance(cards, list):
        raise ValueError(f"{name} must be a list")
    total = 0.0
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError(f"{name} entries must be objects")
        total += max(0, int(card.get("attack", 0) or 0)) * max(0, int(card.get("hp", 0) or 0))
    return float(total)


def _hero_hp(player: dict[str, Any]) -> int:
    hero = player.get("hero")
    return int(hero.get("hp", 0) or 0) if isinstance(hero, dict) else 0


def _save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


__all__ = [
    "ASSEMBLER_LABEL",
    "AssemblerCandidate",
    "AssemblerDatasetRow",
    "DeckMatchupEvaluator",
    "DESIRERER_LABEL",
    "DesirererDatasetRow",
    "DrawAssistController",
    "DrawDesirerer",
    "DrawScore",
    "build_assembler_rows_from_matchup_summaries",
    "build_assembler_rows_from_v5_trace",
    "build_desirerer_rows_from_v5_trace",
    "evaluate_assembler_baseline",
    "evaluate_desirerer_baseline",
    "load_assembler_dataset",
    "load_desirerer_dataset",
    "save_assembler_dataset",
    "save_assembler_dataset_with_manifest",
    "save_desirerer_dataset",
    "save_desirerer_dataset_with_manifest",
]
