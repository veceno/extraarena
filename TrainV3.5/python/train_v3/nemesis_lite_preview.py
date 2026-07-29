"""Nemesis Lite Preview training contract.

Nemesis predicts a terminal three-way outcome from information available
before the first action:

* both exact initial decks (card ids and explicit effective levels);
* which side starts.

Catalog and ruleset identities are deliberately *not* model inputs.  They are
immutable compatibility gates recorded in the training and ONNX manifests.

The architecture is swap-equivariant by construction.  Both decks pass
through one shared, permutation-invariant encoder.  The p1/p2 advantage head
is an odd, bias-free network over ``deck_1 - deck_2`` and the starting-side
sign, while the draw head sees only swap-invariant features.  Swapping decks,
flipping the starter and exchanging the p1/p2 classes therefore produces the
same prediction up to floating-point roundoff.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from core.nemesis_dataset import (
    NEMESIS_EXPORT_FORMAT,
    NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME,
    NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME,
    NEMESIS_SCHEMA,
    validate_nemesis_record,
)


UNIFIED_ROW_SCHEMA = NEMESIS_SCHEMA
SYNTHETIC_ROW_SCHEMA = "extra_lr_nemesis_battle_v1"
TRAINING_MANIFEST_SCHEMA = "extra_lr_nemesis_lite_preview_training_v1"
MODEL_ID = "extra-lr-nemesis-lite-preview"
ARCHITECTURE_VERSION = "shared_deck_odd_advantage_v1"
CLASS_NAMES = ("p1_win", "draw", "p2_win")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SWAPPED_CLASS_INDEX = np.asarray([2, 1, 0], dtype=np.int64)


class NemesisRowExcluded(ValueError):
    """A valid unified row is explicitly ineligible for Nemesis Lite."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CatalogContract:
    path: str
    sha256: str
    ordered_card_ids: tuple[int, ...]
    max_card_id: int
    hero_card_ids: tuple[int, ...]
    max_levels: dict[int, int]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": Path(self.path).name,
            "sha256": self.sha256,
            "card_count": len(self.ordered_card_ids),
            "ordered_card_ids": list(self.ordered_card_ids),
            "max_card_id": self.max_card_id,
            "hero_card_ids": list(self.hero_card_ids),
            "level_policy": {
                str(card_id): int(maximum)
                for card_id, maximum in sorted(self.max_levels.items())
            },
        }


def load_catalog_contract(path: str | Path) -> CatalogContract:
    catalog_path = Path(path).expanduser().resolve()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{catalog_path}: card catalog must be a non-empty list")
    cards: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"{catalog_path}: card {index} must be an object")
        card_id = raw.get("id")
        if not isinstance(card_id, int) or isinstance(card_id, bool) or card_id <= 0:
            raise ValueError(f"{catalog_path}: card {index} has invalid id")
        if card_id in cards:
            raise ValueError(f"{catalog_path}: duplicate card id {card_id}")
        cards[card_id] = raw
    ordered = tuple(sorted(cards))
    heroes = tuple(
        card_id
        for card_id in ordered
        if str(cards[card_id].get("card_type")) == "hero"
    )
    if not heroes:
        raise ValueError(f"{catalog_path}: catalog has no hero cards")
    max_levels = {
        card_id: 2 if bool(cards[card_id].get("simplified_levelup")) else 10
        for card_id in ordered
    }
    return CatalogContract(
        path=str(catalog_path),
        sha256=sha256_file(catalog_path),
        ordered_card_ids=ordered,
        max_card_id=max(ordered),
        hero_card_ids=heroes,
        max_levels=max_levels,
    )


@dataclass(frozen=True)
class NemesisBattleRow:
    battle_id: str
    p1_card_ids: tuple[int, ...]
    p1_levels: tuple[int, ...]
    p2_card_ids: tuple[int, ...]
    p2_levels: tuple[int, ...]
    starting_side: int
    target: int
    sample_weight: float
    catalog_verified: bool
    checkpoint_mix: tuple[str, ...]
    matchup_key: str
    source_matchup_group_id: str
    source_path: str
    source_line: int

    def swapped(self) -> "NemesisBattleRow":
        return NemesisBattleRow(
            battle_id=f"{self.battle_id}::swap",
            p1_card_ids=self.p2_card_ids,
            p1_levels=self.p2_levels,
            p2_card_ids=self.p1_card_ids,
            p2_levels=self.p1_levels,
            starting_side=1 - self.starting_side,
            target=int(SWAPPED_CLASS_INDEX[self.target]),
            sample_weight=self.sample_weight,
            catalog_verified=self.catalog_verified,
            checkpoint_mix=self.checkpoint_mix,
            matchup_key=self.matchup_key,
            source_matchup_group_id=self.source_matchup_group_id,
            source_path=self.source_path,
            source_line=self.source_line,
        )


def _deck_signature(
    card_ids: Sequence[int],
    levels: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(zip(card_ids, levels, strict=True)))


def matchup_key_for_decks(
    p1_card_ids: Sequence[int],
    p1_levels: Sequence[int],
    p2_card_ids: Sequence[int],
    p2_levels: Sequence[int],
) -> str:
    """Hash an unordered exact deck pair for leakage-safe splitting."""

    signatures = sorted(
        [
            _deck_signature(p1_card_ids, p1_levels),
            _deck_signature(p2_card_ids, p2_levels),
        ]
    )
    return _canonical_json_sha256(signatures)


def _parse_deck(
    value: Any,
    *,
    field: str,
    deck_size: int,
    catalog: CatalogContract,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, list) or len(value) != deck_size:
        raise ValueError(f"{field} must contain exactly {deck_size} cards")
    card_ids: list[int] = []
    levels: list[int] = []
    for index, card in enumerate(value):
        if not isinstance(card, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        card_id = card.get("card_id")
        level = card.get("level")
        if (
            not isinstance(card_id, int)
            or isinstance(card_id, bool)
            or card_id not in catalog.max_levels
        ):
            raise ValueError(f"{field}[{index}].card_id is outside the catalog")
        maximum = catalog.max_levels[card_id]
        if (
            not isinstance(level, int)
            or isinstance(level, bool)
            or not 1 <= level <= maximum
        ):
            raise ValueError(
                f"{field}[{index}].level must be in [1, {maximum}]"
            )
        card_ids.append(card_id)
        levels.append(level)
    if len(set(card_ids)) != len(card_ids):
        raise ValueError(f"{field} contains duplicate card ids")
    hero_count = sum(card_id in catalog.hero_card_ids for card_id in card_ids)
    if hero_count != 1:
        raise ValueError(f"{field} must contain exactly one hero")
    return tuple(card_ids), tuple(levels)


def _terminal_target(
    value: Any,
    *,
    field: str,
    winner_field: str,
) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    status = str(value.get("status") or "")
    winner_side = value.get(winner_field)
    if status == "p1_win":
        expected_side = "p1"
        label = "p1_win"
    elif status == "p2_win":
        expected_side = "p2"
        label = "p2_win"
    elif status in {"draw", "stalemate"}:
        expected_side = None
        label = "draw"
    else:
        raise ValueError(f"{field}.status is not terminal")
    if winner_side != expected_side:
        raise ValueError(
            f"{field}.winner_side does not match status={status!r}"
        )
    return CLASS_INDEX[label]


def parse_unified_row(
    raw: Mapping[str, Any],
    *,
    catalog: CatalogContract,
    ruleset: str,
    deck_size: int = 9,
    source_path: str = "<memory>",
    source_line: int = 1,
) -> NemesisBattleRow:
    schema = raw.get("schema_version", raw.get("schema"))
    if schema not in {UNIFIED_ROW_SCHEMA, SYNTHETIC_ROW_SCHEMA}:
        raise ValueError(
            f"{source_path}:{source_line}: unsupported nemesis schema {schema!r}"
        )
    if schema == UNIFIED_ROW_SCHEMA:
        canonical = validate_nemesis_record(raw, require_terminal=True)
        base = canonical["features"]["base"]
        seats = base["seats"]
        quality = canonical["quality"]
        exclusion_reasons = [
            str(reason)
            for reason in quality["exclusion_reasons"]
            if str(reason).strip()
        ]
        if not quality["eligible_lite"] or float(quality["sample_weight"]) <= 0.0:
            reasons = ",".join(exclusion_reasons) or "eligible_lite=false"
            raise NemesisRowExcluded(
                f"{source_path}:{source_line}: excluded from Nemesis Lite "
                f"({reasons})"
            )
        flattened = {
            "battle_id": canonical["battle_id"],
            "catalog_hash": base["catalog_hash"],
            "catalog_available": base["catalog_available"],
            "ruleset": base["ruleset"],
            "starting_player": base["starting_player"],
            "p1_deck": seats["p1"]["initial_deck"],
            "p2_deck": seats["p2"]["initial_deck"],
            "terminal": canonical["label"],
            "sample_weight": quality["sample_weight"],
            "matchup_group_id": canonical["provenance"]["split_group"],
            "checkpoint_mix": canonical["provenance"]["checkpoint_mix"],
        }
        winner_field = "winner_seat"
    else:
        flattened = dict(raw)
        flattened["catalog_available"] = True
        winner_field = "winner_side"
    battle_id = flattened.get("battle_id")
    if not isinstance(battle_id, str) or not battle_id.strip():
        raise ValueError(f"{source_path}:{source_line}: battle_id is required")
    row_catalog_hash = flattened.get(
        "catalog_hash",
        flattened.get("catalog_sha256"),
    )
    catalog_verified = bool(flattened.get("catalog_available"))
    if catalog_verified and row_catalog_hash != catalog.sha256:
        raise ValueError(
            f"{source_path}:{source_line}: catalog hash mismatch"
        )
    if not catalog_verified and row_catalog_hash not in {None, ""}:
        raise ValueError(
            f"{source_path}:{source_line}: catalog availability/hash mismatch"
        )
    row_ruleset = flattened.get(
        "ruleset",
        flattened.get("ruleset_hash"),
    )
    if row_ruleset != ruleset:
        raise ValueError(
            f"{source_path}:{source_line}: ruleset mismatch"
        )
    starting_player = flattened.get(
        "starting_player",
        flattened.get("starting_side"),
    )
    if starting_player not in {"p1", "p2"}:
        raise ValueError(
            f"{source_path}:{source_line}: starting_player must be p1 or p2"
        )
    p1_card_ids, p1_levels = _parse_deck(
        flattened.get("p1_deck"),
        field=f"{source_path}:{source_line}:p1_deck",
        deck_size=deck_size,
        catalog=catalog,
    )
    p2_card_ids, p2_levels = _parse_deck(
        flattened.get("p2_deck"),
        field=f"{source_path}:{source_line}:p2_deck",
        deck_size=deck_size,
        catalog=catalog,
    )
    target = _terminal_target(
        flattened.get("terminal"),
        field=f"{source_path}:{source_line}:terminal",
        winner_field=winner_field,
    )
    sample_weight = flattened.get("sample_weight", 1.0)
    if (
        isinstance(sample_weight, bool)
        or not isinstance(sample_weight, (int, float))
        or not math.isfinite(float(sample_weight))
        or float(sample_weight) <= 0.0
    ):
        raise ValueError(
            f"{source_path}:{source_line}: sample_weight must be positive"
        )
    raw_checkpoint_mix = flattened.get("checkpoint_mix") or []
    if not isinstance(raw_checkpoint_mix, list):
        raise ValueError(
            f"{source_path}:{source_line}: checkpoint_mix must be a list"
        )
    checkpoint_mix = tuple(
        sorted(
            {
                str(checkpoint).strip()
                for checkpoint in raw_checkpoint_mix
                if str(checkpoint).strip()
            }
        )
    )
    return NemesisBattleRow(
        battle_id=battle_id.strip(),
        p1_card_ids=p1_card_ids,
        p1_levels=p1_levels,
        p2_card_ids=p2_card_ids,
        p2_levels=p2_levels,
        starting_side=1 if starting_player == "p1" else 0,
        target=target,
        sample_weight=float(sample_weight),
        catalog_verified=catalog_verified,
        checkpoint_mix=checkpoint_mix,
        matchup_key=matchup_key_for_decks(
            p1_card_ids,
            p1_levels,
            p2_card_ids,
            p2_levels,
        ),
        source_matchup_group_id=str(
            flattened.get("matchup_group_id") or ""
        ),
        source_path=str(source_path),
        source_line=int(source_line),
    )


def load_unified_jsonl(
    paths: Sequence[str | Path],
    *,
    catalog: CatalogContract,
    ruleset: str,
    deck_size: int = 9,
) -> tuple[list[NemesisBattleRow], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one unified nemesis JSONL path is required")
    rows: list[NemesisBattleRow] = []
    sources: list[dict[str, Any]] = []
    battle_ids: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        before = len(rows)
        expected_battles: int | None = None
        source_battles = 0
        excluded_battles = 0
        catalog_unverified_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                if raw.get("record_type") == "header":
                    if line_number != 1 or expected_battles is not None:
                        raise ValueError(f"{path}:{line_number}: invalid header position")
                    if raw.get("format") != NEMESIS_EXPORT_FORMAT:
                        raise ValueError(f"{path}:{line_number}: unsupported export format")
                    if (
                        raw.get("identity_scheme")
                        != "side_pseudonyms_p1_1_p2_2"
                        or raw.get("include_players") is not False
                        or raw.get("record_id_scheme")
                        != NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
                        or raw.get("player_group_scheme")
                        != NEMESIS_PSEUDONYMIZED_PLAYER_GROUP_SCHEME
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: unsafe privacy header"
                        )
                    expected_battles = int(raw.get("battle_count") or 0)
                    continue
                if raw.get("record_type") == "battle":
                    raw = {
                        key: item
                        for key, item in raw.items()
                        if key != "record_type"
                    }
                source_battles += 1
                try:
                    row = parse_unified_row(
                        raw,
                        catalog=catalog,
                        ruleset=ruleset,
                        deck_size=deck_size,
                        source_path=str(path),
                        source_line=line_number,
                    )
                except NemesisRowExcluded:
                    excluded_battles += 1
                    continue
                if row.battle_id in battle_ids:
                    raise ValueError(f"{path}:{line_number}: duplicate battle_id")
                battle_ids.add(row.battle_id)
                rows.append(row)
                if not row.catalog_verified:
                    catalog_unverified_rows += 1
        if (
            expected_battles is not None
            and expected_battles != source_battles
        ):
            raise ValueError(
                f"{path}: header battle_count={expected_battles}, "
                f"loaded={source_battles}"
            )
        sources.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(rows) - before,
                "records": source_battles,
                "excluded_rows": excluded_battles,
                "catalog_unverified_rows": catalog_unverified_rows,
            }
        )
    if not rows:
        raise ValueError("no unified nemesis rows loaded")
    return rows, sources


@dataclass(frozen=True)
class GroupedSplit:
    train: tuple[NemesisBattleRow, ...]
    validation: tuple[NemesisBattleRow, ...]
    test: tuple[NemesisBattleRow, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        assignments = {
            "train": list(self.train_groups),
            "validation": list(self.validation_groups),
            "test": list(self.test_groups),
        }
        return {
            "method": "unordered_exact_decks_with_levels_group_holdout_v1",
            "rows": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "groups": {
                "train": len(self.train_groups),
                "validation": len(self.validation_groups),
                "test": len(self.test_groups),
            },
            "assignment_sha256": _canonical_json_sha256(assignments),
            "leakage_free": not (
                set(self.train_groups) & set(self.validation_groups)
                or set(self.train_groups) & set(self.test_groups)
                or set(self.validation_groups) & set(self.test_groups)
            ),
        }


def grouped_matchup_split(
    rows: Sequence[NemesisBattleRow],
    *,
    seed: int = 20260728,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> GroupedSplit:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be in (0, 0.5)")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation+test fractions leave too little training data")
    grouped: dict[str, list[NemesisBattleRow]] = defaultdict(list)
    for row in rows:
        grouped[row.matchup_key].append(row)
    group_ids = sorted(grouped)
    if len(group_ids) < 3:
        raise ValueError(
            "grouped holdout requires at least 3 unique deck matchups"
        )
    generator = np.random.default_rng(int(seed))
    shuffled = [group_ids[int(index)] for index in generator.permutation(len(group_ids))]
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    while validation_count + test_count >= len(shuffled):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    validation_groups = tuple(sorted(shuffled[:validation_count]))
    test_groups = tuple(
        sorted(shuffled[validation_count : validation_count + test_count])
    )
    train_groups = tuple(
        sorted(shuffled[validation_count + test_count :])
    )

    def materialize(groups: Iterable[str]) -> tuple[NemesisBattleRow, ...]:
        return tuple(
            row
            for group in groups
            for row in grouped[group]
        )

    split = GroupedSplit(
        train=materialize(train_groups),
        validation=materialize(validation_groups),
        test=materialize(test_groups),
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )
    if not split.manifest()["leakage_free"]:
        raise AssertionError("matchup groups leaked across splits")
    return split


@dataclass(frozen=True)
class ModelConfig:
    max_card_id: int
    deck_size: int = 9
    embedding_dim: int = 24
    deck_hidden_dim: int = 48
    deck_output_dim: int = 32
    outcome_hidden_dim: int = 48

    def __post_init__(self) -> None:
        for field in (
            "max_card_id",
            "deck_size",
            "embedding_dim",
            "deck_hidden_dim",
            "deck_output_dim",
            "outcome_hidden_dim",
        ):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")


class SharedDeckEncoder(nn.Module):
    """Permutation-invariant card+level encoder shared by both seats."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.max_card_id + 1,
            config.embedding_dim,
            padding_idx=0,
        )
        self.card_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim + 1, config.deck_hidden_dim),
            nn.GELU(),
            nn.Linear(config.deck_hidden_dim, config.deck_output_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        mask = card_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(card_ids)
        normalized_level = levels.to(embedded.dtype).unsqueeze(-1) / 10.0
        encoded = self.card_mlp(
            torch.cat([embedded, normalized_level], dim=-1)
        )
        encoded = encoded * mask.to(encoded.dtype)
        count = mask.sum(dim=1).clamp_min(1).to(encoded.dtype)
        return encoded.sum(dim=1) / count


class NemesisLitePreviewModel(nn.Module):
    """Exactly swap-consistent three-class matchup predictor."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.deck_encoder = SharedDeckEncoder(config)
        odd_dim = config.deck_output_dim + 1
        self.advantage_head = nn.Sequential(
            nn.Linear(odd_dim, config.outcome_hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(config.outcome_hidden_dim, 1, bias=False),
        )
        symmetric_dim = config.deck_output_dim * 3
        self.draw_head = nn.Sequential(
            nn.Linear(symmetric_dim, config.outcome_hidden_dim),
            nn.GELU(),
            nn.Linear(config.outcome_hidden_dim, 1),
        )

    def forward(
        self,
        p1_card_ids: torch.Tensor,
        p1_levels: torch.Tensor,
        p2_card_ids: torch.Tensor,
        p2_levels: torch.Tensor,
        starting_side: torch.Tensor,
    ) -> torch.Tensor:
        p1 = self.deck_encoder(p1_card_ids, p1_levels)
        p2 = self.deck_encoder(p2_card_ids, p2_levels)
        difference = p1 - p2
        starter = starting_side.reshape(-1, 1).to(p1.dtype) * 2.0 - 1.0
        advantage = self.advantage_head(
            torch.cat([difference, starter], dim=-1)
        )
        symmetric = torch.cat(
            [
                p1 + p2,
                torch.abs(difference),
                difference * starter,
            ],
            dim=-1,
        )
        draw = self.draw_head(symmetric)
        return torch.cat([0.5 * advantage, draw, -0.5 * advantage], dim=-1)


def _rows_to_numpy(
    rows: Sequence[NemesisBattleRow],
) -> dict[str, np.ndarray]:
    return {
        "p1_card_ids": np.asarray(
            [row.p1_card_ids for row in rows],
            dtype=np.int64,
        ),
        "p1_levels": np.asarray(
            [row.p1_levels for row in rows],
            dtype=np.int64,
        ),
        "p2_card_ids": np.asarray(
            [row.p2_card_ids for row in rows],
            dtype=np.int64,
        ),
        "p2_levels": np.asarray(
            [row.p2_levels for row in rows],
            dtype=np.int64,
        ),
        "starting_side": np.asarray(
            [[row.starting_side] for row in rows],
            dtype=np.int64,
        ),
        "target": np.asarray([row.target for row in rows], dtype=np.int64),
        "sample_weight": np.asarray(
            [row.sample_weight for row in rows],
            dtype=np.float32,
        ),
    }


def _torch_batch(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.ascontiguousarray(value[indices])).to(device)
        for name, value in arrays.items()
    }


def predict_probabilities(
    model: NemesisLitePreviewModel,
    rows: Sequence[NemesisBattleRow],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    if not rows:
        return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
    target_device = torch.device(device)
    arrays = _rows_to_numpy(rows)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            indices = np.arange(start, min(len(rows), start + int(batch_size)))
            batch = _torch_batch(arrays, indices, device=target_device)
            logits = model(
                batch["p1_card_ids"],
                batch["p1_levels"],
                batch["p2_card_ids"],
                batch["p2_levels"],
                batch["starting_side"],
            )
            outputs.append(
                torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            )
    return np.concatenate(outputs, axis=0)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    ece_bins: int = 15,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (targets.size, len(CLASS_NAMES)):
        raise ValueError("probabilities must have shape [rows, 3]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    row_sums = probabilities.sum(axis=1)
    if np.any(probabilities < 0.0) or not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError("probabilities must be normalized")
    if np.any((targets < 0) | (targets >= len(CLASS_NAMES))):
        raise ValueError("targets contain an unknown class")
    weights = (
        np.ones(targets.size, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if weights.shape != targets.shape or np.any(weights <= 0.0):
        raise ValueError("sample_weight must be positive and row-aligned")
    weight_sum = float(weights.sum())
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    chosen = np.clip(
        probabilities[np.arange(targets.size), targets],
        1.0e-12,
        1.0,
    )
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[targets]
    brier_per_row = np.square(probabilities - one_hot).sum(axis=1)
    confidence = probabilities.max(axis=1)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for index in range(int(ece_bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower)
            & (
                confidence <= upper
                if index == int(ece_bins) - 1
                else confidence < upper
            )
        )
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        accuracy = float(np.average(correct[mask], weights=weights[mask]))
        mean_confidence = float(
            np.average(confidence[mask], weights=weights[mask])
        )
        ece += (bin_weight / weight_sum) * abs(accuracy - mean_confidence)
    return {
        "rows": int(targets.size),
        "accuracy": float(np.average(correct, weights=weights)),
        "logloss": float(np.average(-np.log(chosen), weights=weights)),
        "brier": float(np.average(brier_per_row, weights=weights)),
        "ece": float(ece),
        "ece_bins": int(ece_bins),
    }


def class_prevalence_report(
    rows: Sequence[NemesisBattleRow],
) -> dict[str, Any]:
    """Describe observed labels without claiming population-level rarity."""

    if not rows:
        counts = {name: 0 for name in CLASS_NAMES}
        return {
            "rows": 0,
            "total_sample_weight": 0.0,
            "class_counts": counts,
            "class_row_prevalence": {name: None for name in CLASS_NAMES},
            "class_weighted_counts": {
                name: 0.0 for name in CLASS_NAMES
            },
            "class_weighted_prevalence": {
                name: None for name in CLASS_NAMES
            },
            "draw_observation": {
                "count": 0,
                "row_prevalence": None,
                "weighted_prevalence": None,
                "note": (
                    "descriptive split frequency only; no rarity or "
                    "calibration claim"
                ),
            },
        }
    arrays = _rows_to_numpy(rows)
    targets = arrays["target"]
    weights = arrays["sample_weight"].astype(np.float64)
    raw_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    weighted_counts = np.bincount(
        targets,
        weights=weights,
        minlength=len(CLASS_NAMES),
    )
    rows_count = int(len(rows))
    total_weight = float(weights.sum())
    counts = {
        name: int(raw_counts[index])
        for index, name in enumerate(CLASS_NAMES)
    }
    row_prevalence = {
        name: float(raw_counts[index] / rows_count)
        for index, name in enumerate(CLASS_NAMES)
    }
    weighted_count_map = {
        name: float(weighted_counts[index])
        for index, name in enumerate(CLASS_NAMES)
    }
    weighted_prevalence = {
        name: float(weighted_counts[index] / total_weight)
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "rows": rows_count,
        "total_sample_weight": total_weight,
        "class_counts": counts,
        "class_row_prevalence": row_prevalence,
        "class_weighted_counts": weighted_count_map,
        "class_weighted_prevalence": weighted_prevalence,
        "draw_observation": {
            "count": counts["draw"],
            "row_prevalence": row_prevalence["draw"],
            "weighted_prevalence": weighted_prevalence["draw"],
            "note": (
                "descriptive split frequency only; no rarity or "
                "calibration claim"
            ),
        },
    }


def _empirical_class_distribution(
    rows: Sequence[NemesisBattleRow],
) -> np.ndarray:
    if not rows:
        raise ValueError("empirical distribution requires at least one row")
    arrays = _rows_to_numpy(rows)
    weighted_counts = np.bincount(
        arrays["target"],
        weights=arrays["sample_weight"].astype(np.float64),
        minlength=len(CLASS_NAMES),
    )
    return weighted_counts / weighted_counts.sum()


def _evaluate_fixed_probabilities(
    rows: Sequence[NemesisBattleRow],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    arrays = _rows_to_numpy(rows)
    return {
        **classification_metrics(
            arrays["target"],
            probabilities,
            sample_weight=arrays["sample_weight"],
        ),
        "observed_labels": class_prevalence_report(rows),
    }


def evaluate_test_baselines(
    train_rows: Sequence[NemesisBattleRow],
    test_rows: Sequence[NemesisBattleRow],
) -> dict[str, Any]:
    """Fit label-only baselines on train and evaluate once on grouped test."""

    if not train_rows or not test_rows:
        raise ValueError("baseline evaluation requires non-empty train and test")
    train_prior = _empirical_class_distribution(train_rows)
    majority_probabilities = np.repeat(
        train_prior.reshape(1, -1),
        len(test_rows),
        axis=0,
    )

    starter_priors: dict[int, np.ndarray] = {}
    starter_fallback: dict[int, bool] = {}
    for starter in (0, 1):
        matching = [
            row for row in train_rows if row.starting_side == starter
        ]
        starter_fallback[starter] = not matching
        starter_priors[starter] = (
            _empirical_class_distribution(matching)
            if matching
            else train_prior
        )
    starter_probabilities = np.stack(
        [starter_priors[row.starting_side] for row in test_rows]
    )

    def distribution_payload(values: np.ndarray) -> dict[str, float]:
        return {
            name: float(values[index])
            for index, name in enumerate(CLASS_NAMES)
        }

    return {
        "fit_scope": "grouped_train_only",
        "evaluation_scope": "grouped_test_only",
        "majority_class": {
            "kind": "constant_train_empirical_class_prior",
            "predicted_class": CLASS_NAMES[int(np.argmax(train_prior))],
            "train_probabilities": distribution_payload(train_prior),
            "metrics": _evaluate_fixed_probabilities(
                test_rows,
                majority_probabilities,
            ),
        },
        "starter_only_empirical": {
            "kind": "train_empirical_outcome_prior_conditioned_on_starter",
            "starting_side_values": {"p2": 0, "p1": 1},
            "train_probabilities": {
                "p2_starts": distribution_payload(starter_priors[0]),
                "p1_starts": distribution_payload(starter_priors[1]),
            },
            "fallback_to_global_prior": {
                "p2_starts": starter_fallback[0],
                "p1_starts": starter_fallback[1],
            },
            "metrics": _evaluate_fixed_probabilities(
                test_rows,
                starter_probabilities,
            ),
        },
    }


def swap_consistency_metrics(
    model: NemesisLitePreviewModel,
    rows: Sequence[NemesisBattleRow],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    original = predict_probabilities(model, rows, device=device)
    swapped = predict_probabilities(
        model,
        [row.swapped() for row in rows],
        device=device,
    )
    restored = swapped[:, SWAPPED_CLASS_INDEX]
    error = np.abs(original - restored)
    return {
        "rows": len(rows),
        "mean_abs": float(error.mean()) if error.size else 0.0,
        "max_abs": float(error.max()) if error.size else 0.0,
    }


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260728
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 12
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    device: str = "cpu"
    embedding_dim: int = 24
    deck_hidden_dim: int = 48
    deck_output_dim: int = 32
    outcome_hidden_dim: int = 48

    def __post_init__(self) -> None:
        for field in (
            "epochs",
            "batch_size",
            "patience",
            "embedding_dim",
            "deck_hidden_dim",
            "deck_output_dim",
            "outcome_hidden_dim",
        ):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        for field in ("learning_rate", "weight_decay"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")


def _evaluate_rows(
    model: NemesisLitePreviewModel,
    rows: Sequence[NemesisBattleRow],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    arrays = _rows_to_numpy(rows)
    probabilities = predict_probabilities(model, rows, device=device)
    return {
        **classification_metrics(
            arrays["target"],
            probabilities,
            sample_weight=arrays["sample_weight"],
        ),
        "swap_consistency": swap_consistency_metrics(
            model,
            rows,
            device=device,
        ),
        "observed_labels": class_prevalence_report(rows),
    }


def _test_breakdown_metrics(
    model: NemesisLitePreviewModel,
    rows: Sequence[NemesisBattleRow],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    by_source: dict[str, list[NemesisBattleRow]] = defaultdict(list)
    by_checkpoint: dict[tuple[str, ...], list[NemesisBattleRow]] = defaultdict(
        list
    )
    by_source_checkpoint: dict[
        tuple[str, tuple[str, ...]],
        list[NemesisBattleRow],
    ] = defaultdict(list)
    for row in rows:
        source = row.source_path
        checkpoints = row.checkpoint_mix or ("<unspecified>",)
        by_source[source].append(row)
        by_checkpoint[checkpoints].append(row)
        by_source_checkpoint[(source, checkpoints)].append(row)

    def source_payload(source: str) -> dict[str, str | None]:
        path = Path(source)
        return {
            "source": path.name if source != "<memory>" else source,
            "source_sha256": (
                sha256_file(path)
                if source != "<memory>" and path.is_file()
                else None
            ),
        }

    return {
        "scope": "grouped_test_only",
        "by_source": [
            {
                **source_payload(source),
                "metrics": _evaluate_rows(model, grouped, device=device),
            }
            for source, grouped in sorted(by_source.items())
        ],
        "by_checkpoint_mix": [
            {
                "checkpoint_mix": list(checkpoints),
                "metrics": _evaluate_rows(model, grouped, device=device),
            }
            for checkpoints, grouped in sorted(by_checkpoint.items())
        ],
        "by_source_checkpoint": [
            {
                **source_payload(source),
                "checkpoint_mix": list(checkpoints),
                "metrics": _evaluate_rows(model, grouped, device=device),
            }
            for (source, checkpoints), grouped in sorted(
                by_source_checkpoint.items()
            )
        ],
        "small_group_note": (
            "breakdowns are descriptive; small row counts are not evidence "
            "of reliable subgroup performance"
        ),
    }


def train_model(
    rows: Sequence[NemesisBattleRow],
    *,
    catalog: CatalogContract,
    deck_size: int = 9,
    config: TrainingConfig = TrainingConfig(),
) -> tuple[NemesisLitePreviewModel, GroupedSplit, dict[str, Any]]:
    split = grouped_matchup_split(
        rows,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train_rows = tuple(
        item
        for row in split.train
        for item in (row, row.swapped())
    )
    if not train_rows:
        raise ValueError("training split is empty")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model_config = ModelConfig(
        max_card_id=catalog.max_card_id,
        deck_size=deck_size,
        embedding_dim=config.embedding_dim,
        deck_hidden_dim=config.deck_hidden_dim,
        deck_output_dim=config.deck_output_dim,
        outcome_hidden_dim=config.outcome_hidden_dim,
    )
    model = NemesisLitePreviewModel(model_config)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    arrays = _rows_to_numpy(train_rows)
    class_counts = np.bincount(
        arrays["target"],
        minlength=len(CLASS_NAMES),
    ).astype(np.float64)
    class_weight = np.sqrt(
        class_counts.sum() / (len(CLASS_NAMES) * np.maximum(class_counts, 1.0))
    )
    class_weight_tensor = torch.from_numpy(class_weight.astype(np.float32)).to(
        device
    )
    best_logloss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    permutation_generator = np.random.default_rng(config.seed)

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = permutation_generator.permutation(len(train_rows))
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = _torch_batch(arrays, indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["p1_card_ids"],
                batch["p1_levels"],
                batch["p2_card_ids"],
                batch["p2_levels"],
                batch["starting_side"],
            )
            per_row = nn.functional.cross_entropy(
                logits,
                batch["target"],
                weight=class_weight_tensor,
                reduction="none",
            )
            row_weight = batch["sample_weight"]
            loss = torch.sum(per_row * row_weight) / torch.sum(row_weight)
            loss.backward()
            optimizer.step()
            batch_weight = float(row_weight.sum().detach().cpu())
            weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight

        validation = _evaluate_rows(
            model,
            split.validation,
            device=device,
        )
        validation_logloss = float(validation["logloss"])
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": weighted_loss_sum / max(weight_sum, 1.0),
                "validation_logloss": validation_logloss,
            }
        )
        if validation_logloss < best_logloss - 1.0e-7:
            best_logloss = validation_logloss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation model")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    report = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "selection_metric": "validation_logloss",
        "class_weight": {
            name: float(class_weight[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "swap_augmentation": "train_rows_doubled_with_deck_and_starter_swap",
        "history": history,
        "metrics": {
            "train_unaugmented": _evaluate_rows(
                model,
                split.train,
                device=device,
            ),
            "validation": _evaluate_rows(
                model,
                split.validation,
                device=device,
            ),
            "test": _evaluate_rows(
                model,
                split.test,
                device=device,
            ),
            "test_baselines": evaluate_test_baselines(
                split.train,
                split.test,
            ),
            "test_breakdowns": _test_breakdown_metrics(
                model,
                split.test,
                device=device,
            ),
            "observed_class_prevalence": {
                "all": class_prevalence_report(rows),
                "train": class_prevalence_report(split.train),
                "validation": class_prevalence_report(split.validation),
                "test": class_prevalence_report(split.test),
                "interpretation": (
                    "descriptive counts and frequencies only; draw frequency "
                    "does not by itself establish draw calibration"
                ),
            },
        },
    }
    model.to("cpu")
    return model, split, report


def save_model_artifact(
    model: NemesisLitePreviewModel,
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"state::{name}": tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez_compressed(output, **arrays)
    payload = dict(manifest)
    payload.update(
        {
            "schema": TRAINING_MANIFEST_SCHEMA,
            "model_id": MODEL_ID,
            "artifact": output.name,
            "artifact_sha256": sha256_file(output),
            "architecture": {
                "version": ARCHITECTURE_VERSION,
                "config": asdict(model.config),
                "shared_deck_encoder": True,
                "deck_order": "permutation_invariant",
                "swap_equivariance": (
                    "swap decks, flip starting_side, exchange p1/p2 classes"
                ),
            },
            "input_contract": {
                "p1_card_ids": [None, model.config.deck_size],
                "p1_levels": [None, model.config.deck_size],
                "p2_card_ids": [None, model.config.deck_size],
                "p2_levels": [None, model.config.deck_size],
                "starting_side": [None, 1],
                "starting_side_values": {"p2": 0, "p1": 1},
                "dtype": "int64",
            },
            "output_contract": {
                "class_order": list(CLASS_NAMES),
                "logits": [None, len(CLASS_NAMES)],
                "probabilities": [None, len(CLASS_NAMES)],
            },
        }
    )
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, sidecar


def load_model_artifact(
    path: str | Path,
) -> tuple[NemesisLitePreviewModel, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    sidecar = source.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema") != TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"{sidecar}: unsupported training manifest schema")
    if metadata.get("artifact_sha256") != sha256_file(source):
        raise ValueError(f"{source}: artifact sha256 mismatch")
    architecture = metadata.get("architecture") or {}
    if architecture.get("version") != ARCHITECTURE_VERSION:
        raise ValueError(f"{sidecar}: unsupported architecture")
    raw_config = architecture.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"{sidecar}: missing model config")
    model = NemesisLitePreviewModel(ModelConfig(**raw_config))
    with np.load(source, allow_pickle=False) as loaded:
        state = {
            name.removeprefix("state::"): torch.from_numpy(
                np.asarray(loaded[name])
            )
            for name in loaded.files
            if name.startswith("state::")
        }
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            f"{source}: state keys mismatch "
            f"(missing={sorted(expected - set(state))}, "
            f"extra={sorted(set(state) - expected)})"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def build_training_manifest(
    *,
    catalog: CatalogContract,
    ruleset: str,
    deck_size: int,
    dataset_sources: Sequence[Mapping[str, Any]],
    rows: Sequence[NemesisBattleRow],
    split: GroupedSplit,
    training_config: TrainingConfig,
    training_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "preview",
        "training_readiness": "preview_candidate",
        "dataset_contract": {
            "schema_version": UNIFIED_ROW_SCHEMA,
            "rows": len(rows),
            "catalog_verified_rows": sum(row.catalog_verified for row in rows),
            "catalog_unverified_lite_rows": sum(
                not row.catalog_verified for row in rows
            ),
            "sources": [dict(source) for source in dataset_sources],
            "explicit_card_levels_required": True,
            "quality_policy": (
                "eligible_lite=true and sample_weight>0; catalog-unavailable "
                "Lite rows retain their contract weight"
            ),
            "terminal_stalemate_maps_to": "draw",
        },
        "catalog": catalog.to_manifest(),
        "ruleset": {
            "id": ruleset,
            "delivery": "manifest_static_not_dynamic_input",
        },
        "deck_size": int(deck_size),
        "split": split.manifest(),
        "training_config": asdict(training_config),
        "training": dict(training_report),
        "metrics_definition": {
            "accuracy": "sample_weighted_top1",
            "logloss": "sample_weighted_natural_log",
            "brier": "sample_weighted_sum_squared_error_over_3_classes",
            "ece": "sample_weighted_top_label_15_bin",
            "swap_consistency": (
                "absolute probability drift after swapping decks, flipping "
                "starter, and restoring p1/draw/p2 class order"
            ),
            "test_baselines": (
                "majority/class-prior and starter-only empirical "
                "probabilities fitted on grouped train and scored on grouped "
                "test using the same weighted metrics"
            ),
            "subgroup_breakdowns": (
                "descriptive grouped-test metrics by source and checkpoint "
                "mix; no reliability claim for small groups"
            ),
        },
    }


__all__ = [
    "ARCHITECTURE_VERSION",
    "CLASS_INDEX",
    "CLASS_NAMES",
    "CatalogContract",
    "GroupedSplit",
    "MODEL_ID",
    "ModelConfig",
    "NemesisBattleRow",
    "NemesisLitePreviewModel",
    "NemesisRowExcluded",
    "TRAINING_MANIFEST_SCHEMA",
    "TrainingConfig",
    "UNIFIED_ROW_SCHEMA",
    "build_training_manifest",
    "class_prevalence_report",
    "classification_metrics",
    "evaluate_test_baselines",
    "grouped_matchup_split",
    "load_catalog_contract",
    "load_model_artifact",
    "load_unified_jsonl",
    "matchup_key_for_decks",
    "parse_unified_row",
    "predict_probabilities",
    "save_model_artifact",
    "sha256_file",
    "swap_consistency_metrics",
    "train_model",
]
