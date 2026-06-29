"""V5 acceptance gauntlet contracts with Rust-first hot-path policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


EXPLOIT_AGENT_KINDS = (
    "face_rush",
    "board_control",
    "greedy_trade",
    "stall",
    "punish_empty_board",
    "anti_draw_greed",
    "anti_hand_leak_overfit",
)


@dataclass(frozen=True)
class ExploitLaneConfig:
    kind: str
    weight: float = 1.0
    runtime: str = "rust"

    def validate(self) -> "ExploitLaneConfig":
        if self.kind not in EXPLOIT_AGENT_KINDS:
            raise ValueError(f"unknown exploit lane kind: {self.kind}")
        if self.runtime != "rust":
            raise ValueError("exploit lanes must use rust runtime in production gauntlets")
        if float(self.weight) <= 0.0:
            raise ValueError("exploit lane weight must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class V5GauntletConfig:
    require_rust_hot_path: bool = True
    no_assist_min_score_rate: float = 0.45
    p1_p2_max_score_gap: float = 0.12
    level_handicap_min_score_rate: float = 0.35
    invalid_action_max_rate: float = 0.0
    exploit_resistance_min_score_rate: float = 0.42
    adaptive_strength_min_margin: float = 0.03

    def validate(self) -> "V5GauntletConfig":
        for name, value in asdict(self).items():
            if name == "require_rust_hot_path":
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_default_exploit_gauntlet() -> list[ExploitLaneConfig]:
    return [ExploitLaneConfig(kind=kind).validate() for kind in EXPLOIT_AGENT_KINDS]


__all__ = [
    "EXPLOIT_AGENT_KINDS",
    "ExploitLaneConfig",
    "V5GauntletConfig",
    "build_default_exploit_gauntlet",
]
