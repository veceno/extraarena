from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai.bot_brain import BerserkInference
from ai.bot_factory import BotGenerator
from ai.train_v2.classic_rl_env import ClassicRLEnv
from infrastructure.config import BOT_DIFFICULTY_PROFILES


DEFAULT_RUN_DIR = Path("ai/train_v2/runs/m4_balanced_from_0950_20260522_144431")


@dataclass(frozen=True)
class BotSpec:
    key: str
    label: str
    model_name: str
    profile: dict[str, Any]
    difficulty: str
    trophy_range: str
    player_max_level: int


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    candidate_level: int | None = None
    opponent_level: int | None = None
    production_like: bool = False


SCENARIOS = [
    Scenario("fair_l1", "Fair L1", candidate_level=1, opponent_level=1),
    Scenario("fair_l5", "Fair L5", candidate_level=5, opponent_level=5),
    Scenario("fair_l10", "Fair L10", candidate_level=10, opponent_level=10),
    Scenario("bot_underleveled", "Bot L5 vs Opp L8", candidate_level=5, opponent_level=8),
    Scenario("bot_overleveled", "Bot L8 vs Opp L5", candidate_level=8, opponent_level=5),
    Scenario("production_like", "Production-like levels", production_like=True),
]


def _train_v2_profile(model_path: str, *, selection: str, temperature: tuple[float, float]) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "temperature_range": temperature,
        "selection": selection,
        "placement_mode": "append_only",
        "verify_mask": False,
    }


def _legacy_profile(model_path: str, obs_dim: int, *, selection: str = "argmax") -> dict[str, Any]:
    return {
        "model_path": model_path,
        "obs_dim": obs_dim,
        "temperature_range": (0.01, 0.01),
        "selection": selection,
    }


def generation_specs() -> list[BotSpec]:
    return [
        BotSpec(
            key="extra_lr_v1",
            label="extra-lr-v1",
            model_name="OnlyVersusRandomBiggest",
            profile=_legacy_profile("ai/models/OnlyVersusRandomBiggest.onnx", 621),
            difficulty="lite",
            trophy_range="0-50",
            player_max_level=1,
        ),
        BotSpec(
            key="extra_lr_v2",
            label="extra-lr-v2",
            model_name="extra-lr-v3-medium",
            profile=_legacy_profile("ai/models/extra-lr-v3-medium.onnx", 997),
            difficulty="medium",
            trophy_range="151-500",
            player_max_level=5,
        ),
        BotSpec(
            key="extra_lr_v3_max",
            label="extra-lr-v3-max",
            model_name="extra-lr-v3-max",
            profile=_legacy_profile("ai/models/extra-lr-v3-max.onnx", 997),
            difficulty="max",
            trophy_range="1001+",
            player_max_level=10,
        ),
        BotSpec(
            key="extra_lr_v4_max",
            label="extra-lr-v4-max",
            model_name="extra-lr-v4-max",
            profile=_train_v2_profile(
                "ai/models/extra-lr-v4-max.onnx",
                selection="argmax",
                temperature=(0.01, 0.01),
            ),
            difficulty="max",
            trophy_range="1001+",
            player_max_level=10,
        ),
    ]


def difficulty_specs() -> list[BotSpec]:
    trophy_ranges = {
        "lite": "0-50",
        "easy": "51-150",
        "medium": "151-500",
        "hard": "501-1000",
        "max": "1001+",
    }
    player_max_levels = {
        "lite": 1,
        "easy": 2,
        "medium": 5,
        "hard": 7,
        "max": 10,
    }

    specs: list[BotSpec] = []
    for difficulty in ("lite", "easy", "medium", "hard", "max"):
        profile = dict(BOT_DIFFICULTY_PROFILES[difficulty])
        specs.append(
            BotSpec(
                key=f"difficulty_{difficulty}",
                label=difficulty,
                model_name=Path(profile["model_path"]).stem,
                profile=profile,
                difficulty=difficulty,
                trophy_range=trophy_ranges[difficulty],
                player_max_level=player_max_levels[difficulty],
            )
        )
    return specs


class BrainCorePolicy:
    def __init__(self, spec: BotSpec):
        self.spec = spec
        self.brain = BerserkInference(profiles={spec.key: spec.profile})
        self.invalid_actions = 0

    def reset(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def select_core_action(self, env: ClassicRLEnv, player_id: int):
        state = env.clone_state()
        legal = env._env.get_legal_actions(player_id)
        if not legal:
            return None
        idx = self.brain.get_action(state, player_id, legal, difficulty=self.spec.key)
        if idx < 0 or idx >= len(legal):
            self.invalid_actions += 1
            idx = 0
        return legal[idx]


def _all_card_levels(env: ClassicRLEnv, level: int) -> dict[int, int]:
    level = max(1, min(10, int(level)))
    return {int(card_id): level for card_id in env._cards_data.keys()}


def _default_deck_for_seed(seed: int) -> list[int]:
    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")
    episode_rng = random.Random(seed)
    deck_rng = random.Random(episode_rng.randint(0, 2**31 - 1))
    return env._generate_default_deck(deck_rng)


def _production_levels(spec: BotSpec, deck_ids: list[int], seed: int) -> dict[int, int]:
    state = random.getstate()
    try:
        random.seed(seed + sum(ord(c) for c in spec.key) * 17)
        levels = BotGenerator._build_bot_card_levels(
            spec.difficulty,
            spec.player_max_level,
            len(deck_ids),
        )
    finally:
        random.setstate(state)
    return {int(card_id): int(level) for card_id, level in zip(deck_ids, levels)}


def _levels_for(spec: BotSpec, scenario: Scenario, deck_ids: list[int], seed: int, *, is_candidate: bool) -> dict[int, int]:
    if scenario.production_like:
        return _production_levels(spec, deck_ids, seed)
    level = scenario.candidate_level if is_candidate else scenario.opponent_level
    if level is None:
        level = 1
    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")
    return _all_card_levels(env, level)


def _score_for_candidate(winner_id: int | None, candidate_player_id: int) -> float:
    if winner_id == candidate_player_id:
        return 1.0
    if winner_id is None:
        return 0.5
    return 0.0


def _run_game(
    *,
    candidate: BotSpec,
    opponent: BotSpec,
    candidate_policy: BrainCorePolicy,
    opponent_policy: BrainCorePolicy,
    scenario: Scenario,
    seed: int,
    candidate_player_id: int,
    starting_player_id: int,
    max_steps: int,
) -> dict[str, Any]:
    random.seed(seed * 1009 + candidate_player_id * 37 + starting_player_id * 101)
    np.random.seed((seed * 9173 + candidate_player_id * 311 + starting_player_id * 557) % (2**32 - 1))

    deck_ids = _default_deck_for_seed(seed)
    candidate_levels = _levels_for(candidate, scenario, deck_ids, seed, is_candidate=True)
    opponent_levels = _levels_for(opponent, scenario, deck_ids, seed + 13, is_candidate=False)

    if candidate_player_id == 1:
        p1_spec, p2_spec = candidate, opponent
        p1_policy, p2_policy = candidate_policy, opponent_policy
        p1_levels, p2_levels = candidate_levels, opponent_levels
    else:
        p1_spec, p2_spec = opponent, candidate
        p1_policy, p2_policy = opponent_policy, candidate_policy
        p1_levels, p2_levels = opponent_levels, candidate_levels

    p1_policy.reset(seed * 3 + 1)
    p2_policy.reset(seed * 3 + 2)

    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")
    env.reset(
        seed=seed,
        p1_deck_ids=deck_ids,
        p2_deck_ids=deck_ids,
        p1_levels=p1_levels,
        p2_levels=p2_levels,
        starting_player_id=starting_player_id,
    )

    invalid_before = candidate_policy.invalid_actions + opponent_policy.invalid_actions
    steps = 0
    terminated = False
    truncated = False
    for steps in range(1, max_steps + 1):
        cp = env.current_player_id()
        policy = p1_policy if cp == 1 else p2_policy
        action = policy.select_core_action(env, cp)
        if action is None:
            break
        _, _, terminated, truncated, info = env.step_core_action(action)
        if info.get("invalid_action"):
            if cp == candidate_player_id:
                candidate_policy.invalid_actions += 1
            else:
                opponent_policy.invalid_actions += 1
        if terminated or truncated:
            break

    winner_id = env.winner_id()
    state = env._env.state
    invalid_after = candidate_policy.invalid_actions + opponent_policy.invalid_actions
    return {
        "candidate": candidate.key,
        "opponent": opponent.key,
        "scenario": scenario.key,
        "seed": seed,
        "candidate_player_id": candidate_player_id,
        "starting_player_id": starting_player_id,
        "p1": p1_spec.key,
        "p2": p2_spec.key,
        "winner_id": winner_id,
        "winner": p1_spec.key if winner_id == 1 else p2_spec.key if winner_id == 2 else None,
        "candidate_score": _score_for_candidate(winner_id, candidate_player_id),
        "candidate_win": winner_id == candidate_player_id,
        "candidate_loss": winner_id is not None and winner_id != candidate_player_id,
        "draw": winner_id is None,
        "steps": steps,
        "turns": state.turn_number,
        "p1_hp": state.p1.hero.hp,
        "p2_hp": state.p2.hero.hp,
        "candidate_hp_margin": (state.p1.hero.hp - state.p2.hero.hp) if candidate_player_id == 1 else (state.p2.hero.hp - state.p1.hero.hp),
        "invalid_actions": invalid_after - invalid_before,
    }


def _empty_metric_row(spec: BotSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "model_name": spec.model_name,
        "difficulty": spec.difficulty,
        "trophy_range": spec.trophy_range,
        "format": spec.profile.get("format", "legacy"),
        "selection": spec.profile.get("selection", "softmax"),
        "temperature_range": list(spec.profile.get("temperature_range", (1.0, 1.0))),
        "games": 0,
        "score": 0.0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "p1_games": 0,
        "p1_wins": 0,
        "p2_games": 0,
        "p2_wins": 0,
        "hp_margin": 0.0,
        "invalid_actions": 0,
        "scenarios": {},
    }


def _add_game(row: dict[str, Any], game: dict[str, Any]) -> None:
    row["games"] += 1
    row["score"] += game["candidate_score"]
    row["wins"] += int(game["candidate_win"])
    row["losses"] += int(game["candidate_loss"])
    row["draws"] += int(game["draw"])
    row["hp_margin"] += float(game["candidate_hp_margin"])
    row["invalid_actions"] += int(game["invalid_actions"])
    if game["candidate_player_id"] == 1:
        row["p1_games"] += 1
        row["p1_wins"] += int(game["candidate_win"])
    else:
        row["p2_games"] += 1
        row["p2_wins"] += int(game["candidate_win"])

    scenario = game["scenario"]
    srow = row["scenarios"].setdefault(
        scenario,
        {"games": 0, "score": 0.0, "wins": 0, "losses": 0, "draws": 0, "hp_margin": 0.0, "invalid_actions": 0},
    )
    srow["games"] += 1
    srow["score"] += game["candidate_score"]
    srow["wins"] += int(game["candidate_win"])
    srow["losses"] += int(game["candidate_loss"])
    srow["draws"] += int(game["draw"])
    srow["hp_margin"] += float(game["candidate_hp_margin"])
    srow["invalid_actions"] += int(game["invalid_actions"])


def _finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    games = max(1, row["games"])
    row["score_rate"] = row["score"] / games
    row["winrate"] = row["wins"] / games
    row["draw_rate"] = row["draws"] / games
    row["p1_winrate"] = row["p1_wins"] / row["p1_games"] if row["p1_games"] else 0.0
    row["p2_winrate"] = row["p2_wins"] / row["p2_games"] if row["p2_games"] else 0.0
    row["avg_hp_margin"] = row["hp_margin"] / games
    for srow in row["scenarios"].values():
        sgames = max(1, srow["games"])
        srow["score_rate"] = srow["score"] / sgames
        srow["winrate"] = srow["wins"] / sgames
        srow["draw_rate"] = srow["draws"] / sgames
        srow["avg_hp_margin"] = srow["hp_margin"] / sgames
    return row


def run_report_group(
    name: str,
    specs: list[BotSpec],
    *,
    seeds: list[int],
    max_steps: int,
    scenarios: list[Scenario],
) -> dict[str, Any]:
    policies = {spec.key: BrainCorePolicy(spec) for spec in specs}
    rows = {spec.key: _empty_metric_row(spec) for spec in specs}
    games_written = 0

    for scenario_idx, scenario in enumerate(scenarios, 1):
        print(f"[{name}] scenario {scenario_idx}/{len(scenarios)} {scenario.key}", flush=True)
        for candidate in specs:
            for opponent in specs:
                if opponent.key == candidate.key:
                    continue
                for seed in seeds:
                    for candidate_player_id in (1, 2):
                        for starting_player_id in (1, 2):
                            game = _run_game(
                                candidate=candidate,
                                opponent=opponent,
                                candidate_policy=policies[candidate.key],
                                opponent_policy=policies[opponent.key],
                                scenario=scenario,
                                seed=seed,
                                candidate_player_id=candidate_player_id,
                                starting_player_id=starting_player_id,
                                max_steps=max_steps,
                            )
                            _add_game(rows[candidate.key], game)
                            games_written += 1
        print(f"[{name}] completed {scenario.key}; games={games_written}", flush=True)

    final_rows = [_finalize_row(row) for row in rows.values()]
    final_rows.sort(key=lambda r: (r["score_rate"], r["winrate"], r["avg_hp_margin"]), reverse=True)
    for idx, row in enumerate(final_rows, 1):
        row["rank"] = idx

    return {
        "name": name,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "seed_count": len(seeds),
        "max_steps": max_steps,
        "scenarios": [s.key for s in scenarios],
        "games": games_written,
        "rows": final_rows,
    }


def _load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_bar_chart(rows: list[dict[str, Any]], title: str, output_png: Path, output_svg: Path) -> None:
    width = max(1200, 250 * len(rows))
    height = 820
    margin_l, margin_r, margin_t, margin_b = 90, 60, 90, 245
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    bg = (248, 250, 252)
    ink = (22, 28, 36)
    grid = (214, 222, 232)
    bar = (67, 112, 219)
    bar2 = (38, 166, 154)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    title_font = _load_font(30, bold=True)
    label_font = _load_font(18, bold=True)
    small_font = _load_font(14)
    value_font = _load_font(17, bold=True)

    draw.text((margin_l, 28), title, fill=ink, font=title_font)
    max_score = max([row["score_rate"] for row in rows] + [1.0])
    y_max = max(1.0, math.ceil(max_score * 10) / 10)

    for i in range(0, 11):
        val = y_max * i / 10
        y = margin_t + plot_h - (val / y_max) * plot_h
        draw.line((margin_l, y, width - margin_r, y), fill=grid, width=1)
        draw.text((20, y - 8), f"{val * 100:.0f}%", fill=(90, 99, 112), font=small_font)

    bar_gap = 34
    slot_w = plot_w / max(1, len(rows))
    bar_w = max(54, min(130, slot_w - bar_gap))

    for idx, row in enumerate(rows):
        x = margin_l + idx * slot_w + (slot_w - bar_w) / 2
        score = row["score_rate"]
        h = (score / y_max) * plot_h
        y = margin_t + plot_h - h
        color = bar2 if row.get("format") == "train_v2_classic_v1" else bar
        draw.rounded_rectangle((x, y, x + bar_w, margin_t + plot_h), radius=8, fill=color)
        draw.text((x + bar_w / 2 - 28, y - 26), f"{score * 100:.1f}%", fill=ink, font=value_font)

        label_y = margin_t + plot_h + 16
        lines = [
            row["label"],
            f"trophies {row['trophy_range']}",
            row["model_name"],
            f"{row['selection']} T={row['temperature_range'][0]}",
        ]
        for li, line in enumerate(lines):
            font = label_font if li == 0 else small_font
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((x + bar_w / 2 - (bbox[2] - bbox[0]) / 2, label_y + li * 28), line, fill=ink, font=font)

    img.save(output_png)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{margin_l}" y="50" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#161c24">{title}</text>',
    ]
    for i in range(0, 11):
        val = y_max * i / 10
        y = margin_t + plot_h - (val / y_max) * plot_h
        svg_parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#d6dee8" stroke-width="1"/>')
        svg_parts.append(f'<text x="20" y="{y + 5:.1f}" font-family="Arial, sans-serif" font-size="14" fill="#5a6370">{val * 100:.0f}%</text>')
    for idx, row in enumerate(rows):
        x = margin_l + idx * slot_w + (slot_w - bar_w) / 2
        score = row["score_rate"]
        h = (score / y_max) * plot_h
        y = margin_t + plot_h - h
        color = "#26a69a" if row.get("format") == "train_v2_classic_v1" else "#4370db"
        svg_parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="8" fill="{color}"/>')
        svg_parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" font-family="Arial, sans-serif" font-size="17" font-weight="700" text-anchor="middle" fill="#161c24">{score * 100:.1f}%</text>')
        label_y = margin_t + plot_h + 34
        lines = [
            (row["label"], 18, 700),
            (f"trophies {row['trophy_range']}", 14, 400),
            (row["model_name"], 14, 400),
            (f"{row['selection']} T={row['temperature_range'][0]}", 14, 400),
        ]
        for li, (line, size, weight) in enumerate(lines):
            svg_parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{label_y + li * 28:.1f}" font-family="Arial, sans-serif" '
                f'font-size="{size}" font-weight="{weight}" text-anchor="middle" fill="#161c24">{line}</text>'
            )
    svg_parts.append("</svg>")
    output_svg.write_text("\n".join(svg_parts), encoding="utf-8")


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bot_power_report.json"
    csv_path = output_dir / "bot_power_report.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "report", "rank", "label", "score_rate", "winrate", "draw_rate",
            "p1_winrate", "p2_winrate", "avg_hp_margin", "invalid_actions",
            "games", "trophy_range", "model_name", "format", "selection", "temperature_range",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report_name in ("generations", "difficulties"):
            for row in report[report_name]["rows"]:
                writer.writerow({
                    "report": report_name,
                    "rank": row["rank"],
                    "label": row["label"],
                    "score_rate": row["score_rate"],
                    "winrate": row["winrate"],
                    "draw_rate": row["draw_rate"],
                    "p1_winrate": row["p1_winrate"],
                    "p2_winrate": row["p2_winrate"],
                    "avg_hp_margin": row["avg_hp_margin"],
                    "invalid_actions": row["invalid_actions"],
                    "games": row["games"],
                    "trophy_range": row["trophy_range"],
                    "model_name": row["model_name"],
                    "format": row["format"],
                    "selection": row["selection"],
                    "temperature_range": row["temperature_range"],
                })

    _draw_bar_chart(
        report["generations"]["rows"],
        "Generational Flagships: balanced score across scenarios",
        output_dir / "bot_power_generations.png",
        output_dir / "bot_power_generations.svg",
    )
    _draw_bar_chart(
        report["difficulties"]["rows"],
        "Current Difficulty Presets: production settings",
        output_dir / "bot_power_difficulties.png",
        output_dir / "bot_power_difficulties.svg",
    )


def build_report(*, output_dir: Path, seed_start: int, seed_count: int, max_steps: int) -> dict[str, Any]:
    seeds = list(range(seed_start, seed_start + seed_count))
    report = {
        "version": "bot_power_report_v1",
        "seed_start": seed_start,
        "seed_count": seed_count,
        "max_steps": max_steps,
        "scenario_weights": "equal",
        "scenarios": [{"key": s.key, "label": s.label} for s in SCENARIOS],
        "generations": run_report_group(
            "generations",
            generation_specs(),
            seeds=seeds,
            max_steps=max_steps,
            scenarios=SCENARIOS,
        ),
        "difficulties": run_report_group(
            "difficulties",
            difficulty_specs(),
            seeds=seeds,
            max_steps=max_steps,
            scenarios=SCENARIOS,
        ),
    }
    write_outputs(report, output_dir)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build realistic bot power charts for production and TrainV2 models")
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=220)
    args = parser.parse_args()

    report = build_report(
        output_dir=Path(args.output_dir),
        seed_start=args.seed_start,
        seed_count=args.seeds,
        max_steps=args.max_steps,
    )
    print("\nGenerations")
    for row in report["generations"]["rows"]:
        print(f"#{row['rank']} {row['label']:<18} score={row['score_rate'] * 100:5.1f}% invalid={row['invalid_actions']}")
    print("\nDifficulties")
    for row in report["difficulties"]["rows"]:
        print(f"#{row['rank']} {row['label']:<8} score={row['score_rate'] * 100:5.1f}% invalid={row['invalid_actions']}")
    print(f"\nSaved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    _main()
