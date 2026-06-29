#!/usr/bin/env python3
"""Run fun showmatches: frontier LLM policies vs Extra-LR-V4-Max.

This is not a training script. It runs visible game logs where each frontier
model chooses one legal action_id per turn from the TrainV2/TrainV3 action
space. Each model keeps one persistent conversation across all of its games.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))

from ai.train_v2.classic_actions_v1 import decode_action  # noqa: E402
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from core.actions import AttackAction, EndTurnAction, PlayCardAction  # noqa: E402
from core.state import CardInstance, CardType  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402


DEFAULT_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_V4_MAX = ROOT / "ai" / "models" / "extra-lr-v4-max.onnx"
DEFAULT_DOC = ROOT / "docs" / "FRONTIER_LLM_SHOWMATCH_GUIDE.md"
DEFAULT_CARD_CATALOG = ROOT / "docs" / "frontier_llm_card_catalog.csv"
DEFAULT_API_MODELS = [
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.5-flash",
    "moonshotai/kimi-k2.6",
    "sber/gigachat-2-max",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "anthropic/claude-opus-4.8",
]
CODEX_PARTICIPANT = "codex_cli"
MODEL_PROVIDER_OVERRIDES = {
    "deepseek/deepseek-v4-pro": {"only": ["DeepSeek"], "allow_fallbacks": False},
}
FULL_STATIC_CONTEXT_PREFIXES = (
    "deepseek/",
    "google/",
    "anthropic/",
)


@dataclass
class TurnDecision:
    action_id: int
    battle_line: str = ""
    raw: dict[str, Any] | None = None
    fallback: bool = False
    error: str = ""


class PersistentOpenAICompatiblePolicy:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        run_dir: Path,
        guide: str,
        catalog: str,
        cache_strategy: str,
        games: int,
        opponent_meta: dict[str, Any],
        timeout: float = 120.0,
        thinking_mode: bool = True,
        max_tokens: int = 0,
        repair_max_tokens: int = 0,
    ):
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.thinking_mode = bool(thinking_mode)
        self.max_tokens = int(max_tokens)
        self.repair_max_tokens = int(repair_max_tokens)
        self.cache_strategy = cache_strategy
        self.turns_path = run_dir / "turns.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        self.requests_path = run_dir / "requests.jsonl"
        self.raw_responses_path = run_dir / "raw_responses.jsonl"
        self.system_message = _system_message(
            participant=model,
            guide=guide,
            catalog=catalog,
            games=games,
            opponent_meta=opponent_meta,
            via_codex_cli=False,
        )
        self.messages: list[dict[str, str]] = [self.system_message]

    def reset_for_match(self, seed: int) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "event": "MODEL_RESET_FOR_NEXT_GAME",
                        "seed": int(seed),
                        "note": "This starts the next battle. Previous battle history remains in this same conversation context.",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    def observe_game_result(self, result: dict[str, Any]) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "event": "GAME_RESULT",
                        "result": _compact_game_result(result),
                        "note": "Remember this result for the remaining battles.",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    def select_action(self, payload: dict[str, Any], legal: list[int]) -> TurnDecision:
        self.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "event": "TURN_REQUEST",
                        "context_policy": {
                            "one_battle_one_context": True,
                            "all_battles_one_context": True,
                            "payload_only_contains_current_state_and_legal_actions": True,
                        },
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        request = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        }
        provider_override = MODEL_PROVIDER_OVERRIDES.get(self.model)
        if provider_override is not None:
            request["provider"] = dict(provider_override)
        if self.max_tokens > 0:
            request["max_tokens"] = self.max_tokens
        if self.thinking_mode:
            request["reasoning"] = {"effort": "high", "enabled": True, "exclude": False}
        try:
            response = self._call_chat("turn", request)
        except Exception as exc:
            if "reasoning" in request:
                request.pop("reasoning", None)
                try:
                    response = self._call_chat("turn_retry_without_reasoning", request)
                except Exception as retry_exc:
                    return self._fallback(legal, f"{type(retry_exc).__name__}: {retry_exc}")
            else:
                return self._fallback(legal, f"{type(exc).__name__}: {exc}")

        content = _extract_chat_content(response)
        decision = _parse_decision(content, legal)
        if decision is None:
            repair_response = self._repair_decision(payload=payload, legal=legal, bad_response=response, bad_content=content)
            if repair_response is not None:
                response = repair_response
                content = _extract_chat_content(response)
                decision = _parse_decision(content, legal)
        if decision is None:
            return self._fallback(
                legal,
                "response did not contain legal action_id: "
                + json.dumps({"content": content[:500], "response": _compact_response_for_log(response)}, ensure_ascii=False),
            )
        decision.raw = {"content": content, "usage": response.get("usage", {})}
        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"action_id": int(decision.action_id), "battle_line": decision.battle_line},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        _append_jsonl(self.turns_path, {"payload": payload, "decision": decision_to_json(decision)})
        return decision

    def _repair_decision(
        self,
        *,
        payload: dict[str, Any],
        legal: list[int],
        bad_response: dict[str, Any],
        bad_content: str,
    ) -> dict[str, Any] | None:
        repair_messages = [
            self.system_message,
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "event": "REPAIR_TURN_REQUEST",
                        "instruction": (
                            "Your previous response was invalid or empty. Return strict JSON only. "
                            "Choose exactly one action_id from legal_action_ids."
                        ),
                        "legal_action_ids": [int(item) for item in legal],
                        "turn_payload": payload,
                        "previous_content": bad_content[:500],
                        "previous_finish_reason": _finish_reason(bad_response),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        request = {
            "model": self.model,
            "messages": repair_messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        provider_override = MODEL_PROVIDER_OVERRIDES.get(self.model)
        if provider_override is not None:
            request["provider"] = dict(provider_override)
        if self.repair_max_tokens > 0:
            request["max_tokens"] = self.repair_max_tokens
        try:
            return self._call_chat("repair", request)
        except Exception as exc:
            _append_jsonl(self.errors_path, {"model": self.model, "repair_error": f"{type(exc).__name__}: {exc}"})
            return None

    def _call_chat(self, kind: str, request_json: dict[str, Any]) -> dict[str, Any]:
        _append_jsonl(
            self.requests_path,
            {
                "kind": kind,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "request": request_json,
            },
        )
        response = self._post_chat(request_json)
        _append_jsonl(
            self.raw_responses_path,
            {
                "kind": kind,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "response": response,
            },
        )
        return response

    def _post_chat(self, request_json: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(request_json).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc

    def _fallback(self, legal: list[int], error: str) -> TurnDecision:
        _append_jsonl(self.errors_path, {"model": self.model, "error": error})
        return TurnDecision(action_id=_fallback_action(legal), fallback=True, error=error)


class CodexCliPolicy:
    def __init__(
        self,
        *,
        run_dir: Path,
        guide: str,
        catalog: str,
        games: int,
        opponent_meta: dict[str, Any],
        timeout: float = 180.0,
    ):
        self.name = CODEX_PARTICIPANT
        self.run_dir = run_dir
        self.timeout = float(timeout)
        self.turns_path = run_dir / "turns.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        self.raw_cli_path = run_dir / "codex_cli_raw.jsonl"
        self.context_log: list[dict[str, Any]] = [
            {
                "event": "SYSTEM_CONTEXT",
                "content": _system_prompt(
                    participant=CODEX_PARTICIPANT,
                    guide=guide,
                    catalog=catalog,
                    games=games,
                    opponent_meta=opponent_meta,
                    via_codex_cli=True,
                ),
            }
        ]
        self.schema_path = run_dir / "codex_action_schema.json"
        self.schema_path.write_text(
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "integer"},
                        "battle_line": {"type": "string"},
                    },
                    "required": ["action_id", "battle_line"],
                    "additionalProperties": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def reset_for_match(self, seed: int) -> None:
        self.context_log.append({"event": "MODEL_RESET_FOR_NEXT_GAME", "seed": int(seed)})

    def observe_game_result(self, result: dict[str, Any]) -> None:
        self.context_log.append({"event": "GAME_RESULT", "result": result})

    def select_action(self, payload: dict[str, Any], legal: list[int]) -> TurnDecision:
        prompt = (
            "You are playing Extra Arena as the Codex CLI participant. "
            "Use the full context below. Think privately, then return only JSON that matches the schema.\n\n"
            + json.dumps({"context": self.context_log, "turn_request": payload}, ensure_ascii=False, sort_keys=True)
        )
        out_path = self.run_dir / f"codex_last_{payload['game_index']:02d}_{payload['step']:03d}.json"
        cmd = [
            "codex",
            "-a",
            "never",
            "exec",
            "--cd",
            str(ROOT),
            "--sandbox",
            "read-only",
            "--output-schema",
            str(self.schema_path),
            "--output-last-message",
            str(out_path),
            "-",
        ]
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                cwd=str(ROOT),
            )
        except Exception as exc:
            return self._fallback(legal, f"{type(exc).__name__}: {exc}")
        raw_text = out_path.read_text(encoding="utf-8") if out_path.exists() else completed.stdout
        _append_jsonl(
            self.raw_cli_path,
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "payload": payload,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "output_last_message": raw_text,
            },
        )
        decision = _parse_decision(raw_text, legal)
        if completed.returncode != 0 or decision is None:
            err = (completed.stderr or completed.stdout or raw_text)[:2000]
            return self._fallback(legal, f"codex failed rc={completed.returncode}: {err}")
        decision.raw = {"content": raw_text, "stderr_tail": completed.stderr[-1000:]}
        self.context_log.append({"event": "TURN_REQUEST", "payload": payload})
        self.context_log.append({"event": "TURN_DECISION", "decision": decision_to_json(decision)})
        _append_jsonl(self.turns_path, {"payload": payload, "decision": decision_to_json(decision)})
        return decision

    def _fallback(self, legal: list[int], error: str) -> TurnDecision:
        _append_jsonl(self.errors_path, {"model": self.name, "error": error})
        return TurnDecision(action_id=_fallback_action(legal), fallback=True, error=error)


def run_showmatches(config: argparse.Namespace) -> dict[str, Any]:
    _validate_v4max(config.v4_model)
    run_dir = config.output_dir or ROOT / "TrainV3" / "runs" / f"frontier_llm_showmatch_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    guide = config.guide.read_text(encoding="utf-8")
    full_catalog = _read_catalog_compact(config.card_catalog)
    compact_catalog = _read_mechanics_glossary(config.card_catalog)
    opponent_meta = _opponent_meta(config.v4_model)
    participants = _resolve_participants(config.participants)

    api_key = _resolve_api_key()
    policies = []
    for participant in participants:
        pdir = _participant_dir(run_dir, participant)
        pdir.mkdir(parents=True, exist_ok=True)
        if participant == CODEX_PARTICIPANT:
            policies.append(
                CodexCliPolicy(
                    run_dir=pdir,
                    guide=guide,
                    catalog=full_catalog,
                    games=config.games,
                    opponent_meta=opponent_meta,
                    timeout=config.codex_timeout,
                )
            )
        else:
            if not api_key:
                raise ValueError("API participants require SHOWMATCH_API_KEY, NANO_GPT_API_KEY, or OPENAI_API_KEY")
            cache_strategy = _cache_strategy_for_model(participant)
            policies.append(
                PersistentOpenAICompatiblePolicy(
                    model=participant,
                    base_url=config.base_url,
                    api_key=api_key,
                    run_dir=pdir,
                    guide=guide if cache_strategy != "compact_unknown_provider" else _compact_guide(),
                    catalog=full_catalog if cache_strategy != "compact_unknown_provider" else compact_catalog,
                    cache_strategy=cache_strategy,
                    games=config.games,
                    opponent_meta=opponent_meta,
                    timeout=config.api_timeout,
                    thinking_mode=not config.no_thinking,
                    max_tokens=config.max_tokens,
                    repair_max_tokens=config.repair_max_tokens,
                )
            )

    top_summary: dict[str, Any] = {
        "schema": "extra_arena_frontier_llm_showmatch_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "games_per_participant": int(config.games),
        "max_steps": int(config.max_steps),
        "base_url": config.base_url.rstrip("/"),
        "opponent": opponent_meta,
        "participants": participants,
        "thinking_mode": not config.no_thinking,
        "results": [],
    }
    (run_dir / "run_config_public.json").write_text(json.dumps(top_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for policy in policies:
        participant_result = _run_participant(policy, config, run_dir, opponent_meta)
        top_summary["results"].append(participant_result)
        (run_dir / "summary.json").write_text(json.dumps(top_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(_format_participant_line(participant_result), flush=True)

    return top_summary


def _run_participant(policy: Any, config: argparse.Namespace, run_dir: Path, opponent_meta: dict[str, Any]) -> dict[str, Any]:
    pdir = _participant_dir(run_dir, policy.name)
    games_path = pdir / "games.jsonl"
    v4_policy = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    games = []
    for game_idx in range(1, int(config.games) + 1):
        seed = int(config.seed) + game_idx - 1
        llm_player_id = 1 if game_idx % 2 == 1 else 2
        starting_player_id = 1 if ((game_idx - 1) // 2) % 2 == 0 else 2
        print(f"[{policy.name}] game {game_idx}/{config.games}: llm=p{llm_player_id} start=p{starting_player_id} seed={seed}", flush=True)
        policy.reset_for_match(seed)
        v4_policy.reset(seed * 17 + game_idx)
        result = _run_one_game(
            policy=policy,
            v4_policy=v4_policy,
            seed=seed,
            game_idx=game_idx,
            llm_player_id=llm_player_id,
            starting_player_id=starting_player_id,
            max_steps=int(config.max_steps),
            total_games=int(config.games),
            opponent_meta=opponent_meta,
        )
        policy.observe_game_result(result)
        games.append(result)
        _append_jsonl(games_path, result)

    summary = _summarize_games(policy.name, games)
    (pdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _run_one_game(
    *,
    policy: Any,
    v4_policy: OnnxActionPolicy,
    seed: int,
    game_idx: int,
    llm_player_id: int,
    starting_player_id: int,
    max_steps: int,
    total_games: int,
    opponent_meta: dict[str, Any],
) -> dict[str, Any]:
    v4_player_id = 2 if llm_player_id == 1 else 1
    env = TrainV3ClassicEnv(TrainV3EnvConfig(seed=seed, verify_mask=False, placement_mode="append_only"))
    env.reset(seed=seed, p1_is_bot=True, p2_is_bot=True, starting_player_id=starting_player_id)
    steps = 0
    invalid = 0
    fallback_count = 0
    transcript = []
    for steps in range(1, max_steps + 1):
        current = env.current_player_id()
        if current == llm_player_id:
            legal = env.env.legal_action_ids(current)
            payload = _build_turn_payload(
                env=env,
                game_idx=game_idx,
                step=steps,
                seed=seed,
                llm_player_id=llm_player_id,
                v4_player_id=v4_player_id,
                starting_player_id=starting_player_id,
                total_games=total_games,
                opponent_meta=opponent_meta,
            )
            decision = policy.select_action(payload, legal)
            action_id = int(decision.action_id)
            fallback_count += int(decision.fallback)
            actor_name = policy.name
        else:
            action_id = int(v4_policy.select_action(env.env, current))
            decision = TurnDecision(action_id=action_id, battle_line="", raw=None)
            actor_name = "extra-lr-v4-max"

        _obs, _reward, terminated, truncated, info = env.step(action_id)
        invalid += int(bool(info.get("invalid_action")))
        transcript.append(
            {
                "step": steps,
                "actor_player_id": current,
                "actor": actor_name,
                "action_id": action_id,
                "action_text": _describe_action_from_pre_info(info),
                "invalid": bool(info.get("invalid_action")),
                "fallback": bool(decision.fallback),
                "battle_line": decision.battle_line,
            }
        )
        if terminated or truncated:
            break

    state = env.env._env.state
    winner = env.env.winner_id()
    result = {
        "game_index": game_idx,
        "seed": seed,
        "llm_model": policy.name,
        "llm_player_id": llm_player_id,
        "v4_player_id": v4_player_id,
        "starting_player_id": starting_player_id,
        "winner_id": winner,
        "winner_name": policy.name if winner == llm_player_id else "extra-lr-v4-max" if winner == v4_player_id else None,
        "llm_win": winner == llm_player_id,
        "draw": winner is None,
        "steps": steps,
        "turn_number": state.turn_number,
        "p1_hp": state.p1.hero.hp,
        "p2_hp": state.p2.hero.hp,
        "llm_hp": state.p1.hero.hp if llm_player_id == 1 else state.p2.hero.hp,
        "v4_hp": state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp,
        "invalid_actions": invalid,
        "llm_fallbacks": fallback_count,
        "opponent": opponent_meta,
        "transcript": transcript,
    }
    return result


def _build_turn_payload(
    *,
    env: TrainV3ClassicEnv,
    game_idx: int,
    step: int,
    seed: int,
    llm_player_id: int,
    v4_player_id: int,
    starting_player_id: int,
    total_games: int,
    opponent_meta: dict[str, Any],
) -> dict[str, Any]:
    state = env.env._env.state
    me, enemy = _players_for(state, llm_player_id)
    legal_ids = env.env.legal_action_ids(llm_player_id)
    actions = [_describe_legal_action(env, llm_player_id, action_id) for action_id in legal_ids]
    return {
        "game_index": int(game_idx),
        "total_games": int(total_games),
        "step": int(step),
        "seed": int(seed),
        "player_id": int(llm_player_id),
        "opponent_player_id": int(v4_player_id),
        "opponent_name": "Extra-LR-V4-Max",
        "opponent_confirmation": opponent_meta,
        "starting_player_id": int(starting_player_id),
        "turn_number": int(state.turn_number),
        "current_player_id": int(state.current_turn_owner_id),
        "you": _player_public_summary(me, include_hand=True, include_deck_summary=True),
        "opponent": _player_public_summary(enemy, include_hand=False, include_deck_summary=False),
        "recent_actions": list(state.action_history[-10:]),
        "legal_action_count": len(legal_ids),
        "legal_actions": actions,
        "response_schema": {"action_id": "integer from legal_actions", "battle_line": "short optional string"},
    }


def _player_public_summary(player: Any, *, include_hand: bool, include_deck_summary: bool) -> dict[str, Any]:
    out = {
        "player_id": int(player.user_id),
        "hero": _card_summary(player.hero, zone="hero"),
        "mana": int(player.mana),
        "max_mana": int(player.max_mana),
        "hand_size": len(player.hand),
        "board": [_card_summary(card, zone="board", slot=idx) for idx, card in enumerate(player.board)],
        "deck_size": len(player.deck),
        "graveyard": [_card_summary(card, zone="graveyard", slot=idx) for idx, card in enumerate(player.graveyard[-8:])],
        "graveyard_size": len(player.graveyard),
    }
    if include_hand:
        out["hand"] = [_card_summary(card, zone="hand", slot=idx) for idx, card in enumerate(player.hand)]
    if include_deck_summary:
        out["own_deck_known_cards_remaining"] = [_card_public_name(card) for card in player.deck]
    return out


def _card_summary(card: CardInstance, *, zone: str, slot: int | None = None) -> dict[str, Any]:
    data = {
        "zone": zone,
        "card_id": int(card.card_id),
        "name": card.name,
        "card_type": card.card_type.value if isinstance(card.card_type, CardType) else str(card.card_type),
        "mana_cost": int(card.mana_cost),
        "attack": int(card.attack),
        "hp": int(card.hp),
        "max_hp": int(card.max_hp),
        "mechanics": list(card.mechanics),
        "is_ready": bool(card.is_ready),
        "is_frozen": bool(card.is_frozen),
    }
    if slot is not None:
        data["slot"] = int(slot)
    return data


def _card_public_name(card: CardInstance) -> dict[str, Any]:
    return {"card_id": int(card.card_id), "name": card.name, "card_type": card.card_type.value}


def _describe_legal_action(env: TrainV3ClassicEnv, player_id: int, action_id: int) -> dict[str, Any]:
    state = env.env._env.state
    action = decode_action(state, player_id, int(action_id))
    text = "Unknown action"
    source = None
    target = None
    action_type = "unknown"
    if isinstance(action, EndTurnAction):
        text = "End turn"
        action_type = "end_turn"
    elif isinstance(action, PlayCardAction):
        me, enemy = _players_for(state, player_id)
        source_card = me.hand[action.hand_index] if 0 <= action.hand_index < len(me.hand) else None
        source = _card_summary(source_card, zone="hand", slot=action.hand_index) if source_card else None
        target_card = _find_target_card(me, enemy, action.target_id)
        target = _card_summary(target_card, zone="target") if target_card else None
        action_type = "play_card"
        target_text = f" targeting {target_card.name}" if target_card else " with no target"
        pos_text = f" to board position {action.position}" if action.position is not None else ""
        text = f"Play {source_card.name if source_card else 'card'} from hand slot {action.hand_index}{pos_text}{target_text}"
    elif isinstance(action, AttackAction):
        me, enemy = _players_for(state, player_id)
        attacker = _find_card_by_instance_id(me.board, action.attacker_id)
        target_card = enemy.hero if action.target_is_hero else _find_card_by_instance_id(enemy.board, action.target_id)
        source = _card_summary(attacker, zone="board") if attacker else None
        target = _card_summary(target_card, zone="target") if target_card else None
        action_type = "attack"
        text = f"Attack {target_card.name if target_card else 'target'} with {attacker.name if attacker else 'attacker'}"
    preview = _preview_delta(env, action) if action is not None else {}
    return {
        "action_id": int(action_id),
        "type": action_type,
        "text": text,
        "source": source,
        "target": target,
        "preview_hp_delta": preview,
    }


def _preview_delta(env: TrainV3ClassicEnv, action: Any) -> dict[str, Any]:
    try:
        delta = env.env._env.get_preview_delta(action)
    except Exception:
        return {}
    if not delta:
        return {}
    state = env.env._env.state
    cards = [state.p1.hero, state.p2.hero, *state.p1.board, *state.p2.board]
    name_by_id = {str(card.instance_id): card.name for card in cards}
    return {name_by_id.get(str(instance_id), str(instance_id)): int(value) for instance_id, value in delta.items()}


def _describe_action_from_pre_info(info: dict[str, Any]) -> str:
    action = info.get("action")
    if action is None:
        return str(info.get("error", ""))
    try:
        return json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(action)


def _players_for(state: Any, player_id: int) -> tuple[Any, Any]:
    if state.p1.user_id == int(player_id):
        return state.p1, state.p2
    return state.p2, state.p1


def _find_target_card(me: Any, enemy: Any, target_id: str | None) -> CardInstance | None:
    return _find_card_by_instance_id([me.hero, enemy.hero, *me.board, *enemy.board], target_id)


def _find_card_by_instance_id(cards: list[CardInstance], instance_id: str | None) -> CardInstance | None:
    if not instance_id:
        return None
    target = str(instance_id)
    for card in cards:
        if str(card.instance_id) == target:
            return card
    return None


def _extract_chat_content(response: dict[str, Any]) -> str:
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _finish_reason(response: dict[str, Any]) -> str:
    try:
        return str(response.get("choices", [{}])[0].get("finish_reason", ""))
    except Exception:
        return ""


def _compact_response_for_log(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list):
        compact_choices = []
        for choice in choices[:2]:
            if isinstance(choice, dict):
                msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                compact_choices.append(
                    {
                        "finish_reason": choice.get("finish_reason"),
                        "message_keys": sorted(msg.keys()) if isinstance(msg, dict) else [],
                        "content_preview": str(msg.get("content", ""))[:200] if isinstance(msg, dict) else "",
                    }
                )
        choices = compact_choices
    return {"id": response.get("id"), "choices": choices, "usage": response.get("usage")}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _parse_decision(text: str, legal: list[int]) -> TurnDecision | None:
    payload = _parse_json_object(text)
    if not isinstance(payload, dict):
        return None
    try:
        action_id = int(payload.get("action_id"))
    except Exception:
        return None
    if action_id not in set(int(x) for x in legal):
        return None
    return TurnDecision(
        action_id=action_id,
        battle_line=str(payload.get("battle_line", payload.get("taunt", payload.get("reason", ""))))[:300],
        raw=payload,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except Exception:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                return None
    return None


def _fallback_action(legal: list[int]) -> int:
    legal = [int(x) for x in legal]
    return 0 if 0 in legal else legal[0] if legal else 0


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def decision_to_json(decision: TurnDecision) -> dict[str, Any]:
    return {
        "action_id": int(decision.action_id),
        "battle_line": decision.battle_line,
        "fallback": bool(decision.fallback),
        "error": decision.error,
        "raw": decision.raw,
    }


def _system_prompt(
    *,
    participant: str,
    guide: str,
    catalog: str,
    games: int,
    opponent_meta: dict[str, Any],
    via_codex_cli: bool,
) -> str:
    return f"""You are {participant}, a frontier LLM playing Extra Arena for fun against Extra-LR-V4-Max.

You will play exactly {games} games. All games for you share one persistent context, so learn from previous game results.
Each individual game must also stay in this same context. The opponent is exactly Extra-LR-V4-Max:
{json.dumps(opponent_meta, ensure_ascii=False, sort_keys=True)}

Thinking mode is enabled: think privately and carefully before choosing, but never reveal chain-of-thought.
Return strict JSON only: {{"action_id": <legal integer>, "battle_line": "<short optional line>"}}.
You may only choose an action_id from the provided legal_actions list.
{"You are being called via Codex CLI, not via the external provider key." if via_codex_cli else "You are being called through an OpenAI-compatible API provider."}

=== SHOWMATCH GUIDE ===
{guide}

=== CARD CATALOG CSV ===
{catalog}
"""


def _system_message(
    *,
    participant: str,
    guide: str,
    catalog: str,
    games: int,
    opponent_meta: dict[str, Any],
    via_codex_cli: bool,
) -> dict[str, Any]:
    prompt = _system_prompt(
        participant=participant,
        guide=guide,
        catalog=catalog,
        games=games,
        opponent_meta=opponent_meta,
        via_codex_cli=via_codex_cli,
    )
    if participant.startswith("anthropic/"):
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    return {"role": "system", "content": prompt}


def _cache_strategy_for_model(model: str) -> str:
    if model.startswith("deepseek/"):
        return "full_static_context_auto_cache_deepseek"
    if model.startswith("google/"):
        return "full_static_context_auto_cache_google"
    if model.startswith("anthropic/"):
        return "full_static_context_manual_cache_control"
    return "compact_unknown_provider"


def _compact_guide() -> str:
    return """Extra Arena showmatch rules:
- Choose exactly one legal action_id from legal_actions.
- Return strict JSON only: {"action_id": <int>, "battle_line": "<short optional line>"}.
- Win by reducing the enemy hero to 0 HP. If both heroes die, draw.
- Board limit is 7, hand limit is 4, mana grows up to 10.
- Play warriors to board; most cannot attack until later unless they have charge.
- Potions resolve immediately and go to graveyard.
- Taunt must be attacked first unless attacker has bypass_taunt.
- Shield blocks one harmful damage/effect. Armor reduces incoming damage.
- Check lethal first, then prevent losing, then remove taunt/threats, then develop board.
The current turn payload is authoritative for state and legal actions."""


def _read_mechanics_glossary(path: Path) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    seen: dict[str, str] = {}
    for row in rows:
        raw = row.get("mechanics", "[]")
        try:
            mechanics = json.loads(raw)
        except Exception:
            mechanics = []
        explanation = row.get("mechanics_explained", "")
        for mechanic in mechanics:
            if mechanic and mechanic not in seen:
                seen[mechanic] = explanation
    lines = ["mechanic,meaning"]
    for mechanic, meaning in sorted(seen.items()):
        lines.append(f"{mechanic},{meaning.replace(chr(10), ' ')}")
    return "\n".join(lines)


def _read_catalog_compact(path: Path) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fields = ["id", "name", "card_type", "mana_cost", "base_attack", "base_hp", "mechanics", "mechanics_explained", "targeting", "tactical_notes"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "")).replace("\n", " ") for field in fields))
    return "\n".join(lines)


def _compact_game_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_index": result.get("game_index"),
        "seed": result.get("seed"),
        "llm_model": result.get("llm_model"),
        "llm_player_id": result.get("llm_player_id"),
        "starting_player_id": result.get("starting_player_id"),
        "winner_name": result.get("winner_name"),
        "llm_win": result.get("llm_win"),
        "draw": result.get("draw"),
        "steps": result.get("steps"),
        "turn_number": result.get("turn_number"),
        "llm_hp": result.get("llm_hp"),
        "v4_hp": result.get("v4_hp"),
        "invalid_actions": result.get("invalid_actions"),
        "llm_fallbacks": result.get("llm_fallbacks"),
    }



def _opponent_meta(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "name": "Extra-LR-V4-Max",
        "onnx_path": str(path.resolve()),
        "onnx_sha256": hashlib.sha256(data).hexdigest(),
        "policy_class": "ai.train_v2.onnx_policy.OnnxActionPolicy",
        "policy_mode": "argmax",
    }


def _validate_v4max(path: Path) -> None:
    if path.name != "extra-lr-v4-max.onnx":
        raise ValueError(f"opponent must be extra-lr-v4-max.onnx, got {path}")
    if not path.exists():
        raise FileNotFoundError(path)


def _resolve_api_key() -> str:
    return os.environ.get("SHOWMATCH_API_KEY") or os.environ.get("NANO_GPT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def _resolve_participants(raw: str) -> list[str]:
    if raw == "all":
        return [*DEFAULT_API_MODELS, CODEX_PARTICIPANT]
    if raw == "api":
        return list(DEFAULT_API_MODELS)
    if raw == "codex":
        return [CODEX_PARTICIPANT]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _participant_dir(run_dir: Path, participant: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", participant)
    return run_dir / safe


def _summarize_games(name: str, games: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(games)
    llm_wins = sum(1 for game in games if game["llm_win"])
    draws = sum(1 for game in games if game["draw"])
    v4_wins = n - llm_wins - draws
    return {
        "participant": name,
        "games": n,
        "llm_wins": llm_wins,
        "v4_wins": v4_wins,
        "draws": draws,
        "llm_winrate": llm_wins / n if n else 0.0,
        "avg_steps": sum(int(game["steps"]) for game in games) / n if n else 0.0,
        "invalid_actions": sum(int(game["invalid_actions"]) for game in games),
        "llm_fallbacks": sum(int(game["llm_fallbacks"]) for game in games),
        "games_detail": [
            {
                key: game[key]
                for key in (
                    "game_index",
                    "seed",
                    "llm_player_id",
                    "starting_player_id",
                    "winner_name",
                    "steps",
                    "turn_number",
                    "llm_hp",
                    "v4_hp",
                    "llm_fallbacks",
                )
            }
            for game in games
        ],
    }


def _format_participant_line(summary: dict[str, Any]) -> str:
    return (
        f"{summary['participant']}: "
        f"{summary['llm_wins']}-{summary['v4_wins']}-{summary['draws']} "
        f"(winrate={summary['llm_winrate']:.3f}, fallbacks={summary['llm_fallbacks']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frontier LLM showmatches vs Extra-LR-V4-Max.")
    parser.add_argument("--participants", default="all", help="'all', 'api', 'codex', or comma-separated model names")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--codex-timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means omit max_tokens and let provider/model decide.")
    parser.add_argument("--repair-max-tokens", type=int, default=0, help="0 means omit max_tokens and let provider/model decide.")
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--guide", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--card-catalog", type=Path, default=DEFAULT_CARD_CATALOG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-thinking", action="store_true", help="Disable reasoning_effort request field and thinking prompt note.")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_showmatches(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
