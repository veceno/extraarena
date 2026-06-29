"""Offline OpenAI-compatible teacher labeling for V5 hard states."""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class OpenAICompatibleTeacherConfig:
    base_url: str = "https://api.openai.com"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    prompt_version: str = "v5-teacher-v1"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    cache_dir: str | Path | None = None

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"


@dataclass(frozen=True)
class TeacherPreferenceRow:
    state_sha256: str
    legal_action_ids: list[int]
    preferred_action_id: int
    action_ranking: list[int]
    rationale_summary: str
    confidence: float
    model: str
    provider_base_url: str
    prompt_version: str

    def validate(self) -> "TeacherPreferenceRow":
        if not self.state_sha256:
            raise ValueError("state_sha256 must not be empty")
        legal = [int(item) for item in self.legal_action_ids]
        if not legal:
            raise ValueError("legal_action_ids must not be empty")
        if int(self.preferred_action_id) not in set(legal):
            raise ValueError("preferred_action_id must be in legal_action_ids")
        ranking = [int(item) for item in self.action_ranking]
        if not ranking or ranking[0] != int(self.preferred_action_id):
            raise ValueError("action_ranking must start with preferred_action_id")
        if any(action_id not in set(legal) for action_id in ranking):
            raise ValueError("action_ranking entries must be legal")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeacherPreferenceRow":
        return cls(
            state_sha256=str(data["state_sha256"]),
            legal_action_ids=[int(item) for item in data["legal_action_ids"]],
            preferred_action_id=int(data["preferred_action_id"]),
            action_ranking=[int(item) for item in data["action_ranking"]],
            rationale_summary=str(data.get("rationale_summary", "")),
            confidence=float(data.get("confidence", 0.0)),
            model=str(data["model"]),
            provider_base_url=str(data["provider_base_url"]),
            prompt_version=str(data["prompt_version"]),
        ).validate()


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class OpenAICompatibleTeacherClient:
    def __init__(
        self,
        config: OpenAICompatibleTeacherConfig | None = None,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
    ):
        self.config = config or OpenAICompatibleTeacherConfig()
        self.api_key = api_key if api_key is not None else os.environ.get(self.config.api_key_env, "")
        self.transport = transport or self._default_transport

    def label_state(
        self,
        *,
        state_sha256: str,
        legal_action_ids: list[int],
        state_summary: dict[str, Any],
    ) -> TeacherPreferenceRow:
        cache_path = self._cache_path(state_sha256)
        if cache_path is not None and cache_path.exists():
            return TeacherPreferenceRow.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

        request = self._build_request(
            state_sha256=state_sha256,
            legal_action_ids=legal_action_ids,
            state_summary=state_summary,
        )
        response = self._call_with_retries(request)
        row = self._parse_response(
            response,
            state_sha256=state_sha256,
            legal_action_ids=legal_action_ids,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(row.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        return row

    def cache_key(self, state_sha256: str) -> str:
        payload = {
            "state_sha256": str(state_sha256),
            "prompt_version": self.config.prompt_version,
            "model": self.config.model,
            "base_url": self.config.base_url.rstrip("/"),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, state_sha256: str) -> Path | None:
        if self.config.cache_dir is None:
            return None
        return Path(self.config.cache_dir) / f"{self.cache_key(state_sha256)}.json"

    def _build_request(
        self,
        *,
        state_sha256: str,
        legal_action_ids: list[int],
        state_summary: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_payload = {
            "state_sha256": state_sha256,
            "legal_action_ids": [int(item) for item in legal_action_ids],
            "state_summary": state_summary,
            "response_schema": {
                "preferred_action_id": "int legal action id",
                "action_ranking": "list[int] legal action ids",
                "confidence": "float in [0, 1]",
                "rationale_summary": "short string",
            },
        }
        return {
            "url": self.config.chat_completions_url,
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            "timeout": float(self.config.timeout_seconds),
            "json": {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an offline Extra-LR V5 tactical teacher. "
                            "Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, sort_keys=True)},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        }

    def _call_with_retries(self, request: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(max(1, int(self.config.max_retries) + 1)):
            try:
                return self.transport(request)
            except Exception as exc:  # pragma: no cover - exercised by integration callers.
                last_error = exc
        raise RuntimeError(f"teacher request failed after retries: {last_error}") from last_error

    def _parse_response(
        self,
        response: dict[str, Any],
        *,
        state_sha256: str,
        legal_action_ids: list[int],
    ) -> TeacherPreferenceRow:
        try:
            content = response["choices"][0]["message"]["content"]
            payload = _parse_json_object_content(str(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("teacher response must contain JSON object content") from exc
        return TeacherPreferenceRow(
            state_sha256=str(state_sha256),
            legal_action_ids=[int(item) for item in legal_action_ids],
            preferred_action_id=int(payload["preferred_action_id"]),
            action_ranking=_normalize_action_ranking(
                preferred_action_id=int(payload["preferred_action_id"]),
                action_ranking=[int(item) for item in payload.get("action_ranking", [])],
                legal_action_ids=[int(item) for item in legal_action_ids],
            ),
            rationale_summary=str(payload.get("rationale_summary", "")),
            confidence=float(payload.get("confidence", 0.0)),
            model=self.config.model,
            provider_base_url=self.config.base_url.rstrip("/"),
            prompt_version=self.config.prompt_version,
        ).validate()

    def _default_transport(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(f"missing API key env {self.config.api_key_env}")
        data = json.dumps(request["json"]).encode("utf-8")
        req = urllib.request.Request(
            request["url"],
            data=data,
            headers=request["headers"],
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(request["timeout"]), context=_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"teacher HTTP request failed: {exc}") from exc


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _parse_json_object_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("teacher content JSON must be an object", text, 0)
    return payload


def _normalize_action_ranking(
    *,
    preferred_action_id: int,
    action_ranking: list[int],
    legal_action_ids: list[int],
) -> list[int]:
    legal = [int(item) for item in legal_action_ids]
    legal_set = set(legal)
    preferred = int(preferred_action_id)
    if preferred not in legal_set:
        return [preferred]
    normalized = [preferred]
    for action_id in action_ranking:
        aid = int(action_id)
        if aid in legal_set and aid not in normalized:
            normalized.append(aid)
    return normalized


def save_teacher_preferences(rows: list[TeacherPreferenceRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return out


def load_teacher_preferences(path: str | Path) -> list[TeacherPreferenceRow]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(TeacherPreferenceRow.from_dict(json.loads(line)))
    return rows


__all__ = [
    "OpenAICompatibleTeacherClient",
    "OpenAICompatibleTeacherConfig",
    "TeacherPreferenceRow",
    "load_teacher_preferences",
    "save_teacher_preferences",
]
