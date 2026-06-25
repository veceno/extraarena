"""Манифест группы боёв для RLHF-среды.

Создаёт JSON-файл `manifest.json` в директории группы боёв и обновляет
его по мере завершения отдельных матчей. Также пишет `summary.json`
с финальной агрегированной статистикой.

Формат manifest.json см. в плане проекта / DOCS.md.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlhf_env.components.log_schema import MANIFEST_VERSION, validate_manifest

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_git_commit(cwd: Path | str) -> str:
    """Возвращает короткий git-commit текущего репо или "unknown"."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _safe_package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


class ManifestWriter:
    """Записывает и обновляет manifest.json + summary.json для группы боёв."""

    def __init__(
        self,
        *,
        group_id: str,
        spec: Dict[str, Any],
        group_dir: Path | str,
        repo_root: Optional[Path | str] = None,
        rlhf_version: str = "0.1.0",
    ):
        self.group_id = group_id
        self.spec = spec
        self.group_dir = Path(group_dir)
        self.group_dir.mkdir(parents=True, exist_ok=True)
        (self.group_dir / "battles").mkdir(exist_ok=True)

        self.manifest_path = self.group_dir / "manifest.json"
        self.summary_path = self.group_dir / "summary.json"

        # Если уже есть — загружаем (для resume/append)
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if "battles_results" not in self.manifest:
                    self.manifest["battles_results"] = []
                return
            except Exception as exc:
                logger.warning("[ManifestWriter] bad existing manifest, recreating: %s", exc)

        self.manifest = {
            "manifest_version": MANIFEST_VERSION,
            "group_id": group_id,
            "created_at": _utc_now_iso(),
            "finished_at": None,
            # spec хранится в манифесте как метаданные серии. Убираем transient
            # UI-настройки (audio — выключатели музыки/SFX из меню среды), которые
            # не относятся к боевым параметрам и не должны попадать в записи.
            "spec": {k: v for k, v in spec.items() if k != "audio"},
            "env": self._build_env_info(repo_root or Path.cwd(), rlhf_version),
            "results": {
                "battles_finished": 0,
                "battles_planned": int(spec.get("battles_planned", 0)),
                "p1_wins": 0,
                "p2_wins": 0,
                "draws": 0,
                "winrate_p1": 0.0,
                "winrate_p2": 0.0,
                "avg_turns": 0.0,
                "avg_duration_seconds": 0.0,
            },
            "battle_ids": [],
            "battles_results": [],
        }
        self._flush()

    @staticmethod
    def _build_env_info(repo_root: Path | str, rlhf_version: str) -> Dict[str, Any]:
        return {
            "rlhf_env_version": rlhf_version,
            "core_engine_commit": _safe_git_commit(repo_root),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime_version": _safe_package_version("onnxruntime"),
            "numpy_version": _safe_package_version("numpy"),
            "aiohttp_version": _safe_package_version("aiohttp"),
        }

    def append_battle_result(
        self,
        *,
        battle_id: str,
        battle_log_path: str,
        winner_user_id: Optional[int],
        loser_user_id: Optional[int],
        status: str,
        turns: int,
        duration_seconds: float,
        v5_dir: Optional[str] = None,
        v5_meta_path: Optional[str] = None,
        decks_cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Добавляет результат одного боя и обновляет агрегаты."""
        result = {
            "battle_id": battle_id,
            "battle_log_path": battle_log_path,
            "winner_user_id": winner_user_id,
            "loser_user_id": loser_user_id,
            "status": status,
            "turns": int(turns),
            "duration_seconds": float(duration_seconds),
        }
        if v5_dir is not None:
            result["v5_dir"] = v5_dir
        if v5_meta_path is not None:
            result["v5_meta_path"] = v5_meta_path
        if decks_cache is not None:
            # ДЕНОРАЛИЗОВАННЫЙ кэш колод для быстрого lookup при батч-обучении;
            # АВТОРИТАТИВНЫЕ resolved-колоды лежат в v5/meta.json, а в manifest.spec
            # только стратегия колоды (для random стратегий resolved card_ids здесь).
            result["decks_cache"] = decks_cache
        self.manifest["battles_results"].append(result)
        if battle_id not in self.manifest["battle_ids"]:
            self.manifest["battle_ids"].append(battle_id)

        # обновляем агрегаты
        self.manifest["results"]["battles_finished"] = len(self.manifest["battles_results"])
        if status == "P1_WIN":
            self.manifest["results"]["p1_wins"] += 1
        elif status == "P2_WIN":
            self.manifest["results"]["p2_wins"] += 1
        elif status == "DRAW":
            self.manifest["results"]["draws"] += 1

        finished = self.manifest["results"]["battles_finished"]
        if finished > 0:
            self.manifest["results"]["winrate_p1"] = round(
                self.manifest["results"]["p1_wins"] / finished, 4
            )
            self.manifest["results"]["winrate_p2"] = round(
                self.manifest["results"]["p2_wins"] / finished, 4
            )
            self.manifest["results"]["avg_turns"] = round(
                sum(b["turns"] for b in self.manifest["battles_results"]) / finished, 2
            )
            self.manifest["results"]["avg_duration_seconds"] = round(
                sum(b["duration_seconds"] for b in self.manifest["battles_results"]) / finished, 2
            )

        self._flush()

        # Авто-финализация: когда все запланированные бои серии записаны,
        # манифест закрывается сам (finished_at + summary.json) — не ждём
        # отдельного next_match/finish. Иначе серия, сыгранная до конца,
        # оставалась в статусе «running»: finalize() вызывался только из
        # next_match (arena_match_manager.next_match), а человек после
        # последнего боя жмёт «Завершить» (выход в меню), а не «Следующий бой».
        planned = int(self.manifest["results"].get("battles_planned", 0) or 0)
        finished = int(self.manifest["results"].get("battles_finished", 0) or 0)
        if planned > 0 and finished >= planned and not self.manifest.get("finished_at"):
            self.finalize()

    def finalize(self) -> Dict[str, Any]:
        """Закрывает манифест: ставит finished_at и пишет summary.json.

        Идемпотентно: повторный вызов (напр. из finish_series после авто-
        финализации) ничего не перезаписывает — серия остаётся с первым
        finished_at. Иначе «Завершить» после сыгранного до конца боя двигал
        бы finished_at на более позднее время.
        """
        if self.manifest.get("finished_at"):
            return self.manifest
        self.manifest["finished_at"] = _utc_now_iso()
        self._flush()

        summary = {
            "group_id": self.group_id,
            "battles_finished": self.manifest["results"]["battles_finished"],
            "winrate_p1": self.manifest["results"]["winrate_p1"],
            "winrate_p2": self.manifest["results"]["winrate_p2"],
            "draws": self.manifest["results"]["draws"],
            "avg_turns": self.manifest["results"]["avg_turns"],
            "avg_duration_seconds": self.manifest["results"]["avg_duration_seconds"],
            "finished_at": self.manifest["finished_at"],
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.manifest

    def _flush(self) -> None:
        """Атомарная запись manifest.json (через tmp + rename)."""
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, self.manifest_path)

    @property
    def status(self) -> str:
        if self.manifest["finished_at"]:
            return "completed"
        return "running"


__all__ = ["ManifestWriter", "_utc_now_iso"]