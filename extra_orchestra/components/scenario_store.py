"""Сохранение/загрузка сценариев в ``scenarios/*.json`` (files-only, без БД).

Имя файла = slug имени сценария (или ``id`` из JSON, если задан). Атомарная
запись через tmp+replace. Никаких внешних зависимостей.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SLUG_RE = re.compile(r"[^\w]+", re.UNICODE)
SCHEMA = "extra_orchestra.scenario.v1"
V2_SCHEMA = "extra_orchestra.scenario.v2"


def slugify(name: str) -> str:
    # preserve unicode letters/digits (incl. Cyrillic), drop spaces/punct
    s = SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return s or "scenario"


class ScenarioStore:
    def __init__(self, scenarios_dir: Path) -> None:
        self.dir = Path(scenarios_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, scenario: Dict[str, Any]) -> Path:
        sid = scenario.get("id")
        if sid and isinstance(sid, str) and sid.isidentifier():
            slug = sid
        else:
            slug = slugify(scenario.get("name", "scenario"))
        return self.dir / f"{slug}.json"

    def save(self, scenario: Dict[str, Any]) -> Path:
        scenario = dict(scenario)
        # не штампуем v1-схему поверх v2-тела без явного поля schema (есть graph → v2)
        if "schema" not in scenario:
            scenario["schema"] = V2_SCHEMA if scenario.get("graph") is not None else SCHEMA
        path = self._path_for(scenario)
        data = json.dumps(scenario, ensure_ascii=False, indent=2, sort_keys=False)
        # atomic write
        fd, tmp = tempfile.mkstemp(prefix=".orch-", suffix=".json", dir=str(self.dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("scenario saved: %s", path)
        return path

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.dir / f"{slugify(name)}.json"
        if not path.exists():
            # может быть передано уже с расширением; confine к basename —
            # иначе "../foo" читает файл вне scenarios/ (path traversal)
            alt = self.dir / Path(name).name
            if alt.exists():
                path = alt
            else:
                return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("bad scenario json: %s", p)
                continue
            schema = s.get("schema", SCHEMA)
            entry = {
                "id": s.get("id", p.stem),
                "name": s.get("name", p.stem),
                "schema": schema,
                "file": p.name,
                "turns": len(s.get("turns", [])),
            }
            if schema == "extra_orchestra.scenario.v2":
                entry["nodes"] = len((s.get("graph") or {}).get("nodes", []))
            out.append(entry)
        return out

    def delete(self, name: str) -> bool:
        path = self.dir / f"{slugify(name)}.json"
        if path.exists():
            path.unlink()
            return True
        return False