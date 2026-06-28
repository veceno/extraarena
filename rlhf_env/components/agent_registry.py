"""AgentRegistry — кодовые имена «играющих» суб-агентов оркестратора.

Оркестратор полу-синтетических боёв пинит цепочку боев (серию) за конкретным
суб-агентом и даёт ему кодовое имя (codename). Пул имён:

  1. Фиксированный список («крутые» кодовые имена):
     Veceno, Mentalist, Pvwell, Sinaf, Movi, Ilya, Oguzok, Milita, dranik,
     sukunyata, absolute.
  2. Названия карт из ai/cards.json (50 шт).
  3. Random-fallback ``Agent-<hex>`` когда пул исчерпан.

In-memory ``_busy: Dict[name, {group_id, claimed_at}]`` + persist
``sessions/agents_index.json`` (атомарно tmp+rename, threading.Lock).

Проброс: spec.agent_name → create_series → _GroupLive/ArenaMatch → manifest
(top-level + per-battle) → v5/meta.json. finish_series освобождает имя.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # Unix (darwin/linux) — cross-process файловый lock.
    import fcntl as _fcntl
    _HAVE_FCNTL = True
except Exception:  # noqa: BLE001 — Windows/no-fcntl → no-op lock (in-process only).
    _fcntl = None
    _HAVE_FCNTL = False

logger = logging.getLogger(__name__)


# Фиксированный пул кодовых имён (приоритет 1).
_FIXED_CODENAMES: List[str] = [
    "Veceno", "Mentalist", "Pvwell", "Sinaf", "Movi", "Ilya",
    "Oguzok", "Milita", "dranik", "sukunyata", "absolute",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_card_names(cards_path: Path | str) -> List[str]:
    """Имена карт из ai/cards.json (приоритет 2 пула). Молча [] при ошибке."""
    try:
        data = json.loads(Path(cards_path).read_text(encoding="utf-8"))
        cards = data if isinstance(data, list) else data.get("cards", [])
        out: List[str] = []
        seen: set[str] = set()
        for c in cards:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if not name or not isinstance(name, str):
                continue
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(name.strip())
        return out
    except Exception:  # noqa: BLE001
        return []


def _build_codename_pool(cards_path: Path | str) -> List[str]:
    """Пул = fixed list + имена карт (дедуп case-insensitive, фиксированные сначала)."""
    pool: List[str] = []
    seen: set[str] = set()
    for name in _FIXED_CODENAMES + _load_card_names(cards_path):
        key = name.lower()
        if key not in seen:
            seen.add(key)
            pool.append(name)
    return pool


class AgentRegistry:
    """Реестр кодовых имён суб-агентов + persist + status-aggregate из манифеста."""

    def __init__(
        self,
        index_path: Path | str,
        *,
        sessions_dir: Optional[Path | str] = None,
        cards_path: Path | str = "ai/cards.json",
    ) -> None:
        self.index_path = Path(index_path)
        self.sessions_dir = Path(sessions_dir) if sessions_dir else self.index_path.parent
        self._pool = _build_codename_pool(cards_path)
        self._busy: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # F9: при старте терпим испорченный/нечитаемый index — стартуем empty,
        # чтобы mcp_server не падал; мутации всё равно пробросят ошибку _load.
        try:
            self._load()
        except Exception:  # noqa: BLE001
            logger.warning("[AgentRegistry] init load failed, starting empty: %s", exc_info=True)
            self._busy = {}

    # -- persist -----------------------------------------------------------

    def _load(self) -> None:
        # F9: НЕ сбрасываем _busy в {} молча при ошибке чтения существующего
        # файла — иначе следующая мутация persist-ит пустой/одноэлементный index
        # и стирает чужие записи. Пробрасываем исключение; __init__ ловит.
        if not self.index_path.exists():
            self._busy = {}
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            self._busy = {k: v for k, v in data.items() if isinstance(v, dict)}
        else:
            self._busy = {}

    def _persist(self) -> None:
        # F8: НЕ проглатываем — пробрасываем, чтобы claim/claim_auto/pin_group
        # не сообщали об успехе когда на диске ничего не записано (иначе другой
        # процесс _load-ит stale и клеймит то же имя = дивергенция BUG2).
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._busy, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.index_path)

    # -- cross-process lock (BUG2) -----------------------------------------
    # threading.Lock сериализует в пределах процесса; fcntl.flock — между
    # процессами (несколько mcp_server stdio на одном sessions-dir = несколько
    # процессов, иначе last-write-wins терял записи/кодовые имена коллизионили).

    @property
    def _lock_path(self) -> Path:
        return self.index_path.with_suffix(self.index_path.suffix + ".lock")

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self._lock_path, "a+", encoding="utf-8")
        try:
            if _HAVE_FCNTL:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
            yield
        finally:
            if _HAVE_FCNTL:
                try:
                    _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
                except Exception:  # noqa: BLE001
                    pass
            f.close()

    def _is_free(self, name: str) -> bool:
        """Имя свободно: записи нет ИЛИ она помечена finished (BUG6: history kept)."""
        info = self._busy.get(name)
        return info is None or bool(info.get("finished"))

    def _self_heal_locked(self, name: str, info: Optional[Dict[str, Any]]) -> None:
        """Под lock + после _load: если имя не finished, но серия его группы
        доиграна/финализирована (манифест на диске) — помечаем finished + persist.

        Cross-process recovery: agents_index.json персистится across processes,
        а manager._groups/_matches — in-memory. Упавший процесс (краш клиента
        mid-series) оставляет busy-запись, которую НОВЫЙ процесс не может
        освободить через manager.reap_completed (нет _groups-записи). Registry
        сам читает манифест группы и освобождает имя при первом обращении к нему
        (is_busy/claim/claim_auto/status). Без этого codename-pool истощался
        навсегда после любого краша клиента. Манифест уже финализирован
        (finished_at) или battles_finished >= battles_planned → серия завершена.
        """
        if info is None or info.get("finished"):
            return
        gid = info.get("group_id")
        if not gid:
            return
        man = self._read_manifest(gid)
        if man is None:
            return
        res = man.get("results", {}) or {}
        finished_battles = int(res.get("battles_finished", 0) or 0)
        planned = int(res.get("battles_planned", 0) or 0)
        if man.get("finished_at") or (planned and finished_battles >= planned):
            info["finished"] = True
            info["released_at"] = _utc_now_iso()
            try:
                self._persist()
                logger.info("[AgentRegistry] self-heal released stale busy name=%s group=%s", name, gid)
            except Exception:  # noqa: BLE001
                logger.warning("[AgentRegistry] self-heal persist failed %s: %s", name, exc_info=True)

    # -- claim / release ---------------------------------------------------

    def _claim_locked(self, name: str) -> bool:
        """Предполагает удержание _lock + _file_lock и уже сделанный _load.
        Клеймит имя (если свободно) и persist-ит. True = успешно.
        F8: при ошибке persist откатывает in-memory запись и возвращает False
        (claim/claim_auto сообщают неудачу, а не «успех без записи на диске»)."""
        if not self._is_free(name):
            return False
        self._busy[name] = {"group_id": None, "claimed_at": _utc_now_iso()}
        try:
            self._persist()
        except Exception:  # noqa: BLE001
            self._busy.pop(name, None)
            logger.warning("[AgentRegistry] persist failed for %s: %s", name, exc_info=True)
            return False
        return True

    def claim(self, name: str) -> bool:
        """Явное имя. True если свободно (или finished — переиспользование), False если занято."""
        name = str(name).strip()
        if not name:
            return False
        try:
            with self._lock, self._file_lock():
                self._load()  # перечитываем диск — могли писать другие процессы
                self._self_heal_locked(name, self._busy.get(name))
                return self._claim_locked(name)
        except Exception:  # noqa: BLE001 — _load/_file_lock упали → клейм не состоялся
            logger.warning("[AgentRegistry] claim(%s) failed: %s", name, exc_info=True)
            return False

    def claim_auto(self) -> str:
        """Auto-assign: первый свободный из пула, иначе random-fallback (F10:
        всегда проверяем _is_free — последний fallback не должен перезаписывать
        активную запись). F8: при ошибке persist пробуем следующего кандидата."""
        with self._lock, self._file_lock():
            self._load()
            for name in self._pool:
                # self-heal: имя может быть занято «призраком» упавшего процесса
                # (agents_index busy, но серия уже завершена) — освободим перед
                # проверкой, иначе пул истощается leaked-именами.
                self._self_heal_locked(name, self._busy.get(name))
                if self._claim_locked(name):
                    return name
            # пул исчерпан → random-fallback; цикл до свободного (F10).
            for _ in range(32):
                name = f"Agent-{secrets.token_hex(8)}"
                if self._claim_locked(name):
                    return name
        raise RuntimeError("claim_auto: cannot claim a free codename (persist failing?)")

    def release(self, name: str, *, purge: bool = False) -> None:
        """Освобождает имя. По умолчанию (BUG6) помечает finished — история
        доступна через status() после finish_series. purge=True — полное
        удаление (для rollback провалившегося create_series)."""
        with self._lock, self._file_lock():
            self._load()
            info = self._busy.get(name)
            if info is None:
                return
            if purge:
                self._busy.pop(name, None)
            else:
                if info.get("finished"):
                    return  # идемпотентно
                info["finished"] = True
                info["released_at"] = _utc_now_iso()
            self._persist()

    def release_group(self, group_id: str, *, purge: bool = False) -> None:
        # F2: атомарно — одна критическая секция под обоими lock'ами и один
        # _persist (раньше отпускали lock и звали self.release по имени —
        # между enumerate и release другой процесс мог pin_group имя в новую
        # группу, и purge=True затирал чужой клейм; mid-loop crash оставлял
        # частичный rollback). re-check group_id/finished под lock'ом.
        with self._lock, self._file_lock():
            self._load()
            for n, info in list(self._busy.items()):
                if info.get("group_id") != group_id:
                    continue
                if purge:
                    self._busy.pop(n, None)
                elif not info.get("finished"):
                    info["finished"] = True
                    info["released_at"] = _utc_now_iso()
            self._persist()

    def is_busy(self, name: str) -> bool:
        with self._lock, self._file_lock():
            self._load()
            info = self._busy.get(name)
            self._self_heal_locked(name, info)
            return info is not None and not info.get("finished")

    def pin_group(self, name: str, group_id: str) -> None:
        with self._lock, self._file_lock():
            self._load()
            info = self._busy.get(name)
            if info is None:
                # имя не было claim — claim неявно (явный agent_name из spec).
                info = {"group_id": None, "claimed_at": _utc_now_iso()}
                self._busy[name] = info
            info["group_id"] = group_id
            info.pop("finished", None)
            info.pop("released_at", None)
            self._persist()

    def group_of(self, name: str) -> Optional[str]:
        with self._lock, self._file_lock():
            self._load()
            info = self._busy.get(name)
            return info.get("group_id") if info else None

    # -- status / list -----------------------------------------------------

    def _read_manifest(self, group_id: str) -> Optional[Dict[str, Any]]:
        if not group_id:
            return None
        try:
            mp = self.sessions_dir / group_id / "manifest.json"
            if mp.exists():
                return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return None

    def _status_from_info(self, name: str, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Статус из уже загруженного info (без повторного _load/lock — F6).
        manifest читается с диска (per-group, вне registry-index) — это OK
        делать без lock'а индекса."""
        if info is None:
            return {"agent_name": name, "busy": False, "group_id": None,
                    "battles_finished": 0, "battles_planned": 0,
                    "wins": 0, "losses": 0, "draws": 0,
                    "decks": {"p1": None, "p2": None},
                    "opponent_model": None, "p1_actor_type": None,
                    "current_match_id": None, "status": "unknown"}
        group_id = info.get("group_id")
        finished = bool(info.get("finished"))
        out: Dict[str, Any] = {
            "agent_name": name,
            "busy": not finished,
            "group_id": group_id,
            "claimed_at": info.get("claimed_at"),
            "released_at": info.get("released_at"),
        }
        man = self._read_manifest(group_id) if group_id else None
        if man:
            res = man.get("results", {}) or {}
            spec = man.get("spec", {}) or {}
            # wins/losses: p1 — это агент (p1_actor_type rl/llm/human). losses = p2_wins.
            out["battles_finished"] = res.get("battles_finished", 0)
            out["battles_planned"] = res.get("battles_planned", 0)
            out["wins"] = res.get("p1_wins", 0)
            out["losses"] = res.get("p2_wins", 0)
            out["draws"] = res.get("draws", 0)
            out["winrate"] = res.get("winrate_p1", 0.0)
            out["opponent_model"] = spec.get("p2_model")
            out["p1_actor_type"] = spec.get("p1_actor_type")
            out["decks"] = {
                "p1": spec.get("custom_deck_p1") or spec.get("deck_strategy_p1"),
                "p2": spec.get("custom_deck_p2") or spec.get("deck_strategy_p2"),
            }
            out["finished_at"] = man.get("finished_at")
            out["status"] = "completed" if man.get("finished_at") else "running"
        else:
            out["battles_finished"] = 0
            out["battles_planned"] = 0
            out["wins"] = 0
            out["losses"] = 0
            out["draws"] = 0
            out["decks"] = {"p1": None, "p2": None}
            out["opponent_model"] = None
            out["p1_actor_type"] = None
            out["status"] = "completed" if finished else "registered"
        return out

    def status(self, name: str) -> Dict[str, Any]:
        """Полный статус агента: registry-info + aggregate из манифеста группы.

        BUG6: после release имя остаётся в index помеченным finished. status()
        всё равно возвращает историю (читает манифест по group_id), но busy=False.
        Так оркестратор видит итог завершённой серии, а не пустой ответ.
        """
        with self._lock, self._file_lock():
            self._load()
            info = self._busy.get(name)
            self._self_heal_locked(name, info)
            # F6: снимаем снапшот info под lock'ом и считаем статус из него —
            # не пере-заходим в status() снаружи (где имя могли purge'нуть).
            return self._status_from_info(name, info)

    def list_active(self) -> List[Dict[str, Any]]:
        with self._lock, self._file_lock():
            self._load()
            # F6: атомарный снапшот — считаем статусы прямо под lock'ом из уже
            # загруженного _busy, а не зовём self.status() по имени снаружи
            # (между enumerate и status имя могли purge/claim в другом процессе).
            return [
                self._status_from_info(n, info)
                for n, info in self._busy.items()
                if not info.get("finished")
            ]


__all__ = ["AgentRegistry", "_FIXED_CODENAMES"]