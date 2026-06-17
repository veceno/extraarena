from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SUPPORT_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "support" / "support.json"


class SupportJsonDatabase:
    """Permanent JSON-backed support store used when PostgreSQL is unavailable."""

    def __init__(self, path: Path | str = DEFAULT_SUPPORT_JSON_PATH) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._data = self._empty()

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        else:
            self._data = self._empty()
            await self._save()

    async def disconnect(self) -> None:
        await self._save()

    async def close(self) -> None:
        await self.disconnect()

    async def init_schema(self) -> None:
        await self._save()

    async def fetchval(self, query: str, *args) -> Any:
        return None

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        normalized = self._normalize(query)
        async with self._lock:
            if "insert into support_identities" in normalized:
                row = self._upsert_identity(args)
                await self._save_locked()
                return row
            if "insert into support_tickets" in normalized:
                row = self._insert(
                    "tickets",
                    {
                        "topic": args[0],
                        "status": args[1],
                        "account_scope": args[2],
                        "requester_identity_id": args[3],
                        "game_user_id": args[4],
                        "channel": args[5],
                        "channel_id": str(args[6]),
                        "subject": args[7],
                        "priority_score": int(args[8]),
                        "priority_tier": args[9],
                        "profile_snapshot": self._json(args[10]),
                        "metadata": self._json(args[11]),
                    },
                )
                row["last_message_at"] = row["created_at"]
                await self._save_locked()
                return row
            if "insert into support_messages" in normalized:
                row = self._insert(
                    "messages",
                    {
                        "ticket_id": args[0],
                        "identity_id": args[1],
                        "direction": args[2],
                        "body": args[3],
                        "metadata": self._json(args[4]),
                    },
                )
                self._touch_ticket(args[0])
                await self._save_locked()
                return row
            if "insert into support_delivery_outbox" in normalized:
                row = self._insert(
                    "outbox",
                    {
                        "ticket_id": args[0],
                        "message_id": args[1],
                        "channel": args[2],
                        "channel_id": str(args[3]),
                        "payload": self._json(args[4]),
                        "status": "pending",
                        "attempts": 0,
                        "next_attempt_at": self._now(),
                        "last_error": "",
                        "metadata": {},
                    },
                )
                await self._save_locked()
                return row
            if "select * from support_tickets where id" in normalized:
                return self._find("tickets", args[0])
            if "from support_tickets" in normalized and "status <> 'closed'" in normalized:
                rows = [
                    row for row in self._data["tickets"]
                    if row.get("channel") == args[0] and row.get("channel_id") == str(args[1]) and row.get("status") != "closed"
                ]
                rows.sort(key=lambda row: (row.get("last_message_at") or "", row.get("created_at") or ""), reverse=True)
                return dict(rows[0]) if rows else None
            if "insert into support_attachments" in normalized:
                row = self._insert(
                    "attachments",
                    {
                        "ticket_id": args[0],
                        "message_id": args[1],
                        "uploader_identity_id": args[2],
                        "storage_path": args[3],
                        "original_filename": args[4],
                        "content_type": args[5],
                        "sha256": args[6],
                        "size_bytes": int(args[7]),
                        "width": args[8],
                        "height": args[9],
                        "metadata": self._json(args[10]),
                    },
                )
                await self._save_locked()
                return row
            if "insert into support_auth_codes" in normalized:
                row = self._insert(
                    "auth_codes",
                    {
                        "identity_id": args[0],
                        "game_user_id": args[1],
                        "purpose": args[2],
                        "code_hash": args[3],
                        "expires_at": self._serialize_dt(args[4]),
                        "used_at": None,
                        "metadata": self._json(args[5]),
                    },
                )
                await self._save_locked()
                return row
            if "insert into support_admin_login_codes" in normalized:
                row = self._insert(
                    "admin_codes",
                    {
                        "admin_channel_id": str(args[0]),
                        "code_hash": args[1],
                        "expires_at": self._serialize_dt(args[2]),
                        "used_at": None,
                        "metadata": self._json(args[3]),
                    },
                )
                await self._save_locked()
                return row
            if "update support_auth_codes" in normalized:
                row = self._consume_code("auth_codes", code_hash=args[0], purpose=args[1])
                await self._save_locked()
                return row
            if "update support_admin_login_codes" in normalized:
                row = self._consume_code("admin_codes", code_hash=args[0])
                await self._save_locked()
                return row
            if "insert into support_audit_events" in normalized:
                row = self._insert(
                    "audit_events",
                    {
                        "admin_channel_id": args[0],
                        "ticket_id": args[1],
                        "event_type": args[2],
                        "event_data": self._json(args[3]),
                        "metadata": self._json(args[4]),
                    },
                )
                await self._save_locked()
                return row
        raise AssertionError(f"Unsupported support JSON fetchrow query: {query}")

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        normalized = self._normalize(query)
        async with self._lock:
            if "from support_delivery_outbox" in normalized:
                rows = [row for row in self._data["outbox"] if row.get("status") == "pending"]
                rows.sort(key=lambda row: row.get("created_at") or "")
                return [dict(row) for row in rows[: int(args[0])]]
            if "from support_messages" in normalized and "from support_tickets" not in normalized:
                rows = [row for row in self._data["messages"] if row.get("ticket_id") == args[0]]
                if "direction <> 'internal'" in normalized:
                    rows = [row for row in rows if row.get("direction") != "internal"]
                    return [
                        {
                            "id": row["id"],
                            "ticket_id": row["ticket_id"],
                            "direction": row["direction"],
                            "body": row["body"],
                            "created_at": row["created_at"],
                        }
                        for row in sorted(rows, key=lambda row: row.get("created_at") or "")
                    ]
                return [dict(row) for row in sorted(rows, key=lambda row: row.get("created_at") or "")]
            if "from support_attachments" in normalized:
                rows = [row for row in self._data["attachments"] if row.get("ticket_id") == args[0]]
                return [dict(row) for row in sorted(rows, key=lambda row: row.get("created_at") or "")]
            if "from support_tickets" in normalized:
                rows = []
                for ticket in self._data["tickets"]:
                    latest = self._latest_message(ticket["id"])
                    row = dict(ticket)
                    row["latest_message_body"] = latest.get("body") if latest else None
                    row["latest_message_direction"] = latest.get("direction") if latest else None
                    row["latest_message_created_at"] = latest.get("created_at") if latest else None
                    rows.append(row)
                rows.sort(
                    key=lambda row: (
                        -int(row.get("priority_score") or 0),
                        row.get("last_message_at") or "",
                        row.get("created_at") or "",
                    )
                )
                limit = int(args[0]) if args else 100
                offset = int(args[1]) if len(args) > 1 else 0
                return rows[offset: offset + limit]
        raise AssertionError(f"Unsupported support JSON fetch query: {query}")

    async def execute(self, query: str, *args) -> str:
        normalized = self._normalize(query)
        async with self._lock:
            if "update support_tickets set last_message_at" in normalized:
                self._touch_ticket(args[0])
                await self._save_locked()
                return "OK"
            if "update support_tickets set status" in normalized:
                ticket = self._find("tickets", args[1])
                if ticket:
                    ticket["status"] = args[0]
                    ticket["updated_at"] = self._now()
                await self._save_locked()
                return "OK"
            if "update support_delivery_outbox" in normalized and "status = 'sent'" in normalized:
                row = self._find("outbox", args[0])
                if row:
                    row["status"] = "sent"
                    row["attempts"] = int(row.get("attempts") or 0) + 1
                    row["updated_at"] = self._now()
                await self._save_locked()
                return "OK"
            if "update support_delivery_outbox" in normalized:
                row = self._find("outbox", args[0])
                if row:
                    row["attempts"] = int(row.get("attempts") or 0) + 1
                    row["last_error"] = str(args[1])
                    row["updated_at"] = self._now()
                await self._save_locked()
                return "OK"
        return "OK"

    def _upsert_identity(self, args: tuple[Any, ...]) -> dict[str, Any]:
        scope, external_user_id = args[0], args[1]
        for row in self._data["identities"]:
            if row.get("scope") == scope and row.get("external_user_id") == external_user_id:
                row.update(
                    {
                        "game_user_id": args[2],
                        "display_name": args[3],
                        "channel": args[4],
                        "channel_id": str(args[5]),
                        "is_verified": bool(args[6]),
                        "metadata": self._json(args[7]),
                        "updated_at": self._now(),
                    }
                )
                return dict(row)
        return self._insert(
            "identities",
            {
                "scope": scope,
                "external_user_id": external_user_id,
                "game_user_id": args[2],
                "display_name": args[3],
                "channel": args[4],
                "channel_id": str(args[5]),
                "is_verified": bool(args[6]),
                "metadata": self._json(args[7]),
            },
        )

    def _insert(self, collection: str, row: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        payload = {"id": str(uuid.uuid4()), **row, "created_at": now, "updated_at": now}
        self._data[collection].append(payload)
        return dict(payload)

    def _touch_ticket(self, ticket_id: Any) -> None:
        ticket = self._find("tickets", ticket_id)
        if ticket:
            now = self._now()
            ticket["last_message_at"] = now
            ticket["updated_at"] = now

    def _latest_message(self, ticket_id: Any) -> dict[str, Any] | None:
        rows = [row for row in self._data["messages"] if row.get("ticket_id") == ticket_id]
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows[0] if rows else None

    def _find(self, collection: str, row_id: Any) -> dict[str, Any] | None:
        for row in self._data[collection]:
            if str(row.get("id")) == str(row_id):
                return row
        return None

    def _consume_code(self, collection: str, *, code_hash: str, purpose: str | None = None) -> dict[str, Any] | None:
        now = self._now()
        for row in self._data[collection]:
            if row.get("code_hash") != code_hash or row.get("used_at"):
                continue
            if purpose is not None and row.get("purpose") != purpose:
                continue
            if str(row.get("expires_at") or "") <= now:
                continue
            row["used_at"] = now
            return dict(row)
        return None

    async def _save(self) -> None:
        async with self._lock:
            await self._save_locked()

    async def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _empty() -> dict[str, list[dict[str, Any]]]:
        return {
            "identities": [],
            "tickets": [],
            "messages": [],
            "attachments": [],
            "auth_codes": [],
            "admin_codes": [],
            "outbox": [],
            "audit_events": [],
        }

    @staticmethod
    def _json(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(str(query).lower().split())

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _serialize_dt(value: Any) -> str:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        return str(value)
