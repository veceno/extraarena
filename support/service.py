from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from support.codes import hash_support_code
from support.constants import (
    AccountScope,
    MESSAGE_INBOUND,
    MESSAGE_INTERNAL,
    MESSAGE_OUTBOUND,
    SupportStatus,
)


class SupportService:
    def __init__(self, *, support_db: Any, game_db: Any | None = None, notifier: Any | None = None) -> None:
        self.support_db = support_db
        self.game_db = game_db
        self.notifier = notifier

    async def create_ticket(
        self,
        *,
        topic: str,
        channel: str,
        channel_id: Any,
        external_user_id: str | None,
        display_name: str,
        body: str,
        game_user_id: int | None = None,
        account_scope: str | None = None,
        subject: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_snapshot = await self._profile_snapshot(game_user_id)
        effective_scope, status, priority_tier, priority_score = self._priority_for_profile(
            profile_snapshot,
            requested_scope=account_scope,
            has_game_user_id=game_user_id is not None,
        )

        identity = await self._create_identity(
            scope=effective_scope,
            external_user_id=external_user_id,
            game_user_id=game_user_id if effective_scope == AccountScope.VERIFIED else None,
            display_name=display_name,
            channel=channel,
            channel_id=channel_id,
            is_verified=effective_scope == AccountScope.VERIFIED,
            metadata=metadata or {},
        )
        ticket = await self._insert_ticket(
            topic=topic,
            status=status,
            account_scope=effective_scope,
            requester_identity_id=identity["id"],
            game_user_id=game_user_id if effective_scope == AccountScope.VERIFIED else None,
            channel=channel,
            channel_id=channel_id,
            subject=subject,
            priority_score=priority_score,
            priority_tier=priority_tier,
            profile_snapshot=profile_snapshot,
            metadata=metadata or {},
        )
        message = await self.add_message(
            ticket_id=ticket["id"],
            identity_id=identity["id"],
            direction=MESSAGE_INBOUND,
            body=body,
            metadata=metadata or {},
        )
        if self.notifier is not None:
            try:
                await self.notifier.notify_new_ticket(ticket)
            except Exception:
                pass
        return {"identity": identity, "ticket": ticket, "message": message}

    async def add_message(
        self,
        *,
        ticket_id: Any,
        identity_id: Any | None,
        direction: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_messages (ticket_id, identity_id, direction, body, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """,
            ticket_id,
            identity_id,
            direction,
            str(body or ""),
            json.dumps(metadata or {}, ensure_ascii=False),
        )
        await self.support_db.execute(
            "UPDATE support_tickets SET last_message_at = now(), updated_at = now() WHERE id = $1",
            ticket_id,
        )
        return dict(row)

    async def create_admin_reply(
        self,
        *,
        ticket: dict[str, Any],
        body: str,
        admin_channel_id: Any,
    ) -> dict[str, Any]:
        message = await self.add_message(
            ticket_id=ticket["id"],
            identity_id=None,
            direction=MESSAGE_OUTBOUND,
            body=body,
            metadata={"admin_channel_id": str(admin_channel_id)},
        )
        outbox = await self.support_db.fetchrow(
            """
            INSERT INTO support_delivery_outbox (ticket_id, message_id, channel, channel_id, payload)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """,
            ticket["id"],
            message["id"],
            str(ticket.get("channel") or ""),
            str(ticket.get("channel_id") or ""),
            json.dumps({"text": str(body or "")}, ensure_ascii=False),
        ) if str(ticket.get("channel") or "") in {"telegram", "max"} else None
        await self._audit(
            ticket_id=ticket["id"],
            event_type="admin_reply",
            admin_channel_id=admin_channel_id,
            event_data={"message_id": message["id"]},
        )
        return {"message": message, "outbox": dict(outbox) if outbox else None}

    async def create_admin_note(
        self,
        *,
        ticket_id: Any,
        body: str,
        admin_channel_id: Any,
    ) -> dict[str, Any]:
        note = await self.add_message(
            ticket_id=ticket_id,
            identity_id=None,
            direction=MESSAGE_INTERNAL,
            body=body,
            metadata={"admin_channel_id": str(admin_channel_id)},
        )
        await self._audit(
            ticket_id=ticket_id,
            event_type="admin_note",
            admin_channel_id=admin_channel_id,
            event_data={"message_id": note["id"]},
        )
        return note

    async def update_ticket_status(
        self,
        *,
        ticket_id: Any,
        status: str,
        admin_channel_id: Any,
    ) -> dict[str, Any]:
        await self.support_db.execute(
            "UPDATE support_tickets SET status = $1, updated_at = now() WHERE id = $2",
            status,
            ticket_id,
        )
        await self._audit(
            ticket_id=ticket_id,
            event_type="status_update",
            admin_channel_id=admin_channel_id,
            event_data={"status": status},
        )
        return {"id": ticket_id, "status": status}

    async def list_admin_tickets(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = await self.support_db.fetch(
            """
            SELECT
                t.*,
                latest.body AS latest_message_body,
                latest.direction AS latest_message_direction,
                latest.created_at AS latest_message_created_at,
                i.display_name AS requester_display_name,
                i.external_user_id AS requester_external_user_id,
                i.game_user_id AS requester_game_user_id,
                i.channel AS requester_channel,
                i.channel_id AS requester_channel_id
            FROM support_tickets t
            LEFT JOIN LATERAL (
                SELECT body, direction, created_at
                FROM support_messages m
                WHERE m.ticket_id = t.id
                ORDER BY m.created_at DESC
                LIMIT 1
            ) latest ON TRUE
            LEFT JOIN support_identities i ON i.id = t.requester_identity_id
            ORDER BY t.priority_score DESC, t.last_message_at ASC, t.created_at ASC
            LIMIT $1 OFFSET $2
            """,
            int(limit),
            int(offset),
        )
        return [dict(row) for row in rows]

    async def get_ticket(self, *, ticket_id: Any) -> dict[str, Any] | None:
        row = await self.support_db.fetchrow(
            """
            SELECT t.*, i.display_name AS requester_display_name,
                   i.external_user_id AS requester_external_user_id,
                   i.game_user_id AS requester_game_user_id,
                   i.channel AS requester_channel,
                   i.channel_id AS requester_channel_id
            FROM support_tickets t
            LEFT JOIN support_identities i ON i.id = t.requester_identity_id
            WHERE t.id = $1
            """,
            ticket_id,
        )
        return dict(row) if row else None

    async def get_active_ticket_for_channel(self, *, channel: str, channel_id: Any) -> dict[str, Any] | None:
        row = await self.support_db.fetchrow(
            """
            SELECT *
            FROM support_tickets
            WHERE channel = $1 AND channel_id = $2 AND status <> 'closed'
            ORDER BY last_message_at DESC, created_at DESC
            LIMIT 1
            """,
            str(channel),
            str(channel_id),
        )
        return dict(row) if row else None

    async def add_inbound_channel_message(
        self,
        *,
        ticket: dict[str, Any],
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message = await self.add_message(
            ticket_id=ticket["id"],
            identity_id=ticket.get("requester_identity_id"),
            direction=MESSAGE_INBOUND,
            body=body,
            metadata=metadata or {},
        )
        if self.notifier is not None:
            try:
                await self.notifier.notify_new_inbound_message(ticket, message)
            except Exception:
                pass
        return message

    async def add_user_message(
        self,
        *,
        ticket: dict[str, Any],
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if str(ticket.get("status") or "") == "closed":
            raise ValueError("ticket_closed")
        message = await self.add_message(
            ticket_id=ticket["id"],
            identity_id=ticket.get("requester_identity_id"),
            direction=MESSAGE_INBOUND,
            body=body,
            metadata=metadata or {},
        )
        if self.notifier is not None:
            try:
                await self.notifier.notify_new_inbound_message(ticket, message)
            except Exception:
                pass
        return message

    async def list_ticket_messages(self, *, ticket_id: Any, public: bool = False) -> list[dict[str, Any]]:
        if public:
            rows = await self.support_db.fetch(
                """
                SELECT id, ticket_id, direction, body, created_at
                FROM support_messages
                WHERE ticket_id = $1 AND direction <> 'internal'
                ORDER BY created_at ASC
                """,
                ticket_id,
            )
            return [dict(row) for row in rows]
        rows = await self.support_db.fetch(
            """
            SELECT *
            FROM support_messages
            WHERE ticket_id = $1
            ORDER BY created_at ASC
            """,
            ticket_id,
        )
        return [dict(row) for row in rows]

    async def list_ticket_attachments(self, *, ticket_id: Any) -> list[dict[str, Any]]:
        rows = await self.support_db.fetch(
            """
            SELECT *
            FROM support_attachments
            WHERE ticket_id = $1
            ORDER BY created_at ASC
            """,
            ticket_id,
        )
        return [dict(row) for row in rows]

    async def list_public_ticket_attachments(self, *, ticket_id: Any) -> list[dict[str, Any]]:
        rows = await self.support_db.fetch(
            """
            SELECT a.*
            FROM support_attachments a
            LEFT JOIN support_messages m ON m.id = a.message_id
            WHERE a.ticket_id = $1
              AND (m.direction IS NULL OR m.direction <> 'internal')
            ORDER BY a.created_at ASC
            """,
            ticket_id,
        )
        return [dict(row) for row in rows]

    async def record_attachment(
        self,
        *,
        ticket_id: Any,
        message_id: Any | None,
        uploader_identity_id: Any | None = None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_attachments (
                ticket_id, message_id, uploader_identity_id, storage_path, original_filename,
                content_type, sha256, size_bytes, width, height, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            RETURNING *
            """,
            ticket_id,
            message_id,
            uploader_identity_id,
            str(metadata.get("storage_path") or ""),
            str(metadata.get("original_filename") or ""),
            str(metadata.get("content_type") or "application/octet-stream"),
            str(metadata.get("sha256") or ""),
            int(metadata.get("size_bytes") or 0),
            metadata.get("width"),
            metadata.get("height"),
            json.dumps(metadata, ensure_ascii=False),
        )
        return dict(row)

    async def issue_auth_code(
        self,
        *,
        identity_id: Any | None,
        game_user_id: int | None,
        code: str,
        ttl_seconds: int = 300,
        purpose: str = "verify_account",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_seconds)))
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_auth_codes (identity_id, game_user_id, purpose, code_hash, expires_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING *
            """,
            identity_id,
            game_user_id,
            purpose,
            hash_support_code(code),
            expires_at,
            json.dumps(metadata or {}, ensure_ascii=False),
        )
        payload = dict(row)
        payload["code"] = code
        return payload

    async def consume_auth_code(self, code: str, *, purpose: str = "verify_account") -> dict[str, Any] | None:
        row = await self.support_db.fetchrow(
            """
            UPDATE support_auth_codes
            SET used_at = now()
            WHERE code_hash = $1 AND purpose = $2 AND used_at IS NULL AND expires_at > now()
            RETURNING *
            """,
            hash_support_code(code),
            purpose,
        )
        return dict(row) if row else None

    async def issue_admin_login_code(
        self,
        *,
        admin_channel_id: Any,
        code: str,
        ttl_seconds: int = 300,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_seconds)))
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_admin_login_codes (admin_channel_id, code_hash, expires_at, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING *
            """,
            str(admin_channel_id),
            hash_support_code(code),
            expires_at,
            json.dumps(metadata or {}, ensure_ascii=False),
        )
        payload = dict(row)
        payload["code"] = code
        return payload

    async def consume_admin_login_code(self, code: str) -> dict[str, Any] | None:
        row = await self.support_db.fetchrow(
            """
            UPDATE support_admin_login_codes
            SET used_at = now()
            WHERE code_hash = $1 AND used_at IS NULL AND expires_at > now()
            RETURNING *
            """,
            hash_support_code(code),
        )
        return dict(row) if row else None

    def sort_tickets_for_inbox(self, tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(ticket: dict[str, Any]) -> tuple[int, str]:
            last = str(ticket.get("last_message_at") or ticket.get("created_at") or "")
            return (-int(ticket.get("priority_score") or 0), last)

        return sorted(tickets, key=key)

    async def _profile_snapshot(self, game_user_id: int | None) -> dict[str, Any]:
        if not game_user_id or not self.game_db:
            return {}
        getter = getattr(self.game_db, "get_user_profile", None) or getattr(self.game_db, "get_user_info", None)
        if not getter:
            return {}
        profile = await getter(int(game_user_id))
        return dict(profile) if profile else {}

    def _priority_for_profile(
        self,
        profile: dict[str, Any],
        *,
        requested_scope: str | None,
        has_game_user_id: bool,
    ) -> tuple[str, str, str, int]:
        if requested_scope == AccountScope.UNVERIFIED:
            return AccountScope.UNVERIFIED, SupportStatus.QUEUED_UNVERIFIED, "unverified", 10
        if requested_scope == AccountScope.GUEST or not has_game_user_id:
            return AccountScope.GUEST, SupportStatus.QUEUED_GUEST, "guest", 0
        trophies = int(profile.get("trophies") or 0)
        if trophies < 100:
            return AccountScope.UNVERIFIED, SupportStatus.QUEUED_UNVERIFIED, "unverified", 10
        mode = str(profile.get("extra_pass") or "inactive").lower()
        if mode == "ultra":
            return AccountScope.VERIFIED, SupportStatus.OPEN, "ultra", 300
        if mode == "active":
            return AccountScope.VERIFIED, SupportStatus.OPEN, "extra_pass", 200
        return AccountScope.VERIFIED, SupportStatus.OPEN, "verified", 100

    async def _create_identity(
        self,
        *,
        scope: str,
        external_user_id: str | None,
        game_user_id: int | None,
        display_name: str,
        channel: str,
        channel_id: Any,
        is_verified: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_external_user_id = external_user_id or f"{channel}:{channel_id}"
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_identities
                (scope, external_user_id, game_user_id, display_name, channel, channel_id, is_verified, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (scope, external_user_id) DO UPDATE SET
                game_user_id = EXCLUDED.game_user_id,
                display_name = EXCLUDED.display_name,
                channel = EXCLUDED.channel,
                channel_id = EXCLUDED.channel_id,
                is_verified = EXCLUDED.is_verified,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING *
            """,
            scope,
            normalized_external_user_id,
            game_user_id,
            str(display_name or ""),
            channel,
            str(channel_id),
            bool(is_verified),
            json.dumps(metadata, ensure_ascii=False),
        )
        return dict(row)

    async def _insert_ticket(
        self,
        *,
        topic: str,
        status: str,
        account_scope: str,
        requester_identity_id: Any,
        game_user_id: int | None,
        channel: str,
        channel_id: Any,
        subject: str,
        priority_score: int,
        priority_tier: str,
        profile_snapshot: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        row = await self.support_db.fetchrow(
            """
            INSERT INTO support_tickets
                (topic, status, account_scope, requester_identity_id, game_user_id, channel, channel_id,
                 subject, priority_score, priority_tier, profile_snapshot, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb)
            RETURNING *
            """,
            topic,
            status,
            account_scope,
            requester_identity_id,
            game_user_id,
            channel,
            str(channel_id),
            str(subject or ""),
            int(priority_score),
            priority_tier,
            json.dumps(profile_snapshot, ensure_ascii=False, default=str),
            json.dumps(metadata, ensure_ascii=False),
        )
        return dict(row)

    async def _audit(
        self,
        *,
        ticket_id: Any,
        event_type: str,
        admin_channel_id: Any | None = None,
        event_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self.support_db.fetchrow(
                """
                INSERT INTO support_audit_events (admin_channel_id, ticket_id, event_type, event_data, metadata)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                RETURNING *
                """,
                str(admin_channel_id) if admin_channel_id is not None else None,
                ticket_id,
                event_type,
                json.dumps(event_data or {}, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        except Exception:
            return
