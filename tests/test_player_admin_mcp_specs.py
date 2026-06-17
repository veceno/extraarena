from __future__ import annotations

import pytest


class FakePlayerAdminDB:
    def __init__(self) -> None:
        self.bans: list[dict] = []
        self.unbans: list[dict] = []
        self.warnings: list[dict] = []
        self.account_updates: list[dict] = []

    async def admin_ban_user(self, admin_user_id, target_user_id, reason=None, until=None):
        self.bans.append(
            {
                "admin_user_id": admin_user_id,
                "target_user_id": target_user_id,
                "reason": reason,
                "until": until,
            }
        )
        return {"status": "ok", "action": "ban"}

    async def admin_unban_user(self, admin_user_id, target_user_id, reason=None):
        self.unbans.append(
            {
                "admin_user_id": admin_user_id,
                "target_user_id": target_user_id,
                "reason": reason,
            }
        )
        return {"status": "ok", "action": "unban"}

    async def admin_warn_user(self, admin_user_id, target_user_id, reason):
        self.warnings.append(
            {
                "admin_user_id": admin_user_id,
                "target_user_id": target_user_id,
                "reason": reason,
            }
        )
        return {"status": "ok", "action": "warn"}

    async def admin_update_user_account(self, admin_user_id, target_user_id, fields=None, reason=None):
        self.account_updates.append(
            {
                "admin_user_id": admin_user_id,
                "target_user_id": target_user_id,
                "fields": dict(fields or {}),
                "reason": reason,
            }
        )
        return {"status": "ok", "action": "update_account", "fields": dict(fields or {})}


def _spec_by_id():
    from web.player_admin_action_specs import PLAYER_ADMIN_ACTION_CAPABILITY_SPECS

    return {spec["id"]: spec for spec in PLAYER_ADMIN_ACTION_CAPABILITY_SPECS}


def test_player_admin_specs_cover_missing_mutating_actions():
    specs = _spec_by_id()

    assert {
        "admin.players.ban",
        "admin.players.unban",
        "admin.players.warn",
        "admin.players.account.update",
    }.issubset(specs)
    assert "admin.extrapass.players.entitlement.set" not in specs
    assert "admin.players.note.create" not in specs
    assert "admin.players.resource.grant" not in specs

    for spec in specs.values():
        schema = spec["input_schema"]
        properties = schema["properties"]
        assert spec["required_scope"] == "admin:players:write"
        assert spec["read_only"] is False
        assert spec["mutating"] is True
        assert spec["dry_run_required"] is True
        assert spec["idempotency_required"] is True
        assert "dry_run" in properties
        assert "idempotency_key" in properties
        assert "confirmation_token" in properties

    assert specs["admin.players.ban"]["input_schema"]["required"] == ["user_id", "reason", "dry_run"]
    assert specs["admin.players.unban"]["input_schema"]["required"] == ["user_id", "dry_run"]
    assert specs["admin.players.warn"]["input_schema"]["required"] == ["user_id", "reason", "dry_run"]
    assert specs["admin.players.account.update"]["input_schema"]["required"] == [
        "user_id",
        "fields",
        "reason",
        "dry_run",
    ]


@pytest.mark.asyncio
async def test_ban_adapter_dry_run_parses_until_and_does_not_mutate():
    from web.player_admin_action_specs import adapter_ban_player_account

    db = FakePlayerAdminDB()
    result = await adapter_ban_player_account(
        {"db": db},
        101,
        {
            "user_id": 201,
            "reason": "spam",
            "until": "2026-06-08T12:30:00+00:00",
            "dry_run": True,
        },
    )

    assert result["dry_run"] is True
    assert result["target_user_id"] == 201
    assert result["until"] == "2026-06-08T12:30:00+00:00"
    assert db.bans == []


@pytest.mark.asyncio
async def test_ban_adapter_apply_calls_existing_db_method():
    from web.player_admin_action_specs import adapter_ban_player_account

    db = FakePlayerAdminDB()
    result = await adapter_ban_player_account(
        {"db": db},
        101,
        {
            "user_id": 201,
            "reason": "spam",
            "dry_run": False,
            "idempotency_key": "player-ban-201-spam",
            "confirmation_token": "confirmation-token",
        },
    )

    assert result["dry_run"] is False
    assert result["result"] == {"status": "ok", "action": "ban"}
    assert db.bans == [
        {
            "admin_user_id": 101,
            "target_user_id": 201,
            "reason": "spam",
            "until": None,
        }
    ]


@pytest.mark.asyncio
async def test_warn_adapter_requires_reason():
    from web.mcp_admin_tools import MCPToolInputError
    from web.player_admin_action_specs import adapter_warn_player_account

    with pytest.raises(MCPToolInputError, match="reason_required"):
        await adapter_warn_player_account({"db": FakePlayerAdminDB()}, 101, {"user_id": 201, "dry_run": True})


@pytest.mark.asyncio
async def test_account_update_adapter_validates_fields_and_self_ban_confirmation():
    from web.mcp_admin_tools import MCPToolInputError
    from web.player_admin_action_specs import adapter_update_player_account

    with pytest.raises(MCPToolInputError, match="unsupported_account_field"):
        await adapter_update_player_account(
            {"db": FakePlayerAdminDB()},
            101,
            {"user_id": 201, "fields": {"gems": 999}, "reason": "bad field", "dry_run": True},
        )

    with pytest.raises(MCPToolInputError, match="self_ban_requires_confirm"):
        await adapter_update_player_account(
            {"db": FakePlayerAdminDB()},
            101,
            {"user_id": 101, "fields": {"status": "banned"}, "reason": "self ban", "dry_run": True},
        )


@pytest.mark.asyncio
async def test_account_update_adapter_apply_normalizes_fields():
    from web.player_admin_action_specs import adapter_update_player_account

    db = FakePlayerAdminDB()
    result = await adapter_update_player_account(
        {"db": db},
        101,
        {
            "user_id": 201,
            "fields": {
                "username": " pilot ",
                "trophies": "15",
                "max_trophies": "-3",
                "status": "WARN",
            },
            "reason": "support correction",
            "dry_run": False,
            "idempotency_key": "account-update-201",
            "confirmation_token": "confirmation-token",
        },
    )

    assert result["dry_run"] is False
    assert db.account_updates == [
        {
            "admin_user_id": 101,
            "target_user_id": 201,
            "fields": {
                "username": "pilot",
                "trophies": 15,
                "max_trophies": 0,
                "status": "warn",
            },
            "reason": "support correction",
        }
    ]
