import logging
from datetime import datetime, timedelta, timezone

import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database
from infrastructure.payments_logic import _grant_rewards_for_item, process_successful_payment


class ClaimFailedPaymentDB:
    def __init__(self):
        self.gems_added = 0
        self.claim_attempted = False

    async def claim_payment_for_processing(self, payment_id):
        self.claim_attempted = True
        return None

    async def execute(self, query, *args):
        if "SET gems =" in query and "+ $1" in query:
            self.gems_added += int(args[0])
        return "UPDATE 1"

    async def fetchval(self, query, *args):
        return None


class GrantRaisesAfterClaimDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.claim_attempted = False
        self.released = False
        self.marked_processed = False

    async def claim_payment_for_processing(self, payment_id):
        self.claim_attempted = True
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released = True

    async def execute(self, query, *args):
        raise RuntimeError("grant exploded")

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
        return None


class ClaimSucceedsPaymentDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.gems_added = 0
        self.marked_processed = False
        self.released = False

    async def claim_payment_for_processing(self, payment_id):
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released = True

    async def execute(self, query, *args):
        if "SET gems =" in query and "+ $1" in query:
            self.gems_added += int(args[0])
        return "UPDATE 1"

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
            return 1
        return None

    async def create_mail(self, **kwargs):
        return {"success": True}

    async def track_economy_event(self, **kwargs):
        return {"success": True}


class NoRewardsAlertDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.released = False
        self.marked_processed = False
        self.alerts = []
        self.admin_actions = []

    async def claim_payment_for_processing(self, payment_id):
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released = True

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
        return None

    async def record_payment_processing_alert(self, **kwargs):
        self.alerts.append(kwargs)
        return {"success": True}

    async def record_admin_account_action(
        self,
        admin_user_id,
        target_user_id,
        action_type,
        reason=None,
        payload=None,
    ):
        self.admin_actions.append(
            {
                "admin_user_id": admin_user_id,
                "target_user_id": target_user_id,
                "action_type": action_type,
                "reason": reason,
                "payload": payload,
            }
        )
        return {"success": True}


class EconomyEventFailsAfterGrantDB(ClaimSucceedsPaymentDB):
    async def track_economy_event(self, **kwargs):
        raise RuntimeError("economy event failed")


class PassEntitlementDB:
    def __init__(self, profile):
        self.profile = profile
        self.pass_updates = []
        self.gems_added = 0
        self.coins_added = 0

    async def get_user_profile(self, user_id):
        return dict(self.profile)

    async def fetchval(self, query, *args):
        compact_query = " ".join(str(query).split())
        if "UPDATE users SET extra_pass = 'active'" in query:
            self.pass_updates.append({"mode": "active", "expires_at": args[0], "user_id": args[1]})
            return 1
        if "UPDATE users SET extra_pass = 'ultra'" in query:
            self.pass_updates.append({"mode": "ultra", "expires_at": args[0], "user_id": args[1]})
            return 1
        if "UPDATE users SET extra_pass = $1" in compact_query:
            if len(args) >= 4:
                self.gems_added += int(args[2])
            self.pass_updates.append({"mode": args[0], "expires_at": args[1], "user_id": args[3] if len(args) >= 4 else args[2]})
            return 1
        return None

    async def execute(self, query, *args):
        if "SET gems =" in query and "+ 1200" in query:
            self.gems_added += 1200
        elif "SET gems =" in query and "+ 500" in query:
            self.gems_added += 500
        elif "SET coins =" in query and "+ 3000" in query:
            self.coins_added += 3000
        elif "extra_pass = 'active'" in query:
            self.pass_updates.append({"mode": "active", "expires_at": args[0], "user_id": args[1]})
        return "UPDATE 1"

    async def get_admin_case_id(self, tier):
        return None


class StarterBoostStepLedgerDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.payment_record["metadata"] = dict(payment_record.get("metadata") or {})
        self.fail_coins_once = True
        self.released = 0
        self.marked_processed = False
        self.pass_updates = 0
        self.gems_added = 0
        self.coins_added = 0
        self.marked_steps = []

    async def run_payment_reward_step(self, payment_id, step_id, apply_fn):
        metadata = self.payment_record.setdefault("metadata", {})
        if metadata.setdefault("reward_steps", {}).get(step_id):
            return {"success": True, "applied": False, "status": "already_applied"}
        await apply_fn(self)
        metadata["reward_steps"][step_id] = True
        self.marked_steps.append(step_id)
        return {"success": True, "applied": True, "status": "applied"}

    async def claim_payment_for_processing(self, payment_id):
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released += 1

    async def mark_payment_reward_step(self, payment_id, step_id):
        metadata = self.payment_record.setdefault("metadata", {})
        metadata.setdefault("reward_steps", {})[step_id] = True
        self.marked_steps.append(step_id)
        return True

    async def get_user_profile(self, user_id):
        return {"user_id": user_id, "extra_pass": "inactive", "extra_pass_expires_at": None}

    async def fetchval(self, query, *args):
        if "UPDATE users" in query and "extra_pass = $1" in query:
            self.pass_updates += 1
            if len(args) >= 4:
                self.gems_added += int(args[2])
            return 1
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
            return 1
        return None

    async def execute(self, query, *args):
        if "UPDATE users SET coins" in query and "+ 3000" in query:
            if self.fail_coins_once:
                self.fail_coins_once = False
                raise RuntimeError("coins exploded")
            self.coins_added += 3000
        return "UPDATE 1"

    async def get_admin_case_id(self, tier):
        return None

    async def create_mail(self, **kwargs):
        return {"success": True}

    async def track_economy_event(self, **kwargs):
        return {"success": True}


class StarterBoostMarkerCrashDB(StarterBoostStepLedgerDB):
    def __init__(self, payment_record):
        super().__init__(payment_record)
        self.fail_marker_once = True

    async def mark_payment_reward_step(self, payment_id, step_id):
        if step_id == "starter_boost_coins" and self.fail_marker_once:
            self.fail_marker_once = False
            self.payment_record.setdefault("metadata", {}).setdefault("reward_steps", {}).pop(step_id, None)
            raise RuntimeError("marker exploded")
        return await super().mark_payment_reward_step(payment_id, step_id)


class ShopSetPaymentProcessingDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.payment_record["metadata"] = dict(payment_record.get("metadata") or {})
        self.released = 0
        self.marked_processed = False
        self.marked_steps = []
        self.shop_set_grants = []
        self.mail = None
        self.economy_events = []

    async def claim_payment_for_processing(self, payment_id):
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released += 1

    async def run_payment_reward_step(self, payment_id, step_id, apply_fn):
        metadata = self.payment_record.setdefault("metadata", {})
        if metadata.setdefault("reward_steps", {}).get(step_id):
            return {"success": True, "applied": False, "status": "already_applied"}
        await apply_fn(self)
        metadata["reward_steps"][step_id] = True
        self.marked_steps.append(step_id)
        return {"success": True, "applied": True, "status": "applied"}

    async def grant_shop_set_rewards_on_conn(self, conn, user_id, set_id):
        self.shop_set_grants.append({"conn": conn, "user_id": user_id, "set_id": set_id})
        return {
            "success": True,
            "granted": [
                {"type": "gems", "amount": 25},
                {"type": "cosmetic", "cosmetic_slug": "avatar_paid_test", "acquired": True},
            ],
        }

    async def fetchrow(self, query=None, *args):
        return {"id": 1}

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
            return 1
        return None

    async def create_mail(self, **kwargs):
        self.mail = kwargs
        return {"success": True}

    async def track_economy_event(self, **kwargs):
        self.economy_events.append(kwargs)
        return {"success": True}


class SquadBoostPaymentProcessingDB:
    def __init__(self, payment_record):
        self.payment_record = dict(payment_record)
        self.payment_record["metadata"] = dict(payment_record.get("metadata") or {})
        self.released = 0
        self.marked_processed = False
        self.marked_steps = []
        self.activations = []
        self.mail = None
        self.economy_events = []

    async def claim_payment_for_processing(self, payment_id):
        return dict(self.payment_record)

    async def release_payment_processing_claim(self, payment_id):
        self.released += 1

    async def run_payment_reward_step(self, payment_id, step_id, apply_fn):
        metadata = self.payment_record.setdefault("metadata", {})
        if metadata.setdefault("reward_steps", {}).get(step_id):
            return {"success": True, "applied": False, "status": "already_applied"}
        await apply_fn(self)
        metadata["reward_steps"][step_id] = True
        self.marked_steps.append(step_id)
        return {"success": True, "applied": True, "status": "applied"}

    async def activate_clan_boost_from_purchase(self, user_id, *, executor=None):
        self.activations.append({"user_id": user_id, "executor": executor})
        return {"status": "activated", "clan_id": 10, "boost_public_id": 777, "member_slots_added": 5}

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.marked_processed = True
            return 1
        return None

    async def create_mail(self, **kwargs):
        self.mail = kwargs
        return {"success": True}

    async def track_economy_event(self, **kwargs):
        self.economy_events.append(kwargs)
        return {"success": True}


class DuplicateRetryDB(ClaimSucceedsPaymentDB):
    async def claim_payment_for_processing(self, payment_id):
        if self.marked_processed:
            return None
        return dict(self.payment_record)


class ShopSetRewardApplyDB(Database):
    def __init__(self):
        super().__init__(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))
        self.cosmetic_grants = []

    async def grant_cosmetic_by_slug(self, user_id, slug, *, source="grant", auto_equip=False):
        self.cosmetic_grants.append(
            {
                "user_id": int(user_id),
                "slug": str(slug),
                "source": source,
                "auto_equip": bool(auto_equip),
            }
        )
        return {
            "item": {"id": 77, "slug": str(slug), "item_type": "avatar", "name": "Gift Avatar"},
            "equipped": {"slug": str(slug)} if auto_equip else None,
            "acquired": True,
        }


class ShopSetRewardConn:
    def __init__(self):
        self.execute_calls = []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"


class ShopSetOwnedCardFallbackConn(ShopSetRewardConn):
    async def fetchrow(self, query, *args):
        if "FROM cosmetic_items" in query:
            return {
                "id": 88,
                "slug": str(args[0]),
                "item_type": "profile_background",
                "class": "gold",
                "name": "Gold Background",
                "asset_path": "/static/bg_gold.png",
                "media_type": "image",
            }
        return None

    async def fetchval(self, query, *args):
        if "FROM user_cards" in query:
            return 1
        if "FROM user_cosmetics" in query:
            return None
        return None


class ShopSetOwnedCardParticleConn(ShopSetRewardConn):
    def __init__(self):
        super().__init__()
        self.particles = 10

    async def fetchrow(self, query, *args):
        normalized = " ".join(str(query).split())
        if "SELECT id, name, rarity, simplified_levelup FROM cards" in normalized:
            return {
                "id": 77,
                "name": "Duplicate Rare",
                "rarity": "rare",
                "simplified_levelup": False,
            }
        if "PERCENTILE_DISC" in normalized:
            return {"eligible_count": 9, "reference_level": 1}
        if "INSERT INTO user_cards" in normalized:
            return None
        if "SELECT name, rarity FROM cards" in normalized:
            return {"name": "Duplicate Rare", "rarity": "rare"}
        if "FROM users" in normalized and "FOR UPDATE" in normalized:
            return {"coins": 1000}
        if "FROM user_cards uc" in normalized and "FOR UPDATE OF uc" in normalized:
            return {
                "card_id": 77,
                "level": 1,
                "name": "Duplicate Rare",
                "simplified_levelup": False,
            }
        if "UPDATE user_cards" in normalized and "RETURNING particles" in normalized:
            self.particles += int(args[0])
            return {"particles": self.particles}
        return None


@pytest.mark.asyncio
async def test_successful_payment_does_not_grant_rewards_when_processing_claim_fails():
    db = ClaimFailedPaymentDB()
    payment_record = {
        "payment_id": "pay-race",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "100 gems",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "rewards_processed": False,
    }

    result = await process_successful_payment(
        db,
        payment_id="pay-race",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert db.claim_attempted is True
    assert db.gems_added == 0
    assert result["status"] == "already_processed"


@pytest.mark.asyncio
async def test_successful_payment_releases_processing_claim_when_grant_raises():
    payment_record = {
        "payment_id": "pay-retry",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "100 gems",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "rewards_processed": False,
    }
    db = GrantRaisesAfterClaimDB(payment_record)

    with pytest.raises(RuntimeError, match="grant exploded"):
        await process_successful_payment(
            db,
            payment_id="pay-retry",
            payment_record=payment_record,
            source="test",
            logger=logging.getLogger("test"),
        )

    assert db.claim_attempted is True
    assert db.released is True
    assert db.marked_processed is False


@pytest.mark.asyncio
async def test_successful_payment_marks_processed_only_after_successful_grant():
    payment_record = {
        "payment_id": "pay-ok",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "100 gems",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "rewards_processed": False,
    }
    db = ClaimSucceedsPaymentDB(payment_record)

    result = await process_successful_payment(
        db,
        payment_id="pay-ok",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert db.gems_added == 100
    assert db.marked_processed is True
    assert db.released is False


@pytest.mark.asyncio
async def test_successful_payment_records_admin_alert_when_no_rewards_given(caplog):
    payment_record = {
        "payment_id": "pay-no-rewards",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "legacy_unknown_sku",
        "metadata": {},
        "rewards_processed": False,
    }
    db = NoRewardsAlertDB(payment_record)
    logger = logging.getLogger("test.payment.no_rewards")

    with caplog.at_level(logging.CRITICAL, logger=logger.name):
        result = await process_successful_payment(
            db,
            payment_id="pay-no-rewards",
            payment_record=payment_record,
            source="test_gateway",
            logger=logger,
        )

    assert result["status"] == "no_rewards"
    assert result["item_type"] == "legacy_unknown_sku"
    assert db.released is True
    assert db.marked_processed is False
    assert db.alerts == [
        {
            "alert_type": "payment_no_rewards",
            "payment_id": "pay-no-rewards",
            "user_id": 1001,
            "item_type": "legacy_unknown_sku",
            "source": "test_gateway",
            "description": "legacy_unknown_sku",
            "metadata": {"payment_id": "pay-no-rewards"},
        }
    ]
    assert db.admin_actions == [
        {
            "admin_user_id": 0,
            "target_user_id": 1001,
            "action_type": "payment_no_rewards",
            "reason": "successful_payment_without_rewards",
            "payload": {
                "payment_id": "pay-no-rewards",
                "source": "test_gateway",
                "item_type": "legacy_unknown_sku",
                "amount": 99,
                "currency": "RUB",
                "description": "legacy_unknown_sku",
                "metadata": {"payment_id": "pay-no-rewards"},
            },
        }
    ]
    assert any("PAYMENT_NO_REWARDS" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_successful_payment_does_not_retry_reward_when_economy_event_fails_after_grant():
    payment_record = {
        "payment_id": "pay-event-warning",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "100 gems",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "rewards_processed": False,
    }
    db = EconomyEventFailsAfterGrantDB(payment_record)

    result = await process_successful_payment(
        db,
        payment_id="pay-event-warning",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert result["warnings"] == ["economy event failed"]
    assert db.gems_added == 100
    assert db.marked_processed is True
    assert db.released is False


@pytest.mark.asyncio
async def test_extrapass_purchase_over_active_ultra_preserves_tier_without_expiry():
    future_expiry = datetime.now(timezone.utc) + timedelta(days=10)
    db = PassEntitlementDB(
        {
            "user_id": 1001,
            "extra_pass": "ultra",
            "extra_pass_expires_at": future_expiry,
        }
    )

    result = await _grant_rewards_for_item(
        db,
        user_id=1001,
        item_type="extrapass",
        metadata={},
        logger=logging.getLogger("test"),
    )

    assert result["rewards_given"] is True
    assert db.pass_updates[-1]["mode"] == "ultra"
    assert db.pass_updates[-1]["expires_at"] is None


@pytest.mark.asyncio
async def test_extrapass_ultra_purchase_grants_season_entitlement_without_expiry():
    db = PassEntitlementDB(
        {
            "user_id": 1001,
            "extra_pass": "inactive",
            "extra_pass_expires_at": None,
        }
    )

    result = await _grant_rewards_for_item(
        db,
        user_id=1001,
        item_type="extrapass_ultra",
        metadata={},
        logger=logging.getLogger("test"),
    )

    assert result["rewards_given"] is True
    assert db.pass_updates[-1] == {"mode": "ultra", "expires_at": None, "user_id": 1001}
    assert db.gems_added == 500


@pytest.mark.asyncio
async def test_starter_boost_treats_active_ultra_as_extra_pass_without_downgrade():
    future_expiry = datetime.now(timezone.utc) + timedelta(days=10)
    db = PassEntitlementDB(
        {
            "user_id": 1001,
            "extra_pass": "ultra",
            "extra_pass_expires_at": future_expiry,
        }
    )

    result = await _grant_rewards_for_item(
        db,
        user_id=1001,
        item_type="starter_boost",
        metadata={},
        logger=logging.getLogger("test"),
    )

    assert result["rewards_given"] is True
    assert db.pass_updates == []
    assert db.gems_added == 1200
    assert db.coins_added == 3000


@pytest.mark.asyncio
async def test_payment_reward_step_ledger_retries_without_duplicate_grants():
    payment_record = {
        "payment_id": "pay-starter-step",
        "user_id": 1001,
        "amount": 499,
        "currency": "RUB",
        "description": "Starter Boost",
        "metadata": {"item_type": "starter_boost"},
        "rewards_processed": False,
    }
    db = StarterBoostStepLedgerDB(payment_record)

    with pytest.raises(RuntimeError, match="coins exploded"):
        await process_successful_payment(
            db,
            payment_id="pay-starter-step",
            payment_record=payment_record,
            source="test",
            logger=logging.getLogger("test"),
        )

    assert db.released == 1
    assert db.gems_added == 500
    assert db.pass_updates == 1
    assert "starter_boost_pass_or_gems" in db.payment_record["metadata"]["reward_steps"]
    assert db.marked_processed is False

    result = await process_successful_payment(
        db,
        payment_id="pay-starter-step",
        payment_record=db.payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert db.gems_added == 500
    assert db.pass_updates == 1
    assert db.coins_added == 3000
    assert db.marked_processed is True


@pytest.mark.asyncio
async def test_reward_step_marker_crash_after_side_effect_does_not_duplicate_on_retry():
    payment_record = {
        "payment_id": "pay-marker-crash",
        "user_id": 1001,
        "amount": 499,
        "currency": "RUB",
        "description": "Starter Boost",
        "metadata": {
            "item_type": "starter_boost",
            "reward_steps": {"starter_boost_pass_or_gems": True},
        },
        "rewards_processed": False,
    }
    db = StarterBoostMarkerCrashDB(payment_record)
    db.fail_coins_once = False
    db.gems_added = 500
    db.pass_updates = 1

    result = await process_successful_payment(
        db,
        payment_id="pay-marker-crash",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert db.coins_added == 3000
    assert "starter_boost_coins" in db.payment_record["metadata"].get("reward_steps", {})
    assert db.marked_processed is True

    second = await process_successful_payment(
        db,
        payment_id="pay-marker-crash",
        payment_record=db.payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert second["status"] == "processed"
    assert db.coins_added == 3000


@pytest.mark.asyncio
async def test_duplicate_successful_payment_retry_does_not_duplicate_grant():
    payment_record = {
        "payment_id": "pay-duplicate-retry",
        "user_id": 1001,
        "amount": 99,
        "currency": "RUB",
        "description": "100 gems",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "rewards_processed": False,
    }
    db = DuplicateRetryDB(payment_record)

    first = await process_successful_payment(
        db,
        payment_id="pay-duplicate-retry",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )
    second = await process_successful_payment(
        db,
        payment_id="pay-duplicate-retry",
        payment_record=payment_record,
        source="test",
        logger=logging.getLogger("test"),
    )

    assert first["status"] == "processed"
    assert second["status"] == "already_processed"
    assert db.gems_added == 100


@pytest.mark.asyncio
async def test_successful_shop_set_payment_grants_pack_and_marks_reward_step():
    payment_record = {
        "payment_id": "pay-shop-set-10rub",
        "user_id": 1001,
        "amount": 10,
        "currency": "RUB",
        "description": "Test Shop Set",
        "metadata": {"item_type": "shop_set_7", "shop_set_id": 7, "item_name": "Test Shop Set"},
        "rewards_processed": False,
    }
    db = ShopSetPaymentProcessingDB(payment_record)

    result = await process_successful_payment(
        db,
        payment_id="pay-shop-set-10rub",
        payment_record=payment_record,
        source="yookassa_webhook",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert result["attachments"]["shop_set_id"] == 7
    assert result["attachments"]["granted"] == [
        {"type": "gems", "amount": 25},
        {"type": "cosmetic", "cosmetic_slug": "avatar_paid_test", "acquired": True},
    ]
    assert db.shop_set_grants == [{"conn": db, "user_id": 1001, "set_id": 7}]
    assert db.marked_steps == ["shop_set_7"]
    assert db.payment_record["metadata"]["reward_steps"]["shop_set_7"] is True
    assert db.marked_processed is True
    assert db.released == 0
    assert db.mail["attachments"]["shop_set_id"] == 7


@pytest.mark.asyncio
async def test_successful_squad_boost_payment_activates_current_clan_and_marks_reward_step():
    payment_record = {
        "payment_id": "pay-squad-boost",
        "user_id": 1001,
        "amount": 299,
        "currency": "RUB",
        "description": "Boost сквада",
        "metadata": {"item_type": "squad_boost", "item_name": "Boost сквада"},
        "rewards_processed": False,
    }
    db = SquadBoostPaymentProcessingDB(payment_record)

    result = await process_successful_payment(
        db,
        payment_id="pay-squad-boost",
        payment_record=payment_record,
        source="yookassa_webhook",
        logger=logging.getLogger("test"),
    )

    assert result["status"] == "processed"
    assert result["attachments"]["squad_boost"] is True
    assert result["attachments"]["clan_id"] == 10
    assert result["attachments"]["boost_public_id"] == 777
    assert result["attachments"]["member_slots_added"] == 5
    assert result["rewards_text"] == ["⚡ Boost сквада активирован", "+5 мест в скваде"]
    assert db.activations == [{"user_id": 1001, "executor": db}]
    assert db.marked_steps == ["squad_boost_activation"]
    assert db.payment_record["metadata"]["reward_steps"]["squad_boost_activation"] is True
    assert db.marked_processed is True
    assert db.released == 0
    assert db.mail["attachments"]["squad_boost"] is True


@pytest.mark.asyncio
async def test_shop_set_rewards_support_cosmetics_and_preserve_case_type():
    db = ShopSetRewardApplyDB()
    conn = ShopSetRewardConn()
    rewards, error = db._normalize_shop_set_rewards(
        [
            {"type": "case", "amount": 2},
            {"type": "cosmetic", "cosmetic_slug": "avatar_gold", "auto_equip": True},
        ]
    )
    assert error is None

    granted = await db._apply_shop_set_rewards_on_conn(conn, 1001, 7, rewards)

    assert {"type": "case", "amount": 2} in granted
    assert {
        "type": "cosmetic",
        "cosmetic_slug": "avatar_gold",
        "auto_equip": True,
        "acquired": True,
    } in granted
    assert db.cosmetic_grants == [
        {"user_id": 1001, "slug": "avatar_gold", "source": "shop_set", "auto_equip": True}
    ]
    assert any("UPDATE users SET keys" in query for query, _args in conn.execute_calls)


@pytest.mark.asyncio
async def test_shop_set_card_background_reward_replaces_owned_card_with_gems():
    db = ShopSetRewardApplyDB()
    conn = ShopSetOwnedCardFallbackConn()
    rewards, error = db._normalize_shop_set_rewards(
        [
            {"type": "card", "card_id": 77},
            {"type": "cosmetic", "cosmetic_slug": "bg_gold", "auto_equip": True},
        ]
    )
    assert error is None

    granted = await db._apply_shop_set_rewards_on_conn(conn, 1001, 12, rewards)

    assert {"type": "gems", "amount": 50, "fallback_for": "owned_card", "card_id": 77} in granted
    assert not any("INSERT INTO user_cards" in query for query, _args in conn.execute_calls)
    assert any("UPDATE users SET gems" in query for query, _args in conn.execute_calls)


@pytest.mark.asyncio
async def test_shop_set_owned_card_without_background_converts_to_particles():
    db = ShopSetRewardApplyDB()
    conn = ShopSetOwnedCardParticleConn()
    rewards, error = db._normalize_shop_set_rewards([{"type": "card", "card_id": 77}])
    assert error is None

    granted = await db._apply_shop_set_rewards_on_conn(conn, 1001, 13, rewards)

    assert granted == [{
        "type": "particles",
        "amount": 3,
        "fallback_for": "owned_card",
        "card_id": 77,
        "card_name": "Duplicate Rare",
        "rarity": "rare",
    }]
    assert conn.particles == 13
    assert not any("SET level = user_cards.level + 1" in query for query, _args in conn.execute_calls)


@pytest.mark.asyncio
async def test_shop_set_reward_validation_rejects_missing_card_and_inactive_cosmetic():
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))
    db._pool = object()

    async def fake_fetchval(query, *args):
        normalized = " ".join(str(query).split())
        if "FROM cards" in normalized:
            return 1 if int(args[0]) == 10 else None
        if "FROM cosmetic_items" in normalized:
            return 1 if str(args[0]) == "avatar_active" else None
        return None

    db.fetchval = fake_fetchval

    missing_card = await db.validate_shop_set_rewards([{"type": "card", "card_id": 404}])
    missing_particles_card = await db.validate_shop_set_rewards(
        [{"type": "particles", "card_id": 405, "amount": 10}]
    )
    inactive_cosmetic = await db.validate_shop_set_rewards(
        [{"type": "cosmetic", "cosmetic_slug": "avatar_inactive"}]
    )
    valid = await db.validate_shop_set_rewards(
        [
            {"type": "card", "card_id": 10},
            {"type": "cosmetic", "cosmetic_slug": "avatar_active", "auto_equip": False},
        ]
    )

    assert missing_card == ([], "reward_card_not_found")
    assert missing_particles_card == ([], "reward_card_not_found")
    assert inactive_cosmetic == ([], "reward_cosmetic_not_found")
    assert valid[1] is None


def test_payment_processing_claim_can_reclaim_stale_processing_marker():
    source = __import__("pathlib").Path("infrastructure/database.py").read_text()

    assert "rewards_processing_started_at" in source
    assert "900" in source or "15 minutes" in source
    assert "EXTRACT(EPOCH FROM NOW())" in source
