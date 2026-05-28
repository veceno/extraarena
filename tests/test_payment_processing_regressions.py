import logging

import pytest

from infrastructure.payments_logic import process_successful_payment


class ClaimFailedPaymentDB:
    def __init__(self):
        self.gems_added = 0
        self.claim_attempted = False

    async def claim_payment_for_processing(self, payment_id):
        self.claim_attempted = True
        return None

    async def execute(self, query, *args):
        if "SET gems = gems + $1" in query:
            self.gems_added += int(args[0])
        return "UPDATE 1"

    async def fetchval(self, query, *args):
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
