"""Runtime verification of reward behavior across all reward types and fallbacks.

This script is NOT a pytest test — it is an integration verifier that talks to the
local dev PostgreSQL (port 5432) and exercises `claim_reward_entries_transaction`
directly. It synthesizes test users and scenarios, then asserts the resulting
granted payload + side effects in DB match expectations.

Coverage:
- coins / gems / keys (basic currencies)
- case (T1–T5)
- particles (with card_id, with fallback when target card not owned)
- card (random rarity, with fallback when all cards of rarity already owned)
- specific_card (with fallback when target card already owned)
- cosmetic (with fallback when slug not found)

Run with: .venv/bin/python tests/runtime_verify_glory_rewards.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path so we can import infrastructure
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infrastructure.config import DatabaseSettings  # noqa: E402
from infrastructure.database import Database  # noqa: E402


DB_URL = os.environ.get(
    "RUNTIME_VERIFY_DATABASE_URL",
    "postgresql://user:password@127.0.0.1:5432/extraarena",
)


def _settings_from_url(url: str) -> DatabaseSettings:
    # url: postgresql://user:password@host:port/db
    rest = url.split("://", 1)[1]
    creds, hostdb = rest.split("@", 1)
    user, password = creds.split(":", 1)
    host_port, database = hostdb.split("/", 1)
    host, port = host_port.split(":", 1)
    return DatabaseSettings(host=host, port=int(port), user=user, password=password, database=database)


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def step(label: str) -> None:
    print(f"\n--- {label}")


def assert_eq(got, expected, label: str) -> None:
    if got != expected:
        raise AssertionError(f"{label}: expected={expected!r}, got={got!r}")
    print(f"  OK: {label} == {expected!r}")


def assert_in(needle, haystack, label: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: expected to find {needle!r} in {haystack!r}")
    print(f"  OK: {label} contains {needle!r}")


async def reset_test_user(conn, user_id: int) -> None:
    await conn.execute(
        "DELETE FROM claimed_rewards WHERE user_id = $1 AND track_type = 'test_runtime'",
        user_id,
    )
    await conn.execute(
        "DELETE FROM economy_events WHERE user_id = $1 AND metadata->>'track_type' = 'test_runtime'",
        user_id,
    )
    await conn.execute(
        """
        UPDATE users
        SET coins = 0, gems = 0, keys = 0
        WHERE user_id = $1
        """,
        user_id,
    )
    await conn.execute("DELETE FROM user_cards WHERE user_id = $1", user_id)
    await conn.execute("DELETE FROM user_cases WHERE user_id = $1", user_id)


async def ensure_test_user(db: Database, user_id: int, label: str) -> None:
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        await conn.execute(
            """
            INSERT INTO users (user_id, username, first_name, last_name, trophies,
                               max_trophies, league, is_bot, status)
            VALUES ($1, $2, 'Test', $2, 12000, 12000, 10, TRUE, 'active')
            ON CONFLICT (user_id) DO UPDATE
            SET trophies = 12000,
                max_trophies = 12000,
                league = 10,
                is_bot = TRUE,
                status = 'active',
                updated_at = NOW()
            """,
            user_id,
            label,
        )
        await reset_test_user(conn, user_id)


async def test_coins_gems_keys(db: Database) -> None:
    banner("TEST: coins / gems / keys — basic currencies grant correctly")
    user_id = 900001
    await ensure_test_user(db, user_id, "currencies")
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=1,
        entries=[
            {"reward_type": "coins", "reward_amount": 150, "reward_meta": None},
            {"reward_type": "gems", "reward_amount": 25, "reward_meta": None},
            {"reward_type": "keys", "reward_amount": 3, "reward_meta": None},
        ],
    )
    assert_eq(result.get("success"), True, "claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 3, "granted length")
    assert_eq(granted[0]["reward_type"], "coins", "first grant type")
    assert_eq(granted[0]["reward_amount"], 150, "coins amount")
    assert_eq(granted[1]["reward_type"], "gems", "second grant type")
    assert_eq(granted[1]["reward_amount"], 25, "gems amount")
    assert_eq(granted[2]["reward_type"], "keys", "third grant type")
    assert_eq(granted[2]["reward_amount"], 3, "keys amount")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        coins = await conn.fetchval("SELECT coins FROM users WHERE user_id = $1", user_id)
        gems = await conn.fetchval("SELECT gems FROM users WHERE user_id = $1", user_id)
        keys = await conn.fetchval("SELECT keys FROM users WHERE user_id = $1", user_id)
    assert_eq(coins, 150, "user.coins after grant")
    assert_eq(gems, 25, "user.gems after grant")
    assert_eq(keys, 3, "user.keys after grant")


async def test_case_tiers(db: Database) -> None:
    banner("TEST: case — T1..T5 creates user_cases row with correct tier")
    user_id = 900002
    await ensure_test_user(db, user_id, "cases")
    for tier in (1, 2, 3, 4, 5):
        position = 100 + tier  # unique position per tier
        result = await db.claim_reward_entries_transaction(
            user_id=user_id,
            track_type="test_runtime",
            position=position,
            entries=[{"reward_type": "case", "reward_amount": tier, "reward_meta": None}],
        )
        assert_eq(result.get("success"), True, f"claim success tier={tier}")
        granted = result.get("granted") or []
        assert_eq(len(granted), 1, f"granted length tier={tier}")
        assert_eq(granted[0]["reward_type"], "case", f"type tier={tier}")
        assert_eq(granted[0]["case_tier"], tier, f"case_tier tier={tier}")
        assert "user_case_id" in granted[0], f"user_case_id present tier={tier}"

        async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
            row = await conn.fetchrow(
                "SELECT case_id, tier, status FROM user_cases WHERE id = $1",
                int(granted[0]["user_case_id"]),
            )
        assert_eq(row["case_id"], tier, f"DB case_id tier={tier}")
        assert_eq(row["tier"], tier, f"DB tier tier={tier}")
        assert_eq(row["status"], "pending", f"DB status tier={tier}")


async def test_case_invalid_tier(db: Database) -> None:
    banner("TEST: case — invalid tier (0, 6) is rejected with error")
    user_id = 900002
    for bad_tier in (0, 6):
        result = await db.claim_reward_entries_transaction(
            user_id=user_id,
            track_type="test_runtime",
            position=200 + bad_tier,
            entries=[{"reward_type": "case", "reward_amount": bad_tier, "reward_meta": None}],
        )
        assert_eq(result.get("success"), False, f"success for tier={bad_tier}")
        assert_eq(result.get("error"), "invalid_case_tier", f"error code tier={bad_tier}")


async def test_particles_owned(db: Database) -> None:
    banner("TEST: particles — when target card is owned, grants particles")
    user_id = 900003
    await ensure_test_user(db, user_id, "particles_owned")
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        # Card 46 is Уссоп, common. Give it to user so particles target resolves.
        await conn.execute(
            "INSERT INTO user_cards (user_id, card_id, level, particles) VALUES ($1, 46, 1, 0)",
            user_id,
        )
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=300,
        entries=[
            {
                "reward_type": "particles",
                "reward_amount": 30,
                "reward_meta": {"card_id": 46, "rarity": "superrare", "card_name": "Уссоп"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "particles", "grant type")
    assert_eq(granted[0]["reward_amount"], 30, "particles amount")
    assert_eq(granted[0]["card_id"], 46, "card_id")
    assert "fallback_for" not in granted[0], "no fallback when owned"

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        particles = await conn.fetchval(
            "SELECT particles FROM user_cards WHERE user_id = $1 AND card_id = 46",
            user_id,
        )
    assert_eq(particles, 30, "DB particles on card 46")


async def test_particles_not_owned(db: Database) -> None:
    banner("TEST: particles — fallback to coins when target card NOT owned")
    user_id = 900004
    await ensure_test_user(db, user_id, "particles_fallback")
    # Card 46 deliberately NOT granted.
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=301,
        entries=[
            {
                "reward_type": "particles",
                "reward_amount": 30,
                "reward_meta": {"card_id": 46, "rarity": "superrare", "card_name": "Уссоп"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success (fallback is in-grant)")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "coins", "fallback grant type")
    assert_eq(granted[0]["fallback_for"], "particles", "fallback_for marker")
    # 30 particles * 10 * 1.2 (superrare) = 360
    assert_eq(granted[0]["reward_amount"], 360, "fallback coins amount")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        coins = await conn.fetchval("SELECT coins FROM users WHERE user_id = $1", user_id)
        particles = await conn.fetchval(
            "SELECT particles FROM user_cards WHERE user_id = $1 AND card_id = 46",
            user_id,
        )
    assert_eq(coins, 360, "user.coins after fallback")
    assert_eq(particles, None, "no particles row created (card never owned)")


async def test_random_card_grant(db: Database) -> None:
    banner("TEST: card (random) — picks from rarity and grants if not owned")
    user_id = 900005
    await ensure_test_user(db, user_id, "card_random")
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=400,
        entries=[
            {
                "reward_type": "card",
                "reward_amount": 1,
                "reward_meta": {"rarity": "epic"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "card", "grant type")
    assert_eq(granted[0]["rarity"], "epic", "rarity matched")
    assert_in("card_id", granted[0], "card_id present in granted payload")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        cards = await conn.fetch(
            "SELECT card_id FROM user_cards WHERE user_id = $1", user_id,
        )
    assert_eq(len(cards), 1, "exactly one card granted")
    epic_card_ids = {18, 30, 43, 44, 45}
    assert_eq(int(cards[0]["card_id"]) in epic_card_ids, True, "granted card is in epic set")


async def test_random_card_fallback_all_owned(db: Database) -> None:
    banner("TEST: card (random) — fallback to coins when all cards of rarity owned")
    user_id = 900006
    await ensure_test_user(db, user_id, "card_random_fallback")
    # Own every epic warrior card.
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        for cid in (18, 30, 43, 44, 45):
            await conn.execute(
                "INSERT INTO user_cards (user_id, card_id, level, particles) VALUES ($1, $2, 1, 0)",
                user_id, cid,
            )

    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=401,
        entries=[
            {
                "reward_type": "card",
                "reward_amount": 1,
                "reward_meta": {"rarity": "epic"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success (fallback in-grant)")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "coins", "fallback grant type")
    assert_eq(granted[0]["fallback_for"], "card", "fallback_for marker")
    assert_eq(granted[0]["fallback_rarity"], "epic", "fallback rarity")
    # epic is rarity_ordinal = 5 → 5 * 100 = 500 coins
    assert_eq(granted[0]["reward_amount"], 500, "fallback coins amount")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        coins = await conn.fetchval("SELECT coins FROM users WHERE user_id = $1", user_id)
    assert_eq(coins, 500, "user.coins after fallback")


async def test_specific_card_not_owned(db: Database) -> None:
    banner("TEST: specific_card — grants card when not owned")
    user_id = 900007
    await ensure_test_user(db, user_id, "specific_card_grant")
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=500,
        entries=[
            {
                "reward_type": "specific_card",
                "reward_amount": 1,
                "reward_meta": {"card_id": 20, "card_name": "Канеки Кен", "rarity": "legendary"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "specific_card", "grant type")
    assert_eq(granted[0]["card_id"], 20, "card_id")
    assert_eq(granted[0]["rarity"], "legendary", "rarity")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        owned = await conn.fetchval(
            "SELECT 1 FROM user_cards WHERE user_id = $1 AND card_id = 20",
            user_id,
        )
    assert_eq(owned, 1, "card 20 owned in DB")


async def test_specific_card_already_owned(db: Database) -> None:
    banner("TEST: specific_card — duplicate converts to particles")
    user_id = 900008
    await ensure_test_user(db, user_id, "specific_card_owned")
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        await conn.execute(
            "INSERT INTO user_cards (user_id, card_id, level, particles) VALUES ($1, 20, 1, 0)",
            user_id,
        )

    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=501,
        entries=[
            {
                "reward_type": "specific_card",
                "reward_amount": 1,
                "reward_meta": {"card_id": 20, "card_name": "Канеки Кен", "rarity": "legendary"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success (duplicate conversion)")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "particles", "fallback type")
    assert_eq(granted[0]["fallback_for"], "specific_card", "fallback_for marker")
    assert_eq(granted[0]["rarity"], "legendary", "fallback rarity")
    assert_eq(granted[0]["reward_amount"], 26, "duplicate particles amount")

    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        coins = await conn.fetchval("SELECT coins FROM users WHERE user_id = $1", user_id)
        particles = await conn.fetchval(
            "SELECT particles FROM user_cards WHERE user_id = $1 AND card_id = 20",
            user_id,
        )
    assert_eq(coins, 0, "user.coins unchanged after duplicate")
    assert_eq(particles, 26, "duplicate particles added")


async def test_specific_card_invalid_id(db: Database) -> None:
    banner("TEST: specific_card — error when card_id is missing or unknown")
    user_id = 900009
    await ensure_test_user(db, user_id, "specific_card_bad")

    # Missing card_id
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=600,
        entries=[{"reward_type": "specific_card", "reward_amount": 1, "reward_meta": {}}],
    )
    assert_eq(result.get("success"), False, "missing card_id → fail")
    assert_eq(result.get("error"), "specific_card_id_required", "missing card_id error code")

    # Unknown card_id
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=601,
        entries=[
            {
                "reward_type": "specific_card",
                "reward_amount": 1,
                "reward_meta": {"card_id": 999999},
            }
        ],
    )
    assert_eq(result.get("success"), False, "unknown card_id → fail")
    assert_eq(result.get("error"), "specific_card_not_found", "unknown card_id error code")


async def test_cosmetic_grant(db: Database) -> None:
    banner("TEST: cosmetic — graceful path (uses slug resolver, may fall back)")
    user_id = 900010
    await ensure_test_user(db, user_id, "cosmetic")

    # Discover a valid cosmetic slug from the table.
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        row = await conn.fetchrow(
            "SELECT slug FROM cosmetic_items WHERE slug IS NOT NULL LIMIT 1"
        )
    if not row:
        print("  SKIP: no cosmetics rows in DB — nothing to test")
        return
    slug = row["slug"]
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=700,
        entries=[
            {
                "reward_type": "cosmetic",
                "reward_amount": 1,
                "reward_meta": {"cosmetic_slug": slug, "auto_equip": False, "fallback_rarity": "rare"},
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    # Either granted cosmetic or fallback to coins — both are valid behaviour.
    grant_type = granted[0]["reward_type"]
    assert_in(grant_type, ("cosmetic", "coins"), "grant type is cosmetic or fallback")
    if grant_type == "cosmetic":
        assert_eq(granted[0]["cosmetic_slug"], slug, "cosmetic slug echoed")
        print("  Cosmetic granted directly (acquired=%s)" % granted[0].get("acquired"))
    else:
        assert_eq(granted[0]["fallback_for"], "cosmetic", "fallback marker")


async def test_cosmetic_unknown_slug(db: Database) -> None:
    banner("TEST: cosmetic — unknown slug falls back to coins")
    user_id = 900011
    await ensure_test_user(db, user_id, "cosmetic_fallback")
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=701,
        entries=[
            {
                "reward_type": "cosmetic",
                "reward_amount": 1,
                "reward_meta": {
                    "cosmetic_slug": "definitely-not-a-real-slug-xyz",
                    "auto_equip": False,
                    "fallback_rarity": "epic",
                },
            }
        ],
    )
    assert_eq(result.get("success"), True, "claim success (fallback in-grant)")
    granted = result.get("granted") or []
    assert_eq(len(granted), 1, "granted length")
    assert_eq(granted[0]["reward_type"], "coins", "fallback type")
    assert_eq(granted[0]["fallback_for"], "cosmetic", "fallback marker")
    assert_eq(granted[0]["fallback_rarity"], "epic", "fallback rarity respected")
    # epic ordinal 5 → 500 coins
    assert_eq(granted[0]["reward_amount"], 500, "fallback coins amount")


async def test_already_claimed(db: Database) -> None:
    banner("TEST: claim — second attempt at same position returns already_claimed")
    user_id = 900012
    await ensure_test_user(db, user_id, "duplicate")
    entries = [{"reward_type": "coins", "reward_amount": 50, "reward_meta": None}]
    r1 = await db.claim_reward_entries_transaction(
        user_id=user_id, track_type="test_runtime", position=800, entries=entries,
    )
    assert_eq(r1.get("success"), True, "first claim success")
    r2 = await db.claim_reward_entries_transaction(
        user_id=user_id, track_type="test_runtime", position=800, entries=entries,
    )
    assert_eq(r2.get("success"), False, "second claim fails")
    assert_eq(r2.get("error"), "already_claimed", "already_claimed error")


async def test_multi_entry_set(db: Database) -> None:
    banner("TEST: combined set — coins + case + particles + card, each branch executes")
    user_id = 900013
    await ensure_test_user(db, user_id, "combo")
    async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
        # Own card 46 so particles work.
        await conn.execute(
            "INSERT INTO user_cards (user_id, card_id, level, particles) VALUES ($1, 46, 1, 0)",
            user_id,
        )
    result = await db.claim_reward_entries_transaction(
        user_id=user_id,
        track_type="test_runtime",
        position=900,
        entries=[
            {"reward_type": "coins", "reward_amount": 100, "reward_meta": None},
            {"reward_type": "case", "reward_amount": 2, "reward_meta": None},
            {
                "reward_type": "particles",
                "reward_amount": 15,
                "reward_meta": {"card_id": 46, "rarity": "superrare", "card_name": "Уссоп"},
            },
            {"reward_type": "card", "reward_amount": 1, "reward_meta": {"rarity": "rare"}},
        ],
    )
    assert_eq(result.get("success"), True, "combo claim success")
    granted = result.get("granted") or []
    assert_eq(len(granted), 4, "all 4 entries granted")
    types = [g["reward_type"] for g in granted]
    assert_eq(types, ["coins", "case", "particles", "card"], "branch order preserved")


async def main() -> None:
    settings = _settings_from_url(DB_URL)
    db = Database(settings)
    await db.connect()
    try:
        await test_coins_gems_keys(db)
        await test_case_tiers(db)
        await test_case_invalid_tier(db)
        await test_particles_owned(db)
        await test_particles_not_owned(db)
        await test_random_card_grant(db)
        await test_random_card_fallback_all_owned(db)
        await test_specific_card_not_owned(db)
        await test_specific_card_already_owned(db)
        await test_specific_card_invalid_id(db)
        await test_cosmetic_grant(db)
        await test_cosmetic_unknown_slug(db)
        await test_already_claimed(db)
        await test_multi_entry_set(db)
    finally:
        # Clean up
        async with db._pool.acquire() as conn:  # type: ignore[attr-defined]
            test_user_ids = tuple(range(900001, 900020))
            await conn.execute(
                "DELETE FROM claimed_rewards WHERE track_type = 'test_runtime' AND user_id = ANY($1::int[])",
                list(test_user_ids),
            )
            await conn.execute(
                "DELETE FROM economy_events WHERE metadata->>'track_type' = 'test_runtime' AND user_id = ANY($1::int[])",
                list(test_user_ids),
            )
            await conn.execute("DELETE FROM user_cards WHERE user_id = ANY($1::int[])", list(test_user_ids))
            await conn.execute("DELETE FROM user_cases WHERE user_id = ANY($1::int[])", list(test_user_ids))
            await conn.execute("DELETE FROM users WHERE user_id = ANY($1::int[])", list(test_user_ids))
        await db.close()
    print("\nAll runtime reward checks passed.\n")


if __name__ == "__main__":
    asyncio.run(main())
