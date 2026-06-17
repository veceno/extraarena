# Beta Release Checklist

Use this checklist before promoting a beta build. Do not treat a checked box as a substitute for logs, command output, or a smoke-test note.

## Environment And Secrets

- [ ] `ENVIRONMENT` is set explicitly. Use `production` or the agreed beta environment name for non-local binds; do not run a public beta as `development`.
- [ ] `WEBAPP_HOST`, `WEBAPP_PORT`, `WEBAPP_URL`, and `EXTRA_SHOP_URL` point at the beta host and public URLs.
- [ ] `JWT_SECRET` and `ADMIN_SESSION_SECRET` are both set, strong, non-default, and different from each other.
- [ ] Database settings are set for the beta PostgreSQL instance: `DATABASE_URL` or `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`; plus `EXTRAID_DATABASE_URL` or `EXTRAID_DB_HOST`, `EXTRAID_DB_PORT`, `EXTRAID_DB_USER`, `EXTRAID_DB_PASSWORD`, `EXTRAID_DB_NAME` if ExtraID uses the split DB.
- [ ] Payment secrets are set only for the intended beta mode: `ROBOKASSA_MERCHANT_LOGIN`, `ROBOKASSA_PASSWORD1`, `ROBOKASSA_PASSWORD2`, `ROBOKASSA_HASH_ALGO`, `ROBOKASSA_TEST_MODE`, `ROBOKASSA_RESULT_URL`, `ROBOKASSA_SUCCESS_URL`, `ROBOKASSA_FAIL_URL`; optional YooKassa fallback `YOOKASSA_SHOP_ID`, `YOOKASSA_SECRET_KEY`, `YOOKASSA_TEST_MODE`; `RUSTORE_PUBLIC_TOKEN`, `RUSTORE_CONSOLE_APP_ID`, `RUSTORE_KEY_ID`, `RUSTORE_PRIVATE_KEY` or `RUSTORE_PRIVATE_KEY_FILE`, `RUSTORE_SANDBOX`; `STARS_RATE_RUB`, `STARS_MARKUP`, `STARS_TEST_MODE`; and `PAYMENT_PRIMARY_PROVIDER`, `PAYMENT_FALLBACK_PROVIDER`, `PAYMENT_PROVIDER_ORDER`.
- [ ] Telegram `BOT_TOKEN` is set for the beta bot and not shared with unrelated environments.
- [ ] If UGC/community submissions, squad announcements, or image uploads are enabled, `POLZA_AI_KEY` is set and moderation requests are observable in logs. If `POLZA_AI_KEY` is missing, hide/disable UGC entry points rather than adding a broad moderation fallback.
- [ ] CORS allowed origins are explicit: `CORS_ALLOWED_ORIGINS` or `WEBAPP_ALLOWED_ORIGINS` contains only beta web origins, shop origins, and approved local smoke origins.

## Runtime Policy

- [ ] Run one web/bot worker only: set `WEB_CONCURRENCY=1`. In-memory match, matchmaking, pending invite, bot task, and socket state are process-local.
- [ ] Use a drain window before deploy: stop accepting new matches, let active matches finish or timeout, then restart.
- [ ] No horizontal scale for battle or matchmaking traffic until shared Redis/Postgres state replaces in-memory match state.

## Backup, Rollback, And Deploy Safety

- [ ] Take and verify a beta PostgreSQL backup before deploy. Record the backup artifact, restore command, and owner in the release note.
- [ ] Rollback plan is written before deploy: previous image/build id, config snapshot, migration compatibility note, and the person authorized to trigger rollback.
- [ ] Staging smoke passes on the release candidate using beta-like config before production beta deploy.
- [ ] Post-deploy smoke passes on the beta host after deploy; attach timestamp, operator, and any known anomalies.

## Schema, Seed, And Verification Commands

Run migrations/seeds explicitly before switching production traffic. App startup should use the read-only schema gate (`AUTO_MIGRATE_ON_START=false`) after this command succeeds:

```bash
python - <<'PY'
import asyncio

from infrastructure.config import get_settings
from infrastructure.database import Database, SCHEMA_VERSION


async def main():
    settings = get_settings()
    db = Database(settings.database)
    await db.connect()
    try:
        changed = await db.init_schema()
        print(f"schema_version={SCHEMA_VERSION} schema_changed={changed}")
    finally:
        await db.close()


asyncio.run(main())
PY
```

Targeted policy regression:

```bash
pytest -q tests/test_community_news_polls.py::test_beta_release_checklist_documents_config_policy_risks
```

Compile check:

```bash
python -m compileall main.py run_web.py infrastructure web core ai battle_engine.py
```

## Local Smoke Steps

- [ ] Staging smoke: repeat login, deck, battle, payment test-mode, rating, and admin-boundary checks on staging before promoting the build.
- [ ] Login: open the WebApp with a valid Telegram init flow or approved local auth path; verify user profile loads and session survives refresh.
- [ ] Deck: create or edit a deck, save it, refresh, and confirm the saved deck is selected for battle.
- [ ] Battle: start a bot battle and a PvP/friendly flow if available; verify board, hand, mana, action legality, result, and rewards.
- [ ] End-turn/timeout/surrender: complete one manual end turn, force or wait for one timeout path, and surrender once; verify result handling and no stuck match state.
- [ ] Cases/generator: open a case, verify duplicate/economy event behavior, then run the generator upgrade/claim path.
- [ ] Shop/payment test mode: run checkout in the configured test mode for YooKassa/RuStore/Stars as applicable; verify webhook or verifier status and item delivery.
- [ ] Glory/Trophy Road: verify battle progress updates visible Glory or Trophy Road progress and rewards can be claimed once.
- [ ] Admin boundary: verify non-admin users cannot access admin APIs or admin UI actions; verify the configured admin can complete the intended beta operations.
- [ ] Post-deploy smoke: repeat login, deck, battle, rating, and payment delivery checks on the live beta host immediately after deploy.

## Accepted Statuses

- [ ] F-55 FIX_NOW/VERIFIED: community poll/vote/reaction toggles use transaction-safe guards with concurrency regression coverage. Monitor anomalies during beta.
- [ ] F-57 ACCEPTED_RISK/CONFIG_POLICY: UGC requires `POLZA_AI_KEY` when UGC enabled or operators must hide/disable UGC entry points. Do not implement broad moderation fallback before beta.
- [ ] F-02 ACCEPTED_RISK: single worker/drain/no horizontal scale until Redis/Postgres shared state is implemented for match, matchmaking, pending invite, bot task, and socket state.
- [ ] F-12 ACCEPTED_RISK/CONSTRAINT: no god-module refactor before beta, minimal localized patches only.
- [ ] F-35 VERIFIED: deterministic starting hand behavior is covered by regression tests and included in battle smoke observations.
- [ ] F-54 VERIFIED: bot replayability/reproducibility markers are covered by regression tests and included in bot battle smoke observations.
