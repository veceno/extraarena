# Notification System

## Architecture

New player notifications use `notification_outbox` as a durable delivery queue. Event producers enqueue records with `user_id`, `category`, `event_type`, JSON payload, and a unique `dedupe_key`. The dispatcher formats one canonical message and sends it to the Android app through FCM when the player has a registered app device; otherwise it falls back to the Telegram WebApp button flow. Rows are marked as `sent`, `failed`, or `blocked`.

Legacy `notifications` remains for old cooldown-style notifications such as dice readiness.

## Settings

`user_settings` contains category toggles:

| Column | Default | Scope |
| --- | --- | --- |
| `notif_shop` | `false` | Daily shop particles |
| `notif_generator` | `true` | Key generator events |
| `notif_reminders` | `false` | Daily spontaneous reminders |
| `notif_daily_rewards` | `false` | Daily login reward availability |
| `notif_extra_arena_modifiers` | `true` | ExtraArena modifier change events |
| `notif_squad_member_role` | `true` | Squad promote/demote notifications |
| `notif_squad_new_member` | `true` | New squad member notifications |
| `notif_squad_disbanded` | `true` | Squad disband notifications |
| `notif_squad_boost` | `true` | Squad Boost notifications |
| `notif_squad_weekly_tokens` | `true` | Weekly squad CBRP token settlement |
| `notif_game_invites` | `true` | Friendly battle invites |
| `notif_friend_requests` | `true` | Incoming friend requests |

Old notification columns remain for compatibility with previous clients.

## Schedules

`notification_schedules` stores per-user `shop_particles` and `daily_reminder` cadence. Due schedules are enqueued by a background task, then advanced by 24 hours plus a small random jitter. Shop notifications do not create or mutate the particles rotation; they only invite the player to open the shop.

## Event Sources

Generator checks classify events as:

| Event type | Meaning |
| --- | --- |
| `generator_new_key` | A key became available and capacity is not full |
| `generator_full_on_new_key` | A key became available and filled the generator |
| `generator_full_blocked_key` | A key would have been produced, but capacity was already full |

Squad endpoints enqueue notifications after successful join, accepted request, promote, demote, disband, and Boost purchase.

## Delivery

`main.py` runs:

| Task | Responsibility |
| --- | --- |
| `_generator_notifications_task` | Scans generator state and enqueues generator events |
| `_scheduled_notifications_task` | Enqueues shop, reminder, daily-login and extra-arena modifier notifications when schedules are due |
| `_notification_outbox_task` | Sends queued messages through Android FCM first, then Telegram fallback, honoring per-user `notification_delivery_mode` (`app_then_telegram` / `app_only` / `telegram_only`) |

## Android FCM

The Android client registers its FCM token at `/api/push/register` after ExtraID or anonymous JWT auth. Tokens are stored in `push_devices`; invalid tokens are disabled after permanent FCM errors.

Server credentials are configured with env only:

- `FIREBASE_SERVICE_ACCOUNT_FILE`
- `FIREBASE_SERVICE_ACCOUNT_JSON`
- `FIREBASE_SERVICE_ACCOUNT_B64`
- optional `FIREBASE_PROJECT_ID`

Use `GET /api/admin/push/status` to verify Firebase Admin configuration and registered Android device count. Use `POST /api/admin/push/app-update` for the required-update notification:

```json
{
  "title": "Хорошие новости!",
  "body": "Вышло обновление, скачай новую версию, чтобы продолжить игру",
  "url": "https://t.me/extraarena"
}
```

Telegram `Forbidden` or `BadRequest` errors are treated as blocked delivery so the same row is not retried forever. Other send failures return the row to pending until the attempt limit is reached.

## Audit (2026-06-25)

- Verified `infrastructure/notifications.py` (default map + reminder payload) and `infrastructure/push_notifications.py` (FCM sender, env vars, build/broadcast helpers).
- Cross-checked `main.py:343-345, 442-497` for the three delivery task names — names match doc.
- Cross-checked `web/server.py:13602-13695` for `/api/admin/push/{status,app-update}` and `web/extraid_handlers.py:1177-1179` for `/api/push/{register,unregister,test}`.
- Cross-checked DB defaults in `infrastructure/database.py:2819-2833` (notif_* columns) and `infrastructure/notifications.py:41-53` (NOTIFICATION_DEFAULTS).
- Cross-checked dispatcher semantics in `infrastructure/database.py:4568-4626` (status flow: pending → sending → sent/failed/blocked; max 5 attempts; permanent FCM errors mark `failed`).

Fixed:
- `docs/notifications/SYSTEM.md:18` — `notif_daily_rewards` default corrected from `true` to `false` (DB schema: `notif_daily_rewards BOOLEAN NOT NULL DEFAULT false`).
- `docs/notifications/SYSTEM.md:23` — table extended with `notif_squad_weekly_tokens`, `notif_game_invites`, `notif_friend_requests` (DB columns 2821, 2822, 2832, all default `true`).
- `docs/notifications/SYSTEM.md:50` — `_scheduled_notifications_task` row updated to mention daily-login and extra-arena modifier enqueueing (matches `main.py:458-475` + `db.enqueue_due_daily_login_notifications` / `db.enqueue_due_scheduled_notifications`).
- `docs/notifications/SYSTEM.md:51` — `_notification_outbox_task` row annotated with the per-user `notification_delivery_mode` enum (matches `main.py:524-557` and the `user_settings.notification_delivery_mode` column at `infrastructure/database.py:2843`).

Not changed but verified:
- Generator event types (`generator_new_key`, `generator_full_on_new_key`, `generator_full_blocked_key`) match `infrastructure/notifications.py:79-87`.
- FCM env vars (`FIREBASE_SERVICE_ACCOUNT_FILE` / `_JSON` / `_B64`, `FIREBASE_PROJECT_ID`) match `infrastructure/push_notifications.py:111-126`; `GOOGLE_APPLICATION_CREDENTIALS` is also accepted as a fallback (line 116) — not doc'd; left as-is to keep changes minimal.
- `app_update` request shape for `/api/admin/push/app-update` matches `web/server.py:13632-13644` (handler also reads `limit` and `dry_run`; doc shows only the visible fields).
- Permanent FCM error markers and status transitions match `infrastructure/push_notifications.py:129-143` and `infrastructure/database.py:4598-4626`.
