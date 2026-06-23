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
| `notif_daily_rewards` | `true` | Daily login reward availability |
| `notif_extra_arena_modifiers` | `true` | ExtraArena modifier change events |
| `notif_squad_member_role` | `true` | Squad promote/demote notifications |
| `notif_squad_new_member` | `true` | New squad member notifications |
| `notif_squad_disbanded` | `true` | Squad disband notifications |
| `notif_squad_boost` | `true` | Squad Boost notifications |

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
| `_scheduled_notifications_task` | Enqueues shop and reminder notifications when schedules are due |
| `_notification_outbox_task` | Sends queued messages through Android FCM first, then Telegram fallback |

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
