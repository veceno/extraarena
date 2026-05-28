# ExtraArena App Architecture Notes

## Client Split

Both clients should use the same game backend.

- Telegram WebApp authenticates with Telegram `initData`.
- Android App authenticates with ExtraID/JWT.
- Android App must not depend on Telegram services at runtime.

The Android shell now owns the first-run account flow:

- anonymous play: `POST /api/auth/anonymous` with a nickname, then use the returned JWT
- login: `POST /api/extraid/login`
- registration: `POST /api/extraid/register`, then login to obtain a JWT
- JWT is stored locally and passed into the web frontend as `_auth`
- connection profile switching clears the local JWT
- hidden connection switching is available only while the native loading screen
  is visible; the live WebView never receives a transparent dev overlay

The Android shell loads the same web frontend with:

```text
_auth=<jwt>
ea_platform=android_app
ea_shell=android
ea_telegram=0
ea_app_version=<version>
```

The web frontend should treat Telegram APIs as optional and use
`window.ExtraArenaApp` when present.

## Native Bridge

The Android bridge is exposed as `window.ExtraArenaApp`:

- `getPlatform()`
- `getBaseUrl()`
- `getConnectionProfile()`
- `getAppVersion()`
- `isTestServer()`
- `isWhitelistEnabled()`
- `getWhitelistCode()`
- `setAuthToken(token)`
- `requestPushRegistration()`
- `haptic(style)`
- `openExternal(url)`

The first web-side integration should call `setAuthToken(extra_id_token)` after
login and after token refresh. The native shell also attempts to read
`localStorage.extra_id_token` after page load as a compatibility fallback.

## Push Backend Contract

Add Android push registration endpoints:

```text
POST /api/push/register
POST /api/push/unregister
POST /api/push/test
```

Suggested registration payload:

```json
{
  "auth": "<jwt>",
  "platform": "android",
  "token": "<fcm-token>",
  "app_version": "0.1.0",
  "device_label": "Google Pixel",
  "os_name": "Android",
  "os_version": "16"
}
```

Suggested table:

```text
push_devices:
  id uuid / bigserial
  user_id bigint
  session_id uuid nullable
  platform text
  token_hash text unique
  token_encrypted text
  app_version text
  device_label text
  os_name text
  os_version text
  enabled boolean
  last_seen_at timestamptz
  revoked_at timestamptz nullable
  created_at timestamptz
```

Existing `notification_outbox` can remain the event source. The backend can add
a second delivery channel:

- Telegram dispatcher for Telegram players.
- FCM dispatcher for Android devices.

## Update Flow

No binary update manager is required for now. The backend can send an FCM data
message:

```json
{
  "type": "app_update_required",
  "title": "Хорошие новости!",
  "body": "Вышло обновление, скачай новую версию, чтобы продолжить игру",
  "url": "https://t.me/extraarena"
}
```

The Android app shows a high-priority notification and opens the provided URL.
