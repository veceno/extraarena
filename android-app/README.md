# ExtraArena Android App

Native Android shell for ExtraArena.

The app is intentionally Telegram-free:

- auth is expected to use ExtraID/JWT;
- anonymous play calls `/api/auth/anonymous` and receives the same JWT session shape;
- native ExtraID login/registration calls `/api/extraid/login` and `/api/extraid/register`;
- push notifications are expected to use Firebase Cloud Messaging;
- the WebView receives `ea_platform=android_app` and `ea_telegram=0`;
- `window.ExtraArenaApp` exposes a small Android bridge for app-only features.

## Build

Open `android-app/` in Android Studio.

Requirements:

- JDK 17 or newer;
- Android SDK with API 34+;
- Gradle 8.7+ for AGP 8.6.x.

CLI build:

```bash
./gradlew :app:assembleDebug
```

Firebase setup is optional for the first local run. To enable FCM, put the real
`google-services.json` into `android-app/app/`. The Gradle script applies the
Google Services plugin only when that file exists. For this workspace the file
is mirrored from `android-app/Google Firebase/google-services.json`; the private
Admin SDK key stays server-side and is configured through env, not bundled into
the APK.

## Server Switching

Tap the top-left 72dp area five times quickly on the native loading screen to
open connection profiles. The hotspot is not mounted over the live WebView, so
in-game controls in that corner keep working. Fresh installs contain one default
profile: `ExtraArena Worldwide`. Developer profiles can store a title,
`base_url`, and optional WhiteList code. Changing the active profile clears the
local JWT so dev/prod accounts do not get mixed.

Defaults live in `app/build.gradle.kts`:

- `DEFAULT_BASE_URL`
- `TEST_BASE_URL`
- `UPDATE_CHANNEL_URL`

## Update Push

The update push is data-driven. Send an FCM message with this data:

```json
{
  "type": "app_update_required",
  "title": "Хорошие новости!",
  "body": "Вышло обновление, скачай новую версию, чтобы продолжить игру",
  "url": "https://t.me/extraarena"
}
```

The notification opens the Telegram channel URL.

Backend env for Firebase Admin:

```bash
FIREBASE_SERVICE_ACCOUNT_FILE="android-app/Google Firebase/extraarena-94cd6-firebase-adminsdk-fbsvc-9b70609fe9.json"
# or FIREBASE_SERVICE_ACCOUNT_JSON / FIREBASE_SERVICE_ACCOUNT_B64
FIREBASE_PROJECT_ID=extraarena-94cd6
EXTRAARENA_UPDATE_CHANNEL_URL=https://t.me/extraarena
```

Admin helpers:

- `GET /api/admin/push/status`
- `POST /api/admin/push/app-update`
