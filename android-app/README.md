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
- Android SDK with API 35;
- Gradle 8.7+ for AGP 8.6.x.
- Python 3 with Pillow (the release build verifies the optimized card pack).

CLI build:

```bash
EXTRAARENA_PYTHON=python3 ./gradlew \
  :app:assembleDirectRelease :app:bundleDirectRelease \
  :app:assembleRustoreRelease :app:bundleRustoreRelease
```

Every build first verifies that `webapp/index.compiled.js` matches the current
game and that `optimized-assets/` matches every canonical card by stem,
dimensions and SHA-256. Regenerate the Android-only full-size WebP pack with:

```bash
python3 scripts/optimize_assets.py \
  --source ../DesignAssets \
  --output optimized-assets
```

The shell excludes duplicate archives/copies, legacy arena themes and the
full-size welcome-carousel duplicates. It also omits the obsolete runtime Babel
compiler because the current game shell is precompiled. Canonical source art is
not modified.

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

## Native updates

`GET /api/mobile/client-version` is the release manifest. Availability and
enforcement are deliberately separate:

```text
update_available = installed < latest_version_code
required         = installed < min_supported_version_code
```

The minimum-supported floor is monotonic. Publishing an optional release after
a mandatory one therefore never unlocks an older client.

The `direct` flavor downloads the published APK through Android
`DownloadManager`, displays real progress, verifies size, SHA-256, package,
version and signer, requests the per-app unknown-sources permission when needed,
then opens the system package installer through `FileProvider`. It never
uninstalls the app. The `REQUEST_INSTALL_PACKAGES` permission and provider are
absent from the `rustore` flavor, which continues to use the RuStore SDK. AAB is
an admin/store artifact and is never offered to Android's package installer.
The RuStore SDK listener completes downloaded flexible updates and recovers a
downloaded update after process restart. Publish the matching backend manifest
only after the exact version is already live in RuStore Console; the release
service requires that explicit confirmation and ExtraAdmin disables RuStore
auto-publish.

In-place upgrades preserve anonymous/ExtraID credentials and connection
profiles only when package name and signing lineage match. Direct and RuStore
release signing can therefore be configured independently:

```text
ANDROID_DIRECT_RELEASE_STORE_FILE / _STORE_PASSWORD / _KEY_ALIAS / _KEY_PASSWORD
ANDROID_RUSTORE_RELEASE_STORE_FILE / _STORE_PASSWORD / _KEY_ALIAS / _KEY_PASSWORD
```

The generic `ANDROID_RELEASE_*` signing variables remain fallback values.

## Update push

The update push is data-driven. Send an FCM message with this data:

```json
{
  "type": "app_update_required",
  "title": "Хорошие новости!",
  "body": "Вышло обновление, скачай новую версию, чтобы продолжить игру",
  "release_id": "published-release-id"
}
```

The notification opens `MainActivity`; the client always reloads the signed
manifest and derives optional/required state there. Push payloads do not decide
the gate and do not send the user to a browser.

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

## Menu music

The Android shell owns `DesignAssets/Sounds/main_theme.mp3` while the app is on
native loading, update, welcome, login and registration screens as well as the
web main menu. The WebView bridge synchronizes the setting and audio scene so
menu and arena music never overlap. Audio focus and Activity lifecycle pause,
duck, resume and release playback normally.
