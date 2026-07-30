# ExtraArena Telegram iOS Fullsize harness

This development-only app runs the local ExtraArena web app in the real iOS
Simulator `WKWebView`. It adds a Telegram-like top bar and injects a Telegram
WebApp mock with:

- `platform = "ios"`
- `isFullscreen = false`
- a deliberately stale/oversized Telegram viewport height
- a rejected `requestFullscreen()` call
- iPhone safe-area insets

The harness is for layout and WebKit regression testing only. It does not
contain Telegram credentials and does not connect to production.

Build and run:

```sh
xcodebuild \
  -project tests/ios_simulator_harness/ExtraArenaHarness.xcodeproj \
  -scheme ExtraArenaHarness \
  -configuration Debug \
  -sdk iphonesimulator \
  -derivedDataPath /tmp/extraarena-ios-simulator-harness \
  build

xcrun simctl install booted \
  /tmp/extraarena-ios-simulator-harness/Build/Products/Debug-iphonesimulator/ExtraArenaHarness.app
xcrun simctl launch booted ru.extraarena.iosharness
```

The default URL is `http://127.0.0.1:8081/` with a synthetic local user ID.
Override it at launch without changing source:

```sh
SIMCTL_CHILD_EXTRAARENA_URL='http://127.0.0.1:8081/?user_id=123' \
  xcrun simctl launch booted ru.extraarena.iosharness
```
