# Mobile Performance Shell Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Android client feel faster by shipping the heavy UI shell and common assets inside the APK, while keeping live gameplay/API data on the server.

**Architecture:** WebView keeps the same production/test profile origin, but `MainActivity` serves root HTML, JS/CSS, vendor scripts, fonts, sounds, card art, and static design assets from APK assets through `shouldInterceptRequest`. The web client keeps a persistent JSON cache in `localStorage` and uses stale-while-revalidate for profile/runtime/shop/collection bootstrap data.

**Tech Stack:** Android Java WebView, Gradle asset sync, WebView request interception, aiohttp static headers, browser `localStorage`.

---

### Task 1: Regression Tests

**Files:**
- Modify: `tests/test_mobile_client_frontend.py`

- [ ] Add a static test that requires Android to sync `webapp`, `DesignAssets`, and vendor libraries into APK assets.
- [ ] Add a static test that requires `MainActivity` to intercept root shell, webapp files, card images, CDN scripts, Telegram script, and Google Fonts.
- [ ] Add a static test that requires persistent stale-while-revalidate cache APIs in `webapp/index.html`.

### Task 2: APK Shell Assets

**Files:**
- Modify: `android-app/app/build.gradle.kts`
- Create during build: `android-app/app/build/generated/extraArenaShellAssets/**`

- [ ] Add a Gradle `Sync` task that copies `webapp/*.html`, `webapp/*.js`, `webapp/*.css`, `DesignAssets/**`, and `android-app/app/src/main/assets/ea_vendor/**` into generated APK assets.
- [ ] Register the generated asset directory in the Android `main` source set.
- [ ] Make `preBuild` depend on the sync task.

### Task 3: WebView Local Shell

**Files:**
- Modify: `android-app/app/src/main/java/ru/extraarena/app/MainActivity.java`

- [ ] Add `shouldInterceptRequest` routing for the selected connection-profile host.
- [ ] Serve `/`, `/index.html`, `/arena.html`, `/safe-area.js`, `/matchmaking-tips.config.js`, `/arena.js`, `/arena-styles.css`, and other packaged webapp files from APK assets.
- [ ] Serve `/DesignAssets/**` and `/api/cards/image?card_id=N` from APK assets when available.
- [ ] Serve local vendor replacements for React, ReactDOM, Babel, DOMPurify, Socket.IO client, Telegram WebApp stub, and Google Fonts CSS.
- [ ] Fall through to network for `/api/**` except local card images.

### Task 4: Persistent JSON Cache

**Files:**
- Modify: `webapp/index.html`

- [ ] Extend `eaCachedJson` to persist successful JSON responses in `localStorage`.
- [ ] Add stale-while-revalidate behavior that returns stale data immediately and refreshes in background.
- [ ] Dispatch `ea-json-cache-updated` events after background refresh.
- [ ] Wire root profile/runtime state to those events.
- [ ] Use persistent cache for `loadProfile`, `loadRuntimeStatus`, and Android warm-up endpoints.

### Task 5: Server Static Cache

**Files:**
- Modify: `web/server.py`

- [ ] Replace `no-store` for static webapp and DesignAssets responses with long-lived public cache headers.
- [ ] Keep `no-store` on dynamic HTML/API where necessary.

### Task 6: Verification

**Commands:**
- `python3 -m py_compile web/server.py`
- `pytest -q tests/test_mobile_client_frontend.py`
- `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :app:assembleDebug`
- `aapt dump xmltree android-app/app/build/outputs/apk/debug/app-debug.apk AndroidManifest.xml`
