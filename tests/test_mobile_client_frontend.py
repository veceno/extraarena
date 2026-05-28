from pathlib import Path


INDEX = Path("webapp/index.html")
ARENA = Path("webapp/arena.js")
SAFE_AREA = Path("webapp/safe-area.js")
MAIN_ACTIVITY = Path("android-app/app/src/main/java/ru/extraarena/app/MainActivity.java")
DEVICE_REGISTRAR = Path("android-app/app/src/main/java/ru/extraarena/app/DeviceRegistrar.java")
MESSAGING_SERVICE = Path("android-app/app/src/main/java/ru/extraarena/app/ExtraArenaMessagingService.java")
ANDROID_MANIFEST = Path("android-app/app/src/main/AndroidManifest.xml")
APP_BUILD = Path("android-app/app/build.gradle.kts")
WIDGET_PROVIDER = Path("android-app/app/src/main/java/ru/extraarena/app/ExtraArenaModifierWidgetProvider.java")
WIDGET_UPDATER = Path("android-app/app/src/main/java/ru/extraarena/app/ExtraArenaModifierWidgetUpdater.java")
WIDGET_LAYOUT = Path("android-app/app/src/main/res/layout/widget_extra_arena_modifier.xml")
WIDGET_INFO = Path("android-app/app/src/main/res/xml/extra_arena_modifier_widget.xml")
WEB_SERVER = Path("web/server.py")
SQUAD_WIDGET_UPDATER = Path("android-app/app/src/main/java/ru/extraarena/app/SquadWidgetUpdater.java")
SQUAD_PERSONAL_PROVIDER = Path("android-app/app/src/main/java/ru/extraarena/app/SquadPersonalCbrpWidgetProvider.java")
SQUAD_OVERVIEW_PROVIDER = Path("android-app/app/src/main/java/ru/extraarena/app/SquadOwnerOverviewWidgetProvider.java")
SQUAD_CBRP_PROVIDER = Path("android-app/app/src/main/java/ru/extraarena/app/SquadOwnerCbrpWidgetProvider.java")
SQUAD_WIDGET_LAYOUT = Path("android-app/app/src/main/res/layout/widget_squad_summary.xml")
SQUAD_PERSONAL_INFO = Path("android-app/app/src/main/res/xml/squad_personal_cbrp_widget.xml")
SQUAD_OVERVIEW_INFO = Path("android-app/app/src/main/res/xml/squad_owner_overview_widget.xml")
SQUAD_CBRP_INFO = Path("android-app/app/src/main/res/xml/squad_owner_cbrp_widget.xml")
SHOP_PARTICLES_PROVIDER = Path("android-app/app/src/main/java/ru/extraarena/app/ShopParticlesWidgetProvider.java")
SHOP_PARTICLES_UPDATER = Path("android-app/app/src/main/java/ru/extraarena/app/ShopParticlesWidgetUpdater.java")
SHOP_PARTICLES_LAYOUT = Path("android-app/app/src/main/res/layout/widget_shop_particles.xml")
SHOP_PARTICLES_INFO = Path("android-app/app/src/main/res/xml/shop_particles_widget.xml")
STRINGS = Path("android-app/app/src/main/res/values/strings.xml")


def test_android_external_links_are_forced_outside_webview():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "window.openExternalLink" in source
    assert "document.addEventListener('click', handleExternalAnchorClick" in source
    assert "window.openExternalLink(historyUrl)" in source
    assert "settings.setSupportMultipleWindows(true)" in native
    assert "onCreateWindow" in native


def test_mobile_client_has_afk_disconnect_and_bad_ping_ui():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "AFK_TIMEOUT_MS = 120000" in source
    assert "BAD_CONNECTION_THRESHOLD_MS = 1200" in source
    assert "ConnectionHealthOverlay" in source
    assert "Видимо ты отошел" in source
    assert "Соединение разорвано" in source
    assert "Плохое соединение" in source
    assert "stopConnectionPing" in source
    assert "maintenance_mode" in source
    assert "maintenanceBlocksClient" in source
    assert "showConnectivityError()" in native
    assert "notifyAndroidNoConnection" in source
    assert "fetchWithTimeout" in source


def test_arena_page_monitors_disconnect_and_runtime_maintenance():
    source = ARENA.read_text(encoding="utf-8")

    assert "startArenaHealthMonitor()" in source
    assert "showArenaConnectionModal" in source
    assert "/api/runtime/status" in source
    assert "Плохое соединение" in source
    assert "Соединение разорвано" in source
    assert "maintenance_mode" in source
    assert "connect_error" in source
    assert "arenaBadPingDismissed = true" in source


def test_android_welcome_screen_uses_legal_links_instead_of_create_extraid_cta():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "authLegalNotice" in native
    assert "Продолжая, ты соглашаешься" in native
    assert "офертой" in native
    assert "политикой конфиденциальности" in native
    assert "https://google.com" in native
    assert "authSecondaryAction.setVisibility(step == AuthStep.WELCOME ? View.GONE : View.VISIBLE)" in native


def test_android_haptics_can_be_disabled_from_mobile_settings():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "Виброотдача" in source
    assert "extra_haptics_enabled" in source
    assert "window.toggleHaptics" in source
    assert "setHapticsEnabled" in native
    assert "isHapticsEnabled" in native
    assert "KEY_HAPTICS_ENABLED" in native
    assert "if (!isHapticsEnabled())" in native


def test_android_only_notification_settings_are_conditionally_visible_and_include_modifiers():
    source = INDEX.read_text(encoding="utf-8")
    server = WEB_SERVER.read_text(encoding="utf-8")
    database = Path("infrastructure/database.py").read_text(encoding="utf-8")
    notifications = Path("infrastructure/notifications.py").read_text(encoding="utf-8")
    main = Path("main.py").read_text(encoding="utf-8")

    assert "notif_extra_arena_modifiers" in source
    assert "Модификаторы ExtraArena" in source
    assert "canUseTelegramDelivery" in source
    assert "extraid_linked_telegram" in server
    assert "notif_extra_arena_modifiers BOOLEAN NOT NULL DEFAULT true" in database
    assert "\"extra_arena_modifier\": \"notif_extra_arena_modifiers\"" in notifications
    assert "extra_arena_modifier_changed" in notifications
    assert "mobile_only" in main


def test_android_extraid_management_and_connection_prompts_are_native():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "openNativeExtraIdManager" in source
    assert "public void openExtraIdManager" in native
    assert "showExtraIdManagerDialog" in native
    assert "showAddExtraIdAccountDialog" in native
    assert "Добавить аккаунт" in native
    assert "showBadConnectionSplash" in source
    assert "public void showBadConnectionSplash" in native
    assert "showNativeAfkDialog" in source
    assert "public void showNativeAfkDialog" in native


def test_android_push_registration_sends_device_timezone():
    registrar = DEVICE_REGISTRAR.read_text(encoding="utf-8")

    assert "TimeZone.getDefault()" in registrar
    assert "timezone" in registrar
    assert "utc_offset_minutes" in registrar
    assert 'body.put("auth"' not in registrar


def test_android_push_notification_intents_are_hardened():
    service = MESSAGING_SERVICE.read_text(encoding="utf-8")

    assert "sanitizeUpdateUrl" in service
    assert "BuildConfig.UPDATE_CHANNEL_URL.equals(url)" in service
    assert "BuildConfig.UPDATE_APK_URL.equals(url)" in service
    assert "intent.setPackage(getPackageName())" in service
    assert "PendingIntent.getActivity(\n                this,\n                notificationId" in service
    assert "Math.floorMod(seed.hashCode(), 100000)" in service


def test_android_native_email_validation_uses_android_patterns():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "android.util.Patterns" in native
    assert "Patterns.EMAIL_ADDRESS.matcher" in native


def test_android_configuration_rebuild_has_no_redundant_auth_branch():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "if (DeviceRegistrar.getAuthToken(this).isEmpty()) {\n            launchAfterUpdateGate(getIntent());\n        } else {\n            launchAfterUpdateGate(getIntent());\n        }" not in native


def test_android_shell_uses_native_share_for_result_and_invite_actions():
    arena = ARENA.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "window.shareExtraArena" in index
    assert "bridge.shareText" in index
    assert "function shareArenaResult" in arena
    assert "window.ExtraArenaApp.shareText" in arena
    assert "window.shareExtraArena = shareArenaResult" in arena
    assert "shareBtn.addEventListener('click'" in arena
    assert "window.shareExtraArena(shareText" in arena
    assert "tg.openTelegramLink(shareUrl)" in arena
    assert "shareSquad" in index
    assert "window.shareExtraArena(text" in index
    assert "Intent.ACTION_SEND" in native
    assert "Intent.createChooser" in native
    assert "public void shareText" in native


def test_android_app_declares_resizable_extra_arena_modifier_widget():
    manifest = ANDROID_MANIFEST.read_text(encoding="utf-8")
    build = APP_BUILD.read_text(encoding="utf-8")
    provider = WIDGET_PROVIDER.read_text(encoding="utf-8")
    updater = WIDGET_UPDATER.read_text(encoding="utf-8")
    layout = WIDGET_LAYOUT.read_text(encoding="utf-8")
    info = WIDGET_INFO.read_text(encoding="utf-8")
    server = WEB_SERVER.read_text(encoding="utf-8")

    assert "EXTRA_ARENA_WIDGET_PATH" in build
    assert "/api/mobile/extra-arena-widget" in server
    assert ".ExtraArenaModifierWidgetProvider" in manifest
    assert "android.appwidget.action.APPWIDGET_UPDATE" in manifest
    assert "ExtraArenaModifierWidgetUpdater.refresh" in provider
    assert "BuildConfig.EXTRA_ARENA_WIDGET_PATH" in updater
    assert "MODE_DESCRIPTIONS" in updater
    assert "AlarmManager" in updater
    assert "widget_description" in layout
    assert "android:resizeMode=\"horizontal|vertical\"" in info
    assert "android:minWidth=\"110dp\"" in info


def test_android_app_declares_personal_squad_widget_with_authenticated_mobile_api():
    manifest = ANDROID_MANIFEST.read_text(encoding="utf-8")
    build = APP_BUILD.read_text(encoding="utf-8")
    server = WEB_SERVER.read_text(encoding="utf-8")
    database = Path("infrastructure/database.py").read_text(encoding="utf-8")
    updater = SQUAD_WIDGET_UPDATER.read_text(encoding="utf-8")
    layout = SQUAD_WIDGET_LAYOUT.read_text(encoding="utf-8")

    assert "SQUAD_PERSONAL_WIDGET_PATH" in build
    assert "/api/mobile/squad/personal-cbrp-widget" in server
    assert "get_mobile_squad_personal_cbrp_widget" in database
    assert "DeviceRegistrar.getAuthToken" in updater
    assert "Authorization" in updater
    assert "request.headers.get(\"Authorization\"" in server
    assert "BuildConfig.SQUAD_PERSONAL_WIDGET_PATH" in updater
    assert ".SquadPersonalCbrpWidgetProvider" in manifest
    assert "SquadWidgetUpdater.refreshPersonalCbrp" in SQUAD_PERSONAL_PROVIDER.read_text(encoding="utf-8")
    assert ".SquadOwnerOverviewWidgetProvider" not in manifest
    assert ".SquadOwnerCbrpWidgetProvider" not in manifest
    assert "widget_squad_detail_1" in layout
    assert "android:resizeMode=\"horizontal|vertical\"" in SQUAD_PERSONAL_INFO.read_text(encoding="utf-8")


def test_android_app_declares_shop_particles_widget_and_russian_widget_names():
    manifest = ANDROID_MANIFEST.read_text(encoding="utf-8")
    build = APP_BUILD.read_text(encoding="utf-8")
    server = WEB_SERVER.read_text(encoding="utf-8")
    strings = STRINGS.read_text(encoding="utf-8")
    updater = SHOP_PARTICLES_UPDATER.read_text(encoding="utf-8")
    layout = SHOP_PARTICLES_LAYOUT.read_text(encoding="utf-8")
    info = SHOP_PARTICLES_INFO.read_text(encoding="utf-8")

    assert "SHOP_PARTICLES_WIDGET_PATH" in build
    assert "/api/mobile/shop-particles-widget" in server
    assert "mobile_shop_particles_widget_handler" in server
    assert ".ShopParticlesWidgetProvider" in manifest
    assert "ShopParticlesWidgetUpdater.refresh" in SHOP_PARTICLES_PROVIDER.read_text(encoding="utf-8")
    assert "BuildConfig.SHOP_PARTICLES_WIDGET_PATH" in updater
    assert "Authorization" in updater
    assert "next_rotation_ts" in updater
    assert "widget_particles_card_1" in layout
    assert "android:resizeMode=\"horizontal|vertical\"" in info
    assert "Модификатор ExtraArena" in strings
    assert "Твой вклад CBRP в скваде" in strings
    assert "Новые частицы карт в магазине" in strings


def test_android_app_packages_heavy_local_shell_assets():
    build = APP_BUILD.read_text(encoding="utf-8")

    assert "syncExtraArenaShellAssets" in build
    assert "generated/extraArenaShellAssets" in build
    assert "ea_webapp" in build
    assert "DesignAssets" in build
    assert "ea_vendor" in build
    assert "assets.srcDir(extraArenaShellAssetsDir)" in build
    assert "preBuild" in build


def test_android_webview_serves_shell_and_common_assets_from_apk():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "shouldInterceptRequest" in native
    assert "interceptExtraArenaRequest" in native
    assert "servePackagedWebappAsset" in native
    assert "servePackagedDesignAsset" in native
    assert "servePackagedCardImage" in native
    assert "servePackagedVendorAsset" in native
    assert "serveTelegramStub" in native
    assert "serveLocalFontsCss" in native
    assert "ea_webapp/index.html" in native
    assert "DesignAssets/Cards/" in native
    assert "ea_vendor/react.production.min.js" in native
    assert "api/cards/image" in native
    assert "return null;" in native


def test_android_webview_vendor_assets_include_cors_and_console_logging():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "crossorigin=\"anonymous\"" in INDEX.read_text(encoding="utf-8")
    assert "corsHeaders()" in native
    assert "\"Access-Control-Allow-Origin\", \"*\"" in native
    assert "servePackagedVendorAsset" in native
    assert "servePackagedAsset(assetPath, corsHeaders())" in native
    assert "onConsoleMessage" in native
    assert "EAWebView" in native
    assert "BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), \"DesignAssets/Font/FuturaPT-Medium.ttf\")" in native


def test_real_money_purchases_have_modular_success_modal_contract():
    source = INDEX.read_text(encoding="utf-8")

    assert "REAL_MONEY_SUCCESS_THEMES" in source
    assert "extrapassUltra" in source
    assert "linear-gradient(135deg,#ff5fa2,#2dd4bf)" in source
    assert "linear-gradient(135deg,#ff4d2e,#f5921e)" in source
    assert "function buildRealMoneyPurchaseSuccessPayload" in source
    assert "function showRealMoneyPurchaseSuccess" in source
    assert "window.showRealMoneyPurchaseSuccess" in source
    assert "data-provider" in source
    assert "markRealMoneySuccessModalShown" in source
    assert "triggerPaymentSuccessFromStatus" in source
    assert "triggerPaymentSuccessFromRecent" in source
    assert "value:'1 сезон'" in source


def test_android_webview_auth_prefers_native_session_after_apk_update():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "function getNativeAuthToken()" in source
    assert "getAndroidBridge()?.getAuthToken?.()" in source
    assert "getUrlAuthToken() || getNativeAuthToken() || localStorage.getItem('extra_id_token')" in source
    assert "add('auth', getNativeAuthToken(), 'native_extra_id')" in source
    assert "clearExtraToken({native: auth.source === 'native_extra_id'})" in source
    assert "const token = getNativeAuthToken() || localStorage.getItem('extra_id_token')" in source
    assert "public String getAuthToken()" in native
    assert "new URL(location.href).searchParams.get('_auth')" in native
    assert "localStorage.getItem('extra_id_token')||''" in native
    assert "eaJsonCache:" in native


def test_android_webview_recovers_viewport_after_keyboard_closes():
    safe_area = SAFE_AREA.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    manifest = ANDROID_MANIFEST.read_text(encoding="utf-8")

    assert "visualViewport" in safe_area
    assert "isKeyboardLikelyOpen" in safe_area
    assert "lastStableViewportHeight" in safe_area
    assert "scheduleViewportRecovery" in safe_area
    assert "root.style.setProperty('--ea-viewport-height'" in safe_area
    assert "if (!root.style.getPropertyValue('--ea-viewport-height'))" not in safe_area
    assert "document.addEventListener('focusout', scheduleViewportRecovery, true)" in safe_area

    assert 'android:windowSoftInputMode="adjustResize"' in manifest
    assert "hideSoftKeyboardAndClearAuthFocus" in native
    assert "InputMethodManager" in native
    assert "imm.hideSoftInputFromWindow" in native
    assert "focused.clearFocus()" in native


def test_battle_deck_picker_has_retry_and_no_infinite_spinner_on_auth_errors():
    source = INDEX.read_text(encoding="utf-8")

    assert "decksLoading" in source
    assert "deckLoadError" in source
    assert "loadBattleDeckPresets" in source
    assert source.count("loadBattleDeckPresets") >= 2
    assert "fetchWithTimeout(_buildAuthUrl('/api/deck/presets')" in source
    assert "window.eaInvalidateJson?.('/api/mobile/battle-bootstrap')" in source
    assert "Не удалось загрузить колоды" in source
    assert "Повторить" in source


def test_background_music_recovers_from_webview_audio_interruptions():
    source = INDEX.read_text(encoding="utf-8")
    arena = ARENA.read_text(encoding="utf-8")

    assert "ensureBgMusicWatchdog" in source
    assert "['ended', 'stalled', 'suspend', 'emptied']" in source
    assert "audio.addEventListener(eventName" in source
    assert "audio.paused || audio.ended" in source
    assert "_musicStarted && !audio.paused && !audio.ended" in source
    assert "ensureArenaMusicWatchdog" in arena
    assert "music.addEventListener(eventName" in arena
    assert "music.paused || music.ended" in arena
    assert "setTimeout(startMusic, 150)" in arena
    assert "document.addEventListener('pointerdown', startMusic" in arena
    assert "capture: true" in arena


def test_mobile_profile_mutations_invalidate_stale_cached_profile():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateProfileCaches()" in source
    assert "window.eaInvalidateJson?.('/api/profile')" in source
    assert "window.eaInvalidateJson?.('/api/mobile/bootstrap')" in source
    assert "window.reloadFreshProfile" in source
    assert "window.eaInvalidateJson?.('/api/mobile/shop-bootstrap')" in source
    assert "window.reloadFreshProfile().then(function(data)" in source
    assert "window.reloadFreshProfile().then(p=>{if(p)window.__updateProfile(p);});" in source
    assert "onDone={()=>{invalidateInventoryCaches();setShowCaseOpen(false);window.reloadFreshProfile()" in source


def test_mobile_extraid_account_manager_is_exposed_in_menu():
    source = INDEX.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    account_store = Path("android-app/app/src/main/java/ru/extraarena/app/ExtraIdAccountStore.java").read_text(encoding="utf-8")

    assert "getMobileExtraIdAccounts" in source
    assert "switchMobileExtraIdAccount" in source
    assert "forgetMobileExtraIdAccount" in source
    assert "resolveExtraIDRegistrationAuthData" in source
    assert "mobileAccounts" in source
    assert "select" in source
    assert "Сменить аккаунт" in source
    assert "client:'android_app'" in source
    assert "telegramAuth || resolveUserId()" not in source
    assert "Откройте игру через Telegram и попробуйте снова" not in source

    assert "public String getExtraIdAccounts()" in native
    assert "public boolean switchExtraIdAccount" in native
    assert "public boolean forgetExtraIdAccount" in native
    assert "public void saveExtraIdAccount" in native
    assert "public void reloadWithActiveAccount" in native
    assert "ExtraIdAccountStore.toJson" in native
    assert "ExtraIdAccountStore.activateAccount" in native
    assert "ExtraIdAccountStore.removeAccount" in native

    assert "static JSONArray toJson" in account_store
    assert "static boolean activateAccount" in account_store
    assert "static boolean removeAccount" in account_store
    assert "static void saveAccount(" in account_store


def test_mobile_web_client_uses_persistent_stale_while_revalidate_cache():
    source = INDEX.read_text(encoding="utf-8")

    assert "eaJsonCache:" in source
    assert "localStorage.getItem(storageKey(url))" in source
    assert "localStorage.setItem(storageKey(url)" in source
    assert "staleWhileRevalidate" in source
    assert "ea-json-cache-updated" in source
    assert "window.addEventListener('ea-json-cache-updated'" in source
    assert "window.eaWarmAndroidSections" in source
    assert "maxAgeMs: 120000" in source


def test_mobile_mutable_inventory_and_decks_are_loaded_network_first():
    source = INDEX.read_text(encoding="utf-8")

    assert "forceRefresh === true" in source
    assert "window.loadMobileCollectionBootstrap({forceFresh: true})" in source
    assert source.count("loadBattleDeckPresets(true)") >= 2
    assert "forceFresh ? true : false" in source
    assert "staleWhileRevalidate: forceFresh ? false : true" in source


def test_mobile_inventory_and_deck_mutations_invalidate_dependent_caches():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateInventoryCaches()" in source
    assert "function invalidateDeckCaches()" in source
    assert "window.eaInvalidateJson?.('/api/mobile/collection-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/mobile/battle-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/cards/collection')" in source
    assert "window.eaInvalidateJson?.('/api/deck/presets')" in source
    assert "invalidateInventoryCaches();\n        setResult(d);" in source
    assert "const handleCardUpgraded = React.useCallback(() => { invalidateInventoryCaches(); loadCards(); }" in source
    assert "invalidateDeckCaches();\n      await onReload();" in source
    assert "onDone={()=>{invalidateInventoryCaches();setShowCaseOpen(false);window.reloadFreshProfile()" in source


def test_mobile_shop_mutations_use_network_first_data_and_invalidate_caches():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateShopCaches()" in source
    assert "window.loadMobileShopBootstrap({forceFresh: true})" in source
    assert "window.eaInvalidateJson?.('/api/mobile/shop-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/shop/sets')" in source
    assert "window.eaInvalidateJson?.('/api/shop/particles/daily')" in source
    assert "invalidateShopCaches();\n            invalidateInventoryCaches();" in source
    assert "invalidateShopCaches();\n        invalidateInventoryCaches();" in source
    assert source.count("await loadData();") >= 2


def test_mobile_squad_mutations_use_network_first_data_and_invalidate_caches():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateSquadCaches()" in source
    assert "window.loadMobileSquadsBootstrap({forceFresh: true})" in source
    assert "window.eaInvalidateJson?.('/api/mobile/squads-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/squads/me')" in source
    assert "window.eaInvalidateJson?.('/api/squads/shop')" in source
    assert source.count("invalidateSquadCaches();\n      await loadMe();") >= 7
    assert source.count("invalidateSquadCaches();\n      await loadShop();") >= 2


def test_mobile_social_and_community_fetches_bypass_browser_cache_after_mutations():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateFriendsCaches()" in source
    assert "function invalidateCommunityCaches()" in source
    assert "fetch(_buildAuthUrl('/api/friends/list'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/recent-opponents'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/friends/requests'), {cache:'no-store'})" in source
    assert source.count("invalidateFriendsCaches();\n        loadAll();") >= 3
    assert "window.showToast('Заявка отозвана');\n      invalidateFriendsCaches();\n      loadAll();" in source
    assert "fetch(_buildAuthUrl('/api/community/news?limit=30'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/community/announcements?limit=20'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl(`/api/community/ideas?limit=30&sort=${sortBy}`), {cache:'no-store'})" in source
    assert source.count("invalidateCommunityCaches();") >= 3


def test_server_allows_static_assets_to_be_cached_by_mobile_shell():
    server = WEB_SERVER.read_text(encoding="utf-8")

    assert "STATIC_ASSET_CACHE_HEADERS" in server
    assert "\"Cache-Control\": \"public, max-age=31536000, immutable\"" in server
    assert "return web.FileResponse(file_path, headers=STATIC_ASSET_CACHE_HEADERS)" in server


def test_mobile_bootstrap_endpoints_aggregate_existing_game_handlers_without_forking_logic():
    server = WEB_SERVER.read_text(encoding="utf-8")
    source = INDEX.read_text(encoding="utf-8")

    assert "mobile_bootstrap_handler" in server
    assert "mobile_shop_bootstrap_handler" in server
    assert "mobile_collection_bootstrap_handler" in server
    assert "mobile_battle_bootstrap_handler" in server
    assert "mobile_squads_bootstrap_handler" in server
    assert "profile_handler, request" in server
    assert "runtime_status_handler, request" in server
    assert "shop_sets_public_handler, request" in server
    assert "particles_daily_handler, request" in server
    assert "collection_with_status_handler, request" in server
    assert "deck_presets_list_handler, request" in server
    assert "match_modes_handler, request" in server
    assert "squads_me_handler, request" in server
    assert "squads_shop_handler, request" in server
    assert "/api/mobile/bootstrap" in server
    assert "/api/mobile/shop-bootstrap" in server
    assert "/api/mobile/collection-bootstrap" in server
    assert "/api/mobile/battle-bootstrap" in server
    assert "/api/mobile/squads-bootstrap" in server

    assert "window.loadMobileBootstrap" in source
    assert "window.loadMobileShopBootstrap" in source
    assert "window.loadMobileCollectionBootstrap" in source
    assert "window.loadMobileBattleBootstrap" in source
    assert "window.loadMobileSquadsBootstrap" in source
    assert "/api/mobile/bootstrap" in source
    assert "/api/mobile/shop-bootstrap" in source
    assert "/api/mobile/collection-bootstrap" in source
    assert "/api/mobile/battle-bootstrap" in source
    assert "/api/mobile/squads-bootstrap" in source
    assert "loadDataFallback" in source
