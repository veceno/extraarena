import re
from pathlib import Path


INDEX = Path("webapp/index.html")
ARENA = Path("webapp/arena.js")
SAFE_AREA = Path("webapp/safe-area.js")
MAIN_ACTIVITY = Path("android-app/app/src/main/java/ru/extraarena/app/MainActivity.java")
AUTH_CLIENT = Path("android-app/app/src/main/java/ru/extraarena/app/AuthClient.java")
EXTRA_ID_ACCOUNT_STORE = Path("android-app/app/src/main/java/ru/extraarena/app/ExtraIdAccountStore.java")
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
CONNECTION_PROFILE_STORE = Path("android-app/app/src/main/java/ru/extraarena/app/ConnectionProfileStore.java")
BASE_URL_STORE = Path("android-app/app/src/main/java/ru/extraarena/app/BaseUrlStore.java")
REGION_DETECTOR = Path("android-app/app/src/main/java/ru/extraarena/app/RegionDetector.java")


def _battle_history_sheet_source():
    source = INDEX.read_text(encoding="utf-8")
    return source.split("const BattleHistorySheet", 1)[1].split(
        "// ── Analytics Controller ──",
        1,
    )[0]


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


def test_league_info_sheet_uses_handoff_redesign():
    source = INDEX.read_text(encoding="utf-8")
    sheet = source.split("const LeagueInfoSheet", 1)[1].split(
        "ReactDOM.createRoot",
        1,
    )[0]

    assert "league-redesign-screen" in source
    assert "ExtraArena Glory Path" in sheet
    assert "LEAGUE_ART_BY_ID[viewId]" in sheet
    assert "Награды Trophy Road" in sheet
    assert "onOpenGloryPath" in sheet


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
    assert "BuildConfig.LEGAL_OFFER_URL" in native
    assert "BuildConfig.LEGAL_PRIVACY_URL" in native
    assert "openExternal(legalUrlForTarget(target))" in native
    assert "authSecondaryAction.setVisibility(step == AuthStep.WELCOME ? View.GONE : View.VISIBLE)" in native


def test_webapp_info_sheet_links_existing_legal_docs_and_is_scrollable():
    source = INDEX.read_text(encoding="utf-8")
    sheet = source.split("const InfoSheet = ({onClose}) =>", 1)[1].split(
        "// ═══════════════════════════════════════════\n// SUPPORT SHEET",
        1,
    )[0]

    assert "Запроси документ через поддержку" not in sheet
    assert "/legal/offer" in sheet
    assert "/legal/privacy" in sheet
    assert "/legal/refund" in sheet
    assert "window.openExternalLink?.(href)" in sheet
    assert "new URL(url, location.origin).href" in sheet
    assert "openTelegramLink?.(url)" not in sheet
    assert "maxHeight:'calc(var(--ea-viewport-height, 100dvh) - var(--ea-safe-top))'" in sheet
    assert "overflowY:'auto'" in sheet
    assert "WebkitOverflowScrolling:'touch'" in sheet


def test_webapp_info_and_support_sheets_use_current_public_links():
    source = INDEX.read_text(encoding="utf-8")
    info_sheet = source.split("const InfoSheet = ({onClose}) =>", 1)[1].split(
        "// ═══════════════════════════════════════════\n// SUPPORT SHEET",
        1,
    )[0]
    support_sheet = source.split("const SupportSheet = ({onClose}) =>", 1)[1].split(
        "// ═══════════════════════════════════════════\n// ARENA SCREEN",
        1,
    )[0]

    assert "https://t.me/extraarena" in info_sheet
    assert "t.me/extraarena" in info_sheet
    assert "https://max.ru/se13279035_biz" in info_sheet
    assert "max.ru/se13279035_biz" in info_sheet
    assert "https://t.me/extraarena_supbot" in support_sheet
    assert "t.me/extraarena_supbot" in support_sheet
    assert "https://max.ru/se13279035_1_bot" in support_sheet
    assert "max.ru/se13279035_1_bot" in support_sheet
    assert "support.laveqox.ru" in support_sheet
    assert "скоро" in support_sheet.lower()
    assert "openLink('https://support.laveqox.ru')" not in support_sheet
    assert "openLink('http://support.laveqox.ru')" not in support_sheet
    assert "t.me/extracards" not in info_sheet
    assert "t.me/extracards_chat" not in info_sheet
    assert "t.me/lqsup" not in source


def test_android_legal_links_are_derived_from_current_base_url():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    build = APP_BUILD.read_text(encoding="utf-8")

    assert "clumsily-deft-guan.cloudpub.ru/legal" not in build
    assert 'propOrEnv("EXTRAARENA_OFFER_URL", "")' in build
    assert 'propOrEnv("EXTRAARENA_PRIVACY_URL", "")' in build
    assert 'propOrEnv("EXTRAARENA_REFUND_URL", "")' in build
    assert "configuredLegalUrlOrDefault(BuildConfig.LEGAL_OFFER_URL, \"/legal/offer\")" in native
    assert "configuredLegalUrlOrDefault(BuildConfig.LEGAL_PRIVACY_URL, \"/legal/privacy\")" in native
    assert "BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), path)" in native


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
    assert "notif_squad_weekly_tokens" in source
    assert "Недельные токены" in source
    assert "Модификаторы ExtraArena" in source
    assert "canUseTelegramDelivery" in source
    assert "extraid_linked_telegram" in server
    assert "notif_extra_arena_modifiers BOOLEAN NOT NULL DEFAULT true" in database
    assert "notif_squad_weekly_tokens BOOLEAN NOT NULL DEFAULT true" in database
    assert "\"extra_arena_modifier\": \"notif_extra_arena_modifiers\"" in notifications
    assert "\"squad_weekly_tokens\": \"notif_squad_weekly_tokens\"" in notifications
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


def test_legacy_collection_card_images_use_card_id_when_file_id_missing():
    source = Path("webapp/main.js").read_text(encoding="utf-8")

    assert "async function getCardImageUrlForCard(card, options = {})" in source
    assert 'const variant = options.variant || "preview"' in source
    assert "`/api/cards/image?card_id=${encodeURIComponent(card.id)}${variantSuffix}`" in source
    assert "getCardImageUrlForCard(card)" in source
    assert 'getCardImageUrlForCard(card, { variant: "full" })' in source


def test_collection_card_detail_uses_clean_open_sfx():
    index = INDEX.read_text(encoding="utf-8")
    main = Path("webapp/main.js").read_text(encoding="utf-8")

    assert 'id="collection-card-detail-sound"' in index
    assert "/DesignAssets/Sounds/collection/card_detail_open.wav" in index
    assert "playCollectionCardDetailSfx()" in index
    assert "window._playSfx?.('collection-card-detail-sound');" in index
    assert "window._playSfx?.('collection-card-detail-sound');" in main


def test_android_packaged_card_image_lookup_allows_jpg_assets():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    block = native.split("private WebResourceResponse servePackagedCardImage", 1)[1].split(
        "private WebResourceResponse servePackagedVendorAsset",
        1,
    )[0]

    assert '"DesignAssets/Cards/" + cardId + ".png"' in block
    assert '"DesignAssets/Cards/" + cardId + ".jpg"' in block
    assert '"DesignAssets/Cards/" + cardId + ".jpeg"' in block
    assert '"DesignAssets/Cards/" + cardId + ".webp"' in block


def test_android_friendly_invite_notification_forwards_accept_link_metadata():
    service = MESSAGING_SERVICE.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert 'data.get("invite_id")' in service
    assert 'intent.putExtra("invite_id", inviteId)' in service
    assert 'intent.putExtra("invite_action", inviteAction)' in service
    assert 'getStringExtra("invite_id")' in native
    assert 'getStringExtra("invite_action")' in native
    assert 'query.put("invite_id", inviteId)' in native
    assert 'query.put("invite_action", inviteAction)' in native


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


def test_profile_extra_pass_card_lifts_text_above_subscription_bar():
    source = INDEX.read_text(encoding="utf-8")
    profile_block = source.split("const ProfileScreen", 1)[1].split(
        "const ConnectionHealthOverlay",
        1,
    )[0]

    assert "const passCardLift = hasExtraPass || isUltra ? 18 : 0" in profile_block
    assert "bottom: passCardLift ? '28px' : '10px'" in profile_block
    assert "bottom: passCardLift ? '51px' : '33px'" in profile_block
    assert "bottom: passCardLift ? '34px' : '16px'" in profile_block


def test_battle_history_upsell_is_hidden_for_empty_history_and_mentions_hidden_modes():
    history_block = _battle_history_sheet_source()

    assert "isLimited && battles.length > 0" in history_block
    assert "включая скрытые тренировки и дружеские бои" in history_block
    assert "расширенную историю до 50 боёв" in history_block
    assert "полную историю (50 боёв)" not in history_block


def test_battle_history_load_error_has_dedicated_retry_state():
    history_block = _battle_history_sheet_source()
    compact = "".join(history_block.split())

    assert "loadError" in history_block
    assert "setLoadError" in history_block
    assert "catch(e)" in history_block
    assert "setLoadError(" in history_block
    assert "data-battle-history-load-error" in history_block
    assert "Повторить" in history_block
    assert "onClick={load}" in compact


def test_collection_potion_cards_use_normal_tile_height():
    source = INDEX.read_text(encoding="utf-8")
    card_tile = source.split("const CardTile = ({card", 1)[1].split(
        "const GeneratorScreen",
        1,
    )[0]

    assert "aspectRatio: isPotion ? '3/4'" not in card_tile
    assert "inset:'-7% -6% -9%'" not in card_tile
    assert "borderRadius: outerRadius" in card_tile
    assert "aspectRatio: '3/4'" in card_tile


def test_collection_uses_preview_images_but_detail_uses_full_images():
    source = INDEX.read_text(encoding="utf-8")
    card_tile = source.split("const CardTile = ({card", 1)[1].split("const GeneratorScreen", 1)[0]
    detail_block = source.split("const CardArtLightbox", 1)[1].split("const DeckPreviewSlot", 1)[0]
    deck_block = source.split("const DeckPreviewSlot", 1)[1].split("/* ──── Deck Metrics", 1)[0]

    assert "const cardPreviewUrl" in source
    assert "const cardFullUrl" in source
    assert "variant:'preview'" in source
    assert "cardPreviewUrl(card)" in card_tile
    assert "fallbackPreviewToFull" in card_tile
    assert "cardPreviewUrl(card)" in deck_block
    assert "cardFullUrl(cardId)" in detail_block
    assert "cardFullUrl(localCard)" in detail_block
    assert "cardPreviewUrl(localCard)" not in detail_block


def test_collection_deck_validity_blocks_draft_primary_and_battle_autoselect():
    source = INDEX.read_text(encoding="utf-8")

    assert "const getDeckPresetValidity" in source
    assert "canSetPrimary = deckValidity.isCompletePlayable" in source
    assert "ДРАФТ" in source
    assert "onNewbieTaskComplete?.('save_first_deck')" in source
    assert "if (draftValidity.isCompletePlayable)" in source
    assert "p.is_primary && getDeckPresetValidity(p).isCompletePlayable" in source


def test_safe_zone_covers_rating_generator_and_extrapass_controls():
    source = INDEX.read_text(encoding="utf-8")

    assert "height:'var(--ea-viewport-height, 100dvh)'" in source
    assert "padding:'0 12px calc(16px + var(--ea-safe-bottom-soft))'" in source
    assert "padding:'calc(24px + var(--ea-safe-top)) calc(24px + var(--ea-safe-right)) calc(24px + var(--ea-safe-bottom-soft)) calc(24px + var(--ea-safe-left))'" in source
    assert "height: var(--ea-viewport-height, 100dvh);" in source
    assert "padding:'calc(18px + var(--ea-safe-top)) calc(12px + var(--ea-safe-right)) calc(18px + var(--ea-safe-bottom-soft)) calc(12px + var(--ea-safe-left))'" in source


def test_safe_area_resists_transient_telegram_zero_insets():
    source = SAFE_AREA.read_text(encoding="utf-8")

    assert "const insetFallbacks" in source
    assert "hasInset(normalized) || !hasInset(state.values)" in source
    assert "tg.onEvent(eventName, scheduleResync)" in source
    assert "document.addEventListener('pointerup', scheduleResync" in source
    assert "window.addEventListener('pageshow', scheduleResync" in source
    assert "syncSoon: scheduleResync" in source


def test_generator_screen_uses_safe_json_and_http_status_for_api_mutations():
    source = INDEX.read_text(encoding="utf-8")
    generator_block = source.split("const GeneratorScreen", 1)[1].split(
        "const CardDetailScreen",
        1,
    )[0]
    fetch_status = generator_block.split("const fetchStatus", 1)[1].split(
        "React.useEffect",
        1,
    )[0]
    claim = generator_block.split("const doClaim", 1)[1].split(
        "const doUpgrade",
        1,
    )[0]
    upgrade = generator_block.split("const doUpgrade", 1)[1].split(
        "const fmtTimer",
        1,
    )[0]

    for request_block in (fetch_status, claim, upgrade):
        assert "const data = await res.json();" not in request_block
        assert "res.ok" in request_block
        assert (
            ".json().catch" in request_block
            or "readJsonOrError" in request_block
            or "readGeneratorJsonOrError" in request_block
        )

    assert "apiUrl('/api/generator/status')" in fetch_status
    assert "apiUrl('/api/generator/claim')" in claim
    assert "apiUrl('/api/generator/upgrade')" in upgrade


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
    assert "getNativeAuthToken() || getUrlAuthToken() || getStoredExtraToken()" in source
    assert "add('auth', getNativeAuthToken(), 'native_extra_id')" in source
    assert "clearExtraToken({native: auth.source === 'native_extra_id'})" in source
    assert "const token = getNativeAuthToken() || getStoredExtraToken()" in source
    assert "sessionStorage.getItem(EXTRA_ID_TOKEN_SESSION_KEY)" in source
    assert "public String getAuthToken()" in native
    assert 'webView.addJavascriptInterface(new AndroidBridge(), "ExtraArenaApp")' in native


def test_android_extraid_registration_reuses_active_identity_and_revokes_sessions():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    auth_client = AUTH_CLIENT.read_text(encoding="utf-8")
    account_store = EXTRA_ID_ACCOUNT_STORE.read_text(encoding="utf-8")
    launch_block = native.split("private String buildLaunchUrl(", 2)[2].split(
        "private void probeAndLoadArena",
        1,
    )[0]
    logout_block = native.split("private void logoutCurrentDevice()", 1)[1].split("\n    }\n}", 1)[0]
    remove_block = account_store.split("static boolean removeAccount(", 1)[1].split(
        "static void touchAccountByToken",
        1,
    )[0]

    assert 'appendQueryParameter("ea_platform", "android_app")' in launch_block
    assert 'appendQueryParameter("_auth"' not in launch_block
    assert 'registerBody.put("client", "android_app")' in auth_client
    assert '"/api/extraid/register"' in auth_client
    assert "activeAuthToken" in auth_client
    assert 'connection.setRequestProperty("Authorization", "Bearer " + cleanAuthToken)' in auth_client
    assert '"/api/auth/logout"' in auth_client
    assert "AuthClient.logoutBestEffort(context, account.token)" in remove_block
    assert "AuthClient.logoutBestEffort(this, activeToken)" in logout_block
    assert logout_block.index("AuthClient.logoutBestEffort(this, activeToken)") < logout_block.index(
        "DeviceRegistrar.clearAuthToken(this)"
    )


def test_telegram_collection_auth_drops_stale_url_session_token():
    source = INDEX.read_text(encoding="utf-8")
    candidates_block = source.split("function getUiAuthCandidates", 1)[1].split(
        "function resolveUserId",
        1,
    )[0]

    assert "TELEGRAM_AUTH_SESSION_MAX_AGE_SECONDS = 23 * 60 * 60" in source
    assert "function isStaleTelegramInitDataToken(token)" in source
    assert "sessionStorage.removeItem(EXTRA_URL_AUTH_SESSION_KEY)" in source
    assert "if (isStaleTelegramInitDataToken(token))" in source
    assert "if (sessionToken && isStaleTelegramInitDataToken(sessionToken))" in source
    assert "clean.searchParams.delete('_auth')" in source
    assert "if (isTelegramGameClient())" in candidates_block
    assert "return candidates;" in candidates_block
    assert candidates_block.index("add('auth', getNativeAuthToken(), 'native_extra_id')") < candidates_block.index(
        "add('auth', getUrlAuthToken(), 'url_auth')"
    )
    assert candidates_block.index("add('auth', tg?.initData, 'telegram')") < candidates_block.index(
        "add('auth', getUrlAuthToken(), 'url_auth')"
    )


def test_arena_prefers_current_telegram_identity_and_purges_cached_launch_auth():
    arena = ARENA.read_text(encoding="utf-8")
    init_block = arena.split("document.addEventListener('DOMContentLoaded'", 1)[1].split(
        "console.log('[ARENA] Match ID:'",
        1,
    )[0]

    assert "function isStaleArenaTelegramInitDataToken(token)" in arena
    assert "TELEGRAM_AUTH_SESSION_MAX_AGE_SECONDS = 23 * 60 * 60" in arena
    assert "if (hasTelegramInitData)" in init_block
    assert "authToken = telegramAuth" in init_block
    assert "sessionStorage.removeItem('arena_auth')" in init_block
    assert "clean.searchParams.delete('_auth')" in init_block
    assert "if (!authToken && !hasTelegramInitData)" in init_block
    assert init_block.index("authToken = telegramAuth") < init_block.index(
        "sessionStorage.getItem('arena_auth')"
    )


def test_versioned_third_party_scripts_use_sri_but_mutable_telegram_does_not():
    index = INDEX.read_text(encoding="utf-8")
    arena_html = Path("webapp/arena.html").read_text(encoding="utf-8")

    def script_tag(markup: str, src: str) -> str:
        match = re.search(
            rf'<script\b[^>]*\bsrc="{re.escape(src)}"[^>]*></script>',
            markup,
        )
        assert match is not None, f"missing third-party script: {src}"
        return match.group(0)

    # Telegram serves this unversioned URL as a mutable asset. Pinning its
    # current bytes would make every future upstream update fail closed in the
    # browser, so keep it covered by CSP but deliberately omit SRI.
    for markup in (index, arena_html):
        telegram = script_tag(
            markup,
            "https://telegram.org/js/telegram-web-app.js",
        )
        assert "integrity=" not in telegram

    for src in (
        "https://unpkg.com/react@18.3.1/umd/react.production.min.js",
        "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js",
        "https://unpkg.com/dompurify@3.2.6/dist/purify.min.js",
    ):
        versioned = script_tag(index, src)
        assert 'integrity="sha384-' in versioned
        assert 'crossorigin="anonymous"' in versioned

    socket_io = script_tag(
        arena_html,
        "https://cdn.socket.io/4.5.4/socket.io.min.js",
    )
    assert 'integrity="sha384-' in socket_io
    assert 'crossorigin="anonymous"' in socket_io


def test_webapp_moves_sensitive_auth_query_params_to_dedicated_headers():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    arena = ARENA.read_text(encoding="utf-8")
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")

    assert "function installJwtQueryAuthHeaderBridge()" in source
    assert "Authorization" in source
    assert "looksLikeJwtBearer" in source
    assert "url.origin !== window.location.origin" in source
    assert "headers.set('X-Telegram-Init-Data', telegramToken)" in source
    assert "Sensitive auth never stays in API URLs" in source
    build_url_block = source.split("function buildUiAuthUrl(path, explicitAuth)", 1)[1].split(
        "function looksLikeJwtBearer",
        1,
    )[0]
    assert "if (auth.type === 'auth')" in build_url_block
    assert "auth.type === 'auth' ? '_auth'" not in build_url_block
    assert "function installArenaJwtQueryAuthHeaderBridge()" in arena
    assert "looksLikeArenaJwtBearer" in arena
    assert "headers.set('X-Telegram-Init-Data', telegramToken)" in arena
    assert "Sensitive auth never stays in API URLs" in arena
    assert "localStorage.getItem('extra_id_token')||''" not in native
    assert "public String getAuthToken()" in native
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


def test_safe_area_ignores_telegram_viewport_while_keyboard_or_transition_is_unstable():
    safe_area = SAFE_AREA.read_text(encoding="utf-8")

    keyboard_guard = safe_area.index("if (isKeyboardLikelyOpen())")
    telegram_height = safe_area.index("const stableHeight = Number(tg.viewportStableHeight || tg.viewportHeight)")
    assert keyboard_guard < telegram_height
    assert "lastStableViewportHeight = Math.max(lastStableViewportHeight, stableHeight);" in safe_area


def test_profile_display_name_uses_telegram_fetched_name_before_generic_fallback():
    source = INDEX.read_text(encoding="utf-8")

    assert "profile?.custom_nickname || profile?.display_name || profile?.first_name || profile?.username || 'ExtraArena'" in source


def test_collection_upgrade_errors_are_user_facing_not_raw_codes():
    source = INDEX.read_text(encoding="utf-8")
    detail_block = source.split("const CardDetailScreen", 1)[1].split("const DeckPreviewSlot", 1)[0]

    assert "formatCardUpgradeError" in detail_block
    assert "data.message" in detail_block
    assert "insufficient_coins" in detail_block


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
    assert "window.reloadFreshProfile().then(d => { if (d && window.__updateProfile) window.__updateProfile(d); });" in source
    assert "onDone={()=>{invalidateInventoryCaches();setShowCaseOpen(false);window.reloadFreshProfile()" in source


def test_case_open_reveal_spotlights_new_cards_above_resource_rewards():
    source = INDEX.read_text(encoding="utf-8")
    case_block = source.split("const CaseOpenScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// BATTLE PICK SHEET",
        1,
    )[0]

    assert "const revealedCards = result?.rewards?.cards || [];" in case_block
    assert "const hasResourceRewards" in case_block
    assert "case-card-spotlight" in case_block
    assert "Новая карта в коллекции" in case_block
    assert "case-resource-rewards" in case_block
    assert case_block.index("case-card-spotlight") < case_block.index("case-resource-rewards")


def test_case_open_ultra_manual_reroll_uses_pending_claim_flow():
    source = INDEX.read_text(encoding="utf-8")
    case_block = source.split("const CaseOpenScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// BATTLE PICK SHEET",
        1,
    )[0]

    assert "const [openingToken, setOpeningToken]" in case_block
    assert "const [rerollToken, setRerollToken]" in case_block
    assert "const [canReroll, setCanReroll]" in case_block
    assert ("apiUrl('/api/cases/reroll-from-keys')" in case_block
            or "/api/cases/reroll-from-keys" in case_block)
    assert ("apiUrl('/api/cases/open-reroll-from-keys')" in case_block
            or "/api/cases/open-reroll-from-keys" in case_block)
    assert ("apiUrl('/api/cases/claim-from-keys')" in case_block
            or "/api/cases/claim-from-keys" in case_block)
    assert ("apiUrl('/api/cases/reroll')" in case_block
            or "/api/cases/reroll" in case_block)
    assert ("apiUrl('/api/cases/apply-reroll')" in case_block
            or "/api/cases/apply-reroll" in case_block)
    assert ("apiUrl('/api/cases/claim')" in case_block
            or "/api/cases/claim" in case_block)
    assert "Переоткрыть кейс" in case_block
    assert "claimCurrentOpening({closeAfter:true, notifyDone:true})" in case_block
    assert "Текущие награды сохранены" in case_block
    assert "<button onClick={closeCaseScreen} disabled={claiming}" in case_block
    assert case_block.count("setSkipping(false);") >= 4
    assert "openAnotherCase" in case_block


def test_case_open_uses_new_compressed_case_sfx_without_global_click_overlap():
    source = INDEX.read_text(encoding="utf-8")
    case_block = source.split("const CaseOpenScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// BATTLE PICK SHEET",
        1,
    )[0]
    sound_dir = Path("DesignAssets/Sounds/cases")
    expected = [
        "case-open-init",
        "case-tap",
        "case-reroll-ultra",
        "case-reward-resources",
        "case-reward-card-start",
        "case-reward-card-common",
        "case-reward-card-rare",
        "case-reward-card-superrare",
        "case-reward-card-epic",
        "case-reward-card-legendary",
        "case-reward-card-mythic",
        "case-reward-card-divine",
        "case-reward-card-limited",
    ]

    assert "window._playCaseSfx = function(id, options)" in source
    assert "if (!window._sfxEnabled) return;" in source
    assert "window._stopCaseSfx(id);" in source
    assert "data-no-global-click-sound" in case_block
    assert "caseRewardSfxId(d.rewards)" in case_block
    assert "CASE_REWARD_RARITY_WEIGHT" in source
    assert "case-open-sound" not in case_block
    assert "case-reward-sound" not in case_block

    for name in expected:
        asset = sound_dir / f"{name}.mp3"
        assert asset.exists(), f"missing case SFX asset: {asset}"
        assert 1_000 < asset.stat().st_size < 45_000
        assert f"/DesignAssets/Sounds/cases/{name}.mp3" in source


def test_mail_attachment_badges_accept_season_reset_aliases():
    source = INDEX.read_text(encoding="utf-8")
    mail_block = source.split("const AttachmentBadge", 1)[1].split(
        "const MailScreen",
        1,
    )[0]

    assert "const keys = Number(attachments.granted_keys ?? attachments.keys ?? 0);" in mail_block
    assert "const coins = Number(attachments.granted_coins ?? attachments.coins ?? 0);" in mail_block
    assert "if (keys)   items.push" in mail_block
    assert "if (coins)  items.push" in mail_block


def test_new_season_modal_uses_existing_mail_and_is_once_per_user_reset():
    source = INDEX.read_text(encoding="utf-8")
    modal_block = source.split("const NewSeasonModal", 1)[1].split(
        "// ═══════════════════════════════════════════\n// MAIL SCREEN",
        1,
    )[0]
    app_block = source.split("const App = () =>", 1)[1].split(
        "{/* League Up Modal */}",
        1,
    )[0]

    assert "function newSeasonSeenKey(userId, resetId)" in source
    assert "`ea:new-season-modal:v1:${userId}:${resetId}`" in source
    assert "fetch(_buildAuthUrl('/api/mail?category=system&limit=20'), {cache:'no-store'})" in app_block
    assert "const mail = (data.mail || []).find(item => {" in app_block
    assert "extractSeasonResetFromMail" in app_block
    assert "function newSeasonResetSeen(userId, resetId)" in source
    assert "function markSeasonResetSeen(reset)" in source
    assert "mailId: mail?.id || mail?.mail_id" in source
    assert "fetch(_buildAuthUrl('/api/mail/read')," in source
    assert "body: JSON.stringify({mail_id: reset.mailId})" in source
    assert "if (reset && !mail.is_read && !newSeasonResetSeen(userId, reset.resetId))" in app_block
    assert "setSeasonResetNotice(reset);" in app_block
    assert "markSeasonResetSeen(reset);" in app_block
    assert "<NewSeasonModal reset={seasonResetNotice}" in app_block
    assert "reset.grantedKeys" in modal_block
    assert "reset.grantedCoins" in modal_block


def test_extra_pass_case_reward_art_does_not_show_tier_as_quantity_badge():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]
    reward_art = sheet_block.split("const RewardArt = ({reward, locked, size = 42}) =>", 1)[1].split(
        "const renderLaneTile",
        1,
    )[0]

    assert "reward.type === 'case' || reward.type === 'keys'" not in reward_art
    assert "reward.type === 'keys' || reward.type === 'card' || reward.type === 'specific_card'" in reward_art
    assert "x{reward.amount}" in reward_art
    assert "{reward.tier}</span>" in reward_art


def test_extra_pass_claim_invalidates_inventory_caches_and_shows_granted_result():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]
    claim_block = sheet_block.split("const handleClaim = async (laneState) =>", 1)[1].split(
        "const laneStatus",
        1,
    )[0]

    assert "const describeGrantedRewards = (granted) =>" in sheet_block
    assert "const grantedSummary = describeGrantedRewards(d.granted);" in claim_block
    assert "invalidateInventoryCaches();" in claim_block
    assert "window.eaInvalidateJson?.('/api/rewards/extra-pass')" in claim_block
    assert "window.showToast?.(grantedSummary || 'Награда получена', 'success')" in claim_block
    assert claim_block.index("invalidateInventoryCaches();") < claim_block.index("await load();")


def test_real_money_payment_success_publishes_fresh_profile_to_react_state():
    source = INDEX.read_text(encoding="utf-8")
    status_block = source.split("async function triggerPaymentSuccessFromStatus", 1)[1].split(
        "async function triggerPaymentSuccessFromRecent",
        1,
    )[0]
    recent_block = source.split("async function triggerPaymentSuccessFromRecent", 1)[1].split(
        "function normalizeRuStorePaymentEvent",
        1,
    )[0]

    assert "async function refreshProfileStateAfterPaymentSuccess" in source
    assert "const profile = await window.reloadFreshProfile?.(authData)" in source
    assert "if (profile && window.__updateProfile) window.__updateProfile(profile);" in source
    assert "await refreshProfileStateAfterPaymentSuccess();" in status_block
    assert "await refreshProfileStateAfterPaymentSuccess(authData);" in recent_block


def test_legacy_main_js_treats_ultra_and_expiry_as_effective_extrapass():
    source = Path("webapp/main.js").read_text(encoding="utf-8")

    assert "function isExtraPassActive(profile)" in source
    assert 'mode !== "active" && mode !== "ultra"' in source
    assert "profile.extra_pass_expires_at" in source
    assert "const hasExtraPass = isExtraPassActive(data);" in source
    assert "const hasExtraPass = isExtraPassActive(currentProfile);" in source


def test_battle_pass_load_error_has_retry_state_and_disables_purchase_cta_without_payload():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]

    assert "const [battlePassLoadError, setBattlePassLoadError]" in sheet_block
    assert "setBattlePassLoadError(null);" in sheet_block
    assert "setBattlePassLoadError(e?.message || 'load_failed');" in sheet_block
    assert "const canUseBattlePassCta = !!payload && !battlePassLoadError;" in sheet_block
    assert "disabled={!canUseBattlePassCta}" in sheet_block
    assert "data-extra-pass-load-error" in sheet_block
    assert "onClick={load}" in sheet_block
    assert "Не удалось загрузить ExtraPass" in sheet_block


def test_battle_pass_upsell_is_hidden_for_active_pass_modes():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]
    topbar_block = sheet_block.split("<div style={s.topbar}>", 1)[1].split(
        "<div style={s.main}>",
        1,
    )[0]

    assert "{passMode === 'f2p' && (" in sheet_block
    assert "ExtraPass ждёт" in sheet_block
    assert "💎 {gems.toLocaleString()}" not in topbar_block
    assert "aria-hidden=\"true\" style={{width:38,height:38}}" in topbar_block


def test_battle_pass_reward_tiles_do_not_render_internal_status_chips():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]
    tile_block = sheet_block.split("const renderLaneTile = (lane) =>", 1)[1].split(
        "const ctaTitle",
        1,
    )[0]

    assert "{lane.label}" not in tile_block
    assert "{actionText}" not in tile_block
    assert "const helperText =" in tile_block
    assert "lane.access_locked ? (lane.access === 'ultra' ? 'откроется с Ultra' : 'откроется с ExtraPass')" in tile_block
    assert "RewardArt reward={rewards[0]} locked={locked} size={36}" in tile_block
    assert "helperText + ' · +'" in tile_block


def test_battle_pass_key_reward_art_inverts_dark_key_asset():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]
    reward_art_block = sheet_block.split("const RewardArt = ({reward, locked, size = 42}) =>", 1)[1].split(
        "const renderLaneTile = (lane) =>",
        1,
    )[0]

    assert "const keyRewardImageStyle =" in reward_art_block
    assert "brightness(0) invert(1)" in reward_art_block
    assert '<img src="/DesignAssets/MainMenu/Generator/Key.png" alt="" style={keyRewardImageStyle}/>' in reward_art_block


def test_battle_pass_sheet_distinguishes_specific_cards_and_highlights_ultra_zone():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]

    assert "reward?.reward_type === 'specific_card'" in sheet_block
    assert "Случайная карта" in sheet_block
    assert "Конкретная карта" in sheet_block
    assert "fetch(_buildAuthUrl('/api/rewards/extra-pass'), {cache:'no-store'})" in sheet_block
    assert "data-extra-pass-ultra-header" in sheet_block
    assert "data-extra-pass-ultra-divider" in sheet_block
    assert "Ultra-финал {ultraStart}-{ultraEnd}" in sheet_block
    assert "summary.claimable_with_ultra && !summary.claimable_with_extra_pass" in sheet_block
    assert ") : detail.access_locked ? (" in sheet_block


def test_new_season_modal_uses_mail_reset_notice_once_per_user_and_reset():
    source = INDEX.read_text(encoding="utf-8")
    app_block = source.split("const App = () => {", 1)[1].split(
        "// ═══════════════════════════════════════════\n// TROPHY ROAD SHEET",
        1,
    )[0]

    assert "const NewSeasonModal" in source
    assert "seasonResetNotice" in app_block
    assert "fetch(_buildAuthUrl('/api/mail?category=system&limit=20'), {cache:'no-store'})" in app_block
    assert "function newSeasonLegacySeenKey(userId, resetId)" in source
    assert "newSeasonResetSeen(userId, resetId)" in source
    assert "markSeasonResetSeen(reset)" in app_block
    assert "attachments.reset_id" in app_block
    assert "profile?.user_id" in app_block
    assert "setShowBattlePass(true)" in source


def test_mail_badges_accept_season_reset_granted_aliases():
    source = INDEX.read_text(encoding="utf-8")
    badge_block = source.split("const AttachmentBadge", 1)[1].split(
        "const MailScreen",
        1,
    )[0]

    assert "attachments.granted_coins" in badge_block
    assert "attachments.granted_keys" in badge_block


def test_battle_pass_claim_invalidates_inventory_and_displays_granted_payload_without_case_count_badge():
    source = INDEX.read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePassSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// LEAGUE INFO SHEET",
        1,
    )[0]

    assert "const describeGrantedRewards = (granted) =>" in sheet_block
    assert "describeGrantedRewards(d.granted)" in sheet_block
    assert "invalidateInventoryCaches();" in sheet_block
    assert "window.showToast?.(grantedSummary || 'Награда получена', 'success')" in sheet_block
    assert "reward.type === 'keys' || reward.type === 'card' || reward.type === 'specific_card'" in sheet_block


def test_shop_basic_pass_cta_blocks_ultra_downgrade():
    source = INDEX.read_text(encoding="utf-8")
    shop_block = source.split("const ShopScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// ARENA MAIN",
        1,
    )[0]

    assert "var isPassPurchaseDisabled = function(tab)" in shop_block
    assert "if (hasUltraPass && tab === 'basic') return true;" in shop_block
    assert "var passPurchaseLabel = function(tab)" in shop_block
    assert "if (hasUltraPass && tab === 'basic') return 'Ultra уже активен';" in shop_block
    assert "openPassPurchase('basic', true)" in shop_block
    assert "disabled={isPassPurchaseDisabled('basic')}" in shop_block
    assert "disabled={buying!=null || isPassPurchaseDisabled(passTab)}" in shop_block
    assert "window.__openExtraPassModal = function(tab, options)" in shop_block
    assert "setPendingPassPurchase(nextTab)" in shop_block
    assert "if (!pendingPassPurchase || !passProductFor(pendingPassPurchase)) return;" in shop_block
    assert "openPassPurchase(nextTab, true)" in shop_block
    assert "options && options.showDetails" in shop_block


def test_shop_orders_paid_packs_before_gifts():
    source = INDEX.read_text(encoding="utf-8")
    shop_block = source.split("const ShopScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// ARENA MAIN",
        1,
    )[0]

    assert "var paidPacksFirst = function(packs)" in shop_block
    assert "var giftA = a.pack && a.pack.isGift ? 1 : 0;" in shop_block
    assert "var visiblePacks = paidPacksFirst(rawVisiblePacks.map(resolvePackForOwnedCards).filter(Boolean));" in shop_block


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


def test_browser_user_id_auth_fallback_is_local_dev_only():
    source = INDEX.read_text(encoding="utf-8")
    legacy = Path("webapp/main.js").read_text(encoding="utf-8")
    server = WEB_SERVER.read_text(encoding="utf-8")
    start_battle_block = source.split("window.startBattle = async function", 1)[1].split(
        "// ── VS Bot Search",
        1,
    )[0]
    start_vs_bot_block = source.split("window.startVsBot = async function", 1)[1].split(
        "var overlayText =",
        1,
    )[0]
    prebattle_block = source.split("window._openPreBattle = function", 1)[1].split(
        "window.__analytics?.battlePlayed",
        1,
    )[0]
    friendly_battle_block = source.split("function openFriendlyInviteBattle", 1)[1].split(
        "const authData = resolveExtraIDAuthData()",
        1,
    )[0]
    send_friend_invite_block = source.split("const sendFriendInvite = async", 1)[1].split(
        "const selectedFriendlyDeckId",
        1,
    )[0]
    send_invite_block = source.split("const sendInvite = async", 1)[1].split(
        "setSending(true); setError(null);",
        1,
    )[0]
    incoming_invite_respond_block = source.split("var respond = async (action) =>", 1)[1].split(
        "setResponding(true);",
        1,
    )[0]
    onboarding_start_block = source.split("const startOnboardingBattle = React.useCallback", 1)[1].split(
        "const auth = resolveUiAuth();",
        1,
    )[0]

    assert "function allowLocalDevUserIdAuth()" in source
    assert "['localhost', '127.0.0.1', '::1'].includes(location.hostname)" in source
    assert "function canLaunchArenaBattleHere()" in source
    assert "return isAndroidAppShell() || !!getTelegramInitData() || allowLocalDevUserIdAuth();" in source
    assert "function showBrowserArenaUnavailable()" in source
    assert "Арена недоступна в обычном браузере" in source
    assert "const allowUserIdFallback = allowLocalDevUserIdAuth();" in source
    assert "add(allowUserIdFallback && urlId && explicitValue === urlId ? 'user_id' : 'auth'" in source
    assert "if (allowUserIdFallback) {" in source
    assert "add('user_id', urlId, 'url');" in source
    assert "add('user_id', window._getUserId?.() || tg?.initDataUnsafe?.user?.id, 'unsafe_user');" in source
    assert "allowLocalDevUserIdAuth() && urlId ? parseInt(urlId) : null" in source
    assert "if (!canLaunchArenaBattleHere())" in start_battle_block
    assert start_battle_block.index("if (!canLaunchArenaBattleHere())") < start_battle_block.index("var body =")
    assert "if (!canLaunchArenaBattleHere())" in start_vs_bot_block
    assert "if (!canLaunchArenaBattleHere())" in prebattle_block
    assert "if (!canLaunchArenaBattleHere())" in friendly_battle_block
    assert "if (!canLaunchArenaBattleHere())" in send_friend_invite_block
    assert "if (!canLaunchArenaBattleHere())" in send_invite_block
    assert "if (action === 'accept' && !canLaunchArenaBattleHere())" in incoming_invite_respond_block
    assert "if (!canLaunchArenaBattleHere())" in onboarding_start_block

    assert "function allowLocalDevUserIdAuth()" in legacy
    assert "['localhost', '127.0.0.1', '::1'].includes(window.location.hostname)" in legacy
    assert "if (urlId && allowLocalDevUserIdAuth())" in legacy

    assert "def _allow_dev_user_id_auth" in server
    assert "settings.environment == \"development\"" in server
    assert "request.remote in {\"127.0.0.1\", \"::1\", \"localhost\"}" in server


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
    claim_block = source.split("const claimCurrentOpening", 1)[1].split(
        "const openAnotherCase",
        1,
    )[0]

    assert "function invalidateInventoryCaches()" in source
    assert "function invalidateDeckCaches()" in source
    assert "window.eaInvalidateJson?.('/api/mobile/collection-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/mobile/battle-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/cards/collection')" in source
    assert "window.eaInvalidateJson?.('/api/deck/presets')" in source
    assert "invalidateInventoryCaches();" in claim_block
    assert claim_block.index("invalidateInventoryCaches();") < claim_block.index("setResult(d);", claim_block.index("invalidateInventoryCaches();"))
    assert "const handleCardUpgraded = React.useCallback(() => { invalidateInventoryCaches(); loadCards(); }" in source
    assert "invalidateDeckCaches();\n      await onReload();" in source
    assert "onDone={()=>{invalidateInventoryCaches();setShowCaseOpen(false);window.reloadFreshProfile()" in source


def test_mobile_shop_mutations_use_network_first_data_and_invalidate_caches():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateShopCaches()" in source
    assert "window.loadMobileShopBootstrap({forceFresh: true})" in source
    assert "data?.shop_sets && data?.particles_daily && data?.ruble_products" in source
    assert "window.eaInvalidateJson?.('/api/mobile/shop-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/shop/sets?surface=game')" in source
    assert "window.eaInvalidateJson?.('/api/shop/particles/daily')" in source
    assert len(re.findall(r"invalidateShopCaches\(\);\s*invalidateInventoryCaches\(\);", source)) >= 2
    assert source.count("await loadData();") >= 2


def test_profile_avatar_picker_normalizes_image_size_without_cropping():
    source = INDEX.read_text(encoding="utf-8")
    picker_block = source.split("const CosmeticPickerSheet", 1)[1].split(
        "type === 'profile_background'",
        1,
    )[0]

    assert "gridTemplateColumns:'repeat(3,minmax(0,1fr))'" in picker_block
    assert "boxSizing:'border-box'" in picker_block
    assert "width:'64px'" in picker_block
    assert "height:'64px'" in picker_block
    assert "width:'54px'" in picker_block
    assert "height:'54px'" in picker_block
    assert "objectFit:'contain'" in picker_block
    assert "margin:0" in picker_block
    assert "width:'76%'" not in picker_block
    assert "margin:'12%'" not in picker_block
    assert "objectFit:'cover'" not in picker_block


def test_mobile_squad_mutations_use_network_first_data_and_invalidate_caches():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateSquadCaches()" in source
    assert "const getSquadAvatarUrl = (s) => s?.avatar_url || s?.clan_avatar_url || null" in source
    assert "const getSquadBannerUrl = (s) => s?.banner_url || s?.clan_banner_url || null" in source
    assert "url={getSquadAvatarUrl(clan)}" in source
    assert "getSquadBannerUrl(clan)" in source
    assert "url={getSquadAvatarUrl(squadPreview.clan)}" in source
    assert "getSquadBannerUrl(squadPreview.clan)" in source
    assert "window.loadMobileSquadsBootstrap({forceFresh: true})" in source
    assert "window.eaInvalidateJson?.('/api/mobile/squads-bootstrap')" in source
    assert "window.eaInvalidateJson?.('/api/squads/me')" in source
    assert "window.eaInvalidateJson?.('/api/squads/shop')" in source
    assert source.count("invalidateSquadCaches();\n      await loadMe();") >= 7
    assert source.count("invalidateSquadCaches();\n      await loadShop();") >= 2


def test_squad_beta_polish_ui_copy_and_removed_shop_upgrades_are_present():
    source = INDEX.read_text(encoding="utf-8")
    legacy_main = Path("webapp/main.js").read_text(encoding="utf-8")

    assert "title: 'Глава'" in source
    assert "short: 'Владелец сквада'" in source
    assert ">Удерживай</HoldSquadButton>" in source
    assert "aria-hidden=\"true\" style={{position:'absolute',inset:0,background:`center/cover url(${m.profile_background_url})`}}" in source
    assert ".filter(([key]) => !['boost','customization'].includes(key))" in source
    assert "['boost','Boost'],['shop','Магазин']" in source
    assert "if(tab === 'shop') loadShop();" in source
    assert "Boost клана" in source
    assert "Фон · Boost" in source
    assert "linear-gradient(135deg,#14b8a6,#38bdf8)" in source
    assert "gridTemplateColumns:'repeat(2,minmax(0,1fr))'" in source
    assert "gridTemplateRows:'repeat(3,1fr)'" in source
    assert "aspectRatio:'1 / 1'" in source
    assert "Преимущества Boost" in source
    assert "6 бонусов" in source
    assert "title:'+5 мест'" in source
    assert "title:'×1.2 CBRP'" in source
    assert "caption:'множитель вклада'" in source
    assert "×1.2 CBRP за вклад" in source
    shop_block = source.split("{clan && tab === 'shop'", 1)[1].split("{clan && tab === 'boost'", 1)[0]
    boost_block = source.split("{clan && tab === 'boost'", 1)[1].split("{SQUAD_WARS_BETA_ENABLED", 1)[0]
    assert "Бонусы сквада" in shop_block
    assert "Бонусы сквада" not in boost_block
    assert "Общак" not in boost_block
    assert "Мои токены" not in boost_block
    assert "токен" not in boost_block.lower()
    assert "customizationUnlocked" not in source
    assert "Кастомизация сквада не открыта" not in source
    assert "Аватар закрыт" not in source
    assert "Фон закрыт" not in source
    assert "notif_squad_weekly_tokens: true" in legacy_main
    assert 'data-setting="notif_squad_weekly_tokens"' in legacy_main


def test_mobile_social_and_community_fetches_bypass_browser_cache_after_mutations():
    source = INDEX.read_text(encoding="utf-8")

    assert "function invalidateFriendsCaches()" in source
    assert "function invalidateCommunityCaches()" in source
    assert "const ANNOUNCE_PIN_BASE_COST = 1500" in source
    assert "Math.max(0, pinPrice - ANNOUNCE_PIN_BASE_COST)" in source
    assert "const sub = 500 + wExtra + iExtra + dExtra + pBase" in source
    assert "total:sub-disc+extraPin" in source
    assert "fetch(_buildAuthUrl('/api/friends/list'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/recent-opponents'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/friends/requests'), {cache:'no-store'})" in source
    assert source.count("invalidateFriendsCaches();\n        loadAll();") >= 3
    assert "window.showToast('Заявка отозвана');\n      invalidateFriendsCaches();\n      loadAll();" in source
    assert "fetch(_buildAuthUrl('/api/community/news?limit=30'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/community/announcements?limit=20'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl(`/api/community/ideas?limit=30&sort=${sortBy}`), {cache:'no-store'})" in source
    assert source.count("invalidateCommunityCaches();") >= 3


def test_webapp_ideas_section_has_read_more_and_admin_controls_and_bug_reports():
    source = INDEX.read_text(encoding="utf-8")

    assert "const IdeaCard = ({item, onVote, isAdmin, onStatusChange, onDelete}) => {" in source
    assert "const BugCard = ({item, onStatusChange, onDelete}) => {" in source
    assert "'Читать далее...'" in source
    assert "const IdeasSubScreen = ({isAdmin}) => {" in source
    assert "isAdmin={userRole.is_admin}" in source
    assert "fetch(_buildAuthUrl('/api/community/bugs?limit=50'), {cache:'no-store'})" in source
    assert "fetch(_buildAuthUrl('/api/community/ideas/admin/delete'),{method:'POST'" in source
    assert "fetch(_buildAuthUrl('/api/community/ideas/admin/status'),{method:'POST'" in source
    assert "<option value=\"reviewing\">На рассмотрении</option>" in source
    assert "<option value=\"in_progress\">В работе</option>" in source
    assert "<option value=\"rejected\">Отвергнута</option>" in source
    assert "setView('bugs')" in source
    assert "setView('ideas')" in source
    assert "loadBugs" in source
    assert "handleDelete" in source
    assert "handleStatusChange" in source


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


def test_android_app_has_two_built_in_connection_profiles_for_ru_and_worldwide():
    build = APP_BUILD.read_text(encoding="utf-8")
    store = CONNECTION_PROFILE_STORE.read_text(encoding="utf-8")

    # Two production hosts compiled in: worldwide (Cloudflare tunnel) + RU-direct.
    assert "https://app.extraarena.space/" in build
    assert "https://app.laveqox.ru/" in build
    assert 'RU_BASE_URL' in build
    # Staging host is no longer the default.
    assert "clumsily-deft-guan.cloudpub.ru" not in build

    # Both profiles are seeded as built-ins.
    assert 'DEFAULT_PROFILE_ID = "extraarena_worldwide"' in store
    assert 'RU_PROFILE_ID = "extraarena_ru"' in store
    assert "BUILT_IN_IDS" in store
    assert "ensureBuiltInProfiles" in store
    assert "ruProfile()" in store
    assert "autoSelectIfNeeded" in store
    assert "otherBuiltIn" in store
    assert "isBuiltIn" in store


def test_android_main_activity_auto_selects_profile_by_region_with_health_fallback():
    native = MAIN_ACTIVITY.read_text(encoding="utf-8")
    region = REGION_DETECTOR.read_text(encoding="utf-8")
    store = CONNECTION_PROFILE_STORE.read_text(encoding="utf-8")

    assert "RegionDetector" in native
    assert "selectInitialProfileIfNeeded" in native
    assert "autoSelectIfNeeded" in native
    # Health-probe fallback to the other built-in host when the selected one is unreachable.
    assert "otherBuiltIn" in native
    assert "isBuiltIn" in native
    assert "isServerAvailable" in native

    assert "isLikelyRu" in region
    assert "getSimCountryIso" in region
    assert "getNetworkCountryIso" in region
    assert "RU_TIMEZONES" in region

    # autoSelect runs at most once and respects a pre-existing manual selection.
    assert "pref_auto_selected_profile" in store
    assert 'KEY_AUTO_SELECTED' in store


def test_android_base_url_store_treats_both_production_hosts_as_non_test():
    base = BASE_URL_STORE.read_text(encoding="utf-8")
    assert "isBuiltIn" in base
    assert "RU_BASE_URL" in base
    assert "DEFAULT_BASE_URL" in base


def test_webapp_cosmetics_and_squad_images_fall_back_to_bundled_defaults_on_error():
    source = INDEX.read_text(encoding="utf-8")

    # Bundled default cosmetics used as the standard fallback (tier 3).
    assert "ratingDefaultAvatar" in source
    assert "ratingDefaultBackground" in source
    assert "/DesignAssets/PlayerCosmetics/Avatars/1.png" in source
    assert "/DesignAssets/PlayerCosmetics/Background/7.png" in source
    # Cosmetic <img> renders swap to the bundled default on error instead of showing a broken image.
    assert source.count("e.currentTarget.src=ratingDefaultAvatar") >= 1
    assert source.count("e.currentTarget.src=ratingDefaultBackground") >= 2
    # Squad avatar degrades to the letter-avatar on fetch failure (no empty circle).
    assert "const [imgFailed, setImgFailed]" in source
    assert "onError={()=>setImgFailed(true)}" in source
    # Squad banner layers a solid gradient under the image so failure degrades to the gradient.
    assert source.count(", linear-gradient(145deg,${T.purple3},${T.bgDeep})") >= 3


def test_arena_prebattle_background_falls_back_to_default_background_on_error():
    arena = ARENA.read_text(encoding="utf-8")

    assert "fallbackUrl" in arena
    assert "img.onerror" in arena
    assert "/DesignAssets/PlayerCosmetics/Background/7.png" in arena
    # The prebattle background render passes the default as the fallback URL.
    assert "profile?.background_url, '/DesignAssets/PlayerCosmetics/Background/7.png'" in arena


def test_server_serves_cosmetic_detail_by_id_or_slug_for_mobile_fallback_chain():
    server = WEB_SERVER.read_text(encoding="utf-8")

    assert "cosmetic_detail_handler" in server
    assert '"/api/cosmetics/{identity}"' in server or '"/api/cosmetics/{identity}"' in server
    assert "get_cosmetic_item" in server
    assert "is_active" in server
    # Squad images cached long enough to stay "saved locally" on mobile between sessions.
    assert "max-age=2592000" in server
