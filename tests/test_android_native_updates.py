from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "android-app/app/src/main/java/ru/extraarena/app/MainActivity.java"
DIRECT_INSTALLER = (
    ROOT
    / "android-app/app/src/direct/java/ru/extraarena/app/DirectApkUpdateInstaller.java"
)
DIRECT_MANIFEST = ROOT / "android-app/app/src/direct/AndroidManifest.xml"
RUSTORE_MANIFEST = ROOT / "android-app/app/src/rustore/AndroidManifest.xml"
RUSTORE_INTEGRATION = (
    ROOT
    / "android-app/app/src/rustore/java/ru/extraarena/app/RuStoreIntegrationImpl.java"
)
MUSIC = ROOT / "android-app/app/src/main/java/ru/extraarena/app/NativeMenuMusic.java"
WEBAPP = ROOT / "webapp/index.html"


def test_update_policy_keeps_latest_and_required_floor_independent():
    source = MAIN.read_text(encoding="utf-8")

    assert "BuildConfig.VERSION_CODE < minCode" in source
    assert "BuildConfig.VERSION_CODE < latestCode" in source
    assert "Math.max(latestCode, minCode)" not in source
    assert "showOptionalUpdateDialog" in source
    assert "minSupportedVersionCode" in source
    assert "fetchMobileUpdateInfoFrom(\n                BuildConfig.DEFAULT_BASE_URL" in source
    assert "fetchMobileUpdateInfoFrom(BuildConfig.RU_BASE_URL)" in source
    assert "MobileUpdateInfo.merge(worldwide, ru)" in source
    assert "MobileUpdateInfo cached = readCachedMobileUpdateInfo()" in source
    assert "effective.toCacheJson()" in source
    assert "fetch order can replace a newer regional APK" in source
    assert "cachedCandidate.latestVersionCode >= effectiveFloor" in source
    assert "if (effective.hasUsableTarget())" in source
    assert "SecurePrefs.putString(this, KEY_UPDATE_MANIFEST_CACHE, raw)" not in source
    assert source.count("cancelOptionalUpdateWork();") >= 2
    assert "no download or installer may surface after game entry" in source
    assert "!updateDialogRequired" in source
    assert "optional prompt is open" in source
    continue_gate = source.split("private void continueAfterUpdateGate(Intent intent)", 1)[1].split(
        "private void showUpdateGateLoading()", 1
    )[0]
    assert "if (updateBlocked)" in continue_gate
    assert "showRequiredUpdateDialog(blockedUpdateInfo)" in continue_gate
    assert continue_gate.index("if (updateBlocked)") < continue_gate.index("loadArena(intent)")


def test_update_policy_fails_closed_without_a_live_official_authority():
    source = MAIN.read_text(encoding="utf-8")
    fetch = source.split("private MobileUpdateInfo fetchMobileUpdateInfo()", 1)[1].split(
        "private MobileUpdateInfo readCachedMobileUpdateInfo()", 1
    )[0]
    launch = source.split("private void launchAfterUpdateGate(Intent intent)", 1)[1].split(
        "private void continueAfterUpdateGate(Intent intent)", 1
    )[0]

    assert "if (live != null)" in fetch
    assert "cachedEffective != null && cachedEffective.required" in fetch
    assert "isDebugLoopbackProfile(selectedProfile)" in fetch
    assert "BuildConfig.DEBUG" in fetch
    assert '"10.0.2.2".equals(host)' in fetch
    assert '"http".equalsIgnoreCase(uri.getScheme())' in fetch
    assert "return MobileUpdateInfo.policyUnavailable(cachedEffective);" in fetch
    assert 'data.has("latest_version_code")' in source
    assert 'data.has("min_supported_version_code")' in source
    assert "if (info.policyUnavailable)" in launch
    assert launch.index("if (info.policyUnavailable)") < launch.index("if (info.required)")
    assert "showRequiredUpdateDialog(info)" in launch


def test_direct_update_is_downloaded_and_verified_before_system_installer():
    source = DIRECT_INSTALLER.read_text(encoding="utf-8")

    for contract in (
        "DownloadManager",
        "sha256(apk)",
        "getPackageArchiveInfo",
        "archiveVersion != expected.versionCode",
        "signingDigests",
        "canRequestPackageInstalls",
        "ACTION_MANAGE_UNKNOWN_APP_SOURCES",
        "FileProvider.getUriForFile",
        "Intent.ACTION_INSTALL_PACKAGE",
    ):
        assert contract in source
    assert "ACTION_DELETE" not in source
    assert "uninstall" not in source.lower()
    assert "cleanupInstalledUpdateArtifacts();" in source
    assert "savedVersion > BuildConfig.VERSION_CODE" in source
    assert "discardSavedDownload(updateDir, savedId, savedPath)" in source
    assert "!candidate.equals(preservedPath)" in source


def test_unknown_sources_permission_and_file_provider_are_direct_only():
    direct = DIRECT_MANIFEST.read_text(encoding="utf-8")
    rustore = RUSTORE_MANIFEST.read_text(encoding="utf-8")

    assert "android.permission.REQUEST_INSTALL_PACKAGES" in direct
    assert "androidx.core.content.FileProvider" in direct
    assert '${applicationId}.updates' in direct
    assert "android.permission.REQUEST_INSTALL_PACKAGES" not in rustore
    assert "FileProvider" not in rustore


def test_rustore_flexible_update_completes_and_recovers_downloaded_install():
    source = RUSTORE_INTEGRATION.read_text(encoding="utf-8")

    assert "manager.registerListener(installStateUpdateListener)" in source
    assert "manager.unregisterListener(installStateUpdateListener)" in source
    assert "state.getInstallStatus() == InstallStatus.DOWNLOADED" in source
    assert "info.getInstallStatus() == InstallStatus.DOWNLOADED" in source
    assert "recoverPendingUpdate(manager)" in source
    assert "manager.completeUpdate(updateOptions(AppUpdateType.FLEXIBLE))" in source
    assert "a cancelled installer must not reopen in a loop" in source


def test_rustore_immediate_update_resumes_and_surfaces_terminal_failures():
    source = RUSTORE_INTEGRATION.read_text(encoding="utf-8")

    assert "UpdateAvailability.DEVELOPER_TRIGGERED_UPDATE_IN_PROGRESS" in source
    assert "startImmediateFlow(manager, info)" in source
    assert "private Runnable requiredUpdateFallback" in source
    assert "immediateFlowAccepted = true" in source
    assert "immediateFlowActive = false" in source
    assert "Activity.RESULT_CANCELED" in source
    assert "InstallStatus.DOWNLOAD_INTERRUPTED" in source
    assert "failRequiredUpdateFlow()" in source


def test_native_menu_music_owns_loading_auth_and_android_web_menu_audio():
    native = MUSIC.read_text(encoding="utf-8")
    activity = MAIN.read_text(encoding="utf-8")
    web = WEBAPP.read_text(encoding="utf-8")

    assert 'openFd("DesignAssets/Sounds/main_theme.mp3")' in native
    assert "AudioFocusRequest" in native
    assert 'nativeMenuMusic.setScene("menu")' in activity
    assert 'nativeMenuMusic.setScene(isArenaUrl(url) ? "arena" : "menu")' in activity
    assert "setMenuMusicEnabled" in activity
    assert "hasNativeMenuMusic" in web
    assert "window.ExtraArenaApp.setAudioScene('menu')" in web
    assert "window.toggleMusic(musicOn)" in web
    assert 'right.setText("72%")' not in activity
    assert 'setNativeLoadingStatus("Проверяем версию")' in activity


def test_installer_requires_same_package_and_strict_version_upgrade():
    source = DIRECT_INSTALLER.read_text(encoding="utf-8")

    assert "activity.getPackageName().equals(candidate.packageName)" in source
    assert "candidate.versionCode <= BuildConfig.VERSION_CODE" in source
    assert 'normalizeDigest(candidate.signingCertSha256).matches("[0-9a-f]{64}")' in source
    assert "archiveVersion <= BuildConfig.VERSION_CODE" in source
    assert "compatibleSigner" in source
    assert "verificationInProgress" in source
    assert "token != verificationToken" in source
    assert "if (verifiedFile != null)" in source
    assert "Returning from the system installer must not reopen it automatically" in source
