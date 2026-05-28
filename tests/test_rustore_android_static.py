from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANDROID_APP = ROOT / "android-app" / "app"


def test_android_gradle_declares_direct_and_rustore_flavors():
    gradle = (ANDROID_APP / "build.gradle.kts").read_text(encoding="utf-8")

    assert 'create("direct")' in gradle
    assert 'create("rustore")' in gradle
    assert 'buildConfigField("String", "DISTRIBUTION_CHANNEL"' in gradle
    assert 'add("rustoreImplementation", platform("ru.rustore.sdk:bom:2026.04.02"))' in gradle
    assert 'add("rustoreImplementation", "ru.rustore.sdk:pay")' in gradle
    assert 'add("rustoreImplementation", "ru.rustore.sdk:appupdate")' in gradle
    assert 'propOrEnv("RUSTORE_CONSOLE_APP_ID", "2063712624")' in gradle
    assert 'ANDROID_RELEASE_STORE_FILE' in gradle
    assert 'signingConfigs' in gradle


def test_android_rustore_manifest_declares_sdk_metadata_and_permission_removals():
    manifest = (ANDROID_APP / "src" / "rustore" / "AndroidManifest.xml").read_text(encoding="utf-8")

    assert "sdk_pay_scheme_value" in manifest
    assert "console_app_id_value" in manifest
    assert 'android:name="android.permission.BLUETOOTH"' in manifest
    assert 'tools:node="remove"' in manifest


def test_android_bridge_exposes_rustore_payment_methods():
    main_activity = (ANDROID_APP / "src" / "main" / "java" / "ru" / "extraarena" / "app" / "MainActivity.java").read_text(encoding="utf-8")

    assert "isRuStorePayAvailable" in main_activity
    assert "startRuStorePayment" in main_activity
    assert "ExtraArenaRuStorePayment" in main_activity
    assert "RUSTORE_APP_URL" in main_activity
