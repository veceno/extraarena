import org.gradle.api.tasks.Sync

plugins {
    id("com.android.application")
}

if (file("google-services.json").exists()) {
    apply(plugin = "com.google.gms.google-services")
}

val repoRoot = rootDir.parentFile
val extraArenaShellAssetsDir = layout.buildDirectory.dir("generated/extraArenaShellAssets")
val optimizedExtraArenaAssetsDir = repoRoot.resolve("android-app/optimized-assets")
fun propOrEnv(name: String, fallback: String): String =
    (findProperty(name) as String?)?.takeIf { it.isNotBlank() }
        ?: System.getenv(name)?.takeIf { it.isNotBlank() }
        ?: fallback

fun quoted(value: String): String = "\"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\""

val rustoreConsoleAppId = propOrEnv("RUSTORE_CONSOLE_APP_ID", "2063712624")
val rustorePayScheme = propOrEnv("RUSTORE_PAY_SCHEME", "ru.extraarena.app.rustore.pay")
val rustoreAppUrl = propOrEnv("RUSTORE_APP_URL", "https://www.rustore.ru/catalog/app/ru.extraarena.app")
val legalOfferUrl = propOrEnv("EXTRAARENA_OFFER_URL", "")
val legalPrivacyUrl = propOrEnv("EXTRAARENA_PRIVACY_URL", "")
val legalRefundUrl = propOrEnv("EXTRAARENA_REFUND_URL", "")
val releaseStoreFile = propOrEnv("ANDROID_RELEASE_STORE_FILE", "")
val releaseStorePassword = propOrEnv("ANDROID_RELEASE_STORE_PASSWORD", "")
val releaseKeyAlias = propOrEnv("ANDROID_RELEASE_KEY_ALIAS", "")
val releaseKeyPassword = propOrEnv("ANDROID_RELEASE_KEY_PASSWORD", "")
val directReleaseStoreFile = propOrEnv("ANDROID_DIRECT_RELEASE_STORE_FILE", releaseStoreFile)
val directReleaseStorePassword = propOrEnv("ANDROID_DIRECT_RELEASE_STORE_PASSWORD", releaseStorePassword)
val directReleaseKeyAlias = propOrEnv("ANDROID_DIRECT_RELEASE_KEY_ALIAS", releaseKeyAlias)
val directReleaseKeyPassword = propOrEnv("ANDROID_DIRECT_RELEASE_KEY_PASSWORD", releaseKeyPassword)
val rustoreReleaseStoreFile = propOrEnv("ANDROID_RUSTORE_RELEASE_STORE_FILE", releaseStoreFile)
val rustoreReleaseStorePassword = propOrEnv("ANDROID_RUSTORE_RELEASE_STORE_PASSWORD", releaseStorePassword)
val rustoreReleaseKeyAlias = propOrEnv("ANDROID_RUSTORE_RELEASE_KEY_ALIAS", releaseKeyAlias)
val rustoreReleaseKeyPassword = propOrEnv("ANDROID_RUSTORE_RELEASE_KEY_PASSWORD", releaseKeyPassword)
fun hasSigning(storeFile: String, storePassword: String, keyAlias: String, keyPassword: String): Boolean =
    storeFile.isNotBlank() && storePassword.isNotBlank() && keyAlias.isNotBlank() && keyPassword.isNotBlank()
val hasDirectReleaseSigning = hasSigning(
    directReleaseStoreFile,
    directReleaseStorePassword,
    directReleaseKeyAlias,
    directReleaseKeyPassword
)
val hasRustoreReleaseSigning = hasSigning(
    rustoreReleaseStoreFile,
    rustoreReleaseStorePassword,
    rustoreReleaseKeyAlias,
    rustoreReleaseKeyPassword
)
val verifyOptimizedExtraArenaAssets by tasks.registering(Exec::class) {
    group = "verification"
    description = "Fails the Android build when optimized card assets are stale or below quality budgets"
    commandLine(
        propOrEnv("EXTRAARENA_PYTHON", "python3"),
        repoRoot.resolve("android-app/scripts/optimize_assets.py").absolutePath,
        "--source",
        repoRoot.resolve("DesignAssets").absolutePath,
        "--output",
        optimizedExtraArenaAssetsDir.absolutePath,
        "--check"
    )
}
val verifyCompiledWebapp by tasks.registering(Exec::class) {
    group = "verification"
    description = "Fails when the bundled web game is not compiled from the current index.html"
    commandLine(
        propOrEnv("EXTRAARENA_PYTHON", "python3"),
        repoRoot.resolve("scripts/precompile_webapp_index.py").absolutePath,
        "--check"
    )
}
val syncExtraArenaShellAssets by tasks.registering(Sync::class) {
    dependsOn(verifyOptimizedExtraArenaAssets)
    dependsOn(verifyCompiledWebapp)
    into(extraArenaShellAssetsDir)
    from(repoRoot.resolve("webapp")) {
        into("ea_webapp")
        include("*.html")
        include("*.js")
        // Keep the arena telemetry module explicit: it is loaded before arena.js.
        include("analytics-v2.js")
        include("*.css")
        exclude("extraid-mockup.html")
        exclude(".DS_Store")
    }
    from(repoRoot.resolve("DesignAssets")) {
        into("DesignAssets")
        exclude("Cards/**")
        exclude("Cards copy/**")
        exclude("Cards.zip")
        exclude("Arena/Sounds/arena_theme_legacy_20260616.wav")
        exclude("Arena/Sounds/arena_theme_v2_loop.wav")
        exclude("**/.DS_Store")
    }
    from(optimizedExtraArenaAssetsDir.resolve("DesignAssets/Cards")) {
        into("DesignAssets/Cards")
    }
    from(repoRoot.resolve("assets/audio")) {
        into("assets/audio")
        exclude("**/.DS_Store")
    }
    from(file("src/main/assets")) {
        // The native welcome carousel now uses the already-packaged w384 previews.
        // The current shell is precompiled, so the 3 MB runtime compiler is dead weight.
        exclude("ea_vendor/babel.min.js")
        exclude("extra_mobile/1.png")
        exclude("extra_mobile/3.png")
        exclude("extra_mobile/10.png")
        exclude("extra_mobile/14.png")
        exclude("extra_mobile/17.png")
        exclude("extra_mobile/21.png")
        exclude("extra_mobile/31.png")
        exclude("extra_mobile/40.png")
        exclude("extra_mobile/card-1.png")
        exclude("extra_mobile/card-40.png")
        exclude("**/.DS_Store")
    }
}

android {
    namespace = "ru.extraarena.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "ru.extraarena.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 50
        versionName = "0.6.1"

        buildConfigField("String", "DEFAULT_BASE_URL", "\"https://app.extraarena.space/\"")
        buildConfigField("String", "RU_BASE_URL", "\"https://app.laveqox.ru/\"")
        buildConfigField("String", "TEST_BASE_URL", "\"http://10.0.2.2:8081/\"")
        buildConfigField("String", "UPDATE_CHANNEL_URL", "\"https://t.me/extraarenamobile\"")
        buildConfigField("String", "UPDATE_APK_URL", "\"https://apk.laveqox.ru\"")
        buildConfigField("String", "APP_VERSION_PATH", "\"/api/mobile/client-version\"")
        buildConfigField("String", "EXTRA_ARENA_WIDGET_PATH", "\"/api/mobile/extra-arena-widget\"")
        buildConfigField("String", "SQUAD_PERSONAL_WIDGET_PATH", "\"/api/mobile/squad/personal-cbrp-widget\"")
        buildConfigField("String", "SHOP_PARTICLES_WIDGET_PATH", "\"/api/mobile/shop-particles-widget\"")
        buildConfigField("String", "PUSH_REGISTER_PATH", "\"/api/push/register\"")
        buildConfigField("String", "PUSH_UNREGISTER_PATH", "\"/api/push/unregister\"")
        buildConfigField("String", "DISTRIBUTION_CHANNEL", quoted("direct"))
        buildConfigField("String", "RUSTORE_CONSOLE_APP_ID", quoted(rustoreConsoleAppId))
        buildConfigField("String", "RUSTORE_PAY_SCHEME", quoted(rustorePayScheme))
        buildConfigField("String", "RUSTORE_APP_URL", quoted(rustoreAppUrl))
        buildConfigField("String", "PAYMENT_PROVIDER_ORDER", quoted("yookassa,stars"))
        buildConfigField("String", "LEGAL_OFFER_URL", quoted(legalOfferUrl))
        buildConfigField("String", "LEGAL_PRIVACY_URL", quoted(legalPrivacyUrl))
        buildConfigField("String", "LEGAL_REFUND_URL", quoted(legalRefundUrl))
    }

    buildFeatures {
        buildConfig = true
    }

    androidResources {
        // Native loading/auth screens play this asset before WebView exists.
        noCompress += "mp3"
    }

    signingConfigs {
        create("directRelease") {
            if (hasDirectReleaseSigning) {
                storeFile = file(directReleaseStoreFile)
                storePassword = directReleaseStorePassword
                keyAlias = directReleaseKeyAlias
                keyPassword = directReleaseKeyPassword
                enableV1Signing = true
                enableV2Signing = true
                enableV3Signing = true
            }
        }
        create("rustoreRelease") {
            if (hasRustoreReleaseSigning) {
                storeFile = file(rustoreReleaseStoreFile)
                storePassword = rustoreReleaseStorePassword
                keyAlias = rustoreReleaseKeyAlias
                keyPassword = rustoreReleaseKeyPassword
                enableV1Signing = true
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        getByName("release")
    }

    flavorDimensions += "distribution"
    productFlavors {
        create("direct") {
            dimension = "distribution"
            if (hasDirectReleaseSigning) {
                signingConfig = signingConfigs.getByName("directRelease")
            }
            buildConfigField("String", "DISTRIBUTION_CHANNEL", quoted("direct"))
            buildConfigField("String", "PAYMENT_PROVIDER_ORDER", quoted("yookassa,stars"))
        }
        create("rustore") {
            dimension = "distribution"
            if (hasRustoreReleaseSigning) {
                signingConfig = signingConfigs.getByName("rustoreRelease")
            }
            buildConfigField("String", "DISTRIBUTION_CHANNEL", quoted("rustore"))
            buildConfigField("String", "RUSTORE_CONSOLE_APP_ID", quoted(rustoreConsoleAppId))
            buildConfigField("String", "RUSTORE_PAY_SCHEME", quoted(rustorePayScheme))
            buildConfigField("String", "RUSTORE_APP_URL", quoted(rustoreAppUrl))
            buildConfigField("String", "PAYMENT_PROVIDER_ORDER", quoted("yookassa,rustore"))
            manifestPlaceholders["sdk_pay_scheme_value"] = rustorePayScheme
            manifestPlaceholders["console_app_id_value"] = rustoreConsoleAppId
        }
    }

    sourceSets {
        getByName("main") {
            assets.setSrcDirs(listOf(extraArenaShellAssetsDir))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

tasks.named("preBuild") {
    dependsOn(syncExtraArenaShellAssets)
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    implementation(platform("com.google.firebase:firebase-bom:34.13.0"))
    implementation("com.google.firebase:firebase-messaging")
    add("directImplementation", "androidx.core:core:1.15.0")
    add("rustoreImplementation", platform("ru.rustore.sdk:bom:2026.04.02"))
    add("rustoreImplementation", "ru.rustore.sdk:pay")
    add("rustoreImplementation", "ru.rustore.sdk:appupdate")
}
