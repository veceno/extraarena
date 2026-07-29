import org.gradle.api.tasks.Sync

plugins {
    id("com.android.application")
}

if (file("google-services.json").exists()) {
    apply(plugin = "com.google.gms.google-services")
}

val repoRoot = rootDir.parentFile
val extraArenaShellAssetsDir = layout.buildDirectory.dir("generated/extraArenaShellAssets")
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
val hasReleaseSigning = releaseStoreFile.isNotBlank()
    && releaseStorePassword.isNotBlank()
    && releaseKeyAlias.isNotBlank()
    && releaseKeyPassword.isNotBlank()
val syncExtraArenaShellAssets by tasks.registering(Sync::class) {
    into(extraArenaShellAssetsDir)
    from(repoRoot.resolve("webapp")) {
        into("ea_webapp")
        include("index.html")
        include("arena.html")
        include("analytics-v2.js")
        include("arena.js")
        include("arena-styles.css")
        include("main.js")
        include("styles.css")
        include("safe-area.js")
        include("matchmaking-tips.config.js")
        include("index.compiled.js")
        exclude(".DS_Store")
    }
    from(repoRoot.resolve("DesignAssets")) {
        into("DesignAssets")
        exclude("**/.DS_Store")
    }
    // ea_vendor stays in src/main/assets to avoid duplicate generated assets.
}

android {
    namespace = "ru.extraarena.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "ru.extraarena.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 47
        versionName = "0.5.1"

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

    signingConfigs {
        create("release") {
            if (hasReleaseSigning) {
                storeFile = file(releaseStoreFile)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
                enableV1Signing = true
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        getByName("release") {
            if (hasReleaseSigning) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    flavorDimensions += "distribution"
    productFlavors {
        create("direct") {
            dimension = "distribution"
            buildConfigField("String", "DISTRIBUTION_CHANNEL", quoted("direct"))
            buildConfigField("String", "PAYMENT_PROVIDER_ORDER", quoted("yookassa,stars"))
        }
        create("rustore") {
            dimension = "distribution"
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
            assets.srcDir(extraArenaShellAssetsDir)
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
    implementation(platform("com.google.firebase:firebase-bom:34.13.0"))
    implementation("com.google.firebase:firebase-messaging")
    add("rustoreImplementation", platform("ru.rustore.sdk:bom:2026.04.02"))
    add("rustoreImplementation", "ru.rustore.sdk:pay")
    add("rustoreImplementation", "ru.rustore.sdk:appupdate")
}
