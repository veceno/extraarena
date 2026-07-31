package ru.extraarena.app;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class DeviceRegistrar {
    private static final String TAG = "EADeviceRegistrar";
    private static final String KEY_LEGACY_AUTH_TOKEN = "auth_token";
    private static final String KEY_AUTH_TOKEN_PREFIX = "auth_token_v2_";
    private static final String KEY_FCM_TOKEN = "fcm_token";
    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();

    private DeviceRegistrar() {
    }

    static final class BackendBinding {
        final String baseUrl;
        final String whitelistCode;
        final String credentialScope;

        BackendBinding(String baseUrl, String whitelistCode) {
            this.baseUrl = BaseUrlStore.normalize(baseUrl);
            this.whitelistCode = whitelistCode == null ? "" : whitelistCode.trim();
            this.credentialScope = BaseUrlStore.identityScope(
                    this.baseUrl,
                    this.whitelistCode
            );
        }

        String endpoint(String path) {
            return BaseUrlStore.join(baseUrl, path);
        }

        boolean sameEndpoint(BackendBinding other) {
            return other != null
                    && baseUrl.equals(other.baseUrl)
                    && whitelistCode.equals(other.whitelistCode);
        }
    }

    private static BackendBinding selectedBackend(Context context) {
        ConnectionProfileStore.ConnectionProfile profile =
                ConnectionProfileStore.getSelectedProfile(context);
        return backendForProfile(profile);
    }

    private static BackendBinding backendForProfile(
            ConnectionProfileStore.ConnectionProfile profile
    ) {
        return new BackendBinding(
                profile.baseUrl,
                profile.whitelistEnabled ? profile.whitelistCode : ""
        );
    }

    static boolean saveAuthToken(Context context, String token) {
        BackendBinding backend = selectedBackend(context);
        return saveAuthToken(context, backend.baseUrl, backend.whitelistCode, token);
    }

    static boolean saveAuthToken(
            Context context,
            String baseUrl,
            String whitelistCode,
            String token
    ) {
        if (token == null || token.trim().isEmpty() || "null".equals(token)) {
            return false;
        }
        BackendBinding backend = new BackendBinding(baseUrl, whitelistCode);
        boolean saved = SecurePrefs.putString(
                context,
                authTokenStorageKey(backend.baseUrl, backend.whitelistCode),
                token
        );
        if (!saved) {
            Log.w(TAG, "Failed to save auth token");
            return false;
        }
        registerIfReady(context, backend);
        return true;
    }

    static String getAuthToken(Context context) {
        Context appContext = context.getApplicationContext();
        BackendBinding backend = selectedBackend(appContext);
        String token = getAuthToken(
                appContext,
                backend.baseUrl,
                backend.whitelistCode
        );
        if (!token.isEmpty()) {
            return token;
        }
        // One-time migration: the legacy global token belongs to the profile
        // selected at upgrade time. Never copy it into any non-selected scope.
        String legacy = SecurePrefs.getString(appContext, KEY_LEGACY_AUTH_TOKEN, "");
        if (legacy == null || legacy.trim().isEmpty()) {
            return "";
        }
        if (!SecurePrefs.putString(
                appContext,
                authTokenStorageKey(backend.baseUrl, backend.whitelistCode),
                legacy
        )) {
            return "";
        }
        SecurePrefs.remove(appContext, KEY_LEGACY_AUTH_TOKEN);
        return legacy;
    }

    static String getAuthToken(Context context, String baseUrl) {
        ConnectionProfileStore.ConnectionProfile selected =
                ConnectionProfileStore.getSelectedProfile(context);
        String whitelistCode = BaseUrlStore.normalize(selected.baseUrl).equals(
                BaseUrlStore.normalize(baseUrl)
        ) && selected.whitelistEnabled ? selected.whitelistCode : "";
        return getAuthToken(context, baseUrl, whitelistCode);
    }

    static String getAuthToken(
            Context context,
            String baseUrl,
            String whitelistCode
    ) {
        return SecurePrefs.getString(
                context,
                authTokenStorageKey(baseUrl, whitelistCode),
                ""
        );
    }

    static void clearAuthToken(Context context) {
        Context appContext = context.getApplicationContext();
        BackendBinding backend = selectedBackend(appContext);
        String authToken = getAuthToken(appContext);
        String fcmToken = SecurePrefs.getString(appContext, KEY_FCM_TOKEN, "");
        if (authToken != null && !authToken.isEmpty() && fcmToken != null && !fcmToken.isEmpty()) {
            postDeviceToken(
                    appContext,
                    backend,
                    BuildConfig.PUSH_UNREGISTER_PATH,
                    authToken,
                    fcmToken
            );
        }
        SecurePrefs.remove(
                appContext,
                authTokenStorageKey(backend.baseUrl, backend.whitelistCode)
        );
        SecurePrefs.remove(appContext, KEY_LEGACY_AUTH_TOKEN);
    }

    static void saveFcmToken(Context context, String token) {
        if (token == null || token.trim().isEmpty()) {
            return;
        }
        SecurePrefs.putString(context, KEY_FCM_TOKEN, token);
        registerIfReady(context);
    }

    static void registerIfReady(Context context) {
        Context appContext = context.getApplicationContext();
        registerIfReady(appContext, selectedBackend(appContext));
    }

    static void registerIfReady(
            Context context,
            String baseUrl,
            String whitelistCode
    ) {
        registerIfReady(
                context.getApplicationContext(),
                new BackendBinding(baseUrl, whitelistCode)
        );
    }

    private static void registerIfReady(Context appContext, BackendBinding backend) {
        String authToken = getAuthToken(
                appContext,
                backend.baseUrl,
                backend.whitelistCode
        );
        String fcmToken = SecurePrefs.getString(appContext, KEY_FCM_TOKEN, "");
        if (authToken == null || authToken.isEmpty() || fcmToken == null || fcmToken.isEmpty()) {
            return;
        }

        postDeviceToken(
                appContext,
                backend,
                BuildConfig.PUSH_REGISTER_PATH,
                authToken,
                fcmToken
        );
    }

    static void syncProfileTransition(
            Context context,
            ConnectionProfileStore.ConnectionProfile oldProfile,
            ConnectionProfileStore.ConnectionProfile newProfile
    ) {
        if (oldProfile == null || newProfile == null) {
            return;
        }
        Context appContext = context.getApplicationContext();
        BackendBinding oldBackend = backendForProfile(oldProfile);
        BackendBinding newBackend = backendForProfile(newProfile);
        if (oldBackend.sameEndpoint(newBackend)) {
            return;
        }
        // Called before selectProfile: this also migrates the one legacy token
        // into the immutable old backend scope.
        String oldAuthToken = getAuthToken(appContext);
        String newAuthToken = getAuthToken(
                appContext,
                newBackend.baseUrl,
                newBackend.whitelistCode
        );
        String fcmToken = SecurePrefs.getString(appContext, KEY_FCM_TOKEN, "");
        if (fcmToken == null || fcmToken.isEmpty()) {
            return;
        }
        if (oldAuthToken != null && !oldAuthToken.isEmpty()) {
            postDeviceToken(
                    appContext,
                    oldBackend,
                    BuildConfig.PUSH_UNREGISTER_PATH,
                    oldAuthToken,
                    fcmToken
            );
        }
        if (newAuthToken != null && !newAuthToken.isEmpty()) {
            postDeviceToken(
                    appContext,
                    newBackend,
                    BuildConfig.PUSH_REGISTER_PATH,
                    newAuthToken,
                    fcmToken
            );
        }
    }

    static String authTokenStorageKey(String baseUrl) {
        return authTokenStorageKey(baseUrl, "");
    }

    static String authTokenStorageKey(String baseUrl, String whitelistCode) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                    BaseUrlStore.identityScope(baseUrl, whitelistCode).getBytes(StandardCharsets.UTF_8)
            );
            StringBuilder encoded = new StringBuilder(digest.length * 2);
            for (byte value : digest) {
                encoded.append(String.format("%02x", value & 0xff));
            }
            return KEY_AUTH_TOKEN_PREFIX + encoded;
        } catch (Exception e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    private static void postDeviceToken(
            Context appContext,
            BackendBinding backend,
            String path,
            String authToken,
            String fcmToken
    ) {
        // Every value used by the worker is immutable. A profile switch after
        // enqueue can no longer redirect an unregister/register to another host.
        final String endpoint = backend.endpoint(path);
        final String whitelistCode = backend.whitelistCode;
        EXECUTOR.execute(() -> {
            HttpURLConnection connection = null;
            try {
                URL url = new URL(endpoint);
                JSONObject body = new JSONObject();
                body.put("platform", "android");
                body.put("token", fcmToken);
                body.put("app_version", BuildConfig.VERSION_NAME);
                body.put("device_label", Build.MANUFACTURER + " " + Build.MODEL);
                body.put("os_name", "Android");
                body.put("os_version", Build.VERSION.RELEASE);
                TimeZone timeZone = TimeZone.getDefault();
                body.put("timezone", timeZone.getID());
                body.put("utc_offset_minutes", timeZone.getOffset(System.currentTimeMillis()) / 60000);

                byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
                connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setConnectTimeout(6000);
                connection.setReadTimeout(6000);
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
                connection.setRequestProperty("Authorization", "Bearer " + authToken);
                if (!whitelistCode.isEmpty()) {
                    connection.setRequestProperty("X-ExtraArena-Whitelist-Code", whitelistCode);
                }
                try (OutputStream output = connection.getOutputStream()) {
                    output.write(payload);
                }
                int status = connection.getResponseCode();
                if (status < 200 || status >= 300) {
                    Log.d(TAG, "Push device sync returned HTTP " + status);
                }
            } catch (Exception e) {
                Log.d(TAG, "Push device sync failed", e);
            } finally {
                if (connection != null) {
                    connection.disconnect();
                }
            }
        });
    }

}
