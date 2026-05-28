package ru.extraarena.app;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class DeviceRegistrar {
    private static final String TAG = "EADeviceRegistrar";
    private static final String KEY_AUTH_TOKEN = "auth_token";
    private static final String KEY_FCM_TOKEN = "fcm_token";
    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();

    private DeviceRegistrar() {
    }

    static boolean saveAuthToken(Context context, String token) {
        if (token == null || token.trim().isEmpty() || "null".equals(token)) {
            return false;
        }
        boolean saved = SecurePrefs.putString(context, KEY_AUTH_TOKEN, token);
        if (!saved) {
            Log.w(TAG, "Failed to save auth token");
            return false;
        }
        registerIfReady(context);
        return true;
    }

    static String getAuthToken(Context context) {
        return SecurePrefs.getString(context, KEY_AUTH_TOKEN, "");
    }

    static void clearAuthToken(Context context) {
        Context appContext = context.getApplicationContext();
        String authToken = SecurePrefs.getString(appContext, KEY_AUTH_TOKEN, "");
        String fcmToken = SecurePrefs.getString(appContext, KEY_FCM_TOKEN, "");
        if (authToken != null && !authToken.isEmpty() && fcmToken != null && !fcmToken.isEmpty()) {
            postDeviceToken(appContext, BuildConfig.PUSH_UNREGISTER_PATH, authToken, fcmToken);
        }
        SecurePrefs.remove(appContext, KEY_AUTH_TOKEN);
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
        String authToken = SecurePrefs.getString(appContext, KEY_AUTH_TOKEN, "");
        String fcmToken = SecurePrefs.getString(appContext, KEY_FCM_TOKEN, "");
        if (authToken == null || authToken.isEmpty() || fcmToken == null || fcmToken.isEmpty()) {
            return;
        }

        postDeviceToken(appContext, BuildConfig.PUSH_REGISTER_PATH, authToken, fcmToken);
    }

    private static void postDeviceToken(Context appContext, String path, String authToken, String fcmToken) {
        EXECUTOR.execute(() -> {
            HttpURLConnection connection = null;
            try {
                URL url = new URL(BaseUrlStore.join(BaseUrlStore.getBaseUrl(appContext), path));
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
                String whitelistCode = ConnectionProfileStore.getWhitelistCode(appContext);
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
