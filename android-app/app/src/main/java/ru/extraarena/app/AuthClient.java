package ru.extraarena.app;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.net.ssl.SSLException;

final class AuthClient {
    private static final String TAG = "EAAuthClient";
    private static final String NETWORK_REQUIRED_MESSAGE = "Нужен интернет. VPN для ExtraArena не нужен.";
    private static final String SERVER_UNAVAILABLE_MESSAGE =
            "Нужен интернет. Если сеть есть, сервер временно недоступен. VPN для игры не нужен.";
    private static final String KEY_PENDING_REVOCATIONS = "pending_auth_revocations_v1";
    private static final int MAX_PENDING_REVOCATIONS = 64;
    private static final Object REVOCATION_QUEUE_LOCK = new Object();

    interface Callback {
        void onSuccess(AuthResult result);
        void onError(String message);
    }

    interface SimpleCallback {
        void onSuccess();
        void onError(String message);
    }

    static final class AuthResult {
        final String token;
        final long userId;
        final String displayId;
        final boolean regBonus;
        final boolean emailVerificationRequired;
        final boolean emailSent;

        AuthResult(
                String token,
                long userId,
                String displayId,
                boolean regBonus,
                boolean emailVerificationRequired,
                boolean emailSent
        ) {
            this.token = token;
            this.userId = userId;
            this.displayId = displayId;
            this.regBonus = regBonus;
            this.emailVerificationRequired = emailVerificationRequired;
            this.emailSent = emailSent;
        }
    }

    private static final class PendingRevocation {
        final String token;
        final String baseUrl;
        final String whitelistCode;
        final long queuedAt;

        PendingRevocation(String token, String baseUrl, String whitelistCode, long queuedAt) {
            this.token = safeAuthToken(token);
            this.baseUrl = BaseUrlStore.normalize(baseUrl);
            this.whitelistCode = whitelistCode == null ? "" : whitelistCode.trim();
            this.queuedAt = queuedAt;
        }

        boolean sameCredential(PendingRevocation other) {
            return other != null
                    && token.equals(other.token)
                    && baseUrl.equals(other.baseUrl);
        }
    }

    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();
    private static final ExecutorService LOGOUT_EXECUTOR = Executors.newSingleThreadExecutor();

    private AuthClient() {
    }

    static void login(Context context, String email, String password, Callback callback) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = credentialsBody(email, password);
                JSONObject response = post(context, "/api/extraid/login", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess(toAuthResult(response));
                } else if (response.optBoolean("email_verification_required", false)) {
                    callback.onSuccess(toAuthResult(response));
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void startAnonymous(Context context, String nickname, Callback callback) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("nickname", nickname.trim());
                body.put("device_label", Build.MANUFACTURER + " " + Build.MODEL);
                JSONObject response = post(context, "/api/auth/anonymous", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess(toAuthResult(response));
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void register(Context context, String email, String password, String nickname, Callback callback) {
        register(
                context,
                email,
                password,
                nickname,
                DeviceRegistrar.getAuthToken(context),
                callback
        );
    }

    static void register(
            Context context,
            String email,
            String password,
            String nickname,
            String activeAuthToken,
            Callback callback
    ) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject registerBody = credentialsBody(email, password);
                registerBody.put("client", "android_app");
                if (nickname != null && !nickname.trim().isEmpty()) {
                    registerBody.put("nickname", nickname.trim());
                }
                JSONObject registerResponse = post(
                        context,
                        "/api/extraid/register",
                        registerBody,
                        activeAuthToken
                );
                if (!registerResponse.optBoolean("ok")) {
                    callback.onError(errorMessage(registerResponse));
                    return;
                }
                AuthResult registerResult = toAuthResult(registerResponse);
                if (registerResult.emailVerificationRequired) {
                    callback.onSuccess(registerResult);
                    return;
                }

                JSONObject loginResponse = post(context, "/api/extraid/login", credentialsBody(email, password));
                if (loginResponse.optBoolean("ok")) {
                    callback.onSuccess(toAuthResult(loginResponse));
                } else {
                    callback.onError("Аккаунт создан, но вход не выполнен. Попробуй войти.");
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    /**
     * Persists a server-session revocation before returning so an offline
     * logout is retried after the next app start or authentication operation.
     * The caller can still clear local state immediately without blocking UI.
     */
    static void logoutBestEffort(Context context, String authToken) {
        String cleanToken = safeAuthToken(authToken);
        if (cleanToken.isEmpty()) {
            return;
        }
        Context appContext = context.getApplicationContext();
        PendingRevocation pending = new PendingRevocation(
                cleanToken,
                BaseUrlStore.getBaseUrl(appContext),
                ConnectionProfileStore.getWhitelistCode(appContext),
                System.currentTimeMillis()
        );
        if (!enqueuePendingRevocation(appContext, pending)) {
            Log.w(TAG, "Failed to persist pending server logout");
        }
        flushPendingRevocations(appContext);
    }

    static void flushPendingRevocations(Context context) {
        Context appContext = context.getApplicationContext();
        LOGOUT_EXECUTOR.execute(() -> flushPendingRevocationsNow(appContext));
    }

    static boolean isRevocationPending(Context context, String baseUrl, String authToken) {
        PendingRevocation candidate = new PendingRevocation(authToken, baseUrl, "", 0L);
        if (candidate.token.isEmpty()) {
            return false;
        }
        for (PendingRevocation pending : readPendingRevocations(context)) {
            if (pending.sameCredential(candidate)) {
                return true;
            }
        }
        return false;
    }

    static void requestTelegramCode(Context context, String telegramId, SimpleCallback callback) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("telegram_id", telegramId.trim());
                JSONObject response = post(context, "/api/telegram-transfer/request-code", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess();
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void resendVerificationEmail(Context context, String email, SimpleCallback callback) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("email", email == null ? "" : email.trim().toLowerCase(Locale.ROOT));
                JSONObject response = post(context, "/api/extraid/email/resend", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess();
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void verifyEmailCode(
            Context context,
            String email,
            String code,
            SimpleCallback callback
    ) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("email", email == null ? "" : email.trim().toLowerCase(Locale.ROOT));
                body.put("code", code == null ? "" : code.trim());
                JSONObject response = post(context, "/api/extraid/email/verify", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess();
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void changeUnverifiedEmail(
            Context context,
            String authToken,
            String newEmail,
            SimpleCallback callback
    ) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put(
                        "email",
                        newEmail == null ? "" : newEmail.trim().toLowerCase(Locale.ROOT)
                );
                JSONObject response = post(
                        context,
                        "/api/extraid/email/change",
                        body,
                        authToken
                );
                if (response.optBoolean("ok")) {
                    callback.onSuccess();
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    static void completeTelegramTransfer(
            Context context,
            String telegramId,
            String code,
            String email,
            String password,
            Callback callback
    ) {
        flushPendingRevocations(context);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = credentialsBody(email, password);
                body.put("telegram_id", telegramId.trim());
                body.put("code", code.trim());
                JSONObject response = post(context, "/api/telegram-transfer/complete", body);
                if (response.optBoolean("ok")) {
                    callback.onSuccess(toAuthResult(response));
                } else {
                    callback.onError(errorMessage(response));
                }
            } catch (Exception e) {
                callback.onError(connectionMessage(e));
            }
        });
    }

    private static JSONObject credentialsBody(String email, String password) throws Exception {
        JSONObject body = new JSONObject();
        body.put("email", email.trim().toLowerCase(Locale.ROOT));
        body.put("password", password);
        body.put("device_label", Build.MANUFACTURER + " " + Build.MODEL);
        return body;
    }

    private static AuthResult toAuthResult(JSONObject response) {
        return new AuthResult(
                extractToken(response),
                response.optLong("user_id", 0L),
                response.optString("display_id", ""),
                response.optBoolean("reg_bonus", false),
                response.optBoolean("email_verification_required", false),
                response.optBoolean("email_sent", false)
        );
    }

    private static String extractToken(JSONObject response) {
        String token = response.optString("token", "");
        if (!token.trim().isEmpty()) {
            return token;
        }
        token = response.optString("access_token", "");
        if (!token.trim().isEmpty()) {
            return token;
        }
        token = response.optString("auth_token", "");
        if (!token.trim().isEmpty()) {
            return token;
        }
        JSONObject session = response.optJSONObject("session");
        if (session != null) {
            token = session.optString("token", "");
            if (!token.trim().isEmpty()) {
                return token;
            }
        }
        return "";
    }

    private static JSONObject post(Context context, String path, JSONObject body) throws Exception {
        return post(context, path, body, null);
    }

    private static JSONObject post(
            Context context,
            String path,
            JSONObject body,
            String authToken
    ) throws Exception {
        return post(
                context,
                BaseUrlStore.getBaseUrl(context),
                path,
                body,
                authToken,
                ConnectionProfileStore.getWhitelistCode(context)
        );
    }

    private static JSONObject post(
            Context context,
            String baseUrl,
            String path,
            JSONObject body,
            String authToken,
            String whitelistCode
    ) throws Exception {
        HttpURLConnection connection = null;
        try {
            URL url = new URL(BaseUrlStore.join(baseUrl, path));
            byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
            connection = (HttpURLConnection) url.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(10000);
            connection.setReadTimeout(10000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            String cleanAuthToken = safeAuthToken(authToken);
            if (!cleanAuthToken.isEmpty()) {
                connection.setRequestProperty("Authorization", "Bearer " + cleanAuthToken);
            }
            String cleanWhitelistCode = whitelistCode == null ? "" : whitelistCode.trim();
            if (!cleanWhitelistCode.isEmpty()) {
                connection.setRequestProperty("X-ExtraArena-Whitelist-Code", cleanWhitelistCode);
            }
            try (OutputStream output = connection.getOutputStream()) {
                output.write(payload);
            }
            int status = connection.getResponseCode();
            InputStream stream = status >= 400
                    ? connection.getErrorStream()
                    : connection.getInputStream();
            String text = readText(stream);
            if (status == HttpURLConnection.HTTP_UNAVAILABLE || status == 502 || status == 504 || status >= 500) {
                return errorResponse("server_unavailable", status);
            }
            if (text.isEmpty()) {
                return new JSONObject();
            }
            try {
                return new JSONObject(text);
            } catch (Exception ignored) {
                if (status >= 400) {
                    return errorResponse("server_unavailable", status);
                }
                throw ignored;
            }
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private static void flushPendingRevocationsNow(Context context) {
        List<PendingRevocation> snapshot = readPendingRevocations(context);
        // Prefer the session the user just logged out from. If connectivity is
        // down, stop after the first failure instead of blocking this worker
        // for every older entry in the bounded queue.
        for (int i = snapshot.size() - 1; i >= 0; i--) {
            PendingRevocation pending = snapshot.get(i);
            if (pending.token.isEmpty()) {
                continue;
            }
            try {
                JSONObject response = post(
                        context,
                        pending.baseUrl,
                        "/api/auth/logout",
                        new JSONObject(),
                        pending.token,
                        pending.whitelistCode
                );
                String error = response.optString("error", "");
                if (response.optBoolean("ok") || "invalid_jwt_session".equals(error)) {
                    removePendingRevocation(context, pending);
                } else {
                    Log.d(TAG, "Pending server logout was not acknowledged");
                    if ("server_unavailable".equals(error) || "rate_limited".equals(error)) {
                        break;
                    }
                }
            } catch (Exception e) {
                // Retain the encrypted queue entry for the next start/network operation.
                Log.d(TAG, "Pending server logout retained for retry", e);
                break;
            }
        }
    }

    private static boolean enqueuePendingRevocation(Context context, PendingRevocation pending) {
        if (pending == null || pending.token.isEmpty()) {
            return false;
        }
        synchronized (REVOCATION_QUEUE_LOCK) {
            ArrayList<PendingRevocation> queue = readPendingRevocationsUnlocked(context);
            for (PendingRevocation existing : queue) {
                if (existing.sameCredential(pending)) {
                    return true;
                }
            }
            while (queue.size() >= MAX_PENDING_REVOCATIONS) {
                queue.remove(0);
            }
            queue.add(pending);
            return persistPendingRevocationsUnlocked(context, queue);
        }
    }

    private static List<PendingRevocation> readPendingRevocations(Context context) {
        synchronized (REVOCATION_QUEUE_LOCK) {
            return new ArrayList<>(readPendingRevocationsUnlocked(context));
        }
    }

    private static ArrayList<PendingRevocation> readPendingRevocationsUnlocked(Context context) {
        ArrayList<PendingRevocation> queue = new ArrayList<>();
        String raw = SecurePrefs.getString(context, KEY_PENDING_REVOCATIONS, "");
        if (raw == null || raw.trim().isEmpty()) {
            return queue;
        }
        try {
            JSONArray array = new JSONArray(raw);
            int start = Math.max(0, array.length() - MAX_PENDING_REVOCATIONS);
            for (int i = start; i < array.length(); i++) {
                JSONObject item = array.optJSONObject(i);
                if (item == null) {
                    continue;
                }
                PendingRevocation pending = new PendingRevocation(
                        item.optString("token", ""),
                        item.optString("base_url", BuildConfig.DEFAULT_BASE_URL),
                        item.optString("whitelist_code", ""),
                        item.optLong("queued_at", 0L)
                );
                if (pending.token.isEmpty()) {
                    continue;
                }
                boolean duplicate = false;
                for (PendingRevocation existing : queue) {
                    if (existing.sameCredential(pending)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    queue.add(pending);
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Ignoring malformed pending logout queue", e);
            queue.clear();
        }
        return queue;
    }

    private static void removePendingRevocation(Context context, PendingRevocation completed) {
        synchronized (REVOCATION_QUEUE_LOCK) {
            ArrayList<PendingRevocation> queue = readPendingRevocationsUnlocked(context);
            for (int i = queue.size() - 1; i >= 0; i--) {
                if (queue.get(i).sameCredential(completed)) {
                    queue.remove(i);
                }
            }
            if (!persistPendingRevocationsUnlocked(context, queue)) {
                Log.w(TAG, "Failed to remove completed pending logout");
            }
        }
    }

    private static boolean persistPendingRevocationsUnlocked(
            Context context,
            List<PendingRevocation> queue
    ) {
        if (queue.isEmpty()) {
            SecurePrefs.remove(context, KEY_PENDING_REVOCATIONS);
            return true;
        }
        JSONArray array = new JSONArray();
        for (PendingRevocation pending : queue) {
            try {
                JSONObject item = new JSONObject();
                item.put("token", pending.token);
                item.put("base_url", pending.baseUrl);
                item.put("whitelist_code", pending.whitelistCode);
                item.put("queued_at", pending.queuedAt);
                array.put(item);
            } catch (Exception ignored) {
            }
        }
        return SecurePrefs.putString(context, KEY_PENDING_REVOCATIONS, array.toString());
    }

    private static String safeAuthToken(String authToken) {
        if (authToken == null) {
            return "";
        }
        String clean = authToken.trim();
        if (clean.isEmpty()
                || "null".equals(clean)
                || clean.indexOf('\r') >= 0
                || clean.indexOf('\n') >= 0) {
            return "";
        }
        return clean;
    }

    private static String readText(InputStream stream) throws Exception {
        if (stream == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
        }
        return builder.toString();
    }

    private static JSONObject errorResponse(String error, int status) throws Exception {
        JSONObject response = new JSONObject();
        response.put("ok", false);
        response.put("error", error);
        response.put("status", status);
        return response;
    }

    private static String connectionMessage(Exception exception) {
        if (isOfflineLike(exception)) {
            return NETWORK_REQUIRED_MESSAGE;
        }
        return SERVER_UNAVAILABLE_MESSAGE;
    }

    private static boolean isOfflineLike(Throwable throwable) {
        Throwable cursor = throwable;
        while (cursor != null) {
            if (cursor instanceof UnknownHostException
                    || cursor instanceof ConnectException
                    || cursor instanceof SocketTimeoutException
                    || cursor instanceof SSLException) {
                return true;
            }
            if (cursor instanceof IOException && cursor.getMessage() != null) {
                String message = cursor.getMessage().toLowerCase(Locale.ROOT);
                if (message.contains("network is unreachable")
                        || message.contains("failed to connect")
                        || message.contains("unable to resolve host")) {
                    return true;
                }
            }
            cursor = cursor.getCause();
        }
        return false;
    }

    private static String errorMessage(JSONObject response) {
        String error = response.optString("error", "");
        if ("invalid_credentials".equals(error)) return "Неверный email или пароль";
        if ("email_taken".equals(error)) return "Email уже занят";
        if ("nickname_taken".equals(error)) return "Никнейм занят";
        if ("password_too_short".equals(error)) return "Пароль должен быть не короче 8 символов";
        if ("password_too_long".equals(error)) return "Пароль должен занимать не больше 72 байт";
        if ("invalid_email".equals(error)) return "Проверь email";
        if ("invalid_nickname".equals(error)) return response.optString("message", "Никнейм должен быть от 3 до 20 символов");
        if ("invalid_telegram_id".equals(error)) return "Проверь Telegram ID";
        if ("telegram_user_not_found".equals(error)) return "Игрок с таким Telegram ID не найден";
        if ("telegram_delivery_failed".equals(error)) return "Не удалось отправить код. Открой бота и попробуй еще раз";
        if ("invalid_code".equals(error)) return "Неверный или просроченный код";
        if ("code_user_mismatch".equals(error)) return "Код выдан для другого Telegram ID";
        if ("extraid_already_exists".equals(error)) return "У этого Telegram-аккаунта уже есть ExtraID";
        if ("email_not_verified".equals(error)) return "Введи 6-значный код из письма, затем войди снова";
        if ("invalid_or_expired_code".equals(error)) return "Неверный или просроченный код";
        if ("email_delivery_failed".equals(error)) return "Не удалось отправить письмо. Попробуй создать ExtraID позже";
        if ("email_delivery_unavailable".equals(error)) return "Отправка писем временно недоступна";
        if ("email_change_not_allowed".equals(error)) return "Email уже подтверждён или его нельзя изменить";
        if ("extra_account_not_found".equals(error)) return "ExtraID для текущей игры не найден";
        if ("invalid_auth".equals(error) || "invalid_jwt_session".equals(error)) return "Сессия истекла. Войди снова";
        if ("rate_limited".equals(error)) return "Слишком много попыток. Попробуй чуть позже";
        if ("server_unavailable".equals(error)) return SERVER_UNAVAILABLE_MESSAGE;
        String message = response.optString("message", "");
        return message.isEmpty() ? "Что-то пошло не так" : message;
    }
}
