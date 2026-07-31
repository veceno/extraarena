package ru.extraarena.app;

import android.content.Context;
import android.os.Build;
import android.util.Base64;
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
import java.security.MessageDigest;
import java.security.SecureRandom;
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
    private static final String KEY_ANONYMOUS_BOOTSTRAP_PREFIX = "anonymous_bootstrap_v1_";
    private static final int MAX_PENDING_REVOCATIONS = 64;
    private static final int BOOTSTRAP_SECRET_BYTES = 32;
    private static final Object REVOCATION_QUEUE_LOCK = new Object();
    private static final Object BOOTSTRAP_LOCK = new Object();
    private static final SecureRandom SECURE_RANDOM = new SecureRandom();

    interface Callback {
        void onSuccess(AuthResult result);
        void onError(String message);
    }

    interface SimpleCallback {
        void onSuccess();
        void onError(String message);
    }

    interface BootstrapCallback {
        void onSuccess(AuthResult result);
        void onError(String errorCode, String message);
    }

    static final class AuthResult {
        final String token;
        final long userId;
        final String displayId;
        final boolean regBonus;
        final boolean emailVerificationRequired;
        final boolean emailSent;
        final boolean emailQueued;
        final long bootstrapGeneration;

        AuthResult(
                String token,
                long userId,
                String displayId,
                boolean regBonus,
                boolean emailVerificationRequired,
                boolean emailSent,
                boolean emailQueued,
                long bootstrapGeneration
        ) {
            this.token = token;
            this.userId = userId;
            this.displayId = displayId;
            this.regBonus = regBonus;
            this.emailVerificationRequired = emailVerificationRequired;
            this.emailSent = emailSent;
            this.emailQueued = emailQueued;
            this.bootstrapGeneration = bootstrapGeneration;
        }
    }

    private static final class AnonymousBootstrap {
        final String baseUrl;
        final String whitelistCode;
        final String id;
        final String secret;
        final String nickname;

        AnonymousBootstrap(
                String baseUrl,
                String whitelistCode,
                String id,
                String secret,
                String nickname
        ) {
            this.baseUrl = baseUrl;
            this.whitelistCode = whitelistCode;
            this.id = id;
            this.secret = secret;
            this.nickname = nickname;
        }
    }

    private static final class BackendTarget {
        final String baseUrl;
        final String whitelistCode;

        BackendTarget(ConnectionProfileStore.ConnectionProfile profile) {
            this.baseUrl = BaseUrlStore.normalize(profile.baseUrl);
            this.whitelistCode = profile.whitelistEnabled ? profile.whitelistCode : "";
        }
    }

    private static final class PendingRevocation {
        final String token;
        final String baseUrl;
        final String whitelistCode;
        final String credentialScope;
        final long queuedAt;

        PendingRevocation(String token, String baseUrl, String whitelistCode, long queuedAt) {
            this.token = safeAuthToken(token);
            this.baseUrl = BaseUrlStore.normalize(baseUrl);
            this.whitelistCode = whitelistCode == null ? "" : whitelistCode.trim();
            this.credentialScope = BaseUrlStore.identityScope(
                    this.baseUrl,
                    this.whitelistCode
            );
            this.queuedAt = queuedAt;
        }

        boolean sameCredential(PendingRevocation other) {
            return other != null
                    && token.equals(other.token)
                    && credentialScope.equals(other.credentialScope);
        }
    }

    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();
    private static final ExecutorService LOGOUT_EXECUTOR = Executors.newSingleThreadExecutor();

    private AuthClient() {
    }

    private static BackendTarget selectedBackend(Context context) {
        return new BackendTarget(ConnectionProfileStore.getSelectedProfile(context));
    }

    static void login(Context context, String email, String password, Callback callback) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = credentialsBody(email, password);
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/login",
                        body,
                        null,
                        backend.whitelistCode
                );
                if (response.optBoolean("ok")) {
                    AuthResult result = toAuthResult(response);
                    clearAnonymousBootstrapAfterVerifiedLogin(
                            appContext,
                            backend.baseUrl,
                            backend.whitelistCode,
                            result
                    );
                    callback.onSuccess(result);
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
        startAnonymousBootstrap(context, nickname, new BootstrapCallback() {
            @Override
            public void onSuccess(AuthResult result) {
                callback.onSuccess(result);
            }

            @Override
            public void onError(String errorCode, String message) {
                callback.onError(message);
            }
        });
    }

    /**
     * Creates the game session required by production ExtraID registration.
     *
     * The credential is encrypted before the request and scoped to the selected backend.
     * Retrying after a lost response sends the same high-entropy pair, allowing the server to
     * rotate a JWT for the same anonymous user instead of allocating a duplicate account.
     */
    static void startAnonymousBootstrap(Context context, BootstrapCallback callback) {
        startAnonymousBootstrap(context, "", callback);
    }

    static void startAnonymousBootstrap(
            Context context,
            String requestedNickname,
            BootstrapCallback callback
    ) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        ConnectionProfileStore.ConnectionProfile selectedProfile =
                ConnectionProfileStore.getSelectedProfile(appContext);
        EXECUTOR.execute(() -> {
            try {
                AnonymousBootstrap bootstrap = getOrCreateAnonymousBootstrap(
                        appContext,
                        requestedNickname,
                        selectedProfile
                );
                if (bootstrap == null) {
                    callback.onError(
                            "bootstrap_storage_unavailable",
                            "Не удалось безопасно подготовить игровую сессию. "
                                    + "Нажми «Создать аккаунт» ещё раз."
                    );
                    return;
                }
                JSONObject body = new JSONObject();
                body.put("nickname", bootstrap.nickname);
                body.put("device_label", Build.MANUFACTURER + " " + Build.MODEL);
                body.put("bootstrap_id", bootstrap.id);
                body.put("bootstrap_secret", bootstrap.secret);
                JSONObject response = post(
                        appContext,
                        bootstrap.baseUrl,
                        "/api/auth/anonymous",
                        body,
                        null,
                        bootstrap.whitelistCode
                );
                if (response.optBoolean("ok")) {
                    callback.onSuccess(toAuthResult(response));
                } else {
                    callback.onError(
                            response.optString("error", ""),
                            errorMessage(response)
                    );
                }
            } catch (Exception e) {
                callback.onError("", connectionMessage(e));
            }
        });
    }

    /** Saves only the newest server generation for this backend/profile. */
    static boolean saveAnonymousBootstrapAuth(Context context, AuthResult result) {
        if (result == null
                || safeAuthToken(result.token).isEmpty()
                || result.bootstrapGeneration <= 0L) {
            return false;
        }
        Context appContext = context.getApplicationContext();
        synchronized (BOOTSTRAP_LOCK) {
            ConnectionProfileStore.ConnectionProfile selectedProfile =
                    ConnectionProfileStore.getSelectedProfile(appContext);
            String baseUrl = BaseUrlStore.normalize(selectedProfile.baseUrl);
            String whitelistCode = selectedProfile.whitelistEnabled
                    ? selectedProfile.whitelistCode
                    : "";
            String storageKey = anonymousBootstrapStorageKey(baseUrl, whitelistCode);
            String raw = SecurePrefs.getString(appContext, storageKey, "");
            if (raw.isEmpty()) {
                return false;
            }
            try {
                JSONObject saved = new JSONObject(raw);
                long acceptedGeneration = saved.optLong("accepted_generation", 0L);
                if (result.bootstrapGeneration < acceptedGeneration) {
                    Log.w(TAG, "Ignoring stale anonymous bootstrap response");
                    return false;
                }
                String requestBaseUrl = BaseUrlStore.normalize(
                        saved.optString("base_url", baseUrl)
                );
                whitelistCode = saved.optString("whitelist_code", "");
                if (!DeviceRegistrar.saveAuthToken(
                        appContext,
                        requestBaseUrl,
                        whitelistCode,
                        result.token
                )) {
                    return false;
                }
                saved.put("accepted_generation", result.bootstrapGeneration);
                if (result.userId > 0L) {
                    saved.put("user_id", result.userId);
                }
                return SecurePrefs.putString(appContext, storageKey, saved.toString());
            } catch (Exception e) {
                Log.w(TAG, "Failed to accept anonymous bootstrap response", e);
                return false;
            }
        }
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
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject registerBody = credentialsBody(email, password);
                registerBody.put("client", "android_app");
                if (nickname != null && !nickname.trim().isEmpty()) {
                    registerBody.put("nickname", nickname.trim());
                }
                JSONObject registerResponse = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/register",
                        registerBody,
                        activeAuthToken,
                        backend.whitelistCode
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

                JSONObject loginResponse = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/login",
                        credentialsBody(email, password),
                        null,
                        backend.whitelistCode
                );
                if (loginResponse.optBoolean("ok")) {
                    AuthResult loginResult = toAuthResult(loginResponse);
                    clearAnonymousBootstrapAfterVerifiedLogin(
                            appContext,
                            backend.baseUrl,
                            backend.whitelistCode,
                            loginResult
                    );
                    callback.onSuccess(loginResult);
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
        ConnectionProfileStore.ConnectionProfile selected =
                ConnectionProfileStore.getSelectedProfile(context);
        String whitelistCode = BaseUrlStore.normalize(selected.baseUrl).equals(
                BaseUrlStore.normalize(baseUrl)
        ) && selected.whitelistEnabled ? selected.whitelistCode : "";
        return isRevocationPendingForScope(
                context,
                BaseUrlStore.identityScope(baseUrl, whitelistCode),
                authToken
        );
    }

    static boolean isRevocationPendingForScope(
            Context context,
            String credentialScope,
            String authToken
    ) {
        String token = safeAuthToken(authToken);
        if (token.isEmpty()) {
            return false;
        }
        for (PendingRevocation pending : readPendingRevocations(context)) {
            if (pending.token.equals(token)
                    && pending.credentialScope.equals(credentialScope)) {
                return true;
            }
        }
        return false;
    }

    static void requestTelegramCode(Context context, String telegramId, SimpleCallback callback) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("telegram_id", telegramId.trim());
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/telegram-transfer/request-code",
                        body,
                        null,
                        backend.whitelistCode
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

    static void resendVerificationEmail(Context context, String email, SimpleCallback callback) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("email", email == null ? "" : email.trim().toLowerCase(Locale.ROOT));
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/email/resend",
                        body,
                        null,
                        backend.whitelistCode
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

    static void verifyEmailCode(
            Context context,
            String email,
            String code,
            SimpleCallback callback
    ) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("email", email == null ? "" : email.trim().toLowerCase(Locale.ROOT));
                body.put("code", code == null ? "" : code.trim());
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/email/verify",
                        body,
                        null,
                        backend.whitelistCode
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

    static void changeUnverifiedEmail(
            Context context,
            String authToken,
            String newEmail,
            SimpleCallback callback
    ) {
        flushPendingRevocations(context);
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put(
                        "email",
                        newEmail == null ? "" : newEmail.trim().toLowerCase(Locale.ROOT)
                );
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/extraid/email/change",
                        body,
                        authToken,
                        backend.whitelistCode
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
        Context appContext = context.getApplicationContext();
        BackendTarget backend = selectedBackend(appContext);
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = credentialsBody(email, password);
                body.put("telegram_id", telegramId.trim());
                body.put("code", code.trim());
                JSONObject response = post(
                        appContext,
                        backend.baseUrl,
                        "/api/telegram-transfer/complete",
                        body,
                        null,
                        backend.whitelistCode
                );
                if (response.optBoolean("ok")) {
                    AuthResult result = toAuthResult(response);
                    clearAnonymousBootstrapAfterVerifiedLogin(
                            appContext,
                            backend.baseUrl,
                            backend.whitelistCode,
                            result
                    );
                    callback.onSuccess(result);
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

    private static AnonymousBootstrap getOrCreateAnonymousBootstrap(
            Context context,
            String requestedNickname,
            ConnectionProfileStore.ConnectionProfile profile
    ) {
        synchronized (BOOTSTRAP_LOCK) {
            String baseUrl = BaseUrlStore.normalize(profile.baseUrl);
            String whitelistCode = profile.whitelistEnabled ? profile.whitelistCode : "";
            String storageKey = anonymousBootstrapStorageKey(baseUrl, whitelistCode);
            String raw = SecurePrefs.getString(context, storageKey, "");
            if (!raw.isEmpty()) {
                try {
                    JSONObject saved = new JSONObject(raw);
                    String id = saved.optString("id", "");
                    String secret = saved.optString("secret", "");
                    String nickname = saved.optString(
                            "nickname",
                            bootstrapNickname(id)
                    );
                    if (isBootstrapCredential(id) && isBootstrapCredential(secret)) {
                        String requested = requestedNickname == null
                                ? ""
                                : requestedNickname.trim();
                        if (isBootstrapNickname(requested)
                                && !requested.equals(nickname)) {
                            // Persist before sending so a lost response retries
                            // the same owner and the same user-chosen nickname.
                            saved.put("nickname", requested);
                            nickname = requested;
                        }
                        // The credential identity is shared by official
                        // entrypoints, but each request binding is not. Persist
                        // the exact captured endpoint/whitelist before POST so
                        // its eventual token registration cannot drift.
                        saved.put("nickname", nickname);
                        saved.put("base_url", baseUrl);
                        saved.put(
                                "whitelist_code",
                                profile.whitelistEnabled ? profile.whitelistCode : ""
                        );
                        if (!SecurePrefs.putString(
                                context,
                                storageKey,
                                saved.toString()
                        )) {
                            return null;
                        }
                        return new AnonymousBootstrap(
                                baseUrl,
                                profile.whitelistEnabled ? profile.whitelistCode : "",
                                id,
                                secret,
                                isBootstrapNickname(nickname)
                                        ? nickname
                                        : bootstrapNickname(id)
                        );
                    }
                } catch (Exception ignored) {
                }
                // Never overwrite a corrupt pending credential: the server may already have
                // committed it and a replacement could allocate a duplicate anonymous user.
                Log.w(TAG, "Unreadable anonymous bootstrap credential for selected backend");
                return null;
            }

            String id = randomBootstrapCredential();
            String secret = randomBootstrapCredential();
            String nickname = requestedNickname == null ? "" : requestedNickname.trim();
            if (!isBootstrapNickname(nickname)) {
                nickname = bootstrapNickname(id);
            }
            JSONObject value = new JSONObject();
            try {
                value.put("id", id);
                value.put("secret", secret);
                value.put("nickname", nickname);
                value.put("accepted_generation", 0L);
                value.put("base_url", baseUrl);
                value.put(
                        "whitelist_code",
                        profile.whitelistEnabled ? profile.whitelistCode : ""
                );
            } catch (Exception ignored) {
                return null;
            }
            if (!SecurePrefs.putString(context, storageKey, value.toString())) {
                return null;
            }
            return new AnonymousBootstrap(
                    baseUrl,
                    profile.whitelistEnabled ? profile.whitelistCode : "",
                    id,
                    secret,
                    nickname
            );
        }
    }

    private static void clearAnonymousBootstrapAfterVerifiedLogin(
            Context context,
            String baseUrl,
            String whitelistCode,
            AuthResult result
    ) {
        if (result == null || result.userId <= 0L || safeAuthToken(result.token).isEmpty()) {
            return;
        }
        synchronized (BOOTSTRAP_LOCK) {
            String storageKey = anonymousBootstrapStorageKey(baseUrl, whitelistCode);
            String raw = SecurePrefs.getString(context, storageKey, "");
            if (raw.isEmpty()) {
                return;
            }
            try {
                JSONObject saved = new JSONObject(raw);
                if (saved.optLong("user_id", 0L) == result.userId) {
                    // The server has now created a verified email/password
                    // session for this exact owner and disabled the bootstrap.
                    SecurePrefs.remove(context, storageKey);
                }
            } catch (Exception e) {
                Log.w(TAG, "Keeping unreadable bootstrap after verified login", e);
            }
        }
    }

    private static String anonymousBootstrapStorageKey(
            String normalizedBaseUrl,
            String whitelistCode
    ) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                    BaseUrlStore.identityScope(
                            normalizedBaseUrl,
                            whitelistCode
                    ).getBytes(StandardCharsets.UTF_8)
            );
            return KEY_ANONYMOUS_BOOTSTRAP_PREFIX
                    + Base64.encodeToString(
                    digest,
                    Base64.URL_SAFE | Base64.NO_WRAP | Base64.NO_PADDING
            );
        } catch (Exception e) {
            // SHA-256 is guaranteed by Android. Refuse to collapse backend scopes if the
            // provider is unexpectedly unavailable.
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    private static String randomBootstrapCredential() {
        byte[] bytes = new byte[BOOTSTRAP_SECRET_BYTES];
        SECURE_RANDOM.nextBytes(bytes);
        return Base64.encodeToString(
                bytes,
                Base64.URL_SAFE | Base64.NO_WRAP | Base64.NO_PADDING
        );
    }

    private static boolean isBootstrapCredential(String value) {
        return value != null && value.matches("[A-Za-z0-9_-]{43}");
    }

    private static String bootstrapNickname(String bootstrapId) {
        long suffix = 0L;
        for (int i = 0; i < Math.min(12, bootstrapId.length()); i++) {
            suffix = (suffix * 131L + bootstrapId.charAt(i)) % 100_000_000L;
        }
        return String.format(Locale.ROOT, "Arena%08d", suffix);
    }

    private static boolean isBootstrapNickname(String value) {
        return value != null && value.matches("[A-Za-z0-9_-]{3,20}");
    }

    private static AuthResult toAuthResult(JSONObject response) {
        return new AuthResult(
                extractToken(response),
                response.optLong("user_id", 0L),
                response.optString("display_id", ""),
                response.optBoolean("reg_bonus", false),
                response.optBoolean("email_verification_required", false),
                response.optBoolean("email_sent", false),
                response.optBoolean("email_queued", false),
                response.optLong("bootstrap_generation", 0L)
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
            if (text.isEmpty()) {
                return status >= 500
                        ? errorResponse("server_unavailable", status)
                        : new JSONObject();
            }
            try {
                JSONObject response = new JSONObject(text);
                if (status >= 500
                        && response.optString("error", "").trim().isEmpty()
                        && response.optString("message", "").trim().isEmpty()) {
                    return errorResponse("server_unavailable", status);
                }
                if (status >= 400) {
                    response.put("ok", false);
                    response.put("status", status);
                }
                return response;
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
        if ("email_delivery_failed".equals(error)) return response.optString(
                "message",
                "Не удалось отправить письмо. Попробуй ещё раз чуть позже"
        );
        if ("email_delivery_unavailable".equals(error)) return "Отправка писем временно недоступна";
        if ("email_change_not_allowed".equals(error)) return "Email уже подтверждён или его нельзя изменить";
        if ("extra_account_not_found".equals(error)) return "ExtraID для текущей игры не найден";
        if ("invalid_auth".equals(error) || "invalid_jwt_session".equals(error)) return "Сессия истекла. Войди снова";
        if ("rate_limited".equals(error)) return "Слишком много попыток. Попробуй чуть позже";
        if ("bootstrap_upgraded".equals(error)) return "Игровая сессия уже защищена ExtraID. Войди с email";
        if ("server_unavailable".equals(error)) return SERVER_UNAVAILABLE_MESSAGE;
        String message = response.optString("message", "");
        return message.isEmpty() ? "Что-то пошло не так" : message;
    }
}
