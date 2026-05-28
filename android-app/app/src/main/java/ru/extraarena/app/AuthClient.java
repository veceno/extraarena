package ru.extraarena.app;

import android.content.Context;
import android.os.Build;

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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.net.ssl.SSLException;

final class AuthClient {
    private static final String NETWORK_REQUIRED_MESSAGE = "Нужен интернет. VPN для ExtraArena не нужен.";
    private static final String SERVER_UNAVAILABLE_MESSAGE =
            "Нужен интернет. Если сеть есть, сервер временно недоступен. VPN для игры не нужен.";

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

        AuthResult(String token, long userId, String displayId, boolean regBonus) {
            this.token = token;
            this.userId = userId;
            this.displayId = displayId;
            this.regBonus = regBonus;
        }
    }

    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();

    private AuthClient() {
    }

    static void login(Context context, String email, String password, Callback callback) {
        EXECUTOR.execute(() -> {
            try {
                JSONObject body = credentialsBody(email, password);
                JSONObject response = post(context, "/api/extraid/login", body);
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

    static void startAnonymous(Context context, String nickname, Callback callback) {
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
        EXECUTOR.execute(() -> {
            try {
                JSONObject registerBody = credentialsBody(email, password);
                if (nickname != null && !nickname.trim().isEmpty()) {
                    registerBody.put("nickname", nickname.trim());
                }
                JSONObject registerResponse = post(context, "/api/extraid/register", registerBody);
                if (!registerResponse.optBoolean("ok")) {
                    callback.onError(errorMessage(registerResponse));
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

    static void requestTelegramCode(Context context, String telegramId, SimpleCallback callback) {
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

    static void completeTelegramTransfer(
            Context context,
            String telegramId,
            String code,
            String email,
            String password,
            Callback callback
    ) {
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
        body.put("email", email.trim().toLowerCase());
        body.put("password", password);
        body.put("device_label", Build.MANUFACTURER + " " + Build.MODEL);
        return body;
    }

    private static AuthResult toAuthResult(JSONObject response) {
        return new AuthResult(
                extractToken(response),
                response.optLong("user_id", 0L),
                response.optString("display_id", ""),
                response.optBoolean("reg_bonus", false)
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
        HttpURLConnection connection = null;
        try {
            URL url = new URL(BaseUrlStore.join(BaseUrlStore.getBaseUrl(context), path));
            byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(10000);
            connection.setReadTimeout(10000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            String whitelistCode = ConnectionProfileStore.getWhitelistCode(context);
            if (!whitelistCode.isEmpty()) {
                connection.setRequestProperty("X-ExtraArena-Whitelist-Code", whitelistCode);
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
                String message = cursor.getMessage().toLowerCase();
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
        if ("invalid_email".equals(error)) return "Проверь email";
        if ("invalid_nickname".equals(error)) return response.optString("message", "Никнейм должен быть от 3 до 20 символов");
        if ("invalid_telegram_id".equals(error)) return "Проверь Telegram ID";
        if ("telegram_user_not_found".equals(error)) return "Игрок с таким Telegram ID не найден";
        if ("telegram_delivery_failed".equals(error)) return "Не удалось отправить код. Открой бота и попробуй еще раз";
        if ("invalid_code".equals(error)) return "Неверный или просроченный код";
        if ("code_user_mismatch".equals(error)) return "Код выдан для другого Telegram ID";
        if ("extraid_already_exists".equals(error)) return "У этого Telegram-аккаунта уже есть ExtraID";
        if ("rate_limited".equals(error)) return "Слишком много попыток. Попробуй чуть позже";
        if ("server_unavailable".equals(error)) return SERVER_UNAVAILABLE_MESSAGE;
        String message = response.optString("message", "");
        return message.isEmpty() ? "Что-то пошло не так" : message;
    }
}
