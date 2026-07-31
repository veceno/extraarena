from pathlib import Path


ANDROID_SOURCE = (
    Path("android-app")
    / "app"
    / "src"
    / "main"
    / "java"
    / "ru"
    / "extraarena"
    / "app"
)


def _source(name: str) -> str:
    return (ANDROID_SOURCE / name).read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_android_logout_purges_saved_token_and_retries_server_revocation():
    auth = _source("AuthClient.java")
    accounts = _source("ExtraIdAccountStore.java")
    activity = _source("MainActivity.java")

    logout = _between(
        auth,
        "static void logoutBestEffort(Context context, String authToken)",
        "static void flushPendingRevocations(Context context)",
    )
    assert "enqueuePendingRevocation(appContext, pending)" in logout
    assert "flushPendingRevocations(appContext)" in logout
    assert logout.index("enqueuePendingRevocation") < logout.index("flushPendingRevocations")

    assert 'KEY_PENDING_REVOCATIONS = "pending_auth_revocations_v1"' in auth
    assert "MAX_PENDING_REVOCATIONS = 64" in auth
    assert "SecurePrefs.getString(context, KEY_PENDING_REVOCATIONS" in auth
    assert "SecurePrefs.putString(context, KEY_PENDING_REVOCATIONS" in auth
    assert 'pending.baseUrl,\n                        "/api/auth/logout"' in auth
    assert 'response.optBoolean("ok") || "invalid_jwt_session".equals(error)' in auth
    assert "connection.setInstanceFollowRedirects(false);" in auth
    assert "removePendingRevocation(context, pending)" in auth
    assert "static boolean isRevocationPending(" in auth
    assert "!AuthClient.isRevocationPending(context, account.baseUrl, account.token)" in accounts
    assert "purgePendingLocalCredential();" in activity
    assert "AuthClient.flushPendingRevocations(this);" in activity

    remove_by_token = _between(
        accounts,
        "static boolean removeAccountByToken(Context context, String baseUrl, String token)",
        "static void touchAccountByToken",
    )
    assert "account.token.equals(cleanToken)" in remove_by_token
    assert "persistAccounts(context, accounts)" in remove_by_token

    current_logout = _between(
        activity,
        "private void logoutCurrentDevice()",
        "\n    }\n}",
    )
    assert "AuthClient.logoutBestEffort(this, activeToken)" in current_logout
    assert "ExtraIdAccountStore.removeAccountByToken(" in current_logout
    assert "DeviceRegistrar.clearAuthToken(this)" in current_logout
    assert current_logout.index("logoutBestEffort") < current_logout.index("removeAccountByToken")
    assert current_logout.index("removeAccountByToken") < current_logout.index("clearAuthToken")


def test_android_registration_handles_email_verification_without_replacing_active_session():
    auth = _source("AuthClient.java")
    activity = _source("MainActivity.java")

    assert "final boolean emailVerificationRequired;" in auth
    assert "final boolean emailSent;" in auth
    assert 'response.optBoolean("email_verification_required", false)' in auth
    assert 'response.optBoolean("email_sent", false)' in auth
    assert '"email_not_verified".equals(error)' in auth
    login = _between(
        auth,
        "static void login(Context context, String email, String password, Callback callback)",
        "static void startAnonymous",
    )
    assert 'response.optBoolean("email_verification_required", false)' in login
    assert "callback.onSuccess(toAuthResult(response))" in login

    registration = _between(
        auth,
        "static void register(\n            Context context,",
        "/**\n     * Persists a server-session revocation",
    )
    assert "AuthResult registerResult = toAuthResult(registerResponse)" in registration
    assert "if (registerResult.emailVerificationRequired)" in registration
    assert registration.index("callback.onSuccess(registerResult)") < registration.index(
        "JSONObject loginResponse"
    )

    auth_callback = _between(
        activity,
        "private AuthClient.Callback authCallback(boolean rememberExtraId)",
        "private boolean persistNativeAuthToken",
    )
    assert "if (result.emailVerificationRequired)" in auth_callback
    assert auth_callback.index("if (result.emailVerificationRequired)") < auth_callback.index(
        "resetWebViewForNewAuthSession()"
    )

    native_callback = _between(
        activity,
        "private AuthClient.Callback nativeExtraIdCallback(",
        "private boolean saveNativeAuthTokenForDialog",
    )
    assert "if (result.emailVerificationRequired)" in native_callback
    assert native_callback.index("if (result.emailVerificationRequired)") < native_callback.index(
        "saveNativeAuthTokenForDialog"
    )
    assert "Текущая игра и прогресс на этом устройстве сохранены." in activity
    assert '"/api/extraid/email/resend"' in auth
    assert '"/api/extraid/email/verify"' in auth
    assert "static void verifyEmailCode(" in auth
    assert 'body.put("code"' in auth
    assert '"Новый код"' in activity
    assert "Если ExtraID ожидает подтверждения, новый код отправлен" in activity
    assert "code.matches(\"\\\\d{6}\")" in activity
    assert "new InputFilter.LengthFilter(6)" in activity
    assert '"/api/extraid/email/change"' in auth
    assert "body,\n                        authToken" in auth
    assert '"Исправить email"' in activity
    assert "DeviceRegistrar.getAuthToken(MainActivity.this)" in activity
    assert "AuthClient.changeUnverifiedEmail(" in activity
    assert "Email изменён, новый код отправлен" in activity


def test_android_webview_bridge_is_not_exposed_to_third_party_frames():
    activity = _source("MainActivity.java")

    assert "settings.setAllowFileAccessFromFileURLs(false)" in activity
    assert "settings.setAllowUniversalAccessFromFileURLs(false)" in activity
    assert "+ \"frame-src 'self'; \"" in activity
    assert 'headers.put("Content-Security-Policy", WEBAPP_CONTENT_SECURITY_POLICY)' in activity
    assert 'headers.put("X-Content-Type-Options", "nosniff")' in activity
