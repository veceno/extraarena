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
    assert "static boolean isRevocationPendingForScope(" in auth
    assert "pending.credentialScope.equals(credentialScope)" in auth
    assert "!AuthClient.isRevocationPendingForScope(" in accounts
    assert "account.credentialScope" in accounts
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
    assert "Email изменён, готовим новый код" in activity
    assert "письмо придёт автоматически" in activity
    assert "showEmailVerificationPending(newEmail, false, true)" in activity
    assert "if (!cleanEmail.isEmpty() && !emailQueued)" in activity


def test_android_clean_install_can_choose_profile_before_auth_without_webview():
    activity = _source("MainActivity.java")

    welcome = _between(
        activity,
        "if (step == AuthStep.WELCOME)",
        "} else if (step == AuthStep.ANONYMOUS_NICKNAME)",
    )
    topbar = _between(
        activity,
        "private void handleTopbarAction()",
        "private void setFieldVisible",
    )
    load_arena = _between(
        activity,
        "private void loadArena(Intent intent)",
        "private String buildLaunchUrl(",
    )

    assert 'setTopbar("ExtraArena", "Сервер")' in welcome
    assert "if (authStep == AuthStep.WELCOME)" in topbar
    assert "showServerSwitcher();" in topbar
    assert "Данные входа отправляются выбранному серверу" in activity
    assert load_arena.index("DeviceRegistrar.getAuthToken(this).isEmpty()") < load_arena.index(
        "probeAndLoadArena"
    )
    assert "showAuth();" in load_arena
    assert "reloadAfterServerChange()" in activity
    assert "DeviceRegistrar.clearAuthToken(this)" in activity


def test_android_native_shell_respects_system_bar_insets_without_touching_webview():
    activity = _source("MainActivity.java")
    shell = _between(
        activity,
        "private FrameLayout createShellScreen()",
        "private View createTopbar(",
    )

    assert "applyNativeSystemBarInsets(screen);" in shell
    assert "screen.setOnApplyWindowInsetsListener" in shell
    assert "insets.getSystemWindowInsetTop()" in shell
    assert "insets.getSystemWindowInsetBottom()" in shell
    assert "view.setPadding(" in shell
    assert "webView.setPadding(" not in activity


def test_android_native_topbars_are_above_full_screen_content_for_real_taps():
    activity = _source("MainActivity.java")
    auth = _between(activity, "private View createAuthView()", "private TextView createLegalNotice")
    loading = _between(activity, "private View createLoadingView()", "private View createLoadingProgress")
    error = _between(activity, "private View createErrorView()", "private GradientDrawable makeBackground")

    for block in (auth, loading, error):
        assert "View topbar = createTopbar(" in block
        assert "screen.addView(topbar);" in block
        assert block.rindex("screen.addView(topbar);") > block.index("screen.addView(")
        assert block.rindex("screen.addView(topbar);") > block.index("ViewGroup.LayoutParams.MATCH_PARENT")


def test_android_first_run_extraid_uses_retry_safe_anonymous_bootstrap():
    auth = _source("AuthClient.java")
    activity = _source("MainActivity.java")

    bootstrap = _between(
        auth,
        "static void startAnonymousBootstrap",
        "/** Saves only the newest server generation",
    )
    accept = _between(
        auth,
        "static boolean saveAnonymousBootstrapAuth",
        "static void register(Context context",
    )
    registration = _between(
        activity,
        "private void registerExtraIdWithGameSession",
        "private AuthClient.Callback authCallback",
    )

    assert "SecureRandom" in auth
    assert "BOOTSTRAP_SECRET_BYTES = 32" in auth
    assert "Base64.URL_SAFE | Base64.NO_WRAP | Base64.NO_PADDING" in auth
    assert 'KEY_ANONYMOUS_BOOTSTRAP_PREFIX = "anonymous_bootstrap_v1_"' in auth
    assert 'MessageDigest.getInstance("SHA-256")' in auth
    assert 'body.put("bootstrap_id", bootstrap.id)' in bootstrap
    assert 'body.put("bootstrap_secret", bootstrap.secret)' in bootstrap
    assert 'body.put("nickname", bootstrap.nickname)' in bootstrap
    assert 'value.put("nickname", nickname)' in auth
    assert 'String.format(Locale.ROOT, "Arena%08d", suffix)' in auth
    assert 'response.optLong("bootstrap_generation", 0L)' in auth
    assert 'response.optBoolean("email_queued", false)' in auth
    assert "result.bootstrapGeneration <= 0L" in accept
    assert "result.bootstrapGeneration < acceptedGeneration" in accept
    assert accept.index("result.bootstrapGeneration < acceptedGeneration") < accept.index(
        "DeviceRegistrar.saveAuthToken"
    )
    assert "anonymousBootstrapStorageKey(baseUrl, whitelistCode)" in auth
    assert 'saved.optLong("user_id", 0L) == result.userId' in auth
    assert "SecurePrefs.remove(context, storageKey)" in auth

    assert "if (!activeToken.isEmpty())" in registration
    assert "AuthClient.startAnonymousBootstrap" in registration
    assert "new AuthClient.BootstrapCallback()" in registration
    assert "AuthClient.saveAnonymousBootstrapAuth" in registration
    bootstrap_callback = registration.split("AuthClient.startAnonymousBootstrap", 1)[1]
    assert bootstrap_callback.index("AuthClient.saveAnonymousBootstrapAuth") < bootstrap_callback.index(
        "AuthClient.register"
    )
    assert "прогресс не задублируется" in registration
    assert '"bootstrap_upgraded".equals(errorCode)' in registration
    assert "updateAuthStep(AuthStep.LOGIN_EMAIL)" in registration
    assert "выбери сохранённый аккаунт" in registration

    anonymous = _between(
        activity,
        "private void submitAnonymousProfile()",
        "private void submitTelegramId()",
    )
    assert "AuthClient.startAnonymousBootstrap(" in anonymous
    assert "AuthClient.saveAnonymousBootstrapAuth" in anonymous
    assert "AuthClient.startAnonymous(this" not in anonymous
    assert "профиль не задублируется" in anonymous


def test_android_preserves_json_domain_errors_from_http_5xx():
    auth = _source("AuthClient.java")
    post = _between(
        auth,
        "private static JSONObject post(\n            Context context,\n            String baseUrl",
        "private static void flushPendingRevocationsNow",
    )

    assert "JSONObject response = new JSONObject(text);" in post
    assert 'response.optString("error", "").trim().isEmpty()' in post
    assert 'response.optString("message", "").trim().isEmpty()' in post
    parsed_index = post.index("JSONObject response = new JSONObject(text);")
    assert post.index("if (status >= 500", parsed_index) > parsed_index
    assert post.index('errorResponse("server_unavailable", status)', parsed_index) > parsed_index
    assert '"email_delivery_failed".equals(error)' in auth
    assert 'response.optString(\n                "message"' in auth
    assert '"email_delivery_unavailable".equals(error)' in auth
    assert "Отправка писем временно недоступна" in auth


def test_android_canonicalizes_backend_identity_scope_and_queued_email_state():
    base_url = _source("BaseUrlStore.java")
    auth = _source("AuthClient.java")
    activity = _source("MainActivity.java")
    accounts = _source("ExtraIdAccountStore.java")

    assert "new URI(value)" in base_url
    assert "scheme.toLowerCase(Locale.ROOT)" in base_url
    assert "host.toLowerCase(Locale.ROOT)" in base_url
    assert '"https".equals(scheme) && port == 443' in base_url
    assert '"http".equals(scheme) && port == 80' in base_url
    assert "normalizePathSafely(path)" in base_url
    assert 'rawPath.split("/", -1)' in base_url
    assert "parsed.getRawPath()" in base_url
    assert "BaseUrlStore.normalize(baseUrl)" in accounts
    assert "static String identityScope(String baseUrl, String whitelistCode)" in base_url
    assert "normalizedWhitelist.isEmpty()" in base_url
    assert 'OFFICIAL_IDENTITY_SCOPE = "extraarena_official_v1"' in base_url
    assert 'item.put("credential_scope", account.credentialScope)' in accounts
    assert "account.credentialScope.equals(selectedScope)" in accounts
    assert "anonymousBootstrapStorageKey(baseUrl, whitelistCode)" in auth

    assert "final boolean emailQueued;" in auth
    assert 'response.optBoolean("email_queued", false)' in auth
    assert "result.emailQueued" in activity
    assert "showEmailVerificationPending(newEmail, false, true)" in activity


def test_android_profile_switches_cannot_retarget_inflight_auth_or_push_work():
    activity = _source("MainActivity.java")
    auth = _source("AuthClient.java")
    registrar = _source("DeviceRegistrar.java")

    assert "final class WebAuthContext" in activity
    assert "authContextGeneration" in activity
    assert "isWebAuthContextCurrent(expectedAuthContext)" in activity
    assert 'Ignoring stale JavaScript auth token after profile/auth switch' in activity
    assert "DeviceRegistrar.syncProfileTransition(" in activity
    assert activity.index("DeviceRegistrar.syncProfileTransition(") < activity.index(
        "ConnectionProfileStore.selectProfile(this, profile.id)"
    )

    assert "private static final class BackendTarget" in auth
    assert "BackendTarget backend = selectedBackend(appContext)" in auth
    assert "backend.baseUrl" in auth
    assert "backend.whitelistCode" in auth
    assert "final class BackendBinding" in registrar
    assert "final String endpoint = backend.endpoint(path)" in registrar
    assert "BaseUrlStore.identityScope(baseUrl, whitelistCode)" in registrar


def test_android_webview_bridge_is_not_exposed_to_third_party_frames():
    activity = _source("MainActivity.java")

    assert "settings.setAllowFileAccessFromFileURLs(false)" in activity
    assert "settings.setAllowUniversalAccessFromFileURLs(false)" in activity
    assert "+ \"frame-src 'self'; \"" in activity
    assert 'headers.put("Content-Security-Policy", WEBAPP_CONTENT_SECURITY_POLICY)' in activity
    assert 'headers.put("X-Content-Type-Options", "nosniff")' in activity
