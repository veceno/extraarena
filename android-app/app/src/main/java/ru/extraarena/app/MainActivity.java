package ru.extraarena.app;

import android.Manifest;
import android.app.Activity;
import android.app.Dialog;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Message;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.text.InputFilter;
import android.text.InputType;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.TextPaint;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.util.Log;
import android.util.Patterns;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.webkit.JavascriptInterface;
import android.webkit.CookieManager;
import android.webkit.ConsoleMessage;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebStorage;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.messaging.FirebaseMessaging;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final String TAG_WEBVIEW = "EAWebView";
    private static final String TAG = "ExtraArenaApp";
    private static final int SECRET_TAP_WINDOW_MS = 1800;
    private static final int SECRET_TAP_TARGET_DP = 72;
    private static final int EA_BG = Color.rgb(9, 5, 18);
    private static final int EA_TEXT = Color.rgb(240, 236, 255);
    private static final int EA_MUTED = Color.rgb(196, 184, 232);
    private static final int EA_SOFT = Color.rgb(122, 111, 160);
    private static final int EA_ACCENT = Color.rgb(245, 146, 30);
    private static final int EA_PINK = Color.rgb(244, 114, 182);
    private static final int EA_SURFACE = Color.rgb(26, 16, 48);
    private static final long UPDATE_GATE_CACHE_MS = 10 * 60 * 1000L;
    private static final long NOTIFICATION_PROMPT_COOLDOWN_MS = 3L * 24L * 60L * 60L * 1000L;
    private static final String NOTIFICATION_PROMPT_PREFS = "extraarena_notification_prompt";
    private static final String KEY_NOTIFICATION_PROMPT_ACCEPTED = "accepted";
    private static final String KEY_NOTIFICATION_PROMPT_LAST_SHOWN = "last_shown_at";
    private static final String HAPTICS_PREFS = "extraarena_haptics";
    private static final String KEY_HAPTICS_ENABLED = "enabled";

    private enum AuthStep {
        WELCOME,
        ANONYMOUS_NICKNAME,
        LOGIN_EMAIL,
        LOGIN_PASSWORD,
        REGISTER_EMAIL,
        REGISTER_PASSWORD,
        TELEGRAM_ID,
        TELEGRAM_CODE,
        TELEGRAM_EMAIL,
        TELEGRAM_PASSWORD
    }

    private FrameLayout root;
    private WebView webView;
    private View loadingView;
    private View loadingDevHotspot;
    private View errorView;
    private View authView;
    private EditText authEmail;
    private EditText authPassword;
    private EditText authNickname;
    private EditText authTelegramId;
    private EditText authTelegramCode;
    private TextView authTitle;
    private TextView authSubtitle;
    private TextView authKicker;
    private TextView topbarLabel;
    private TextView topbarAction;
    private TextView authAction;
    private TextView authSecondaryAction;
    private TextView authTelegramAction;
    private TextView authModeSwitch;
    private TextView authLegalNotice;
    private TextView authError;
    private TextView authHint;
    private TextView accountSectionTitle;
    private FrameLayout authStage;
    private LinearLayout authEmailField;
    private LinearLayout authPasswordField;
    private LinearLayout authNicknameField;
    private LinearLayout authTelegramIdField;
    private LinearLayout authTelegramCodeField;
    private LinearLayout savedAccountsList;
    private LinearLayout loginSteps;
    private AuthStep authStep = AuthStep.WELCOME;
    private String stagedEmail = "";
    private String stagedTelegramId = "";
    private String stagedTelegramCode = "";
    private Typeface futuraMedium;
    private Typeface futuraBold;
    private Typeface futuraExtraBold;
    private int secretTapCount = 0;
    private long firstSecretTapAt = 0L;
    private boolean loadingPausedForProfileSwitcher = false;
    private boolean arenaLoadBlockedByConnectivity = false;
    private int arenaLoadGeneration = 0;
    private int updateGateGeneration = 0;
    private long updateGatePassedAt = 0L;
    private boolean updateBlocked = false;
    private Dialog updateDialog;
    private Dialog notificationPromptDialog;
    private Dialog extraIdManagerDialog;
    private Dialog addExtraIdDialog;
    private Dialog badConnectionDialog;
    private Dialog nativeAfkDialog;
    private MobileUpdateInfo blockedUpdateInfo;
    private String lastFinishedUrl = "";
    private RuStoreIntegration ruStoreIntegration;
    private boolean rustoreOptionalUpdateChecked = false;
    private final ExecutorService arenaProbeExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        configureWindowColors();
        NotificationChannels.ensure(this);
        loadTypefaces();
        buildUi();
        configureWebView();
        ruStoreIntegration = RuStoreIntegrationFactory.create(this);
        ruStoreIntegration.onCreate(savedInstanceState, getIntent());
        fetchFcmToken();
        handlePushIntent(getIntent());
        launchAfterUpdateGate(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (ruStoreIntegration != null) {
            ruStoreIntegration.onNewIntent(intent);
        }
        handlePushIntent(intent);
        launchAfterUpdateGate(intent);
    }

    @Override
    public void onConfigurationChanged(android.content.res.Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        destroyWebContent();
        configureWindowColors();
        buildUi();
        configureWebView();
        launchAfterUpdateGate(getIntent());
    }

    @Override
    public void onBackPressed() {
        if (authView != null && authView.getVisibility() == View.VISIBLE && authStep != AuthStep.WELCOME) {
            updateAuthStep(AuthStep.WELCOME);
            vibrate("selection");
            return;
        }
        if (webView != null && webView.getVisibility() == View.VISIBLE) {
            handleWebBack();
            return;
        }
        super.onBackPressed();
    }

    @Override
    protected void onPause() {
        pauseWebContent();
        super.onPause();
    }

    @Override
    protected void onStop() {
        pauseWebContent();
        super.onStop();
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumeWebContent();
        if (updateBlocked) {
            showRequiredUpdateDialog(blockedUpdateInfo);
        }
    }

    @Override
    protected void onDestroy() {
        arenaProbeExecutor.shutdownNow();
        destroyWebContent();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != 7001) {
            return;
        }
        boolean granted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
        SharedPreferences prefs = getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE);
        prefs.edit()
                .putBoolean(KEY_NOTIFICATION_PROMPT_ACCEPTED, granted)
                .putLong(KEY_NOTIFICATION_PROMPT_LAST_SHOWN, System.currentTimeMillis())
                .apply();
        if (granted) {
            fetchFcmToken();
            DeviceRegistrar.registerIfReady(this);
            Toast.makeText(this, "Уведомления включены", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Ок, напомним позже", Toast.LENGTH_SHORT).show();
        }
    }

    private void pauseWebContent() {
        if (webView == null) {
            return;
        }
        pauseWebMedia(false);
        try {
            webView.onPause();
        } catch (Exception ignored) {
        }
        try {
            webView.pauseTimers();
        } catch (Exception ignored) {
        }
    }

    private void resumeWebContent() {
        if (webView == null) {
            return;
        }
        try {
            webView.resumeTimers();
        } catch (Exception ignored) {
        }
        try {
            webView.onResume();
        } catch (Exception ignored) {
        }
        try {
            webView.evaluateJavascript(
                    "try{if(window.ExtraArenaAppResume){window.ExtraArenaAppResume();}}catch(e){}",
                    null
            );
        } catch (Exception ignored) {
        }
        webView.postDelayed(() -> {
            try {
                if (webView != null) {
                    webView.evaluateJavascript(
                            "try{if(window.ExtraArenaAppResume){window.ExtraArenaAppResume();}}catch(e){}",
                            null
                    );
                }
            } catch (Exception ignored) {
            }
        }, 220);
    }

    private void destroyWebContent() {
        if (webView == null) {
            return;
        }
        pauseWebMedia(true);
        try {
            webView.stopLoading();
        } catch (Exception ignored) {
        }
        try {
            webView.loadUrl("about:blank");
        } catch (Exception ignored) {
        }
        try {
            if (webView.getParent() instanceof ViewGroup) {
                ((ViewGroup) webView.getParent()).removeView(webView);
            }
        } catch (Exception ignored) {
        }
        try {
            webView.destroy();
        } catch (Exception ignored) {
        }
        webView = null;
    }

    private void pauseWebMedia(boolean resetPosition) {
        if (webView == null) {
            return;
        }
        String reset = resetPosition ? "true" : "false";
        try {
            webView.evaluateJavascript(
                    "try{"
                            + "if(" + reset + "&&window.stopBgMusic){window.stopBgMusic();}"
                            + "if(" + reset + "&&window.stopArenaMusic){window.stopArenaMusic();}"
                            + "if(!" + reset + "&&window.ExtraArenaAppPause){window.ExtraArenaAppPause();}"
                            + "document.querySelectorAll('audio,video').forEach(function(media){"
                            + "try{media.pause();if(" + reset + ")media.currentTime=0;}catch(e){}"
                            + "});"
                            + "}catch(e){}",
                    null
            );
        } catch (Exception ignored) {
        }
    }

    private void configureWindowColors() {
        Window window = getWindow();
        if (window == null) {
            return;
        }
        window.setStatusBarColor(EA_BG);
        window.setNavigationBarColor(EA_BG);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.setNavigationBarDividerColor(EA_BG);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            int flags = window.getDecorView().getSystemUiVisibility();
            flags &= ~View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                flags &= ~View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
            }
            window.getDecorView().setSystemUiVisibility(flags);
        }
    }

    private void buildUi() {
        configureWindowColors();
        root = new FrameLayout(this);
        root.setBackgroundColor(EA_BG);

        webView = new WebView(this);
        webView.setVisibility(View.INVISIBLE);
        root.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        loadingView = createLoadingView();
        root.addView(loadingView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        errorView = createErrorView();
        errorView.setVisibility(View.GONE);
        root.addView(errorView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        authView = createAuthView();
        root.addView(authView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        loadingDevHotspot = new View(this);
        loadingDevHotspot.setBackgroundColor(Color.TRANSPARENT);
        loadingDevHotspot.setVisibility(View.GONE);
        loadingDevHotspot.setOnClickListener(v -> registerSecretTap());
        FrameLayout.LayoutParams hotspotParams = new FrameLayout.LayoutParams(
                dp(SECRET_TAP_TARGET_DP),
                dp(SECRET_TAP_TARGET_DP),
                Gravity.TOP | Gravity.LEFT
        );
        root.addView(loadingDevHotspot, hotspotParams);

        setContentView(root);
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setLoadsImagesAutomatically(true);
        settings.setBlockNetworkImage(false);
        settings.setLoadWithOverviewMode(false);
        settings.setUseWideViewPort(true);
        settings.setSupportZoom(false);
        settings.setTextZoom(100);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setSupportMultipleWindows(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            settings.setOffscreenPreRaster(true);
        }
        settings.setUserAgentString(settings.getUserAgentString() + " ExtraArenaApp/" + BuildConfig.VERSION_NAME);

        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        webView.addJavascriptInterface(new AndroidBridge(), "ExtraArenaApp");
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage consoleMessage) {
                if (consoleMessage == null) {
                    return false;
                }
                String message = consoleMessage.message()
                        + " @ "
                        + consoleMessage.sourceId()
                        + ":"
                        + consoleMessage.lineNumber();
                if (consoleMessage.messageLevel() == ConsoleMessage.MessageLevel.ERROR) {
                    Log.e(TAG_WEBVIEW, message);
                } else if (consoleMessage.messageLevel() == ConsoleMessage.MessageLevel.WARNING) {
                    Log.w(TAG_WEBVIEW, message);
                } else {
                    Log.d(TAG_WEBVIEW, message);
                }
                return true;
            }

            @Override
            public boolean onCreateWindow(WebView view, boolean isDialog, boolean isUserGesture, Message resultMsg) {
                String hintedUrl = null;
                try {
                    WebView.HitTestResult result = view == null ? null : view.getHitTestResult();
                    hintedUrl = result == null ? null : result.getExtra();
                } catch (Exception ignored) {
                }
                if (hintedUrl != null && !hintedUrl.trim().isEmpty()) {
                    openExternal(hintedUrl);
                    return false;
                }

                WebView popup = new WebView(MainActivity.this);
                popup.setWebViewClient(new WebViewClient() {
                    @Override
                    public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && request != null && request.getUrl() != null) {
                            openExternal(request.getUrl().toString());
                        }
                        popup.destroy();
                        return true;
                    }

                    @Override
                    public boolean shouldOverrideUrlLoading(WebView view, String url) {
                        openExternal(url);
                        popup.destroy();
                        return true;
                    }
                });
                WebView.WebViewTransport transport = (WebView.WebViewTransport) resultMsg.obj;
                transport.setWebView(popup);
                resultMsg.sendToTarget();
                return true;
            }
        });
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && request != null && request.getUrl() != null) {
                    return handleNavigation(request.getUrl().toString());
                }
                return false;
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                return handleNavigation(url);
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && request != null) {
                    return interceptExtraArenaRequest(request.getUrl());
                }
                return null;
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                return interceptExtraArenaRequest(Uri.parse(url));
            }

            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                if (!loadingPausedForProfileSwitcher && url != null && !"about:blank".equals(url)) {
                    arenaLoadBlockedByConnectivity = false;
                }
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                if (loadingPausedForProfileSwitcher) {
                    webView.setVisibility(View.INVISIBLE);
                    return;
                }
                if (arenaLoadBlockedByConnectivity) {
                    webView.setVisibility(View.INVISIBLE);
                    return;
                }
                webView.setVisibility(View.VISIBLE);
                loadingView.setVisibility(View.GONE);
                setLoadingDevHotspotVisible(false);
                errorView.setVisibility(View.GONE);
                injectAppContext();
                injectHapticsBridge();
                captureAuthTokenFromLocalStorage();
                maybeShowNotificationOptInPrompt();
                String previousUrl = lastFinishedUrl;
                lastFinishedUrl = url == null ? "" : url;
                if (isAppHomeUrl(lastFinishedUrl) && isArenaUrl(previousUrl)) {
                    webView.clearHistory();
                }
            }

            @Override
            public void onReceivedError(WebView view, int errorCode, String description, String failingUrl) {
                if (loadingPausedForProfileSwitcher) {
                    return;
                }
                if (failingUrl != null && failingUrl.equals(view.getUrl())) {
                    showConnectivityError();
                }
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (loadingPausedForProfileSwitcher) {
                    return;
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && request != null && request.isForMainFrame()) {
                    showConnectivityError();
                }
            }

            @Override
            public void onReceivedHttpError(WebView view, WebResourceRequest request, WebResourceResponse errorResponse) {
                if (loadingPausedForProfileSwitcher) {
                    return;
                }
                if (request == null || errorResponse == null) {
                    return;
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && request.isForMainFrame()) {
                    int status = errorResponse.getStatusCode();
                    if (status == 503 || status >= 500) {
                        showConnectivityError();
                    }
                }
            }
        });
    }

    private WebResourceResponse interceptExtraArenaRequest(Uri uri) {
        if (uri == null) {
            return null;
        }

        String host = uri.getHost() == null ? "" : uri.getHost().toLowerCase();
        String path = uri.getPath() == null || uri.getPath().isEmpty() ? "/" : uri.getPath();

        if ("telegram.org".equals(host) && "/js/telegram-web-app.js".equals(path)) {
            return serveTelegramStub();
        }
        if ("fonts.googleapis.com".equals(host)) {
            return serveLocalFontsCss();
        }
        if ("unpkg.com".equals(host)) {
            return servePackagedVendorAsset(vendorAssetForUnpkgPath(path));
        }
        if ("cdn.socket.io".equals(host) && path.endsWith("/socket.io.min.js")) {
            return servePackagedVendorAsset("ea_vendor/socket.io.min.js");
        }

        if (!isSelectedConnectionProfileUri(uri)) {
            return null;
        }

        if ("/api/cards/image".equals(path)) {
            return servePackagedCardImage(uri);
        }
        if (path.startsWith("/api/")) {
            return null;
        }
        if (path.startsWith("/DesignAssets/")) {
            return servePackagedDesignAsset(path);
        }
        return servePackagedWebappAsset(webappAssetForPath(path));
    }

    private boolean isSelectedConnectionProfileUri(Uri uri) {
        Uri base = Uri.parse(BaseUrlStore.getBaseUrl(this));
        String uriHost = uri.getHost();
        String baseHost = base.getHost();
        if (uriHost == null || baseHost == null || !uriHost.equalsIgnoreCase(baseHost)) {
            return false;
        }
        String uriScheme = uri.getScheme() == null ? "https" : uri.getScheme();
        String baseScheme = base.getScheme() == null ? "https" : base.getScheme();
        if (!uriScheme.equalsIgnoreCase(baseScheme)) {
            return false;
        }
        return normalizedPort(uri) == normalizedPort(base);
    }

    private int normalizedPort(Uri uri) {
        int port = uri.getPort();
        if (port > 0) {
            return port;
        }
        return "http".equalsIgnoreCase(uri.getScheme()) ? 80 : 443;
    }

    private String webappAssetForPath(String path) {
        String clean = path == null ? "/" : path.trim();
        if (clean.isEmpty() || "/".equals(clean)) {
            return "ea_webapp/index.html";
        }
        if (clean.startsWith("/")) {
            clean = clean.substring(1);
        }
        if ("arena".equals(clean)) {
            clean = "arena.html";
        }
        if ("index.html".equals(clean)
                || "arena.html".equals(clean)
                || "arena.js".equals(clean)
                || "arena-styles.css".equals(clean)
                || "main.js".equals(clean)
                || "styles.css".equals(clean)
                || "safe-area.js".equals(clean)
                || "matchmaking-tips.config.js".equals(clean)
                || "index.compiled.js".equals(clean)) {
            return "ea_webapp/" + clean;
        }
        return null;
    }

    private WebResourceResponse servePackagedWebappAsset(String assetPath) {
        return servePackagedAsset(assetPath);
    }

    private WebResourceResponse servePackagedDesignAsset(String path) {
        String clean = path == null ? "" : path.replaceFirst("^/+", "");
        return servePackagedAsset(clean);
    }

    private WebResourceResponse servePackagedCardImage(Uri uri) {
        String cardId = uri.getQueryParameter("card_id");
        if (cardId == null || !cardId.matches("\\d+")) {
            return null;
        }
        WebResourceResponse response = servePackagedAsset("DesignAssets/Cards/" + cardId + ".png");
        if (response != null) return response;
        response = servePackagedAsset("DesignAssets/Cards/" + cardId + ".jpg");
        if (response != null) return response;
        response = servePackagedAsset("DesignAssets/Cards/" + cardId + ".jpeg");
        if (response != null) return response;
        response = servePackagedAsset("DesignAssets/Cards/" + cardId + ".webp");
        if (response != null) return response;
        return null;
    }

    private WebResourceResponse servePackagedVendorAsset(String assetPath) {
        return servePackagedAsset(assetPath, corsHeaders());
    }

    private String vendorAssetForUnpkgPath(String path) {
        String clean = path == null ? "" : path;
        if (clean.contains("react-dom@")) {
            return "ea_vendor/react-dom.production.min.js";
        }
        if (clean.contains("react@") && clean.endsWith("react.production.min.js")) {
            return "ea_vendor/react.production.min.js";
        }
        if (clean.contains("@babel/standalone@") && clean.endsWith("babel.min.js")) {
            return "ea_vendor/babel.min.js";
        }
        if (clean.contains("dompurify@") && clean.endsWith("purify.min.js")) {
            return "ea_vendor/purify.min.js";
        }
        return null;
    }

    private WebResourceResponse servePackagedAsset(String assetPath) {
        return servePackagedAsset(assetPath, null);
    }

    private WebResourceResponse servePackagedAsset(String assetPath, Map<String, String> headers) {
        if (assetPath == null || assetPath.trim().isEmpty() || assetPath.contains("..")) {
            return null;
        }
        try {
            InputStream stream = getAssets().open(assetPath);
            if (headers != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                return new WebResourceResponse(
                        mimeTypeForAsset(assetPath),
                        encodingForAsset(assetPath),
                        200,
                        "OK",
                        headers,
                        stream
                );
            }
            return new WebResourceResponse(mimeTypeForAsset(assetPath), encodingForAsset(assetPath), stream);
        } catch (Exception ignored) {
            return null;
        }
    }

    private Map<String, String> corsHeaders() {
        Map<String, String> headers = new HashMap<>();
        headers.put("Access-Control-Allow-Origin", "*");
        headers.put("Access-Control-Allow-Methods", "GET, OPTIONS");
        headers.put("Access-Control-Allow-Headers", "*");
        headers.put("Cache-Control", "public, max-age=31536000, immutable");
        return headers;
    }

    private WebResourceResponse serveTelegramStub() {
        String stub = ""
                + "(function(){"
                + "var tg=window.Telegram=window.Telegram||{};"
                + "tg.WebApp=tg.WebApp||{"
                + "initData:'',initDataUnsafe:null,platform:'android',version:'android-app',"
                + "ready:function(){},expand:function(){},onEvent:function(){},offEvent:function(){},"
                + "setHeaderColor:function(){},setBackgroundColor:function(){},setBottomBarColor:function(){},"
                + "openLink:function(url){window.open(url,'_blank');},"
                + "openTelegramLink:function(url){window.open(url,'_blank');},"
                + "HapticFeedback:{impactOccurred:function(){},notificationOccurred:function(){},selectionChanged:function(){}}"
                + "};"
                + "})();";
        return stringResponse("application/javascript", stub);
    }

    private WebResourceResponse serveLocalFontsCss() {
        String medium = BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), "DesignAssets/Font/FuturaPT-Medium.ttf");
        String demi = BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), "DesignAssets/Font/FuturaPT-Demi.ttf");
        String bold = BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), "DesignAssets/Font/FuturaPT-Bold.ttf");
        String extraBold = BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), "DesignAssets/Font/FuturaPT-ExtraBold.ttf");
        String css = ""
                + "@font-face{font-family:'Inter';src:url('" + medium + "') format('truetype');font-weight:400;font-style:normal;font-display:swap;}"
                + "@font-face{font-family:'Inter';src:url('" + demi + "') format('truetype');font-weight:600;font-style:normal;font-display:swap;}"
                + "@font-face{font-family:'Exo 2';src:url('" + bold + "') format('truetype');font-weight:700;font-style:normal;font-display:swap;}"
                + "@font-face{font-family:'Exo 2';src:url('" + extraBold + "') format('truetype');font-weight:900;font-style:normal;font-display:swap;}";
        return stringResponse("text/css", css);
    }

    private WebResourceResponse stringResponse(String mimeType, String body) {
        byte[] payload = body.getBytes(StandardCharsets.UTF_8);
        return new WebResourceResponse(mimeType, "UTF-8", new ByteArrayInputStream(payload));
    }

    private String encodingForAsset(String assetPath) {
        String mimeType = mimeTypeForAsset(assetPath);
        return mimeType.startsWith("text/") || mimeType.contains("javascript") || mimeType.contains("json")
                ? "UTF-8"
                : null;
    }

    private String mimeTypeForAsset(String assetPath) {
        String lower = assetPath.toLowerCase();
        if (lower.endsWith(".html")) return "text/html";
        if (lower.endsWith(".js")) return "application/javascript";
        if (lower.endsWith(".css")) return "text/css";
        if (lower.endsWith(".json")) return "application/json";
        if (lower.endsWith(".png")) return "image/png";
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
        if (lower.endsWith(".webp")) return "image/webp";
        if (lower.endsWith(".gif")) return "image/gif";
        if (lower.endsWith(".svg")) return "image/svg+xml";
        if (lower.endsWith(".mp3")) return "audio/mpeg";
        if (lower.endsWith(".wav")) return "audio/wav";
        if (lower.endsWith(".ttf")) return "font/ttf";
        if (lower.endsWith(".otf")) return "font/otf";
        return "application/octet-stream";
    }

    private boolean handleNavigation(String url) {
        if (url == null) {
            return false;
        }
        String baseUrl = BaseUrlStore.getBaseUrl(this);
        if (url.startsWith(baseUrl)) {
            return false;
        }
        if (url.startsWith("extraarena://")) {
            Uri uri = Uri.parse(url);
            String section = uri.getQueryParameter("section");
            webView.loadUrl(buildLaunchUrl(section));
            return true;
        }
        openExternal(url);
        return true;
    }

    private void handleWebBack() {
        if (webView == null) {
            super.onBackPressed();
            return;
        }
        try {
            webView.evaluateJavascript(
                    "(function(){try{return !!(window.ExtraArenaAppBack&&window.ExtraArenaAppBack());}catch(e){return false;}})()",
                    value -> runOnUiThread(() -> {
                        if ("true".equals(stripJsString(value))) {
                            vibrate("selection");
                            return;
                        }
                        handleFallbackWebBack();
                    })
            );
        } catch (Exception ignored) {
            handleFallbackWebBack();
        }
    }

    private void handleFallbackWebBack() {
        if (webView == null) {
            return;
        }
        String url = webView.getUrl();
        if (isArenaUrl(url) || isAppHomeUrl(url)) {
            moveTaskToBack(true);
            return;
        }
        if (webView.canGoBack()) {
            webView.goBack();
            return;
        }
        moveTaskToBack(true);
    }

    private boolean isAppHomeUrl(String url) {
        if (url == null || url.trim().isEmpty() || "about:blank".equals(url)) {
            return false;
        }
        try {
            Uri uri = Uri.parse(url);
            Uri base = Uri.parse(BaseUrlStore.getBaseUrl(this));
            if (uri.getHost() == null || base.getHost() == null || !uri.getHost().equalsIgnoreCase(base.getHost())) {
                return false;
            }
            String path = uri.getPath();
            return path == null || path.isEmpty() || "/".equals(path) || "/index.html".equals(path);
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isArenaUrl(String url) {
        if (url == null || url.trim().isEmpty()) {
            return false;
        }
        try {
            String path = Uri.parse(url).getPath();
            return path != null && ("/arena".equals(path) || path.startsWith("/arena/"));
        } catch (Exception ignored) {
            return false;
        }
    }

    private void injectHapticsBridge() {
        if (webView == null) {
            return;
        }
        String script = "(function(){"
                + "if(window.__eaHapticsInstalled)return;"
                + "window.__eaHapticsInstalled=true;"
                + "function hapticsEnabled(){try{if(localStorage.getItem('extra_haptics_enabled')==='false')return false;}catch(e){}try{if(window.ExtraArenaApp&&window.ExtraArenaApp.isHapticsEnabled&&window.ExtraArenaApp.isHapticsEnabled()===false)return false;}catch(e){}return true;}"
                + "function setHapticsEnabled(enabled){try{localStorage.setItem('extra_haptics_enabled',enabled?'true':'false');}catch(e){}try{if(window.ExtraArenaApp&&window.ExtraArenaApp.setHapticsEnabled)window.ExtraArenaApp.setHapticsEnabled(!!enabled);}catch(e){}}"
                + "function nativeHaptic(style){if(!hapticsEnabled())return true;try{if(window.ExtraArenaApp&&window.ExtraArenaApp.haptic){window.ExtraArenaApp.haptic(style||'light');return true;}}catch(e){}return false;}"
                + "window.ExtraArenaHaptics={isEnabled:hapticsEnabled,setEnabled:setHapticsEnabled,impact:function(style){if(!hapticsEnabled())return;if(nativeHaptic(style))return;try{var h=window.Telegram&&window.Telegram.WebApp&&window.Telegram.WebApp.HapticFeedback;if(!h)return;if(style==='success'||style==='warning'||style==='error')h.notificationOccurred(style);else h.impactOccurred(style||'light');}catch(e){}}};"
                + "document.addEventListener('pointerdown',function(e){"
                + "var t=e.target;if(!t||!t.closest)return;"
                + "var el=t.closest('[data-haptic],button,[role=\"button\"],a,input,select,textarea');"
                + "if(!el||el.disabled||el.getAttribute('aria-disabled')==='true')return;"
                + "var style=el.getAttribute('data-haptic')||(el.tagName==='INPUT'||el.tagName==='TEXTAREA'||el.tagName==='SELECT'?'selection':'light');"
                + "window.ExtraArenaHaptics.impact(style);"
                + "},{capture:true,passive:true});"
                + "})();";
        try {
            webView.evaluateJavascript(script, null);
        } catch (Exception ignored) {
        }
    }

    private void handlePushIntent(Intent intent) {
        if (intent == null) {
            return;
        }
        String type = intent.getStringExtra("type");
        if ("app_update_required".equals(type) || "app_update".equals(type)) {
            String url = intent.getStringExtra("url");
            if (url == null || url.trim().isEmpty()) {
                url = isRuStoreBuild() ? BuildConfig.RUSTORE_APP_URL : BuildConfig.UPDATE_CHANNEL_URL;
            }
            openExternal(url);
        }
    }

    private void selectInitialProfileIfNeeded() {
        RegionDetector.Result region = RegionDetector.isLikelyRu(this);
        ConnectionProfileStore.autoSelectIfNeeded(this, region.ru);
        ConnectionProfileStore.ConnectionProfile selected = ConnectionProfileStore.getSelectedProfile(this);
        Log.d(TAG, "Connection profile: " + selected.id + " (" + selected.baseUrl + ") region=" + region.source);
    }

    private void launchAfterUpdateGate(Intent intent) {
        selectInitialProfileIfNeeded();
        if (updateBlocked) {
            showRequiredUpdateDialog(blockedUpdateInfo);
            return;
        }

        if (isRuStoreBuild() && !rustoreOptionalUpdateChecked && ruStoreIntegration != null) {
            rustoreOptionalUpdateChecked = true;
            showUpdateGateLoading();
            ruStoreIntegration.checkOptionalUpdate(() -> launchAfterUpdateGate(intent));
            return;
        }

        long now = System.currentTimeMillis();
        if (updateGatePassedAt > 0L && now - updateGatePassedAt < UPDATE_GATE_CACHE_MS) {
            loadArena(intent);
            return;
        }

        showUpdateGateLoading();
        final int generation = ++updateGateGeneration;
        arenaProbeExecutor.execute(() -> {
            MobileUpdateInfo info = fetchMobileUpdateInfo();
            runOnUiThread(() -> {
                if (generation != updateGateGeneration) {
                    return;
                }
                if (info.required) {
                    updateBlocked = true;
                    blockedUpdateInfo = info;
                    if (isRuStoreBuild() && ruStoreIntegration != null) {
                        ruStoreIntegration.startImmediateUpdate(() -> showRequiredUpdateDialog(info));
                    } else {
                        showRequiredUpdateDialog(info);
                    }
                    return;
                }
                updateGatePassedAt = System.currentTimeMillis();
                loadArena(intent);
            });
        });
    }

    private void showUpdateGateLoading() {
        loadingPausedForProfileSwitcher = false;
        arenaLoadBlockedByConnectivity = false;
        if (webView != null) {
            try {
                webView.stopLoading();
            } catch (Exception ignored) {
            }
            webView.setVisibility(View.INVISIBLE);
        }
        if (authView != null) {
            authView.setVisibility(View.GONE);
        }
        if (errorView != null) {
            errorView.setVisibility(View.GONE);
        }
        if (loadingView != null) {
            loadingView.setVisibility(View.VISIBLE);
        }
        setLoadingDevHotspotVisible(false);
    }

    private MobileUpdateInfo fetchMobileUpdateInfo() {
        HttpURLConnection connection = null;
        try {
            Uri uri = Uri.parse(BaseUrlStore.join(BuildConfig.DEFAULT_BASE_URL, BuildConfig.APP_VERSION_PATH))
                    .buildUpon()
                    .appendQueryParameter("platform", "android")
                    .appendQueryParameter("version_code", String.valueOf(BuildConfig.VERSION_CODE))
                    .appendQueryParameter("version_name", BuildConfig.VERSION_NAME)
                    .build();
            URL url = new URL(uri.toString());
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);
            connection.setInstanceFollowRedirects(true);
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("Cache-Control", "no-cache");
            connection.setRequestProperty("User-Agent", "ExtraArenaApp/" + BuildConfig.VERSION_NAME);
            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                return MobileUpdateInfo.notRequired();
            }
            String raw = readStream(connection.getInputStream());
            JSONObject data = new JSONObject(raw);
            return MobileUpdateInfo.from(data);
        } catch (Exception ignored) {
            return MobileUpdateInfo.notRequired();
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private String readStream(InputStream input) throws Exception {
        if (input == null) {
            return "";
        }
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[2048];
        int read;
        while ((read = input.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return output.toString(StandardCharsets.UTF_8.name());
    }

    private void showRequiredUpdateDialog(MobileUpdateInfo info) {
        MobileUpdateInfo updateInfo = info == null ? MobileUpdateInfo.requiredFallback() : info;
        blockedUpdateInfo = updateInfo;
        showUpdateGateLoading();

        if (updateDialog != null && updateDialog.isShowing()) {
            return;
        }

        Dialog dialog = new Dialog(this);
        updateDialog = dialog;
        dialog.setCancelable(false);
        dialog.setCanceledOnTouchOutside(false);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.setBackgroundColor(Color.TRANSPARENT);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(22), dp(20), dp(18));
        panel.setBackground(makePanelBackground());
        scroll.addView(panel, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        TextView title = createDialogTitle("Нужно обновиться");
        panel.addView(title);

        String latestLabel = updateInfo.latestVersionName.isEmpty()
                ? "последнюю версию"
                : "версию " + updateInfo.latestVersionName;
        TextView subtitle = createDialogSubtitle(
                "Установлен клиент " + BuildConfig.VERSION_NAME + ". Доступна " + latestLabel
                        + ". Подключение к серверам заблокировано до обновления."
        );
        panel.addView(subtitle);

        TextView message = createDialogSubtitle(updateInfo.message);
        panel.addView(message);

        if (isRuStoreBuild()) {
            TextView rustoreLink = createLinkLabel("rustore.ru/catalog/app/ru.extraarena.app");
            panel.addView(rustoreLink, inputParams());

            TextView rustore = createDialogButton("Открыть RuStore", true);
            rustore.setOnClickListener(v -> openExternal(updateInfo.rustoreUrl));
            LinearLayout.LayoutParams rustoreParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            rustoreParams.topMargin = dp(14);
            panel.addView(rustore, rustoreParams);
        } else {
            TextView channelLink = createLinkLabel("t.me/extraarenamobile");
            panel.addView(channelLink, inputParams());

            TextView apkLink = createLinkLabel("apk.laveqox.ru");
            panel.addView(apkLink, inputParams());

            TextView apk = createDialogButton("Скачать APK", true);
            apk.setOnClickListener(v -> openExternal(updateInfo.apkUrl));
            LinearLayout.LayoutParams apkParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            apkParams.topMargin = dp(14);
            panel.addView(apk, apkParams);

            TextView channel = createDialogButton("Telegram канал", false);
            channel.setOnClickListener(v -> openExternal(updateInfo.telegramUrl));
            panel.addView(channel, inputParams());
        }

        TextView retry = createDialogButton("Проверить снова", false);
        retry.setOnClickListener(v -> {
            dialog.dismiss();
            updateDialog = null;
            updateBlocked = false;
            blockedUpdateInfo = null;
            updateGatePassedAt = 0L;
            launchAfterUpdateGate(getIntent());
        });
        panel.addView(retry, inputParams());

        dialog.setContentView(scroll);
        dialog.setOnDismissListener(d -> {
            if (updateDialog == dialog) {
                updateDialog = null;
            }
        });
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawableResource(android.R.color.transparent);
        }
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(430)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private void loadArena(Intent intent) {
        if (DeviceRegistrar.getAuthToken(this).isEmpty()) {
            showAuth();
            return;
        }
        if (!hasNetworkConnection()) {
            showConnectivityError();
            return;
        }
        hideSoftKeyboardAndClearAuthFocus();
        arenaLoadBlockedByConnectivity = false;
        loadingPausedForProfileSwitcher = false;
        authView.setVisibility(View.GONE);
        loadingView.setVisibility(View.VISIBLE);
        setLoadingDevHotspotVisible(true);
        errorView.setVisibility(View.GONE);
        String section = intent == null ? null : intent.getStringExtra("section");
        String inviteId = intent == null ? null : intent.getStringExtra("invite_id");
        String inviteAction = intent == null ? null : intent.getStringExtra("invite_action");
        if (section == null && intent != null && intent.getData() != null) {
            section = intent.getData().getQueryParameter("section");
        }
        if (inviteId == null && intent != null && intent.getData() != null) {
            inviteId = intent.getData().getQueryParameter("invite_id");
        }
        if (inviteAction == null && intent != null && intent.getData() != null) {
            inviteAction = intent.getData().getQueryParameter("invite_action");
        }
        probeAndLoadArena(section, inviteId, inviteAction);
    }

    private String buildLaunchUrl(String section) {
        return buildLaunchUrl(section, null, null);
    }

    private String buildLaunchUrl(String section, String inviteId, String inviteAction) {
        Uri.Builder builder = Uri.parse(BaseUrlStore.getBaseUrl(this)).buildUpon()
                .appendQueryParameter("_auth", DeviceRegistrar.getAuthToken(this))
                .appendQueryParameter("ea_platform", "android_app")
                .appendQueryParameter("ea_shell", "android")
                .appendQueryParameter("ea_telegram", "0")
                .appendQueryParameter("ea_app_version", BuildConfig.VERSION_NAME)
                .appendQueryParameter("ea_distribution", BuildConfig.DISTRIBUTION_CHANNEL);
        String whitelistCode = ConnectionProfileStore.getWhitelistCode(this);
        if (!whitelistCode.isEmpty()) {
            builder.appendQueryParameter("ea_whitelist_code", whitelistCode);
        }
        if (section != null && !section.trim().isEmpty()) {
            builder.appendQueryParameter("section", section);
        }
        Map<String, String> query = new HashMap<>();
        if (inviteId != null && !inviteId.trim().isEmpty()) {
            query.put("invite_id", inviteId);
        }
        if (inviteAction != null && !inviteAction.trim().isEmpty()) {
            query.put("invite_action", inviteAction);
        }
        for (Map.Entry<String, String> entry : query.entrySet()) {
            builder.appendQueryParameter(entry.getKey(), entry.getValue());
        }
        return builder.build().toString();
    }

    private void probeAndLoadArena(String section, String inviteId, String inviteAction) {
        final int generation = ++arenaLoadGeneration;
        if (generation != arenaLoadGeneration || loadingPausedForProfileSwitcher) {
            return;
        }
        arenaProbeExecutor.execute(() -> {
            ConnectionProfileStore.ConnectionProfile selected = ConnectionProfileStore.getSelectedProfile(this);
            boolean available = isServerAvailable(BaseUrlStore.join(selected.baseUrl, "health"));
            String switchedTo = null;
            if (!available && ConnectionProfileStore.isBuiltIn(selected.id)) {
                // The selected built-in host is unreachable: try the other built-in host so the app
                // keeps working when one entrypoint (e.g. the Cloudflare tunnel) is down.
                ConnectionProfileStore.ConnectionProfile other =
                        ConnectionProfileStore.otherBuiltIn(this, selected.id);
                if (other != null
                        && isServerAvailable(BaseUrlStore.join(other.baseUrl, "health"))) {
                    selected = other;
                    available = true;
                    switchedTo = other.id;
                }
            }
            final boolean finalAvailable = available;
            final String finalSwitchedTo = switchedTo;
            final ConnectionProfileStore.ConnectionProfile finalSelected = selected;
            runOnUiThread(() -> {
                if (generation != arenaLoadGeneration || loadingPausedForProfileSwitcher || updateBlocked) {
                    return;
                }
                if (finalAvailable) {
                    if (finalSwitchedTo != null) {
                        // Persist the fallback selection under the generation guard so a manual
                        // profile choice made during the probe window is not silently overwritten.
                        ConnectionProfileStore.selectProfile(this, finalSwitchedTo);
                        Log.w(TAG, "Selected host unreachable; fell back to profile " + finalSwitchedTo);
                    }
                    webView.loadUrl(buildLaunchUrl(section, inviteId, inviteAction));
                } else {
                    showConnectivityError();
                }
            });
        });
    }

    private boolean isServerAvailable(String url) {
        HttpURLConnection connection = null;
        try {
            URL target = new URL(url);
            connection = (HttpURLConnection) target.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);
            connection.setInstanceFollowRedirects(true);
            connection.setRequestProperty("Range", "bytes=0-256");
            connection.setRequestProperty("User-Agent", "ExtraArenaApp/" + BuildConfig.VERSION_NAME);
            int status = connection.getResponseCode();
            return status > 0 && status < 500 && status != HttpURLConnection.HTTP_UNAVAILABLE;
        } catch (Exception ignored) {
            return false;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private void pingConnectionProfile(
            ConnectionProfileStore.ConnectionProfile profile,
            TextView statusView,
            TextView button
    ) {
        if (profile == null) {
            return;
        }
        button.setEnabled(false);
        button.setAlpha(0.65f);
        statusView.setText("Пингуем...");
        statusView.setTextColor(Color.rgb(255, 218, 157));
        arenaProbeExecutor.execute(() -> {
            PingResult result = pingProfile(profile);
            runOnUiThread(() -> {
                button.setEnabled(true);
                button.setAlpha(1f);
                if (result.ok) {
                    statusView.setText(result.elapsedMs + " мс");
                    statusView.setTextColor(Color.rgb(114, 228, 169));
                } else {
                    statusView.setText(result.statusCode > 0
                            ? "HTTP " + result.statusCode
                            : "нет ответа");
                    statusView.setTextColor(Color.rgb(255, 132, 132));
                }
            });
        });
    }

    private PingResult pingProfile(ConnectionProfileStore.ConnectionProfile profile) {
        HttpURLConnection connection = null;
        long started = System.nanoTime();
        int status = -1;
        try {
            URL target = new URL(BaseUrlStore.join(profile.baseUrl, "health"));
            connection = (HttpURLConnection) target.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(4000);
            connection.setReadTimeout(4000);
            connection.setInstanceFollowRedirects(true);
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("Cache-Control", "no-cache");
            connection.setRequestProperty("User-Agent", "ExtraArenaApp/" + BuildConfig.VERSION_NAME);
            if (profile.whitelistEnabled && !profile.whitelistCode.isEmpty()) {
                connection.setRequestProperty("X-ExtraArena-Whitelist-Code", profile.whitelistCode);
            }
            status = connection.getResponseCode();
            long elapsedMs = Math.max(1L, Math.round((System.nanoTime() - started) / 1_000_000.0));
            return new PingResult(status > 0 && status < 500 && status != HttpURLConnection.HTTP_UNAVAILABLE, elapsedMs, status);
        } catch (Exception ignored) {
            long elapsedMs = Math.max(1L, Math.round((System.nanoTime() - started) / 1_000_000.0));
            return new PingResult(false, elapsedMs, status);
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private void injectAppContext() {
        ConnectionProfileStore.ConnectionProfile profile = ConnectionProfileStore.getSelectedProfile(this);
        String payload = new JSONArray()
                .put("android_app")
                .put(BuildConfig.VERSION_NAME)
                .put(BaseUrlStore.getBaseUrl(this))
                .put(BaseUrlStore.isTestServer(this))
                .put(profile.name)
                .put(profile.whitelistEnabled)
                .put(profile.whitelistCode)
                .put(BuildConfig.DISTRIBUTION_CHANNEL)
                .put(BuildConfig.PAYMENT_PROVIDER_ORDER)
                .toString();
        String script = "window.__EXTRA_ARENA_APP__={"
                + "platform:" + payload + "[0],"
                + "version:" + payload + "[1],"
                + "baseUrl:" + payload + "[2],"
                + "isTestServer:" + payload + "[3],"
                + "connectionProfile:" + payload + "[4],"
                + "whitelistEnabled:" + payload + "[5],"
                + "whitelistCode:" + payload + "[6],"
                + "distributionChannel:" + payload + "[7],"
                + "paymentProviderOrder:" + payload + "[8],"
                + "telegram:false"
                + "};";
        webView.evaluateJavascript(script, null);
    }

    private void captureAuthTokenFromLocalStorage() {
        webView.evaluateJavascript(
                "(function(){try{var p=new URL(location.href).searchParams.get('_auth');return p||localStorage.getItem('extra_id_token')||'';}catch(e){return localStorage.getItem('extra_id_token')||'';}})()",
                value -> {
                    String token = stripJsString(value);
                    if (!token.isEmpty()) {
                        DeviceRegistrar.saveAuthToken(this, token);
                    }
                }
        );
    }

    private String stripJsString(String value) {
        if (value == null || "null".equals(value)) {
            return "";
        }
        String result = value;
        if (result.length() >= 2 && result.startsWith("\"") && result.endsWith("\"")) {
            result = result.substring(1, result.length() - 1);
        }
        return result.replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private void registerSecretTap() {
        if (loadingView.getVisibility() != View.VISIBLE) {
            return;
        }
        long now = System.currentTimeMillis();
        if (firstSecretTapAt == 0L || now - firstSecretTapAt > SECRET_TAP_WINDOW_MS) {
            firstSecretTapAt = now;
            secretTapCount = 0;
        }
        secretTapCount += 1;
        if (secretTapCount >= 5) {
            secretTapCount = 0;
            firstSecretTapAt = 0L;
            pauseLoadingForProfileSwitcher();
            showServerSwitcher();
        }
    }

    private void pauseLoadingForProfileSwitcher() {
        loadingPausedForProfileSwitcher = true;
        webView.stopLoading();
        webView.loadUrl("about:blank");
        webView.setVisibility(View.INVISIBLE);
        loadingView.setVisibility(View.GONE);
        setLoadingDevHotspotVisible(false);
        errorView.setVisibility(View.GONE);
    }

    private void showServerSwitcher() {
        Dialog dialog = new Dialog(this);
        final boolean[] profileChanged = {false};
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.setBackgroundColor(Color.TRANSPARENT);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(20), dp(20), dp(18));
        panel.setBackground(makePanelBackground());
        scroll.addView(panel, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        TextView title = createDialogTitle("Профили подключения");
        panel.addView(title);

        TextView subtitle = createDialogSubtitle("Выбери сервер или добавь отдельный профиль для разработки.");
        panel.addView(subtitle);

        ConnectionProfileStore.ConnectionProfile selected = ConnectionProfileStore.getSelectedProfile(this);
        List<ConnectionProfileStore.ConnectionProfile> profiles = ConnectionProfileStore.getProfiles(this);
        for (ConnectionProfileStore.ConnectionProfile profile : profiles) {
            panel.addView(createProfileCard(profile, profile.id.equals(selected.id), dialog, profileChanged));
        }

        TextView addTitle = createDialogSectionTitle("Новый профиль");
        LinearLayout.LayoutParams addTitleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        addTitleParams.topMargin = dp(16);
        panel.addView(addTitle, addTitleParams);

        EditText profileName = createInput("Название профиля", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_NORMAL);
        panel.addView(profileName, inputParams());

        EditText profileUrl = createInput("base_url", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        profileUrl.setText(BuildConfig.TEST_BASE_URL);
        profileUrl.setSelectAllOnFocus(true);
        panel.addView(profileUrl, inputParams());

        LinearLayout whitelistRow = new LinearLayout(this);
        whitelistRow.setGravity(Gravity.CENTER_VERTICAL);
        whitelistRow.setPadding(dp(12), dp(6), dp(6), dp(4));
        whitelistRow.setBackground(makeInputBackground());
        TextView whitelistLabel = new TextView(this);
        whitelistLabel.setText("WhiteList");
        whitelistLabel.setTextColor(Color.rgb(246, 241, 255));
        whitelistLabel.setTextSize(14);
        whitelistLabel.setTypeface(Typeface.DEFAULT_BOLD);
        whitelistRow.addView(whitelistLabel, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Switch whitelistSwitch = new Switch(this);
        whitelistRow.addView(whitelistSwitch);
        panel.addView(whitelistRow, inputParams());

        EditText whitelistCode = createInput("Код WhiteList", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_NORMAL);
        whitelistCode.setVisibility(View.GONE);
        panel.addView(whitelistCode, inputParams());
        whitelistSwitch.setOnCheckedChangeListener((buttonView, checked) ->
                whitelistCode.setVisibility(checked ? View.VISIBLE : View.GONE)
        );

        TextView save = createDialogButton("Сохранить профиль", true);
        save.setOnClickListener(v -> {
            String name = profileName.getText().toString().trim();
            String baseUrl = profileUrl.getText().toString().trim();
            boolean whitelistEnabled = whitelistSwitch.isChecked();
            String code = whitelistCode.getText().toString().trim();
            if (name.isEmpty()) {
                Toast.makeText(this, "Укажи название профиля", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!baseUrl.startsWith("http://") && !baseUrl.startsWith("https://")) {
                Toast.makeText(this, "base_url должен начинаться с http:// или https://", Toast.LENGTH_SHORT).show();
                return;
            }
            if (whitelistEnabled && code.isEmpty()) {
                Toast.makeText(this, "Укажи код WhiteList", Toast.LENGTH_SHORT).show();
                return;
            }
            ConnectionProfileStore.ConnectionProfile profile = new ConnectionProfileStore.ConnectionProfile(
                    ConnectionProfileStore.newProfileId(),
                    name,
                    baseUrl,
                    whitelistEnabled,
                    code
            );
            ConnectionProfileStore.saveProfile(this, profile);
            ConnectionProfileStore.selectProfile(this, profile.id);
            Toast.makeText(this, "Профиль подключен", Toast.LENGTH_SHORT).show();
            profileChanged[0] = true;
            dialog.dismiss();
            reloadAfterServerChange();
        });
        LinearLayout.LayoutParams saveParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        saveParams.topMargin = dp(16);
        panel.addView(save, saveParams);

        TextView close = createDialogButton("Закрыть", false);
        close.setOnClickListener(v -> dialog.dismiss());
        panel.addView(close, inputParams());

        dialog.setOnDismissListener(d -> {
            if (loadingPausedForProfileSwitcher && !profileChanged[0]) {
                launchAfterUpdateGate(getIntent());
            }
        });
        dialog.setContentView(scroll);
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawableResource(android.R.color.transparent);
        }
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(430)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private void reloadAfterServerChange() {
        loadingPausedForProfileSwitcher = false;
        webView.clearCache(true);
        webView.clearHistory();
        DeviceRegistrar.clearAuthToken(this);
        launchAfterUpdateGate(getIntent());
    }

    private void setLoadingDevHotspotVisible(boolean visible) {
        if (loadingDevHotspot != null) {
            loadingDevHotspot.setVisibility(visible ? View.VISIBLE : View.GONE);
        }
    }

    private View createProfileCard(
            ConnectionProfileStore.ConnectionProfile profile,
            boolean selected,
            Dialog dialog,
            boolean[] profileChanged
    ) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        card.setBackground(makeProfileBackground(selected));
        card.setOnClickListener(v -> {
            ConnectionProfileStore.selectProfile(this, profile.id);
            Toast.makeText(this, "Подключено: " + profile.name, Toast.LENGTH_SHORT).show();
            profileChanged[0] = true;
            dialog.dismiss();
            reloadAfterServerChange();
        });

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);

        TextView name = new TextView(this);
        name.setText(profile.name);
        name.setTextColor(Color.rgb(246, 241, 255));
        name.setTextSize(15);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        top.addView(name, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        if (selected) {
            TextView badge = new TextView(this);
            badge.setText("Активен");
            badge.setTextColor(Color.rgb(20, 12, 28));
            badge.setTextSize(11);
            badge.setTypeface(Typeface.DEFAULT_BOLD);
            badge.setPadding(dp(8), dp(4), dp(8), dp(4));
            badge.setBackground(makeButtonBackground(Color.rgb(255, 184, 116)));
            top.addView(badge);
        }
        card.addView(top);

        TextView url = new TextView(this);
        url.setText(profile.baseUrl);
        url.setTextColor(Color.rgb(171, 159, 197));
        url.setTextSize(12);
        url.setSingleLine(false);
        LinearLayout.LayoutParams urlParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        urlParams.topMargin = dp(5);
        card.addView(url, urlParams);

        if (profile.whitelistEnabled) {
            TextView whitelist = new TextView(this);
            whitelist.setText("WhiteList включен");
            whitelist.setTextColor(Color.rgb(114, 228, 169));
            whitelist.setTextSize(12);
            LinearLayout.LayoutParams whitelistParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            whitelistParams.topMargin = dp(5);
            card.addView(whitelist, whitelistParams);
        }

        LinearLayout pingRow = new LinearLayout(this);
        pingRow.setGravity(Gravity.CENTER_VERTICAL);
        pingRow.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams pingRowParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        pingRowParams.topMargin = dp(9);

        TextView pingStatus = new TextView(this);
        pingStatus.setText("Пинг не проверен");
        pingStatus.setTextColor(Color.rgb(183, 169, 210));
        pingStatus.setTextSize(12);
        pingRow.addView(pingStatus, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView pingButton = createSmallDialogButton("Пинг");
        pingButton.setOnClickListener(v -> pingConnectionProfile(profile, pingStatus, pingButton));
        pingRow.addView(pingButton);
        card.addView(pingRow, pingRowParams);

        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.topMargin = dp(10);
        card.setLayoutParams(params);
        return card;
    }

    private TextView createDialogTitle(String text) {
        TextView title = new TextView(this);
        title.setText(text);
        title.setTextColor(Color.rgb(246, 241, 255));
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setGravity(Gravity.CENTER);
        return title;
    }

    private TextView createDialogSubtitle(String text) {
        TextView subtitle = new TextView(this);
        subtitle.setText(text);
        subtitle.setTextColor(Color.rgb(183, 169, 210));
        subtitle.setTextSize(13);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setLineSpacing(2, 1);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.topMargin = dp(6);
        params.bottomMargin = dp(10);
        subtitle.setLayoutParams(params);
        return subtitle;
    }

    private TextView createDialogSectionTitle(String text) {
        TextView title = new TextView(this);
        title.setText(text);
        title.setTextColor(Color.rgb(255, 184, 116));
        title.setTextSize(13);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        return title;
    }

    private TextView createDialogButton(String text, boolean primary) {
        TextView button = new TextView(this);
        button.setText(text);
        button.setGravity(Gravity.CENTER);
        button.setTextSize(15);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setTextColor(primary ? Color.WHITE : Color.rgb(255, 184, 116));
        button.setPadding(dp(16), dp(12), dp(16), dp(12));
        button.setBackground(makeButtonBackground(primary ? Color.rgb(255, 138, 61) : Color.rgb(33, 22, 55)));
        addHapticTouch(button, primary ? "medium" : "light");
        return button;
    }

    private TextView createSmallDialogButton(String text) {
        TextView button = new TextView(this);
        button.setText(text);
        button.setGravity(Gravity.CENTER);
        button.setTextSize(12);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setTextColor(Color.rgb(255, 218, 157));
        button.setPadding(dp(12), dp(7), dp(12), dp(7));
        button.setBackground(makeButtonBackground(Color.rgb(33, 22, 55)));
        addHapticTouch(button, "selection");
        return button;
    }

    private TextView createLinkLabel(String text) {
        TextView label = new TextView(this);
        label.setText(text);
        label.setGravity(Gravity.CENTER);
        label.setTextSize(13);
        label.setTypeface(Typeface.DEFAULT_BOLD);
        label.setTextColor(Color.rgb(173, 255, 244));
        label.setPadding(dp(12), dp(8), dp(12), dp(8));
        label.setBackground(makeInputBackground());
        return label;
    }

    private FrameLayout createShellScreen() {
        FrameLayout screen = new FrameLayout(this);
        screen.addView(new ShellBackgroundView(this), new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));
        return screen;
    }

    private View createTopbar(String label, String action) {
        LinearLayout topbar = new LinearLayout(this);
        topbar.setGravity(Gravity.CENTER_VERTICAL);
        topbar.setPadding(dp(shellPaddingDp()), 0, dp(shellPaddingDp()), 0);

        LinearLayout brand = new LinearLayout(this);
        brand.setGravity(Gravity.CENTER_VERTICAL);

        View dot = new View(this);
        dot.setBackground(makeBrandDotBackground());
        int dotSize = isCompactHeight() ? 26 : 28;
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dp(dotSize), dp(dotSize));
        brand.addView(dot, dotParams);

        topbarLabel = new TextView(this);
        topbarLabel.setText(label);
        topbarLabel.setTextColor(Color.argb(204, 240, 236, 255));
        topbarLabel.setTextSize(12);
        topbarLabel.setTypeface(futuraExtraBold);
        topbarLabel.setLetterSpacing(0.08f);
        topbarLabel.setAllCaps(true);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        labelParams.leftMargin = dp(9);
        brand.addView(topbarLabel, labelParams);

        topbar.addView(brand, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        topbarAction = new TextView(this);
        topbarAction.setTextColor(Color.argb(235, 196, 184, 232));
        topbarAction.setTextSize(13);
        topbarAction.setTypeface(futuraBold);
        topbarAction.setGravity(Gravity.CENTER);
        topbarAction.setMinHeight(dp(44));
        topbarAction.setPadding(dp(8), 0, dp(4), 0);
        topbarAction.setOnClickListener(v -> handleTopbarAction());
        addHapticTouch(topbarAction, "selection");
        topbar.addView(topbarAction);
        setTopbarAction(action);

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(topbarHeightDp()),
                Gravity.TOP
        );
        topbar.setLayoutParams(params);
        return topbar;
    }

    private TextView createKicker() {
        TextView kicker = new TextView(this);
        kicker.setTextColor(Color.rgb(255, 217, 239));
        kicker.setTextSize(11);
        kicker.setTypeface(futuraExtraBold);
        kicker.setLetterSpacing(0.08f);
        kicker.setAllCaps(true);
        kicker.setPadding(dp(10), dp(isCompactHeight() ? 6 : 7), dp(10), dp(isCompactHeight() ? 5 : 6));
        kicker.setBackground(makeKickerBackground());
        return kicker;
    }

    private LinearLayout createLoginSteps() {
        LinearLayout steps = new LinearLayout(this);
        steps.setOrientation(LinearLayout.HORIZONTAL);
        steps.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.topMargin = dp(isCompactHeight() ? 10 : 16);
        steps.setLayoutParams(params);
        for (int i = 0; i < 3; i++) {
            View bar = new View(this);
            LinearLayout.LayoutParams barParams = new LinearLayout.LayoutParams(0, dp(isCompactHeight() ? 4 : 5), 1f);
            if (i > 0) {
                barParams.leftMargin = dp(7);
            }
            steps.addView(bar, barParams);
        }
        return steps;
    }

    private LinearLayout createField(String label, EditText input) {
        LinearLayout field = new LinearLayout(this);
        field.setOrientation(LinearLayout.VERTICAL);
        field.setPadding(
                dp(15),
                dp(isCompactHeight() ? 11 : 14),
                dp(15),
                dp(isCompactHeight() ? 10 : 12)
        );
        field.setBackground(makeFieldBackground(false));

        TextView labelView = new TextView(this);
        labelView.setText(label);
        labelView.setTextColor(Color.argb(205, 196, 184, 232));
        labelView.setTextSize(11);
        labelView.setTypeface(futuraExtraBold);
        labelView.setLetterSpacing(0.08f);
        labelView.setAllCaps(true);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        labelParams.bottomMargin = dp(7);
        field.addView(labelView, labelParams);
        field.addView(input, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        return field;
    }

    private void stylePrimaryButton(TextView button) {
        button.setGravity(Gravity.CENTER);
        button.setTextColor(Color.rgb(42, 18, 0));
        button.setTextSize(isCompactHeight() ? 16 : 17);
        button.setTypeface(futuraExtraBold);
        button.setLetterSpacing(0.02f);
        button.setMinHeight(dp(buttonHeightDp()));
        button.setPadding(dp(18), dp(isCompactHeight() ? 11 : 14), dp(18), dp(isCompactHeight() ? 11 : 14));
        button.setBackground(makePrimaryButtonBackground());
        button.setElevation(dp(8));
        addHapticTouch(button, "medium");
    }

    private void styleSecondaryButton(TextView button) {
        button.setGravity(Gravity.CENTER);
        button.setTextColor(EA_TEXT);
        button.setTextSize(isCompactHeight() ? 15 : 16);
        button.setTypeface(futuraExtraBold);
        button.setMinHeight(dp(buttonHeightDp()));
        button.setPadding(dp(18), dp(isCompactHeight() ? 11 : 14), dp(18), dp(isCompactHeight() ? 11 : 14));
        button.setBackground(makeSecondaryButtonBackground());
        addHapticTouch(button, "light");
    }

    private void styleTelegramButton(TextView button) {
        button.setGravity(Gravity.CENTER);
        button.setTextColor(Color.WHITE);
        button.setTextSize(isCompactHeight() ? 15 : 16);
        button.setTypeface(futuraExtraBold);
        button.setMinHeight(dp(telegramButtonHeightDp()));
        button.setPadding(dp(18), dp(isCompactHeight() ? 10 : 13), dp(18), dp(isCompactHeight() ? 10 : 13));
        button.setBackground(makeTelegramButtonBackground());
        button.setElevation(dp(6));
        addHapticTouch(button, "medium");
    }

    private View createAuthView() {
        FrameLayout screen = createShellScreen();
        screen.addView(createTopbar("ExtraArena", null));

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setOverScrollMode(View.OVER_SCROLL_NEVER);
        screen.addView(scroll, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(
                dp(shellPaddingDp()),
                dp(topbarHeightDp()),
                dp(shellPaddingDp()),
                dp(isCompactHeight() ? 14 : 22)
        );
        scroll.addView(layout, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        authKicker = createKicker();
        layout.addView(authKicker);

        authTitle = new TextView(this);
        authTitle.setTextColor(Color.WHITE);
        authTitle.setTextSize(authTitleSizeSp());
        authTitle.setTypeface(futuraExtraBold);
        authTitle.setGravity(Gravity.LEFT);
        authTitle.setLineSpacing(0, 0.90f);
        authTitle.setShadowLayer(dp(3), 0, dp(2), Color.argb(170, 59, 22, 96));
        layout.addView(authTitle, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        authSubtitle = new TextView(this);
        authSubtitle.setTextColor(Color.argb(210, 240, 236, 255));
        authSubtitle.setTextSize(bodyTextSizeSp());
        authSubtitle.setTypeface(futuraMedium);
        authSubtitle.setGravity(Gravity.LEFT);
        authSubtitle.setLineSpacing(dp(2), 1.04f);
        LinearLayout.LayoutParams subtitleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        subtitleParams.topMargin = dp(isCompactHeight() ? 10 : 14);
        layout.addView(authSubtitle, subtitleParams);

        loginSteps = createLoginSteps();
        layout.addView(loginSteps);

        authStage = new FrameLayout(this);
        LinearLayout.LayoutParams stageParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(274)
        );
        stageParams.topMargin = dp(fieldGapDp());
        layout.addView(authStage, stageParams);

        authEmail = createInput("Email", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS);
        authEmailField = createField("Почта", authEmail);
        layout.addView(authEmailField, inputParams());

        authPassword = createInput("Пароль", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        authPasswordField = createField("Пароль", authPassword);
        layout.addView(authPasswordField, inputParams());

        authNickname = createInput("Никнейм", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_NORMAL);
        authNicknameField = createField("Никнейм", authNickname);
        layout.addView(authNicknameField, inputParams());

        authTelegramId = createInput("Telegram ID", InputType.TYPE_CLASS_NUMBER);
        authTelegramIdField = createField("Telegram ID", authTelegramId);
        layout.addView(authTelegramIdField, inputParams());

        authTelegramCode = createInput("Код", InputType.TYPE_CLASS_NUMBER);
        authTelegramCode.setFilters(new InputFilter[]{new InputFilter.LengthFilter(6)});
        authTelegramCodeField = createField("Код из бота", authTelegramCode);
        layout.addView(authTelegramCodeField, inputParams());

        authHint = new TextView(this);
        authHint.setTextColor(Color.argb(194, 196, 184, 232));
        authHint.setTextSize(isCompactHeight() ? 13 : 14);
        authHint.setTypeface(futuraMedium);
        authHint.setLineSpacing(dp(2), 1f);
        LinearLayout.LayoutParams hintParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        hintParams.topMargin = dp(isCompactHeight() ? 8 : 12);
        layout.addView(authHint, hintParams);

        accountSectionTitle = new TextView(this);
        accountSectionTitle.setText("Сохраненные ExtraID");
        accountSectionTitle.setTextColor(Color.rgb(173, 255, 244));
        accountSectionTitle.setTextSize(13);
        accountSectionTitle.setTypeface(futuraExtraBold);
        accountSectionTitle.setLetterSpacing(0.04f);
        LinearLayout.LayoutParams accountTitleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        accountTitleParams.topMargin = dp(isCompactHeight() ? 10 : 16);
        layout.addView(accountSectionTitle, accountTitleParams);

        savedAccountsList = new LinearLayout(this);
        savedAccountsList.setOrientation(LinearLayout.VERTICAL);
        layout.addView(savedAccountsList);

        authError = new TextView(this);
        authError.setTextColor(Color.rgb(239, 68, 68));
        authError.setTextSize(13);
        authError.setTypeface(futuraBold);
        authError.setGravity(Gravity.LEFT);
        authError.setVisibility(View.GONE);
        LinearLayout.LayoutParams errorParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        errorParams.topMargin = dp(isCompactHeight() ? 6 : 8);
        layout.addView(authError, errorParams);

        View bottomSpacer = new View(this);
        layout.addView(bottomSpacer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        LinearLayout bottomPanel = new LinearLayout(this);
        bottomPanel.setOrientation(LinearLayout.VERTICAL);
        bottomPanel.setGravity(Gravity.CENTER_HORIZONTAL);
        layout.addView(bottomPanel, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        authAction = new TextView(this);
        stylePrimaryButton(authAction);
        authAction.setOnClickListener(v -> submitAuth());
        bottomPanel.addView(authAction, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        authSecondaryAction = new TextView(this);
        styleSecondaryButton(authSecondaryAction);
        authSecondaryAction.setOnClickListener(v -> submitSecondaryAuthAction());
        bottomPanel.addView(authSecondaryAction, inputParams());

        authTelegramAction = new TextView(this);
        styleTelegramButton(authTelegramAction);
        authTelegramAction.setText("Я из Telegram");
        authTelegramAction.setOnClickListener(v -> updateAuthStep(AuthStep.TELEGRAM_ID));
        bottomPanel.addView(authTelegramAction, inputParams());

        authModeSwitch = new TextView(this);
        authModeSwitch.setGravity(Gravity.CENTER);
        authModeSwitch.setTextColor(Color.rgb(255, 210, 138));
        authModeSwitch.setTextSize(isCompactHeight() ? 14 : 15);
        authModeSwitch.setTypeface(futuraBold);
        authModeSwitch.setPadding(dp(12), dp(isCompactHeight() ? 8 : 12), dp(12), 0);
        authModeSwitch.setOnClickListener(v -> submitAuthModeSwitch());
        bottomPanel.addView(authModeSwitch, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        authLegalNotice = createLegalNotice();
        bottomPanel.addView(authLegalNotice, inputParams());

        updateAuthStep(AuthStep.WELCOME);
        return screen;
    }

    private TextView createLegalNotice() {
        TextView text = new TextView(this);
        text.setGravity(Gravity.CENTER);
        text.setTextSize(isCompactHeight() ? 11 : 12);
        text.setTypeface(futuraMedium);
        text.setTextColor(Color.argb(178, 196, 184, 232));
        text.setLineSpacing(dp(2), 1f);
        text.setHighlightColor(Color.TRANSPARENT);
        text.setMovementMethod(LinkMovementMethod.getInstance());
        String copy = "Продолжая, ты соглашаешься с офертой и политикой конфиденциальности";
        SpannableString span = new SpannableString(copy);
        applyLegalLink(span, copy, "офертой");
        applyLegalLink(span, copy, "политикой конфиденциальности");
        text.setText(span);
        return text;
    }

    private void applyLegalLink(SpannableString span, String copy, String target) {
        int start = copy.indexOf(target);
        if (start < 0) {
            return;
        }
        int end = start + target.length();
        span.setSpan(new ClickableSpan() {
            @Override
            public void onClick(View widget) {
                openExternal(legalUrlForTarget(target));
            }

            @Override
            public void updateDrawState(TextPaint ds) {
                super.updateDrawState(ds);
                ds.setColor(Color.rgb(255, 210, 138));
                ds.setUnderlineText(true);
            }
        }, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }

    private String legalUrlForTarget(String target) {
        if (target != null && target.toLowerCase().contains("политик")) {
            return configuredLegalUrlOrDefault(BuildConfig.LEGAL_PRIVACY_URL, "/legal/privacy");
        }
        if (target != null && target.toLowerCase().contains("возврат")) {
            return configuredLegalUrlOrDefault(BuildConfig.LEGAL_REFUND_URL, "/legal/refund");
        }
        return configuredLegalUrlOrDefault(BuildConfig.LEGAL_OFFER_URL, "/legal/offer");
    }

    private String configuredLegalUrlOrDefault(String configuredUrl, String path) {
        if (configuredUrl != null && !configuredUrl.trim().isEmpty()) {
            return configuredUrl.trim();
        }
        return BaseUrlStore.join(BaseUrlStore.getBaseUrl(this), path);
    }

    private EditText createInput(String hint, int inputType) {
        EditText input = new EditText(this);
        input.setHint(hint);
        input.setSingleLine(true);
        input.setInputType(inputType);
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.argb(96, 196, 184, 232));
        input.setTextSize(isCompactHeight() ? 20 : 22);
        input.setTypeface(futuraExtraBold);
        input.setPadding(0, 0, 0, 0);
        input.setBackgroundColor(Color.TRANSPARENT);
        return input;
    }

    private LinearLayout.LayoutParams inputParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.topMargin = dp(fieldGapDp());
        return params;
    }

    private void setTopbar(String label, String action) {
        if (topbarLabel != null) {
            topbarLabel.setText(label);
        }
        setTopbarAction(action);
    }

    private void setTopbarAction(String action) {
        if (topbarAction == null) {
            return;
        }
        if (action == null || action.trim().isEmpty()) {
            topbarAction.setVisibility(View.INVISIBLE);
            topbarAction.setText("");
        } else {
            topbarAction.setVisibility(View.VISIBLE);
            topbarAction.setText(action);
        }
    }

    private void handleTopbarAction() {
        if (authStep == AuthStep.WELCOME) {
            return;
        }
        if (authStep == AuthStep.TELEGRAM_CODE) {
            updateAuthStep(AuthStep.TELEGRAM_ID);
            return;
        }
        if (authStep == AuthStep.TELEGRAM_EMAIL) {
            updateAuthStep(AuthStep.TELEGRAM_CODE);
            return;
        }
        if (authStep == AuthStep.TELEGRAM_PASSWORD) {
            updateAuthStep(AuthStep.TELEGRAM_EMAIL);
            return;
        }
        updateAuthStep(AuthStep.WELCOME);
    }

    private void setFieldVisible(LinearLayout field, boolean visible, boolean active) {
        field.setVisibility(visible ? View.VISIBLE : View.GONE);
        field.setBackground(makeFieldBackground(active));
    }

    private void setStageHeight(int heightDp) {
        ViewGroup.LayoutParams params = authStage.getLayoutParams();
        params.height = heightDp <= 0 ? 0 : dp(heightDp);
        authStage.setLayoutParams(params);
        authStage.setVisibility(heightDp <= 0 ? View.GONE : View.VISIBLE);
    }

    private void updateLoginSteps(int activeStep) {
        loginSteps.setVisibility(activeStep <= 0 ? View.GONE : View.VISIBLE);
        for (int i = 0; i < loginSteps.getChildCount(); i++) {
            View bar = loginSteps.getChildAt(i);
            if (i < activeStep) {
                bar.setBackground(makeStepBackground(true));
            } else {
                bar.setBackground(makeStepBackground(false));
            }
        }
    }

    private void renderWelcomeStage() {
        authStage.removeAllViews();
        FrameLayout art = new FrameLayout(this);
        art.setBackground(makeWelcomeArtBackground());
        FrameLayout.LayoutParams artParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(welcomeArtHeightDp()),
                Gravity.TOP
        );
        artParams.leftMargin = -dp(18);
        artParams.rightMargin = -dp(18);
        artParams.topMargin = dp(isCompactHeight() ? 4 : 8);
        authStage.addView(art, artParams);

        ImageView image = createAssetImage("extra_mobile/midoriya-waving.jpg", ImageView.ScaleType.CENTER_CROP);
        image.setAlpha(0.82f);
        art.addView(image, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        View fade = new View(this);
        fade.setBackground(new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{Color.TRANSPARENT, Color.argb(235, 9, 5, 18)}
        ));
        art.addView(fade, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        HorizontalScrollView carousel = new HorizontalScrollView(this);
        carousel.setHorizontalScrollBarEnabled(false);
        carousel.setOverScrollMode(View.OVER_SCROLL_NEVER);
        LinearLayout track = new LinearLayout(this);
        track.setOrientation(LinearLayout.HORIZONTAL);
        track.setPadding(dp(16), 0, dp(16), 0);
        String[] cards = {"card-1.png", "3.png", "10.png", "14.png", "17.png", "21.png", "31.png", "card-40.png"};
        for (int i = 0; i < cards.length; i++) {
            ImageView card = createAssetImage("extra_mobile/" + cards[i], ImageView.ScaleType.CENTER_CROP);
            card.setRotation(i % 2 == 0 ? -5f : 4f);
            card.setTranslationY(dp(i % 2 == 0 ? 12 : 2));
            card.setElevation(dp(8));
            LinearLayout.LayoutParams cardParams = new LinearLayout.LayoutParams(dp(cardWidthDp()), dp(cardHeightDp()));
            if (i > 0) {
                cardParams.leftMargin = dp(isCompactHeight() ? 8 : 10);
            }
            track.addView(card, cardParams);
        }
        carousel.addView(track, new HorizontalScrollView.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));
        FrameLayout.LayoutParams carouselParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(welcomeCarouselHeightDp()),
                Gravity.BOTTOM
        );
        carouselParams.leftMargin = -dp(22);
        carouselParams.rightMargin = -dp(22);
        authStage.addView(carousel, carouselParams);
    }

    private void renderAnonymousStage() {
        authStage.removeAllViews();
    }

    private void renderExtraIdStage() {
        authStage.removeAllViews();
        TextView badge = new TextView(this);
        badge.setText("ID");
        badge.setTextColor(Color.WHITE);
        badge.setTextSize(isCompactHeight() ? 34 : 42);
        badge.setTypeface(futuraExtraBold);
        badge.setGravity(Gravity.CENTER);
        badge.setShadowLayer(dp(4), 0, dp(3), Color.argb(120, 52, 18, 84));
        badge.setRotation(-3f);
        badge.setBackground(makeExtraIdBadgeBackground());
        badge.setElevation(dp(12));
        int badgeSize = isCompactHeight() ? 88 : 112;
        FrameLayout.LayoutParams badgeParams = new FrameLayout.LayoutParams(dp(badgeSize), dp(badgeSize), Gravity.CENTER);
        authStage.addView(badge, badgeParams);
    }

    private void renderTelegramStage() {
        authStage.removeAllViews();
        TextView badge = new TextView(this);
        badge.setText("TG");
        badge.setTextColor(Color.WHITE);
        badge.setTextSize(isCompactHeight() ? 31 : 38);
        badge.setTypeface(futuraExtraBold);
        badge.setGravity(Gravity.CENTER);
        badge.setShadowLayer(dp(4), 0, dp(3), Color.argb(110, 0, 58, 114));
        badge.setRotation(3f);
        badge.setBackground(makeTelegramBadgeBackground());
        badge.setElevation(dp(12));
        int badgeSize = isCompactHeight() ? 88 : 112;
        FrameLayout.LayoutParams badgeParams = new FrameLayout.LayoutParams(dp(badgeSize), dp(badgeSize), Gravity.CENTER);
        authStage.addView(badge, badgeParams);
    }

    private void refreshSavedAccounts(boolean show) {
        savedAccountsList.removeAllViews();
        List<ExtraIdAccountStore.ExtraIdAccount> accounts = ExtraIdAccountStore.getAccounts(this, BaseUrlStore.getBaseUrl(this));
        boolean visible = show && !accounts.isEmpty();
        accountSectionTitle.setVisibility(visible ? View.VISIBLE : View.GONE);
        savedAccountsList.setVisibility(visible ? View.VISIBLE : View.GONE);
        if (!visible) {
            return;
        }
        for (ExtraIdAccountStore.ExtraIdAccount account : accounts) {
            savedAccountsList.addView(createAccountCard(account));
        }
    }

    private View createAccountCard(ExtraIdAccountStore.ExtraIdAccount account) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(13), dp(14), dp(13));
        card.setBackground(makeAccountCardBackground());
        card.setOnClickListener(v -> {
            resetWebViewForNewAuthSession();
            if (!persistNativeAuthToken(account.token)) {
                return;
            }
            Toast.makeText(this, "Входим как " + account.email, Toast.LENGTH_SHORT).show();
            launchAfterUpdateGate(getIntent());
        });

        TextView title = new TextView(this);
        title.setText(account.email);
        title.setTextColor(Color.rgb(173, 255, 244));
        title.setTextSize(15);
        title.setTypeface(futuraExtraBold);
        card.addView(title);

        TextView subtitle = new TextView(this);
        String display = account.displayId.isEmpty() ? "ExtraID" : "ExtraID " + account.displayId;
        subtitle.setText(display + " · нажми, чтобы продолжить");
        subtitle.setTextColor(Color.argb(194, 240, 236, 255));
        subtitle.setTextSize(13);
        subtitle.setTypeface(futuraMedium);
        LinearLayout.LayoutParams subtitleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        subtitleParams.topMargin = dp(4);
        card.addView(subtitle, subtitleParams);

        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.topMargin = dp(10);
        card.setLayoutParams(params);
        return card;
    }

    private void updateAuthStep(AuthStep step) {
        authStep = step;
        authError.setVisibility(View.GONE);

        setFieldVisible(authEmailField, false, false);
        setFieldVisible(authPasswordField, false, false);
        setFieldVisible(authNicknameField, false, false);
        setFieldVisible(authTelegramIdField, false, false);
        setFieldVisible(authTelegramCodeField, false, false);
        authEmail.setEnabled(true);
        authTelegramId.setEnabled(true);
        authTelegramCode.setEnabled(true);
        authPassword.setText("");
        authTelegramAction.setVisibility(View.GONE);
        authHint.setVisibility(View.GONE);
        refreshSavedAccounts(false);
        authSecondaryAction.setVisibility(step == AuthStep.WELCOME ? View.GONE : View.VISIBLE);
        authLegalNotice.setVisibility(step == AuthStep.WELCOME ? View.VISIBLE : View.GONE);

        if (step == AuthStep.WELCOME) {
            setTopbar("ExtraArena", null);
            authKicker.setText("Карточная арена");
            authTitle.setText("Залетай\nв бой");
            authSubtitle.setText("Собери отряд, разыграй карты и переиграй соперника за пару ходов.");
            updateLoginSteps(0);
            setStageHeight(welcomeStageHeightDp());
            renderWelcomeStage();
            authAction.setText("Погнали");
            authSecondaryAction.setText("Создать ExtraID");
            authTelegramAction.setVisibility(View.VISIBLE);
            authModeSwitch.setText("Уже играл? Войти");
        } else if (step == AuthStep.ANONYMOUS_NICKNAME) {
            setTopbar("Начинаем игру", "Позже");
            authKicker.setText("Быстрый старт");
            authTitle.setText("Как тебя\nзвать?");
            authSubtitle.setText("Ник будет виден в бою. Потом спокойно изменишь его в настройках.");
            updateLoginSteps(0);
            setStageHeight(isCompactHeight() ? 18 : 36);
            renderAnonymousStage();
            setFieldVisible(authNicknameField, true, true);
            authNickname.requestFocus();
            authAction.setText("Вперед");
            authSecondaryAction.setText("Создать ExtraID");
            authModeSwitch.setText("Уже играл? Войти");
        } else if (step == AuthStep.LOGIN_EMAIL) {
            setTopbar("ExtraID", "Назад");
            authKicker.setText("Сохранить прогресс");
            authTitle.setText("Войди\nв ExtraID");
            authSubtitle.setText("Введи почту — и продолжим.");
            updateLoginSteps(1);
            setStageHeight(isCompactHeight() ? 88 : 112);
            renderExtraIdStage();
            setFieldVisible(authEmailField, true, true);
            authEmail.requestFocus();
            authHint.setText("Можно вернуться к игре без входа в любой момент.");
            authHint.setVisibility(View.VISIBLE);
            refreshSavedAccounts(true);
            authAction.setText("Дальше");
            authSecondaryAction.setText("Играть без ExtraID");
            authModeSwitch.setText("Нет аккаунта? Создать ExtraID");
        } else if (step == AuthStep.LOGIN_PASSWORD) {
            setTopbar("ExtraID", "Назад");
            authKicker.setText("Сохранить прогресс");
            authTitle.setText("Введи\nпароль");
            authSubtitle.setText("Последний шаг для входа в аккаунт.");
            updateLoginSteps(2);
            setStageHeight(0);
            setFieldVisible(authEmailField, true, false);
            authEmail.setText(stagedEmail);
            authEmail.setEnabled(false);
            setFieldVisible(authPasswordField, true, true);
            authPassword.requestFocus();
            authAction.setText("Войти");
            authSecondaryAction.setText("Изменить email");
            authModeSwitch.setText("Нет аккаунта? Создать ExtraID");
        } else if (step == AuthStep.REGISTER_EMAIL) {
            setTopbar("ExtraID", "Назад");
            authKicker.setText("Новый аккаунт");
            authTitle.setText("Создай\nExtraID");
            authSubtitle.setText("ExtraID сохранит прогресс и даст вход без Telegram-сервисов.");
            updateLoginSteps(1);
            setStageHeight(isCompactHeight() ? 88 : 112);
            renderExtraIdStage();
            setFieldVisible(authEmailField, true, true);
            authEmail.requestFocus();
            authHint.setText("Анонимная игра остается доступной в любой момент.");
            authHint.setVisibility(View.VISIBLE);
            authAction.setText("Дальше");
            authSecondaryAction.setText("Играть без ExtraID");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        } else if (step == AuthStep.REGISTER_PASSWORD) {
            setTopbar("ExtraID", "Назад");
            authKicker.setText("Новый аккаунт");
            authTitle.setText("Придумай\nпароль");
            authSubtitle.setText("На этот email будет создан ExtraID для входа без Telegram.");
            updateLoginSteps(2);
            setStageHeight(0);
            setFieldVisible(authEmailField, true, false);
            authEmail.setText(stagedEmail);
            authEmail.setEnabled(false);
            setFieldVisible(authPasswordField, true, true);
            authPassword.requestFocus();
            authAction.setText("Создать аккаунт");
            authSecondaryAction.setText("Изменить email");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        } else if (step == AuthStep.TELEGRAM_ID) {
            setTopbar("Telegram", "Назад");
            authKicker.setText("Перенос");
            authTitle.setText("Играл\nв Telegram?");
            authSubtitle.setText("Введи свой Telegram ID. Напиши /id боту или найди ID в профиле.");
            updateLoginSteps(1);
            setStageHeight(isCompactHeight() ? 88 : 112);
            renderTelegramStage();
            authTelegramId.setText(stagedTelegramId);
            setFieldVisible(authTelegramIdField, true, true);
            authTelegramId.requestFocus();
            authHint.setText("Бот отправит одноразовый код в личные сообщения.");
            authHint.setVisibility(View.VISIBLE);
            authAction.setText("Получить код");
            authSecondaryAction.setText("Играть без ExtraID");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        } else if (step == AuthStep.TELEGRAM_CODE) {
            setTopbar("Telegram", "Назад");
            authKicker.setText("Проверка");
            authTitle.setText("Введи\nкод");
            authSubtitle.setText("Мы отправили код в личные сообщения бота.");
            updateLoginSteps(2);
            setStageHeight(0);
            authTelegramId.setText(stagedTelegramId);
            authTelegramId.setEnabled(false);
            setFieldVisible(authTelegramIdField, true, false);
            authTelegramCode.setText(stagedTelegramCode);
            setFieldVisible(authTelegramCodeField, true, true);
            authTelegramCode.requestFocus();
            authHint.setText("Код действует 5 минут. Если не пришел, проверь, что бот уже был открыт.");
            authHint.setVisibility(View.VISIBLE);
            authAction.setText("Продолжить");
            authSecondaryAction.setText("Изменить ID");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        } else if (step == AuthStep.TELEGRAM_EMAIL) {
            setTopbar("Telegram", "Назад");
            authKicker.setText("Новый ExtraID");
            authTitle.setText("Укажи\nпочту");
            authSubtitle.setText("На нее создадим ExtraID и привяжем Telegram-прогресс.");
            updateLoginSteps(3);
            setStageHeight(isCompactHeight() ? 88 : 112);
            renderTelegramStage();
            setFieldVisible(authEmailField, true, true);
            authEmail.requestFocus();
            authAction.setText("Дальше");
            authSecondaryAction.setText("Изменить код");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        } else if (step == AuthStep.TELEGRAM_PASSWORD) {
            setTopbar("Telegram", "Назад");
            authKicker.setText("Новый ExtraID");
            authTitle.setText("Создай\nпароль");
            authSubtitle.setText("После создания аккаунта сразу запустим игру.");
            updateLoginSteps(3);
            setStageHeight(0);
            setFieldVisible(authEmailField, true, false);
            authEmail.setText(stagedEmail);
            authEmail.setEnabled(false);
            setFieldVisible(authPasswordField, true, true);
            authPassword.requestFocus();
            authAction.setText("Создать ExtraID");
            authSecondaryAction.setText("Изменить email");
            authModeSwitch.setText("Уже есть ExtraID? Войти");
        }
    }

    private void showAuth() {
        webView.setVisibility(View.INVISIBLE);
        loadingView.setVisibility(View.GONE);
        setLoadingDevHotspotVisible(false);
        errorView.setVisibility(View.GONE);
        authView.setVisibility(View.VISIBLE);
        updateAuthStep(AuthStep.WELCOME);
    }

    private void hideSoftKeyboardAndClearAuthFocus() {
        View focused = getCurrentFocus();
        if (focused == null && authView != null) {
            focused = authView.findFocus();
        }
        if (focused != null) {
            focused.clearFocus();
            try {
                InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.hideSoftInputFromWindow(focused.getWindowToken(), 0);
                }
            } catch (Exception ignored) {
            }
        }
        if (root != null) {
            root.requestFocus();
        }
    }

    private void submitAuth() {
        if (authStep == AuthStep.WELCOME) {
            updateAuthStep(AuthStep.ANONYMOUS_NICKNAME);
            return;
        }
        if (authStep == AuthStep.ANONYMOUS_NICKNAME) {
            submitAnonymousProfile();
            return;
        }
        if (authStep == AuthStep.LOGIN_EMAIL || authStep == AuthStep.REGISTER_EMAIL) {
            String email = authEmail.getText().toString().trim();
            if (!isValidEmail(email)) {
                showAuthError("Проверь email");
                return;
            }
            stagedEmail = email.toLowerCase();
            updateAuthStep(authStep == AuthStep.LOGIN_EMAIL ? AuthStep.LOGIN_PASSWORD : AuthStep.REGISTER_PASSWORD);
            return;
        }
        if (authStep == AuthStep.TELEGRAM_ID) {
            submitTelegramId();
            return;
        }
        if (authStep == AuthStep.TELEGRAM_CODE) {
            submitTelegramCode();
            return;
        }
        if (authStep == AuthStep.TELEGRAM_EMAIL) {
            String email = authEmail.getText().toString().trim();
            if (!isValidEmail(email)) {
                showAuthError("Проверь email");
                return;
            }
            stagedEmail = email.toLowerCase();
            updateAuthStep(AuthStep.TELEGRAM_PASSWORD);
            return;
        }
        if (authStep == AuthStep.TELEGRAM_PASSWORD) {
            submitTelegramPassword();
            return;
        }
        submitExtraIdPassword();
    }

    private void submitSecondaryAuthAction() {
        if (authStep == AuthStep.WELCOME) {
            updateAuthStep(AuthStep.REGISTER_EMAIL);
        } else if (authStep == AuthStep.ANONYMOUS_NICKNAME) {
            updateAuthStep(AuthStep.REGISTER_EMAIL);
        } else if (authStep == AuthStep.LOGIN_EMAIL || authStep == AuthStep.REGISTER_EMAIL) {
            updateAuthStep(AuthStep.ANONYMOUS_NICKNAME);
        } else if (authStep == AuthStep.LOGIN_PASSWORD) {
            updateAuthStep(AuthStep.LOGIN_EMAIL);
        } else if (authStep == AuthStep.REGISTER_PASSWORD) {
            updateAuthStep(AuthStep.REGISTER_EMAIL);
        } else if (authStep == AuthStep.TELEGRAM_ID) {
            updateAuthStep(AuthStep.ANONYMOUS_NICKNAME);
        } else if (authStep == AuthStep.TELEGRAM_CODE) {
            updateAuthStep(AuthStep.TELEGRAM_ID);
        } else if (authStep == AuthStep.TELEGRAM_EMAIL) {
            updateAuthStep(AuthStep.TELEGRAM_CODE);
        } else if (authStep == AuthStep.TELEGRAM_PASSWORD) {
            updateAuthStep(AuthStep.TELEGRAM_EMAIL);
        }
    }

    private void submitAuthModeSwitch() {
        if (authStep == AuthStep.WELCOME || authStep == AuthStep.ANONYMOUS_NICKNAME) {
            updateAuthStep(AuthStep.LOGIN_EMAIL);
        } else if (authStep == AuthStep.LOGIN_EMAIL || authStep == AuthStep.LOGIN_PASSWORD) {
            updateAuthStep(AuthStep.REGISTER_EMAIL);
        } else {
            updateAuthStep(AuthStep.LOGIN_EMAIL);
        }
    }

    private void submitAnonymousProfile() {
        String nickname = authNickname.getText().toString().trim();
        if (nickname.length() < 3 || nickname.length() > 20) {
            showAuthError("Никнейм должен быть от 3 до 20 символов");
            return;
        }
        if (!ensureNetworkForAuth()) {
            return;
        }
        setAuthLoading(true);
        AuthClient.startAnonymous(this, nickname, authCallback(false));
    }

    private void submitTelegramId() {
        String telegramId = authTelegramId.getText().toString().trim();
        if (telegramId.isEmpty()) {
            showAuthError("Введи Telegram ID");
            return;
        }
        try {
            long parsed = Long.parseLong(telegramId);
            if (parsed <= 0L) {
                showAuthError("Telegram ID должен быть числом");
                return;
            }
        } catch (NumberFormatException ignored) {
            showAuthError("Telegram ID должен быть числом");
            return;
        }

        stagedTelegramId = telegramId;
        stagedTelegramCode = "";
        if (!ensureNetworkForAuth()) {
            return;
        }
        setAuthLoading(true);
        AuthClient.requestTelegramCode(this, stagedTelegramId, new AuthClient.SimpleCallback() {
            @Override
            public void onSuccess() {
                runOnUiThread(() -> {
                    setAuthLoading(false);
                    updateAuthStep(AuthStep.TELEGRAM_CODE);
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(() -> {
                    setAuthLoading(false);
                    showAuthError(message);
                });
            }
        });
    }

    private void submitTelegramCode() {
        String code = authTelegramCode.getText().toString().trim();
        if (code.length() != 6) {
            showAuthError("Код должен состоять из 6 цифр");
            return;
        }
        stagedTelegramCode = code;
        updateAuthStep(AuthStep.TELEGRAM_EMAIL);
    }

    private void submitTelegramPassword() {
        String password = authPassword.getText().toString();
        if (password.length() < 8) {
            showAuthError("Пароль должен быть не короче 8 символов");
            return;
        }
        if (!ensureNetworkForAuth()) {
            return;
        }
        setAuthLoading(true);
        AuthClient.completeTelegramTransfer(
                this,
                stagedTelegramId,
                stagedTelegramCode,
                stagedEmail,
                password,
                authCallback(true)
        );
    }

    private void submitExtraIdPassword() {
        String password = authPassword.getText().toString();
        if (password.length() < 8) {
            showAuthError("Пароль должен быть не короче 8 символов");
            return;
        }
        if (!ensureNetworkForAuth()) {
            return;
        }
        setAuthLoading(true);
        if (authStep == AuthStep.REGISTER_PASSWORD) {
            AuthClient.register(this, stagedEmail, password, "", authCallback(true));
        } else {
            AuthClient.login(this, stagedEmail, password, authCallback(true));
        }
    }

    private AuthClient.Callback authCallback(boolean rememberExtraId) {
        String emailForAccount = stagedEmail;
        return new AuthClient.Callback() {
            @Override
            public void onSuccess(AuthClient.AuthResult result) {
                runOnUiThread(() -> {
                    setAuthLoading(false);
                    hideSoftKeyboardAndClearAuthFocus();
                    resetWebViewForNewAuthSession();
                    if (!persistNativeAuthToken(result.token)) {
                        return;
                    }
                    if (rememberExtraId) {
                        ExtraIdAccountStore.saveAccount(
                                MainActivity.this,
                                emailForAccount,
                                result,
                                BaseUrlStore.getBaseUrl(MainActivity.this)
                        );
                    }
                    if (result.regBonus) {
                        Toast.makeText(MainActivity.this, "Бонус: +3 ключа", Toast.LENGTH_SHORT).show();
                    }
                    launchAfterUpdateGate(getIntent());
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(() -> {
                    setAuthLoading(false);
                    showAuthError(message);
                });
            }
        };
    }

    private boolean persistNativeAuthToken(String token) {
        if (token == null || token.trim().isEmpty() || "null".equals(token)) {
            showAuthError("Сервер не вернул токен входа. Попробуй еще раз.");
            return false;
        }
        if (!DeviceRegistrar.saveAuthToken(this, token)) {
            showAuthError("Не удалось сохранить вход на устройстве. Попробуй еще раз.");
            return false;
        }
        if (DeviceRegistrar.getAuthToken(this).isEmpty()) {
            showAuthError("Не удалось прочитать сохраненный вход. Попробуй еще раз.");
            return false;
        }
        return true;
    }

    private boolean isValidEmail(String email) {
        if (email == null) {
            return false;
        }
        String value = email.trim();
        return !value.isEmpty() && Patterns.EMAIL_ADDRESS.matcher(value).matches();
    }

    private void setAuthLoading(boolean loading) {
        authAction.setEnabled(!loading);
        authSecondaryAction.setEnabled(!loading);
        authTelegramAction.setEnabled(!loading);
        authModeSwitch.setEnabled(!loading);
        authAction.setText(loading ? "Подключаемся..." : actionTextForStep(authStep));
        authAction.setAlpha(loading ? 0.7f : 1f);
        authSecondaryAction.setAlpha(loading ? 0.55f : 1f);
        authTelegramAction.setAlpha(loading ? 0.55f : 1f);
        authModeSwitch.setAlpha(loading ? 0.55f : 1f);
    }

    private String actionTextForStep(AuthStep step) {
        if (step == AuthStep.WELCOME) return "Погнали";
        if (step == AuthStep.ANONYMOUS_NICKNAME) return "Вперед";
        if (step == AuthStep.LOGIN_EMAIL || step == AuthStep.REGISTER_EMAIL || step == AuthStep.TELEGRAM_EMAIL) return "Дальше";
        if (step == AuthStep.REGISTER_PASSWORD) return "Создать аккаунт";
        if (step == AuthStep.TELEGRAM_ID) return "Получить код";
        if (step == AuthStep.TELEGRAM_CODE) return "Продолжить";
        if (step == AuthStep.TELEGRAM_PASSWORD) return "Создать ExtraID";
        return "Войти";
    }

    private void showAuthError(String message) {
        authError.setText(message);
        authError.setVisibility(View.VISIBLE);
    }

    private boolean ensureNetworkForAuth() {
        if (hasNetworkConnection()) {
            return true;
        }
        showAuthError("Нужен интернет. VPN для ExtraArena не нужен.");
        return false;
    }

    private View createLoadingView() {
        FrameLayout screen = createShellScreen();
        screen.addView(createTopbar("Клиент", null));

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setOverScrollMode(View.OVER_SCROLL_NEVER);
        screen.addView(scroll, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_HORIZONTAL);
        layout.setPadding(
                dp(shellPaddingDp()),
                dp(topbarHeightDp()),
                dp(shellPaddingDp()),
                dp(isCompactHeight() ? 14 : 22)
        );
        scroll.addView(layout, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        ImageView logo = createAssetImage("extra_mobile/logotype.png", ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams logoParams = new LinearLayout.LayoutParams(
                dp(isCompactHeight() ? 178 : 204),
                dp(isCompactHeight() ? 110 : 144)
        );
        layout.addView(logo, logoParams);

        LoadingRingView ring = new LoadingRingView(this);
        int ringSize = isCompactHeight() ? 154 : 188;
        LinearLayout.LayoutParams ringParams = new LinearLayout.LayoutParams(dp(ringSize), dp(ringSize));
        ringParams.topMargin = dp(isCompactHeight() ? 4 : 8);
        ringParams.bottomMargin = dp(isCompactHeight() ? 6 : 10);
        layout.addView(ring, ringParams);

        TextView title = new TextView(this);
        title.setText("Готовим\nарену");
        title.setTextColor(Color.WHITE);
        title.setTextSize(authTitleSizeSp());
        title.setTypeface(futuraExtraBold);
        title.setGravity(Gravity.LEFT);
        title.setLineSpacing(0, 0.90f);
        title.setShadowLayer(dp(3), 0, dp(2), Color.argb(170, 59, 22, 96));
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        titleParams.topMargin = dp(isCompactHeight() ? 9 : 13);
        layout.addView(title, titleParams);

        TextView lead = new TextView(this);
        lead.setText("Собираем колоду, прогреваем арену и зовем соперника.");
        lead.setTextColor(Color.argb(210, 240, 236, 255));
        lead.setTextSize(bodyTextSizeSp());
        lead.setTypeface(futuraMedium);
        lead.setGravity(Gravity.LEFT);
        lead.setLineSpacing(dp(2), 1.04f);
        LinearLayout.LayoutParams leadParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        leadParams.topMargin = dp(isCompactHeight() ? 10 : 14);
        layout.addView(lead, leadParams);

        View spacer = new View(this);
        layout.addView(spacer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        layout.addView(createLoadingProgress());

        return screen;
    }

    private View createLoadingProgress() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        TextView left = new TextView(this);
        left.setText("Загрузка");
        left.setTextColor(Color.argb(199, 240, 236, 255));
        left.setTextSize(13);
        left.setTypeface(futuraExtraBold);
        left.setAllCaps(true);
        left.setLetterSpacing(0.04f);
        top.addView(left, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView right = new TextView(this);
        right.setText("72%");
        right.setTextColor(Color.argb(199, 240, 236, 255));
        right.setTextSize(13);
        right.setTypeface(futuraExtraBold);
        top.addView(right);
        box.addView(top);

        FrameLayout bar = new FrameLayout(this);
        bar.setPadding(dp(3), dp(3), dp(3), dp(3));
        bar.setBackground(makeProgressTrackBackground());
        LinearLayout.LayoutParams barParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(16)
        );
        barParams.topMargin = dp(12);
        box.addView(bar, barParams);

        View fill = new View(this);
        fill.setBackground(makeProgressFillBackground());
        FrameLayout.LayoutParams fillParams = new FrameLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.MATCH_PARENT
        );
        bar.addView(fill, fillParams);
        bar.post(() -> {
            FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) fill.getLayoutParams();
            int available = Math.max(0, bar.getWidth() - bar.getPaddingLeft() - bar.getPaddingRight());
            params.width = Math.round(available * 0.72f);
            fill.setLayoutParams(params);
        });

        TextView tips = new TextView(this);
        tips.setText("Совет: держи добивку до конца хода — иногда одна карта решает всю партию.");
        tips.setTextColor(Color.argb(199, 240, 236, 255));
        tips.setTextSize(isCompactHeight() ? 13 : 14);
        tips.setTypeface(futuraMedium);
        tips.setLineSpacing(dp(2), 1f);
        tips.setSingleLine(false);
        tips.setPadding(dp(14), dp(isCompactHeight() ? 11 : 13), dp(14), dp(isCompactHeight() ? 12 : 13));
        tips.setBackground(makeTipBackground());
        LinearLayout.LayoutParams tipsParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        tipsParams.topMargin = dp(isCompactHeight() ? 9 : 12);
        box.addView(tips, tipsParams);

        return box;
    }

    private View createErrorView() {
        FrameLayout screen = createShellScreen();
        screen.addView(createTopbar("Клиент", null));

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_HORIZONTAL);
        layout.setPadding(
                dp(shellPaddingDp()),
                dp(topbarHeightDp() + (isCompactHeight() ? 14 : 24)),
                dp(shellPaddingDp()),
                dp(isCompactHeight() ? 18 : 26)
        );
        screen.addView(layout, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        OfflineDinoView dino = new OfflineDinoView(this);
        dino.setContentDescription("Мини-игра без интернета");
        LinearLayout.LayoutParams dinoParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
        );
        dinoParams.bottomMargin = dp(isCompactHeight() ? 12 : 18);
        layout.addView(dino, dinoParams);

        TextView title = new TextView(this);
        title.setText("Сервер недоступен");
        title.setTextColor(Color.rgb(246, 241, 255));
        title.setTextSize(isCompactHeight() ? 34 : 42);
        title.setTypeface(futuraExtraBold);
        title.setGravity(Gravity.CENTER);
        title.setShadowLayer(dp(3), 0, dp(2), Color.argb(170, 59, 22, 96));
        layout.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Похоже, соединение с ареной не прошло. Проверь интернет или попробуй еще раз чуть позже.");
        subtitle.setTextColor(Color.argb(215, 240, 236, 255));
        subtitle.setTextSize(bodyTextSizeSp());
        subtitle.setTypeface(futuraMedium);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setLineSpacing(dp(2), 1.04f);
        LinearLayout.LayoutParams subtitleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        subtitleParams.topMargin = dp(isCompactHeight() ? 8 : 12);
        layout.addView(subtitle, subtitleParams);

        TextView retry = new TextView(this);
        retry.setText("Повторить");
        stylePrimaryButton(retry);
        retry.setOnClickListener(v -> launchAfterUpdateGate(getIntent()));
        LinearLayout.LayoutParams retryParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        retryParams.topMargin = dp(isCompactHeight() ? 18 : 24);
        layout.addView(retry, retryParams);

        TextView hint = new TextView(this);
        hint.setText("Если интернет есть, сервер может просыпаться. Нажми «Повторить» через минуту.");
        hint.setTextColor(Color.rgb(255, 218, 157));
        hint.setTextSize(isCompactHeight() ? 12 : 13);
        hint.setTypeface(futuraBold);
        hint.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams hintParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        hintParams.topMargin = dp(10);
        layout.addView(hint, hintParams);

        return screen;
    }

    private GradientDrawable makeBackground() {
        return new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{Color.rgb(24, 10, 50), Color.rgb(15, 10, 26), EA_BG}
        );
    }

    private GradientDrawable makeBrandDotBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{EA_ACCENT, EA_PINK, Color.rgb(124, 92, 191)}
        );
        drawable.setCornerRadius(dp(9));
        return drawable;
    }

    private GradientDrawable makeKickerBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(31, 244, 114, 182));
        drawable.setCornerRadius(dp(999));
        drawable.setStroke(dp(1), Color.argb(97, 244, 114, 182));
        return drawable;
    }

    private GradientDrawable makeWelcomeArtBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(160, 26, 16, 48));
        drawable.setCornerRadius(dp(34));
        return drawable;
    }

    private GradientDrawable makeFieldBackground(boolean active) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(199, 16, 10, 30));
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(1), active ? Color.argb(224, 245, 146, 30) : Color.argb(148, 122, 111, 160));
        return drawable;
    }

    private GradientDrawable makePrimaryButtonBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{Color.rgb(255, 189, 69), EA_ACCENT, Color.rgb(217, 117, 16)}
        );
        drawable.setCornerRadius(dp(18));
        return drawable;
    }

    private GradientDrawable makeSecondaryButtonBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(184, 45, 31, 82));
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(1), Color.argb(140, 122, 111, 160));
        return drawable;
    }

    private GradientDrawable makeTelegramButtonBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{Color.rgb(42, 171, 238), Color.rgb(34, 158, 217)}
        );
        drawable.setCornerRadius(dp(18));
        return drawable;
    }

    private GradientDrawable makeStepBackground(boolean active) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setCornerRadius(dp(999));
        drawable.setColor(active ? EA_ACCENT : Color.argb(107, 122, 111, 160));
        return drawable;
    }

    private GradientDrawable makeExtraIdBadgeBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{EA_ACCENT, EA_PINK, Color.rgb(124, 92, 191)}
        );
        drawable.setCornerRadius(dp(36));
        return drawable;
    }

    private GradientDrawable makeTelegramBadgeBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{Color.rgb(42, 171, 238), Color.rgb(34, 158, 217), Color.rgb(78, 205, 255)}
        );
        drawable.setCornerRadius(dp(36));
        return drawable;
    }

    private GradientDrawable makeAccountCardBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{Color.argb(31, 45, 212, 191), Color.argb(168, 45, 31, 82)}
        );
        drawable.setCornerRadius(dp(22));
        drawable.setStroke(dp(1), Color.argb(66, 45, 212, 191));
        return drawable;
    }

    private GradientDrawable makeProgressTrackBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(209, 16, 10, 30));
        drawable.setCornerRadius(dp(999));
        drawable.setStroke(dp(1), Color.argb(107, 122, 111, 160));
        return drawable;
    }

    private GradientDrawable makeProgressFillBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{EA_ACCENT, EA_PINK, Color.rgb(45, 212, 191)}
        );
        drawable.setCornerRadius(dp(999));
        return drawable;
    }

    private GradientDrawable makeTipBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.argb(173, 26, 16, 48));
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(1), Color.argb(82, 122, 111, 160));
        return drawable;
    }

    private GradientDrawable makePanelBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{Color.rgb(22, 14, 38), Color.rgb(12, 8, 24)}
        );
        drawable.setCornerRadius(dp(22));
        drawable.setStroke(dp(1), Color.rgb(79, 55, 126));
        return drawable;
    }

    private GradientDrawable makeProfileBackground(boolean selected) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(selected ? Color.rgb(43, 29, 72) : Color.rgb(22, 14, 38));
        drawable.setCornerRadius(dp(14));
        drawable.setStroke(dp(1), selected ? Color.rgb(255, 138, 61) : Color.rgb(64, 43, 113));
        return drawable;
    }

    private GradientDrawable makeInputBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.rgb(18, 11, 36));
        drawable.setCornerRadius(dp(12));
        drawable.setStroke(dp(1), Color.rgb(64, 43, 113));
        return drawable;
    }

    private GradientDrawable makeButtonBackground(int color) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(14));
        return drawable;
    }

    private boolean hasNetworkConnection() {
        ConnectivityManager manager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (manager == null) {
            return true;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Network network = manager.getActiveNetwork();
            if (network == null) {
                return false;
            }
            NetworkCapabilities capabilities = manager.getNetworkCapabilities(network);
            return capabilities != null && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET);
        }
        NetworkInfo info = manager.getActiveNetworkInfo();
        return info != null && info.isConnected();
    }

    private void showConnectivityError() {
        arenaLoadBlockedByConnectivity = true;
        loadingPausedForProfileSwitcher = false;
        if (webView != null) {
            try {
                webView.stopLoading();
            } catch (Exception ignored) {
            }
        }
        if (webView != null) {
            webView.setVisibility(View.INVISIBLE);
        }
        if (loadingView != null) {
            loadingView.setVisibility(View.GONE);
        }
        if (authView != null) {
            authView.setVisibility(View.GONE);
        }
        setLoadingDevHotspotVisible(false);
        errorView.setVisibility(View.VISIBLE);
    }

    private void fetchFcmToken() {
        try {
            FirebaseMessaging.getInstance().getToken()
                    .addOnSuccessListener(token -> DeviceRegistrar.saveFcmToken(this, token));
        } catch (IllegalStateException ignored) {
            // Local debug builds may run without google-services.json.
        }
    }

    private void maybeShowNotificationOptInPrompt() {
        if (updateBlocked || isFinishing()) {
            return;
        }
        if (DeviceRegistrar.getAuthToken(this).isEmpty()) {
            return;
        }
        if (webView == null || webView.getVisibility() != View.VISIBLE) {
            return;
        }
        if (authView != null && authView.getVisibility() == View.VISIBLE) {
            return;
        }
        if (hasNotificationPermission()) {
            getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE)
                    .edit()
                    .putBoolean(KEY_NOTIFICATION_PROMPT_ACCEPTED, true)
                    .apply();
            DeviceRegistrar.registerIfReady(this);
            return;
        }
        SharedPreferences prefs = getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE);
        long lastShown = prefs.getLong(KEY_NOTIFICATION_PROMPT_LAST_SHOWN, 0L);
        if (lastShown > 0L && System.currentTimeMillis() - lastShown < NOTIFICATION_PROMPT_COOLDOWN_MS) {
            return;
        }
        showNotificationOptInDialog();
    }

    private boolean hasNotificationPermission() {
        return Build.VERSION.SDK_INT < 33
                || checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
    }

    private void showNotificationOptInDialog() {
        if (notificationPromptDialog != null && notificationPromptDialog.isShowing()) {
            return;
        }
        getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE)
                .edit()
                .putLong(KEY_NOTIFICATION_PROMPT_LAST_SHOWN, System.currentTimeMillis())
                .apply();

        Dialog dialog = new Dialog(this);
        notificationPromptDialog = dialog;
        dialog.setCancelable(false);
        dialog.setCanceledOnTouchOutside(false);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(22), dp(20), dp(18));
        panel.setBackground(makePanelBackground());

        TextView badge = new TextView(this);
        badge.setText("!");
        badge.setGravity(Gravity.CENTER);
        badge.setTextColor(Color.rgb(9, 5, 18));
        badge.setTextSize(24);
        badge.setTypeface(Typeface.DEFAULT_BOLD);
        badge.setBackground(makeButtonBackground(Color.rgb(61, 219, 198)));
        LinearLayout.LayoutParams badgeParams = new LinearLayout.LayoutParams(dp(48), dp(48));
        badgeParams.gravity = Gravity.CENTER_HORIZONTAL;
        badgeParams.bottomMargin = dp(12);
        panel.addView(badge, badgeParams);

        TextView title = createDialogTitle("Не пропусти награды");
        panel.addView(title);
        TextView subtitle = createDialogSubtitle(
                "ExtraArena будет напоминать о генераторе ключей, важных событиях сквада и обновлениях без Telegram и без VPN."
        );
        panel.addView(subtitle);

        addNotificationBenefit(panel, "Ключи", "Узнаешь, когда генератор снова готов.");
        addNotificationBenefit(panel, "Сквад", "Не пропустишь заявки, роли и Boost-события.");
        addNotificationBenefit(panel, "Обновления", "Сразу увидишь, когда вышла новая версия APK.");

        TextView enable = createDialogButton("Включить уведомления", true);
        enable.setOnClickListener(v -> {
            dialog.dismiss();
            acceptNotificationPrompt();
        });
        LinearLayout.LayoutParams enableParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        enableParams.topMargin = dp(16);
        panel.addView(enable, enableParams);

        TextView later = createDialogButton("Не сейчас", false);
        later.setOnClickListener(v -> {
            getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE)
                    .edit()
                    .putBoolean(KEY_NOTIFICATION_PROMPT_ACCEPTED, false)
                    .putLong(KEY_NOTIFICATION_PROMPT_LAST_SHOWN, System.currentTimeMillis())
                    .apply();
            dialog.dismiss();
        });
        panel.addView(later, inputParams());

        dialog.setContentView(panel);
        dialog.setOnDismissListener(d -> {
            if (notificationPromptDialog == dialog) {
                notificationPromptDialog = null;
            }
        });
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(430)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawableResource(android.R.color.transparent);
        }
        dialog.show();
    }

    private void addNotificationBenefit(LinearLayout panel, String title, String text) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(12), dp(10), dp(12), dp(10));
        row.setBackground(makeInputBackground());

        View dot = new View(this);
        dot.setBackground(makeButtonBackground(Color.rgb(245, 146, 30)));
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dp(9), dp(9));
        dotParams.rightMargin = dp(10);
        row.addView(dot, dotParams);

        LinearLayout copy = new LinearLayout(this);
        copy.setOrientation(LinearLayout.VERTICAL);
        TextView name = new TextView(this);
        name.setText(title);
        name.setTextColor(Color.rgb(246, 241, 255));
        name.setTextSize(13);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        copy.addView(name);
        TextView body = new TextView(this);
        body.setText(text);
        body.setTextColor(Color.rgb(183, 169, 210));
        body.setTextSize(12);
        body.setLineSpacing(2, 1f);
        copy.addView(body);
        row.addView(copy, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        rowParams.topMargin = dp(7);
        panel.addView(row, rowParams);
    }

    private void acceptNotificationPrompt() {
        if (Build.VERSION.SDK_INT >= 33 && !hasNotificationPermission()) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 7001);
            return;
        }
        getSharedPreferences(NOTIFICATION_PROMPT_PREFS, MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_NOTIFICATION_PROMPT_ACCEPTED, true)
                .putLong(KEY_NOTIFICATION_PROMPT_LAST_SHOWN, System.currentTimeMillis())
                .apply();
        fetchFcmToken();
        DeviceRegistrar.registerIfReady(this);
        Toast.makeText(this, "Уведомления включены", Toast.LENGTH_SHORT).show();
    }

    private void showExtraIdManagerDialog() {
        if (extraIdManagerDialog != null && extraIdManagerDialog.isShowing()) {
            return;
        }
        Dialog dialog = new Dialog(this);
        extraIdManagerDialog = dialog;
        dialog.setCanceledOnTouchOutside(true);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.setBackgroundColor(Color.TRANSPARENT);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(22), dp(20), dp(18));
        panel.setBackground(makePanelBackground());
        scroll.addView(panel, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        panel.addView(createDialogTitle("ExtraID"));
        panel.addView(createDialogSubtitle("Управляй учетными записями клиента без Telegram: переключайся, добавляй новую или создай ExtraID для текущей игры."));

        String baseUrl = BaseUrlStore.getBaseUrl(this);
        String activeToken = DeviceRegistrar.getAuthToken(this);
        List<ExtraIdAccountStore.ExtraIdAccount> accounts = ExtraIdAccountStore.getAccounts(this, baseUrl);

        if (accounts.isEmpty()) {
            TextView empty = createDialogSubtitle("На этом сервере пока нет сохраненных ExtraID.");
            panel.addView(empty);
        } else {
            for (ExtraIdAccountStore.ExtraIdAccount account : accounts) {
                boolean active = !activeToken.isEmpty() && activeToken.equals(account.token);
                panel.addView(createNativeAccountRow(dialog, account, active), inputParams());
            }
        }

        TextView add = createDialogButton("Добавить аккаунт", true);
        add.setOnClickListener(v -> showAddExtraIdAccountDialog(false));
        LinearLayout.LayoutParams addParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        addParams.topMargin = dp(16);
        panel.addView(add, addParams);

        TextView create = createDialogButton("Создать ExtraID", false);
        create.setOnClickListener(v -> showAddExtraIdAccountDialog(true));
        panel.addView(create, inputParams());

        if (!activeToken.isEmpty()) {
            TextView logout = createDialogButton("Выйти на этом устройстве", false);
            logout.setOnClickListener(v -> {
                dialog.dismiss();
                clearWebAuthState();
                DeviceRegistrar.clearAuthToken(MainActivity.this);
                showAuth();
            });
            panel.addView(logout, inputParams());
        }

        TextView close = createDialogButton("Закрыть", false);
        close.setOnClickListener(v -> dialog.dismiss());
        panel.addView(close, inputParams());

        dialog.setContentView(scroll);
        dialog.setOnDismissListener(d -> {
            if (extraIdManagerDialog == dialog) {
                extraIdManagerDialog = null;
            }
        });
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(430)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private View createNativeAccountRow(Dialog ownerDialog, ExtraIdAccountStore.ExtraIdAccount account, boolean active) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        card.setBackground(makeProfileBackground(active));
        addHapticTouch(card, "selection");

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);

        LinearLayout text = new LinearLayout(this);
        text.setOrientation(LinearLayout.VERTICAL);
        TextView name = new TextView(this);
        String display = account.displayId == null || account.displayId.isEmpty() ? account.email : account.displayId;
        name.setText(display);
        name.setTextColor(Color.rgb(246, 241, 255));
        name.setTextSize(15);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        text.addView(name);

        TextView email = new TextView(this);
        email.setText(account.email);
        email.setTextColor(Color.rgb(183, 169, 210));
        email.setTextSize(12);
        text.addView(email);
        top.addView(text, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView chip = createSmallDialogButton(active ? "Активен" : "Выбрать");
        chip.setTextColor(active ? Color.rgb(173, 255, 244) : Color.rgb(255, 218, 157));
        chip.setOnClickListener(v -> activateNativeAccount(ownerDialog, account.email));
        top.addView(chip);
        card.addView(top);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setGravity(Gravity.RIGHT);
        TextView remove = createSmallDialogButton("Убрать");
        remove.setOnClickListener(v -> {
            ExtraIdAccountStore.removeAccount(
                    MainActivity.this,
                    BaseUrlStore.getBaseUrl(MainActivity.this),
                    account.email,
                    DeviceRegistrar.getAuthToken(MainActivity.this)
            );
            ownerDialog.dismiss();
            if (DeviceRegistrar.getAuthToken(MainActivity.this).isEmpty()) {
                showAuth();
            } else {
                showExtraIdManagerDialog();
            }
        });
        actions.addView(remove);
        LinearLayout.LayoutParams actionsParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        actionsParams.topMargin = dp(10);
        card.addView(actions, actionsParams);

        card.setOnClickListener(v -> activateNativeAccount(ownerDialog, account.email));
        return card;
    }

    private void activateNativeAccount(Dialog ownerDialog, String email) {
        boolean ok = ExtraIdAccountStore.activateAccount(
                this,
                BaseUrlStore.getBaseUrl(this),
                email
        );
        if (!ok) {
            Toast.makeText(this, "Не удалось переключить аккаунт", Toast.LENGTH_SHORT).show();
            return;
        }
        ownerDialog.dismiss();
        resetWebViewForNewAuthSession();
        launchAfterUpdateGate(getIntent());
    }

    private void showAddExtraIdAccountDialog(boolean createMode) {
        if (addExtraIdDialog != null && addExtraIdDialog.isShowing()) {
            addExtraIdDialog.dismiss();
        }
        Dialog dialog = new Dialog(this);
        addExtraIdDialog = dialog;
        dialog.setCanceledOnTouchOutside(true);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(22), dp(20), dp(18));
        panel.setBackground(makePanelBackground());

        panel.addView(createDialogTitle(createMode ? "Создать ExtraID" : "Добавить ExtraID"));
        panel.addView(createDialogSubtitle(createMode
                ? "Укажи почту и пароль. Аккаунт сохранится в клиенте и появится в переключателе."
                : "Войди в существующий ExtraID, чтобы быстро переключаться между учетками."));

        EditText email = createDialogInput("Email", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS);
        panel.addView(email, inputParams());
        EditText password = createDialogInput("Пароль", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        panel.addView(password, inputParams());
        EditText nickname = null;
        if (createMode) {
            nickname = createDialogInput("Никнейм в игре", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_NORMAL);
            nickname.setFilters(new InputFilter[]{new InputFilter.LengthFilter(20)});
            panel.addView(nickname, inputParams());
        }

        TextView error = new TextView(this);
        error.setTextColor(Color.rgb(255, 142, 142));
        error.setTextSize(12);
        error.setGravity(Gravity.CENTER);
        error.setVisibility(View.GONE);
        panel.addView(error, inputParams());

        TextView submit = createDialogButton(createMode ? "Создать и войти" : "Войти", true);
        EditText nicknameInput = nickname;
        submit.setOnClickListener(v -> {
            String cleanEmail = email.getText().toString().trim().toLowerCase();
            String pass = password.getText().toString();
            String nick = nicknameInput == null ? "" : nicknameInput.getText().toString().trim();
            if (!isValidEmail(cleanEmail)) {
                showDialogError(error, "Проверь email.");
                return;
            }
            if (pass.length() < 8) {
                showDialogError(error, "Пароль должен быть не короче 8 символов.");
                return;
            }
            submit.setEnabled(false);
            submit.setAlpha(0.65f);
            submit.setText("Подключаемся...");
            AuthClient.Callback callback = nativeExtraIdCallback(dialog, cleanEmail, submit, createMode ? "Создать и войти" : "Войти", error);
            if (createMode) {
                AuthClient.register(MainActivity.this, cleanEmail, pass, nick, callback);
            } else {
                AuthClient.login(MainActivity.this, cleanEmail, pass, callback);
            }
        });
        LinearLayout.LayoutParams submitParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        submitParams.topMargin = dp(14);
        panel.addView(submit, submitParams);

        TextView cancel = createDialogButton("Отмена", false);
        cancel.setOnClickListener(v -> dialog.dismiss());
        panel.addView(cancel, inputParams());

        dialog.setContentView(panel);
        dialog.setOnDismissListener(d -> {
            if (addExtraIdDialog == dialog) {
                addExtraIdDialog = null;
            }
        });
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(410)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private AuthClient.Callback nativeExtraIdCallback(
            Dialog dialog,
            String email,
            TextView submit,
            String submitText,
            TextView error
    ) {
        return new AuthClient.Callback() {
            @Override
            public void onSuccess(AuthClient.AuthResult result) {
                runOnUiThread(() -> {
                    submit.setEnabled(true);
                    submit.setAlpha(1f);
                    submit.setText(submitText);
                    if (!saveNativeAuthTokenForDialog(result.token, error)) {
                        return;
                    }
                    ExtraIdAccountStore.saveAccount(
                            MainActivity.this,
                            email,
                            result,
                            BaseUrlStore.getBaseUrl(MainActivity.this)
                    );
                    if (extraIdManagerDialog != null) {
                        extraIdManagerDialog.dismiss();
                    }
                    dialog.dismiss();
                    resetWebViewForNewAuthSession();
                    launchAfterUpdateGate(getIntent());
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(() -> {
                    submit.setEnabled(true);
                    submit.setAlpha(1f);
                    submit.setText(submitText);
                    showDialogError(error, message);
                });
            }
        };
    }

    private boolean saveNativeAuthTokenForDialog(String token, TextView error) {
        if (token == null || token.trim().isEmpty() || "null".equals(token)) {
            showDialogError(error, "Сервер не вернул токен входа. Попробуй еще раз.");
            return false;
        }
        if (!DeviceRegistrar.saveAuthToken(this, token)) {
            showDialogError(error, "Не удалось сохранить вход на устройстве. Попробуй еще раз.");
            return false;
        }
        if (DeviceRegistrar.getAuthToken(this).isEmpty()) {
            showDialogError(error, "Не удалось прочитать сохраненный вход. Попробуй еще раз.");
            return false;
        }
        return true;
    }

    private EditText createDialogInput(String hint, int inputType) {
        EditText input = createInput(hint, inputType);
        input.setTextSize(16);
        input.setTypeface(Typeface.DEFAULT_BOLD);
        input.setPadding(dp(14), dp(11), dp(14), dp(11));
        input.setBackground(makeInputBackground());
        return input;
    }

    private void showDialogError(TextView error, String message) {
        if (error == null) {
            return;
        }
        error.setText(message == null || message.trim().isEmpty() ? "Что-то пошло не так." : message);
        error.setVisibility(View.VISIBLE);
    }

    private void showBadConnectionSplash(int latencyMs) {
        if (badConnectionDialog != null && badConnectionDialog.isShowing()) {
            return;
        }
        Dialog dialog = new Dialog(this);
        badConnectionDialog = dialog;
        dialog.setCanceledOnTouchOutside(true);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.HORIZONTAL);
        panel.setGravity(Gravity.CENTER_VERTICAL);
        panel.setPadding(dp(14), dp(10), dp(14), dp(10));
        GradientDrawable bg = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{Color.rgb(58, 39, 11), Color.rgb(30, 19, 9)}
        );
        bg.setCornerRadius(dp(999));
        bg.setStroke(dp(1), Color.argb(112, 251, 191, 36));
        panel.setBackground(bg);
        addHapticTouch(panel, "selection");

        LinearLayout bars = new LinearLayout(this);
        bars.setOrientation(LinearLayout.HORIZONTAL);
        bars.setGravity(Gravity.BOTTOM);
        int[] heights = {dp(7), dp(11), dp(15)};
        for (int i = 0; i < heights.length; i++) {
            View bar = new View(this);
            bar.setBackground(makeButtonBackground(i == 2 ? Color.rgb(245, 158, 11) : Color.rgb(251, 191, 36)));
            LinearLayout.LayoutParams barParams = new LinearLayout.LayoutParams(dp(4), heights[i]);
            barParams.rightMargin = dp(3);
            bars.addView(bar, barParams);
        }
        panel.addView(bars);

        TextView label = new TextView(this);
        label.setText(latencyMs > 0 ? "Плохое соединение · " + latencyMs + " мс" : "Плохое соединение");
        label.setTextColor(Color.rgb(253, 230, 138));
        label.setTextSize(12);
        label.setTypeface(Typeface.DEFAULT_BOLD);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        labelParams.leftMargin = dp(8);
        panel.addView(label, labelParams);

        panel.setOnClickListener(v -> {
            dialog.dismiss();
            evaluateJsSilently("try{window.__extraArenaDismissBadConnection&&window.__extraArenaDismissBadConnection()}catch(e){}");
        });

        dialog.setContentView(panel);
        dialog.setOnDismissListener(d -> {
            if (badConnectionDialog == dialog) {
                badConnectionDialog = null;
            }
        });
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.clearFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND);
                WindowManager.LayoutParams attrs = shownWindow.getAttributes();
                attrs.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
                attrs.y = dp(18);
                shownWindow.setAttributes(attrs);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(32), dp(300)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private void showNativeAfkDialog() {
        if (nativeAfkDialog != null && nativeAfkDialog.isShowing()) {
            return;
        }
        Dialog dialog = new Dialog(this);
        nativeAfkDialog = dialog;
        dialog.setCancelable(false);
        dialog.setCanceledOnTouchOutside(false);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(20), dp(22), dp(20), dp(18));
        panel.setBackground(makePanelBackground());

        TextView badge = new TextView(this);
        badge.setText("⏱");
        badge.setGravity(Gravity.CENTER);
        badge.setTextSize(28);
        badge.setBackground(makeButtonBackground(Color.rgb(245, 146, 30)));
        LinearLayout.LayoutParams badgeParams = new LinearLayout.LayoutParams(dp(54), dp(54));
        badgeParams.gravity = Gravity.CENTER_HORIZONTAL;
        badgeParams.bottomMargin = dp(12);
        panel.addView(badge, badgeParams);

        panel.addView(createDialogTitle("Похоже, ты отошел"));
        panel.addView(createDialogSubtitle("Мы остановили фоновые проверки, чтобы не мучить соединение. Перезапусти клиент и возвращайся в игру свежим входом."));

        TextView restart = createDialogButton("Перезапустить", true);
        restart.setOnClickListener(v -> {
            dialog.dismiss();
            evaluateJsSilently("try{window.__extraArenaRestartClient&&window.__extraArenaRestartClient()}catch(e){location.reload()}");
        });
        panel.addView(restart, inputParams());

        dialog.setContentView(panel);
        dialog.setOnDismissListener(d -> {
            if (nativeAfkDialog == dialog) {
                nativeAfkDialog = null;
            }
        });
        dialog.setOnShowListener(d -> {
            Window shownWindow = dialog.getWindow();
            if (shownWindow != null) {
                shownWindow.setBackgroundDrawableResource(android.R.color.transparent);
                shownWindow.setLayout(
                        Math.min(getResources().getDisplayMetrics().widthPixels - dp(28), dp(380)),
                        WindowManager.LayoutParams.WRAP_CONTENT
                );
            }
        });
        dialog.show();
    }

    private void evaluateJsSilently(String script) {
        if (webView == null) {
            return;
        }
        try {
            webView.evaluateJavascript(script, null);
        } catch (Exception ignored) {
        }
    }

    void emitRuStorePaymentEvent(JSONObject event) {
        if (event == null) {
            return;
        }
        String json = event.toString();
        runOnUiThread(() -> evaluateJsSilently(
                "(function(){try{"
                        + "var detail=" + json + ";"
                        + "if(window.ExtraArenaRuStorePaymentEvent){window.ExtraArenaRuStorePaymentEvent(detail);}"
                        + "window.dispatchEvent(new CustomEvent('ExtraArenaRuStorePayment',{detail:detail}));"
                        + "}catch(e){console.warn('RuStore event bridge failed',e);}})();"
        ));
    }

    private boolean isRuStoreBuild() {
        return "rustore".equals(BuildConfig.DISTRIBUTION_CHANNEL);
    }

    private void openExternal(String url) {
        if (url == null || url.trim().isEmpty()) {
            return;
        }
        try {
            Uri uri = Uri.parse(url.trim());
            String scheme = uri.getScheme();
            if (scheme == null
                    || !("https".equalsIgnoreCase(scheme)
                    || "http".equalsIgnoreCase(scheme)
                    || "tg".equalsIgnoreCase(scheme)
                    || "telegram".equalsIgnoreCase(scheme))) {
                return;
            }
            startActivity(new Intent(Intent.ACTION_VIEW, uri));
        } catch (ActivityNotFoundException | IllegalArgumentException ignored) {
        }
    }

    private void shareText(String text, String url, String title) {
        String cleanText = text == null ? "" : text.trim();
        String cleanUrl = url == null ? "" : url.trim();
        String cleanTitle = title == null || title.trim().isEmpty() ? "ExtraArena" : title.trim();
        if (cleanText.isEmpty() && cleanUrl.isEmpty()) {
            return;
        }
        String body = cleanText;
        if (!cleanUrl.isEmpty() && !body.contains(cleanUrl)) {
            body = body.isEmpty() ? cleanUrl : body + "\n\n" + cleanUrl;
        }
        Intent sendIntent = new Intent(Intent.ACTION_SEND);
        sendIntent.setType("text/plain");
        sendIntent.putExtra(Intent.EXTRA_SUBJECT, cleanTitle);
        sendIntent.putExtra(Intent.EXTRA_TEXT, body);
        try {
            startActivity(Intent.createChooser(sendIntent, cleanTitle));
        } catch (ActivityNotFoundException | IllegalArgumentException ignored) {
        }
    }

    private void vibrate() {
        vibrate("light");
    }

    private boolean isHapticsEnabled() {
        return getSharedPreferences(HAPTICS_PREFS, MODE_PRIVATE)
                .getBoolean(KEY_HAPTICS_ENABLED, true);
    }

    private void setHapticsEnabled(boolean enabled) {
        getSharedPreferences(HAPTICS_PREFS, MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_HAPTICS_ENABLED, enabled)
                .apply();
    }

    private void addHapticTouch(View view, String style) {
        if (view == null) {
            return;
        }
        view.setHapticFeedbackEnabled(true);
        view.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN && v.isEnabled()) {
                vibrate(style);
            }
            return false;
        });
    }

    private void vibrate(String style) {
        if (!isHapticsEnabled()) {
            return;
        }
        Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (vibrator == null) {
            return;
        }
        try {
            if (!vibrator.hasVibrator()) {
                return;
            }
        } catch (Exception ignored) {
        }
        String normalized = style == null ? "light" : style.trim().toLowerCase();
        long[] pattern = null;
        int amplitude = 96;
        long duration = 14L;
        if ("selection".equals(normalized)) {
            duration = 8L;
            amplitude = 58;
        } else if ("medium".equals(normalized)) {
            duration = 22L;
            amplitude = 142;
        } else if ("heavy".equals(normalized)) {
            duration = 34L;
            amplitude = 210;
        } else if ("success".equals(normalized)) {
            pattern = new long[]{0, 18, 42, 24};
            amplitude = 150;
        } else if ("warning".equals(normalized)) {
            pattern = new long[]{0, 28, 34, 28};
            amplitude = 185;
        } else if ("error".equals(normalized)) {
            pattern = new long[]{0, 34, 28, 38, 28, 34};
            amplitude = 220;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (pattern != null) {
                int[] amplitudes = new int[pattern.length];
                for (int i = 0; i < amplitudes.length; i++) {
                    amplitudes[i] = i % 2 == 0 ? 0 : amplitude;
                }
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, amplitudes, -1));
            } else {
                vibrator.vibrate(VibrationEffect.createOneShot(duration, amplitude));
            }
        } else {
            if (pattern != null) {
                vibrator.vibrate(pattern, -1);
            } else {
                vibrator.vibrate(duration);
            }
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private int screenHeightDp() {
        return Math.round(getResources().getDisplayMetrics().heightPixels / getResources().getDisplayMetrics().density);
    }

    private boolean isCompactHeight() {
        return screenHeightDp() < 820;
    }

    private boolean isVeryCompactHeight() {
        return screenHeightDp() < 720;
    }

    private int shellPaddingDp() {
        return isCompactHeight() ? 16 : 22;
    }

    private int topbarHeightDp() {
        return isCompactHeight() ? 60 : 70;
    }

    private int fieldGapDp() {
        return isCompactHeight() ? 8 : 10;
    }

    private int authTitleSizeSp() {
        return isVeryCompactHeight() ? 36 : (isCompactHeight() ? 40 : 46);
    }

    private int bodyTextSizeSp() {
        return isCompactHeight() ? 15 : 17;
    }

    private int buttonHeightDp() {
        return isCompactHeight() ? 50 : 58;
    }

    private int telegramButtonHeightDp() {
        return isCompactHeight() ? 48 : 54;
    }

    private int welcomeStageHeightDp() {
        if (isVeryCompactHeight()) {
            return 218;
        }
        return isCompactHeight() ? 252 : 344;
    }

    private int welcomeArtHeightDp() {
        if (isVeryCompactHeight()) {
            return 188;
        }
        return isCompactHeight() ? 218 : 292;
    }

    private int welcomeCarouselHeightDp() {
        return isCompactHeight() ? 104 : 128;
    }

    private int cardWidthDp() {
        return isCompactHeight() ? 58 : 72;
    }

    private int cardHeightDp() {
        return isCompactHeight() ? 88 : 108;
    }

    private void loadTypefaces() {
        futuraMedium = loadTypeface("extra_mobile/fonts/FuturaPT-Medium.ttf", Typeface.DEFAULT);
        futuraBold = loadTypeface("extra_mobile/fonts/FuturaPT-Bold.ttf", Typeface.DEFAULT_BOLD);
        futuraExtraBold = loadTypeface("extra_mobile/fonts/FuturaPT-ExtraBold.ttf", Typeface.DEFAULT_BOLD);
    }

    private Typeface loadTypeface(String path, Typeface fallback) {
        try {
            return Typeface.createFromAsset(getAssets(), path);
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private ImageView createAssetImage(String path, ImageView.ScaleType scaleType) {
        ImageView image = new ImageView(this);
        image.setScaleType(scaleType);
        image.setAdjustViewBounds(false);
        try (InputStream stream = getAssets().open(path, AssetManager.ACCESS_BUFFER)) {
            Bitmap bitmap = BitmapFactory.decodeStream(stream);
            if (bitmap != null) {
                image.setImageBitmap(bitmap);
            }
        } catch (Exception ignored) {
            image.setBackgroundColor(Color.argb(80, 45, 31, 82));
        }
        return image;
    }

    private final class ShellBackgroundView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        ShellBackgroundView(Context context) {
            super(context);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            int width = getWidth();
            int height = getHeight();
            paint.setShader(new LinearGradient(
                    0, 0, 0, height,
                    new int[]{Color.rgb(24, 10, 50), Color.rgb(15, 10, 26), EA_BG},
                    new float[]{0f, 0.46f, 1f},
                    Shader.TileMode.CLAMP
            ));
            canvas.drawRect(0, 0, width, height, paint);

            paint.setShader(new RadialGradient(width * 0.20f, height * 0.12f, width * 0.34f,
                    Color.argb(51, 244, 114, 182), Color.TRANSPARENT, Shader.TileMode.CLAMP));
            canvas.drawCircle(width * 0.20f, height * 0.12f, width * 0.34f, paint);

            paint.setShader(new RadialGradient(width * 0.82f, height * 0.16f, width * 0.34f,
                    Color.argb(41, 96, 165, 250), Color.TRANSPARENT, Shader.TileMode.CLAMP));
            canvas.drawCircle(width * 0.82f, height * 0.16f, width * 0.34f, paint);

            paint.setShader(null);
            paint.setStrokeWidth(1f);
            paint.setColor(Color.argb(9, 255, 255, 255));
            int step = dp(34);
            for (int x = -step; x < width + step; x += step) {
                canvas.drawLine(x, 0, x + dp(150), height, paint);
            }
            for (int y = 0; y < height; y += step) {
                canvas.drawLine(0, y, width, y + dp(42), paint);
            }

            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.WHITE);
            float[][] stars = {
                    {0.16f, 0.09f, 1.0f}, {0.78f, 0.18f, 0.8f}, {0.11f, 0.31f, 0.7f},
                    {0.88f, 0.43f, 0.9f}, {0.21f, 0.72f, 0.65f}, {0.74f, 0.82f, 0.8f}
            };
            for (float[] star : stars) {
                paint.setAlpha((int) (180 * star[2]));
                canvas.drawCircle(width * star[0], height * star[1], dp(1), paint);
            }
            paint.setAlpha(255);

            paint.setShader(new RadialGradient(width * 0.5f, height * 0.98f, width * 0.60f,
                    Color.argb(92, 245, 146, 30), Color.TRANSPARENT, Shader.TileMode.CLAMP));
            canvas.drawCircle(width * 0.5f, height * 0.98f, width * 0.60f, paint);
            paint.setShader(null);
        }
    }

    private final class LoadingRingView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private float angle = 0f;

        LoadingRingView(Context context) {
            super(context);
        }

        @Override
        protected void onAttachedToWindow() {
            super.onAttachedToWindow();
            postInvalidateOnAnimation();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float width = getWidth();
            float height = getHeight();
            float size = Math.min(width, height);
            float cx = width / 2f;
            float cy = height / 2f;

            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(2));
            paint.setColor(Color.argb(51, 196, 184, 232));
            canvas.drawCircle(cx, cy, size * 0.38f, paint);

            RectF outer = new RectF(cx - size * 0.46f, cy - size * 0.46f, cx + size * 0.46f, cy + size * 0.46f);
            paint.setStrokeWidth(dp(3));
            paint.setStrokeCap(Paint.Cap.ROUND);
            paint.setColor(EA_ACCENT);
            canvas.drawArc(outer, angle, 105, false, paint);
            paint.setColor(EA_PINK);
            canvas.drawArc(outer, angle + 118, 68, false, paint);

            RectF inner = new RectF(cx - size * 0.27f, cy - size * 0.27f, cx + size * 0.27f, cy + size * 0.27f);
            paint.setColor(Color.rgb(45, 212, 191));
            canvas.drawArc(inner, -angle * 1.35f, 118, false, paint);

            angle = (angle + 2.4f) % 360f;
            postInvalidateDelayed(16);
        }
    }

    private final class OfflineDinoView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Random random = new Random();
        private final RectF runnerRect = new RectF();
        private final RectF obstacleRect = new RectF();
        private final RectF bonusRect = new RectF();
        private float runnerTop = 0f;
        private float velocityY = 0f;
        private float obstacleX = -1f;
        private float bonusX = -1f;
        private float groundOffset = 0f;
        private long lastFrameAt = 0L;
        private int score = 0;
        private int bestScore = 0;
        private int combo = 0;
        private boolean gameOver = false;
        private boolean shield = false;

        OfflineDinoView(Context context) {
            super(context);
            setClickable(true);
            bestScore = getSharedPreferences("extraarena_offline_runner", MODE_PRIVATE).getInt("best_score", 0);
        }

        @Override
        protected void onAttachedToWindow() {
            super.onAttachedToWindow();
            lastFrameAt = 0L;
            postInvalidateOnAnimation();
        }

        @Override
        public boolean performClick() {
            super.performClick();
            if (gameOver) {
                restartRun(true);
            } else {
                jump();
            }
            return true;
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                performClick();
                return true;
            }
            return true;
        }

        private void jump() {
            float groundTop = getGroundY() - getRunnerHeight();
            if (Math.abs(runnerTop - groundTop) < dp(3)) {
                velocityY = -dp(760);
                combo += 1;
                vibrate(combo % 5 == 0 ? "medium" : "selection");
            }
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            int width = getWidth();
            int height = getHeight();
            if (width <= 0 || height <= 0) {
                return;
            }

            updateRun();
            float groundY = getGroundY();
            float runnerW = getRunnerWidth();
            float runnerH = getRunnerHeight();
            float runnerX = width * 0.18f;
            runnerRect.set(runnerX, runnerTop, runnerX + runnerW, runnerTop + runnerH);

            paint.setStyle(Paint.Style.FILL);
            paint.setShader(new LinearGradient(
                    0, 0, width, height,
                    new int[]{Color.argb(155, 245, 146, 30), Color.argb(95, 244, 114, 182), Color.argb(120, 45, 212, 191)},
                    new float[]{0f, 0.52f, 1f},
                    Shader.TileMode.CLAMP
            ));
            canvas.drawRoundRect(dp(6), dp(6), width - dp(6), height - dp(6), dp(28), dp(28), paint);
            paint.setShader(null);

            paint.setColor(Color.argb(205, 16, 10, 30));
            canvas.drawRoundRect(dp(10), dp(10), width - dp(10), height - dp(10), dp(24), dp(24), paint);

            drawStars(canvas, width, height);
            drawRoad(canvas, width, groundY);
            drawBonus(canvas, groundY);
            drawObstacle(canvas, groundY);
            drawRunner(canvas, runnerRect, shield);

            paint.setStyle(Paint.Style.FILL);
            paint.setTypeface(futuraBold);
            paint.setTextSize(isCompactHeight() ? dp(12) : dp(13));
            paint.setColor(Color.argb(218, 240, 236, 255));
            canvas.drawText("score " + score, dp(22), dp(32), paint);
            paint.setColor(Color.rgb(255, 218, 157));
            canvas.drawText("best " + bestScore, width - dp(94), dp(32), paint);
            if (shield) {
                paint.setColor(Color.rgb(45, 212, 191));
                canvas.drawText("shield", dp(22), dp(52), paint);
            }

            if (gameOver) {
                drawGameOver(canvas, width, height);
            }

            postInvalidateOnAnimation();
        }

        private void updateRun() {
            long now = System.nanoTime();
            if (lastFrameAt == 0L) {
                lastFrameAt = now;
                ensureRunReady();
                return;
            }
            float dt = Math.min(0.033f, (now - lastFrameAt) / 1_000_000_000f);
            lastFrameAt = now;
            ensureRunReady();
            if (gameOver) {
                return;
            }

            float groundTop = getGroundY() - getRunnerHeight();
            velocityY += dp(1900) * dt;
            runnerTop += velocityY * dt;
            if (runnerTop >= groundTop) {
                runnerTop = groundTop;
                velocityY = 0f;
                combo = 0;
            }
            float runnerW = getRunnerWidth();
            float runnerH = getRunnerHeight();
            float runnerX = getWidth() * 0.18f;
            runnerRect.set(runnerX, runnerTop, runnerX + runnerW, runnerTop + runnerH);

            float speed = dp(285) + Math.min(dp(190), score * 0.42f);
            obstacleX -= speed * dt;
            bonusX -= speed * dt;
            groundOffset = (groundOffset + speed * dt) % dp(42);

            if (obstacleX < -dp(80)) {
                score += 12;
                resetObstacle();
                if (score % 60 == 0) {
                    vibrate("selection");
                }
            }
            if (bonusX < -dp(50) && random.nextFloat() < 0.018f) {
                bonusX = getWidth() + dp(32);
            }

            float groundY = getGroundY();
            float obstacleW = dp(24 + (score / 90) % 3 * 5);
            float obstacleH = dp(42 + (score / 120) % 3 * 8);
            obstacleRect.set(obstacleX, groundY - obstacleH, obstacleX + obstacleW, groundY);
            if (bonusX > 0f) {
                bonusRect.set(bonusX, groundY - dp(112), bonusX + dp(30), groundY - dp(82));
            } else {
                bonusRect.setEmpty();
            }

            RectF hitRunner = new RectF(runnerRect);
            hitRunner.inset(dp(8), dp(7));
            RectF hitObstacle = new RectF(obstacleRect);
            hitObstacle.inset(dp(4), dp(2));
            if (RectF.intersects(hitRunner, hitObstacle)) {
                if (shield) {
                    shield = false;
                    score += 28;
                    resetObstacle();
                    vibrate("success");
                } else {
                    gameOver = true;
                    bestScore = Math.max(bestScore, score);
                    getSharedPreferences("extraarena_offline_runner", MODE_PRIVATE)
                            .edit()
                            .putInt("best_score", bestScore)
                            .apply();
                    vibrate("error");
                }
            }
            if (!bonusRect.isEmpty() && RectF.intersects(hitRunner, bonusRect)) {
                bonusX = -1f;
                shield = true;
                score += 40;
                vibrate("success");
            }
        }

        private void ensureRunReady() {
            if (runnerTop <= 0f) {
                runnerTop = getGroundY() - getRunnerHeight();
            }
            if (obstacleX < 0f) {
                resetObstacle();
            }
        }

        private void restartRun(boolean feedback) {
            score = 0;
            combo = 0;
            velocityY = 0f;
            shield = false;
            gameOver = false;
            runnerTop = getGroundY() - getRunnerHeight();
            resetObstacle();
            bonusX = getWidth() + dp(260);
            lastFrameAt = 0L;
            if (feedback) {
                vibrate("success");
            }
            postInvalidateOnAnimation();
        }

        private void resetObstacle() {
            obstacleX = getWidth() + dp(120 + random.nextInt(170));
        }

        private float getGroundY() {
            return getHeight() * 0.72f;
        }

        private float getRunnerWidth() {
            return Math.min(dp(78), getWidth() * 0.19f);
        }

        private float getRunnerHeight() {
            return getRunnerWidth() * 0.88f;
        }

        private void drawStars(Canvas canvas, int width, int height) {
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.argb(120, 240, 236, 255));
            float drift = (groundOffset * 0.18f) % width;
            for (int i = 0; i < 9; i++) {
                float x = (i * width * 0.17f - drift + width) % width;
                float y = dp(54) + (i % 4) * height * 0.10f;
                canvas.drawCircle(x, y, dp(i % 3 == 0 ? 2 : 1), paint);
            }
        }

        private void drawRoad(Canvas canvas, int width, float groundY) {
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(2));
            paint.setStrokeCap(Paint.Cap.ROUND);
            paint.setColor(Color.argb(120, 255, 218, 157));
            canvas.drawLine(dp(24), groundY, width - dp(24), groundY, paint);
            paint.setStrokeWidth(dp(3));
            paint.setColor(Color.argb(110, 45, 212, 191));
            for (int i = 0; i < 9; i++) {
                float x = (i * dp(56) - groundOffset + width) % width;
                canvas.drawLine(x, groundY + dp(10), x + dp(22), groundY + dp(10), paint);
            }
        }

        private void drawRunner(Canvas canvas, RectF r, boolean protectedRun) {
            paint.setStyle(Paint.Style.FILL);
            if (protectedRun) {
                paint.setColor(Color.argb(80, 45, 212, 191));
                canvas.drawCircle(r.centerX(), r.centerY(), r.width() * 0.78f, paint);
            }
            paint.setColor(Color.rgb(240, 236, 255));
            canvas.drawRoundRect(r.left, r.top + r.height() * 0.24f, r.left + r.width() * 0.72f, r.bottom, dp(10), dp(10), paint);
            canvas.drawRoundRect(r.left + r.width() * 0.48f, r.top, r.right, r.top + r.height() * 0.48f, dp(9), dp(9), paint);
            canvas.drawRect(r.left + r.width() * 0.18f, r.bottom - dp(3), r.left + r.width() * 0.30f, r.bottom + dp(14), paint);
            canvas.drawRect(r.left + r.width() * 0.48f, r.bottom - dp(3), r.left + r.width() * 0.60f, r.bottom + dp(14), paint);
            canvas.drawRoundRect(r.left - r.width() * 0.20f, r.top + r.height() * 0.50f, r.left + r.width() * 0.10f, r.top + r.height() * 0.64f, dp(5), dp(5), paint);

            paint.setColor(EA_ACCENT);
            canvas.drawRoundRect(r.left + r.width() * 0.18f, r.top + r.height() * 0.26f, r.left + r.width() * 0.62f, r.top + r.height() * 0.38f, dp(4), dp(4), paint);

            paint.setColor(Color.rgb(16, 10, 30));
            canvas.drawCircle(r.left + r.width() * 0.82f, r.top + r.height() * 0.16f, Math.max(2f, r.width() * 0.035f), paint);
        }

        private void drawObstacle(Canvas canvas, float groundY) {
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(45, 212, 191));
            float x = obstacleRect.left;
            float h = obstacleRect.height();
            float w = obstacleRect.width();
            canvas.drawRoundRect(obstacleRect, dp(6), dp(6), paint);
            paint.setColor(Color.rgb(173, 255, 244));
            canvas.drawRoundRect(x - w * 0.62f, groundY - h * 0.66f, x, groundY - h * 0.52f, dp(5), dp(5), paint);
            canvas.drawRoundRect(x + w, groundY - h * 0.48f, x + w + w * 0.62f, groundY - h * 0.34f, dp(5), dp(5), paint);
        }

        private void drawBonus(Canvas canvas, float groundY) {
            if (bonusRect.isEmpty()) {
                return;
            }
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(255, 218, 157));
            canvas.drawRoundRect(bonusRect, dp(8), dp(8), paint);
            paint.setColor(EA_ACCENT);
            canvas.drawCircle(bonusRect.centerX(), bonusRect.centerY(), dp(6), paint);
        }

        private void drawGameOver(Canvas canvas, int width, int height) {
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.argb(178, 5, 2, 12));
            canvas.drawRoundRect(dp(28), height * 0.26f, width - dp(28), height * 0.60f, dp(22), dp(22), paint);
            paint.setTypeface(futuraExtraBold);
            paint.setTextAlign(Paint.Align.CENTER);
            paint.setTextSize(dp(isCompactHeight() ? 25 : 30));
            paint.setColor(Color.WHITE);
            canvas.drawText("Сеть потерялась", width / 2f, height * 0.38f, paint);
            paint.setTypeface(futuraBold);
            paint.setTextSize(dp(13));
            paint.setColor(Color.rgb(255, 218, 157));
            canvas.drawText("Тапни, чтобы пробежать еще", width / 2f, height * 0.48f, paint);
            paint.setTextAlign(Paint.Align.LEFT);
        }
    }

    private static final class PingResult {
        final boolean ok;
        final long elapsedMs;
        final int statusCode;

        PingResult(boolean ok, long elapsedMs, int statusCode) {
            this.ok = ok;
            this.elapsedMs = elapsedMs;
            this.statusCode = statusCode;
        }
    }

    private static final class MobileUpdateInfo {
        final boolean required;
        final int latestVersionCode;
        final String latestVersionName;
        final String message;
        final String telegramUrl;
        final String apkUrl;
        final String rustoreUrl;

        private MobileUpdateInfo(
                boolean required,
                int latestVersionCode,
                String latestVersionName,
                String message,
                String telegramUrl,
                String apkUrl,
                String rustoreUrl
        ) {
            this.required = required;
            this.latestVersionCode = latestVersionCode;
            this.latestVersionName = latestVersionName == null ? "" : latestVersionName.trim();
            this.message = message == null || message.trim().isEmpty()
                    ? "Хорошие новости! Вышло обновление, скачай новую версию, чтобы продолжить игру."
                    : message.trim();
            this.telegramUrl = normalizeUrl(telegramUrl, BuildConfig.UPDATE_CHANNEL_URL);
            this.apkUrl = normalizeUrl(apkUrl, BuildConfig.UPDATE_APK_URL);
            this.rustoreUrl = normalizeUrl(rustoreUrl, BuildConfig.RUSTORE_APP_URL);
        }

        static MobileUpdateInfo from(JSONObject data) {
            int latestCode = data.optInt("latest_version_code", BuildConfig.VERSION_CODE);
            int minCode = data.optInt("min_supported_version_code", latestCode);
            boolean required = data.optBoolean(
                    "required",
                    data.optBoolean("update_required", BuildConfig.VERSION_CODE < Math.max(latestCode, minCode))
            );
            if (BuildConfig.VERSION_CODE < Math.max(latestCode, minCode)) {
                required = true;
            }
            return new MobileUpdateInfo(
                    required,
                    latestCode,
                    data.optString("latest_version_name", ""),
                    data.optString("message", ""),
                    data.optString("telegram_url", data.optString("update_url", BuildConfig.UPDATE_CHANNEL_URL)),
                    data.optString("apk_url", BuildConfig.UPDATE_APK_URL),
                    data.optString("rustore_url", data.optString("rustore_app_url", BuildConfig.RUSTORE_APP_URL))
            );
        }

        static MobileUpdateInfo notRequired() {
            return new MobileUpdateInfo(
                    false,
                    BuildConfig.VERSION_CODE,
                    BuildConfig.VERSION_NAME,
                    "",
                    BuildConfig.UPDATE_CHANNEL_URL,
                    BuildConfig.UPDATE_APK_URL,
                    BuildConfig.RUSTORE_APP_URL
            );
        }

        static MobileUpdateInfo requiredFallback() {
            return new MobileUpdateInfo(
                    true,
                    BuildConfig.VERSION_CODE + 1,
                    "",
                    "",
                    BuildConfig.UPDATE_CHANNEL_URL,
                    BuildConfig.UPDATE_APK_URL,
                    BuildConfig.RUSTORE_APP_URL
            );
        }

        private static String normalizeUrl(String value, String fallback) {
            String result = value == null || value.trim().isEmpty() ? fallback : value.trim();
            if (result.startsWith("http://") || result.startsWith("https://")
                    || result.startsWith("tg:") || result.startsWith("telegram:")) {
                return result;
            }
            return "https://" + result;
        }
    }

    public final class AndroidBridge {
        @JavascriptInterface
        public String getPlatform() {
            return "android_app";
        }

        @JavascriptInterface
        public String getBaseUrl() {
            return BaseUrlStore.getBaseUrl(MainActivity.this);
        }

        @JavascriptInterface
        public String getAppVersion() {
            return BuildConfig.VERSION_NAME;
        }

        @JavascriptInterface
        public String getDistributionChannel() {
            return BuildConfig.DISTRIBUTION_CHANNEL;
        }

        @JavascriptInterface
        public String getPaymentProviderOrder() {
            return BuildConfig.PAYMENT_PROVIDER_ORDER;
        }

        @JavascriptInterface
        public boolean isRuStorePayAvailable() {
            return ruStoreIntegration != null && ruStoreIntegration.isPayAvailable();
        }

        @JavascriptInterface
        public String startRuStorePayment(String payloadJson) {
            if (ruStoreIntegration == null) {
                return "{\"accepted\":false,\"error\":\"rustore_unavailable\"}";
            }
            return ruStoreIntegration.startPayment(payloadJson);
        }

        @JavascriptInterface
        public boolean isTestServer() {
            return BaseUrlStore.isTestServer(MainActivity.this);
        }

        @JavascriptInterface
        public String getConnectionProfile() {
            return ConnectionProfileStore.getSelectedProfile(MainActivity.this).name;
        }

        @JavascriptInterface
        public boolean isWhitelistEnabled() {
            return ConnectionProfileStore.isWhitelistEnabled(MainActivity.this);
        }

        @JavascriptInterface
        public String getWhitelistCode() {
            return ConnectionProfileStore.getWhitelistCode(MainActivity.this);
        }

        @JavascriptInterface
        public void setAuthToken(String token) {
            DeviceRegistrar.saveAuthToken(MainActivity.this, token);
            ExtraIdAccountStore.touchAccountByToken(
                    MainActivity.this,
                    BaseUrlStore.getBaseUrl(MainActivity.this),
                    token
            );
        }

        @JavascriptInterface
        public String getAuthToken() {
            return DeviceRegistrar.getAuthToken(MainActivity.this);
        }

        @JavascriptInterface
        public String getExtraIdAccounts() {
            return ExtraIdAccountStore.toJson(
                    MainActivity.this,
                    BaseUrlStore.getBaseUrl(MainActivity.this),
                    DeviceRegistrar.getAuthToken(MainActivity.this)
            ).toString();
        }

        @JavascriptInterface
        public boolean switchExtraIdAccount(String email) {
            return ExtraIdAccountStore.activateAccount(
                    MainActivity.this,
                    BaseUrlStore.getBaseUrl(MainActivity.this),
                    email
            );
        }

        @JavascriptInterface
        public boolean forgetExtraIdAccount(String email) {
            return ExtraIdAccountStore.removeAccount(
                    MainActivity.this,
                    BaseUrlStore.getBaseUrl(MainActivity.this),
                    email,
                    DeviceRegistrar.getAuthToken(MainActivity.this)
            );
        }

        @JavascriptInterface
        public void saveExtraIdAccount(String email, String token, String displayId, String userId) {
            long parsedUserId = 0L;
            try {
                parsedUserId = Long.parseLong(userId == null ? "" : userId.trim());
            } catch (Exception ignored) {
            }
            ExtraIdAccountStore.saveAccount(
                    MainActivity.this,
                    email,
                    token,
                    displayId,
                    parsedUserId,
                    BaseUrlStore.getBaseUrl(MainActivity.this)
            );
        }

        @JavascriptInterface
        public void reloadWithActiveAccount() {
            runOnUiThread(() -> {
                resetWebViewForNewAuthSession();
                launchAfterUpdateGate(getIntent());
            });
        }

        @JavascriptInterface
        public void openExtraIdManager() {
            runOnUiThread(MainActivity.this::showExtraIdManagerDialog);
        }

        @JavascriptInterface
        public void clearAuthToken() {
            runOnUiThread(() -> {
                clearWebAuthState();
                DeviceRegistrar.clearAuthToken(MainActivity.this);
                showAuth();
            });
        }

        @JavascriptInterface
        public void requestPushRegistration() {
            DeviceRegistrar.registerIfReady(MainActivity.this);
        }

        @JavascriptInterface
        public void haptic(String style) {
            runOnUiThread(() -> MainActivity.this.vibrate(style));
        }

        @JavascriptInterface
        public boolean isHapticsEnabled() {
            return MainActivity.this.isHapticsEnabled();
        }

        @JavascriptInterface
        public void setHapticsEnabled(boolean enabled) {
            MainActivity.this.setHapticsEnabled(enabled);
        }

        @JavascriptInterface
        public void showConnectivityError() {
            runOnUiThread(MainActivity.this::showConnectivityError);
        }

        @JavascriptInterface
        public void showBadConnectionSplash(int latencyMs) {
            runOnUiThread(() -> MainActivity.this.showBadConnectionSplash(latencyMs));
        }

        @JavascriptInterface
        public void showNativeAfkDialog() {
            runOnUiThread(MainActivity.this::showNativeAfkDialog);
        }

        @JavascriptInterface
        public void openExternal(String url) {
            runOnUiThread(() -> MainActivity.this.openExternal(url));
        }

        @JavascriptInterface
        public void shareText(String text, String url, String title) {
            runOnUiThread(() -> MainActivity.this.shareText(text, url, title));
        }
    }

    private void resetWebViewForNewAuthSession() {
        try {
            CookieManager manager = CookieManager.getInstance();
            manager.removeAllCookies(null);
            manager.flush();
        } catch (Exception ignored) {
        }
        if (webView != null) {
            try {
                webView.stopLoading();
                webView.clearHistory();
                webView.clearCache(false);
                webView.evaluateJavascript(
                        "try{for(var i=localStorage.length-1;i>=0;i--){var k=localStorage.key(i);if(k==='extra_id_token'||(k&&k.indexOf('eaJsonCache:')===0))localStorage.removeItem(k);}sessionStorage.clear();}catch(e){}",
                        null
                );
            } catch (Exception ignored) {
            }
        }
    }

    private void clearWebAuthState() {
        try {
            WebStorage.getInstance().deleteAllData();
        } catch (Exception ignored) {
        }
        try {
            CookieManager manager = CookieManager.getInstance();
            manager.removeAllCookies(null);
            manager.flush();
        } catch (Exception ignored) {
        }
        if (webView != null) {
            try {
                webView.stopLoading();
                webView.clearHistory();
                webView.clearCache(true);
                webView.evaluateJavascript(
                        "try{localStorage.removeItem('extra_id_token');sessionStorage.clear();}catch(e){}",
                        null
                );
            } catch (Exception ignored) {
            }
        }
    }
}
