package ru.extraarena.app;

import android.content.Context;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Locale;

final class BaseUrlStore {
    private static final String OFFICIAL_IDENTITY_SCOPE = "extraarena_official_v1";

    private BaseUrlStore() {
    }

    static String getBaseUrl(Context context) {
        return ConnectionProfileStore.getSelectedProfile(context).baseUrl;
    }

    static void setBaseUrl(Context context, String baseUrl) {
        ConnectionProfileStore.ConnectionProfile profile = new ConnectionProfileStore.ConnectionProfile(
                "custom_" + System.currentTimeMillis(),
                "Custom server",
                normalize(baseUrl),
                false,
                ""
        );
        ConnectionProfileStore.saveProfile(context, profile);
        ConnectionProfileStore.selectProfile(context, profile.id);
    }

    static void resetToProduction(Context context) {
        ConnectionProfileStore.selectProfile(context, ConnectionProfileStore.DEFAULT_PROFILE_ID);
    }

    static boolean isTestServer(Context context) {
        ConnectionProfileStore.ConnectionProfile selected = ConnectionProfileStore.getSelectedProfile(context);
        if (ConnectionProfileStore.isBuiltIn(selected.id)) {
            String url = normalize(selected.baseUrl);
            return !url.equals(normalize(BuildConfig.DEFAULT_BASE_URL))
                    && !url.equals(normalize(BuildConfig.RU_BASE_URL));
        }
        return true;
    }

    static String join(String baseUrl, String path) {
        String normalized = normalize(baseUrl);
        String cleanPath = path == null ? "" : path.trim();
        if (cleanPath.startsWith("/")) {
            cleanPath = cleanPath.substring(1);
        }
        return normalized + cleanPath;
    }

    static String normalize(String value) {
        String result = value == null || value.trim().isEmpty()
                ? BuildConfig.DEFAULT_BASE_URL
                : value.trim();
        String canonical = canonicalHttpBaseUrl(result);
        if (canonical != null) {
            return canonical;
        }
        return result.endsWith("/") ? result : result + "/";
    }

    /**
     * Stable credential/account identity shared by the two official network
     * entrypoints. Endpoint selection may change, but it is still one backend
     * and must not fork anonymous bootstrap or ExtraID state.
     */
    static String identityScope(String baseUrl) {
        return identityScope(baseUrl, "");
    }

    static String identityScope(String baseUrl, String whitelistCode) {
        String normalized = normalize(baseUrl);
        String normalizedWhitelist = whitelistCode == null ? "" : whitelistCode.trim();
        if (normalizedWhitelist.isEmpty()
                && (normalized.equals(normalize(BuildConfig.DEFAULT_BASE_URL))
                || normalized.equals(normalize(BuildConfig.RU_BASE_URL)))) {
            return OFFICIAL_IDENTITY_SCOPE;
        }
        String tenantBinding = normalized
                + "\u0000"
                + normalizedWhitelist;
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                    tenantBinding.getBytes(StandardCharsets.UTF_8)
            );
            StringBuilder encoded = new StringBuilder(digest.length * 2);
            for (byte value : digest) {
                encoded.append(String.format(Locale.ROOT, "%02x", value & 0xff));
            }
            return "extraarena_custom_v1_" + encoded;
        } catch (Exception e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    static String identityBaseUrl(String baseUrl) {
        return OFFICIAL_IDENTITY_SCOPE.equals(identityScope(baseUrl))
                ? normalize(BuildConfig.DEFAULT_BASE_URL)
                : normalize(baseUrl);
    }

    private static String canonicalHttpBaseUrl(String value) {
        try {
            URI parsed = new URI(value);
            String scheme = parsed.getScheme();
            String host = parsed.getHost();
            if (scheme == null || host == null) {
                return null;
            }
            scheme = scheme.toLowerCase(Locale.ROOT);
            if (!"http".equals(scheme) && !"https".equals(scheme)) {
                return null;
            }
            host = host.toLowerCase(Locale.ROOT);
            int port = parsed.getPort();
            if (("http".equals(scheme) && port == 80)
                    || ("https".equals(scheme) && port == 443)) {
                port = -1;
            }
            String path = parsed.getRawPath();
            if (path == null || path.isEmpty()) {
                path = "/";
            }
            path = normalizePathSafely(path);
            String authorityHost = host.indexOf(':') >= 0 && !host.startsWith("[")
                    ? "[" + host + "]"
                    : host;
            String result = scheme
                    + "://"
                    + authorityHost
                    + (port >= 0 ? ":" + port : "")
                    + path;
            return result.endsWith("/") ? result : result + "/";
        } catch (Exception ignored) {
            return null;
        }
    }

    /** Resolves literal dot segments without conflating meaningful duplicate slashes. */
    private static String normalizePathSafely(String rawPath) {
        boolean absolute = rawPath.startsWith("/");
        boolean trailingDirectory = rawPath.endsWith("/")
                || rawPath.endsWith("/.")
                || rawPath.endsWith("/..");
        String[] segments = rawPath.split("/", -1);
        ArrayList<String> normalized = new ArrayList<>();
        int start = absolute ? 1 : 0;
        for (int i = start; i < segments.length; i++) {
            String segment = segments[i];
            if (".".equals(segment)) {
                continue;
            }
            if ("..".equals(segment)) {
                if (!normalized.isEmpty()) {
                    normalized.remove(normalized.size() - 1);
                }
                continue;
            }
            normalized.add(segment);
        }
        if (trailingDirectory
                && (normalized.isEmpty() || !normalized.get(normalized.size() - 1).isEmpty())) {
            normalized.add("");
        }
        String joined = String.join("/", normalized);
        return absolute ? "/" + joined : joined;
    }
}
