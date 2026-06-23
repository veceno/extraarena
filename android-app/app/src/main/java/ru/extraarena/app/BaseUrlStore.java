package ru.extraarena.app;

import android.content.Context;

final class BaseUrlStore {
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
        return result.endsWith("/") ? result : result + "/";
    }
}
