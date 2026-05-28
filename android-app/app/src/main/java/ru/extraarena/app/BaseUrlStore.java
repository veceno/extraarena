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
        return !ConnectionProfileStore.DEFAULT_PROFILE_ID.equals(selected.id)
                || !normalize(BuildConfig.DEFAULT_BASE_URL).equals(selected.baseUrl);
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
