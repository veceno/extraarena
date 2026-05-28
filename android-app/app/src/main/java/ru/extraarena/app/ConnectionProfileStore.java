package ru.extraarena.app;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

final class ConnectionProfileStore {
    static final String DEFAULT_PROFILE_ID = "extraarena_worldwide";

    private static final String KEY_PROFILES = "connection_profiles";
    private static final String KEY_SELECTED_PROFILE_ID = "selected_connection_profile_id";
    private static final String KEY_LEGACY_BASE_URL = "base_url";

    private ConnectionProfileStore() {
    }

    static final class ConnectionProfile {
        final String id;
        final String name;
        final String baseUrl;
        final boolean whitelistEnabled;
        final String whitelistCode;

        ConnectionProfile(String id, String name, String baseUrl, boolean whitelistEnabled, String whitelistCode) {
            this.id = cleanId(id);
            this.name = cleanName(name);
            this.baseUrl = BaseUrlStore.normalize(baseUrl);
            this.whitelistEnabled = whitelistEnabled;
            this.whitelistCode = whitelistCode == null ? "" : whitelistCode.trim();
        }
    }

    static List<ConnectionProfile> getProfiles(Context context) {
        ArrayList<ConnectionProfile> profiles = parseProfiles(SecurePrefs.getString(context, KEY_PROFILES, ""));
        ensureDefaultProfile(profiles);
        migrateLegacyBaseUrl(context, profiles);
        persistProfiles(context, profiles);
        return profiles;
    }

    static ConnectionProfile getSelectedProfile(Context context) {
        List<ConnectionProfile> profiles = getProfiles(context);
        String selectedId = SecurePrefs.getString(context, KEY_SELECTED_PROFILE_ID, DEFAULT_PROFILE_ID);
        for (ConnectionProfile profile : profiles) {
            if (profile.id.equals(selectedId)) {
                return profile;
            }
        }
        return profiles.get(0);
    }

    static void selectProfile(Context context, String profileId) {
        SecurePrefs.putString(context, KEY_SELECTED_PROFILE_ID, cleanId(profileId));
    }

    static void saveProfile(Context context, ConnectionProfile profile) {
        ArrayList<ConnectionProfile> profiles = new ArrayList<>(getProfiles(context));
        boolean replaced = false;
        for (int i = 0; i < profiles.size(); i++) {
            if (profiles.get(i).id.equals(profile.id)) {
                profiles.set(i, profile);
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            profiles.add(profile);
        }
        ensureDefaultProfile(profiles);
        persistProfiles(context, profiles);
    }

    static boolean isWhitelistEnabled(Context context) {
        return getSelectedProfile(context).whitelistEnabled;
    }

    static String getWhitelistCode(Context context) {
        ConnectionProfile profile = getSelectedProfile(context);
        return profile.whitelistEnabled ? profile.whitelistCode : "";
    }

    static String newProfileId() {
        return "profile_" + System.currentTimeMillis();
    }

    private static ArrayList<ConnectionProfile> parseProfiles(String raw) {
        ArrayList<ConnectionProfile> profiles = new ArrayList<>();
        if (raw == null || raw.trim().isEmpty()) {
            return profiles;
        }
        try {
            JSONArray array = new JSONArray(raw);
            for (int i = 0; i < array.length(); i++) {
                JSONObject item = array.optJSONObject(i);
                if (item == null) {
                    continue;
                }
                profiles.add(new ConnectionProfile(
                        item.optString("id", ""),
                        item.optString("name", ""),
                        item.optString("base_url", BuildConfig.DEFAULT_BASE_URL),
                        item.optBoolean("whitelist_enabled", false),
                        item.optString("whitelist_code", "")
                ));
            }
        } catch (Exception ignored) {
            profiles.clear();
        }
        return profiles;
    }

    private static void ensureDefaultProfile(ArrayList<ConnectionProfile> profiles) {
        ConnectionProfile defaultProfile = defaultProfile();
        for (int i = 0; i < profiles.size(); i++) {
            if (DEFAULT_PROFILE_ID.equals(profiles.get(i).id)) {
                profiles.set(i, defaultProfile);
                if (i != 0) {
                    profiles.remove(i);
                    profiles.add(0, defaultProfile);
                }
                return;
            }
        }
        profiles.add(0, defaultProfile);
    }

    private static void migrateLegacyBaseUrl(Context context, ArrayList<ConnectionProfile> profiles) {
        String legacy = SecurePrefs.getString(context, KEY_LEGACY_BASE_URL, "");
        String normalizedLegacy = BaseUrlStore.normalize(legacy);
        if (legacy == null || legacy.trim().isEmpty()
                || normalizedLegacy.equals(BaseUrlStore.normalize(BuildConfig.DEFAULT_BASE_URL))) {
            return;
        }
        for (ConnectionProfile profile : profiles) {
            if (profile.baseUrl.equals(normalizedLegacy)) {
                return;
            }
        }
        profiles.add(new ConnectionProfile(
                "migrated_" + Math.abs(normalizedLegacy.hashCode()),
                "Migrated server",
                normalizedLegacy,
                false,
                ""
        ));
    }

    private static void persistProfiles(Context context, List<ConnectionProfile> profiles) {
        JSONArray array = new JSONArray();
        for (ConnectionProfile profile : profiles) {
            JSONObject item = new JSONObject();
            try {
                item.put("id", profile.id);
                item.put("name", profile.name);
                item.put("base_url", profile.baseUrl);
                item.put("whitelist_enabled", profile.whitelistEnabled);
                item.put("whitelist_code", profile.whitelistCode);
                array.put(item);
            } catch (Exception ignored) {
            }
        }
        SecurePrefs.putString(context, KEY_PROFILES, array.toString());
    }

    private static ConnectionProfile defaultProfile() {
        return new ConnectionProfile(
                DEFAULT_PROFILE_ID,
                "ExtraArena Worldwide",
                BuildConfig.DEFAULT_BASE_URL,
                false,
                ""
        );
    }

    private static String cleanId(String value) {
        if (value == null || value.trim().isEmpty()) {
            return newProfileId();
        }
        return value.trim();
    }

    private static String cleanName(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "Connection profile";
        }
        return value.trim();
    }
}
