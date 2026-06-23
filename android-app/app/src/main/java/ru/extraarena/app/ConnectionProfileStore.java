package ru.extraarena.app;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class ConnectionProfileStore {
    static final String DEFAULT_PROFILE_ID = "extraarena_worldwide";
    static final String RU_PROFILE_ID = "extraarena_ru";
    // Java 8-safe construction (Set.of is a Java 9+ library API and the project has no
    // core library desugaring, so it would throw NoSuchMethodError on API 26-32).
    static final Set<String> BUILT_IN_IDS = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(DEFAULT_PROFILE_ID, RU_PROFILE_ID)));

    private static final String KEY_PROFILES = "connection_profiles";
    private static final String KEY_SELECTED_PROFILE_ID = "selected_connection_profile_id";
    private static final String KEY_LEGACY_BASE_URL = "base_url";
    private static final String KEY_AUTO_SELECTED = "pref_auto_selected_profile";

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
        ensureBuiltInProfiles(profiles);
        migrateLegacyBaseUrl(context, profiles);
        persistProfiles(context, profiles);
        return profiles;
    }

    static boolean isBuiltIn(String id) {
        return id != null && BUILT_IN_IDS.contains(id);
    }

    static ConnectionProfile otherBuiltIn(Context context, String currentId) {
        if (!isBuiltIn(currentId)) {
            return null;
        }
        String otherId = DEFAULT_PROFILE_ID.equals(currentId) ? RU_PROFILE_ID : DEFAULT_PROFILE_ID;
        for (ConnectionProfile profile : getProfiles(context)) {
            if (profile.id.equals(otherId)) {
                return profile;
            }
        }
        return null;
    }

    /**
     * On first run (no persisted selection yet), pick the built-in profile that matches the
     * device region so RU users hit the direct RU host and everyone else the Cloudflare tunnel.
     * Runs at most once (guarded by KEY_AUTO_SELECTED); a pre-existing selection is respected.
     */
    static void autoSelectIfNeeded(Context context, boolean ru) {
        if ("1".equals(SecurePrefs.getString(context, KEY_AUTO_SELECTED, ""))) {
            return;
        }
        SecurePrefs.putString(context, KEY_AUTO_SELECTED, "1");
        if (!SecurePrefs.getString(context, KEY_SELECTED_PROFILE_ID, "").isEmpty()) {
            return;
        }
        selectProfile(context, ru ? RU_PROFILE_ID : DEFAULT_PROFILE_ID);
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
        ensureBuiltInProfiles(profiles);
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

    private static void ensureBuiltInProfiles(ArrayList<ConnectionProfile> profiles) {
        ensureBuiltIn(profiles, defaultProfile(), 0);
        ensureBuiltIn(profiles, ruProfile(), 1);
    }

    private static void ensureBuiltIn(
            ArrayList<ConnectionProfile> profiles,
            ConnectionProfile builtIn,
            int preferredIndex
    ) {
        for (int i = 0; i < profiles.size(); i++) {
            if (builtIn.id.equals(profiles.get(i).id)) {
                profiles.set(i, builtIn);
                if (i != preferredIndex) {
                    profiles.remove(i);
                    profiles.add(preferredIndex, builtIn);
                }
                return;
            }
        }
        profiles.add(preferredIndex, builtIn);
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

    private static ConnectionProfile ruProfile() {
        return new ConnectionProfile(
                RU_PROFILE_ID,
                "ExtraArena RU",
                BuildConfig.RU_BASE_URL,
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
