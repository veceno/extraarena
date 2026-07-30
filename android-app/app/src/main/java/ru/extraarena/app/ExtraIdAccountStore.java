package ru.extraarena.app;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

final class ExtraIdAccountStore {
    private static final String KEY_ACCOUNTS = "extraid_accounts";

    private ExtraIdAccountStore() {
    }

    static final class ExtraIdAccount {
        final String email;
        final String token;
        final String displayId;
        final long userId;
        final String baseUrl;
        final long lastUsedAt;

        ExtraIdAccount(String email, String token, String displayId, long userId, String baseUrl, long lastUsedAt) {
            this.email = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
            this.token = token == null ? "" : token;
            this.displayId = displayId == null ? "" : displayId.trim();
            this.userId = userId;
            this.baseUrl = BaseUrlStore.normalize(baseUrl);
            this.lastUsedAt = lastUsedAt;
        }
    }

    static List<ExtraIdAccount> getAccounts(Context context, String baseUrl) {
        String normalizedBaseUrl = BaseUrlStore.normalize(baseUrl);
        ArrayList<ExtraIdAccount> result = new ArrayList<>();
        for (ExtraIdAccount account : parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""))) {
            if (account.baseUrl.equals(normalizedBaseUrl)
                    && !account.token.isEmpty()
                    && !AuthClient.isRevocationPending(context, account.baseUrl, account.token)) {
                result.add(account);
            }
        }
        result.sort((left, right) -> Long.compare(right.lastUsedAt, left.lastUsedAt));
        return result;
    }

    static void saveAccount(
            Context context,
            String email,
            AuthClient.AuthResult result,
            String baseUrl
    ) {
        if (result == null || result.token == null || result.token.isEmpty()) {
            return;
        }
        saveAccount(context, email, result.token, result.displayId, result.userId, baseUrl);
    }

    static void saveAccount(
            Context context,
            String email,
            String token,
            String displayId,
            long userId,
            String baseUrl
    ) {
        if (token == null || token.isEmpty()) {
            return;
        }
        String cleanEmail = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
        if (cleanEmail.isEmpty()) {
            return;
        }

        ArrayList<ExtraIdAccount> accounts = parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""));
        ExtraIdAccount next = new ExtraIdAccount(
                cleanEmail,
                token,
                displayId,
                userId,
                baseUrl,
                System.currentTimeMillis()
        );

        boolean replaced = false;
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.email.equals(next.email) && account.baseUrl.equals(next.baseUrl)) {
                accounts.set(i, next);
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            accounts.add(next);
        }
        persistAccounts(context, accounts);
    }

    static JSONArray toJson(Context context, String baseUrl, String activeToken) {
        String active = activeToken == null ? "" : activeToken;
        JSONArray array = new JSONArray();
        for (ExtraIdAccount account : getAccounts(context, baseUrl)) {
            try {
                JSONObject item = new JSONObject();
                item.put("email", account.email);
                item.put("display_id", account.displayId);
                item.put("user_id", account.userId);
                item.put("last_used_at", account.lastUsedAt);
                item.put("active", !active.isEmpty() && active.equals(account.token));
                array.put(item);
            } catch (Exception ignored) {
            }
        }
        return array;
    }

    static boolean activateAccount(Context context, String baseUrl, String email) {
        String normalizedBaseUrl = BaseUrlStore.normalize(baseUrl);
        String cleanEmail = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
        if (cleanEmail.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""));
        ExtraIdAccount selected = null;
        long now = System.currentTimeMillis();
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.baseUrl.equals(normalizedBaseUrl) && account.email.equals(cleanEmail) && !account.token.isEmpty()) {
                selected = new ExtraIdAccount(
                        account.email,
                        account.token,
                        account.displayId,
                        account.userId,
                        account.baseUrl,
                        now
                );
                accounts.set(i, selected);
                break;
            }
        }
        if (selected == null) {
            return false;
        }
        if (!persistAccounts(context, accounts)) {
            return false;
        }
        return DeviceRegistrar.saveAuthToken(context, selected.token);
    }

    static boolean removeAccount(Context context, String baseUrl, String email, String activeToken) {
        String normalizedBaseUrl = BaseUrlStore.normalize(baseUrl);
        String cleanEmail = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
        String active = activeToken == null ? "" : activeToken;
        if (cleanEmail.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""));
        boolean removed = false;
        boolean removedActive = false;
        for (int i = accounts.size() - 1; i >= 0; i--) {
            ExtraIdAccount account = accounts.get(i);
            if (account.baseUrl.equals(normalizedBaseUrl) && account.email.equals(cleanEmail)) {
                removedActive = removedActive || (!active.isEmpty() && active.equals(account.token));
                AuthClient.logoutBestEffort(context, account.token);
                accounts.remove(i);
                removed = true;
            }
        }
        if (!removed) {
            return false;
        }
        if (!persistAccounts(context, accounts)) {
            return false;
        }
        if (removedActive) {
            DeviceRegistrar.clearAuthToken(context);
        }
        return true;
    }

    static boolean removeAccountByToken(Context context, String baseUrl, String token) {
        String normalizedBaseUrl = BaseUrlStore.normalize(baseUrl);
        String cleanToken = token == null ? "" : token.trim();
        if (cleanToken.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""));
        boolean removed = false;
        for (int i = accounts.size() - 1; i >= 0; i--) {
            ExtraIdAccount account = accounts.get(i);
            if (account.baseUrl.equals(normalizedBaseUrl) && account.token.equals(cleanToken)) {
                accounts.remove(i);
                removed = true;
            }
        }
        return removed && persistAccounts(context, accounts);
    }

    static void touchAccountByToken(Context context, String baseUrl, String token) {
        String normalizedBaseUrl = BaseUrlStore.normalize(baseUrl);
        String activeToken = token == null ? "" : token;
        if (activeToken.isEmpty()) {
            return;
        }

        ArrayList<ExtraIdAccount> accounts = parseAccounts(SecurePrefs.getString(context, KEY_ACCOUNTS, ""));
        boolean changed = false;
        long now = System.currentTimeMillis();
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.baseUrl.equals(normalizedBaseUrl) && activeToken.equals(account.token)) {
                accounts.set(i, new ExtraIdAccount(
                        account.email,
                        account.token,
                        account.displayId,
                        account.userId,
                        account.baseUrl,
                        now
                ));
                changed = true;
                break;
            }
        }
        if (changed) {
            persistAccounts(context, accounts);
        }
    }

    private static ArrayList<ExtraIdAccount> parseAccounts(String raw) {
        ArrayList<ExtraIdAccount> accounts = new ArrayList<>();
        if (raw == null || raw.trim().isEmpty()) {
            return accounts;
        }
        try {
            JSONArray array = new JSONArray(raw);
            for (int i = 0; i < array.length(); i++) {
                JSONObject item = array.optJSONObject(i);
                if (item == null) {
                    continue;
                }
                accounts.add(new ExtraIdAccount(
                        item.optString("email", ""),
                        item.optString("token", ""),
                        item.optString("display_id", ""),
                        item.optLong("user_id", 0L),
                        item.optString("base_url", BuildConfig.DEFAULT_BASE_URL),
                        item.optLong("last_used_at", 0L)
                ));
            }
        } catch (Exception ignored) {
            accounts.clear();
        }
        return accounts;
    }

    private static boolean persistAccounts(Context context, List<ExtraIdAccount> accounts) {
        JSONArray array = new JSONArray();
        for (ExtraIdAccount account : accounts) {
            try {
                JSONObject item = new JSONObject();
                item.put("email", account.email);
                item.put("token", account.token);
                item.put("display_id", account.displayId);
                item.put("user_id", account.userId);
                item.put("base_url", account.baseUrl);
                item.put("last_used_at", account.lastUsedAt);
                array.put(item);
            } catch (Exception ignored) {
            }
        }
        return SecurePrefs.putString(context, KEY_ACCOUNTS, array.toString());
    }
}
