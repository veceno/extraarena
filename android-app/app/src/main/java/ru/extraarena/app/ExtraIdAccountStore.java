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
        final String credentialScope;
        final long lastUsedAt;

        ExtraIdAccount(
                String email,
                String token,
                String displayId,
                long userId,
                String baseUrl,
                String credentialScope,
                long lastUsedAt
        ) {
            this.email = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
            this.token = token == null ? "" : token;
            this.displayId = displayId == null ? "" : displayId.trim();
            this.userId = userId;
            this.baseUrl = BaseUrlStore.identityBaseUrl(baseUrl);
            this.credentialScope = credentialScope == null || credentialScope.trim().isEmpty()
                    ? BaseUrlStore.identityScope(baseUrl)
                    : credentialScope.trim();
            this.lastUsedAt = lastUsedAt;
        }
    }

    static List<ExtraIdAccount> getAccounts(Context context, String baseUrl) {
        String selectedScope = credentialScope(context, baseUrl);
        ArrayList<ExtraIdAccount> result = new ArrayList<>();
        for (ExtraIdAccount account : loadAccounts(context)) {
            if (account.credentialScope.equals(selectedScope)
                    && !account.token.isEmpty()
                    && !AuthClient.isRevocationPendingForScope(
                    context,
                    account.credentialScope,
                    account.token
            )) {
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

        ArrayList<ExtraIdAccount> accounts = loadAccounts(context);
        ExtraIdAccount next = new ExtraIdAccount(
                cleanEmail,
                token,
                displayId,
                userId,
                baseUrl,
                credentialScope(context, baseUrl),
                System.currentTimeMillis()
        );

        boolean replaced = false;
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.email.equals(next.email)
                    && account.credentialScope.equals(next.credentialScope)) {
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
        String selectedScope = credentialScope(context, baseUrl);
        String cleanEmail = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
        if (cleanEmail.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = loadAccounts(context);
        ExtraIdAccount selected = null;
        long now = System.currentTimeMillis();
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.credentialScope.equals(selectedScope)
                    && account.email.equals(cleanEmail)
                    && !account.token.isEmpty()) {
                selected = new ExtraIdAccount(
                        account.email,
                        account.token,
                        account.displayId,
                        account.userId,
                        account.baseUrl,
                        account.credentialScope,
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
        String selectedScope = credentialScope(context, baseUrl);
        String cleanEmail = email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
        String active = activeToken == null ? "" : activeToken;
        if (cleanEmail.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = loadAccounts(context);
        boolean removed = false;
        boolean removedActive = false;
        for (int i = accounts.size() - 1; i >= 0; i--) {
            ExtraIdAccount account = accounts.get(i);
            if (account.credentialScope.equals(selectedScope)
                    && account.email.equals(cleanEmail)) {
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
        String selectedScope = credentialScope(context, baseUrl);
        String cleanToken = token == null ? "" : token.trim();
        if (cleanToken.isEmpty()) {
            return false;
        }

        ArrayList<ExtraIdAccount> accounts = loadAccounts(context);
        boolean removed = false;
        for (int i = accounts.size() - 1; i >= 0; i--) {
            ExtraIdAccount account = accounts.get(i);
            if (account.credentialScope.equals(selectedScope)
                    && account.token.equals(cleanToken)) {
                accounts.remove(i);
                removed = true;
            }
        }
        return removed && persistAccounts(context, accounts);
    }

    static void touchAccountByToken(Context context, String baseUrl, String token) {
        String selectedScope = credentialScope(context, baseUrl);
        String activeToken = token == null ? "" : token;
        if (activeToken.isEmpty()) {
            return;
        }

        ArrayList<ExtraIdAccount> accounts = loadAccounts(context);
        boolean changed = false;
        long now = System.currentTimeMillis();
        for (int i = 0; i < accounts.size(); i++) {
            ExtraIdAccount account = accounts.get(i);
            if (account.credentialScope.equals(selectedScope)
                    && activeToken.equals(account.token)) {
                accounts.set(i, new ExtraIdAccount(
                        account.email,
                        account.token,
                        account.displayId,
                        account.userId,
                        account.baseUrl,
                        account.credentialScope,
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

    private static ArrayList<ExtraIdAccount> loadAccounts(Context context) {
        String raw = SecurePrefs.getString(context, KEY_ACCOUNTS, "");
        ArrayList<ExtraIdAccount> accounts = new ArrayList<>();
        if (raw == null || raw.trim().isEmpty()) {
            return accounts;
        }
        boolean migrated = false;
        try {
            JSONArray array = new JSONArray(raw);
            for (int i = 0; i < array.length(); i++) {
                JSONObject item = array.optJSONObject(i);
                if (item == null) {
                    continue;
                }
                String baseUrl = item.optString(
                        "base_url",
                        BuildConfig.DEFAULT_BASE_URL
                );
                String storedScope = item.optString("credential_scope", "").trim();
                if (storedScope.isEmpty()) {
                    // Legacy rows had no tenant binding. Bind them exactly once
                    // to the profile selected at upgrade time, just like the
                    // legacy global auth token migration.
                    storedScope = credentialScope(context, baseUrl);
                    migrated = true;
                }
                accounts.add(new ExtraIdAccount(
                        item.optString("email", ""),
                        item.optString("token", ""),
                        item.optString("display_id", ""),
                        item.optLong("user_id", 0L),
                        baseUrl,
                        storedScope,
                        item.optLong("last_used_at", 0L)
                ));
            }
        } catch (Exception ignored) {
            accounts.clear();
            return accounts;
        }
        if (migrated) {
            persistAccounts(context, accounts);
        }
        return accounts;
    }

    private static String credentialScope(Context context, String baseUrl) {
        ConnectionProfileStore.ConnectionProfile selected =
                ConnectionProfileStore.getSelectedProfile(context);
        String whitelistCode = BaseUrlStore.normalize(selected.baseUrl).equals(
                BaseUrlStore.normalize(baseUrl)
        ) && selected.whitelistEnabled ? selected.whitelistCode : "";
        return BaseUrlStore.identityScope(baseUrl, whitelistCode);
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
                item.put("credential_scope", account.credentialScope);
                item.put("last_used_at", account.lastUsedAt);
                array.put(item);
            } catch (Exception ignored) {
            }
        }
        return SecurePrefs.putString(context, KEY_ACCOUNTS, array.toString());
    }
}
