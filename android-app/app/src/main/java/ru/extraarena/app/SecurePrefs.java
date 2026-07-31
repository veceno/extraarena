package ru.extraarena.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import android.util.Log;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.util.Map;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

final class SecurePrefs {
    private static final String TAG = "EASecurePrefs";
    private static final String SECURE_PREFS = "extraarena_secure";
    private static final String LEGACY_PREFS = "extraarena_app";
    private static final String KEY_ALIAS_V1 = "extraarena_app_prefs_v1";
    private static final String KEY_ALIAS_V2 = "extraarena_app_prefs_v2";
    private static final String ANDROID_KEYSTORE = "AndroidKeyStore";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";
    private static final int IV_BYTES = 12;
    private static final int GCM_TAG_BITS = 128;
    private static final int GCM_TAG_BYTES = GCM_TAG_BITS / 8;
    private static final int CRYPTO_ATTEMPTS = 2;
    private static final Object STORAGE_LOCK = new Object();
    private static final Object KEYSTORE_LOCK = new Object();

    private SecurePrefs() {
    }

    static String getString(Context context, String key, String defaultValue) {
        Context appContext = context.getApplicationContext();
        synchronized (STORAGE_LOCK) {
            SharedPreferences securePrefs = securePrefs(appContext);
            StoredValue stored = readStoredValue(securePrefs, key);
            if (stored.present) {
                if (stored.value != null) {
                    String decrypted = decrypt(key, stored.value);
                    if (decrypted != null) {
                        return decrypted;
                    }
                }
                // The encrypted value remains authoritative. Falling back to a possibly stale
                // legacy value, deleting the ciphertext, or rewriting a default here can turn a
                // transient Android Keystore failure into permanent account loss.
                Log.w(TAG, "Keeping unreadable encrypted value for key=" + key);
                return defaultValue;
            }

            SharedPreferences legacyPrefs = legacyPrefs(appContext);
            String legacy = legacyPrefs.getString(key, null);
            if (legacy != null) {
                // putString removes the legacy copy only after the encrypted commit succeeds.
                putString(appContext, key, legacy);
                return legacy;
            }
            return defaultValue;
        }
    }

    static boolean putString(Context context, String key, String value) {
        Context appContext = context.getApplicationContext();
        if (value == null) {
            remove(appContext, key);
            return true;
        }
        synchronized (STORAGE_LOCK) {
            SharedPreferences securePrefs = securePrefs(appContext);
            StoredValue existing = readStoredValue(securePrefs, key);
            if (existing.present
                    && (existing.value == null || decrypt(key, existing.value) == null)) {
                // Callers such as ConnectionProfileStore persist defaults after reads. Refuse to
                // replace an unreadable value so that a later Keystore retry can still recover it.
                Log.w(TAG, "Refusing to overwrite unreadable encrypted value for key=" + key);
                return false;
            }

            boolean mayCreateCurrentKey = !hasStoredVersion(
                    securePrefs,
                    SecurePrefsEnvelope.CURRENT_VERSION
            );
            String encrypted = encrypt(key, value, mayCreateCurrentKey);
            if (encrypted == null) {
                Log.w(TAG, "Failed to encrypt value for key=" + key);
                return false;
            }
            boolean saved = securePrefs.edit().putString(key, encrypted).commit();
            if (!saved) {
                Log.w(TAG, "Failed to persist encrypted value for key=" + key);
                return false;
            }
            legacyPrefs(appContext).edit().remove(key).apply();
            return true;
        }
    }

    static void remove(Context context, String key) {
        Context appContext = context.getApplicationContext();
        synchronized (STORAGE_LOCK) {
            // Removal is intentionally the only destructive recovery path and must be an explicit
            // caller action (for example, logout), never a side effect of a crypto exception.
            securePrefs(appContext).edit().remove(key).apply();
            legacyPrefs(appContext).edit().remove(key).apply();
        }
    }

    private static SharedPreferences securePrefs(Context context) {
        return context.getSharedPreferences(SECURE_PREFS, Context.MODE_PRIVATE);
    }

    private static SharedPreferences legacyPrefs(Context context) {
        return context.getSharedPreferences(LEGACY_PREFS, Context.MODE_PRIVATE);
    }

    private static StoredValue readStoredValue(SharedPreferences preferences, String key) {
        Map<String, ?> values = preferences.getAll();
        if (!values.containsKey(key)) {
            return StoredValue.absent();
        }
        Object value = values.get(key);
        return value instanceof String
                ? StoredValue.present((String) value)
                : StoredValue.presentUnreadable();
    }

    private static boolean hasStoredVersion(SharedPreferences preferences, String version) {
        String prefix = version + ":";
        for (Object value : preferences.getAll().values()) {
            if (value instanceof String && ((String) value).startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private static String encrypt(String preferenceKey, String value, boolean mayCreateKey) {
        for (int attempt = 1; attempt <= CRYPTO_ATTEMPTS; attempt++) {
            String encrypted = encryptOnce(preferenceKey, value, mayCreateKey);
            if (encrypted != null) {
                return encrypted;
            }
            if (attempt < CRYPTO_ATTEMPTS) {
                Log.w(TAG, "Retrying encryption without changing Keystore state");
            }
        }
        return null;
    }

    private static String encryptOnce(String preferenceKey, String value, boolean mayCreateKey) {
        try {
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(
                    Cipher.ENCRYPT_MODE,
                    getKey(KEY_ALIAS_V2, mayCreateKey)
            );
            cipher.updateAAD(associatedData(preferenceKey));
            byte[] iv = cipher.getIV();
            if (iv == null || iv.length != IV_BYTES) {
                Log.w(TAG, "Encryption produced invalid IV");
                return null;
            }
            byte[] ciphertext = cipher.doFinal(value.getBytes(StandardCharsets.UTF_8));
            return SecurePrefsEnvelope.encodeCurrent(
                    Base64.encodeToString(iv, Base64.NO_WRAP),
                    Base64.encodeToString(ciphertext, Base64.NO_WRAP)
            );
        } catch (Exception e) {
            Log.w(TAG, "Encryption failed", e);
            return null;
        }
    }

    private static String decrypt(String preferenceKey, String encoded) {
        SecurePrefsEnvelope.Parsed parsed = SecurePrefsEnvelope.parse(encoded);
        if (parsed == null) {
            Log.w(TAG, "Encrypted value has an invalid envelope");
            return null;
        }
        String alias = keyAliasForVersion(parsed.version);
        if (alias == null) {
            Log.w(TAG, "Encrypted value uses unsupported version=" + parsed.version);
            return null;
        }
        for (int attempt = 1; attempt <= CRYPTO_ATTEMPTS; attempt++) {
            String decrypted = decryptOnce(preferenceKey, parsed, alias);
            if (decrypted != null) {
                return decrypted;
            }
            if (attempt < CRYPTO_ATTEMPTS) {
                Log.w(TAG, "Retrying decryption without changing stored data or Keystore state");
            }
        }
        return null;
    }

    private static String decryptOnce(
            String preferenceKey,
            SecurePrefsEnvelope.Parsed parsed,
            String alias
    ) {
        try {
            byte[] iv = Base64.decode(parsed.ivBase64, Base64.NO_WRAP);
            byte[] ciphertext = Base64.decode(parsed.ciphertextBase64, Base64.NO_WRAP);
            if (iv.length != IV_BYTES || ciphertext.length < GCM_TAG_BYTES) {
                Log.w(TAG, "Encrypted value has invalid IV or ciphertext length");
                return null;
            }
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(
                    Cipher.DECRYPT_MODE,
                    getKey(alias, false),
                    new GCMParameterSpec(GCM_TAG_BITS, iv)
            );
            if (SecurePrefsEnvelope.VERSION_V2.equals(parsed.version)) {
                cipher.updateAAD(associatedData(preferenceKey));
            }
            byte[] plaintext = cipher.doFinal(ciphertext);
            return new String(plaintext, StandardCharsets.UTF_8);
        } catch (Exception e) {
            Log.w(TAG, "Decryption failed", e);
            return null;
        }
    }

    private static byte[] associatedData(String preferenceKey) {
        return (SECURE_PREFS + ":" + preferenceKey).getBytes(StandardCharsets.UTF_8);
    }

    private static String keyAliasForVersion(String version) {
        if (SecurePrefsEnvelope.VERSION_V1.equals(version)) {
            return KEY_ALIAS_V1;
        }
        if (SecurePrefsEnvelope.VERSION_V2.equals(version)) {
            return KEY_ALIAS_V2;
        }
        return null;
    }

    private static SecretKey getKey(String alias, boolean mayCreate) throws Exception {
        synchronized (KEYSTORE_LOCK) {
            KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
            keyStore.load(null);
            KeyStore.Entry entry = keyStore.getEntry(alias, null);
            if (entry instanceof KeyStore.SecretKeyEntry) {
                return ((KeyStore.SecretKeyEntry) entry).getSecretKey();
            }
            if (entry != null) {
                throw new KeyStoreException("Unexpected entry type for alias=" + alias);
            }
            if (!mayCreate) {
                throw new KeyStoreException("Missing existing key alias=" + alias);
            }

            KeyGenerator keyGenerator = KeyGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_AES,
                    ANDROID_KEYSTORE
            );
            KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
                    alias,
                    KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT
            )
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256)
                    .build();
            keyGenerator.init(spec);
            return keyGenerator.generateKey();
        }
    }

    private static final class StoredValue {
        final boolean present;
        final String value;

        private StoredValue(boolean present, String value) {
            this.present = present;
            this.value = value;
        }

        static StoredValue absent() {
            return new StoredValue(false, null);
        }

        static StoredValue present(String value) {
            return new StoredValue(true, value);
        }

        static StoredValue presentUnreadable() {
            return new StoredValue(true, null);
        }
    }
}
