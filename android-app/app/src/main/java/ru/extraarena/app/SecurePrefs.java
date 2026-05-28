package ru.extraarena.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import android.util.Log;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

final class SecurePrefs {
    private static final String TAG = "EASecurePrefs";
    private static final String SECURE_PREFS = "extraarena_secure";
    private static final String LEGACY_PREFS = "extraarena_app";
    private static final String KEY_ALIAS = "extraarena_app_prefs_v1";
    private static final String ANDROID_KEYSTORE = "AndroidKeyStore";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";
    private static final int IV_BYTES = 12;
    private static final int GCM_TAG_BITS = 128;

    private SecurePrefs() {
    }

    static String getString(Context context, String key, String defaultValue) {
        Context appContext = context.getApplicationContext();
        SharedPreferences securePrefs = securePrefs(appContext);
        String encrypted = securePrefs.getString(key, null);
        if (encrypted != null) {
            String decrypted = decrypt(encrypted);
            if (decrypted != null) {
                return decrypted;
            }
            securePrefs.edit().remove(key).apply();
        }

        SharedPreferences legacyPrefs = legacyPrefs(appContext);
        String legacy = legacyPrefs.getString(key, null);
        if (legacy != null) {
            if (putString(appContext, key, legacy)) {
                legacyPrefs.edit().remove(key).apply();
            }
            return legacy;
        }
        return defaultValue;
    }

    static boolean putString(Context context, String key, String value) {
        Context appContext = context.getApplicationContext();
        if (value == null) {
            remove(appContext, key);
            return true;
        }
        String encrypted = encrypt(value);
        if (encrypted == null) {
            Log.w(TAG, "Failed to encrypt value for key=" + key);
            return false;
        }
        boolean saved = securePrefs(appContext).edit().putString(key, encrypted).commit();
        if (!saved) {
            Log.w(TAG, "Failed to persist encrypted value for key=" + key);
            return false;
        }
        legacyPrefs(appContext).edit().remove(key).apply();
        return true;
    }

    static void remove(Context context, String key) {
        Context appContext = context.getApplicationContext();
        securePrefs(appContext).edit().remove(key).apply();
        legacyPrefs(appContext).edit().remove(key).apply();
    }

    private static SharedPreferences securePrefs(Context context) {
        return context.getSharedPreferences(SECURE_PREFS, Context.MODE_PRIVATE);
    }

    private static SharedPreferences legacyPrefs(Context context) {
        return context.getSharedPreferences(LEGACY_PREFS, Context.MODE_PRIVATE);
    }

    private static String encrypt(String value) {
        String encrypted = encryptWithCurrentKey(value);
        if (encrypted != null) {
            return encrypted;
        }
        if (deleteKeyAlias()) {
            Log.w(TAG, "Reset invalid Android Keystore alias; retrying encryption");
            return encryptWithCurrentKey(value);
        }
        return null;
    }

    private static String encryptWithCurrentKey(String value) {
        try {
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey());
            byte[] iv = cipher.getIV();
            if (iv == null || iv.length != IV_BYTES) {
                Log.w(TAG, "Encryption produced invalid IV");
                return null;
            }
            byte[] ciphertext = cipher.doFinal(value.getBytes(StandardCharsets.UTF_8));
            return Base64.encodeToString(iv, Base64.NO_WRAP)
                    + ":"
                    + Base64.encodeToString(ciphertext, Base64.NO_WRAP);
        } catch (Exception e) {
            Log.w(TAG, "Encryption failed", e);
            return null;
        }
    }

    private static String decrypt(String encoded) {
        try {
            String[] parts = encoded.split(":", 2);
            if (parts.length != 2) {
                return null;
            }
            byte[] iv = Base64.decode(parts[0], Base64.NO_WRAP);
            byte[] ciphertext = Base64.decode(parts[1], Base64.NO_WRAP);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), new GCMParameterSpec(GCM_TAG_BITS, iv));
            byte[] plaintext = cipher.doFinal(ciphertext);
            return new String(plaintext, StandardCharsets.UTF_8);
        } catch (Exception e) {
            Log.w(TAG, "Decryption failed", e);
            return null;
        }
    }

    private static SecretKey getOrCreateKey() throws Exception {
        KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
        keyStore.load(null);
        KeyStore.Entry entry = keyStore.getEntry(KEY_ALIAS, null);
        if (entry instanceof KeyStore.SecretKeyEntry) {
            return ((KeyStore.SecretKeyEntry) entry).getSecretKey();
        }
        if (entry != null) {
            keyStore.deleteEntry(KEY_ALIAS);
        }

        KeyGenerator keyGenerator = KeyGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_AES,
                ANDROID_KEYSTORE
        );
        KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT
        )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build();
        keyGenerator.init(spec);
        return keyGenerator.generateKey();
    }

    private static boolean deleteKeyAlias() {
        try {
            KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
            keyStore.load(null);
            if (keyStore.containsAlias(KEY_ALIAS)) {
                keyStore.deleteEntry(KEY_ALIAS);
            }
            return true;
        } catch (Exception e) {
            Log.w(TAG, "Failed to reset Android Keystore alias", e);
            return false;
        }
    }
}
