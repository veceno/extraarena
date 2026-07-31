package ru.extraarena.app;

/**
 * Versioned, Android-independent wire format for values stored by {@link SecurePrefs}.
 *
 * <p>Values written before the envelope existed used {@code iv:ciphertext}; those values are
 * treated as v1 and are never rewritten merely because they were read successfully. New v2
 * values use {@code v2:iv:ciphertext}. Keeping parsing separate makes future key migrations
 * additive: old aliases remain available for old records.</p>
 */
final class SecurePrefsEnvelope {
    static final String VERSION_V1 = "v1";
    static final String VERSION_V2 = "v2";
    static final String CURRENT_VERSION = VERSION_V2;

    private SecurePrefsEnvelope() {
    }

    static String encodeCurrent(String ivBase64, String ciphertextBase64) {
        if (isEmpty(ivBase64) || isEmpty(ciphertextBase64)) {
            throw new IllegalArgumentException("Encrypted envelope fields must not be empty");
        }
        return CURRENT_VERSION + ":" + ivBase64 + ":" + ciphertextBase64;
    }

    static Parsed parse(String encoded) {
        if (isEmpty(encoded)) {
            return null;
        }
        int firstSeparator = encoded.indexOf(':');
        if (firstSeparator <= 0 || firstSeparator == encoded.length() - 1) {
            return null;
        }

        String firstField = encoded.substring(0, firstSeparator);
        int secondSeparator = encoded.indexOf(':', firstSeparator + 1);
        if (isVersionToken(firstField)) {
            if (secondSeparator <= firstSeparator + 1
                    || secondSeparator == encoded.length() - 1
                    || encoded.indexOf(':', secondSeparator + 1) >= 0) {
                return null;
            }
            return new Parsed(
                    firstField,
                    encoded.substring(firstSeparator + 1, secondSeparator),
                    encoded.substring(secondSeparator + 1),
                    false
            );
        }

        // Legacy v1 values have exactly two Base64 fields and no explicit version.
        if (secondSeparator >= 0) {
            return null;
        }
        return new Parsed(
                VERSION_V1,
                firstField,
                encoded.substring(firstSeparator + 1),
                true
        );
    }

    private static boolean isVersionToken(String value) {
        if (value.length() < 2 || value.charAt(0) != 'v') {
            return false;
        }
        for (int i = 1; i < value.length(); i++) {
            if (!Character.isDigit(value.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean isEmpty(String value) {
        return value == null || value.isEmpty();
    }

    static final class Parsed {
        final String version;
        final String ivBase64;
        final String ciphertextBase64;
        final boolean legacy;

        Parsed(String version, String ivBase64, String ciphertextBase64, boolean legacy) {
            this.version = version;
            this.ivBase64 = ivBase64;
            this.ciphertextBase64 = ciphertextBase64;
            this.legacy = legacy;
        }
    }
}
