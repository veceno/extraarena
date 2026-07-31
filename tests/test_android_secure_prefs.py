import shutil
import subprocess
from pathlib import Path

import pytest


ANDROID_SOURCE = (
    Path("android-app")
    / "app"
    / "src"
    / "main"
    / "java"
    / "ru"
    / "extraarena"
    / "app"
)


def _source(name: str) -> str:
    return (ANDROID_SOURCE / name).read_text(encoding="utf-8")


def _method(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def _working_javac() -> Path | None:
    candidates = [
        shutil.which("javac"),
        "/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/javac",
        "/Applications/Android Studio Preview.app/Contents/jbr/Contents/Home/bin/javac",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            continue
        probe = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return path
    return None


def test_secure_prefs_crypto_failures_are_non_destructive():
    source = _source("SecurePrefs.java")
    get_string = _method(
        source,
        "static String getString(Context context, String key, String defaultValue)",
        "static boolean putString",
    )
    put_string = _method(
        source,
        "static boolean putString(Context context, String key, String value)",
        "static void remove",
    )

    assert "deleteEntry(" not in source
    assert "deleteKeyAlias" not in source
    assert ".remove(key)" not in get_string
    assert "Keeping unreadable encrypted value" in get_string
    assert get_string.index("return defaultValue") < get_string.index("SharedPreferences legacyPrefs")

    assert "decrypt(key, existing.value) == null" in put_string
    assert "Refusing to overwrite unreadable encrypted value" in put_string
    assert put_string.index("decrypt(key, existing.value)") < put_string.index(".putString(key, encrypted)")
    assert "Retrying encryption without changing Keystore state" in source
    assert "Retrying decryption without changing stored data or Keystore state" in source


def test_secure_prefs_uses_additive_versioned_keys_and_authenticated_v2_records():
    source = _source("SecurePrefs.java")
    envelope = _source("SecurePrefsEnvelope.java")

    assert 'KEY_ALIAS_V1 = "extraarena_app_prefs_v1"' in source
    assert 'KEY_ALIAS_V2 = "extraarena_app_prefs_v2"' in source
    assert "CURRENT_VERSION = VERSION_V2" in envelope
    assert "SecurePrefsEnvelope.VERSION_V1.equals(version)" in source
    assert "SecurePrefsEnvelope.VERSION_V2.equals(version)" in source
    assert "getKey(alias, false)" in source
    assert "cipher.updateAAD(associatedData(preferenceKey))" in source
    assert "!hasStoredVersion(" in source
    assert "if (!mayCreate)" in source


def test_secure_prefs_envelope_codec_keeps_legacy_values_readable(tmp_path: Path):
    javac = _working_javac()
    java = javac.with_name("java") if javac is not None else None
    if javac is None or java is None or not java.is_file():
        pytest.skip("JDK is required for the pure-Java envelope unit test")

    package_dir = tmp_path / "ru" / "extraarena" / "app"
    package_dir.mkdir(parents=True)
    production_source = ANDROID_SOURCE / "SecurePrefsEnvelope.java"
    copied_source = package_dir / production_source.name
    copied_source.write_text(production_source.read_text(encoding="utf-8"), encoding="utf-8")

    harness = package_dir / "SecurePrefsEnvelopeHarness.java"
    harness.write_text(
        """
package ru.extraarena.app;

public final class SecurePrefsEnvelopeHarness {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        String encoded = SecurePrefsEnvelope.encodeCurrent("iv64", "cipher64");
        check("v2:iv64:cipher64".equals(encoded), "new writes must be v2");

        SecurePrefsEnvelope.Parsed current = SecurePrefsEnvelope.parse(encoded);
        check(current != null, "v2 must parse");
        check("v2".equals(current.version), "v2 version");
        check(!current.legacy, "v2 is not legacy");

        SecurePrefsEnvelope.Parsed legacy = SecurePrefsEnvelope.parse("oldIv64:oldCipher64");
        check(legacy != null, "legacy v1 must parse");
        check("v1".equals(legacy.version), "legacy maps to v1 key alias");
        check(legacy.legacy, "legacy marker");
        check("oldIv64".equals(legacy.ivBase64), "legacy IV preserved");
        check("oldCipher64".equals(legacy.ciphertextBase64), "legacy ciphertext preserved");

        SecurePrefsEnvelope.Parsed explicitV1 = SecurePrefsEnvelope.parse("v1:oldIv:oldCipher");
        check(explicitV1 != null, "explicit v1 must parse");
        check("v1".equals(explicitV1.version), "explicit v1 version");
        check(!explicitV1.legacy, "explicit v1 uses versioned envelope");

        SecurePrefsEnvelope.Parsed future = SecurePrefsEnvelope.parse("v3:newIv:newCipher");
        check(future != null, "future envelope remains identifiable");
        check("v3".equals(future.version), "future version preserved");

        check(SecurePrefsEnvelope.parse("broken") == null, "missing separator rejected");
        check(SecurePrefsEnvelope.parse("v2::cipher") == null, "empty IV rejected");
        check(SecurePrefsEnvelope.parse("v2:iv:") == null, "empty ciphertext rejected");
        check(SecurePrefsEnvelope.parse("v2:iv:cipher:extra") == null, "extra field rejected");
    }
}
""".strip(),
        encoding="utf-8",
    )

    subprocess.run(
        [str(javac), "-d", str(tmp_path), str(copied_source), str(harness)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(java), "-cp", str(tmp_path), "ru.extraarena.app.SecurePrefsEnvelopeHarness"],
        check=True,
        capture_output=True,
        text=True,
    )
