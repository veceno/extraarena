package ru.extraarena.app;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

public final class BaseUrlStoreTest {
    @Test
    public void normalizeCanonicalizesEquivalentHttpBackendScopes() {
        assertEquals(
                "https://example.com/api/",
                BaseUrlStore.normalize("HTTPS://Example.COM:443/edge/../api#ignored")
        );
        assertEquals(
                "https://example.com/api/",
                BaseUrlStore.normalize("https://example.com/api/?ignored=true")
        );
        assertEquals(
                "http://example.com/",
                BaseUrlStore.normalize("http://EXAMPLE.com:80")
        );
    }

    @Test
    public void normalizeKeepsMeaningfulPortAndPathDistinctions() {
        assertEquals(
                "https://example.com:8443/api/",
                BaseUrlStore.normalize("https://Example.com:8443/api")
        );
        assertEquals(
                "https://example.com/api//v1/",
                BaseUrlStore.normalize("https://example.com/api//v1")
        );
        assertEquals(
                "https://example.com/api/a%2Fb/",
                BaseUrlStore.normalize("https://example.com/api/a%2Fb")
        );
        assertEquals(
                "http://[::1]:8080/api/",
                BaseUrlStore.normalize("http://[::1]:8080/api")
        );
    }

    @Test
    public void officialEntrypointsShareOneDurableCredentialScope() {
        assertEquals(
                BaseUrlStore.identityScope(BuildConfig.DEFAULT_BASE_URL),
                BaseUrlStore.identityScope(BuildConfig.RU_BASE_URL)
        );
        assertEquals(
                DeviceRegistrar.authTokenStorageKey(BuildConfig.DEFAULT_BASE_URL, ""),
                DeviceRegistrar.authTokenStorageKey(BuildConfig.RU_BASE_URL, "")
        );
    }

    @Test
    public void customWhitelistTenantsNeverShareCredentials() {
        String baseUrl = "https://custom.example/api/";
        assertNotEquals(
                BaseUrlStore.identityScope(baseUrl, "tenant-a"),
                BaseUrlStore.identityScope(baseUrl, "tenant-b")
        );
        assertNotEquals(
                DeviceRegistrar.authTokenStorageKey(baseUrl, "tenant-a"),
                DeviceRegistrar.authTokenStorageKey(baseUrl, "tenant-b")
        );
        assertNotEquals(
                BaseUrlStore.identityScope(BuildConfig.DEFAULT_BASE_URL, "tenant-a"),
                BaseUrlStore.identityScope(BuildConfig.DEFAULT_BASE_URL, "")
        );
        assertNotEquals(
                DeviceRegistrar.authTokenStorageKey(BuildConfig.RU_BASE_URL, "tenant-a"),
                DeviceRegistrar.authTokenStorageKey(BuildConfig.DEFAULT_BASE_URL, "")
        );
    }

    @Test
    public void queuedBackendBindingKeepsItsCapturedEndpointAndTenant() {
        DeviceRegistrar.BackendBinding first = new DeviceRegistrar.BackendBinding(
                "https://first.example/api/",
                "tenant-a"
        );
        DeviceRegistrar.BackendBinding second = new DeviceRegistrar.BackendBinding(
                "https://second.example/",
                "tenant-b"
        );

        assertEquals(
                "https://first.example/api/push/register",
                first.endpoint("/push/register")
        );
        assertNotEquals(first.credentialScope, second.credentialScope);
    }
}
