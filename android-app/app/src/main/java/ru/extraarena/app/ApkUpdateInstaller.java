package ru.extraarena.app;

interface ApkUpdateInstaller {
    enum State {
        IDLE,
        DOWNLOADING,
        VERIFYING,
        PERMISSION_REQUIRED,
        READY,
        INSTALLING,
        ERROR
    }

    final class Release {
        final String releaseId;
        final int versionCode;
        final String versionName;
        final String downloadUrl;
        final long sizeBytes;
        final String sha256;
        final String packageName;
        final String signingCertSha256;

        Release(
                String releaseId,
                int versionCode,
                String versionName,
                String downloadUrl,
                long sizeBytes,
                String sha256,
                String packageName,
                String signingCertSha256
        ) {
            this.releaseId = releaseId == null ? "" : releaseId.trim();
            this.versionCode = versionCode;
            this.versionName = versionName == null ? "" : versionName.trim();
            this.downloadUrl = downloadUrl == null ? "" : downloadUrl.trim();
            this.sizeBytes = sizeBytes;
            this.sha256 = sha256 == null ? "" : sha256.trim();
            this.packageName = packageName == null ? "" : packageName.trim();
            this.signingCertSha256 = signingCertSha256 == null ? "" : signingCertSha256.trim();
        }
    }

    interface Listener {
        void onState(State state, int progressPercent, long downloadedBytes, long totalBytes, String message);
    }

    boolean isSupported();

    void start(Release release, Listener listener);

    void onResume();

    void cancel();

    void destroy();
}
