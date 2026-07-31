package ru.extraarena.app;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;

import androidx.core.content.FileProvider;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileInputStream;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class DirectApkUpdateInstaller implements ApkUpdateInstaller {
    private static final String PREFS = "extraarena_apk_update";
    private static final String KEY_DOWNLOAD_ID = "download_id";
    private static final String KEY_VERSION_CODE = "version_code";
    private static final String KEY_SHA256 = "sha256";
    private static final String KEY_FILE_PATH = "file_path";
    private static final long POLL_INTERVAL_MS = 500L;

    private final MainActivity activity;
    private final DownloadManager downloadManager;
    private final SharedPreferences prefs;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ExecutorService verifier = Executors.newSingleThreadExecutor();
    private final BroadcastReceiver completionReceiver;

    private Release release;
    private Listener listener;
    private long downloadId = -1L;
    private File destination;
    private File verifiedFile;
    private boolean destroyed;
    private boolean receiverRegistered;
    private boolean awaitingPermission;
    private boolean verificationInProgress;
    private long verificationToken;

    DirectApkUpdateInstaller(MainActivity activity) {
        this.activity = activity;
        this.downloadManager = (DownloadManager) activity.getSystemService(Context.DOWNLOAD_SERVICE);
        this.prefs = activity.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        this.completionReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent == null || !DownloadManager.ACTION_DOWNLOAD_COMPLETE.equals(intent.getAction())) {
                    return;
                }
                long completedId = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L);
                if (completedId == downloadId) {
                    inspectDownload();
                }
            }
        };
        registerReceiver();
        cleanupInstalledUpdateArtifacts();
    }

    @Override
    public boolean isSupported() {
        return downloadManager != null;
    }

    @Override
    public void start(Release requestedRelease, Listener requestedListener) {
        listener = requestedListener;
        release = requestedRelease;
        verifiedFile = null;
        awaitingPermission = false;
        verificationInProgress = false;
        verificationToken++;

        String validationError = validateManifest(requestedRelease);
        if (validationError != null) {
            emit(State.ERROR, 0, 0L, 0L, validationError);
            return;
        }
        if (downloadManager == null) {
            emit(State.ERROR, 0, 0L, requestedRelease.sizeBytes, "Системный загрузчик Android недоступен");
            return;
        }

        File baseDir = activity.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (baseDir == null) {
            emit(State.ERROR, 0, 0L, requestedRelease.sizeBytes, "Не удалось открыть защищённое хранилище обновлений");
            return;
        }
        File updateDir = new File(baseDir, "updates");
        if (!updateDir.exists() && !updateDir.mkdirs()) {
            emit(State.ERROR, 0, 0L, requestedRelease.sizeBytes, "Не удалось подготовить папку обновлений");
            return;
        }
        destination = new File(
                updateDir,
                "extraarena-" + requestedRelease.versionCode + "-" + normalizeDigest(requestedRelease.sha256).substring(0, 12) + ".apk"
        );

        long savedId = prefs.getLong(KEY_DOWNLOAD_ID, -1L);
        int savedVersion = prefs.getInt(KEY_VERSION_CODE, -1);
        String savedSha = prefs.getString(KEY_SHA256, "");
        String savedPath = prefs.getString(KEY_FILE_PATH, "");
        if (savedId >= 0L
                && savedVersion == requestedRelease.versionCode
                && normalizeDigest(savedSha).equals(normalizeDigest(requestedRelease.sha256))
                && destination.getAbsolutePath().equals(savedPath)) {
            downloadId = savedId;
            inspectDownload();
            return;
        }

        // A different requested release supersedes any app-owned in-flight file.
        // Do not leave an orphaned DownloadManager job after replacing its prefs.
        discardSavedDownload(updateDir, savedId, savedPath);

        if (destination.exists() && !destination.delete()) {
            emit(State.ERROR, 0, 0L, requestedRelease.sizeBytes, "Не удалось заменить старый файл обновления");
            return;
        }

        try {
            DownloadManager.Request request = new DownloadManager.Request(Uri.parse(requestedRelease.downloadUrl));
            request.setTitle("ExtraArena " + requestedRelease.versionName);
            request.setDescription("Загрузка обновления игры");
            request.setMimeType("application/vnd.android.package-archive");
            request.setAllowedOverMetered(true);
            request.setAllowedOverRoaming(false);
            request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE);
            request.setDestinationUri(Uri.fromFile(destination));
            downloadId = downloadManager.enqueue(request);
            prefs.edit()
                    .putLong(KEY_DOWNLOAD_ID, downloadId)
                    .putInt(KEY_VERSION_CODE, requestedRelease.versionCode)
                    .putString(KEY_SHA256, normalizeDigest(requestedRelease.sha256))
                    .putString(KEY_FILE_PATH, destination.getAbsolutePath())
                    .apply();
            emit(State.DOWNLOADING, 0, 0L, requestedRelease.sizeBytes, "Загружаем обновление…");
            schedulePoll();
        } catch (Exception error) {
            clearDownloadState(false);
            emit(State.ERROR, 0, 0L, requestedRelease.sizeBytes, "Не удалось начать загрузку. Проверь подключение.");
        }
    }

    @Override
    public void onResume() {
        if (destroyed) {
            return;
        }
        if (awaitingPermission && verifiedFile != null) {
            if (canRequestPackageInstalls()) {
                awaitingPermission = false;
                launchInstaller(verifiedFile);
            } else {
                emit(
                        State.PERMISSION_REQUIRED,
                        100,
                        verifiedFile.length(),
                        release == null ? verifiedFile.length() : release.sizeBytes,
                        "Разрешение ещё не включено. Нажми кнопку, когда будешь готов повторить."
                );
            }
            return;
        }
        // Returning from the system installer must not reopen it automatically.
        // The dialog remains visible and lets the user explicitly try again.
        if (verifiedFile != null) {
            return;
        }
        if (downloadId >= 0L && release != null) {
            inspectDownload();
        }
    }

    @Override
    public void cancel() {
        if (downloadManager != null && downloadId >= 0L) {
            try {
                downloadManager.remove(downloadId);
            } catch (Exception ignored) {
            }
        }
        clearDownloadState(true);
        emit(State.IDLE, 0, 0L, release == null ? 0L : release.sizeBytes, "Загрузка отменена");
    }

    @Override
    public void destroy() {
        destroyed = true;
        verificationToken++;
        handler.removeCallbacksAndMessages(null);
        verifier.shutdownNow();
        if (receiverRegistered) {
            try {
                activity.unregisterReceiver(completionReceiver);
            } catch (Exception ignored) {
            }
            receiverRegistered = false;
        }
        listener = null;
    }

    private void registerReceiver() {
        try {
            IntentFilter filter = new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE);
            ContextCompat.registerReceiver(
                    activity,
                    completionReceiver,
                    filter,
                    ContextCompat.RECEIVER_NOT_EXPORTED
            );
            receiverRegistered = true;
        } catch (Exception ignored) {
            receiverRegistered = false;
        }
    }

    private void cleanupInstalledUpdateArtifacts() {
        File baseDir = activity.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (baseDir == null) {
            return;
        }
        File updateDir = new File(baseDir, "updates");
        if (!updateDir.isDirectory()) {
            return;
        }

        long savedId = prefs.getLong(KEY_DOWNLOAD_ID, -1L);
        int savedVersion = prefs.getInt(KEY_VERSION_CODE, -1);
        String savedPath = prefs.getString(KEY_FILE_PATH, "");
        String preservedPath = null;
        if (savedId >= 0L && savedVersion > BuildConfig.VERSION_CODE && !savedPath.isEmpty()) {
            preservedPath = canonicalOwnedPath(updateDir, savedPath);
        } else {
            discardSavedDownload(updateDir, savedId, savedPath);
        }

        File[] files = updateDir.listFiles();
        if (files == null) {
            return;
        }
        for (File file : files) {
            if (!file.isFile() || !file.getName().endsWith(".apk")) {
                continue;
            }
            String candidate = canonicalOwnedPath(updateDir, file.getAbsolutePath());
            if (candidate != null && !candidate.equals(preservedPath)) {
                file.delete();
            }
        }
    }

    private void discardSavedDownload(File updateDir, long savedId, String savedPath) {
        if (downloadManager != null && savedId >= 0L) {
            try {
                downloadManager.remove(savedId);
            } catch (Exception ignored) {
            }
        }
        String ownedPath = canonicalOwnedPath(updateDir, savedPath);
        if (ownedPath != null) {
            File savedFile = new File(ownedPath);
            if (savedFile.isFile()) {
                savedFile.delete();
            }
        }
        prefs.edit().clear().apply();
    }

    private String canonicalOwnedPath(File updateDir, String candidatePath) {
        if (candidatePath == null || candidatePath.isEmpty()) {
            return null;
        }
        try {
            File candidate = new File(candidatePath).getCanonicalFile();
            File ownedDir = updateDir.getCanonicalFile();
            if (ownedDir.equals(candidate.getParentFile()) && candidate.getName().endsWith(".apk")) {
                return candidate.getAbsolutePath();
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private String validateManifest(Release candidate) {
        if (candidate == null) {
            return "Сервер не передал данные обновления";
        }
        Uri uri;
        try {
            uri = Uri.parse(candidate.downloadUrl);
        } catch (Exception error) {
            return "Некорректная ссылка обновления";
        }
        if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null) {
            return "APK можно загрузить только по защищённому HTTPS-соединению";
        }
        if (candidate.versionCode <= BuildConfig.VERSION_CODE) {
            return "Сервер предложил не более новую версию приложения";
        }
        if (!activity.getPackageName().equals(candidate.packageName)) {
            return "Обновление предназначено для другого приложения";
        }
        if (!normalizeDigest(candidate.sha256).matches("[0-9a-f]{64}")) {
            return "У обновления отсутствует проверочная SHA-256 сумма";
        }
        if (!normalizeDigest(candidate.signingCertSha256).matches("[0-9a-f]{64}")) {
            return "Сервер не передал доверенную подпись релиза";
        }
        if (candidate.sizeBytes <= 0L) {
            return "Сервер не передал размер APK";
        }
        return null;
    }

    private void schedulePoll() {
        handler.removeCallbacksAndMessages(null);
        handler.postDelayed(this::inspectDownload, POLL_INTERVAL_MS);
    }

    private void inspectDownload() {
        if (destroyed || release == null || downloadId < 0L || downloadManager == null) {
            return;
        }
        DownloadManager.Query query = new DownloadManager.Query().setFilterById(downloadId);
        try (Cursor cursor = downloadManager.query(query)) {
            if (cursor == null || !cursor.moveToFirst()) {
                clearDownloadState(false);
                emit(State.ERROR, 0, 0L, release.sizeBytes, "Загрузка больше не найдена. Запусти её заново.");
                return;
            }
            int status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS));
            long downloaded = Math.max(0L, cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)));
            long total = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES));
            if (total <= 0L) {
                total = release.sizeBytes;
            }
            int progress = total > 0L ? (int) Math.min(100L, downloaded * 100L / total) : 0;
            if (status == DownloadManager.STATUS_SUCCESSFUL) {
                verifyDownloadedApk();
                return;
            }
            if (status == DownloadManager.STATUS_FAILED) {
                int reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON));
                clearDownloadState(false);
                emit(State.ERROR, progress, downloaded, total, "Загрузка прервана (код " + reason + "). Можно повторить.");
                return;
            }
            String message = status == DownloadManager.STATUS_PAUSED
                    ? "Загрузка приостановлена системой"
                    : status == DownloadManager.STATUS_PENDING
                    ? "Ожидаем системный загрузчик…"
                    : "Загружаем обновление…";
            emit(State.DOWNLOADING, progress, downloaded, total, message);
            schedulePoll();
        } catch (Exception error) {
            clearDownloadState(false);
            emit(State.ERROR, 0, 0L, release.sizeBytes, "Не удалось прочитать состояние загрузки");
        }
    }

    private void verifyDownloadedApk() {
        if (verificationInProgress) {
            return;
        }
        if (destination == null || !destination.isFile()) {
            clearDownloadState(false);
            emit(State.ERROR, 0, 0L, release.sizeBytes, "Загруженный APK не найден");
            return;
        }
        verificationInProgress = true;
        emit(State.VERIFYING, 100, destination.length(), release.sizeBytes, "Проверяем файл и подпись…");
        File apk = destination;
        Release expected = release;
        long token = ++verificationToken;
        verifier.execute(() -> {
            String error = verifyApk(apk, expected);
            activity.runOnUiThread(() -> {
                if (destroyed || token != verificationToken || expected != release) {
                    return;
                }
                verificationInProgress = false;
                if (error != null) {
                    clearDownloadState(true);
                    emit(State.ERROR, 0, 0L, expected.sizeBytes, error);
                    return;
                }
                verifiedFile = apk;
                emit(State.READY, 100, apk.length(), expected.sizeBytes, "APK проверен. Открываем установщик Android…");
                requestInstall(apk);
            });
        });
    }

    private String verifyApk(File apk, Release expected) {
        try {
            if (apk.length() != expected.sizeBytes) {
                return "Размер загруженного APK не совпал с релизом";
            }
            if (!sha256(apk).equals(normalizeDigest(expected.sha256))) {
                return "SHA-256 APK не совпала. Файл удалён — загрузка безопасно остановлена.";
            }

            PackageManager packageManager = activity.getPackageManager();
            int signingFlags = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
                    ? PackageManager.GET_SIGNING_CERTIFICATES
                    : PackageManager.GET_SIGNATURES;
            PackageInfo archive = packageManager.getPackageArchiveInfo(
                    apk.getAbsolutePath(),
                    signingFlags
            );
            if (archive == null || !activity.getPackageName().equals(archive.packageName)) {
                return "APK относится к другому приложению";
            }
            long archiveVersion = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
                    ? archive.getLongVersionCode()
                    : archive.versionCode;
            if (archiveVersion != expected.versionCode || archiveVersion <= BuildConfig.VERSION_CODE) {
                return "Версия внутри APK не совпала с опубликованным релизом";
            }

            PackageInfo installed = packageManager.getPackageInfo(
                    activity.getPackageName(),
                    signingFlags
            );
            Set<String> installedLineage = signingDigests(installed, true);
            Set<String> archiveLineage = signingDigests(archive, true);
            Set<String> archiveCurrent = signingDigests(archive, false);
            boolean compatibleSigner = false;
            for (String digest : installedLineage) {
                if (archiveLineage.contains(digest)) {
                    compatibleSigner = true;
                    break;
                }
            }
            if (!compatibleSigner || archiveCurrent.isEmpty()) {
                return "Подпись APK не совпадает с установленной ExtraArena";
            }
            String pinnedSigner = normalizeDigest(expected.signingCertSha256);
            if (!pinnedSigner.isEmpty() && !archiveCurrent.contains(pinnedSigner)) {
                return "Подпись APK не совпадает с подписью опубликованного релиза";
            }
            return null;
        } catch (Exception error) {
            return "Android не смог проверить APK. Установка остановлена.";
        }
    }

    private Set<String> signingDigests(PackageInfo info, boolean includeHistory) throws Exception {
        Set<String> result = new HashSet<>();
        if (info == null) {
            return result;
        }
        Signature[] signatures = null;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && info.signingInfo != null) {
            signatures = includeHistory && !info.signingInfo.hasMultipleSigners()
                    ? info.signingInfo.getSigningCertificateHistory()
                    : info.signingInfo.getApkContentsSigners();
        } else if (info.signatures != null) {
            signatures = info.signatures;
        }
        if (signatures == null) {
            return result;
        }
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (Signature signature : signatures) {
            if (signature != null) {
                result.add(hex(digest.digest(signature.toByteArray())));
                digest.reset();
            }
        }
        return result;
    }

    private void requestInstall(File apk) {
        if (!canRequestPackageInstalls()) {
            awaitingPermission = true;
            emit(
                    State.PERMISSION_REQUIRED,
                    100,
                    apk.length(),
                    release == null ? apk.length() : release.sizeBytes,
                    "Разреши установку обновлений от ExtraArena. Мы вернёмся к установке автоматически."
            );
            try {
                Intent permissionIntent = new Intent(
                        Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                        Uri.parse("package:" + activity.getPackageName())
                );
                activity.startActivity(permissionIntent);
            } catch (Exception error) {
                emit(State.ERROR, 100, apk.length(), release.sizeBytes, "Открой разрешение «Установка неизвестных приложений» в настройках Android");
            }
            return;
        }
        launchInstaller(apk);
    }

    private boolean canRequestPackageInstalls() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.O
                || activity.getPackageManager().canRequestPackageInstalls();
    }

    private void launchInstaller(File apk) {
        try {
            Uri contentUri = FileProvider.getUriForFile(
                    activity,
                    activity.getPackageName() + ".updates",
                    apk
            );
            Intent intent = new Intent(Intent.ACTION_INSTALL_PACKAGE);
            intent.setData(contentUri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.putExtra(Intent.EXTRA_NOT_UNKNOWN_SOURCE, true);
            intent.putExtra(Intent.EXTRA_RETURN_RESULT, false);
            emit(State.INSTALLING, 100, apk.length(), release == null ? apk.length() : release.sizeBytes, "Подтверди обновление в системном установщике");
            activity.startActivity(intent);
        } catch (Exception error) {
            emit(State.ERROR, 100, apk.length(), release == null ? apk.length() : release.sizeBytes, "Не удалось открыть системный установщик Android");
        }
    }

    private void emit(State state, int progress, long downloaded, long total, String message) {
        Listener current = listener;
        if (current != null && !destroyed) {
            current.onState(state, progress, downloaded, total, message == null ? "" : message);
        }
    }

    private void clearDownloadState(boolean deleteFile) {
        handler.removeCallbacksAndMessages(null);
        downloadId = -1L;
        prefs.edit().clear().apply();
        if (deleteFile && destination != null && destination.isFile()) {
            destination.delete();
        }
        verifiedFile = null;
        awaitingPermission = false;
        verificationInProgress = false;
        verificationToken++;
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[1024 * 1024];
        try (FileInputStream input = new FileInputStream(file)) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        return hex(digest.digest());
    }

    private static String normalizeDigest(String value) {
        return value == null
                ? ""
                : value.replace(":", "").replaceAll("\\s+", "").toLowerCase(Locale.ROOT);
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return result.toString();
    }
}
