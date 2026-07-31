package ru.extraarena.app;

final class ApkUpdateInstallerFactory {
    private ApkUpdateInstallerFactory() {
    }

    static ApkUpdateInstaller create(MainActivity activity) {
        return new DirectApkUpdateInstaller(activity);
    }
}
