package ru.extraarena.app;

final class ApkUpdateInstallerFactory {
    private ApkUpdateInstallerFactory() {
    }

    static ApkUpdateInstaller create(MainActivity activity) {
        return new NoopApkUpdateInstaller();
    }

    private static final class NoopApkUpdateInstaller implements ApkUpdateInstaller {
        @Override
        public boolean isSupported() {
            return false;
        }

        @Override
        public void start(Release release, Listener listener) {
            if (listener != null) {
                listener.onState(State.ERROR, 0, 0L, 0L, "Для этой сборки обновление устанавливается через RuStore");
            }
        }

        @Override
        public void onResume() {
        }

        @Override
        public void cancel() {
        }

        @Override
        public void destroy() {
        }
    }
}
