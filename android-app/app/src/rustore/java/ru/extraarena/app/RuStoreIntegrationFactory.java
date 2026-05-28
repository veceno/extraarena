package ru.extraarena.app;

final class RuStoreIntegrationFactory {
    private RuStoreIntegrationFactory() {
    }

    static RuStoreIntegration create(MainActivity activity) {
        return new RuStoreIntegrationImpl(activity);
    }
}
