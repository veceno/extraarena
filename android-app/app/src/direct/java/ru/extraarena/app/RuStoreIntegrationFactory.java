package ru.extraarena.app;

import android.content.Intent;
import android.os.Bundle;

final class RuStoreIntegrationFactory {
    private RuStoreIntegrationFactory() {
    }

    static RuStoreIntegration create(MainActivity activity) {
        return new NoopRuStoreIntegration();
    }

    private static final class NoopRuStoreIntegration implements RuStoreIntegration {
        @Override
        public void onCreate(Bundle savedInstanceState, Intent intent) {
        }

        @Override
        public void onNewIntent(Intent intent) {
        }

        @Override
        public void checkOptionalUpdate(Runnable continueFlow) {
            if (continueFlow != null) {
                continueFlow.run();
            }
        }

        @Override
        public void startImmediateUpdate(Runnable fallback) {
            if (fallback != null) {
                fallback.run();
            }
        }

        @Override
        public boolean isPayAvailable() {
            return false;
        }

        @Override
        public String startPayment(String payloadJson) {
            return "{\"accepted\":false,\"error\":\"rustore_unavailable\",\"message\":\"RuStore Pay is not available in this build\"}";
        }
    }
}
