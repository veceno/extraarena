package ru.extraarena.app;

import android.content.Intent;
import android.os.Bundle;

interface RuStoreIntegration {
    void onCreate(Bundle savedInstanceState, Intent intent);

    void onNewIntent(Intent intent);

    void checkOptionalUpdate(Runnable continueFlow);

    void startImmediateUpdate(Runnable fallback);

    boolean isPayAvailable();

    String startPayment(String payloadJson);
}
