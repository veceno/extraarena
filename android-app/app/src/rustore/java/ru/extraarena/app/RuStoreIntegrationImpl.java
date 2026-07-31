package ru.extraarena.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

import org.json.JSONObject;

import java.util.Collections;

import ru.rustore.sdk.appupdate.manager.RuStoreAppUpdateManager;
import ru.rustore.sdk.appupdate.manager.factory.RuStoreAppUpdateManagerFactory;
import ru.rustore.sdk.appupdate.listener.InstallStateUpdateListener;
import ru.rustore.sdk.appupdate.model.AppUpdateInfo;
import ru.rustore.sdk.appupdate.model.AppUpdateOptions;
import ru.rustore.sdk.appupdate.model.AppUpdateType;
import ru.rustore.sdk.appupdate.model.InstallState;
import ru.rustore.sdk.appupdate.model.InstallStatus;
import ru.rustore.sdk.appupdate.model.UpdateAvailability;
import ru.rustore.sdk.pay.RuStorePayClient;
import ru.rustore.sdk.pay.RuStorePayClientProvider;
import ru.rustore.sdk.pay.callback.PurchaseEventListener;
import ru.rustore.sdk.pay.model.AppUserEmail;
import ru.rustore.sdk.pay.model.AppUserId;
import ru.rustore.sdk.pay.model.ConsoleApplicationId;
import ru.rustore.sdk.pay.model.DeveloperPayload;
import ru.rustore.sdk.pay.model.InvoiceId;
import ru.rustore.sdk.pay.model.OrderId;
import ru.rustore.sdk.pay.model.PreferredPurchaseType;
import ru.rustore.sdk.pay.model.ProductId;
import ru.rustore.sdk.pay.model.ProductPurchaseParams;
import ru.rustore.sdk.pay.model.ProductPurchaseResult;
import ru.rustore.sdk.pay.model.PurchaseId;
import ru.rustore.sdk.pay.model.Quantity;
import ru.rustore.sdk.pay.model.RuStorePaymentException;
import ru.rustore.sdk.pay.model.SdkTheme;

final class RuStoreIntegrationImpl implements RuStoreIntegration {
    private static final String TAG = "EARuStore";

    private final MainActivity activity;
    private RuStorePayClient payClient;
    private RuStoreAppUpdateManager appUpdateManager;
    private final InstallStateUpdateListener installStateUpdateListener;
    private boolean updateListenerRegistered;
    private boolean updateRecoveryInProgress;
    private boolean flexibleCompletionInProgress;
    private boolean immediateFlowActive;
    private boolean immediateFlowAccepted;
    private Runnable requiredUpdateFallback;
    private boolean destroyed;
    private boolean availabilityKnown = false;
    private boolean payAvailable = false;

    RuStoreIntegrationImpl(MainActivity activity) {
        this.activity = activity;
        this.installStateUpdateListener = this::handleInstallState;
    }

    @Override
    public void onCreate(Bundle savedInstanceState, Intent intent) {
        ensurePayClient();
        RuStoreAppUpdateManager manager = ensureAppUpdateManager();
        recoverPendingUpdate(manager);
        proceedPaymentIntent(intent);
        refreshPaymentAvailability();
    }

    @Override
    public void onNewIntent(Intent intent) {
        proceedPaymentIntent(intent);
    }

    @Override
    public void onResume() {
        if (destroyed) {
            return;
        }
        // Keep listening while the Activity is alive. Pending work is recovered
        // once in onCreate; a cancelled installer must not reopen in a loop.
        registerUpdateListener(ensureAppUpdateManager());
    }

    @Override
    public void destroy() {
        destroyed = true;
        RuStoreAppUpdateManager manager = appUpdateManager;
        if (manager != null && updateListenerRegistered) {
            try {
                manager.unregisterListener(installStateUpdateListener);
            } catch (Throwable error) {
                Log.d(TAG, "RuStore update listener cleanup ignored", error);
            }
        }
        updateListenerRegistered = false;
        requiredUpdateFallback = null;
    }

    @Override
    public void checkOptionalUpdate(Runnable continueFlow) {
        RuStoreAppUpdateManager manager = ensureAppUpdateManager();
        if (manager == null) {
            runOnUi(continueFlow);
            return;
        }
        manager.getAppUpdateInfo()
                .addOnSuccessListener(info -> {
                    if (info != null && info.getInstallStatus() == InstallStatus.DOWNLOADED) {
                        completeFlexibleUpdate(manager);
                        runOnUi(continueFlow);
                        return;
                    }
                    if (isUpdateAvailable(info, AppUpdateType.FLEXIBLE)) {
                        manager.startUpdateFlow(info, updateOptions(AppUpdateType.FLEXIBLE))
                                .addOnSuccessListener(result -> runOnUi(continueFlow))
                                .addOnFailureListener(error -> runOnUi(continueFlow));
                    } else {
                        runOnUi(continueFlow);
                    }
                })
                .addOnFailureListener(error -> runOnUi(continueFlow));
    }

    @Override
    public void startImmediateUpdate(Runnable fallback) {
        requiredUpdateFallback = fallback;
        if (immediateFlowActive || immediateFlowAccepted) {
            return;
        }
        immediateFlowActive = true;
        RuStoreAppUpdateManager manager = ensureAppUpdateManager();
        if (manager == null) {
            failRequiredUpdateFlow();
            return;
        }
        manager.getAppUpdateInfo()
                .addOnSuccessListener(info -> {
                    if (!isUpdateAvailable(info, AppUpdateType.IMMEDIATE)) {
                        failRequiredUpdateFlow();
                        return;
                    }
                    startImmediateFlow(manager, info);
                })
                .addOnFailureListener(error -> failRequiredUpdateFlow());
    }

    @Override
    public boolean isPayAvailable() {
        return isRustoreConfigured() && (!availabilityKnown || payAvailable);
    }

    @Override
    public String startPayment(String payloadJson) {
        if (!isRustoreConfigured()) {
            return jsonResult(false, "rustore_app_id_missing", "RuStore Console App ID is not configured");
        }
        RuStorePayClient client = ensurePayClient();
        if (client == null) {
            return jsonResult(false, "rustore_client_unavailable", "RuStore Pay SDK is not initialized");
        }
        final JSONObject payload;
        try {
            payload = new JSONObject(payloadJson == null ? "{}" : payloadJson);
        } catch (Exception e) {
            return jsonResult(false, "invalid_payload", "Invalid RuStore payment payload");
        }
        String paymentId = payload.optString("payment_id", payload.optString("order_id", "")).trim();
        String productId = payload.optString("product_id", "").trim();
        if (paymentId.isEmpty() || productId.isEmpty()) {
            return jsonResult(false, "missing_fields", "payment_id and product_id are required");
        }

        activity.runOnUiThread(() -> startSdkPurchase(client, payload, paymentId, productId));
        return jsonResult(true, "", "");
    }

    private RuStorePayClient ensurePayClient() {
        if (payClient != null) {
            return payClient;
        }
        if (!isRustoreConfigured()) {
            return null;
        }
        try {
            payClient = RuStorePayClient.Companion.getInstance();
            return payClient;
        } catch (Throwable ignored) {
        }
        try {
            payClient = new RuStorePayClientProvider().provide(
                    activity,
                    new ConsoleApplicationId(BuildConfig.RUSTORE_CONSOLE_APP_ID),
                    Collections.emptyMap()
            );
            return payClient;
        } catch (Throwable error) {
            Log.w(TAG, "RuStore Pay init failed", error);
            return null;
        }
    }

    private RuStoreAppUpdateManager ensureAppUpdateManager() {
        if (appUpdateManager != null) {
            registerUpdateListener(appUpdateManager);
            return appUpdateManager;
        }
        try {
            appUpdateManager = RuStoreAppUpdateManagerFactory.INSTANCE.create(activity);
            registerUpdateListener(appUpdateManager);
            return appUpdateManager;
        } catch (Throwable error) {
            Log.w(TAG, "RuStore AppUpdate init failed", error);
            return null;
        }
    }

    private void registerUpdateListener(RuStoreAppUpdateManager manager) {
        if (manager == null || updateListenerRegistered || destroyed) {
            return;
        }
        try {
            manager.registerListener(installStateUpdateListener);
            updateListenerRegistered = true;
        } catch (Throwable error) {
            Log.w(TAG, "RuStore update listener registration failed", error);
        }
    }

    private void handleInstallState(InstallState state) {
        if (state == null || destroyed) {
            return;
        }
        if (state.getInstallStatus() == InstallStatus.DOWNLOADED && !immediateFlowActive) {
            if (requiredUpdateFallback == null && !immediateFlowAccepted) {
                completeFlexibleUpdate(ensureAppUpdateManager());
            }
        } else if (state.getInstallStatus() == InstallStatus.FAILED
                || state.getInstallStatus() == InstallStatus.DOWNLOAD_INTERRUPTED) {
            Log.w(TAG, "RuStore update download failed with code " + state.getInstallErrorCode());
            flexibleCompletionInProgress = false;
            if (requiredUpdateFallback != null
                    || immediateFlowActive
                    || immediateFlowAccepted) {
                failRequiredUpdateFlow();
            }
        }
    }

    private void recoverPendingUpdate(RuStoreAppUpdateManager manager) {
        if (manager == null || destroyed || updateRecoveryInProgress) {
            return;
        }
        updateRecoveryInProgress = true;
        manager.getAppUpdateInfo()
                .addOnSuccessListener(info -> {
                    updateRecoveryInProgress = false;
                    if (info != null && info.getInstallStatus() == InstallStatus.DOWNLOADED) {
                        completeFlexibleUpdate(manager);
                    } else if (info != null
                            && info.getUpdateAvailability()
                            == UpdateAvailability.DEVELOPER_TRIGGERED_UPDATE_IN_PROGRESS
                            && info.isUpdateTypeAllowed(AppUpdateType.IMMEDIATE)) {
                        // The Activity/process may have been recreated while the
                        // mandatory RuStore UI was open. Resume the same immediate
                        // flow; startImmediateUpdate will attach its fallback when
                        // the server manifest confirms the mandatory floor.
                        immediateFlowActive = true;
                        startImmediateFlow(manager, info);
                    }
                    // DOWNLOADING/PENDING flows are recovered by the registered listener.
                })
                .addOnFailureListener(error -> {
                    updateRecoveryInProgress = false;
                    Log.d(TAG, "RuStore pending update recovery unavailable", error);
                });
    }

    private void completeFlexibleUpdate(RuStoreAppUpdateManager manager) {
        if (manager == null || destroyed || flexibleCompletionInProgress) {
            return;
        }
        flexibleCompletionInProgress = true;
        runOnUi(() -> manager.completeUpdate(updateOptions(AppUpdateType.FLEXIBLE))
                .addOnSuccessListener(result -> Log.d(TAG, "RuStore update installation requested"))
                .addOnFailureListener(error -> {
                    flexibleCompletionInProgress = false;
                    Log.w(TAG, "RuStore update installation failed", error);
                }));
    }

    private void refreshPaymentAvailability() {
        RuStorePayClient client = ensurePayClient();
        if (client == null) {
            availabilityKnown = true;
            payAvailable = false;
            return;
        }
        try {
            client.getPurchaseInteractor().getPurchaseAvailability()
                    .addOnSuccessListener(result -> {
                        availabilityKnown = true;
                        payAvailable = result != null && result.getClass().getName().endsWith("$Available");
                    })
                    .addOnFailureListener(error -> {
                        availabilityKnown = true;
                        payAvailable = false;
                    });
        } catch (Throwable error) {
            availabilityKnown = true;
            payAvailable = false;
        }
    }

    private void startSdkPurchase(
            RuStorePayClient client,
            JSONObject payload,
            String paymentId,
            String productId
    ) {
        try {
            int quantity = Math.max(1, payload.optInt("quantity", 1));
            String orderId = payload.optString("order_id", paymentId);
            String developerPayload = payload.optString("developer_payload", paymentId);
            String appUserId = payload.optString("app_user_id", "");
            String appUserEmail = payload.optString("app_user_email", "");

            ProductPurchaseParams params = new ProductPurchaseParams(
                    new ProductId(productId),
                    new Quantity(quantity),
                    new OrderId(orderId),
                    new DeveloperPayload(developerPayload),
                    appUserId.isEmpty() ? null : new AppUserId(appUserId),
                    appUserEmail.isEmpty() ? null : new AppUserEmail(appUserEmail)
            );

            PurchaseEventListener listener = new PurchaseEventListener() {
                @Override
                public void onPurchaseCreated(PurchaseId purchaseId, InvoiceId invoiceId) {
                    emitPaymentEvent("purchase_created", paymentId, orderId, productId, purchaseId, invoiceId, "");
                }

                @Override
                public void onPaymentStarted(PurchaseId purchaseId, InvoiceId invoiceId) {
                    emitPaymentEvent("payment_started", paymentId, orderId, productId, purchaseId, invoiceId, "");
                }

                @Override
                public void onPaymentCompleted(PurchaseId purchaseId, InvoiceId invoiceId) {
                    emitPaymentEvent("payment_completed", paymentId, orderId, productId, purchaseId, invoiceId, "");
                }

                @Override
                public void onPaymentFailed(PurchaseId purchaseId, InvoiceId invoiceId) {
                    emitPaymentEvent("failed", paymentId, orderId, productId, purchaseId, invoiceId, "Payment failed");
                }

                @Override
                public void onPurchaseCancelled(PurchaseId purchaseId, InvoiceId invoiceId) {
                    emitPaymentEvent("cancelled", paymentId, orderId, productId, purchaseId, invoiceId, "");
                }
            };

            client.getPurchaseInteractor()
                    .purchase(params, PreferredPurchaseType.ONE_STEP, SdkTheme.DARK, listener)
                    .addOnSuccessListener(result -> emitPurchaseResult("completed", paymentId, result, ""))
                    .addOnFailureListener(error -> emitPurchaseError(paymentId, orderId, productId, error));
        } catch (Throwable error) {
            emitPaymentEvent("failed", paymentId, payload.optString("order_id", paymentId), productId, null, null, error.getMessage());
        }
    }

    private void emitPurchaseResult(String type, String paymentId, ProductPurchaseResult result, String message) {
        if (result == null) {
            emitPaymentEvent(type, paymentId, paymentId, "", null, null, message);
            return;
        }
        emitPaymentEvent(
                type,
                paymentId,
                stringValue(result.getOrderId()),
                stringValue(result.getProductId()),
                result.getPurchaseId(),
                result.getInvoiceId(),
                message
        );
    }

    private void emitPurchaseError(String paymentId, String orderId, String productId, Throwable error) {
        String message = error == null || error.getMessage() == null ? "RuStore Pay error" : error.getMessage();
        if (error instanceof RuStorePaymentException.ProductPurchaseCancelled) {
            RuStorePaymentException.ProductPurchaseCancelled canceled = (RuStorePaymentException.ProductPurchaseCancelled) error;
            emitPaymentEvent("cancelled", paymentId, orderId, productId, canceled.getPurchaseId(), null, message);
            return;
        }
        if (error instanceof RuStorePaymentException.ProductPurchaseException) {
            RuStorePaymentException.ProductPurchaseException failed = (RuStorePaymentException.ProductPurchaseException) error;
            emitPaymentEvent(
                    "failed",
                    paymentId,
                    stringValue(failed.getOrderId()).isEmpty() ? orderId : stringValue(failed.getOrderId()),
                    stringValue(failed.getProductId()).isEmpty() ? productId : stringValue(failed.getProductId()),
                    failed.getPurchaseId(),
                    failed.getInvoiceId(),
                    message
            );
            return;
        }
        emitPaymentEvent("failed", paymentId, orderId, productId, null, null, message);
    }

    private void emitPaymentEvent(
            String type,
            String paymentId,
            String orderId,
            String productId,
            PurchaseId purchaseId,
            InvoiceId invoiceId,
            String message
    ) {
        try {
            JSONObject event = new JSONObject()
                    .put("type", type)
                    .put("payment_id", paymentId)
                    .put("order_id", orderId)
                    .put("product_id", productId)
                    .put("purchase_id", stringValue(purchaseId))
                    .put("invoice_id", stringValue(invoiceId))
                    .put("message", message == null ? "" : message);
            activity.emitRuStorePaymentEvent(event);
        } catch (Exception error) {
            Log.w(TAG, "Failed to emit RuStore payment event", error);
        }
    }

    private void proceedPaymentIntent(Intent intent) {
        if (intent == null) {
            return;
        }
        try {
            RuStorePayClient client = ensurePayClient();
            if (client != null) {
                client.getIntentInteractor().proceedIntent(intent, SdkTheme.DARK);
            }
        } catch (Throwable error) {
            Log.d(TAG, "RuStore Pay intent ignored", error);
        }
    }

    private boolean isUpdateAvailable(AppUpdateInfo info, int updateType) {
        if (info == null) {
            return false;
        }
        int availability = info.getUpdateAvailability();
        return (availability == UpdateAvailability.UPDATE_AVAILABLE
                || availability == UpdateAvailability.DEVELOPER_TRIGGERED_UPDATE_IN_PROGRESS)
                && info.isUpdateTypeAllowed(updateType);
    }

    private void startImmediateFlow(
            RuStoreAppUpdateManager manager,
            AppUpdateInfo info
    ) {
        manager.startUpdateFlow(info, updateOptions(AppUpdateType.IMMEDIATE))
                .addOnSuccessListener(result -> {
                    immediateFlowActive = false;
                    if (result != null && result == Activity.RESULT_OK) {
                        // RESULT_OK means RuStore accepted control of the
                        // immediate install. Keep the required fallback attached
                        // so a later FAILED/INTERRUPTED state can surface retry UI.
                        immediateFlowAccepted = true;
                        return;
                    }
                    // Includes Activity.RESULT_CANCELED and malformed results.
                    failRequiredUpdateFlow();
                })
                .addOnFailureListener(error -> failRequiredUpdateFlow());
    }

    private void failRequiredUpdateFlow() {
        immediateFlowActive = false;
        immediateFlowAccepted = false;
        Runnable fallback = requiredUpdateFallback;
        requiredUpdateFallback = null;
        runOnUi(fallback);
    }

    private AppUpdateOptions updateOptions(int updateType) {
        return new AppUpdateOptions.Builder()
                .appUpdateType(updateType)
                .build();
    }

    private boolean isRustoreConfigured() {
        String appId = BuildConfig.RUSTORE_CONSOLE_APP_ID == null ? "" : BuildConfig.RUSTORE_CONSOLE_APP_ID.trim();
        return "rustore".equals(BuildConfig.DISTRIBUTION_CHANNEL) && !appId.isEmpty() && !"0".equals(appId);
    }

    private void runOnUi(Runnable runnable) {
        if (runnable != null) {
            activity.runOnUiThread(runnable);
        }
    }

    private String stringValue(Object value) {
        if (value == null) {
            return "";
        }
        try {
            return String.valueOf(value.getClass().getMethod("getValue").invoke(value));
        } catch (Exception ignored) {
            return String.valueOf(value);
        }
    }

    private String jsonResult(boolean accepted, String error, String message) {
        try {
            JSONObject result = new JSONObject()
                    .put("accepted", accepted);
            if (error != null && !error.isEmpty()) {
                result.put("error", error);
            }
            if (message != null && !message.isEmpty()) {
                result.put("message", message);
            }
            return result.toString();
        } catch (Exception ignored) {
            return accepted ? "{\"accepted\":true}" : "{\"accepted\":false}";
        }
    }
}
