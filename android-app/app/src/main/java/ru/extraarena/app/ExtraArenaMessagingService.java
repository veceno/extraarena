package ru.extraarena.app;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.util.Map;

public class ExtraArenaMessagingService extends FirebaseMessagingService {
    private static final int UPDATE_NOTIFICATION_ID = 7101;

    @Override
    public void onNewToken(String token) {
        super.onNewToken(token);
        DeviceRegistrar.saveFcmToken(this, token);
    }

    @Override
    public void onMessageReceived(RemoteMessage message) {
        super.onMessageReceived(message);
        NotificationChannels.ensure(this);

        Map<String, String> data = message.getData();
        String type = data.get("type");
        if ("app_update_required".equals(type) || "app_update".equals(type)) {
            showUpdateNotification(data);
            return;
        }

        RemoteMessage.Notification notification = message.getNotification();
        String title = data.containsKey("title")
                ? data.get("title")
                : notification != null ? notification.getTitle() : "ExtraArena";
        String body = data.containsKey("body")
                ? data.get("body")
                : notification != null ? notification.getBody() : "🔔 На арене новое событие";
        showGameNotification(title, body, data);
    }

    private void showUpdateNotification(Map<String, String> data) {
        String title = data.containsKey("title") ? data.get("title") : "⬇️ Хорошие новости!";
        String body = data.containsKey("body")
                ? data.get("body")
                : "⬇️ Вышло обновление, скачай новую версию, чтобы продолжить игру";
        String url = sanitizeUpdateUrl(data.get("url"));

        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        intent.addCategory(Intent.CATEGORY_BROWSABLE);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this,
                100,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        Notification.Builder builder = new Notification.Builder(this)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(new Notification.BigTextStyle().bigText(body))
                .setContentIntent(pendingIntent)
                .setAutoCancel(true);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            builder.setChannelId(NotificationChannels.UPDATES);
        } else {
            builder.setPriority(Notification.PRIORITY_HIGH);
        }

        notify(UPDATE_NOTIFICATION_ID, builder.build());
    }

    private void showGameNotification(String title, String body, Map<String, String> data) {
        String section = data == null ? null : data.get("section");
        String inviteId = data == null ? null : data.get("invite_id");
        String inviteAction = data == null ? null : data.get("invite_action");
        String decisionId = data == null ? null : data.get("rc_decision_id");
        if ((decisionId == null || decisionId.trim().isEmpty()) && data != null) {
            decisionId = data.get("decision_id");
        }
        String outboxNotificationId = data == null ? null : data.get("notification_id");
        String deliveryId = data == null ? null : data.get("delivery_id");
        String entrypoint = data == null ? null : data.get("entrypoint");
        String category = data == null ? null : data.get("category");
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        intent.setAction("ru.extraarena.app.PUSH");
        intent.setPackage(getPackageName());
        if (section != null && !section.trim().isEmpty()) {
            intent.putExtra("section", section);
        }
        if (inviteId != null && !inviteId.trim().isEmpty()) {
            intent.putExtra("invite_id", inviteId);
        }
        if (inviteAction != null && !inviteAction.trim().isEmpty()) {
            intent.putExtra("invite_action", inviteAction);
        }
        if (decisionId != null && !decisionId.trim().isEmpty()) {
            intent.putExtra("rc_decision_id", decisionId);
        }
        if (outboxNotificationId != null && !outboxNotificationId.trim().isEmpty()) {
            intent.putExtra("notification_id", outboxNotificationId);
        }
        if (deliveryId != null && !deliveryId.trim().isEmpty()) {
            intent.putExtra("delivery_id", deliveryId);
        }
        if ("notification".equals(entrypoint)) {
            intent.putExtra("entrypoint", "notification");
        }
        int notificationId = makeGameNotificationId(title, body, section, category);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this,
                notificationId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        Notification.Builder builder = new Notification.Builder(this)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle(title == null || title.isEmpty() ? "ExtraArena" : title)
                .setContentText(body == null || body.isEmpty() ? "🔔 На арене новое событие" : body)
                .setStyle(new Notification.BigTextStyle().bigText(body == null || body.isEmpty() ? "🔔 На арене новое событие" : body))
                .setContentIntent(pendingIntent)
                .setAutoCancel(true);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            builder.setChannelId(NotificationChannels.channelForCategory(category));
        } else {
            builder.setPriority(Notification.PRIORITY_DEFAULT);
        }

        notify(notificationId, builder.build());
    }

    private int makeGameNotificationId(String title, String body, String section, String category) {
        String seed = String.valueOf(title) + "|" + String.valueOf(body) + "|" + String.valueOf(section) + "|" + String.valueOf(category) + "|" + System.currentTimeMillis();
        return 7200 + Math.floorMod(seed.hashCode(), 100000);
    }

    private String sanitizeUpdateUrl(String rawUrl) {
        String url = rawUrl == null ? "" : rawUrl.trim();
        if ("rustore".equals(BuildConfig.DISTRIBUTION_CHANNEL)) {
            return BuildConfig.RUSTORE_APP_URL.equals(url) ? url : BuildConfig.RUSTORE_APP_URL;
        }
        if (BuildConfig.UPDATE_CHANNEL_URL.equals(url) || BuildConfig.UPDATE_APK_URL.equals(url)) {
            return url;
        }
        return BuildConfig.UPDATE_CHANNEL_URL;
    }

    private void notify(int id, Notification notification) {
        NotificationManager manager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (manager != null) {
            manager.notify(id, notification);
        }
    }
}
