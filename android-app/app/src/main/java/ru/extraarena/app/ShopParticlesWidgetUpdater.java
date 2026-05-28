package ru.extraarena.app;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.RemoteViews;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class ShopParticlesWidgetUpdater {
    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();

    private ShopParticlesWidgetUpdater() {
    }

    static void refresh(Context context, AppWidgetManager appWidgetManager, int appWidgetId) {
        Context appContext = context.getApplicationContext();
        render(appContext, appWidgetManager, appWidgetId,
                "Новые частицы",
                "Загрузка...",
                "Обновляем магазин",
                new String[0],
                "Связываемся с ареной");
        EXECUTOR.execute(() -> {
            String authToken = DeviceRegistrar.getAuthToken(appContext);
            if (authToken == null || authToken.trim().isEmpty()) {
                render(appContext, appWidgetManager, appWidgetId,
                        "Новые частицы",
                        "Войдите в ExtraID",
                        "Откройте приложение",
                        new String[0],
                        "Требуется игровой профиль");
                return;
            }
            try {
                JSONObject payload = fetchPayload(appContext, authToken);
                JSONObject daily = payload.optJSONObject("particles_daily");
                if (daily == null) {
                    daily = payload;
                }
                JSONArray cards = daily.optJSONArray("cards");
                long nextRotation = daily.optLong("next_rotation_ts", 0L);
                scheduleNextRefresh(appContext, appWidgetId, nextRotation);
                render(appContext, appWidgetManager, appWidgetId,
                        "Новые частицы",
                        cards == null || cards.length() == 0 ? "Ротация пуста" : firstCardTitle(cards),
                        formatRemaining(nextRotation),
                        cardLines(cards),
                        "Тап по виджету откроет магазин");
            } catch (Exception ignored) {
                render(appContext, appWidgetManager, appWidgetId,
                        "Новые частицы",
                        "Нет связи",
                        "Не удалось обновить ротацию",
                        new String[0],
                        "Нажми ↻ для повторной попытки");
            }
        });
    }

    private static JSONObject fetchPayload(Context context, String authToken) throws Exception {
        URL url = new URL(BaseUrlStore.join(BaseUrlStore.getBaseUrl(context), BuildConfig.SHOP_PARTICLES_WIDGET_PATH));
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(6000);
        connection.setReadTimeout(6000);
        connection.setRequestProperty("Accept", "application/json");
        connection.setRequestProperty("Authorization", "Bearer " + authToken);
        connection.setRequestProperty("User-Agent", "ExtraArenaApp/" + BuildConfig.VERSION_NAME + " Widget");
        String whitelistCode = ConnectionProfileStore.getWhitelistCode(context);
        if (!whitelistCode.isEmpty()) {
            connection.setRequestProperty("X-ExtraArena-Whitelist-Code", whitelistCode);
        }
        try {
            int status = connection.getResponseCode();
            InputStream stream = status >= 200 && status < 300
                    ? connection.getInputStream()
                    : connection.getErrorStream();
            String body = readAll(stream);
            if (status < 200 || status >= 300) {
                throw new IllegalStateException("widget endpoint returned " + status);
            }
            return new JSONObject(body == null || body.trim().isEmpty() ? "{}" : body);
        } finally {
            connection.disconnect();
        }
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) {
            return "{}";
        }
        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
        }
        return builder.toString();
    }

    private static void render(
            Context context,
            AppWidgetManager appWidgetManager,
            int appWidgetId,
            String title,
            String primary,
            String timer,
            String[] details,
            String status
    ) {
        RemoteViews views = new RemoteViews(context.getPackageName(), R.layout.widget_shop_particles);
        views.setTextViewText(R.id.widget_particles_title, title);
        views.setTextViewText(R.id.widget_particles_primary, primary);
        views.setTextViewText(R.id.widget_particles_timer, timer);
        views.setTextViewText(R.id.widget_particles_status, status);

        boolean showDetails = shouldShowDetails(appWidgetManager, appWidgetId);
        setDetail(views, R.id.widget_particles_card_1, true, details, 0);
        setDetail(views, R.id.widget_particles_card_2, showDetails, details, 1);
        setDetail(views, R.id.widget_particles_card_3, showDetails, details, 2);
        views.setOnClickPendingIntent(R.id.widget_particles_root, openAppIntent(context));
        views.setOnClickPendingIntent(R.id.widget_particles_refresh_button, refreshIntent(context, appWidgetId));
        appWidgetManager.updateAppWidget(appWidgetId, views);
    }

    private static void setDetail(RemoteViews views, int viewId, boolean showDetails, String[] details, int index) {
        String value = details != null && index < details.length ? details[index] : "";
        boolean visible = showDetails && value != null && !value.trim().isEmpty();
        views.setViewVisibility(viewId, visible ? View.VISIBLE : View.GONE);
        views.setTextViewText(viewId, visible ? value : "");
    }

    private static PendingIntent openAppIntent(Context context) {
        Intent intent = new Intent(context, MainActivity.class);
        intent.setAction(Intent.ACTION_VIEW);
        intent.putExtra("section", "shop");
        return PendingIntent.getActivity(
                context,
                8700,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | immutableFlag()
        );
    }

    private static PendingIntent refreshIntent(Context context, int appWidgetId) {
        Intent intent = new Intent(context, ShopParticlesWidgetProvider.class);
        intent.setAction(ShopParticlesWidgetProvider.ACTION_REFRESH);
        intent.putExtra(ShopParticlesWidgetProvider.EXTRA_WIDGET_ID, appWidgetId);
        return PendingIntent.getBroadcast(
                context,
                8800 + appWidgetId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | immutableFlag()
        );
    }

    private static void scheduleNextRefresh(Context context, int appWidgetId, long nextRotationAtSeconds) {
        if (nextRotationAtSeconds <= 0) {
            return;
        }
        AlarmManager alarmManager = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        if (alarmManager == null) {
            return;
        }
        long triggerAtMillis = Math.max(
                System.currentTimeMillis() + 60_000L,
                nextRotationAtSeconds * 1000L + 5_000L
        );
        alarmManager.set(AlarmManager.RTC, triggerAtMillis, refreshIntent(context, appWidgetId));
    }

    private static boolean shouldShowDetails(AppWidgetManager manager, int appWidgetId) {
        Bundle options = manager.getAppWidgetOptions(appWidgetId);
        int minWidth = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0);
        int minHeight = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_HEIGHT, 0);
        return minWidth >= 230 || minHeight >= 150;
    }

    private static String firstCardTitle(JSONArray cards) {
        JSONObject card = cards.optJSONObject(0);
        if (card == null) {
            return "Ротация пуста";
        }
        String name = card.optString("name", "").trim();
        return name.isEmpty() ? "Карты на частицы" : name;
    }

    private static String[] cardLines(JSONArray cards) {
        if (cards == null || cards.length() == 0) {
            return new String[0];
        }
        int size = Math.min(3, cards.length());
        String[] lines = new String[size];
        for (int i = 0; i < size; i++) {
            JSONObject card = cards.optJSONObject(i);
            if (card == null) {
                lines[i] = "";
                continue;
            }
            String name = card.optString("name", "Карта");
            String rarity = rarityLabel(card.optString("rarity", ""));
            long particles = card.optLong("particles", 0);
            long coins = card.optLong("coins", 0);
            lines[i] = name + " · " + rarity + " · " + particles + " частиц за " + coins;
        }
        return lines;
    }

    private static String rarityLabel(String rarity) {
        String value = rarity == null ? "" : rarity.trim().toLowerCase(Locale.US);
        if ("rare".equals(value)) return "редкая";
        if ("epic".equals(value)) return "эпическая";
        if ("legendary".equals(value)) return "легендарная";
        return "обычная";
    }

    private static String formatRemaining(long nextRotationAtSeconds) {
        if (nextRotationAtSeconds <= 0) {
            return "Смена скоро";
        }
        long seconds = Math.max(0L, nextRotationAtSeconds - (System.currentTimeMillis() / 1000L));
        long hours = seconds / 3600L;
        long minutes = Math.max(0L, (seconds % 3600L) / 60L);
        if (hours > 0L) {
            return String.format(Locale.US, "Смена через %dч %02dм", hours, minutes);
        }
        if (minutes > 0L) {
            return String.format(Locale.US, "Смена через %dм", minutes);
        }
        return "Смена меньше чем через минуту";
    }

    private static int immutableFlag() {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0;
    }
}
