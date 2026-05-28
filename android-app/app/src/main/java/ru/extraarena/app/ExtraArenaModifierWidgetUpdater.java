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

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class ExtraArenaModifierWidgetUpdater {
    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();
    private static final Map<String, String> MODE_DESCRIPTIONS = createModeDescriptions();

    private ExtraArenaModifierWidgetUpdater() {
    }

    static void refresh(Context context, AppWidgetManager appWidgetManager, int appWidgetId) {
        Context appContext = context.getApplicationContext();
        render(appContext, appWidgetManager, appWidgetId, "Загрузка...", "Обновляем цикл", "", "Связываемся с ареной");
        EXECUTOR.execute(() -> {
            try {
                JSONObject payload = fetchPayload(appContext);
                if (!payload.optBoolean("enabled", false)) {
                    render(appContext, appWidgetManager, appWidgetId,
                            "ExtraArena",
                            "Режим недоступен",
                            "Особый режим временно выключен.",
                            "Нажми, чтобы открыть игру");
                    return;
                }

                String modeId = payload.optString("mode_id", "");
                String label = cleanLabel(payload.optString("label", "ExtraArena"));
                int seconds = payload.optInt("seconds_to_rotation", -1);
                scheduleNextRefresh(appContext, appWidgetId, payload.optLong("next_rotation_at", 0L));
                render(appContext, appWidgetManager, appWidgetId,
                        label,
                        formatRemaining(seconds),
                        MODE_DESCRIPTIONS.getOrDefault(modeId, "Особый режим с уникальными правилами."),
                        "Тап по виджету откроет ExtraArena");
            } catch (Exception ignored) {
                render(appContext, appWidgetManager, appWidgetId,
                        "ExtraArena",
                        "Нет связи",
                        "Не удалось обновить текущий модификатор.",
                        "Нажми ↻ для повторной попытки");
            }
        });
    }

    private static JSONObject fetchPayload(Context context) throws Exception {
        URL url = new URL(BaseUrlStore.join(BaseUrlStore.getBaseUrl(context), BuildConfig.EXTRA_ARENA_WIDGET_PATH));
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(6000);
        connection.setReadTimeout(6000);
        connection.setRequestProperty("Accept", "application/json");
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
            return new JSONObject(body);
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
            String label,
            String timer,
            String description,
            String status
    ) {
        RemoteViews views = new RemoteViews(context.getPackageName(), R.layout.widget_extra_arena_modifier);
        views.setTextViewText(R.id.widget_mode_label, label);
        views.setTextViewText(R.id.widget_timer, timer);
        views.setTextViewText(R.id.widget_description, description);
        views.setTextViewText(R.id.widget_status, status);
        views.setViewVisibility(
                R.id.widget_description,
                shouldShowDescription(appWidgetManager, appWidgetId) && !description.trim().isEmpty()
                        ? View.VISIBLE
                        : View.GONE
        );
        views.setOnClickPendingIntent(R.id.widget_root, openAppIntent(context));
        views.setOnClickPendingIntent(R.id.widget_refresh_button, refreshIntent(context, appWidgetId));
        appWidgetManager.updateAppWidget(appWidgetId, views);
    }

    private static PendingIntent openAppIntent(Context context) {
        Intent intent = new Intent(context, MainActivity.class);
        intent.setAction(Intent.ACTION_VIEW);
        intent.putExtra("section", "battle");
        return PendingIntent.getActivity(
                context,
                8100,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | immutableFlag()
        );
    }

    private static PendingIntent refreshIntent(Context context, int appWidgetId) {
        Intent intent = new Intent(context, ExtraArenaModifierWidgetProvider.class);
        intent.setAction(ExtraArenaModifierWidgetProvider.ACTION_REFRESH);
        intent.putExtra(ExtraArenaModifierWidgetProvider.EXTRA_WIDGET_ID, appWidgetId);
        return PendingIntent.getBroadcast(
                context,
                8200 + appWidgetId,
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
        alarmManager.set(
                AlarmManager.RTC,
                triggerAtMillis,
                refreshIntent(context, appWidgetId)
        );
    }

    private static int immutableFlag() {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0;
    }

    private static boolean shouldShowDescription(AppWidgetManager appWidgetManager, int appWidgetId) {
        Bundle options = appWidgetManager.getAppWidgetOptions(appWidgetId);
        int minWidth = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0);
        int minHeight = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_HEIGHT, 0);
        return minWidth >= 230 || minHeight >= 150;
    }

    private static String cleanLabel(String label) {
        String value = label == null ? "" : label.trim();
        if (value.startsWith("ExtraArena ")) {
            value = value.substring("ExtraArena ".length());
        }
        return value.isEmpty() ? "ExtraArena" : value;
    }

    private static String formatRemaining(int seconds) {
        if (seconds < 0) {
            return "Смена скоро";
        }
        int hours = seconds / 3600;
        int minutes = Math.max(0, (seconds % 3600) / 60);
        if (hours > 0) {
            return String.format(Locale.US, "Смена через %dч %02dм", hours, minutes);
        }
        if (minutes > 0) {
            return String.format(Locale.US, "Смена через %dм", minutes);
        }
        return "Смена меньше чем через минуту";
    }

    private static Map<String, String> createModeDescriptions() {
        Map<String, String> values = new HashMap<>();
        values.put("extra_arena:powermax", "Все карты выходят на максимум. Здесь решает не прокачка, а темп и расчет.");
        values.put("extra_arena:blitzkrieg", "Ходы быстрее, существа атакуют сразу. Ошибаться почти некогда.");
        values.put("extra_arena:spellstorm", "Заклинания бесплатные. Комбо становятся резче, а стол меняется каждую секунду.");
        values.put("extra_arena:sudden_death", "Герои теряют здоровье каждый ход. Затягивать партию опасно.");
        return values;
    }
}
