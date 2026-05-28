package ru.extraarena.app;

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

final class SquadWidgetUpdater {
    static final String ACTION_REFRESH_PERSONAL = BuildConfig.APPLICATION_ID + ".widget.SQUAD_PERSONAL_REFRESH";
    static final String EXTRA_WIDGET_ID = "extra_widget_id";

    private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor();

    private SquadWidgetUpdater() {
    }

    static void refreshPersonalCbrp(Context context, AppWidgetManager appWidgetManager, int appWidgetId) {
        refresh(context, appWidgetManager, appWidgetId, WidgetKind.PERSONAL_CBRP);
    }

    private static void refresh(
            Context context,
            AppWidgetManager appWidgetManager,
            int appWidgetId,
            WidgetKind kind
    ) {
        Context appContext = context.getApplicationContext();
        render(appContext, appWidgetManager, appWidgetId, kind, "Загрузка...", "Сквад", "", "Обновляем данные", new String[0]);
        EXECUTOR.execute(() -> {
            String authToken = DeviceRegistrar.getAuthToken(appContext);
            if (authToken == null || authToken.trim().isEmpty()) {
                render(appContext, appWidgetManager, appWidgetId, kind,
                        "Войдите в ExtraID",
                        "Сквад",
                        "",
                        "Откройте приложение",
                        new String[0]);
                return;
            }
            try {
                JSONObject payload = fetchPayload(appContext, kind.path, authToken);
                renderPayload(appContext, appWidgetManager, appWidgetId, kind, payload);
            } catch (Exception ignored) {
                render(appContext, appWidgetManager, appWidgetId, kind,
                        "Нет связи",
                        "Сквад",
                        "",
                        "Не удалось обновить виджет",
                        new String[0]);
            }
        });
    }

    private static JSONObject fetchPayload(Context context, String path, String authToken) throws Exception {
        URL url = new URL(BaseUrlStore.join(BaseUrlStore.getBaseUrl(context), path));
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
                if (body == null || body.trim().isEmpty()) {
                    throw new IllegalStateException("widget endpoint returned " + status);
                }
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

    private static void renderPayload(
            Context context,
            AppWidgetManager manager,
            int appWidgetId,
            WidgetKind kind,
            JSONObject payload
    ) {
        if (!payload.optBoolean("in_squad", true)) {
            render(context, manager, appWidgetId, kind, "Вы не в скваде", kind.kicker, "", "Откройте поиск сквадов", new String[0]);
            return;
        }
        if ("owner_required".equals(payload.optString("error", ""))) {
            render(context, manager, appWidgetId, kind, "Только владельцу", kind.kicker, "", "Этот виджет для создателя сквада", new String[0]);
            return;
        }

        JSONObject squad = payload.optJSONObject("squad");
        String squadName = squad == null ? "Сквад" : squad.optString("name", "Сквад");
        if (kind == WidgetKind.PERSONAL_CBRP) {
            render(context, manager, appWidgetId, kind,
                    squadName,
                    formatNumber(payload.optLong("personal_cbrp", 0)) + " CBRP",
                    signed(payload.optLong("delta_24h", 0)) + " за 24ч",
                    "Личный вклад",
                    eventLines(payload.optJSONArray("events"), false));
            return;
        }

        render(context, manager, appWidgetId, kind,
                squadName,
                formatNumber(payload.optLong("personal_cbrp", 0)) + " CBRP",
                signed(payload.optLong("delta_24h", 0)) + " за 24ч",
                "Личный вклад",
                eventLines(payload.optJSONArray("events"), false));
    }

    private static void render(
            Context context,
            AppWidgetManager manager,
            int appWidgetId,
            WidgetKind kind,
            String title,
            String primary,
            String secondary,
            String status,
            String[] details
    ) {
        RemoteViews views = new RemoteViews(context.getPackageName(), R.layout.widget_squad_summary);
        views.setTextViewText(R.id.widget_squad_kicker, kind.kicker);
        views.setTextViewText(R.id.widget_squad_title, title);
        views.setTextViewText(R.id.widget_squad_primary, primary);
        views.setTextViewText(R.id.widget_squad_secondary, secondary);
        views.setTextViewText(R.id.widget_squad_status, status);

        boolean showDetails = shouldShowDetails(manager, appWidgetId);
        setDetail(views, R.id.widget_squad_detail_1, showDetails, details, 0);
        setDetail(views, R.id.widget_squad_detail_2, showDetails, details, 1);
        setDetail(views, R.id.widget_squad_detail_3, showDetails, details, 2);
        views.setViewVisibility(R.id.widget_squad_secondary, secondary == null || secondary.trim().isEmpty() ? View.GONE : View.VISIBLE);
        views.setOnClickPendingIntent(R.id.widget_squad_root, openAppIntent(context));
        views.setOnClickPendingIntent(R.id.widget_squad_refresh_button, refreshIntent(context, appWidgetId, kind));
        manager.updateAppWidget(appWidgetId, views);
    }

    private static void setDetail(RemoteViews views, int viewId, boolean showDetails, String[] details, int index) {
        String value = details != null && index < details.length ? details[index] : "";
        boolean visible = showDetails && value != null && !value.trim().isEmpty();
        views.setViewVisibility(viewId, visible ? View.VISIBLE : View.GONE);
        views.setTextViewText(viewId, visible ? value : "");
    }

    private static boolean shouldShowDetails(AppWidgetManager manager, int appWidgetId) {
        Bundle options = manager.getAppWidgetOptions(appWidgetId);
        int minWidth = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0);
        int minHeight = options.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_HEIGHT, 0);
        return minWidth >= 230 || minHeight >= 150;
    }

    private static PendingIntent openAppIntent(Context context) {
        Intent intent = new Intent(context, MainActivity.class);
        intent.setAction(Intent.ACTION_VIEW);
        intent.putExtra("section", "squads");
        return PendingIntent.getActivity(
                context,
                8300,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | immutableFlag()
        );
    }

    private static PendingIntent refreshIntent(Context context, int appWidgetId, WidgetKind kind) {
        Intent intent = new Intent(context, kind.providerClass);
        intent.setAction(kind.refreshAction);
        intent.putExtra(EXTRA_WIDGET_ID, appWidgetId);
        return PendingIntent.getBroadcast(
                context,
                kind.requestCodeBase + appWidgetId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | immutableFlag()
        );
    }

    private static int immutableFlag() {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0;
    }

    private static String[] eventLines(JSONArray events, boolean includeNick) {
        return linesFromArray(events, (item) -> {
            String label = eventLabel(item.optString("event_type", ""));
            long cbrp = item.optLong("cbrp", 0);
            String nick = includeNick ? item.optString("nick", "") + ": " : "";
            return nick + label + " · +" + cbrp;
        });
    }

    private static String[] linesFromArray(JSONArray array, LineFormatter formatter) {
        if (array == null || array.length() == 0) {
            return new String[0];
        }
        int size = Math.min(3, array.length());
        String[] lines = new String[size];
        for (int i = 0; i < size; i++) {
            lines[i] = formatter.format(array.optJSONObject(i) == null ? new JSONObject() : array.optJSONObject(i));
        }
        return lines;
    }

    private static String eventLabel(String eventType) {
        if ("battle_win".equals(eventType)) return "Победа";
        if ("battle_loss".equals(eventType)) return "Бой";
        if ("case_open".equals(eventType)) return "Кейс";
        if ("card_upgrade".equals(eventType)) return "Улучшение";
        if ("weekly_trophy_delta".equals(eventType)) return "Неделя";
        if ("new_card".equals(eventType)) return "Новая карта";
        return "Вклад";
    }

    private static String signed(long value) {
        return (value >= 0 ? "+" : "") + formatNumber(value);
    }

    private static String formatNumber(long value) {
        return String.format(Locale.US, "%,d", value).replace(',', ' ');
    }

    private enum WidgetKind {
        PERSONAL_CBRP(
                "МОЙ CBRP",
                BuildConfig.SQUAD_PERSONAL_WIDGET_PATH,
                ACTION_REFRESH_PERSONAL,
                SquadPersonalCbrpWidgetProvider.class,
                8400
        );

        final String kicker;
        final String path;
        final String refreshAction;
        final Class<?> providerClass;
        final int requestCodeBase;

        WidgetKind(String kicker, String path, String refreshAction, Class<?> providerClass, int requestCodeBase) {
            this.kicker = kicker;
            this.path = path;
            this.refreshAction = refreshAction;
            this.providerClass = providerClass;
            this.requestCodeBase = requestCodeBase;
        }
    }

    private interface LineFormatter {
        String format(JSONObject item);
    }
}
