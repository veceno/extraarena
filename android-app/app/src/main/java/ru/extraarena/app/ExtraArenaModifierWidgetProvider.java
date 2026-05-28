package ru.extraarena.app;

import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

public class ExtraArenaModifierWidgetProvider extends AppWidgetProvider {
    static final String ACTION_REFRESH = "ru.extraarena.app.EXTRA_ARENA_WIDGET_REFRESH";
    static final String EXTRA_WIDGET_ID = "extraarena_widget_id";

    @Override
    public void onUpdate(Context context, AppWidgetManager appWidgetManager, int[] appWidgetIds) {
        for (int appWidgetId : appWidgetIds) {
            ExtraArenaModifierWidgetUpdater.refresh(context, appWidgetManager, appWidgetId);
        }
    }

    @Override
    public void onAppWidgetOptionsChanged(
            Context context,
            AppWidgetManager appWidgetManager,
            int appWidgetId,
            Bundle newOptions
    ) {
        ExtraArenaModifierWidgetUpdater.refresh(context, appWidgetManager, appWidgetId);
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        super.onReceive(context, intent);
        if (intent == null || !ACTION_REFRESH.equals(intent.getAction())) {
            return;
        }

        AppWidgetManager manager = AppWidgetManager.getInstance(context);
        int requestedId = intent.getIntExtra(EXTRA_WIDGET_ID, AppWidgetManager.INVALID_APPWIDGET_ID);
        if (requestedId != AppWidgetManager.INVALID_APPWIDGET_ID) {
            ExtraArenaModifierWidgetUpdater.refresh(context, manager, requestedId);
            return;
        }

        ComponentName componentName = new ComponentName(context, ExtraArenaModifierWidgetProvider.class);
        int[] appWidgetIds = manager.getAppWidgetIds(componentName);
        for (int appWidgetId : appWidgetIds) {
            ExtraArenaModifierWidgetUpdater.refresh(context, manager, appWidgetId);
        }
    }
}
