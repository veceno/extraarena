package ru.extraarena.app;

import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

public class SquadPersonalCbrpWidgetProvider extends AppWidgetProvider {
    @Override
    public void onReceive(Context context, Intent intent) {
        super.onReceive(context, intent);
        if (!SquadWidgetUpdater.ACTION_REFRESH_PERSONAL.equals(intent == null ? null : intent.getAction())) {
            return;
        }
        int appWidgetId = intent.getIntExtra(SquadWidgetUpdater.EXTRA_WIDGET_ID, AppWidgetManager.INVALID_APPWIDGET_ID);
        if (appWidgetId != AppWidgetManager.INVALID_APPWIDGET_ID) {
            SquadWidgetUpdater.refreshPersonalCbrp(context, AppWidgetManager.getInstance(context), appWidgetId);
        }
    }

    @Override
    public void onUpdate(Context context, AppWidgetManager appWidgetManager, int[] appWidgetIds) {
        for (int appWidgetId : appWidgetIds) {
            SquadWidgetUpdater.refreshPersonalCbrp(context, appWidgetManager, appWidgetId);
        }
    }

    @Override
    public void onAppWidgetOptionsChanged(
            Context context,
            AppWidgetManager appWidgetManager,
            int appWidgetId,
            Bundle newOptions
    ) {
        SquadWidgetUpdater.refreshPersonalCbrp(context, appWidgetManager, appWidgetId);
    }
}
