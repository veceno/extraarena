package ru.extraarena.app;

import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

public class ShopParticlesWidgetProvider extends AppWidgetProvider {
    static final String ACTION_REFRESH = BuildConfig.APPLICATION_ID + ".widget.SHOP_PARTICLES_REFRESH";
    static final String EXTRA_WIDGET_ID = "shop_particles_widget_id";

    @Override
    public void onUpdate(Context context, AppWidgetManager appWidgetManager, int[] appWidgetIds) {
        for (int appWidgetId : appWidgetIds) {
            ShopParticlesWidgetUpdater.refresh(context, appWidgetManager, appWidgetId);
        }
    }

    @Override
    public void onAppWidgetOptionsChanged(
            Context context,
            AppWidgetManager appWidgetManager,
            int appWidgetId,
            Bundle newOptions
    ) {
        ShopParticlesWidgetUpdater.refresh(context, appWidgetManager, appWidgetId);
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
            ShopParticlesWidgetUpdater.refresh(context, manager, requestedId);
            return;
        }

        ComponentName componentName = new ComponentName(context, ShopParticlesWidgetProvider.class);
        int[] appWidgetIds = manager.getAppWidgetIds(componentName);
        for (int appWidgetId : appWidgetIds) {
            ShopParticlesWidgetUpdater.refresh(context, manager, appWidgetId);
        }
    }
}
