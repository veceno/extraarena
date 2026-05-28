package ru.extraarena.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.os.Build;

final class NotificationChannels {
    static final String GAME = "extraarena_game";
    static final String UPDATES = "extraarena_updates";

    private NotificationChannels() {
    }

    static void ensure(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return;
        }

        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null) {
            return;
        }

        NotificationChannel game = new NotificationChannel(
                GAME,
                "ExtraArena",
                NotificationManager.IMPORTANCE_DEFAULT
        );
        game.setDescription("Игровые события ExtraArena");

        NotificationChannel updates = new NotificationChannel(
                UPDATES,
                "Обновления",
                NotificationManager.IMPORTANCE_HIGH
        );
        updates.setDescription("Важные обновления приложения");

        manager.createNotificationChannel(game);
        manager.createNotificationChannel(updates);
    }
}
