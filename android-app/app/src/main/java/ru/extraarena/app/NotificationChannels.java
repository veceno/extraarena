package ru.extraarena.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.os.Build;

final class NotificationChannels {
    static final String GAME = "extraarena_game";
    static final String UPDATES = "extraarena_updates";
    static final String GAME_DAILY = "extraarena_daily_rewards";
    static final String GAME_MODIFIERS = "extraarena_modifiers";
    static final String GAME_REMINDERS = "extraarena_reminders";

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

        NotificationChannel daily = new NotificationChannel(
                GAME_DAILY,
                "Ежедневные награды",
                NotificationManager.IMPORTANCE_DEFAULT
        );
        daily.setDescription("Напоминания о ежедневных наградах");

        NotificationChannel modifiers = new NotificationChannel(
                GAME_MODIFIERS,
                "Модификаторы ExtraArena",
                NotificationManager.IMPORTANCE_LOW
        );
        modifiers.setDescription("Уведомления о ротации модификаторов ExtraArena");

        NotificationChannel reminders = new NotificationChannel(
                GAME_REMINDERS,
                "Напоминания",
                NotificationManager.IMPORTANCE_LOW
        );
        reminders.setDescription("Общие напоминания");

        manager.createNotificationChannel(game);
        manager.createNotificationChannel(updates);
        manager.createNotificationChannel(daily);
        manager.createNotificationChannel(modifiers);
        manager.createNotificationChannel(reminders);
    }

    static String channelForCategory(String category) {
        if (category == null) {
            return GAME;
        }
        switch (category) {
            case "daily_rewards":
                return GAME_DAILY;
            case "extra_arena_modifiers":
                return GAME_MODIFIERS;
            case "reminders":
                return GAME_REMINDERS;
            default:
                return GAME;
        }
    }
}