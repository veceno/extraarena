package ru.extraarena.app;

import android.content.Context;
import android.telephony.TelephonyManager;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.TimeZone;

/**
 * Best-effort, permission-free detection of whether the device is likely in Russia, used to
 * auto-select the RU-direct connection profile (app.laveqox.ru) vs the worldwide Cloudflare-tunnel
 * profile (app.extraarena.space).
 *
 * Signals, strongest first:
 *  - TelephonyManager.getSimCountryIso() / getNetworkCountryIso() == "ru" (no permission required;
 *    return "" when there is no SIM / no network).
 *  - Locale.getDefault().getCountry() == "RU" (device locale).
 *  - TimeZone.getDefault().getID() is a Russian timezone (weak signal; used only on devices with no
 *    SIM and a non-RU locale, e.g. emulators).
 */
final class RegionDetector {
    private RegionDetector() {
    }

    static final class Result {
        final boolean ru;
        final String source;

        Result(boolean ru, String source) {
            this.ru = ru;
            this.source = source;
        }
    }

    private static final Set<String> RU_TIMEZONES = new HashSet<>(Arrays.asList(
            "Europe/Kaliningrad", "Europe/Moscow", "Europe/Simferopol", "Europe/Kirov",
            "Europe/Volgograd", "Europe/Saratov", "Europe/Astrakhan", "Europe/Ulyanovsk",
            "Europe/Samara", "Asia/Yekaterinburg", "Asia/Omsk", "Asia/Novosibirsk",
            "Asia/Barnaul", "Asia/Tomsk", "Asia/Novokuznetsk", "Asia/Krasnoyarsk",
            "Asia/Irkutsk", "Asia/Chita", "Asia/Yakutsk", "Asia/Khandyga",
            "Asia/Vladivostok", "Asia/Ust-Nera", "Asia/Magadan", "Asia/Sakhalin",
            "Asia/Srednekolymsk", "Asia/Kamchatka", "Asia/Anadyr"));

    static Result isLikelyRu(Context context) {
        try {
            TelephonyManager tm = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (tm != null) {
                String sim = tm.getSimCountryIso();
                if (sim != null && sim.equalsIgnoreCase("ru")) {
                    return new Result(true, "sim:" + sim);
                }
                String net = tm.getNetworkCountryIso();
                if (net != null && net.equalsIgnoreCase("ru")) {
                    return new Result(true, "net:" + net);
                }
            }
        } catch (Exception ignored) {
            // TelephonyManager unavailable (e.g. tablet / emulator without telephony) — fall through.
        }
        String localeCountry = Locale.getDefault().getCountry();
        if (localeCountry != null && localeCountry.equalsIgnoreCase("RU")) {
            return new Result(true, "locale:" + localeCountry);
        }
        String tz = TimeZone.getDefault().getID();
        if (tz != null && RU_TIMEZONES.contains(tz)) {
            return new Result(true, "tz:" + tz);
        }
        return new Result(false, "none");
    }
}