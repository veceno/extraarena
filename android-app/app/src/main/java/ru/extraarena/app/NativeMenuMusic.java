package ru.extraarena.app;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.media.AudioAttributes;
import android.media.AudioFocusRequest;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.util.Log;

import java.util.Locale;

final class NativeMenuMusic {
    private static final String TAG = "EAMenuMusic";
    private static final String KEY_ENABLED = "native_menu_music_enabled_v1";
    private static final float NORMAL_VOLUME = 0.34f;
    private static final float DUCK_VOLUME = 0.10f;

    private final Context context;
    private final AudioManager audioManager;
    private final AudioFocusRequest focusRequest;
    private MediaPlayer player;
    private boolean prepared;
    private boolean preparing;
    private boolean lifecyclePaused;
    private boolean hasAudioFocus;
    private String scene = "menu";

    NativeMenuMusic(Context context) {
        this.context = context.getApplicationContext();
        this.audioManager = (AudioManager) this.context.getSystemService(Context.AUDIO_SERVICE);
        AudioAttributes attributes = new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_GAME)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                .build();
        this.focusRequest = new AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
                .setAudioAttributes(attributes)
                .setAcceptsDelayedFocusGain(true)
                .setOnAudioFocusChangeListener(this::onAudioFocusChange)
                .build();
    }

    boolean isEnabled() {
        return !"false".equals(SecurePrefs.getString(context, KEY_ENABLED, "true"));
    }

    void setEnabled(boolean enabled) {
        SecurePrefs.putString(context, KEY_ENABLED, enabled ? "true" : "false");
        if (enabled) {
            startIfNeeded();
        } else {
            pausePlayer();
            abandonAudioFocus();
        }
    }

    void setScene(String requestedScene) {
        String clean = requestedScene == null ? "" : requestedScene.trim().toLowerCase(Locale.ROOT);
        scene = "menu".equals(clean) ? "menu" : "silent";
        if ("menu".equals(scene)) {
            startIfNeeded();
        } else {
            pausePlayer();
            abandonAudioFocus();
        }
    }

    void onResume() {
        lifecyclePaused = false;
        startIfNeeded();
    }

    void onPause() {
        lifecyclePaused = true;
        pausePlayer();
        abandonAudioFocus();
    }

    void release() {
        lifecyclePaused = true;
        abandonAudioFocus();
        MediaPlayer current = player;
        player = null;
        prepared = false;
        preparing = false;
        if (current != null) {
            try {
                current.stop();
            } catch (Exception ignored) {
            }
            try {
                current.release();
            } catch (Exception ignored) {
            }
        }
    }

    private void startIfNeeded() {
        if (lifecyclePaused || !"menu".equals(scene) || !isEnabled()) {
            return;
        }
        if (player == null) {
            preparePlayer();
            return;
        }
        if (!prepared || preparing || player.isPlaying()) {
            return;
        }
        if (!requestAudioFocus()) {
            return;
        }
        try {
            player.setVolume(NORMAL_VOLUME, NORMAL_VOLUME);
            player.start();
        } catch (Exception error) {
            Log.w(TAG, "Unable to resume menu music", error);
        }
    }

    private void preparePlayer() {
        if (preparing || player != null) {
            return;
        }
        preparing = true;
        AssetFileDescriptor descriptor = null;
        try {
            descriptor = context.getAssets().openFd("DesignAssets/Sounds/main_theme.mp3");
            MediaPlayer created = new MediaPlayer();
            created.setAudioAttributes(new AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_GAME)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build());
            created.setDataSource(
                    descriptor.getFileDescriptor(),
                    descriptor.getStartOffset(),
                    descriptor.getLength()
            );
            created.setLooping(true);
            created.setVolume(NORMAL_VOLUME, NORMAL_VOLUME);
            created.setOnPreparedListener(mediaPlayer -> {
                preparing = false;
                prepared = true;
                startIfNeeded();
            });
            created.setOnErrorListener((mediaPlayer, what, extra) -> {
                Log.w(TAG, "Menu music playback error what=" + what + " extra=" + extra);
                preparing = false;
                prepared = false;
                if (player == mediaPlayer) {
                    player = null;
                }
                try {
                    mediaPlayer.release();
                } catch (Exception ignored) {
                }
                abandonAudioFocus();
                return true;
            });
            player = created;
            created.prepareAsync();
        } catch (Exception error) {
            Log.w(TAG, "Unable to prepare bundled menu music", error);
            preparing = false;
            prepared = false;
            if (player != null) {
                try {
                    player.release();
                } catch (Exception ignored) {
                }
                player = null;
            }
        } finally {
            if (descriptor != null) {
                try {
                    descriptor.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    private boolean requestAudioFocus() {
        if (audioManager == null) {
            return true;
        }
        if (hasAudioFocus) {
            return true;
        }
        int result = audioManager.requestAudioFocus(focusRequest);
        hasAudioFocus = result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED;
        return hasAudioFocus;
    }

    private void abandonAudioFocus() {
        if (audioManager == null || !hasAudioFocus) {
            return;
        }
        try {
            audioManager.abandonAudioFocusRequest(focusRequest);
        } catch (Exception ignored) {
        }
        hasAudioFocus = false;
    }

    private void onAudioFocusChange(int change) {
        if (change == AudioManager.AUDIOFOCUS_GAIN) {
            hasAudioFocus = true;
            if (player != null) {
                try {
                    player.setVolume(NORMAL_VOLUME, NORMAL_VOLUME);
                } catch (Exception ignored) {
                }
            }
            startIfNeeded();
            return;
        }
        if (change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK) {
            if (player != null) {
                try {
                    player.setVolume(DUCK_VOLUME, DUCK_VOLUME);
                } catch (Exception ignored) {
                }
            }
            return;
        }
        pausePlayer();
        if (change == AudioManager.AUDIOFOCUS_LOSS) {
            hasAudioFocus = false;
        }
    }

    private void pausePlayer() {
        if (player == null || !prepared) {
            return;
        }
        try {
            if (player.isPlaying()) {
                player.pause();
            }
        } catch (Exception ignored) {
        }
    }
}
