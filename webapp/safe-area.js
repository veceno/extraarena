(function () {
  const root = document.documentElement;
  const sides = ['top', 'right', 'bottom', 'left'];
  const KEYBOARD_DELTA_PX = 120;
  const VIEWPORT_RECOVERY_DELAY_MS = 260;
  const RESYNC_DELAYS_MS = [0, 80, 180, 360, 720];
  const FULLSCREEN_MIN_VERSION = '8.0';
  const insetFallbacks = {
    safe: {
      out: '--ea-tg-safe-',
      css: '--tg-safe-area-inset-',
      values: { top: 0, right: 0, bottom: 0, left: 0 },
    },
    content: {
      out: '--ea-tg-content-safe-',
      css: '--tg-content-safe-area-inset-',
      values: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  };
  let boundTelegram = null;
  let lastStableViewportHeight = 0;
  let fullscreenRequestAttempted = false;
  let recoveryTimer = 0;
  let resyncTimers = [];

  const toPx = (value) => {
    const number = Number(value);
    return `${Number.isFinite(number) && number > 0 ? number : 0}px`;
  };

  const toNumber = (value) => {
    const number = Number.parseFloat(String(value ?? '').trim());
    return Number.isFinite(number) && number > 0 ? number : 0;
  };

  const hasInset = (insets) => sides.some((side) => toNumber(insets && insets[side]) > 0);

  const isTelegramIos = (tg) => String(tg && tg.platform || '').toLowerCase() === 'ios';

  const readCssInset = (name) => {
    try {
      return toNumber(window.getComputedStyle(root).getPropertyValue(name));
    } catch (error) {
      return 0;
    }
  };

  const readFallbackInsets = (state) => {
    sides.forEach((side) => {
      const cssValue = readCssInset(`${state.css}${side}`);
      if (cssValue > 0) {
        state.values[side] = cssValue;
      }
    });
  };

  const normalizeInsets = (insets) => {
    if (!insets || typeof insets !== 'object') return null;
    let hasKnownSide = false;
    const normalized = {};
    sides.forEach((side) => {
      if (Object.prototype.hasOwnProperty.call(insets, side)) {
        hasKnownSide = true;
        normalized[side] = toNumber(insets[side]);
      }
    });
    return hasKnownSide ? normalized : null;
  };

  const applyInsets = (state, insets) => {
    readFallbackInsets(state);

    const normalized = normalizeInsets(insets);
    if (normalized && (hasInset(normalized) || !hasInset(state.values))) {
      sides.forEach((side) => {
        state.values[side] = normalized[side] || 0;
      });
    }

    sides.forEach((side) => {
      root.style.setProperty(`${state.out}${side}`, toPx(state.values[side]));
    });
  };

  const usableViewportHeight = () => {
    const visualHeight = Number(window.visualViewport && window.visualViewport.height);
    const innerHeight = Number(window.innerHeight);
    if (Number.isFinite(visualHeight) && visualHeight > 0) {
      return Math.round(visualHeight);
    }
    if (Number.isFinite(innerHeight) && innerHeight > 0) {
      return Math.round(innerHeight);
    }
    return 0;
  };

  const resolveTelegramViewportHeight = (tg, currentHeight) => {
    const telegramHeight = Number(tg && (tg.viewportStableHeight || tg.viewportHeight));
    const hasTelegramHeight = Number.isFinite(telegramHeight) && telegramHeight > 0;
    const hasCurrentHeight = Number.isFinite(currentHeight) && currentHeight > 0;

    if (!hasTelegramHeight) return hasCurrentHeight ? currentHeight : 0;
    if (!isTelegramIos(tg) || !hasCurrentHeight) return telegramHeight;

    // Telegram iOS can remain in Fullsize even when BotFather requests
    // Fullscreen. Never let a stale/optimistic Telegram height extend the
    // fixed app shell beyond WKWebView's actually visible viewport.
    return Math.min(telegramHeight, currentHeight);
  };

  const isKeyboardLikelyOpen = () => {
    const visualHeight = Number(window.visualViewport && window.visualViewport.height);
    const innerHeight = Number(window.innerHeight);
    const screenHeight = Number(window.screen && window.screen.height);
    const hasFocusedTextInput = !!(document.activeElement && document.activeElement.matches && document.activeElement.matches('input, textarea, select, [contenteditable="true"]'));

    if (Number.isFinite(visualHeight) && Number.isFinite(innerHeight) && innerHeight - visualHeight > KEYBOARD_DELTA_PX) {
      return true;
    }
    if (hasFocusedTextInput && Number.isFinite(screenHeight) && Number.isFinite(innerHeight) && screenHeight - innerHeight > KEYBOARD_DELTA_PX) {
      return true;
    }
    if (hasFocusedTextInput && Number.isFinite(visualHeight) && lastStableViewportHeight - visualHeight > KEYBOARD_DELTA_PX) {
      return true;
    }
    return false;
  };

  const setViewportHeight = (height) => {
    if (!Number.isFinite(height) || height <= 0) return;
    root.style.setProperty('--ea-viewport-height', `${Math.round(height)}px`);
  };

  const scheduleViewportRecovery = () => {
    window.clearTimeout(recoveryTimer);
    recoveryTimer = window.setTimeout(syncSafeArea, VIEWPORT_RECOVERY_DELAY_MS);
  };

  const scheduleResync = () => {
    resyncTimers.forEach((timer) => window.clearTimeout(timer));
    resyncTimers = RESYNC_DELAYS_MS.map((delay) => window.setTimeout(syncSafeArea, delay));
  };

  const applyStableViewportHeight = (tg, height) => {
    if (!Number.isFinite(height) || height <= 0) return false;
    lastStableViewportHeight = isTelegramIos(tg)
      ? height
      : Math.max(lastStableViewportHeight, height);
    setViewportHeight(lastStableViewportHeight);
    return true;
  };

  const syncTelegramModeClasses = (tg) => {
    const telegramIos = isTelegramIos(tg);
    root.classList.toggle('ea-telegram-ios', telegramIos);
    root.classList.toggle('ea-telegram-ios-fullsize', telegramIos && !tg.isFullscreen);
  };

  const syncSafeArea = () => {
    const tg = window.Telegram && window.Telegram.WebApp;
    const currentHeight = usableViewportHeight();
    syncTelegramModeClasses(tg);

    if (tg) {
      applyInsets(insetFallbacks.safe, tg.safeAreaInset);
      applyInsets(insetFallbacks.content, tg.contentSafeAreaInset);
    }

    if (isKeyboardLikelyOpen()) {
      if (lastStableViewportHeight > 0) {
        setViewportHeight(lastStableViewportHeight);
      }
      scheduleViewportRecovery();
      return;
    }

    if (tg) {
      const stableHeight = resolveTelegramViewportHeight(tg, currentHeight);
      if (applyStableViewportHeight(tg, stableHeight)) return;
    }

    if (!currentHeight) {
      return;
    }

    lastStableViewportHeight = currentHeight;
    setViewportHeight(currentHeight);
  };

  const requestFullscreenForIos = () => {
    const tg = window.Telegram && window.Telegram.WebApp;
    if (!isTelegramIos(tg) || tg.isFullscreen || fullscreenRequestAttempted) return false;
    if (typeof tg.requestFullscreen !== 'function') return false;
    if (typeof tg.isVersionAtLeast === 'function') {
      try {
        if (!tg.isVersionAtLeast(FULLSCREEN_MIN_VERSION)) return false;
      } catch (error) {
        return false;
      }
    }

    fullscreenRequestAttempted = true;
    try {
      tg.requestFullscreen();
      scheduleResync();
      return true;
    } catch (error) {
      fullscreenRequestAttempted = false;
      console.warn('[safe-area] iOS fullscreen request failed:', error);
      scheduleResync();
      return false;
    }
  };

  const bindTelegramEvents = () => {
    const tg = window.Telegram && window.Telegram.WebApp;
    if (!tg || typeof tg.onEvent !== 'function' || boundTelegram === tg) return;
    boundTelegram = tg;

    ['viewportChanged', 'safeAreaChanged', 'contentSafeAreaChanged', 'fullscreenChanged', 'fullscreenFailed'].forEach((eventName) => {
      try {
        tg.onEvent(eventName, scheduleResync);
      } catch (error) {
        console.warn('[safe-area] event bind failed:', eventName, error);
      }
    });
  };

  window.ExtraArenaSafeArea = {
    sync: syncSafeArea,
    syncSoon: scheduleResync,
    requestFullscreenForIos,
    isTelegramIos: () => isTelegramIos(window.Telegram && window.Telegram.WebApp),
    getInsets: () => ({
      safe: { ...insetFallbacks.safe.values },
      content: { ...insetFallbacks.content.values },
    }),
  };

  syncSafeArea();
  bindTelegramEvents();

  window.addEventListener('resize', syncSafeArea, { passive: true });
  window.addEventListener('orientationchange', syncSafeArea, { passive: true });
  window.addEventListener('pageshow', scheduleResync, { passive: true });
  window.addEventListener('focus', scheduleResync, { passive: true });
  window.addEventListener('hashchange', scheduleResync, { passive: true });
  window.addEventListener('popstate', scheduleResync, { passive: true });
  document.addEventListener('visibilitychange', scheduleResync, { passive: true });
  document.addEventListener('pointerup', scheduleResync, { passive: true, capture: true });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', syncSafeArea, { passive: true });
    window.visualViewport.addEventListener('scroll', scheduleViewportRecovery, { passive: true });
  }
  document.addEventListener('focusout', scheduleViewportRecovery, true);
  document.addEventListener('DOMContentLoaded', () => {
    syncSafeArea();
    bindTelegramEvents();
    scheduleResync();
  });
})();
