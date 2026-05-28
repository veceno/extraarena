(function () {
  const root = document.documentElement;
  const sides = ['top', 'right', 'bottom', 'left'];
  const KEYBOARD_DELTA_PX = 120;
  const VIEWPORT_RECOVERY_DELAY_MS = 260;
  let eventsBound = false;
  let lastStableViewportHeight = 0;
  let recoveryTimer = 0;

  const toPx = (value) => {
    const number = Number(value);
    return `${Number.isFinite(number) && number > 0 ? number : 0}px`;
  };

  const applyInsets = (prefix, insets) => {
    if (!insets || typeof insets !== 'object') return;
    sides.forEach((side) => {
      if (Object.prototype.hasOwnProperty.call(insets, side)) {
        root.style.setProperty(`${prefix}${side}`, toPx(insets[side]));
      }
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

  const syncSafeArea = () => {
    const tg = window.Telegram && window.Telegram.WebApp;

    if (tg) {
      applyInsets('--ea-tg-safe-', tg.safeAreaInset);
      applyInsets('--ea-tg-content-safe-', tg.contentSafeAreaInset);

      const stableHeight = Number(tg.viewportStableHeight || tg.viewportHeight);
      if (Number.isFinite(stableHeight) && stableHeight > 0) {
        setViewportHeight(stableHeight);
        return;
      }
    }

    const currentHeight = usableViewportHeight();
    if (!currentHeight) {
      return;
    }

    if (isKeyboardLikelyOpen()) {
      if (lastStableViewportHeight > 0) {
        setViewportHeight(lastStableViewportHeight);
      }
      scheduleViewportRecovery();
      return;
    }

    lastStableViewportHeight = currentHeight;
    setViewportHeight(currentHeight);
  };

  const bindTelegramEvents = () => {
    const tg = window.Telegram && window.Telegram.WebApp;
    if (!tg || typeof tg.onEvent !== 'function' || eventsBound) return;
    eventsBound = true;

    ['viewportChanged', 'safeAreaChanged', 'contentSafeAreaChanged', 'fullscreenChanged'].forEach((eventName) => {
      try {
        tg.onEvent(eventName, syncSafeArea);
      } catch (error) {
        console.warn('[safe-area] event bind failed:', eventName, error);
      }
    });
  };

  window.ExtraArenaSafeArea = { sync: syncSafeArea };

  syncSafeArea();
  bindTelegramEvents();

  window.addEventListener('resize', syncSafeArea, { passive: true });
  window.addEventListener('orientationchange', syncSafeArea, { passive: true });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', syncSafeArea, { passive: true });
    window.visualViewport.addEventListener('scroll', scheduleViewportRecovery, { passive: true });
  }
  document.addEventListener('focusout', scheduleViewportRecovery, true);
  document.addEventListener('DOMContentLoaded', () => {
    syncSafeArea();
    bindTelegramEvents();
  });
})();
