(function () {
  const MAX_SESSION_KEY = 'max_game_session_token';
  const PLATFORM_SESSION_KEY = 'extraarena_launch_platform';
  const telegram = () => window.Telegram && window.Telegram.WebApp;
  const max = () => window.WebApp;

  const hasInitData = (app) => !!(
    app
    && typeof app.initData === 'string'
    && app.initData.trim()
  );

  const platformFromUrl = (rawUrl) => {
    if (!rawUrl) return null;
    try {
      const url = new URL(String(rawUrl), window.location.href);
      const hash = new URLSearchParams(url.hash.replace(/^#/, ''));
      const telegramMarker = hash.get('tgWebAppData');
      const maxMarker = hash.get('WebAppData');
      if (telegramMarker === null && maxMarker === null) return null;
      if (telegramMarker !== null && maxMarker !== null) return 'web';
      if (telegramMarker !== null) {
        return (
          telegramMarker.trim()
          && hasInitData(telegram())
          && telegramMarker === telegram().initData
        ) ? 'telegram' : 'web';
      }
      return (
        maxMarker.trim()
        && hasInitData(max())
        && maxMarker === max().initData
      ) ? 'max' : 'web';
    } catch (_) {
      return null;
    }
  };

  const explicitLaunchPlatform = () => {
    const current = platformFromUrl(window.location.href);
    if (current) return current;
    try {
      const navigation = window.performance?.getEntriesByType?.('navigation')?.[0];
      return platformFromUrl(navigation?.name);
    } catch (_) {
      return null;
    }
  };

  const getStoredLaunchPlatform = () => {
    try {
      const value = sessionStorage.getItem(PLATFORM_SESSION_KEY);
      return value === 'telegram' || value === 'max' ? value : null;
    } catch (_) {
      return null;
    }
  };

  const rememberLaunchPlatform = (value) => {
    if (value !== 'telegram' && value !== 'max') return;
    try { sessionStorage.setItem(PLATFORM_SESSION_KEY, value); } catch (_) {}
  };

  const clearLaunchPlatform = () => {
    try { sessionStorage.removeItem(PLATFORM_SESSION_KEY); } catch (_) {}
  };

  const resolvePlatform = () => {
    const explicit = explicitLaunchPlatform();
    if (explicit) {
      if (explicit === 'web') clearLaunchPlatform();
      else rememberLaunchPlatform(explicit);
      return explicit;
    }

    // Full-page navigation to /arena drops the Mini App launch hash. Only our
    // own marker may carry the already resolved platform across that boundary.
    const stored = getStoredLaunchPlatform();
    if (stored) return stored;

    const hasTelegramInitData = hasInitData(telegram());
    const hasMaxInitData = hasInitData(max());
    if (hasTelegramInitData === hasMaxInitData) {
      // Both SDKs restore initData from their own sessionStorage. When both are
      // populated without an explicit launch marker, choosing either identity
      // would risk authenticating the wrong platform account.
      return 'web';
    }
    return hasTelegramInitData ? 'telegram' : 'max';
  };

  const api = {
    kind() {
      return resolvePlatform();
    },

    isMax() {
      return this.kind() === 'max';
    },

    isTelegram() {
      return this.kind() === 'telegram';
    },

    getInitData() {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      if (hasInitData(app)) return app.initData;
      return null;
    },

    getUnsafeUser() {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      if (app) return app.initDataUnsafe && app.initDataUnsafe.user;
      return null;
    },

    getStoredAuthToken() {
      if (!this.isMax()) return null;
      try {
        const token = sessionStorage.getItem(MAX_SESSION_KEY);
        return token && token.trim() ? token : null;
      } catch (_) {
        return null;
      }
    },

    async ensureAuthSession() {
      if (!this.isMax()) return null;
      if (this._authPromise) return this._authPromise;
      this._authPromise = (async () => {
        const maxApp = max();
        if (!hasInitData(maxApp)) throw new Error('max_init_data_missing');
        const response = await fetch('/api/auth/max', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          cache: 'no-store',
          body: JSON.stringify({
            init_data: maxApp.initData,
            device_label: `MAX ${maxApp.platform || 'Mini App'} ${maxApp.version || ''}`.trim(),
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.token) {
          try { sessionStorage.removeItem(MAX_SESSION_KEY); } catch (_) {}
          throw new Error(payload.error || `max_auth_http_${response.status}`);
        }
        try { sessionStorage.setItem(MAX_SESSION_KEY, payload.token); } catch (_) {}
        window.dispatchEvent(new CustomEvent('extraarena:max-auth-ready', {detail: payload}));
        return payload.token;
      })();
      return this._authPromise;
    },

    ready() {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try { app && app.ready && app.ready(); } catch (_) {}
      try { app && app.expand && app.expand(); } catch (_) {}
    },

    openLink(url) {
      const href = String(url || '');
      if (!href) return false;
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try {
        if (this.isTelegram() && /^https:\/\/t\.me\//i.test(href) && app.openTelegramLink) {
          app.openTelegramLink(href);
          return true;
        }
        if (app && app.openLink) {
          app.openLink(href);
          return true;
        }
      } catch (_) {}
      return false;
    },

    async share(text, link) {
      const payload = {text: String(text || ''), link: String(link || '')};
      try {
        if (this.isMax() && max().shareMaxContent) {
          await max().shareMaxContent(payload);
          return true;
        }
      } catch (_) {}
      return false;
    },

    close() {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try {
        if (app && app.close) {
          app.close();
          return true;
        }
      } catch (_) {}
      return false;
    },

    impact(style) {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try { app && app.HapticFeedback && app.HapticFeedback.impactOccurred(style || 'light'); } catch (_) {}
    },

    notification(type) {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try { app && app.HapticFeedback && app.HapticFeedback.notificationOccurred(type); } catch (_) {}
    },

    selection() {
      const app = this.isMax() ? max() : (this.isTelegram() ? telegram() : null);
      try { app && app.HapticFeedback && app.HapticFeedback.selectionChanged(); } catch (_) {}
    },
  };

  window.ExtraArenaPlatform = api;
  if (api.isMax()) {
    api.ready();
    // Start the signed exchange while the rest of the application assets load.
    api.ensureAuthSession().catch((error) => {
      console.warn('[MAX] launch authentication failed:', error && error.message);
    });
  }
})();
