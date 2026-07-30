(function () {
  const MAX_SESSION_KEY = 'max_game_session_token';
  const telegram = () => window.Telegram && window.Telegram.WebApp;
  const max = () => window.WebApp;

  const hasInitData = (app) => !!(
    app
    && typeof app.initData === 'string'
    && app.initData.trim()
  );

  const api = {
    kind() {
      if (hasInitData(max())) return 'max';
      if (hasInitData(telegram())) return 'telegram';
      return 'web';
    },

    isMax() {
      return this.kind() === 'max';
    },

    isTelegram() {
      return this.kind() === 'telegram';
    },

    getInitData() {
      if (this.isMax()) return max().initData;
      if (this.isTelegram()) return telegram().initData;
      return null;
    },

    getUnsafeUser() {
      if (this.isMax()) return max().initDataUnsafe && max().initDataUnsafe.user;
      if (this.isTelegram()) return telegram().initDataUnsafe && telegram().initDataUnsafe.user;
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
        const response = await fetch('/api/auth/max', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          cache: 'no-store',
          body: JSON.stringify({
            init_data: max().initData,
            device_label: `MAX ${max().platform || 'Mini App'} ${max().version || ''}`.trim(),
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
      const app = this.isMax() ? max() : telegram();
      try { app && app.ready && app.ready(); } catch (_) {}
      try { app && app.expand && app.expand(); } catch (_) {}
    },

    openLink(url) {
      const href = String(url || '');
      if (!href) return false;
      const app = this.isMax() ? max() : telegram();
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
      const app = this.isMax() ? max() : telegram();
      try {
        if (app && app.close) {
          app.close();
          return true;
        }
      } catch (_) {}
      return false;
    },

    impact(style) {
      const app = this.isMax() ? max() : telegram();
      try { app && app.HapticFeedback && app.HapticFeedback.impactOccurred(style || 'light'); } catch (_) {}
    },

    notification(type) {
      const app = this.isMax() ? max() : telegram();
      try { app && app.HapticFeedback && app.HapticFeedback.notificationOccurred(type); } catch (_) {}
    },

    selection() {
      const app = this.isMax() ? max() : telegram();
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
