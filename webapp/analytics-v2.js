(function (global) {
  'use strict';

  const ANALYTICS_VERSION = 2;
  const SESSION_INACTIVITY_MS = 30 * 60 * 1000;
  const HEARTBEAT_INTERVAL_MS = 30 * 1000;
  const UPDATE_STALE_MS = 25 * 1000;

  function cleanParam(params, name, maxLength) {
    const value = params && params.get(name);
    if (typeof value !== 'string') return null;
    const cleaned = value.trim();
    return cleaned ? cleaned.slice(0, maxLength) : null;
  }

  function readLaunchContext() {
    let params = null;
    try {
      params = new URLSearchParams(global.location.search);
    } catch (_) {}

    const platform = cleanParam(params, 'ea_platform', 64);
    let timezone = null;
    try {
      timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || null;
    } catch (_) {}

    return {
      source: platform
        || (global.ExtraArenaApp ? 'android_app' : null)
        || (global.Telegram && global.Telegram.WebApp && global.Telegram.WebApp.initData ? 'telegram_webapp' : 'web'),
      timezone,
      entrypoint: cleanParam(params, 'entrypoint', 128),
      returnclockDecisionId:
        cleanParam(params, 'rc_decision_id', 128)
        || cleanParam(params, 'returnclock_decision_id', 128)
        || cleanParam(params, 'decision_id', 128),
      returnclockDeliveryId:
        cleanParam(params, 'delivery_id', 128)
        || cleanParam(params, 'notification_id', 128),
      notificationId: cleanParam(params, 'notification_id', 128),
    };
  }

  // Capture attribution before arena bootstrap removes authentication or other
  // launch parameters from the visible URL.
  const documentLaunchContext = readLaunchContext();

  function clearLaunchAttributionFromUrl(context) {
    if (
      !context.returnclockDecisionId
      && !context.returnclockDeliveryId
      && !context.notificationId
      && context.entrypoint !== 'notification'
    ) {
      return;
    }
    try {
      const url = new URL(global.location.href);
      [
        'rc_decision_id',
        'returnclock_decision_id',
        'decision_id',
        'delivery_id',
        'notification_id',
      ].forEach((key) => url.searchParams.delete(key));
      if (url.searchParams.get('entrypoint') === 'notification') {
        url.searchParams.delete('entrypoint');
      }
      global.history.replaceState(
        global.history.state,
        '',
        `${url.pathname}${url.search}${url.hash}`,
      );
    } catch (_) {}
  }

  function create(options) {
    const config = options || {};
    if (typeof config.apiUrl !== 'function') {
      throw new TypeError('ExtraArenaAnalyticsV2 requires apiUrl(path)');
    }

    const initialScreen = String(config.initialScreen || '').trim() || 'unknown';
    const launchContext = config.launchContext || documentLaunchContext;
    const terminalBattleIds = new Set();
    const currentSessionBattleIds = new Set();

    let sessionId = null;
    let screens = [];
    let battlesPlayed = 0;
    let casesOpenedCount = 0;
    let lastUpdate = 0;
    let updateTimer = null;
    let heartbeatTimer = null;
    let startRetryTimer = null;
    let startRequest = null;
    let hiddenAt = null;
    let ended = false;
    let started = false;
    let startConfirmed = false;
    let launchAttributionConsumed = false;

    function newSessionId() {
      if (global.crypto && typeof global.crypto.randomUUID === 'function') {
        return global.crypto.randomUUID();
      }
      return `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    }

    function getSessionId() {
      if (!sessionId) sessionId = newSessionId();
      return sessionId;
    }

    function apiUrl(path) {
      try {
        return config.apiUrl(path);
      } catch (_) {
        return null;
      }
    }

    function postJSON(path, body) {
      const url = apiUrl(path);
      if (!url) return Promise.resolve(null);
      return global.fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
        keepalive: true,
      }).catch(() => null);
    }

    async function postJSONChecked(path, body) {
      const url = apiUrl(path);
      if (!url) throw new Error('analytics_auth_unavailable');
      const response = await global.fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
        keepalive: true,
      });
      if (!response.ok) throw new Error(`analytics_http_${response.status}`);
      const result = await response.json();
      if (!result || !result.success) throw new Error('analytics_start_rejected');
      return result;
    }

    function sessionPayload(extra) {
      return Object.assign({
        session_id: getSessionId(),
        screens_visited: screens.slice(-200),
        battles_played: battlesPlayed,
        battle_ids: Array.from(currentSessionBattleIds).slice(-200),
        cases_opened: casesOpenedCount,
      }, extra || {});
    }

    function flushUpdate(extra) {
      if (ended || !started || !startConfirmed) return;
      lastUpdate = Date.now();
      postJSON('/api/analytics/session/update', sessionPayload(Object.assign({
        heartbeat: true,
      }, extra || {})));
    }

    function scheduleUpdate() {
      if (ended || !started || !startConfirmed) return;
      const now = Date.now();
      if (updateTimer) global.clearTimeout(updateTimer);
      if (now - lastUpdate > UPDATE_STALE_MS) {
        flushUpdate();
        return;
      }
      updateTimer = global.setTimeout(flushUpdate, 3000);
    }

    function stopSessionTimers() {
      if (startRetryTimer) global.clearTimeout(startRetryTimer);
      if (updateTimer) global.clearTimeout(updateTimer);
      startRetryTimer = null;
      updateTimer = null;
    }

    function dispatchEnd(url, payload) {
      if (!url) return;
      const body = JSON.stringify(payload);
      // A JWT is added by the arena fetch bridge as an Authorization header.
      // sendBeacon cannot carry that header, so only query-auth URLs are
      // beacon-safe; all other clients use a keepalive fetch.
      const beaconSafe = /(?:[?&])_auth=/.test(url);
      if (beaconSafe && global.navigator && typeof global.navigator.sendBeacon === 'function') {
        try {
          const blob = new Blob([body], {type: 'application/json'});
          if (!global.navigator.sendBeacon(url, blob)) {
            postJSON('/api/analytics/session/end', payload);
          }
        } catch (_) {
          postJSON('/api/analytics/session/end', payload);
        }
      } else {
        postJSON('/api/analytics/session/end', payload);
      }
    }

    function sendEnd(endedAtMs, reason) {
      if (ended || !started) return false;
      const currentSessionId = getSessionId();
      const hasClientBoundary = Number.isFinite(endedAtMs);
      const clientEndedAt = hasClientBoundary ? endedAtMs : Date.now();
      const wasStartConfirmed = startConfirmed;
      const pendingStart = startRequest;
      ended = true;
      started = false;
      startConfirmed = false;
      startRequest = null;
      stopSessionTimers();

      const payload = sessionPayload({
        ended_at: new Date(clientEndedAt).toISOString(),
        metadata: reason ? {end_reason: reason} : {},
      });
      const url = apiUrl('/api/analytics/session/end');
      if (wasStartConfirmed) {
        dispatchEnd(url, payload);
      } else if (pendingStart) {
        // A quick navigation can finish the document before /start inserts its
        // row. Preserve ordering so /end cannot become a lost UPDATE.
        pendingStart.then((result) => {
          if (result && result.started) dispatchEnd(url, payload);
        });
      }
      if (sessionId === currentSessionId) sessionId = null;
      return true;
    }

    function startCurrentSession(resumed) {
      if (started && !ended) return startRequest;
      ended = false;
      started = true;
      startConfirmed = false;
      lastUpdate = Date.now();
      const includeLaunchAttribution = !launchAttributionConsumed;
      const currentSessionId = getSessionId();
      const payload = {
        session_id: currentSessionId,
        analytics_version: ANALYTICS_VERSION,
        source: launchContext.source,
        timezone: launchContext.timezone,
        utc_offset_minutes: -new Date().getTimezoneOffset(),
        entrypoint: includeLaunchAttribution ? launchContext.entrypoint : null,
        returnclock_decision_id: includeLaunchAttribution ? launchContext.returnclockDecisionId : null,
        returnclock_delivery_id: includeLaunchAttribution ? launchContext.returnclockDeliveryId : null,
        notification_id: includeLaunchAttribution ? launchContext.notificationId : null,
        resumed: resumed === true,
      };
      startRequest = postJSONChecked('/api/analytics/session/start', payload)
        .then((result) => {
          if (!result || !result.started) throw new Error('analytics_session_not_started');
          if (!ended && sessionId === currentSessionId) startConfirmed = true;
          if (includeLaunchAttribution) {
            launchAttributionConsumed = true;
            clearLaunchAttributionFromUrl(launchContext);
          }
          if (screens.length && !ended && sessionId === currentSessionId) scheduleUpdate();
          return result;
        })
        .catch(() => {
          if (!ended && sessionId === currentSessionId) {
            started = false;
            startRequest = null;
            if (startRetryTimer) global.clearTimeout(startRetryTimer);
            startRetryTimer = global.setTimeout(() => startCurrentSession(resumed), 5000);
          }
          return null;
        });
      return startRequest;
    }

    function startFreshSession(resumed) {
      sessionId = null;
      screens = [{screen: initialScreen, ts: Date.now()}];
      battlesPlayed = 0;
      currentSessionBattleIds.clear();
      casesOpenedCount = 0;
      lastUpdate = 0;
      ended = false;
      started = false;
      startConfirmed = false;
      startRequest = null;
      stopSessionTimers();
      return startCurrentSession(resumed);
    }

    function handleVisible() {
      const now = Date.now();
      const hiddenStartedAt = hiddenAt;
      const hiddenFor = hiddenStartedAt == null ? 0 : now - hiddenStartedAt;
      hiddenAt = null;
      if (ended) {
        startFreshSession(true);
      } else if (started && hiddenFor >= SESSION_INACTIVITY_MS) {
        // End at the moment of backgrounding. Ending at resume time would hide
        // the actual return gap from ReturnClock labels.
        sendEnd(hiddenStartedAt, 'background_inactivity');
        startFreshSession(true);
      } else if (started) {
        flushUpdate({resumed: hiddenStartedAt != null});
      }
    }

    function start() {
      if (!screens.length) screens = [{screen: initialScreen, ts: Date.now()}];
      return startCurrentSession(false);
    }

    function screen(name) {
      const cleaned = String(name || '').trim();
      if (!cleaned || (screens.length && screens[screens.length - 1].screen === cleaned)) return;
      screens.push({screen: cleaned, ts: Date.now()});
      if (screens.length > 250) screens = screens.slice(-200);
      scheduleUpdate();
    }

    function battleFinished(battleId) {
      if (ended) return false;
      const key = String(battleId || 'arena-document-battle');
      if (terminalBattleIds.has(key)) return false;
      terminalBattleIds.add(key);
      currentSessionBattleIds.add(key);
      battlesPlayed += 1;
      scheduleUpdate();
      return true;
    }

    function caseOpened() {
      if (ended) return false;
      casesOpenedCount += 1;
      scheduleUpdate();
      return true;
    }

    if (global.addEventListener && global.document) {
      const handlePageExit = () => sendEnd(
        Number.isFinite(hiddenAt) ? hiddenAt : Date.now(),
        'page_exit',
      );
      global.addEventListener('pagehide', handlePageExit);
      global.addEventListener('beforeunload', handlePageExit);
      global.addEventListener('pageshow', handleVisible);
      global.document.addEventListener('visibilitychange', () => {
        if (global.document.visibilityState === 'hidden') {
          hiddenAt = Date.now();
          flushUpdate();
        } else if (global.document.visibilityState === 'visible') {
          handleVisible();
        }
      });
      heartbeatTimer = global.setInterval(() => {
        if (!ended && started && global.document.visibilityState === 'visible' && Date.now() - lastUpdate > UPDATE_STALE_MS) {
          flushUpdate();
        }
      }, HEARTBEAT_INTERVAL_MS);
    }

    return {
      start,
      screen,
      battleFinished,
      caseOpened,
      end: () => sendEnd(null, 'explicit'),
      destroy: () => {
        sendEnd(null, 'destroy');
        if (heartbeatTimer) global.clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      },
    };
  }

  global.ExtraArenaAnalyticsV2 = Object.freeze({
    version: ANALYTICS_VERSION,
    create,
  });
})(window);
