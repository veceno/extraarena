import json
import subprocess
from pathlib import Path


ARENA_HTML = Path("webapp/arena.html")
ARENA_JS = Path("webapp/arena.js")
ANALYTICS_JS = Path("webapp/analytics-v2.js")
ANDROID_BUILD = Path("android-app/app/build.gradle.kts")
ANDROID_ACTIVITY = Path(
    "android-app/app/src/main/java/ru/extraarena/app/MainActivity.java"
)


def test_arena_loads_analytics_before_bootstrap_and_starts_only_after_auth():
    markup = ARENA_HTML.read_text(encoding="utf-8")
    arena = ARENA_JS.read_text(encoding="utf-8")

    assert markup.index('src="analytics-v2.js?') < markup.index('src="arena.js?')
    assert "analyticsFactory.create({" in arena
    assert "apiUrl: buildArenaAuthUrl" in arena
    assert "initialScreen: 'arena'" in arena

    auth_guard = arena.split("if (!authToken) {", 1)[1].split("initTalkies();", 1)[0]
    assert "startArenaAnalytics();" in auth_guard
    assert auth_guard.index("return;") < auth_guard.index("startArenaAnalytics();")


def test_android_shell_packages_and_routes_arena_analytics_module():
    build = ANDROID_BUILD.read_text(encoding="utf-8")
    activity = ANDROID_ACTIVITY.read_text(encoding="utf-8")

    assert 'include("analytics-v2.js")' in build
    assert '|| "analytics-v2.js".equals(clean)' in activity


def test_arena_counts_a_terminal_battle_once_at_result_boundary():
    arena = ARENA_JS.read_text(encoding="utf-8")
    main = Path("webapp/index.html").read_text(encoding="utf-8")
    result_block = arena.split("function showBattleResult(", 1)[1].split(
        "function animateNumber",
        1,
    )[0]

    assert "arenaAnalytics?.battleFinished(matchId);" in result_block
    assert "terminalBattleIds.has(key)" in ANALYTICS_JS.read_text(encoding="utf-8")
    assert "terminalBattleIds.add(key)" in ANALYTICS_JS.read_text(encoding="utf-8")
    assert "currentSessionBattleIds.add(key)" in ANALYTICS_JS.read_text(
        encoding="utf-8"
    )
    assert "battle_ids: Array.from(currentSessionBattleIds)" in (
        ANALYTICS_JS.read_text(encoding="utf-8")
    )
    assert "window.__analytics?.battlePlayed()" not in arena
    assert "window.__analytics?.battlePlayed()" not in main


def test_arena_analytics_v2_runtime_splits_long_background_and_preserves_terminal_dedup():
    script = r"""
const fs = require('fs');
const vm = require('vm');

let now = 1_000;
let uuidCounter = 0;
const requests = [];
const windowListeners = {};
const documentListeners = {};
const historyCalls = [];

class FakeDate extends Date {
  constructor(...args) {
    super(...(args.length ? args : [now]));
  }
  static now() { return now; }
  getTimezoneOffset() { return -180; }
}

const document = {
  visibilityState: 'visible',
  addEventListener(name, callback) { documentListeners[name] = callback; },
};
const window = {
  location: {
    search: '?id=match-1&ea_platform=android_app&entrypoint=notification&rc_decision_id=decision-1&delivery_id=delivery-1&notification_id=outbox-1',
    href: 'https://example.test/arena?id=match-1&ea_platform=android_app&entrypoint=notification&rc_decision_id=decision-1&delivery_id=delivery-1&notification_id=outbox-1',
  },
  history: {
    state: null,
    replaceState(state, title, url) { historyCalls.push(url); },
  },
  document,
  crypto: {randomUUID() { uuidCounter += 1; return `session-${uuidCounter}`; }},
  ExtraArenaApp: {},
  addEventListener(name, callback) { windowListeners[name] = callback; },
  setTimeout(callback) { return {callback}; },
  clearTimeout() {},
  setInterval(callback) { return {callback}; },
  clearInterval() {},
  fetch(url, options) {
    requests.push({url, body: JSON.parse(options.body)});
    return Promise.resolve({
      ok: true,
      json: async () => ({success: true, started: true}),
    });
  },
  navigator: {sendBeacon() { throw new Error('JWT path must use keepalive fetch'); }},
};

const context = {
  window,
  URL,
  URLSearchParams,
  Intl,
  Date: FakeDate,
  Blob,
  Math,
  Set,
  Object,
  String,
  TypeError,
  Error,
  Promise,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('webapp/analytics-v2.js', 'utf8'), context);

(async () => {
  const analytics = window.ExtraArenaAnalyticsV2.create({
    apiUrl: (path) => path,
    initialScreen: 'arena',
  });
  await analytics.start();
  const firstTerminal = analytics.battleFinished('match-1');
  const duplicateTerminal = analytics.battleFinished('match-1');

  now = 5_000;
  document.visibilityState = 'hidden';
  documentListeners.visibilitychange();

  now = 5_000 + 30 * 60 * 1_000;
  document.visibilityState = 'visible';
  documentListeners.visibilitychange();
  await new Promise((resolve) => setImmediate(resolve));

  const duplicateAfterResume = analytics.battleFinished('match-1');
  analytics.end();
  await new Promise((resolve) => setImmediate(resolve));

  process.stdout.write(JSON.stringify({
    requests,
    historyCalls,
    firstTerminal,
    duplicateTerminal,
    duplicateAfterResume,
  }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""
    result = json.loads(
        subprocess.check_output(["node", "-e", script], text=True, cwd=Path.cwd())
    )
    starts = [
        request["body"]
        for request in result["requests"]
        if request["url"] == "/api/analytics/session/start"
    ]
    ends = [
        request["body"]
        for request in result["requests"]
        if request["url"] == "/api/analytics/session/end"
    ]
    updates = [
        request["body"]
        for request in result["requests"]
        if request["url"] == "/api/analytics/session/update"
    ]

    assert len(starts) == 2
    assert starts[0] == {
        "session_id": "session-1",
        "analytics_version": 2,
        "source": "android_app",
        "timezone": starts[0]["timezone"],
        "utc_offset_minutes": 180,
        "entrypoint": "notification",
        "returnclock_decision_id": "decision-1",
        "returnclock_delivery_id": "delivery-1",
        "notification_id": "outbox-1",
        "resumed": False,
    }
    assert starts[0]["timezone"]
    assert starts[1]["session_id"] == "session-2"
    assert starts[1]["resumed"] is True
    assert starts[1]["entrypoint"] is None
    assert starts[1]["returnclock_decision_id"] is None
    assert starts[1]["returnclock_delivery_id"] is None
    assert starts[1]["notification_id"] is None

    assert result["firstTerminal"] is True
    assert result["duplicateTerminal"] is False
    assert result["duplicateAfterResume"] is False

    assert updates
    assert updates[-1]["heartbeat"] is True
    assert updates[-1]["screens_visited"][0]["screen"] == "arena"

    assert len(ends) == 2
    assert ends[0]["session_id"] == "session-1"
    assert ends[0]["battles_played"] == 1
    assert ends[0]["battle_ids"] == ["match-1"]
    assert ends[0]["ended_at"] == "1970-01-01T00:00:05.000Z"
    assert ends[0]["metadata"] == {"end_reason": "background_inactivity"}
    assert ends[1]["session_id"] == "session-2"
    assert ends[1]["battles_played"] == 0
    assert ends[1]["battle_ids"] == []
    assert ends[1]["metadata"] == {"end_reason": "explicit"}

    assert result["historyCalls"]
    assert "rc_decision_id" not in result["historyCalls"][0]
    assert "delivery_id" not in result["historyCalls"][0]
    assert "notification_id" not in result["historyCalls"][0]


def test_arena_analytics_v2_has_heartbeat_and_document_end_boundaries():
    source = ANALYTICS_JS.read_text(encoding="utf-8")

    assert "const ANALYTICS_VERSION = 2;" in source
    assert "const SESSION_INACTIVITY_MS = 30 * 60 * 1000;" in source
    assert "const HEARTBEAT_INTERVAL_MS = 30 * 1000;" in source
    assert "heartbeat: true" in source
    assert "const handlePageExit = () => sendEnd(" in source
    assert "Number.isFinite(hiddenAt) ? hiddenAt : Date.now()" in source
    assert "global.addEventListener('pagehide', handlePageExit)" in source
    assert "global.addEventListener('beforeunload', handlePageExit)" in source
    assert "hiddenFor >= SESSION_INACTIVITY_MS" in source
    assert "sendEnd(hiddenStartedAt, 'background_inactivity')" in source
    assert "screens = [{screen: initialScreen, ts: Date.now()}];" in source
    assert "currentSessionBattleIds.clear();" in source
