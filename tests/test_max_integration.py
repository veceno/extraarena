import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import time
import uuid
from types import SimpleNamespace
from urllib.parse import urlencode

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import main
from bot.max_client import normalize_max_update
from web import max_integration


BOT_TOKEN = "max-test-bot-token"


def _signed_init_data(
    *,
    user_id: int = 67890,
    auth_date: int | None = None,
    bot_token: str = BOT_TOKEN,
) -> str:
    values = {
        "auth_date": str(auth_date or int(time.time())),
        "query_id": str(uuid.uuid4()),
        "user": json.dumps(
            {
                "id": user_id,
                "first_name": "Max",
                "last_name": "Player",
                "username": "max_player",
                "language_code": "ru",
                "photo_url": "https://example.com/max-player.jpg",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    launch_params = "\n".join(
        f"{key}={value}" for key, value in sorted(values.items())
    )
    secret = hmac.new(
        b"WebAppData",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    values["hash"] = hmac.new(
        secret,
        launch_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return urlencode(values)


def _settings(**overrides):
    values = {
        "max_bot_token": BOT_TOKEN,
        "max_bot_webhook_secret": "webhook-secret",
        "jwt_secret": "max-session-jwt-secret-long-enough-for-hs256-tests",
        "jwt_expiry_days": 30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_max_init_data_signature_and_user_are_verified():
    verified = max_integration.verify_max_init_data(_signed_init_data(), BOT_TOKEN)

    assert verified is not None
    assert max_integration.extract_max_user(verified) == {
        "id": 67890,
        "first_name": "Max",
        "last_name": "Player",
        "username": "max_player",
        "language_code": "ru",
        "photo_url": "https://example.com/max-player.jpg",
    }


def test_max_init_data_rejects_tampering_duplicates_and_stale_launches():
    valid = _signed_init_data()
    assert max_integration.verify_max_init_data(
        valid.replace("Max%22", "Evil%22"),
        BOT_TOKEN,
    ) is None
    assert max_integration.verify_max_init_data(
        valid + "&auth_date=1",
        BOT_TOKEN,
    ) is None
    assert max_integration.verify_max_init_data(
        _signed_init_data(auth_date=int(time.time()) - 7200),
        BOT_TOKEN,
    ) is None


def test_max_webhook_update_normalization_supports_bot_and_message_shapes():
    started = normalize_max_update(
        {
            "update_type": "bot_started",
            "user": {"user_id": 123, "first_name": "Игрок"},
        }
    )
    message = normalize_max_update(
        {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 456, "first_name": "Макс"},
                "body": {"text": "/start payload"},
            },
        }
    )

    assert started["user_id"] == "123"
    assert started["display_name"] == "Игрок"
    assert message["user_id"] == "456"
    assert message["text"] == "/start payload"


def test_shared_webapp_bootstraps_max_before_render_and_keeps_identity_bound():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    arena = Path("webapp/arena.js").read_text(encoding="utf-8")
    server = Path("web/server.py").read_text(encoding="utf-8")

    assert '<script src="https://st.max.ru/js/max-web-app.js"></script>' in index
    assert '<script src="platform-bridge.js?v=auth-platform-20260730"></script>' in index
    assert "await window.ExtraArenaPlatform.ensureAuthSession();" in index
    bootstrap = index.split("const renderExtraArenaApp = () =>", 1)[1].split(
        "</script>",
        1,
    )[0]
    assert bootstrap.index("persistAppAuthFromUrl();") < bootstrap.index(
        "await window.ExtraArenaPlatform.ensureAuthSession();"
    )
    assert "if (isMaxGameClient())" in index
    assert "add('auth', getMaxAuthToken(), 'max');" in index
    assert "const isPlatformBound = isPlatformFlow" in index
    assert "hasLocalExtraSession && !isPlatformFlow" in index
    assert "cannot_delete_max_bound" in Path(
        "web/extraid_handlers.py"
    ).read_text(encoding="utf-8")
    assert "register_max_routes(app)" in server
    assert "await platform.ensureAuthSession();" in arena
    assert "window.ExtraArenaPlatform?.isMax?.()" in arena


def test_platform_bridge_resolves_live_launch_before_stale_other_sdk_data():
    script = r"""
const fs = require('fs');
const vm = require('vm');
const bridge = fs.readFileSync('webapp/platform-bridge.js', 'utf8');

async function runScenario(options) {
  const storage = new Map(Object.entries(options.storage || {}));
  const sessionStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); },
  };
  const fetchCalls = [];
  const location = {href: options.url};
  const window = {
    location,
    performance: {
      getEntriesByType(type) {
        return type === 'navigation' && options.navigationUrl
          ? [{name: options.navigationUrl}]
          : [];
      },
    },
    Telegram: {
      WebApp: {
        initData: options.telegramInit || '',
        initDataUnsafe: {user: {id: 101}},
      },
    },
    WebApp: {
      initData: options.maxInit || '',
      initDataUnsafe: {user: {id: 202}},
      platform: 'test',
      version: '1',
    },
    dispatchEvent() {},
  };
  const context = {
    window,
    sessionStorage,
    URL,
    URLSearchParams,
    CustomEvent: function CustomEvent() {},
    fetch: async (...args) => {
      fetchCalls.push(args);
      return {ok: true, status: 200, json: async () => ({token: 'max-jwt'})};
    },
    console,
  };
  vm.createContext(context);
  vm.runInContext(bridge, context);
  await Promise.resolve();
  await Promise.resolve();
  return {
    kind: window.ExtraArenaPlatform.kind(),
    initData: window.ExtraArenaPlatform.getInitData(),
    maxAuthCalls: fetchCalls.filter(([url]) => url === '/api/auth/max').length,
    storedPlatform: storage.get('extraarena_launch_platform') || null,
  };
}

(async () => {
  const results = {
    telegramLive: await runScenario({
      url: 'https://game.example/#tgWebAppData=tg-live',
      telegramInit: 'tg-live',
      maxInit: 'max-stale',
    }),
    maxLive: await runScenario({
      url: 'https://game.example/#WebAppData=max-live',
      telegramInit: 'tg-stale',
      maxInit: 'max-live',
    }),
    navigationMarker: await runScenario({
      url: 'https://game.example/',
      navigationUrl: 'https://game.example/#tgWebAppData=tg-live',
      telegramInit: 'tg-live',
      maxInit: 'max-stale',
    }),
    arenaInherited: await runScenario({
      url: 'https://game.example/arena?id=match-1',
      storage: {extraarena_launch_platform: 'telegram'},
      telegramInit: 'tg-live',
      maxInit: 'max-stale',
    }),
    ambiguous: await runScenario({
      url: 'https://game.example/',
      telegramInit: 'tg-stale',
      maxInit: 'max-stale',
    }),
    conflictingMarkers: await runScenario({
      url: 'https://game.example/#tgWebAppData=tg-live&WebAppData=max-live',
      storage: {extraarena_launch_platform: 'max'},
      telegramInit: 'tg-live',
      maxInit: 'max-live',
    }),
    emptyMarker: await runScenario({
      url: 'https://game.example/#WebAppData=',
      storage: {extraarena_launch_platform: 'telegram'},
      telegramInit: 'tg-stale',
      maxInit: 'max-stale',
    }),
    querySpoofIgnored: await runScenario({
      url: 'https://game.example/?WebAppData=max-live',
      storage: {extraarena_launch_platform: 'telegram'},
      telegramInit: 'tg-live',
      maxInit: 'max-live',
    }),
  };
  process.stdout.write(JSON.stringify(results));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""
    result = json.loads(
        subprocess.check_output(["node", "-e", script], text=True, cwd=Path.cwd())
    )

    assert result["telegramLive"] == {
        "kind": "telegram",
        "initData": "tg-live",
        "maxAuthCalls": 0,
        "storedPlatform": "telegram",
    }
    assert result["maxLive"] == {
        "kind": "max",
        "initData": "max-live",
        "maxAuthCalls": 1,
        "storedPlatform": "max",
    }
    assert result["navigationMarker"]["kind"] == "telegram"
    assert result["navigationMarker"]["maxAuthCalls"] == 0
    assert result["arenaInherited"]["kind"] == "telegram"
    assert result["arenaInherited"]["initData"] == "tg-live"
    assert result["ambiguous"] == {
        "kind": "web",
        "initData": None,
        "maxAuthCalls": 0,
        "storedPlatform": None,
    }
    assert result["conflictingMarkers"] == {
        "kind": "web",
        "initData": None,
        "maxAuthCalls": 0,
        "storedPlatform": None,
    }
    assert result["emptyMarker"] == {
        "kind": "web",
        "initData": None,
        "maxAuthCalls": 0,
        "storedPlatform": None,
    }
    assert result["querySpoofIgnored"] == {
        "kind": "telegram",
        "initData": "tg-live",
        "maxAuthCalls": 0,
        "storedPlatform": "telegram",
    }


def test_shared_auth_consumers_honor_resolved_platform():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    arena = Path("webapp/arena.js").read_text(encoding="utf-8")

    telegram_helper = index.split("function getTelegramInitData()", 1)[1].split(
        "function getMaxAuthToken",
        1,
    )[0]
    arena_bootstrap = arena.split("document.addEventListener('DOMContentLoaded'", 1)[1].split(
        "console.log('[ARENA] Match ID:'",
        1,
    )[0]
    browser_guard = arena.split("function isUnsupportedExternalArenaBrowser", 1)[1].split(
        "function showArenaLaunchError",
        1,
    )[0]

    assert "!platform.isTelegram?.()" in telegram_helper
    assert "platform?.getInitData?.()" in telegram_helper
    assert "platform?.isTelegram?.()" in arena_bootstrap
    assert "window.ExtraArenaPlatform?.isTelegram?.()" in browser_guard
    assert '<script src="platform-bridge.js?v=auth-platform-20260730"></script>' in Path(
        "webapp/arena.html"
    ).read_text(encoding="utf-8")


def test_max_launch_scrubs_foreign_url_auth_before_generic_persistence():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    persist_block = index.split("function persistAppAuthFromUrl()", 1)[1].split(
        "async function loadExtraIDProfile",
        1,
    )[0]
    max_guard = persist_block.split("if (isMaxGameClient())", 1)[1].split(
        "if (getTelegramInitData())",
        1,
    )[0]
    candidates_block = index.split("function getUiAuthCandidates", 1)[1].split(
        "function resolveUserId",
        1,
    )[0]

    assert persist_block.index("if (isMaxGameClient())") < persist_block.index(
        "const token = getUrlAuthToken();"
    )
    assert "sessionStorage.removeItem(EXTRA_URL_AUTH_SESSION_KEY)" in max_guard
    assert "sessionStorage.removeItem(EXTRA_ID_TOKEN_SESSION_KEY)" in max_guard
    assert "localStorage.removeItem('extra_id_token')" in max_guard
    assert "clean.searchParams.delete('_auth')" in max_guard
    assert "history.replaceState(null, '', clean.pathname + clean.search + clean.hash)" in max_guard
    assert "return;" in max_guard
    assert "if (isMaxGameClient())" in candidates_block
    assert "add('auth', getMaxAuthToken(), 'max');" in candidates_block
    assert candidates_block.index("add('auth', getMaxAuthToken(), 'max');") < candidates_block.index(
        "return candidates;"
    )


def test_max_real_button_taps_delegate_haptics_without_android_double_fire():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    bridge = Path("webapp/platform-bridge.js").read_text(encoding="utf-8")
    haptic_block = index.split("function playMaxControlHaptic(target)", 1)[1].split(
        "// Глобальный обработчик",
        1,
    )[0]
    pointer_block = index.split(
        "document.addEventListener('pointerdown', e => {",
        1,
    )[1].split("}, {passive: true});", 1)[0]
    click_block = index.split(
        "document.addEventListener('click', e => {",
        1,
    )[1].split("}, {passive: true});", 1)[0]

    assert "isAndroidAppShell()) return;" in haptic_block
    assert "if (!platform?.isMax?.()) return;" in haptic_block
    assert "!window.isHapticsEnabled()" in haptic_block
    assert "control.closest('[data-no-global-haptic]')" in haptic_block
    assert "control.getAttribute('data-haptic') || 'selection'" in haptic_block
    assert "platform.selection?.()" in haptic_block
    assert "platform.impact?.(impact)" in haptic_block
    assert "platform.notification?.(feedback)" in haptic_block
    assert "window.playClick();" in pointer_block
    assert "playMaxControlHaptic(control);" not in pointer_block
    assert "if (!e.isTrusted) return;" in click_block
    assert "playMaxControlHaptic(control);" in click_block
    assert "aria-label=\"Меню\"" in index
    assert haptic_block.index("isAndroidAppShell()) return;") < haptic_block.index(
        "platform.selection?.()"
    )
    assert haptic_block.index("!window.isHapticsEnabled()") < haptic_block.index(
        "platform.selection?.()"
    )

    assert "impactOccurred(style || 'light')" in bridge
    assert "notificationOccurred(type)" in bridge
    assert "selectionChanged()" in bridge


def test_max_settings_expose_haptic_toggle_in_both_native_shells():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    settings_block = index.split("const SettingsScreen =", 1)[1].split(
        "// CASE OPENING OVERLAY",
        1,
    )[0]

    assert "const isMaxShell = isMaxGameClient();" in settings_block
    assert "const canUseHaptics = isAndroidShell || isMaxShell;" in settings_block
    assert '{canUseHaptics && <Row label="Виброотдача"' in settings_block


@pytest.mark.asyncio
async def test_max_auth_exchange_issues_internal_session(monkeypatch):
    class GameDB:
        def __init__(self):
            self.calls = []

        async def resolve_or_create_platform_user(self, **kwargs):
            self.calls.append(kwargs)
            return 8_000_000_000_000_001, True

    class ExtraIDDB:
        def __init__(self):
            self.executed = []
            self.session = None

        async def execute(self, query, *args):
            self.executed.append((query, args))

        async def create_auth_session(self, *args, **kwargs):
            self.session = (args, kwargs)

        async def get_any_extra_account_by_user_id(self, user_id):
            assert user_id == 8_000_000_000_000_001
            return None

    game_db = GameDB()
    extra_db = ExtraIDDB()
    app = web.Application()
    app["db"] = game_db
    app["extraid_db"] = extra_db
    app.router.add_post("/api/auth/max", max_integration.max_auth_exchange_handler)
    monkeypatch.setattr(max_integration, "get_settings", lambda: _settings())

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/auth/max",
            json={"init_data": _signed_init_data(user_id=987654321)},
        )
        body = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert body["ok"] is True
    assert body["provider"] == "max"
    assert body["user_id"] == 8_000_000_000_000_001
    decoded = jwt.decode(
        body["token"],
        "max-session-jwt-secret-long-enough-for-hs256-tests",
        algorithms=["HS256"],
    )
    assert decoded["user_id"] == 8_000_000_000_000_001
    assert game_db.calls[0]["provider"] == "max"
    assert game_db.calls[0]["subject"] == "987654321"
    assert extra_db.session[0][1] == "max"


@pytest.mark.asyncio
async def test_max_webhook_requires_secret_and_sends_open_app(monkeypatch):
    class Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, user_id, text, *, open_app=False, **kwargs):
            self.sent.append((user_id, text, open_app))
            return {"ok": True, "status": 200, "data": {}}

    bot = Bot()
    app = web.Application()
    app["max_bot_client"] = bot
    app.router.add_post("/api/max/webhook", max_integration.max_bot_webhook_handler)
    monkeypatch.setattr(max_integration, "get_settings", lambda: _settings())

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        rejected = await client.post(
            "/api/max/webhook",
            json={"update_type": "bot_started", "user": {"user_id": 123}},
        )
        accepted = await client.post(
            "/api/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "webhook-secret"},
            json={"update_type": "bot_started", "user": {"user_id": 123}},
        )
    finally:
        await client.close()

    assert rejected.status == 401
    assert accepted.status == 200
    assert bot.sent == [
        (
            "123",
            "Добро пожаловать в ExtraArena!\n\n"
            "Открой игру кнопкой ниже. MAX-аккаунт будет подтверждён автоматически, "
            "а ExtraID можно создать и навсегда привязать внутри игры.",
            True,
        )
    ]


@pytest.mark.asyncio
async def test_notification_outbox_routes_max_identity_to_max_bot():
    class DB:
        def __init__(self):
            self.sent = []
            self.failed = []

        async def get_notification_delivery_mode(self, user_id):
            return "telegram_only"

        async def get_platform_identity_for_user(self, user_id, provider):
            assert provider == "max"
            return {"subject": "1122334455", "user_id": user_id}

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_failed(self, notification_id):
            self.failed.append(notification_id)

    class MaxBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, user_id, text, **kwargs):
            self.sent.append((user_id, text, kwargs))
            return {"ok": True, "status": 200, "data": {"body": {"mid": "m1"}}}

    class TelegramBot:
        async def send_message(self, **kwargs):
            raise AssertionError("MAX user must not be sent to Telegram")

    db = DB()
    max_bot = MaxBot()
    await main._deliver_notification(
        TelegramBot(),
        db,
        "https://example.com/game",
        {
            "id": 77,
            "user_id": 8_000_000_000_000_777,
            "category": "generator",
            "event_type": "generator_new_key",
            "payload": {"keys": 1},
            "attempts": 1,
        },
        max_bot=max_bot,
    )

    assert db.sent == [77]
    assert db.failed == []
    assert max_bot.sent[0][0] == "1122334455"
    assert max_bot.sent[0][2] == {"open_app": True, "text_format": "html"}
