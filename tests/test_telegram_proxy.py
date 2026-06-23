from __future__ import annotations

from infrastructure import telegram_proxy


def _reset_proxy_cache() -> None:
    telegram_proxy.get_telegram_proxy_url.cache_clear()


def test_telegram_proxy_uses_reachable_happ_by_default(monkeypatch):
    monkeypatch.delenv("TELEGRAM_PROXY_URL", raising=False)
    monkeypatch.delenv("HAPP_PROXY_URL", raising=False)
    monkeypatch.setattr(telegram_proxy, "_proxy_reachable", lambda url: True)
    _reset_proxy_cache()

    assert telegram_proxy.get_telegram_proxy_url() == telegram_proxy.DEFAULT_HAPP_PROXY_URL


def test_telegram_proxy_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("TELEGRAM_PROXY_URL", "direct")
    monkeypatch.setattr(telegram_proxy, "_proxy_reachable", lambda url: True)
    _reset_proxy_cache()

    assert telegram_proxy.get_telegram_proxy_url() is None


def test_telegram_proxy_kwargs_only_apply_to_telegram_urls():
    proxy = "http://127.0.0.1:10808"

    assert telegram_proxy._telegram_proxy_kwargs("https://api.telegram.org/bot/x", proxy, {}) == {
        "proxy": proxy,
    }
    assert telegram_proxy._telegram_proxy_kwargs("https://example.com/api", proxy, {}) == {}


def test_support_bot_uses_telegram_proxy_factory(monkeypatch):
    from support.channels import telegram as support_telegram

    calls = []

    class FakeDispatcher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def include_router(self, router):
            self.router = router

    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def message(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def callback_query(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    monkeypatch.setattr(support_telegram, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(support_telegram, "Router", FakeRouter)

    def fake_create_telegram_bot(token, **kwargs):
        calls.append((token, kwargs))
        return object()

    monkeypatch.setattr(support_telegram, "create_telegram_bot", fake_create_telegram_bot)

    bot, dispatcher = support_telegram.create_support_bot("123:token")

    assert bot is not None
    assert isinstance(dispatcher, FakeDispatcher)
    assert calls and calls[0][0] == "123:token"
