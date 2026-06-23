from __future__ import annotations

import os
import socket
from functools import lru_cache
from urllib.parse import urlsplit

import aiohttp
import aiogram
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError


DEFAULT_HAPP_PROXY_URL = "http://127.0.0.1:10808"


def _is_disabled(value: str) -> bool:
    return value.strip().lower() in {"", "0", "false", "no", "off", "none", "direct"}


def _proxy_reachable(proxy_url: str) -> bool:
    parsed = urlsplit(proxy_url)
    if parsed.scheme not in {"http", "https", "socks4", "socks5"}:
        return False
    if not parsed.hostname:
        return False
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    try:
        with socket.create_connection((parsed.hostname, port), timeout=0.25):
            return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def get_telegram_proxy_url() -> str | None:
    explicit = os.getenv("TELEGRAM_PROXY_URL")
    if explicit is not None:
        explicit = explicit.strip()
        return None if _is_disabled(explicit) else explicit

    happ_proxy = os.getenv("HAPP_PROXY_URL", DEFAULT_HAPP_PROXY_URL).strip()
    if happ_proxy and _proxy_reachable(happ_proxy):
        return happ_proxy
    return None


class TelegramProxyAiohttpSession(AiohttpSession):
    def __init__(self, *, proxy: str):
        super().__init__()
        self._telegram_proxy = proxy

    async def make_request(self, bot, method, timeout=None):
        session = await self.create_session()
        url = self.api.api_url(token=bot.token, method=method.__api_method__)
        form = self.build_form_data(bot=bot, method=method)

        try:
            async with session.post(
                url,
                data=form,
                timeout=self.timeout if timeout is None else timeout,
                proxy=self._telegram_proxy,
            ) as resp:
                raw_result = await resp.text()
        except TimeoutError:
            raise TelegramNetworkError(method=method, message="Request timeout error")
        except aiohttp.ClientError as exc:
            raise TelegramNetworkError(method=method, message=f"{type(exc).__name__}: {exc}")

        response = self.check_response(
            bot=bot,
            method=method,
            status_code=resp.status,
            content=raw_result,
        )
        return response.result

    async def stream_content(
        self,
        url: str,
        headers=None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ):
        if headers is None:
            headers = {}
        session = await self.create_session()
        async with session.get(
            url,
            timeout=timeout,
            headers=headers,
            raise_for_status=raise_for_status,
            proxy=self._telegram_proxy,
        ) as resp:
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk


def create_telegram_aiogram_session() -> AiohttpSession:
    proxy = get_telegram_proxy_url()
    return TelegramProxyAiohttpSession(proxy=proxy) if proxy else AiohttpSession()


def create_telegram_bot(token: str, **kwargs) -> aiogram.Bot:
    return aiogram.Bot(token=token, session=create_telegram_aiogram_session(), **kwargs)


def _is_telegram_url(url: object) -> bool:
    hostname = urlsplit(str(url)).hostname or ""
    return hostname == "telegram.org" or hostname.endswith(".telegram.org")


def _telegram_proxy_kwargs(url: object, proxy: str | None, kwargs: dict) -> dict:
    if proxy and "proxy" not in kwargs and _is_telegram_url(url):
        kwargs = dict(kwargs)
        kwargs["proxy"] = proxy
    return kwargs


class TelegramAiohttpSession:
    def __init__(self, *, insecure_ssl: bool = False, **session_kwargs):
        connector = None
        if insecure_ssl:
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)
        if connector is not None and "connector" not in session_kwargs:
            session_kwargs["connector"] = connector
        self._session = aiohttp.ClientSession(**session_kwargs)
        self._proxy = get_telegram_proxy_url()

    async def __aenter__(self):
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._session.__aexit__(exc_type, exc, tb)

    def get(self, url, *args, **kwargs):
        kwargs = _telegram_proxy_kwargs(url, self._proxy, kwargs)
        return self._session.get(url, *args, **kwargs)

    def post(self, url, *args, **kwargs):
        kwargs = _telegram_proxy_kwargs(url, self._proxy, kwargs)
        return self._session.post(url, *args, **kwargs)


def create_telegram_aiohttp_session(*, insecure_ssl: bool = False, **session_kwargs) -> TelegramAiohttpSession:
    return TelegramAiohttpSession(insecure_ssl=insecure_ssl, **session_kwargs)
