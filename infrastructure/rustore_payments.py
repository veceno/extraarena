"""RuStore Pay Public API helpers.

The Android Pay SDK creates the purchase UI and returns invoice identifiers.
Rewards are granted only after this module verifies the invoice through the
RuStore Public API and checks it against the payment row created by our server.
"""

from __future__ import annotations

import json
import logging
import base64
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

RUSTORE_PUBLIC_API_BASE = "https://public-api.rustore.ru"
RUSTORE_AUTH_SKEW_SECONDS = 60
RUSTORE_CONFIRMED_STATUS = "CONFIRMED"
RUSTORE_PENDING_STATUSES = {"CREATED", "EXECUTED", "PAID", "REFUNDING"}
RUSTORE_CANCELED_STATUSES = {"CANCELLED", "REJECTED", "EXPIRED", "REVERSED", "REFUNDED"}


@dataclass(frozen=True)
class RuStorePaymentSettings:
    public_token: str = ""
    console_app_id: str = ""
    sandbox: bool = False
    api_base: str = RUSTORE_PUBLIC_API_BASE
    key_id: str = ""
    private_key: str = ""
    private_key_file: str = ""


def resolve_rustore_product_id(product: dict[str, Any] | None) -> str:
    """Return the RuStore Console product id for a ruble product row."""
    if not product:
        return ""
    metadata = product.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        override = str(metadata.get("rustore_product_id") or "").strip()
        if override:
            return override
    return str(product.get("code") or "").strip()


def _kopeks(amount_rub: float | int | str) -> int:
    return int(round(float(amount_rub or 0) * 100))


class RuStorePublicTokenProvider:
    """Fetch and cache short-lived RuStore Public API JWE tokens."""

    def __init__(
        self,
        settings: RuStorePaymentSettings,
        *,
        http_post: Callable[..., Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.http_post = http_post or requests.post
        self.now = now or time.time
        self._cached_token = ""
        self._expires_at = 0.0
        self.last_ttl = 0

    def get_token(self) -> str:
        static_token = str(self.settings.public_token or "").strip()
        if static_token:
            return static_token

        current_time = self.now()
        if self._cached_token and current_time < self._expires_at - RUSTORE_AUTH_SKEW_SECONDS:
            return self._cached_token

        key_id = str(getattr(self.settings, "key_id", "") or "").strip()
        private_key = self._read_private_key()
        if not key_id or not private_key:
            return ""

        payload = self._build_auth_payload(key_id=key_id, private_key=private_key)
        url = self.settings.api_base.rstrip("/") + "/public/auth"
        response = self.http_post(
            url,
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=(5, 15),
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code < 200 or status_code >= 300:
            text = str(getattr(response, "text", ""))
            raise RuntimeError(f"RuStore auth HTTP {status_code}: {text[:200]}")

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("RuStore auth returned non-object JSON")
        code = str(data.get("code") or "").upper()
        if code not in {"OK", "ОК"}:
            raise RuntimeError(str(data.get("message") or code or "RuStore auth failed"))

        body = data.get("body") or {}
        if not isinstance(body, dict):
            raise RuntimeError("RuStore auth response body is empty")
        token = str(body.get("jwe") or "").strip()
        if not token:
            raise RuntimeError("RuStore auth response does not contain jwe")

        try:
            ttl = int(body.get("ttl") or 900)
        except (TypeError, ValueError):
            ttl = 900
        self._cached_token = token
        self.last_ttl = ttl
        self._expires_at = current_time + ttl
        return token

    def _build_auth_payload(self, *, key_id: str, private_key: str) -> dict[str, str]:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        message = f"{key_id}{timestamp}".encode("utf-8")
        signature = self._sign(private_key, message)
        return {
            "keyId": key_id,
            "timestamp": timestamp,
            "signature": signature,
        }

    def _read_private_key(self) -> str:
        inline_key = str(getattr(self.settings, "private_key", "") or "").strip()
        if inline_key:
            return inline_key
        key_file = str(getattr(self.settings, "private_key_file", "") or "").strip()
        if not key_file:
            return ""
        return Path(key_file).read_text(encoding="utf-8").strip()

    def _sign(self, private_key: str, message: bytes) -> str:
        normalized = private_key.strip().replace("\\n", "\n")
        if "BEGIN" in normalized:
            key = serialization.load_pem_private_key(normalized.encode("utf-8"), password=None)
        else:
            key_der = base64.b64decode("".join(normalized.split()), validate=True)
            key = serialization.load_der_private_key(key_der, password=None)
        signature = key.sign(message, padding.PKCS1v15(), hashes.SHA512())
        return base64.b64encode(signature).decode("ascii")


class RuStoreInvoiceVerifier:
    def __init__(
        self,
        settings: RuStorePaymentSettings,
        *,
        http_get: Callable[..., Any] | None = None,
        http_post: Callable[..., Any] | None = None,
        token_provider: RuStorePublicTokenProvider | None = None,
    ) -> None:
        self.settings = settings
        self.http_get = http_get or requests.get
        self.token_provider = token_provider or RuStorePublicTokenProvider(settings, http_post=http_post)

    def verify_invoice(
        self,
        *,
        invoice_id: str,
        expected_payment_id: str,
        expected_amount_rub: float,
        expected_currency: str,
        expected_product_id: str,
    ) -> dict[str, Any]:
        """Verify a RuStore invoice and map it to our payment status vocabulary."""
        try:
            public_token = self.token_provider.get_token()
        except Exception as exc:
            logger.warning("RuStore Public API auth failed: %s", exc, exc_info=True)
            return self._error("public_token_request_failed", str(exc))
        if not public_token:
            return self._error("missing_public_token", "RuStore Public API token/auth key is not configured")
        if not str(invoice_id or "").strip():
            return self._error("missing_invoice_id", "invoice_id is required")

        try:
            payload = self._fetch_invoice(str(invoice_id).strip(), public_token)
        except Exception as exc:
            logger.warning("RuStore invoice fetch failed for %s: %s", invoice_id, exc, exc_info=True)
            return self._error("provider_request_failed", str(exc))

        code = str(payload.get("code") or "").upper()
        if code not in {"OK", "ОК"}:
            return self._error("provider_error", payload.get("message") or code or "unknown")

        body = payload.get("body") or {}
        if not isinstance(body, dict):
            return self._error("missing_invoice_body", "RuStore response body is empty")

        order = body.get("order") or {}
        if not isinstance(order, dict):
            order = {}

        app_id = str(body.get("appId") or "").strip()
        expected_app_id = str(self.settings.console_app_id or "").strip()
        if expected_app_id and expected_app_id != "0" and app_id and app_id != expected_app_id:
            return self._error("app_id_mismatch", f"expected {expected_app_id}, got {app_id}")

        order_id = str(order.get("orderId") or "").strip()
        if order_id and order_id != str(expected_payment_id):
            return self._error("order_id_mismatch", f"expected {expected_payment_id}, got {order_id}")

        item_code = str(order.get("itemCode") or "").strip()
        if item_code and item_code != str(expected_product_id):
            return self._error("product_id_mismatch", f"expected {expected_product_id}, got {item_code}")

        currency = str(order.get("currency") or "").upper()
        expected_currency = str(expected_currency or "RUB").upper()
        if currency and currency != expected_currency:
            return self._error("currency_mismatch", f"expected {expected_currency}, got {currency}")

        provider_amount = order.get("amountCurrent", order.get("amountCreate"))
        if provider_amount is not None:
            try:
                amount_kopeks = int(round(float(provider_amount)))
            except (TypeError, ValueError):
                return self._error("invalid_amount", f"invalid amountCurrent: {provider_amount}")
            expected_kopeks = _kopeks(expected_amount_rub)
            if amount_kopeks != expected_kopeks:
                return self._error("amount_mismatch", f"expected {expected_kopeks}, got {amount_kopeks}")

        invoice_status = str(body.get("invoiceStatus") or "").upper()
        mapped_status = self._map_invoice_status(invoice_status)
        if mapped_status == "unknown":
            return self._error("unknown_invoice_status", invoice_status or "empty")

        return {
            "success": True,
            "paid": mapped_status == "succeeded",
            "status": mapped_status,
            "invoice_status": invoice_status,
            "invoice_id": str(body.get("invoiceId") or invoice_id),
            "purchase_id": str(body.get("purchaseId") or ""),
            "order_id": order_id,
            "product_id": item_code,
            "amount_kopeks": int(round(float(provider_amount))) if provider_amount is not None else None,
            "currency": currency or expected_currency,
            "sandbox": bool(self.settings.sandbox),
            "raw": payload,
        }

    def _fetch_invoice(self, invoice_id: str, public_token: str) -> dict[str, Any]:
        path = "/public/sandbox/v2/invoices/" if self.settings.sandbox else "/public/v2/invoices/"
        url = self.settings.api_base.rstrip("/") + path + invoice_id
        response = self.http_get(
            url,
            headers={
                "Accept": "application/json",
                "Public-Token": public_token,
            },
            timeout=(5, 15),
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code < 200 or status_code >= 300:
            text = str(getattr(response, "text", ""))
            raise RuntimeError(f"RuStore API HTTP {status_code}: {text[:200]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("RuStore API returned non-object JSON")
        return data

    def _map_invoice_status(self, invoice_status: str) -> str:
        if invoice_status == RUSTORE_CONFIRMED_STATUS:
            return "succeeded"
        if invoice_status in RUSTORE_PENDING_STATUSES:
            return "pending"
        if invoice_status in RUSTORE_CANCELED_STATUSES:
            return "canceled"
        return "unknown"

    def _error(self, reason: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "paid": False,
            "status": "verification_failed",
            "reason": reason,
            "message": message,
        }
