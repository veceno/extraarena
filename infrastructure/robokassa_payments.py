"""Robokassa payment form and ResultURL helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobokassaSettings:
    merchant_login: str
    password1: str
    password2: str
    password3: str = ""
    hash_algo: str = "sha256"
    test_mode: bool = False
    result_url: str = ""
    success_url: str = ""
    fail_url: str = ""
    payment_url: str = "https://auth.robokassa.ru/Merchant/Index.aspx"
    payment_object: str = "service"


class RobokassaPaymentService:
    """Create signed Robokassa payment forms and verify ResultURL callbacks."""

    def __init__(self, settings: RobokassaSettings) -> None:
        self.settings = settings
        self.hash_algo = self._normalize_hash_algo(settings.hash_algo)

    @staticmethod
    def _normalize_hash_algo(value: str) -> str:
        algo = (value or "sha256").strip().lower().replace("-", "")
        if algo == "sha256":
            return "sha256"
        if algo == "md5":
            return "md5"
        raise ValueError(f"Unsupported Robokassa hash algorithm: {value}")

    def _hash(self, value: str) -> str:
        digest = hashlib.new(self.hash_algo)
        digest.update(value.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _amount_string(amount: float | int | str) -> str:
        return f"{float(amount):.2f}"

    @staticmethod
    def _invoice_id(inv_id: int | str | None = None) -> str:
        if inv_id is not None:
            raw = str(inv_id).strip()
            if raw:
                return raw
        return str(uuid.uuid4().int % 10_000_000_000)

    def _password1(self) -> str:
        return self.settings.password1

    def _password2(self) -> str:
        return self.settings.password2

    def _shp_signature_tail(self, shp_params: dict[str, Any]) -> str:
        if not shp_params:
            return ""
        parts = []
        for key in sorted(shp_params):
            if not key.startswith("Shp_"):
                continue
            parts.append(f"{key}={shp_params[key]}")
        return "".join(f":{part}" for part in parts)

    def _receipt(self, *, name: str, amount: float) -> str:
        item_name = str(name or "Покупка ExtraArena").strip()[:128] or "Покупка ExtraArena"
        receipt = {
            "items": [
                {
                    "name": item_name,
                    "quantity": 1,
                    "sum": round(float(amount), 2),
                    "tax": "none",
                    "payment_method": "full_payment",
                    "payment_object": self.settings.payment_object or "service",
                }
            ]
        }
        receipt_json = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"))
        return quote(receipt_json, safe="")

    def _payment_signature(
        self,
        *,
        out_sum: str,
        inv_id: str,
        receipt: str,
        success_url2: str = "",
        success_url2_method: str = "",
        fail_url2: str = "",
        fail_url2_method: str = "",
        shp_params: dict[str, Any],
    ) -> str:
        modifiers = [receipt]
        if success_url2:
            modifiers.extend([success_url2, success_url2_method or "GET"])
        if fail_url2:
            modifiers.extend([fail_url2, fail_url2_method or "GET"])
        base = ":".join(
            [
                self.settings.merchant_login,
                out_sum,
                inv_id,
                *modifiers,
                self._password1(),
            ]
        )
        base = f"{base}{self._shp_signature_tail(shp_params)}"
        return self._hash(base)

    def _result_signature(self, *, out_sum: str, inv_id: str, shp_params: dict[str, Any]) -> str:
        base = f"{out_sum}:{inv_id}:{self._password2()}{self._shp_signature_tail(shp_params)}"
        return self._hash(base)

    def _success_signature(self, *, out_sum: str, inv_id: str, shp_params: dict[str, Any]) -> str:
        base = f"{out_sum}:{inv_id}:{self._password1()}{self._shp_signature_tail(shp_params)}"
        return self._hash(base)

    def create_payment(
        self,
        *,
        amount: float,
        currency: str = "RUB",
        description: str,
        return_url: str,
        metadata: dict[str, Any] | None = None,
        idempotence_key: str | None = None,
        inv_id: int | str | None = None,
    ) -> dict[str, Any]:
        if str(currency or "RUB").upper() != "RUB":
            return {"success": False, "error": "robokassa_supports_rub_only"}

        invoice_id = self._invoice_id(inv_id)
        payment_id = f"robokassa_{invoice_id}"
        out_sum = self._amount_string(amount)
        meta = dict(metadata or {})
        item_name = str(meta.get("item_name") or description or "Покупка ExtraArena")
        receipt = self._receipt(name=item_name, amount=float(amount))
        shp_params = {"Shp_payment_id": payment_id}
        success_url2 = self.settings.success_url.strip()
        fail_url2 = self.settings.fail_url.strip()
        success_url2_method = "GET" if success_url2 else ""
        fail_url2_method = "GET" if fail_url2 else ""

        signature = self._payment_signature(
            out_sum=out_sum,
            inv_id=invoice_id,
            receipt=receipt,
            success_url2=success_url2,
            success_url2_method=success_url2_method,
            fail_url2=fail_url2,
            fail_url2_method=fail_url2_method,
            shp_params=shp_params,
        )

        form: dict[str, str] = {
            "MerchantLogin": self.settings.merchant_login,
            "OutSum": out_sum,
            "InvId": invoice_id,
            "Description": str(description or item_name)[:100],
            "SignatureValue": signature,
            "Receipt": receipt,
            "Culture": "ru",
            "Encoding": "utf-8",
            "Shp_payment_id": payment_id,
        }
        if self.settings.test_mode:
            form["IsTest"] = "1"
        if meta.get("buyer_email"):
            form["Email"] = str(meta["buyer_email"])[:128]
        if success_url2:
            form["SuccessUrl2"] = success_url2
            form["SuccessUrl2Method"] = success_url2_method
        if fail_url2:
            form["FailUrl2"] = fail_url2
            form["FailUrl2Method"] = fail_url2_method

        payment_url = f"{self.settings.payment_url}?{urlencode(form)}"
        return {
            "success": True,
            "payment_id": payment_id,
            "provider": "robokassa",
            "confirmation_url": payment_url,
            "payment_url": payment_url,
            "status": "pending",
            "amount": out_sum,
            "currency": "RUB",
            "form": form,
            "form_action": self.settings.payment_url,
            "robokassa_inv_id": invoice_id,
            "test_mode": self.settings.test_mode,
        }

    def parse_result_notification(self, payload: dict[str, Any]) -> dict[str, Any]:
        out_sum = str(payload.get("OutSum") or payload.get("out_sum") or "").strip()
        inv_id = str(payload.get("InvId") or payload.get("InvID") or payload.get("inv_id") or "").strip()
        signature = str(payload.get("SignatureValue") or "").strip()
        if not out_sum or not inv_id or not signature:
            return {"success": False, "error": "missing_required_fields"}

        shp_params = {
            str(key): str(value)
            for key, value in payload.items()
            if str(key).startswith("Shp_")
        }
        expected = self._result_signature(out_sum=out_sum, inv_id=inv_id, shp_params=shp_params)
        if not hmac.compare_digest(expected.lower(), signature.lower()):
            logger.warning("Robokassa ResultURL signature mismatch for InvId=%s", inv_id)
            return {"success": False, "error": "invalid_signature"}

        payment_id = shp_params.get("Shp_payment_id") or f"robokassa_{inv_id}"
        try:
            amount = float(out_sum)
        except (TypeError, ValueError):
            return {"success": False, "error": "invalid_amount"}

        return {
            "success": True,
            "payment_id": payment_id,
            "inv_id": inv_id,
            "amount": amount,
            "currency": "RUB",
            "status": "succeeded",
            "paid": True,
        }

    def parse_success_redirect(self, payload: dict[str, Any]) -> dict[str, Any]:
        out_sum = str(payload.get("OutSum") or payload.get("out_sum") or "").strip()
        inv_id = str(payload.get("InvId") or payload.get("InvID") or payload.get("inv_id") or "").strip()
        signature = str(payload.get("SignatureValue") or "").strip()
        if not out_sum or not inv_id or not signature:
            return {"success": False, "error": "missing_required_fields"}

        shp_params = {
            str(key): str(value)
            for key, value in payload.items()
            if str(key).startswith("Shp_")
        }
        expected = self._success_signature(out_sum=out_sum, inv_id=inv_id, shp_params=shp_params)
        if not hmac.compare_digest(expected.lower(), signature.lower()):
            logger.warning("Robokassa SuccessURL signature mismatch for InvId=%s", inv_id)
            return {"success": False, "error": "invalid_signature"}

        payment_id = shp_params.get("Shp_payment_id") or f"robokassa_{inv_id}"
        try:
            amount = float(out_sum)
        except (TypeError, ValueError):
            return {"success": False, "error": "invalid_amount"}

        return {
            "success": True,
            "payment_id": payment_id,
            "inv_id": inv_id,
            "amount": amount,
            "currency": "RUB",
            "status": "succeeded",
            "paid": True,
        }
