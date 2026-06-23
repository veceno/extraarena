"""Модуль для работы с YooKassa API."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

import requests

from infrastructure.config import YooKassaSettings

logger = logging.getLogger(__name__)


class PaymentService:
    """Сервис для работы с платежами YooKassa."""

    def __init__(self, settings: YooKassaSettings) -> None:
        self.settings = settings
        self.shop_id = settings.shop_id
        self.secret_key = settings.secret_key
        self.test_mode = settings.test_mode
        self.api_base = "https://api.yookassa.ru/v3"

        if self.test_mode:
            logger.info("YooKassa работает в тестовом режиме")

    def create_payment(
        self,
        *,
        amount: float,
        currency: str = "RUB",
        description: str,
        return_url: str,
        metadata: Optional[dict[str, Any]] = None,
        idempotence_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Создать платеж в YooKassa.
        
        Args:
            amount: Сумма платежа
            currency: Валюта (по умолчанию RUB)
            description: Описание платежа
            return_url: URL для возврата после оплаты
            metadata: Дополнительные метаданные
            idempotence_key: Внешний ключ идемпотентности; если не задан — генерируется UUID
        """
        try:
            # Используем переданный ключ идемпотентности или генерируем UUID
            _idempotence_key = idempotence_key or str(uuid.uuid4())
            
            logger.info(f"Создание платежа в YooKassa: amount={amount}, currency={currency}, description={description[:50]}...")

            payload = {
                "amount": {
                    "value": f"{amount:.2f}",
                    "currency": currency
                },
                "confirmation": {
                    "type": "redirect",
                    "return_url": return_url
                },
                "capture": True,
                "description": description,
                "metadata": metadata or {}
            }

            response = requests.post(
                "https://api.yookassa.ru/v3/payments",
                auth=(self.settings.shop_id, self.settings.secret_key),
                headers={
                    "Idempotence-Key": _idempotence_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(5, 15),
            )

            try:
                payment_dict = response.json()
            except json.JSONDecodeError:
                payment_dict = {}

            if response.status_code >= 400:
                error_msg = (
                    payment_dict.get("description")
                    or payment_dict.get("message")
                    or response.text
                    or f"HTTP {response.status_code}"
                )
                logger.error("YooKassa API error %s: %s", response.status_code, error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": f"HTTP_{response.status_code}",
                }

            return self._format_payment_result(payment_dict)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Ошибка создания платежа в YooKassa: {error_type}: {error_msg}", exc_info=True)
            
            # Более детальная информация об ошибке
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    logger.error(f"Детали ошибки от YooKassa: {error_data}")
                    if isinstance(error_data, dict) and 'description' in error_data:
                        error_msg = error_data.get('description', error_msg)
                except:
                    pass
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": error_type
            }

    def _format_payment_result(self, payment_dict: dict[str, Any]) -> dict[str, Any]:

        logger.info(f"Платеж создан успешно: payment_id={payment_dict.get('id')}, status={payment_dict.get('status')}")

        confirmation_url = payment_dict.get("confirmation", {}).get("confirmation_url")
        if not confirmation_url:
            logger.warning(f"confirmation_url отсутствует в ответе YooKassa. Полный ответ: {json.dumps(payment_dict, indent=2)}")

        return {
            "success": True,
            "payment_id": payment_dict.get("id"),
            "confirmation_url": confirmation_url,
            "status": payment_dict.get("status"),
            "amount": payment_dict.get("amount", {}).get("value"),
            "currency": payment_dict.get("amount", {}).get("currency"),
        }

    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        """
        Получить статус платежа напрямую через YooKassa API (без SDK).

        Args:
            payment_id: ID платежа в YooKassa

        Returns:
            dict с информацией о платеже
        """
        url = f"{self.api_base}/payments/{payment_id}"
        try:
            response = requests.get(
                url,
                auth=(self.shop_id, self.secret_key),
                headers={"Content-Type": "application/json"},
                timeout=(5, 15),
            )

            try:
                payment_dict = response.json()
            except json.JSONDecodeError:
                payment_dict = {}

            if response.status_code >= 400:
                err = payment_dict.get("description") or payment_dict.get("message") or str(response.status_code)
                logger.error("YooKassa get status error %s: %s", response.status_code, err)
                return {"success": False, "error": err, "error_type": f"HTTP_{response.status_code}"}

            return self._format_status_result(payment_dict)
        except Exception as e:
            logger.error("Ошибка получения статуса платежа %s: %s", payment_id, e, exc_info=True)
            return {"success": False, "error": str(e)}

    def _format_status_result(self, payment_dict: dict[str, Any]) -> dict[str, Any]:
        amount = payment_dict.get("amount") or {}
        return {
            "success": True,
            "payment_id": payment_dict.get("id"),
            "status": payment_dict.get("status"),
            "paid": payment_dict.get("paid", False),
            "amount": amount.get("value"),
            "currency": amount.get("currency", "RUB"),
            "metadata": payment_dict.get("metadata", {}),
        }

    def parse_webhook(self, request_body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Распарсить вебхук от YooKassa (прямой JSON-парсинг, без SDK).

        Args:
            request_body: Тело запроса от YooKassa

        Returns:
            dict с данными уведомления или None при ошибке
        """
        try:
            if not isinstance(request_body, dict):
                logger.error("parse_webhook: тело запроса не dict, тип=%s", type(request_body))
                return None

            event = request_body.get("event")
            obj = request_body.get("object")
            if not isinstance(obj, dict):
                logger.error("parse_webhook: отсутствует объект 'object' в теле webhook, keys=%s", list(request_body.keys())[:10])
                return None

            payment_id = obj.get("id")
            status = obj.get("status")
            paid = obj.get("paid", False)
            amount_data = obj.get("amount") or {}
            metadata = obj.get("metadata") or {}

            result: dict[str, Any] = {
                "event": event,
                "payment_id": payment_id,
                "status": status,
                "paid": bool(paid),
                "amount": float(amount_data.get("value", 0)) if amount_data.get("value") else 0.0,
                "currency": amount_data.get("currency", "RUB"),
                "metadata": metadata,
            }

            logger.info(
                "parse_webhook: event=%s payment_id=%s status=%s paid=%s metadata_keys=%s",
                event, payment_id, status, paid, list(metadata.keys())[:10] if isinstance(metadata, dict) else "not_dict",
            )
            return result
        except Exception as e:
            logger.error("Ошибка парсинга вебхука YooKassa: %s, raw_body=%s", e,
                         json.dumps(request_body, ensure_ascii=False, default=str)[:1000], exc_info=True)
            return None
