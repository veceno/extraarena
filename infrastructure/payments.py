"""Модуль для работы с YooKassa API."""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from yookassa import Configuration, Payment
    from yookassa.domain.notification import WebhookNotificationFactory
    YOOKASSA_AVAILABLE = True
except ImportError:
    YOOKASSA_AVAILABLE = False
    logging.warning("YooKassa SDK не установлен. Установите: pip install yookassa")

from infrastructure.config import YooKassaSettings

logger = logging.getLogger(__name__)


class PaymentService:
    """Сервис для работы с платежами YooKassa."""

    def __init__(self, settings: YooKassaSettings) -> None:
        if not YOOKASSA_AVAILABLE:
            raise ImportError("YooKassa SDK не установлен. Установите: pip install yookassa")
        
        self.settings = settings
        Configuration.account_id = settings.shop_id
        Configuration.secret_key = settings.secret_key
        self.test_mode = settings.test_mode
        
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
    ) -> dict[str, Any]:
        """
        Создать платеж в YooKassa.
        
        Args:
            amount: Сумма платежа
            currency: Валюта (по умолчанию RUB)
            description: Описание платежа
            return_url: URL для возврата после оплаты
            metadata: Дополнительные метаданные
            
        Returns:
            dict с результатом создания платежа
        """
        try:
            import uuid
            import json
            # Генерируем уникальный ключ идемпотентности для предотвращения дублирования платежей
            idempotence_key = str(uuid.uuid4())
            
            logger.info(f"Создание платежа в YooKassa: amount={amount}, currency={currency}, description={description[:50]}...")
            
            payment = Payment.create({
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
            }, idempotence_key)
            
            # Получаем данные платежа
            # В YooKassa SDK объект Payment имеет атрибуты напрямую
            # Используем dict() или обращаемся к атрибутам
            try:
                # Пробуем получить как словарь через json()
                payment_dict = payment.json()
                # Если это строка, парсим
                if isinstance(payment_dict, str):
                    payment_dict = json.loads(payment_dict)
            except (AttributeError, TypeError):
                # Если json() не работает, используем атрибуты объекта напрямую
                payment_dict = {
                    "id": getattr(payment, "id", None),
                    "status": getattr(payment, "status", None),
                    "amount": {
                        "value": getattr(payment.amount, "value", None) if hasattr(payment, "amount") else None,
                        "currency": getattr(payment.amount, "currency", None) if hasattr(payment, "amount") else None,
                    },
                    "confirmation": {
                        "confirmation_url": getattr(payment.confirmation, "confirmation_url", None) if hasattr(payment, "confirmation") else None,
                    } if hasattr(payment, "confirmation") else {}
                }
            
            # Логируем для отладки
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

    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        """
        Получить статус платежа.
        
        Args:
            payment_id: ID платежа в YooKassa
            
        Returns:
            dict с информацией о платеже
        """
        try:
            payment = Payment.find_one(payment_id)
            
            # Получаем данные платежа
            # payment.json() может возвращать строку или dict в зависимости от версии SDK
            payment_data = payment.json()
            
            # Если это строка, парсим её
            if isinstance(payment_data, str):
                import json
                payment_dict = json.loads(payment_data)
            else:
                payment_dict = payment_data
            
            # Если payment_dict все еще не словарь, пробуем получить атрибуты напрямую
            if not isinstance(payment_dict, dict):
                payment_dict = {
                    "id": getattr(payment, "id", payment_id),
                    "status": getattr(payment, "status", "unknown"),
                    "paid": getattr(payment, "paid", False),
                    "amount": {
                        "value": getattr(payment.amount, "value", None) if hasattr(payment, "amount") else None,
                        "currency": getattr(payment.amount, "currency", "RUB") if hasattr(payment, "amount") else "RUB"
                    } if hasattr(payment, "amount") else {"value": None, "currency": "RUB"},
                    "metadata": getattr(payment, "metadata", {}) if hasattr(payment, "metadata") else {}
                }
            
            return {
                "success": True,
                "payment_id": payment_dict.get("id"),
                "status": payment_dict.get("status"),
                "paid": payment_dict.get("paid", False),
                "amount": payment_dict.get("amount", {}).get("value") if isinstance(payment_dict.get("amount"), dict) else None,
                "currency": payment_dict.get("amount", {}).get("currency") if isinstance(payment_dict.get("amount"), dict) else "RUB",
                "metadata": payment_dict.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"Ошибка получения статуса платежа {payment_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def parse_webhook(self, request_body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Распарсить вебхук от YooKassa.
        
        Args:
            request_body: Тело запроса от YooKassa
            
        Returns:
            dict с данными уведомления или None при ошибке
        """
        try:
            if not YOOKASSA_AVAILABLE:
                logger.error("YooKassa SDK не доступен для парсинга вебхука")
                return None
                
            notification = WebhookNotificationFactory().create(request_body)
            notification_object = notification.object
            
            return {
                "event": notification.event,
                "payment_id": notification_object.id if hasattr(notification_object, 'id') else None,
                "status": notification_object.status if hasattr(notification_object, 'status') else None,
                "paid": notification_object.paid if hasattr(notification_object, 'paid') else False,
                "amount": float(notification_object.amount.value) if hasattr(notification_object, 'amount') and hasattr(notification_object.amount, 'value') else None,
                "currency": notification_object.amount.currency if hasattr(notification_object, 'amount') and hasattr(notification_object.amount, 'currency') else None,
                "metadata": notification_object.metadata if hasattr(notification_object, 'metadata') else {},
            }
        except Exception as e:
            logger.error(f"Ошибка парсинга вебхука YooKassa: {e}", exc_info=True)
            return None

