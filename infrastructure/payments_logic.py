"""Дополнительные утилиты для универсальной обработки успешных платежей.

В этом модуле сосредоточена логика выдачи наград, создания писем и фиксации
статуса платежа, чтобы один и тот же код использовали и Telegram Stars, и YooKassa.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # Импортируем только для подсказок типов, чтобы избежать циклов
    from infrastructure.database import Database


async def process_successful_payment(
    db: "Database",
    *,
    payment_id: str,
    payment_record: Dict[str, Any],
    source: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Унифицированная обработка подтвержденного платежа.

    Args:
        db: Подключение к БД.
        payment_id: Уникальный идентификатор платежа.
        payment_record: Строка из таблицы payments (уже считанная снаружи).
        source: Текстовое описание источника ("telegram_stars", "yookassa_webhook").
        logger: Логгер контекста вызова.

    Returns:
        Подробный словарь: статус обработки, текст наград, сформированные вложения и т.д.
    """
    result: Dict[str, Any] = {
        "status": "unknown",
        "rewards_text": [],
        "attachments": {},
        "mail_created": False,
        "warnings": [],
    }

    if not payment_record:
        result["status"] = "missing_payment"
        logger.warning("Попытка обработать платеж %s, но записи в БД нет", payment_id)
        return result

    if payment_record.get("rewards_processed"):
        result["status"] = "already_processed"
        logger.info("Платеж %s уже был обработан ранее, пропускаем", payment_id)
        return result

    metadata = _ensure_metadata(payment_record.get("metadata"))
    item_type = _resolve_item_type(payment_record, metadata, logger)

    reward_ctx = await _grant_rewards_for_item(
        db=db,
        user_id=payment_record["user_id"],
        item_type=item_type,
        metadata=metadata,
        logger=logger,
    )

    result.update(
        {
            "rewards_text": reward_ctx["rewards_text"],
            "attachments": reward_ctx["attachments"],
            "item_type": item_type,
        }
    )

    mail_content = None
    if reward_ctx["rewards_text"]:
        mail_content = await _create_purchase_mail(
            db=db,
            user_id=payment_record["user_id"],
            amount=float(payment_record.get("amount") or 0),
            currency=payment_record.get("currency", "RUB"),
            rewards_text=reward_ctx["rewards_text"],
            attachments=reward_ctx["attachments"],
            metadata=metadata,
            source=source,
            logger=logger,
        )
        result["mail_created"] = mail_content is not None

    # Помечаем платеж как обработанный (тот же подход, что и раньше в коде)
    try:
        updated_rows = await db.fetchval(
            """
            UPDATE payments
            SET rewards_processed = TRUE
            WHERE payment_id = $1 AND rewards_processed = FALSE
            RETURNING 1
            """,
            payment_id,
        )
        if updated_rows:
            logger.info("Платеж %s помечен как обработанный", payment_id)
        else:
            logger.warning(
                "Платеж %s не удалось пометить как обработанный (возможно, параллельный процесс успел раньше)",
                payment_id,
            )
    except Exception as mark_err:  # pragma: no cover - защитный блок
        logger.error("Ошибка фиксации статуса платежа %s: %s", payment_id, mark_err, exc_info=True)
        result["warnings"].append(str(mark_err))

    result["status"] = "processed"
    return result


def _ensure_metadata(raw_metadata: Any) -> Dict[str, Any]:
    """Приводим метаданные к словарю, даже если они пришли строкой из БД."""
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        try:
            return json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
    return dict(raw_metadata)


def _resolve_item_type(
    payment_record: Dict[str, Any], 
    metadata: Dict[str, Any],
    logger: logging.Logger
) -> Optional[str]:
    """Пытаемся определить тип товара, чтобы понимать какую награду выдавать."""
    # Логируем для отладки
    logger.info(
        "Определение item_type: metadata.item_type=%s, description=%s, metadata=%s",
        metadata.get("item_type"),
        payment_record.get("description"),
        metadata
    )
    
    if metadata.get("item_type"):
        item_type = str(metadata["item_type"])
        logger.info("Используем item_type из metadata: %s", item_type)
        return item_type
    
    # Иногда описание платежа хранит полезную строку (например, "gems_120").
    description = payment_record.get("description")
    if isinstance(description, str) and description:
        logger.info("Используем description как item_type: %s", description)
        return description
    
    logger.warning(
        "Не удалось определить item_type: metadata=%s, description=%s",
        metadata,
        payment_record.get("description")
    )
    return None


async def _grant_rewards_for_item(
    db: "Database",
    *,
    user_id: int,
    item_type: Optional[str],
    metadata: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Начисляет ресурсы в зависимости от item_type и собирает описание наград."""
    rewards_text: List[str] = []
    attachments: Dict[str, Any] = {}
    rewards_given = False

    def _append_currency(kind: str, amount: int, label: str) -> None:
        """Локальный хэлпер для компактной записи повторяющейся логики."""
        if amount <= 0:
            return
        attachments[kind] = attachments.get(kind, 0) + amount
        rewards_text.append(f"{amount} {label}")

    if item_type == "test_payment":
        gems_amount = int(metadata.get("gems_amount") or 10)
        await db.execute("UPDATE users SET gems = gems + $1 WHERE user_id = $2", gems_amount, user_id)
        _append_currency("gems", gems_amount, "💎 гемов (тест)")
        rewards_given = True

    elif item_type in {"gems", "coins", "keys"}:
        amount_field = f"{item_type}_amount"
        amount_value = int(metadata.get(amount_field) or metadata.get("amount") or 0)
        if amount_value > 0:
            if item_type == "keys":
                # Для кейсов используем специальную функцию, которая также синхронизирует user_cases
                await db.increment_user_keys(user_id, amount_value)
                await db.sync_user_key_cases(user_id)
            else:
                column = {"gems": "gems", "coins": "coins"}[item_type]
                await db.execute(
                    f"UPDATE users SET {column} = {column} + $1 WHERE user_id = $2",
                    amount_value,
                    user_id,
                )
            emoji = {"gems": "💎", "coins": "💰", "keys": "📦"}[item_type]
            label = {"gems": "гемов", "coins": "монет", "keys": "кейсов"}[item_type]
            _append_currency("gems" if item_type == "gems" else item_type, amount_value, f"{emoji} {label}")
            rewards_given = True

    elif item_type and item_type.startswith("keys_"):
        # Покупка кейсов в формате keys_1, keys_3, keys_10 и т.д.
        try:
            keys_amount = int(item_type.split("_")[1])
            if keys_amount > 0:
                await db.increment_user_keys(user_id, keys_amount)
                await db.sync_user_key_cases(user_id)
                _append_currency("keys", keys_amount, f"📦 кейсов")
                rewards_given = True
        except (ValueError, IndexError):
            logger.error("Не удалось извлечь количество кейсов из item_type=%s", item_type)

    elif item_type and item_type.startswith("gems_"):
        try:
            gems_amount = int(item_type.split("_")[1])
            await db.execute("UPDATE users SET gems = gems + $1 WHERE user_id = $2", gems_amount, user_id)
            _append_currency("gems", gems_amount, "💎 гемов")
            rewards_given = True
        except (ValueError, IndexError):
            logger.error("Не удалось извлечь количество гемов из item_type=%s", item_type)

    elif item_type == "extrapass":
        from datetime import datetime, timedelta

        expires_at = datetime.now() + timedelta(days=30)
        await db.execute(
            "UPDATE users SET extra_pass = 'active', extra_pass_expires_at = $1 WHERE user_id = $2",
            expires_at,
            user_id,
        )
        attachments["extrapass"] = True
        rewards_text.append("⭐ ExtraPass (30 дней)")
        rewards_given = True

    elif item_type == "extrapass_ultra":
        from datetime import datetime, timedelta

        expires_at = datetime.now() + timedelta(days=30)
        await db.execute(
            "UPDATE users SET extra_pass = 'active', extra_pass_expires_at = $1 WHERE user_id = $2",
            expires_at,
            user_id,
        )
        # ExtraPass Ultra дает дополнительные бонусы
        await db.execute("UPDATE users SET gems = gems + 500 WHERE user_id = $1", user_id)
        _append_currency("gems", 500, "💎 гемов (бонус Ultra)")
        attachments["extrapass"] = True
        attachments["extrapass_ultra"] = True
        rewards_text.append("💫 ExtraPass Ultra (30 дней)")
        rewards_given = True

    elif item_type == "starter_boost":
        from datetime import datetime, timedelta

        user_profile = await db.get_user_profile(user_id)
        has_extra_pass = user_profile and user_profile.get("extra_pass") == "active"

        if has_extra_pass:
            await db.execute("UPDATE users SET gems = gems + 1200 WHERE user_id = $1", user_id)
            _append_currency("gems", 1200, "💎 гемов (700 бонус за активный ExtraPass + 500)")
        else:
            expires_at = datetime.now() + timedelta(days=30)
            await db.execute(
                "UPDATE users SET extra_pass = 'active', extra_pass_expires_at = $1 WHERE user_id = $2",
                expires_at,
                user_id,
            )
            attachments["extrapass"] = True
            rewards_text.append("⭐ ExtraPass (30 дней)")
            await db.execute("UPDATE users SET gems = gems + 500 WHERE user_id = $1", user_id)
            _append_currency("gems", 500, "💎 гемов")

        await db.execute("UPDATE users SET coins = coins + 3000 WHERE user_id = $1", user_id)
        _append_currency("coins", 3000, "💰 монет")

        case_t2_id = await db.get_admin_case_id(2)
        case_t3_id = await db.get_admin_case_id(3)
        granted_cases: List[int] = []
        if case_t2_id:
            await db.add_user_case(user_id, case_t2_id, 2)
            granted_cases.append(2)
            rewards_text.append("1×T2 кейс")
        if case_t3_id:
            await db.add_user_case(user_id, case_t3_id, 3)
            granted_cases.append(3)
            rewards_text.append("1×T3 кейс")
        if granted_cases:
            attachments["cases"] = granted_cases
        rewards_given = True

    elif item_type and item_type.startswith("shop_set_"):
        try:
            set_id = int(item_type.split("_")[-1])
            set_result = await db.grant_shop_set_rewards(user_id, set_id)
            if set_result.get("success"):
                rewards_given = True
                granted = set_result.get("granted", [])
                rewards_text.extend(_describe_shop_set_grants(granted))
                attachments["shop_set_id"] = set_id
                if granted:
                    attachments["granted"] = granted
            else:
                logger.error(
                    "Ошибка выдачи наград набора %s пользователю %s: %s",
                    set_id,
                    user_id,
                    set_result.get("error"),
                )
        except (ValueError, IndexError):
            logger.error("Не удалось определить ID набора из item_type=%s", item_type)

    else:
        if item_type:
            logger.warning("Неизвестный тип товара %s, награды не выданы", item_type)
        else:
            logger.warning("Тип товара не указан в metadata, награды не выданы")

    return {
        "rewards_given": rewards_given,
        "rewards_text": rewards_text,
        "attachments": attachments,
    }


def _describe_shop_set_grants(granted: List[Dict[str, Any]]) -> List[str]:
    """Конвертирует структуру grant_shop_set_rewards в человекочитаемый список."""
    descriptions: List[str] = []
    for reward in granted:
        r_type = reward.get("type")
        if r_type == "gems":
            descriptions.append(f"{reward.get('amount', 0)} 💎 гемов (набор)")
        elif r_type == "coins":
            descriptions.append(f"{reward.get('amount', 0)} 💰 монет (набор)")
        elif r_type == "card":
            descriptions.append(f"🃏 Карта ID {reward.get('card_id')}")
        elif r_type == "particles":
            descriptions.append(
                f"{reward.get('amount', 0)} частиц для карты {reward.get('card_id')}"
            )
        elif r_type == "case":
            tier = reward.get("tier")
            descriptions.append(f"Кейс T{tier} (ID {reward.get('case_id')})")
        else:
            descriptions.append(f"Награда набора: {reward}")
    return descriptions


async def _create_purchase_mail(
    db: "Database",
    *,
    user_id: int,
    amount: float,
    currency: str,
    rewards_text: List[str],
    attachments: Dict[str, Any],
    metadata: Dict[str, Any],
    source: str,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Собирает красивое письмо в игровую почту о совершенной покупке."""
    subject = metadata.get("mail_subject") or "Покупка успешно оплачена"

    item_name = metadata.get("item_name") or metadata.get("display_name") or metadata.get("item_type")
    amount_line = f"Сумма: {amount:.2f} {currency}"
    item_line = f"Товар: {item_name}" if item_name else None
    source_line = "Источник: Telegram Stars" if source == "telegram_stars" else "Источник: YooKassa"

    text_lines = ["Ваш платеж успешно обработан.", amount_line]
    if item_line:
        text_lines.append(item_line)
    text_lines.append(source_line)
    text_lines.append("")  # пустая строка перед списком наград
    text_lines.append("Получено:")
    text_lines.extend(f"• {reward}" for reward in rewards_text)

    content = "\n".join(text_lines)

    try:
        return await db.create_mail(
            user_id=user_id,
            sender="Система платежей",
            subject=subject,
            content=content,
            category="rewards",
            icon="💳" if source == "yookassa_webhook" else "⭐",
            attachments=attachments if attachments else None,
        )
    except Exception as mail_err:  # pragma: no cover - страховка
        logger.error("Не удалось создать письмо о покупке пользователю %s: %s", user_id, mail_err, exc_info=True)
        return None


