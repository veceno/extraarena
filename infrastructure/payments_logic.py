"""Дополнительные утилиты для универсальной обработки успешных платежей.

В этом модуле сосредоточена логика выдачи наград, создания писем и фиксации
статуса платежа, чтобы один и тот же код использовали и Telegram Stars, и YooKassa.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
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

    claim_payment = getattr(db, "claim_payment_for_processing", None)
    if claim_payment:
        claimed_record = await claim_payment(payment_id)
        if not claimed_record:
            result["status"] = "already_processed"
            logger.info("Платеж %s уже был обработан ранее или обрабатывается сейчас", payment_id)
            return result
        payment_record = claimed_record
    elif payment_record.get("rewards_processed"):
        result["status"] = "already_processed"
        logger.info("Платеж %s уже был обработан ранее, пропускаем", payment_id)
        return result

    claim_reserved = bool(claim_payment)

    try:
        metadata = _ensure_metadata(payment_record.get("metadata"))
        item_type = _resolve_item_type(payment_record, metadata, logger)

        metadata.setdefault("payment_id", payment_id)
        reward_ctx = await _grant_rewards_for_item(
            db=db,
            user_id=payment_record["user_id"],
            item_type=item_type,
            metadata=metadata,
            payment_id=payment_id,
            logger=logger,
        )
    except Exception:
        if claim_reserved:
            await _release_payment_processing_claim(db, payment_id, logger)
        raise

    result.update(
        {
            "rewards_text": reward_ctx["rewards_text"],
            "attachments": reward_ctx["attachments"],
            "item_type": item_type,
        }
    )

    if not reward_ctx.get("rewards_given"):
        result["status"] = "no_rewards"
        logger.error(
            "Платеж %s не обработан: для item_type=%s награды не были выданы. rewards_processed не меняем.",
            payment_id,
            item_type,
        )
        await _record_payment_no_rewards_alert(
            db=db,
            payment_id=payment_id,
            payment_record=payment_record,
            item_type=item_type,
            metadata=metadata,
            source=source,
            logger=logger,
        )
        if claim_reserved:
            await _release_payment_processing_claim(db, payment_id, logger)
        return result

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

    # Трекаем economy_events для каждой начисленной валюты. Награды уже выданы,
    # поэтому сбой аудита не должен открывать повторную выдачу товара.
    try:
        await _track_purchase_economy_events(
            db=db,
            user_id=payment_record["user_id"],
            item_type=item_type,
            attachments=reward_ctx["attachments"],
            metadata=metadata,
            logger=logger,
        )
    except Exception as event_err:
        logger.error("Ошибка записи economy_events для платежа %s: %s", payment_id, event_err, exc_info=True)
        result["warnings"].append(str(event_err))

    mark_warning = await _mark_payment_rewards_processed(db, payment_id, logger)
    if mark_warning:
        result["warnings"].append(mark_warning)

    result["status"] = "processed"
    return result


async def _release_payment_processing_claim(db: "Database", payment_id: str, logger: logging.Logger) -> None:
    release_claim = getattr(db, "release_payment_processing_claim", None)
    if not release_claim:
        return
    try:
        await release_claim(payment_id)
    except Exception as release_err:  # pragma: no cover - защитный блок
        logger.error("Ошибка сброса processing-claim платежа %s: %s", payment_id, release_err, exc_info=True)


async def _record_payment_no_rewards_alert(
    *,
    db: "Database",
    payment_id: str,
    payment_record: Dict[str, Any],
    item_type: Optional[str],
    metadata: Dict[str, Any],
    source: str,
    logger: logging.Logger,
) -> None:
    alert_payload = {
        "alert_type": "payment_no_rewards",
        "payment_id": payment_id,
        "user_id": payment_record.get("user_id"),
        "item_type": item_type,
        "source": source,
        "description": payment_record.get("description"),
        "metadata": dict(metadata),
    }
    logger.critical(
        "PAYMENT_NO_REWARDS payment_id=%s user_id=%s item_type=%s source=%s description=%s metadata=%s",
        alert_payload["payment_id"],
        alert_payload["user_id"],
        alert_payload["item_type"],
        alert_payload["source"],
        alert_payload["description"],
        alert_payload["metadata"],
    )

    record_alert = getattr(db, "record_payment_processing_alert", None)
    if record_alert:
        try:
            await record_alert(**alert_payload)
        except Exception as alert_err:  # pragma: no cover - защитный блок
            logger.error("Ошибка записи payment_no_rewards alert для платежа %s: %s", payment_id, alert_err, exc_info=True)

    record_admin_action = getattr(db, "record_admin_account_action", None)
    if not record_admin_action:
        return
    try:
        await record_admin_action(
            0,
            int(payment_record.get("user_id")),
            "payment_no_rewards",
            reason="successful_payment_without_rewards",
            payload={
                "payment_id": payment_id,
                "source": source,
                "item_type": item_type,
                "amount": payment_record.get("amount"),
                "currency": payment_record.get("currency"),
                "description": payment_record.get("description"),
                "metadata": dict(metadata),
            },
        )
    except Exception as admin_alert_err:  # pragma: no cover - защитный блок
        logger.error(
            "Ошибка записи admin payment_no_rewards alert для платежа %s: %s",
            payment_id,
            admin_alert_err,
            exc_info=True,
        )


async def _mark_payment_rewards_processed(db: "Database", payment_id: str, logger: logging.Logger) -> str | None:
    try:
        updated_rows = await db.fetchval(
            """
            UPDATE payments
            SET rewards_processed = TRUE,
                metadata = COALESCE(metadata, '{}'::jsonb) - 'rewards_processing' - 'rewards_processing_started_at',
                updated_at = NOW()
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
        return str(mark_err)
    return None


def _reward_step_done(metadata: Dict[str, Any], step_id: str) -> bool:
    steps = metadata.get("reward_steps") or {}
    return isinstance(steps, dict) and bool(steps.get(step_id))


async def _mark_payment_reward_step(
    db: "Database",
    *,
    payment_id: str | None,
    metadata: Dict[str, Any],
    step_id: str,
    logger: logging.Logger,
) -> None:
    metadata.setdefault("reward_steps", {})[step_id] = True
    if not payment_id:
        return
    marker = getattr(db, "mark_payment_reward_step", None)
    if marker:
        await marker(payment_id, step_id)
        return
    try:
        await db.fetchval(
            """
            UPDATE payments
            SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{reward_steps}',
                    COALESCE(metadata->'reward_steps', '{}'::jsonb) || jsonb_build_object($2::text, true),
                    true
                ),
                updated_at = NOW()
            WHERE payment_id = $1
            RETURNING 1
            """,
            payment_id,
            step_id,
        )
    except Exception as exc:
        logger.error("Ошибка фиксации payment reward-step %s для %s: %s", step_id, payment_id, exc, exc_info=True)
        raise


async def _run_payment_reward_step(
    db: "Database",
    *,
    payment_id: str | None,
    metadata: Dict[str, Any],
    step_id: str,
    logger: logging.Logger,
    grant_fn,
) -> bool:
    if _reward_step_done(metadata, step_id):
        logger.info("Платеж %s: reward-step %s уже применен, пропускаем", payment_id, step_id)
        if payment_id and hasattr(db, "mark_payment_reward_step"):
            try:
                await db.mark_payment_reward_step(payment_id, step_id)
            except Exception as exc:
                logger.warning(
                    "Платеж %s: не удалось backfill payment_reward_steps для %s: %s",
                    payment_id,
                    step_id,
                    exc,
                )
        return False
    runner = getattr(db, "run_payment_reward_step", None)
    if runner and payment_id:
        async def apply_step(executor):
            return await grant_fn(executor)

        result = await runner(payment_id, step_id, apply_step)
        if result.get("applied"):
            metadata.setdefault("reward_steps", {})[step_id] = True
            return True
        metadata.setdefault("reward_steps", {})[step_id] = True
        return False
    try:
        await grant_fn(db)
    except TypeError:
        await grant_fn()
    await _mark_payment_reward_step(
        db,
        payment_id=payment_id,
        metadata=metadata,
        step_id=step_id,
        logger=logger,
    )
    return True


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


def _coerce_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _extra_pass_is_effective(profile: Dict[str, Any] | None, now: datetime | None = None) -> bool:
    if not profile:
        return False
    mode = str(profile.get("extra_pass") or "inactive")
    if mode not in {"active", "ultra"}:
        return False
    expires_at = _coerce_utc_datetime(profile.get("extra_pass_expires_at"))
    if not expires_at:
        return True
    return expires_at > (now or datetime.now(timezone.utc))


async def _grant_extra_pass_access(
    db: "Database",
    *,
    user_id: int,
    requested_mode: str,
    logger: logging.Logger,
    bonus_gems: int = 0,
    executor=None,
) -> tuple[bool, str, datetime]:
    now = datetime.now(timezone.utc)
    profile = None
    get_profile = getattr(db, "get_user_profile", None)
    if get_profile:
        profile = await get_profile(user_id)

    current_mode = str((profile or {}).get("extra_pass") or "inactive")
    current_expiry = _coerce_utc_datetime((profile or {}).get("extra_pass_expires_at"))
    current_active = _extra_pass_is_effective(profile, now)
    tier_order = {"inactive": 0, "active": 1, "ultra": 2}
    final_mode = requested_mode
    if current_active and tier_order.get(current_mode, 0) > tier_order.get(requested_mode, 0):
        final_mode = current_mode

    base_expiry = current_expiry if current_expiry and current_expiry > now else now
    expires_at = base_expiry + timedelta(days=30)

    if bonus_gems > 0:
        target = executor or db
        updated = await target.fetchval(
            """
            UPDATE users
            SET extra_pass = $1,
                extra_pass_expires_at = $2,
                gems = gems + $3
            WHERE user_id = $4
            RETURNING 1
            """,
            final_mode,
            expires_at,
            bonus_gems,
            user_id,
        )
    else:
        target = executor or db
        updated = await target.fetchval(
            "UPDATE users SET extra_pass = $1, extra_pass_expires_at = $2 WHERE user_id = $3 RETURNING 1",
            final_mode,
            expires_at,
            user_id,
        )
    if not updated:
        logger.error("extra_pass: пользователь %s не найден", user_id)
    return bool(updated), final_mode, expires_at


async def _grant_rewards_for_item(
    db: "Database",
    *,
    user_id: int,
    item_type: Optional[str],
    metadata: Dict[str, Any],
    logger: logging.Logger,
    payment_id: str | None = None,
) -> Dict[str, Any]:
    """Начисляет ресурсы в зависимости от item_type и собирает описание наград."""
    logger.info("GRANT_REWARDS: user_id=%s item_type=%s metadata=%s", user_id, item_type, {k: metadata.get(k) for k in ("item_name","package_type","recipient_id") if k in metadata})
    rewards_text: List[str] = []
    attachments: Dict[str, Any] = {}
    rewards_given = False
    had_completed_steps = bool(metadata.get("reward_steps"))

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
            async def grant_currency(executor):
                column = {"gems": "gems", "coins": "coins", "keys": "keys"}[item_type]
                await executor.execute(
                    f"UPDATE users SET {column} = COALESCE({column}, 0) + $1 WHERE user_id = $2",
                    amount_value,
                    user_id,
                )
                if item_type == "keys" and hasattr(db, "sync_user_key_cases"):
                    await db.sync_user_key_cases(user_id)
                emoji = {"gems": "💎", "coins": "💰", "keys": "📦"}[item_type]
                label = {"gems": "гемов", "coins": "монет", "keys": "кейсов"}[item_type]
                _append_currency("gems" if item_type == "gems" else item_type, amount_value, f"{emoji} {label}")

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id=f"currency_{item_type}",
                logger=logger,
                grant_fn=grant_currency,
            ):
                rewards_given = True

    elif item_type and item_type.startswith("keys_"):
        # Покупка кейсов в формате keys_1, keys_3, keys_10 и т.д.
        try:
            keys_amount = int(item_type.split("_")[1])
            if keys_amount > 0:
                async def grant_keys_item(executor):
                    await executor.execute(
                        "UPDATE users SET keys = COALESCE(keys, 0) + $1 WHERE user_id = $2",
                        keys_amount,
                        user_id,
                    )
                    if hasattr(db, "sync_user_key_cases"):
                        await db.sync_user_key_cases(user_id)
                    _append_currency("keys", keys_amount, f"📦 кейсов")

                if await _run_payment_reward_step(
                    db,
                    payment_id=payment_id,
                    metadata=metadata,
                    step_id="keys_item",
                    logger=logger,
                    grant_fn=grant_keys_item,
                ):
                    rewards_given = True
        except (ValueError, IndexError):
            logger.error("Не удалось извлечь количество кейсов из item_type=%s", item_type)

    elif item_type == "gems_package":
        from infrastructure.shop_config import GEM_PACKAGES

        package_type = metadata.get("package_type", "")
        pkg = GEM_PACKAGES.get(package_type, {})
        gems = int(pkg.get("gems") or metadata.get("package_gems") or 0)
        if gems > 0:
            async def grant_gems_package(executor):
                if pkg.get("one_time") and not metadata.get("skip_starter_mark"):
                    consumed = True
                    consume = getattr(db, "consume_one_time_payment_reservation", None)
                    if consume and payment_id:
                        consumed = await consume(
                            user_id=user_id,
                            product_key=str(metadata.get("product_key") or "gems_package:starter_once"),
                            payment_id=payment_id,
                            executor=executor,
                        )
                    await executor.execute(
                        """
                        INSERT INTO user_settings (user_id, starter_pack_used)
                        VALUES ($1, FALSE)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        user_id,
                    )
                    starter_claimed = await executor.fetchval(
                        """
                        UPDATE user_settings
                        SET starter_pack_used = TRUE
                        WHERE user_id = $1
                          AND starter_pack_used = FALSE
                        RETURNING 1
                        """,
                        user_id,
                    )
                    if not consumed or not starter_claimed:
                        raise RuntimeError("starter_once_already_used")
                await executor.execute("UPDATE users SET gems = COALESCE(gems, 0) + $1 WHERE user_id = $2", gems, user_id)
                _append_currency("gems", gems, "💎 гемов")

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id="gems_package",
                logger=logger,
                grant_fn=grant_gems_package,
            ):
                rewards_given = True
        else:
            logger.error("gems_package: не удалось определить количество гемов для package_type=%s", package_type)

    elif item_type and item_type.startswith("gems_"):
        try:
            gems_amount = int(item_type.split("_")[1])

            async def grant_gems_item(executor):
                await executor.execute("UPDATE users SET gems = COALESCE(gems, 0) + $1 WHERE user_id = $2", gems_amount, user_id)
                _append_currency("gems", gems_amount, "💎 гемов")

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id="gems_item",
                logger=logger,
                grant_fn=grant_gems_item,
            ):
                rewards_given = True
        except (ValueError, IndexError):
            logger.error("Не удалось извлечь количество гемов из item_type=%s", item_type)

    elif item_type == "extrapass":
        async def grant_extrapass(executor):
            updated, final_mode, _expires_at = await _grant_extra_pass_access(
                db,
                user_id=user_id,
                requested_mode="active",
                logger=logger,
                executor=executor,
            )
            if not updated:
                raise RuntimeError("extrapass_user_not_found")
            attachments["extrapass"] = True
            if final_mode == "ultra":
                attachments["extrapass_ultra"] = True
                rewards_text.append("💫 ExtraPass Ultra продлен на 30 дней")
            else:
                rewards_text.append("⭐ ExtraPass (30 дней)")

        if await _run_payment_reward_step(
            db,
            payment_id=payment_id,
            metadata=metadata,
            step_id="extrapass_access",
            logger=logger,
            grant_fn=grant_extrapass,
        ):
            rewards_given = True

    elif item_type == "extrapass_ultra":
        async def grant_ultra_pass(executor):
            updated, _final_mode, _expires_at = await _grant_extra_pass_access(
                db,
                user_id=user_id,
                requested_mode="ultra",
                logger=logger,
                bonus_gems=500,
                executor=executor,
            )
            if not updated:
                raise RuntimeError("extrapass_ultra_user_not_found")
            _append_currency("gems", 500, "💎 гемов (бонус Ultra)")
            attachments["extrapass"] = True
            attachments["extrapass_ultra"] = True
            rewards_text.append("💫 ExtraPass Ultra (30 дней)")

        if await _run_payment_reward_step(
            db,
            payment_id=payment_id,
            metadata=metadata,
            step_id="extrapass_ultra_access",
            logger=logger,
            grant_fn=grant_ultra_pass,
        ):
            rewards_given = True

    elif item_type == "extrapass_gift":
        recipient_id = int(metadata.get("recipient_id", 0))
        if recipient_id <= 0:
            logger.error("extrapass_gift: recipient_id не указан или равен 0, отмена")
        else:
            is_ultra = metadata.get("ultra") in {True, "true", "1"}

            async def grant_extrapass_gift(executor):
                pass_mode = "ultra" if is_ultra else "active"
                updated, final_mode, _expires_at = await _grant_extra_pass_access(
                    db,
                    user_id=recipient_id,
                    requested_mode=pass_mode,
                    logger=logger,
                    bonus_gems=500 if is_ultra else 0,
                    executor=executor,
                )
                if not updated:
                    raise RuntimeError("extrapass_gift_recipient_not_found")

                if is_ultra:
                    _append_currency("gems", 500, "💎 гемов (бонус Ultra)")
                    attachments["extrapass_ultra"] = True
                    rewards_text.append(f"💫 ExtraPass Ultra (подарок для ID {recipient_id})")
                elif final_mode == "ultra":
                    attachments["extrapass_ultra"] = True
                    rewards_text.append(f"💫 ExtraPass Ultra продлен подарком для ID {recipient_id}")
                else:
                    rewards_text.append(f"\u2b50 ExtraPass (подарок для ID {recipient_id})")

                attachments["extrapass_gift"] = True
                attachments["gift_recipient_id"] = recipient_id
                logger.info(
                    "extrapass_gift: покупатель=%s получатель=%s ultra=%s",
                    user_id, recipient_id, is_ultra,
                )

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id="extrapass_gift_access",
                logger=logger,
                grant_fn=grant_extrapass_gift,
            ):
                rewards_given = True

    elif item_type == "starter_boost":
        user_profile = await db.get_user_profile(user_id)
        has_extra_pass = _extra_pass_is_effective(user_profile)

        async def grant_starter_pass_or_gems(executor):
            if has_extra_pass:
                await executor.execute("UPDATE users SET gems = COALESCE(gems, 0) + 1200 WHERE user_id = $1", user_id)
                _append_currency("gems", 1200, "💎 гемов (700 бонус за активный ExtraPass + 500)")
                return
            updated, final_mode, _expires_at = await _grant_extra_pass_access(
                db,
                user_id=user_id,
                requested_mode="active",
                logger=logger,
                bonus_gems=500,
                executor=executor,
            )
            if not updated:
                raise RuntimeError("starter_boost_user_not_found")
            attachments["extrapass"] = True
            if final_mode == "ultra":
                attachments["extrapass_ultra"] = True
                rewards_text.append("💫 ExtraPass Ultra продлен на 30 дней")
            else:
                rewards_text.append("⭐ ExtraPass (30 дней)")
            _append_currency("gems", 500, "💎 гемов")

        if await _run_payment_reward_step(
            db,
            payment_id=payment_id,
            metadata=metadata,
            step_id="starter_boost_pass_or_gems",
            logger=logger,
            grant_fn=grant_starter_pass_or_gems,
        ):
            rewards_given = True

        async def grant_starter_coins(executor):
            await executor.execute("UPDATE users SET coins = COALESCE(coins, 0) + 3000 WHERE user_id = $1", user_id)
            _append_currency("coins", 3000, "💰 монет")

        if await _run_payment_reward_step(
            db,
            payment_id=payment_id,
            metadata=metadata,
            step_id="starter_boost_coins",
            logger=logger,
            grant_fn=grant_starter_coins,
        ):
            rewards_given = True

        case_t2_id = await db.get_admin_case_id(2)
        case_t3_id = await db.get_admin_case_id(3)
        granted_cases: List[int] = []
        if case_t2_id:
            async def grant_t2_case(executor):
                if hasattr(executor, "fetchrow"):
                    await executor.fetchrow(
                        """
                        INSERT INTO user_cases (user_id, case_id, tier, status)
                        VALUES ($1, $2, $3, 'pending')
                        RETURNING id
                        """,
                        user_id,
                        case_t2_id,
                        2,
                    )
                else:
                    await db.add_user_case(user_id, case_t2_id, 2)
                granted_cases.append(2)
                rewards_text.append("1×T2 кейс")

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id="starter_boost_case_t2",
                logger=logger,
                grant_fn=grant_t2_case,
            ):
                rewards_given = True
        if case_t3_id:
            async def grant_t3_case(executor):
                if hasattr(executor, "fetchrow"):
                    await executor.fetchrow(
                        """
                        INSERT INTO user_cases (user_id, case_id, tier, status)
                        VALUES ($1, $2, $3, 'pending')
                        RETURNING id
                        """,
                        user_id,
                        case_t3_id,
                        3,
                    )
                else:
                    await db.add_user_case(user_id, case_t3_id, 3)
                granted_cases.append(3)
                rewards_text.append("1×T3 кейс")

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id="starter_boost_case_t3",
                logger=logger,
                grant_fn=grant_t3_case,
            ):
                rewards_given = True
        if granted_cases:
            attachments["cases"] = granted_cases

    elif item_type and item_type.startswith("shop_set_"):
        try:
            set_id = int(item_type.split("_")[-1])

            async def grant_shop_set_step(executor):
                if hasattr(db, "grant_shop_set_rewards_on_conn") and hasattr(executor, "fetchrow"):
                    set_result = await db.grant_shop_set_rewards_on_conn(executor, user_id, set_id)
                else:
                    set_result = await db.grant_shop_set_rewards(user_id, set_id)
                if not set_result.get("success"):
                    raise RuntimeError(str(set_result.get("error") or "shop_set_grant_failed"))
                granted = set_result.get("granted", [])
                rewards_text.extend(_describe_shop_set_grants(granted))
                attachments["shop_set_id"] = set_id
                if granted:
                    attachments["granted"] = granted

            if await _run_payment_reward_step(
                db,
                payment_id=payment_id,
                metadata=metadata,
                step_id=f"shop_set_{set_id}",
                logger=logger,
                grant_fn=grant_shop_set_step,
            ):
                rewards_given = True
        except (ValueError, IndexError):
            logger.error("Не удалось определить ID набора из item_type=%s", item_type)

    else:
        if item_type:
            logger.warning("Неизвестный тип товара %s, награды не выданы", item_type)
        else:
            logger.warning("Тип товара не указан в metadata, награды не выданы")

    return {
        "rewards_given": rewards_given or had_completed_steps,
        "rewards_text": rewards_text,
        "attachments": attachments,
    }


async def _track_purchase_economy_events(
    db: "Database",
    user_id: int,
    item_type: Optional[str],
    attachments: Dict[str, Any],
    metadata: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Записывает economy_events для каждого начисленного ресурса при покупке."""
    base_meta = {
        "item_type": item_type,
        "payment_id": metadata.get("payment_id"),
        "item_name": metadata.get("item_name") or metadata.get("display_name"),
        "package_type": metadata.get("package_type"),
        "provider": metadata.get("provider"),
    }

    if attachments.get("gems"):
        await db.track_economy_event(
            user_id=user_id,
            event_type="earn",
            resource="gems",
            amount=float(attachments["gems"]),
            source="purchase",
            metadata=base_meta,
        )
    if attachments.get("coins"):
        await db.track_economy_event(
            user_id=user_id,
            event_type="earn",
            resource="coins",
            amount=float(attachments["coins"]),
            source="purchase",
            metadata=base_meta,
        )
    if attachments.get("keys"):
        await db.track_economy_event(
            user_id=user_id,
            event_type="earn",
            resource="keys",
            amount=float(attachments["keys"]),
            source="purchase",
            metadata=base_meta,
        )
    if attachments.get("extrapass"):
        await db.track_economy_event(
            user_id=user_id,
            event_type="earn",
            resource="extrapass",
            amount=1,
            source="purchase",
            metadata=base_meta,
        )



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
    source_labels = {
        "telegram_stars": "Telegram Stars",
        "robokassa_webhook": "Robokassa",
        "robokassa_success_return": "Robokassa",
        "robokassa_status_check": "Robokassa",
        "rustore_webhook": "RuStore",
        "rustore_status_check": "RuStore",
        "yookassa_webhook": "YooKassa",
        "yookassa_status_check": "YooKassa",
    }
    source_line = f"Источник: {source_labels.get(source, 'YooKassa')}"

    text_lines = ["Ваш платеж успешно обработан.", amount_line]
    if item_line:
        text_lines.append(item_line)
    text_lines.append(source_line)
    text_lines.append("")  # пустая строка перед списком наград
    text_lines.append("Получено:")
    text_lines.extend(f"• {reward}" for reward in rewards_text)

    content = "\n".join(text_lines)

    try:
        mail_result = await db.create_mail(
            user_id=user_id,
            sender="Система платежей",
            subject=subject,
            content=content,
            category="rewards",
            icon="⭐" if source == "telegram_stars" else "💳",
            attachments=attachments if attachments else None,
        )
        if not mail_result.get("success"):
            logger.error(
                "Не удалось создать письмо о покупке пользователю %s: %s",
                user_id,
                mail_result.get("error", "unknown"),
            )
            return None
        return mail_result
    except Exception as mail_err:  # pragma: no cover - страховка
        logger.error("Не удалось создать письмо о покупке пользователю %s: %s", user_id, mail_err, exc_info=True)
        return None
