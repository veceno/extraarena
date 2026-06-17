from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/|discord\.gg/|vk\.com/)", re.I)


def _normalize_policy_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _local_policy_rejection(text: str, category: str) -> Optional[str]:
    """Fast deterministic guard for obvious violations before model moderation."""
    normalized = _normalize_policy_text(text)
    if not normalized:
        return None

    sexual_terms = (
        "порно", "порнограф", "эрот", "секс", "nsfw", "xxx", "onlyfans",
        "nude", "nudity", "naked", "обнажен", "голый", "голая", "интим",
    )
    minor_terms = (
        "дет", "ребен", "несовершеннолет", "школьник", "школьниц",
        "minor", "child", "children", "kid", "teen",
    )
    fraud_terms = ("скам", "мошен", "casino", "казино", "ставк", "betting")
    hate_terms = ("наци", "фаш", "hitler", "гитлер", "расист")

    has_sexual = any(term in normalized for term in sexual_terms)
    has_minor = any(term in normalized for term in minor_terms)
    if has_sexual and has_minor:
        return "Сексуальный контент с упоминанием несовершеннолетних запрещён"
    if category in ("SQUAD", "ANNOUNCEMENT") and has_sexual:
        return "Сексуальный контент запрещён в сквадах"
    if category in ("SQUAD", "ANNOUNCEMENT") and _URL_RE.search(normalized):
        return "Ссылки и внешние контакты запрещены"
    if any(term in normalized for term in fraud_terms):
        return "Мошеннический или азартный контент запрещён"
    if any(term in normalized for term in hate_terms):
        return "Ненавистнические символы и агитация запрещены"
    return None


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


async def moderate_content(
    text: str,
    category: str,
    image_b64: Optional[str] = None,
    image_mime: str = "image/jpeg",
) -> dict[str, Any]:
    """
    Отправляет контент на автомодерацию через PolzaAI (OpenAI-compatible).

    Returns:
        {"decision": "approve"|"reject", "reason": str}

    При любой ошибке API — fail-closed (reject).
    """
    from infrastructure.community_config import (
        MODERATION_API_BASE_URL,
        MODERATION_API_KEY,
        MODERATION_MODEL,
        MODERATION_PROMPTS,
    )

    system_prompt = MODERATION_PROMPTS.get(category, MODERATION_PROMPTS["IDEA"])
    system_prompt += (
        "\n\nВерни только JSON. Схема: "
        '{"decision":"approve|reject","reason":"строка","image_checked":true|false}. '
        "Если передано изображение, image_checked=true можно ставить только когда изображение реально просмотрено. "
        "Если изображение недоступно, не поддерживается или не было просмотрено — decision=reject."
    )

    local_reason = _local_policy_rejection(text, category)
    if local_reason:
        return {"decision": "reject", "reason": local_reason}
    if not MODERATION_API_KEY:
        logger.error("Moderation API key is not configured")
        return {"decision": "reject", "reason": "Модерация не настроена"}

    # Build messages
    user_content: Any
    if image_b64:
        image_mime = image_mime if image_mime in ("image/png", "image/jpeg", "image/webp") else "image/jpeg"
        user_content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
            },
        ]
    else:
        user_content = text

    payload = {
        "model": MODERATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    try:
        import aiohttp
        import ssl
        import certifi

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        timeout = aiohttp.ClientTimeout(total=15)
        headers = {
            "Authorization": f"Bearer {MODERATION_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{MODERATION_API_BASE_URL}/chat/completions"

        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Moderation API error status=%s body=%s", resp.status, body[:200]
                    )
                    return {"decision": "reject", "reason": "Модерация временно недоступна"}

                data = await resp.json()
                raw = data["choices"][0]["message"]["content"]
                try:
                    result = _extract_json_object(raw)
                    decision = result.get("decision", "reject")
                    reason = result.get("reason", "")
                    if decision not in ("approve", "reject"):
                        decision = "reject"
                    if image_b64 and decision == "approve" and result.get("image_checked") is not True:
                        return {
                            "decision": "reject",
                            "reason": "Изображение не удалось надёжно проверить",
                        }
                    logger.info(
                        "Moderation decision category=%s has_image=%s decision=%s reason=%s",
                        category,
                        bool(image_b64),
                        decision,
                        str(reason)[:120],
                    )
                    return {"decision": decision, "reason": reason}
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Moderation response parse error: %s", raw[:200])
                    return {"decision": "reject", "reason": "Модерация вернула некорректный ответ"}

    except Exception as exc:
        logger.error("Moderation API exception: %s", exc, exc_info=True)
        return {"decision": "reject", "reason": "Модерация временно недоступна"}


async def check_rate_limit(db: Any, user_id: int) -> dict[str, Any]:
    """
    Проверяет rate limit для отправок на модерацию.

    Returns:
        {"allowed": bool, "remaining": int, "retry_after_seconds": int}
    """
    from infrastructure.community_config import SUBMISSION_RATE_LIMIT, SUBMISSION_RATE_WINDOW_MINUTES

    count = await db.count_recent_submissions(user_id, minutes=SUBMISSION_RATE_WINDOW_MINUTES)
    remaining = max(0, SUBMISSION_RATE_LIMIT - count)
    allowed = count < SUBMISSION_RATE_LIMIT

    retry_after = 0
    if not allowed:
        # Find oldest submission in window to calculate when slot frees up
        try:
            oldest = await db.fetchval(
                """
                SELECT created_at FROM community_submissions
                WHERE user_id = $1
                  AND created_at > NOW() - ($2 || ' minutes')::INTERVAL
                ORDER BY created_at ASC
                LIMIT 1
                """,
                user_id,
                str(SUBMISSION_RATE_WINDOW_MINUTES),
            )
            if oldest:
                import datetime as _dt
                from datetime import timezone as _tz
                window_end = oldest.replace(tzinfo=_tz.utc) + _dt.timedelta(
                    minutes=SUBMISSION_RATE_WINDOW_MINUTES
                )
                now = _dt.datetime.now(_tz.utc)
                retry_after = max(0, int((window_end - now).total_seconds()))
        except Exception:
            retry_after = SUBMISSION_RATE_WINDOW_MINUTES * 60

    return {
        "allowed": allowed,
        "remaining": remaining,
        "retry_after_seconds": retry_after,
    }
