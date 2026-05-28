from __future__ import annotations

import random
from typing import Any
from urllib.parse import urlencode

from infrastructure.config import LEAGUE_CONFIG, LEAGUE_NEXT_TROPHIES


WEBAPP_SECTION_BY_CATEGORY = {
    "generator": "generator",
    "shop": "shop",
    "reminders": "arena",
    "squad_member_role": "squads",
    "squad_new_member": "squads",
    "squad_disbanded": "squads",
    "squad_boost": "squads",
    "extra_arena_modifier": "arena",
}

NOTIFICATION_SETTING_BY_CATEGORY = {
    "generator": "notif_generator",
    "shop": "notif_shop",
    "reminders": "notif_reminders",
    "squad_member_role": "notif_squad_member_role",
    "squad_new_member": "notif_squad_new_member",
    "squad_disbanded": "notif_squad_disbanded",
    "squad_boost": "notif_squad_boost",
    "extra_arena_modifier": "notif_extra_arena_modifiers",
}

NOTIFICATION_DEFAULTS = {
    "notif_shop": False,
    "notif_reminders": True,
    "notif_squad_member_role": True,
    "notif_squad_new_member": True,
    "notif_squad_disbanded": True,
    "notif_squad_boost": True,
    "notif_extra_arena_modifiers": True,
}

REMINDER_DUSTY_WEIGHT = 1


def wins_required_for_case(extra_pass: str | None) -> int:
    if extra_pass == "ultra":
        return 3
    if extra_pass == "active":
        return 4
    return 5


def next_league_trophies(trophies: int) -> int | None:
    for threshold in LEAGUE_NEXT_TROPHIES:
        if trophies < threshold:
            return threshold - trophies
    return None


def wins_to_next_case(wins_since_last_case: int, extra_pass: str | None) -> int:
    required = wins_required_for_case(extra_pass)
    return max(1, required - max(0, wins_since_last_case))


def classify_generator_event(*, stored_keys: int, new_keys: int, cap: int) -> str | None:
    if new_keys <= 0:
        return None
    keys_before_latest_tick = min(stored_keys + max(0, new_keys - 1), cap)
    if keys_before_latest_tick >= cap:
        return "generator_full_blocked_key"
    if keys_before_latest_tick + 1 >= cap:
        return "generator_full_on_new_key"
    return "generator_new_key"


def choose_reminder_payload(profile: dict[str, Any], *, rng: random.Random | None = None) -> dict[str, Any]:
    rng = rng or random
    trophies = int(profile.get("trophies") or 0)
    wins = int(profile.get("wins_since_last_case") or 0)
    extra_pass = profile.get("extra_pass") or "inactive"
    squad_id = int(profile.get("squad_id") or 0)

    candidates: list[tuple[int, dict[str, Any]]] = []
    league_remaining = next_league_trophies(trophies)
    if league_remaining is not None:
        candidates.append((
            3,
            {
                "template": "arena_next_league",
                "text": f"До новой лиги осталось {league_remaining} кубков - поднажми, вперед в бой!",
                "section": "arena",
            },
        ))

    wins_remaining = wins_to_next_case(wins, extra_pass)
    candidates.append((
        3,
        {
            "template": "arena_case_reward",
            "text": f"До награды за бои осталось всего {wins_remaining} победы! Вперед в бой",
            "section": "arena",
        },
    ))
    candidates.append((
        2,
        {
            "template": "general_memory",
            "text": "О твоих похождениях помнит ВСЯ арена! Но скоро забудет... Вперед в бой",
            "section": "arena",
        },
    ))
    candidates.append((
        REMINDER_DUSTY_WEIGHT,
        {
            "template": "general_dusty_deck",
            "text": "Скорее в бой - колода уже покрылась пылью, а скоро и вовсе отсыреет",
            "section": "arena",
        },
    ))
    if squad_id:
        candidates.append((
            2,
            {
                "template": "squad_missed",
                "text": "В твоем скваде по тебе соскучились - вперед в бой!",
                "section": "squads",
            },
        ))

    total = sum(weight for weight, _ in candidates)
    pick = rng.uniform(0, total)
    upto = 0.0
    for weight, payload in candidates:
        upto += weight
        if pick <= upto:
            return payload
    return candidates[-1][1]


def format_notification_message(event_type: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    if event_type in {"app_update", "app_update_required"}:
        return str(payload.get("body") or "Хорошие новости! Вышло обновление, скачай новую версию, чтобы продолжить игру")
    if event_type == "dice_ready":
        return "🎲 Эй! Самое время бросить кости!"
    if event_type == "generator_new_key":
        count = int(payload.get("keys") or 1)
        return f"Новый ключ готов! В генераторе уже {count} ключ(ей)."
    if event_type == "generator_full_on_new_key":
        cap = int(payload.get("cap") or payload.get("keys") or 0)
        return f"Новый ключ готов, и генератор переполнился! Забери ключи, максимум сейчас: {cap}."
    if event_type == "generator_full_blocked_key":
        return "Щас бы был новый ключ, но генератор уже переполнен!"
    if event_type == "shop_particles":
        return "Новые частицы карт уже в магазине!"
    if event_type == "extra_arena_modifier_changed":
        label = payload.get("label") or payload.get("mode_name") or "новый модификатор"
        return f"В ExtraArena сменился режим: {label}. Самое время проверить колоду!"
    if event_type == "daily_reminder":
        return str(payload.get("text") or "Пора вернуться на арену!")
    if event_type == "squad_member_role":
        nick = payload.get("nick") or "Участник"
        action = payload.get("action") or "изменил роль"
        return f"{nick}: {action} в твоем скваде."
    if event_type == "squad_new_member":
        nick = payload.get("nick") or "Новый игрок"
        squad = payload.get("squad_name") or "сквад"
        return f"{nick} вступил в {squad}!"
    if event_type == "squad_disbanded":
        squad = payload.get("squad_name") or "Твой сквад"
        return f"{squad} расформирован."
    if event_type == "squad_boost":
        squad = payload.get("squad_name") or "У сквада"
        return f"{squad} активировал Boost!"
    return str(payload.get("text") or "В ExtraArena новое событие!")


def notification_section(category: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    return str(payload.get("section") or WEBAPP_SECTION_BY_CATEGORY.get(category) or "arena")


def build_webapp_url(base_url: str, *, section: str) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{urlencode({'section': section})}"
