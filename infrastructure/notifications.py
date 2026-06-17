from __future__ import annotations

import random
from typing import Any
from html import escape
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
    "game_invites": "friends",
    "friend_requests": "friends",
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
    "game_invites": "notif_game_invites",
    "friend_requests": "notif_friend_requests",
}

NOTIFICATION_DEFAULTS = {
    "notif_shop": False,
    "notif_reminders": True,
    "notif_squad_member_role": True,
    "notif_squad_new_member": True,
    "notif_squad_disbanded": True,
    "notif_squad_boost": True,
    "notif_extra_arena_modifiers": True,
    "notif_game_invites": True,
    "notif_friend_requests": True,
}

REMINDER_DUSTY_WEIGHT = 1
REMINDER_TITLES = ("Вперед в бой", "Задай им тряски!")


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
            return _with_reminder_title(payload, rng)
    return _with_reminder_title(candidates[-1][1], rng)


def _with_reminder_title(payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {**payload, "title": rng.choice(REMINDER_TITLES)}


def format_notification_message(event_type: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    if event_type in {"app_update", "app_update_required"}:
        return str(payload.get("body") or "Хорошие новости! Вышло обновление, скачай новую версию, чтобы продолжить игру")
    if event_type == "generator_new_key":
        count = int(payload.get("keys") or 1)
        return f"Новый ключ уже готов! - скорее открой кейс! В генераторе уже {count} ключ(ей)."
    if event_type == "generator_full_on_new_key":
        return "Генератор уже переполнен - собери ключ и открой кейс, чтобы генератор заработал!"
    if event_type == "generator_full_blocked_key":
        return "Ты бы мог получить новый ключ, но генератор уже переполнен!"
    if event_type == "shop_particles":
        return "Новые частицы карт уже в магазине!"
    if event_type == "extra_arena_modifier_changed":
        label = payload.get("label") or payload.get("mode_name") or "новый модификатор"
        return f"В ExtraArena сменился модификатор: {label}! Ну что, задашь им жару?"
    if event_type == "friendly_battle_invite":
        from_name = payload.get("from_name") or "Друг"
        return f"{from_name} вызывает тебя на дружеский бой!"
    if event_type == "friend_request_received":
        from_name = payload.get("from_name") or "Игрок"
        return f"{from_name} отправил заявку в друзья."
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


def format_telegram_notification_message(event_type: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    if event_type == "generator_new_key":
        count = int(payload.get("keys") or 1)
        return f"<b>Новый ключ уже готов!</b> - скорее открой кейс! В генераторе уже {count} ключ(ей)."
    if event_type == "generator_full_on_new_key":
        return "<b>Генератор уже переполнен</b> - собери ключ и открой кейс, чтобы генератор заработал!"
    if event_type == "extra_arena_modifier_changed":
        label = escape(str(payload.get("label") or payload.get("mode_name") or "новый модификатор"))
        return f"В ExtraArena сменился модификатор: {label}! Ну что, задашь им жару?"
    if event_type == "friendly_battle_invite":
        from_name = escape(str(payload.get("from_name") or "Друг"))
        return f"{from_name} вызывает тебя на дружеский бой!"
    if event_type == "friend_request_received":
        from_name = escape(str(payload.get("from_name") or "Игрок"))
        return f"{from_name} отправил заявку в друзья."
    if event_type == "daily_reminder":
        return escape(str(payload.get("text") or "Пора вернуться на арену!"))
    if event_type == "squad_member_role":
        nick = escape(str(payload.get("nick") or "Участник"))
        action = escape(str(payload.get("action") or "изменил роль"))
        return f"{nick}: {action} в твоем скваде."
    if event_type == "squad_new_member":
        nick = escape(str(payload.get("nick") or "Новый игрок"))
        squad = escape(str(payload.get("squad_name") or "сквад"))
        return f"{nick} вступил в {squad}!"
    if event_type == "squad_disbanded":
        squad = escape(str(payload.get("squad_name") or "Твой сквад"))
        return f"{squad} расформирован."
    if event_type == "squad_boost":
        squad = escape(str(payload.get("squad_name") or "У сквада"))
        return f"{squad} активировал Boost!"
    return escape(format_notification_message(event_type, payload))


def format_android_notification_title(category: str, event_type: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    if payload.get("title"):
        return str(payload.get("title"))
    if event_type == "generator_new_key":
        return "Новый ключ готов!"
    if event_type == "generator_full_on_new_key":
        return "Новый ключ готов!"
    if event_type == "generator_full_blocked_key":
        return "Генератор переполнен!"
    if event_type == "shop_particles":
        return "Новые частицы карт в магазине!"
    if event_type == "extra_arena_modifier_changed":
        label = payload.get("label") or payload.get("mode_name") or "ExtraArena"
        return f"{label} в ExtraArena!"
    if event_type == "friendly_battle_invite":
        from_name = payload.get("from_name") or "Друг"
        return f"{from_name} вызвал тебя на бой!"
    if event_type == "friend_request_received":
        return "Заявка в друзья"
    if event_type == "daily_reminder":
        return random.choice(REMINDER_TITLES)
    if category.startswith("squad_") or event_type.startswith("squad_"):
        squad = payload.get("squad_name") or "Сквад"
        event_labels = {
            "squad_new_member": "новый участник!",
            "squad_member_role": "роль изменена!",
            "squad_disbanded": "расформирован!",
            "squad_boost": "Boost!",
        }
        return f"{squad}: {event_labels.get(event_type, 'событие!')}"
    return "ExtraArena"


def notification_section(category: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    return str(payload.get("section") or WEBAPP_SECTION_BY_CATEGORY.get(category) or "arena")


def build_webapp_url(base_url: str, *, section: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    query = {"section": section}
    for key in ("invite_id", "invite_action", "request_id"):
        if payload.get(key) is not None:
            query[key] = str(payload.get(key))
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{urlencode(query)}"
