from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from support.constants import AccountScope, SupportChannel, SupportTopic


TOPIC_OPTIONS: tuple[tuple[str, str], ...] = (
    (SupportTopic.ACCOUNT, "Аккаунт / вход"),
    (SupportTopic.PAYMENTS, "Платежи"),
    (SupportTopic.TECHNICAL, "Техническая проблема"),
    (SupportTopic.COMPLAINT, "Жалоба"),
    (SupportTopic.OTHER, "Другое"),
)

ACCOUNT_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = (
    (AccountScope.OWN, "Мой аккаунт"),
    (AccountScope.OTHER, "Другой аккаунт"),
    (AccountScope.GUEST, "Не могу войти / без ID"),
)


@dataclass(frozen=True)
class BotChoice:
    id: str
    label: str


@dataclass(frozen=True)
class BotReply:
    text: str
    choices: tuple[BotChoice, ...] = ()
    ticket: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None


class SupportBotConversationManager:
    """Small shared state machine for support bot clients.

    Only the onboarding state is in memory. Once a ticket exists, the active chat
    target is loaded from the support database by channel/channel_id.
    """

    def __init__(self, support_service: Any) -> None:
        self.support_service = support_service
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}

    def session_for(self, *, channel: str, channel_id: Any) -> dict[str, Any] | None:
        return self._sessions.get((str(channel), str(channel_id)))

    def clear_session(self, *, channel: str, channel_id: Any) -> None:
        self._sessions.pop((str(channel), str(channel_id)), None)

    async def begin(
        self,
        *,
        channel: str,
        channel_id: Any,
        display_name: str,
        external_user_id: str,
    ) -> BotReply:
        self._sessions[(str(channel), str(channel_id))] = {
            "step": "topic",
            "channel": str(channel),
            "channel_id": str(channel_id),
            "display_name": str(display_name or "Player"),
            "external_user_id": str(external_user_id or f"{channel}:{channel_id}"),
        }
        return self.topic_prompt()

    def topic_prompt(self) -> BotReply:
        return BotReply(
            "Выберите тему обращения:",
            choices=tuple(BotChoice(f"topic:{topic}", label) for topic, label in TOPIC_OPTIONS),
        )

    def account_scope_prompt(self) -> BotReply:
        return BotReply(
            "Обращение про ваш аккаунт или про другой?",
            choices=tuple(BotChoice(f"scope:{scope}", label) for scope, label in ACCOUNT_SCOPE_OPTIONS),
        )

    async def select_topic(self, *, channel: str, channel_id: Any, topic: str) -> BotReply:
        session = self._sessions.get((str(channel), str(channel_id)))
        if not session:
            return self.topic_prompt()
        if topic not in SupportTopic.ALL:
            return BotReply("Такой темы нет. Выберите одну из кнопок ниже.", self.topic_prompt().choices)
        session["topic"] = topic
        session["step"] = "account_scope"
        return self.account_scope_prompt()

    async def select_account_scope(self, *, channel: str, channel_id: Any, scope: str) -> BotReply:
        session = self._sessions.get((str(channel), str(channel_id)))
        if not session:
            return self.topic_prompt()
        if scope not in {AccountScope.OWN, AccountScope.OTHER, AccountScope.GUEST}:
            return BotReply("Выберите вариант аккаунта кнопкой ниже.", self.account_scope_prompt().choices)
        session["account_relation"] = scope
        if scope == AccountScope.GUEST:
            session["step"] = "body"
            session["claimed_account"] = ""
            return BotReply("Опишите проблему одним сообщением. Можно приложить детали, номер заказа или ник.")
        session["step"] = "account_id"
        if scope == AccountScope.OWN:
            return BotReply(
                "Напишите ваш игровой ID или ExtraID. Если сейчас не можете его узнать, напишите «нет»."
            )
        return BotReply("Напишите игровой ID / ExtraID другого аккаунта. Если ID неизвестен, напишите «нет».")

    async def receive_text(
        self,
        *,
        channel: str,
        channel_id: Any,
        external_user_id: str,
        display_name: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> BotReply:
        text = str(body or "").strip()
        key = (str(channel), str(channel_id))
        lowered = text.lower()
        if lowered in {"/start", "start", "начать"}:
            return await self.begin(
                channel=channel,
                channel_id=channel_id,
                display_name=display_name,
                external_user_id=external_user_id,
            )
        if lowered in {"/new", "новое", "новое обращение"}:
            self.clear_session(channel=channel, channel_id=channel_id)
            return await self.begin(
                channel=channel,
                channel_id=channel_id,
                display_name=display_name,
                external_user_id=external_user_id,
            )

        session = self._sessions.get(key)
        if not session:
            active = await self.support_service.get_active_ticket_for_channel(
                channel=channel,
                channel_id=str(channel_id),
            )
            if active:
                message = await self.support_service.add_inbound_channel_message(
                    ticket=active,
                    body=text,
                    metadata=metadata or {},
                )
                return BotReply(
                    "Сообщение добавлено в обращение. Поддержка ответит здесь же.\n\n"
                    "Чтобы создать отдельное обращение, отправьте /new.",
                    ticket=active,
                    message=message,
                )
            matched_topic = _match_topic(text)
            await self.begin(
                channel=channel,
                channel_id=channel_id,
                display_name=display_name,
                external_user_id=external_user_id,
            )
            if matched_topic:
                return await self.select_topic(channel=channel, channel_id=channel_id, topic=matched_topic)
            return self.topic_prompt()

        step = str(session.get("step") or "topic")
        if step == "topic":
            matched_topic = _match_topic(text)
            if matched_topic:
                return await self.select_topic(channel=channel, channel_id=channel_id, topic=matched_topic)
            return BotReply("Выберите тему кнопкой или отправьте номер темы.", self.topic_prompt().choices)
        if step == "account_scope":
            matched_scope = _match_account_scope(text)
            if matched_scope:
                return await self.select_account_scope(channel=channel, channel_id=channel_id, scope=matched_scope)
            return BotReply("Выберите, про какой аккаунт обращение.", self.account_scope_prompt().choices)
        if step == "account_id":
            session["claimed_account"] = "" if lowered in {"нет", "no", "skip", "-"} else text
            session["step"] = "body"
            return BotReply("Теперь опишите проблему одним сообщением.")
        if step == "body":
            if not text:
                return BotReply("Нужно текстом описать проблему, чтобы я создал обращение.")
            result = await self._create_ticket_from_session(session, text, metadata=metadata or {})
            self.clear_session(channel=channel, channel_id=channel_id)
            ticket = result.get("ticket") or {}
            return BotReply(
                "Обращение создано.\n"
                f"Номер: {ticket.get('id', 'support')}\n\n"
                "Пишите сюда дополнительные сообщения — они попадут в этот же чат поддержки. "
                "Для нового обращения отправьте /new.",
                ticket=ticket,
                message=result.get("message"),
                identity=result.get("identity"),
            )
        return self.topic_prompt()

    async def _create_ticket_from_session(
        self,
        session: dict[str, Any],
        body: str,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        account_relation = str(session.get("account_relation") or AccountScope.GUEST)
        account_scope = AccountScope.GUEST if account_relation == AccountScope.GUEST else AccountScope.UNVERIFIED
        merged_metadata = {
            **metadata,
            "source": str(session.get("channel") or ""),
            "account_relation": account_relation,
            "claimed_account": str(session.get("claimed_account") or ""),
        }
        return await self.support_service.create_ticket(
            topic=str(session.get("topic") or SupportTopic.OTHER),
            channel=str(session.get("channel") or SupportChannel.SITE),
            channel_id=str(session.get("channel_id") or ""),
            external_user_id=str(session.get("external_user_id") or ""),
            display_name=str(session.get("display_name") or "Player"),
            body=body,
            game_user_id=None,
            account_scope=account_scope,
            subject=body[:120],
            metadata=merged_metadata,
        )


def format_text_menu(reply: BotReply) -> str:
    if not reply.choices:
        return reply.text
    lines = [reply.text, ""]
    for index, choice in enumerate(reply.choices, start=1):
        lines.append(f"{index}. {choice.label}")
    return "\n".join(lines)


def _match_topic(text: str) -> str | None:
    normalized = text.strip().lower()
    for index, (topic, label) in enumerate(TOPIC_OPTIONS, start=1):
        if normalized in {str(index), topic, label.lower()}:
            return topic
    return None


def _match_account_scope(text: str) -> str | None:
    normalized = text.strip().lower()
    aliases = {
        "1": AccountScope.OWN,
        AccountScope.OWN: AccountScope.OWN,
        "мой": AccountScope.OWN,
        "мой аккаунт": AccountScope.OWN,
        "2": AccountScope.OTHER,
        AccountScope.OTHER: AccountScope.OTHER,
        "другой": AccountScope.OTHER,
        "другой аккаунт": AccountScope.OTHER,
        "3": AccountScope.GUEST,
        AccountScope.GUEST: AccountScope.GUEST,
        "нет": AccountScope.GUEST,
        "без id": AccountScope.GUEST,
        "не могу войти": AccountScope.GUEST,
    }
    return aliases.get(normalized)
