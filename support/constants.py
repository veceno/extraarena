from __future__ import annotations


class SupportTopic:
    ACCOUNT = "account"
    PAYMENTS = "payments"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    OTHER = "other"

    ALL = {ACCOUNT, PAYMENTS, TECHNICAL, COMPLAINT, OTHER}


class AccountScope:
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    GUEST = "guest"
    OWN = "own"
    OTHER = "other"


class SupportStatus:
    OPEN = "open"
    QUEUED_UNVERIFIED = "queued_unverified"
    QUEUED_GUEST = "queued_guest"
    PENDING_ADMIN = "pending_admin"
    CLOSED = "closed"


class SupportChannel:
    TELEGRAM = "telegram"
    MAX = "max"
    SITE = "site"


MESSAGE_INBOUND = "inbound"
MESSAGE_OUTBOUND = "outbound"
MESSAGE_INTERNAL = "internal"
