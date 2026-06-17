"""
Конфигурация матчмейкера в одном месте, чтобы числа не расползались по коду.
"""

# Жесткий дедлайн, после которого игрок гарантированно получает бота
MM_BOT_TIMEOUT: int = 15
# UX-задержка для low-trophy soft-start, чтобы мгновенный бот не выглядел подозрительно
MM_SOFT_START_BOT_DELAY_RANGE: tuple[float, float] = (2.0, 4.0)
# Шаги «расширяющегося» поиска по трофеям
SEARCH_WINDOWS: tuple[int, int, int] = (50, 200, 500)
# Периодичность повторных попыток подбора соперника
QUEUE_POLL_INTERVAL: int = 3

__all__ = [
    "MM_BOT_TIMEOUT",
    "MM_SOFT_START_BOT_DELAY_RANGE",
    "SEARCH_WINDOWS",
    "QUEUE_POLL_INTERVAL",
]






