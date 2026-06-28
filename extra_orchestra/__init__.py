"""ExtraOrchestra — scenario replay + mp4 recording utility for ExtraArena.

Наследует арену (webapp/arena.*) verbatim по образцу rlhf_env/: frozen-снапшот
в ``extra_orchestra/webapp_borrow/``, aiohttp-сервер реплицирует HTTP/Socket.IO
контракт, сериализаторы скопированы из ``battle_engine.py``. Поверх — сценарный
движок (core.engine.ArenaEnvironment), визуальный node-редактор, предпросмотр и
экспорт mp4 (Playwright + ffmpeg).

См. DOCS.md и /Users/laveqox/.claude/plans/quizzical-soaring-rain.md.
"""

__version__ = "0.1.0"