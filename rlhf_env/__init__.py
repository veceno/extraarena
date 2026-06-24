"""RLHF-среда ExtraArena.

Автономная среда для сбора human-vs-model данных, расположенная в
отдельной директории rlhf_env/. Не зависит от прод-БД, бота и прод-веба.

Запуск:
    ./rlhf_env/start_rlhf_env.sh                # web @ :8090
    ./rlhf_env/start_rlhf_env.sh mcp            # MCP stdio
    python -m rlhf_env.server --port 8090       # прямой запуск web

См. rlhf_env/README.md и rlhf_env/DOCS.md.
"""

__version__ = "0.1.0"
