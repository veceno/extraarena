"""Smoke-тесты для RLHF-среды.

Покрывают критические компоненты:
- log_schema: валидация battle_log/manifest
- deck_builder: загрузка каталога, генерация, парсинг
- policy_factory: V4/Baselines, ошибки
- battle_runner: 1 бой end-to-end с записью battle_log
- manifest: финализация, winrate
- session_manager: старт/стоп/статус

Запуск:
    python3 -m pytest rlhf_env/tests/ -v
    python3 -m pytest rlhf_env/tests/test_battle_runner.py -v
"""