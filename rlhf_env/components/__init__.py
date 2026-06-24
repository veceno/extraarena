"""Базовые компоненты RLHF-среды.

battle_runner      — запуск одного боя с записью лога
policy_factory     — адаптер сторонних политик поверх ai.model_benchmark.create_policy
policy_registry    — реестр доступных ONNX-моделей из указанной директории
deck_builder       — генерация и парсинг колод
manifest           — запись манифеста группы боёв
inference_params   — дефолты инференса из sidecar
log_schema         — схема battle_log.json
session_manager    — asyncio-реестр активных групп
"""
