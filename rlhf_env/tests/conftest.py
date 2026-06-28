"""Конftest для in-process тестов rlhf_env.

autouse-фикстура убирает реальные bot-turn delay'и (classic.bot_turn_delay_range
по умолчанию 4–6с на ход) — для in-process run_auto они только тормозят тесты,
прод-семантика (WS-broadcast pacing) здесь не нужна. Прод-код не меняется:
патчится только атрибут ``asyncio`` модуля match_runner на время теста.
"""
from __future__ import annotations

import types

import pytest

import rlhf_env.components.match_runner as _mr


@pytest.fixture(autouse=True)
def _fast_bot_sleep():
    real = _mr.asyncio
    _mr.asyncio = types.SimpleNamespace(
        sleep=lambda d: real.sleep(0),
        create_task=real.create_task,
        Lock=real.Lock,
    )
    try:
        yield
    finally:
        _mr.asyncio = real