"""
Тестовый конфиг добавляет корень репозитория в sys.path.
Так pytest всегда найдёт модули вроде battle_engine при импортировании.
"""

import sys
from pathlib import Path

import pytest

# Явно добавляем путь к корню проекта в начало sys.path.
# Это устраняет ModuleNotFoundError при запуске тестов из любой директории.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mocker():
    """
    Локальный аналог фикстуры из pytest-mock, чтобы не тянуть зависимость.
    Возвращаем модуль unittest.mock, у которого есть patch/patch.dict и Mock.
    """
    from unittest import mock

    return mock







