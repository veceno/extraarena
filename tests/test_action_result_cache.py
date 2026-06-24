"""
Unit-тесты для ACTION_RESULT_CACHE (web.server).

H4 fix: failed action cache poisoning.
Раньше _action_cache_set сохранял ответы с любым status, включая
4xx (turn_expired, not_your_turn, game_already_ended). Клиент с
повторно отправленным client_action_id получал устаревший error
снимок даже после того, как состояние матча изменилось и ошибка
была уже неактуальна — игрок не мог продолжить до истечения TTL.
Фикс: _action_cache_set отбрасывает любые ответы со status >= 400,
кэшируя только успешные (status < 400).
"""
import pytest

import web.server as web_server


@pytest.fixture(autouse=True)
def _clear_cache():
    """Каждый тест начинается с чистого кэша."""
    web_server.ACTION_RESULT_CACHE.clear()
    yield
    web_server.ACTION_RESULT_CACHE.clear()


class TestActionResultCacheH4:
    def test_successful_response_is_cached(self):
        """status=200 → запись попадает в кэш."""
        web_server._action_cache_set(
            match_id="m1", user_id=1, client_action_id="cid-1",
            payload={"success": True, "state": {}}, status=200,
        )
        cached = web_server._action_cache_get("m1", 1, "cid-1")
        assert cached is not None
        assert cached["status"] == 200
        assert cached["payload"]["success"] is True

    def test_409_response_is_not_cached(self):
        """H4 fix: 4xx (turn_expired / not_your_turn) НЕ попадают в кэш."""
        web_server._action_cache_set(
            match_id="m2", user_id=1, client_action_id="cid-2",
            payload={"error": "turn_expired"}, status=409,
        )
        cached = web_server._action_cache_get("m2", 1, "cid-2")
        assert cached is None, (
            "Failed-action cache poisoning: 409 записан в кэш, "
            "повтор того же client_action_id вернёт устаревший turn_expired."
        )

    def test_400_response_is_not_cached(self):
        web_server._action_cache_set(
            match_id="m3", user_id=1, client_action_id="cid-3",
            payload={"error": "bad_request"}, status=400,
        )
        assert web_server._action_cache_get("m3", 1, "cid-3") is None

    def test_403_response_is_not_cached(self):
        web_server._action_cache_set(
            match_id="m4", user_id=1, client_action_id="cid-4",
            payload={"error": "unauthorized"}, status=403,
        )
        assert web_server._action_cache_get("m4", 1, "cid-4") is None

    def test_500_response_is_not_cached(self):
        """H4 (расширение): 5xx тоже не должны кэшироваться — это
        transient/internal errors, не retryable payload."""
        web_server._action_cache_set(
            match_id="m5", user_id=1, client_action_id="cid-5",
            payload={"error": "internal"}, status=500,
        )
        assert web_server._action_cache_get("m5", 1, "cid-5") is None

    def test_no_client_action_id_means_no_cache_at_all(self):
        """Совместимость: без client_action_id ни успех, ни ошибка
        не должны попадать в кэш."""
        web_server._action_cache_set(
            match_id="m6", user_id=1, client_action_id=None,
            payload={"ok": True}, status=200,
        )
        assert web_server._action_cache_get("m6", 1, None) is None

    def test_cache_is_per_client_action_id(self):
        """Разные client_action_id изолированы друг от друга."""
        web_server._action_cache_set(
            match_id="m7", user_id=1, client_action_id="cid-a",
            payload={"success": True, "tag": "a"}, status=200,
        )
        web_server._action_cache_set(
            match_id="m7", user_id=1, client_action_id="cid-b",
            payload={"success": True, "tag": "b"}, status=200,
        )
        assert web_server._action_cache_get("m7", 1, "cid-a")["payload"]["tag"] == "a"
        assert web_server._action_cache_get("m7", 1, "cid-b")["payload"]["tag"] == "b"

    def test_recovery_after_failed_action_does_not_return_stale_error(self):
        """H4 fix: интеграционный сценарий — клиент отправил действие,
        получил 409 (не его ход), затем ход перешёл к нему, он
        отправляет тот же client_action_id и должен получить НОВЫЙ
        ответ, а не устаревший 409 из кэша."""
        # Сначала клиент получил turn_expired (его ход истёк).
        web_server._action_cache_set(
            match_id="m8", user_id=1, client_action_id="cid-retry",
            payload={"error": "turn_expired"}, status=409,
        )
        # Затем клиент снова отправляет то же действие (повтор после
        # обновления состояния). Сервер успешно его обработал и
        # закэшировал успех.
        web_server._action_cache_set(
            match_id="m8", user_id=1, client_action_id="cid-retry",
            payload={"success": True, "new_state": "x"}, status=200,
        )
        cached = web_server._action_cache_get("m8", 1, "cid-retry")
        # Должен вернуться success, не stale turn_expired.
        assert cached is not None
        assert cached["status"] == 200
        assert cached["payload"]["success"] is True