from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_support_wiring_uses_separate_support_bots_not_game_handlers():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    run_web = (ROOT / "run_web.py").read_text(encoding="utf-8")
    game_handlers = (ROOT / "bot" / "handlers.py").read_text(encoding="utf-8")

    assert "SupportDatabase" in main
    assert "create_support_bot" in main
    assert "support_telegram_bot_token" in main
    assert "SupportDatabase" in run_web
    assert "SupportDeliveryDispatcher" in run_web
    assert "SupportAdminNotifier" in run_web
    assert "MaxSupportClient" in run_web
    assert "register_support_routes" in (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert "create_support_bot" not in game_handlers
