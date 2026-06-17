from pathlib import Path


WEBAPP_INDEX = Path(__file__).resolve().parents[1] / "webapp" / "index.html"


def test_shop_screen_contains_rustore_payment_flow_hooks():
    html = WEBAPP_INDEX.read_text(encoding="utf-8")

    assert "/api/payments/rustore/create" in html
    assert "/api/payments/rustore/attach" in html
    assert "/api/payments/rustore/complete" in html
    assert "startRuStorePayment" in html
    assert "ExtraArenaRuStorePayment" in html


def test_payment_success_provider_detection_includes_rustore_ids():
    html = WEBAPP_INDEX.read_text(encoding="utf-8")

    assert "startsWith('rustore_') ? 'rustore'" in html


def test_rustore_payment_events_are_deduplicated_by_terminal_key():
    html = WEBAPP_INDEX.read_text(encoding="utf-8")
    block = html.split("async function handleRuStorePaymentEvent", 1)[1].split(
        "window.ExtraArenaRuStorePaymentEvent",
        1,
    )[0]

    assert "const ruStoreHandledEventKeys = new Map();" in html
    assert "function shouldHandleRuStorePaymentEvent(event)" in html
    assert "shouldHandleRuStorePaymentEvent(event)" in block
    assert "event.payment_id || event.order_id || event.invoice_id" in html
