from pathlib import Path


INDEX = Path("webapp/index.html")


def test_real_money_success_modal_waits_for_processed_rewards():
    source = INDEX.read_text(encoding="utf-8")
    status_block = source.split("async function triggerPaymentSuccessFromStatus", 1)[1].split(
        "async function triggerPaymentSuccessFromRecent",
        1,
    )[0]
    recent_block = source.split("async function triggerPaymentSuccessFromRecent", 1)[1].split(
        "function normalizeRuStorePaymentEvent",
        1,
    )[0]

    assert "status.rewards_processed !== true" in status_block
    assert "p?.rewards_processed === true" in recent_block


def test_checkout_session_polling_keeps_pending_until_rewards_processed():
    source = INDEX.read_text(encoding="utf-8")
    checkout_poll = source.split("var startAppCheckoutSessionCheck = function(jti, provider)", 1)[1].split(
        "var doStars",
        1,
    )[0]
    app_visibility_poll = source.split("const pendingCheckoutJti = sessionStorage.getItem('pending_checkout_jti');", 1)[1].split(
        "await triggerPaymentSuccessFromRecent(authData);",
        1,
    )[0]

    assert "d.payment_id" in checkout_poll
    assert "startAppPaymentStatusCheck(d.payment_id" in checkout_poll
    assert "d.rewards_processed === true" in checkout_poll
    assert "session.rewards_processed === true" in app_visibility_poll
    assert "'/api/payments/status?payment_id=' + encodeURIComponent(session.payment_id)" in app_visibility_poll
    assert "sessionStorage.setItem('pending_payment_method', session.provider" in app_visibility_poll
