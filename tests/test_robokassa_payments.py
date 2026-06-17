import hashlib
import json
from urllib.parse import parse_qs, unquote, urlparse

from infrastructure.config import get_settings
from infrastructure.robokassa_payments import RobokassaPaymentService, RobokassaSettings


def test_robokassa_test_payment_includes_receipt_and_istest_in_signed_form():
    service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="pass1",
            password2="pass2",
            hash_algo="sha256",
            test_mode=True,
            result_url="https://laveqox.ru/api/payments/robokassa/result",
            success_url="https://laveqox.ru/extraShop/payment-success",
            fail_url="https://laveqox.ru/extraShop/payment-fail",
        )
    )

    result = service.create_payment(
        amount=179,
        currency="RUB",
        description="ExtraPass",
        return_url="https://game.example",
        metadata={"item_name": "ExtraPass", "item_type": "extrapass"},
        inv_id=12345,
        payment_page_url="https://laveqox.ru/api/payments/robokassa/pay/robokassa_12345",
    )

    assert result["success"] is True
    assert result["payment_id"] == "robokassa_12345"
    assert result["confirmation_url"] == result["payment_url"]
    assert result["payment_page_url"] == "https://laveqox.ru/api/payments/robokassa/pay/robokassa_12345"
    assert result["status"] == "pending"

    form = result["form"]
    assert form["MerchantLogin"] == "extraarena"
    assert form["OutSum"] == "179.00"
    assert form["InvId"] == "12345"
    assert form["IsTest"] == "1"
    assert form["Culture"] == "ru"
    assert form["Encoding"] == "utf-8"
    assert form["SuccessUrl2"] == "https://laveqox.ru/extraShop/payment-success"
    assert form["SuccessUrl2Method"] == "GET"
    assert form["FailUrl2"] == "https://laveqox.ru/extraShop/payment-fail"
    assert form["FailUrl2Method"] == "GET"

    receipt = json.loads(unquote(form["Receipt"]))
    assert receipt["items"] == [
        {
            "name": "ExtraPass",
            "quantity": 1,
            "sum": 179.0,
            "tax": "none",
            "payment_method": "full_payment",
            "payment_object": "service",
        }
    ]

    expected = hashlib.sha256(
        (
            f"extraarena:179.00:12345:{form['Receipt']}:"
            "https://laveqox.ru/extraShop/payment-success:GET:"
            "https://laveqox.ru/extraShop/payment-fail:GET:"
            "pass1:Shp_payment_id=robokassa_12345"
        ).encode()
    ).hexdigest()
    assert form["SignatureValue"] == expected
    assert form["Shp_payment_id"] == "robokassa_12345"


def test_robokassa_result_signature_uses_password2_and_shp_parameters():
    service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="pass1",
            password2="pass2",
            hash_algo="sha256",
            test_mode=True,
        )
    )
    payload = {
        "OutSum": "179.00",
        "InvId": "12345",
        "Shp_payment_id": "robokassa_12345",
    }
    payload["SignatureValue"] = hashlib.sha256(
        "179.00:12345:pass2:Shp_payment_id=robokassa_12345".encode()
    ).hexdigest().upper()

    parsed = service.parse_result_notification(payload)

    assert parsed == {
        "success": True,
        "payment_id": "robokassa_12345",
        "inv_id": "12345",
        "amount": 179.0,
        "currency": "RUB",
        "status": "succeeded",
        "paid": True,
    }


def test_robokassa_success_signature_uses_password1_and_shp_parameters():
    service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="pass1",
            password2="pass2",
            hash_algo="sha256",
            test_mode=True,
        )
    )
    payload = {
        "OutSum": "179.00",
        "InvId": "12345",
        "Shp_payment_id": "robokassa_12345",
    }
    payload["SignatureValue"] = hashlib.sha256(
        "179.00:12345:pass1:Shp_payment_id=robokassa_12345".encode()
    ).hexdigest().upper()

    parsed = service.parse_success_redirect(payload)

    assert parsed == {
        "success": True,
        "payment_id": "robokassa_12345",
        "inv_id": "12345",
        "amount": 179.0,
        "currency": "RUB",
        "status": "succeeded",
        "paid": True,
    }


def test_robokassa_payment_url_contains_no_passwords():
    service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="pass1",
            password2="pass2",
            hash_algo="sha256",
            test_mode=True,
        )
    )

    result = service.create_payment(
        amount=49,
        currency="RUB",
        description="Starter gems",
        return_url="https://game.example",
        metadata={"item_name": "Starter gems"},
        inv_id=777,
        payment_page_url="https://laveqox.ru/api/payments/robokassa/pay/robokassa_777",
    )
    parsed = urlparse(result["payment_url"])
    query = parse_qs(parsed.query)

    assert parsed.netloc == "auth.robokassa.ru"
    assert "pass1" not in result["payment_url"]
    assert "pass2" not in result["payment_url"]
    assert query["IsTest"] == ["1"]
    assert query["MerchantLogin"] == ["extraarena"]


def test_settings_prefers_robokassa_primary_when_configured_without_explicit_provider(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ROBOKASSA_MERCHANT_LOGIN", "extraarena")
    monkeypatch.setenv("ROBOKASSA_TEST_PASSWORD1", "test-pass-1")
    monkeypatch.setenv("ROBOKASSA_TEST_PASSWORD2", "test-pass-2")
    monkeypatch.setenv("ROBOKASSA_TEST_MODE", "true")
    monkeypatch.delenv("PAYMENT_PRIMARY_PROVIDER", raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.robokassa is not None
        assert settings.payment_primary_provider == "robokassa"
        assert settings.payment_fallback_provider == "yookassa"
    finally:
        get_settings.cache_clear()
