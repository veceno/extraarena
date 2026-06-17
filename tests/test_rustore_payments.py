import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from infrastructure.rustore_payments import (
    RuStoreInvoiceVerifier,
    RuStorePaymentSettings,
    RuStorePublicTokenProvider,
    resolve_rustore_product_id,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "fake"

    def json(self):
        return self._payload


def confirmed_invoice_payload(**overrides):
    body = {
        "invoiceId": "123456",
        "invoiceStatus": "CONFIRMED",
        "developerPayload": "rustore_payment",
        "appId": 42,
        "paymentInfo": {},
        "purchaseId": "purchase-uuid",
        "order": {
            "orderId": "rustore_payment",
            "amountCreate": 9900,
            "amountCurrent": 9900,
            "currency": "RUB",
            "itemCode": "gems_100",
            "description": "100 gems",
        },
    }
    body.update(overrides)
    return {"code": "OK", "body": body}


def test_resolve_rustore_product_id_prefers_metadata_override():
    product = {"code": "gems_100", "metadata": {"rustore_product_id": "ea.gems.100"}}

    assert resolve_rustore_product_id(product) == "ea.gems.100"


def test_resolve_rustore_product_id_falls_back_to_product_code():
    product = {"code": "gems_100", "metadata": {}}

    assert resolve_rustore_product_id(product) == "gems_100"


def test_public_token_provider_signs_key_id_auth_request_and_caches_token():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_key_b64 = base64.b64encode(private_key_der).decode("ascii")
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        assert url.endswith("/public/auth")
        assert json["keyId"] == "2351028774"
        signed_message = f"{json['keyId']}{json['timestamp']}".encode("utf-8")
        private_key.public_key().verify(
            base64.b64decode(json["signature"]),
            signed_message,
            padding.PKCS1v15(),
            hashes.SHA512(),
        )
        return FakeResponse({"code": "OK", "body": {"jwe": "generated-token", "ttl": 900}})

    provider = RuStorePublicTokenProvider(
        RuStorePaymentSettings(key_id="2351028774", private_key=private_key_b64),
        http_post=fake_post,
        now=lambda: 1000.0,
    )

    assert provider.get_token() == "generated-token"
    assert provider.get_token() == "generated-token"
    assert len(calls) == 1
    assert calls[0][2]["Content-Type"] == "application/json"


def test_verify_invoice_confirms_matching_successful_payment():
    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return FakeResponse(confirmed_invoice_payload())

    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token", console_app_id="42", sandbox=True),
        http_get=fake_get,
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is True
    assert result["paid"] is True
    assert result["status"] == "succeeded"
    assert result["purchase_id"] == "purchase-uuid"
    assert calls[0][0].endswith("/public/sandbox/v2/invoices/123456")
    assert calls[0][1]["Public-Token"] == "token"


@pytest.mark.parametrize("invoice_status", ["CREATED", "EXECUTED", "PAID", "REFUNDING"])
def test_verify_invoice_keeps_non_final_statuses_pending(invoice_status):
    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token", console_app_id="42", sandbox=False),
        http_get=lambda *args, **kwargs: FakeResponse(
            confirmed_invoice_payload(invoiceStatus=invoice_status)
        ),
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is True
    assert result["paid"] is False
    assert result["status"] == "pending"


@pytest.mark.parametrize("invoice_status", ["CANCELLED", "REJECTED", "EXPIRED", "REVERSED", "REFUNDED"])
def test_verify_invoice_maps_terminal_failures_to_canceled(invoice_status):
    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token", console_app_id="42"),
        http_get=lambda *args, **kwargs: FakeResponse(
            confirmed_invoice_payload(invoiceStatus=invoice_status)
        ),
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is True
    assert result["paid"] is False
    assert result["status"] == "canceled"


@pytest.mark.parametrize(
    "payload, reason",
    [
        (confirmed_invoice_payload(order={"orderId": "other", "amountCurrent": 9900, "currency": "RUB", "itemCode": "gems_100"}), "order_id_mismatch"),
        (confirmed_invoice_payload(order={"orderId": "rustore_payment", "amountCurrent": 10000, "currency": "RUB", "itemCode": "gems_100"}), "amount_mismatch"),
        (confirmed_invoice_payload(order={"orderId": "rustore_payment", "amountCurrent": 9900, "currency": "USD", "itemCode": "gems_100"}), "currency_mismatch"),
        (confirmed_invoice_payload(order={"orderId": "rustore_payment", "amountCurrent": 9900, "currency": "RUB", "itemCode": "other"}), "product_id_mismatch"),
        (confirmed_invoice_payload(appId=7), "app_id_mismatch"),
    ],
)
def test_verify_invoice_rejects_mismatched_invoice_fields(payload, reason):
    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token", console_app_id="42"),
        http_get=lambda *args, **kwargs: FakeResponse(payload),
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is False
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "payload, reason",
    [
        (
            confirmed_invoice_payload(
                developerPayload="",
                order={"amountCurrent": 9900, "currency": "RUB", "itemCode": "gems_100"},
            ),
            "missing_order_id",
        ),
        (
            confirmed_invoice_payload(
                order={"orderId": "rustore_payment", "amountCurrent": 9900, "currency": "RUB"}
            ),
            "missing_product_id",
        ),
        (
            confirmed_invoice_payload(
                order={"orderId": "rustore_payment", "amountCurrent": 9900, "itemCode": "gems_100"}
            ),
            "missing_currency",
        ),
        (
            confirmed_invoice_payload(
                order={"orderId": "rustore_payment", "currency": "RUB", "itemCode": "gems_100"}
            ),
            "missing_amount",
        ),
    ],
)
def test_verify_invoice_requires_identity_fields(payload, reason):
    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token", console_app_id="42"),
        http_get=lambda *args, **kwargs: FakeResponse(payload),
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is False
    assert result["reason"] == reason


def test_verify_invoice_requires_public_token():
    verifier = RuStoreInvoiceVerifier(RuStorePaymentSettings(public_token=""))

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is False
    assert result["reason"] == "missing_public_token"


def test_verify_invoice_marks_provider_request_failure_retryable():
    def failing_get(*args, **kwargs):
        raise RuntimeError("temporary rustore outage")

    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token="token"),
        http_get=failing_get,
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is False
    assert result["reason"] == "provider_request_failed"
    assert result["retryable"] is True
    assert result["message"] == "RuStore provider request failed"
    assert "temporary rustore outage" not in str(result)


def test_verify_invoice_hides_public_token_exception_text():
    class FailingTokenProvider:
        def get_token(self):
            raise RuntimeError("secret rustore private key path")

    verifier = RuStoreInvoiceVerifier(
        RuStorePaymentSettings(public_token=""),
        token_provider=FailingTokenProvider(),
    )

    result = verifier.verify_invoice(
        invoice_id="123456",
        expected_payment_id="rustore_payment",
        expected_amount_rub=99,
        expected_currency="RUB",
        expected_product_id="gems_100",
    )

    assert result["success"] is False
    assert result["reason"] == "public_token_request_failed"
    assert result["retryable"] is True
    assert result["message"] == "RuStore Public API request failed"
    assert "secret rustore private key path" not in str(result)
