from pathlib import Path


SERVER = Path("web/server.py")


def test_checkout_urls_do_not_put_signed_token_in_query_string():
    server = SERVER.read_text(encoding="utf-8")

    assert '/extraShop?checkout={token}' not in server
    assert '/extraShop#checkout={token}' not in server
    assert '/extraShop#checkout_jti=' in server
    assert '"token": token' not in server
