from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from infrastructure.database import Database


ROOT = Path(__file__).resolve().parents[1]


class ShopSetDBHarness(Database):
    def __init__(self, rows=None, row=None):
        self._pool = object()
        self.rows = list(rows or [])
        self.row = row

    async def fetch(self, _query):
        return self.rows

    async def fetchrow(self, _query, _set_id):
        return self.row


@pytest.mark.asyncio
async def test_shop_sets_are_json_safe():
    created_at = datetime(2026, 5, 24, 12, 30, tzinfo=timezone.utc)
    db = ShopSetDBHarness(
        rows=[
            {
                "id": 1,
                "price": Decimal("199.90"),
                "created_at": created_at,
                "rewards": [{"type": "gems", "amount": Decimal("5")}],
            }
        ],
        row={
            "id": 2,
            "price": Decimal("10"),
            "updated_at": created_at,
        },
    )

    sets = await db.get_shop_sets(active_only=False)
    one_set = await db.get_shop_set(2)

    assert sets[0]["price"] == 199.9
    assert sets[0]["created_at"] == created_at.isoformat()
    assert sets[0]["rewards"][0]["amount"] == 5.0
    assert one_set["price"] == 10.0
    assert one_set["updated_at"] == created_at.isoformat()


def test_hidden_admin_frontend_uses_stable_set_editor_state():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "editingSet=null" in html
    assert "async function saveSet(pub){\n        var s=editingSet;" in html
    assert "function(){var s=editingSet;if(!s)return;" in html
    assert "if(s)body.set_id" in html


def test_hidden_admin_bot_analytics_accepts_backend_field_names():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "botByMode" in html
    assert "m.bot_wins!=null?m.bot_wins:m.wins" in html
    assert "m.bot_losses!=null?m.bot_losses:m.losses" in html


def test_profile_admin_visibility_contract_is_wired():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    index = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")

    assert '"is_admin": await _is_admin_user(db, user_id),' in server
    assert "profile?.is_admin === true || profile?.user_id === ADMIN_ID" in index


def test_promocode_admin_list_is_global():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert "promocodes = await db.get_promocodes_list()" in server
    assert "promocodes = await db.get_promocodes_list(created_by=user_id)" not in server


def test_hidden_admin_frontend_keeps_auth_out_of_api_urls():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "opts.headers.Authorization='Bearer '+AUTH_PARAM.value" in html
    assert "history.replaceState" in html
    assert "u.searchParams.set(AUTH_PARAM.key,AUTH_PARAM.value)" not in html
    assert "p.get('user_id')" not in html
    assert "if(!AUTH_PARAM)" not in html


def test_hidden_admin_frontend_avoids_inline_player_onclick_and_raw_toast_html():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "e.textContent=m" in html
    assert "e.innerHTML=m" not in html
    assert "data-player-open" in html
    assert "onclick=\"openPlayerDetail" not in html


def test_product_editor_uses_guided_product_selectors():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert '<select id="pe-item-type"' in html
    assert '<input id="pe-item-type"' not in html
    assert '<select id="pe-package-type"' in html
    assert '<select id="pe-shop-set-id"' in html
    assert "var productOptions=" in html
    assert "loadProductOptions" in html
    assert "/api/admin/ruble-products/options" in html


def test_admin_frontend_displays_partial_block_warnings():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "renderAdminBlockWarnings" in html
    assert "cfg-partial-warnings" in html
    assert "sq-partial-warnings" in html


def test_legacy_admin_players_post_route_is_not_registered():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert 'app.router.add_get("/api/admin/players", admin_players_handler)' in server
    assert 'app.router.add_post("/api/admin/players", admin_players_handler)' not in server


def test_admin_routes_have_central_auth_middleware():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert "async def admin_auth_middleware" in server
    assert 'request.path.startswith("/api/admin/")' in server
    assert "app.middlewares.append(admin_auth_middleware)" in server


def test_admin_adjust_resource_uses_explicit_column_map():
    database = (ROOT / "infrastructure" / "database.py").read_text(encoding="utf-8")

    assert "resource_columns = {" in database
    assert 'f"SELECT {resource} FROM users WHERE user_id = $1"' not in database
