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


class AdminSearchHarness(Database):
    def __init__(self):
        self._pool = object()
        self.count_query = ""
        self.count_args = ()
        self.data_query = ""
        self.data_args = ()

    async def fetchval(self, query, *args):
        self.count_query = query
        self.count_args = args
        return 0

    async def fetch(self, query, *args):
        self.data_query = query
        self.data_args = args
        return []


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


@pytest.mark.asyncio
async def test_admin_player_search_uses_unified_sql_params_for_activity_filters():
    db = AdminSearchHarness()

    result = await db.search_admin_players(query="alice", activity="active_7d", limit=25, offset=5)

    assert result == {"players": [], "total": 0}
    assert "started_at >= $2" in db.count_query
    assert len(db.count_args) == 2
    assert "started_at >= $2" in db.data_query
    assert "started_at >= $3" in db.data_query
    assert "LIMIT $4 OFFSET $5" in db.data_query
    assert db.data_args[-2:] == (25, 5)


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
    assert "p.get('_auth')" not in html
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
    assert '<input id="pe-rustore-product-id"' in html
    assert "rustore_product_id" in html
    assert "var productOptions=" in html
    assert "loadProductOptions" in html
    assert "/api/admin/ruble-products/options" in html


def test_admin_shop_sets_render_content_aware_previews_and_creation_path():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert 'id="btn-create-shop-set"' in html
    assert "openSetEditor(null)" in html
    assert "function classifyShopSetPreview(set)" in html
    assert "function renderShopSetPreview(set)" in html
    assert "resources-only" in html
    assert "cosmetics-resources" in html
    assert "card-pack" in html
    assert "reward-card-media" in html
    assert "reward-cosmetic-chip" in html
    assert "type+(r.card_id?'#'+r.card_id:'')+':'+(r.amount||1)" not in html


def test_admin_shop_set_editor_supports_cosmetic_rewards():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "VALID_REWARD_TYPES=['gems','coins','keys','case','card','particles','cosmetic']" in html
    assert "var cosmeticCatalog=[]" in html
    assert "async function loadCosmetics()" in html
    assert 'data-add-reward="cosmetic"' in html
    assert "need cosmetic_slug" in html
    assert 'data-rk="cosmetic_slug"' in html
    assert '<select data-rf="\'+i+\'" data-rk="cosmetic_slug"' in html
    assert 'data-rk="auto_equip"' in html
    assert "bl.cosmetic_slug=''" in html
    assert "delete editorRewards[idx].slug" in html
    assert "_ri(t){return{gems:" in html and "cosmetic:" in html
    assert "editorRewards=JSON.parse(JSON.stringify(rewardList(s)))" in html
    assert "cosmeticCatalog.find(function(c)" in html
    assert "if(item.item_type==='profile_background')return'background'" in html


def test_admin_cosmetics_tab_is_wired():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert 'data-nav="cosmetics"' in html
    assert 'id="cosmetics-view"' in html
    assert 'id="cosmetics-tbody"' in html
    assert 'id="btn-create-cosmetic"' in html
    assert "'cosmetics-view'" in html
    assert "showView('cosmetics-view');loadCosmetics()" in html
    assert "/api/admin/cosmetics" in html
    assert "/api/admin/cosmetics/upload-image" in html
    assert "/api/admin/cosmetics/create" in html
    assert "/api/admin/cosmetics/delete" in html
    assert "asset&&c.item_type!=='title'&&c.is_active!==!1" in html


def test_bot_equipped_cosmetics_ignore_inactive_catalog_items():
    source = (ROOT / "infrastructure" / "database.py").read_text(encoding="utf-8")

    start = source.index("async def _get_bot_equipped_cosmetics")
    end = source.index("async def get_cosmetic_catalog_by_class")
    function_source = source[start:end]

    assert "JOIN cosmetic_items ci ON ci.id = uec.cosmetic_id" in function_source
    assert "AND ci.is_active = TRUE" in function_source


def test_admin_pack_preview_treats_particles_as_resources_not_cosmetics():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "r.type==='cosmetic'||r.type==='particles'" not in html
    assert "['gems','coins','keys','case','particles']" in html


def test_admin_product_editor_supports_gift_shop_set_semantics():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "value:'gift_shop_set'" in html
    assert "isGiftShopSetType(type)" in html
    assert "$('#pe-shop-set-field').style.display=isShopSetProductType(type)?'':'none'" in html
    assert "Gift / Free" in html
    assert "gift_shop_set_" in html
    assert "if(isShopSetProductType(it)&&!$('#pe-shop-set-id').value)" in html
    assert "else if(isShopSetProductType(it))" in html


def test_extra_pass_admin_reward_type_selectors_include_case_rewards():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    for select_id in ("sr-reward-type", "rt-reward-type"):
        select_markup = html.split(f'<select id="{select_id}">', 1)[1].split("</select>", 1)[0]
        for reward_type in ("coins", "gems", "keys", "case", "card", "specific_card", "particles", "cosmetic", "guaranteed_card"):
            assert f'<option value="{reward_type}"' in select_markup


def test_extra_pass_admin_exposes_season_reset_controls():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert 'data-season-tab="reset"' in html
    assert 'data-season-panel="reset"' in html
    assert 'id="btn-season-reset-preview"' in html
    assert 'id="btn-season-reset-execute"' in html
    assert "/api/admin/seasons/'+Number(seasonId)+'/reset-preview" in html
    assert "/api/admin/seasons/'+Number(seasonId)+'/reset" in html
    assert "seasonResetPreview=null" in html
    assert "confirm_season_id:Number(seasonId)" in html
    assert "Load a fresh reset preview first" in html
    assert "setSeasonResetBusy" in html
    assert "btn.disabled=seasonResetBusy" in html
    assert "shown of '+fmtNum(total)+' players" in html
    assert "setSwitch($('#season-reset-confirm'),false)" in html
    assert "Reset preview: '+escHtml(e.message)" not in html
    assert "Season reset: '+escHtml(e.message)" not in html
    assert 'app.router.add_get("/api/admin/seasons/{season_id:\\\\d+}/reset-preview", admin_season_reset_preview_handler)' in server
    assert 'app.router.add_post("/api/admin/seasons/{season_id:\\\\d+}/reset", admin_season_reset_execute_handler)' in server


def test_admin_frontend_displays_partial_block_warnings():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "renderAdminBlockWarnings" in html
    assert "cfg-partial-warnings" in html
    assert "sq-partial-warnings" in html


def test_admin_frontend_exposes_resumable_android_release_management():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert 'data-nav="releases"' in html
    assert 'id="releases-view"' in html
    assert 'id="release-required-switch"' in html
    assert 'id="release-progress-bar"' in html
    assert "file.slice(androidReleaseState.offset,end)" in html
    assert "'Upload-Offset':String(offset)" in html
    assert "X-Android-Upload-Token" in html
    assert "publishAndroidRelease" in html
    assert "Type RETIRE '+versionCode" in html
    assert "min_supported_version_code" in html
    assert "AAB is staging/storage for an external console workflow only in V1" in html
    assert "releaseUploadStatus" in html
    assert "Connection interrupted · resuming" in html
    assert "retryAttempt>4" in html
    assert "resumable?'Resume upload':'Upload & verify'" in html
    assert "RUSTORE LIVE '+versionCode" in html
    assert "store_release_confirmed:rustore" in html
    assert "channel==='direct'&&kind==='apk'" in html
    assert "RuStore releases can be published here only after that exact version is live" in html


def test_profile_admin_entry_bootstraps_cookie_session_without_auth_query():
    index = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")

    assert "openAdminPanel" in index
    assert "'/api/admin/session'" in index
    assert "buildUiAuthUrl('/extraShop/admin')" not in index
    assert "/extraShop/admin?_auth" not in index


def test_admin_frontend_api_handles_expired_session_and_non_json_errors():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "await r.text()" in html
    assert "JSON.parse(text)" in html
    assert "showSessionExpired" in html
    assert "r.status===401||r.status===403" in html
    assert "d.data&&d.data.error" in html


def test_admin_frontend_escapes_single_quotes_and_uses_js_string_callbacks():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "replace(/'/g,'&#39;')" in html
    assert "function jsString" in html
    assert "saveSquadConfig('+jsString(k)+')" in html
    assert "toggleRuntimeFeature('+jsString(key)+'," in html
    assert "toggleMatchMode('+jsString(String(m.mode_id))+'" in html


def test_admin_frontend_toasts_do_not_render_literal_html_entities():
    html = (ROOT / "extraShop" / "admin.html").read_text(encoding="utf-8")

    assert "&mdash;" not in html


def test_legacy_admin_players_post_route_is_not_registered():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert 'app.router.add_get("/api/admin/players", admin_players_handler)' in server
    assert 'app.router.add_post("/api/admin/players", admin_players_handler)' not in server


def test_admin_routes_have_central_auth_middleware():
    server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")

    assert "async def admin_auth_middleware" in server
    assert "def _is_admin_api_path" in server
    assert 'path.startswith("/api/admin/")' in server
    assert "COMMUNITY_ADMIN_API_PATHS" in server
    assert "app.middlewares.append(admin_auth_middleware)" in server


def test_admin_adjust_resource_uses_explicit_column_map():
    database = (ROOT / "infrastructure" / "database.py").read_text(encoding="utf-8")

    assert "resource_columns = {" in database
    assert 'f"SELECT {resource} FROM users WHERE user_id = $1"' not in database
