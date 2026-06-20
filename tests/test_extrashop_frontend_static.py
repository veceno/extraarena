import re
from pathlib import Path


EXTRASHOP_INDEX = Path(__file__).resolve().parents[1] / "extraShop" / "index.html"
EXTRASHOP_ADMIN = Path(__file__).resolve().parents[1] / "extraShop" / "admin.html"
EXTRASHOP_DIR = Path(__file__).resolve().parents[1] / "extraShop"

SERVER_RUBLE_PRODUCT_DESCRIPTIONS = {
    "extrapass": "Премиальная дорожка Battle Pass, 5 пресетов колод, кейс за 4 победы и премиальное свечение ника.",
    "extrapass_ultra": "Всё из ExtraPass, 500 гемов, Ultra-финал Battle Pass, улучшенное свечение ника и переоткрытие кейса.",
    "starter_boost": "ExtraPass на 1 сезон, гемы, монеты и кейсы.",
}


def test_extrashop_gifts_are_hidden_while_beta_disabled():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")

    assert "var GIFTS_BETA_ENABLED = false;" in html
    assert 'href="#gifts"' not in html
    assert 'data-route="gifts"' not in html
    assert 'class="secondary-btn gift-btn"' not in html
    assert 'id="gifts"' not in html
    assert "Подарок соклановцам" not in html
    assert "подарок соклановцам" not in html

    routes_match = re.search(r"var routes = \[(?P<routes>[^\]]+)\];", html)
    assert routes_match is not None
    assert '"gifts"' not in routes_match.group("routes")

    gift_handler = html.split("var giftBtn = event.target.closest('.gift-btn');", 1)[1].split(
        "var gemCard = event.target.closest('.gem-card');",
        1,
    )[0]
    assert "if (!GIFTS_BETA_ENABLED) return;" in gift_handler
    assert "setRoute('gifts');" in gift_handler


def test_dynamic_product_images_are_created_with_dom_api():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")

    assert '\'<img src="\' + esc(p.image_url)' not in html
    assert 'var image = document.createElement(\'img\');' in html
    assert "image.width = 48;" in html
    assert "image.height = 48;" in html
    assert "image.style.borderRadius = '8px';" in html
    assert "image.style.objectFit = 'cover';" in html
    assert "image.alt = '';" in html
    assert "image.src = p.image_url || display.image_url;" in html
    assert "gemMain.appendChild(image);" in html


def test_mobile_product_card_selection_routes_to_checkout_when_side_panel_hidden():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")
    public_shop_block = html.split("// ==================== PUBLIC SHOP (no ?checkout) ====================", 1)[1]
    card_handler = html.split("var gemCard = event.target.closest('.gem-card');", 1)[1].split(
        "document.getElementById('checkout-pay-btn').addEventListener",
        1,
    )[0]

    assert "function catalogSelectionNeedsCheckoutRoute()" in html
    assert "window.getComputedStyle(side).display === 'none'" in html
    assert "if (catalogSelectionNeedsCheckoutRoute()) setRoute('checkout');" in card_handler
    assert "function escAttr(s)" in public_shop_block
    assert "'<button class=\"buy-btn\" type=\"button\" data-item=\"' + escAttr(p.code)" in html
    assert "actionLabel = isGift ? 'Получить в игре' : 'Купить'" in html
    assert "console.warn('[ExtraShop] Dynamic products unavailable:', error);" in html


def test_public_shop_uses_safe_dynamic_selectors_and_pack_previews():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")
    public_shop_block = html.split("// ==================== PUBLIC SHOP (no ?checkout) ====================", 1)[1]

    assert "function safeDataItemSelector(dataItem)" in public_shop_block
    assert "window.CSS && CSS.escape" in public_shop_block
    assert "document.querySelector(safeDataItemSelector(dataItem) + '[data-item-name]')" in public_shop_block
    assert "'[data-item=\"' + dataItem" not in public_shop_block
    assert "function renderPackPreview(product)" in public_shop_block
    assert "normalizePackRewards(product)" in public_shop_block
    assert "pack-preview resources-only" in public_shop_block
    assert "pack-preview cosmetics-resources" in public_shop_block
    assert "pack-preview card-pack" in public_shop_block
    assert "reward.card_image_url || reward.image_url" in public_shop_block
    assert "reward.card_description || reward.description" in public_shop_block
    assert "reward.mechanics || reward.mechanics_desc" in public_shop_block
    assert "pack-card-desc" in public_shop_block
    assert "pack-card-mechanics" in public_shop_block
    assert "reward.cosmetic_slug || reward.slug" in public_shop_block


def test_public_shop_gift_products_do_not_start_public_checkout():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")
    start_block = html.split("function startPublicCheckout()", 1)[1].split(
        "// Click handlers for \"Купить\" / \"Подарить\" buttons",
        1,
    )[0]

    assert "function isGiftProductSelection()" in html
    assert "function giftProductActionMessage()" in html
    assert "selectedItemType === 'gift_shop_set'" in html
    assert "showCheckoutError(giftProductActionMessage());" in start_block
    assert "fetch('/api/payments/checkout/public/start'" in start_block
    assert start_block.index("showCheckoutError(giftProductActionMessage());") < start_block.index("fetch('/api/payments/checkout/public/start'")
    assert "Подарок" in html
    assert "Бесплатно" in html
    assert "Получить в игре" in html
    assert "Robokassa · карта / СБП" in html


def test_payment_open_uses_single_popup_attempt():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")
    open_block = html.split("function openExternalPayment(url)", 1)[1].split(
        "function setCopyState",
        1,
    )[0]

    assert "window.open(url, '_blank', 'noopener,noreferrer')" in open_block
    assert "link.click()" not in open_block
    assert "document.createElement('a')" not in open_block


def test_checkout_session_uses_jti_and_cleans_legacy_token_url():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")
    overlay_block = html.split("var CHECKOUT_JTI_STORAGE_KEY = 'extra_shop_checkout_jti';", 1)[1].split(
        "var historyToken = params.get('history');",
        1,
    )[0]
    start_block = html.split("function startPublicCheckout()", 1)[1].split(
        "function selectProduct",
        1,
    )[0]

    assert "var hashParams = new URLSearchParams((location.hash || '').replace(/^#/, ''));" in html
    assert "params.get('checkout_jti') || hashParams.get('checkout_jti') || hashParams.get('checkout')" in overlay_block
    assert "sessionStorage.setItem(CHECKOUT_JTI_STORAGE_KEY, checkoutJti)" in overlay_block
    assert "sessionStorage.setItem(CHECKOUT_LEGACY_TOKEN_STORAGE_KEY, token)" in overlay_block
    assert "params.delete('checkout')" in overlay_block
    assert "params.delete('checkout_jti')" in overlay_block
    assert "history.replaceState(null, '', cleanUrl || location.pathname)" in overlay_block
    assert "sessionStorage.getItem(CHECKOUT_JTI_STORAGE_KEY)" in overlay_block
    assert "sessionStorage.setItem(CHECKOUT_JTI_STORAGE_KEY, result.data.checkout_jti)" in start_block
    assert "body: JSON.stringify(checkoutJti ? { checkout_jti: checkoutJti } : { token: token })" in html
    assert "'/extraShop#checkout_jti=' + encodeURIComponent(result.data.checkout_jti)" in start_block
    assert "'/extraShop?checkout=' + encodeURIComponent(result.data.token)" not in html


def test_static_product_descriptions_match_in_game_ruble_catalog():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")

    for item_code, description in SERVER_RUBLE_PRODUCT_DESCRIPTIONS.items():
        item_match = re.search(
            rf'<article class="product bundle-card"[^>]+data-item="{re.escape(item_code)}"[^>]+>',
            html,
        )
        assert item_match is not None
        assert f'data-item-desc="{description}"' in item_match.group(0)

        product_block = html.split(f'data-item="{item_code}"', 1)[1].split("</article>", 1)[0]
        assert f"<p>{description}</p>" in product_block


def test_static_catalog_cards_have_stable_polished_layout_rules():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")

    assert ".catalog-grid { display: grid; gap: 10px; }" in html
    assert "grid-template-columns: minmax(128px, 36%) minmax(0, 1fr);" in html
    assert "min-height: 176px;" in html
    assert ".bundle-content .product-top {" in html
    assert "grid-template-columns: minmax(0, 1fr) auto;" in html
    assert ".gem-card .price { justify-self: end; align-self: end; }" in html
    assert ".benefit-grid {" in html
    assert "grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));" in html


def test_pack_preview_treats_particles_as_resources_not_cosmetics():
    html = EXTRASHOP_INDEX.read_text(encoding="utf-8")

    assert "return ['gems', 'coins', 'keys', 'case', 'particles'];" in html
    assert "reward.type === 'cosmetic' || reward.type === 'particles'" not in html


def test_admin_extra_pass_reward_editor_splits_random_and_specific_card_rewards():
    html = EXTRASHOP_ADMIN.read_text(encoding="utf-8")

    assert '<option value="card">random card by rarity</option>' in html
    assert '<option value="specific_card">specific card</option>' in html
    assert '"reward_type":"specific_card"' in html or "reward_type:'specific_card'" in html
    assert "type==='card'||type==='specific_card'?1" in html
    assert '{"rarity":["epic"]} или {"card_id":46}' in html


def test_legal_documents_use_registered_routes_and_canonical_support_email():
    html_by_name = {
        path.name: path.read_text(encoding="utf-8")
        for path in EXTRASHOP_DIR.glob("*.html")
        if path.name in {"index.html", "extrashop.html", "oferta.html", "privacy.html", "refund.html"}
    }
    all_html = "\n".join(html_by_name.values())

    assert 'href="/privacy.html"' not in all_html
    assert 'href="/oferta.html"' not in all_html
    assert 'href="/refund.html"' not in all_html
    assert "personal@laveqox.ru" not in all_html
    assert "support@laveqox.ru" in all_html
    assert 'href="/extraShop/privacy"' in html_by_name["oferta.html"]
    assert 'href="/extraShop/oferta"' in html_by_name["refund.html"]


def test_extrashop_info_sections_link_to_current_legal_documents():
    for filename in ("index.html", "extrashop.html"):
        html = (EXTRASHOP_DIR / filename).read_text(encoding="utf-8")
        legal_section = html.split('id="legal"', 1)[1].split("</section>", 1)[0]

        assert '<a class="legal-link" href="/extraShop/oferta" target="_blank" rel="noopener noreferrer">' in legal_section
        assert '<a class="legal-link" href="/extraShop/privacy" target="_blank" rel="noopener noreferrer">' in legal_section
        assert '<a class="legal-link" href="/extraShop/refund" target="_blank" rel="noopener noreferrer">' in legal_section
        assert "<div class=\"legal-link\"><strong>Пользовательское соглашение / оферта</strong>" not in legal_section
        assert "<div class=\"legal-link\"><strong>Политика конфиденциальности</strong>" not in legal_section
        assert "<div class=\"legal-link\"><strong>Возвраты</strong>" not in legal_section


def test_extrashop_support_contacts_use_current_channels_and_soon_online_support():
    html_by_name = {
        path.name: path.read_text(encoding="utf-8")
        for path in EXTRASHOP_DIR.glob("*.html")
        if path.name in {"index.html", "extrashop.html", "oferta.html", "privacy.html", "refund.html"}
    }
    all_html = "\n".join(html_by_name.values())

    assert "https://t.me/extraarena_supbot" in all_html
    assert "https://max.ru/se13279035_1_bot" in all_html
    assert "support.laveqox.ru" in all_html
    assert "lqsup" not in all_html
    assert 'href="https://support.laveqox.ru"' not in all_html
    assert 'href="http://support.laveqox.ru"' not in all_html
    assert "support.laveqox.ru (скоро)" in all_html
