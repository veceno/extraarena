/* tests/e2e/squad_chat_link_member_modal.js — member opens the chat_link modal.

The squad already has a chat_link set; a non-owner (member) sees the «Ссылка
на чат» button in the hub. Clicking it opens a modal with the link, a
«Перейти» CTA and a disclaimer. Capture that modal.
*/
const { chromium } = require('playwright');

const json = (payload, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify(payload),
});

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 412, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  const CHAT_URL = 'https://t.me/audit_squad_chat';
  const clanWithLink = {
    id: 10, name: 'Audit Squad', tag: 'AUD', displayId: 10, publicId: 10,
    // member role: NOT creator — to verify members see the link too
    member_role: 'member', memberRole: 'member',
    description: 'Клан для альфа-тестеров проекта',
    type: 'open', minTrophies: 0,
    has_boost: false, boost: false, members: 8, max: 15,
    cbrp: 0, treasury: 0, myTokens: 0,
    avatar_url: '', banner_url: '',
    chat_link: CHAT_URL, owner_id: 99,
  };
  const profile = {
    user_id: 42, display_name: 'QA Member', first_name: 'Member', username: 'qa_member',
    gems: 100, coins: 1000, trophies: 100, max_trophies: 100,
    extra_pass: 'inactive', settings: {},
  };
  const runtimeStatus = {
    maintenance_mode: { enabled: false },
    feature_availability: { shop: true, collection: true, squads: true, community: true },
  };

  await page.route('**/api/mobile/bootstrap**', (route) => route.fulfill(json({
    profile,
    runtime_status: runtimeStatus,
    profile_status: 200,
    runtime_status_code: 200,
    server_time: Date.now() / 1000,
  })));
  await page.route('**/api/profile**', (route) => route.fulfill(json(profile)));
  await page.route('**/api/runtime/status**', (route) => route.fulfill(json(runtimeStatus)));
  await page.route('**/api/squads/me**', (route) => route.fulfill(json({
    clan: clanWithLink, members: [], activity: [], notice_count: 0,
  })));
  await page.route('**/api/squads/search**', (route) => route.fulfill(json({ clans: [] })));
  await page.route('**/api/squads/shop**', (route) => route.fulfill(json({
    upgrade_catalog: {}, upgrades: {}, personal_rewards: [],
  })));
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (
      url.includes('/api/mobile/bootstrap') || url.includes('/api/profile') ||
      url.includes('/api/runtime/status') || url.includes('/api/squads/')
    ) return route.fallback();
    return route.fulfill(json({ success: true }));
  });

  await page.goto('http://127.0.0.1:8095/index.html?user_id=42');
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(2500);

  // Open squad hub
  await page.locator('button')
    .filter({ has: page.locator('span', { hasText: /^Сквады$/ }) })
    .first().click();
  await page.waitForTimeout(1500);

  // Capture hub (the button is visible here)
  await page.screenshot({
    path: '/Users/laveqox/Documents/ExtraArenaRaS/output/squad-chat-link-member-hub.png',
    fullPage: true,
  });

  // Click «Ссылка на чат» button
  const chatBtn = page.locator('button').filter({ hasText: /^Ссылка на чат$/ }).first();
  await chatBtn.click();
  await page.waitForTimeout(1000);

  // Capture the modal
  await page.screenshot({
    path: '/Users/laveqox/Documents/ExtraArenaRaS/output/squad-chat-link-member-modal.png',
    fullPage: true,
  });

  await browser.close();
  console.log('Member-modal screenshot saved.');
})().catch((e) => { console.error('FAIL:', e.message); process.exit(1); });
