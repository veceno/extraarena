/* tests/e2e/squad_chat_link_owner_form.js — owner fills chat_link form and saves.

Mocks /api/squads/me + /api/squads/settings via page.route, opens the squad
hub, switches to the «Настройки» tab, types a Telegram URL, checks the
confirmation checkbox, and captures the form before save.
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

  const clanOwner = {
    id: 10, name: 'Audit Squad', tag: 'AUD', displayId: 10, publicId: 10,
    member_role: 'creator', memberRole: 'creator', description: '', type: 'open',
    minTrophies: 0, has_boost: false, boost: false, members: 1, max: 15,
    cbrp: 0, treasury: 0, myTokens: 0, avatar_url: '', banner_url: '',
    chat_link: '', owner_id: 42,
  };
  const profile = {
    user_id: 42, display_name: 'QA Owner', first_name: 'Owner', username: 'qa_owner',
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
    clan: clanOwner, members: [], activity: [], notice_count: 0,
  })));
  await page.route('**/api/squads/search**', (route) => route.fulfill(json({ clans: [] })));
  await page.route('**/api/squads/shop**', (route) => route.fulfill(json({
    upgrade_catalog: {}, upgrades: {}, personal_rewards: [],
  })));
  // Intercept settings POST: capture payload, respond success with chat_link normalized.
  let capturedBody = null;
  await page.route('**/api/squads/settings**', (route) => {
    if (route.request().method() === 'POST') {
      capturedBody = route.request().postData();
      const updatedClan = { ...clanOwner, chat_link: 'https://t.me/audit_squad_chat' };
      return route.fulfill(json({ success: true, clan: updatedClan }));
    }
    return route.fulfill(json({ success: true, clan: clanOwner }));
  });
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

  // Click «Настройки» tab via direct onClick
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button'))
      .find((b) => /^Настройки$/.test((b.textContent || '').trim()) && b.closest('[style*="overflow-x"]'));
    const propsKey = Object.keys(btn).find((k) => k.startsWith('__reactProps$'));
    btn[propsKey].onClick({ stopPropagation: () => {}, preventDefault: () => {} });
  });
  await page.waitForTimeout(1500);

  // Fill chat_link input
  const chatInput = page.locator('input[placeholder*="your_squad_chat"]');
  await chatInput.fill('https://t.me/audit_squad_chat');
  await page.waitForTimeout(400);

  // Tick the confirmation checkbox (custom span[role="checkbox"])
  const confirmBox = page.locator('span[role="checkbox"]').first();
  await confirmBox.click();
  await page.waitForTimeout(400);

  // Capture the form
  await page.screenshot({
    path: '/Users/laveqox/Documents/ExtraArenaRaS/output/squad-chat-link-owner-form.png',
    fullPage: true,
  });

  // Save and verify payload
  const saveBtn = page.locator('button').filter({ hasText: /^Сохранить$/ }).first();
  await saveBtn.click();
  await page.waitForTimeout(1500);

  console.log('Captured POST body:', capturedBody);

  // After save, capture hub to confirm chat_link appears (sanity, optional)
  await page.waitForTimeout(800);
  await page.screenshot({
    path: '/Users/laveqox/Documents/ExtraArenaRaS/output/squad-chat-link-owner-after-save.png',
    fullPage: true,
  });

  await browser.close();
  console.log('Owner-form screenshots saved.');
})().catch((e) => { console.error('FAIL:', e.message); process.exit(1); });
