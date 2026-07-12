/* tests/e2e/casesfix_admin_case_config.js — CasesFix admin panel e2e.

Validates the REAL extraShop/admin.html case-config panel (the real-time
case configuration UI added by CasesFix) against a mocked /api/admin/configs +
/api/admin/case-config backend.

Covers the CasesFix frontend changes:
  1. renderCaseConfig renders the live case_config from /api/admin/configs:
     limited base particles (150), T5 jackpot particles (125, the post-960a5e8e
     value the old >=500 threshold was stale against), limited-event switch
     state, probability input, and the human-readable summary.
  2. The limited-event toggle posts a structured patch
     ({limited_event_active:true}) to /api/admin/case-config and re-renders.
  3. "Save limited-event" posts {limited_event_active, limited_event_probability}
     with the edited probability.
  4. "Apply JSON patch" posts an arbitrary structured patch and re-renders
     (e.g. t5_common_jackpot_particles -> 200 updates the readonly jackpot field).

The served backend is mocked: admin.html is served statically from the worktree
root; /api/admin/* return fixed/merged JSON. No real server, no real DB, no
real admin token. Auth is bypassed (admin.html only attaches Authorization when
window.Telegram.WebApp.initData is present, which it is not in a plain browser).

DOM / JS-state assertions only — NO visual_input (not a vision model).

Run from the worktree root:
    node tests/e2e/casesfix_admin_case_config.js
*/
const { chromium } = require('playwright');
const fs = require('fs');
const http = require('http');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const PORT = 8099;

const json = (payload, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify(payload),
});

// ── Mirror infrastructure/case_config.py post-CasesFix defaults ──
// The −40% epic+ rebalance (2026-07-12) lives in tier_rarity_probabilities.
// limited base = 150, T5 jackpot = 125 (was 500 before 960a5e8e).
const CASE_CONFIG = {
  limited_event_active: false,
  limited_event_probability: 0.0015,
  base_particles_by_rarity: {
    common: 2, rare: 3, start: 4, superrare: 5,
    epic: 10, legendary: 20, mythic: 40, divine: 100, limited: 150,
  },
  t5_common_jackpot_particles: 125,
  tier_rarity_probabilities: {
    1: { common: 0.820, rare: 0.162, superrare: 0.018 },
    2: { common: 0.644, rare: 0.272, superrare: 0.073, epic: 0.011 },
    3: { common: 0.525, rare: 0.281, superrare: 0.140, epic: 0.043, legendary: 0.011 },
    4: { common: 0.402, rare: 0.294, superrare: 0.196, epic: 0.065, legendary: 0.032, mythic: 0.011 },
    5: { common: 0.322, rare: 0.258, superrare: 0.258, epic: 0.081, legendary: 0.043, mythic: 0.027, divine: 0.011, limited: 0.00 },
  },
  tier_particles_multiplier: { 1: 1.30, 2: 0.65, 3: 0.81, 4: 1.14, 5: 1.95 },
  tier_rewards_count: {},
  start_rarity_replacement: { rare: 0.05, superrare: 0.02 },
  max_rarity_by_tier: { 1: 'superrare', 2: 'epic', 3: 'legendary', 4: 'mythic', 5: 'divine' },
  tier_upgrade_chances: { 1: 0.25, 2: 0.20, 3: 0.15, 4: 0.10 },
};

// In-memory server-side config state + captured POSTs.
function startServer() {
  let stored = JSON.parse(JSON.stringify(CASE_CONFIG));
  const posts = []; // {path, patch}

  const types = {
    '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp',
    '.svg': 'image/svg+xml', '.json': 'application/json', '.ico': 'image/x-icon',
  };

  const serveFile = (urlPath, res) => {
    const filePath = path.join(ROOT, urlPath);
    if (!filePath.startsWith(ROOT)) { res.statusCode = 403; return res.end(); }
    fs.readFile(filePath, (err, data) => {
      if (err) { res.statusCode = 404; return res.end('not found'); }
      res.setHeader('Content-Type', types[path.extname(filePath).toLowerCase()] || 'application/octet-stream');
      res.end(data);
    });
  };

  // Shallow top-level merge of scalar/dict patch keys into stored config.
  // (The real merge_case_config_patch does structured per-tier/per-rarity
  // deep-merge; this mock only needs scalar patches, so shallow is sufficient
  // and faithful for the patches exercised here.)
  const applyPatch = (patch) => {
    const next = JSON.parse(JSON.stringify(stored));
    for (const [k, v] of Object.entries(patch)) next[k] = v;
    stored = next;
    return stored;
  };

  return new Promise((resolve, reject) => {
    const server = http.createServer(async (req, res) => {
      const finish = (r) => {
        res.statusCode = r.status || 200;
        res.setHeader('Content-Type', r.contentType || 'application/json');
        res.end(r.body);
      };
      let urlPath = decodeURIComponent(req.url.split('?')[0]);
      if (urlPath === '/') urlPath = '/extraShop/admin.html';
      if (!urlPath.startsWith('/api/')) return serveFile(urlPath, res);

      let body = '';
      for await (const chunk of req) body += chunk;
      let payload = {};
      try { payload = body ? JSON.parse(body) : {}; } catch (_) {}

      if (urlPath === '/api/admin/configs') {
        return finish(json({ data: {
          case_config: stored,
          runtime_config: { maintenance_mode: { enabled: false }, feature_availability: {}, disabled_card_ids: [] },
          cards: [], match_modes: [], reward_tracks: [],
          shop_sets: [], ruble_products: [], promocodes_count: 0, excluded: [],
        } }));
      }
      if (urlPath === '/api/admin/case-config') {
        posts.push({ path: urlPath, patch: payload.patch || null, raw: payload });
        const merged = applyPatch(payload.patch || {});
        return finish(json({ data: merged }));
      }
      // loadAll() + lazy fetches: empty 200s so the panel boots without crashing.
      if (urlPath === '/api/admin/shop/sets') return finish(json({ sets: [] }));
      if (urlPath === '/api/admin/ruble-products') return finish(json({ products: [] }));
      if (urlPath === '/api/admin/ruble-products/options') return finish(json({ item_types: [], package_types: {}, shop_sets: [] }));
      if (urlPath === '/api/admin/cosmetics') return finish(json({ items: [] }));
      if (urlPath === '/api/admin/runtime-config') return finish(json({ data: stored }));
      return finish(json({}));
    });
    server.on('error', reject);
    server.listen(PORT, '127.0.0.1', () => resolve({ server, getPosts: () => posts }));
  });
}

(async () => {
  const { server, getPosts } = await startServer();
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') pageErrors.push('console.error: ' + m.text()); });

  let passed = 0;
  const assert = (cond, msg) => {
    if (!cond) throw new Error('ASSERT FAILED: ' + msg);
    passed += 1;
    console.log('  ✓ ' + msg);
  };
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  try {
    console.log('\n▶ Loading admin panel (mocked /api/admin/*)…');
    await page.goto(`http://127.0.0.1:${PORT}/extraShop/admin.html`, { waitUntil: 'domcontentloaded' });

    // Open the Configs view (triggers loadConfigs -> renderCaseConfig).
    console.log('▶ Opening Configs view…');
    await page.click('[data-nav="configs"]');
    // Wait for renderCaseConfig to populate the readonly summary (non-default text).
    await page.waitForFunction(() => {
      const el = document.querySelector('#cfg-case-summary');
      return !!el && el.value && el.value.indexOf('Loading') === -1;
    }, { timeout: 8000 });

    console.log('\n── renderCaseConfig: live case_config rendering ──');
    const limitedBase = await page.inputValue('#cfg-case-limited-base');
    assert(limitedBase === '150', `limited base particles renders 150 (got "${limitedBase}")`);
    const jackpot = await page.inputValue('#cfg-case-jackpot');
    assert(jackpot === '125', `T5 jackpot particles renders 125 — the post-960a5e8e value the old >=500 threshold was stale against (got "${jackpot}")`);
    const summary = await page.inputValue('#cfg-case-summary');
    assert(summary.includes('Limited event OFF'), `summary shows "Limited event OFF" (got "${summary}")`);
    assert(summary.includes('limited base: 150'), `summary shows "limited base: 150" (got "${summary}")`);
    const switchOn = await page.getAttribute('#cfg-case-limited-switch', 'data-on');
    assert(switchOn === '0', `limited-event switch is OFF (data-on="${switchOn}")`);
    const prob = await page.inputValue('#cfg-case-limited-prob');
    assert(prob === '0.0015', `limited-event probability renders 0.0015 (got "${prob}")`);

    console.log('\n── limited-event toggle: structured patch POST ──');
    await page.click('#cfg-case-limited-switch');
    await wait(200);
    const togglePosts = getPosts().filter((p) => p.patch && 'limited_event_active' in p.patch);
    assert(togglePosts.length === 1, `limited-event toggle POSTed exactly once to /api/admin/case-config`);
    assert(togglePosts[0].patch.limited_event_active === true, `toggle patch is {limited_event_active:true} (got ${JSON.stringify(togglePosts[0].patch)})`);
    // After re-render, the switch reflects the server-returned state.
    const switchOnAfter = await page.getAttribute('#cfg-case-limited-switch', 'data-on');
    assert(switchOnAfter === '1', `after toggle+re-render, switch is ON (data-on="${switchOnAfter}")`);
    const summaryAfter = await page.inputValue('#cfg-case-summary');
    assert(summaryAfter.includes('Limited event ON'), `after toggle, summary shows "Limited event ON" (got "${summaryAfter}")`);

    console.log('\n── Save limited-event: edited probability POST ──');
    await page.fill('#cfg-case-limited-prob', '0.002');
    await page.click('#btn-case-limited-save');
    await wait(200);
    const savePosts = getPosts().filter((p) => p.patch && 'limited_event_probability' in p.patch);
    assert(savePosts.length === 1, `save-limited-event POSTed exactly once`);
    assert(savePosts[0].patch.limited_event_probability === 0.002, `save patch carries edited probability 0.002 (got ${savePosts[0].patch.limited_event_probability})`);
    assert(savePosts[0].patch.limited_event_active === true, `save patch also carries limited_event_active:true (got ${savePosts[0].patch.limited_event_active})`);

    console.log('\n── Apply JSON patch: arbitrary structured patch + re-render ──');
    await page.fill('#cfg-case-patch-json', JSON.stringify({ t5_common_jackpot_particles: 200 }));
    await page.click('#btn-case-patch-apply');
    // Wait for the status line to flip to "Applied …".
    await page.waitForFunction(() => {
      const el = document.querySelector('#cfg-case-patch-status');
      return !!el && el.textContent.indexOf('Applied') === 0;
    }, { timeout: 8000 });
    const patchPosts = getPosts().filter((p) => p.patch && 't5_common_jackpot_particles' in p.patch);
    assert(patchPosts.length === 1, `apply-JSON-patch POSTed exactly once`);
    assert(patchPosts[0].patch.t5_common_jackpot_particles === 200, `JSON patch carries t5_common_jackpot_particles:200 (got ${patchPosts[0].patch.t5_common_jackpot_particles})`);
    const jackpotAfter = await page.inputValue('#cfg-case-jackpot');
    assert(jackpotAfter === '200', `after patch+re-render, readonly jackpot field updates to 200 (got "${jackpotAfter}")`);
    const statusText = await page.textContent('#cfg-case-patch-status');
    assert(statusText.startsWith('Applied'), `patch status shows "Applied …" (got "${statusText}")`);

    console.log('\n── reject invalid JSON patch (no POST, error status) ──');
    await page.fill('#cfg-case-patch-json', 'not-json');
    const beforeBad = getPosts().length;
    await page.click('#btn-case-patch-apply');
    await wait(150);
    const afterBad = getPosts().length;
    assert(afterBad === beforeBad, `invalid JSON patch did NOT POST (before=${beforeBad}, after=${afterBad})`);
    // Invalid JSON path emits a transient toast (toast('Patch must be valid JSON','error')),
    // NOT the #cfg-case-patch-status element (which retains its prior "Applied" text).
    await page.waitForFunction(() => {
      const t = document.querySelector('.toast.error');
      return !!t && t.textContent.toLowerCase().includes('json');
    }, { timeout: 4000 });
    const badToast = await page.textContent('.toast.error');
    assert(badToast.toLowerCase().includes('valid json'), `invalid-patch toast reports JSON error (got "${badToast}")`);

    console.log(`\n✅ CasesFix admin case-config e2e PASSED (${passed} assertions)`);
  } catch (err) {
    console.error('\n❌ e2e FAILED:', err.message);
    try {
      await page.screenshot({ path: path.join(__dirname, 'casesfix_admin_case_config_fail.png'), fullPage: false });
      console.error('   screenshot: tests/e2e/casesfix_admin_case_config_fail.png');
    } catch (_) {}
    console.error('   pageErrors:', pageErrors.slice(0, 10).join('\n           '));
    process.exitCode = 1;
  } finally {
    await browser.close();
    server.close();
  }
})();