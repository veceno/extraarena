/* tests/e2e/onboarding_tutorial_arena.js — graph-driven onboarding tutorial e2e.

Validates the REAL webapp/arena.js frontend against REAL TutorialBattleEngine
output (dumped by _dump_onboarding_fixtures.py into onboarding_fixtures.json):
the arena is served statically, /api/battle/state + /api/onboarding/tutorial/action
are mocked with the real engine's per-step states + action responses, and the
tutorial is driven through the arena's OWN action-submission functions
(sendOnboardingTutorialAction / the "Понятно" + "В меню" buttons). After each
step we assert the Midoria dialog text, the board card ids / opponent hero HP
(from `currentState`), the step-progress label, and finally the lethal → victory
modal → "В меню" → /?onboarding_menu=1 handoff.

New 11-step scenario (final_step=10): goal → play_attacker → sleep → threat →
choose_target (NEW: tap Стив = wrong → 409 custom feedback → retry hero) → tempo
→ danger → taunt_intro → taunt_demo (auto) → lethal → victory.

DOM / JS-state assertions only — NO visual_input (not a vision model).

Run from the worktree root:
    node tests/e2e/onboarding_tutorial_arena.js
Prereq: python3 tests/e2e/_dump_onboarding_fixtures.py  (regenerates fixtures)
*/
const playwright = require('playwright');
const fs = require('fs');
const http = require('http');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const FIXTURES_PATH = path.join(__dirname, 'onboarding_fixtures.json');
const PORT = 8097;
const BROWSER_NAME = process.env.EA_E2E_BROWSER || 'chromium';
const VIEWPORT_WIDTH = Number(process.env.EA_E2E_VIEWPORT_WIDTH || 412);
const VIEWPORT_HEIGHT = Number(process.env.EA_E2E_VIEWPORT_HEIGHT || 900);
const TELEGRAM_PLATFORM = process.env.EA_E2E_TELEGRAM_PLATFORM || '';
const TELEGRAM_VIEWPORT_HEIGHT = Number(process.env.EA_E2E_TG_VIEWPORT_HEIGHT || VIEWPORT_HEIGHT);
const IS_IOS_FULLSIZE = TELEGRAM_PLATFORM === 'ios';

const json = (payload, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify(payload),
});

function startStaticServer() {
  // Minimal static file server rooted at the worktree root, so that
  // /webapp/arena.html, /webapp/arena.js, /webapp/arena-style.css and
  // /DesignAssets/** all resolve exactly as in prod.
  const types = {
    '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp',
    '.gif': 'image/gif', '.svg': 'image/svg+xml', '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg', '.json': 'application/json', '.ico': 'image/x-icon',
  };
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      let urlPath = decodeURIComponent(req.url.split('?')[0]);
      if (urlPath === '/') urlPath = '/index.html';
      const filePath = path.join(ROOT, urlPath);
      if (!filePath.startsWith(ROOT)) { res.statusCode = 403; return res.end(); }
      fs.readFile(filePath, (err, data) => {
        if (err) { res.statusCode = 404; return res.end('not found'); }
        res.setHeader('Content-Type', types[path.extname(filePath).toLowerCase()] || 'application/octet-stream');
        res.end(data);
      });
    });
    server.on('error', reject);
    server.listen(PORT, '127.0.0.1', () => resolve(server));
  });
}

(async () => {
  const fixtures = JSON.parse(fs.readFileSync(FIXTURES_PATH, 'utf8'));
  const { user_id: UID, match_id: MATCH_ID, final_step: FINAL_STEP, step0_state: STEP0, responses } = fixtures;

  // The wrong-path (tap Стив) is response index 4 and leaves the engine on
  // step4, so the simple responses[n-1] mapping breaks. Use an explicit
  // step-number → response-index map for the state AFTER reaching each step.
  // (step4's steady state is responses[3]; responses[4] is the 409 wrong tap,
  //  same step4 state.)
  const STEP_RESP = { 1: 0, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10 };
  const WRONG_RESP = 4; // choose_target wrong-minion tap → 409

  const snapshot = (n) => (n === 0 ? STEP0 : responses[STEP_RESP[n]].state);
  const tutorialAt = (n) => snapshot(n).tutorial;
  const boardAt = (n) => (snapshot(n).player.board || []).map((c) => c.card_id);
  const oppBoardAt = (n) => (snapshot(n).opponent.board || []).map((c) => c.card_id);
  const oppHeroHpAt = (n) => snapshot(n).opponent.hero.hp;

  let passed = 0;
  const assert = (cond, msg) => {
    if (!cond) throw new Error('ASSERT FAILED: ' + msg);
    passed += 1;
    console.log('  ✓ ' + msg);
  };

  const server = await startStaticServer();
  const browserType = playwright[BROWSER_NAME];
  if (!browserType) throw new Error(`Unknown Playwright browser: ${BROWSER_NAME}`);
  const browser = await browserType.launch();
  const context = await browser.newContext({
    viewport: { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));

  // Stub window.io BEFORE any page script runs (socket.io CDN is blocked below,
  // so `io` would be undefined and initSocketIO() would throw).
  await page.addInitScript(({ telegramPlatform, telegramViewportHeight, uid }) => {
    window.io = function fakeIo() {
      // `socket.io` is the Manager on the real socket.io client (arena calls
      // socket.io.on('reconnect_failed', …)); expose a stub so initSocketIO
      // doesn't throw. The onboarding flow is pure-HTTP and never uses these.
      const manager = { on: () => {}, off: () => {}, emit: () => {} };
      return {
        id: 'fake-socket',
        connected: false,
        io: manager,
        on: () => {},
        off: () => {},
        emit: () => {},
        disconnect: () => {},
        close: () => {},
      };
    };

    if (telegramPlatform) {
      const handlers = {};
      window.__requestFullscreenCount = 0;
      window.__telegramEventHandlers = handlers;
      window.Telegram = {
        WebApp: {
          platform: telegramPlatform,
          version: '9.6',
          initData: 'dev',
          initDataUnsafe: { user: { id: uid } },
          viewportHeight: telegramViewportHeight,
          viewportStableHeight: telegramViewportHeight,
          isFullscreen: false,
          safeAreaInset: { top: 0, right: 0, bottom: 34, left: 0 },
          contentSafeAreaInset: { top: 52, right: 0, bottom: 34, left: 0 },
          ready() {},
          expand() {},
          setHeaderColor() {},
          setBackgroundColor() {},
          setBottomBarColor() {},
          isVersionAtLeast() { return true; },
          requestFullscreen() { window.__requestFullscreenCount += 1; },
          onEvent(name, callback) {
            handlers[name] = handlers[name] || [];
            handlers[name].push(callback);
          },
          offEvent(name, callback) {
            handlers[name] = (handlers[name] || []).filter((item) => item !== callback);
          },
        },
      };
    }
  }, {
    telegramPlatform: TELEGRAM_PLATFORM,
    telegramViewportHeight: TELEGRAM_VIEWPORT_HEIGHT,
    uid: UID,
  });

  // Block the CDN scripts (offline) so they don't hang or redefine our stub.
  await page.route('https://telegram.org/**', (route) => route.fulfill({ status: 200, body: '' }));
  await page.route('https://cdn.socket.io/**', (route) => route.fulfill({ status: 200, body: '' }));

  // Mock /api/battle/state (initial load) with the real step0 state.
  await page.route('**/api/battle/state**', (route) => route.fulfill(json(STEP0)));

  // Mock /api/onboarding/tutorial/action — serve the real engine responses in
  // order, honoring _status (the real server returns 409 for tutorial_wrong_action).
  let actionIdx = 0;
  await page.route('**/api/onboarding/tutorial/action**', (route) => {
    const resp = responses[actionIdx];
    actionIdx += 1;
    if (!resp) { return route.fulfill(json({ result: { success: false } }, 500)); }
    const status = resp._status || 200;
    return route.fulfill(json(resp, status));
  });

  // Mock the background pings / settings fetches.
  await page.route('**/api/runtime/status**', (route) => route.fulfill(json({
    maintenance_mode: { enabled: false },
    feature_availability: { shop: true, collection: true, squads: true, community: true },
  })));
  await page.route('**/api/settings**', (route) => route.fulfill(json({ success: true })));
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/api/battle/state') || url.includes('/api/onboarding/tutorial/action')
        || url.includes('/api/runtime/status') || url.includes('/api/settings')) {
      return route.fallback();
    }
    return route.fulfill(json({ success: true }));
  });

  // The "В меню" handoff navigates to /?onboarding_menu=1 — serve a stub page
  // so the main-menu bundle doesn't load (we only assert the URL).
  await page.route((url) => url.href.includes('onboarding_menu=1'), (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>menu tour</body></html>' }));

  try {
    const platformQuery = IS_IOS_FULLSIZE ? '' : '&ea_platform=android_app';
    const arenaUrl = `http://127.0.0.1:${PORT}/webapp/arena.html?id=${MATCH_ID}&onboarding=1&_auth=dev${platformQuery}`;
    await page.goto(arenaUrl, { waitUntil: 'domcontentloaded' });

    // Helper: wait until the onboarding bubble contains the steady-state step
    // message. (No step in the new scenario uses an `after` followup, so the
    // bubble settles on `message` directly.)
    const waitForStepMessage = async (n, { timeout = 7000 } = {}) => {
      const expected = tutorialAt(n).message;
      await page.waitForFunction((exp) => {
        const bubble = document.querySelector('.arena-onboarding-bubble');
        return !!bubble && bubble.textContent.includes(exp);
      }, expected, { timeout });
    };

    const bubbleText = () => page.locator('.arena-onboarding-bubble').first().textContent();
    const stepLabel = () => page.locator('.arena-onboarding-step').first().textContent();
    const currentState = () => page.evaluate(() => currentState);
    const assertInsideViewport = async (selector, label) => {
      const box = await page.locator(selector).first().boundingBox();
      const viewport = page.viewportSize();
      assert(!!box, `${label}: bounding box exists`);
      assert(
        box.y >= -1 && box.y + box.height <= viewport.height + 1,
        `${label}: visible inside ${viewport.height}px viewport`,
      );
    };

    // ---- step 0 (goal) --------------------------------------------------------
    await waitForStepMessage(0);
    console.log('step 0 — goal');
    assert((await bubbleText()).includes(tutorialAt(0).message), 'step0 Midoria message (goal)');
    assert((await stepLabel()).trim() === 'Старт', 'step0 label "Старт"');
    {
      const st = await currentState();
      assert(JSON.stringify(st.player.hand.map((c) => c.card_id)) === JSON.stringify([37, 39]),
        'step0 hand [Слайм, Альфонс]');
      assert(st.opponent.hero.hp === 8 && st.opponent.hero.max_hp === 8, 'step0 opponent hero 8/8');
    }
    if (IS_IOS_FULLSIZE) {
      const viewportMetrics = await page.evaluate(() => ({
        cssHeight: Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--ea-viewport-height')),
        visualHeight: window.visualViewport?.height || window.innerHeight,
        fullscreenRequests: window.__requestFullscreenCount,
      }));
      assert(viewportMetrics.cssHeight <= viewportMetrics.visualHeight + 1,
        'iOS CSS shell height is clamped to the visible WKWebView viewport');
      assert(viewportMetrics.fullscreenRequests === 1,
        'iOS requests Telegram fullscreen exactly once');
      await assertInsideViewport('.arena-onboarding-action', 'step0 onboarding action');
    }

    // step0 → 1: click "Понятно" (goal → play_attacker).
    await page.locator('.arena-onboarding-action').first().click();
    await waitForStepMessage(1);
    console.log('step 1 — play_attacker');
    assert((await bubbleText()).includes(tutorialAt(1).message), 'step1 Midoria message (play_attacker)');
    assert((await stepLabel()).trim() === 'Шаг 1/10', 'step1 label "Шаг 1/10"');

    // step1 → 2: play Слайм via the arena's own action submission.
    await page.evaluate(async () => {
      await sendOnboardingTutorialAction({
        type: 'play_card', hand_index: 0, card_id: 37,
        target_position: 0, target_id: null, target_is_hero: false,
      });
    });
    await waitForStepMessage(2);
    console.log('step 2 — sleep');
    assert((await bubbleText()).includes(tutorialAt(2).message), 'step2 Midoria message (sleep)');
    assert((await stepLabel()).trim() === 'Шаг 2/10', 'step2 label "Шаг 2/10"');
    {
      const st = await currentState();
      assert(JSON.stringify(st.player.board.map((c) => c.card_id)) === JSON.stringify([37]),
        'step2 Слайм on board');
    }

    // step2 → 3: end turn (p2 plays Стив → threat beat).
    if (IS_IOS_FULLSIZE) {
      await assertInsideViewport('#end-turn-button', 'step2 end-turn button');
    }
    await page.evaluate(async () => { await sendOnboardingTutorialAction({ type: 'end_turn' }); });
    await waitForStepMessage(3);
    console.log('step 3 — threat');
    assert((await bubbleText()).includes(tutorialAt(3).message), 'step3 Midoria message (threat)');
    assert((await stepLabel()).trim() === 'Шаг 3/10', 'step3 label "Шаг 3/10"');
    {
      const st = await currentState();
      assert(JSON.stringify(st.opponent.board.map((c) => c.card_id)) === JSON.stringify([40]),
        'step3 Стив on opponent board');
    }

    // step3 → 4: click "Понято" (threat → choose_target). Mid-flow continue
    // beats render a "Понято" button (allowed.type === 'continue').
    await page.locator('.arena-onboarding-action').first().click();
    await waitForStepMessage(4);
    console.log('step 4 — choose_target');
    assert((await bubbleText()).includes(tutorialAt(4).message), 'step4 Midoria message (choose_target)');
    assert((await stepLabel()).trim() === 'Шаг 4/10', 'step4 label "Шаг 4/10"');
    {
      const st = await currentState();
      assert(st.opponent.hero.hp === 8, 'step4 opponent hero still 8 (pre-attack)');
      assert(st.opponent.board.length === 1, 'step4 Стив still on opponent board');
    }

    // ---- WRONG PATH: tap Стив (wrong target) → 409 custom feedback, no state change.
    {
      const stBefore = await currentState();
      const steveId = stBefore.opponent.board[0].instance_id;
      const attackerId = stBefore.tutorial.attacker_instance_id;
      assert(steveId && attackerId, 'step4 have Стив instance_id + attacker_instance_id');

      // Drive the wrong tap; sendOnboardingTutorialAction throws on !response.ok,
      // handleOnboardingActionError shows a toast (sets bare onboardingFeedbackMessage)
      // and does NOT re-render, so currentState is untouched.
      await page.evaluate(async ({ sid, aid }) => {
        try {
          await sendOnboardingTutorialAction({
            type: 'attack', attacker_id: aid, target_id: sid, target_is_hero: false,
          });
        } catch (e) { /* expected: 409 tutorial_wrong_action */ }
      }, { sid: steveId, aid: attackerId });

      const expectedFeedback = responses[WRONG_RESP].feedback;
      await page.waitForFunction((exp) => onboardingFeedbackMessage === exp, expectedFeedback, { timeout: 5000 });
      assert((await page.evaluate(() => onboardingFeedbackMessage)) === expectedFeedback,
        'step4 wrong-minion tap → custom feedback toast text');

      const stAfter = await currentState();
      assert(stAfter.opponent.hero.hp === 8, 'step4 wrong tap: opponent hero HP unchanged (8)');
      assert(stAfter.opponent.board.length === 1, 'step4 wrong tap: Стив still on board');
      assert((stAfter.tutorial.step_id || stAfter.tutorial.id) === 'choose_target',
        'step4 wrong tap: still on choose_target step');
    }

    // step4 → 5: attack opponent hero (correct target, 8 → 4) → tempo beat.
    await page.evaluate(async () => {
      const attackerId = currentState.tutorial.attacker_instance_id;
      await sendOnboardingTutorialAction({
        type: 'attack', attacker_id: attackerId, target_id: null, target_is_hero: true,
      });
    });
    await waitForStepMessage(5);
    console.log('step 5 — tempo');
    assert((await bubbleText()).includes(tutorialAt(5).message), 'step5 Midoria message (tempo)');
    assert((await stepLabel()).trim() === 'Шаг 5/10', 'step5 label "Шаг 5/10"');
    assert(oppHeroHpAt(5) === 4, 'step5 opponent hero HP 4 (real first attack)');
    {
      const st = await currentState();
      assert(st.opponent.hero.hp === 4, 'step5 currentState opponent hero HP 4');
    }

    // step5 → 6: click "Понято" (tempo → danger).
    await page.locator('.arena-onboarding-action').first().click();
    await waitForStepMessage(6);
    console.log('step 6 — danger');
    assert((await bubbleText()).includes(tutorialAt(6).message), 'step6 Midoria message (danger)');
    assert((await stepLabel()).trim() === 'Шаг 6/10', 'step6 label "Шаг 6/10"');

    // step6 → 7: click "Понято" (danger → taunt_intro).
    await page.locator('.arena-onboarding-action').first().click();
    await waitForStepMessage(7);
    console.log('step 7 — taunt_intro');
    assert((await bubbleText()).includes(tutorialAt(7).message), 'step7 Midoria message (taunt_intro)');
    assert((await stepLabel()).trim() === 'Шаг 7/10', 'step7 label "Шаг 7/10"');
    {
      const st = await currentState();
      assert(JSON.stringify(st.player.board.map((c) => c.card_id)) === JSON.stringify([37]),
        'step7 board still [Слайм] (Альфонс in hand)');
    }

    // step7 → 8: play Альфонс (taunt_intro → taunt_demo).
    await page.evaluate(async () => {
      await sendOnboardingTutorialAction({
        type: 'play_card', hand_index: 0, card_id: 39,
        target_position: 1, target_id: null, target_is_hero: false,
      });
    });
    await waitForStepMessage(8);
    console.log('step 8 — taunt_demo (auto)');
    assert((await bubbleText()).includes(tutorialAt(8).message), 'step8 Midoria message (taunt_demo)');
    assert((await stepLabel()).trim() === 'Демо 8/10', 'step8 label "Демо 8/10"');
    {
      const st = await currentState();
      assert(JSON.stringify(st.player.board.map((c) => c.card_id)) === JSON.stringify([37, 39]),
        'step8 Альфонс joins board [Слайм, Альфонс]');
      // Auto-step: shows "Ход противника..." and no "Понято" action button.
      assert((await page.locator('.arena-onboarding-status').first().textContent()).includes('Ход противника'),
        'step8 auto-step status "Ход противника..."');
      assert(await page.locator('.arena-onboarding-action').count() === 0, 'step8 no action button (auto)');
    }

    // step8 → 9: auto_continue (taunt demo — Стив attacks Альфонс, Альфонс dies, p2 ends).
    await page.evaluate(async () => { await sendOnboardingTutorialAction({ type: 'auto_continue' }); });
    await waitForStepMessage(9);
    console.log('step 9 — lethal');
    assert((await bubbleText()).includes(tutorialAt(9).message), 'step9 Midoria message (lethal)');
    assert((await stepLabel()).trim() === 'Шаг 9/10', 'step9 label "Шаг 9/10"');
    assert(JSON.stringify(boardAt(9)) === JSON.stringify([37]), 'step9 Альфонс removed by real death processing');
    {
      const st = await currentState();
      assert(JSON.stringify(st.player.board.map((c) => c.card_id)) === JSON.stringify([37]),
        'step9 currentState board [Слайм] (Альфонс gone)');
      assert(st.opponent.hero.hp === 4, 'step9 opponent hero still 4 (Стив hit Альфонс, not hero)');
    }

    // step9 → 10: lethal attack on opponent hero (4 → 0, real P1_WIN).
    await page.evaluate(async () => {
      const attackerId = currentState.tutorial.attacker_instance_id;
      await sendOnboardingTutorialAction({
        type: 'attack', attacker_id: attackerId, target_id: null, target_is_hero: true,
      });
    });
    console.log('step 10 — victory');
    await page.waitForSelector('.arena-onboarding-victory', { timeout: 5000 });
    assert((await page.locator('.arena-onboarding-victory-title').first().textContent()).trim() === 'Победа',
      'step10 victory modal title "Победа"');
    assert((await page.locator('.arena-onboarding-victory-text').first().textContent()).includes(tutorialAt(10).message),
      'step10 victory text = scenario victory message');
    assert((await page.locator('.arena-onboarding-victory-action').first().textContent()).trim() === 'В меню',
      'step10 "В меню" button present');
    if (IS_IOS_FULLSIZE) {
      await assertInsideViewport('.arena-onboarding-victory-action', 'step10 menu action');
    }
    assert(oppHeroHpAt(10) === 0, 'step10 opponent hero HP 0 (real lethal)');
    {
      const st = await currentState();
      assert(st.is_ended === true && st.game_over === true && st.winner_id === UID,
        'step10 real engine game_over + winner_id = user');
    }

    // step10 → menu tour: click "В меню" → complete → /?onboarding_menu=1.
    await page.locator('.arena-onboarding-victory-action').first().click();
    await page.waitForURL((u) => u.href.includes('onboarding_menu=1'), { timeout: 5000 });
    assert(page.url().includes('onboarding_menu=1'), 'complete → /?onboarding_menu=1 handoff');
    assert(actionIdx === responses.length, `all ${responses.length} action requests consumed (got ${actionIdx})`);

    if (errors.length) {
      console.log('\npage errors (non-fatal):');
      for (const e of errors.slice(0, 10)) console.log('  ⚠ ' + e);
    }

    console.log(`\nALL ${passed} ASSERTIONS PASSED ✅`);
  } catch (err) {
    console.error('\n❌ E2E FAILED:', err.message);
    try { await page.screenshot({ path: path.join(__dirname, 'onboarding_e2e_fail.png') }); } catch (_) {}
    process.exitCode = 1;
  } finally {
    await context.close();
    await browser.close();
    server.close();
  }
})();
