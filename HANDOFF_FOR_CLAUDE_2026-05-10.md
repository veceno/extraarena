# ExtraArena Handoff for Claude — 2026-05-10

## Role / Process

User wants you as a technical partner/reviewer for work done through OpenCode + Deepseek V4 Pro agents. Do not start coding unless explicitly asked. Primary workflow:

- review agent reports against actual code;
- give concise ТЗ/prompts for agents;
- track roadmap/scope;
- prioritize manual QA and security;
- avoid sending UI-rendered "implement this plan" cards; use normal markdown/code blocks.

Language/tone: Russian, informal, direct. User is moving fast and may be low on limits.

Workspace:

```text
/Users/laveqox/Documents/ExtraArenaRaS
```

Last committed snapshot:

```text
4d5d8de chore: snapshot beta systems and assets
```

Current working tree has a large post-commit layer. Do not revert anything without explicit user approval.

## Current Git State

Tracked modified files include:

```text
.gitignore
.opencode/commands/design.md
DesignAssets/Cards/*.png
bot/app.py
bot/handlers.py
infrastructure/config.py
infrastructure/database.py
infrastructure/matchmaking.py
main.py
requirements.txt
webapp/arena.js
webapp/index.html
webapp/main.js
```

Important untracked files:

```text
bot/inline_handlers.py
bot/profile_card_renderer.py
infrastructure/generator_config.py
DesignAssets/Cards/19.png ... 35.png
GeneratedCardArt/trial/*.png
```

Current `git diff --stat` before adding this handoff showed roughly:

```text
23 files changed, 1725 insertions(+), 332 deletions(-)
```

## What Is Already Accepted / Tested by User

Battle/gameplay:

- Blitz works.
- Selected deck propagation worked before latest security changes.
- Trophy loss/win modal worked before latest security changes.
- Trophies are deducted/not deducted correctly in classic/training/friendly before latest security changes.
- Training prebattle and in-battle bot name was patched to "🤖 Тренер".
- Bot lethal now calls `_process_battle_end`, so economy/game_over should emit.

Generator:

- Implemented as "Генератор" main-menu section replacing analytics.
- Passive key generation + claim + lvl1→lvl2 upgrade config.
- Telegram completion notifications.
- Admin debug/reset endpoints.
- Fixed offline generation computation and transactional claim/upgrade.

Deck/collection security:

- Backend deck validation added:
  - card ownership validation;
  - duplicate card rejection;
  - hero/warrior slot validation;
  - card existence check;
  - missing hero rejection;
  - preset limits: free 3, pass/ultra 5;
  - preset name length cap;
  - primary deck validity;
  - cache invalidation;
  - delete primary deck clears `users.primary_deck`.
- Compile/tests reported clean.

Friends:

- Persistent friends MVP implemented:
  - `friend_requests` table;
  - `/api/friends/request`;
  - `/api/friends/request/respond`;
  - `/api/friends/request/cancel`;
  - `/api/friends/list`;
  - `/api/friends/requests`;
  - `/api/friends/remove`;
  - recent opponents now include `is_friend`;
  - UI FriendsScreen now has friends/requests/opponents tabs;
  - friend request notifications and block setting.

Telegram inline profile cards:

- Implemented:
  - `bot/inline_handlers.py`;
  - `bot/profile_card_renderer.py`;
  - `get_public_player_card`;
  - `/generated/inline/` static route;
  - cleanup task;
  - `pillow>=10.0`;
  - separate thumbnail PNGs.
- Patched:
  - no disabled SSL verification;
  - font fallback uses `ImageFont.load_default()`.
- Before real test: BotFather inline mode must be enabled with placeholder like `me или ID игрока`.

## Last Verified Commands

I ran:

```text
python3 -m py_compile web/server.py infrastructure/config.py main.py
pytest -q
```

Result:

```text
91 passed
```

Compile/tests passing does NOT mean security patch is correct.

## Most Important Current Blocker: Security Patch Is Not Ready

The security agent report was too optimistic. It claimed fallbacks were gated/removed, but actual code still has serious issues.

### Critical Finding 1 — raw auth fallbacks still active in production routes

There are still many inline patterns like:

```python
if init_data and init_data.isdigit():
    user_id = int(init_data)

user_id_param = request.rel_url.query.get("user_id")
```

I counted about 54 handlers with this old pattern. These are not all low-priority. Examples:

- `admin_players_handler` around `web/server.py:2088` accepts raw numeric `_auth` and `?user_id`.
- `admin_stats_handler`
- `change_nickname_handler`
- promocode admin/user handlers
- admin cards/items handlers
- user cards and card upgrade handlers
- cases handlers
- debug add key
- admin TPS/rewards/shop handlers
- payments
- mail
- shop buy/particles
- dice
- welcome

This means production impersonation is still possible outside the battle/match endpoints. Example risk:

```text
/api/admin/players?_auth=6803854304
```

appears to pass admin auth if the handler is unchanged.

Useful audit command:

```text
rg -n "init_data and init_data\\.isdigit|rel_url\\.query\\.get\\(\"user_id\"|payload\\.get\\(\"user_id\"|data\\.get\\(\"user_id\"" web/server.py
```

### Critical Finding 2 — arena identity likely broken after auth URL change

`webapp/arena.js` still initializes:

```js
userId = urlParams.get('user_id');
const _auth = urlParams.get('_auth');
if (_auth) {
  sessionStorage.setItem('arena_auth', _auth);
  history.replaceState(null, '', '/arena?id=' + matchId);
}
```

But redirects now go to:

```text
/arena?id=<matchId>&_auth=<initData>
```

No `user_id` is present, and `_auth` is removed from the visible URL. `userId` remains `null`.

Later `arena.js` uses:

```js
const userIdNum = Number(userId); // Number(null) -> 0
state.is_my_turn = Number(state.current_player_id) === userIdNum;
```

and also uses `userId` to:

- choose player/opponent side;
- compute `isWinner`;
- read `data.players[userId]` for economy.

This probably breaks arena perspective and result modal after latest security changes, even though battle worked before.

Recommended fix:

- Add `viewer_id` to `BattleEngine.get_full_state(viewer_id=...)` response, or set it in `battle_state_handler`.
- In `arena.js`, after loading state, set:

```js
userId = String(state.viewer_id || state.player?.user_id || userId || '');
```

- Do not recompute `state.is_my_turn` from URL user_id. Trust server-provided `state.is_my_turn`, or recompute from `viewer_id`.

### Critical Finding 3 — duplicate Socket.IO disconnect handlers

There are two `@sio.event async def disconnect` definitions in `web/server.py`:

- old one around line 101, immediate AFK;
- new multi-session one around line 199.

Even if Python/socketio ends up using the latter, this is a dangerous leftover. Remove the old handler completely.

### High Finding 4 — match lock cleanup tied to disconnect

`_unregister_session()` currently does:

```python
if not MATCH_SESSIONS[match_id]:
    MATCH_SESSIONS.pop(match_id, None)
    MATCH_LOCKS.pop(match_id, None)
```

Do not delete `MATCH_LOCKS` merely because no sockets remain. Match may still be active and mutated by bot/timeout/HTTP surrender. Clean locks on final match cleanup instead.

### Medium Finding 5 — missing auth_date allowed

`_validate_auth_date()` returns `True` if `auth_date` is absent. Telegram WebApp initData should include it. For strict security, missing `auth_date` should be rejected.

## Suggested Next Agent Prompt: Security Fix Follow-up

Send this to the security agent:

```md
Security patch is NOT complete. Please fix the actual code, not the report.

Blocking fixes:

1. Remove all remaining production raw auth fallbacks in `web/server.py`.
   Run:
   `rg -n "init_data and init_data\\.isdigit|rel_url\\.query\\.get\\(\"user_id\"|payload\\.get\\(\"user_id\"|data\\.get\\(\"user_id\"" web/server.py`
   Every auth-related hit must be migrated to `await require_user_id(request)` or `require_user_id_from_payload(request, payload)`.
   Only target IDs such as admin debug `target_user_id` may remain, and only after authenticated admin/user is resolved.

2. Specifically fix admin, cards, cases, payments, mail, shop, dice, welcome, community, promocode routes. Do not call them low priority; they are production impersonation surfaces.

3. Remove the old nested `_resolve_user_id_from_payload()` in `create_web_app`, because it trusts `payload["user_id"]`.

4. Fix arena identity after `_auth` redirect:
   - server must include `viewer_id` in `/api/battle/state`;
   - `arena.js` must set its local `userId` from `state.viewer_id` or `state.player.user_id`;
   - do not rely on URL `user_id`;
   - `handleGameOver` must read `players[String(userId)]` after this is set.

5. Delete the duplicate old `@sio.event disconnect` handler. Keep only the multi-session version.

6. Do not pop `MATCH_LOCKS[match_id]` in `_unregister_session`; cleanup locks only when match is truly ended/removed.

7. Make `auth_date` mandatory for Telegram initData. Missing or expired auth_date must return 401.

Verification:
- `python3 -m py_compile web/server.py infrastructure/config.py main.py`
- `pytest -q`
- `rg` command above should show no raw auth fallback except legitimate target-user fields after auth.
- Manual/local production-mode checks:
  - `/api/admin/players?_auth=6803854304` must be 401 unless this is valid Telegram initData.
  - `/api/cards/user?user_id=<victim>` must be 401 outside dev+localhost.
  - `/api/cases/user?user_id=<victim>` must be 401 outside dev+localhost.
  - `/api/mail?user_id=<victim>` must be 401 outside dev+localhost.
  - battle arena must still orient the player correctly and show correct result economy.
```

## Manual QA Plan After Security Fix

Do this through Telegram WebApp unless explicitly testing dev fallbacks.

1. Open profile.
2. Open deck editor:
   - save valid deck;
   - try duplicate cards if UI allows;
   - set primary;
   - start battle immediately with selected deck.
3. Classic vs bot:
   - check correct side/orientation;
   - check hand/deck;
   - lose/win and verify modal economy.
4. Blitz:
   - 5s timer;
   - mana +2;
   - correct selected deck;
   - trophies deducted/awarded.
5. Training:
   - prebattle and battle name must be "🤖 Тренер";
   - no trophies.
6. Surrender:
   - classic deducts trophies;
   - training/friendly does not.
7. Friends:
   - A sends request to B;
   - B accepts;
   - list updates both sides;
   - friendly invite works;
   - remove friend;
   - block friend requests.
8. Generator:
   - status;
   - claim;
   - upgrade;
   - notification toggle.
9. Inline profile:
   - BotFather `/setinline` first;
   - `@extracards me`;
   - `@extracards <existing_id>`;
   - missing ID no result;
   - verify generated image has no private economy fields.

## Known Context From Earlier Reports

When reviewing agent reports, be skeptical of phrases like "py_compile confirms attack test". Static compile does not prove runtime security.

The project is not released yet, so user is okay with deeper cleanup rather than tiny hotfixes, but wants scope control:

- Do not mix new gameplay features into security patch.
- Do not start Draft yet.
- Current desired feature direction after security: audit deck editor and start friends. Friends MVP is now implemented, but needs QA after auth is fixed.

Potential future features:

- Draft battle mode, but needs design first.
- Telegram inline profile cards are v1; later may add group mention commands.

## Useful Commands

```text
git status --short
git diff --stat
python3 -m py_compile web/server.py infrastructure/config.py main.py
pytest -q
rg -n "init_data and init_data\\.isdigit|rel_url\\.query\\.get\\(\"user_id\"|payload\\.get\\(\"user_id\"|data\\.get\\(\"user_id\"" web/server.py
rg -n "async def disconnect|@sio.event|join_match|client_ready|surrender|MATCH_LOCKS|MATCH_SESSIONS" web/server.py
rg -n "userId =|viewer_id|players\\[String\\(userId\\)|is_my_turn|join_match|_auth" webapp/arena.js
```

