# Daily Quests (Квесты) — Design

**Date:** 2026-07-15
**Branch:** `glm-5.2/RegularTasks`
**Replaces:** the existing single-reward daily-login ("За вход") system.

## Goal

Replace the single-reward daily-login ("За вход") with a daily quest board of 5 fixed quests. Each day (Moscow midnight) the board resets: progress cleared, claims reset. The player completes quests by playing (login, open a case, win battles, win streak) and claims each quest's reward individually.

## Confirmed decisions

1. **Reset schedule:** fixed Moscow midnight (00:00 MSK = 21:00 UTC). `reset_seconds` counts down to next 00:00 MSK. Uses `MOSCOW_TZ` (`infrastructure/database.py:59`).
2. **Case reward tier:** fixed **T1** (basic case) for all case-reward quests. N cases = N `user_cases` rows of tier 1.
3. **Win eligibility:** mirror existing streak rules — bot wins count, surrender/AFK wins count, draws neither count as a win nor break the streak. Exclude only `friendly`/`training` matches (consistent with `_is_ranked_streak_mode`, `database.py:3663`, and the squad/newbie hooks at `web/server.py:3095`).
4. **Assets:** `DesignAssets/Images/quests.jpg` (sheet art, 3000×3000) and `DesignAssets/MainMenu/Icons/Quests.png` (96×96 black-on-transparent RGBA) are committed into the worktree. The icon auto-renders white at runtime via the existing CSS filter `--icon-white: brightness(0) invert(1)` (`webapp/index.html:1833`, applied to every `.quiet-btn img` at `:3516`) — no manual inversion needed, consistent with `EveryDayLogin.png` and all other menu icons.

## The 5 quests (hardcoded constant `DAILY_QUESTS`)

| id | title | description | target | reward |
|----|-------|-------------|--------|--------|
| `login_once` | Зайти в игру | Просто вернись и награда уже твоя. | 1 | 50 coins |
| `open_case_1` | Открой 1 кейс | Отличный шанс попытать удачу и получить награду! | 1 | 100 coins |
| `win_battle_1` | Выиграй 1 бой | Продвигайся вперед по трофейной дороге! | 1 | 1 case (T1) |
| `win_battle_5` | Выиграй 5 боев | Просто ежедневное боевое крещение. | 5 | 3 cases (T1) |
| `win_streak_5` | Выиграй 5 боев подряд | Я в огне! Я сам огонь! | 5 | 5 cases (T1) |

Quests are fixed/static → hardcoded in Python (mirrors `DAILY_LOGIN_REWARD_PRESETS`, `database.py:808`), not a DB table.

## Architecture

**Approach A (chosen):** dedicated `daily_quests_progress` table + hardcoded quest defs + stored progress (incremented at event hooks). Rejected alternatives: (B) on-the-fly progress from `economy_events`/`battle_summary` — no persisted streak, O(N) status reads, draw divergence; (C) JSONB blob on `users` — harder to query/aggregate/reset than rows.

### Data model

New table, added via the code-driven migration pattern (`_ensure_daily_quests_progress_table()`, registered in `init_schema()`, `SCHEMA_VERSION` 48→49):

```sql
CREATE TABLE daily_quests_progress (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  quest_id TEXT NOT NULL,
  reset_date DATE NOT NULL,              -- Moscow-midnight day
  progress INTEGER NOT NULL DEFAULT 0,
  claimed BOOLEAN NOT NULL DEFAULT FALSE,
  claimed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(user_id, quest_id, reset_date)
);
CREATE INDEX idx_daily_quests_progress_user_date ON daily_quests_progress(user_id, reset_date);
```

`reset_date` = current date in `MOSCOW_TZ`. One row per (user, quest, day). The `UNIQUE(user_id, quest_id, reset_date)` constraint gives idempotent progress upserts; `SELECT … FOR UPDATE` on the row serializes claims.

The old 11 `daily_login_*` columns on `users` (`database.py:2784-2817`) stay dormant (harmless; removing columns is riskier than leaving them). The old `/api/daily-login/*` routes + handlers (`web/server.py:20101-20148, 20560-20561`), `get_daily_login_status`/`claim_daily_login_reward`/`_advance_daily_login_cycle`/`_choose_daily_login_reward`/`_daily_login_position_token` (`database.py:832-1331`), and the `main.py:469-472` notification-scheduler call are **removed**.

### Backend — endpoints (`web/server.py`)

Registered next to the removed daily-login routes:

- `GET /api/daily-quests/status` → `db.get_daily_quests_status(user_id)` → `{enabled, reset_at, reset_seconds, quests:[{id,title,description,reward_type,reward_amount,progress,target,claimable,claimed}]}`.
- `POST /api/daily-quests/claim` body `{quest_id}` → `db.claim_daily_quest_reward(user_id, quest_id)` → `{success, granted:{reward_type,reward_amount}, quest_id}`; HTTP 409 `already_claimed`, 400 `not_claimable`.

Handlers mirror the daily-login try/except + `require_user_id` shape. On success, the handler logs economy via `_track_economy_safe(..., source='daily_quest', ...)` (the DB method also writes its own `economy_events` row — pick ONE to avoid double-logging; the DB method is the single source of truth, so the handler does NOT add a second `_track_economy_safe` call).

### Backend — status assembly (`db.get_daily_quests_status`)

1. `today = datetime.now(MOSCOW_TZ).date()`.
2. Lazy-upsert a progress row for each of the 5 quests for `today` (`INSERT … ON CONFLICT (user_id, quest_id, reset_date) DO NOTHING`). For `login_once`, set `progress = GREATEST(progress, 1)` on this touch (opening the quests status = "зашёл в игру").
3. `reset_at` = next 00:00 MSK as a UTC `datetime`; `reset_seconds = max(0, (reset_at - now_utc).total_seconds())`.
4. For each quest: `progress = min(row.progress, target)` (display-capped), `target`, `claimable = row.progress >= target and not row.claimed`, `claimed = row.claimed`, `reward_type`, `reward_amount`.
5. `enabled = True` (feature flag; default on; `enabled=False` makes the frontend hide the tile, matching the daily-login guard at `webapp/index.html:19347`).

### Backend — claim (`db.claim_daily_quest_reward`)

Transaction (`async with pool.acquire() as conn: async with conn.transaction()`):
1. `SELECT … FROM daily_quests_progress WHERE user_id=$1 AND quest_id=$2 AND reset_date=$3 FOR UPDATE`. If missing, lazy-create then re-select.
2. If `row.claimed` → return `already_claimed` (handler → 409).
3. If `row.progress < target` → return `not_claimable` (handler → 400).
4. Grant reward by `reward_type`:
   - `coins`: `UPDATE users SET coins = GREATEST(0, COALESCE(coins,0) + $N), updated_at = NOW() WHERE user_id = $1`.
   - `case`: insert `reward_amount` (=count) rows: `INSERT INTO user_cases (user_id, case_id, tier, status) VALUES ($1, 1, 1, 'pending')` (T1; `case_id = tier = 1` per `get_admin_case_id`).
   - Log `economy_events` with `source='daily_quest'`, `metadata={quest_id, reset_date, reward_type, reward_amount}`.
5. `UPDATE daily_quests_progress SET claimed = TRUE, claimed_at = NOW(), updated_at = NOW() WHERE …`.
6. Return `{success: True, granted: {reward_type, reward_amount}, quest_id}`.

Idempotency = row lock + `claimed` flag + `UNIQUE(user_id, quest_id, reset_date)`. No `claimed_rewards` overloading needed (the per-day row is the idempotency boundary). Case `reward_amount` here is a **count** (not the tier-as-amount convention of `claim_reward_entries_transaction`); we grant that many T1 rows directly.

### Backend — progress increments (all wrapped in try/except, never block the host flow; mirrors `mark_newbie_path_task` `database.py:22015`)

- `login_once`: set to 1 in `get_daily_quests_status` (above).
- `open_case_1`: `progress = LEAST(progress + 1, target)` at the two case-**claim** (committed) sites — `web/server.py:9430` (keys flow `_claim_case_key_opening`) and `:9534` (user_cases flow `_claim_user_case_opening`) — NOT at `/api/cases/open`. Guard once per `opening_token` (the `opening['claimed']` idempotency at `:9457` already prevents double-claim). Wrap in the existing try/except so quest failures never block a case claim.
- `win_battle_1` / `win_battle_5`: `progress = LEAST(progress + 1, target)` at the battle-end hook `web/server.py:3093-3138`, gated by `tx_result.get('applied')` (idempotency — duplicate `match_id` → `applied=False`) and `eligible_mode` (exclude `friendly`/`training`), for the human winner (`winner_id_int` when it's a human, not a bot winner — i.e. the human won). Bot/surrender/AFK wins count; draws (`winner_id_int is None`) do not increment.
- `win_streak_5`: stored daily streak counter at the same battle-end hook, same gates:
  - human win → `progress = LEAST(progress + 1, target)`
  - human loss → `progress = 0`
  - draw → unchanged (draws invisible, mirroring `_streak_result_for_row` `database.py:3669` + scan skip `:3766`)
  - daily reset → new row starts at 0

A new DB method `increment_daily_quest(user_id, quest_id, delta, *, reset_on_loss=False)` does the atomic upsert (`INSERT … ON CONFLICT … DO UPDATE SET progress = …`), used by all three hooks. `win_streak_5` calls it with `delta=+1` on win and `delta=0,reset=True` on loss.

### Frontend (`webapp/index.html`, then `scripts/precompile_webapp_index.py`)

**ICONS map** (`:5749-5776`): add `quests: ICON_ROOT + 'Quests.png'`.

**Main-menu button** (`:10104-10108`, inside `.quiet-grid`): label `Квесты`, `aria-label="Квесты"`, icon `ICONS.quests`. Badge (`<small>`) + accent class driven by `questsStatus`:

| condition | badge text | accent class |
|-----------|-----------|--------------|
| ≥1 quest claimable | `Награда` | `ready-orange` (orange), no blink |
| all 5 claimed | reset-timer (constant) | none |
| 1–4 claimed, none claimable | `Новые` ⇄ reset-timer every 3s (blink keyframe) | none |
| none done | `Новые` (steady) | none |

`claimableCount = quests.filter(q => q.progress >= q.target && !q.claimed).length`; `claimedCount = quests.filter(q => q.claimed).length`. Reset-timer uses the drift-free 1s-clock pattern (`:9877-9901`) against `reset_seconds` (re-pin `{sec, at:Date.now()}` on each status refresh). New CSS `@keyframes questBadgeBlink` toggling `Новые`↔timer every 3s, scoped under `.mm-redesign`. Keep a single-space `" "` placeholder when no text to preserve tile height.

**QuestsSheet** (replaces `DailyLoginSheet` `:10853-11020`, rendered at `:19721`): reuses GeneratorScreen's visual system (`:10601-10851`) —
- art-background layer → `/DesignAssets/Images/quests.jpg` (the `:10713-10716` pattern, with the responsive `compactHeight` branch at `:10694`).
- sticky glass header (`:10718-10732`): back chevron + `Квесты` h1 + `Сброс` chip showing HH:MM:SS countdown from `reset_seconds`.
- card-shell `<section>` (`:10736`): summary block (`Ежедневные задания` eyebrow / `Забери дневной маршрут` h2 / description / `Выполнено X/5` score chip) + daily-progress bar.
- 5 quest rows: per-quest icon (reuse mockup SVGs), title + status badge (`Готово` claimable / `Забрано` claimed / `В процессе` in progress), description, mini progress bar `progress/target`, reward icon+amount (coins via `IconCoin` `:5744`, cases via the case glyph), claim button (gold `:10798-10803` when `claimable && !claimed`, `✓`/disabled when claimed, `Ждёт`/disabled when not yet completable).
- Per-quest claim `POST /api/daily-quests/claim {quest_id}` → on success `window.reloadFreshProfile()` + `onQuestClaimed()` refetch.
- z-index above the mm-redesign main menu but below toasts (mirror DailyLoginSheet z=200).
- analytics `window.__analytics?.screen('daily_quests')` on open.

**Parent App plumbing** (`:18607-18608, :18852, :18896, :19336-19372, :19684, :19721`): rename `showDailyLogin→showQuests`, `dailyStatus→questsStatus`, `fetchDailyLoginStatus→fetchQuestStatus` (hit `/api/daily-quests/status`, 30s poll + on-open + post-claim, keep the `if (data && data.enabled) setQuestsStatus(data)` guard), `onDailyClaimed→onQuestClaimed`; pass `onQuests` + `questsStatus` into ArenaScreen and mount the sheet. Deep-link `?section=daily_quests` → open the sheet (replaces `:19336`). Drop the vestigial parent `dailyTimerLabel` useMemo (`:19363-1970`); the button uses the ArenaScreen-local label.

**Build:** after every JSX edit run `python3 scripts/precompile_webapp_index.py` (regenerates `index.compiled.js` + `?v=hash`). Never edit `index.compiled.js` by hand. `--check` is non-mutating (exit 1 if stale).

### Label convention (the "Готово → Новые" mapping)

"Готово → Новые" applies to the **main-menu button badge** (the old ready-state indicator text → `Новые`, expanded into the Новые/timer/Награда state machine above). The sheet keeps its own labels: per-quest claimable badge `Готово`, claimed `Забрано`, in-progress `В процессе` (per mockup — semantically correct: a completed quest is "готово" = ready to claim), and the score chip reads `Выполнено X/5` (variant-2 label, to avoid a second `Готово`). This is the recommended, least-confusing reading; if the user instead wants `Готово` gone everywhere, the per-quest claimable badge is the only other place it appears and would need a different replacement label (e.g. `Готово` → `Забрать`), but `Новые` on a completed claimable quest reads oddly and is not recommended.

## Testing

New `tests/test_daily_quests.py` (mirror `test_daily_login.py`'s `_DailyLoginFakeConnection` mock-asyncpg style):
- lazy init creates 5 rows for today; `login_once` auto-progresses to 1 on status fetch.
- `reset_date` rollover: status fetch on a new day creates fresh rows (old rows untouched).
- progress caps at target; `claimable = progress>=target && !claimed`.
- claim success grants coins (UPDATE users) / N T1 cases (N `user_cases` inserts) + `economy_events` row `source='daily_quest'`; second claim → `already_claimed`; claim below target → `not_claimable`.
- `win_streak_5`: +1 on win (cap 5), 0 on loss, unchanged on draw.
- increments are idempotent / guarded by `tx_result['applied']` semantics (mock the battle-end + case-claim hook calls).

Also run `python3 scripts/precompile_webapp_index.py --check` after edits (non-mutating freshness gate). Backend tests via pytest (`pytest.ini`).

## Rollout / migration

- `SCHEMA_VERSION` 48→49 + `_ensure_daily_quests_progress_table()` registered in `init_schema()` → dev auto-migrates on startup (`auto_migrate_on_start=True` in dev). Prod requires the migration step before startup (`verify_schema_ready` raises if version mismatch, `database.py:1556`).
- The `enabled` flag in the status payload lets the feature be turned off per-user/overall (frontend hides the tile when `enabled=false`, matching the daily-login guard).
- Old daily-login backend removed in the same change (frontend no longer references it); old `users.daily_login_*` columns left dormant.