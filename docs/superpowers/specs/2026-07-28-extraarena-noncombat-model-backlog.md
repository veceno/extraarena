# ExtraArena — Non-combat ML Backlog

**Status:** ideas / research backlog; not an implementation plan

**Date:** 2026-07-28

**Scope:** user experience outside the arena

**Worktree:** `gpt-v5implantation`

---

## 0. Frozen decision

- The V5 combat runtime and its current sub-model wiring stay unchanged.
- Any new combat assistants, changes to their orchestration, or a shared combat
  encoder are deferred to **V6**.
- The models in this document must improve an out-of-arena user task, not
  maximize session length or spending.
- Exact game/economy rules remain deterministic. ML may rank or forecast only
  where user behaviour, language, or uncertainty is genuinely involved.

---

## 1. What the project already records

The repository already has useful foundations:

- `user_sessions`: start/end time, duration, ordered `screens_visited`,
  battles and cases opened;
- `onboarding_events` and `user_onboarding`: current step, completion and time;
- `economy_events`, collection, card levels/particles, reward tracks, cases,
  generator state and deck presets;
- notification outbox/schedules, timezone, device and last-seen data;
- clans, clan activity, join requests, friends, community posts, votes and
  polls;
- support tickets, message threads, status changes and audit events.

This means some baselines can be built without a new data-collection campaign.
It does **not** mean the existing data is automatically causal or sufficient
for personalized recommendations. Current production volume and label quality
must be measured before training.

---

## 2. Recommended model backlog

### P0 — useful and realistically buildable

#### ExtraUX Navigator V1

Understands a player's question or difficulty and routes them to one verified
answer and one allowed screen/deep-link.

Examples:

- “Где улучшить карту?”
- “Почему не открывается кейс?”
- “Как получить ключ?”
- “Куда делась награда?”
- “Как вступить в сквад?”

**Implementation boundary:** the model returns only a whitelisted
`intent + screen + article_id`. Facts come from audited templates/RAG; the
model cannot mutate state, make purchases, or invent reward/economy rules.

**Training:** this is the strongest synthetic-first candidate. Generate
paraphrases and hard negatives from UI copy, documentation, routes and support
topics, then calibrate on real resolved tickets. A small text encoder is enough.

**Missing labels:** `resolution_code`, `root_cause`, article used, “helped /
did not help”, and repeat contact.

#### ExtraUX ScreenPrefetch V1

Predicts the next screen from the ordered screen sequence and prefetches only
its data/assets.

This is invisible, low-risk personalization with a measurable UX outcome. The
existing `{screen, ts}` session history supplies direct labels.

**Model:** Markov baseline first, then a small sequence/tree model if it
materially improves prediction.

**Acceptance:** next-screen hit rate, saved p95 navigation latency, and wasted
bandwidth. It must never change what the user sees or can access.

#### ExtraUX ReturnClock V1

Predicts the user's likely next natural return window and chooses a quiet,
opt-in notification time. It does not generate manipulative copy and does not
increase notification frequency.

**Model:** survival analysis or GBDT, not an SLM.

**Existing labels:** session timestamps, timezone, weekday, recent cadence and
last-seen time.

**Prerequisite:** fix session continuity after background/foreground
transitions. The current client ends analytics on `visibilitychange=hidden`,
keeps the same `session_id`, and does not reopen that session on return. This
can truncate Telegram/Android sessions.

The current outbox records sent/failed, not delivered/opened/dismissed or the
result of a deep-link. Therefore V1 should initially predict time only; category
or channel optimization needs additional attribution.

#### ExtraUX IdeaGraph V1

Finds semantic duplicates and clusters ideas, bug reports and support topics.
Before submission, the user sees several likely duplicates and may add a vote
or context instead. Administrators receive merged clusters and a digest.

**Training:** synthetic paraphrases, near-duplicates and hard negatives are
useful here. Calibrate on real moderator confirmations.

**Safety:** never auto-delete or auto-merge a submission. Human confirmation is
required.

#### ExtraUX Support Copilot V1

Routes a ticket, retrieves similar resolved cases, summarizes a long thread and
drafts a reply for an operator.

This can reuse the same semantic encoder as Navigator/IdeaGraph.

**Prerequisite labels:** `resolution_code`, `root_cause`,
`solved_by_article`, and lightweight CSAT.

**Safety:** the operator sends the answer. Payments, account ownership,
complaints and security issues remain hard-rule escalations. Spending,
ExtraPass status and trophies must not determine urgency.

### P1 — high potential, but instrument before training

#### ExtraUX Journey V1

Shows one transparent “Что полезно сделать дальше” card:

- continue onboarding;
- claim an available reward;
- finish or save a deck;
- collect a ready key;
- open an available case;
- respond to a friend/squad invitation;
- inspect a relevant progression milestone.

The candidate set and eligibility are deterministic. A model only ranks valid
options.

Training directly on current screen transitions would merely imitate the old
UI. First add exposure and outcome logging, run a randomized/rule-based pilot,
and optimize useful task completion or explicit satisfaction — not session
length, spending, or “engagement at any cost”.

#### ExtraUX Onboarding Assist V1

Predicts when a newcomer is stuck and selects the smallest sufficient
intervention: ordinary hint, expanded explanation, replay demonstration, or
“continue later”.

The current onboarding tables and screen sequences are a good start, but add:

- step start and abandon;
- wrong/failed action and validation error;
- hint variant and exposure;
- completion after hint;
- explicit “понятно / не помогло”.

Start with deterministic repeated-error rules. Train only after collecting
several thousand properly instrumented onboarding paths.

#### ExtraUX Milestone V1

Predicts a median and an honest interval for a player-selected progression
goal:

- next meaningful Glory Path reward;
- next affordable upgrade;
- complete usable deck;
- next case or generator milestone.

Exact scheduled events and resource arithmetic must be calculated by the
existing deterministic systems. The model predicts only the human cadence
component (“usually one or two sessions”), preferably as calibrated quantiles.

Synthetic economy simulation can pretrain the deterministic portion; real
session data is required for calibration.

#### ExtraUX QuestCurator V1

Ranks valid daily-quest sets by feasibility, variety, expected completion time
and frustration risk. Hard constraints remain rules: never issue an impossible,
unowned-card, purchase-gated or otherwise unfair quest.

Daily quests currently live in the separate `glm-5.2+RegularTasks` worktree and
assign a fixed set, so there are no counterfactual labels yet. Before ML:

- merge/stabilize the feature;
- introduce a catalog of valid alternatives;
- log `assignment_id`, eligible set, exposure, reroll/skip and completion;
- run a small randomized pilot.

Synthetic data may validate feasibility, not predict what humans enjoy.

#### ExtraUX SquadFit V1

Ranks eligible squads using coarse activity overlap, language/timezone bucket,
trophy fit, member activity and predicted 14/30-day mutual retention.

Start with explainable rules and three recommendations. Train only after
logging candidate impressions, selections, joins, exits and exit reasons.

Never auto-join, reveal precise schedules, infer sensitive attributes, or
override privacy/blocklist/full-squad/min-trophy rules.

### P2 — worthwhile later or mainly operational

#### ExtraUX SocialConnect V1

Suggests friends or a repeat friendly match from recent opponents, mutual
social connections and coarse activity compatibility. It needs impression,
accept, decline, repeat-match, removal and block outcomes. Repeated declines
must suppress a candidate.

#### ExtraUX FeedCurator V1

Ranks news, polls and community ideas using freshness, verified tags, votes and
semantic relevance. Pinned/admin messages, exploration and viewpoint diversity
remain reserved rule-based slots.

Personalized ranking requires impression/click/dwell/hide/report events; likes
alone are selection-biased.

#### ExtraUX StabilityRadar V1

An admin-facing anomaly detector across app version, device class, screen
journey, frontend failures and support topics. Its UX value is faster discovery
of regressions.

Add privacy-minimized frontend error, crash and performance events first.
Exclude raw IP and unnecessary user-agent details from model features.

#### ExtraUX Collection Curator V1

Explains which owned cards are new, upgrade-ready or relevant to a selected
account goal, and gives a resource-aware upgrade plan.

This is an out-of-arena UX surface, but scoring the competitive benefit of
cards/decks depends on the combat simulation and overlaps with Assembler.
Therefore its full ML version is **V6-adjacent**. A V5-era implementation should
remain a deterministic collection filter/sorter, not introduce new combat
sub-model orchestration.

#### ExtraUX Librarian V1

Parses collection-search queries such as “дешёвые карты с заморозкой” into
structured filters and returns exact catalog matches.

Synthetic paraphrases are sufficient for cold start. With roughly 50 cards,
this should be an intent encoder plus deterministic catalog query, not a large
model.

---

## 3. Shared non-combat architecture

Do not export one neural network for every product idea by default.

A compact **Journey Encoder V1** can consume a privacy-minimized non-combat
event sequence and account-state snapshot, with separate heads for:

- next-screen prediction;
- onboarding friction;
- next useful goal ranking;
- return-time quantiles.

Language-heavy Navigator, IdeaGraph and Support Copilot may share a separate
small semantic encoder. QuestCurator/Notify should remain contextual
bandits/rankers because they need randomized exposure and causal evaluation.

For tabular/cadence tasks, a calibrated GBDT, survival model or Markov chain is
preferable to an SLM when it wins the same acceptance test.

---

## 4. Telemetry contract required before personalization

Add a common append-only `ux_events` contract:

| Field | Purpose |
|---|---|
| `event_id`, `timestamp` | ordering/deduplication |
| pseudonymous `user_id`, `session_id` | user/session split |
| `surface`, `event_type` | impression/click/dismiss/complete/error |
| `candidate_id`, `rank` | what was actually offered |
| `eligible_candidate_ids` | counterfactual candidate set |
| `policy_version` | rule/model version |
| `experiment_id`, `variant` | causal evaluation |
| minimal `context_snapshot` | only features needed for the decision |
| `outcome_id` / completion window | join exposure to result |

Also add:

- notification `delivery_id`, delivered/opened/dismissed/deep-link outcome;
- deck change history instead of only the latest preset;
- upgrade attempt/success/failure and resource ledger events;
- recommendation and hint feedback;
- squad leave reason;
- support resolution/root-cause/helpfulness labels.

Validation must split by user and forward in time. Random row splitting across
the same user's sessions will overstate quality.

---

## 5. Synthetic-data boundary

### Good uses

- intent paraphrases and hard negatives for Navigator/Librarian;
- duplicate/non-duplicate pairs for IdeaGraph;
- documentation-grounded support question variants;
- economy/progression simulations for Milestone;
- quest feasibility and invariant tests;
- cold-start UI-flow validation.

### Bad uses

Synthetic actors cannot supply trustworthy labels for:

- whether a human liked a recommendation;
- onboarding confusion;
- notification receptivity;
- social compatibility;
- return/retention effects;
- willingness to spend.

Those need real exposure/outcome data and controlled experiments.

---

## 6. Keep deterministic, not ML

- case odds/outcomes and reward eligibility;
- exact generator readiness;
- upgrade legality and price;
- permissions, privacy, cooldowns and squad capacity;
- leaderboard/rating calculations;
- notification quiet hours and hard frequency caps;
- personalized prices, paywalls or propensity-to-pay;
- a generic churn score without a concrete user-helping intervention.

For global chat, the immediate need is to apply the existing moderation
pipeline and rules. A local SafeText classifier may be distilled later from a
deliberately designed, redacted and appeal-aware decision log.

---

## 7. Recommended order

1. **Navigator V1 + IdeaGraph V1** — genuinely synthetic-friendly and useful
   immediately.
2. **Fix session continuity + add `ux_events`.**
3. **ScreenPrefetch V1 + ReturnClock V1** — direct labels and clear offline/
   online metrics.
4. Run a rule-based **Journey/Onboarding** exposure pilot.
5. Train **Journey V1 + Onboarding Assist V1** only after causal labels exist.
6. Add **QuestCurator/SquadFit** after their product surfaces and telemetry
   stabilize.
7. Consider **Milestone, SocialConnect, FeedCurator, StabilityRadar** based on
   measured usage and support cost.

---

## 8. Deferred V6 combat-model backlog

No item in this section changes V5.

- shared combat-state encoder and cheaper multi-head assistance;
- Tactician / plan-horizon head;
- LethalGuard / forced-lethal verifier;
- calibrated Oracle/value-confidence head;
- Sentinel/anomaly guard for suspicious decisions;
- live TimeStamp calibration;
- DeckDoctor / UpgradeOptimum backed by combat evaluation;
- BalanceProbe for patch simulation;
- ModePilot for mode-specific policy selection;
- Mimic for controlled human-style variants.

Before generating synthetic labels for those models, fix deterministic RNG
control: the current auxiliary generator resets the environment RNG while
`core/effects.py` also uses module-level `random`. Otherwise paired seeds are
not truly paired and derived labels can be noisy.
