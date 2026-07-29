# Nemesis production dataset contract

Nemesis uses one immutable record per completed battle. The canonical record
is stored once as `meta.nemesis_record` inside the existing V5 action journal;
the compact Nemesis dataset is an exported view, not a second database copy
and not a replacement for the V5 action trace.

`features.base` is frozen before battle start and contains both initial decks
with card levels, starting side, game mode/ruleset, catalog and card-parameter
schemas, actor type and explicit nullable model/checkpoint provenance.
`features.extended` contains nullable per-seat snapshots:

- `profile`: wins, losses and trophies;
- `summary`: an allowlisted aggregate of battles completed before
  `feature_cutoff_at`;
- `recent`: a bounded de-identified sequence with result, opponent actor class,
  game mode, deterministic `completed_at`, duration, turns, trophy delta and
  nullable `started_first`. It contains no match/opponent IDs or names.

Every extended snapshot has `captured_at <= feature_cutoff_at <=
battle_started_at`. Terminal status exists only under `label`; the builder
selects pre-match V5 metadata through an allowlist and cannot copy duration,
winner, final turns or post-match trophies into features.

Domains are explicit:

- `human-human`;
- `human-bot` (one human and one automated/model seat);
- `model-model`.

The default NDJSON export replaces participant IDs with side pseudonyms
`p1=1`, `p2=2`. Names, usernames, contacts, opponent identities and client
tokens are outside the contract. `--include-players` is an authorized
diagnostic mode and produces a sensitive private artifact.

`provenance` carries source/campaign/seed, dataset generation, checkpoint mix
and a deterministic grouped-split fingerprint of the unordered exact deck
pair with levels, catalog and ruleset. Repeated or seat-swapped matchups must
therefore remain in one train/validation/test partition. `quality` carries
lite/standard eligibility, sample weight and exclusion reasons. A missing production
`catalog_hash` is explicit: the record remains lite-eligible at reduced weight
but is fail-closed for standard training with `catalog_unavailable`.
Duration and turn count are terminal-only fields under `label`.

Standard eligibility is deliberately narrower than record compatibility:

- `human-human` is eligible only when both pre-match profile/history snapshots
  are present;
- `human-bot` retains the human extension and full model provenance, but is
  marked `human_bot_standard_auxiliary_only`; it is primary data for Lite and
  audit/future-research material for a possible masked/domain-aware auxiliary
  regime. The current canonical Standard trainer excludes it;
- `model-model` is Lite-only;
- a rehydrated V5 trace generation is stored for audit but has zero weight and
  is ineligible, preventing one gameplay match from becoming two labels.

Production snapshots are captured in parallel with a short timeout before the
engine is exposed. Snapshot or collector failures never block gameplay, but
they are explicit in V5 metadata and fail closed for the affected training
target. The history snapshot is de-identified and unifies canonical
`battle_summary` with deduplicated legacy `battle_results`.

The default view is intended for ExtraArena's closed training perimeter. It is
pseudonymized, not anonymous: exact decks, levels, timestamps and profile
statistics can still be indirect identifiers. A third-party export requires a
separate anonymized view (opaque/HMAC battle IDs, relative or bucketed time,
and reviewed profile-stat buckets).

Python interface:

```python
collector = NemesisBattleCollector.from_v5_meta(
    v5_meta_at_match_start,
    feature_cutoff_at="2026-07-28T12:00:00Z",
    extended_by_seat={"p1": p1_snapshot, "p2": None},
)
record = collector.finalize(status="p1_win")
write_nemesis_export([record], "nemesis.jsonl")
```

Production wiring should capture extended data before the engine is exposed,
finalize beside the V5 terminal seal, and persist/export only complete records.

The CLI accepts either canonical Nemesis records or existing V5 export battle
bundles. For a V5 bundle it reads exactly `meta.nemesis_record`; it never
reconstructs features from terminal V5 metadata. Missing, open, invalid and
duplicate records fail closed before the atomic destination replacement.

## Training and matchmaking interpretation

Model-vs-model and human-vs-bot outcomes are policy-conditional labels. They
are useful for Lite mechanics/deck-interaction pretraining, but bot win/loss
rates are not estimates of human win probability: humans differ in action
quality, familiarity, surrender/AFK behavior and deck/profile distribution.
Standard Nemesis therefore needs a human-vs-human fine-tune/calibration set
and human-domain holdouts before it may influence matchmaking.

Evaluation should combine:

- unordered exact-deck-pair grouped splits (leakage prevention);
- player-disjoint and chronological human holdouts (generalization/drift);
- calibration metrics, not accuracy alone;
- slices by trophies, first mover, history length, ruleset/catalog and actor
  domain.

`split_nemesis_training_dataset` always materializes the Lite deck-grouped
assignment. It adds all three Standard assignments only when their joint gates
pass: at least six distinct players, three pairwise-disjoint human-human
battles, three matchup groups and three cutoff cohorts. Otherwise the artifact
remains Lite-ready and records `standard_readiness_blockers`. The primary
Standard assignment maps each export-local player alias to exactly one of
train/validation/test and excludes battles whose two aliases land in different
partitions. Those excluded battle fingerprints and counts are part of the
validated private manifest; MCP surfaces only exact counts and a bounded
sample. Player aliases are split-only metadata and must never be projected
into model features. No one partitioning is claimed to satisfy every holdout.

Because Nemesis will eventually affect which battles occur, its own
matchmaking decisions create a feedback loop. Production training exports
must retain the selection-policy/version provenance; a rollout should keep a
small randomized control lane or logged selection propensities for unbiased
recalibration. Profile statistics are pre-match features, never causal claims
about skill.

## Nemesis Lite Preview

The Preview is trained only on full Arena simulations from non-Ultra V5
checkpoints. It is a policy-mixture deck outcome estimator, not a human
matchmaking model. Its three-class output remains a forward-compatible
contract, but a Preview corpus with very few draws cannot validate or
calibrate the draw class; targeted draw/stalemate data or an explicit binary
shipping policy is required before promotion.
