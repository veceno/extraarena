---
name: extrarlhf-player
description: "Use when playing ONE battle as the p1 actor (human or llm) in the ExtraArena RLHF arena via MCP player tools: read the mandatory winning guide, poll match status, read state + legal actions, submit indexed actions, advance the opponent, and finish. Not for model-vs-model (rl p1) — that auto-plays via advance_bot."
---

# ExtraRLHF — Level 2: Player Sub-Agent

You play **one battle** as p1 (the left/human side) via the MCP player tools.
Your objective is to **win**, not to generate varied or merely plausible
decisions. Before starting the game loop, read
[`references/arena-strategy-guide.md`](./references/arena-strategy-guide.md)
in full and follow its lethal-first, pressure-oriented policy. The process that
calls `start_series` owns the in-memory match,
so the same persistent MCP session must create **and** play the battle. A fresh
sub-agent/MCP process cannot reuse a `match_id` created elsewhere. Level 1 must
delegate the full lifecycle (start → play → finish), not only a `match_id`.

> **Scope:** p1 must be `human` or `llm`. If `p1_actor_type="rl"` (model-vs-
> model), do **not** use this skill — `submit_action`/`surrender` are rejected
> and p1 is auto-played via `advance_bot`. You'd only call `get_match_status`/
> `get_action_history` to observe.

## Your tools (level-2 set)

| Tool | When |
|---|---|
| `get_match_status` | every poll — lightweight (turn, is_ended, winner_id, is_my_turn, current_player_id, battle_tag, p1_actor_type) |
| `get_state` | each decision: use `compact=true, history_limit=8`; nested state + complete `legal_actions` |
| `get_legal_actions` | compatibility/debug only; redundant after `get_state` |
| `submit_action` | to act; use `legal_action_index` from the latest compact state |
| `advance_bot` | if it's somehow the opponent's turn and didn't auto-advance (rare) |
| `surrender` | to concede (e.g. unwinnable / testing) |
| `get_action_history` | to replay/audit without re-fetching full state |

## Mandatory preparation

Read `references/arena-strategy-guide.md` in full before the first MCP player
call. Treat its exact rules as environment documentation and its decision order
as the default policy. Live state wins over the guide's base-card numbers because
Arena levels scale stats and mechanics.

The player is not an annotator, observer, or diversity generator. It is trying
to defeat the opponent. Prefer immediate lethal, prevention of opposing lethal,
decisive control, then face pressure. A face attack is the default when a trade
has no concrete tactical purpose.

## The game loop

```
loop:
  1. ms = get_match_status(match_id)
  2. if ms.is_ended: stop — read ms.winner_id
  3. if not ms.is_my_turn:
       advance_bot(match_id)           # opponent turn (rare; submit_action usually auto-advances)
       continue
  4. state = get_state(match_id, compact=true, history_limit=8)
     legal = state.legal_actions       # use the complete list, not a truncation
  5. pick action A from legal using the mandatory winning guide
  6. resp = submit_action(match_id, legal_action_index=A.legal_action_index, compact_response=true,
                          history_limit=8)
  7. if resp.result.game_over: stop — read resp.result.winner_id
  → submit_action auto-runs the opponent turn for play_card/attack/end_turn;
    for mana_draw it does NOT pass the turn (you may act again).
```

Safety: cap iterations (~400) to avoid runaway if state misreads.

## Preferred action submission

Every action returned by compact `get_state` and `get_legal_actions` has a
`legal_action_index`. Submit that integer directly:

```
submit_action(match_id, legal_action_index=N, compact_response=true)
```

This is the required path for LLM players: do not copy UUIDs into a new action
object. The server resolves the index against the current legal set and records
the exact native action. Re-read compact state after every accepted action;
never reuse an index after the legal set changes.

## Legacy action payloads (`submit_action` `action`)

```
play_card:  {"type":"play_card", "hand_index":N,
             "target_position":N?, "target_id":id?, "target_is_hero":bool?}
attack:     {"type":"attack", "attacker_id":id, "target_id":id, "target_is_hero":bool?}
end_turn:   {"type":"end_turn"}
mana_draw:  {"type":"mana_draw"}        # spend mana to draw; does NOT end your turn
```
- `hand_index` = 0-based position in `get_state.player.hand`.
- `target_is_hero=true` attacks the hero face; else `target_id` = a board unit id.
- `attacker_id` / `target_id` accept int or string (hero ids are strings).
- A `client_action_id` is auto-set if you omit it.

If `submit_action` is rejected, times out, or returns an error, mark the whole
battle unusable and report `fail_replay`. Never silently substitute
`end_turn`: that produces structurally valid but behaviorally corrupt data.

## Reading state (`get_state`)

Full actor-perspective state (same shape as prod `/api/battle/state`):
- `player.hand`, `player.board`, `player.hero`, `player.mana`,
  `player.max_mana`, `player.deck_count`, `player.mana_draw_count_this_turn`.
- `opponent.board`, `opponent.hero`, `opponent.mana`,
  `opponent.max_mana`, `opponent.deck_count`.
- top-level `turn`, `is_my_turn`, `current_player_id`, `legal_actions`,
  `action_history`, `game_over`, `winner_id`.

`legal_actions[].position` maps to submit field `target_position`. Card play
uses `player.hand[hand_index]`. Do not spend a second MCP call on
`get_legal_actions` every turn.

Use compact mode for generation. It retains names, descriptions, mechanics,
HP/mana, targets, the complete legal set and a bounded recent history, while
dropping UI-only fields. Full state on every tool turn caused a measured
~4.14M input-token single battle in Minimax M3; it is for debugging only.

## Condensed winning doctrine

- Inspect **all** legal actions and scan immediate lethal before every action.
- You are here to beat the bot, not produce diverse decisions. Play slightly
  aggressively: after lethal and essential defence/control, default to face.
- Never leave a ready face attack unused without a concrete tactical reason.
- Prefer attacks and useful card plays over unconditional `end_turn`.
- Spend mana productively (cards, then a justified `mana_draw` as a sink).
- `mana_draw` cost grows per use per turn; it doesn't pass the turn — use it
  before `end_turn` when mana allows.
- Trade only to prevent lethal, clear taunt/high-impact threats, protect more
  damage, exploit triggers, or gain a clearly favourable exchange.
- Track the opponent's likely answers from `action_history`.
- On an empty board with no playable card, `mana_draw` then `end_turn`.
- Preserve card names, mechanics, targets, HP/mana and action history in the
  prompt. Stripping these fields made Minimax M3 skip obvious attacks.
- For non-vision models use state JSON only; do not invoke screenshots.

You're free to play any legal action — the trace records whatever you choose
(`decision_source="llm"` for llm p1, `"human"` for human p1).

## Edge cases
- **Opponent didn't auto-advance**: `submit_action` normally runs the bot turn
  for you; if `get_match_status` shows `is_my_turn=false` after your action,
  call `advance_bot` once.
- **No legal actions on your turn**: shouldn't happen (`end_turn` is always
  legal); if it does, `advance_bot` then re-poll.
- **Stalemate / unwinnable**: `surrender(match_id)` → `{result:{game_over,winner_id}}`.
- **`p1_actor_type="rl"`**: `submit_action` →
  `{"error":"submit_action_unavailable_for_rl_p1"}`; `surrender` →
  `surrender_unavailable_for_rl_p1`. Switch to observe-only (`get_match_status`/
  `get_action_history`) or use `advance_bot` to step the rl p1 model.
- **Match not found after completion**: a completed series' match may be reaped
  from memory; the result is already in the manifest/trace — read
  `get_battle_group_manifest` (Level 1) instead of re-polling.

## Report back to Level 1
When done: `match_id`, `is_ended`, `winner_id`/`winner_side`, `turns_played`,
`my_actions` count, any `fail_replay` reason, and whether `degraded` was seen
on `start_series`. Also report `end_turn_with_attack_legal`,
`end_turn_with_play_legal`, `mana_draw` count, rejections/timeouts and resolved
`weights_hash`. Level 1 aggregates across the fleet.

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md` (action payload details, tool returns),
  `concepts.md` (actor types, decision_source), `data-format.md`
- `../extrarlhf-gen-orchestration/SKILL.md` (L1) — your caller
