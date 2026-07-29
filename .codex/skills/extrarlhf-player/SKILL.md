---
name: extrarlhf-player
description: "Use when playing ONE battle as the p1 actor (human or llm) in the ExtraArena RLHF arena via MCP: read the mandatory winning guide, poll status, read compact state + legal actions, submit indexed actions, advance the opponent, and finish. Not for model-vs-model (rl p1), which auto-plays."
---

# ExtraRLHF — Level 2: Player Sub-Agent

You play **one battle** as p1 (the left/human side) via the MCP player tools.
Your objective is to **win**, not to generate varied or merely plausible
decisions. Before starting the loop, read
[`references/arena-strategy-guide.md`](./references/arena-strategy-guide.md)
in full and follow its lethal-first, pressure-oriented policy. The process that
calls `start_series` owns the in-memory match, so the same persistent MCP
session must create **and** play it. A fresh process cannot reuse a `match_id`
created elsewhere. Level 1 delegates the full lifecycle, not only a match id.

> **Scope:** p1 must be `human` or `llm`. If `p1_actor_type="rl"` (model-vs-
> model), do **not** use this skill — `submit_action`/`surrender` are rejected
> and p1 is auto-played via `advance_bot`. You'd only call `get_match_status`/
> `get_action_history` to observe.

## Your tools (level-2 set)

| Tool | When |
|---|---|
| `get_match_status` | every poll — lightweight (turn, is_ended, winner_id, is_my_turn, current_player_id, battle_tag, p1_actor_type) |
| `get_state` | each decision: `compact=true, history_limit=8`; nested state + complete indexed `legal_actions` |
| `get_legal_actions` | compatibility/debug; redundant after compact `get_state` |
| `submit_action` | act with `legal_action_index` from the latest state and `compact_response=true` |
| `advance_bot` | if it's somehow the opponent's turn and didn't auto-advance (rare) |
| `surrender` | to concede (e.g. unwinnable / testing) |
| `get_action_history` | to replay/audit without re-fetching full state |

## Mandatory preparation

Read the strategy guide in full before the first player tool call. Treat its
rules as environment documentation and its decision order as the default
policy. Live state wins over base-card numbers because Arena levels scale stats
and mechanics.

The player is not an annotator, observer or diversity generator. Prefer
immediate lethal, prevention of opposing lethal, decisive control, then face
pressure. A face attack is the default when a trade has no concrete tactical
purpose.

## The game loop

```
loop:
  1. ms = get_match_status(match_id)
  2. if ms.is_ended: stop — read ms.winner_id
  3. if not ms.is_my_turn:
       advance_bot(match_id)           # opponent turn (rare; submit_action usually auto-advances)
       continue
  4. state = get_state(match_id, compact=true, history_limit=8)
     legal = state.legal_actions
  5. pick A using the mandatory winning guide
  6. resp = submit_action(match_id,
                          legal_action_index=A.legal_action_index,
                          compact_response=true,
                          history_limit=8)
  7. if resp.result.game_over: stop — read resp.result.winner_id
  → submit_action auto-runs the opponent turn for play_card/attack/end_turn;
    for mana_draw it does NOT pass the turn (you may act again).
```

Safety: cap iterations (~400) to avoid runaway if state misreads.

## Preferred action submission

Every action from compact state has a stable-for-that-state
`legal_action_index`. Submit the integer directly:

```
submit_action(match_id, legal_action_index=N, compact_response=true)
```

This is the required LLM path: do not transcribe UUIDs into a fresh action
object. Re-read compact state after every accepted action; never reuse an index
after the legal set changes.

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

If `submit_action` is rejected, times out or returns an error, mark the entire
battle unusable and report `fail_replay`. Never silently substitute
`end_turn`: that creates structurally valid but behaviorally corrupt data.

## Reading state (`get_state`)

Compact actor-perspective state contains:

- `player.hand`, `player.board`, `player.hero`, mana/max mana, deck count and
  mana-draw count;
- `opponent.board`, `opponent.hero`, mana/max mana and deck count;
- top-level `turn`, `is_my_turn`, current player, complete `legal_actions`,
  bounded history, terminal state and winner.

`legal_actions[].position` maps to legacy `target_position`. Use compact mode
for generation: it retains names, descriptions, mechanics, HP/mana, targets and
the complete legal set while dropping UI-only data. Full state on every tool
turn caused a measured ~4.14M input-token single battle in MiniMax M3; reserve
it for debugging.

## Condensed winning doctrine

- Inspect every legal action and scan immediate lethal before each move.
- Play slightly aggressively: after lethal and essential defence/control,
  default to face.
- Never leave a ready face attack unused without a concrete reason.
- Prefer attacks and useful card plays over unconditional `end_turn`.
- Spend mana productively; use a justified `mana_draw` as a sink.
- `mana_draw` cost grows per use per turn; it doesn't pass the turn — use it
  before `end_turn` when mana allows.
- Trade only to prevent lethal, clear taunt/high-impact threats, protect more
  damage, exploit triggers, or gain a clearly favourable exchange.
- Track the opponent's likely answers from `action_history`.
- On an empty board with no playable card, `mana_draw` then `end_turn`.
- For non-vision models, use state JSON only; do not invoke screenshots.

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
`end_turn_with_play_legal`, mana-draw count, rejections/timeouts and resolved
`weights_hash`. Level 1 aggregates across the fleet.

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md` (action payload details, tool returns),
  `concepts.md` (actor types, decision_source), `data-format.md`
- `../extrarlhf-gen-orchestration/SKILL.md` (L1) — your caller
- [`references/arena-strategy-guide.md`](./references/arena-strategy-guide.md)
  — mandatory rules, mechanics, cards and winning decision order
