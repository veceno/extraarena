---
name: extrarlhf-player
description: "Use when playing ONE battle as the p1 actor (human or llm) in the ExtraArena RLHF arena via MCP player tools: poll match status, read full state + legal actions, submit actions, advance the opponent, surrender. Strategy-agnostic — provides the mechanics; the LLM supplies the strategy. Not for model-vs-model (rl p1) — that auto-plays via advance_bot."
---

# ExtraRLHF — Level 2: Player Sub-Agent

You play **one battle** as p1 (the left/human side) via the MCP player tools.
You are strategy-agnostic: this skill gives you the **mechanics** (read state,
pick a legal action, submit, drain the opponent, detect end); you supply the
**strategy**. You are spawned by Level 1 (`extrarlhf-gen-orchestration`) which
hands you a `match_id`.

> **Scope:** p1 must be `human` or `llm`. If `p1_actor_type="rl"` (model-vs-
> model), do **not** use this skill — `submit_action`/`surrender` are rejected
> and p1 is auto-played via `advance_bot`. You'd only call `get_match_status`/
> `get_action_history` to observe.

## Your tools (level-2 set)

| Tool | When |
|---|---|
| `get_match_status` | every poll — lightweight (turn, is_ended, winner_id, is_my_turn, current_player_id, battle_tag, p1_actor_type) |
| `get_state` | when you need the full board (hand, board, hp, mana, deck, action_history, turn) |
| `get_legal_actions` | on your turn — `{legal_actions, is_my_turn}` |
| `submit_action` | to act (play_card / attack / end_turn / mana_draw) |
| `advance_bot` | if it's somehow the opponent's turn and didn't auto-advance (rare) |
| `surrender` | to concede (e.g. unwinnable / testing) |
| `get_action_history` | to replay/audit without re-fetching full state |

## The game loop

```
loop:
  1. ms = get_match_status(match_id)
  2. if ms.is_ended: stop — read ms.winner_id
  3. if not ms.is_my_turn:
       advance_bot(match_id)           # opponent turn (rare; submit_action usually auto-advances)
       continue
  4. legal = get_legal_actions(match_id).legal_actions
  5. pick action A from legal (YOUR strategy)
  6. resp = submit_action(match_id, action=payload(A))
  7. if resp.result.game_over: stop — read resp.result.winner_id
  → submit_action auto-runs the opponent turn for play_card/attack/end_turn;
    for mana_draw it does NOT pass the turn (you may act again).
```

Safety: cap iterations (~400) to avoid runaway if state misreads.

## Action payloads (`submit_action` `action`)

```
play_card:  {"type":"play_card", "hand_index":N,
             "target_position":N?, "target_id":id?, "target_is_hero":bool?}
attack:     {"type":"attack", "attacker_id":id, "target_id":id, "target_is_hero":bool?}
end_turn:   {"type":"end_turn"}
mana_draw:  {"type":"mana_draw"}        # spend mana to draw; does NOT end your turn
```
- `hand_index` = 0-based position in `get_state.hand`.
- `target_is_hero=true` attacks the hero face; else `target_id` = a board unit id.
- `attacker_id` / `target_id` accept int or string (hero ids are strings).
- A `client_action_id` is auto-set if you omit it.

If `submit_action` returns `result.success=false`, retry with `end_turn`; if
that also fails, `surrender` (mark `fail_replay="action_rejected"`).

## Reading state (`get_state`)

Full actor-perspective state (same shape as prod `/api/battle/state`):
- `hand` — your cards (index → `hand_index`).
- `board` / `opponent_board` — units in play with ids, hp, atk, keywords.
- `hp` / `opponent_hp` (or hero objects), `mana` / `max_mana`, `turn`.
- `deck` count, `action_history` (recent moves both sides).
Read what your strategy needs; you don't have to call `get_state` every turn —
`get_match_status` is cheap for polling, `get_state` for decisions.

## Strategy pointers (the skill is mechanics, but)

- Prefer spending mana each turn (cards + `mana_draw` to dig for answers).
- `mana_draw` cost grows per use per turn; it doesn't pass the turn — use it
  before `end_turn` when mana allows.
- Trade favorably on board; go face when you're the beatdown.
- Track the opponent's likely answers from `action_history`.
- On an empty board with no playable card, `mana_draw` then `end_turn`.

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
on `start_series`. Level 1 aggregates across the fleet.

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md` (action payload details, tool returns),
  `concepts.md` (actor types, decision_source), `data-format.md`
- `../extrarlhf-gen-orchestration/SKILL.md` (L1) — your caller