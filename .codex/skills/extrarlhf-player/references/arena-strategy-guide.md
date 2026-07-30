# ExtraArena Arena RLHF: rules and winning guide

This is the mandatory decision guide for an LLM playing `p1_actor_type="llm"`.
Its purpose is to win the battle, not to produce varied, plausible, educational,
or aesthetically pleasing actions. Read the current compact state after every
action and use this guide as policy; the live state's scaled stats and mechanics
always override the base-card reference below.

## Mission and decision order

At every decision, inspect the complete `legal_actions` list and answer these in
order:

1. **Can I kill the enemy hero now?** Count every ready attack, charge attack,
   targeted damage, deathrattle damage, taunt, shield and armor. If lethal is
   safe, take it immediately. Do not draw or improve the board first.
2. **Can the opponent kill me on its next turn?** Remove only the attackers,
   engines or taunts needed to prevent that, prioritizing efficient trades.
3. **Can I create or preserve a decisive advantage?** Remove a high-impact
   unit, open a taunt, protect a lethal attacker, or make a strongly favorable
   trade.
4. **Otherwise, push damage.** If a face attack is legal and a trade is not
   required by items 2-3, attack the hero. A ready attacker has no stored value
   after the turn, and face attacks receive no retaliation.
5. **Before `end_turn`, use remaining resources.** Make all useful attacks and
   plays, and use a sensible `mana_draw` as a mana sink. Never end with ready
   attackers, a clearly useful playable card, or avoidably wasted mana.

Play slightly more aggressively than a balanced control player. The default is
face pressure; trading requires a concrete reason: preventing lethal, clearing
taunt, killing a dangerous engine, protecting more future damage, exploiting
cleave/death triggers, or obtaining a clearly favorable exchange.

This policy is grounded in 204 valid MiniMax-vs-u29250 battles. Wins and losses
made almost the same number of trades per turn (0.390 vs 0.406), but wins made
2.32x as many face attacks and 2.75x as much face damage per turn. Thirty losses
never attacked face at all; no win had zero face attacks. Therefore the needed
change is **adding pressure**, not blindly eliminating all trades.

## Exact match rules

- The current catalogue has **50 cards**: 7 heroes, 38 warriors and 5 potions.
  Each battle deck has exactly **9 unique cards**: 1 hero plus 8 non-heroes.
  A generated random deck has 5-7 warriors and 1-3 potions.
- Arena levels use a shared centre (normally 3, 5, or 8) with each card varied
  by up to two levels. Warrior stats and mechanic magnitudes scale, so use the
  current state rather than memorized catalogue values.
- The opening hand contains up to the 3 cheapest warriors. Potions and remaining
  warriors begin in the draw pile.
- Hand limit is 4; board limit is 5. With a full hand, the automatic draw is
  skipped rather than burned. Make room before ending the turn when useful.
- A turn allows any number of legal plays, attacks and mana draws. Only
  `end_turn` passes. Unspent mana and unused attacks do not carry over.
- At the start of a player's turn, maximum mana increases by 1 up to 10 and
  current mana refills. Units become ready; a frozen unit instead loses freeze
  and remains unready for that turn. Refreshing shields and regeneration trigger,
  then one card is drawn.
- `mana_draw` draws a card without ending the turn. Successive uses in one turn
  cost 2, then 4, then 6, and so on; the counter resets on the player's next
  turn. It is legal only with hand space, enough mana and a card in deck or
  graveyard. A drawn card can be played immediately.
- Drawing is weighted, not FIFO/top-deck deterministic. Cards absent for longer
  are more likely, and missing cheap/expensive cost bands receive bias. Do not
  assume a specific next card.
- When the deck empties, the graveyard is reset to base state, shuffled, and
  becomes the deck. There is no fatigue damage.
- Normal warriors have summoning sickness. `charge` warriors are ready
  immediately. A unit can attack once while ready and with effective attack >0.
- `taunt` forces ordinary attackers to target a taunt unit and blocks face.
  `bypass_taunt` ignores it. Combat against a unit deals simultaneous return
  damage; combat against the hero has no return damage.
- Shield fully consumes the first incoming damage or blockable effect. Armor is
  applied after shield and reduces each separate damage event. Reflect happens
  only after damage gets through. Lifesteal heals by actual damage dealt.
- Death cleanup cascades after every action. Rebirth resolves before deathrattle;
  then deathrattle and the hero's Crime and Punishment trigger. Hero HP is checked
  after the cascade, so a careless kill can turn a win into a draw or loss.
- Hero HP at or below zero loses; simultaneous hero death is a draw.

## Strong sequencing

Re-read state after **every** submitted action. Effects, deaths, shields, hand
indices, legal targets and lethal math may all change; never reuse a previous
`legal_action_index`.

Recommended sequence is contextual, but these rules prevent common losses:

- Count lethal before doing anything. Use direct damage or bypass-taunt to avoid
  unnecessary trades.
- Pop a shield with the cheapest damage/effect before a large hit, freeze,
  delete or one-time instant kill.
- If taunt blocks lethal, use the minimum resources needed to clear it, then send
  all remaining attackers face.
- Do not replace a free attack with playing a card. Usually secure safe face
  attacks first; change the order only when a buff, charge or targeted effect
  increases current-turn damage or when attack order affects triggers.
- Prefer trades where the target dies and the attacker survives. A suicide that
  does not even kill its target is almost never useful.
- Aim cleave at a central target when that damages both neighbours; reconsider
  after its deathrattle cascade.
- Play charge before ending and attack with it immediately. A charge unit left
  ready is usually pure lost damage.
- Use healing/control only when it changes the race or protects an important
  unit. Do not heal harmless chip damage instead of developing pressure.
- Develop board/direct damage before speculative draw. Use the first 2-mana draw
  when no stronger use exists, especially if it can find a playable answer.
  Avoid paying 4 or 6 unless the search is strategically justified.
- Do not finish with four cards if a useful play can create space for the free
  next-turn draw.
- Never end with an empty hand, a non-empty deck/graveyard and a legal 2-mana
  draw. In the audited wave this happened 108 times, 93 of them in losses.
- With a board of five, `consume_ally` can still be played because it frees a
  slot. It consumes without deathrattle, which may be either an advantage or a
  lost trigger.

## Mechanic tactics and traps

- **Freeze:** makes a unit miss its next readiness. Shield consumes freeze.
- **Delete:** shield consumes it; otherwise the target goes directly to the
  graveyard without deathrattle or rebirth. Prefer it against Ban, Creeper and
  other death-value targets after stripping shield.
- **Consume:** transfers the ally's attack, HP and max HP to Kaneki, sends the
  ally to graveyard, does not trigger its deathrattle, and can free a full board.
- **Creeper:** ordinary death damages every enemy unit and the enemy hero. Do not
  cluster fragile units into its explosion without a reason. Use its death as
  reach when it produces lethal; delete/consume bypasses the explosion.
- **Dostoevsky:** whenever one of that hero's units truly dies, the opposing hero
  directly loses the displayed `crime_and_punishment_N`, bypassing armor,
  reflect and lifesteal. Rebirth is not a true death; delete/consume bypass the
  death trigger. With Dostoevsky, profitable sacrifices become direct face
  damage. Against him, every normal enemy-unit kill also hurts your hero.
- **Saitama:** the first enemy **unit** he attacks during his lifetime is killed
  after combat unless shield blocks it. The one-time power is still spent when
  shield blocks it. It never instant-kills a hero. Preserve or spend the one-time
  kill deliberately; do not waste it on a trivial or shielded unit if a major
  threat is coming.
- **Gojo:** starts shielded and refreshes shield at the start of its owner's turn.
  Remove the shield and finish/control it within the same turn when possible.
- **Soldier Boy (`Солдатик`):** silences mechanics on up to three enemy units;
  shield does not stop silence. Silence removes mechanics, not already stored
  state flags.
- **Saul Goodman (`Соул Гудман`):** shields up to three *other* allied units,
  not himself. Board development before playing him increases value.
- **Geralt:** no target gives Geralt shield; a valid enemy target deals the
  displayed damage. Choose damage for lethal/tempo, shield when it preserves
  more value.
- **Krista:** raises an ally unit's or hero's max HP but does not heal current
  HP. Do not treat it as immediate sustain.
- **Mana Drain:** transfers available mana immediately and records any shortage
  as a drain at the start of the opponent's next turn. Its practical value is
  tempo and next-turn denial, not immediate face damage; gained mana can be
  wasted if already near the cap.
- **Random effects** (Toka, Midoriya) are not deterministic. Take guaranteed
  lethal first and avoid relying on a favourable random roll when a safe line
  exists.

## Card reference

Numbers here are level-1 catalogue values for orientation only. The battle state
contains the actual level-scaled cost, attack, HP and mechanic magnitude; always
calculate from state.

### Heroes

| ID | Card | Base HP | Role |
|---:|---|---:|---|
| 1 | Искатель | 35 | No special mechanic. |
| 3 | Жанна д'Арк | 35 | Adds attack to allied units; wide boards and face attacks gain value. |
| 4 | Аскеладд | 40 | Reflects damage that gets through; repeated small damage can be costly. |
| 5 | Даркнесс | 45 | Armor reduces every damage event; prefer larger hits and board advantage. |
| 6 | Росомаха | 37 | Regenerates each own turn; sustained low pressure is worse than burst. |
| 7 | Олег Тиньков | 23 | Starts with extra mana but low HP; exploit or apply early tempo. |
| 49 | Достоевский | 32 | Allied true deaths deal direct damage to the opposing hero. |

### Potions

| ID | Card | Cost | Effect / tactical use |
|---:|---|---:|---|
| 8 | Глитч-Удар | 1 | Targeted damage to enemy unit or hero; efficient shield pop or reach. |
| 10 | Импульс Бездны | 4 | AOE damage to enemy units, not the hero; use on multi-unit value or a lethal cascade. |
| 11 | Заморозка | 2 | Freeze an enemy unit; shield blocks it. |
| 12 | Кража Маны | 3 | Mana drain; best when next-turn denial changes the race. |
| 13 | Черная Дыра | 5 | Delete an enemy unit without deathrattle/rebirth; shield blocks it. |

### Warriors

| ID | Card | Cost | Base A/H | Main use |
|---:|---|---:|---:|---|
| 14 | Сакура | 2 | 2/2 | Battlecry heals own hero; cheap development. |
| 15 | Тока Киришима | 2 | 2/1 | Random 1-damage battlecry; can pop a shield but is not guaranteed. |
| 16 | Хог Райдер | 4 | 4/3 | Bypasses taunt; face finisher or back-line removal. |
| 17 | Танджиро | 3 | 3/2 | One-use shield; good tempo attacker. |
| 18 | П.Е.К.К.А. | 5 | 5/5 | Armor; durable pressure and efficient trades. |
| 19 | Саб-Зиро | 4 | 3/4 | Targeted battlecry freeze; protect a race or disable a major attacker. |
| 20 | Канеки Кен | 3 | 2/2 | Consumes ally stats without deathrattle and frees a board slot. |
| 21 | Геральт | 5 | 4/5 | Choose self-shield or targeted damage. |
| 22 | Дио Брандо | 8 | 6/6 | Freezes up to three enemies; creates a major attack window. |
| 23 | Сукуна | 7 | 7/5 | Cleaves target neighbours; aim centrally for value. |
| 24 | Годжо Сатору | 9 | 5/6 | Shield that refreshes each own turn. |
| 25 | Сайтама | 10 | 10/10 | One-time instant kill on first attacked enemy unit, never hero. |
| 26 | Мидория | 5 | 5/5 | Casts a random spell on entry; do not depend on a specific result. |
| 27 | Скелет | 1 | 2/1 | Cheap early pressure/trade. |
| 28 | Торфинн | 2 | 4/1 | Fragile high attack; usually pressure face before it is removed. |
| 29 | Штурмовик | 3 | 3/4 | Efficient vanilla body. |
| 30 | Наофуми | 3 | 1/5 | Taunt; protects attackers and alters race math. |
| 31 | Наемник | 4 | 4/5 | Efficient vanilla body. |
| 32 | Зеницу | 4 | 5/3 | Charge; immediate damage/removal. |
| 33 | Маления | 7 | 6/6 | Lifesteal on actual attack damage; race stabilizer. |
| 34 | Крипер | 3 | 2/2 | Deathrattle AOE to all enemies including hero. |
| 35 | Фрирен | 5 | 3/5 | Targeted ally/hero heal; use only when it changes combat/race. |
| 36 | Юни | 2 | 1/2 | Smaller targeted ally/hero heal. |
| 37 | Слайм | 1 | 1/2 | Cheap development. |
| 38 | Хиличурл | 2 | 2/2 | Vanilla development. |
| 39 | Альфонс Элрик | 2 | 1/3 | Cheap taunt. |
| 40 | Стив | 3 | 3/3 | Vanilla development. |
| 41 | Довакин | 4 | 4/4 | Vanilla development. |
| 42 | Атакующий Титан | 6 | 6/6 | Large pressure body. |
| 43 | Лара Крофт | 2 | 3/1 | Bypasses taunt; fragile face reach. |
| 44 | Леви Аккерман | 2 | 3/1 | Charge; immediate face reach or removal. |
| 45 | Солид Снейк | 5 | 5/4 | Taunt with meaningful attack. |
| 46 | Уссоп | 2 | 3/1 | Fragile early pressure. |
| 47 | Солдатик | 7 | 4/5 | Silences mechanics on up to three enemies; shields do not stop it. |
| 48 | Соул Гудман | 7 | 2/4 | Gives shields to up to three other allies. |
| 50 | Бан | 8 | 3/7 | One rebirth before true death/deathrattle processing. |
| 51 | Кинг | 0 | 0/4 | Free taunt; tempo/protection, but no attack at base. |
| 52 | Криста Ленц | 2 | 1/2 | Raises ally unit/hero max HP without healing current HP. |

## Anti-patterns that corrupt playing quality

- Ending the turn while any ready attacker can safely hit face.
- Treating every board unit as something that must be traded away.
- Drawing, healing or developing after a guaranteed lethal is available.
- Saving mana for a future turn; mana does not carry over.
- Paying 4/6 for repeated draws without a specific needed answer.
- Using expensive removal on a shield before stripping it cheaply.
- Killing Creeper or a Dostoevsky unit without including its trigger in hero-HP
  math.
- Assuming catalogue base stats instead of reading scaled live stats.
- Reusing stale hand indices, target IDs or `legal_action_index` after an action.
- Choosing actions to create diverse training data. Quality comes from trying to
  win consistently; the environment supplies deck/state diversity.

## Final pre-`end_turn` checklist

Before submitting `end_turn`, verify all five answers are “no”:

1. Is lethal available now?
2. Is any ready unit still able to attack, especially face?
3. Is there a useful card play that improves damage, tempo or survival?
4. Can a 2-mana draw use otherwise wasted mana and still allow a useful play?
5. Is my hand full, causing the next automatic draw to be skipped unnecessarily?

If any answer is “yes”, act and read the new state again.
