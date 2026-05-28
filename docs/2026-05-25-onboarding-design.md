# Onboarding Design: Mandatory First Battle and Newbie Path

Date: 2026-05-25
Status: discussed and approved as product/design direction
Scope: onboarding UX, script, copy, and technical design notes. This document is not an implementation plan.

## Goal

ExtraArenaRaS needs onboarding because the interface is dense and a new player can easily lose the main idea behind the game. The onboarding should not explain every system at once. Its first job is to make the player understand the core battle goal:

**Victory means reducing the enemy hero to 0 HP.**

Everything else should be taught through short guided actions and later newbie tasks, not through a long tutorial lecture.

## Core Principles

1. The player starts in a controlled battle, not in the menu.
2. The first battle is mandatory for new players and cannot be skipped.
3. The only bypass is logging into an existing account whose onboarding is already completed.
4. The first battle should be short: around 3-5 meaningful player actions.
5. The first battle teaches the goal of combat first, and only lightly introduces card mechanics.
6. The mandatory onboarding should end quickly: welcome, tutorial battle, three menu highlights.
7. Further education happens through a non-blocking "Newbie Path" after the main menu opens.
8. Midoria is the onboarding guide. Her tone is energetic, game-like, and a little bold, but not cringe or toxic.

## Mandatory Flow

The agreed mandatory flow is:

1. Welcome from Midoria.
2. Controlled tutorial battle.
3. Victory and short reinforcement of the core rule.
4. Menu tour through three key sections:
   - Arena
   - Collection
   - Decks
5. Main menu unlocks.
6. Newbie Path appears as a soft next-step route.

The Newbie Path is not part of the hard gate. It should guide without locking the whole app.

## Onboarding Gate

Because there is no skip button, onboarding should be treated as an account gate, not as a closable modal.

Recommended server-side states:

- `not_started`
- `welcome`
- `tutorial_battle`
- `menu_tour`
- `completed`

Rules:

- If onboarding is not `completed`, the main menu should not be freely accessible.
- If the player reloads the app mid-onboarding, they should return to the last saved onboarding step.
- If the player logs into an existing account, the server checks onboarding progress.
- If that account has `completed`, the player enters the game normally.
- If not, the player continues onboarding.

Existing useful infrastructure:

- `user_settings.welcome_shown` already exists, but is too coarse for the new flow.
- `onboarding_events` already exists and can track steps, completion, time spent, and metadata.
- The current analytics endpoint can record onboarding milestones.

## First Battle Design

The selected approach is a hard guided fight. The player can only perform the intended action at each step. Other actions are blocked or corrected with short feedback.

The battle should be deterministic and tuned so the player cannot lose.

### Teaching Objective

The first battle teaches:

- The enemy hero is the main target.
- Cards are played from hand to board.
- A newly played unit may sleep before it can attack.
- Ending the turn is part of combat rhythm.
- Some cards have mechanics.
- Taunt/Provocation means the enemy must attack that unit first.

The first battle does not teach:

- Rarities
- Economy
- Cases
- Shop
- Deckbuilding depth
- Ratings
- Extra Arena modifiers
- Complex mechanics such as Freeze, Shield, Lifesteal, Armor, Cleave

### Key Card: Alphonse Elric

Alphonse Elric is the agreed card for introducing mechanics:

- Card id: `39`
- Name: `Альфонс Элрик`
- Rarity: `start`
- Mana cost: `2`
- Stats: `1/3`
- Mechanics: `["taunt"]`

Reason:

Alphonse is a starter card, so the player already receives him at registration. Taunt/Provocation is intuitive and demonstrates that cards have special behavior without requiring a long explanation.

### Tutorial Battle Script

Meaningful player actions:

1. Player plays an attacking card.
2. Player ends turn.
3. Player attacks the enemy hero.
4. Player plays Alphonse Elric.
5. Player kills the enemy hero.

Between actions:

- The opponent takes scripted turns.
- The opponent plays a strong attacking unit.
- After Alphonse is played, the opponent is forced to attack Alphonse.
- Midoria explains that this happened because of Provocation.

The opponent's attack into Alphonse is not counted as a new player action. It is a visual demonstration.

## Midoria Presentation

The agreed approach is a combination of formats:

1. Cinematic format for welcome and victory.
2. Portrait plus speech bubble for important explanations.
3. Compact bottom bar during battle so the guide does not cover the board.

Existing asset:

- `DesignAssets/MidoriaFixingRobot.png`
- Size: 3000x3000 PNG with alpha

MVP can use this static PNG with CSS animation. Lottie or sprite animation can be added later, but is not required for the first implementation.

Recommended component concept:

- `MascotCoach`
- Supports modes:
  - `cinematic`
  - `portrait`
  - `compact`
- Shows Midoria art, short copy, action button when needed, and optional target/spotlight metadata.

## Midoria Voice Rules

Tone:

- Confident
- Energetic
- Slightly bold
- Game-like
- Not toxic
- Not childish
- Not overloaded with slang

Good pattern:

- One thought per message.
- Short phrases.
- Explain after or during visible action, not before a long abstract lecture.

Avoid:

- Long multi-sentence explanations.
- Babying the player.
- Meme overload.
- Humiliating the player or enemy.
- Technical terms before the player sees the mechanic.

## Approved Copy

### Welcome

Title:

```text
Мидория
```

Body:

```text
Сразу в бой. Меню подождет.
Цель простая: у героя напротив должно стать 0 HP. Я проведу первый раунд.
```

Primary button:

```text
Начать бой
```

Existing account prompt:

```text
Уже есть аккаунт? Войди и продолжим с твоего прогресса.
```

Button:

```text
Войти в аккаунт
```

### Tutorial Battle

Battle start:

```text
Вот твой герой. Вот герой врага. Его HP — наша цель.
```

Step 1: play attacking card.

```text
Ставь бойца на поле. Ему нужно занять позицию, прежде чем бить.
```

UI hint:

```text
Выставь эту карту
```

After card is played:

```text
Он спит до следующего хода. Нормально: только вышел на поле.
```

Step 2: end turn.

```text
Жми конец хода. Пусть враг дернется.
```

Safer alternate:

```text
Жми конец хода. Пусть враг сделает ход.
```

UI hint:

```text
Завершить ход
```

Opponent plays threat:

```text
Враг выставил сильного бойца. Выглядит неприятно, но у нас есть броня потяжелее.
```

Step 3: attack enemy hero.

```text
Наш боец готов. Бей героя, не разменивайся. HP вниз — победа ближе.
```

UI hint:

```text
Выбери бойца и нажми героя врага
```

Step 4: play Alphonse.

```text
Теперь Альфонс. Он закрывает проход к герою.
```

UI hint:

```text
Выставь Альфонса
```

Opponent attacks Taunt:

```text
Видишь? Это Провокация. Враг обязан сначала ударить Альфонса.
```

Step 5: lethal.

```text
Путь открыт. Добивай героя.
```

UI hint:

```text
Атакуй героя врага
```

Victory:

```text
Готово. Главное правило поймано: победа — это 0 HP у героя врага.
Теперь покажу, где собирать отряд и искать новые бои.
```

Button:

```text
В меню
```

### Wrong Action Feedback

Generic:

```text
Сейчас не туда. Следуй подсветке.
```

Wrong target:

```text
Эту цель пока не трогаем. Нам нужен герой.
```

Sleeping unit:

```text
Рано. Эта карта еще спит.
```

Wrong card when Alphonse is required:

```text
Сейчас нужен Альфонс. Он примет удар на себя.
```

Tutorial lock:

```text
Этот бой учебный. Действуем по плану.
```

## Menu Tour

The mandatory menu tour should only highlight three sections. It should not explain the full interface.

### Arena

```text
Арена — сюда за боями. Хочешь прогресс — возвращайся сюда.
```

Button:

```text
Дальше
```

### Collection

```text
Коллекция — здесь все твои карты. Нажимай на карту, чтобы увидеть, что она умеет.
```

Button:

```text
Дальше
```

### Decks

```text
Колоды — здесь собирается твой отряд. Сильные карты сами себя не выберут.
```

Button:

```text
Дальше
```

### End of Mandatory Onboarding

```text
Все, базу ты взял. Дальше игра открыта.
Я оставлю короткий Путь новичка: сделаешь пару задач — получишь награды и быстрее соберешь нормальную колоду.
```

Button:

```text
Открыть Путь новичка
```

## Newbie Path

The Newbie Path appears after the main menu is unlocked. It should be a soft route, not a hard lock.

Recommended number of tasks for MVP: 5.

Title:

```text
Путь новичка
```

Description:

```text
Короткий маршрут без лишней лекции. Делай задачи, забирай награды, усиливай колоду.
```

Tasks:

```text
Открой стартовый кейс
Посмотри новую карту
Сохрани первую колоду
Сыграй обычный бой
Забери награду новичка
```

Completion texts:

```text
Есть. Кейс открыт.
Карта в коллекции. Уже можно думать, кого взять в отряд.
Колода сохранена.
Первый настоящий бой принят.
Маршрут новичка закрыт. Дальше уже по-взрослому.
```

Reward principle:

- Rewards should be small and frequent.
- Good rewards: starter case, coins, 1-2 starter-friendly cards, possibly a cosmetic after a real battle.
- Avoid giving too much currency immediately, because it can pull the player into shop/economy noise too early.

## Technical Direction

### Frontend

Likely frontend pieces:

- `MascotCoach`
- `OnboardingGate`
- `TutorialBattleController`
- `SpotlightOverlay`
- `NewbiePath`

The current app already has modal, overlay, and animation patterns that can be reused. The missing piece is a reusable guided spotlight/tour layer that:

- Highlights a target UI element.
- Blocks non-target actions.
- Positions a compact tooltip or Midoria message.
- Handles mobile viewport constraints.
- Emits analytics events.

### Battle

The tutorial battle should not rely on normal matchmaking. It should be a deterministic scripted match or a special battle mode.

Requirements:

- Fixed player hand.
- Fixed opponent actions.
- Legal actions restricted to the tutorial step.
- Server-side validation of tutorial step, not only client-side blocking.
- Guaranteed lethal setup.
- Resume support if the WebApp reloads.

### Backend

Recommended additions:

- Store onboarding status and current step server-side.
- Use `onboarding_events` for analytics.
- Add tutorial-specific endpoints or extend existing welcome/onboarding endpoints.
- Track completion of:
  - welcome shown
  - tutorial battle started
  - each tutorial battle step
  - tutorial battle completed
  - menu tour completed
  - Newbie Path task completion

### Analytics

Minimum events:

- `welcome_seen`
- `welcome_completed`
- `tutorial_battle_started`
- `tutorial_step_completed`
- `tutorial_wrong_action`
- `tutorial_battle_completed`
- `menu_tour_started`
- `menu_tour_step_completed`
- `mandatory_onboarding_completed`
- `newbie_path_opened`
- `newbie_path_task_completed`

Important metrics:

- Time to complete mandatory onboarding.
- Drop-off by tutorial step.
- Wrong-action count by step.
- Percentage of users who open Newbie Path.
- Percentage of users who complete first real battle.

## Implementation Boundaries

This document does not authorize implementation. The project owner explicitly stated that development should not begin until a separate explicit command is given.

When implementation starts, create a separate implementation plan before writing code.

## Open Decisions for Implementation Planning

These do not block the current design, but should be answered before coding:

1. Which exact attacking starter card should be used in the first tutorial step?
2. What exact enemy hero HP and scripted card stats guarantee the intended 5-action flow?
3. Should the safer alternate phrase "Пусть враг сделает ход" replace "Пусть враг дернется"?
4. Should Newbie Path rewards be configured in code, database, or an admin-controlled config?
5. Should `.superpowers/` brainstorm mockups be ignored permanently in git?
