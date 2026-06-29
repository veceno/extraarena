# Обучение на арене — граф-сценарный туториал

Дата: 2026-06-29
Статус: реализовано и слито в `main`
Файлы: `scenarios/onboarding_basic.json`, `onboarding_tutorial.py`, `webapp/arena.js`, `webapp/arena-styles.css`, `extra_orchestra/components/cards_catalog.py`, `tests/test_onboarding_tutorial.py`, `tests/e2e/`

Этот документ описывает **текущую** реализацию обязательного учебного боя: граф-сценарный движок, который прогоняет новичка через реальный бойевой движок по декларативному сценарию. Дизайнерские намерения (почему именно такой бой, какую механику учить) см. в `docs/2026-05-25-onboarding-design.md`; возможности вебаппа для онбординга в целом — в `docs/ONBOARDING_CAPABILITIES.md`. Здесь — как именно устроен сценарий и движок.

> Суть одной фразой: учебный бой больше не «телепортирует» игрока в зашитые в коде позиции — он описан одним v2-графом (`scenarios/onboarding_basic.json`), а `TutorialBattleEngine` проходит этот граф через **настоящий** `execute_action` → `ArenaEnvironment.step`. Летал — настоящий, смерть бойцов — настоящая, состояние наследуется по пути. Редактировать сценарий можно, не трогая Python.

---

## 1. Файлы и компоненты

| Файл | Роль |
|------|------|
| `scenarios/onboarding_basic.json` | **Сценарий** — v2-граф: init-сцена + узлы + рёбра + реплики Мидории + разрешённые действия. Единственный источник контента учебного боя. |
| `onboarding_tutorial.py` | `TutorialBattleEngine(BattleEngine)` — обходит граф через реальный движок; `TutorialArenaEnvironment(ArenaEnvironment)` — подавляет воскрешение убитых бойцов из сброса; детерминированная сборка состояния; валидация/применение действий новичка. Prod-safe: **не импортирует** `extra_orchestra`. |
| `web/server.py` | HTTP-роуты `POST /api/onboarding/tutorial/start` и `POST /api/onboarding/tutorial/action`; сборка/кэширование движка в `active_matches`; 409-обёртка для неверного действия. |
| `webapp/arena.js` | Клиентский onboarding-слой: облачко Мидории, spotlight-подсветка цели, click-guard, бесконечный таймер, кнопки «Понятно»/«В меню». Чистый HTTP — сокет для состояния не нужен. |
| `webapp/arena-styles.css` | Стили onboarding-слоя (`.arena-onboarding-*`, spotlight-оверлей, pulse-анимация). |
| `extra_orchestra/components/cards_catalog.py` | Поле `max_hp_override` в `build_instance` — чтобы hero-карты с урезанным HP корректно рисовались в orchestra preview/export (тут же используется и сценарием). |
| `tests/test_onboarding_tutorial.py` | 28 unit-тестов: happy-path по шагам, wrong-card, legal_actions, choose_target (правильный + неверный → 409), метки/`final_step`. |
| `tests/e2e/onboarding_tutorial_arena.js` + `_dump_onboarding_fixtures.py` | Playwright e2e (без `visual_input`): дамп реального состояния в `onboarding_fixtures.json`, мок `/api/*`, драйв arena.js его же функциями. |

---

## 2. Сценарий — структура графа (главное)

Сценарий — это файл схемы `extra_orchestra.scenario.v2` (той же, что использует ExtraOrchestra для реплеев/экспорта). Поля верхнего уровня:

```jsonc
{
  "schema": "extra_orchestra.scenario.v2",
  "name": "onboarding_basic",
  "match_id": "onboarding",
  "seed": 42,                 // детерминизм: instance_id карт и RNG боя
  "viewer_side": "p1",        // новичок — p1
  "game_mode": "classic",
  "classic_params": {          // параметры режима боя
    "turn_duration_seconds": 99,
    "mana_per_turn": 1,
    "sudden_death_enabled": false
  },
  "graph": { "start": "s0", "nodes": [...], "edges": [...] }
}
```

`classic_params` попадает в `ClassicParams` движка (`TutorialBattleEngine._classic_params_from`), поэтому и арена, и сериализатор согласованы со сценарием. `turn_duration_seconds: 99` — на клиенте всё равно показывается ∞ (см. §6), это просто «достаточно много» серверного запаса.

### 2.1. Узлы графа

Два `kind`:

**`scene`** — без действия, чисто презентационный/опорный узел.
- `{ "type": "init", ... }` — стартовая сцена (одна, `graph.start`). Описывает обоих игроков, их hero/hand/board/deck, ману, сторону хода. Из неё однажды строится `ArenaEnvironment`.
- `{ "type": "hold", "display_ms": 1200 }` — пауза-сцена; несёт `tutorial` (обычно beat-шаг: Мидория говорит, новичок жмёт «Понятно»).

**`action`** — узел-действие. Имеет `side` (`"p1"` / `"p2"`) и поле `action`. Действие **p1-узла с `tutorial.allowed`** ждёт новичка; **silent-узлы** (без `tutorial` — ходы противника, end_turn-узлы) движок применяет автоматически при продвижении по графу (см. §4).

### 2.2. `tutorial`-объект шага (schema)

Только узлы с `tutorial` становятся «шагами» обучения (см. §3). Поля:

| Поле | Назначение |
|------|------------|
| `id` | Строковый id шага (`goal`, `play_attacker`, `sleep`, `choose_target`, `taunt_demo`, `lethal`, `victory` …). Используется в `scripted_event` и в тестах. |
| `message` | Реплика Мидории (основной текст в облачке). |
| `hint` | Короткая подсказка-действие (на кнопке/над ней) или `null`. |
| `target` | Куда наводит spotlight: `"opponent_hero"`, `"hand_card:<card_id>"`, `"board_card:<card_id>"`, `"end_turn"`, `null`. |
| `allowed` | **Что разрешено сделать новичку на этом шаге** — см. варианты ниже. |
| `after` *(опц.)* | Реплика-«после» для показа после выполнения action-шага (через `previous_message`/`after_message`). |
| `is_auto_step` *(опц.)* | `true` — шаг без действия игрока, авто-продвижение (на клиенте статус «Ход противника…», нет кнопки). |
| `auto_advance_delay_ms` *(опц.)* | Задержка авто-продвижения (мс). Пробрасывается в payload как `auto_advance_delay_ms`. |
| `wrong_feedback` *(опц.)* | Per-step override текста ошибки — `{ "wrong_target" \| "wrong_card" \| "sleeping_unit" \| "generic": "..." }`. См. §5. |

### 2.3. Варианты `allowed`

| `allowed.type` | Поведение |
|----------------|-----------|
| `continue` | Beat-шаг: новичок жмёт «Понятно», происходит `_advance_to_next_step` (применяются промежуточные silent-узлы до следующего шага). |
| `auto_continue` | Авто-шаг: клиент продвигает сам по таймеру (`auto_advance_delay_ms`). Движок возвращает `scripted_event` (`opponent_attack_taunt` при выходе с `taunt_demo` и т.п.). |
| `complete` | Финал: бой завершён, `game_over: true, winner_id: <user>`, клиент показывает победу и уводит в меню-тур. |
| `play_card` | Новичок играет карту `{ "card_id": <id> }`. `validate_tutorial_action` сверяет `card_id` (или `hand_index` → card_id). |
| `end_turn` | Новичок завершает ход. |
| `attack` | Новичок атакует `{ "attacker_card_id": <id>, "target_is_hero": bool, "also_allow_minion_targets"?: bool }`. |

### 2.4. Init-сцена — spec карт и hero

`p1`/`p2` описывают каждого игрока: `user_id`/`nickname`/`is_bot`/`mana`/`max_mana` + `hero` + `hand`/`board`/`deck`. Карта описана spec'ом:

```jsonc
{ "card_id": 37, "level": 1, "attack_override": 4 }   // Слайм, урезанная атака
{ "card_id": 1, "level": 1, "hp_override": 8, "max_hp_override": 8 }  // hero врага 8/8
```

Поддержанные overrides (`_build_card_from_spec`): `mechanics_override`, `attack_override`, `hp_override`, `max_hp_override`, `is_ready`, `is_frozen`. `max_hp_override` — отдельное поле: hero врага в онбординге имеет 8/8 (а не 8/35), и корректная полоска HP нужна и в orchestra preview/export, и в бою.

`instance_id` каждой карты считается **детерминированно** по позиции — `_deterministic_instance_id`: `uuid5(DNS, "extra-orchestra:{seed}:{side}:{zone}:{index}:{card_id}:{level}")`. Та же формула/seed, что `cards_catalog.deterministic_instance_id` в extra_orchestra → dev-реплеи (`preview_frames`) дают **идентичные** instance_id боевому коду. Реальные `user_id` берутся снаружи (новичок + бот-оффсет), не из плейсхолдеров 1001/2002 — на instance_id это не влияет (он side-based).

> Поэтому `scenarios/onboarding_basic.json` можно прогнать через orchestra `validate_scenario`/`preview_frames` и получить ровно то состояние, что будет в проде — это и есть dev-инструмент авторства сценария (см. §8).

### 2.5. Silent-узлы

Узлы без `tutorial` — служебные действия, которые движок применяет сам при `_advance_to_next_step`: `p2_play`, `p2_end`, `p1_end`, `p2_attack`, `p2_end2`. Их `action` описан так же, как action-узлы новичка, но `side: "p2"` (или `p1` для end_turn после taunt_intro). По рёбрам графа они вставлены между шагами в нужном порядке.

### 2.6. Рёбра и линейный путь

`graph.edges` — список `{ "from", "to" }`. Граф здесь линейный (каждый узел имеет ≥ одно исходящее ребро; движок берёт первое исходящее — `outgoing.setdefault(edge["from"], edge["to"])`). `_build_linear_path` разворачивает граф в список узлов от `start` до конца; `_TUTORIAL_STEP_NODES` — фильтр узлов с `tutorial`; `TUTORIAL_STEPS` и `TUTORIAL_FINAL_STEP` выводятся из графа **при загрузке модуля** (`onboarding_tutorial.py` верхнего уровня). Поэтому `web/server.py` (и любой код, читающий `TUTORIAL_STEPS.get(step).get("id")`) менять не нужно — шаги и их id берутся из сценария.

---

## 3. `onboarding_basic` — полный разбор

Граф содержит 16 узлов; 11 из них несут `tutorial` (step0…step10, `final_step = 10`). Путь:

| # | id узла | kind | tutorial.id | allowed | Реплика Мидории (сокр.) / что происходит |
|---|---------|------|-------------|---------|------------------------------------------|
| 0 | step0 | scene(hold) | `goal` | continue | «Побеждает тот, кто опустит HP героя врага до нуля…» |
| 1 | step1 | action p1 | `play_attacker` | play_card 37 | «Ставим бойца на поле. Он не атакует сразу…» — новичок выставляет Слайма |
| 2 | step2 | action p1 | `sleep` | end_turn | «Видишь метку сна?… Теперь, заверши ход.» |
| — | p2_play | action p2 | — | — | *(silent)* противник выставляет Стива |
| 3 | step3 | scene(hold) | `threat` | continue | «У врага появился боец. Но наша цель всё ещё герой…» |
| — | p2_end | action p2 | — | — | *(silent)* end_turn противника |
| 4 | step4 | action p1 | `choose_target` | attack hero (also minions) | «Слайм готов. Попробуй выбрать, куда выгоднее ударить.» — **первый выбор цели** (см. §5) |
| 5 | step5 | scene(hold) | `tempo` | continue | «Отлично… Считай, сколько ударов осталось до 0 HP.» |
| 6 | step6 | scene(hold) | `danger` | continue | «Стив ударит больно… Нужен защитник.» |
| 7 | step7 | action p1 | `taunt_intro` | play_card 39 | «У Альфонса Провокация…» — новичок выставляет Альфонса |
| — | p1_end | action p1 | — | — | *(silent)* end_turn новичка |
| 8 | step8 | scene(hold, 5600ms) | `taunt_demo` | auto_continue | «Враг атакует Альфонса…» — авто-демо Провокации (`is_auto_step`, 5600ms) |
| — | p2_attack | action p2 | — | — | *(silent)* Стив бьёт Альфонса → тот умирает (реальная обработка смерти) |
| — | p2_end2 | action p2 | — | — | *(silent)* end_turn противника |
| 9 | step9 | action p1 | `lethal` | attack hero | «Теперь путь открыт… Добивай.» — реальный летал 4→0 |
| 10 | step10 | scene(hold) | `victory` | complete | «Готово. Победа — это 0 HP у героя врага…» → меню-тур |

Механическая последовательность действий та же, что и в прежнем туториале → экономика маны/летала не изменилась и проверена: шаг4 — настоящий урон героя 8→4 (Слайм `attack_override: 4`); шаг9 — настоящий 4→0 (`P1_WIN`). Стив на шаге4 — `targetable-enemy` (можно тапнуть, но «неправильно»); на шаге9 (lethal) Стив `attack-target-disabled` → подсветка только героя (см. §6).

---

## 4. Движок: `TutorialBattleEngine`

`TutorialBattleEngine(BattleEngine)` — сабкласс **прод-движка** (НЕ `OrchestraBattleEngine`): ходы применяются через настоящий `execute_action` → `ArenaEnvironment.step`, состояние наследуется по пути графа.

### 4.1. Детерминированная сборка и реплей

- `_build_initial_arena` строит `TutorialArenaEnvironment` **однажды** из init-сцены (`apply_start_effects=False` — init остаётся ровно как описал автор; start-of-turn эффекты следующих ходов честно применятся через `end_turn`).
- `_replay_to_step(step)` — детерминированный реплей из init до шага N: monkeypatch-ит `core.effects.random = Random(seed)` (блокирует случайность) и авто-применяет все action-узлы **перед** step-узлом N (`_apply_node` → `execute_action`). При конструкторе `TutorialBattleEngine(user_id=…, tutorial_step=k)` сразу реплеится до k — это и есть «восстановление посредине онбординга» при перезагрузке.
- Если action-узел провалился (`result.success is False`), движок кидает `RuntimeError` — это баг авторства сценария (нелегальный ход/side-guard), не заглушается: сценарий должен быть провалидирован в dev (см. §8).

### 4.2. Продвижение по графу — `_advance_to_next_step`

После действия новичка (или `continue`) движок идёт по пути вперёд и **авто-применяет все silent action-узлы** до ближайшего step-узла. Так `p2_play`/`p2_end`/`p2_attack`/`p1_end` выполняются сами, без действий игрока. Возвращает результат последнего применённого действия (нужен для `game_over` при летале).

### 4.3. `apply_tutorial_action(action)` — валидация до мутации

`validate_tutorial_action` проверяет действие **до** `execute_action` → неверное действие не меняет состояние:
- `type` должен совпадать с `allowed.type`;
- `play_card`: `card_id` (или `hand_index`→card_id) должен совпадать с `allowed.card_id`;
- `attack`: `bool(action.target_is_hero)` должен совпадать с `allowed.target_is_hero` (← это и ловит тап миньона на hero-шаге), и `attacker_id` должен разрешаться в карту с `card_id == allowed.attacker_card_id`.

Если валидация провалилась → `{"success": False, "error": "tutorial_wrong_action", "feedback": <text>, "tutorial_step": k}`, **состояние не трогается** (новичок повторяет). Сервер оборачивает это в **HTTP 409** (см. §7).

`continue` / `auto_continue` / `complete` — beat-шаги: выполняют `_advance_to_next_step` (complete ещё ставит `game_over`). `play_card`/`attack`/`end_turn` — action-шаги: сначала `execute_action` (реальная мутация), потом `_advance_to_next_step`. При `game_over` или достижении `final_step` в payload добавляется `game_over: true, winner_id`.

### 4.4. `_tutorial_legal_actions` — что подсветить арене

Возвращает список «разрешённых» действий для **текущего шага** — арена по нему рисует targetable-цели и кнопки:
- `play_card` → одна запись с `hand_index` нужной карты;
- `end_turn` → `[{type:"end_turn"}]`;
- `auto_continue` → `[]` (клиент ведёт таймер);
- `attack` → hero-запись (`target_is_hero` из `allowed`); если `also_allow_minion_targets` — **дополнительно** по записи на каждый миньон `state.p2.board` (`target_is_hero: false, target_id: minion.instance_id`). Это расширяет только подсветку/тапабельность — валидацию всё равно блокирует `target_is_hero`-мисматч (см. §5);
- `continue`/`complete` → `[]` (кнопку рисует клиент по `allowed.type`).

### 4.5. `tutorial_payload` — поля для клиента

`step_index`, `step_id`, `message`, `hint`, `target`, `allowed`, `is_auto_step`, `player_step`/`player_steps_total` (= `display_step`/`display_steps_total`, для прогресс-метки «Шаг N/10»), `previous_message`, `final_step`, `wrong_action_feedback` (все `WRONG_ACTION_FEEDBACK` строки), `midoria_asset`, опц. `auto_advance_delay_ms`, и snapshot-ы instance_id: `attacker_instance_id`, `alphonse_instance_id`, `hand_attacker_instance_id`, `hand_alphonse_instance_id`.

### 4.6. `TutorialArenaEnvironment` — почему нельзя убрать

Сценарий скриптовый: колоды пустые, добора карт нет. Базовый `_handle_end_turn` честно добирает 1 карту в начале хода; при пустой колоде `draw_one_from_deck` **reshuffle'ит graveyard → deck → hand** → убитый Альфонс возвращается в руку новичка на следующем конце хода. Override:

```python
class TutorialArenaEnvironment(ArenaEnvironment):
    def _handle_end_turn(self, player, opponent):
        opponent.graveyard.clear()   # drawer = тот, кто добирает → чистим его сброс
        super()._handle_end_turn(player, opponent)
```

Чистим graveyard **того, кто добирает** (`opponent` относительно действующего) → `draw_one_from_deck` идёт по пути fatigue (пустая колода + пустой сброс → `return False`), убитые бойцы не воскресают. Остальная end-of-turn логика (смена хода, мана, пробуждение, реген) — как в базе. Будущий «cleanup», убирающий этот сабкласс, возвращёт баг «Альфонс вновь в руке после смерти».

---

## 5. Механика «первый выбор цели» (особое внимание)

Шаг `choose_target` (step4) — единственная нетривиальная педагогическая механика: новичок может тапнуть **Стива** (миньона врага) или **героя**. Тап Стива механически легален, но педагогически «неправилен» → сервер отклоняет с кастомным фидбэком, состояние не меняется, новичок повторяет, тапая героя.

Связка (всё уже в коде, новой серверной логики не требуется):

1. **`allowed.also_allow_minion_targets: true`** на шаге → `_tutorial_legal_actions` эмитит дополнительно по одной attack-записи на каждый миньон `state.p2.board` (`target_is_hero: false`). → arena `highlightAttackTargets` маркирует Стива `targetable-enemy`.
2. **Per-step `wrong_feedback.wrong_target`** = «Можно, но сейчас выгоднее бить героя: его HP — условие победы.» → едет в payload шага → при ошибке поднимается как `result.feedback`.
3. **`validate_tutorial_action`** отклоняет тап миньона через `bool(action.target_is_hero) != bool(allowed.target_is_hero)` (на hero-шаге `target_is_hero` игрока = false ≠ allowed true). Валидация **до мутации** → сервер возвращает **409** `{result:{success:false, error:"tutorial_wrong_action", feedback:<custom>}}`, клиент ловит `!response.ok` → toast 2400мс (`onboardingFeedbackMessage`), **ре-рендера нет**, шаг не меняется → повтор.
4. **Клиент пропускает тап миньона к серверу**: `getOnboardingAllowedClickSelectors` для ATTACK-mode включает `#opponent-board-zone .board-unit-card.targetable-enemy` (иначе click-guard бы проглотил тап). Spotlight остаётся на правильной цели (герой), но миньон тапабелен — это и есть «можно, но не надо».

Шаг `lethal` (step9) `also_allow_minion_targets` **не ставит** → Стив там `attack-target-disabled` → подсветка только героя (см. §6.3). Таким образом флаг включает «лишнюю» цель ровно там, где нужна педагогика выбора.

Per-step `wrong_feedback` — индексо-независимый override: `_wrong_feedback_for` сначала ищет причину (`wrong_target`/`wrong_card`/`sleeping_unit`/`generic`) в `step.wrong_feedback`, иначе берёт `WRONG_ACTION_FEEDBACK[...]`. Константы (`onboarding_tutorial.py`):

```python
WRONG_ACTION_FEEDBACK = {
    "generic":        "Сейчас не туда. Следуй подсветке.",
    "wrong_target":   "Эту цель пока не трогаем. Нам нужен герой.",
    "sleeping_unit":  "Рано. Эта карта еще спит.",
    "wrong_alphonse": "Сейчас нужен Альфонс. Он примет удар на себя.",
    "tutorial_lock":  "Этот бой учебный. Действуем по плану.",
}
```

---

## 6. Клиент — `webapp/arena.js` onboarding-слой

Чистый HTTP: арена читает `payload.state` из каждого ответа `/api/onboarding/tutorial/action` (через `sendOnboardingTutorialAction`); сокет для состояния не нужен. Ложный `window.io` обязан отдавать `socket.io` Manager-stub (arena зовёт `socket.io.on('reconnect_failed')`).

### 6.1. Детектор и разводка действий

- `isOnboardingTutorialState(state)` — `is_onboarding_tutorial` || `game_mode==='tutorial'` || `match_id` начинается с `tutorial-`.
- `sendOnboardingTutorialAction(action)` — POST `/api/onboarding/tutorial/action`; на `!response.ok` кидает ошибку с `error.feedback` → `handleOnboardingActionError` показывает toast.
- `renderOnboardingTutorialLayer()` — пересобирается на каждом state-update: облачко Мидории (`.arena-onboarding-coach` + `.arena-onboarding-bubble`), метка «Шаг N/10» (или «Демо N/10» для auto), кнопка «Понятно» (на любом `allowed.type==='continue'`-бите) / «В меню» (на финале). Победа (step10, `allowed.complete`) — отдельный `.arena-onboarding-victory` модал.

### 6.2. Spotlight — подсветка цели (оверлей)

`.arena-onboarding-layer` (z-7600, `pointer-events:none`) держит **один** `.arena-onboarding-spotlight` прямоугольник = `border:2px solid #f5921e` (жёлтая рамка) + `box-shadow:0 0 0 9999px rgba(5,3,14,0.62)` (распространение 9999px **затемняет всё вне** прямоугольника — та самая «затемнённая область»). `getOnboardingSpotlightElement()` выбирает элемент под spotlight по `tutorial.target`; `positionOnboardingSpotlight()` накладывает прямоугольник по `getBoundingClientRect()` и вешает `.arena-onboarding-target-pulse` (оранжевый `drop-shadow`, z-7700 = **над** оверлеем) на цель через `getOnboardingPulseTargets`.

Особый случай — `choose_target` (attack + ATTACK-mode + на поле есть `#opponent-board-zone .board-unit-card.targetable-enemy`): spotlight наводится на `.arena-zone-top` (контейнер, оборачивающий **и** hero-панель, **и** доску врага), а pulse едет **на героя И на каждый** `.targetable-enemy` миньон (подняты к z-7700). Иначе Стив (в отдельном `#opponent-board-zone` вне `.opponent-panel-root`) оказался бы **вне** прямоугольника → под тёмным оверлеем → читался бы «недоступен». Lethal остаётся hero-only (нет `targetable-enemy`) → spotlight на `.opponent-panel-root`, pulse только на герое.

### 6.3. Coach placement, click-guard, таймер (правки 2026-06-29)

- **Облачко сверху/снизу** — `getOnboardingCoachPlacement(tutorial)`: `'top'` для `allowed.type === 'end_turn' || 'play_card'` (действуют по низу — рука/свое поле, нижнее облачко закрыло бы карты), иначе `'bottom'` (цель вверху). Облачко пересобирается каждый рендер → позиция переоценивается по шагу.
- **Click-guard** — `installOnboardingClickGuard` + `getOnboardingAllowedClickSelectors` пропускают только разрешённые клики; ATTACK-mode разрешает `.targetable-enemy` миньонов (см. §5).
- **∞ вместо таймера** — `updateTurnTimer(state)` ранним возвратом ставит `timerText.textContent = '∞'` (без `setInterval`, без warning/critical), когда `isOnboardingTutorialState(state)`; тот же гвард в сокетном `turn_start` и в `renderTurnTimerModal` (модалка: «Обучение — без ограничения времени», кольцо 0°). Реальные бои — обычный отсчёт `turn_duration` (по умолчанию 25с), не трогается.
- **Кнопка «Понятно»** — рендерится на **любом** `tutorial.allowed.type==='continue'`-бите (а не только на первом/последнем шаге), иначе срединные beat-шаги (threat/tempo/danger) остались бы без кнопки и без авто-продвижения → новичок застрял бы.

`arena.js`/`arena-styles.css` входят в `BATTLE_SHELL_STATIC_FILES` → отдаются `no-store` → правки оживают **без** рестарта 8081 и без bump'а `?v=`.

---

## 7. HTTP API

| Метод | Путь | Тело | Ответ |
|-------|------|------|-------|
| POST | `/api/onboarding/tutorial/start` | `{}` | `200` `{success, match_id, redirect_url:"/arena?id=<match>&onboarding=1", state:<full_state>, onboarding}`. Если `completed` — только `{success, onboarding}`; если `menu_tour` — `{success, redirect_url:"/?onboarding_menu=1", onboarding}`. Ставит статус `tutorial_battle`, строит/переиспользует движок в `active_matches`, пишет `track_onboarding_event("tutorial_battle_started")`. |
| POST | `/api/onboarding/tutorial/action` | `{type, hand_index?, card_id?, attacker_id?, target_id?, target_is_hero?}` | `200` `{result:{success:true, tutorial_step, game_over?, winner_id?}, state:<full_state>, onboarding, ...}`. При неверном действии — **`409`** `{result:{success:false, error:"tutorial_wrong_action", feedback}, state, onboarding, feedback}`. |

`type` может передаваться и как `action` (legacy). `complete` идемпотентен: если онбординг уже завершён/в меню-туре — сразу `redirect_url:"/?onboarding_menu=1"` без повтора. Действия кэшируются по `client_action_id` (идемпотентность ретраев).

Движок живёт в `request.app["active_matches"][match_id]` (`game_mode`=`"tutorial"`); `_ensure_tutorial_engine_for_user` реконструирует его по `onboarding_state.tutorial_step`, если выпарился (→ детерминированный реплей восстанавливает состояние).

---

## 8. Dev-инструменты авторства сценария

- **Orchestra `validate_scenario` / `preview_frames`** (`mcp__extra-orchestra__*`) — прогоняют **тот же** граф через тот же `core`-движок с тем же seed/uuid5 → предсказывают прод-состояние **точно**. Автор правит `scenarios/onboarding_basic.json`, валидирует (ходы легальны, мана сходится, Стив на доске после step3, hero 8→4 на choose_target, Альфонс умирает на taunt-атаке, финальный кадр `P1_WIN` с hero 0), и только потом пушит. Неизвестные tutorial-поля (`also_allow_minion_targets`, `wrong_feedback`) оркестра игнорирует → валидация не ломается.
- **E2E** — `tests/e2e/_dump_onboarding_fixtures.py` дампит реальный `TutorialBattleEngine.get_full_state` + ответы действий в `onboarding_fixtures.json`; `onboarding_tutorial_arena.js` мокает `/api/battle/state` + `/api/onboarding/tutorial/action` и драйвит arena.js его же action-функциями + кнопками «Понятно»/«В меню». 47 ассертов (вкл. wrong-path: тап Стива → 409 → toast → состояние не изменилось → тап героя → tempo 8→4). **Без `visual_input`** — только DOM.

---

## 9. Как отредактировать сценарий

1. Правь `scenarios/onboarding_basic.json` (реплики, шаги, `allowed`, `wrong_feedback`, init-сцена). Для новой педагогической «лишней» цели — `also_allow_minion_targets: true` + `wrong_feedback.wrong_target` (см. §5).
2. Прогони `validate_scenario` + `preview_frames` (§8) — убедись, что экономика/летал сходятся.
3. Перегенерируй фикстуры: `PYTHONPATH=. python3 tests/e2e/_dump_onboarding_fixtures.py`; прогони `NODE_PATH=<main-repo>/node_modules node tests/e2e/onboarding_tutorial_arena.js`.
4. Юнит-тесты: `PYTHONPATH=. python3 -m pytest tests/test_onboarding_tutorial.py -q` (28 тестов; шаги/id/сообщения/assert-ы обновятся под новый сценарий — правь тесты под новый layout).

**Рестарт-нюанс:** `onboarding_tutorial.py` + сценарий грузятся **при import** модуля в процессе 8081 → запущенный dev-сервер отдаёт **старый** сценарий, пока его не перезапустить. Правки `arena.js`/`arena-styles.css` — `no-store`, оживают без рестарта. После правок сценария/движка — рестарт 8081.

---

## 10. Тесты

- `tests/test_onboarding_tutorial.py` — 28 unit: happy-path (сообщения/борд/hero HP по шагам), wrong-card (Альфонс на step7, per-step `wrong_card`), legal_actions, choose_target (`also_allow_minion_targets` → hero+Стив записи; неверный тап → 409 без смены состояния → hero-атака → шаг вперёд), метки/`TUTORIAL_FINAL_STEP`.
- `tests/e2e/onboarding_tutorial_arena.js` — 47 ассертов (Playwright, без `visual_input`).
- Проверки на момент слива в `main`: 28/28 unit PASS; 47/47 e2e PASS; `precompile_webapp_index.py --check` exit 0 (онбординг не трогает JSX → бандл не рассинхронен).