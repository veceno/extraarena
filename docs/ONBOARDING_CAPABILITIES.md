# WebApp — возможности для онбординга и хаба новичка

---

## 1. Анимированные попапы/оверлеи поверх интерфейса

**УЖЕ РЕАЛИЗОВАНО.** Есть 4 готовых паттерна:

### A. React full-screen overlay + центрованная карточка (рекомендуемый)

Пример из `ExtraArenaInfoModal` — `webapp/index.html:14039-14120`:

```jsx
<div style={{
  position:'absolute', inset:0,
  background:'rgba(5,3,14,0.72)', backdropFilter:'blur(12px)',
  zIndex:50, display:'flex', alignItems:'center', justifyContent:'center',
  padding:'18px'
}} onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    style={{
      width:'100%', maxWidth:'392px', maxHeight:'84vh',
      borderRadius:'22px',
      border:`1.5px solid ${MODE_COLOR}44`,
      animation:'fadeIn 0.18s'
    }}>
    {/* header + body */}
  </div>
</div>
```

- Клик на задний фон — закрывает
- `stopPropagation` — предотвращает закрытие при клике на контент
- CSS-анимация `fadeIn` на вход

### B. Bottom sheet (React)

`webapp/index.html:14155` — `GameModeSheet`. Слайд снизу с drag-ручкой, `animation:'slideUp 0.28s'`. Другие bottom-sheet паттерны на строках 6992, 7167, 7612, 7637, 7723, 7759, 9608, 12162.

### C. Vanilla JS модальные окна

`webapp/main.js:1901-1902` (а также 2006-2007, 2190-2191) — `document.createElement("div")` с классом `.modal-overlay`. Используется для админ-панелей.

### D. Toast-уведомления

`webapp/main.js:5242` — `showNotification(message, type = "info")`. Фиксировано сверху по центру, авто-скрытие через 5 секунд.

**CSS-анимации (20+), готовые к переиспользованию** — `webapp/index.html:104-128`:
`fadeIn`, `fadeUp`, `slideUp`, `popIn`, `pulse`, `pulse2`, `glowPulse`, `ringExpand`, `runeFloat`, `slideDown`, `opponentReveal`, `timerPulse`, `onbCoachFloat` и др.

**Итог**: попап с приветствием от Мидории при первом входе делается на существующих паттернах. Нужен только контент и триггер на `welcome_shown == false`.

---

## 2. Отслеживание «первого входа» игрока

**ЧАСТИЧНО РЕАЛИЗОВАНО.**

### Что есть

| Уровень | Механизм | Файл |
|---------|----------|------|
| **БД** | `user_settings.welcome_shown BOOLEAN DEFAULT false` | `database.py:2844` |
| **БД** | `get_welcome_status(user_id)` — возвращает `{should_show: true}`, если пользователь новый или `welcome_shown` = false | `database.py:21968` |
| **БД** | `mark_welcome_shown(user_id)` — upsert с `welcome_shown = true` | `database.py:21990` |
| **БД** | `get_onboarding_state(user_id)` / `set_onboarding_state(...)` — новая таблица `user_onboarding` со статусами `not_started`/`welcome`/`tutorial_battle`/`menu_tour`/`completed` | `database.py:21728` / `database.py:21792` |
| **БД** | `mark_newbie_path_task` / `claim_newbie_path_task` — прогресс по Newbie Path (см. `onboarding_tutorial.NEWBIE_PATH_TASKS`) | `database.py:21861` / `database.py:21909` |
| **БД** | `track_onboarding_event(user_id, step, completed, time_spent_seconds, metadata)` — таблица `onboarding_events` | `database.py:19290` |
| **API** | `GET /api/onboarding/status` — полный онбординг-стейт + newbie-path таски | `server.py:19901` |
| **API** | `POST /api/onboarding/welcome/complete` | `server.py:19906` |
| **API** | `POST /api/onboarding/tutorial/start` / `/api/onboarding/tutorial/action` | `server.py:19923` / `server.py:19954` |
| **API** | `POST /api/onboarding/menu-tour/step` и `/api/onboarding/complete` | `server.py:19980` / `server.py:20017` |
| **API** | `GET/POST /api/onboarding/newbie-path` (claim), `POST /api/onboarding/newbie-path/progress` | `server.py:20043` / `server.py:20140` |
| **API** | `GET /api/welcome/status` — легаси проверка, нужно ли показывать приветствие | `server.py:19785` |
| **API** | `POST /api/welcome/mark-shown` — пометить как показанное | `server.py:19817` |
| **API** | `POST /api/welcome/create-user` — создание пользователя после онбординга | `server.py:20169` |
| **Frontend** | `checkAndShowWelcome(authData)` — welcome modal | `main.js:3391` |
| **Frontend** | `MascotCoach` — React-компонент с Мидорией, репликами и CTA (используется онбордингом и меню-туром) | `index.html:15382` |

### Что отсутствует

- Флаг `is_new_player` на таблице `users`
- Флаги гранулярного прохождения (`tutorial_completed`, `first_battle_done`, `first_case_opened`, `first_deck_built`)
- `last_login` timestamp
- **Нет localStorage-флагов** — веб-приложение не использует `localStorage` вообще, вся логика на серверных флагах

**Итог**: серверная таблица `user_onboarding` уже даёт онбординг-стейт, статусы и Newbie Path. Для хаба новичка, если потребуется ещё больше шагов, расширяется `NEWBIE_PATH_TASKS` в `onboarding_tutorial.py` + таблица `onboarding_events`.

---

## 3. Система уведомлений

**УЖЕ РЕАЛИЗОВАНО.** Два канала доставки:

### A. Telegram-уведомления через бота

`bot/handlers.py` — функции оповещения бота (например, `_check_and_notify_dice_ready`, `get_dice_status` ожидаются регресс-тестом `tests/test_release_readiness_static.py:47`). Используйте их как шаблон для любых onboarding-уведомлений.

### B. Внутренние UI-уведомления

| Тип | Функция | Файл | Поведение |
|-----|---------|------|-----------|
| Toast | `showNotification(msg, type = "info")` | `main.js:5242` | Сверху-центр, авто-скрытие 5 сек, 4 типа (success/error/warning/info) |
| Alert | `showGameAlert(msg, icon)` | `main.js:5277` | Модальное окно с кнопкой OK |
| Confirm | `showGameConfirm(msg, icon)` | `main.js:5324` | Модальное окно с Да/Нет |

### C. Настройки уведомлений на пользователя

15+ флагов в `user_settings` (`database.py:2819-2843`): `notif_cases`, `notif_daily_rewards`, `notif_game_invites`, `notif_friend_requests`, `notif_events`, `notif_news`, `notif_generator`, `notif_shop`, `notif_reminders`, `notif_squad_member_role`, `notif_squad_new_member`, `notif_squad_disbanded`, `notif_squad_boost`, `notif_squad_weekly_tokens`, `notif_extra_arena_modifiers` + поле `notification_delivery_mode TEXT DEFAULT 'app_then_telegram'`. UI-переключатели в `webapp/index.html:4118-4220` (`notifs` state) и связанных JSX-блоках секции настроек.

**Итог**: уведомления о заданиях хаба можно слать через оба канала — и внутри приложения, и через Telegram-бота.

---

## 4. Пошаговые подсказки с подсветкой UI (tour/spotlight)

**НУЖНО ДЕЛАТЬ С НУЛЯ.**

### Что есть сейчас

3-шаговый welcome modal (`main.js:3391+` + `index.html` секция `#welcome-modal`) — это **статический текст** с переключением панелей:

```
Шаг 1: Карта (показ стартовой карты)
Шаг 2: Обучение (текстовая инструкция)
Шаг 3: Подарок (принятие награды)
```

### Что отсутствует

- Нет **spotlight-подсветки** конкретных элементов интерфейса
- Нет **тултипов-стрелок**, указывающих на кнопки
- Нет **интерактивного guided tour** (последовательность «нажми сюда → теперь сюда»)
- Нет системы координат/позиционирования тултипов относительно DOM-элементов
- Нет блокировки взаимодействия с нефокусными элементами во время тура

**Итог**: spotlight-систему нужно разрабатывать как отдельный компонент. Она может быть реализована как React-компонент с позиционированием через `getBoundingClientRect()` подсвечиваемого элемента, overlay с вырезанным «окном», и стрелкой-указателем.

---

## 5. Хаб новичка — раздел/экран

**НУЖНО ДЕЛАТЬ С НУЛЯ.**

### Что есть сейчас

Приложение управляется React-состояниями (`webapp/index.html:16723-16724`, `hasOverlay` объединяет их; декларации см. также рядом в `setShow*`/useState). Реальные имена стейтов:

```js
squadsOpen, communityOpen, showAI, showGenerator, showDailyLogin, showMenu,
showSettings, showBattlePick, showGameMode, showMail, showNews, showCaseOpen,
showInfo, showGloryPath, showLeagueInfo, showBattlePass, showPreBattle,
showSupport, showInvite, showBattles, showFriends, showProfile, showExtraID,
onboardingBlocking, onboardingTourActive, newbiePathOpen, seasonResetNotice, pendingCaseOpen
```

`onboardingTourActive`, `onboardingBlocking` и `newbiePathOpen` — это уже существующие стейты для онбординга и Newbie Path; отдельных `showTutorial`/`showQuests`/`showNewbieHub` пока нет, но они реализуемы добавлением useState + интеграцией в `hasOverlay`/`backHandler`.

Существующие разделы:
- **Arena** — бой
- **Collection** — карты
- **Squads** — редактор колод
- **Shop** — магазин
- **Community** — комьюнити

В `webapp/extraid-mockup.html` есть отдельный мокап с вкладкой «Старт» для ExtraID-регистрации, но он не интегрирован в основное приложение.

**Итог**: хаб новичка — это новый раздел/экран, который нужно добавить в систему навигации приложения. Потребуется: новое React-состояние, новый компонент, добавление в таб-бар или в виде отдельного overlay.

---

## 6. Отслеживание действий игрока на сервере

**НУЖНО ДЕЛАТЬ С НУЛЯ.**

### Что можно вычислить из существующих данных

| Событие | Как определить по текущим таблицам |
|---------|-----------------------------------|
| Открыл первый кейс | Появление первой записи в `user_cases` (после открытия) |
| Собрал первую колоду | Первая запись в `deck_presets` |
| Выиграл первый PvP-бой | Первая запись в `battle_results` с `winner_id = user_id` |
| Получил первую карту | Первая запись в `user_cards` |

### Что отсутствует

- **Нет таблицы событий** (`player_events`, `action_log` и т.п.)
- **Нет системы заданий/квестов** — ни `daily_quests`, ни `achievements`, ни `player_milestones`
- **Нет серверной логики детекта milestone** — сервер не триггерит события «игрок впервые X»
- **Нет эндпоинта для получения прогресса** по заданиям

**Итог**: для хаба новичка потребуется либо новая таблица `user_onboarding_progress` с флагами, либо система заданий с таблицей `user_quests`. Самый простой путь — добавить boolean-флаги в `user_settings` для каждого шага онбординга.

---

## 7. Передача данных между ботом и WebApp — Telegram InitData

**УЖЕ РЕАЛИЗОВАНО.**

### Полный пайплайн

| Шаг | Где | Что |
|-----|-----|-----|
| **1. Получение** | `webapp/index.html:2472-2473` | `tg?.initData` берётся из Telegram WebApp API |
| **2. Передача** | `webapp/index.html:2806` | `add('auth', tg?.initData, 'telegram')` — кандидаты на `_auth` собираются через `resolveUiAuth()` и прокидываются в API-запросы |
| **3. Верификация** | `web/server.py:3860` | `_verify_init_data(init_data, bot_token)` — HMAC-проверка подписи |
| **4. Извлечение** | `web/server.py:3892` | `_extract_user_id_from_init_data(data_dict)` — парсинг `user.id` |
| **5. Fallback** | `web/server.py:4115-4135` (`_telegram_init_data_for_request`) | Поддерживает `Authorization: Bearer` (JWT), `initData` и dev-фолбэк через `_auth`; старого пути «_auth = число → user_id» уже нет, только initData/JWT |

### Дополнительные данные из initData

`tg.initDataUnsafe.user` содержит:
- `id` — Telegram user ID
- `first_name`, `last_name`, `username`
- `language_code`

**Итог**: механика идентификации пользователя полностью готова. Хаб новичка может использовать тот же механизм аутентификации без изменений.

---

## 8. Анимации Мидории (Lottie/GIF)

**НЕ РЕАЛИЗОВАНО ДЛЯ ПЕРСОНАЖЕЙ.**

### Что есть

- **Только CSS `@keyframes`** — 20+ анимаций (fadeIn, pulse, popIn, slideUp и т.д.)
- **Статическое изображение**: `DesignAssets/MidoriaFixingRobot.png` — не используется в UI
- **Нет Lottie**: zero references to "lottie", "Lottie", "bodymovin"
- **Нет анимированных GIF персонажей** (единственный GIF — прозрачный 1×1 пиксель для drag-and-drop)
- **Нет спрайтовой системы**

### Что можно сделать без Lottie

CSS-анимации могут оживить статичное изображение Мидории:
- `pulse` — пульсирующее появление
- `popIn` — выскакивание с масштабированием
- `float` — покачивание (нужно добавить keyframe)
- `bounce` — подпрыгивание

### Что нужно для Lottie

1. Добавить библиотеку `lottie-web` или `@dotlottie/player-component` в `<script>` теги
2. Подготовить Lottie JSON-анимацию Мидории (дизайнерская задача)
3. Использовать `<lottie-player>` или `lottie.loadAnimation()` в компоненте

Альтернатива без библиотек: **CSS sprite-sheet анимация** — кадры Мидории в одном PNG, `@keyframes` с `background-position`.

**Итог**: анимации персонажа возможны двумя путями — Lottie (требует библиотеку и дизайн) или CSS-анимации поверх статичных PNG (можно сделать прямо сейчас).

---

## 9. Мидория как маскот проекта

### Текущее использование

| Контекст | Где | Роль |
|----------|-----|------|
| **Игровая карта** | `cards.json` (id=26), `core/effects.py:838-950+` (`@register_effect("cast_random_spell")`) | Воин с `cast_random_spell` (Texas Smash, Recovery, Blackwhip) |
| **AI-модель** | `ai/bot_brain.py` (профили `train_v2_classic_v1` — `obs_dim=1456`, `action_feature_dim=171`, см. `bot_brain.py:13`) | Нейросеть для ботов; конкретные размерности нужно сверять с активным профилем |
| **Боевой лог** | `webapp/arena-styles.css:1545` | CSS-класс `.log-row.log-midoriya` для подсветки действий |
| **UI (выбор чата)** | `webapp/index.html:8171` | Опция `{id:'midoria', name:'Мидория', tag:'Free', ...}` |
| **Онбординг-туториал** | `onboarding_tutorial.py` (`ONBOARDING_MIDORIA_ASSET = "/DesignAssets/MidoriaOnboardingGuide.png"`, реплики в `TUTORIAL_STEPS`) | Midoria ведёт обязательный учебный бой с репликами и feedback'ом |
| **MascotCoach** | `webapp/index.html:15382` (`const MascotCoach`) | React-компонент с Мидорией, title/body, CTA-кнопками; используется в онбординге и меню-туре |
| **Дизайн-ассет** | `DesignAssets/MidoriaFixingRobot.png`, `DesignAssets/MidoriaOnboardingGuide.png` | Статичные картинки (для онбординга уже используется вторая) |

### Что отсутствует для роли onboarding-маскота

- Нет полноценной **диалоговой системы** для общения с игроком от лица Мидории (есть только жёстко прописанные реплики в `TUTORIAL_STEPS` и сообщения внутри `MascotCoach`)

**Итог**: Мидория уже работает как маскот: учебный бой (`onboarding_tutorial.py`) использует её реплики + ассет `MidoriaOnboardingGuide.png`, а в `MascotCoach` есть React-обёртка для показа Мидории с текстом и CTA. Для полноценного хаба новичка остаётся добавить диалоговую систему и анимированные ассеты (Lottie/CSS-sprites).

---

## 10. Паттерны модальных окон — готовые примеры

### Паттерн 1: Полноэкранный overlay + карточка по центру (React)

```jsx
// webapp/index.html:14051-14054
<div style={{
  position:'absolute', inset:0,
  background:'rgba(5,3,14,0.72)', backdropFilter:'blur(12px)',
  zIndex:50, display:'flex', alignItems:'center', justifyContent:'center',
  padding:'18px'
}} onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    style={{ width:'100%', maxWidth:'392px', animation:'fadeIn 0.18s' }}>
    {/* Контент */}
  </div>
</div>
```

### Паттерн 2: Bottom Sheet (React)

```jsx
// webapp/index.html:6992 (один из bottom-sheet вариантов)
<div onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    className="safe-pad-bottom"
    style={{ borderRadius:'24px 24px 0 0', animation:'slideUp 0.28s cubic-bezier(0.32,0.72,0,1)' }}>
    <div style={{width:'36px',height:'4px'}}/> {/* drag handle */}
    {/* Контент */}
  </div>
</div>
```

### Паттерн 3: Динамический vanilla JS modal

```js
// webapp/main.js:1901
const modal = document.createElement("div");
modal.className = "modal-overlay";
modal.innerHTML = `<div class="modal-content">...</div>`;
document.body.appendChild(modal);
```

### Паттерн 4: Toast-уведомление

```js
// webapp/main.js:5242
showNotification("Кейс открыт!", "success");
// → позиция fixed сверху-центр, авто-скрытие 5 сек
```

---

## Сводная таблица готовности

| # | Возможность | Статус | Что нужно |
|---|------------|--------|-----------|
| 1 | Анимированные попапы/оверлеи | ✅ Готово | Только контент и триггер |
| 2 | Первый вход — онбординг-стейт в БД | ✅ Готово (частично) | `user_onboarding` со статусами + `welcome_shown`; расширяется через `NEWBIE_PATH_TASKS` |
| 3 | Система уведомлений | ✅ Готово | Telegram-бот + in-app toast |
| 4 | Пошаговый tour / spotlight | ⚠️ Частично | Есть tutorial battle в `onboarding_tutorial.py` + `MascotCoach`; **spotlight**-подсветка UI пока отсутствует |
| 5 | Хаб новичка (раздел) | ⚠️ Частично | Newbie Path + `newbiePathOpen` overlay готовы; нужен полноценный хаб-экран |
| 6 | Серверный трекинг действий | ✅ Готово | `track_onboarding_event`, таблица `onboarding_events` |
| 7 | Telegram InitData | ✅ Готово | Без изменений |
| 8 | Lottie/GIF анимации | ❌ С нуля | Библиотека + дизайн; или CSS-анимации PNG как fallback |
| 9 | Мидория как маскот | ✅ Готово (учебный бой + MascotCoach) | Не хватает анимированного персонажа и диалоговой системы |
| 10 | Паттерны модальных окон | ✅ Готово | 4 готовых паттерна |

---

## Рекомендуемая архитектура онбординга

На основе анализа существующего кода, минимальный путь реализации хаба новичка:

### Backend (Python)

| Компонент | Что делаем |
|-----------|-----------|
| `user_settings.onboarding_step` | Текущий шаг онбординга (1..N, 0 = завершён) |
| `GET /api/onboarding/status` | Возвращает текущий шаг, данные для хаба |
| `POST /api/onboarding/progress` | Сохраняет выполнение шага, выдаёт награду |
| `infrastructure/onboarding.py` | Логика наград за шаги (карты, монеты, гемы) |

### Frontend (React)

| Компонент | Что делаем |
|-----------|-----------|
| `OnboardingHub` | Основной экран хаба со списком заданий |
| `SpotlightOverlay` | Подсветка элемента интерфейса со стрелкой |
| `MidoriaGuide` | Мидория с облачком диалога (PNG + CSS анимация) |
| `onboarding` (состояние) | Добавить в список `show*` состояний App |

### Персонаж Мидории

| Ассет | Формат |
|-------|--------|
| Мидория (обычная) | PNG 150×200px |
| Мидория (радость) | PNG 150×200px |
| Анимация появления | CSS `@keyframes popIn` (уже есть) |
| Анимация покачивания | CSS `@keyframes float` (добавить) |
| Облачко диалога | CSS border-radius + `::after` треугольник |

---

## Audit (2026-06-25)

Что проверено: учебный onboarding-бой (`onboarding_tutorial.py` + `tests/test_onboarding_tutorial.py`), API-роуты онбординга в `web/server.py`, БД-функции в `infrastructure/database.py`, фронтенд в `webapp/index.html`/`webapp/main.js`/`webapp/arena*.css`, граф `graphify-out/2026-06-25/GRAPH_REPORT.md`, проектный дизайн-документ `docs/2026-05-25-onboarding-design.md`.

Исправлено:
- `docs/ONBOARDING_CAPABILITIES.md:11` — `webapp/index.html:2686-2731` → `14039` для `ExtraArenaInfoModal`; корректировка `maxWidth` 380→392 и background `0.88→0.72` под актуальный JSX.
- `docs/ONBOARDING_CAPABILITIES.md:36` — `GameModeSheet` `:2799-2804` → `:14155`, плюс список других bottom-sheet блоков.
- `docs/ONBOARDING_CAPABILITIES.md:40` — vanilla-modal `webapp/main.js:1618` → `:1901` (+ список альтернатив `:2006`, `:2190`).
- `docs/ONBOARDING_CAPABILITIES.md:44` — `showNotification` `webapp/main.js:5036` → `:5242`.
- `docs/ONBOARDING_CAPABILITIES.md:46-47` — CSS-анимации `webapp/index.html:55-77` → `:104-128`; список пополнен реальными keyframes (`pulse2`, `runeFloat`, `slideDown`, `opponentReveal`, `timerPulse`, `onbCoachFloat`).
- `docs/ONBOARDING_CAPABILITIES.md:62-66` — три строки БД (`welcome_shown`, `get_welcome_status`, `mark_welcome_shown`) теперь указывают на `database.py:2844 / 21968 / 21990` (фактические позиции) и дополнены `get_onboarding_state`/`set_onboarding_state`, `mark_newbie_path_task`, `claim_newbie_path_task`, `track_onboarding_event`.
- `docs/ONBOARDING_CAPABILITIES.md:65-69` — API-роуты: реальные `web/server.py:19785 / 19817 / 20169`, плюс расширенный набор (`/api/onboarding/{status,welcome/complete,tutorial/start,tutorial/action,menu-tour/step,complete,newbie-path,newbie-path/progress}`).
- `docs/ONBOARDING_CAPABILITIES.md:67` — `checkAndShowWelcome` `main.js:3219` → `:3391`, плюс упоминание `MascotCoach` (`index.html:15382`).
- `docs/ONBOARDING_CAPABILITIES.md:76` — итог §2 переписан под существующую `user_onboarding`.
- `docs/ONBOARDING_CAPABILITIES.md:86` — `bot/handlers.py:354 _check_and_notify_dice_ready` → перефразировано, поскольку функции в исходниках нет (она заявлена только в `tests/test_release_readiness_static.py:47`).
- `docs/ONBOARDING_CAPABILITIES.md:92-94` — `showGameAlert` `5071→5277`, `showGameConfirm` `5118→5324`.
- `docs/ONBOARDING_CAPABILITIES.md:98` — секция про 6 флагов нотификаций заменена на 15+ флагов с реальным диапазоном `database.py:2819-2843` и `webapp/index.html:4118-4220`.
- `docs/ONBOARDING_CAPABILITIES.md:111` — welcome modal `main.js:3246-3390` → `:3391+`.
- `docs/ONBOARDING_CAPABILITIES.md:138-145` — список `show*`-стейтов переписан на фактический (включая `onboardingTourActive`, `onboardingBlocking`, `newbiePathOpen`); ссылки на `:3248-3261` заменены на `:16723-16724`.
- `docs/ONBOARDING_CAPABILITIES.md:192-195` — Telegram initData: реальные строки `webapp/index.html:2472-2473 / 2806`, `web/server.py:3860 / 3892 / 4115-4135`; прежний «фолбэк _auth=число» уже не действует.
- `docs/ONBOARDING_CAPABILITIES.md:247-265` — Мидория: `core/effects.py:714-802` → `:838-950+`, AI-параметры `997/200` → `obs_dim=1456, action_feature_dim=171` (`ai/bot_brain.py:13`), midoria-чат `:1744` → `:8171`, log-midoriya → `arena-styles.css:1545`. Добавлены `onboarding_tutorial.py` и `MascotCoach` (`index.html:15382`) как уже существующие маскот-механизмы; прежнее «нет кода-гида» удалено.
- `docs/ONBOARDING_CAPABILITIES.md:277-313` — патерны модалок обновлены по тем же корректным ссылкам.
- `docs/ONBOARDING_CAPABILITIES.md:330-340` — сводная таблица: статусы §2/§4/§5/§6/§9 переоценены с «с нуля/частично» на «готово/частично» в соответствии с реальным состоянием (`user_onboarding`, `onboarding_events`, Newbie Path, `MascotCoach`).

Не проверено / нельзя верифицировать без запуска:
- Реальная длина turn-timer 99 в tutorial battle (`onboarding_tutorial.py:71`) — подтверждена в `tests/test_onboarding_tutorial.py:344`, но в живом рантайме не запускалась.
- Полный список активных AI-профилей `train_v2_classic_v1` (есть только дефолт `:1456/171` в `bot_brain.py:13`).
- Дизайн-ассет `DesignAssets/MidoriaOnboardingGuide.png` физически присутствует (`find` подтверждает), но не открывался в графическом редакторе.
