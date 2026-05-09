# WebApp — возможности для онбординга и хаба новичка

---

## 1. Анимированные попапы/оверлеи поверх интерфейса

**УЖЕ РЕАЛИЗОВАНО.** Есть 4 готовых паттерна:

### A. React full-screen overlay + центрованная карточка (рекомендуемый)

Пример из `ExtraArenaInfoModal` — `webapp/index.html:2686-2731`:

```jsx
<div style={{
  position:'absolute', inset:0,
  background:'rgba(5,3,14,0.88)', backdropFilter:'blur(12px)',
  zIndex:50, display:'flex', alignItems:'center', justifyContent:'center'
}} onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    style={{
      maxWidth:'380px', borderRadius:'22px',
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

`webapp/index.html:2799-2804` — `GameModeSheet`. Слайд снизу с drag-ручкой, `animation:'slideUp 0.28s'`.

### C. Vanilla JS модальные окна

`webapp/main.js:1618` — `document.createElement("div")` с классом `.modal-overlay`. Используется для админ-панелей.

### D. Toast-уведомления

`webapp/main.js:5036` — `showNotification(message, type)`. Фиксировано сверху по центру, авто-скрытие через 5 секунд.

**CSS-анимации (20+), готовые к переиспользованию** — `webapp/index.html:55-77`:
`fadeIn`, `fadeUp`, `slideUp`, `popIn`, `pulse`, `glowPulse`, `ringExpand`, `countBounce` и др.

**Итог**: попап с приветствием от Мидории при первом входе делается на существующих паттернах. Нужен только контент и триггер на `welcome_shown == false`.

---

## 2. Отслеживание «первого входа» игрока

**ЧАСТИЧНО РЕАЛИЗОВАНО.**

### Что есть

| Уровень | Механизм | Файл |
|---------|----------|------|
| **БД** | `user_settings.welcome_shown BOOLEAN DEFAULT false` | `database.py:1173` |
| **БД** | `get_welcome_status(user_id)` — возвращает `{should_show: true}`, если пользователь новый или `welcome_shown` = false | `database.py:3423` |
| **БД** | `mark_welcome_shown(user_id)` — upsert с `welcome_shown = true` | `database.py:3445` |
| **API** | `GET /api/welcome/status` — проверка, нужно ли показывать приветствие | `server.py:7026` |
| **API** | `POST /api/welcome/mark-shown` — пометить как показанное | `server.py:7072` |
| **API** | `POST /api/welcome/create-user` — создание пользователя после онбординга | `server.py:7103` |
| **Frontend** | `checkAndShowWelcome(authData)` — 3-шаговый welcome modal | `main.js:3219` |

### Что отсутствует

- Флаг `is_new_player` на таблице `users`
- Флаги гранулярного прохождения (`tutorial_completed`, `first_battle_done`, `first_case_opened`, `first_deck_built`)
- `last_login` timestamp
- **Нет localStorage-флагов** — веб-приложение не использует `localStorage` вообще, вся логика на серверном флаге

**Итог**: для онбординга хаба новичка потребуется расширить таблицу `user_settings` или создать новую таблицу `user_onboarding` с флагами по шагам.

---

## 3. Система уведомлений

**УЖЕ РЕАЛИЗОВАНО.** Два канала доставки:

### A. Telegram-уведомления через бота

`bot/handlers.py:354` — `_check_and_notify_dice_ready()`. Пока только для кубика, но паттерн готов для любых уведомлений.

### B. Внутренние UI-уведомления

| Тип | Функция | Файл | Поведение |
|-----|---------|------|-----------|
| Toast | `showNotification(msg, type)` | `main.js:5036` | Сверху-центр, авто-скрытие 5 сек, 4 типа (success/error/warning/info) |
| Alert | `showGameAlert(msg, icon)` | `main.js:5071` | Модальное окно с кнопкой OK |
| Confirm | `showGameConfirm(msg)` | `main.js:5118` | Модальное окно с Да/Нет |

### C. Настройки уведомлений на пользователя

6 флагов в `user_settings`: `notif_cases`, `notif_daily_rewards`, `notif_game_invites`, `notif_friend_requests`, `notif_events`, `notif_news`, `notif_dice` — `database.py:1138-1144`. UI-переключатели в `webapp/index.html:750-757`.

**Итог**: уведомления о заданиях хаба можно слать через оба канала — и внутри приложения, и через Telegram-бота.

---

## 4. Пошаговые подсказки с подсветкой UI (tour/spotlight)

**НУЖНО ДЕЛАТЬ С НУЛЯ.**

### Что есть сейчас

3-шаговый welcome modal (`main.js:3246-3390`) — это **статический текст** с переключением панелей:

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

Приложение управляется React-состояниями (`webapp/index.html:3248-3261`):

```js
showAI, showMenu, showSettings, showBattlePick, showGameMode,
showMail, showCaseOpen, showInfo, showGloryPath, showLeagueInfo,
showPreBattle, showSupport
```

Нет состояний `showTutorial`, `showQuests`, `showNewbieHub`.

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
| **1. Получение** | `webapp/index.html:450` | `tg?.initData` берётся из Telegram WebApp API |
| **2. Передача** | `webapp/index.html:474` | `params.set('_auth', initData)` — query-параметр во все API-запросы |
| **3. Верификация** | `web/server.py:848` | `_verify_init_data(init_data, bot_token)` — HMAC-проверка подписи |
| **4. Извлечение** | `web/server.py:880` | `_extract_user_id_from_init_data(data_dict)` — парсинг `user.id` |
| **5. Fallback** | `web/server.py:1534-1549` | Если `_auth` — число (старый формат), используется как `user_id` напрямую |

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
| **Игровая карта** | `cards.json` (id=26), `core/effects.py:714-802` | Воин с `cast_random_spell` |
| **AI-модель** | `ai/bot_brain.py` (997 признаков, 200 действий) | Нейросеть «Мидория v3» для ботов |
| **Боевой лог** | `webapp/arena.js:535` | CSS-класс `.log-midoriya` для подсветки действий |
| **UI (выбор чата)** | `webapp/index.html:1744` | Опция `{id:'midoria', name:'Мидория', tag:'Free'}` |
| **Дизайн-ассет** | `DesignAssets/MidoriaFixingRobot.png` | Статичная картинка (потенциально для экрана загрузки) |

### Что отсутствует для роли onboarding-маскота

- Нет кода, использующего Мидорию как **гида/помощника**
- Нет **диалоговой системы** для общения с игроком от лица Мидории
- Нет **анимированного персонажа** на экране
- Нет **реплик** или character personality в UI

**Итог**: Мидория полностью реализована как игровая карта, но не как onboarding-маскот. Чтобы сделать её лицом хаба новичка, потребуется:
1. Анимированное изображение (PNG с CSS-анимацией минимум)
2. Компонент-обёртка для показа Мидории с текстовыми облачками
3. Тексты реплик для каждого шага онбординга

---

## 10. Паттерны модальных окон — готовые примеры

### Паттерн 1: Полноэкранный overlay + карточка по центру (React)

```jsx
// webapp/index.html:2687
<div style={{
  position:'absolute', inset:0,
  background:'rgba(5,3,14,0.88)', backdropFilter:'blur(12px)',
  zIndex:50, display:'flex', alignItems:'center', justifyContent:'center'
}} onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    style={{ animation:'fadeIn 0.18s' }}>
    {/* Контент */}
  </div>
</div>
```

### Паттерн 2: Bottom Sheet (React)

```jsx
// webapp/index.html:2799
<div onClick={onClose}>
  <div onClick={e=>e.stopPropagation()}
    style={{ borderRadius:'24px 24px 0 0', animation:'slideUp 0.28s' }}>
    <div style={{width:'36px',height:'4px'}}/> {/* drag handle */}
    {/* Контент */}
  </div>
</div>
```

### Паттерн 3: Динамический vanilla JS modal

```js
// webapp/main.js:1618
const modal = document.createElement("div");
modal.className = "modal-overlay";
modal.innerHTML = `<div class="modal-content">...</div>`;
document.body.appendChild(modal);
```

### Паттерн 4: Toast-уведомление

```js
// webapp/main.js:5036
showNotification("Кейс открыт!", "success");
// → позиция fixed сверху-центр, авто-скрытие 5 сек
```

---

## Сводная таблица готовности

| # | Возможность | Статус | Что нужно |
|---|------------|--------|-----------|
| 1 | Анимированные попапы/оверлеи | ✅ Готово | Только контент и триггер |
| 2 | Первый вход — флаг в БД | ⚠️ Частично | `welcome_shown` есть; нужны гранулярные флаги для шагов |
| 3 | Система уведомлений | ✅ Готово | Telegram-бот + in-app toast |
| 4 | Пошаговый tour / spotlight | ❌ С нуля | Spotlight-компонент с позиционированием |
| 5 | Хаб новичка (раздел) | ❌ С нуля | Новый компонент + навигация |
| 6 | Серверный трекинг действий | ❌ С нуля | Таблица событий или флаги в user_settings |
| 7 | Telegram InitData | ✅ Готово | Без изменений |
| 8 | Lottie/GIF анимации | ❌ С нуля | Библиотека + дизайн; или CSS-анимации PNG как fallback |
| 9 | Мидория как маскот | ⚠️ Как карта | Компонент диалогов, реплики, анимированное изображение |
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
