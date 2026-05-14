const tg = window.Telegram?.WebApp;

let currentProfile = null;
let currentSettings = null;
let lastBattleSelection = {
  mode: null,
  matchType: 'ranked',
  deck: null,
  difficulty: null
};

// Функция проверки готовности к бою
function checkBattleReady() {
  const { mode, matchType, deck, difficulty } = lastBattleSelection;
  console.log(`[DEBUG] Ready Check -> Type: ${matchType}, Mode: ${mode}, Deck: ${deck}, Diff: ${difficulty}.`);
  
  const playBtn = document.getElementById("play-battle-btn");
  
  const hasMode = !!mode;
  const hasDeck = !!deck;
  const hasDiff = !!difficulty;

  let isReady = false;
  if (matchType === 'training') {
    isReady = hasMode && hasDeck && hasDiff;
  } else {
    isReady = hasMode && hasDeck;
  }

  if (playBtn) {
    if (isReady) {
      playBtn.disabled = false;
      playBtn.classList.add('active');
    } else {
      playBtn.disabled = true;
      playBtn.classList.remove('active');
    }
  }
}


const CASE_TIER_CONFIG = {
  1: {
    title: "Обычный кейс",
    icon: "📦",
    maxRarityKey: "superrare",
    coinsRange: [50, 150],
    cardsRange: [3, 4],
    accent: "#60a5fa",
  },
  2: {
    title: "Улучшенный кейс",
    icon: "💠",
    maxRarityKey: "epic",
    coinsRange: [150, 300],
    cardsRange: [4, 5],
    accent: "#34d399",
  },
  3: {
    title: "Элитный кейс",
    icon: "💎",
    maxRarityKey: "legendary",
    coinsRange: [300, 700],
    cardsRange: [5, 6],
    accent: "#c084fc",
  },
  4: {
    title: "Мифический кейс",
    icon: "🔥",
    maxRarityKey: "mythic",
    coinsRange: [700, 1500],
    cardsRange: [6, 7],
    accent: "#fb7185",
  },
  5: {
    title: "Божественный кейс",
    icon: "🌟",
    maxRarityKey: "divine",
    coinsRange: [1500, 3000],
    cardsRange: [7, 8],
    accent: "#fbbf24",
    allowsLimited: true,
    bonusNote: "Шанс на гемы и лимитки",
  },
};

const CASE_TAP_CHANCES = {
  1: 25,
  2: 20,
  3: 15,
  4: 10,
};

const CASE_REWARD_ICONS = {
  coins: "💰",
  card: "🃏",
  particles: "✨",
  gems: "💎",
  limited_shards: "🧩",
};

const CASE_TAP_LIMIT = 4;
const CASE_EFFECTS_REDUCED = true;

let userCasesData = [];
let matchmakingStatusTimer = null;

// Функции для работы с поиском матча
function setMatchmakingStatus(text) {
  // Функция может использоваться в будущем для обновления статуса
  console.log("Matchmaking status:", text);
}

function enableMatchmakingButton(isEnabled) {
  // Функция может использоваться в будущем для управления кнопками
  console.log("Matchmaking button enabled:", isEnabled);
}

function toggleSearchOverlay(isVisible, text = "Идет поиск противника...") {
  const overlay = document.getElementById("searchOverlay");
  if (!overlay) {
    console.error("Search overlay not found");
    return;
  }

  if (isVisible) {
    // Показываем оверлей
    overlay.style.display = "flex";

    // Обновляем текст в оверлее
    const titleEl = overlay.querySelector(".search-overlay-title");
    const textEl = overlay.querySelector("[data-role='search-overlay-text']");

    if (titleEl) {
      titleEl.textContent = text;
    }

    if (textEl && text !== "Идет поиск противника...") {
      textEl.textContent = text;
    } else if (textEl) {
      textEl.textContent = "Подбираем соперника, это займет пару секунд";
    }

    console.log("Search overlay shown with text:", text);
      } else {
    // Скрываем оверлей
    overlay.style.display = "none";
    console.log("Search overlay hidden");
  }
}

async function startMatchmaking(selectedDeckId = null, mode = 'classic', matchType = 'ranked', difficulty = null) {
  // Включаем полноэкранный оверлей сразу, чтобы игрок видел старт поиска
  toggleSearchOverlay(true, (matchType === 'training') ? "Подготовка к тренировке..." : "Идет поиск противника...");

  const authData = resolveUserId();
  const userId = typeof authData === "number" ? authData : currentProfile?.user_id;
  const trophies = currentProfile?.trophies ?? 0;
  const avgLevel = currentProfile?.avg_level ?? currentProfile?.level ?? 1;

  if (!userId) {
    setMatchmakingStatus("Статус: Не авторизован");
    toggleSearchOverlay(false);
    return;
  }

  // Показываем игроку, что поиск стартовал, ещё до сетевых запросов.
  setMatchmakingStatus((matchType === 'training') ? "Создание тренировочного матча..." : "Идет поиск противника...");
  enableMatchmakingButton(false);

  if (matchmakingStatusTimer) {
    clearInterval(matchmakingStatusTimer);
    matchmakingStatusTimer = null;
  }

  try {
    console.log("Sending matchmaking request...");
    const requestBody = {
      _auth: authData,
      trophies,
      user_avg_level: avgLevel,
      mode: mode,
      match_type: matchType,
      difficulty: difficulty
    };
    
    // Добавляем selected_deck_id, если указан
    if (selectedDeckId !== null && selectedDeckId !== undefined) {
      requestBody.selected_deck_id = selectedDeckId;
      console.log("Selected deck ID:", selectedDeckId);
    }
    
    const response = await fetch("/api/match/find", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Ошибка запуска матчмейкинга:", errorText);
      setMatchmakingStatus("Статус: Ошибка запроса");
      toggleSearchOverlay(false);
      enableMatchmakingButton(true);
      return;
    }

    const data = await response.json();
    console.log("Server Response:", data);
    console.log("Match data:", data);

    if (data.status === "found") {
      // Если матч найден сразу, уводим игрока в бой
      console.log("Match found immediately. Redirecting...");
      setMatchmakingStatus("Матч найден! Подготовка к бою...");

    // Передаем initData для аутентификации в арене
    const resolvedUserId = userId;

    if (!authData || typeof authData !== "string") {
      console.error("CRITICAL: Could not resolve initData for redirect!");
      alert("Error: Auth data missing. Cannot start battle.");
      return;
    }

    const targetUrl = `${window.location.origin}/arena?id=${encodeURIComponent(data.match_id)}&_auth=${encodeURIComponent(authData)}`;
    console.log(`Redirecting to Arena. Match: ${data.match_id}, url=${targetUrl}`);

      try {
        window.location.replace(targetUrl);
      } catch (err) {
        console.error("Error redirecting to arena:", err);
        // Fallback: попробуем обычный location.href
          window.location.href = targetUrl;
        }
    } else {
      // Если матч не найден сразу, продолжаем ждать
      console.log("Match not found immediately, waiting...");
    }
  } catch (error) {
    console.error("Ошибка при запуске матчмейкинга:", error);
    setMatchmakingStatus("Статус: Ошибка сети");
    toggleSearchOverlay(false);
    enableMatchmakingButton(true);
  }
}
let casesInitialized = false;
let casesLoading = false;
let casesGridBound = false;
let caseModalBound = false;
let caseOpeningState = null;
let casesNeedRefresh = false;
let caseHintTransitionTimeout = null;

// Кэш для визуала кейса (избегаем лишних перерисовок)
const caseVisualCache = {
  select: { tier: null, container: null },
  taps: { tier: null, container: null }
};

// Кэш для DOM элементов
const caseDOMCache = {
  caseDisplay: null,
  caseDisplayTapping: null,
  tierBadge: null,
  tierBadgeTapping: null,
  stageInfo: null,
  tapIndicators: null,
  progressMeter: null,
  tapHint: null
};

// Кэш для конфигурации тиров (избегаем повторных вызовов getCaseTierConfig)
const tierConfigCache = new Map();

// Кэш для HTML строк (stageInfo)
const stageInfoHTMLCache = new Map();

// Предзагруженные визуалы для всех тиров (храним HTML)
const preloadedVisuals = {
  select: new Map(), // tier -> HTML string
  taps: new Map()    // tier -> HTML string
};

// Флаг готовности кейсов
let casesPreloaded = false;
let casesPreloading = false;

// Дебаунсинг для обновлений
let caseUpdateTimeout = null;
let caseUpdateRAF = null;

function appendAuthParams(url, authData) {
  if (authData === undefined || authData === null) {
    return url;
  }
  const separator = url.includes("?") ? "&" : "?";
  if (typeof authData === "string") {
    return `${url}${separator}_auth=${encodeURIComponent(authData)}`;
  }
  if (typeof authData === "number") {
    console.warn("appendAuthParams: numeric userId unsupported; use initData string for auth.");
    return url;
  }
  return url;
}

// Инициализация Telegram WebApp
if (tg) {
  tg.ready();
  tg.expand();
  
  // Ждем события готовности WebApp
  tg.onEvent('viewportChanged', () => {
    console.log("WebApp viewport changed");
  });
  
  // Проверяем, что мы действительно в Telegram
  if (tg.platform === 'unknown' || !tg.initDataUnsafe) {
    console.warn("WebApp может быть открыт не через Telegram. platform:", tg.platform);
  }
}

// Display-only: resolves identity for rendering/client use.
// Auth for API calls should always be the string initData via _auth param.
function resolveUserId() {
  const urlParams = new URLSearchParams(window.location.search);
  
  // Пробуем получить initData разными способами
  let tgInitData = null;
  if (tg) {
    // Основной способ - initData (полная строка для серверной проверки)
    tgInitData = tg.initData;
    
    // Если initData пустой, но есть initDataUnsafe с user, пробуем получить user_id напрямую
    if (!tgInitData && tg.initDataUnsafe) {
      // Проверяем, что initDataUnsafe не пустой объект
      const hasUserData = tg.initDataUnsafe.user && tg.initDataUnsafe.user.id;
      if (hasUserData) {
        const userId = tg.initDataUnsafe.user.id;
        console.log("Используем initDataUnsafe.user.id:", userId);
        console.log("initDataUnsafe полный:", JSON.stringify(tg.initDataUnsafe));
        return userId;
      }
    }
    
    // Если initData есть, используем его
    if (tgInitData && tgInitData.trim() !== "") {
      console.log("Найден initData:", tgInitData.substring(0, 50) + "...");
      return tgInitData; // Return initData for server-side verification
    }
  }

  // Fallback for local testing without Telegram context
  const urlId = urlParams.get("user_id");
  if (urlId) {
    console.log("Используем user_id из URL:", urlId);
    return Number(urlId);
  }
  
  console.warn("Не удалось получить данные авторизации.");
  console.warn("tg:", !!tg);
  console.warn("tg.initData:", tg?.initData ? "есть" : "нет");
  console.warn("tg.initDataUnsafe:", tg?.initDataUnsafe ? JSON.stringify(tg.initDataUnsafe) : "нет");
  console.warn("tg.initDataUnsafe.user:", tg?.initDataUnsafe?.user ? JSON.stringify(tg.initDataUnsafe.user) : "нет");
  console.warn("tg.platform:", tg?.platform);
  console.warn("tg.version:", tg?.version);
  return null;
}

// Загрузка профиля
async function loadProfile(authData) {
  try {
    let url = "/api/profile";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    } else {
      throw new Error("Invalid authentication data");
    }

    const response = await fetch(url);
    if (!response.ok) {
      // Если 404, проверяем, нужно ли показать welcome
      if (response.status === 404) {
        const errorData = await response.json().catch(() => ({}));
        if (errorData.should_show_welcome || errorData.need_registration) {
          // Пользователь не найден - нужно показать welcome
          await checkAndShowWelcome(authData);
          // Возвращаем null, чтобы приложение продолжило работу
          return null;
        }
      }
      throw new Error(`Ошибка ${response.status}`);
    }
    const data = await response.json();
    currentProfile = data;
    renderProfile(data);
    
    // Показываем/скрываем админские товары
    const isAdmin = data.user_id === 6803854304;
    document.querySelectorAll(".admin-only-item").forEach(item => {
      item.style.display = isAdmin ? "block" : "none";
    });
    
    // Обновляем хранилище кейсов
    updateCasesStorage(data);
    
    // Проверяем непрочитанные письма для индикатора
    await updateMailNotificationBadge(authData);
    
    return data;
  } catch (error) {
    console.error("Ошибка загрузки профиля:", error);
    throw error;
  }
}

// Обновление хранилища кейсов
function updateCasesStorage(profileData) {
  const casesCountEl = document.getElementById("cases-count");
  const casesHintBadge = document.getElementById("cases-hint-badge");
  const casesOpenBtn = document.getElementById("cases-open-btn");
  const casesShopBtn = document.getElementById("cases-shop-btn");
  
  // Получаем количество кейсов из поля keys
  const casesCount = profileData.keys || profileData.cases_count || 0;
  
  if (casesCountEl) {
    casesCountEl.textContent = casesCount;
  }
  
  // Показываем/скрываем элементы в зависимости от количества кейсов
  if (casesCount > 0) {
    // Есть кейсы - скрываем подсказку и кнопку "В магазин", показываем кнопку "Открыть"
    if (casesHintBadge) {
      casesHintBadge.style.display = "none";
    }
    if (casesOpenBtn) {
      casesOpenBtn.style.display = "flex";
    }
    if (casesShopBtn) {
      casesShopBtn.style.display = "none";
    }
  } else {
    // Нет кейсов - показываем подсказку и кнопку "В магазин", скрываем кнопку "Открыть"
    if (casesHintBadge) {
      casesHintBadge.style.display = "flex";
    }
    if (casesOpenBtn) {
      casesOpenBtn.style.display = "none";
    }
    if (casesShopBtn) {
      casesShopBtn.style.display = "flex";
    }
  }
}

// Загрузка настроек
async function loadSettings(authData) {
  try {
    let url = "/api/settings";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    } else {
      throw new Error("Invalid authentication data");
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const data = await response.json();
    currentSettings = data;
    return data;
  } catch (error) {
    console.error("Ошибка загрузки настроек:", error);
    // Если настройки не найдены, создаем дефолтные
    currentSettings = getDefaultSettings();
    return currentSettings;
  }
}

// Сохранение настроек
async function saveSettings(authData, settings) {
  try {
    let url = "/api/settings";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    };

    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, options);
    if (!response.ok) {
      const errorText = await response.text();
      console.error("Ошибка ответа сервера:", response.status, errorText);
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = { message: errorText };
      }
      throw new Error(errorData.message || `Ошибка ${response.status}`);
    }
    const data = await response.json();
    console.log("Настройки успешно сохранены:", data);
    // Обновляем настройки из ответа сервера
    if (data.settings) {
      currentSettings = data.settings;
    } else {
      currentSettings = { ...currentSettings, ...settings };
    }
    return data;
  } catch (error) {
    console.error("Ошибка сохранения настроек:", error);
    throw error;
  }
}

// Дефолтные настройки
function getDefaultSettings() {
  return {
    notif_cases: true,
    notif_daily_rewards: true,
    notif_game_invites: true,
    notif_friend_requests: true,
    notif_events: true,
    notif_news: false,
    notif_dice: false,
    notif_generator: true,
    ads_enabled: true,
    sound_music: true,
    sound_sfx: true,
    social_block_friend_requests: false,
  };
}

// Воспроизведение звука нажатия кнопки
function playButtonSound() {
  // Проверяем настройки звуковых эффектов
  if (currentSettings && currentSettings.sound_sfx === false) {
    return; // Звуковые эффекты выключены
  }
  
  // Если настройки еще не загружены, используем дефолтное значение (true)
  const buttonSound = document.getElementById("button-sound");
  if (buttonSound) {
    // Сбрасываем звук на начало для возможности повторного воспроизведения
    buttonSound.currentTime = 0;
    buttonSound.play().catch(error => {
      // Игнорируем ошибки воспроизведения (например, если пользователь не взаимодействовал со страницей)
      console.debug("Не удалось воспроизвести звук кнопки:", error);
    });
  }
}

// Воспроизведение звука открытия меню выбора режима боя
function playBattleModeSound() {
  if (currentSettings && currentSettings.sound_sfx === false) {
    return;
  }
  
  const battleModeSound = document.getElementById("battle-mode-sound");
  if (battleModeSound) {
    battleModeSound.currentTime = 0;
    battleModeSound.play().catch(error => {
      console.debug("Не удалось воспроизвести звук открытия меню боя:", error);
    });
  }
}

// Воспроизведение звука успешной покупки
function playPurchaseSuccessSound() {
  if (currentSettings && currentSettings.sound_sfx === false) {
    return;
  }
  
  const purchaseSound = document.getElementById("purchase-success-sound");
  if (purchaseSound) {
    purchaseSound.currentTime = 0;
    purchaseSound.play().catch(error => {
      console.debug("Не удалось воспроизвести звук успешной покупки:", error);
    });
  }
}

// Воспроизведение звука покупки ресурсов (монет)
function playResourcePurchaseSound() {
  window._playSfx?.('purchase-sound');
}

// Воспроизведение звука показа наград кейса
function playCaseRewardsSound() {
  window._playSfx?.('case-reward-sound');
}

// Воспроизведение звука начала открытия кейса
function playCaseOpenedSound() {
  window._playSfx?.('case-open-sound');
}

// Воспроизведение звука тапа по кейсу
function playCaseTapSound() {
  window._playSfx?.('case-tap-sound');
}
// Воспроизведение звука тапа при открытии кейса
// Воспроизведение звука отправки сообщения в чат
function playChatMessageSound() {
  window._playSfx?.('chat-sent-sound');
}

// Отображение профиля
function renderProfile(data) {
  console.log("renderProfile вызвана с данными:", data);
  
  const playerName = document.getElementById("player-name");
  const playerTitle = document.getElementById("player-title");
  const avatarImg = document.getElementById("avatar-img");
  const playerAvatar = document.getElementById("player-avatar");
  const extrapassBadge = document.getElementById("extrapass-badge");
  const topBar = document.getElementById("top-bar");
  const resourceGems = document.getElementById("resource-gems");
  const resourceCoins = document.getElementById("resource-coins");
  const infoUserId = document.getElementById("info-user-id");

  // Имя пользователя (используем кастомный никнейм, если есть)
  if (playerName) {
    const name = data.custom_nickname || data.first_name || data.username || "Игрок";
    playerName.textContent = name;
    console.log("Имя установлено:", name);
  } else {
    console.error("Элемент player-name не найден!");
  }

  // Титул (вместо трофеев)
  if (playerTitle) {
    const isAdmin = data.user_id === 6803854304;
    const title = isAdmin ? "Администратор" : (data.title || "Игрок");
    playerTitle.textContent = title;
    if (isAdmin) {
      playerTitle.style.color = "var(--chibi-red)";
      playerTitle.style.fontWeight = "bold";
    }
    console.log("Титул установлен:", title);
  } else {
    console.error("Элемент player-title не найден!");
  }

  // Аватарка
  if (data.photo_url && data.photo_url.trim() !== "") {
    console.log("Устанавливаем фото:", data.photo_url);
    if (avatarImg) {
      avatarImg.onload = () => {
        console.log("Аватарка успешно загружена");
        // Скрываем текстовый фон, если есть изображение
        if (playerAvatar) {
          playerAvatar.style.background = "transparent";
          // Очищаем текст, если он был
          const textNodes = Array.from(playerAvatar.childNodes).filter(
            node => node.nodeType === Node.TEXT_NODE && node !== avatarImg
          );
          textNodes.forEach(node => node.remove());
        }
      };
      avatarImg.onerror = () => {
        console.error("Ошибка загрузки изображения:", data.photo_url);
        // Если изображение не загрузилось, показываем букву
        avatarImg.style.display = "none";
        showAvatarLetter(playerAvatar, data);
      };
      avatarImg.src = data.photo_url;
      avatarImg.alt = data.first_name || "Аватар";
      avatarImg.style.display = "block";
    }
  } else {
    console.log("Фото нет, показываем букву. photo_url:", data.photo_url);
    // Если нет фото, показываем первую букву имени
    if (avatarImg) {
      avatarImg.style.display = "none";
    }
    showAvatarLetter(playerAvatar, data);
  }

  // ExtraPass
  const hasExtraPass = data.extra_pass === "active";
  if (hasExtraPass) {
    if (extrapassBadge) {
      extrapassBadge.style.display = "flex";
    }
    if (playerAvatar) {
      playerAvatar.classList.add("extrapass-active");
    }
    if (topBar) {
      topBar.classList.add("extrapass-active");
    }
  } else {
    if (extrapassBadge) {
      extrapassBadge.style.display = "none";
    }
    if (playerAvatar) {
      playerAvatar.classList.remove("extrapass-active");
    }
    if (topBar) {
      topBar.classList.remove("extrapass-active");
    }
  }
  
  // Обновляем видимость ExtraPass в магазине
  updateShopExtraPassVisibility(hasExtraPass);

  // Ресурсы
  if (resourceGems) {
    resourceGems.textContent = data.gems || 0;
  }
  if (resourceCoins) {
    resourceCoins.textContent = data.coins || 0;
  }
  
  // Баттлпасс
  const battlepassSeason = document.getElementById("battlepass-season");
  const battlepassStarsCurrent = document.getElementById("battlepass-stars-current");
  const battlepassStarsNext = document.getElementById("battlepass-stars-next");
  const battlepassProgress = document.getElementById("battlepass-progress");
  
  if (battlepassSeason) {
    battlepassSeason.textContent = data.season || 0;
  }
  if (battlepassStarsCurrent) {
    battlepassStarsCurrent.textContent = data.stars || 0;
  }
  if (battlepassStarsNext) {
    battlepassStarsNext.textContent = "100"; // Заглушка
  }
  if (battlepassProgress) {
    const stars = data.stars || 0;
    const next = 100; // Заглушка
    const percent = Math.min((stars / next) * 100, 100);
    battlepassProgress.style.width = percent + "%";
  }
  
  // Арена
  const arenaName = document.getElementById("arena-name");
  const arenaTrophiesValue = document.getElementById("arena-trophies-value");
  const arenaTrophiesNext = document.getElementById("arena-trophies-next");
  
  // Функция для определения следующей лиги
  function getNextLeagueTrophies(trophies) {
    if (trophies < 300) return 300;
    if (trophies < 600) return 600;
    if (trophies < 1200) return 1200;
    if (trophies < 2000) return 2000;
    if (trophies < 3000) return 3000;
    if (trophies < 4500) return 4500;
    if (trophies < 6000) return 6000;
    if (trophies < 7500) return 7500;
    if (trophies < 9000) return 9000;
    if (trophies < 10000) return 10000;
    return trophies; // Максимальная лига
  }
  
  if (arenaName) {
    const league = data.league != null ? getLeagueById(data.league) : getLeagueByTrophies(data.trophies || 0);
    arenaName.textContent = `${league.emoji} ${league.name}`;
  }
  if (arenaTrophiesValue) {
    arenaTrophiesValue.textContent = data.trophies || 0;
  }
  if (arenaTrophiesNext) {
    const trophies = data.trophies || 0;
    const nextLeague = getNextLeagueTrophies(trophies);
    arenaTrophiesNext.textContent = nextLeague;
  }
  
  // Энергия (компактная версия слева от кнопки)
  const maxEnergy = data.extra_pass === "active" ? 6 : 5;
  const currentEnergy = data.energy || maxEnergy;
  const energyCurrentDisplay = document.getElementById("energy-current-display");
  const energyMaxDisplay = document.getElementById("energy-max-display");
  const energyExtrapassHint = document.getElementById("energy-extrapass-hint");
  
  if (energyCurrentDisplay) {
    energyCurrentDisplay.textContent = currentEnergy;
  }
  if (energyMaxDisplay) {
    energyMaxDisplay.textContent = maxEnergy;
  }
  if (energyExtrapassHint) {
    if (data.extra_pass !== "active") {
      energyExtrapassHint.style.display = "flex";
    } else {
      energyExtrapassHint.style.display = "none";
    }
  }
  
  // Премиум слот
  const premiumSlot = document.getElementById("premium-slot");
  const premiumLockIcon = document.getElementById("premium-lock-icon");
  const premiumExtrapassPromo = document.getElementById("premium-extrapass-promo");
  if (premiumSlot) {
    if (data.extra_pass !== "active") {
      premiumSlot.classList.add("locked");
      premiumSlot.title = "Требуется ExtraPass";
      if (premiumLockIcon) {
        premiumLockIcon.style.display = "block";
      }
      if (premiumExtrapassPromo) {
        premiumExtrapassPromo.style.display = "flex";
      }
    } else {
      premiumSlot.classList.remove("locked");
      if (premiumLockIcon) {
        premiumLockIcon.style.display = "none";
      }
      if (premiumExtrapassPromo) {
        premiumExtrapassPromo.style.display = "none";
      }
    }
  }

  // ID для информации
  if (infoUserId) {
    infoUserId.textContent = data.user_id || "—";
  }

  // Настройки
  if (data.settings) {
    currentSettings = data.settings;
    renderSettings(data.settings);
    // Управляем музыкой в соответствии с настройками
    if (data.settings.sound_music !== false) {
      toggleMusic(true);
    } else {
      toggleMusic(false);
    }
  }
}

// Обновление видимости ExtraPass в магазине
function updateShopExtraPassVisibility(hasExtraPass) {
  const extrapassItem = document.getElementById("extrapass-shop-item");
  if (extrapassItem) {
    extrapassItem.style.display = hasExtraPass ? "none" : "flex";
  }
  
  // Обновляем стартовый буст: если есть ExtraPass, заменяем его на +700 гемов
  const starterBoostItem = document.getElementById("starter-boost-shop-item");
  const starterBoostStats = document.getElementById("starter-boost-stats");
  if (starterBoostItem && starterBoostStats) {
    const extrapassStat = starterBoostStats.querySelector(".extrapass-stat");
    if (hasExtraPass) {
      // Если есть ExtraPass, заменяем его на +700 гемов
      if (extrapassStat) {
        extrapassStat.innerHTML = '💎 +700 гемов';
        extrapassStat.classList.remove("extrapass-stat");
      } else {
        // Если элемента нет, создаем новый
        const firstStat = starterBoostStats.querySelector(".stat-item");
        if (firstStat) {
          const gemsEl = document.createElement("span");
          gemsEl.className = "stat-item";
          gemsEl.textContent = "💎 +700 гемов";
          starterBoostStats.insertBefore(gemsEl, firstStat);
        }
      }
      // Обновляем описание гемов (500 -> 1200)
      const gemsStat = Array.from(starterBoostStats.querySelectorAll(".stat-item")).find(s => s.textContent.includes("500 гемов"));
      if (gemsStat) {
        gemsStat.textContent = "💎 1200 гемов";
      }
    } else {
      // Если нет ExtraPass, возвращаем оригинальное описание
      const gemsStat = Array.from(starterBoostStats.querySelectorAll(".stat-item")).find(s => s.textContent.includes("1200 гемов"));
      if (gemsStat) {
        gemsStat.textContent = "💎 500 гемов";
      }
      // Убираем +700 гемов, если он был добавлен
      const extraGemsStat = Array.from(starterBoostStats.querySelectorAll(".stat-item")).find(s => s.textContent.includes("+700 гемов"));
      if (extraGemsStat && !extrapassStat) {
        extraGemsStat.remove();
        // Возвращаем ExtraPass
        const firstStat = starterBoostStats.querySelector(".stat-item");
        if (firstStat) {
          const extrapassEl = document.createElement("span");
          extrapassEl.className = "stat-item extrapass-stat";
          extrapassEl.textContent = "⭐ ExtraPass";
          starterBoostStats.insertBefore(extrapassEl, firstStat);
        }
      }
    }
  }
}

function showAvatarLetter(playerAvatar, data) {
  if (!playerAvatar) return;
  
  const firstLetter = (data.first_name || data.username || "И")[0].toUpperCase();
  // Очищаем содержимое, кроме img и других элементов
  const existingText = Array.from(playerAvatar.childNodes).find(
    node => node.nodeType === Node.TEXT_NODE
  );
  if (!existingText || existingText.textContent.trim() !== firstLetter) {
    // Удаляем все текстовые узлы
    Array.from(playerAvatar.childNodes).forEach(node => {
      if (node.nodeType === Node.TEXT_NODE) {
        node.remove();
      }
    });
    // Добавляем новую букву
    const textNode = document.createTextNode(firstLetter);
    playerAvatar.appendChild(textNode);
  }
  playerAvatar.style.background = "linear-gradient(135deg, var(--chibi-pink), var(--chibi-purple))";
  
  // Скрываем img, если он есть
  const avatarImg = document.getElementById("avatar-img");
  if (avatarImg) {
    avatarImg.style.display = "none";
  }
}

// Отображение настроек
function renderSettings(settings) {
  const settingsContent = document.getElementById("settings-content");
  if (!settingsContent) return;

  const defaultSettings = getDefaultSettings();
  const mergedSettings = { ...defaultSettings, ...settings };

  const isAdmin = currentProfile?.user_id === 6803854304;
  const hasExtraPass = currentProfile?.extra_pass === "active";
  
  settingsContent.innerHTML = `
    <div class="setting-group">
      <div class="setting-group-title">🔔 Уведомления</div>
      <div class="setting-item">
        <span class="setting-label">Уведомления о кейсах</span>
        <div class="toggle-switch ${mergedSettings.notif_cases ? "active" : ""}" data-setting="notif_cases"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Ежедневные награды</span>
        <div class="toggle-switch ${mergedSettings.notif_daily_rewards ? "active" : ""}" data-setting="notif_daily_rewards"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Приглашения в игру</span>
        <div class="toggle-switch ${mergedSettings.notif_game_invites ? "active" : ""}" data-setting="notif_game_invites"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Запросы в друзья</span>
        <div class="toggle-switch ${mergedSettings.notif_friend_requests ? "active" : ""}" data-setting="notif_friend_requests"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">События</span>
        <div class="toggle-switch ${mergedSettings.notif_events ? "active" : ""}" data-setting="notif_events"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Новости</span>
        <div class="toggle-switch ${mergedSettings.notif_news ? "active" : ""}" data-setting="notif_news"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Генератор ключей</span>
        <div class="toggle-switch ${mergedSettings.notif_generator ? "active" : ""}" data-setting="notif_generator"></div>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-group-title">📺 Реклама</div>
      <div class="setting-item ${hasExtraPass ? "disabled" : ""}">
        <span class="setting-label">Показывать рекламу${hasExtraPass ? " (отключено ExtraPass)" : ""}</span>
        <div class="toggle-switch ${mergedSettings.ads_enabled && !hasExtraPass ? "active" : ""}" data-setting="ads_enabled" ${hasExtraPass ? "style='cursor: not-allowed;'" : ""}></div>
      </div>
      ${!hasExtraPass ? `<button class="ads-place-btn" id="ads-place-btn">Разместить</button>` : ""}
      ${!hasExtraPass ? `
        <div class="extrapass-promo">
          <div class="extrapass-promo-text">⚡ <b>Купите ExtraPass</b>, чтобы отключить рекламу и получить множество бонусов!</div>
          <div class="extrapass-benefits">
            <div>✨ <b>Отключение рекламы</b> - играйте без прерываний</div>
            <div>✨ <b>+1 бой за КД</b> - больше возможностей для побед</div>
            <div>✨ <b>5-й слот для кейсов</b> - храните больше наград</div>
            <div>✨ <b>Эксклюзивные награды</b> - уникальные предметы и бонусы</div>
          </div>
          <button class="extrapass-promo-btn" id="extrapass-shop-btn">Купить ExtraPass</button>
        </div>
      ` : ""}
    </div>

    <div class="setting-group">
      <div class="setting-group-title">🔊 Звук</div>
      <div class="setting-item">
        <span class="setting-label">Музыка</span>
        <div class="toggle-switch ${mergedSettings.sound_music ? "active" : ""}" data-setting="sound_music"></div>
      </div>
      <div class="setting-item">
        <span class="setting-label">Звуковые эффекты</span>
        <div class="toggle-switch ${mergedSettings.sound_sfx ? "active" : ""}" data-setting="sound_sfx"></div>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-group-title">👥 Социальные</div>
      <div class="setting-item">
        <span class="setting-label">Блокировать запросы в друзья</span>
        <div class="toggle-switch ${mergedSettings.social_block_friend_requests ? "active" : ""}" data-setting="social_block_friend_requests"></div>
      </div>
      <div class="setting-item">
        <div style="display: flex; flex-direction: column; gap: 8px; width: 100%;">
          <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <span class="setting-label">Никнейм</span>
            <span class="nickname-cost" id="nickname-cost" style="font-size: 12px; color: var(--chibi-gold);">${currentProfile?.nickname_changed ? "500 💎" : "Бесплатно"}</span>
          </div>
          <div style="display: flex; gap: 8px; width: 100%;">
            <input type="text" id="nickname-input" class="nickname-input" placeholder="Введите никнейм..." value="${currentProfile?.custom_nickname || ""}" maxlength="20" style="flex: 1; padding: 10px; background: rgba(192, 132, 252, 0.1); border: 2px solid var(--chibi-border); border-radius: 8px; color: var(--chibi-text); font-size: 14px;" />
            <button class="btn-secondary" id="change-nickname-btn" style="padding: 10px 16px; white-space: nowrap;" title="Изменить никнейм">✏️</button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Обработчики переключателей
  document.querySelectorAll(".toggle-switch").forEach((toggle) => {
    toggle.addEventListener("click", async () => {
      // Проверяем, не заблокирован ли переключатель
      if (toggle.closest(".setting-item.disabled")) {
        return;
      }
      
      const setting = toggle.dataset.setting;
      const isActive = toggle.classList.contains("active");
      const newValue = !isActive;
      
      // Визуально обновляем сразу
      toggle.classList.toggle("active");
      
      // Специальная обработка для музыки
      if (setting === "sound_music") {
        toggleMusic(newValue);
      }

      const authData = resolveUserId();
      if (authData) {
        try {
          console.log(`Сохранение настройки ${setting} = ${newValue}`);
          await saveSettings(authData, { [setting]: newValue });
          console.log(`Настройка ${setting} успешно сохранена`);
          // Обновляем текущие настройки
          currentSettings = { ...currentSettings, [setting]: newValue };
          try {
            if (tg?.HapticFeedback?.impactOccurred) {
              tg.HapticFeedback.impactOccurred("light");
            }
          } catch (e) {
            // Игнорируем ошибки HapticFeedback
          }
        } catch (error) {
          console.error("Ошибка сохранения настройки:", error);
          toggle.classList.toggle("active"); // Откатываем изменение
          if (setting === "sound_music") {
            toggleMusic(!newValue); // Откатываем музыку
          }
          await showGameAlert("Не удалось сохранить настройку. Попробуйте еще раз.", "❌");
        }
      } else {
        console.error("Нет данных авторизации для сохранения настроек");
        toggle.classList.toggle("active"); // Откатываем изменение
        if (setting === "sound_music") {
          toggleMusic(!newValue); // Откатываем музыку
        }
      }
    });
  });
  
  // Кнопка покупки ExtraPass
  const extrapassShopBtn = document.getElementById("extrapass-shop-btn");
  if (extrapassShopBtn) {
    extrapassShopBtn.addEventListener("click", async () => {
      // TODO: Открыть магазин ExtraPass
      await showGameAlert("Магазин ExtraPass скоро будет доступен!", "ℹ️");
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {
        // Игнорируем ошибки HapticFeedback
      }
    });
  }

  // Кнопка размещения рекламы
  const adsPlaceBtn = document.getElementById("ads-place-btn");
  if (adsPlaceBtn) {
    adsPlaceBtn.addEventListener("click", () => {
      // TODO: Логика размещения рекламы
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {
        // Игнорируем ошибки HapticFeedback
      }
    });
  }
  
  // Обработчик смены никнейма
  const changeNicknameBtn = document.getElementById("change-nickname-btn");
  const nicknameInput = document.getElementById("nickname-input");
  if (changeNicknameBtn && nicknameInput) {
    changeNicknameBtn.addEventListener("click", async () => {
      const newNickname = nicknameInput.value.trim();
      
      if (!newNickname) {
        await showGameAlert("Введите никнейм", "⚠️");
        return;
      }
      
      if (newNickname.length > 20) {
        await showGameAlert("Никнейм не может быть длиннее 20 символов", "⚠️");
        return;
      }
      
      // Фильтр запрещенных слов и паттернов
      const forbiddenPatterns = [
        /евреи\s*пидорасы/i,
        /пидорасы\s*евреи/i,
        /https?:\/\//i,
        /www\./i,
        /\.(com|ru|org|net|io|me|gg)/i,
        /@\w+/i,
        /t\.me\//i,
        /telegram/i,
        /channel/i,
        /bot/i,
        /admin/i,
        /moderator/i,
        /cp\s*[0-9]/i,
        /[0-9]{4,}/, // Длинные числа (возможно, ID)
      ];
      
      const forbiddenWords = [
        "реклама",
        "купить",
        "продать",
        "скидка",
        "промокод",
        "бесплатно",
        "халява",
      ];
      
      // Проверка на запрещенные паттерны
      for (const pattern of forbiddenPatterns) {
        if (pattern.test(newNickname)) {
          await showGameAlert("Никнейм содержит запрещенные символы или ссылки", "❌");
          return;
        }
      }
      
      // Проверка на запрещенные слова (не слишком строго)
      const lowerNickname = newNickname.toLowerCase();
      for (const word of forbiddenWords) {
        if (lowerNickname.includes(word)) {
          await showGameAlert("Никнейм содержит запрещенные слова", "❌");
          return;
        }
      }
      
      const authData = resolveUserId();
      if (!authData) {
        await showGameAlert("Ошибка авторизации", "❌");
        return;
      }
      
      try {
        let url = "/api/change-nickname";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nickname: newNickname })
        });
        
        const result = await response.json();
        
        if (result.success) {
          await showGameAlert(result.is_first_change ? "Никнейм успешно изменен!" : `Никнейм изменен! Потрачено: ${result.cost} 💎`, "✅");
          // Обновляем профиль
          await loadProfile(authData);
          // Обновляем стоимость
          const costElement = document.getElementById("nickname-cost");
          if (costElement) {
            costElement.textContent = "500 💎";
          }
          try {
            if (tg?.HapticFeedback?.impactOccurred) {
              tg.HapticFeedback.impactOccurred("medium");
            }
          } catch (e) {}
        } else {
          if (result.error === "insufficient_gems") {
            await showGameAlert(`Недостаточно гемов! Требуется: ${result.required} 💎, у вас: ${result.current} 💎`, "❌");
          } else {
            await showGameAlert("Ошибка при смене никнейма", "❌");
          }
        }
      } catch (error) {
        console.error("Ошибка смены никнейма:", error);
        await showGameAlert("Ошибка при смене никнейма", "❌");
      }
    });
  }
}

// Функция для определения лиги по трофеям (10 лиг)
function getLeagueByTrophies(trophies) {
  if (trophies >= 9000) return { name: "Extra", emoji: "🏟️", color: "#FFD700", id: 10 };
  if (trophies >= 7500) return { name: "Legendary", emoji: "👑", color: "#FF6B6B", id: 9 };
  if (trophies >= 6000) return { name: "Grandmaster", emoji: "💫", color: "#9B59B6", id: 8 };
  if (trophies >= 4500) return { name: "Champion", emoji: "🏆", color: "#E74C3C", id: 7 };
  if (trophies >= 3000) return { name: "Master", emoji: "⭐", color: "#F39C12", id: 6 };
  if (trophies >= 2000) return { name: "Crystal", emoji: "💎", color: "#3498DB", id: 5 };
  if (trophies >= 1200) return { name: "Gold", emoji: "🥇", color: "#F1C40F", id: 4 };
  if (trophies >= 600)  return { name: "Silver", emoji: "🥈", color: "#95A5A6", id: 3 };
  if (trophies >= 300)  return { name: "Bronze", emoji: "🥉", color: "#E67E22", id: 2 };
  return { name: "Novice", emoji: "🌱", color: "#2ECC71", id: 1 };
}

// Функция для получения данных лиги по ID (из profile.league с сервера)
const LEAGUE_BY_ID = (function() {
  const map = {};
  [0, 300, 600, 1200, 2000, 3000, 4500, 6000, 7500, 9000].forEach((t) => {
    const l = getLeagueByTrophies(t);
    map[l.id] = l;
  });
  return map;
})();
function getLeagueById(leagueId) {
  return LEAGUE_BY_ID[leagueId] || LEAGUE_BY_ID[1];
}

// Структура шагов пути славы с наградами (линейная дорога 0-10000, шаг 500)
const GLORY_PATH_MILESTONES = [
  // 150 - промежуточная награда Novice
  { trophies: 150, rewards: [{ type: "coins", amount: 1000, icon: "💰" }, { type: "case", tier: 1, icon: "📦" }], league: null },
  // 300 - вход в Bronze + кейс T2 + 50 гемов
  { trophies: 300, rewards: [{ type: "case", tier: 2, icon: "💠" }, { type: "gems", amount: 50, icon: "💎" }], league: { name: "Bronze", emoji: "🥉" } },
  // 450 - промежуточная награда Bronze
  { trophies: 450, rewards: [{ type: "coins", amount: 2000, icon: "💰" }, { type: "particles", rarity: "rare", amount: 20, icon: "✨" }], league: null },
  // 600 - вход в Silver + кейс T2 + гарантированная Редкая карта
  { trophies: 600, rewards: [{ type: "case", tier: 2, icon: "💠" }, { type: "guaranteed_card", rarity: "rare", icon: "🃏" }], league: { name: "Silver", emoji: "🥈" } },
  // 900 - промежуточная награда Silver
  { trophies: 900, rewards: [{ type: "case", tier: 3, icon: "💎" }, { type: "shards", amount: 50, icon: "🔮" }], league: null },
  // 1200 - вход в Gold + 3000 монет + 30 частиц Superrare
  { trophies: 1200, rewards: [{ type: "coins", amount: 3000, icon: "💰" }, { type: "particles", rarity: "superrare", amount: 30, icon: "✨" }], league: { name: "Gold", emoji: "🥇" } },
  // 1600 - промежуточная награда Gold
  { trophies: 1600, rewards: [{ type: "case", tier: 3, icon: "💎" }, { type: "gems", amount: 75, icon: "💎" }], league: null },
  // 2000 - вход в Crystal + гарантированная Эпическая карта
  { trophies: 2000, rewards: [{ type: "guaranteed_card", rarity: "epic", icon: "🃏" }], league: { name: "Crystal", emoji: "💎" } },
  // 2500 - промежуточная награда Crystal
  { trophies: 2500, rewards: [{ type: "case", tier: 4, icon: "🔥" }, { type: "coins", amount: 5000, icon: "💰" }], league: null },
  // 3000 - вход в Master + гарантированная Легендарная карта
  { trophies: 3000, rewards: [{ type: "guaranteed_card", rarity: "legendary", icon: "🃏" }], league: { name: "Master", emoji: "⭐" } },
  // 3750 - промежуточная награда Master
  { trophies: 3750, rewards: [{ type: "case", tier: 4, icon: "🔥" }, { type: "shards", amount: 100, icon: "🔮" }], league: null },
  // 4500 - вход в Champion + 8000 монет + 50 частиц Epic
  { trophies: 4500, rewards: [{ type: "coins", amount: 8000, icon: "💰" }, { type: "particles", rarity: "epic", amount: 50, icon: "✨" }], league: { name: "Champion", emoji: "🏆" } },
  // 5250 - промежуточная награда Champion
  { trophies: 5250, rewards: [{ type: "case", tier: 4, icon: "🔥" }, { type: "gems", amount: 125, icon: "💎" }], league: null },
  // 6000 - вход в Grandmaster + кейс T5
  { trophies: 6000, rewards: [{ type: "case", tier: 5, icon: "👑" }], league: { name: "Grandmaster", emoji: "💫" } },
  // 6750 - промежуточная награда Grandmaster
  { trophies: 6750, rewards: [{ type: "coins", amount: 15000, icon: "💰" }, { type: "particles", rarity: "legendary", amount: 100, icon: "✨" }], league: null },
  // 7500 - вход в Legendary + гарантированная Мифическая карта
  { trophies: 7500, rewards: [{ type: "guaranteed_card", rarity: "mythic", icon: "🃏" }], league: { name: "Legendary", emoji: "👑" } },
  // 8250 - промежуточная награда Legendary
  { trophies: 8250, rewards: [{ type: "case", tier: 5, icon: "👑" }, { type: "shards", amount: 200, icon: "🔮" }], league: null },
  // 9000 - вход в Extra + 20000 монет
  { trophies: 9000, rewards: [{ type: "coins", amount: 20000, icon: "💰" }], league: { name: "Extra", emoji: "🏟️" } },
  // 10000 - финальная награда: Божественный кейс T5 + косметика
  { trophies: 10000, rewards: [{ type: "case", tier: 5, divine: true, icon: "✨" }, { type: "cosmetic", icon: "🎨" }], league: null },
];

// Функция для форматирования награды
function formatRewardLabel(reward) {
  if (reward.type === "coins") {
    return "монет";
  } else if (reward.type === "gems") {
    return "гемов";
  } else if (reward.type === "case") {
    const tierNames = { 1: "Обычный", 2: "Улучшенный", 3: "Элитный", 4: "Легендарный", 5: "Божественный" };
    const name = reward.divine ? "Божественный" : tierNames[reward.tier] || `T${reward.tier}`;
    return `кейс ${name}`;
  } else if (reward.type === "particles") {
    const rarityNames = { rare: "Редкой", superrare: "Сверхредкой", epic: "Эпической", legendary: "Легендарной" };
    return `${reward.amount} частиц ${rarityNames[reward.rarity] || reward.rarity}`;
  } else if (reward.type === "shards") {
    return `${reward.amount} осколков`;
  } else if (reward.type === "guaranteed_card") {
    const rarityNames = { rare: "Редкая", epic: "Эпическая", legendary: "Легендарная", mythic: "Мифическая" };
    return `Гарант. ${rarityNames[reward.rarity] || reward.rarity}`;
  } else if (reward.type === "cosmetic") {
    return "Косметика";
  }
  return "";
}

// Отображение пути славы
function renderGloryPath(data) {
  const gloryPathContent = document.getElementById("glory-path-content");
  if (!gloryPathContent) return;

  const currentTrophies = data.trophies || 0;
  const currentLeague = data.league != null ? getLeagueById(data.league) : getLeagueByTrophies(currentTrophies);
  
  // Вычисляем общий прогресс (максимум 10000)
  const maxTrophies = 10000;
  const overallProgress = Math.min(100, (currentTrophies / maxTrophies) * 100);

  let html = `
    <div class="glory-path-container">
      <div class="glory-path-header">
        <div class="glory-path-current-league">
          <div class="league-emoji-large">${currentLeague.emoji}</div>
          <div class="league-info">
            <div class="league-name-large">${currentLeague.name}</div>
            <div class="league-trophies">${currentTrophies.toLocaleString()} 🏆</div>
          </div>
        </div>
        <div class="glory-path-overall-progress">
          <div class="overall-progress-label">Общий прогресс</div>
          <div class="overall-progress-bar">
            <div class="overall-progress-fill" style="width: ${overallProgress}%"></div>
            <div class="overall-progress-text">${overallProgress.toFixed(1)}%</div>
          </div>
        </div>
      </div>
      <div class="glory-path-road">
  `;

  GLORY_PATH_MILESTONES.forEach((milestone, index) => {
    const isCompleted = currentTrophies >= milestone.trophies;
    const prevMilestone = index > 0 ? GLORY_PATH_MILESTONES[index - 1] : { trophies: 0 };
    const isNext = !isCompleted && currentTrophies >= prevMilestone.trophies;
    const isLocked = !isCompleted && !isNext;
    
    // Вычисляем прогресс до следующего шага
    const progressToNext = isNext ? Math.min(100, ((currentTrophies - prevMilestone.trophies) / (milestone.trophies - prevMilestone.trophies)) * 100) : 0;
    
    // Определяем, заполнена ли линия до этого шага
    const isLineFilled = isCompleted || (isNext && progressToNext > 0);
    
    // Вычисляем сколько осталось до шага
    const remaining = milestone.trophies - currentTrophies;

    html += `
      <div class="glory-path-segment">
        <div class="glory-path-line ${isLineFilled ? "filled" : ""}" style="height: ${index === 0 ? "0" : "45px"}">
          ${isNext && progressToNext > 0 ? `<div class="glory-path-line-progress" style="height: ${progressToNext}%"></div>` : ""}
          ${isLineFilled ? `<div class="glory-path-line-glow"></div>` : ""}
        </div>
        <div class="glory-milestone-card ${isCompleted ? "completed" : ""} ${isNext ? "next" : ""} ${isLocked ? "locked" : ""}">
          <div class="milestone-card-bg"></div>
          ${milestone.league ? `
            <div class="milestone-league-header">
              <div class="milestone-league-badge-large">
                <span class="league-emoji-badge">${milestone.league.emoji}</span>
                <span class="league-name-badge">${milestone.league.name}</span>
              </div>
              <div class="league-unlock-text">Новая лига!</div>
            </div>
          ` : ""}
          <div class="milestone-main-content">
            <div class="milestone-trophy-value">
              <span class="trophy-icon-card">🏆</span>
              <span class="milestone-trophies-value">${milestone.trophies.toLocaleString()}</span>
            </div>
            ${!isCompleted && isNext ? `
              <div class="milestone-remaining">
                <span class="remaining-icon">📊</span>
                <span class="remaining-text">Осталось: ${remaining.toLocaleString()}</span>
              </div>
            ` : ""}
            <div class="milestone-rewards-grid">
              ${milestone.rewards.map(reward => {
                const label = formatRewardLabel(reward);
                // Для частиц и осколков amount уже включен в label, для остальных показываем отдельно
                const showAmount = reward.amount && reward.type !== "particles" && reward.type !== "shards";
                const amount = showAmount ? reward.amount.toLocaleString() : "";
                return `
                <div class="milestone-reward-card">
                  <div class="reward-icon-large">${reward.icon}</div>
                  ${amount ? `<div class="reward-amount-large">${amount}</div>` : ""}
                  <div class="reward-type-label">${label}</div>
                </div>
              `;
              }).join("")}
            </div>
          </div>
          ${isCompleted ? `
            <div class="milestone-checkmark">
              <div class="checkmark-icon">✓</div>
              <div class="checkmark-glow"></div>
            </div>
          ` : ""}
          ${isNext ? `
            <div class="milestone-next-indicator">
              <span class="next-indicator-text">Следующий</span>
              <div class="next-indicator-arrow">↓</div>
            </div>
          ` : ""}
          ${isLocked ? `
            <div class="milestone-lock-overlay">
              <div class="lock-icon">🔒</div>
            </div>
          ` : ""}
        </div>
      </div>
    `;
  });

  html += `
      </div>
    </div>
  `;
  gloryPathContent.innerHTML = html;
}

// Отображение аналитики
function renderAnalytics(data) {
  const analyticsContent = document.getElementById("analytics-content");
  if (!analyticsContent) return;

  const isAdmin = data.user_id === 6803854304;
  const league = data.league != null ? getLeagueById(data.league) : getLeagueByTrophies(data.trophies || 0);
  const hasExtraPass = data.extra_pass === "active";
  
  // Данные для графика (примерная кривая прогресса)
  const maxTrophies = Math.max(data.max_trophies || 0, data.trophies || 0, 100);
  const chartData = [];
  for (let i = 0; i <= 7; i++) {
    const dayTrophies = Math.floor((data.trophies || 0) * (0.3 + i * 0.1));
    chartData.push(dayTrophies);
  }

  analyticsContent.innerHTML = `
    <div class="analytics-header">
      <div class="league-badge" style="background: linear-gradient(135deg, ${league.color}, ${league.color}88);">
        <span class="league-emoji">${league.emoji}</span>
        <span class="league-name">${league.name}</span>
      </div>
    </div>
    
    <div class="analytics-chart">
      <div class="chart-title">Прогресс трофеев (7 дней)</div>
      <div class="chart-container">
        <svg class="chart-svg" viewBox="0 0 300 100" preserveAspectRatio="none">
          <polyline
            class="chart-line"
            points="${chartData.map((val, i) => `${(i * 300) / (chartData.length - 1)},${100 - (val / maxTrophies) * 80}`).join(' ')}"
            fill="none"
            stroke="var(--chibi-gold)"
            stroke-width="2"
          />
          ${chartData.map((val, i) => `
            <circle
              cx="${(i * 300) / (chartData.length - 1)}"
              cy="${100 - (val / maxTrophies) * 80}"
              r="3"
              fill="var(--chibi-gold)"
            />
          `).join('')}
        </svg>
      </div>
    </div>

    <div class="analytics-grid">
      <div class="analytics-item">
        <div class="analytics-label">🏆 Трофеи</div>
        <div class="analytics-value">${data.trophies || 0}</div>
        <div class="analytics-sub">Макс: ${data.max_trophies || 0}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">💎 Gems</div>
        <div class="analytics-value">${data.gems || 0}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">💰 Coins</div>
        <div class="analytics-value">${data.coins || 0}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">👥 Сквад</div>
        <div class="analytics-value">${data.squad_id || "—"}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">📅 Регистрация</div>
        <div class="analytics-value">${data.reg_date ? new Date(data.reg_date).toLocaleDateString("ru-RU") : "—"}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">⚡ ExtraPass</div>
        <div class="analytics-value">${hasExtraPass ? "Активирован" : "Не активирован"}</div>
      </div>
      <div class="analytics-item">
        <div class="analytics-label">📛 Титул</div>
        <div class="analytics-value">${isAdmin ? "Администратор" : (data.title || "Игрок")}</div>
      </div>
    </div>
    
    ${!hasExtraPass ? `
      <div class="extrapass-promo" style="margin-top: 20px;">
        <div class="extrapass-promo-text">⚡ Активируй ExtraPass для эксклюзивных преимуществ!</div>
        <button class="extrapass-promo-btn" id="analytics-extrapass-btn">Купить ExtraPass</button>
      </div>
    ` : ""}
    
    ${isAdmin ? `
      <div class="admin-panel" style="margin-top: 20px; padding: 16px; background: rgba(239, 68, 68, 0.1); border: 2px solid var(--chibi-red); border-radius: 12px;">
        <div class="admin-title" style="font-weight: bold; margin-bottom: 12px; color: var(--chibi-red);">⚙️ Панель администратора</div>
        <button class="btn-primary" id="admin-players-btn" style="width: 100%; margin-bottom: 8px;">Управление игроками</button>
        <button class="btn-secondary" id="admin-stats-btn" style="width: 100%; margin-bottom: 8px;">Статистика бота</button>
        <button class="btn-secondary" id="admin-tps-btn" style="width: 100%;">📊 Производительность (TPS)</button>
      </div>
    ` : ""}
  `;
  
  // Обработчики для админ панели
  if (isAdmin) {
    document.getElementById("admin-players-btn")?.addEventListener("click", async () => {
      const authData = resolveUserId();
      if (!authData) {
        await showGameAlert("Ошибка авторизации", "❌");
        return;
      }
      
      try {
        let url = "/api/admin/players";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Ошибка ${response.status}`);
        }
        const data = await response.json();
        
        // Сохраняем всех игроков для фильтрации
        const allPlayers = data.players;
        
        // Функция для отображения игроков
        const renderPlayers = (players) => {
          const playersList = modal.querySelector(".admin-players-list");
          if (!playersList) return;
          
          if (players.length === 0) {
            playersList.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--chibi-text-muted);">Игроки не найдены</div>';
            return;
          }
          
          playersList.innerHTML = players.map(p => `
            <div class="admin-player-item">
              <div class="player-info-admin">
                <div class="player-name-admin">${p.first_name || p.username || `ID: ${p.user_id}`}</div>
                <div class="player-stats-admin">
                  🏆 ${p.trophies || 0} | ${p.extra_pass === "active" ? "⚡" : ""} | ${p.status || "active"}
                </div>
              </div>
              <div class="player-actions-admin">
                <button class="btn-small" data-action="warn" data-user="${p.user_id}">⚠️</button>
                <button class="btn-small" data-action="ban" data-user="${p.user_id}">🚫</button>
              </div>
            </div>
          `).join('');
          
          // Переустанавливаем обработчики действий
          modal.querySelectorAll("[data-action]").forEach(btn => {
            btn.addEventListener("click", async () => {
              const action = btn.dataset.action;
              const targetUserId = parseInt(btn.dataset.user);
              
              try {
                let url = "/api/admin/players";
                if (typeof authData === "string") {
                  url += `?_auth=${encodeURIComponent(authData)}`;
                } else if (typeof authData === "number") {
                  console.warn("auth: numeric userId unsupported, skipping auth param");
                }
                
                const response = await fetch(url, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ action, user_id: targetUserId })
                });
                
                if (response.ok) {
                  alert(`Действие "${action}" выполнено`);
                  modal.remove();
                }
              } catch (error) {
                console.error("Ошибка:", error);
                alert("Ошибка выполнения действия");
              }
            });
          });
        };
        
        // Показываем модальное окно с игроками
        const modal = document.createElement("div");
        modal.className = "modal-overlay";
        modal.style.display = "flex";
        modal.innerHTML = `
          <div class="modal-content modal-large">
            <div class="modal-header">
              <h2>👥 Управление игроками</h2>
              <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <div class="modal-body">
              <div class="admin-search-container" style="margin-bottom: 16px;">
                <input 
                  type="text" 
                  id="admin-player-search" 
                  class="admin-search-input" 
                  placeholder="🔍 Поиск по ID или нику..."
                  style="width: 100%; padding: 12px; border-radius: 8px; border: 1px solid var(--chibi-border); background: var(--chibi-card); color: var(--chibi-text); font-size: 14px;"
                />
              </div>
              <div class="admin-players-list">
                <!-- Список игроков будет загружен через renderPlayers -->
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(modal);
        
        // Обработчик поиска
        const searchInput = modal.querySelector("#admin-player-search");
        if (searchInput) {
          searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            
            if (!query) {
              // Показываем первые 20 игроков, если поиск пустой
              renderPlayers(allPlayers.slice(0, 20));
              return;
            }
            
            // Фильтруем игроков по ID или нику
            const filtered = allPlayers.filter(p => {
              const userId = String(p.user_id);
              const firstName = (p.first_name || "").toLowerCase();
              const username = (p.username || "").toLowerCase();
              
              return userId.includes(query) || 
                     firstName.includes(query) || 
                     username.includes(query);
            });
            
            renderPlayers(filtered);
          });
        }
        
        // Инициализируем обработчики для начального списка
        renderPlayers(allPlayers.slice(0, 20));
        
        // Закрытие по клику вне модального окна
        modal.addEventListener("click", (e) => {
          if (e.target === modal) {
            modal.remove();
          }
        });
      } catch (error) {
        console.error("Ошибка загрузки игроков:", error);
        await showGameAlert("Не удалось загрузить список игроков", "❌");
      }
    });
    
    document.getElementById("admin-tps-btn")?.addEventListener("click", async () => {
      const authData = resolveUserId();
      if (!authData) {
        await showGameAlert("Ошибка авторизации", "❌");
        return;
      }
      
      try {
        let url = "/api/admin/tps";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Ошибка ${response.status}`);
        }
        
        const stats = await response.json();
        
        // Определяем цвет статуса
        let statusColor = "#10b981"; // зеленый
        if (stats.average_tps_1m < 15.0) {
          statusColor = "#ef4444"; // красный
        } else if (stats.average_tps_1m < 18.0) {
          statusColor = "#f59e0b"; // оранжевый
        } else if (stats.average_tps_1m < 19.0) {
          statusColor = "#eab308"; // желтый
        }
        
        // Форматируем время работы
        const uptime = stats.uptime_formatted || "0:00:00";
        
        // Создаем модальное окно
        const modal = document.createElement("div");
        modal.className = "modal-overlay";
        modal.style.display = "flex";
        modal.innerHTML = `
          <div class="modal-content" style="max-width: 500px;">
            <div class="modal-header">
              <h2>📊 Производительность сервера (TPS)</h2>
              <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <div class="modal-body">
              <div style="margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                  <span style="font-size: 24px;">${stats.status_emoji}</span>
                  <div>
                    <div style="font-weight: bold; font-size: 18px; color: ${statusColor};">Статус: ${stats.status}</div>
                    <div style="font-size: 12px; color: var(--chibi-text-muted);">Идеальное значение: 20.0 TPS</div>
                  </div>
                </div>
              </div>
              
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                <div style="padding: 12px; background: rgba(192, 132, 252, 0.1); border-radius: 8px; border: 1px solid rgba(192, 132, 252, 0.3);">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Текущий TPS</div>
                  <div style="font-size: 24px; font-weight: bold; color: var(--chibi-purple);">${stats.current_tps}</div>
                </div>
                <div style="padding: 12px; background: rgba(96, 165, 250, 0.1); border-radius: 8px; border: 1px solid rgba(96, 165, 250, 0.3);">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Средний (1 мин)</div>
                  <div style="font-size: 24px; font-weight: bold; color: var(--chibi-blue);">${stats.average_tps_1m}</div>
                </div>
                <div style="padding: 12px; background: rgba(34, 211, 153, 0.1); border-radius: 8px; border: 1px solid rgba(34, 211, 153, 0.3);">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Средний (5 мин)</div>
                  <div style="font-size: 24px; font-weight: bold; color: #22d3a5;">${stats.average_tps_5m}</div>
                </div>
                <div style="padding: 12px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Мин. (1 мин)</div>
                  <div style="font-size: 24px; font-weight: bold; color: var(--chibi-red);">${stats.min_tps_1m}</div>
                </div>
              </div>
              
              <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px; margin-bottom: 12px;">
                <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Макс. TPS (1 мин)</div>
                <div style="font-size: 20px; font-weight: bold;">${stats.max_tps_1m}</div>
              </div>
              
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px;">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Всего тиков</div>
                  <div style="font-size: 18px; font-weight: bold;">${stats.total_ticks.toLocaleString()}</div>
                </div>
                <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px;">
                  <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Время работы</div>
                  <div style="font-size: 18px; font-weight: bold;">${uptime}</div>
                </div>
              </div>
              
              <div style="padding: 12px; background: rgba(192, 132, 252, 0.05); border-radius: 8px; border: 1px solid rgba(192, 132, 252, 0.2);">
                <div style="font-size: 11px; color: var(--chibi-text-muted); line-height: 1.5;">
                  <strong>TPS (Ticks Per Second)</strong> - метрика производительности сервера, показывающая количество циклов обработки в секунду. Идеальное значение - 20.0 TPS. Значения ниже 15.0 указывают на проблемы с производительностью.
                </div>
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(modal);
        
        // Закрытие по клику вне модального окна
        modal.addEventListener("click", (e) => {
          if (e.target === modal) {
            modal.remove();
          }
        });
        
        // Автообновление каждые 5 секунд
        const updateInterval = setInterval(async () => {
          try {
            const response = await fetch(url);
            if (response.ok) {
              const newStats = await response.json();
              
              // Обновляем значения в модальном окне
              const statusEl = modal.querySelector(".modal-body");
              if (statusEl) {
                let statusColor = "#10b981";
                if (newStats.average_tps_1m < 15.0) {
                  statusColor = "#ef4444";
                } else if (newStats.average_tps_1m < 18.0) {
                  statusColor = "#f59e0b";
                } else if (newStats.average_tps_1m < 19.0) {
                  statusColor = "#eab308";
                }
                
                statusEl.innerHTML = `
                  <div style="margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                      <span style="font-size: 24px;">${newStats.status_emoji}</span>
                      <div>
                        <div style="font-weight: bold; font-size: 18px; color: ${statusColor};">Статус: ${newStats.status}</div>
                        <div style="font-size: 12px; color: var(--chibi-text-muted);">Идеальное значение: 20.0 TPS</div>
                      </div>
                    </div>
                  </div>
                  
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                    <div style="padding: 12px; background: rgba(192, 132, 252, 0.1); border-radius: 8px; border: 1px solid rgba(192, 132, 252, 0.3);">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Текущий TPS</div>
                      <div style="font-size: 24px; font-weight: bold; color: var(--chibi-purple);">${newStats.current_tps}</div>
                    </div>
                    <div style="padding: 12px; background: rgba(96, 165, 250, 0.1); border-radius: 8px; border: 1px solid rgba(96, 165, 250, 0.3);">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Средний (1 мин)</div>
                      <div style="font-size: 24px; font-weight: bold; color: var(--chibi-blue);">${newStats.average_tps_1m}</div>
                    </div>
                    <div style="padding: 12px; background: rgba(34, 211, 153, 0.1); border-radius: 8px; border: 1px solid rgba(34, 211, 153, 0.3);">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Средний (5 мин)</div>
                      <div style="font-size: 24px; font-weight: bold; color: #22d3a5;">${newStats.average_tps_5m}</div>
                    </div>
                    <div style="padding: 12px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Мин. (1 мин)</div>
                      <div style="font-size: 24px; font-weight: bold; color: var(--chibi-red);">${newStats.min_tps_1m}</div>
                    </div>
                  </div>
                  
                  <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px; margin-bottom: 12px;">
                    <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Макс. TPS (1 мин)</div>
                    <div style="font-size: 20px; font-weight: bold;">${newStats.max_tps_1m}</div>
                  </div>
                  
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                    <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px;">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Всего тиков</div>
                      <div style="font-size: 18px; font-weight: bold;">${newStats.total_ticks.toLocaleString()}</div>
                    </div>
                    <div style="padding: 12px; background: rgba(45, 27, 78, 0.3); border-radius: 8px;">
                      <div style="font-size: 12px; color: var(--chibi-text-muted); margin-bottom: 4px;">Время работы</div>
                      <div style="font-size: 18px; font-weight: bold;">${newStats.uptime_formatted}</div>
                    </div>
                  </div>
                  
                  <div style="padding: 12px; background: rgba(192, 132, 252, 0.05); border-radius: 8px; border: 1px solid rgba(192, 132, 252, 0.2);">
                    <div style="font-size: 11px; color: var(--chibi-text-muted); line-height: 1.5;">
                      <strong>TPS (Ticks Per Second)</strong> - метрика производительности сервера, показывающая количество циклов обработки в секунду. Идеальное значение - 20.0 TPS. Значения ниже 15.0 указывают на проблемы с производительностью.
                    </div>
                  </div>
                `;
              }
            }
          } catch (error) {
            console.error("Ошибка обновления TPS:", error);
          }
        }, 5000);
        
        // Очищаем интервал при закрытии модального окна
        modal.addEventListener("click", (e) => {
          if (e.target === modal || e.target.closest(".modal-close")) {
            clearInterval(updateInterval);
          }
        });
      } catch (error) {
        console.error("Ошибка загрузки TPS:", error);
        await showGameAlert("Не удалось загрузить статистику TPS", "❌");
      }
    });
    
    document.getElementById("admin-stats-btn")?.addEventListener("click", async () => {
      const authData = resolveUserId();
      if (!authData) {
        await showGameAlert("Ошибка авторизации", "❌");
        return;
      }
      
      try {
        let url = "/api/admin/stats";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Ошибка ${response.status}`);
        }
        const data = await response.json();
        
        // Показываем красивое модальное окно со статистикой
        const modal = document.createElement("div");
        modal.className = "modal-overlay";
        modal.style.display = "flex";
        modal.innerHTML = `
          <div class="modal-content">
            <div class="modal-header">
              <h2>📊 Статистика бота</h2>
              <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <div class="modal-body">
              <div class="admin-stats-grid">
                <div class="admin-stat-card">
                  <div class="admin-stat-icon">👥</div>
                  <div class="admin-stat-label">Игроков</div>
                  <div class="admin-stat-value">${data.players || 0}</div>
                </div>
                <div class="admin-stat-card">
                  <div class="admin-stat-icon">⚡</div>
                  <div class="admin-stat-label">ExtraPass</div>
                  <div class="admin-stat-value">${data.extra_pass_active || 0}</div>
                </div>
                <div class="admin-stat-card">
                  <div class="admin-stat-icon">🏆</div>
                  <div class="admin-stat-label">Всего трофеев</div>
                  <div class="admin-stat-value">${data.total_trophies || 0}</div>
                </div>
                <div class="admin-stat-card">
                  <div class="admin-stat-icon">⭐</div>
                  <div class="admin-stat-label">Макс. трофеев</div>
                  <div class="admin-stat-value">${data.max_trophies_global || 0}</div>
                </div>
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(modal);
        
        // Закрытие по клику вне модального окна
        modal.addEventListener("click", (e) => {
          if (e.target === modal) {
            modal.remove();
          }
        });
      } catch (error) {
        console.error("Ошибка загрузки статистики:", error);
        await showGameAlert("Не удалось загрузить статистику", "❌");
      }
    });
  }
  
  // Обработчик кнопки ExtraPass в аналитике
  document.getElementById("analytics-extrapass-btn")?.addEventListener("click", () => {
    alert("Магазин ExtraPass скоро будет доступен!");
  });
}

// Управление модальными окнами
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "flex";
    if (tg?.HapticFeedback) {
      tg.HapticFeedback.impactOccurred("light");
    }
    
    // Сброс состояния модального окна боя при открытии
    if (modalId === "battle-mode-modal") {
      // Скрываем секции выбора колоды и футер
      const deckSelection = document.getElementById("deck-selection");
      const battlePlayFooter = document.getElementById("battle-play-footer");
      
      if (deckSelection) {
        deckSelection.style.display = "none";
      }
      if (battlePlayFooter) {
        battlePlayFooter.style.display = "none";
      }
      
      // Снимаем выделение со всех режимов
      document.querySelectorAll(".battle-mode-item").forEach((item) => {
        item.classList.remove("selected");
        const radio = item.querySelector('input[type="radio"]');
        if (radio) radio.checked = false;
      });
      
      // Сбрасываем выбор
      lastBattleSelection.mode = null;
      lastBattleSelection.deck = null;
      lastBattleSelection.matchType = 'ranked';
      lastBattleSelection.difficulty = null;

      // Скрываем выбор сложности
      const difficultySelection = document.getElementById("difficulty-selection");
      if (difficultySelection) {
        difficultySelection.style.display = "none";
      }
      
      // Сбрасываем выделение сложности
      document.querySelectorAll(".difficulty-item").forEach((d) => {
        const card = d.querySelector(".difficulty-card");
        if (card) {
          card.style.borderColor = "rgba(255,255,255,0.1)";
          card.style.background = "transparent";
        }
        const radio = d.querySelector('input[type="radio"]');
        if (radio) radio.checked = false;
      });
    }
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "none";
    if (tg?.HapticFeedback) {
      tg.HapticFeedback.impactOccurred("light");
    }
  }
}

// Инициализация обработчиков событий (работает даже без авторизации)
function initEventHandlers() {
  // Глобальный обработчик кликов для воспроизведения звука кнопок
  // Используем делегирование событий для работы со всеми кнопками, включая динамически созданные
  document.addEventListener("click", (e) => {
    const target = e.target;
    
    // Пропускаем клики на интерактивные элементы ввода
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable) {
      return;
    }
    
    // Пропускаем клики на overlay (фон модальных окон)
    if (target.classList.contains("modal-overlay") || target.classList.contains("game-alert-modal") || target.classList.contains("game-confirm-modal") || target.classList.contains("dice-notification-prompt-overlay") || target.classList.contains("welcome-modal-overlay")) {
      return;
    }
    
    // Проверяем, является ли элемент кнопкой или находится внутри кнопки
    const button = target.closest("button, [role='button']");
    
    // Также проверяем элементы с классами кнопок
    if (!button) {
      const buttonLike = target.closest(".btn-primary, .btn-secondary, .menu-item, .nav-item, .shop-category, .collection-tab, .community-tab, .friends-tab, .mail-filter-btn, .item-buy-btn, .card-item, .deck-slot, .case-open-btn, .case-skip-btn, .case-close-btn, .dice-modal-btn, .friend-action-btn, .friend-request-btn, .friend-add-btn, .chat-send-btn, .promocode-submit-btn, .post-submit-btn, .card-submit-btn, .welcome-btn, .battle-mode-item, .deck-item, .training-mode-item, .difficulty-item, .training-option, .play-battle-btn, .play-training-btn, .energy-reset-btn, .copy-btn, .info-btn, .arena-info-btn, .target-btn, .battle-btn, .cases-action-btn, .cases-shop-btn, .cases-open-btn, .deck-save-btn, .preset-create-btn, .sort-direction-btn, .filter-toggle-btn, .rarity-filter, .sort-option, .admin-create-card-btn, .admin-get-all-cards-btn, .admin-delete-all-cards-btn, .create-preset-btn, .create-post-btn, .chat-fullscreen-btn, .chat-fullscreen-close-btn, .dice-notification-prompt-btn");
      if (buttonLike) {
        // Исключаем кнопки закрытия модальных окон и кнопку "В БОЙ" (у неё свой звук)
        const isStartBattleBtn = buttonLike.id === "start-battle";
        if (!buttonLike.classList.contains("modal-close") && !buttonLike.classList.contains("game-alert-close") && !buttonLike.classList.contains("game-confirm-close") && !buttonLike.disabled && !isStartBattleBtn) {
          playButtonSound();
        }
        return;
      }
    }
    
    if (button && !button.disabled) {
      // Исключаем кнопки закрытия модальных окон и кнопку "В БОЙ" (у неё свой звук)
      const isStartBattleBtn = button.id === "start-battle";
      if (!button.classList.contains("modal-close") && !button.classList.contains("game-alert-close") && !button.classList.contains("game-confirm-close") && !isStartBattleBtn) {
        playButtonSound();
      }
    }
  }, true); // Используем capture phase для раннего перехвата
  console.log("Инициализация обработчиков событий...");
  
  // Клик на аватарку - открывает аналитику
  const playerAvatar = document.getElementById("player-avatar");
  if (playerAvatar) {
    playerAvatar.addEventListener("click", () => {
      console.log("Клик на аватарку");
      if (currentProfile) {
        renderAnalytics(currentProfile);
        openModal("analytics-modal");
      } else {
        console.warn("Профиль еще не загружен");
      }
    });
  }

  // Клик на трофеи - открывает "Путь славы"
  const arenaTrophies = document.querySelector(".arena-trophies");
  if (arenaTrophies) {
    arenaTrophies.style.cursor = "pointer";
    arenaTrophies.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Клик на трофеи - открываем Путь славы");
      if (currentProfile) {
        renderGloryPath(currentProfile);
        openModal("glory-path-modal");
      } else {
        console.warn("Профиль еще не загружен");
      }
    });
  } else {
    console.error("player-avatar не найден!");
  }

  // Клик на щит арены - визуальный эффект
  const arenaShield = document.querySelector(".arena-shield");
  if (arenaShield) {
    arenaShield.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Клик на щит арены");
      
      // Добавляем класс для анимации
      arenaShield.classList.add("shield-clicked");
      
      // Создаем частицы эффекта
      const shieldIcon = arenaShield.querySelector(".shield-icon");
      if (shieldIcon) {
        for (let i = 0; i < 8; i++) {
          const particle = document.createElement("div");
          particle.className = "shield-particle";
          particle.style.cssText = `
            position: absolute;
            width: 6px;
            height: 6px;
            background: var(--chibi-gold);
            border-radius: 50%;
            pointer-events: none;
            z-index: 100;
          `;
          arenaShield.appendChild(particle);
          
          const angle = (i / 8) * Math.PI * 2;
          const distance = 40;
          const x = Math.cos(angle) * distance;
          const y = Math.sin(angle) * distance;
          
          particle.style.left = "50%";
          particle.style.top = "50%";
          particle.style.transform = `translate(-50%, -50%)`;
          
          setTimeout(() => {
            particle.style.transition = "all 0.6s ease-out";
            particle.style.transform = `translate(${x}px, ${y}px)`;
            particle.style.opacity = "0";
            setTimeout(() => particle.remove(), 600);
          }, 10);
        }
      }
      
      // Убираем класс через время
      setTimeout(() => {
        arenaShield.classList.remove("shield-clicked");
      }, 300);
      
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("medium");
        }
      } catch (e) {}
    });
  }

  // Меню
  const menuButton = document.getElementById("menu-button");
  const menuOverlay = document.getElementById("menu-overlay");
  const menuClose = document.getElementById("menu-close");

  if (menuButton) {
    menuButton.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Кнопка меню нажата");
      if (menuOverlay) {
        menuOverlay.style.display = "flex";
      }
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {
        // Игнорируем ошибки HapticFeedback
      }
    });
  } else {
    console.error("Кнопка menu-button не найдена!");
  }

  if (menuClose) {
    menuClose.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Закрытие меню");
      if (menuOverlay) {
        menuOverlay.style.display = "none";
      }
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {
        // Игнорируем ошибки HapticFeedback
      }
    });
  }

  // Пункты меню
  const menuItems = document.querySelectorAll(".menu-item");
  console.log("Найдено пунктов меню:", menuItems.length);
  menuItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const menuType = item.dataset.menu;
      console.log("Пункт меню нажат:", menuType);
      if (menuOverlay) {
        menuOverlay.style.display = "none";
      }

      if (menuType === "settings") {
        openModal("settings-modal");
      } else if (menuType === "mail") {
        openModal("mail-modal");
        // Загружаем почту при открытии
        const authData = resolveUserId();
        if (authData) {
          // Загружаем почту и автоматически помечаем все непрочитанные письма как прочитанные
          loadMailAndMarkAsRead(authData);
        }
      } else if (menuType === "support") {
        openModal("support-modal");
      } else if (menuType === "info") {
        openModal("info-modal");
      } else if (menuType === "friends") {
        openModal("friends-modal");
        // Инициализируем табы друзей
        initFriendsTabs();
      }
      
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {
        // Игнорируем ошибки HapticFeedback
      }
    });
  });

  // Закрытие модальных окон
  document.querySelectorAll(".modal-close").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const modalId = btn.dataset.modal;
      console.log("Закрытие модального окна:", modalId);
      
      if (modalId === "item-detail") {
        closeItemDetailModal();
        return;
      }
      if (modalId) {
        closeModal(`${modalId}-modal`);
      }
    });
  });

  // Клик вне модального окна для закрытия
  document.querySelectorAll(".modal-overlay").forEach((overlay) => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) {
        console.log("Клик вне модального окна");
        overlay.style.display = "none";
      }
    });
  });
  
  // Клик вне меню для закрытия (menuOverlay уже объявлен выше)
  if (menuOverlay) {
    menuOverlay.addEventListener("click", (e) => {
      if (e.target === menuOverlay) {
        console.log("Клик вне меню");
        menuOverlay.style.display = "none";
      }
    });
  }

  // Кнопка "По умолчанию" в настройках
  const resetSettingsBtn = document.getElementById("reset-settings-btn");
  if (resetSettingsBtn) {
    resetSettingsBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Сброс настроек");
      const defaultSettings = getDefaultSettings();
      const authData = resolveUserId();
      if (authData) {
        try {
          await saveSettings(authData, defaultSettings);
          renderSettings(defaultSettings);
          if (tg?.HapticFeedback) {
            tg.HapticFeedback.impactOccurred("medium");
          }
        } catch (error) {
          console.error("Ошибка сброса настроек:", error);
        }
      } else {
        console.warn("Нет авторизации для сброса настроек");
      }
    });
  }

  // Обработчики для инфо-кнопок
  document.addEventListener("click", (e) => {
    if (e.target.closest(".info-btn")) {
      const btn = e.target.closest(".info-btn");
      e.preventDefault();
      e.stopPropagation();
      const infoType = btn.dataset.info;
      
      const modal = document.getElementById("info-tooltip-modal");
      const title = document.getElementById("info-tooltip-title");
      const content = document.getElementById("info-tooltip-content");
      
      if (modal && title && content) {
        if (infoType === "arena") {
          title.textContent = "🏛️ Арена";
          content.innerHTML = `
            <p><strong>Арена</strong> - это место, где вы сражаетесь с другими игроками и зарабатываете трофеи.</p>
            <p>Чем больше трофеев, тем выше ваша лига — от Novice до Extra!</p>
            <p>Каждая победа приносит трофеи, а поражение их отнимает.</p>
          `;
        } else if (infoType === "energy") {
          title.textContent = "⚡ Энергия";
          content.innerHTML = `
            <p><strong>Энергия</strong> - это ресурс, необходимый для участия в боях.</p>
            <p>• Обычные игроки имеют 5 единиц энергии</p>
            <p>• С ExtraPass: 6 единиц энергии</p>
            <p>• Энергия восстанавливается со временем</p>
            <p>• Можно сбросить кулдаун за 5 💎</p>
            <p>• В тренировке энергия не тратится</p>
          `;
        }
        
        modal.style.display = "flex";
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
      }
    }
  });

  // Обработчики для ресурсов
  document.addEventListener("click", (e) => {
    if (e.target.closest(".resource-item")) {
      const item = e.target.closest(".resource-item");
      e.preventDefault();
      e.stopPropagation();
      const resourceType = item.dataset.resource;
      
      const modal = document.getElementById("resource-info-modal");
      const title = document.getElementById("resource-info-title");
      const content = document.getElementById("resource-info-content");
      
      if (modal && title && content) {
        if (resourceType === "gems") {
          title.textContent = "💎 Gems";
          content.innerHTML = `
            <p><strong>Gems (Гемы)</strong> - это премиальная валюта игры.</p>
            <p>Используются для:</p>
            <ul style="text-align: left; margin: 10px 0;">
              <li>Покупки карт в магазине</li>
              <li>Ускорения открытия кейсов</li>
              <li>Сброса кулдауна энергии</li>
              <li>Приобретения эксклюзивных предметов</li>
            </ul>
            <p>Получить гемы можно через покупки, достижения и награды за события.</p>
          `;
        } else if (resourceType === "coins") {
          title.textContent = "💰 Coins";
          content.innerHTML = `
            <p><strong>Coins (Монеты)</strong> - основная валюта игры.</p>
            <p>Используются для:</p>
            <ul style="text-align: left; margin: 10px 0;">
              <li>Улучшения карт</li>
              <li>Покупки базовых предметов</li>
              <li>Создания новых колод</li>
              <li>Обновления карт</li>
            </ul>
            <p>Монеты можно заработать в боях, открывая кейсы и выполняя ежедневные задания.</p>
          `;
        }
        
        modal.style.display = "flex";
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
      }
    }
  });

  // Инициализация табов друзей
  function initFriendsTabs() {
    const tabs = document.querySelectorAll(".friends-tab");
    const tabContents = document.querySelectorAll(".friends-tab-content");
    
    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const targetTab = tab.dataset.tab;
        
        // Убираем активный класс со всех табов
        tabs.forEach(t => t.classList.remove("active"));
        tabContents.forEach(content => content.style.display = "none");
        
        // Активируем выбранный таб
        tab.classList.add("active");
        const targetContent = document.getElementById(`${targetTab}-tab-content`);
        if (targetContent) {
          targetContent.style.display = "block";
        }
        
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
      });
    });
    
    // Обработчики для кнопок друзей (заглушки)
    document.querySelectorAll(".friend-action-btn, .friend-request-btn, .friend-add-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
        // Заглушка - просто визуальная обратная связь
        btn.style.transform = "scale(0.95)";
        setTimeout(() => {
          btn.style.transform = "";
        }, 100);
      });
    });
    
    // Обработчики для писем (заглушки)
    document.querySelectorAll(".mail-item").forEach(item => {
      item.addEventListener("click", () => {
        // Убираем статус "непрочитано" при клике
        item.classList.remove("unread");
        const unreadDot = item.querySelector(".unread-dot");
        if (unreadDot) {
          unreadDot.style.display = "none";
        }
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
      });
    });
    
    // Обработчики для фильтров почты (будут переустановлены в renderMail)
    // Обработчики добавляются динамически в renderMail после рендеринга писем
  }

  // Копирование ID и версии
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const copyType = btn.dataset.copy;
      console.log("Копирование:", copyType);
      let textToCopy = "";
      if (copyType === "user-id") {
        textToCopy = currentProfile?.user_id || "";
      } else if (copyType === "version") {
        textToCopy = "1.0.0";
      }

      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          console.log("Скопировано:", textToCopy);
          if (tg?.HapticFeedback) {
            tg.HapticFeedback.impactOccurred("light");
          }
        });
      }
    });
  });

  // Навигация по разделам
  const navItems = document.querySelectorAll(".nav-item");
  console.log("Найдено элементов навигации:", navItems.length);
  navItems.forEach((item, index) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const sectionId = item.dataset.section;
      console.log(`Нажата кнопка навигации: ${sectionId}`);
      if (!sectionId) {
        console.warn("У элемента навигации нет data-section");
        return;
      }

      // Анимация перехода
      const currentSection = document.querySelector(".section.active");
      if (currentSection) {
        currentSection.classList.add("slide-out");
        setTimeout(() => {
          currentSection.classList.remove("active", "slide-out");
        }, 300);
      }

      // Убираем активный класс со всех разделов и кнопок
      document.querySelectorAll(".section").forEach((section) => {
        section.classList.remove("active");
      });
      document.querySelectorAll(".nav-item").forEach((navItem) => {
        navItem.classList.remove("active");
      });

      // Активируем выбранный раздел с анимацией
      setTimeout(() => {
        const targetSection = document.getElementById(`${sectionId}-section`);
        if (targetSection) {
          targetSection.classList.add("active");
          console.log(`Раздел ${sectionId} активирован`);
          
          // Загружаем карты и колоды при открытии коллекции
          if (sectionId === "collection") {
            initCollection();
          }
          
          // Инициализируем раздел коммьюнити
          if (sectionId === "community") {
            initCommunity();
          }
        } else {
          console.error(`Раздел ${sectionId}-section не найден!`);
        }
        item.classList.add("active");

        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {
          // Игнорируем ошибки HapticFeedback
        }
      }, 0);
    });
  });

  // Кнопки арены
  const startBattleBtn = document.getElementById("start-battle");
  if (startBattleBtn) {
    startBattleBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Кнопка 'В БОЙ' нажата");
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("medium");
        }
      } catch (e) {}
      // Воспроизводим звук начала боя
      playBattleModeSound();
      // Открываем модальное окно выбора режима боя и колоды
      openModal("battle-mode-modal");
    });
  } else {
    console.error("Кнопка start-battle не найдена!");
  }

  // Кнопка "В магазин" в хранилище кейсов
  const casesShopBtn = document.getElementById("cases-shop-btn");
  if (casesShopBtn) {
    casesShopBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Кнопка 'В магазин' нажата");
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {}
      
      // Находим кнопку навигации "Магазин" и кликаем на неё
      const shopNavItem = document.querySelector('.nav-item[data-section="shop"]');
      if (shopNavItem) {
        shopNavItem.click();
      } else {
        console.error("Кнопка навигации 'Магазин' не найдена!");
      }
    });
  }

  const casesOpenBtn = document.getElementById("cases-open-btn");
  if (casesOpenBtn) {
    casesOpenBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("medium");
        }
      } catch (err) {
        console.warn(err);
      }
      openCasesShortcut();
    });
  }

  const trainingBattleBtn = document.getElementById("training-battle");
  if (trainingBattleBtn) {
    trainingBattleBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Кнопка 'Тренировка' нажата");
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {}
      openModal("training-modal");
    });
  } else {
    console.error("Кнопка training-battle не найдена!");
  }
  
  // Обработчики режимов боя (добавляются динамически после загрузки)
  setTimeout(() => {
    document.querySelectorAll(".battle-mode-item").forEach((item) => {
      item.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const mode = item.dataset.mode;
        const matchType = item.dataset.matchType || 'ranked';
        console.log("Выбран режим боя:", mode, "Тип:", matchType);
        
        // Сохраняем выбранный режим и тип
        lastBattleSelection.mode = mode;
        lastBattleSelection.matchType = matchType;
        
        // Снимаем выделение со всех режимов
        document.querySelectorAll(".battle-mode-item").forEach((m) => {
          m.classList.remove("selected");
          const radio = m.querySelector('input[type="radio"]');
          if (radio) radio.checked = false;
        });
        
        // Выделяем выбранный режим
        item.classList.add("selected");
        const radio = item.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
        
        // Показываем/скрываем выбор сложности
        const difficultySelection = document.getElementById("difficulty-selection");
        if (difficultySelection) {
          difficultySelection.style.display = (matchType === 'training') ? "block" : "none";
        }
        
        // Загружаем и показываем выбор колоды
        const deckSelection = document.getElementById("deck-selection");
        const battlePlayFooter = document.getElementById("battle-play-footer");
        
        if (deckSelection) {
          deckSelection.style.display = "block";
          
          // Загружаем колоды
          const presets = await loadBattleDecks();
          renderBattleDecks(presets);
          
          // Проверяем готовность к бою
          checkBattleReady();
        }
        
        if (battlePlayFooter) {
          battlePlayFooter.style.display = "block";
        }
        
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
      });
    });

    // Обработчик выбора сложности
    document.querySelectorAll(".difficulty-item").forEach((item) => {
      item.addEventListener("click", () => {
        const difficulty = item.dataset.difficulty;
        console.log("Выбрана сложность:", difficulty);
        lastBattleSelection.difficulty = difficulty;
        
        // Снимаем выделение со всех уровней сложности
        document.querySelectorAll(".difficulty-item").forEach((d) => {
          const card = d.querySelector(".difficulty-card");
          if (card) {
            card.style.borderColor = "rgba(255,255,255,0.1)";
            card.style.background = "transparent";
          }
          const radio = d.querySelector('input[type="radio"]');
          if (radio) radio.checked = false;
        });
        
        // Выделяем выбранный уровень
        const selectedCard = item.querySelector(".difficulty-card");
        if (selectedCard) {
          selectedCard.style.borderColor = "var(--chibi-accent)";
          selectedCard.style.background = "rgba(255, 107, 0, 0.1)";
        }
        const radio = item.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
        
        checkBattleReady();
      });
    });
    
    // Делегированный обработчик для выбора колоды (динамические элементы)
    document.getElementById("decks-list")?.addEventListener("change", (e) => {
      if (e.target.name === "battle-deck") {
        const deckId = parseInt(e.target.value);
        console.log("Выбрана колода:", deckId);
        lastBattleSelection.deck = deckId;
        
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
        
        checkBattleReady();
      }
    });
    
    // Обработчик кнопки "ИГРАТЬ"
    document.getElementById("play-battle-btn")?.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      const { mode, deck, matchType, difficulty } = lastBattleSelection;
      
      if (!mode) {
        await showGameAlert("Выберите режим боя", "⚠️");
        return;
      }
      
      if (!deck) {
        await showGameAlert("Выберите колоду для боя", "⚠️");
        return;
      }

      if (matchType === 'training' && !difficulty) {
        await showGameAlert("Выберите сложность для тренировки", "⚠️");
        return;
      }
      
      console.log("Запуск боя:", { mode, deck, matchType, difficulty });
      
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("medium");
        }
      } catch (e) {}
      
      // Закрываем модальное окно
      closeModal("battle-mode-modal");
      
      // Запускаем матчмейкинг с выбранной колодой и параметрами
      await startMatchmaking(deck, mode, matchType, difficulty);
    });
    
    // Функция проверки готовности к тренировке
    function checkTrainingReady() {
      const playBtn = document.getElementById("play-training-btn");
      const footer = document.getElementById("training-play-footer");
      
      const friendsSelected = document.getElementById("training-friends")?.classList.contains("selected");
      const botSelected = document.getElementById("training-bot")?.classList.contains("selected");
      
      let ready = false;
      
      if (friendsSelected) {
        const modeSelected = document.querySelector('input[name="training-mode"]:checked');
        // Для друзей пока не требуется выбор друга (заглушка)
        ready = !!modeSelected;
      } else if (botSelected) {
        const difficultySelected = document.querySelector('input[name="difficulty"]:checked');
        ready = !!difficultySelected;
      }
      
      if (ready && playBtn && footer) {
        playBtn.disabled = false;
        footer.style.display = "block";
      } else if (footer) {
        footer.style.display = "none";
        if (playBtn) playBtn.disabled = true;
      }
    }
    
    // Обработчики выбора типа тренировки
    document.getElementById("training-friends")?.addEventListener("click", () => {
      const friendsOption = document.getElementById("training-friends");
      const botOption = document.getElementById("training-bot");
      const friendsSelection = document.getElementById("friends-mode-selection");
      const botDifficulty = document.getElementById("bot-difficulty");
      const trainingOptions = document.getElementById("training-options");
      
      // Снимаем выделение с бота
      botOption?.classList.remove("selected");
      
      // Выделяем друзей
      friendsOption?.classList.add("selected");
      
      // Показываем выбор режима для друзей
      if (friendsSelection) {
        friendsSelection.style.display = "block";
      }
      
      // Скрываем выбор сложности бота
      if (botDifficulty) {
        botDifficulty.style.display = "none";
      }
      
      // Скрываем опции выбора
      if (trainingOptions) {
        trainingOptions.style.display = "none";
      }
      
      checkTrainingReady();
    });
    
    document.getElementById("training-bot")?.addEventListener("click", () => {
      const friendsOption = document.getElementById("training-friends");
      const botOption = document.getElementById("training-bot");
      const friendsSelection = document.getElementById("friends-mode-selection");
      const botDifficulty = document.getElementById("bot-difficulty");
      const trainingOptions = document.getElementById("training-options");
      
      // Снимаем выделение с друзей
      friendsOption?.classList.remove("selected");
      
      // Выделяем бота
      botOption?.classList.add("selected");
      
      // Показываем выбор сложности бота
      if (botDifficulty) {
        botDifficulty.style.display = "block";
      }
      
      // Скрываем выбор режима для друзей
      if (friendsSelection) {
        friendsSelection.style.display = "none";
      }
      
      // Скрываем опции выбора
      if (trainingOptions) {
        trainingOptions.style.display = "none";
      }
      
      checkTrainingReady();
    });
    
    // Обработчики режимов для друзей
    document.addEventListener("change", (e) => {
      if (e.target.classList.contains("training-mode-radio")) {
        const mode = e.target.value;
        console.log("Выбран режим для друзей:", mode);
        const friendsSelection = document.getElementById("friends-selection");
        if (friendsSelection) {
          friendsSelection.style.display = "block";
        }
        checkTrainingReady();
      }
    });
    
    // Обработчики сложности бота
    document.addEventListener("change", (e) => {
      if (e.target.classList.contains("difficulty-radio")) {
        const difficulty = e.target.value;
        console.log("Выбрана сложность:", difficulty);
        lastBattleSelection.difficulty = difficulty;
        lastBattleSelection.matchType = 'training';
        checkTrainingReady();
      }
    });
    
    // Кнопка "Играть" для тренировки
    document.getElementById("play-training-btn")?.addEventListener("click", () => {
      const friendsSelected = document.getElementById("training-friends")?.classList.contains("selected");
      const botSelected = document.getElementById("training-bot")?.classList.contains("selected");
      
      if (friendsSelected) {
        const mode = document.querySelector('input[name="training-mode"]:checked')?.value;
        console.log("Тренировка с другом:", mode);
        closeModal("training-modal");
        alert(`Тренировка с другом (${mode}) скоро будет доступна!`);
      } else if (botSelected) {
        const difficulty = document.querySelector('input[name="difficulty"]:checked')?.value;
        console.log("Тренировка с ботом:", difficulty);
        closeModal("training-modal");
        alert(`Тренировка с ботом (${difficulty}) скоро будет доступна!`);
      }
    });
  }, 100);
  
  // Обработчик премиум слота
  document.getElementById("premium-slot")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const slot = e.currentTarget;
    if (slot.classList.contains("locked")) {
      showGameAlert("Требуется ExtraPass для доступа к 5-му слоту!", "⚠️");
      openModal("settings-modal");
    } else {
      console.log("Открыт премиум слот");
    }
  });
  
  // Обработчик подсказки "+1" для энергии
  document.getElementById("energy-extrapass-hint")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    openModal("settings-modal");
    try {
      if (tg?.HapticFeedback?.impactOccurred) {
        tg.HapticFeedback.impactOccurred("light");
      }
    } catch (e) {}
  });
  
  // Свайпы для навигации
  let touchStartX = 0;
  let touchEndX = 0;
  const contentArea = document.querySelector(".content-area");
  
  if (contentArea) {
    contentArea.addEventListener("touchstart", (e) => {
      touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });
    
    contentArea.addEventListener("touchend", (e) => {
      touchEndX = e.changedTouches[0].screenX;
      handleSwipe();
    }, { passive: true });
  }
  
  function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
      const sections = ["shop", "collection", "arena", "squads", "community"];
      const currentSection = document.querySelector(".section.active");
      if (!currentSection) return;
      
      const currentIndex = sections.findIndex(s => currentSection.id === `${s}-section`);
      if (currentIndex === -1) return;
      
      let newIndex;
      if (diff > 0) {
        // Свайп влево - следующий раздел
        newIndex = (currentIndex + 1) % sections.length;
      } else {
        // Свайп вправо - предыдущий раздел
        newIndex = (currentIndex - 1 + sections.length) % sections.length;
      }
      
      const newSection = sections[newIndex];
      const navItem = document.querySelector(`.nav-item[data-section="${newSection}"]`);
      if (navItem) {
        try {
          if (tg?.HapticFeedback?.impactOccurred) {
            tg.HapticFeedback.impactOccurred("light");
          }
        } catch (e) {}
        navItem.click();
      }
    }
  }
  
  // Обработчики кликабельных карточек товаров (для просмотра деталей)
  setTimeout(() => {
    document.querySelectorAll(".shop-item-clickable").forEach((card) => {
      card.addEventListener("click", (e) => {
        // Не открываем модальное окно, если клик был на кнопку купить
        if (e.target.closest(".item-buy-btn")) {
          return;
        }
        
        const itemDetail = card.dataset.itemDetail;
        if (itemDetail) {
          showItemDetailModal(card);
        }
      });
    });
  }, 100);

  // Обработчики кнопок покупки в магазине
  setTimeout(() => {
    document.querySelectorAll(".item-buy-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const itemCard = btn.closest(".shop-item-card");
        if (!itemCard) return;
        
        const itemType = itemCard.dataset.itemType;
        // Убираем пробелы из цены перед парсингом (для "1 790" -> "1790")
        const priceStr = (itemCard.dataset.itemPrice || "0").replace(/\s+/g, "");
        const itemPrice = parseFloat(priceStr);
        const itemCurrency = itemCard.dataset.currency || "rubles"; // rubles или gems
        const itemName = itemCard.querySelector(".item-name")?.textContent || "Товар";
        
        if (!itemType) {
          console.warn("У товара не указан item-type");
          return;
        }
        
        if (itemPrice <= 0) {
          console.warn("У товара не указана цена или она равна 0");
          return;
        }
        
        // Если покупка за гемы, используем специальный API
        if (itemCurrency === "gems") {
          console.log("Покупка товара за гемы:", itemType, "за", itemPrice, "💎");
          buyWithGems(itemType, itemPrice, itemName);
        } else {
          // Покупка за реальные деньги - показываем модальное окно выбора метода оплаты
          console.log("Покупка товара:", itemType, "за", itemPrice, "₽");
          showPaymentMethodModal(itemType, itemPrice, itemName);
        }
      });
    });
  }, 100);
  
  console.log("Все обработчики событий установлены");
}

// Функция для скрытия экрана загрузки
function hideLoadingScreen() {
  console.log("hideLoadingScreen вызвана");
  const ls = document.getElementById("loading-screen");
  const appEl = document.getElementById("app");
  
  if (!ls) {
    console.error("Элемент loading-screen не найден!");
    return;
  }
  
  // Проверяем, не скрыт ли уже экран загрузки
  const currentDisplay = window.getComputedStyle(ls).display;
  if (currentDisplay === "none") {
    console.log("Экран загрузки уже скрыт");
    return;
  }
  
  console.log("Скрываем экран загрузки");
  ls.style.opacity = "0";
  ls.style.transition = "opacity 0.5s ease-out";
  setTimeout(() => {
    ls.style.display = "none";
    console.log("Экран загрузки полностью скрыт");
  }, 500);
  
  if (appEl) {
    appEl.style.display = "flex";
    appEl.style.opacity = "0";
    appEl.style.transition = "opacity 0.5s ease-in";
    setTimeout(() => {
      appEl.style.opacity = "1";
      console.log("Приложение показано");
    }, 100);
  } else {
    console.error("Элемент app не найден!");
  }
}

// Функция для загрузки профиля и настроек
async function loadUserData(authData) {
  try {
    await loadProfile(authData);
    
    try {
      const settings = await loadSettings(authData);
      if (settings) {
        renderSettings(settings);
      }
    } catch (e) {
      console.error("Ошибка загрузки настроек:", e);
    }
    
    return true;
  } catch (error) {
    console.error("Ошибка загрузки профиля:", error);
    return false;
  }
}

// Функция предзагрузки всех данных приложения во время загрузки
async function preloadAppData(authData) {
  const loadingText = document.getElementById("loading-text");
  const loadingPercentage = document.getElementById("loading-percentage");
  const loadingProgressBar = document.getElementById("loading-progress-bar");
  const totalSteps = 4; // Профиль, настройки, карты, кейсы
  let currentStep = 0;
  let hasErrors = false;
  
  const updateProgress = (step, text) => {
    currentStep = step;
    const progress = Math.round((currentStep / totalSteps) * 100);
    if (loadingText && text) {
      loadingText.textContent = text;
    }
    if (loadingPercentage) {
      loadingPercentage.textContent = `${progress}%`;
    }
    if (loadingProgressBar) {
      loadingProgressBar.style.width = `${progress}%`;
    }
  };
  
  // Шаг 1: Загрузка профиля (критично - если не загрузится, вернем false)
  updateProgress(1, "Загрузка профиля...");
  try {
    await loadProfile(authData);
    await new Promise(resolve => setTimeout(resolve, 50)); // Небольшая задержка для плавности
  } catch (error) {
    console.error("Ошибка загрузки профиля:", error);
    hasErrors = true;
    // Профиль критичен, но продолжаем загрузку остальных данных
  }
  
  // Шаг 2: Загрузка настроек
  updateProgress(2, "Загрузка настроек...");
  try {
    const settings = await loadSettings(authData);
    if (settings) {
      renderSettings(settings);
    }
  } catch (e) {
    console.error("Ошибка загрузки настроек:", e);
    hasErrors = true;
  }
  await new Promise(resolve => setTimeout(resolve, 50));
  
  // Шаг 3: Предзагрузка карт пользователя
  updateProgress(3, "Загрузка карт...");
  try {
    await loadUserCards();
  } catch (e) {
    console.error("Ошибка предзагрузки карт:", e);
    hasErrors = true;
  }
  await new Promise(resolve => setTimeout(resolve, 50));
  
  // Шаг 4: Предзагрузка кейсов пользователя
  updateProgress(4, "Загрузка кейсов...");
  try {
    await loadUserCases();
  } catch (e) {
    console.error("Ошибка предзагрузки кейсов:", e);
    hasErrors = true;
  }
  await new Promise(resolve => setTimeout(resolve, 50));
  
  updateProgress(totalSteps, "Готово!");
  // Возвращаем true, даже если были ошибки (кроме критических), чтобы приложение продолжило работу
  return !hasErrors || currentProfile !== null;
}

// Управление фоновой музыкой
let backgroundMusic = null;
let musicEnabled = true;

function initBackgroundMusic() {
  backgroundMusic = document.getElementById("background-music");
  if (backgroundMusic) {
    backgroundMusic.volume = 0.3; // 30% громкости для фоновой музыки
    // Пробуем включить музыку сразу
    const tryPlay = () => {
      if (backgroundMusic && musicEnabled) {
        backgroundMusic.play().catch(e => {
            console.log("Автовоспроизведение музыки заблокировано:", e);
        });
      }
    };
    tryPlay();
    // Также пробуем после первого взаимодействия пользователя
    document.addEventListener("click", tryPlay, { once: true });
    document.addEventListener("touchstart", tryPlay, { once: true });
  }
}

function toggleMusic(enabled) {
  musicEnabled = enabled;
  if (backgroundMusic) {
    if (enabled) {
      backgroundMusic.play().catch(e => {
        console.log("Не удалось включить музыку:", e);
      });
    } else {
      backgroundMusic.pause();
    }
  }
}

// Проверка и показ welcome модального окна
async function checkAndShowWelcome(authData) {
  try {
    let url = "/api/welcome/status";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    } else {
      return;
    }

    const response = await fetch(url);
    if (!response.ok) {
      return;
    }
    
    const data = await response.json();
    if (data.should_show && data.start_card) {
      if (window.__analytics) {
        window.__analytics.onboarding('welcome_seen', false, null, { source: 'welcome_status' });
      }
      // Показываем welcome модальное окно
      showWelcomeModal(data.start_card);
    }
  } catch (error) {
    console.error("Ошибка проверки welcome статуса:", error);
  }
}

// Показать welcome модальное окно
function showWelcomeModal(startCard) {
  const welcomeModal = document.getElementById("welcome-modal");
  const welcomeStepCard = document.getElementById("welcome-step-card");
  const welcomeCardImage = document.getElementById("welcome-card-image");
  const welcomeCardName = document.getElementById("welcome-card-name");
  const welcomeCardRarity = document.getElementById("welcome-card-rarity");
  
  if (!welcomeModal || !welcomeStepCard) {
    console.error("Welcome модальное окно не найдено");
    return;
  }
  
  // Заполняем данные о карте
  if (startCard) {
    if (welcomeCardName) {
      welcomeCardName.textContent = startCard.name || "Стартовая карта";
    }
    if (welcomeCardRarity) {
      const rarityNames = {
        "start": "Стартовая",
        "common": "Обычная",
        "rare": "Редкая",
        "epic": "Эпическая",
        "legendary": "Легендарная",
        "mythic": "Мифическая",
        "divine": "Божественная",
        "limited": "Лимитированная"
      };
      welcomeCardRarity.textContent = rarityNames[startCard.rarity] || "Стартовая";
      welcomeCardRarity.className = `welcome-card-rarity ${startCard.rarity || "start"}`;
    }
    if (welcomeCardImage) {
      // Очищаем предыдущее содержимое
      welcomeCardImage.innerHTML = "";
      
      // Создаем img элемент для правильного отображения изображения
      const img = document.createElement("img");
      img.style.width = "100%";
      img.style.height = "100%";
      img.style.objectFit = "cover";
      img.style.borderRadius = "16px";
      img.style.display = "block";
      
      // Устанавливаем источник изображения
      if (startCard.id) {
        // Используем card_id для получения изображения через API
        img.src = `/api/cards/image?card_id=${startCard.id}`;
        img.alt = startCard.name || "Карта";
      } else if (startCard.image_file_id) {
        // Если есть file_id, используем его
        img.src = `/api/cards/image?file_id=${encodeURIComponent(startCard.image_file_id)}`;
        img.alt = startCard.name || "Карта";
      } else {
        // Если изображения нет, используем заглушку
        img.src = "/DesignAssets/Cases/Case.png";
        img.alt = "Заглушка";
      }
      
      // Обработка ошибки загрузки изображения
      img.onerror = function() {
        this.src = "/DesignAssets/Cases/Case.png";
        this.alt = "Заглушка";
      };
      
      // Добавляем изображение в контейнер
      welcomeCardImage.appendChild(img);
    }
  }
  
  // Показываем первый шаг (выдача карты)
  welcomeStepCard.style.display = "block";
  welcomeModal.style.display = "flex";
}

// Инициализация обработчиков welcome модального окна
function initWelcomeHandlers() {
  const welcomeCardNext = document.getElementById("welcome-card-next");
  const welcomeTutorialNext = document.getElementById("welcome-tutorial-next");
  const welcomeGiftNext = document.getElementById("welcome-gift-next");
  const welcomeModal = document.getElementById("welcome-modal");
  const welcomeStepCard = document.getElementById("welcome-step-card");
  const welcomeStepTutorial = document.getElementById("welcome-step-tutorial");
  const welcomeStepGift = document.getElementById("welcome-step-gift");
  
  // Переход от карты к обучению
  if (welcomeCardNext) {
    welcomeCardNext.addEventListener("click", () => {
      if (welcomeStepCard) welcomeStepCard.style.display = "none";
      if (welcomeStepTutorial) welcomeStepTutorial.style.display = "block";
    });
  }
  
  // Переход от обучения к подарку
  if (welcomeTutorialNext) {
    welcomeTutorialNext.addEventListener("click", () => {
      if (welcomeStepTutorial) welcomeStepTutorial.style.display = "none";
      if (welcomeStepGift) welcomeStepGift.style.display = "block";
    });
  }
  
  // Завершение приветствия и создание пользователя
  if (welcomeGiftNext) {
    welcomeGiftNext.addEventListener("click", async () => {
      const authData = resolveUserId();
      if (!authData) {
        console.error("Не удалось получить данные авторизации");
        return;
      }
      
      try {
        let url = "/api/welcome/create-user";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        const response = await fetch(url, { method: "POST" });
        if (!response.ok) {
          throw new Error(`Ошибка ${response.status}`);
        }
        
        const data = await response.json();
        currentProfile = data;
        renderProfile(data);
        
        // Обновляем хранилище кейсов
        updateCasesStorage(data);
        
        // Скрываем модальное окно
        if (welcomeModal) {
          welcomeModal.style.display = "none";
        }
        
        // Перезагружаем профиль для получения актуальных данных
        await loadProfile(authData);
        
        showNotification("Добро пожаловать в ChibiArena! 🎉", "success");
      } catch (error) {
        console.error("Ошибка создания пользователя:", error);
        showNotification("Ошибка при создании пользователя", "error");
      }
    });
  }
}

// Инициализация при загрузке
document.addEventListener("DOMContentLoaded", async () => {
  console.log("DOM загружен, начинаем инициализацию");
  
  // Инициализируем обработчики событий
  initEventHandlers();
  
  // Инициализируем обработчики welcome модального окна
  initWelcomeHandlers();
  
  // Инициализируем модальное окно выбора метода оплаты
  initPaymentMethodModal();
  
  // Инициализируем фоновую музыку
  initBackgroundMusic();
  
  // Показываем сообщение пользователю
  const playerName = document.getElementById("player-name");
  if (playerName) {
    playerName.textContent = "Загрузка...";
  }
  
  // Таймер для показа сообщения о долгой загрузке
  let longLoadTimer = null;
  const startLongLoadTimer = () => {
    longLoadTimer = setTimeout(() => {
      const loadingMessage = document.getElementById("loading-message");
      const loadingScreen = document.getElementById("loading-screen");
      if (loadingMessage && loadingScreen) {
        const currentDisplay = window.getComputedStyle(loadingScreen).display;
        if (currentDisplay !== "none") {
          loadingMessage.style.display = "block";
          console.log("Показано сообщение о долгой загрузке");
        }
      }
    }, 5000);
  };
  
  // Запускаем таймер сразу
  startLongLoadTimer();
  
  // Пробуем получить данные авторизации
  let authData = null;
  let attempts = 0;
  const maxAttempts = 10;
  
  while (!authData && attempts < maxAttempts) {
    attempts++;
    authData = resolveUserId();
    
    if (!authData) {
      await new Promise(resolve => setTimeout(resolve, 300));
    }
  }
  
  // Если получили данные авторизации, предзагружаем все данные
  if (authData) {
    const loaded = await preloadAppData(authData);
    
    if (loaded) {
      console.log("Данные предзагружены успешно");
      // Включаем музыку сразу, если она включена в настройках
      if (currentSettings && currentSettings.sound_music !== false) {
        musicEnabled = true;
        if (backgroundMusic) {
          backgroundMusic.play().catch(e => {
            console.log("Автовоспроизведение музыки заблокировано:", e);
          });
        }
      }
      
      // Обновляем индикатор непрочитанных писем при загрузке приложения
      await updateMailNotificationBadge(authData);
      
      // Проверяем welcome статус и показываем модальное окно, если нужно
      await checkAndShowWelcome(authData);
    } else {
      console.error("Не удалось предзагрузить данные");
      if (playerName) {
        playerName.textContent = "Ошибка загрузки";
      }
    }
  } else {
    console.error("Не удалось получить данные авторизации");
    if (playerName) {
      playerName.textContent = "Ошибка авторизации";
    }
  }
  
  // Очищаем таймер долгой загрузки, если он еще работает
  if (longLoadTimer) {
    clearTimeout(longLoadTimer);
  }
  
  // В любом случае скрываем экран загрузки через небольшую задержку
  setTimeout(() => {
    console.log("Скрываем экран загрузки");
    hideLoadingScreen();
  }, 500);
  
  // Защита от зависания - принудительно скрываем через 10 секунд
  setTimeout(() => {
    const ls = document.getElementById("loading-screen");
    if (ls) {
      const currentDisplay = window.getComputedStyle(ls).display;
      if (currentDisplay !== "none") {
        console.warn("Принудительное скрытие экрана загрузки после таймаута");
        hideLoadingScreen();
      }
    }
  }, 10000);
  
  // Проверяем статус ожидающего платежа при загрузке
  const paymentAuthData = resolveUserId();
  if (paymentAuthData) {
    // Проверяем параметры из URL (если вернулись с YooKassa)
    checkPaymentFromUrl(paymentAuthData);
    // Проверяем ожидающие платежи
    checkPendingPayment(paymentAuthData);
  }
});

// Проверка платежа из URL параметров (когда пользователь возвращается с YooKassa)
async function checkPaymentFromUrl(authData) {
  const urlParams = new URLSearchParams(window.location.search);
  const paymentId = urlParams.get("payment_id") || urlParams.get("paymentId");
  
  if (!paymentId) return;
  
  console.log("Обнаружен payment_id в URL:", paymentId);
  
  try {
    let url = `/api/payments/status?payment_id=${paymentId}`;
    if (typeof authData === "string") {
      url += `&_auth=${encodeURIComponent(authData)}`;
    }
    
    const response = await fetch(url);
    if (!response.ok) {
      console.error("Ошибка проверки статуса платежа из URL:", response.status);
      return;
    }
    
    const result = await response.json();
    console.log("Статус платежа из URL:", result);
    
    if (result.status === "succeeded" && result.paid) {
      // Проверяем, были ли награды уже выданы
      if (result.rewards_processed) {
        console.log("Платеж уже обработан, награды уже выданы. Пропускаем уведомление.");
        sessionStorage.removeItem('pending_payment_id');
        sessionStorage.removeItem('pending_payment_item');
        sessionStorage.removeItem('pending_payment_timestamp');
        sessionStorage.removeItem('pending_payment_method');
        
        // Удаляем параметры из URL
        const newUrl = window.location.pathname;
        window.history.replaceState({}, document.title, newUrl);
        return;
      }
      
      // Платеж успешен - обрабатываем
      sessionStorage.removeItem('pending_payment_id');
      sessionStorage.removeItem('pending_payment_item');
      sessionStorage.removeItem('pending_payment_timestamp');
      sessionStorage.removeItem('pending_payment_method');
      
      // Удаляем параметры из URL
      const newUrl = window.location.pathname;
      window.history.replaceState({}, document.title, newUrl);
      
      // Обрабатываем успешный платеж
      sessionStorage.setItem('pending_payment_method', 'yookassa');
      await handleSuccessfulPayment(authData);
      
      // Показываем модальное окно успеха
      showPaymentWaitingModal(paymentId);
      setTimeout(() => {
        showPaymentSuccess();
      }, 500);
    } else if (result.status === "pending" || result.status === "waiting_for_capture") {
      // Платеж еще обрабатывается - показываем модальное окно
      showPaymentWaitingModal(paymentId);
    }
  } catch (error) {
    console.error("Ошибка проверки платежа из URL:", error);
  }
}

// ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ПЛАТЕЖАМИ ====================

// Покупка товара за гемы
async function buyWithGems(itemType, gemsAmount, itemName) {
  try {
    const authData = resolveUserId();
    if (!authData) {
      showNotification("Ошибка авторизации", "error");
      return;
    }

    // Проверяем баланс гемов
    if (!currentProfile || currentProfile.gems < gemsAmount) {
      showNotification(`Недостаточно гемов! Нужно ${gemsAmount} 💎, у вас ${currentProfile?.gems || 0} 💎`, "error");
      return;
    }

    let url = "/api/shop/buy";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_type: itemType,
        gems_amount: gemsAmount,
        item_name: itemName
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showNotification(errorData.message || "Ошибка покупки", "error");
      return;
    }

    const result = await response.json();
    if (result.success) {
      // Звук успешной покупки
      window._playSfx?.('success-sound');
      // Воспроизводим звук покупки для монет
      if (itemType && itemType.startsWith("coins_")) {
        playResourcePurchaseSound();
      }
      showNotification(result.message || `Успешно куплено: ${itemName}`, "success");
      await loadProfile(authData);
      // Обновляем кейсы, если купили кейсы (обычные или админские)
      if (itemType === "case" || 
          (typeof itemType === "string" && (
            itemType.startsWith("admin_case_tier_") || 
            itemType.startsWith("keys_")
          ))) {
        casesNeedRefresh = true;
        await loadUserCases(true);
      }
    } else {
      showNotification(result.message || "Ошибка покупки", "error");
    }
  } catch (error) {
    console.error("Ошибка покупки за гемы:", error);
    showNotification("Ошибка покупки", "error");
  }
}

// Создание платежа
async function createPayment(itemType, amount, description, metadata = {}) {
  try {
    const authData = resolveUserId();
    let authParam = null;

    if (typeof authData === "string") {
      authParam = encodeURIComponent(authData);
    } else if (typeof authData === "number") {
      authParam = authData.toString();
    }

    if (!authParam) {
      console.error("Не удалось получить данные для аутентификации");
      showNotification("Ошибка: не удалось определить пользователя. Убедитесь, что вы открыли игру через Telegram.", "error");
      return;
    }

    const normalizedMetadata = { ...(metadata || {}) };
    normalizedMetadata.item_type = normalizedMetadata.item_type || itemType;
    normalizedMetadata.item_name = normalizedMetadata.item_name || (currentPaymentData.itemName || description);
    normalizedMetadata.amount_rub = normalizedMetadata.amount_rub || amount;

    const response = await fetch(`/api/payments/checkout/start?_auth=${authParam}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_type: itemType || "",
        package_type: metadata.package_type || "",
        recipient_id: metadata.recipient_id || undefined,
        ultra: metadata.ultra
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Ошибка HTTP при старте checkout:", response.status, errorText);
      let errorMessage = "Ошибка создания заказа";
      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.message || errorData.error || errorMessage;
      } catch (e) {
        if (response.status === 401) {
          errorMessage = "Ошибка авторизации. Убедитесь, что вы открыли игру через Telegram.";
        } else if (response.status === 400) {
          errorMessage = "Некорректные данные для создания заказа";
        } else if (response.status === 503) {
          errorMessage = "Платежный сервис недоступен. Пожалуйста, попробуйте позже.";
        }
      }
      showNotification(`Ошибка создания заказа: ${errorMessage}`, "error");
      return;
    }

    const result = await response.json();

    if (!result.success) {
      showNotification(`Ошибка создания заказа: ${result.error || result.message || "неизвестная ошибка"}`, "error");
      return;
    }

    if (result.checkout_url) {
      const checkoutFullUrl = result.checkout_url;
      if (result.checkout_jti) {
        sessionStorage.setItem("pending_checkout_jti", result.checkout_jti);
      }
      const link = document.createElement("a");
      link.href = checkoutFullUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      document.body.appendChild(link);
      link.click();
      link.remove();
      showNotification("Страница оплаты открыта. Баланс обновится после оплаты.", "success", 5000);
    } else {
      showNotification("Ошибка: не получен URL для оплаты", "error");
    }
  } catch (error) {
    console.error("Ошибка создания заказа:", error);
    showNotification("Ошибка при создании заказа. Попробуйте позже.", "error");
  }
}

// Показать модальное окно ожидания платежа
function showPaymentWaitingModal(paymentId) {
  const modal = document.getElementById("payment-waiting-modal");
  if (!modal) return;
  
  modal.style.display = "flex";
  const statusDiv = document.getElementById("payment-status");
  const statusIcon = document.getElementById("payment-status-icon");
  const statusText = document.getElementById("payment-status-text");
  const waitingText = document.getElementById("payment-waiting-text");
  const closeBtn = document.getElementById("payment-waiting-close");
  const cancelBtn = document.getElementById("payment-waiting-cancel");
  
  if (statusDiv) statusDiv.style.display = "none";
  if (waitingText) waitingText.style.display = "block";
  if (closeBtn) closeBtn.style.display = "none";
  if (cancelBtn) cancelBtn.style.display = "block";
  
  // Обработчик для кнопки "Отмена"
  if (cancelBtn) {
    cancelBtn.onclick = () => {
      closePaymentWaitingModal();
    };
  }
  
  // Добавляем обработчик закрытия по клику вне модального окна
  // Но проверка статуса продолжается в фоне
  const handleModalClick = (e) => {
    if (e.target === modal) {
      // Закрываем модальное окно, но продолжаем проверку в фоне
      modal.style.display = "none";
      modal.removeEventListener("click", handleModalClick);
    }
  };
  
  // Удаляем старый обработчик, если есть
  modal.removeEventListener("click", handleModalClick);
  // Добавляем новый обработчик
  modal.addEventListener("click", handleModalClick);
  
  // Начинаем проверку статуса (она будет продолжаться даже если модальное окно закрыто)
  startPaymentStatusCheck(paymentId);
}

// Закрыть модальное окно ожидания платежа
function closePaymentWaitingModal() {
  const modal = document.getElementById("payment-waiting-modal");
  if (modal) {
    modal.style.display = "none";
  }
  // Проверка статуса продолжается в фоне, не останавливаем её
}

// ==================== ФУНКЦИИ ДЛЯ МОДАЛЬНОГО ОКНА ВЫБОРА МЕТОДА ОПЛАТЫ ====================

// Глобальные переменные для текущей покупки
let currentPaymentData = {
  itemType: null,
  amount: null,
  description: null,
  itemName: null
};

// Показать модальное окно выбора метода оплаты
async function showPaymentMethodModal(itemType, amount, itemName) {
  const modal = document.getElementById("payment-method-modal");
  if (!modal) {
    console.error("Модальное окно выбора метода оплаты не найдено");
    // Fallback: используем старый метод (YooKassa)
    const description = `Покупка: ${itemName}`;
    createPayment(itemType, amount, description, { item_name: itemName });
    return;
  }

  // Сохраняем данные о покупке, убеждаясь что amount - это число
  let amountNum = amount;
  if (typeof amountNum === "string") {
    // Убираем пробелы и преобразуем в число
    amountNum = parseFloat(amountNum.replace(/\s+/g, ""));
  } else if (typeof amountNum !== "number") {
    amountNum = parseFloat(amountNum) || 0;
  }

  if (!amountNum || amountNum <= 0 || isNaN(amountNum)) {
    console.error("Некорректная сумма платежа:", amount);
    showNotification("Ошибка: некорректная сумма платежа", "error");
    return;
  }

  // Проверяем, что itemType установлен
  if (!itemType) {
    console.error("ОШИБКА: itemType не установлен для товара:", itemName);
    showNotification("Ошибка: тип товара не указан. Обратитесь в поддержку.", "error");
    return;
  }
  
  currentPaymentData = {
    itemType: itemType,
    amount: amountNum,
    description: `Покупка: ${itemName}`,
    itemName: itemName
  };
  
  // Логируем для отладки
  console.log("showPaymentMethodModal: установлен currentPaymentData:", currentPaymentData);

  // Загружаем конфигурацию платежей для расчета цены в Stars
  let starsPrice = null;
  try {
    const configResponse = await fetch("/api/payments/config");
    if (configResponse.ok) {
      const config = await configResponse.json();
      const starsRateRub = config.stars_rate_rub || 1.5;
      const starsMarkup = config.stars_markup || 1.2;
      starsPrice = Math.ceil((amount / starsRateRub) * starsMarkup);
      if (config.stars_test_mode) {
        // В тестовом режиме для админов цена может быть 1 Star
        starsPrice = 1;
      }
    }
  } catch (error) {
    console.error("Ошибка загрузки конфигурации платежей:", error);
  }

  // Обновляем цены в модальном окне
  const starsPriceEl = document.getElementById("payment-method-stars-price");
  const cardPriceEl = document.getElementById("payment-method-card-price");

  if (starsPriceEl) {
    const priceValue = starsPriceEl.querySelector(".payment-price-value");
    if (priceValue) {
      priceValue.textContent = starsPrice || "—";
    }
  }

  if (cardPriceEl) {
    const priceValue = cardPriceEl.querySelector(".payment-price-value");
    if (priceValue) {
      priceValue.textContent = amount.toLocaleString("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
  }

  // Сбрасываем выбор
  const options = modal.querySelectorAll(".payment-method-option");
  options.forEach(opt => opt.classList.remove("active"));

  // Показываем модальное окно
  modal.style.display = "flex";

  // Воспроизводим звук кнопки
  playButtonSound();
}

// Закрыть модальное окно выбора метода оплаты
function closePaymentMethodModal() {
  const modal = document.getElementById("payment-method-modal");
  if (!modal) return;

  // Анимация закрытия
  modal.style.opacity = "0";
  setTimeout(() => {
    modal.style.display = "none";
    modal.style.opacity = "1"; // Восстанавливаем для следующего открытия
  }, 300);

  // Сбрасываем данные
  currentPaymentData = {
    itemType: null,
    amount: null,
    description: null,
    itemName: null
  };
}

// Инициализация обработчиков для модального окна выбора метода оплаты
function initPaymentMethodModal() {
  const modal = document.getElementById("payment-method-modal");
  if (!modal) return;

  // Обработчик закрытия по кнопке крестика
  const closeBtn = document.getElementById("payment-method-modal-close");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      closePaymentMethodModal();
      playButtonSound();
    });
  }

  // Обработчик кнопки "Отмена"
  const cancelBtn = document.getElementById("payment-method-cancel-btn");
  if (cancelBtn) {
    cancelBtn.addEventListener("click", () => {
      closePaymentMethodModal();
      playButtonSound();
    });
  }

  // Обработчики клика на опции методов оплаты
  const options = modal.querySelectorAll(".payment-method-option");
  options.forEach(option => {
    option.addEventListener("click", () => {
      // Убираем выделение с других опций
      options.forEach(opt => opt.classList.remove("active"));
      
      // Выделяем выбранную опцию
      option.classList.add("active");
      
      // Воспроизводим звук
      playButtonSound();
      
      // Получаем выбранный метод
      const method = option.dataset.method;
      console.log("Выбран метод оплаты:", method);

      // Обрабатываем выбор метода оплаты
      if (method === "stars") {
        handleStarsPayment();
      } else if (method === "yookassa") {
        handleYooKassaPayment();
      }
    });
  });

  // Закрытие по клику вне модального окна
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closePaymentMethodModal();
    }
  });
}

// Обработка платежа через Telegram Stars
async function handleStarsPayment() {
  if (!currentPaymentData.itemType || !currentPaymentData.amount) {
    showNotification("Ошибка: данные о покупке не найдены", "error");
    closePaymentMethodModal();
    return;
  }

  // Проверяем и преобразуем amount в число
  let amountRub = currentPaymentData.amount;
  if (typeof amountRub === "string") {
    // Убираем пробелы и преобразуем в число
    amountRub = parseFloat(amountRub.replace(/\s+/g, ""));
  } else if (typeof amountRub !== "number") {
    amountRub = parseFloat(amountRub) || 0;
  }

  if (!amountRub || amountRub <= 0 || isNaN(amountRub)) {
    showNotification("Ошибка: некорректная сумма платежа", "error");
    closePaymentMethodModal();
    return;
  }

  try {
    const authData = resolveUserId();
    if (!authData) {
      showNotification("Ошибка: не удалось определить пользователя", "error");
      closePaymentMethodModal();
      return;
    }

    let authParam = null;
    if (typeof authData === "string") {
      authParam = encodeURIComponent(authData);
    } else if (typeof authData === "number") {
      authParam = authData.toString();
    }

    if (!authParam) {
      showNotification("Ошибка: не удалось получить данные для аутентификации", "error");
      closePaymentMethodModal();
      return;
    }

    // Сохраняем данные о покупке ПЕРЕД закрытием модального окна (которое сбрасывает currentPaymentData)
    const paymentItemType = currentPaymentData.itemType;
    const paymentItemName = currentPaymentData.itemName;
    const paymentDescription = currentPaymentData.description || "Покупка в магазине";

    // Закрываем модальное окно выбора метода
    closePaymentMethodModal();

    // Показываем уведомление о загрузке
    showNotification("Создание платежа через Telegram Stars...", "info");

    // Формируем metadata, используя сохраненные данные
    const metadata = {};
    if (paymentItemType) {
      metadata.item_type = paymentItemType;
    }
    if (paymentItemName || paymentDescription) {
      metadata.item_name = paymentItemName || paymentDescription;
    }
    
    // Логируем для отладки
    console.log("Создание платежа Stars:", {
      amount_rub: amountRub,
      item_type: paymentItemType,
      item_name: paymentItemName,
      metadata: metadata,
      saved_before_close: true
    });

    // Отправляем запрос на создание инвойса Stars
    const response = await fetch(`/api/payments/stars/create?_auth=${authParam}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        amount_rub: amountRub,
        description: paymentDescription,
        metadata: metadata
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showNotification(errorData.message || "Ошибка создания платежа через Stars", "error");
      return;
    }

    const result = await response.json();

    if (!result.success) {
      showNotification(result.message || "Ошибка создания платежа через Stars", "error");
      return;
    }

    // Сохраняем payment_id для проверки статуса
    if (result.payment_id) {
      sessionStorage.setItem('pending_payment_id', result.payment_id);
      sessionStorage.setItem('pending_payment_item', paymentItemType);
      sessionStorage.setItem('pending_payment_timestamp', Date.now().toString());
      sessionStorage.setItem('pending_payment_method', 'stars');
    }

    // Показываем модальное окно с инструкциями (вместо простого уведомления)
    showStarsPaymentInstructionModal(result.payment_id);

    // Начинаем проверку статуса платежа (она обработает выдачу наград при успехе)
    startPaymentStatusCheck(result.payment_id);

  } catch (error) {
    console.error("Ошибка создания платежа через Stars:", error);
    showNotification("Ошибка при создании платежа. Попробуйте позже.", "error");
  }
}

// Обработка платежа через YooKassa
async function handleYooKassaPayment() {
  if (!currentPaymentData.itemType || !currentPaymentData.amount) {
    showNotification("Ошибка: данные о покупке не найдены", "error");
    closePaymentMethodModal();
    return;
  }

  // Проверяем и преобразуем amount в число
  let amountNum = currentPaymentData.amount;
  if (typeof amountNum === "string") {
    // Убираем пробелы и преобразуем в число
    amountNum = parseFloat(amountNum.replace(/\s+/g, ""));
  } else if (typeof amountNum !== "number") {
    amountNum = parseFloat(amountNum) || 0;
  }

  if (!amountNum || amountNum <= 0 || isNaN(amountNum)) {
    console.error("Некорректная сумма платежа:", currentPaymentData.amount);
    showNotification("Ошибка: некорректная сумма платежа", "error");
    closePaymentMethodModal();
    return;
  }

  // Сохраняем данные о покупке ПЕРЕД закрытием модального окна (которое сбрасывает currentPaymentData)
  const paymentItemType = currentPaymentData.itemType;
  const paymentDescription = currentPaymentData.description || `Покупка: ${currentPaymentData.itemName || "Товар"}`;

  // Закрываем модальное окно выбора метода
  closePaymentMethodModal();

  // Используем существующую функцию createPayment с сохраненными данными
  createPayment(
    paymentItemType,
    amountNum,
    paymentDescription
  );
}

// Проверка статуса платежа с интервалом
let paymentCheckInterval = null;
let activePaymentChecks = new Map(); // Храним активные проверки платежей

function startPaymentStatusCheck(paymentId) {
  // Если уже есть активная проверка для этого платежа, не создаем новую
  if (activePaymentChecks.has(paymentId)) {
    console.log(`Проверка статуса платежа ${paymentId} уже активна`);
    return;
  }
  
  // Помечаем, что проверка началась
  activePaymentChecks.set(paymentId, true);
  
  const authData = resolveUserId();
  if (!authData) {
    activePaymentChecks.delete(paymentId);
    return;
  }
  
  let checkCount = 0;
  const maxChecks = 120; // Проверяем до 120 раз (10 минут при интервале 5 секунд)
  
  const checkInterval = setInterval(async () => {
    checkCount++;
    
    try {
      let url = `/api/payments/status?payment_id=${paymentId}`;
      if (typeof authData === "string") {
        url += `&_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `&user_id=${authData}`;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        if (checkCount >= maxChecks) {
          clearInterval(paymentCheckInterval);
          paymentCheckInterval = null;
          showPaymentError("Превышено время ожидания. Проверьте статус платежа в истории.");
        }
        return;
      }
      
      const result = await response.json();
      
      if (result.status === "succeeded" && result.paid) {
        // Проверяем, были ли награды уже выданы
        if (result.rewards_processed) {
          console.log("Платеж уже обработан, награды уже выданы. Пропускаем уведомление.");
          clearInterval(checkInterval);
          activePaymentChecks.delete(paymentId);
          
          // Очищаем sessionStorage
          sessionStorage.removeItem('pending_payment_id');
          sessionStorage.removeItem('pending_payment_item');
          sessionStorage.removeItem('pending_payment_timestamp');
          sessionStorage.removeItem('pending_payment_method');
          
          // Закрываем модальное окно, если открыто
          closePaymentWaitingModal();
          // Закрываем модальное окно инструкций Stars, если открыто
          closeStarsPaymentInstructionModal();
          return;
        }
        
        // Платеж успешен
        clearInterval(checkInterval);
        activePaymentChecks.delete(paymentId);
        
        // Проверяем, какой метод оплаты использовался
        const paymentMethod = sessionStorage.getItem('pending_payment_method');
        
        // Закрываем модальное окно инструкций Stars, если открыто
        if (paymentMethod === 'stars') {
          closeStarsPaymentInstructionModal();
        }
        
        // Показываем модальное окно ожидания, если оно было закрыто
        const modal = document.getElementById("payment-waiting-modal");
        if (modal) {
          modal.style.display = "flex";
        }
        
        // Показываем успех
        showPaymentSuccess();
        
        // Загружаем письма и показываем награды (с визуалом выдачи)
        await handleSuccessfulPayment(authData);
        
        // Очищаем sessionStorage после обработки
        sessionStorage.removeItem('pending_payment_id');
        sessionStorage.removeItem('pending_payment_item');
        sessionStorage.removeItem('pending_payment_timestamp');
        sessionStorage.removeItem('pending_payment_method');
        
        // Автоматически закрываем модальное окно ожидания через 2 секунды
        setTimeout(() => {
          closePaymentWaitingModal();
        }, 2000);
      } else if (result.status === "canceled") {
        // Платеж отменен
        clearInterval(checkInterval);
        activePaymentChecks.delete(paymentId);
        
        // Показываем модальное окно, если оно было закрыто
        const modal = document.getElementById("payment-waiting-modal");
        if (modal) {
          modal.style.display = "flex";
        }
        
        showPaymentError("Платеж был отменен.");
      }
    } catch (error) {
      console.error("Ошибка проверки статуса платежа:", error);
      if (checkCount >= maxChecks) {
        clearInterval(checkInterval);
        activePaymentChecks.delete(paymentId);
      }
    }
  }, 5000); // Проверяем каждые 5 секунд
  
  // Сохраняем интервал для этого платежа
  paymentCheckInterval = checkInterval;
}

// Показать успешный статус платежа
function showPaymentSuccess() {
  const statusDiv = document.getElementById("payment-status");
  const statusIcon = document.getElementById("payment-status-icon");
  const statusText = document.getElementById("payment-status-text");
  const waitingText = document.getElementById("payment-waiting-text");
  const closeBtn = document.getElementById("payment-waiting-close");
  
  if (statusDiv) {
    statusDiv.style.display = "block";
    statusDiv.className = "payment-status success";
  }
  if (statusIcon) statusIcon.textContent = "✅";
  if (statusText) statusText.textContent = "Платеж успешно обработан!";
  if (waitingText) waitingText.style.display = "none";
  if (closeBtn) {
    closeBtn.style.display = "block";
    closeBtn.onclick = () => {
      closePaymentWaitingModal();
    };
  }
  
  // Автоматически закрываем модальное окно через 3 секунды
  setTimeout(() => {
    closePaymentWaitingModal();
  }, 3000);
}

// Показать ошибку платежа
function showPaymentError(message) {
  const statusDiv = document.getElementById("payment-status");
  const statusIcon = document.getElementById("payment-status-icon");
  const statusText = document.getElementById("payment-status-text");
  const waitingText = document.getElementById("payment-waiting-text");
  const closeBtn = document.getElementById("payment-waiting-close");
  
  if (statusDiv) {
    statusDiv.style.display = "block";
    statusDiv.className = "payment-status error";
  }
  if (statusIcon) statusIcon.textContent = "❌";
  if (statusText) statusText.textContent = message || "Произошла ошибка при обработке платежа";
  if (waitingText) waitingText.style.display = "none";
  if (closeBtn) {
    closeBtn.style.display = "block";
    closeBtn.onclick = () => {
      closePaymentWaitingModal();
    };
  }
}

// Обработка успешного платежа
async function handleSuccessfulPayment(authData) {
  console.log("Обработка успешного платежа...");
  
  const pendingPaymentId = sessionStorage.getItem('pending_payment_id');

  const oldProfile = currentProfile ? {...currentProfile} : null;
  await loadProfile(authData);
  const newProfile = currentProfile;
  
  // Проверяем метод оплаты для определения времени ожидания
  const paymentMethod = sessionStorage.getItem('pending_payment_method');
  // Для Stars платежей нужно больше времени, так как они обрабатываются через бота
  const waitTime = paymentMethod === 'stars' ? 5000 : 2000;
  
  // Ждем немного, чтобы webhook/bot успел обработаться
  await new Promise(resolve => setTimeout(resolve, waitTime));
  
  // Загружаем письма несколько раз для Stars платежей (они могут обрабатываться с задержкой)
  let purchaseMail = null;
  const maxRetries = paymentMethod === 'stars' ? 3 : 1;
  
  for (let retry = 0; retry < maxRetries; retry++) {
    // Загружаем письма
    try {
      await loadMail(authData);
    } catch (error) {
      console.error("Ошибка загрузки почты:", error);
    }
    
    // Получаем последнее письмо о покупке
    try {
      // Для Stars платежей ищем все письма, не только непрочитанные
      const unreadOnly = paymentMethod !== 'stars';
      let mailUrl = `/api/mail?limit=10&category=rewards${unreadOnly ? '&unread_only=true' : ''}`;
      if (typeof authData === "string") {
        mailUrl += `&_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        mailUrl += `&user_id=${authData}`;
      }
      
      const mailResponse = await fetch(mailUrl);
      if (mailResponse.ok) {
        const mailResult = await mailResponse.json();
        if (mailResult.mail && mailResult.mail.length > 0) {
          // Ищем письма о покупках (более широкий поиск)
          const purchaseMails = mailResult.mail.filter(mail => 
            (mail.subject && (
              mail.subject.includes("Покупка") || 
              mail.subject.includes("покупка") ||
              mail.subject.includes("Платеж") ||
              mail.subject.includes("платеж")
            )) || 
            mail.icon === "💳" ||
            mail.icon === "⭐" ||
            mail.category === "rewards"
          );
          
          if (purchaseMails.length > 0) {
            // Берем самое свежее письмо
            purchaseMail = purchaseMails[0];
            console.log("Найдено письмо о покупке:", purchaseMail);
            break; // Нашли письмо, выходим из цикла
          }
        }
      }
    } catch (error) {
      console.error("Ошибка получения письма о покупке:", error);
    }
    
    // Если не нашли письмо и это не последняя попытка, ждем еще
    if (retry < maxRetries - 1 && !purchaseMail) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
  
  // Вычисляем награды из изменений профиля
  const rewards = [];
  if (oldProfile && newProfile) {
    const gemsDiff = (newProfile.gems || 0) - (oldProfile.gems || 0);
    const coinsDiff = (newProfile.coins || 0) - (oldProfile.coins || 0);
    const oldExtraPass = oldProfile.extra_pass === "active";
    const newExtraPass = newProfile.extra_pass === "active";
    
    if (gemsDiff > 0) rewards.push({ type: "gems", amount: gemsDiff, icon: "💎" });
    if (coinsDiff > 0) rewards.push({ type: "coins", amount: coinsDiff, icon: "💰" });
    // Проверяем активацию ExtraPass (только если его не было)
    if (!oldExtraPass && newExtraPass) {
      rewards.push({ type: "extrapass", amount: 1, icon: "⭐", label: "ExtraPass" });
    }
  }
  
  // Дополняем награды из письма (если что-то не было обнаружено из профиля)
  if (purchaseMail && purchaseMail.attachments) {
    // Проверяем, есть ли уже гемы в наградах
    const hasGems = rewards.some(r => r.type === "gems");
    if (purchaseMail.attachments.gems) {
      if (!hasGems) {
        // Если гемы не были обнаружены из профиля, добавляем из письма
        rewards.push({ type: "gems", amount: purchaseMail.attachments.gems, icon: "💎" });
      } else {
        // Если гемы уже есть, но в письме указано больше (например, 1200 вместо 500), обновляем
        const existingGems = rewards.find(r => r.type === "gems");
        if (existingGems && purchaseMail.attachments.gems > existingGems.amount) {
          existingGems.amount = purchaseMail.attachments.gems;
        }
      }
    }
    // Проверяем, есть ли уже монеты в наградах
    const hasCoins = rewards.some(r => r.type === "coins");
    if (!hasCoins && purchaseMail.attachments.coins) {
      rewards.push({ type: "coins", amount: purchaseMail.attachments.coins, icon: "💰" });
    }
    // Проверяем ExtraPass
    const hasExtraPass = rewards.some(r => r.type === "extrapass");
    if (!hasExtraPass && purchaseMail.attachments.extrapass) {
      rewards.push({ type: "extrapass", amount: 1, icon: "⭐", label: "ExtraPass" });
    }
    // Обработка кейсов из стартового буста
    if (purchaseMail.attachments.cases && Array.isArray(purchaseMail.attachments.cases)) {
      purchaseMail.attachments.cases.forEach(tier => {
        const tierNames = { 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5" };
        rewards.push({ type: "case", amount: 1, icon: "📦", label: `Кейс ${tierNames[tier] || `T${tier}`}` });
      });
    }
  }
  
  // Если награды все еще не найдены, но есть письмо, показываем общее сообщение
  if (rewards.length === 0 && purchaseMail) {
    // Пытаемся извлечь награды из текста письма
    const mailContent = purchaseMail.content || purchaseMail.text || "";
    if (mailContent.includes("ExtraPass")) {
      rewards.push({ type: "extrapass", amount: 1, icon: "⭐", label: "ExtraPass" });
    }
    if (mailContent.includes("гемов")) {
      const gemsMatch = mailContent.match(/(\d+)\s*💎\s*гемов/);
      if (gemsMatch) {
        rewards.push({ type: "gems", amount: parseInt(gemsMatch[1]), icon: "💎" });
      }
    }
    if (mailContent.includes("монет")) {
      const coinsMatch = mailContent.match(/(\d+)\s*💰\s*монет/);
      if (coinsMatch) {
        rewards.push({ type: "coins", amount: parseInt(coinsMatch[1]), icon: "💰" });
      }
    }
    if (mailContent.includes("кейс")) {
      const caseMatches = mailContent.match(/(\d+)×T(\d+)\s*кейс/g);
      if (caseMatches) {
        caseMatches.forEach(match => {
          const tierMatch = match.match(/T(\d+)/);
          if (tierMatch) {
            const tier = parseInt(tierMatch[1]);
            const tierNames = { 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5" };
            rewards.push({ type: "case", amount: 1, icon: "📦", label: `Кейс ${tierNames[tier] || `T${tier}`}` });
          }
        });
      }
    }
  }
  
  // Если все еще нет наград, но платеж успешен, пытаемся еще раз загрузить профиль и письма
  if (rewards.length === 0) {
    console.log("Награды не найдены, повторная попытка загрузки...");
    // Еще раз обновляем профиль
    await loadProfile(authData);
    const finalProfile = currentProfile;
    
    // Сравниваем с начальным профилем
    if (oldProfile && finalProfile) {
      const gemsDiff = (finalProfile.gems || 0) - (oldProfile.gems || 0);
      const coinsDiff = (finalProfile.coins || 0) - (oldProfile.coins || 0);
      const oldExtraPass = oldProfile.extra_pass === "active";
      const newExtraPass = finalProfile.extra_pass === "active";
      
      if (gemsDiff > 0) rewards.push({ type: "gems", amount: gemsDiff, icon: "💎" });
      if (coinsDiff > 0) rewards.push({ type: "coins", amount: coinsDiff, icon: "💰" });
      if (!oldExtraPass && newExtraPass) {
        rewards.push({ type: "extrapass", amount: 1, icon: "⭐", label: "ExtraPass" });
      }
    }
    
    // Если все еще нет наград, создаем общее сообщение об успехе
    if (rewards.length === 0) {
      rewards.push({ type: "success", amount: 1, icon: "✅", label: "Покупка успешна" });
    }
  }
  
  // Показываем красивое всплывающее окно с наградами (всегда)
  console.log("Показываем модальное окно с наградами:", rewards);
  showPurchaseSuccessModal(rewards, purchaseMail);
  
  // Помечаем платёж: модалка показана
  if (pendingPaymentId) {
    try {
      let markUrl = '/api/payments/modal-shown';
      if (typeof authData === "string") {
        markUrl += `?_auth=${encodeURIComponent(authData)}`;
      }
      await fetch(markUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payment_id: pendingPaymentId }),
      });
    } catch(e) { console.error("Не удалось пометить модалку:", e); }
  }

  // Еще раз обновляем профиль для отображения новых ресурсов
  setTimeout(async () => {
    await loadProfile(authData);
  }, 1000);
}

// Проверка статуса ожидающего платежа при загрузке страницы
async function checkPendingPayment(authData) {
  let pendingPaymentId = sessionStorage.getItem('pending_payment_id');
  if (!pendingPaymentId) {
    let jti = sessionStorage.getItem("pending_checkout_jti");
    if (jti) {
      try {
        var jtiUrl = `/api/payments/checkout/session-status?jti=${encodeURIComponent(jti)}`;
        if (typeof authData === "string") jtiUrl += `&_auth=${encodeURIComponent(authData)}`;
        var jtiRes = await fetch(jtiUrl);
        if (jtiRes.ok) {
          var jtiData = await jtiRes.json();
          if (jtiData.payment_id && jtiData.payment_status === "succeeded") {
            pendingPaymentId = jtiData.payment_id;
            sessionStorage.setItem("pending_payment_id", jtiData.payment_id);
            sessionStorage.setItem("pending_payment_method", "yookassa");
            sessionStorage.setItem("pending_payment_timestamp", String(Date.now()));
          }
        }
      } catch(_) {}
    }
    if (!pendingPaymentId) {
      try {
        var succRes = await fetch("/api/payments/recent-success?" + (typeof authData === "string" ? "_auth=" + encodeURIComponent(authData) : ""));
        if (succRes.ok) {
          var succData = await succRes.json();
          if (succData.payments && succData.payments.length > 0) {
            pendingPaymentId = succData.payments[0].payment_id;
            sessionStorage.setItem("pending_payment_id", pendingPaymentId);
            sessionStorage.setItem("pending_payment_method", "yookassa");
            sessionStorage.setItem("pending_payment_timestamp", String(Date.now()));
          }
        }
      } catch(_) {}
    }
  }
  if (!pendingPaymentId) {
    await checkUnreadPurchaseMails(authData);
    return;
  }
  
  const timestamp = parseInt(sessionStorage.getItem('pending_payment_timestamp') || "0");
  const now = Date.now();
  const timeDiff = now - timestamp;
  
  // Если прошло больше 10 минут, удаляем из sessionStorage
  if (timeDiff > 10 * 60 * 1000) {
    sessionStorage.removeItem('pending_payment_id');
    sessionStorage.removeItem('pending_payment_item');
    sessionStorage.removeItem('pending_payment_timestamp');
    sessionStorage.removeItem('pending_payment_method');
    // Все равно проверяем непрочитанные письма
    await checkUnreadPurchaseMails(authData);
    return;
  }
  
  try {
    let url = `/api/payments/status?payment_id=${pendingPaymentId}`;
    if (typeof authData === "string") {
      url += `&_auth=${encodeURIComponent(authData)}`;
    }
    
    const response = await fetch(url);
    if (!response.ok) {
      // Если платеж еще обрабатывается, показываем модальное окно ожидания
      showPaymentWaitingModal(pendingPaymentId);
      return;
    }
    
    const result = await response.json();
    
    if (result.status === "succeeded" && result.paid) {
      // Платеж успешен - обрабатываем
      await handleSuccessfulPayment(authData);

      sessionStorage.removeItem('pending_payment_id');
      sessionStorage.removeItem('pending_payment_item');
      sessionStorage.removeItem('pending_payment_timestamp');
      sessionStorage.removeItem('pending_payment_method');
      sessionStorage.removeItem('pending_checkout_jti');
    } else if (result.status === "pending" || result.status === "waiting_for_capture") {
      // Платеж еще обрабатывается - показываем модальное окно
      showPaymentWaitingModal(pendingPaymentId);
    } else {
      // Платеж отменен или ошибка
      sessionStorage.removeItem('pending_payment_id');
      sessionStorage.removeItem('pending_payment_item');
      sessionStorage.removeItem('pending_payment_timestamp');
      sessionStorage.removeItem('pending_payment_method');
      sessionStorage.removeItem('pending_checkout_jti');
      await checkUnreadPurchaseMails(authData);
    }
  } catch (error) {
    console.error("Ошибка проверки статуса платежа:", error);
    // Проверяем непрочитанные письма на всякий случай
    await checkUnreadPurchaseMails(authData);
  }
}

// Проверка непрочитанных писем о покупках
async function checkUnreadPurchaseMails(authData) {
  try {
    let mailUrl = "/api/mail?category=rewards&unread_only=true&limit=10";
    if (typeof authData === "string") {
      mailUrl += `&_auth=${encodeURIComponent(authData)}`;
    } else if (typeof authData === "number") {
      mailUrl += `&user_id=${authData}`;
    }
    
    const mailResponse = await fetch(mailUrl);
    if (mailResponse.ok) {
      const mailResult = await mailResponse.json();
      if (mailResult.mail && mailResult.mail.length > 0) {
        // Ищем письма о покупках
        const purchaseMails = mailResult.mail.filter(mail => 
          (mail.subject && mail.subject.includes("Покупка")) || mail.icon === "💳"
        );
        
        if (purchaseMails.length > 0) {
          // Показываем уведомление о самом новом письме
          const latestMail = purchaseMails[0];
          showPurchaseNotification(latestMail);
        }
      }
    }
  } catch (error) {
    console.error("Ошибка проверки непрочитанных писем:", error);
  }
}

// Показать уведомление о покупке
function showPurchaseNotification(mail) {
  if (!mail) return;
  
  let message = "Покупка успешно обработана!";
  if (mail.subject) {
    message = mail.subject;
  }
  
  showNotification(message, "success");
  
  // Обновляем профиль для отображения новых ресурсов
  const authData = resolveUserId();
  if (authData) {
    loadProfile(authData);
  }
}

// Показать красивое всплывающее окно успешной покупки
function showPurchaseSuccessModal(rewards, mail) {
  const modal = document.getElementById("purchase-success-modal");
  if (!modal) return;
  
  const messageEl = document.getElementById("purchase-success-message");
  const rewardsContainer = document.getElementById("purchase-rewards-container");
  const okBtn = document.getElementById("purchase-success-ok-btn");
  
  // Устанавливаем сообщение
  if (messageEl) {
    if (mail && mail.subject) {
      messageEl.textContent = mail.subject;
    } else {
      messageEl.textContent = "Ваш платеж успешно обработан!";
    }
  }
  
  // Отображаем награды
  if (rewardsContainer) {
    if (rewards.length > 0) {
      rewardsContainer.innerHTML = rewards.map(reward => {
        // Специальное отображение для ExtraPass
        if (reward.type === "extrapass") {
          return `
            <div class="purchase-reward-item purchase-reward-extrapass" data-reward-type="${reward.type}">
              <div class="reward-icon-large reward-icon-extrapass">${reward.icon}</div>
              <div class="reward-amount-large">
                <div class="reward-extrapass-title">ExtraPass</div>
                <div class="reward-extrapass-subtitle">Активирован на 30 дней</div>
              </div>
            </div>
          `;
        }
        // Специальное отображение для кейсов
        if (reward.type === "case") {
          return `
            <div class="purchase-reward-item purchase-reward-case" data-reward-type="${reward.type}">
              <div class="reward-icon-large">${reward.icon}</div>
              <div class="reward-amount-large">
                <span class="reward-amount-value">${reward.amount}×</span>
                <span class="reward-amount-label">${reward.label || "кейс"}</span>
              </div>
            </div>
          `;
        }
        // Общее сообщение об успехе (если нет конкретных наград)
        if (reward.type === "success") {
          return `
            <div class="purchase-reward-item" data-reward-type="${reward.type}">
              <div class="reward-icon-large">${reward.icon}</div>
              <div class="reward-amount-large">
                <span class="reward-amount-label">${reward.label || "Покупка успешна"}</span>
              </div>
            </div>
          `;
        }
        // Обычное отображение для гемов, монет и т.д.
        const label = reward.type === "gems" ? "гемов" : reward.type === "coins" ? "монет" : "ключей";
        return `
        <div class="purchase-reward-item" data-reward-type="${reward.type}" data-reward-amount="${reward.amount}">
          <div class="reward-icon-large">${reward.icon}</div>
          <div class="reward-amount-large">
              <span class="reward-amount-value">${reward.amount.toLocaleString()}</span>
              <span class="reward-amount-label">${label}</span>
          </div>
        </div>
        `;
      }).join("");
    } else {
      rewardsContainer.innerHTML = '<div style="text-align: center; color: var(--chibi-text-muted); padding: 20px;">Награды начислены</div>';
    }
  }
  
  // Обработчик кнопки "Отлично"
  if (okBtn) {
    okBtn.onclick = () => {
      // Воспроизводим звук успешной покупки при нажатии на кнопку "Отлично"
      playPurchaseSuccessSound();
      closePurchaseSuccessModal();
    };
  }
  
  // Показываем модальное окно с анимацией
  modal.style.display = "flex";
  
  // Запускаем анимацию появления
  setTimeout(() => {
    modal.classList.add("purchase-success-show");
    const content = modal.querySelector(".purchase-success-content");
    if (content) {
      content.classList.add("purchase-success-content-show");
    }
    
    // Анимация начисления валют
    if (rewards.length > 0) {
      animateRewards(rewards);
    }
  }, 50);
}

// Закрыть модальное окно успешной покупки
function closePurchaseSuccessModal() {
  const modal = document.getElementById("purchase-success-modal");
  if (modal) {
    modal.classList.remove("purchase-success-show");
    const content = modal.querySelector(".purchase-success-content");
    if (content) {
      content.classList.remove("purchase-success-content-show");
    }
    setTimeout(() => {
      modal.style.display = "none";
    }, 300);
  }
}

// ==================== ФУНКЦИИ ДЛЯ МОДАЛЬНОГО ОКНА ИНСТРУКЦИЙ TELEGRAM STARS ====================

// Показать модальное окно инструкций по оплате через Telegram Stars
function showStarsPaymentInstructionModal(paymentId) {
  const modal = document.getElementById("stars-payment-instruction-modal");
  if (!modal) return;

  // Показываем модальное окно
  modal.style.display = "flex";

  // Воспроизводим звук кнопки
  playButtonSound();

  // Обработчик кнопки закрытия
  const closeBtn = document.getElementById("stars-payment-instruction-close");
  if (closeBtn) {
    closeBtn.onclick = () => {
      closeStarsPaymentInstructionModal();
      playButtonSound();
    };
  }

  // Обработчик кнопки "Понятно"
  const okBtn = document.getElementById("stars-payment-instruction-ok-btn");
  if (okBtn) {
    okBtn.onclick = () => {
      closeStarsPaymentInstructionModal();
      playButtonSound();
    };
  }

  // Закрытие по клику вне модального окна
  const handleModalClick = (e) => {
    if (e.target === modal) {
      closeStarsPaymentInstructionModal();
      modal.removeEventListener("click", handleModalClick);
    }
  };
  modal.addEventListener("click", handleModalClick);
}

// Закрыть модальное окно инструкций по оплате через Telegram Stars
function closeStarsPaymentInstructionModal() {
  const modal = document.getElementById("stars-payment-instruction-modal");
  if (!modal) return;

  // Анимация закрытия
  modal.style.opacity = "0";
  setTimeout(() => {
    modal.style.display = "none";
    modal.style.opacity = "1"; // Восстанавливаем для следующего открытия
  }, 300);
}

// Показать модальное окно с деталями товара
function showItemDetailModal(itemCard) {
  const modal = document.getElementById("item-detail-modal");
  if (!modal) return;
  
  const itemType = itemCard.dataset.itemType;
  const itemDetail = itemCard.dataset.itemDetail;
  const priceStr = (itemCard.dataset.itemPrice || "0").replace(/\s+/g, "");
  const itemPrice = parseFloat(priceStr);
  const itemCurrency = itemCard.dataset.currency || "rubles";
  const itemName = itemCard.querySelector(".item-name")?.textContent || "Товар";
  const itemDescription = itemCard.querySelector(".item-description")?.textContent || "";
  const itemRarity = itemCard.dataset.rarity || "common";
  const itemImage = itemCard.querySelector(".item-image");
  const itemStats = itemCard.querySelector(".item-stats");
  
  // Заполняем заголовок
  const titleEl = document.getElementById("item-detail-title");
  if (titleEl) titleEl.textContent = itemName;
  
  // Заполняем визуальную часть
  const visualEl = document.getElementById("item-detail-visual");
  if (visualEl && itemImage) {
    visualEl.innerHTML = itemImage.outerHTML;
    visualEl.querySelector(".item-image")?.classList.add("item-detail-image-large");
  }
  
  // Заполняем название
  const nameEl = document.getElementById("item-detail-name");
  if (nameEl) nameEl.textContent = itemName;
  
  // Заполняем описание
  const descEl = document.getElementById("item-detail-description");
  if (descEl) descEl.textContent = itemDescription;
  
  // Заполняем особенности и награды
  const featuresEl = document.getElementById("item-detail-features");
  const rewardsEl = document.getElementById("item-detail-rewards");
  
  if (itemDetail === "extrapass") {
    if (featuresEl) {
      featuresEl.innerHTML = `
        <div class="item-detail-feature">
          <div class="feature-icon">⏱️</div>
          <div class="feature-text">
            <div class="feature-title">Длительность</div>
            <div class="feature-desc">30 дней активной подписки</div>
          </div>
        </div>
        <div class="item-detail-feature">
          <div class="feature-icon">⚡</div>
          <div class="feature-text">
            <div class="feature-title">Дополнительная энергия</div>
            <div class="feature-desc">+1 к максимальной энергии</div>
          </div>
        </div>
        <div class="item-detail-feature">
          <div class="feature-icon">🎁</div>
          <div class="feature-text">
            <div class="feature-title">Эксклюзивные награды</div>
            <div class="feature-desc">Доступ к уникальным наградам в боевом пропуске</div>
          </div>
        </div>
        <div class="item-detail-feature">
          <div class="feature-icon">⭐</div>
          <div class="feature-text">
            <div class="feature-title">Приоритетная поддержка</div>
            <div class="feature-desc">Быстрая обработка запросов</div>
          </div>
        </div>
      `;
    }
    if (rewardsEl) {
      rewardsEl.innerHTML = `
        <div class="item-detail-rewards-title">Что вы получите:</div>
        <div class="item-detail-reward-item">
          <span class="reward-icon">⚡</span>
          <span class="reward-text">+1 к максимальной энергии</span>
        </div>
        <div class="item-detail-reward-item">
          <span class="reward-icon">🎁</span>
          <span class="reward-text">Эксклюзивные награды в боевом пропуске</span>
        </div>
        <div class="item-detail-reward-item">
          <span class="reward-icon">⭐</span>
          <span class="reward-text">Приоритетная поддержка</span>
        </div>
      `;
    }
  } else if (itemDetail === "starter_boost") {
    if (featuresEl) {
      featuresEl.innerHTML = `
        <div class="item-detail-feature">
          <div class="feature-icon">🚀</div>
          <div class="feature-text">
            <div class="feature-title">Идеальный старт</div>
            <div class="feature-desc">Все необходимое для быстрого прогресса</div>
          </div>
        </div>
        <div class="item-detail-feature">
          <div class="feature-icon">💎</div>
          <div class="feature-text">
            <div class="feature-title">Большой набор ресурсов</div>
            <div class="feature-desc">Гемы, монеты и кейсы для начала игры</div>
          </div>
        </div>
      `;
    }
    if (rewardsEl && itemStats) {
      const statsHTML = Array.from(itemStats.querySelectorAll(".stat-item"))
        .map(stat => {
          const text = stat.textContent.trim();
          const icon = text.match(/[⭐💎📦💰]/)?.[0] || "🎁";
          const label = text.replace(/[⭐💎📦💰]/g, "").trim();
          return `
            <div class="item-detail-reward-item">
              <span class="reward-icon">${icon}</span>
              <span class="reward-text">${label}</span>
            </div>
          `;
        }).join("");
      rewardsEl.innerHTML = `
        <div class="item-detail-rewards-title">Содержимое набора:</div>
        ${statsHTML}
      `;
    }
  }
  
  // Заполняем кнопку покупки
  const buyBtn = document.getElementById("item-detail-buy-btn");
  const priceIconEl = document.getElementById("item-detail-price-icon");
  const priceValueEl = document.getElementById("item-detail-price-value");
  
  if (buyBtn && priceIconEl && priceValueEl) {
    priceIconEl.textContent = itemCurrency === "gems" ? "💎" : "₽";
    priceValueEl.textContent = itemPrice.toLocaleString();
    
    buyBtn.onclick = (e) => {
      e.stopPropagation();
      closeItemDetailModal();
      
      // Имитируем клик на кнопку покупки в карточке
      const originalBtn = itemCard.querySelector(".item-buy-btn");
      if (originalBtn) {
        originalBtn.click();
      }
    };
  }
  
  // Показываем модальное окно
  modal.style.display = "flex";
}

// Закрыть модальное окно деталей товара
function closeItemDetailModal() {
  const modal = document.getElementById("item-detail-modal");
  if (modal) {
    modal.style.display = "none";
  }
}

// Анимация начисления валют
function animateRewards(rewards) {
  rewards.forEach((reward, index) => {
    const rewardItem = document.querySelector(`[data-reward-type="${reward.type}"][data-reward-amount="${reward.amount}"]`);
    if (!rewardItem) return;
    
    setTimeout(() => {
      rewardItem.classList.add("reward-animate");
      
      // Анимация счетчика
      const amountValue = rewardItem.querySelector(".reward-amount-value");
      if (amountValue) {
        animateCounter(amountValue, 0, reward.amount, 1000);
      }
    }, index * 200);
  });
  
  // Обновляем значения в профиле с анимацией
  setTimeout(() => {
    const authData = resolveUserId();
    if (authData) {
      loadProfile(authData).then(() => {
        animateProfileValues(rewards);
      });
    }
  }, rewards.length * 200 + 500);
}

// Анимация счетчика
function animateCounter(element, from, to, duration) {
  const startTime = Date.now();
  const animate = () => {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = Math.floor(from + (to - from) * progress);
    element.textContent = current;
    
    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      element.textContent = to;
    }
  };
  animate();
}

// Анимация значений в профиле
function animateProfileValues(rewards) {
  rewards.forEach(reward => {
    let selector = null;
    if (reward.type === "gems") {
      selector = ".profile-gems, #profile-gems, [data-resource='gems']";
    } else if (reward.type === "coins") {
      selector = ".profile-coins, #profile-coins, [data-resource='coins']";
    }
    
    if (selector) {
      const elements = document.querySelectorAll(selector);
      elements.forEach(el => {
        el.classList.add("resource-animate");
        setTimeout(() => {
          el.classList.remove("resource-animate");
        }, 1000);
      });
    }
  });
}

// Функция для показа уведомлений
function showNotification(message, type = "info") {
  const notification = document.getElementById("game-notification");
  if (!notification) {
    console.warn("Элемент game-notification не найден");
    return;
  }
  
  const icon = notification.querySelector("#notification-icon");
  const text = notification.querySelector("#notification-text");
  
  if (icon && text) {
    // Устанавливаем иконку в зависимости от типа
  const icons = {
    success: "✅",
    error: "❌",
      warning: "⚠️",
      info: "ℹ️"
  };
  icon.textContent = icons[type] || icons.info;
  text.textContent = message;
  
    // Устанавливаем класс для стилизации
  notification.className = `game-notification ${type}`;
    
    // Показываем уведомление
  notification.style.display = "block";
  
    // Скрываем через 5 секунд
  setTimeout(() => {
    notification.style.display = "none";
    }, 5000);
  }
}

// Кастомная функция для замены alert()
async function showGameAlert(message, icon = "ℹ️") {
  return new Promise((resolve) => {
    const modal = document.getElementById("game-alert-modal");
    const messageEl = document.getElementById("game-alert-message");
    const iconEl = document.getElementById("game-alert-icon");
    const okBtn = document.getElementById("game-alert-ok");
    const closeBtn = document.getElementById("game-alert-close");
    
    if (!modal || !messageEl || !okBtn) {
      // Fallback на стандартный alert (только в крайнем случае)
      console.warn("Модальное окно alert не найдено, используем стандартный alert");
      alert(message);
      resolve();
      return;
    }
    
    messageEl.textContent = message;
    iconEl.textContent = icon;
    modal.style.display = "flex";
    
    const closeModal = () => {
      modal.style.display = "none";
      resolve();
    };
    
    okBtn.onclick = closeModal;
    closeBtn.onclick = closeModal;
    
    // Закрытие по клику на overlay
    modal.onclick = (e) => {
      if (e.target === modal) {
        closeModal();
      }
    };
    
    // Закрытие по Escape
    const handleEscape = (e) => {
      if (e.key === "Escape") {
        closeModal();
        document.removeEventListener("keydown", handleEscape);
      }
    };
    document.addEventListener("keydown", handleEscape);
  });
}

// Кастомная функция для замены confirm()
async function showGameConfirm(message, icon = "⚠️") {
  return new Promise((resolve) => {
    const modal = document.getElementById("game-confirm-modal");
    const messageEl = document.getElementById("game-confirm-message");
    const iconEl = document.getElementById("game-confirm-icon");
    const okBtn = document.getElementById("game-confirm-ok");
    const cancelBtn = document.getElementById("game-confirm-cancel");
    const closeBtn = document.getElementById("game-confirm-close");
    
    if (!modal || !messageEl || !okBtn || !cancelBtn) {
      // Fallback на стандартный confirm (только в крайнем случае)
      console.warn("Модальное окно confirm не найдено, используем стандартный confirm");
      resolve(confirm(message));
      return;
    }
    
    messageEl.textContent = message;
    iconEl.textContent = icon;
    modal.style.display = "flex";
    
    const closeModal = (result) => {
      modal.style.display = "none";
      resolve(result);
    };
    
    okBtn.onclick = () => closeModal(true);
    cancelBtn.onclick = () => closeModal(false);
    closeBtn.onclick = () => closeModal(false);
    
    // Закрытие по клику на overlay
    modal.onclick = (e) => {
      if (e.target === modal) {
        closeModal(false);
      }
    };
    
    // Закрытие по Escape
    const handleEscape = (e) => {
      if (e.key === "Escape") {
        closeModal(false);
        document.removeEventListener("keydown", handleEscape);
      }
    };
    document.addEventListener("keydown", handleEscape);
  });
}

// Функция для загрузки почты
async function loadMail(authData, category = null) {
  try {
    let url = "/api/mail";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    } else {
      throw new Error("Invalid authentication data");
    }
    
    // Добавляем фильтр категории, если указан
    if (category) {
      url += `&category=${category}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const data = await response.json();
    
    // Обновляем список писем в интерфейсе
    renderMail(data.mail || []);
    
    // Обновляем индикатор непрочитанных писем
    await updateMailNotificationBadge(authData);
    
    return data;
  } catch (error) {
    console.error("Ошибка загрузки почты:", error);
    throw error;
  }
}

// Загрузка почты с автоматическим прочтением всех непрочитанных писем
async function loadMailAndMarkAsRead(authData) {
  try {
    // Сначала загружаем почту
    const data = await loadMail(authData);
    
    // Находим все непрочитанные письма
    const unreadMails = (data.mail || []).filter(mail => !mail.is_read);
    
    if (unreadMails.length > 0) {
      // Помечаем все непрочитанные письма как прочитанные
      const markPromises = unreadMails.map(async (mail) => {
        try {
          const mailId = mail.id || mail.mail_id;
          if (!mailId) return;
          
          let url = "/api/mail/read";
          if (typeof authData === "string") {
            url += `?_auth=${encodeURIComponent(authData)}`;
          } else if (typeof authData === "number") {
            url += `?user_id=${authData}`;
          }
          
          await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mail_id: parseInt(mailId) })
          });
        } catch (error) {
          console.error("Ошибка отметки письма как прочитанного:", error);
        }
      });
      
      // Ждем, пока все письма будут помечены как прочитанные
      await Promise.all(markPromises);
      
      // Перезагружаем почту, чтобы обновить статус
      await loadMail(authData);
    }
  } catch (error) {
    console.error("Ошибка загрузки почты с автоматическим прочтением:", error);
  }
}

// Обновление индикатора непрочитанных писем на кнопке меню и в меню
async function updateMailNotificationBadge(authData) {
  try {
    if (!authData) {
      console.warn("updateMailNotificationBadge: нет данных авторизации");
      return;
    }
    
    let url = "/api/mail/unread-count";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }
    
    console.log("Проверка непрочитанных писем:", url);
    const response = await fetch(url);
    if (!response.ok) {
      console.error("Ошибка получения количества непрочитанных писем:", response.status);
      return;
    }
    
    const data = await response.json();
    const unreadCount = data.count || 0;
    console.log(`Найдено непрочитанных писем: ${unreadCount}`);
    
    // Обновляем индикатор на кнопке меню
    const badge = document.getElementById("menu-notification-badge");
    if (badge) {
      if (unreadCount > 0) {
        badge.textContent = unreadCount > 99 ? "99+" : unreadCount.toString();
        badge.style.display = "flex";
        console.log(`Индикатор меню обновлен: ${unreadCount} непрочитанных писем`);
      } else {
        badge.style.display = "none";
        console.log("Индикатор меню скрыт: нет непрочитанных писем");
      }
    } else {
      console.warn("Элемент menu-notification-badge не найден");
    }
    
    // Обновляем индикатор на кнопке почты в меню
    const mailBadge = document.getElementById("mail-menu-notification-badge");
    if (mailBadge) {
      if (unreadCount > 0) {
        mailBadge.textContent = unreadCount > 99 ? "99+" : unreadCount.toString();
        mailBadge.style.display = "flex";
        console.log(`Индикатор почты обновлен: ${unreadCount} непрочитанных писем`);
      } else {
        mailBadge.style.display = "none";
        console.log("Индикатор почты скрыт: нет непрочитанных писем");
      }
    } else {
      console.warn("Элемент mail-menu-notification-badge не найден");
    }
  } catch (error) {
    console.error("Ошибка проверки непрочитанных писем:", error);
  }
}

// Функция для отображения почты
function renderMail(mailList) {
  const mailListElement = document.querySelector(".mail-list");
  if (!mailListElement) return;
  
  if (mailList.length === 0) {
    mailListElement.innerHTML = "";
    const emptyState = document.querySelector(".mail-empty-state");
    if (emptyState) {
      emptyState.style.display = "block";
    }
    return;
  }
  
  const emptyState = document.querySelector(".mail-empty-state");
  if (emptyState) {
    emptyState.style.display = "none";
  }
  
  mailListElement.innerHTML = mailList.map(mail => {
    const unreadClass = mail.is_read === false ? "unread" : "";
    const timeAgo = mail.created_at ? formatTimeAgo(new Date(mail.created_at)) : "";
    const mailId = mail.id || mail.mail_id; // Поддержка обоих вариантов
    const mailContent = mail.content || mail.body || mail.text || ""; // Поддержка обоих вариантов
    
    // Форматируем attachments для отображения
    let attachmentsHtml = "";
    if (mail.attachments) {
      const att = mail.attachments;
      const attList = [];
      if (att.gems) attList.push(`${att.gems} 💎`);
      if (att.coins) attList.push(`${att.coins} 💰`);
      
      if (attList.length > 0) {
        attachmentsHtml = `
          <div class="mail-attachments">
            ${attList.map(a => `<span class="attachment-badge">${a}</span>`).join("")}
          </div>
        `;
      }
    }
    
    return `
      <div class="mail-item ${unreadClass}" data-mail-id="${mailId}">
        <div class="mail-icon">${mail.icon || "📧"}</div>
        <div class="mail-content">
          <div class="mail-header">
            <div class="mail-sender">${mail.sender || "Система"}</div>
            <div class="mail-time">${timeAgo}</div>
          </div>
          <div class="mail-subject">${mail.subject || "Без темы"}</div>
          <div class="mail-preview">${mailContent}</div>
          ${attachmentsHtml}
        </div>
        ${mail.is_read === false ? '<div class="mail-status unread-dot"></div>' : ''}
      </div>
    `;
  }).join("");
  
  // Добавляем обработчики для писем после рендеринга
  setTimeout(() => {
    document.querySelectorAll(".mail-item").forEach(item => {
      // Удаляем старые обработчики, если есть
      const newItem = item.cloneNode(true);
      item.parentNode.replaceChild(newItem, item);
      
      newItem.addEventListener("click", async () => {
        const mailId = newItem.dataset.mailId;
        if (!mailId) return;
        
        // Убираем статус "непрочитано" при клике
        newItem.classList.remove("unread");
        const unreadDot = newItem.querySelector(".mail-status.unread-dot");
        if (unreadDot) {
          unreadDot.style.display = "none";
        }
        
        // Отмечаем письмо как прочитанное
        const authData = resolveUserId();
        if (authData) {
          try {
            let url = "/api/mail/read";
            if (typeof authData === "string") {
              url += `?_auth=${encodeURIComponent(authData)}`;
            } else if (typeof authData === "number") {
              url += `?user_id=${authData}`;
            }
            
            await fetch(url, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ mail_id: parseInt(mailId) })
            });
            
            // Обновляем индикатор непрочитанных писем
            await updateMailNotificationBadge(authData);
          } catch (error) {
            console.error("Ошибка отметки письма как прочитанного:", error);
          }
      }
      
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {}
    });
  });
    
    // Обработчики для фильтров почты
    document.querySelectorAll(".mail-filter-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        // Убираем активный класс со всех фильтров
        document.querySelectorAll(".mail-filter-btn").forEach(b => b.classList.remove("active"));
        // Добавляем активный класс к выбранному
        btn.classList.add("active");
        
        const filter = btn.textContent.trim();
        const authData = resolveUserId();
        if (!authData) return;
        
        let category = null;
        if (filter === "Награды") {
          category = "rewards";
        } else if (filter === "Новости") {
          category = "news";
        } else if (filter === "События") {
          category = "events";
        }
        
        // Загружаем письма с фильтром
        loadMail(authData, category);
      });
    });
  }, 100);
}

// Вспомогательная функция для форматирования времени
function formatTimeAgo(date) {
  const now = new Date();
  const diff = now - date;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) return `${days} дн. назад`;
  if (hours > 0) return `${hours} ч. назад`;
  if (minutes > 0) return `${minutes} мин. назад`;
  return "только что";
}

// ==================== КЕЙСЫ ====================

function getCaseTierConfig(tier) {
  // Кэшируем конфигурацию
  if (!tierConfigCache.has(tier)) {
    tierConfigCache.set(tier, CASE_TIER_CONFIG[tier] || CASE_TIER_CONFIG[1]);
  }
  return tierConfigCache.get(tier);
}

function getCaseMaxRarityText(config) {
  if (!config || !config.maxRarityKey) {
    return "Обычная";
  }
  const rarityName = getRarityName(config.maxRarityKey);
  if (config.allowsLimited) {
    return `${rarityName} + лимитированные`;
  }
  return rarityName;
}

// Функция предзагрузки всех данных для кейсов
async function preloadCasesData() {
  if (casesPreloaded || casesPreloading) {
    return;
  }
  
  casesPreloading = true;
  const loadingModal = document.getElementById("case-loading-modal");
  const loadingText = document.getElementById("case-loading-text");
  const loadingProgress = document.getElementById("case-loading-progress-bar");
  const loadingStatus = document.getElementById("case-loading-status");
  
  if (loadingModal) {
    loadingModal.style.display = "flex";
  }
  
  const totalSteps = 15; // 5 тиров × 2 этапа (select + taps) + 5 конфигураций
  let currentStep = 0;
  
  const updateProgress = (step, text) => {
    currentStep = step;
    const progress = Math.round((currentStep / totalSteps) * 100);
    if (loadingProgress) {
      loadingProgress.style.width = `${progress}%`;
    }
    if (loadingStatus) {
      loadingStatus.textContent = `${progress}%`;
    }
    if (loadingText && text) {
      loadingText.textContent = text;
    }
  };
  
  try {
    // Шаг 1-5: Предзагрузка конфигураций тиров
    for (let tier = 1; tier <= 5; tier++) {
      getCaseTierConfig(tier);
      updateProgress(tier, `Загрузка конфигурации T${tier}...`);
      await new Promise(resolve => setTimeout(resolve, 10)); // Небольшая задержка для плавности
    }
    
    // Шаг 6-10: Предзагрузка визуалов для этапа select
    const modal = document.getElementById("case-opening-modal");
    if (modal) {
      // Создаем скрытые контейнеры для предзагрузки
      const preloadContainer = document.createElement("div");
      preloadContainer.style.position = "absolute";
      preloadContainer.style.visibility = "hidden";
      preloadContainer.style.pointerEvents = "none";
      preloadContainer.style.top = "-9999px";
      document.body.appendChild(preloadContainer);
      
      for (let tier = 1; tier <= 5; tier++) {
        const tempContainer = document.createElement("div");
        tempContainer.className = "case-display";
        preloadContainer.appendChild(tempContainer);
        renderCaseVisual(tempContainer, tier, { animate: false });
        // Сохраняем HTML вместо самого элемента
        preloadedVisuals.select.set(tier, tempContainer.innerHTML);
        updateProgress(5 + tier, `Подготовка визуала T${tier} (выбор)...`);
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      // Шаг 11-15: Предзагрузка визуалов для этапа taps
      for (let tier = 1; tier <= 5; tier++) {
        const tempContainer = document.createElement("div");
        tempContainer.className = "case-display case-display-tapping";
        preloadContainer.appendChild(tempContainer);
        renderCaseVisual(tempContainer, tier, { animate: false });
        // Сохраняем HTML вместо самого элемента
        preloadedVisuals.taps.set(tier, tempContainer.innerHTML);
        updateProgress(10 + tier, `Подготовка визуала T${tier} (тапы)...`);
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      // Предзагружаем DOM элементы модального окна
      updateProgress(15, "Финальная подготовка...");
      caseDOMCache.caseDisplay = document.getElementById("case-display");
      caseDOMCache.caseDisplayTapping = document.getElementById("case-display-tapping");
      caseDOMCache.tierBadge = document.getElementById("case-tier-badge");
      caseDOMCache.tierBadgeTapping = document.getElementById("case-tier-badge-tapping");
      caseDOMCache.stageInfo = document.getElementById("case-stage-info");
      caseDOMCache.tapIndicators = document.getElementById("case-tap-indicators")?.querySelectorAll(".tap-indicator");
      caseDOMCache.progressMeter = document.getElementById("case-progress-meter");
      caseDOMCache.tapHint = document.getElementById("case-tap-hint");
      
      // Предзагружаем HTML для stageInfo для всех тиров
      for (let tier = 1; tier <= 5; tier++) {
        const config = getCaseTierConfig(tier);
        const cacheKey = `${tier}-${config.coinsRange?.[0]}-${config.coinsRange?.[1]}-${config.cardsRange?.[0]}-${config.cardsRange?.[1]}`;
        if (!stageInfoHTMLCache.has(cacheKey)) {
          const coinsRange = config.coinsRange ? `${config.coinsRange[0]}–${config.coinsRange[1]}` : "—";
          const cardsRange = config.cardsRange ? `${config.cardsRange[0]}–${config.cardsRange[1]}` : "—";
          const html = `
            <div>💰 Монеты: ${coinsRange}</div>
            <div>🃏 Карт: ${cardsRange}</div>
            <div>🌟 Макс редкость: ${escapeHtml(getCaseMaxRarityText(config))}</div>
          `;
          stageInfoHTMLCache.set(cacheKey, html);
        }
      }
      
      // Удаляем временный контейнер после небольшой задержки
      setTimeout(() => {
        if (preloadContainer.parentNode) {
          preloadContainer.parentNode.removeChild(preloadContainer);
        }
      }, 100);
    }
    
    updateProgress(totalSteps, "Готово!");
    casesPreloaded = true;
    
    // Скрываем экран загрузки с небольшой задержкой
    await new Promise(resolve => setTimeout(resolve, 300));
    if (loadingModal) {
      loadingModal.style.display = "none";
    }
  } catch (error) {
    console.error("Ошибка предзагрузки кейсов:", error);
    if (loadingModal) {
      loadingModal.style.display = "none";
    }
  } finally {
    casesPreloading = false;
  }
}

async function initCasesTab(forceReload = false) {
  bindCasesGrid();
  bindCaseModalControls();
  const shouldForce = forceReload || casesNeedRefresh;
  if (shouldForce) {
    await loadUserCases(true);
    return;
  }
  if (!userCasesData.length) {
    await loadUserCases();
  } else {
    renderCasesGrid();
  }
}

function bindCasesGrid() {
  if (casesGridBound) {
    return;
  }
  const grid = document.getElementById("cases-grid");
  if (!grid) {
    return;
  }
  grid.addEventListener("click", (event) => {
    const actionButton = event.target.closest("[data-action='open-case']");
    if (!actionButton) {
      return;
    }
    const caseId = Number(actionButton.dataset.caseId);
    if (!caseId) {
      return;
    }
    const selectedCase = userCasesData.find(item => item.user_case_id === caseId);
    if (selectedCase) {
      openCaseModal(selectedCase);
    }
  });
  casesGridBound = true;
}

function bindCaseModalControls() {
  if (caseModalBound) {
    return;
  }
  document.getElementById("case-open-start-btn")?.addEventListener("click", startCaseTapping);
  document.getElementById("case-skip-btn")?.addEventListener("click", handleCaseSkip);
  document.getElementById("case-skip-btn-taps")?.addEventListener("click", handleCaseSkip);
  document.getElementById("case-display-tapping")?.addEventListener("click", handleCaseTap);
  document.getElementById("case-opening-close")?.addEventListener("click", closeCaseModal);
  document.getElementById("case-rewards-close-btn")?.addEventListener("click", closeCaseModal);
  caseModalBound = true;
}

async function loadUserCases(forceReload = false) {
  if (casesLoading) {
    return;
  }
  if (!forceReload && userCasesData.length) {
    renderCasesGrid();
    return;
  }
  const authData = resolveUserId();
  if (!authData) {
    return;
  }
  casesLoading = true;
  try {
    const url = appendAuthParams("/api/cases/user", authData);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const data = await response.json();
    userCasesData = Array.isArray(data.cases) ? data.cases : [];
    renderCasesGrid();
    casesNeedRefresh = false;
  } catch (error) {
    console.error("Ошибка загрузки кейсов:", error);
    showNotification("Не удалось загрузить кейсы", "error");
  } finally {
    casesLoading = false;
  }
}

function renderCasesGrid() {
  const grid = document.getElementById("cases-grid");
  const emptyState = document.getElementById("cases-empty-state");
  if (!grid) {
    return;
  }
  if (!userCasesData.length) {
    grid.innerHTML = "";
    if (emptyState) {
      emptyState.style.display = "block";
    }
    return;
  }
  if (emptyState) {
    emptyState.style.display = "none";
  }
  grid.innerHTML = userCasesData.map(createCaseCardHTML).join("");
}

function createCaseCardHTML(userCase) {
  const tier = userCase.tier || 1;
  const config = getCaseTierConfig(tier);
  const caseName = userCase.case_name || config.title;
  const description = userCase.description || `Монеты и карты уровня ${tier}`;
  return `
    <div class="case-card" data-tier="${tier}">
      <div class="case-card-image-wrapper">
        <img src="/DesignAssets/Cases/Case.png" alt="${escapeHtml(caseName)}" class="case-card-image" />
        <div class="case-tier-chip">T${tier}</div>
      </div>
      <div class="case-card-header">
        <div class="case-card-title">${escapeHtml(caseName)}</div>
      </div>
      <div class="case-card-body">
        ${escapeHtml(description)}
      </div>
      <div class="case-card-actions">
        <button class="case-open-action" data-action="open-case" data-case-id="${userCase.user_case_id}">
          Открыть
        </button>
      </div>
    </div>
  `;
}

async function openCaseModal(userCase) {
  const modal = document.getElementById("case-opening-modal");
  if (!modal) {
    return;
  }
  
  // Если кейсы еще не загружены, запускаем предзагрузку
  if (!casesPreloaded && !casesPreloading) {
    await preloadCasesData();
  } else if (casesPreloading) {
    // Если идет загрузка, ждем её завершения
    while (casesPreloading) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  
  // Предзагружаем DOM элементы сразу (если еще не загружены)
  if (!caseDOMCache.caseDisplay) {
    caseDOMCache.caseDisplay = document.getElementById("case-display");
    caseDOMCache.caseDisplayTapping = document.getElementById("case-display-tapping");
    caseDOMCache.tierBadge = document.getElementById("case-tier-badge");
    caseDOMCache.tierBadgeTapping = document.getElementById("case-tier-badge-tapping");
    caseDOMCache.stageInfo = document.getElementById("case-stage-info");
    caseDOMCache.tapIndicators = document.getElementById("case-tap-indicators")?.querySelectorAll(".tap-indicator");
    caseDOMCache.progressMeter = document.getElementById("case-progress-meter");
    caseDOMCache.tapHint = document.getElementById("case-tap-hint");
  }
  
  // Предзагружаем звук тапа кейса для быстрого воспроизведения
  const caseTapSound = document.getElementById("case-tap-sound");
  if (caseTapSound) {
    try {
      // Загружаем звук, чтобы он был готов к воспроизведению
      caseTapSound.load();
    } catch (error) {
      console.debug("Не удалось предзагрузить звук тапа кейса:", error);
    }
  }
  
  document.body.classList.add("case-opening-active");
  caseOpeningState = {
    userCase,
    tapResults: [],
    upgradeFlags: [],
    currentTier: userCase.tier || 1,
    tapInProgress: false,
    finalizing: false,
    skipInProgress: false,
    stage: "select",
  };
  
  // Сбрасываем кэш визуала при открытии нового кейса
  caseVisualCache.select = { tier: null, container: null };
  caseVisualCache.taps = { tier: null, container: null };
  
  modal.style.display = "flex";
  const titleEl = document.getElementById("case-opening-title");
  if (titleEl) {
    titleEl.textContent = `Открытие: ${userCase.case_name || getCaseTierConfig(caseOpeningState.currentTier).title}`;
  }
  renderCaseRewards(null);
  const startBtn = document.getElementById("case-open-start-btn");
  if (startBtn) {
    startBtn.disabled = false;
    startBtn.textContent = "Нажми, чтобы открыть";
  }
  document.getElementById("case-skip-btn")?.removeAttribute("disabled");
  document.getElementById("case-skip-btn-taps")?.removeAttribute("disabled");
  const contentRoot = document.querySelector(".case-opening-content");
  if (contentRoot) {
    contentRoot.classList.toggle("reduced-effects", CASE_EFFECTS_REDUCED);
  }
  
  // Используем предзагруженные визуалы, если они есть
  const initialTier = caseOpeningState.currentTier;
  if (caseDOMCache.caseDisplay) {
    // Если есть предзагруженный визуал, используем его HTML
    const preloadedSelectHTML = preloadedVisuals.select.get(initialTier);
    if (preloadedSelectHTML) {
      caseDOMCache.caseDisplay.innerHTML = preloadedSelectHTML;
      caseVisualCache.select.tier = initialTier;
      caseVisualCache.select.container = caseDOMCache.caseDisplay;
    } else {
      // Иначе рендерим как обычно
      renderCaseVisual(caseDOMCache.caseDisplay, initialTier, { animate: false });
      caseVisualCache.select.tier = initialTier;
      caseVisualCache.select.container = caseDOMCache.caseDisplay;
    }
  }
  if (caseDOMCache.caseDisplayTapping) {
    const preloadedTapsHTML = preloadedVisuals.taps.get(initialTier);
    if (preloadedTapsHTML) {
      caseDOMCache.caseDisplayTapping.innerHTML = preloadedTapsHTML;
      caseVisualCache.taps.tier = initialTier;
      caseVisualCache.taps.container = caseDOMCache.caseDisplayTapping;
    } else {
      renderCaseVisual(caseDOMCache.caseDisplayTapping, initialTier, { animate: false });
      caseVisualCache.taps.tier = initialTier;
      caseVisualCache.taps.container = caseDOMCache.caseDisplayTapping;
    }
  }
  
  setCaseStage("select");
  // Используем requestIdleCallback для обновления, если доступен
  if (window.requestIdleCallback) {
    requestIdleCallback(() => updateCaseStageInfo(), { timeout: 100 });
  } else {
    requestAnimationFrame(() => updateCaseStageInfo());
  }
}

function setCaseStage(stage) {
  // Кэшируем элементы этапов
  const selectStage = document.getElementById("case-opening-stage-select");
  const tapsStage = document.getElementById("case-opening-stage-taps");
  const rewardsStage = document.getElementById("case-opening-stage-rewards");
  const stages = {
    select: selectStage,
    taps: tapsStage,
    rewards: rewardsStage,
  };
  
  // Используем один requestAnimationFrame для всех изменений
  requestAnimationFrame(() => {
    Object.entries(stages).forEach(([key, element]) => {
      if (!element) {
        return;
      }
      const isActive = key === stage;
      // Используем visibility вместо display для лучшей производительности
      if (isActive) {
        element.style.display = "block";
        element.style.visibility = "visible";
        element.style.opacity = "1";
        element.classList.add("active");
      } else {
        element.style.opacity = "0";
        element.style.visibility = "hidden";
        element.classList.remove("active");
        // Используем setTimeout только для скрытия display
        setTimeout(() => {
          if (!element.classList.contains("active")) {
            element.style.display = "none";
          }
        }, 200);
      }
    });
  });
  
  if (caseOpeningState) {
    caseOpeningState.stage = stage;
  }
  const contentRoot = document.querySelector(".case-opening-content");
  if (contentRoot) {
    contentRoot.setAttribute("data-case-stage", stage);
  }
  updateCaseTapHint();
  if (stage === "taps") {
    updateCaseProgressMeter();
    // Визуал уже предзагружен, просто убеждаемся что он актуален
    if (caseOpeningState && caseDOMCache.caseDisplayTapping) {
      const tier = caseOpeningState.currentTier;
      if (caseVisualCache.taps.tier !== tier) {
        renderCaseVisual(caseDOMCache.caseDisplayTapping, tier, { animate: false });
        caseVisualCache.taps.tier = tier;
      }
    }
  }
  if (stage === "rewards") {
    playCaseRewardsTransition();
  }
}

function updateCaseStageInfo() {
  if (!caseOpeningState) {
    return;
  }
  
  // Отменяем предыдущие запланированные обновления
  if (caseUpdateRAF) {
    cancelAnimationFrame(caseUpdateRAF);
  }
  if (caseUpdateTimeout) {
    clearTimeout(caseUpdateTimeout);
  }
  
  // Используем requestIdleCallback если доступен, иначе requestAnimationFrame
  const scheduleUpdate = (window.requestIdleCallback && window.requestIdleCallback.bind(window)) || requestAnimationFrame;
  
  caseUpdateRAF = scheduleUpdate(() => {
    const tier = caseOpeningState.currentTier;
    const config = getCaseTierConfig(tier);
    
    // Кэшируем HTML для stageInfo
    const cacheKey = `${tier}-${config.coinsRange?.[0]}-${config.coinsRange?.[1]}-${config.cardsRange?.[0]}-${config.cardsRange?.[1]}`;
    if (caseDOMCache.stageInfo) {
      let newHTML = stageInfoHTMLCache.get(cacheKey);
      if (!newHTML) {
        const coinsRange = config.coinsRange ? `${config.coinsRange[0]}–${config.coinsRange[1]}` : "—";
        const cardsRange = config.cardsRange ? `${config.cardsRange[0]}–${config.cardsRange[1]}` : "—";
        newHTML = `
          <div>💰 Монеты: ${coinsRange}</div>
          <div>🃏 Карт: ${cardsRange}</div>
          <div>🌟 Макс редкость: ${escapeHtml(getCaseMaxRarityText(config))}</div>
        `;
        stageInfoHTMLCache.set(cacheKey, newHTML);
      }
      // Обновляем только если изменилось
      if (caseDOMCache.stageInfo.innerHTML !== newHTML) {
        caseDOMCache.stageInfo.innerHTML = newHTML;
      }
    }
    
    // Обновляем бейджи только если тир изменился (используем кэшированные элементы)
    if (caseDOMCache.tierBadge) {
      const newText = `T${tier}`;
      const currentTier = caseDOMCache.tierBadge.getAttribute("data-tier");
      if (currentTier !== String(tier)) {
        caseDOMCache.tierBadge.textContent = newText;
        caseDOMCache.tierBadge.setAttribute("data-tier", tier);
      }
    }
    
    if (caseDOMCache.tierBadgeTapping) {
      const newText = `T${tier}`;
      const currentTier = caseDOMCache.tierBadgeTapping.getAttribute("data-tier");
      if (currentTier !== String(tier)) {
        caseDOMCache.tierBadgeTapping.textContent = newText;
        caseDOMCache.tierBadgeTapping.setAttribute("data-tier", tier);
      }
    }
    
    // Оптимизация: обновляем визуал только если тир изменился
    const shouldAnimate = Boolean(caseOpeningState.shouldPulseVisual && !CASE_EFFECTS_REDUCED);
    
    // Обновляем только визуал активного этапа и только если тир изменился
    if (caseOpeningState.stage === "select") {
      if (caseDOMCache.caseDisplay && caseVisualCache.select.tier !== tier) {
        renderCaseVisual(caseDOMCache.caseDisplay, tier, { animate: shouldAnimate });
        caseVisualCache.select.tier = tier;
        caseVisualCache.select.container = caseDOMCache.caseDisplay;
      } else if (shouldAnimate && caseDOMCache.caseDisplay) {
        // Если только анимация нужна, запускаем её без полной перерисовки
        const shell = caseDOMCache.caseDisplay.querySelector(".case-visual-shell");
        if (shell) {
          shell.classList.remove("case-tier-surge");
          requestAnimationFrame(() => {
            shell.classList.add("case-tier-surge");
          });
        }
      }
    } else if (caseOpeningState.stage === "taps") {
      if (caseDOMCache.caseDisplayTapping && caseVisualCache.taps.tier !== tier) {
        renderCaseVisual(caseDOMCache.caseDisplayTapping, tier, { animate: shouldAnimate });
        caseVisualCache.taps.tier = tier;
        caseVisualCache.taps.container = caseDOMCache.caseDisplayTapping;
      } else if (shouldAnimate && caseDOMCache.caseDisplayTapping) {
        // Если только анимация нужна, запускаем её без полной перерисовки
        const shell = caseDOMCache.caseDisplayTapping.querySelector(".case-visual-shell");
        if (shell) {
          shell.classList.remove("case-tier-surge");
          requestAnimationFrame(() => {
            shell.classList.add("case-tier-surge");
          });
        }
      }
    }
    
    caseOpeningState.shouldPulseVisual = false;
    updateCaseTapIndicators();
    caseUpdateRAF = null;
  }, { timeout: 16 }); // Максимум 16мс для requestIdleCallback
}

function ensureCaseVisualStructure(container) {
  if (!container) {
    return null;
  }
  let shell = container.querySelector(".case-visual-shell");
  if (shell) {
    return shell;
  }
  shell = document.createElement("div");
  shell.className = "case-visual-shell";
  shell.innerHTML = `
    <div class="case-visual-aurora"></div>
    <div class="case-orbit case-orbit-outer"></div>
    <div class="case-orbit case-orbit-inner"></div>
    <div class="case-visual-core" data-tier="1">
      <span class="case-visual-glyph">📦</span>
      <span class="case-visual-core-glow"></span>
    </div>
    <div class="case-visual-sparks"></div>
  `;
  container.innerHTML = "";
  container.appendChild(shell);
  return shell;
}

function renderCaseVisual(container, tier, options = {}) {
  if (!container) {
    return;
  }
  const shell = ensureCaseVisualStructure(container);
  if (!shell) {
    return;
  }
  
  // Используем will-change для оптимизации анимаций
  if (options.animate) {
    shell.style.willChange = "transform, opacity, filter";
  }
  
  const config = getCaseTierConfig(tier);
  const core = shell.querySelector(".case-visual-core");
  if (core) {
    const currentTier = core.dataset.tier;
    // Обновляем только если тир изменился
    if (currentTier !== String(tier)) {
      // Используем requestAnimationFrame для плавного обновления
      requestAnimationFrame(() => {
        core.dataset.tier = tier;
        // Убираем will-change после анимации
        if (!options.animate) {
          shell.style.willChange = "auto";
        }
      });
    }
  }
  const glyph = shell.querySelector(".case-visual-glyph");
  if (glyph) {
    const newIcon = config.icon || "📦";
    // Обновляем иконку только если она изменилась
    if (glyph.textContent !== newIcon) {
      glyph.textContent = newIcon;
    }
  }
  if (options.animate) {
    // Удаляем класс перед добавлением для перезапуска анимации
    shell.classList.remove("case-tier-surge");
    // Используем requestAnimationFrame для плавной анимации
    requestAnimationFrame(() => {
      shell.classList.add("case-tier-surge");
      // Убираем will-change после завершения анимации
      setTimeout(() => {
        shell.style.willChange = "auto";
      }, 600);
    });
  }
}

function spawnCaseTapSparks(display, accentColor) {
  if (CASE_EFFECTS_REDUCED) {
    return;
  }
  if (!display) {
    return;
  }
  const shell = display.querySelector(".case-visual-shell");
  const sparksHolder = shell?.querySelector(".case-visual-sparks");
  if (!sparksHolder) {
    return;
  }
  const sparkCount = 3;
  for (let i = 0; i < sparkCount; i += 1) {
    const spark = document.createElement("span");
    spark.className = "case-tap-spark";
    spark.style.setProperty("--spark-translate-x", `${(Math.random() - 0.5) * 90}px`);
    spark.style.setProperty("--spark-translate-y", `${(Math.random() - 0.5) * 90}px`);
    if (accentColor) {
      spark.style.setProperty("--spark-color", accentColor);
    }
    sparksHolder.appendChild(spark);
    setTimeout(() => spark.remove(), 700);
  }
}

function spawnCaseUpgradeBurst(tier) {
  if (CASE_EFFECTS_REDUCED) {
    return;
  }
  const display = document.getElementById("case-display-tapping");
  if (!display) {
    return;
  }
  const burst = document.createElement("span");
  burst.className = "case-upgrade-burst";
  burst.dataset.tier = String(tier);
  display.appendChild(burst);
  setTimeout(() => burst.remove(), 1000);
}

function updateCaseTapIndicators() {
  if (!caseDOMCache.tapIndicators) {
    const indicatorsRoot = document.getElementById("case-tap-indicators");
    if (!indicatorsRoot) {
      return;
    }
    caseDOMCache.tapIndicators = Array.from(indicatorsRoot.querySelectorAll(".tap-indicator"));
  }
  
  if (!caseOpeningState || !caseDOMCache.tapIndicators) {
    return;
  }
  
  // Используем DocumentFragment для батчинга изменений
  const fragment = document.createDocumentFragment();
  const changes = [];
  
  caseDOMCache.tapIndicators.forEach((indicator, index) => {
    const wasCompleted = indicator.classList.contains("completed");
    const wasUpgraded = indicator.classList.contains("upgraded");
    const wasActive = indicator.classList.contains("active");
    
    const isCompleted = index < caseOpeningState.tapResults.length;
    const isUpgraded = isCompleted && caseOpeningState.upgradeFlags[index];
    const isActive = caseOpeningState.stage === "taps" && index === caseOpeningState.tapResults.length;
    
    // Сохраняем изменения для батчинга
    if (wasCompleted !== isCompleted || wasUpgraded !== isUpgraded || wasActive !== isActive) {
      changes.push({ indicator, isCompleted, isUpgraded, isActive });
    }
  });
  
  // Применяем все изменения в одном кадре
  if (changes.length > 0) {
    requestAnimationFrame(() => {
      changes.forEach(({ indicator, isCompleted, isUpgraded, isActive }) => {
        indicator.classList.toggle("completed", isCompleted);
        indicator.classList.toggle("upgraded", isUpgraded);
        indicator.classList.toggle("active", isActive);
      });
    });
  }
  
  if (!caseDOMCache.tapCurrent) {
    caseDOMCache.tapCurrent = document.getElementById("case-tap-current");
  }
  if (caseDOMCache.tapCurrent) {
    const newText = String(Math.min(caseOpeningState.tapResults.length, CASE_TAP_LIMIT));
    if (caseDOMCache.tapCurrent.textContent !== newText) {
      caseDOMCache.tapCurrent.textContent = newText;
    }
  }
  
  updateCaseProgressMeter();
}

function updateCaseProgressMeter() {
  if (!caseDOMCache.progressMeter) {
    caseDOMCache.progressMeter = document.getElementById("case-progress-meter");
  }
  if (!caseDOMCache.progressMeter || !caseOpeningState) {
    return;
  }
  
  const progress = Math.min(caseOpeningState.tapResults.length / CASE_TAP_LIMIT, 1);
  const width = `${progress * 100}%`;
  
  // Обновляем только если изменилось
  if (caseDOMCache.progressMeter.style.width !== width) {
    requestAnimationFrame(() => {
      caseDOMCache.progressMeter.style.width = width;
    });
  }
  
  if (!CASE_EFFECTS_REDUCED && progress > 0) {
    // Restart CSS animation for glow pulse только если нужно
    if (!caseDOMCache.progressMeter.classList.contains("is-pulsing")) {
      caseDOMCache.progressMeter.classList.add("is-pulsing");
      clearTimeout(caseDOMCache.progressMeter._pulseTimeout);
      caseDOMCache.progressMeter._pulseTimeout = setTimeout(() => {
        caseDOMCache.progressMeter.classList.remove("is-pulsing");
      }, 320);
    }
  } else {
    caseDOMCache.progressMeter.classList.remove("is-pulsing");
  }
}

function animateCaseHintChange(hintEl, newText) {
  if (!hintEl) {
    return;
  }
  if (hintEl.dataset.currentText === newText) {
    return;
  }
  hintEl.dataset.currentText = newText;
  if (CASE_EFFECTS_REDUCED) {
    hintEl.textContent = newText;
    hintEl.classList.remove("is-changing", "hint-enter");
    return;
  }
  hintEl.classList.add("is-changing");
  if (caseHintTransitionTimeout) {
    clearTimeout(caseHintTransitionTimeout);
  }
  caseHintTransitionTimeout = setTimeout(() => {
    hintEl.textContent = newText;
    hintEl.classList.remove("is-changing");
    hintEl.classList.add("hint-enter");
    setTimeout(() => hintEl.classList.remove("hint-enter"), 240);
  }, 140);
}

function updateCaseTapHint() {
  if (!caseDOMCache.tapHint) {
    caseDOMCache.tapHint = document.getElementById("case-tap-hint");
  }
  if (!caseDOMCache.tapHint || !caseOpeningState) {
    return;
  }
  
  let newText = "";
  if (caseOpeningState.stage !== "taps") {
    if (caseHintTransitionTimeout) {
      clearTimeout(caseHintTransitionTimeout);
      caseHintTransitionTimeout = null;
    }
    newText = "Заряжай кейс серией ударов";
    if (caseDOMCache.tapHint.textContent !== newText) {
      caseDOMCache.tapHint.dataset.currentText = newText;
      caseDOMCache.tapHint.textContent = newText;
      caseDOMCache.tapHint.classList.remove("is-changing", "hint-enter");
    }
    return;
  }
  
  const nextTap = caseOpeningState.tapResults.length + 1;
  if (nextTap > CASE_TAP_LIMIT) {
    newText = "Кейс готов к раскрытию!";
  } else {
    const tapMessages = [
      "Запусти первый импульс — пробуди артефакт!",
      "Энергия накапливается, держи темп!",
      "Коллектор сияет все ярче — еще немного!",
      "Финальный удар зарядит кейс до предела!"
    ];
    const message = tapMessages[nextTap - 1] || "Продолжай, и кейс вспыхнет!";
    newText = `Тап ${nextTap}/${CASE_TAP_LIMIT}. ${message}`;
  }
  
  // Обновляем только если текст изменился
  if (caseDOMCache.tapHint.dataset.currentText !== newText) {
    animateCaseHintChange(caseDOMCache.tapHint, newText);
  }
}

function triggerCaseTapEffects(event) {
  const display = document.getElementById("case-display-tapping");
  if (!display) {
    return;
  }
  
  // Воспроизводим звук тапа при каждом нажатии
  playCaseTapSound();
  
  display.classList.add("case-hit");
  setTimeout(() => display.classList.remove("case-hit"), 350);

  const ripple = document.createElement("span");
  ripple.className = "case-tap-ripple";
  const rect = display.getBoundingClientRect();
  const x = event ? event.clientX - rect.left : rect.width / 2;
  const y = event ? event.clientY - rect.top : rect.height / 2;
  ripple.style.left = `${x}px`;
  ripple.style.top = `${y}px`;
  display.appendChild(ripple);
  setTimeout(() => ripple.remove(), 600);

  const accent = getCaseTierConfig(caseOpeningState?.currentTier || 1)?.accent;
  spawnCaseTapSparks(display, accent);
}

function playCaseRewardsTransition() {
  const container = document.querySelector(".case-opening-content");
  if (!container) {
    return;
  }
  const transitionEl = document.createElement("div");
  transitionEl.className = "case-stage-transition";
  transitionEl.innerHTML = `<span class="case-stage-transition-flare"></span>`;
  container.appendChild(transitionEl);
  setTimeout(() => transitionEl.remove(), 650);
}

function toggleUpgradeNotification(show, tier) {
  // Удалено - используем цветовую индикацию вместо всплывашки
}

function flashUpgradeNotification(tier) {
  // Показываем цветовую индикацию через изменение цвета кейса
  const displays = [
    document.getElementById("case-display"),
    document.getElementById("case-display-tapping")
  ];
  
  displays.forEach(display => {
    if (!display) return;
    const shell = display.querySelector(".case-visual-shell");
    if (shell) {
      shell.classList.add("case-tier-upgraded");
      setTimeout(() => {
        shell.classList.remove("case-tier-upgraded");
      }, 800);
    }
  });
  
  // Обновляем бейдж с анимацией
  const badges = [
    document.getElementById("case-tier-badge"),
    document.getElementById("case-tier-badge-tapping")
  ];
  
  badges.forEach(badge => {
    if (!badge) return;
    badge.classList.add("tier-upgraded");
    setTimeout(() => {
      badge.classList.remove("tier-upgraded");
    }, 800);
  });
  
  spawnCaseUpgradeBurst(tier);
}

async function startCaseTapping() {
  if (!caseOpeningState) {
    return;
  }
  
  // Воспроизводим звук начала открытия кейса
  playCaseOpenedSound();
  
  caseOpeningState.tapResults = [];
  caseOpeningState.upgradeFlags = [];
  caseOpeningState.tapInProgress = false;
  setCaseStage("taps");
  updateCaseStageInfo();
  const startBtn = document.getElementById("case-open-start-btn");
  if (startBtn) {
    startBtn.disabled = true;
  }
}

async function handleCaseTap(event) {
  if (!caseOpeningState || caseOpeningState.stage !== "taps") {
    return;
  }
  if (caseOpeningState.tapInProgress || caseOpeningState.finalizing) {
    return;
  }
  if (caseOpeningState.tapResults.length >= CASE_TAP_LIMIT) {
    await finalizeCaseOpening();
    return;
  }
  const authData = resolveUserId();
  if (!authData) {
    showNotification("Ошибка авторизации", "error");
    return;
  }
  const tapNumber = caseOpeningState.tapResults.length + 1;
  caseOpeningState.tapInProgress = true;
  triggerCaseTapEffects(event);
  try {
    const url = appendAuthParams("/api/cases/tap", authData);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_case_id: caseOpeningState.userCase.user_case_id,
        current_tier: caseOpeningState.currentTier,
        tap_number: tapNumber,
      }),
    });
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const result = await response.json();
    if (!result.success) {
      showNotification("Не удалось обработать тап", "error");
      return;
    }
    caseOpeningState.currentTier = result.new_tier;
    caseOpeningState.userCase.tier = result.new_tier;
    caseOpeningState.tapResults.push(result.new_tier);
    caseOpeningState.upgradeFlags.push(Boolean(result.upgraded));
    if (!CASE_EFFECTS_REDUCED) {
      caseOpeningState.shouldPulseVisual = true;
    }
    updateCaseStageInfo();
    updateCaseTapHint();
    if (result.upgraded) {
      // Не обновляем сетку кейсов при каждом апгрейде - только после финализации
      // renderCasesGrid(); // Убрано для оптимизации
      // Показываем цветовую индикацию вместо всплывашки
      flashUpgradeNotification(result.new_tier);
    }
    if (caseOpeningState.tapResults.length >= CASE_TAP_LIMIT) {
      await finalizeCaseOpening();
    }
  } catch (error) {
    console.error("Ошибка тапа кейса:", error);
    showNotification("Не удалось выполнить тап", "error");
  } finally {
    caseOpeningState.tapInProgress = false;
  }
}

async function finalizeCaseOpening() {
  if (!caseOpeningState || caseOpeningState.finalizing) {
    return;
  }
  if (caseOpeningState.tapResults.length < CASE_TAP_LIMIT) {
    return;
  }
  const authData = resolveUserId();
  if (!authData) {
    showNotification("Ошибка авторизации", "error");
    return;
  }
  caseOpeningState.finalizing = true;
  try {
    const url = appendAuthParams("/api/cases/open", authData);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_case_id: caseOpeningState.userCase.user_case_id,
        tap_results: caseOpeningState.tapResults,
      }),
    });
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const result = await response.json();
    if (!result.success) {
      const errorMsg = result.error || "Не удалось открыть кейс";
      console.error("Ошибка открытия кейса:", errorMsg, result);
      showNotification(`Ошибка: ${errorMsg}`, "error");
      caseOpeningState.finalizing = false;
      return;
    }
    
    // Проверяем, что награды присутствуют
    if (!result.rewards) {
      console.error("Награды отсутствуют в ответе:", result);
      showNotification("Ошибка: награды не получены", "error");
      caseOpeningState.finalizing = false;
      return;
    }
    
    caseOpeningState.tapResults = result.tap_results || caseOpeningState.tapResults;
    caseOpeningState.currentTier = result.final_tier || caseOpeningState.currentTier;
    renderCaseRewards(result.rewards);
    setCaseStage("rewards");
    
    // Воспроизводим звук показа наград
    playCaseRewardsSound();
    
    // Воспроизводим звук показа наград
    playCaseRewardsSound();
    
    // Обновляем профиль и кейсы после успешного открытия (с задержкой для плавности)
    requestAnimationFrame(async () => {
      try {
        await loadProfile(authData);
        // Обновляем сетку кейсов только после финализации
        await loadUserCases(true);
        renderCasesGrid();
      } catch (loadError) {
        console.error("Ошибка обновления данных после открытия кейса:", loadError);
        // Не показываем ошибку пользователю, так как награды уже выданы
      }
    });
  } catch (error) {
    console.error("Ошибка финального открытия кейса:", error);
    showNotification(`Ошибка выдачи наград: ${error.message || "неизвестная ошибка"}`, "error");
  } finally {
    caseOpeningState.finalizing = false;
  }
}

function renderCaseRewards(rewards) {
  const list = document.getElementById("case-rewards-list");
  const noteEl = document.getElementById("case-rewards-note");
  if (!list) {
    return;
  }
  if (!rewards) {
    list.innerHTML = "";
    if (noteEl) {
      noteEl.textContent = "";
    }
    return;
  }
  const items = [];
  if (rewards.coins) {
    items.push(buildRewardItemHtml({
      icon: CASE_REWARD_ICONS.coins,
      title: `+${rewards.coins} монет`,
      description: "Монеты уже добавлены на баланс",
    }));
  }
  if (rewards.gems) {
    items.push(buildRewardItemHtml({
      icon: CASE_REWARD_ICONS.gems,
      title: `+${rewards.gems} гемов`,
      description: "Премиальная валюта",
    }));
  }
  if (rewards.limited_shards) {
    items.push(buildRewardItemHtml({
      icon: CASE_REWARD_ICONS.limited_shards,
      title: `+${rewards.limited_shards} осколков`,
      description: "Для лимитированных героев",
    }));
  }
  (rewards.cards || []).forEach(card => {
    items.push(buildRewardItemHtml({
      icon: CASE_REWARD_ICONS.card,
      title: `${card.card_name || "Новая карта"}`,
      description: card.is_new ? "Новая карта в коллекции" : "Карта получена",
      rarity: card.rarity,
    }));
  });
  (rewards.particles || []).forEach(particle => {
    const isJackpot = particle.rarity === "common" && particle.particles >= 500 && rewards.jackpot;
    items.push(buildRewardItemHtml({
      icon: CASE_REWARD_ICONS.particles,
      title: `${particle.card_name || "Дубликат"}`,
      description: `+${particle.particles} частиц`,
      rarity: particle.rarity,
      extraClass: isJackpot ? "jackpot" : "",
    }));
  });
  list.innerHTML = items.join("");
  if (noteEl) {
    noteEl.textContent = "Награды уже зачислены на ваш аккаунт.";
  }
}

function buildRewardItemHtml({ icon, title, description, rarity, extraClass }) {
  const rarityBadge = rarity ? `<span class="rarity-tag rarity-${rarity}">${escapeHtml(getRarityName(rarity))}</span>` : "";
  return `
    <div class="reward-item ${extraClass || ""}">
      <div class="reward-icon">${icon || "🎁"}</div>
      <div class="reward-content">
        <div class="reward-title">${escapeHtml(title)} ${rarityBadge}</div>
        ${description ? `<div class="reward-description">${escapeHtml(description)}</div>` : ""}
      </div>
    </div>
  `;
}

async function handleCaseSkip() {
  if (!caseOpeningState || caseOpeningState.skipInProgress) {
    return;
  }
  const authData = resolveUserId();
  if (!authData) {
    showNotification("Ошибка авторизации", "error");
    return;
  }
  caseOpeningState.skipInProgress = true;
  document.getElementById("case-skip-btn")?.setAttribute("disabled", "true");
  document.getElementById("case-skip-btn-taps")?.setAttribute("disabled", "true");
  try {
    const url = appendAuthParams("/api/cases/skip", authData);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_case_id: caseOpeningState.userCase.user_case_id,
      }),
    });
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    const result = await response.json();
    if (!result.success) {
      showNotification("Не удалось пропустить анимацию", "error");
      return;
    }
    caseOpeningState.tapResults = result.tap_results || [];
    caseOpeningState.currentTier = result.final_tier || caseOpeningState.currentTier;
    renderCaseRewards(result.rewards);
    setCaseStage("rewards");
    
    // Воспроизводим звук показа наград
    playCaseRewardsSound();
    await loadProfile(authData);
    await loadUserCases(true);
  } catch (error) {
    console.error("Ошибка пропуска открытия кейса:", error);
    showNotification("Ошибка пропуска", "error");
  } finally {
    caseOpeningState.skipInProgress = false;
    document.getElementById("case-skip-btn")?.removeAttribute("disabled");
    document.getElementById("case-skip-btn-taps")?.removeAttribute("disabled");
  }
}

function closeCaseModal() {
  const modal = document.getElementById("case-opening-modal");
  if (modal) {
    modal.style.display = "none";
  }
  document.body.classList.remove("case-opening-active");
  caseOpeningState = null;
  document.getElementById("case-skip-btn")?.removeAttribute("disabled");
  document.getElementById("case-skip-btn-taps")?.removeAttribute("disabled");
  document.getElementById("case-open-start-btn")?.removeAttribute("disabled");
  const selectStage = document.getElementById("case-opening-stage-select");
  const tapsStage = document.getElementById("case-opening-stage-taps");
  const rewardsStage = document.getElementById("case-opening-stage-rewards");
  [selectStage, tapsStage, rewardsStage].forEach((stage, index) => {
    if (!stage) return;
    const isSelect = index === 0;
    stage.style.display = isSelect ? "block" : "none";
    stage.classList.toggle("active", isSelect);
  });
  document.querySelector(".case-opening-content")?.setAttribute("data-case-stage", "select");
  if (caseHintTransitionTimeout) {
    clearTimeout(caseHintTransitionTimeout);
    caseHintTransitionTimeout = null;
  }
}

function openCasesShortcut() {
  const collectionNav = document.querySelector('.nav-item[data-section="collection"]');
  const alreadyActive = collectionNav?.classList.contains("active");
  if (collectionNav) {
    collectionNav.click();
  }
  setTimeout(() => {
    activateCollectionTab("cases");
    initCasesTab(true);
  }, alreadyActive ? 0 : 200);
}

// ==================== РАБОТА С КАРТАМИ И КОЛОДАМИ ====================

let collectionTabsBound = false;

function activateCollectionTab(tabName = "cards") {
  const tabs = document.querySelectorAll(".collection-tab");
  const contents = document.querySelectorAll(".collection-tab-content");
  tabs.forEach(tab => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
  contents.forEach(content => {
    const targetId = `${tabName}-tab-content`;
    content.classList.toggle("active", content.id === targetId);
  });
}

function setupCollectionTabs() {
  if (collectionTabsBound) {
    return;
  }
  const tabs = document.querySelectorAll(".collection-tab");
  if (!tabs.length) {
    return;
  }
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.dataset.tab || "cards";
      activateCollectionTab(targetTab);
      if (targetTab === "cases") {
        initCasesTab(true);
      }
    });
  });
  collectionTabsBound = true;
}

let userCards = [];
let allCardsMap = new Map();
let deckPresets = [];
let currentPresetNumber = null; // При первом заходе будет установлена первая колода
let cardImageCache = new Map(); // Кэш для URL изображений карт
let currentCardSort = "power"; // Текущая категория сортировки (power, level, upgrade, date, name)
let currentSortDirection = "desc"; // Направление сортировки: "desc" (🔽) или "asc" (🔼)
let currentRarityFilter = "all"; // Текущий фильтр по редкости

// Получить URL изображения карты (теперь API проксирует изображение напрямую)
async function getCardImageUrl(imageFileId) {
  if (!imageFileId) return null;
  
  // Проверяем кэш
  if (cardImageCache.has(imageFileId)) {
    return cardImageCache.get(imageFileId);
  }
  
  try {
    // API теперь возвращает само изображение, а не JSON с URL
    // Используем URL API напрямую для кэширования
    const imageUrl = `/api/cards/image?file_id=${encodeURIComponent(imageFileId)}`;
    cardImageCache.set(imageFileId, imageUrl);
    return imageUrl;
  } catch (error) {
    console.error("Ошибка получения изображения карты:", error);
  }
  
  return null;
}

// Сортировка карт
function sortCards(cards, sortBy) {
  const sorted = [...cards];
  
  // Определяем направление сортировки
  const [sortType, direction] = sortBy.includes("-") ? sortBy.split("-") : [sortBy, "desc"];
  const isAsc = direction === "asc";
  
  switch (sortType) {
    case "power":
      sorted.sort((a, b) => {
        const aPower = calculateCardPower(a.power || 0, a.level || 1);
        const bPower = calculateCardPower(b.power || 0, b.level || 1);
        return isAsc ? aPower - bPower : bPower - aPower;
      });
      break;
    case "level":
      sorted.sort((a, b) => {
        const aLevel = a.level || 1;
        const bLevel = b.level || 1;
        if (bLevel !== aLevel) return isAsc ? aLevel - bLevel : bLevel - aLevel;
        // Если уровень одинаковый, сортируем по мощи
        const aPower = calculateCardPower(a.power || 0, aLevel);
        const bPower = calculateCardPower(b.power || 0, bLevel);
        return isAsc ? aPower - bPower : bPower - aPower;
      });
      break;
    case "rarity":
      const rarityOrder = { common: 1, rare: 2, epic: 3, legendary: 4, mythic: 5, divine: 6, limited: 7 };
      sorted.sort((a, b) => {
        const aRarity = rarityOrder[a.rarity] || 0;
        const bRarity = rarityOrder[b.rarity] || 0;
        if (aRarity !== bRarity) return isAsc ? aRarity - bRarity : bRarity - aRarity;
        // Если редкость одинаковая, сортируем по мощи
        const aPower = calculateCardPower(a.power || 0, a.level || 1);
        const bPower = calculateCardPower(b.power || 0, b.level || 1);
        return isAsc ? aPower - bPower : bPower - aPower;
      });
      break;
    case "upgrade":
      sorted.sort((a, b) => {
        const aLevel = a.level || 1;
        const bLevel = b.level || 1;
        const aParticles = a.particles || 0;
        const bParticles = b.particles || 0;
        const aRequired = calculateUpgradeParticles(a.rarity, aLevel);
        const bRequired = calculateUpgradeParticles(b.rarity, bLevel);
        const aProgress = aRequired > 0 ? aParticles / aRequired : 0;
        const bProgress = bRequired > 0 ? bParticles / bRequired : 0;
        return isAsc ? aProgress - bProgress : bProgress - aProgress;
      });
      break;
    case "date":
      sorted.sort((a, b) => {
        const aDate = a.obtained_at ? new Date(a.obtained_at).getTime() : 0;
        const bDate = b.obtained_at ? new Date(b.obtained_at).getTime() : 0;
        return isAsc ? aDate - bDate : bDate - aDate;
      });
      break;
    case "name":
      sorted.sort((a, b) => {
        const aName = (a.name || "").toLowerCase();
        const bName = (b.name || "").toLowerCase();
        return isAsc ? aName.localeCompare(bName) : bName.localeCompare(aName);
      });
      break;
  }
  
  return sorted;
}

// Рассчитать необходимые частицы для улучшения карты
function calculateUpgradeParticles(rarity, level) {
  // Базовые значения частиц для каждого перехода уровня (для Обычной карты)
  const baseParticlesByLevel = {
    1: 5,    // 1 → 2
    2: 10,   // 2 → 3
    3: 20,   // 3 → 4
    4: 40,   // 4 → 5
    5: 80,   // 5 → 6
    6: 160,  // 6 → 7
    7: 320,  // 7 → 8
    8: 640,  // 8 → 9
    9: 2500  // 9 → 10 (ценовой обрыв)
  };
  
  // Множители частиц по редкостям
  const rarityMultipliers = {
    common: 1.0,
    rare: 1.3,
    start: 1.4,
    superrare: 1.6,
    epic: 2.0,
    legendary: 3.0,
    mythic: 4.0,
    divine: 5.0,
    limited: 6.0
  };
  
  // Получаем базовое значение для текущего уровня
  const baseParticles = baseParticlesByLevel[level] || 5;
  
  // Получаем множитель редкости
  const rarityMult = rarityMultipliers[rarity] || 1.0;
  
  // Вычисляем финальное количество частиц (округление вверх)
  return Math.ceil(baseParticles * rarityMult);
}

function calculateUpgradeCoins(rarity, level) {
  // Базовые значения монет для каждого перехода уровня (для Обычной карты)
  const baseCoinsByLevel = {
    1: 50,      // 1 → 2
    2: 150,     // 2 → 3
    3: 400,     // 3 → 4
    4: 900,     // 4 → 5
    5: 2000,    // 5 → 6
    6: 4500,    // 6 → 7
    7: 8000,    // 7 → 8
    8: 13000,   // 8 → 9
    9: 40000    // 9 → 10 (ценовой обрыв)
  };
  
  // Множители монет по редкостям
  const rarityMultipliers = {
    common: 1.0,
    rare: 1.2,
    start: 1.3,
    superrare: 1.5,
    epic: 2.0,
    legendary: 3.0,
    mythic: 4.0,
    divine: 5.0,
    limited: 6.0
  };
  
  // Получаем базовое значение для текущего уровня
  const baseCoins = baseCoinsByLevel[level] || 50;
  
  // Получаем множитель редкости
  const rarityMult = rarityMultipliers[rarity] || 1.0;
  
  // Вычисляем финальное количество монет (округление вверх)
  return Math.ceil(baseCoins * rarityMult);
}

// Рассчитать мощность карты с учетом уровня
// Формула: Power(n) = Power_base × 1.10^(n-1)
function calculateCardPower(basePower, level) {
  const cardLevel = level || 1;
  const powerMultiplier = Math.pow(1.10, cardLevel - 1);
  return Math.floor(basePower * powerMultiplier);
}

// Получить название редкости
function getRarityName(rarity) {
  const names = {
    start: "Стартовая",
    common: "Обычная",
    rare: "Редкая",
    superrare: "Сверхредкая",
    epic: "Эпическая",
    legendary: "Легендарная",
    mythic: "Мифическая",
    divine: "Божественная",
    limited: "Лимитированная"
  };
  return names[rarity] || rarity;
}

// Загрузка карт пользователя
async function loadUserCards() {
  const authData = resolveUserId();
  if (!authData) {
    console.error("Нет авторизации для загрузки карт");
    return;
      }
      
      try {
    let url = "/api/cards/user";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ошибка ${response.status}: ${errorText}`);
    }
    const data = await response.json();

    if (!data || !Array.isArray(data.cards)) {
      throw new Error("Некорректный формат данных от сервера: ожидался массив карт.");
    }

    userCards = data.cards || [];
    // Создаем карту для быстрого поиска (используем строковые ключи для единообразия)
    allCardsMap.clear();
    userCards.forEach(card => {
      if (card && card.id) {
        // Всегда используем строковый ключ для единообразия
        allCardsMap.set(card.id.toString(), card);
      }
    });
    
    renderCardsGrid();
  } catch (error) {
    console.error("Ошибка загрузки карт:", error);
  }
}

// Отображение карт в сетке
function renderCardsGrid() {
  const grid = document.getElementById("cards-grid");
  if (!grid) return;

  // Получаем список карт, используемых в текущей колоде
  const deckGrid = document.getElementById("deck-grid");
  const usedCardIds = new Set();
  if (deckGrid) {
    const slots = deckGrid.querySelectorAll(".deck-slot");
    slots.forEach(slot => {
      const cardId = slot.dataset.cardId;
      if (cardId && cardId !== "") {
        // Добавляем как строку для надежности сравнения
        usedCardIds.add(cardId.toString());
      }
    });
  }

  // Фильтруем карты - показываем только те, которые не используются в текущей колоде
  // Игрок всегда имеет только одну карту одного айди, поэтому исключаем по card.id
  // Также убираем дубликаты по card.id на случай, если они есть в массиве
  const seenCardIds = new Set();
  let availableCards = userCards.filter(card => {
    if (!card || !card.id) return false;
    const cardIdStr = card.id.toString();
    // Проверяем, не является ли это дубликатом
    if (seenCardIds.has(cardIdStr)) {
      return false; // Пропускаем дубликат
    }
    seenCardIds.add(cardIdStr);
    // Проверяем, не используется ли карта с таким id в колоде
    return !usedCardIds.has(cardIdStr);
  });
  
  // Фильтруем по редкости
  if (currentRarityFilter !== "all") {
    availableCards = availableCards.filter(card => card.rarity === currentRarityFilter);
  }
  
  // Сортируем карты
  const sortBy = `${currentCardSort}-${currentSortDirection}`;
  availableCards = sortCards(availableCards, sortBy);

  if (availableCards.length === 0 && userCards.length === 0) {
    grid.innerHTML = `
      <div class="empty-state-card-collection">
        <div class="empty-state-content">
          <div class="empty-icon">🃏</div>
          <div class="empty-info">
            <h3 class="empty-title">Ваша коллекция пуста</h3>
            <p class="empty-text">Получайте карты из кейсов и начинайте собирать свою непобедимую колоду!</p>
          </div>
        </div>
      </div>
    `;
    return;
  }

  if (availableCards.length === 0) {
    grid.innerHTML = `
      <div class="empty-state-card-collection">
        <div class="empty-state-content">
          <div class="empty-icon">✅</div>
          <div class="empty-info">
            <h3 class="empty-title">Все карты в колоде</h3>
            <p class="empty-text">Вы использовали все доступные карты в текущей колоде!</p>
          </div>
        </div>
      </div>
    `;
    return;
  }

  // Рендерим карты асинхронно с изображениями
  grid.innerHTML = '';
  
  // Используем Promise.all для параллельной загрузки изображений
  Promise.all(availableCards.map(async (card) => {
    const level = card.level || 1;
    const basePower = card.power || 0;
    const currentPower = calculateCardPower(basePower, level);
    const particles = card.particles || 0;
    // Для карт максимального уровня показываем прогресс для последнего апгрейда (9->10)
    const requiredParticles = level < 10 ? calculateUpgradeParticles(card.rarity, level) : calculateUpgradeParticles(card.rarity, 9);
    // Убеждаемся, что прогресс не превышает 100% и правильно рассчитывается
    // Если частиц больше или равно требуемым, показываем 100%
    // Если requiredParticles равен 0, но есть частицы, показываем 100%
    let progressPercent = 0;
    if (requiredParticles > 0) {
      progressPercent = Math.min(Math.max((particles / requiredParticles) * 100, 0), 100);
    } else if (particles > 0) {
      progressPercent = 100;
    }
    // Проверяем, можно ли улучшить карту (только по частицам, монеты не учитываем)
    const canUpgradeByParticles = level < 10 && particles >= requiredParticles;
    const isMaxLevel = level >= 10;
    
    // Получаем URL изображения
    const imageUrl = card.image_file_id ? await getCardImageUrl(card.image_file_id) : null;
    const imageSrc = imageUrl || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='300' viewBox='0 0 200 300'%3E%3Crect fill='%232d1b4e' width='200' height='300'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-size='80' fill='%23c084fc'%3E🃏%3C/text%3E%3C/svg%3E";
    
    const cardElement = document.createElement("div");
    let cardClasses = "card-item draggable-card";
    if (canUpgradeByParticles) {
      cardClasses += " card-upgradeable";
    }
    if (isMaxLevel) {
      cardClasses += " card-max-level";
    }
    cardElement.className = cardClasses;
    cardElement.draggable = true;
    cardElement.dataset.cardId = card.id;
    cardElement.dataset.rarity = card.rarity;
    // Предотвращаем контекстное меню Telegram при долгом нажатии
    cardElement.addEventListener("contextmenu", (e) => {
      e.preventDefault();
    });
    // Предотвращаем выделение текста при drag
    cardElement.style.userSelect = "none";
    cardElement.style.webkitUserSelect = "none";
    cardElement.innerHTML = `
      <div class="card-item-image-container">
        <img src="${imageSrc}" alt="${card.name}" class="card-item-image" draggable="false" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'300\\' viewBox=\\'0 0 200 300\\'%3E%3Crect fill=\\'%232d1b4e\\' width=\\'200\\' height=\\'300\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' font-size=\\'80\\' fill=\\'%23c084fc\\'%3E🃏%3C/text%3E%3C/svg%3E';" />
        ${level > 1 ? `<div class="card-item-level-badge ${isMaxLevel ? 'card-max-level-badge' : ''}">Lv.${level}</div>` : ''}
        <div class="card-upgrade-progress-bottom">
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${Math.max(0, Math.min(100, progressPercent))}%"></div>
          </div>
          <div class="progress-text">${particles}/${requiredParticles} ⚡</div>
        </div>
      </div>
      <div class="card-item-info ${isMaxLevel ? 'card-max-level-info' : ''}">
        <div class="card-name ${isMaxLevel ? 'card-max-level-name' : ''}">${card.name}</div>
        <div class="card-power">⚔️ ${currentPower}</div>
      </div>
    `;
    
    // Добавляем обработчик клика для открытия детального просмотра (не при drag)
    let dragStarted = false;
    cardElement.addEventListener("dragstart", (e) => {
      dragStarted = true;
      // Предотвращаем контекстное меню Telegram
      e.dataTransfer.effectAllowed = "move";
      // Добавляем пустое изображение для drag preview
      const dragImage = new Image();
      dragImage.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
      e.dataTransfer.setDragImage(dragImage, 0, 0);
    });
    cardElement.addEventListener("dragend", () => {
      setTimeout(() => { dragStarted = false; }, 100);
    });
    cardElement.addEventListener("click", async (e) => {
      if (!dragStarted) {
        e.stopPropagation();
        await openCardDetail(card);
      }
    });
    
    return cardElement;
  })).then(cardElements => {
    cardElements.forEach(element => grid.appendChild(element));
    // Инициализируем drag and drop после добавления всех карт
    initDragAndDrop();
  });
}

// Загрузка пресетов колод
async function loadDeckPresets() {
  const authData = resolveUserId();
  if (!authData) {
    console.error("Нет авторизации для загрузки пресетов");
    return;
  }

  try {
    let url = "/api/deck/presets";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Ошибка ${response.status}`);
    const data = await response.json();

    deckPresets = data.presets || [];
    
    // Если нет пресетов, создаем пресет по умолчанию
    if (deckPresets.length === 0) {
      try {
        let createUrl = "/api/deck/presets/create";
        if (typeof authData === "string") {
          createUrl += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          createUrl += `?user_id=${authData}`;
        }

        const createResponse = await fetch(createUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ preset_name: "Моя колода" })
        });

        const createResult = await createResponse.json();
        if (createResult.success) {
          // Перезагружаем пресеты
          const reloadResponse = await fetch(url);
          if (reloadResponse.ok) {
            const reloadData = await reloadResponse.json();
            deckPresets = reloadData.presets || [];
          }
        }
      } catch (error) {
        console.error("Ошибка создания пресета по умолчанию:", error);
      }
    }
    
    // Если есть пресеты, при первом заходе всегда выбираем первую колоду
    if (deckPresets.length > 0) {
      // При первом заходе всегда выбираем первую колоду из списка
      // Если currentPresetNumber не установлен или не найден, берем первую колоду
      let activePreset = deckPresets.find(p => p.preset_number === currentPresetNumber);
      if (!activePreset) {
        // При первом заходе выбираем первую колоду
        activePreset = deckPresets[0];
      currentPresetNumber = activePreset.preset_number;
      }
      
      // Рендерим пресеты с правильным активным пресетом
      renderDeckPresets();
      
      if (allCardsMap.size > 0) {
        await loadPresetDeck(activePreset);
        // Перерисовываем сетку карт после загрузки колоды
        setTimeout(() => {
          renderCardsGrid();
        }, 150);
      } else {
        // Если карты еще не загружены, ждем их загрузки
        await loadUserCards();
        if (allCardsMap.size > 0) {
          await loadPresetDeck(activePreset);
          // Перерисовываем сетку карт после загрузки колоды
          setTimeout(() => {
            renderCardsGrid();
          }, 150);
        }
      }
    } else {
      renderDeckPresets();
    }
  } catch (error) {
    console.error("Ошибка загрузки пресетов:", error);
  }
}

// Загрузка колод для модального окна боя
async function loadBattleDecks() {
  const authData = resolveUserId();
  if (!authData) {
    console.error("Нет авторизации для загрузки колод");
    return [];
  }

  try {
    let url = "/api/deck/presets";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Ошибка ${response.status}`);
    const data = await response.json();

    return data.presets || [];
  } catch (error) {
    console.error("Ошибка загрузки колод для боя:", error);
    return [];
  }
}

// Рендер колод в модальном окне боя
function renderBattleDecks(presets) {
  const decksList = document.getElementById("decks-list");
  if (!decksList) return;

  decksList.innerHTML = "";

  if (!presets || presets.length === 0) {
    decksList.innerHTML = `
      <div style="text-align: center; padding: 20px; color: var(--chibi-text-muted);">
        У вас пока нет колод. Создайте колоду в разделе "Коллекция"
      </div>
    `;
    return;
  }

  presets.forEach((preset) => {
    const cardIds = preset.card_ids || [];
    const filledSlots = cardIds.length;
    const deckComplete = filledSlots === 9;
    
    const deckItem = document.createElement("label");
    deckItem.className = "deck-item";
    deckItem.dataset.deck = preset.preset_number;
    if (!deckComplete) {
      deckItem.classList.add("deck-incomplete");
      deckItem.title = "Колода неполная (нужно 9 карт)";
    }

    deckItem.innerHTML = `
      <input type="radio" name="battle-deck" value="${preset.preset_number}" class="deck-radio" ${!deckComplete ? "disabled" : ""}>
      <div class="deck-checkmark"></div>
      <div class="deck-name">
        ${preset.preset_name || `Колода #${preset.preset_number}`}
        <span class="deck-cards-count">(${filledSlots}/9)</span>
      </div>
    `;

    // Добавляем обработчик клика для активации кнопки
    deckItem.addEventListener("click", () => {
      if (!deckComplete) return;

      const deckId = preset.id || preset.preset_number;
      console.log("[DEBUG] Выбрана колода:", deckId);
      
      lastBattleSelection.deck = deckId;
      
      const radio = deckItem.querySelector(".deck-radio");
      if (radio) radio.checked = true;
      
      checkBattleReady();
    });

    decksList.appendChild(deckItem);
  });

  // Активируем первую полную колоду по умолчанию
  const firstCompleteRadio = decksList.querySelector('input[type="radio"]:not([disabled])');
  if (firstCompleteRadio) {
    firstCompleteRadio.checked = true;
    const presetNumber = firstCompleteRadio.value;
    const preset = presets.find(p => String(p.preset_number) === String(presetNumber));
    if (preset) {
      lastBattleSelection.deck = preset.id || preset.preset_number;
    } else {
      lastBattleSelection.deck = isNaN(presetNumber) ? presetNumber : parseInt(presetNumber);
    }
    // Сразу проверяем готовность
    checkBattleReady();
  }
}

// Отображение пресетов колод
function renderDeckPresets() {
  const list = document.getElementById("deck-presets-list");
  if (!list) return;

  if (deckPresets.length === 0) {
    list.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--chibi-text-muted);">Нет пресетов</div>';
    return;
  }

  list.innerHTML = deckPresets.map(preset => {
    const isActive = preset.preset_number === currentPresetNumber;
    return `
      <div class="deck-preset-item ${isActive ? 'active' : ''}" data-preset-number="${preset.preset_number}">
      <div class="preset-name-wrapper">
        <input type="text" class="preset-name-input" value="${preset.preset_name}" 
               data-preset-number="${preset.preset_number}" readonly />
      </div>
      <div class="preset-actions">
          <button class="preset-rename-btn" data-preset-number="${preset.preset_number}" title="Переименовать">✏️</button>
          <button class="preset-delete-btn" data-preset-number="${preset.preset_number}" title="Удалить">🗑️</button>
        </div>
      </div>
    `;
  }).join('');
    
  // Добавляем обработчики клика на пресеты
  list.querySelectorAll(".deck-preset-item").forEach(item => {
    item.addEventListener("click", (e) => {
      // Игнорируем клики на кнопки действий
      if (e.target.closest(".preset-rename-btn") || e.target.closest(".preset-delete-btn")) {
        return;
      }
      
      if (e.target.closest(".preset-name-input")) {
        const input = e.target.closest(".preset-name-input");
        if (input && !input.hasAttribute("readonly")) {
        return;
      }
      }
      const presetNumber = parseInt(item.dataset.presetNumber);
      selectPreset(presetNumber);
    });
  });

  // Обработчики переименования
  list.querySelectorAll(".preset-rename-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const presetNumber = parseInt(btn.dataset.presetNumber);
      const preset = deckPresets.find(p => p.preset_number === presetNumber);
      if (preset) {
        const input = btn.closest(".deck-preset-item")?.querySelector(".preset-name-input");
      if (input) {
        input.removeAttribute("readonly");
        input.focus();
        input.select();
        
          const finishRename = () => {
            input.setAttribute("readonly", "readonly");
            const newName = input.value.trim() || preset.preset_name;
            if (newName !== preset.preset_name) {
              renamePreset(presetNumber, newName);
          } else {
              input.value = preset.preset_name;
          }
        };
        
          input.addEventListener("blur", finishRename, { once: true });
          input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
              e.preventDefault();
              finishRename();
            } else if (e.key === "Escape") {
              input.value = preset.preset_name;
              finishRename();
            }
          });
        }
      }
    });
  });

  // Обработчики удаления
  list.querySelectorAll(".preset-delete-btn").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const presetNumber = parseInt(btn.dataset.presetNumber);
      const preset = deckPresets.find(p => p.preset_number === presetNumber);
      if (preset) {
        if (await showGameConfirm(`Вы уверены, что хотите удалить колоду "${preset.preset_name}"?`)) {
          await deletePreset(presetNumber);
        }
      }
    });
  });
}

// Выбор пресета
async function selectPreset(presetNumber) {
  currentPresetNumber = presetNumber;
  const preset = deckPresets.find(p => p.preset_number === presetNumber);
  if (preset) {
    loadPresetDeck(preset);
    renderDeckPresets();
  }
}

// Загрузка колоды из пресета
async function loadPresetDeck(preset) {
  const deckGrid = document.getElementById("deck-grid");
  if (!deckGrid) return;

  const slots = deckGrid.querySelectorAll(".deck-slot");
  const cardIds = [
    preset.card_slot_1,
    preset.card_slot_2,
    preset.card_slot_3,
    preset.card_slot_4,
    preset.card_slot_5,
    preset.card_slot_6,
    preset.card_slot_7,
    preset.card_slot_8,
    preset.card_slot_9
  ];

  // Сначала очищаем все слоты
  slots.forEach(slot => {
    slot.dataset.cardId = "";
    renderEmptySlot(slot);
  });

  // Затем загружаем карты последовательно
  for (let index = 0; index < slots.length; index++) {
    const slot = slots[index];
    const cardId = cardIds[index];
    if (cardId) {
      // Пробуем найти карту по cardId (может быть число или строка)
      const cardIdStr = cardId.toString();
      let card = allCardsMap.get(cardIdStr);
      
      // Если не найдена, пробуем найти по числовому ключу (для обратной совместимости)
      if (!card && !isNaN(cardId)) {
        const cardIdNum = parseInt(cardId);
        card = allCardsMap.get(cardIdNum.toString());
      }
      
      // Если не найдена, ищем в массиве userCards
      if (!card) {
        card = userCards.find(c => c.id == cardId || c.id == parseInt(cardId));
        // Если нашли, добавляем в allCardsMap для будущих поисков
      if (card) {
          allCardsMap.set(card.id.toString(), card);
        }
      }
      
      if (card) {
        slot.dataset.cardId = card.id.toString();
        await renderCardInSlot(slot, card);
        // Добавляем спецэффект при загрузке карты из пресета
        setTimeout(() => {
          slot.classList.add("card-add-effect", `rarity-${card.rarity}`);
          setTimeout(() => {
            slot.classList.remove("card-add-effect");
          }, 1000);
        }, index * 100); // Задержка для последовательного появления эффектов
      } else {
        slot.dataset.cardId = "";
        renderEmptySlot(slot);
      }
    } else {
      slot.dataset.cardId = "";
      renderEmptySlot(slot);
    }
  }
  
  updateDeckPower();
  initDragAndDrop();
  updateLastRowCentering();
  // Перерисовываем сетку карт после загрузки колоды, чтобы исключить карты из колоды
  // Добавляем небольшую задержку, чтобы все карты успели загрузиться в слоты
  setTimeout(() => {
    renderCardsGrid();
  }, 100);
}

// Обновление центрирования последнего ряда
function updateLastRowCentering() {
  const deckGrid = document.getElementById("deck-grid");
  if (!deckGrid) return;
  
  const slots = Array.from(deckGrid.querySelectorAll(".deck-slot"));
  let filledCount = 0;
  slots.forEach(slot => {
    if (slot.dataset.cardId && slot.dataset.cardId !== "") {
      filledCount++;
    }
  });
  
  // Если заполнено ровно 7 или 8 слотов, центрируем последний ряд
  // Для 9 слотов (3x3) центрирование не требуется
  if (filledCount === 7 || filledCount === 8) {
    deckGrid.classList.add("last-row-2");
  } else {
    deckGrid.classList.remove("last-row-2");
  }
}

// Отображение карты в слоте
async function renderCardInSlot(slot, card) {
  const level = card.level || 1;
  const particles = card.particles || 0;
  const basePower = card.power || 0;
  const currentPower = calculateCardPower(basePower, level);
  // Для карт максимального уровня показываем прогресс для последнего апгрейда (9->10)
  const requiredParticles = level < 10 ? calculateUpgradeParticles(card.rarity, level) : calculateUpgradeParticles(card.rarity, 9);
  // Прогресс всегда показываем: для карт не на макс уровне - текущий прогресс, для макс уровня - 100%
  // Убеждаемся, что прогресс не превышает 100% и правильно рассчитывается
  // Если частиц больше или равно требуемым, показываем 100%
  // Если requiredParticles равен 0, но есть частицы, показываем 100%
  let progressPercent = 0;
  if (requiredParticles > 0) {
    progressPercent = Math.min(Math.max((particles / requiredParticles) * 100, 0), 100);
  } else if (particles > 0) {
    progressPercent = 100;
  }
  // Проверяем, можно ли улучшить карту (только по частицам, монеты не учитываем)
  const canUpgradeByParticles = level < 10 && particles >= requiredParticles;
  const isMaxLevel = level >= 10;
  
  // Получаем URL изображения
  const imageUrl = card.image_file_id ? await getCardImageUrl(card.image_file_id) : null;
  const imageSrc = imageUrl || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='300' viewBox='0 0 200 300'%3E%3Crect fill='%232d1b4e' width='200' height='300'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-size='80' fill='%23c084fc'%3E🃏%3C/text%3E%3C/svg%3E";
  
  slot.innerHTML = `
    <div class="deck-slot-content card-in-slot">
      <div class="card-slot-image-container">
        <img src="${imageSrc}" alt="${card.name}" class="card-slot-image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'300\\' viewBox=\\'0 0 200 300\\'%3E%3Crect fill=\\'%232d1b4e\\' width=\\'200\\' height=\\'300\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' font-size=\\'80\\' fill=\\'%23c084fc\\'%3E🃏%3C/text%3E%3C/svg%3E';" />
        <div class="card-slot-level-badge ${isMaxLevel ? 'card-max-level-badge' : ''}">${level}</div>
      <button class="card-remove-btn" onclick="removeCardFromSlot(this)"></button>
        <div class="card-upgrade-progress-bottom">
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${Math.max(0, Math.min(100, progressPercent))}%"></div>
          </div>
          <div class="progress-text">${particles}/${requiredParticles} ⚡</div>
        </div>
      </div>
    </div>
    <div class="card-slot-footer ${isMaxLevel ? 'card-max-level-footer' : ''}">
      <div class="card-slot-name ${isMaxLevel ? 'card-max-level-name' : ''}">${card.name}</div>
      <div class="card-slot-power">⚔️ ${currentPower}</div>
    </div>
  `;
  slot.classList.add("has-card", `rarity-${card.rarity}`);
  // Удаляем все классы состояния перед добавлением новых
  slot.classList.remove("card-upgradeable", "card-max-level");
  // Выделяем карту, если её можно улучшить (хватает частиц, монеты не учитываем для выделения)
  if (canUpgradeByParticles) {
    slot.classList.add("card-upgradeable");
  }
  if (isMaxLevel) {
    slot.classList.add("card-max-level");
  }
  slot.dataset.cardId = card.id;
  
  // Добавляем обработчик клика на карту для открытия суб-меню
  const cardContent = slot.querySelector(".card-in-slot");
  if (cardContent) {
    cardContent.addEventListener("click", (e) => {
      if (e.target.closest(".card-remove-btn")) return;
      e.stopPropagation();
      showCardSlotMenu(slot, card);
    });
  }
}

// Показать суб-меню для карты в слоте
function showCardSlotMenu(slot, card) {
  // Удаляем существующее меню, если есть
  const existingMenu = document.querySelector(".card-slot-menu");
  if (existingMenu) {
    existingMenu.remove();
  }
  
  const menu = document.createElement("div");
  menu.className = "card-slot-menu";
  menu.innerHTML = `
    <button class="card-menu-btn" data-action="replace">🔄 Заменить</button>
    <button class="card-menu-btn" data-action="open">👁️ Открыть</button>
  `;
  
  // Привязываем меню к слоту через position: absolute
  // Находим родительский контейнер колоды
  const deckGrid = slot.closest(".deck-grid");
  if (!deckGrid) {
    // Fallback на fixed позиционирование, если не найден контейнер
    const rect = slot.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.top = `${rect.bottom + 8}px`;
    menu.style.left = `${rect.left}px`;
    menu.style.zIndex = "10000";
    document.body.appendChild(menu);
  } else {
    // Используем absolute позиционирование относительно контейнера колоды
    deckGrid.style.position = "relative";
    menu.style.position = "absolute";
    
    // Вычисляем позицию относительно контейнера
    const slotRect = slot.getBoundingClientRect();
    const gridRect = deckGrid.getBoundingClientRect();
    
    menu.style.top = `${slotRect.bottom - gridRect.top + 8}px`;
    menu.style.left = `${slotRect.left - gridRect.left}px`;
    menu.style.zIndex = "10000";
    
    deckGrid.appendChild(menu);
    
    // Обновляем позицию при прокрутке
    const updateMenuPosition = () => {
      if (menu.parentElement) {
        const newSlotRect = slot.getBoundingClientRect();
        const newGridRect = deckGrid.getBoundingClientRect();
        menu.style.top = `${newSlotRect.bottom - newGridRect.top + 8}px`;
        menu.style.left = `${newSlotRect.left - newGridRect.left}px`;
      }
    };
    
    // Слушаем события прокрутки
    const scrollHandler = () => {
      updateMenuPosition();
    };
    
    window.addEventListener("scroll", scrollHandler, { passive: true });
    deckGrid.addEventListener("scroll", scrollHandler, { passive: true });
    
    // Сохраняем обработчик для последующего удаления
    menu._scrollHandler = scrollHandler;
  }
  
  // Обработчики действий
  menu.querySelector('[data-action="replace"]')?.addEventListener("click", () => {
    if (menu._scrollHandler) {
      window.removeEventListener("scroll", menu._scrollHandler);
      const deckGrid = slot.closest(".deck-grid");
      if (deckGrid) {
        deckGrid.removeEventListener("scroll", menu._scrollHandler);
      }
    }
    menu.remove();
    openCardSelectionModal(slot);
  });
  
  menu.querySelector('[data-action="open"]')?.addEventListener("click", () => {
    if (menu._scrollHandler) {
      window.removeEventListener("scroll", menu._scrollHandler);
      const deckGrid = slot.closest(".deck-grid");
      if (deckGrid) {
        deckGrid.removeEventListener("scroll", menu._scrollHandler);
      }
    }
    menu.remove();
    openCardDetail(card);
  });
  
  // Закрываем меню при клике вне его
  const closeMenu = (e) => {
    if (!menu.contains(e.target) && !slot.contains(e.target)) {
      if (menu._scrollHandler) {
        window.removeEventListener("scroll", menu._scrollHandler);
        const deckGrid = slot.closest(".deck-grid");
        if (deckGrid) {
          deckGrid.removeEventListener("scroll", menu._scrollHandler);
        }
      }
      menu.remove();
      document.removeEventListener("click", closeMenu);
    }
  };
  
  setTimeout(() => {
    document.addEventListener("click", closeMenu);
  }, 100);
}

// Детальный просмотр карты (как в Clash Royale)
async function openCardDetail(card) {
  const level = card.level || 1;
  const particles = card.particles || 0;
  const basePower = card.power || 0;
  const currentPower = calculateCardPower(basePower, level);
  const requiredParticles = calculateUpgradeParticles(card.rarity, level);
  const requiredCoins = level < 10 ? calculateUpgradeCoins(card.rarity, level) : 0;
  const progressPercent = Math.min((particles / requiredParticles) * 100, 100);
  const nextLevelPower = level < 10 ? calculateCardPower(basePower, level + 1) : currentPower;
  
  // Получаем текущее количество монет пользователя
  const userCoins = currentProfile?.coins || 0;
  const canUpgrade = level < 10 && particles >= requiredParticles && userCoins >= requiredCoins;
  
  // Получаем URL изображения
  const imageUrl = card.image_file_id ? await getCardImageUrl(card.image_file_id) : null;
  const imageSrc = imageUrl || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='600' viewBox='0 0 400 600'%3E%3Crect fill='%232d1b4e' width='400' height='600'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-size='150' fill='%23c084fc'%3E🃏%3C/text%3E%3C/svg%3E";
  
  const modal = document.createElement("div");
  modal.className = "modal-overlay card-detail-modal";
  modal.style.display = "flex";
  // Определяем класс редкости для модального окна
  const rarityClass = card.rarity || "common";
  const isDivine = rarityClass === "divine";
  
  modal.innerHTML = `
    <div class="modal-content card-detail-content rarity-${rarityClass}">
      <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
      <div class="card-detail-image-container rarity-${rarityClass}">
        ${isDivine ? `
          <div class="divine-glow"></div>
          <div class="divine-particles">
            <span class="particle">✨</span>
            <span class="particle">⭐</span>
            <span class="particle">✨</span>
            <span class="particle">⭐</span>
            <span class="particle">✨</span>
            <span class="particle">⭐</span>
          </div>
        ` : ''}
        <img src="${imageSrc}" alt="${card.name}" class="card-detail-image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'600\\' viewBox=\\'0 0 400 600\\'%3E%3Crect fill=\\'%232d1b4e\\' width=\\'400\\' height=\\'600\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' font-size=\\'150\\' fill=\\'%23c084fc\\'%3E🃏%3C/text%3E%3C/svg%3E';" />
        <div class="card-detail-level-badge rarity-${rarityClass}">Уровень ${level}</div>
        <div class="card-detail-rarity-decoration rarity-${rarityClass}"></div>
      </div>
      <div class="card-detail-info rarity-${rarityClass}">
        <h2 class="card-detail-name rarity-${rarityClass}">${card.name}</h2>
        <div class="card-detail-rarity ${card.rarity}">${getRarityName(card.rarity)}</div>
        ${card.description ? `<p class="card-detail-description">${card.description}</p>` : ''}
        <div class="card-detail-stats">
          <div class="card-detail-stat">
            <div class="stat-label">⚔️ Мощь</div>
            <div class="stat-value">${currentPower}</div>
            ${level < 10 ? `
              <div class="stat-next">→ ${nextLevelPower}</div>
            ` : ''}
          </div>
          <div class="card-detail-stat">
            <div class="stat-label">⚡ Частицы</div>
            <div class="stat-value ${particles >= requiredParticles ? 'ready' : ''}">${particles}/${requiredParticles}</div>
          </div>
          ${level < 10 ? `
          <div class="card-detail-stat card-detail-stat-full">
            <div class="stat-label">💰 Монеты</div>
            <div class="stat-value ${userCoins >= requiredCoins ? 'ready' : ''}">${userCoins.toLocaleString()}/${requiredCoins.toLocaleString()}</div>
        </div>
        ` : `
        <div class="card-detail-stat card-detail-stat-full">
          <div class="stat-label">⭐ Максимальный уровень</div>
          <div class="stat-value">Достигнут</div>
        </div>
        `}
        </div>
        ${level < 10 ? `
        <div class="card-detail-upgrade-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${progressPercent}%"></div>
          </div>
        </div>
        ${canUpgrade ? `
          <button class="btn-primary card-upgrade-btn" id="card-upgrade-btn">⬆️ Улучшить (${requiredCoins.toLocaleString()} 💰)</button>
        ` : `
          <div class="card-upgrade-requirements">
            ${particles < requiredParticles ? `<div class="requirement-not-met">⚠️ Недостаточно частиц: ${particles}/${requiredParticles}</div>` : ''}
            ${userCoins < requiredCoins ? `<div class="requirement-not-met">⚠️ Недостаточно монет: ${userCoins.toLocaleString()}/${requiredCoins.toLocaleString()}</div>` : ''}
          </div>
        `}
        ` : ''}
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  
  // Обработчик улучшения карты
  const upgradeBtn = modal.querySelector("#card-upgrade-btn");
  if (upgradeBtn) {
    upgradeBtn.addEventListener("click", async () => {
  const authData = resolveUserId();
  if (!authData) {
        alert("Ошибка авторизации");
    return;
  }

  try {
        let url = "/api/cards/upgrade";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ card_id: card.id })
        });
        
        const result = await response.json();
        
        if (result.success) {
          // Воспроизводим звук левел-апа
          window._playSfx?.('levelup-sound');
          
          const powerIncrease = result.power_increase || 0;
          const coinsSpent = result.coins_spent || 0;
          await showGameAlert(
            `Карта улучшена до уровня ${result.new_level}! Мощь: ${result.old_power || currentPower} → ${result.new_power} (+${powerIncrease})\nПотрачено: ${coinsSpent.toLocaleString()} 💰`, 
            "✅"
          );
          modal.remove();
          // Перезагружаем профиль для обновления монет
          const authData = resolveUserId();
          if (authData) {
            await loadProfile(authData);
          }
          // Перезагружаем карты и обновляем отображение
          await loadUserCards();
          const deckGrid = document.getElementById("deck-grid");
          if (deckGrid) {
            const slots = deckGrid.querySelectorAll(".deck-slot");
            slots.forEach(async (slot) => {
              const cardId = slot.dataset.cardId;
              if (cardId && parseInt(cardId) === card.id) {
                // Используем строковый ключ для поиска в allCardsMap
                const updatedCard = allCardsMap.get(card.id.toString());
                if (updatedCard) {
                  await renderCardInSlot(slot, updatedCard);
                }
              }
            });
            updateDeckPower();
          }
    renderCardsGrid();
        } else {
          let errorMessage = result.message || "Ошибка улучшения карты";
          if (result.error === "insufficient_coins") {
            errorMessage = `Недостаточно монет. Нужно: ${result.required?.toLocaleString() || 0}, имеется: ${result.current?.toLocaleString() || 0}`;
          } else if (result.error === "insufficient_particles") {
            errorMessage = `Недостаточно частиц. Нужно: ${result.required || 0}, имеется: ${result.current || 0}`;
          } else if (result.error === "max_level_reached") {
            errorMessage = "Карта уже достигла максимального уровня (10)";
          }
          await showGameAlert(errorMessage, "❌");
        }
  } catch (error) {
        console.error("Ошибка улучшения карты:", error);
        alert("Ошибка при улучшении карты");
      }
    });
  }
  
  // Закрытие по клику вне модального окна
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.remove();
    }
  });
}

// Отображение пустого слота
function renderEmptySlot(slot) {
  slot.innerHTML = `
    <div class="deck-slot-content slot-empty">
      <div class="slot-empty-icon">✕</div>
      <div class="slot-empty-text">Пусто</div>
    </div>
  `;
  slot.classList.remove("has-card");
  slot.classList.remove("rarity-common", "rarity-rare", "rarity-epic", "rarity-legendary", "rarity-mythic", "rarity-divine", "rarity-limited", "rarity-start");
  slot.classList.remove("card-upgradeable", "card-max-level");
  slot.dataset.cardId = "";
}

// Обновление мощности колоды
function updateDeckPower() {
  const deckGrid = document.getElementById("deck-grid");
  if (!deckGrid) return;
  
    const slots = deckGrid.querySelectorAll(".deck-slot");
  let totalPower = 0;
  
    slots.forEach(slot => {
      const cardId = slot.dataset.cardId;
    if (cardId) {
      // Используем строковый ключ для поиска в allCardsMap
      const card = allCardsMap.get(cardId.toString());
      if (card) {
        const level = card.level || 1;
        const basePower = card.power || 0;
        const currentPower = calculateCardPower(basePower, level);
        totalPower += currentPower;
      }
    }
  });
  
  const powerElement = document.getElementById("deck-power");
  if (powerElement) {
    powerElement.textContent = `Мощь: ${totalPower}`;
  }
}

// Инициализация drag and drop
function initDragAndDrop() {
  const cards = document.querySelectorAll(".draggable-card");
  const slots = document.querySelectorAll(".deck-slot");

  cards.forEach(card => {
    card.addEventListener("dragstart", (e) => {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", card.dataset.cardId);
      card.classList.add("dragging");
    });

    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
    });
  });

  slots.forEach(slot => {
    slot.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      slot.classList.add("drag-over");
    });

    slot.addEventListener("dragleave", () => {
      slot.classList.remove("drag-over");
    });

    slot.addEventListener("drop", async (e) => {
      e.preventDefault();
      slot.classList.remove("drag-over");
      
      const cardIdStr = e.dataTransfer.getData("text/plain");
      if (!cardIdStr) return;
      
      // Проверяем, не используется ли карта уже в другом слоте
      const allSlots = document.querySelectorAll(".deck-slot");
      let cardAlreadyUsed = false;
      allSlots.forEach(otherSlot => {
        if (otherSlot !== slot && otherSlot.dataset.cardId === cardIdStr) {
          cardAlreadyUsed = true;
        }
      });
      
      if (cardAlreadyUsed) {
        await showGameAlert("Эта карта уже используется в колоде", "⚠️");
        return;
      }
      
      // Используем строковый ключ для поиска в allCardsMap
      const card = allCardsMap.get(cardIdStr);
      if (card) {
        slot.dataset.cardId = cardIdStr;
        renderCardInSlot(slot, card).then(() => {
          // Добавляем спецэффект при добавлении карты
          slot.classList.add("card-add-effect", `rarity-${card.rarity}`);
          setTimeout(() => {
            slot.classList.remove("card-add-effect");
          }, 1000);
        });
        updateDeckPower();
        updateLastRowCentering();
        renderCardsGrid();
      }
    });

    // Клик на пустой слот для выбора карты
    slot.addEventListener("click", (e) => {
      if (e.target.closest(".card-remove-btn")) return;
      // Предотвращаем множественное открытие
      if (document.querySelector(".card-selection-modal")) return;
      if (!slot.dataset.cardId || slot.dataset.cardId === "") {
        openCardSelectionModal(slot);
      }
    });
  });
}

// Удаление карты из слота (глобальная функция для onclick)
window.removeCardFromSlot = function(btn) {
  const slot = btn.closest(".deck-slot");
  if (slot) {
    slot.dataset.cardId = "";
    renderEmptySlot(slot);
    updateDeckPower();
      updateLastRowCentering();
    renderCardsGrid();
      }
};

// Открытие модального окна выбора карты
async function openCardSelectionModal(slot) {
  // Проверяем, не открыто ли уже модальное окно
  const existingModal = document.querySelector(".card-selection-modal");
  if (existingModal) {
    existingModal.remove();
  }
  
  const allSlots = document.querySelectorAll(".deck-slot");
  const usedCardIds = new Set();
  allSlots.forEach(otherSlot => {
    if (otherSlot !== slot && otherSlot.dataset.cardId) {
      usedCardIds.add(otherSlot.dataset.cardId);
    }
  });
  
  const availableCards = userCards
    .filter(card => !usedCardIds.has(card.id.toString()))
    .sort((a, b) => {
      const aPower = calculateCardPower(a.power || 0, a.level || 1);
      const bPower = calculateCardPower(b.power || 0, b.level || 1);
      return bPower - aPower;
    });
  
  const modal = document.createElement("div");
  modal.className = "modal-overlay card-selection-modal";
  modal.style.display = "flex";
  modal.innerHTML = `
    <div class="modal-content modal-large">
      <div class="modal-header">
        <h2>Выберите карту</h2>
        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
      </div>
      <div class="modal-body">
        ${availableCards.length === 0 ? 
          '<div style="text-align: center; padding: 20px; color: var(--chibi-text-muted);">Все карты уже используются в колоде</div>' :
          `<div class="cards-selection-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; max-height: 400px; overflow-y: auto;">
            ${availableCards.map(card => {
              const level = card.level || 1;
              const currentPower = calculateCardPower(card.power || 0, level);
              return `
              <div class="card-selection-item" data-card-id="${card.id}" style="background: linear-gradient(135deg, rgba(45, 27, 78, 0.8), rgba(192, 132, 252, 0.1)); border: 2px solid var(--chibi-border); border-radius: 12px; padding: 12px; cursor: pointer; transition: all 0.2s;">
                <div style="font-size: 24px; text-align: center; margin-bottom: 8px;">🃏</div>
                <div style="font-size: 13px; font-weight: bold; text-align: center; margin-bottom: 4px;">${card.name}</div>
                <div class="card-slot-rarity ${card.rarity}" style="font-size: 11px; text-align: center; margin-bottom: 4px;">${getRarityName(card.rarity)}</div>
                  <div style="font-size: 12px; text-align: center; color: var(--chibi-gold);">⚔️ ${currentPower}</div>
                  ${level > 1 ? `<div style="font-size: 10px; text-align: center; color: var(--chibi-gold); margin-top: 4px;">Lv.${level}</div>` : ''}
              </div>
              `;
            }).join('')}
          </div>`
        }
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  modal.querySelectorAll(".card-selection-item").forEach(item => {
    item.addEventListener("click", async () => {
      // Используем строковый ключ для поиска в allCardsMap
      const cardIdStr = item.dataset.cardId;
      if (!cardIdStr) return;
      
      const card = allCardsMap.get(cardIdStr);
      if (!card) {
        // Если не найдено по строковому ключу, пробуем найти по числовому (для обратной совместимости)
        const cardId = parseInt(cardIdStr);
        if (cardId && !isNaN(cardId)) {
          // Пробуем найти по числовому ключу
          const cardByNum = allCardsMap.get(cardId.toString());
          if (cardByNum) {
            // Нашли карту, используем её
            slot.dataset.cardId = cardIdStr;
            await renderCardInSlot(slot, cardByNum);
            // Добавляем спецэффект при добавлении карты
            slot.classList.add("card-add-effect", `rarity-${cardByNum.rarity}`);
            setTimeout(() => {
              slot.classList.remove("card-add-effect");
            }, 1000);
            updateDeckPower();
            updateLastRowCentering();
            modal.remove();
            renderCardsGrid();
            return;
          }
        }
        console.error("Карта не найдена в allCardsMap:", cardIdStr);
        return;
      }
      
      slot.dataset.cardId = cardIdStr;
      await renderCardInSlot(slot, card);
      // Добавляем спецэффект при добавлении карты
      slot.classList.add("card-add-effect", `rarity-${card.rarity}`);
      setTimeout(() => {
        slot.classList.remove("card-add-effect");
      }, 1000);
      updateDeckPower();
      updateLastRowCentering();
      modal.remove();
      renderCardsGrid();
    });
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.remove();
    }
  });
}

// Переименование пресета
async function renamePreset(presetNumber, newName) {
  const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
    return;
  }
  
  try {
    let url = "/api/deck/presets/rename";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }
    
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_number: presetNumber,
        preset_name: newName
      })
    });

    const result = await response.json();
    
    if (result.success) {
      await loadDeckPresets();
    } else {
      alert(result.message || "Ошибка переименования колоды");
    }
  } catch (error) {
    console.error("Ошибка переименования колоды:", error);
    alert("Ошибка при переименовании колоды");
  }
}

// Удаление пресета
async function deletePreset(presetNumber) {
  const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
    return;
  }

  try {
    let url = "/api/deck/presets/delete";
      if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
      url += `?user_id=${authData}`;
      }

    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        preset_number: presetNumber
        })
      });

    const result = await response.json();
    
    if (result.success) {
      // Если удалили активный пресет, переключаемся на первый доступный
      if (currentPresetNumber === presetNumber) {
        const remainingPresets = deckPresets.filter(p => p.preset_number !== presetNumber);
        if (remainingPresets.length > 0) {
          currentPresetNumber = remainingPresets[0].preset_number;
        } else {
          currentPresetNumber = 1;
        }
      }
      await loadDeckPresets();
      } else {
      alert(result.message || "Ошибка удаления колоды");
      }
    } catch (error) {
    console.error("Ошибка удаления колоды:", error);
    alert("Ошибка при удалении колоды");
  }
}

// Сохранение текущей колоды
async function saveCurrentDeck() {
  const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
      return;
    }

  const deckGrid = document.getElementById("deck-grid");
  if (!deckGrid) {
    await showGameAlert("Колода не найдена", "❌");
      return;
  }

  const slots = deckGrid.querySelectorAll(".deck-slot");
  const cardSlots = Array.from(slots).map(slot => {
    const cardId = slot.dataset.cardId;
    return cardId && cardId !== "" ? parseInt(cardId) : null;
  });

  let preset = deckPresets.find(p => p.preset_number === currentPresetNumber);
  
  if (!preset) {
    await showGameAlert("Пресет не найден", "❌");
          return;
  }

  try {
    let url = "/api/deck/presets/save";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_number: currentPresetNumber,
        preset_name: preset.preset_name,
        card_slots: cardSlots
      })
    });

    const result = await response.json();
    
    if (result.success) {
      alert("Колода сохранена");
      await loadDeckPresets();
    } else {
      alert(result.message || "Ошибка сохранения колоды");
    }
  } catch (error) {
    console.error("Ошибка сохранения колоды:", error);
    alert("Ошибка при сохранении колоды");
  }
}

// Инициализация коллекции (вызывается при открытии раздела)
async function initCollection() {
  // Сначала загружаем карты, потом пресеты (чтобы карты были доступны при загрузке пресета)
  await loadUserCards();
  await loadDeckPresets();
  setupCollectionTabs();
  const activeTab = document.querySelector(".collection-tab.active");
  if (activeTab?.dataset.tab === "cases") {
    await initCasesTab(casesNeedRefresh);
  }
  
  // Добавляем обработчик кнопки сохранения
  const saveBtn = document.querySelector(".deck-save-btn");
  if (saveBtn) {
    // Удаляем старый обработчик, если есть
    const newSaveBtn = saveBtn.cloneNode(true);
    saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
    newSaveBtn.addEventListener("click", saveCurrentDeck);
  }
  
  // Добавляем обработчик кнопки создания пресета
  const createBtn = document.getElementById("create-preset-btn");
  if (createBtn) {
    // Удаляем старый обработчик, если есть
    const newCreateBtn = createBtn.cloneNode(true);
    createBtn.parentNode.replaceChild(newCreateBtn, createBtn);
    newCreateBtn.addEventListener("click", async () => {
  const authData = resolveUserId();
  if (!authData) {
        alert("Ошибка авторизации");
    return;
  }

  const presetCount = deckPresets.length;
  const presetName = `Колода ${presetCount + 1}`;

  try {
    let url = "/api/deck/presets/create";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ preset_name: presetName })
    });

    const result = await response.json();
    if (result.success) {
      await loadDeckPresets();
      if (result.preset_number) {
        currentPresetNumber = result.preset_number;
        const deckGrid = document.getElementById("deck-grid");
        if (deckGrid) {
          const slots = deckGrid.querySelectorAll(".deck-slot");
          slots.forEach(slot => {
            slot.dataset.cardId = "";
            renderEmptySlot(slot);
          });
          updateDeckPower();
          renderCardsGrid();
        }
        renderDeckPresets();
      }
    } else {
          alert(result.message || "Ошибка создания пресета");
    }
  } catch (error) {
    console.error("Ошибка создания пресета:", error);
        alert("Ошибка при создании пресета");
      }
    });
  }
  
  // Добавляем обработчики для админских кнопок
  const adminGetAllBtn = document.getElementById("admin-get-all-cards-btn");
  if (adminGetAllBtn) {
    const newAdminGetAllBtn = adminGetAllBtn.cloneNode(true);
    adminGetAllBtn.parentNode.replaceChild(newAdminGetAllBtn, adminGetAllBtn);
    newAdminGetAllBtn.addEventListener("click", async () => {
  const authData = resolveUserId();
  if (!authData) {
        alert("Ошибка авторизации");
    return;
  }

  try {
        let url = "/api/admin/cards/get-all";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, {
      method: "POST",
          headers: { "Content-Type": "application/json" }
    });

    const result = await response.json();
    if (result.success) {
          alert(result.message || "Все карты добавлены в вашу коллекцию");
          await loadUserCards();
    } else {
          alert(result.message || "Ошибка получения карт");
    }
  } catch (error) {
        console.error("Ошибка получения всех карт:", error);
        alert("Ошибка при получении карт");
      }
    });
  }
  
  const adminDeleteAllBtn = document.getElementById("admin-delete-all-cards-btn");
  if (adminDeleteAllBtn) {
    const newAdminDeleteAllBtn = adminDeleteAllBtn.cloneNode(true);
    adminDeleteAllBtn.parentNode.replaceChild(newAdminDeleteAllBtn, adminDeleteAllBtn);
    newAdminDeleteAllBtn.addEventListener("click", async () => {
      if (!(await showGameConfirm("Вы уверены, что хотите удалить ВСЕ карты из вашей коллекции? Это действие нельзя отменить!"))) {
    return;
  }

  const authData = resolveUserId();
  if (!authData) {
        alert("Ошибка авторизации");
    return;
  }

  try {
        let url = "/api/admin/cards/delete-all";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url, {
      method: "POST",
          headers: { "Content-Type": "application/json" }
    });

    const result = await response.json();
    if (result.success) {
          alert("Все карты удалены из вашей коллекции");
          await loadUserCards();
    } else {
          alert(result.message || "Ошибка удаления карт");
    }
  } catch (error) {
        console.error("Ошибка удаления карт:", error);
        alert("Ошибка при удалении карт");
      }
    });
  }
  
  // Проверяем доступ админа для показа кнопок
  if (currentProfile && currentProfile.user_id === 6803854304) {
    const adminActions = document.getElementById("admin-cards-actions");
    if (adminActions) {
      adminActions.style.display = "flex";
    }
  } else {
    // Скрываем кнопки админа для не-админов
    const adminActions = document.getElementById("admin-cards-actions");
    if (adminActions) {
      adminActions.style.display = "none";
    }
  }
  
  // Добавляем обработчики кастомного выпадающего списка сортировки
  const sortSelectBtn = document.getElementById("cards-sort-select-btn");
  const sortDropdown = document.getElementById("cards-sort-dropdown");
  const sortSelectText = sortSelectBtn?.querySelector(".sort-select-text");
  
  if (sortSelectBtn && sortDropdown && sortSelectText) {
    // Обновляем текст кнопки
    const updateSortButtonText = () => {
      const options = {
        power: "⚔️ По мощи",
        level: "📊 По уровню",
        upgrade: "⬆️ Близость к апгрейду",
        date: "📅 По дате получения",
        name: "🔤 По имени"
      };
      sortSelectText.textContent = options[currentCardSort] || "⚔️ По мощи";
    };
    updateSortButtonText();
    
    // Открытие/закрытие выпадающего списка
    sortSelectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = sortDropdown.style.display !== "none";
      sortDropdown.style.display = isOpen ? "none" : "block";
      const arrow = sortSelectBtn.querySelector(".sort-select-arrow");
      if (arrow) {
        arrow.textContent = isOpen ? "▼" : "▲";
      }
    });
    
    // Выбор опции
    sortDropdown.querySelectorAll(".sort-option").forEach(option => {
      option.addEventListener("click", () => {
        currentCardSort = option.dataset.value;
        updateSortButtonText();
        sortDropdown.style.display = "none";
        const arrow = sortSelectBtn.querySelector(".sort-select-arrow");
        if (arrow) arrow.textContent = "▼";
        renderCardsGrid();
      });
    });
    
    // Закрытие при клике вне
    document.addEventListener("click", (e) => {
      if (!sortSelectBtn.contains(e.target) && !sortDropdown.contains(e.target)) {
        sortDropdown.style.display = "none";
        const arrow = sortSelectBtn.querySelector(".sort-select-arrow");
        if (arrow) arrow.textContent = "▼";
      }
    });
  }
  
  // Добавляем обработчик кнопки направления сортировки
  const sortDirectionBtn = document.getElementById("sort-direction-btn");
  if (sortDirectionBtn) {
    sortDirectionBtn.textContent = currentSortDirection === "desc" ? "🔽" : "🔼";
    sortDirectionBtn.addEventListener("click", () => {
      currentSortDirection = currentSortDirection === "desc" ? "asc" : "desc";
      sortDirectionBtn.textContent = currentSortDirection === "desc" ? "🔽" : "🔼";
      renderCardsGrid();
    });
  }
  
  // Добавляем обработчик для раскрывающегося блока фильтров
  const filterToggle = document.getElementById("rarity-filter-toggle");
  const rarityFiltersCollapsible = document.getElementById("rarity-filters-collapsible");
  if (filterToggle && rarityFiltersCollapsible) {
    filterToggle.addEventListener("click", () => {
      const isHidden = rarityFiltersCollapsible.style.display === "none";
      rarityFiltersCollapsible.style.display = isHidden ? "block" : "none";
      filterToggle.textContent = isHidden ? "🔼 Редкости" : "🔽 Редкости";
    });
  }
  
  // Добавляем обработчики фильтрации по редкости
  const rarityFilters = document.querySelectorAll(".rarity-filter");
  rarityFilters.forEach(filter => {
    filter.addEventListener("click", () => {
      rarityFilters.forEach(f => f.classList.remove("active"));
      filter.classList.add("active");
      currentRarityFilter = filter.dataset.rarity || "all";
      renderCardsGrid();
    });
  });
}

// ==================== КОММЬЮНИТИ ====================

// Инициализация раздела коммьюнити
function initCommunity() {
  console.log("Инициализация коммьюнити...");
  
  // Инициализируем вкладки
  initCommunityTabs();
  
  // Загружаем посты
  loadCommunityPosts();
  
  // Инициализируем чат
  initChat();
  
  // Проверяем права админа
  checkAdminAccess();
}

// Инициализация вкладок коммьюнити
function initCommunityTabs() {
  const tabs = document.querySelectorAll(".community-tab");
  const tabContents = document.querySelectorAll(".community-tab-content");
  
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.dataset.tab;
      
      // Убираем активный класс со всех табов
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(content => content.classList.remove("active"));
      
      // Активируем выбранный таб
      tab.classList.add("active");
      const targetContent = document.getElementById(`${targetTab}-tab`);
      if (targetContent) {
        targetContent.classList.add("active");
      }
        
      // Загружаем контент в зависимости от вкладки
      if (targetTab === "posts") {
          loadCommunityPosts();
      } else if (targetTab === "chat") {
        loadChatMessages();
      }
      
      try {
        if (tg?.HapticFeedback?.impactOccurred) {
          tg.HapticFeedback.impactOccurred("light");
        }
      } catch (e) {}
    });
  });
}

  // Загрузка постов коммьюнити
  async function loadCommunityPosts() {
  const postsList = document.getElementById("community-posts-list");
  const emptyState = document.getElementById("posts-empty-state");
  if (!postsList) return;
  
      const authData = resolveUserId();
  if (!authData) {
    postsList.innerHTML = '<div class="error-message">Ошибка авторизации</div>';
    return;
  }
  
  try {
      let url = "/api/community/posts?limit=50";
      if (typeof authData === "string") {
        url += `&_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `&user_id=${authData}`;
      }
      
      const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    
      const data = await response.json();
    const posts = data.posts || [];
    
    if (posts.length === 0) {
        postsList.innerHTML = "";
        if (emptyState) emptyState.style.display = "block";
    } else {
      if (emptyState) emptyState.style.display = "none";
      renderPosts(posts);
    }
  } catch (error) {
    console.error("Ошибка загрузки постов:", error);
    postsList.innerHTML = '<div class="error-message">Ошибка загрузки постов</div>';
  }
}

// Отображение постов
function renderPosts(posts) {
  const postsList = document.getElementById("community-posts-list");
  if (!postsList) return;
  
  const authData = resolveUserId();
  let currentUserId = null;
  if (typeof authData === "number") {
    currentUserId = authData;
  } else if (typeof authData === "string") {
    try {
      const urlParams = new URLSearchParams(authData);
      const userParam = urlParams.get("user");
      if (userParam) {
        const userData = JSON.parse(decodeURIComponent(userParam));
        currentUserId = userData.id;
      }
    } catch (e) {}
  }
  
  postsList.innerHTML = posts.map(post => {
    const timeAgo = post.created_at ? formatTimeAgo(new Date(post.created_at)) : "";
        const isLiked = post.is_liked || false;
    const likesCount = post.likes_count || 0;
    const isAdmin = currentUserId === 6803854304;
    const isAuthor = post.author_id === currentUserId;
    const authorName = post.author_name || post.first_name || post.username || "Игрок";
    const isPostAdmin = post.author_id === 6803854304;
    const avatarUrl = post.author_photo_url || null;
    const avatarInitial = authorName.charAt(0).toUpperCase();
    const escapedAuthorName = escapeHtml(authorName);
    const escapedAvatarUrl = avatarUrl ? escapeHtml(avatarUrl) : '';
        
        return `
      <div class="community-post" data-post-id="${post.id}">
        <div class="post-body">
          <div class="post-title">${escapeHtml(post.title || "")}</div>
          <div class="post-content">${escapeHtml(post.content || "")}</div>
          ${post.photo_file_id ? `<div class="post-photo">📷 Фото</div>` : ""}
        </div>
        <div class="post-footer">
              <div class="post-author-info">
            <div class="post-author-avatar-wrapper">
              <div class="post-author-avatar ${isPostAdmin ? 'admin-avatar' : ''}">
                ${avatarUrl ? `<img src="${escapedAvatarUrl}" alt="${escapedAuthorName}" class="avatar-img" onerror="this.style.display='none'; this.parentElement.innerHTML='<span class=\\'avatar-initial\\'>${avatarInitial}</span>';">` : `<span class="avatar-initial">${avatarInitial}</span>`}
              </div>
              ${isPostAdmin ? '<div class="admin-crown">👑</div>' : ''}
              </div>
            <div class="post-author-details">
              <div class="post-author-name">
                ${escapeHtml(authorName)}
                ${isPostAdmin ? '<span class="admin-badge-text">Админ</span>' : ''}
            </div>
              <div class="post-time">${timeAgo}</div>
            </div>
          </div>
          <div class="post-actions">
            <button class="post-like-btn ${isLiked ? "liked" : ""}" data-post-id="${post.id}" data-liked="${isLiked}">
                <span class="like-icon">${isLiked ? "❤️" : "🤍"}</span>
                <span class="like-count">${likesCount}</span>
              </button>
            ${isAdmin || isAuthor ? `
              <button class="post-delete-btn" data-post-id="${post.id}" title="Удалить пост">🗑️</button>
            ` : ""}
          </div>
            </div>
          </div>
        `;
      }).join("");
      
      // Добавляем обработчики лайков
  postsList.querySelectorAll(".post-like-btn").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
          const postId = parseInt(btn.dataset.postId);
      const wasLiked = btn.dataset.liked === "true";
      
      // Анимация лайка
      animateLike(btn, !wasLiked);
      
          await togglePostLike(postId, btn);
        });
      });

  // Добавляем обработчики удаления постов (для админа)
  postsList.querySelectorAll(".post-delete-btn").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
          const postId = parseInt(btn.dataset.postId);
      if (await showGameConfirm("Вы уверены, что хотите удалить этот пост?")) {
          await deletePost(postId);
      }
        });
      });
}

// Анимация лайка
function animateLike(btn, isLiking) {
  const icon = btn.querySelector(".like-icon");
  const count = btn.querySelector(".like-count");
  
  if (!icon || !count) return;
  
  // Добавляем класс для анимации
  btn.classList.add("like-animating");
  
  if (isLiking) {
    // Анимация появления лайка
    icon.textContent = "❤️";
    btn.classList.add("liked");
    btn.dataset.liked = "true";
    
    // Эффект частиц
    createLikeParticles(btn);
    
    // Обновляем счетчик с анимацией
    const currentCount = parseInt(count.textContent) || 0;
    animateCounter(count, currentCount, currentCount + 1, 300);
  } else {
    // Анимация убирания лайка
    icon.textContent = "🤍";
    btn.classList.remove("liked");
    btn.dataset.liked = "false";
    
    // Обновляем счетчик с анимацией
    const currentCount = parseInt(count.textContent) || 0;
    if (currentCount > 0) {
      animateCounter(count, currentCount, currentCount - 1, 300);
    }
  }
  
  // Убираем класс анимации через время
  setTimeout(() => {
    btn.classList.remove("like-animating");
  }, 600);
}

// Создание частиц при лайке
function createLikeParticles(btn) {
  const rect = btn.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  
  for (let i = 0; i < 8; i++) {
    const particle = document.createElement("div");
    particle.className = "like-particle";
    particle.style.cssText = `
      position: fixed;
      width: 8px;
      height: 8px;
      background: #ff6b9d;
      border-radius: 50%;
      pointer-events: none;
      z-index: 10000;
      left: ${centerX}px;
      top: ${centerY}px;
      transform: translate(-50%, -50%);
    `;
    document.body.appendChild(particle);
    
    const angle = (i / 8) * Math.PI * 2;
    const distance = 30 + Math.random() * 20;
    const x = Math.cos(angle) * distance;
    const y = Math.sin(angle) * distance;
    
    setTimeout(() => {
      particle.style.transition = "all 0.6s ease-out";
      particle.style.transform = `translate(${x}px, ${y}px)`;
      particle.style.opacity = "0";
      particle.style.scale = "0";
      setTimeout(() => particle.remove(), 600);
    }, 10);
    }
  }

  // Удаление поста
  async function deletePost(postId) {
  const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
      return;
    }

    try {
      let url = "/api/community/posts/delete";
      if (typeof authData === "string") {
        url += `?_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `?user_id=${authData}`;
      }

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
      });

      const result = await response.json();
    
      if (result.success) {
        await loadCommunityPosts();
      showNotification("Пост удален", "success");
      } else {
      alert(result.message || "Ошибка удаления поста");
      }
    } catch (error) {
      console.error("Ошибка удаления поста:", error);
    alert("Ошибка при удалении поста");
    }
  }

// Переключение лайка поста
async function togglePostLike(postId, btnElement) {
      const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
    return;
  }
  
  try {
      let url = "/api/community/posts/like";
      if (typeof authData === "string") {
        url += `?_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `?user_id=${authData}`;
      }
      
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
      });
      
      if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
      }
      
      const result = await response.json();
      if (result.success) {
      // Обновляем счетчик лайков
      if (btnElement) {
        const countEl = btnElement.querySelector(".like-count");
        if (countEl) {
          const newCount = result.likes_count || 0;
          countEl.textContent = newCount;
        }
        } else {
        // Если кнопка не передана, перезагружаем посты
        await loadCommunityPosts();
      }
    } else {
      // Откатываем анимацию при ошибке
      if (btnElement) {
        const wasLiked = btnElement.dataset.liked === "true";
        animateLike(btnElement, wasLiked);
      }
      }
    } catch (error) {
    console.error("Ошибка лайка поста:", error);
    // Откатываем анимацию при ошибке
    if (btnElement) {
      const wasLiked = btnElement.dataset.liked === "true";
      animateLike(btnElement, wasLiked);
    }
  }
}

// Проверка прав админа
async function checkAdminAccess() {
    const authData = resolveUserId();
  if (!authData) return;
  
  try {
    // Проверяем, является ли пользователь админом (ID: 6803854304)
    let userId = null;
    if (typeof authData === "number") {
      userId = authData;
    } else if (typeof authData === "string") {
      // Пытаемся извлечь user_id из initData
      const urlParams = new URLSearchParams(authData);
      const userParam = urlParams.get("user");
      if (userParam) {
        try {
          const userData = JSON.parse(decodeURIComponent(userParam));
          userId = userData.id;
        } catch (e) {}
      }
    }
    
    const adminContainer = document.getElementById("admin-post-create-btn-container");
    if (adminContainer && userId === 6803854304) {
      adminContainer.style.display = "block";
      
      // Добавляем обработчик создания поста
      const createBtn = document.getElementById("create-post-btn");
      if (createBtn) {
        createBtn.onclick = () => openCreatePostModal();
      }
    } else if (adminContainer) {
      adminContainer.style.display = "none";
    }
  } catch (error) {
    console.error("Ошибка проверки прав админа:", error);
  }
}

// Открытие модального окна создания поста
function openCreatePostModal() {
  const modal = document.createElement("div");
  modal.className = "modal-overlay create-post-modal";
  modal.style.display = "flex";
  modal.innerHTML = `
    <div class="modal-content create-post-content">
      <div class="modal-header">
        <h2>✨ Создать пост</h2>
        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
              </div>
      <div class="modal-body create-post-body">
        <div class="form-group">
          <label for="post-title-input">Заголовок</label>
          <input type="text" id="post-title-input" class="form-input" placeholder="Введите заголовок поста..." maxlength="200" />
          <div class="form-char-count"><span id="title-char-count">0</span>/200</div>
        </div>
        <div class="form-group">
          <label for="post-content-input">Содержание</label>
          <textarea id="post-content-input" class="form-textarea" placeholder="Напишите что-нибудь интересное..." maxlength="2000" rows="8"></textarea>
          <div class="form-char-count"><span id="content-char-count">0</span>/2000</div>
        </div>
        <div class="form-group">
          <label for="post-photo-input">ID фото (опционально)</label>
          <input type="text" id="post-photo-input" class="form-input" placeholder="file_id из Telegram..." />
          <div class="form-hint">Прикрепите фото к сообщению в боте и скопируйте file_id</div>
        </div>
      </div>
      <div class="modal-footer create-post-footer">
        <button class="btn-secondary" onclick="this.closest('.modal-overlay').remove()">Отмена</button>
        <button class="btn-primary" id="submit-post-btn">
          <span class="btn-icon">📝</span>
          Опубликовать
        </button>
      </div>
            </div>
          `;
  document.body.appendChild(modal);
  
  // Счетчики символов
  const titleInput = modal.querySelector("#post-title-input");
  const contentInput = modal.querySelector("#post-content-input");
  const titleCount = modal.querySelector("#title-char-count");
  const contentCount = modal.querySelector("#content-char-count");
  
  titleInput.addEventListener("input", () => {
    titleCount.textContent = titleInput.value.length;
  });
  
  contentInput.addEventListener("input", () => {
    contentCount.textContent = contentInput.value.length;
  });
  
  const submitBtn = modal.querySelector("#submit-post-btn");
  submitBtn.addEventListener("click", async () => {
    await createPost(modal);
  });
}

// Создание поста
async function createPost(modal) {
  const title = document.getElementById("post-title-input").value.trim();
  const content = document.getElementById("post-content-input").value.trim();
  const photoFileId = document.getElementById("post-photo-input").value.trim();
  
  if (!title || !content) {
    await showGameAlert("Заполните заголовок и содержание поста", "⚠️");
        return;
      }
      
  const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
    return;
  }
  
  try {
    let url = "/api/community/posts/create";
      if (typeof authData === "string") {
        url += `?_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `?user_id=${authData}`;
      }
      
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title,
        content: content,
        photo_file_id: photoFileId || null
      })
    });
      
      const result = await response.json();
    
      if (result.success) {
      modal.remove();
      await loadCommunityPosts();
      showNotification("Пост успешно опубликован!", "success");
      } else {
      alert(result.message || "Ошибка создания поста");
      }
    } catch (error) {
    console.error("Ошибка создания поста:", error);
    alert("Ошибка при создании поста");
  }
}

// Инициализация чата
function initChat() {
  const chatSendBtn = document.getElementById("chat-send-btn");
  const chatInput = document.getElementById("chat-input");
  const chatFullscreenBtn = document.getElementById("chat-fullscreen-btn");
  const chatFullscreenClose = document.getElementById("chat-fullscreen-close");
  const chatFullscreenSendBtn = document.getElementById("chat-fullscreen-send-btn");
  const chatFullscreenInput = document.getElementById("chat-fullscreen-input");
  const chatFullscreenModal = document.getElementById("chat-fullscreen-modal");
  
  if (chatSendBtn && chatInput) {
    chatSendBtn.addEventListener("click", () => sendChatMessage(false));
    chatInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        sendChatMessage(false);
      }
    });
  }

  // Полноэкранный чат
  const chatFullscreenCloseBtn = document.getElementById("chat-fullscreen-close-btn");
  
  function toggleFullscreenMode() {
    const communitySection = document.getElementById("community-section");
    const communityTitle = communitySection?.querySelector(".section-title");
    const communityTabs = communitySection?.querySelector(".community-tabs");
    const postsTab = document.getElementById("posts-tab");
    const guidesTab = document.getElementById("guides-tab");
    const chatTab = document.getElementById("chat-tab");
    
    // Переключаем полноэкранный режим
    const isFullscreen = communitySection?.classList.contains("chat-fullscreen-mode");
    
    if (isFullscreen) {
      // Выходим из полноэкранного режима
      communitySection?.classList.remove("chat-fullscreen-mode");
      if (communityTitle) communityTitle.style.display = "";
      if (communityTabs) communityTabs.style.display = "";
      if (postsTab) postsTab.style.display = "";
      if (guidesTab) guidesTab.style.display = "";
      if (chatFullscreenBtn) {
        chatFullscreenBtn.style.display = "";
        chatFullscreenBtn.textContent = "⛶";
        chatFullscreenBtn.title = "Открыть на весь экран";
      }
      if (chatFullscreenCloseBtn) {
        chatFullscreenCloseBtn.style.display = "none";
      }
    } else {
      // Входим в полноэкранный режим
      communitySection?.classList.add("chat-fullscreen-mode");
      if (communityTitle) communityTitle.style.display = "none";
      if (communityTabs) communityTabs.style.display = "none";
      if (postsTab) postsTab.style.display = "none";
      if (guidesTab) guidesTab.style.display = "none";
      if (chatTab) {
        chatTab.style.display = "block";
        chatTab.classList.add("active");
      }
      if (chatFullscreenBtn) {
        chatFullscreenBtn.style.display = "none";
      }
      if (chatFullscreenCloseBtn) {
        chatFullscreenCloseBtn.style.display = "flex";
      }
      // Загружаем сообщения
      loadChatMessages();
    }
  }
  
  if (chatFullscreenBtn) {
    chatFullscreenBtn.addEventListener("click", toggleFullscreenMode);
  }
  
  if (chatFullscreenCloseBtn) {
    chatFullscreenCloseBtn.addEventListener("click", toggleFullscreenMode);
  }


  // Загружаем сообщения
  loadChatMessages();
  
  // Автообновление сообщений каждые 5 секунд
  setInterval(() => {
      const chatTab = document.getElementById("chat-tab");
    const communitySection = document.getElementById("community-section");
    if ((chatTab && chatTab.classList.contains("active")) || 
        (communitySection && communitySection.classList.contains("chat-fullscreen-mode"))) {
      loadChatMessages();
    }
  }, 5000);
}

// Загрузка сообщений чата
async function loadChatMessages() {
  const messagesContainer = document.getElementById("chat-messages");
  if (!messagesContainer) return;
  
  try {
    const response = await fetch("/api/community/chat/messages?limit=100");
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}`);
    }
    
    const data = await response.json();
    const messages = (data.messages || []).reverse(); // Переворачиваем, чтобы новые были внизу
    
        const authData = resolveUserId();
    let currentUserId = null;
    if (typeof authData === "number") {
      currentUserId = authData;
    } else if (typeof authData === "string") {
      try {
        const urlParams = new URLSearchParams(authData);
        const userParam = urlParams.get("user");
        if (userParam) {
          const userData = JSON.parse(decodeURIComponent(userParam));
          currentUserId = userData.id;
        }
      } catch (e) {}
    }
    
    messagesContainer.innerHTML = messages.map(msg => {
      const timeAgo = msg.created_at ? formatTimeAgo(new Date(msg.created_at)) : "";
      const displayName = msg.display_name || msg.first_name || msg.username || "Игрок";
      const isOwnMessage = msg.user_id === currentUserId;
      const avatarUrl = msg.user_photo_url || null;
      const avatarInitial = displayName.charAt(0).toUpperCase();
      const isMsgAdmin = msg.user_id === 6803854304;
      const escapedDisplayName = escapeHtml(displayName);
      const escapedAvatarUrl = avatarUrl ? escapeHtml(avatarUrl) : '';
      
      return `
        <div class="chat-message ${isOwnMessage ? "own-message" : ""}">
          <div class="message-header">
            <div class="message-author-avatar-wrapper">
              <div class="message-author-avatar ${isMsgAdmin ? 'admin-avatar' : ''}">
                ${avatarUrl ? `<img src="${escapedAvatarUrl}" alt="${escapedDisplayName}" class="avatar-img" onerror="this.style.display='none'; this.parentElement.innerHTML='<span class=\\'avatar-initial\\'>${avatarInitial}</span>';">` : `<span class="avatar-initial">${avatarInitial}</span>`}
              </div>
              ${isMsgAdmin ? '<div class="admin-crown-small">👑</div>' : ''}
            </div>
            <div class="message-author-info">
              <div class="message-author-name">
                ${escapeHtml(displayName)}
                ${isMsgAdmin ? '<span class="admin-badge-text-small">Админ</span>' : ''}
              </div>
              <div class="message-time">${timeAgo}</div>
            </div>
          </div>
          <div class="message-text">${escapeHtml(msg.message || msg.text || "")}</div>
        </div>
      `;
    }).join("");
    
    // Прокручиваем вниз
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  } catch (error) {
    console.error("Ошибка загрузки сообщений:", error);
    }
  }

  // Отправка сообщения в чат
let chatCooldownUntil = 0;
let chatSendBtnDisabled = false;

  async function sendChatMessage() {
    const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  
  if (!chatInput || !chatSendBtn) return;
  
  // Проверяем CD
  const now = Date.now();
  if (now < chatCooldownUntil) {
    const remaining = Math.ceil((chatCooldownUntil - now) / 1000);
    await showGameAlert(`Подождите ${remaining} секунд перед отправкой следующего сообщения`, "⚠️");
        return;
      }

  if (chatSendBtnDisabled) {
        return;
      }

  const text = chatInput.value.trim();
  if (!text) return;

      const authData = resolveUserId();
  if (!authData) {
    alert("Ошибка авторизации");
    return;
  }

  // Блокируем кнопку
  chatSendBtnDisabled = true;
  chatSendBtn.disabled = true;
  const originalBtnText = chatSendBtn.textContent;
  chatSendBtn.textContent = "Отправка...";

  try {
      let url = "/api/community/chat/send";
      if (typeof authData === "string") {
        url += `?_auth=${encodeURIComponent(authData)}`;
      } else if (typeof authData === "number") {
        url += `?user_id=${authData}`;
      }
      
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
      });

    const result = await response.json();
      
      if (!response.ok) {
      if (response.status === 429 && result.cooldown_remaining) {
        chatCooldownUntil = Date.now() + (result.cooldown_remaining * 1000);
        await showGameAlert(result.message || `Подождите ${result.cooldown_remaining} секунд`, "⚠️");
      } else {
        await showGameAlert(result.message || `Ошибка ${response.status}`, "❌");
      }
      return;
    }

      if (result.success) {
        chatInput.value = "";
        // Воспроизводим звук отправки сообщения
        playChatMessageSound();
      await loadChatMessages();
      
      // Устанавливаем CD (будет обновлен с сервера при следующей попытке)
      // Пока используем дефолтные значения
      chatCooldownUntil = Date.now() + 15000; // 15 сек по умолчанию
      } else {
      alert(result.message || "Ошибка отправки сообщения");
      }
    } catch (error) {
      console.error("Ошибка отправки сообщения:", error);
    alert("Ошибка отправки сообщения");
    } finally {
    chatSendBtnDisabled = false;
    chatSendBtn.disabled = false;
    chatSendBtn.textContent = originalBtnText;
  }
}

// Вспомогательная функция для экранирования HTML
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Кубик
let diceCooldownInterval = null;

async function loadDiceStatus() {
  try {
    const authData = resolveUserId();
    if (!authData) return;

    let url = "/api/dice/status";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }

    const response = await fetch(url);
    if (!response.ok) return;

    const status = await response.json();
    updateDiceUI(status);
    return status;
  } catch (error) {
    console.error("Ошибка загрузки статуса кубика:", error);
  }
}

function updateDiceUI(status) {
  const diceBtn = document.getElementById("dice-modal-btn");
  const diceIcon = document.getElementById("dice-modal-icon");
  const diceResult = document.getElementById("dice-modal-result");

  if (!diceBtn) return;

  if (status.can_roll) {
    diceBtn.disabled = false;
    if (diceIcon) diceIcon.style.display = "block";
    if (diceResult) diceResult.style.display = "none";
    if (diceCooldownInterval) {
      clearInterval(diceCooldownInterval);
      diceCooldownInterval = null;
    }
  } else {
    diceBtn.disabled = true;
  }
  
  // Проверяем, нужно ли показать модальное окно при первом входе
  if (status.is_first_login_today) {
    showDiceModal();
  }
}

function startDiceCooldown(cooldownUntil) {
  if (!cooldownUntil) return;

  const updateCooldown = () => {
    const now = new Date();
    const until = new Date(cooldownUntil);
    const diff = until - now;

    if (diff <= 0) {
      const diceCooldownOverlay = document.getElementById("dice-cooldown-overlay");
      if (diceCooldownOverlay) diceCooldownOverlay.style.display = "none";
      const diceBtn = document.getElementById("dice-btn");
      if (diceBtn) {
        diceBtn.disabled = false;
      }
      if (diceCooldownInterval) {
        clearInterval(diceCooldownInterval);
        diceCooldownInterval = null;
      }
      loadDiceStatus();
        return;
      }

    // Форматируем время (компактный формат)
    const totalSeconds = Math.floor(diff / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    const diceCooldownText = document.getElementById("dice-cooldown-text");
    if (diceCooldownText) {
      if (hours > 0) {
        diceCooldownText.textContent = `${hours}:${String(minutes).padStart(2, '0')}`;
      } else if (minutes > 0) {
        diceCooldownText.textContent = `${minutes}:${String(seconds).padStart(2, '0')}`;
      } else {
        diceCooldownText.textContent = `${seconds}с`;
      }
    }
  };

  updateCooldown();
  if (diceCooldownInterval) clearInterval(diceCooldownInterval);
  diceCooldownInterval = setInterval(updateCooldown, 100);
}

async function rollDice() {
      try {
        const authData = resolveUserId();
    if (!authData) return;

    const diceBtn = document.getElementById("dice-modal-btn");
    if (!diceBtn || diceBtn.disabled) return;

    diceBtn.disabled = true;

    let url = "/api/dice/roll";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          url += `?user_id=${authData}`;
    }

    const response = await fetch(url, { method: "POST" });
        if (!response.ok) {
          const error = await response.json();
      if (error.error === "cooldown") {
        await loadDiceStatus();
      }
      await showGameAlert("Ошибка при броске кубика", "❌");
      return;
        }

        const result = await response.json();

        if (result.success) {
      // Анимация броска
      const diceBtn = document.getElementById("dice-modal-btn");
      const diceIcon = document.getElementById("dice-modal-icon");
      const diceResult = document.getElementById("dice-modal-result");
      
      if (diceBtn) {
        diceBtn.classList.add("rolling");
      }
      
      // Скрываем иконку и показываем результат после анимации
      setTimeout(() => {
        if (diceIcon) diceIcon.style.display = "none";
        if (diceResult) {
          diceResult.textContent = result.dice_result;
          diceResult.style.display = "block";
        }
        if (diceBtn) {
          diceBtn.classList.remove("rolling");
        }
        
        // Возвращаем иконку через 2 секунды
        setTimeout(() => {
          if (diceResult) diceResult.style.display = "none";
          if (diceIcon) diceIcon.style.display = "block";
        }, 2000);
      }, 1000);

      // Показываем награду с красивым эффектом
      const rewardText = result.reward.gems > 0 
        ? `+${result.reward.gems} 💎` 
        : `+${result.reward.coins} 💰`;
      
      setTimeout(() => {
        // Создаем красивое уведомление вместо alert
        showDiceRewardNotification(result.dice_result, rewardText);
        
        // Обновляем статус и профиль
        loadDiceStatus();
        loadProfile(authData);
      }, 1500);

      // Тактильная отдача
      if (tg?.HapticFeedback?.impactOccurred) {
        tg.HapticFeedback.impactOccurred("heavy");
      }
        }
      } catch (error) {
    console.error("Ошибка броска кубика:", error);
    await showGameAlert("Ошибка при броске кубика", "❌");
    const diceBtn = document.getElementById("dice-modal-btn");
    if (diceBtn) diceBtn.disabled = false;
  }
}

// Показ уведомления о награде
function showDiceRewardNotification(diceResult, rewardText) {
  // Удаляем предыдущее уведомление, если есть
  const existing = document.getElementById("dice-reward-notification");
  if (existing) existing.remove();

  const notification = document.createElement("div");
  notification.id = "dice-reward-notification";
  notification.className = "dice-reward-notification";
  notification.innerHTML = `
    <div class="dice-notification-content">
      <div class="dice-notification-result">🎲 ${diceResult}</div>
      <div class="dice-notification-reward">${rewardText}</div>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Анимация появления
  setTimeout(() => notification.classList.add("show"), 10);
  
  // Удаляем через 3 секунды
  setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => notification.remove(), 500);
  }, 3000);
}

// Показ модального окна кубика
function showDiceModal() {
  const modal = document.getElementById("dice-modal");
  if (!modal) return;
  
  modal.style.display = "flex";
  
  // Инициализируем кнопку
  const diceBtn = document.getElementById("dice-modal-btn");
  if (diceBtn) {
    diceBtn.onclick = rollDice;
  }
  
  // Загружаем статус кубика
  loadDiceStatus();
  
  // Закрытие модального окна
  const closeBtn = document.getElementById("dice-modal-close");
  if (closeBtn) {
    closeBtn.onclick = () => {
      modal.style.display = "none";
      // После закрытия проверяем, нужно ли предложить включить уведомления
      checkDiceNotificationPrompt();
    };
  }
  
  // Закрытие по клику на overlay
  modal.onclick = (e) => {
    if (e.target === modal) {
      modal.style.display = "none";
      checkDiceNotificationPrompt();
    }
  };
  
  // Тактильная отдача
  if (tg?.HapticFeedback?.impactOccurred) {
    tg.HapticFeedback.impactOccurred("medium");
  }
}

// Проверка и показ предложения включить уведомления
async function checkDiceNotificationPrompt() {
  try {
    const authData = resolveUserId();
    if (!authData) return;
    
    // Проверяем, показывалось ли уже предложение
    let url = "/api/dice/notification-prompt-status";
    if (typeof authData === "string") {
      url += `?_auth=${encodeURIComponent(authData)}`;
    }
    
    const response = await fetch(url);
    if (!response.ok) return;
    
    const data = await response.json();
    
    // Если предложение еще не показывалось, показываем его
    if (!data.prompt_shown) {
      showDiceNotificationPrompt();
    }
  } catch (error) {
    console.error("Ошибка проверки статуса предложения:", error);
  }
}

// Показ предложения включить уведомления
function showDiceNotificationPrompt() {
  const promptModal = document.getElementById("dice-notification-prompt");
  if (!promptModal) return;
  
  promptModal.style.display = "flex";
  
  const yesBtn = document.getElementById("dice-notification-prompt-yes");
  const noBtn = document.getElementById("dice-notification-prompt-no");
  
  if (yesBtn) {
    yesBtn.onclick = async () => {
      try {
        const authData = resolveUserId();
        if (!authData) return;
        
        // Включаем уведомления
        await updateSettings({ notif_dice: true });
        
        // Отмечаем, что предложение было показано
        let url = "/api/dice/notification-prompt-mark";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        await fetch(url, { method: "POST" });
        
        promptModal.style.display = "none";
      } catch (error) {
        console.error("Ошибка включения уведомлений:", error);
      }
    };
  }
  
  if (noBtn) {
    noBtn.onclick = async () => {
      try {
        const authData = resolveUserId();
        if (!authData) return;
        
        // Отмечаем, что предложение было показано (но уведомления не включены)
        let url = "/api/dice/notification-prompt-mark";
        if (typeof authData === "string") {
          url += `?_auth=${encodeURIComponent(authData)}`;
        } else if (typeof authData === "number") {
          console.warn("auth: numeric userId unsupported, skipping auth param");
        }
        
        await fetch(url, { method: "POST" });
        
        promptModal.style.display = "none";
      } catch (error) {
        console.error("Ошибка:", error);
      }
    };
  }
  
  // Закрытие по клику на overlay
  promptModal.onclick = (e) => {
    if (e.target === promptModal) {
      promptModal.style.display = "none";
    }
  };
}

// Инициализация кубика при загрузке приложения (проверка первого входа)
async function initDiceOnLoad() {
  try {
    const authData = resolveUserId();
    if (!authData) return;
    
    // Загружаем статус кубика
    await loadDiceStatus();
  } catch (error) {
    console.error("Ошибка инициализации кубика:", error);
  }
}

// ═══ Фоновое обновление: профиль, почта, pending платежи ═══
let _bgPollTimer = null;
let _bgPollBusy = false;

function startBackgroundPolling() {
  if (_bgPollTimer) return;
  _bgPollTimer = setInterval(() => {
    if (_bgPollBusy) return;
    const authData = resolveUserId();
    if (!authData) return;
    _backgroundPoll(authData);
  }, 15000); // каждые 15 секунд
}

function stopBackgroundPolling() {
  if (_bgPollTimer) {
    clearInterval(_bgPollTimer);
    _bgPollTimer = null;
  }
}

async function _backgroundPoll(authData) {
  if (_bgPollBusy) return;
  _bgPollBusy = true;
  try {
    if (document.visibilityState !== "visible") return;

    var jti = sessionStorage.getItem("pending_checkout_jti");
    if (jti) {
      try {
        var jtiUrl = `/api/payments/checkout/session-status?jti=${encodeURIComponent(jti)}`;
        if (typeof authData === "string") jtiUrl += `&_auth=${encodeURIComponent(authData)}`;
        var jtiRes = await fetch(jtiUrl);
        if (jtiRes.ok) {
          var jtiData = await jtiRes.json();
          if (jtiData.payment_status === "succeeded" && !jtiData.rewards_processed) {
            sessionStorage.setItem("pending_payment_id", jtiData.payment_id || "");
            sessionStorage.setItem("pending_payment_method", "yookassa");
            sessionStorage.setItem("pending_payment_timestamp", String(Date.now()));
            console.log("[BG-POLL] Сессия %s успешна, payment_id=%s", jti, jtiData.payment_id);
            await handleSuccessfulPayment(authData);
            sessionStorage.removeItem("pending_checkout_jti");
          } else if (jtiData.payment_status === "canceled") {
            sessionStorage.removeItem("pending_checkout_jti");
          } else if (jtiData.payment_status === "succeeded" && jtiData.rewards_processed && !jtiData.modal_shown) {
            sessionStorage.setItem("pending_payment_id", jtiData.payment_id || "");
            sessionStorage.setItem("pending_payment_method", "yookassa");
            sessionStorage.setItem("pending_payment_timestamp", String(Date.now()));
            console.log("[BG-POLL] Платёж обработан, но модалка не показана");
            await handleSuccessfulPayment(authData);
            sessionStorage.removeItem("pending_checkout_jti");
          }
        }
      } catch(_) {}
    }

    var pendingPaymentId = sessionStorage.getItem("pending_payment_id");
    if (pendingPaymentId) {
      try {
        var payUrl = `/api/payments/status?payment_id=${pendingPaymentId}`;
        if (typeof authData === "string") payUrl += `&_auth=${encodeURIComponent(authData)}`;
        var payRes = await fetch(payUrl);
        if (payRes.ok) {
          var payData = await payRes.json();
          if (payData.status === "succeeded" && payData.rewards_processed) {
            sessionStorage.removeItem("pending_payment_id");
            sessionStorage.removeItem("pending_payment_item");
            sessionStorage.removeItem("pending_payment_timestamp");
            sessionStorage.removeItem("pending_payment_method");
            sessionStorage.removeItem("pending_checkout_jti");
            await loadProfile(authData);
          }
        }
      } catch(_) {}
    }

    try {
      var succRes = await fetch("/api/payments/recent-success?" + (typeof authData === "string" ? "_auth=" + encodeURIComponent(authData) : ""));
      if (succRes.ok) {
        var succData = await succRes.json();
        if (succData.payments && succData.payments.length > 0) {
          var p = succData.payments[0];
          if (!sessionStorage.getItem("pending_payment_id")) {
            sessionStorage.setItem("pending_payment_id", p.payment_id);
            sessionStorage.setItem("pending_payment_method", "yookassa");
            sessionStorage.setItem("pending_payment_timestamp", String(Date.now()));
            console.log("[BG-POLL] Найден неотмеченный succeeded платёж:", p.payment_id);
            await handleSuccessfulPayment(authData);
            sessionStorage.removeItem("pending_checkout_jti");
          }
        }
      }
    } catch(_) {}

    try { await updateMailNotificationBadge(authData); } catch(_) {}
    try { await loadProfile(authData); } catch(_) {}
  } finally {
    _bgPollBusy = false;
  }
}

// Запуск polling при инициализации
document.addEventListener("DOMContentLoaded", () => {
  startBackgroundPolling();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    startBackgroundPolling();
  } else {
    stopBackgroundPolling();
  }
});
