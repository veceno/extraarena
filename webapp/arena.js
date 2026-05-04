/**
 * ============================================
 * ARENA.JS — Фронтенд для боевой арены
 * ============================================
 * 
 * Управляет интерфейсом боя:
 * - Подключение к Socket.IO
 * - Загрузка и рендеринг состояния боя
 * - Обработка действий игрока (play card, attack, end turn, surrender)
 * - Реакция на события сервера (turn_end, card_played, attack, game_over)
 */

// ============================================
// ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
// ============================================

let socket = null;
let matchId = null;
let userId = null;
let currentState = null;

// Для drag & drop
let selectedCard = null;
let selectedAttacker = null;

// Для детекции исцеления/урона
let previousPlayerHP = null;
let previousOpponentHP = null;
let previousUnitHPs = {}; // { instanceId: hp }
let currentTurnCount = 0;

// ЕДИНЫЙ РЕЖИМ ВЗАИМОДЕЙСТВИЯ
let interactionMode = {
  type: 'NONE', // 'NONE' | 'ATTACK' | 'TARGETING'
  data: null    // Данные карты или атакующего
};

// Кеш legal_actions из последнего состояния
let cachedLegalActions = [];

// ============================================
// LEGAL ACTIONS HELPERS
// ============================================

/**
 * Проверяет, можно ли разыграть карту из руки по индексу.
 * @param {number} handIndex - индекс карты в руке
 * @returns {boolean}
 */
function canPlayCard(handIndex) {
  if (!cachedLegalActions || cachedLegalActions.length === 0) return true; // fallback
  return cachedLegalActions.some(a => a.type === 'play_card' && a.hand_index === handIndex);
}

/**
 * Получает возможные цели для карты из руки.
 * @param {number} handIndex - индекс карты в руке
 * @returns {Array} массив action объектов с target_id
 */
function getPlayCardTargets(handIndex) {
  if (!cachedLegalActions) return [];
  return cachedLegalActions.filter(a => a.type === 'play_card' && a.hand_index === handIndex);
}

/**
 * Проверяет, может ли существо атаковать.
 * @param {string} instanceId - instance_id существа
 * @returns {boolean}
 */
function canAttack(instanceId) {
  // ИСПРАВЛЕНО: Доверяем свойству can_attack самого юнита, 
  // так как legal_actions могут запаздывать или быть неполными
  const board = currentState?.player?.board || currentState?.player1_board || currentState?.player2_board || [];
  const unit = board.find(u => String(u.instance_id) === String(instanceId));
  if (unit && unit.can_attack) return true;

  if (!cachedLegalActions || cachedLegalActions.length === 0) return false;
  return cachedLegalActions.some(a => a.type === 'attack' && a.attacker_id === String(instanceId));
}

/**
 * Получает возможные цели атаки для существа.
 * @param {string} instanceId - instance_id атакующего
 * @returns {Array} массив action объектов с target_id/target_is_hero
 */
function getAttackTargets(instanceId) {
  if (!cachedLegalActions) return [];
  return cachedLegalActions.filter(a => a.type === 'attack' && a.attacker_id === String(instanceId));
}

/**
 * Проверяет, является ли существо/герой валидной целью для выбранного атакующего.
 * @param {string} targetId - instance_id цели
 * @param {boolean} isHero - является ли цель героем
 * @returns {boolean}
 */
function isValidAttackTarget(targetId, isHero) {
  if (!interactionMode.data || interactionMode.type !== 'ATTACK') return false;
  const attackerId = interactionMode.data.instance_id;
  const targets = getAttackTargets(attackerId);
  return targets.some(t => {
    if (isHero) return t.target_is_hero === true;
    return t.target_id === String(targetId);
  });
}

// ============================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('[ARENA] Страница арены загружена');
  
  // Извлекаем параметры из URL
  const urlParams = new URLSearchParams(window.location.search);
  matchId = urlParams.get('id');
  userId = urlParams.get('user_id');
  
  console.log('[ARENA] Match ID:', matchId);
  console.log('[ARENA] User ID:', userId);
  
  if (!matchId || !userId) {
    console.error('[ARENA] Отсутствует match_id или user_id в URL');
    alert('Ошибка: параметры боя не найдены');
    return;
  }
  
  // Инициализируем Socket.IO и загружаем состояние
  initSocketIO();
  loadBattleState();
  
  // Привязываем обработчики UI
  bindUIHandlers();
  
  // Инициализируем фоновую музыку (запускается по первому клику)
  initArenaMusic();
});

// ============================================
// SOCKET.IO
// ============================================

function initSocketIO() {
  console.log('[SOCKET.IO] Подключение к серверу...');
  
  socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5
  });
  
  socket.on('connect', () => {
    console.log('[SOCKET.IO] Подключено! Socket ID:', socket.id);
    
    // Входим в комнату матча
    socket.emit('join_match', {
      match_id: matchId,
      user_id: parseInt(userId, 10)
    });
    
    // ВАЖНО: НЕ отправляем client_ready здесь
    // Сигнал будет отправлен после успешной загрузки состояния боя в loadBattleState()
  });
  
  socket.on('disconnect', (reason) => {
    console.warn('[SOCKET.IO] Отключено:', reason);
  });
  
  socket.on('error', (error) => {
    console.error('[SOCKET.IO] Ошибка:', error);
  });
  
  socket.on('joined_match', (data) => {
    console.log('[SOCKET.IO] Вступили в матч:', data);
    
    // КРИТИЧНО: Сразу после входа в комнату отправляем сигнал готовности
    // Это позволяет серверу запустить бота (если он ходит первым)
    console.log('[SOCKET.IO] Отправка сигнала client_ready...');
    socket.emit('client_ready', {
      match_id: matchId,
      user_id: Number(userId) // Приводим к числу для консистентности
    });
  });
  
  socket.on('client_ready_ack', (data) => {
    console.log('[SOCKET.IO] Подтверждение готовности клиента получено:', data);
  });
  
  // События боя
  socket.on('state_changed', (data) => {
    console.log('[SOCKET.IO] state_changed получено:', data);
    handleStateChanged(data);
  });
  
  socket.on('turn_end', (data) => {
    console.log('[SOCKET.IO] turn_end получено:', data);
    
    // ДОБАВЛЕНО: Специальное логирование для автоматического завершения хода
    if (data.auto_ended) {
      console.log('[SOCKET.IO] ⏰ Ход автоматически завершён сервером (время истекло)');
    }
    
    handleStateChanged(data);
  });
  
  // ДОБАВЛЕНО: Обработчик начала нового хода
  socket.on('turn_start', (data) => {
    console.log('[SOCKET.IO] turn_start получено:', data);
    
    // КРИТИЧНО: Принудительно сбрасываем таймер на 25 секунд
    const timerText = document.getElementById('turn-timer-text');
    if (timerText) {
      timerText.textContent = '25';
    }
    
    handleStateChanged(data);
  });
  
  socket.on('card_played', (eventData) => {
    console.log('[SOCKET.IO] card_played получено:', eventData);
    
    // ДОБАВЛЕНО: Обработка анимации для зелий
    const actionData = eventData.data;
    if (actionData && actionData.card_type === 'potion') {
      console.log('[ARENA] Обнаружен розыгрыш зелья, запускаем анимацию');
      const targetId = actionData.target_id;
      const isHero = actionData.target_is_hero;
      const playerWhoPlayed = actionData.player_id;
      
      let targetEl;
      if (isHero) {
        // Если зелье разыграл текущий пользователь, значит цель - герой оппонента
        // Если зелье разыграл оппонент, значит цель - герой текущего пользователя
        if (String(playerWhoPlayed) === String(userId)) {
          targetEl = document.querySelector('.opponent-hp-block');
        } else {
          targetEl = document.querySelector('.player-hp-block');
        }
      } else {
        // Для существ просто ищем по instance_id на поле
        targetEl = document.querySelector(`[data-instance-id="${targetId}"]`);
      }
      
      if (targetEl) {
        triggerPotionDamageFlash(targetEl);
      }
    }
    
    handleStateChanged(eventData);
  });
  
  socket.on('attack', (data) => {
    console.log('[SOCKET.IO] attack получено:', data);
    handleStateChanged(data);
  });
  
  socket.on('game_over', (data) => {
    console.log('[SOCKET.IO] game_over получено:', data);
    handleGameOver(data);
  });
}

function handleStateChanged(eventData) {
  // Извлекаем state из события
  const newState = eventData.state || eventData.state_p1;
  
  if (!newState) {
    console.warn('[ARENA] state_changed получено без state');
    return;
  }
  
  // ДОБАВЛЕНО: Подробное логирование для отслеживания изменений HP
  console.log('[ARENA] Обновление состояния:', newState);
  console.log('[ARENA] 🔍 HP tracking:');
  console.log('  - Player 1 HP:', newState.player1_hp || newState.player?.hp || '???');
  console.log('  - Player 2 HP:', newState.player2_hp || newState.opponent?.hp || '???');
  console.log('  - Current player:', newState.current_player_id);
  console.log('  - Is my turn:', newState.is_my_turn);
  console.log('  - Turn:', newState.turn);
  console.log('  - Player mana:', newState.player?.mana, '/', newState.player?.max_mana);
  console.log('  - Opponent mana:', newState.opponent?.mana, '/', newState.opponent?.max_mana);
  console.log(`[ARENA] ⏰ TIMER: turn_time_remaining=${newState.turn_time_remaining}, turn_duration=${newState.turn_duration}`);
  
  // ИСПРАВЛЕНО: Принудительный сброс таймера при смене хода
  // Проверяем, изменился ли номер хода - если да, обновляем таймер принудительно
  const turnChanged = currentState && currentState.turn !== newState.turn;
  if (turnChanged) {
    console.log(`[ARENA] ⏰ Ход изменился: ${currentState.turn} -> ${newState.turn}, принудительный сброс таймера`);
    
    // ДОБАВЛЕНО: Специальное логирование для автоматического завершения
    if (eventData.auto_ended) {
      console.log('[ARENA] ⏰ Ход был автоматически завершён сервером из-за истечения времени');
    }
  }
  
  currentState = newState;
  renderBattleState(newState);
  
  // ИСПРАВЛЕНО: Если ход изменился, гарантируем обновление таймера после рендера
  if (turnChanged) {
    updateTurnTimer(newState);
  }
}

// ============================================
// ЗАГРУЗКА СОСТОЯНИЯ БОЯБРЯ
// ============================================

async function loadBattleState() {
  try {
    console.log('[ARENA] Загрузка состояния боя...');
    
    const response = await fetch(`/api/battle/state?match_id=${matchId}&user_id=${userId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const state = await response.json();
    console.log('[ARENA] Состояние боя загружено:', state);
    
    currentState = state;
    renderBattleState(state);
    
  } catch (error) {
    console.error('[ARENA] Ошибка загрузки состояния боя:', error);
    alert('Не удалось загрузить бой: ' + error.message);
  }
}

// ============================================
// РЕНДЕРИНГ СОСТОЯНИЯ
// ============================================

function renderBattleState(state) {
  if (!state) {
    console.warn('[ARENA] renderBattleState вызван с пустым state');
    return;
  }
  
  // Валидация полноты state
  const isValidState = (
    state.player_ids && 
    state.player_ids.length === 2 &&
    (state.player || state.player1_hp !== undefined) &&
    (state.opponent || state.player2_hp !== undefined)
  );
  
  if (!isValidState) {
    console.warn('[ARENA] ⚠️ State неполный, пропускаем рендеринг');
    return;
  }
  
  console.log('[ARENA] Рендеринг состояния боя...');
  
  // КРИТИЧНО: Кешируем legal_actions для использования в рендеринге
  cachedLegalActions = state.legal_actions || [];
  console.log('[ARENA] 📋 Legal actions:', cachedLegalActions.length, 'доступных действий');
  
  const userIdNum = Number(userId);
  
  // Локально рассчитываем is_my_turn (на случай broadcast без viewer_id)
  const isMyTurn = Number(state.current_player_id) === userIdNum;
  state.is_my_turn = isMyTurn;
  
  // Определяем, кто игрок, а кто оппонент
  const playerState = state.player || (state.player_ids && state.player_ids[0] == userIdNum ? state : null);
  const opponentState = state.opponent || null;
  
  // Если структура старая, используем player1/player2
  let myState, opponentStateData;
  
  if (playerState && opponentState) {
    myState = playerState;
    opponentStateData = opponentState;
  } else {
    // Определяем по player_ids
    const p1 = {
      user_id: state.player_ids ? state.player_ids[0] : null,
      // КРИТИЧНО: Явная проверка на undefined/null, чтобы 0 HP не превращался в 30
      hp: (state.player1_hp !== undefined && state.player1_hp !== null) ? state.player1_hp : 30,
      mana: state.player1_mana || 0,
      max_mana: state.player?.max_mana || 10,
      hand: state.player1_hand || [],
      board: state.player1_board || [],
      name: state.player?.name || 'Игрок',
      avatar_url: state.player?.avatar_url
    };
    
    const p2 = {
      user_id: state.player_ids ? state.player_ids[1] : null,
      // КРИТИЧНО: Явная проверка на undefined/null, чтобы 0 HP не превращался в 30
      hp: (state.player2_hp !== undefined && state.player2_hp !== null) ? state.player2_hp : 30,
      mana: state.player2_mana || 0,
      max_mana: state.opponent?.max_mana || 10,
      hand: state.player2_hand || [],
      board: state.player2_board || [],
      name: state.opponent?.name || 'Оппонент',
      avatar_url: state.opponent?.avatar_url
    };
    
    if (String(p1.user_id) === String(userIdNum)) {
      myState = p1;
      opponentStateData = p2;
    } else {
      myState = p2;
      opponentStateData = p1;
    }
  }
  
  // КРИТИЧНО: Логируем HP при каждом обновлении для отслеживания изменений
  console.log('[ARENA] 💚 HP TRACKING: Мой HP =', myState.hp, '| Оппонент HP =', opponentStateData.hp);
  console.log('[ARENA] Мой state:', myState);
  console.log('[ARENA] Оппонент state:', opponentStateData);
  
  // Рендерим панели
  renderPlayerPanel(myState);
  renderOpponentPanel(opponentStateData);
  
  // Сохраняем номер хода
  currentTurnCount = state.turn || 0;

  // Рендерим руку
  renderHand(myState.hand || []);
  
  // Рендерим поля
  renderBoard('player', myState.board || []);
  renderBoard('opponent', opponentStateData.board || []);
  
  // Обновляем индикатор хода
  updateTurnIndicator(state);
  
  // Обновляем таймер
  updateTurnTimer(state);

  // Обновляем лог боя
  updateBattleLog(state.action_history || []);
  
  // КРИТИЧНО: Восстанавливаем подсветку целей при режиме TARGETING после полной перерисовки
  if (interactionMode.type === 'TARGETING') {
    console.log('[ARENA] 🔄 КРИТИЧНО: Восстанавливаю подсветку после полной перерисовки состояния');
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    highlightValidTargets(playActions);
  }
  
  // КРИТИЧНО: Проверяем завершение игры и показываем финальный экран
  if (state.is_ended === true || state.game_over === true) {
    console.log('[ARENA] 🏁 Игра завершена, показываем финальный экран');
    
    // Извлекаем данные о победителе
    const winnerId = state.winner_id || state.winner;
    const isWinner = String(winnerId) === String(userId);
    
    // КРИТИЧНО: Извлекаем данные о трофеях из state (синхронизированы с БД)
    // state.trophy_total и state.coins_total берутся напрямую из результатов
    // db.update_user_trophies() и db.update_user_coins() в server.py
    const trophyDelta = parseInt(state.trophy_change || state.trophy_delta, 10) || 0;
    const trophyTotal = parseInt(state.trophy_total || state.new_trophies, 10) || null;
    
    // КРИТИЧНО: Извлекаем данные о монетах из state (синхронизированы с БД)
    const coinsDelta = parseInt(state.coins_change || state.coins_delta, 10) || 0;
    const coinsTotal = parseInt(state.coins_total || state.new_coins, 10) || null;
    
    console.log('[ARENA] 🎯 Результат игры:', { 
      isWinner, 
      winnerId, 
      trophyDelta, 
      trophyTotal, 
      coinsDelta, 
      coinsTotal 
    });
    
    // Показываем экран результата с задержкой для драматического эффекта
    setTimeout(() => {
      showBattleResult(isWinner, trophyDelta, trophyTotal, coinsDelta, coinsTotal);
    }, 1200);
  }
}

// ============================================
// ОБНОВЛЕНИЕ ЛОГА БОЯ
// ============================================

function updateBattleLog(history) {
  const logRows = document.getElementById('battle-log-rows');
  if (!logRows) return;

  // Очищаем лог
  logRows.innerHTML = '';
  
  // Если история пуста, добавляем системную заглушку
  if (!history || history.length === 0) {
    const row = document.createElement('div');
    row.className = 'log-row';
    row.innerHTML = '<span class="log-dot system"></span><span>Бой начался</span>';
    logRows.appendChild(row);
    return;
  }

  // Берем все записи и рендерим (бэкенд уже ограничивает до 100)
  // history может быть массивом строк или массивом [type, text]
  const lastEntries = history.slice().reverse();
  
  lastEntries.forEach(entry => {
    let type = 'system';
    let text = '';
    
    if (Array.isArray(entry)) {
      [type, text] = entry;
    } else if (typeof entry === 'object') {
      type = entry.type || 'system';
      text = entry.text || '';
    } else {
      // Фолбэк для строк: пытаемся определить тип по содержанию или ставим system
      text = entry;
      if (text.includes('Вы ') || text.includes('Ваш')) type = 'player';
      else if (text.includes('Оппонент') || text.includes('Противник')) type = 'opponent';
    }
    
    // Проверяем, является ли это разделителем хода
    if (type === 'system' && text.includes('———')) {
      // Создаём специальный разделитель
      const separator = document.createElement('div');
      separator.className = 'log-separator';
      separator.textContent = text;
      logRows.appendChild(separator);
    } else {
      // Обычная запись лога
      const row = document.createElement('div');
      row.className = 'log-row';
      
      // СТИЛЬ МИДОРИИ: Если есть молния, выделяем цветом
      if (text.includes('⚡')) {
        row.classList.add('log-midoriya');
      }
      
      row.innerHTML = `<span class="log-dot ${type}"></span><span>${text}</span>`;
      logRows.appendChild(row);
    }
  });

  // Автопрокрутка лога вниз
  requestAnimationFrame(() => {
    logRows.scrollTop = logRows.scrollHeight;
  });
}

// ============================================
// РЕНДЕРИНГ ПАНЕЛЕЙ
// ============================================

function renderPlayerPanel(playerState) {
  // HP - поддержка новой структуры с hero объектом
  const hpText = document.getElementById('player-hp-text');
  const hpMaxText = document.getElementById('player-hp-max-text');
  if (hpText) {
    // Новая структура: playerState.hero.hp, старая: playerState.hp
    const hp = playerState.hero?.hp ?? playerState.hp ?? 30;
    const hpValue = Math.max(0, hp);
    hpText.textContent = hpValue;

    // ОБНОВЛЕНО: Динамическое макс. HP
    if (hpMaxText) {
      hpMaxText.textContent = '/' + (playerState.hero?.max_hp ?? 30);
    }
    
    // Детекция изменений HP
    if (previousPlayerHP !== null) {
      if (hpValue > previousPlayerHP && hpValue > 0) {
        console.log('[ARENA] 💚 Healing detected for player:', previousPlayerHP, '->', hpValue);
        triggerHealAnimation(true);
      } else if (hpValue < previousPlayerHP) {
        console.log('[ARENA] 💥 Damage detected for player:', previousPlayerHP, '->', hpValue);
        const playerHeroEl = document.querySelector('.player-hp-block');
        if (playerHeroEl) triggerDamageEffects(playerHeroEl, previousPlayerHP - hpValue);
      }
    }
    previousPlayerHP = hpValue;
  }
  
  // Имя
  const nameText = document.getElementById('player-name-text');
  if (nameText) {
    nameText.textContent = playerState.name || 'Игрок';
  }
  
  // Аватар (первая буква имени)
  const avatarLetter = document.getElementById('player-avatar-letter');
  if (avatarLetter) {
    const firstName = playerState.name || 'И';
    avatarLetter.textContent = firstName[0].toUpperCase();
  }
  
  // ДОБАВЛЕНО: Премиальный визуал для ExtraPass
  const infoBlock = document.querySelector('.player-info-block');
  if (infoBlock) {
    // Проверяем extra_pass из state или из playerState
    const hasExtraPass = currentState?.extra_pass === 'active' || playerState?.extra_pass === 'active';
    if (hasExtraPass) {
      infoBlock.classList.add('extra-pass-active');
      console.log('[ARENA] 💎 ExtraPass визуал активирован для игрока');
    } else {
      infoBlock.classList.remove('extra-pass-active');
    }
  }
  
  // Мана
  const manaText = document.getElementById('player-mana-text');
  const manaMaxText = document.getElementById('player-mana-max-text');
  const manaFill = document.getElementById('player-mana-fill');
  
  if (manaText) {
    const manaValue = playerState.mana || 0;
    // Если мана дробная, показываем 1 знак после запятой
    manaText.textContent = manaValue % 1 === 0 ? manaValue : manaValue.toFixed(1);
  }
  
  if (manaMaxText) {
    manaMaxText.textContent = `/${playerState.max_mana || 10}`;
  }
  
  if (manaFill) {
    const manaPercent = ((playerState.mana || 0) / (playerState.max_mana || 10)) * 100;
    manaFill.style.width = `${Math.min(100, Math.max(0, manaPercent))}%`;
  }

  // КРИТИЧНО: Восстанавливаем подсветку своего героя при режиме TARGETING после перерисовки
  const playerPanel = document.querySelector('.player-panel-root');
  if (playerPanel && interactionMode.type === 'TARGETING') {
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    const playerHeroId = playerState.hero?.instance_id;
    if (playActions.some(a => a.target_is_hero && (String(a.target_id) === String(playerHeroId) || !a.target_id))) {
      // Примечание: если target_id не указан, но есть target_is_hero, обычно это враг,
      // но для хила мы проверяем ID. Если ID совпал или это явно хил (нужно больше инфы от сервера),
      // но пока ориентируемся на ID.
      playerPanel.classList.add('targetable-friendly');
    }
  }
}

function renderOpponentPanel(opponentState) {
  // HP - поддержка новой структуры с hero объектом
  const hpText = document.getElementById('opponent-hp-text');
  const hpMaxText = document.getElementById('opponent-hp-max-text');
  if (hpText) {
    // Новая структура: opponentState.hero.hp, старая: opponentState.hp
    const hp = opponentState.hero?.hp ?? opponentState.hp ?? 30;
    const hpValue = Math.max(0, hp);
    hpText.textContent = hpValue;

    // ОБНОВЛЕНО: Динамическое макс. HP
    if (hpMaxText) {
      hpMaxText.textContent = '/' + (opponentState.hero?.max_hp ?? 30);
    }
    
    // Детекция изменений HP
    if (previousOpponentHP !== null) {
      if (hpValue > previousOpponentHP && hpValue > 0) {
        console.log('[ARENA] 💚 Healing detected for opponent:', previousOpponentHP, '->', hpValue);
        triggerHealAnimation(false);
      } else if (hpValue < previousOpponentHP) {
        console.log('[ARENA] 💥 Damage detected for opponent:', previousOpponentHP, '->', hpValue);
        const opponentHeroEl = document.querySelector('.opponent-hp-block');
        if (opponentHeroEl) triggerDamageEffects(opponentHeroEl, previousOpponentHP - hpValue);
      }
    }
    previousOpponentHP = hpValue;
  }
  
  // Имя
  const nameText = document.getElementById('opponent-name-text');
  if (nameText) {
    nameText.textContent = opponentState.name || 'Оппонент';
  }
  
  // Аватар (первая буква имени)
  const avatarLetter = document.getElementById('opponent-avatar-letter');
  if (avatarLetter) {
    const firstName = opponentState.name || 'О';
    avatarLetter.textContent = firstName[0].toUpperCase();
  }
  
  // Количество карт в руке
  const handCount = document.getElementById('opponent-hand-count');
  if (handCount) {
    handCount.textContent = opponentState.hand ? opponentState.hand.length : 0;
  }
  
  // ДОБАВЛЕНО: Премиальный визуал для ExtraPass оппонента
  const opponentInfoIsland = document.querySelector('.opponent-info-island');
  if (opponentInfoIsland) {
    const hasExtraPass = opponentState?.extra_pass === 'active';
    if (hasExtraPass) {
      opponentInfoIsland.classList.add('extra-pass-active');
      console.log('[ARENA] 💎 ExtraPass визуал активирован для оппонента');
    } else {
      opponentInfoIsland.classList.remove('extra-pass-active');
    }
  }
  
  // КРИТИЧНО: Восстанавливаем подсветку героя при режиме TARGETING после перерисовки
  const opponentPanel = document.querySelector('.opponent-panel-root');
  if (opponentPanel && interactionMode.type === 'TARGETING') {
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    const opponentHeroId = opponentState.hero?.instance_id;
    if (playActions.some(a => a.target_is_hero && (String(a.target_id) === String(opponentHeroId) || !a.target_id))) {
      opponentPanel.classList.add('targetable-enemy');
    }
  }
}

// ============================================
// РЕНДЕРИНГ РУКИ
// ============================================

function renderHand(handCards) {
  const handZone = document.getElementById('player-hand-zone');
  if (!handZone) {
    console.warn('[ARENA] player-hand-zone не найден');
    return;
  }
  
  // Очищаем руку
  handZone.innerHTML = '';
  
  console.log('[ARENA] Рендеринг руки:', handCards);
  
  if (!handCards || handCards.length === 0) {
    console.log('[ARENA] Рука пуста');
    return;
  }
  
  // Лимит вывода: только первые 5 карт
  const cardsToRender = handCards.slice(0, 5);
  
  cardsToRender.forEach((card, index) => {
    const cardEl = createHandCardElement(card, index);
    handZone.appendChild(cardEl);
  });
}

function createHandCardElement(card, index) {
  const cardDiv = document.createElement('div');
  cardDiv.className = 'hand-card';
  cardDiv.dataset.cardId = card.card_id || card.id;
  cardDiv.dataset.index = index;
  cardDiv.dataset.instanceId = card.instance_id || '';
  
  const cardType = card.card_type || 'warrior';
  cardDiv.dataset.cardType = cardType;
  // Убираем overflow: hidden, чтобы мана не обрезалась
  cardDiv.style.overflow = 'visible'; 
  
  if (cardType === 'potion') {
    cardDiv.classList.add('card-potion');
    cardDiv.classList.add('potion-card-shape');
  }
  
  // LEGAL ACTIONS: Проверяем, можно ли разыграть эту карту
  const isMyTurn = currentState?.is_my_turn || false;
  const cardPlayable = isMyTurn && canPlayCard(index);
  
  // Добавляем класс для недоступных карт
  if (!cardPlayable) {
    cardDiv.classList.add('card-disabled');
  }
  
  // Также проверяем ману (дополнительная визуальная подсказка)
  const cardMana = parseInt(card.mana || card.mana_cost || 0);
  const playerMana = currentState?.player?.mana || 0;
  
  if (cardMana > playerMana) {
    cardDiv.classList.add('insufficient-mana');
  }
  
  // ДОБАВЛЕНО: Визуализация механик карты (минималистичные классы)
  const mechanics = card.mechanics || [];
  if (Array.isArray(mechanics)) {
    if (mechanics.includes('taunt')) {
      cardDiv.classList.add('status-taunt');
    }
    if (mechanics.includes('shield') || mechanics.includes('divine_shield')) {
      cardDiv.classList.add('card-shield', 'status-shield');
    }
    if (mechanics.includes('charge')) {
      cardDiv.classList.add('card-charge');
    }
    if (mechanics.includes('deathrattle')) {
      cardDiv.classList.add('card-deathrattle');
      const drIcon = document.createElement('div');
      drIcon.className = 'mechanic-icon deathrattle-icon';
      drIcon.textContent = '💀';
      cardDiv.appendChild(drIcon);
    }
  }
  
  // ДОБАВЛЕНО: Визуализация заморозки в руке
  if (card.is_frozen === true) {
    cardDiv.classList.add('card-frozen', 'status-frozen');
  }
  
  // Создаем обертку для арта, чтобы применить форму к ней, а не к самой карте
  const artWrapper = document.createElement('div');
  artWrapper.className = 'card-art-wrapper';

  // Картинка карты
  const img = document.createElement('img');
  img.className = 'hand-card-art';
  img.src = card.image || '/DesignAssets/Cards/9.png'; // Дефолтная картинка
  img.alt = card.name || 'Card';
  img.draggable = false;

  // ОБНОВЛЕНО: Для зелий заполняем весь ромб
  if (cardType === 'potion') {
    img.style.objectFit = 'cover';
    img.style.height = '100%';
    img.style.width = '100%';
  }

  artWrapper.appendChild(img);

  // Мана (теперь прямой ребенок cardDiv, чтобы не обрезалась)
  const manaDiv = document.createElement('div');
  manaDiv.className = 'mana-circle';
  manaDiv.textContent = cardMana;

  cardDiv.appendChild(artWrapper);
  cardDiv.appendChild(manaDiv);

  // Статы: не добавляем для зелий
  if (cardType !== 'potion') {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'hand-card-stats';

    const atkDiv = document.createElement('div');
    atkDiv.className = 'card-stat attack';
    atkDiv.textContent = card.attack || card.atk || 0;

    const hpDiv = document.createElement('div');
    hpDiv.className = 'card-stat health';
    hpDiv.textContent = card.hp || card.hp_current || 0;

    statsDiv.appendChild(atkDiv);
    statsDiv.appendChild(hpDiv);
    cardDiv.appendChild(statsDiv);
  }
  
  // ДОБАВЛЕНО: Имя карты
  const nameLabel = document.createElement('div');
  nameLabel.className = 'card-name-label';
  nameLabel.textContent = card.name || 'Карта';
  cardDiv.appendChild(nameLabel);
  
  // ДОБАВЛЕНО: Инфо-иконка
  const infoBtn = document.createElement('div');
  infoBtn.className = 'card-info-button';
  infoBtn.textContent = 'i';
  infoBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Предотвращаем срабатывание глобального клика
    showCardDescription(card, infoBtn);
  });
  cardDiv.appendChild(infoBtn);
  
  // Обработчики drag & drop
  cardDiv.draggable = true;
  
  cardDiv.addEventListener('dragstart', (e) => {
    handleCardDragStart(e, card, index);
  });
  
  cardDiv.addEventListener('dragend', (e) => {
    handleCardDragEnd(e);
  });
  
  // Альтернативно: клик для выбора карты
  cardDiv.addEventListener('click', () => {
    handleCardClick(card, index, cardDiv);
  });
  
  addStatusIcons(cardDiv, card);
  return cardDiv;
}

/**
 * Добавляет контейнеры иконок для визуальных статусов
 * (Щит, Заморозка, Сон, Мишень)
 */
function addStatusIcons(cardDiv, card) {
  const effectsPath = '../DesignAssets/Arena/CardEffects/';
  
  const createIcon = (className, fileName) => {
    const container = document.createElement('div');
    container.className = `status-icon-container ${className}`;
    const img = document.createElement('img');
    img.src = effectsPath + fileName;
    img.className = 'status-icon';
    container.appendChild(img);
    return container;
  };

  cardDiv.appendChild(createIcon('status-icon-shield', 'shield.png'));
  cardDiv.appendChild(createIcon('status-icon-taunt', 'provocation.png'));
  
  const frozenIcon = createIcon('status-icon-frozen', 'freeze.png');
  if (card && card.is_frozen) {
    const counter = document.createElement('span');
    counter.className = 'freeze-counter';
    // Берем freeze_turns или 1 как заглушку
    counter.textContent = card.freeze_turns || "1";
    frozenIcon.appendChild(counter);
  }
  cardDiv.appendChild(frozenIcon);

  cardDiv.appendChild(createIcon('status-icon-asleep', 'asleep.png'));
  cardDiv.appendChild(createIcon('status-icon-target', 'target.png'));
  cardDiv.appendChild(createIcon('status-icon-heal', 'toHeal.png'));
}

// ============================================
// РЕНДЕРИНГ ПОЛЕЙ
// ============================================

function renderBoard(side, boardCards) {
  const boardZone = document.getElementById(`${side}-board-zone`);
  if (!boardZone) {
    console.warn(`[ARENA] ${side}-board-zone не найден`);
    return;
  }
  
  const slots = boardZone.querySelectorAll('.board-slot');
  
  console.log(`[ARENA] Рендеринг поля ${side}:`, boardCards);
  
  // Очищаем все слоты
  slots.forEach(slot => {
    slot.innerHTML = '';
    slot.classList.remove('attacker-selected');
  });
  
  if (!boardCards || boardCards.length === 0) {
    return;
  }
  
  // Заполняем слоты картами
  boardCards.forEach((card, index) => {
    if (index >= slots.length) return;
    
    const slot = slots[index];
    const cardEl = createBoardCardElement(card, side);
    slot.appendChild(cardEl);
    
    // КРИТИЧНО: Восстанавливаем подсветку целей при TARGETING режиме после перерисовки
    if (interactionMode.type === 'TARGETING') {
      const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
      if (playActions.some(a => String(a.target_id) === String(card.instance_id))) {
        cardEl.classList.add(side === 'opponent' ? 'targetable-enemy' : 'targetable-friendly');
      }
    }
  });
}

function createBoardCardElement(card, side) {
  const cardDiv = document.createElement('div');
  cardDiv.className = 'board-unit-card';
  cardDiv.dataset.instanceId = card.instance_id;
  cardDiv.dataset.ownerId = card.owner_id;
  cardDiv.style.overflow = 'visible'; // Позволяем элементам выходить за границы

  const cardType = card.card_type || 'warrior';
  if (cardType === 'potion') {
    cardDiv.classList.add('potion-card-shape');
  }
  
  // LEGAL ACTIONS: Определяем, может ли юнит атаковать
  const unitCanAttack = side === 'player' && canAttack(card.instance_id);
  
  // Спящие юниты (не могут атаковать в первый ход)
  if (card.is_asleep === true || (card.can_attack === false && side === 'player')) {
    cardDiv.classList.add('unit-sleeping');
    // ДОБАВЛЕНО: Визуализация сна
    if (card.is_asleep === true) {
      cardDiv.classList.add('card-sleep');
    }
  }

  // ДОБАВЛЕНО: Визуализация заморозки
  if (card.is_frozen === true) {
    cardDiv.classList.add('card-frozen', 'status-frozen');
  }
  
  // Добавляем класс для юнитов, которые могут атаковать
  if (unitCanAttack) {
    cardDiv.classList.add('can-attack');
  }
  
  // ДОБАВЛЕНО: Визуализация механик юнита на столе (минималистичные классы)
  const mechanics = card.mechanics || [];
  if (Array.isArray(mechanics)) {
    if (mechanics.includes('taunt')) {
      cardDiv.classList.add('status-taunt');
      console.log(`[ARENA] Юнит ${card.name || card.instance_id} имеет Provocation`);
    }
    if (mechanics.includes('shield') || mechanics.includes('divine_shield')) {
      cardDiv.classList.add('card-shield', 'status-shield');
      console.log(`[ARENA] Юнит ${card.name || card.instance_id} имеет Divine Shield`);
    }
    if (mechanics.includes('charge')) {
      cardDiv.classList.add('card-charge');
    }
    if (mechanics.includes('deathrattle')) {
      cardDiv.classList.add('card-deathrattle');
      const drIcon = document.createElement('div');
      drIcon.className = 'mechanic-icon deathrattle-icon';
      drIcon.textContent = '💀';
      cardDiv.appendChild(drIcon);
    }
  }
  
  // Враппер для арта
  const artWrapper = document.createElement('div');
  artWrapper.className = 'card-art-wrapper';

  // Картинка карты
  const img = document.createElement('img');
  img.className = 'unit-card-art';
  img.src = card.image || '/DesignAssets/Cards/9.png';
  img.alt = card.name || 'Unit';

  // ОБНОВЛЕНО: Для зелий на поле (если есть) заполняем форму
  if (cardType === 'potion') {
    img.style.objectFit = 'cover';
    img.style.height = '100%';
    img.style.width = '100%';
  }
  
  artWrapper.appendChild(img);
  cardDiv.appendChild(artWrapper);

  // Статы: не добавляем для зелий
  if (card.card_type !== 'potion') {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'unit-card-stats';
    
    const atkDiv = document.createElement('div');
    atkDiv.className = 'unit-stat attack';
    atkDiv.textContent = card.attack || card.atk || 0;
    
    const hpDiv = document.createElement('div');
    hpDiv.className = 'unit-stat health';
    const hpValue = card.hp || card.hp_current || 0;
    hpDiv.textContent = hpValue;
    
    // Детекция урона юниту
    const instanceId = String(card.instance_id);
    const oldHp = previousUnitHPs[instanceId];
    if (oldHp !== undefined && hpValue < oldHp) {
      triggerDamageEffects(cardDiv, oldHp - hpValue);
    }
    previousUnitHPs[instanceId] = hpValue;
    
    statsDiv.appendChild(atkDiv);
    statsDiv.appendChild(hpDiv);
    cardDiv.appendChild(statsDiv);
  }
  
  // ДОБАВЛЕНО: Имя карты на поле
  const nameLabel = document.createElement('div');
  nameLabel.className = 'card-name-label board-card-name';
  nameLabel.textContent = card.name || 'Юнит';
  cardDiv.appendChild(nameLabel);
  
  // ДОБАВЛЕНО: Инфо-иконка
  const infoBtn = document.createElement('div');
  infoBtn.className = 'card-info-button';
  infoBtn.textContent = 'i';
  infoBtn.style.bottom = '14px'; // Для доски чуть ниже
  infoBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Предотвращаем срабатывание глобального клика
    showCardDescription(card, infoBtn);
  });
  cardDiv.appendChild(infoBtn);
  
  // Если это карта игрока, разрешаем атаку
  if (side === 'player' && card.can_attack) {
    cardDiv.style.cursor = 'pointer';
    cardDiv.addEventListener('click', () => {
      handleAttackerClick(card);
    });
  }
  
  // ИСПРАВЛЕНО: Союзные юниты могут быть целью для хила/баффов
  if (side === 'player') {
    cardDiv.addEventListener('click', (e) => {
      // Если это не режим атаки, проверяем TARGETING
      if (interactionMode.type === 'TARGETING') {
        e.stopPropagation();
        handleGlobalTargetClick(card.instance_id, false, e);
      }
    });
  }
  
  // Если это карта оппонента, может быть целью атаки ИЛИ карты с целью
  if (side === 'opponent') {
    cardDiv.addEventListener('click', (e) => {
      e.stopPropagation();
      
      // КРИТИЧНО: Только в режиме ATTACK или TARGETING передаем клик
      if (interactionMode.type === 'ATTACK' || interactionMode.type === 'TARGETING') {
        handleGlobalTargetClick(card.instance_id, false, e);
      }
    });
  }
  
  addStatusIcons(cardDiv, card);
  return cardDiv;
}

// ============================================
// ИНДИКАТОР ХОДА
// ============================================

function updateTurnIndicator(state) {
  const turnText = document.getElementById('turn-status-text');
  const turnPlaque = document.getElementById('turn-indicator-display');
  
  if (!turnText || !turnPlaque) {
    console.warn('[ARENA] Элементы индикатора хода не найдены');
    return;
  }
  
  // КРИТИЧНО: Используем локально рассчитанный isMyTurn из state (уже установлен в renderBattleState)
  const isMyTurn = state.is_my_turn;
  const currentPlayerId = state.current_player_id;
  const userIdNum = Number(userId);
  
  // КРИТИЧНО: Логируем типы для диагностики проблем с is_my_turn
  console.log('[ARENA] Обновление индикатора хода. is_my_turn:', isMyTurn, 'current_player_id:', currentPlayerId, '(type:', typeof currentPlayerId + '), my user_id:', userIdNum, '(type:', typeof userIdNum + ')');
  
  if (isMyTurn) {
    turnText.textContent = 'Ваш ход';
    turnPlaque.classList.add('player-turn');
    turnPlaque.classList.remove('opponent-turn', 'turn-expired');
    
    // Активируем кнопку завершения хода
    const endTurnBtn = document.getElementById('end-turn-button');
    if (endTurnBtn) {
      endTurnBtn.disabled = false;
    }
  } else {
    turnText.textContent = 'Ход противника';
    turnPlaque.classList.add('opponent-turn');
    turnPlaque.classList.remove('player-turn', 'turn-expired');
    
    // Деактивируем кнопку завершения хода
    const endTurnBtn = document.getElementById('end-turn-button');
    if (endTurnBtn) {
      endTurnBtn.disabled = true;
    }
  }
}

// ============================================
// ТАЙМЕР ХОДА
// ============================================

let timerInterval = null;
let lastTurnNumber = null; // Для отслеживания изменений хода

function updateTurnTimer(state) {
  const timerText = document.getElementById('turn-timer-text');
  const timerContainer = document.getElementById('turn-timer-container');
  
  if (!timerText || !timerContainer) {
    return;
  }
  
  // Останавливаем предыдущий таймер
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  
  const turnDuration = state.turn_duration || 25;
  let timeRemaining = state.turn_time_remaining !== undefined ? state.turn_time_remaining : turnDuration;
  
  // КРИТИЧНО: Если номер хода изменился или время <= 0, НЕМЕДЛЕННО сбрасываем таймер на полную длительность
  // Это предотвращает «зависание» таймера при смене хода - не ждем тика интервала!
  const turnChanged = (lastTurnNumber !== null && state.turn !== lastTurnNumber);
  const timeExpired = timeRemaining <= 0;
  
  if (turnChanged || timeExpired) {
    console.log(`[ARENA] ⏰ Ход изменился: ${lastTurnNumber} -> ${state.turn}, или время истекло. НЕМЕДЛЕННЫЙ сброс таймера до ${turnDuration}с`);
    timeRemaining = turnDuration;
  }
  lastTurnNumber = state.turn;
  
  // Логируем время из состояния для отладки
  console.log(`[ARENA] ⏰ Таймер обновлен: turn=${state.turn}, timeRemaining=${timeRemaining}с, from_server=${state.turn_time_remaining}, turnChanged=${turnChanged}`);
  
  // КРИТИЧНО: НЕМЕДЛЕННО обновляем отображение таймера перед запуском интервала
  // Это гарантирует мгновенное обновление при смене хода
  timerText.textContent = Math.ceil(timeRemaining);
  
  // Визуальные предупреждения
  timerContainer.classList.remove('timer-warning', 'timer-critical');
  if (timeRemaining <= 5) {
    timerContainer.classList.add('timer-critical');
  } else if (timeRemaining <= 10) {
    timerContainer.classList.add('timer-warning');
  }
  
  // Функция обновления таймера каждую секунду
  const updateTimer = () => {
    timeRemaining = Math.max(0, timeRemaining - 1);
    timerText.textContent = Math.ceil(timeRemaining);
    
    // Визуальные предупреждения
    timerContainer.classList.remove('timer-warning', 'timer-critical');
    
    if (timeRemaining <= 5) {
      timerContainer.classList.add('timer-critical');
    } else if (timeRemaining <= 10) {
      timerContainer.classList.add('timer-warning');
    }
    
    if (timeRemaining <= 0) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  };
  
  // Запускаем интервал (первый тик через 1 секунду)
  timerInterval = setInterval(updateTimer, 1000);
}

// ============================================
// DRAG & DROP КАРТ
// ============================================

function handleCardDragStart(e, card, index) {
  console.log('[ARENA] Начало перетаскивания карты:', card);
  selectedCard = { card, index };
  e.currentTarget.classList.add('dragging');
  
  const cardType = card.card_type || 'warrior';
  const mechanics = card.mechanics || [];
  
  const targetingMechanics = [
    'battlecry_damage_1', 'consume_ally', 'battlecry_freeze'
  ];
  const isTargetingWarrior = cardType === 'warrior' && mechanics.some(m => targetingMechanics.includes(m));

  // Получаем возможные цели для этой карты
  const playActions = getPlayCardTargets(index);
  const hasTargetingOptions = playActions.length > 0 && playActions.some(a => a.target_id !== null);
  const requiresTarget = playActions.length > 0 && (isTargetingWarrior || playActions.every(a => a.target_id !== null));

  // Если это зелье или воин с целью - активируем режим TARGETING
  if (hasTargetingOptions || requiresTarget) {
    console.log('[ARENA] 🎯 Карта с целью: подсвечиваем валидные цели (drag mode)');
    
    // КРИТИЧНО: Устанавливаем режим TARGETING
    interactionMode = {
      type: 'TARGETING',
      data: { ...card, handIndex: index }
    };
    
    highlightValidTargets(playActions);
    
    // Добавляем обработчики для drop на валидные цели
    document.querySelectorAll('.targetable-enemy, .targetable-friendly').forEach(el => {
      el.addEventListener('dragover', handlePotionTargetDragOver);
      if (el.classList.contains('opponent-panel-root') || el.classList.contains('player-panel-root')) {
        el.addEventListener('drop', handlePotionHeroDrop);
      } else {
        el.addEventListener('drop', handlePotionTargetDrop);
      }
    });

    // Если выбор цели НЕ обязателен, также подсвечиваем слоты для дропа
    if (!requiresTarget) {
      const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
      playerSlots.forEach(slot => {
        if (!slot.querySelector('.board-unit-card')) {
          slot.classList.add('droppable');
          slot.addEventListener('dragover', handleSlotDragOver);
          slot.addEventListener('drop', handleSlotDrop);
        }
      });
    }
  } else {
    // Для обычных карт - подсвечиваем слоты на поле игрока
    console.log('[ARENA] ⚔️ Обычная карта: подсвечиваю слоты (drag mode)');
    
    const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
    playerSlots.forEach(slot => {
      if (!slot.querySelector('.board-unit-card')) {
        slot.classList.add('droppable');
        
        // Обработчики drop
        slot.addEventListener('dragover', handleSlotDragOver);
        slot.addEventListener('drop', handleSlotDrop);
      }
    });
  }
}

function handleCardDragEnd(e) {
  e.currentTarget.classList.remove('dragging');
  
  // Убираем подсветку слотов
  const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
  playerSlots.forEach(slot => {
    slot.classList.remove('droppable');
    slot.removeEventListener('dragover', handleSlotDragOver);
    slot.removeEventListener('drop', handleSlotDrop);
  });
  
  // Убираем подсветку целей
  document.querySelectorAll('.targetable-enemy, .targetable-friendly').forEach(el => {
    el.removeEventListener('dragover', handlePotionTargetDragOver);
    el.removeEventListener('drop', handlePotionTargetDrop);
    el.removeEventListener('drop', handlePotionHeroDrop);
  });
  
  // Сбрасываем режим
  resetInteractionMode();
}

function handleSlotDragOver(e) {
  e.preventDefault();
}

function handleSlotDrop(e) {
  e.preventDefault();
  
  if (!selectedCard) return;
  
  const slotIndex = parseInt(e.currentTarget.dataset.slot, 10);
  console.log('[ARENA] Карта сброшена на слот:', slotIndex);
  
  playCard(selectedCard.card, slotIndex);
  selectedCard = null;
}

// ============================================
// ОБРАБОТЧИКИ ДЛЯ ЗЕЛИЙ
// ============================================

function handlePotionTargetDragOver(e) {
  e.preventDefault();
}

function handlePotionTargetDrop(e) {
  e.preventDefault();
  
  if (!selectedCard) return;
  
  const targetUnit = e.currentTarget;
  const targetId = targetUnit.dataset.instanceId;
  
  console.log('[ARENA] Зелье применено на существо:', targetId);
  
  playPotionCard(selectedCard.card, targetId, false);
  selectedCard = null;
}

function handlePotionHeroDrop(e) {
  e.preventDefault();
  
  if (!selectedCard) return;
  
  console.log('[ARENA] Зелье применено на героя');
  
  // ИСПРАВЛЕНО: Для героя передаем null вместо -1
  playPotionCard(selectedCard.card, null, true);
  selectedCard = null;
}

// ============================================
// КЛИК ПО КАРТЕ (АЛЬТЕРНАТИВА DRAG&DROP)
// ============================================

function handleCardClick(card, index, cardEl) {
  console.log('[ARENA] 🎴 Клик по карте:', card);
  
  // LEGAL ACTIONS: Проверяем, можно ли разыграть эту карту
  if (!canPlayCard(index)) {
    console.warn('[ARENA] ❌ Карта недоступна для розыгрыша (legal_actions)');
    return;
  }
  
  // Сбрасываем предыдущий режим
  resetInteractionMode();
  
  cardEl.classList.add('selected');
  selectedCard = { card, index };
  
  const cardType = card.card_type || 'warrior';
  const mechanics = card.mechanics || [];
  
  // Проверка на необходимость выбора цели для воинов
  const targetingMechanics = [
    'battlecry_damage_1', 'consume_ally', 'battlecry_freeze'
  ];
  
  const isTargetingWarrior = cardType === 'warrior' && mechanics.some(m => targetingMechanics.includes(m));
  
  // Получаем возможные цели для этой карты
  const playActions = getPlayCardTargets(index);
  // Карта требует выбора цели если она в списке или если все её действия требуют цель
  const hasTargetingOptions = playActions.length > 0 && playActions.some(a => a.target_id !== null);
  const requiresTarget = playActions.length > 0 && (isTargetingWarrior || playActions.every(a => a.target_id !== null));
  
  // Если карта имеет цели, мы ВСЕГДА включаем режим TARGETING, 
  // но если она не требует цель (requiresTarget === false), мы также разрешаем клик по слотам.
  if (hasTargetingOptions || requiresTarget) {
    console.log('[ARENA] 🎯 Режим TARGETING активирован');
    
    interactionMode = {
      type: 'TARGETING',
      data: { ...card, handIndex: index }
    };
    
    document.body.classList.add('targeting-active');
    
    // Подсвечиваем валидные цели из legal_actions
    highlightValidTargets(playActions);
    
    // Если выбор цели НЕ обязателен (например, Геральт), разрешаем также клик по пустой клетке
    if (!requiresTarget) {
      console.log('[ARENA] ⚔️ Выбор цели опционален, подсвечиваю слоты');
      const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
      playerSlots.forEach(slot => {
        if (!slot.querySelector('.board-unit-card')) {
          slot.classList.add('droppable');
          slot.onclick = () => {
            const slotIndex = parseInt(slot.dataset.slot, 10);
            playCard(selectedCard.card, slotIndex);
            resetInteractionMode();
          };
        }
      });
    }
  } else {
    // Для обычных карт вообще без цели
    console.log('[ARENA] ⚔️ Карта без целей, подсвечиваю пустые слоты');
    const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
    playerSlots.forEach(slot => {
      if (!slot.querySelector('.board-unit-card')) {
        slot.classList.add('droppable');
        slot.onclick = () => {
          const slotIndex = parseInt(slot.dataset.slot, 10);
          playCard(selectedCard.card, slotIndex);
          resetInteractionMode();
        };
      }
    });
  }
}

function highlightValidTargets(actions) {
  /**
   * LEGAL ACTIONS: Подсвечивает ТОЛЬКО валидные цели из legal_actions
   * @param {Array} actions - массив action объектов с target_id
   */
  console.log('[ARENA] 🎯 highlightValidTargets:', actions.length, 'targets');
  
  // СТРОГИЙ ФИЛЬТР: Саб-Зиро (freeze) и Фрирен (heal)
  const card = interactionMode.data;
  const mechanics = card?.mechanics || [];
  const isHeal = mechanics.some(m => m.includes('heal'));
  const isFreeze = mechanics.some(m => m.includes('freeze'));

  const opponentUnits = document.querySelectorAll('#opponent-board-zone .board-unit-card');
  const playerUnits = document.querySelectorAll('#player-board-zone .board-unit-card');
  const opponentPanel = document.querySelector('.opponent-panel-root');
  const playerPanel = document.querySelector('.player-panel-root');
  
  // Собираем валидные target_id и разделяем на союзников и врагов
  const friendlyTargets = new Set();
  const enemyTargets = new Set();
  let playerHeroTargetable = false;
  let opponentHeroTargetable = false;
  
  const playerHeroId = currentState?.player?.hero?.instance_id;
  const opponentHeroId = currentState?.opponent?.hero?.instance_id;
  
  // Получаем ID юнитов игрока и оппонента для определения принадлежности
  const playerUnitIds = new Set();
  const opponentUnitIds = new Set();
  
  playerUnits.forEach(unit => playerUnitIds.add(String(unit.dataset.instanceId)));
  opponentUnits.forEach(unit => opponentUnitIds.add(String(unit.dataset.instanceId)));

  actions.forEach(a => {
    if (a.target_id) {
      const targetIdStr = String(a.target_id);
      
      // Определяем, союзник это или враг
      if (targetIdStr === String(playerHeroId)) {
        // Лечение разрешено для своего героя, заморозка — нет
        if (!isFreeze) {
          if (isHeal) {
            const currentHp = currentState?.player?.hero?.hp ?? currentState?.player?.hp ?? 30;
            const maxHp = currentState?.player?.hero?.max_hp ?? 30;
            if (currentHp < maxHp) playerHeroTargetable = true;
          } else {
            playerHeroTargetable = true;
          }
        }
      } else if (targetIdStr === String(opponentHeroId)) {
        // Заморозка на героя НЕ разрешена, лечение на врага — нет
        if (!isFreeze && !isHeal) opponentHeroTargetable = true;
      } else if (playerUnitIds.has(targetIdStr)) {
        // Свои юниты — только если это НЕ заморозка
        if (!isFreeze) friendlyTargets.add(targetIdStr);
      } else if (opponentUnitIds.has(targetIdStr)) {
        // Вражеские юниты — только если это НЕ лечение
        if (!isHeal) enemyTargets.add(targetIdStr);
      }
    }
    
    // Обработка target_is_hero без явного target_id
    if (a.target_is_hero && !a.target_id) {
      if (!isHeal && !isFreeze) opponentHeroTargetable = true;
    }
  });
  
  // Подсвечиваем ТОЛЬКО союзников из friendlyTargets
  playerUnits.forEach(unit => {
    const instanceId = unit.dataset.instanceId;
    if (friendlyTargets.has(instanceId)) {
      unit.classList.add('targetable-friendly');
      console.log('[ARENA] 💚 Союзник подсвечен для лечения/баффа:', instanceId);
      
      // Предпросмотр
      unit.onmouseenter = () => showDamagePreview(unit, false, actions.find(a => String(a.target_id) === String(instanceId)));
      unit.onmouseleave = () => hideDamagePreview(unit, false);
    }
  });
  
  // Подсвечиваем ТОЛЬКО врагов из enemyTargets
  opponentUnits.forEach(unit => {
    const instanceId = unit.dataset.instanceId;
    if (enemyTargets.has(instanceId)) {
      unit.classList.add('targetable-enemy');
      console.log('[ARENA] 🎯 Враг подсвечен:', instanceId);

      // Предпросмотр
      unit.onmouseenter = () => showDamagePreview(unit, false, actions.find(a => String(a.target_id) === String(instanceId)));
      unit.onmouseleave = () => hideDamagePreview(unit, false);
    }
  });
  
  // Подсвечиваем героя противника ТОЛЬКО если он в целях
  if (opponentHeroTargetable && opponentPanel) {
    opponentPanel.classList.add('targetable-enemy');
    
    // Предпросмотр
    opponentPanel.onmouseenter = () => showDamagePreview(opponentPanel, true, actions.find(a => a.target_is_hero));
    opponentPanel.onmouseleave = () => hideDamagePreview(opponentPanel, true);
  }
  
  // Подсвечиваем своего героя ТОЛЬКО если он в целях (лечение)
  if (playerHeroTargetable && playerPanel) {
    playerPanel.classList.add('targetable-friendly');
    console.log('[ARENA] 💚 Свой герой подсвечен для лечения');

    // Предпросмотр
    playerPanel.onmouseenter = () => showDamagePreview(playerPanel, true, actions.find(a => a.target_is_hero));
    playerPanel.onmouseleave = () => hideDamagePreview(playerPanel, true);
  }
}

function highlightAttackTargets(attackerId) {
  /**
   * LEGAL ACTIONS: Подсвечивает валидные цели атаки
   */
  const targets = getAttackTargets(attackerId);
  console.log('[ARENA] ⚔️ highlightAttackTargets:', targets.length, 'targets');
  
  const opponentUnits = document.querySelectorAll('#opponent-board-zone .board-unit-card');
  const opponentPanel = document.querySelector('.opponent-panel-root');
  
  // Собираем валидные target_id
  const validTargets = new Set();
  let heroTargetable = false;
  
  targets.forEach(a => {
    if (a.target_id) {
      validTargets.add(String(a.target_id));
    }
    if (a.target_is_hero) {
      heroTargetable = true;
    }
  });
  
  // Подсвечиваем существ
  opponentUnits.forEach(unit => {
    const instanceId = unit.dataset.instanceId;
    if (validTargets.has(instanceId)) {
      unit.classList.add('attack-target', 'targetable-enemy');
      
      // Предпросмотр
      unit.onmouseenter = () => showDamagePreview(unit, false, targets.find(a => a.target_id === instanceId));
      unit.onmouseleave = () => hideDamagePreview(unit, false);
    }
  });
  
  // Подсвечиваем героя
  if (heroTargetable && opponentPanel) {
    opponentPanel.classList.add('attack-target-hero', 'targetable-enemy');

    // Предпросмотр
    opponentPanel.onmouseenter = () => showDamagePreview(opponentPanel, true, targets.find(a => a.target_is_hero));
    opponentPanel.onmouseleave = () => hideDamagePreview(opponentPanel, true);
  }
}

function clearAttackTargets() {
  /**
   * Убирает подсветку целей атаки
   */
  const opponentUnits = document.querySelectorAll('#opponent-board-zone .board-unit-card');
  const opponentPanel = document.querySelector('.opponent-panel-root');
  
  opponentUnits.forEach(unit => {
    unit.classList.remove('attack-target', 'targetable-enemy');
    unit.onmouseenter = null;
    unit.onmouseleave = null;
    hideDamagePreview(unit, false);
  });
  
  if (opponentPanel) {
    opponentPanel.classList.remove('attack-target-hero', 'targetable-enemy');
    opponentPanel.onmouseenter = null;
    opponentPanel.onmouseleave = null;
    hideDamagePreview(opponentPanel, true);
  }
}

function executeTargetingPlay(targetId, isHero) {
  /**
   * КРИТИЧНО: Выполняет отправку запроса на применение карты с целью (зелье или воин)
   * @param {number|null} targetId - ID цели или null для героя
   * @param {boolean} isHero - true если цель герой
   */
  console.log('[ARENA] 🎯 executeTargetingPlay вызвана:', { targetId, isHero });
  
  if (interactionMode.type !== 'TARGETING') {
    console.error('[ARENA] ❌ executeTargetingPlay вызвана не в режиме TARGETING!');
    return;
  }
  
  const card = interactionMode.data;
  if (!card) {
    console.error('[ARENA] ❌ Нет данных карты в interactionMode!');
    return;
  }
  
  if (card.card_type === 'potion') {
    playPotionCard(card, targetId, isHero);
  } else {
    // Для воинов с боевым кличем: отправляем и target_id, и target_is_hero
    playCard(card, null, targetId, isHero);
  }
  
  // КРИТИЧНО: Сбрасываем режим
  resetInteractionMode();
}

function clearAllCardSelections() {
  /**
   * Универсальная функция очистки всех выделений и обработчиков
   */
  // Очищаем подсветку слотов игрока (для обычных карт)
  const playerSlots = document.querySelectorAll('#player-board-zone .board-slot');
  playerSlots.forEach(slot => {
    slot.classList.remove('droppable');
    slot.onclick = null;
  });
  
  // Убираем подсветку всех целей
  document.querySelectorAll('.targetable-friendly, .targetable-enemy').forEach(el => {
    el.classList.remove('targetable-friendly', 'targetable-enemy', 'potion-target', 'potion-target-hero');
    el.onmouseenter = null;
    el.onmouseleave = null;
    
    // Сбрасываем предпросмотр если он был активен
    const isHero = el.classList.contains('opponent-panel-root') || el.classList.contains('player-panel-root');
    hideDamagePreview(el, isHero);
  });
}

// ============================================
// УНИВЕРСАЛЬНАЯ ОБРАБОТКА КЛИКОВ ПО ЦЕЛЯМ
// ============================================

function handleGlobalTargetClick(targetId, isHero, event) {
  /**
   * Единая точка обработки кликов по целям (атака/targeting).
   * @param {string|number|null} targetId - instance_id цели или null для героя
   * @param {boolean} isHero - true если цель - герой
   * @param {Event} event - событие клика
   */
  console.log('[ARENA] 🎯 Target click:', { targetId, isHero, mode: interactionMode.type });
  
  if (interactionMode.type === 'TARGETING') {
    // LEGAL ACTIONS: Проверяем, что цель валидна
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    
    // Определяем, кликнули по союзнику или врагу
    let isFriendlyTarget = false;
    
    if (isHero) {
      const clickedElement = event?.target?.closest('.player-panel-root, .opponent-panel-root');
      isFriendlyTarget = clickedElement?.classList.contains('player-panel-root');
    } else {
      // Для юнитов проверяем, находится ли он в player-board-zone
      const unitElement = document.querySelector(`[data-instance-id="${targetId}"]`);
      isFriendlyTarget = unitElement?.closest('#player-board-zone') !== null;
    }
    
    console.log('[ARENA] 🎯 Клик по цели:', { targetId, isHero, isFriendlyTarget });
    
    // Проверяем валидность цели
    const isValidTarget = playActions.some(a => {
      if (isHero) {
        if (a.target_id) {
          const playerHeroId = currentState?.player?.hero?.instance_id;
          const opponentHeroId = currentState?.opponent?.hero?.instance_id;
          return String(a.target_id) === String(playerHeroId) || String(a.target_id) === String(opponentHeroId);
        }
        return a.target_is_hero === true;
      }
      return String(a.target_id) === String(targetId);
    });
    
    if (!isValidTarget && playActions.length > 0) {
      console.warn('[ARENA] ❌ Цель не в списке валидных для карты');
      resetInteractionMode(); // Сбрасываем режим при клике на недопустимую цель
      return;
    }
    
    // Передаем правильный target_id
    if (isHero) {
      const clickedElement = event?.target?.closest('.player-panel-root, .opponent-panel-root');
      const isPlayerHero = clickedElement?.classList.contains('player-panel-root');
      const heroId = isPlayerHero 
        ? currentState?.player?.hero?.instance_id 
        : currentState?.opponent?.hero?.instance_id;
      
      console.log('[ARENA] 🎯 Разыгрываем карту на героя:', { isPlayerHero, heroId });
      executeTargetingPlay(heroId, true);
    } else {
      console.log('[ARENA] 🎯 Разыгрываем карту на юнита:', targetId);
      executeTargetingPlay(targetId, false);
    }
    
  } else if (interactionMode.type === 'ATTACK') {
    // БЛОКИРОВКА: В режиме атаки НЕ должно быть кликов по врагам из TARGETING
    // Проверяем, что цель валидна для атаки
    if (!isValidAttackTarget(targetId, isHero)) {
      console.warn('[ARENA] ❌ Цель не валидна для атаки (taunt/legal_actions)');
      return;
    }
    
    attack(interactionMode.data.instance_id, isHero ? null : targetId, isHero);
    resetInteractionMode();
    
  } else {
    console.warn('[ARENA] ⚠️ Клик по цели в режиме NONE - игнорируем');
  }
}

// ============================================
// ПРЕДПРОСМОТР УРОНА (DAMAGE PREVIEW)
// ============================================

async function showDamagePreview(targetEl, isHero, targetData) {
  /**
   * Запрашивает предпросмотр урона у сервера и отображает его.
   */
  if (interactionMode.type === 'NONE') return;

  const targetId = isHero ? null : targetEl.dataset.instanceId;
  
  let action = null;
  if (interactionMode.type === 'ATTACK') {
    action = {
      type: 'attack',
      attacker_id: interactionMode.data.instance_id,
      target_id: targetId,
      target_is_hero: isHero
    };
  } else if (interactionMode.type === 'TARGETING') {
    action = {
      type: 'play_card',
      hand_index: interactionMode.data.handIndex ?? selectedCard?.index ?? 0,
      target_id: targetId,
      target_is_hero: isHero
    };
  }

  if (!action) return;

  try {
    const response = await fetch('/api/battle/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        user_id: parseInt(userId, 10),
        action: action
      })
    });

    if (!response.ok) return;

    const result = await response.json();
    if (!result.success || !result.preview_data) return;

    // Ищем дельту для текущей цели
    let targetInstanceId = targetId;
    
    if (isHero) {
      // Определяем, на какого героя навели: оппонента или своего
      const isOpponent = targetEl.classList.contains('opponent-panel-root');
      const hero = isOpponent ? (currentState.opponent?.hero) : (currentState.player?.hero);
      
      if (hero && hero.instance_id) {
        targetInstanceId = String(hero.instance_id);
      } else {
        // Фолбэк: ищем ID, который есть в preview_data и не является юнитом на столе
        const boardUnitIds = new Set();
        document.querySelectorAll('.board-unit-card').forEach(el => boardUnitIds.add(el.dataset.instanceId));
        
        for (const id of Object.keys(result.preview_data)) {
          if (!boardUnitIds.has(id)) {
            targetInstanceId = id;
            break;
          }
        }
      }
    }

    const delta = result.preview_data[targetInstanceId];
    if (delta === undefined || delta === 0) return;

    const hpTextEl = isHero 
      ? (targetEl.querySelector('.hp-value-large') || targetEl.querySelector('#opponent-hp-text'))
      : targetEl.querySelector('.unit-stat.health');

    if (!hpTextEl) return;

    const currentHp = parseInt(hpTextEl.textContent);
    const newHp = Math.max(0, currentHp + delta); // delta уже со знаком
    
    // Сохраняем оригинал если еще не сохранен
    if (!hpTextEl.dataset.originalHp) {
      hpTextEl.dataset.originalHp = hpTextEl.textContent;
    }
    
    const previewClass = delta > 0 ? 'heal-preview-text' : 'hp-preview-text';
    hpTextEl.innerHTML = `${currentHp} <span class="${previewClass}">→ ${newHp}</span>`;
  } catch (error) {
    console.error('[ARENA] Ошибка предпросмотра урона:', error);
  }
}

function hideDamagePreview(targetEl, isHero) {
  const hpTextEl = isHero 
    ? (targetEl.querySelector('.hp-value-large') || targetEl.querySelector('#opponent-hp-text'))
    : targetEl.querySelector('.unit-stat.health');

  if (hpTextEl && hpTextEl.dataset.originalHp) {
    hpTextEl.textContent = hpTextEl.dataset.originalHp;
    delete hpTextEl.dataset.originalHp;
  }
}

function handleAttackerClick(attackerCard) {
  console.log('[ARENA] Выбран атакующий:', attackerCard);
  
  // Если ход не наш, игнорируем
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход, атака невозможна');
    return;
  }
  
  // ИСПРАВЛЕНО: Приоритет свойству can_attack самой карты
  const unitCanAttack = attackerCard.can_attack || canAttack(attackerCard.instance_id);

  if (!unitCanAttack) {
    console.warn('[ARENA] ❌ Существо не может атаковать');
    return;
  }
  
  // Устанавливаем режим атаки
  interactionMode = {
    type: 'ATTACK',
    data: attackerCard
  };
  
  selectedAttacker = attackerCard;
  
  // Подсвечиваем атакующего
  document.querySelectorAll('.board-unit-card').forEach(el => {
    el.parentElement.classList.remove('attacker-selected');
  });
  
  const attackerEl = document.querySelector(`[data-instance-id="${attackerCard.instance_id}"]`);
  if (attackerEl) {
    attackerEl.parentElement.classList.add('attacker-selected');
  }
  
  // LEGAL ACTIONS: Подсвечиваем только валидные цели атаки
  highlightAttackTargets(attackerCard.instance_id);
  
  console.log('[ARENA] ⚔️ Режим атаки активирован - выберите цель');
}

function resetInteractionMode() {
  /**
   * Сбрасывает режим взаимодействия и очищает все подсветки
   */
  console.log('[ARENA] 🔄 Сброс режима взаимодействия. Был:', interactionMode.type);
  
  // Убираем класс targeting-active с body
  document.body.classList.remove('targeting-active');
  
  interactionMode = { type: 'NONE', data: null };
  selectedCard = null;
  selectedAttacker = null;
  
  // Очищаем все подсветки
  clearAllCardSelections();
  clearAttackTargets();
  
  // Убираем подсветку атакующего
  document.querySelectorAll('.board-slot').forEach(slot => {
    slot.classList.remove('attacker-selected');
  });
  
  // Убираем выделение с карт в руке
  document.querySelectorAll('.hand-card').forEach(el => {
    el.classList.remove('selected');
  });
  
  // Убираем подсветку целей для хила
  document.querySelectorAll('.targetable-friendly').forEach(el => {
    el.classList.remove('targetable-friendly', 'potion-target', 'potion-target-hero');
  });
}

// ============================================
// ДЕЙСТВИЯ ИГРОКА
// ============================================

async function playCard(card, position, targetId = null, targetIsHero = false) {
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход');
    return;
  }
  
  try {
    // Определяем индекс карты в руке
    const handIndex = selectedCard?.index ?? card.handIndex ?? 0;
    console.log('[ARENA] Розыгрыш карты:', card.name, 'hand_index:', handIndex, 'на позицию:', position, 'цель:', targetId);
    
    const response = await fetch('/api/battle/play-card', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        user_id: parseInt(userId, 10),
        hand_index: handIndex,
        card_id: card.card_id || card.id || card.instance_id,
        target_position: position,
        target_id: targetId,
        target_is_hero: targetIsHero
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Не удалось разыграть карту');
    }
    
    const result = await response.json();
    console.log('[ARENA] Карта разыграна:', result);
    
    // Обновляем состояние
    if (result.state) {
      currentState = result.state;
      renderBattleState(result.state);
      
      // Рендерим поля для обоих игроков
      renderBoard('player', (currentState.player || currentState).board || []);
      renderBoard('opponent', (currentState.opponent || currentState).board || []);
    }
    
  } catch (error) {
    console.error('[ARENA] Ошибка розыгрыша карты:', error);
    alert('Не удалось разыграть карту: ' + error.message);
  }
}

async function playPotionCard(card, targetId, targetIsHero) {
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход');
    return;
  }
  
  try {
    // Определяем индекс карты в руке
    const handIndex = card.handIndex ?? selectedCard?.index ?? 0;
    
    console.log('[ARENA] 🧪 Розыгрыш зелья:', {
      hand_index: handIndex,
      card_name: card.name,
      target_id: targetId,
      target_is_hero: targetIsHero
    });
    
    // Находим карту в руке для анимации исчезновения
    const handCard = document.querySelector(`.hand-card[data-index="${handIndex}"]`);
    
    const response = await fetch('/api/battle/play-card', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        user_id: parseInt(userId, 10),
        hand_index: handIndex,
        card_id: card.card_id || card.id || card.instance_id,
        target_id: targetId,
        target_is_hero: targetIsHero
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Не удалось разыграть зелье');
    }
    
    const result = await response.json();
    console.log('[ARENA] Зелье разыграно:', result);
    
    // Анимация: карта исчезает
    if (handCard) {
      handCard.style.transition = 'opacity 0.3s, transform 0.3s';
      handCard.style.opacity = '0';
      handCard.style.transform = 'scale(0.5)';
    }
    
    // Анимация: вспышка урона на цели
    let targetElement;
    if (targetIsHero) {
      targetElement = document.querySelector('.opponent-hp-block');
    } else {
      targetElement = document.querySelector(`[data-instance-id="${targetId}"]`);
    }
    
    if (targetElement) {
      triggerPotionDamageFlash(targetElement);
    }
    
    // Обновляем состояние после анимации
    setTimeout(() => {
      if (result.state) {
        currentState = result.state;
        renderBattleState(result.state);

        // ДОБАВЛЕНО: Принудительная перерисовка для обновления статуса атаки
        renderBoard('player', (currentState.player || currentState).board || []);
      }
    }, 400);
    
  } catch (error) {
    console.error('[ARENA] Ошибка розыгрыша зелья:', error);
    alert('Не удалось разыграть зелье: ' + error.message);
  }
}

function triggerPotionDamageFlash(targetElement) {
  /**
   * Анимация вспышки урона от зелья на цели
   */
  console.log('[ARENA] 💥 Вспышка урона от зелья');
  
  // Создаём элемент вспышки
  const flash = document.createElement('div');
  flash.className = 'potion-damage-flash';
  flash.textContent = '💥';
  
  // Позиционируем относительно цели
  const rect = targetElement.getBoundingClientRect();
  flash.style.position = 'fixed';
  flash.style.left = `${rect.left + rect.width / 2}px`;
  flash.style.top = `${rect.top + rect.height / 2}px`;
  flash.style.transform = 'translate(-50%, -50%)';
  flash.style.fontSize = '48px';
  flash.style.zIndex = '9999';
  flash.style.pointerEvents = 'none';
  flash.style.animation = 'potionFlash 0.6s ease-out';
  
  document.body.appendChild(flash);
  
  // Добавляем красную вспышку на саму цель
  targetElement.style.transition = 'box-shadow 0.3s';
  targetElement.style.boxShadow = '0 0 20px 5px rgba(220, 38, 38, 0.8)';
  
  // Удаляем эффекты через 600ms
  setTimeout(() => {
    flash.remove();
    targetElement.style.boxShadow = '';
  }, 600);
}

async function attack(attackerId, targetId, targetIsHero) {
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход');
    return;
  }
  
  try {
    console.log('[ARENA] Атака:', attackerId, '->', targetIsHero ? 'герой' : targetId);
    
    const response = await fetch('/api/battle/attack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        user_id: parseInt(userId, 10),
        attacker_id: attackerId,
        target_id: targetId,
        target_is_hero: targetIsHero
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Не удалось атаковать');
    }
    
    const result = await response.json();
    console.log('[ARENA] Атака выполнена:', result);
    
    // Обновляем состояние
    if (result.state) {
      currentState = result.state;
      renderBattleState(result.state);
      
      // Рендерим поля для обоих игроков
      renderBoard('player', (currentState.player || currentState).board || []);
      renderBoard('opponent', (currentState.opponent || currentState).board || []);
    }
    
  } catch (error) {
    console.error('[ARENA] Ошибка атаки:', error);
    alert('Не удалось атаковать: ' + error.message);
  }
}

async function endTurn() {
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход');
    return;
  }
  
  try {
    console.log('[ARENA] Завершение хода');
    
    const response = await fetch('/api/battle/end-turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        user_id: parseInt(userId, 10)
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Не удалось завершить ход');
    }
    
    const result = await response.json();
    console.log('[ARENA] Ход завершён:', result);
    
    // Обновляем состояние
    if (result.state) {
      currentState = result.state;
      renderBattleState(result.state);
    }
    
  } catch (error) {
    console.error('[ARENA] Ошибка завершения хода:', error);
    alert('Не удалось завершить ход: ' + error.message);
  }
}

async function surrender() {
  try {
    console.log('[ARENA] Сдача через Socket.IO');
    
    // Отправляем событие сдачи через Socket.IO
    socket.emit('surrender', {
      match_id: matchId,
      user_id: parseInt(userId, 10)
    });
    
    // Ждём подтверждения от сервера
    socket.once('surrender_ack', (data) => {
      console.log('[ARENA] Сдача подтверждена:', data);
      console.log('[ARENA] Сдача принята сервером, ждем game_over...');

      // Fallback: если game_over не пришел за 1500мс
      setTimeout(() => {
        const modal = document.getElementById('battle-result-modal');
        const isVisible = modal && (modal.style.display === 'flex' || modal.classList.contains('visible'));
        
        if (!isVisible) {
          console.warn('[ARENA] Server game_over timeout. Forcing local defeat screen.');
          showBattleResult(
            false, // isWinner
            -15,   // trophyDelta (заглушка)
            null,  // trophyTotal
            0,     // coinsDelta
            null   // coinsTotal
          );
        }
      }, 1500);
    });
    
    // Обработка ошибок
    socket.once('error', (error) => {
      console.error('[ARENA] Ошибка сдачи:', error);
      alert('Не удалось сдаться: ' + error.message);
    });
    
  } catch (error) {
    console.error('[ARENA] Ошибка сдачи:', error);
    alert('Не удалось сдаться: ' + error.message);
  }
}

// ============================================
// ЗАВЕРШЕНИЕ ИГРЫ
// ============================================

function handleGameOver(data) {
  console.log('[ARENA] 🏁 Игра завершена:', data);
  
  // Останавливаем таймер
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  
  // Извлекаем данные о победителе
  const winnerId = data.winner_id || data.winner || currentState?.winner_id;
  const isWinner = String(winnerId) === String(userId);
  
  console.log('[ARENA] 🎯 Результат: isWinner =', isWinner, '| winnerId =', winnerId, '| myId =', userId);
  
  // КРИТИЧНО: Извлекаем данные о трофеях из state (синхронизированы с БД)
  // Эти значения приходят из server.py после вызова db.update_user_trophies()
  const trophyDelta = parseInt(data.trophy_change || data.trophy_delta || currentState?.trophy_change || currentState?.trophy_delta, 10) || 0;
  const trophyTotal = parseInt(data.trophy_total || currentState?.trophy_total, 10) || null;
  
  // КРИТИЧНО: Извлекаем данные о монетах из state (синхронизированы с БД)
  // Эти значения приходят из server.py после вызова db.update_user_coins()
  const coinsDelta = parseInt(data.coins_delta || data.coins_change || currentState?.coins_delta || currentState?.coins_change, 10) || 0;
  const coinsTotal = parseInt(data.coins_total || currentState?.coins_total, 10) || null;
  
  console.log('[ARENA] 🏆 Трофеи: delta =', trophyDelta, '| total =', trophyTotal);
  console.log('[ARENA] 🪙 Монеты: delta =', coinsDelta, '| total =', coinsTotal);
  
  // Показываем экран результата с небольшой задержкой для драматического эффекта
  setTimeout(() => {
    showBattleResult(isWinner, trophyDelta, trophyTotal, coinsDelta, coinsTotal);
  }, 800);
}

function showBattleResult(isWinner, trophyDelta, trophyTotal, coinsDelta, coinsTotal) {
  const modal = document.getElementById('battle-result-modal');
  const icon = document.getElementById('result-icon');
  const title = document.getElementById('result-title');
  const trophyDeltaEl = document.getElementById('result-trophy-delta');
  const trophyTotalEl = document.getElementById('result-trophy-total');
  const trophySection = document.getElementById('result-trophy-section');
  const coinsDeltaEl = document.getElementById('result-coins-delta');
  const coinsTotalEl = document.getElementById('result-coins-total');
  const coinsSection = document.getElementById('result-coins-section');
  const shareBtn = document.getElementById('result-share-btn');
  const card = modal.querySelector('.result-card');
  
  if (!modal || !icon || !title) {
    console.error('[ARENA] Элементы модального окна результата не найдены');
    // Фолбэк на старый способ
    alert(isWinner ? '🎉 Победа!' : '💔 Поражение');
    setTimeout(() => window.location.href = '/', 1500);
    return;
  }
  
  // Устанавливаем иконку и заголовок
  if (isWinner) {
    icon.textContent = '🏆';
    title.textContent = 'Победа!';
    card.classList.add('victory');
    card.classList.remove('defeat');
  } else {
    icon.textContent = '💔';
    title.textContent = 'Поражение';
    card.classList.add('defeat');
    card.classList.remove('victory');
  }
  
  // Настройка кнопки "Поделиться"
  if (shareBtn) {
    const opponentName = document.getElementById('opponent-name-text')?.textContent || 'игрока';
    const winnerHP = isWinner ? (document.getElementById('player-hp-text')?.textContent || '0') : (document.getElementById('opponent-hp-text')?.textContent || '0');
    const turnCount = currentTurnCount;
    const botLink = 'https://t.me/extraarena_bot';
    
    let shareText = '';
    if (isWinner) {
      shareText = `Я победил игрока ${opponentName} в @extraarena_bot! 🏆\nБитва длилась ${turnCount} ходов. Мой герой выжил с ${winnerHP} HP!\n\nСможешь лучше? Принимай вызов! ⚔️`;
    } else {
      shareText = `Я сразился с ${opponentName} в @extraarena_bot! ⚔️\nБитва длилась ${turnCount} ходов. В следующий раз победа будет за мной!\n\nПрисоединяйся к битве! 🏆`;
    }
    
    const encodedText = encodeURIComponent(shareText);
    shareBtn.href = `https://t.me/share/url?url=${botLink}&text=${encodedText}`;
  }
  
  // Отображаем трофеи с анимацией счетчика
  const hasTrophyDelta = trophyDelta !== undefined && trophyDelta !== null && trophyDelta !== 0;
  const hasTrophyTotal = trophyTotal !== undefined && trophyTotal !== null;

  if (hasTrophyDelta || hasTrophyTotal) {
    if (trophySection) trophySection.style.display = 'flex';
    
    if (trophyDeltaEl) {
      if (hasTrophyDelta) {
        const deltaSign = trophyDelta > 0 ? '+' : '-';
        const deltaAbsValue = Math.abs(trophyDelta);
        trophyDeltaEl.className = 'trophy-delta ' + (trophyDelta > 0 ? 'positive' : 'negative');
        
        // Анимация счетчика трофеев (с абсолютным значением)
        animateCounter(trophyDeltaEl, 0, deltaAbsValue, 1000, deltaSign);
        
        // Показываем контейнер дельты (вместе с иконкой 🏆)
        if (trophyDeltaEl.parentElement) {
          trophyDeltaEl.parentElement.style.display = 'flex';
        }
      } else {
        // Если дельты нет (0), скрываем весь контейнер дельты
        if (trophyDeltaEl.parentElement) {
          trophyDeltaEl.parentElement.style.display = 'none';
        }
      }
    }
    
    if (trophyTotalEl && hasTrophyTotal) {
      // КРИТИЧНО: Анимация счетчика общих трофеев (используем state.trophy_total из БД)
      const startValue = hasTrophyDelta ? Math.max(0, trophyTotal - trophyDelta) : trophyTotal;
      animateCounter(trophyTotalEl, startValue, trophyTotal, 1000);
    }
  } else if (trophySection) {
    // Скрываем секцию трофеев, только если нет ни дельты, ни общего количества
    trophySection.style.display = 'none';
  }
  
  // ДОБАВЛЕНО: Отображаем монеты с анимацией счетчика
  const hasCoinsDelta = coinsDelta !== undefined && coinsDelta !== null && coinsDelta !== 0;
  const hasCoinsTotal = coinsTotal !== undefined && coinsTotal !== null;

  if (hasCoinsDelta || hasCoinsTotal) {
    if (coinsSection) coinsSection.style.display = 'flex';
    
    if (coinsDeltaEl) {
      if (hasCoinsDelta) {
        const deltaSign = coinsDelta > 0 ? '+' : '-';
        const deltaAbsValue = Math.abs(coinsDelta);
        coinsDeltaEl.className = 'coins-delta ' + (coinsDelta > 0 ? 'positive' : 'negative');
        
        // Анимация счетчика монет (с абсолютным значением)
        animateCounter(coinsDeltaEl, 0, deltaAbsValue, 1200, deltaSign);
        
        if (coinsDeltaEl.parentElement) {
          coinsDeltaEl.parentElement.style.display = 'flex';
        }
      } else {
        if (coinsDeltaEl.parentElement) {
          coinsDeltaEl.parentElement.style.display = 'none';
        }
      }
    }
    
    if (coinsTotalEl && hasCoinsTotal) {
      // КРИТИЧНО: Анимация счетчика общих монет (используем state.coins_total из БД)
      const startValue = hasCoinsDelta ? Math.max(0, coinsTotal - coinsDelta) : coinsTotal;
      animateCounter(coinsTotalEl, startValue, coinsTotal, 1200);
    }
  } else if (coinsSection) {
    // Скрываем секцию монет, если нет ни дельты, ни общего количества
    coinsSection.style.display = 'none';
  }
  
  // Показываем модальное окно
  modal.style.display = 'flex';
  modal.setAttribute('aria-hidden', 'false');
  
  // Добавляем анимацию появления
  requestAnimationFrame(() => {
    modal.classList.add('visible');
  });
  
  console.log('[ARENA] ✅ Модальное окно результата показано');
}

// ДОБАВЛЕНО: Функция анимации счетчика
function animateCounter(element, from, to, duration, prefix = '') {
  if (!element) return;
  
  const startTime = Date.now();
  const diff = to - from;
  
  const updateCounter = () => {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Ease-out функция для плавного замедления
    const easeOut = 1 - Math.pow(1 - progress, 3);
    
    const current = Math.round(from + diff * easeOut);
    element.textContent = `${prefix}${current}`;
    
    if (progress < 1) {
      requestAnimationFrame(updateCounter);
    } else {
      element.textContent = `${prefix}${to}`;
    }
  };
  
  requestAnimationFrame(updateCounter);
}

// ============================================
// ВИЗУАЛЬНЫЙ ЭФФЕКТ ИСЦЕЛЕНИЯ
// ============================================

function triggerHealAnimation(isPlayer) {
  /**
   * Запускает анимацию исцеления для указанного игрока.
   * 
   * @param {boolean} isPlayer - true для игрока, false для оппонента
   */
  console.log('[ARENA] 💚 Triggering heal animation for', isPlayer ? 'player' : 'opponent');
  
  // Получаем HP блок
  const hpBlock = isPlayer 
    ? document.querySelector('.player-hp-block')
    : document.querySelector('.opponent-hp-block');
  
  if (!hpBlock) {
    console.warn('[ARENA] HP block not found for healing animation');
    return;
  }
  
  // Добавляем класс анимации
  hpBlock.classList.add('heal-active');
  
  // Создаём всплывающий индикатор "+HP"
  const healIndicator = document.createElement('div');
  healIndicator.className = 'heal-indicator';
  healIndicator.textContent = '+';
  hpBlock.appendChild(healIndicator);
  
  // Удаляем класс и индикатор через 1.5 секунды
  setTimeout(() => {
    hpBlock.classList.remove('heal-active');
    if (healIndicator.parentNode) {
      healIndicator.remove();
    }
  }, 1500);
}

// ============================================
// ВИЗУАЛЬНЫЙ ЭФФЕКТ УРОНА (DAMAGE)
// ============================================

function triggerDamageEffects(element, damageAmount = null) {
  /**
   * Запускает сочные эффекты при получении урона
   */
  if (!element) return;

  // 1. Reddening flash + Shake
  element.classList.add('damage-flash');
  element.classList.add('shake-heavy');
  
  setTimeout(() => {
    element.classList.remove('damage-flash');
    element.classList.remove('shake-heavy');
  }, 400);

  // 2. Particles
  spawnDamageParticles(element);

  // 3. Floating Damage Text (Красные цифры)
  if (damageAmount !== null && damageAmount > 0) {
    spawnFloatingText(element, `-${damageAmount}`, 'damage-text-float');
  }
}

/**
 * Создает всплывающий текст над элементом (урон/хил)
 */
function spawnFloatingText(element, text, className) {
  const rect = element.getBoundingClientRect();
  const floatingText = document.createElement('div');
  floatingText.className = `floating-text ${className}`;
  floatingText.textContent = text;
  
  // Позиционируем в центре элемента по горизонтали и сверху по вертикали
  floatingText.style.position = 'fixed';
  floatingText.style.left = `${rect.left + rect.width / 2}px`;
  floatingText.style.top = `${rect.top}px`;
  floatingText.style.zIndex = '10002';
  
  document.body.appendChild(floatingText);
  
  // Удаляем через секунду
  setTimeout(() => {
    floatingText.remove();
  }, 1000);
}

function spawnDamageParticles(element) {
  /**
   * Создает разлетающиеся частицы из центра элемента
   */
  const rect = element.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  
  const particleCount = 12 + Math.floor(Math.random() * 6); // 12-18 частиц
  
  // Варианты цветов (от желтого к темно-красному)
  const colors = ['#FFD700', '#FFA500', '#FF4500', '#FF0000', '#8B0000'];
  
  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'damage-particle';
    
    // Случайный размер
    const size = 3 + Math.random() * 6;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Случайный цвет и свечение
    const color = colors[Math.floor(Math.random() * colors.length)];
    particle.style.background = color;
    particle.style.boxShadow = `0 0 10px ${color}`;
    
    // Начальная позиция
    particle.style.left = centerX + 'px';
    particle.style.top = centerY + 'px';
    
    document.body.appendChild(particle);
    
    // Случайное направление разлета (более взрывное)
    const angle = Math.random() * Math.PI * 2;
    const velocity = 4 + Math.random() * 8;
    let vx = Math.cos(angle) * velocity;
    let vy = Math.sin(angle) * velocity;
    
    const gravity = 0.3; // Сила притяжения
    let opacity = 1;
    let posX = centerX;
    let posY = centerY;
    
    const animate = () => {
      posX += vx;
      posY += vy;
      vy += gravity; // Падение вниз со временем
      
      opacity -= 0.025; // Постепенное исчезновение
      
      particle.style.left = posX + 'px';
      particle.style.top = posY + 'px';
      particle.style.opacity = opacity;
      
      // Масштабирование при исчезновении
      particle.style.transform = `scale(${opacity})`;
      
      if (opacity > 0) {
        requestAnimationFrame(animate);
      } else {
        particle.remove();
      }
    };
    
    requestAnimationFrame(animate);
  }
}

// ============================================
// ОПИСАНИЕ КАРТ (TOOLTIP / MODAL)
// ============================================

/**
 * Показывает глобальное модальное окно с информацией
 */
function showGlobalModal(title, text) {
  let tooltip = document.getElementById('card-description-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'card-description-tooltip';
    tooltip.className = 'card-description-tooltip';
    document.body.appendChild(tooltip);
  }

  tooltip.innerHTML = `<h3 style="color: #FF9400; margin-bottom: 12px; font-size: 20px;">${title}</h3><p style="line-height: 1.4;">${text}</p><div style="margin-top: 15px; font-size: 12px; color: rgba(255,255,255,0.5);">Нажмите в любом месте, чтобы закрыть</div>`;
  tooltip.style.display = 'block';
  tooltip.setAttribute('aria-hidden', 'false');
}

function showCardDescription(card, anchorEl) {
  // Добавляем лог данных карты для отладки
  console.log('[ARENA] Full card data:', card);

  const cardName = card.name || 'Карта';
  const cardLevel = card.level || 1;
  const description = card.description || card.text || 'Нет описания';
  
  const title = `${cardName} (Ур. ${cardLevel})`;
  showGlobalModal(title, description);
}

// ============================================
// ФОНОВАЯ МУЗЫКА АРЕНЫ
// ============================================

function initArenaMusic() {
  const music = document.getElementById('arena-bg-music');
  if (!music) {
    console.warn('[ARENA] Элемент arena-bg-music не найден');
    return;
  }
  
  let musicStarted = false;
  
  // Запускаем музыку по первому клику пользователя (обход ограничений браузера)
  const startMusic = () => {
    if (musicStarted) return;
    
    music.volume = 0.3; // Умеренная громкость
    music.play().then(() => {
      console.log('[ARENA] 🎵 Фоновая музыка запущена');
      musicStarted = true;
    }).catch(err => {
      console.warn('[ARENA] Не удалось запустить музыку:', err);
    });
    
    // Удаляем обработчик после первого запуска
    document.body.removeEventListener('click', startMusic);
  };
  
  document.body.addEventListener('click', startMusic, { once: true });
}

// ============================================
// ОБРАБОТЧИКИ UI
// ============================================

function bindUIHandlers() {
  // Глобальный слушатель для закрытия окон
  document.addEventListener('click', (e) => {
    const logOverlay = document.getElementById('battle-log-overlay');
    const tooltip = document.getElementById('card-description-tooltip');

    if (logOverlay && logOverlay.getAttribute('aria-hidden') === 'false') {
      if (!logOverlay.contains(e.target)) {
        logOverlay.setAttribute('aria-hidden', 'true');
      }
    }

    if (tooltip && tooltip.style.display === 'block') {
      if (!tooltip.contains(e.target)) {
        tooltip.style.display = 'none';
        tooltip.setAttribute('aria-hidden', 'true');
      }
    }
  });

  // Кнопка завершения хода
  const endTurnBtn = document.getElementById('end-turn-button');
  if (endTurnBtn) {
    endTurnBtn.addEventListener('click', () => {
      endTurn();
    });
  }
  
  // Кнопка сдачи
  const surrenderBtn = document.getElementById('surrender-button');
  if (surrenderBtn) {
    surrenderBtn.addEventListener('click', () => {
      openSurrenderModal();
    });
  }

  // Кнопка лога боя
  const logBtn = document.getElementById('battle-log-btn');
  const logOverlay = document.getElementById('battle-log-overlay');
  if (logBtn && logOverlay) {
    logBtn.addEventListener('click', (e) => {
      e.stopPropagation(); // Важно: предотвращаем закрытие сразу после открытия
      const isHidden = logOverlay.getAttribute('aria-hidden') === 'true';
      
      if (isHidden) {
        // Открываем лог
        logOverlay.setAttribute('aria-hidden', 'false');
      } else {
        // Закрываем лог
        logOverlay.setAttribute('aria-hidden', 'true');
      }
    });
  }
  
  // Модальное окно сдачи
  const surrenderModal = document.getElementById('surrender-modal');
  if (surrenderModal) {
    // Кнопка "Продолжить бой"
    const cancelBtn = surrenderModal.querySelector('[data-action="cancel"]');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => {
        closeSurrenderModal();
      });
    }
    
    // Кнопка "Сдаться"
    const confirmBtn = surrenderModal.querySelector('[data-action="confirm"]');
    if (confirmBtn) {
      confirmBtn.addEventListener('click', () => {
        closeSurrenderModal();
        surrender();
      });
    }
    
    // Закрытие по клику на overlay
    const overlay = surrenderModal.querySelector('.surrender-modal-overlay');
    if (overlay) {
      overlay.addEventListener('click', () => {
        closeSurrenderModal();
      });
    }
  }
  
  // Клик по герою оппонента - ЕДИНАЯ ТОЧКА ОБРАБОТКИ
  const opponentPanel = document.querySelector('.opponent-panel-root');
  if (opponentPanel) {
    opponentPanel.addEventListener('click', (e) => {
      e.stopPropagation();
      
      // КРИТИЧНО: Только в режиме ATTACK или TARGETING передаем клик
      if (interactionMode.type === 'ATTACK' || interactionMode.type === 'TARGETING') {
        handleGlobalTargetClick(null, true, e);
      }
    });
  }
  
  // ДОБАВЛЕНО: Клик по своему герою (для хила)
  const playerPanel = document.querySelector('.player-panel-root');
  if (playerPanel) {
    playerPanel.addEventListener('click', (e) => {
      // Только в режиме TARGETING (хил/бафф)
      if (interactionMode.type === 'TARGETING') {
        e.stopPropagation();
        handleGlobalTargetClick(null, true, e);
      }
    });
  }
  
  // Кнопка закрытия модального окна результата боя
  const resultCloseBtn = document.getElementById('result-close-btn');
  if (resultCloseBtn) {
    resultCloseBtn.addEventListener('click', () => {
      console.log('[ARENA] Возврат в главное меню');
      window.location.href = '/';
    });
  }
  
  // Закрытие по клику на overlay
  const resultOverlay = document.querySelector('.result-overlay-bg');
  if (resultOverlay) {
    resultOverlay.addEventListener('click', () => {
      console.log('[ARENA] Возврат в главное меню (клик на overlay)');
      window.location.href = '/';
    });
  }
}

function openSurrenderModal() {
  const modal = document.getElementById('surrender-modal');
  if (modal) {
    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');
  }
}

function closeSurrenderModal() {
  const modal = document.getElementById('surrender-modal');
  if (modal) {
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
  }
}


