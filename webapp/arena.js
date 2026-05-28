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
let authToken = null;
let currentState = null;
let socketJoined = false;
let clientReadySent = false;
let socketJoinRetryTimer = null;
let prebattleRendered = false;
let prebattleSequenceStarted = false;
let prebattleComplete = false;
let activeBattleModal = null;
let pendingInitialBattleState = null;
let arenaWaitingOverlay = null;

// Для drag & drop
let selectedCard = null;
let selectedAttacker = null;

// Для детекции исцеления/урона
let previousPlayerHP = null;
let previousOpponentHP = null;
let previousUnitHPs = {}; // { instanceId: hp }
let previousArenaSoundSnapshot = null;
let previousArenaHapticSnapshot = null;
let pendingArenaStateHaptic = null;
let lastArenaHapticAt = 0;
let arenaLastHapticKey = '';
let currentTurnCount = 0;

const ARENA_SFX = {
  battleStart: 'arena-sfx-battle-start',
  cardAttacked: 'arena-sfx-card-attacked',
  cardDeath: 'arena-sfx-card-death',
  cardFrozen: 'arena-sfx-card-frozen',
  cardHeal: 'arena-sfx-card-heal',
  cardSelected: 'arena-sfx-card-selected',
  heroDamage: 'arena-sfx-hero-damage',
  heroDeath: 'arena-sfx-hero-death',
  nextMove: 'arena-sfx-next-move'
};

const SURRENDER_TROPHY_TIERS = [
  { min: 0, max: 300, penalty: -7 },
  { min: 301, max: 700, penalty: -10 },
  { min: 701, max: 2500, penalty: -25 },
  { min: 2501, max: 5000, penalty: -25 },
  { min: 5001, max: 8000, penalty: -30 },
  { min: 8001, max: 99999, penalty: -30 }
];

const ARENA_BAD_CONNECTION_THRESHOLD_MS = 1200;
const ARENA_HEALTH_PING_INTERVAL_MS = 15000;
const ARENA_CONNECTION_FAILURE_DELAY_MS = 10000;
let arenaHealthInterval = null;
let arenaHealthStopped = false;
let arenaBadPingDismissed = false;
let arenaConnectionIssueSince = null;
let arenaConnectionIssueTimer = null;

const ARENA_HAPTIC_PRIORITY = {
  light: 1,
  selection: 1,
  success: 2,
  medium: 3,
  warning: 4,
  heavy: 5,
  error: 6
};

function isAndroidArenaShell() {
  return !!(window.ExtraArenaApp && typeof window.ExtraArenaApp.haptic === 'function');
}

function arenaHaptic(style, options = {}) {
  if (!isAndroidArenaShell()) return;
  const normalized = style || 'light';
  const key = options.key || normalized;
  const minInterval = Number(options.minInterval ?? 55);
  const now = Date.now();
  if (key === arenaLastHapticKey && now - lastArenaHapticAt < minInterval) return;
  arenaLastHapticKey = key;
  lastArenaHapticAt = now;
  try {
    window.ExtraArenaApp.haptic(normalized);
  } catch (e) {}
}

function queueArenaStateHaptic(style, reason) {
  if (!style) return;
  const nextPriority = ARENA_HAPTIC_PRIORITY[style] || 0;
  const currentPriority = pendingArenaStateHaptic
    ? (ARENA_HAPTIC_PRIORITY[pendingArenaStateHaptic.style] || 0)
    : -1;
  if (!pendingArenaStateHaptic || nextPriority >= currentPriority) {
    pendingArenaStateHaptic = { style, reason: reason || style };
  }
}

function flushArenaStateHaptic() {
  if (!pendingArenaStateHaptic) return;
  const effect = pendingArenaStateHaptic;
  pendingArenaStateHaptic = null;
  arenaHaptic(effect.style, { key: 'state-' + effect.reason, minInterval: 180 });
}

function recordArenaStateHaptic(playerState, opponentState) {
  const next = createArenaSoundSnapshot(playerState, opponentState);
  const prev = previousArenaHapticSnapshot;
  previousArenaHapticSnapshot = next;
  pendingArenaStateHaptic = null;

  if (!prev) return;

  if (next.playerHeroHp <= 0 && prev.playerHeroHp > 0) {
    queueArenaStateHaptic('error', 'hero-death');
  } else if (next.playerHeroHp < prev.playerHeroHp) {
    queueArenaStateHaptic('heavy', 'hero-damage');
  } else if (next.playerHeroHp > prev.playerHeroHp && next.playerHeroHp > 0) {
    queueArenaStateHaptic('success', 'hero-heal');
  }

  Object.entries(prev.units).forEach(([id, oldUnit]) => {
    if (oldUnit.side !== 'player') return;
    if (!next.units[id] && oldUnit.hp > 0) {
      queueArenaStateHaptic('heavy', 'card-death');
    }
  });

  Object.entries(next.units).forEach(([id, newUnit]) => {
    if (newUnit.side !== 'player') return;
    const oldUnit = prev.units[id];
    if (!oldUnit) return;
    if (newUnit.hp <= 0 && oldUnit.hp > 0) {
      queueArenaStateHaptic('heavy', 'card-death');
    } else if (newUnit.hp < oldUnit.hp) {
      queueArenaStateHaptic('medium', 'card-damage');
    } else if (newUnit.hp > oldUnit.hp) {
      queueArenaStateHaptic('success', 'card-heal');
    }
  });

  flushArenaStateHaptic();
}

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
  if (isArenaWaitingForPlayers(currentState)) return false;
  if (!cachedLegalActions || cachedLegalActions.length === 0) return true; // fallback
  return cachedLegalActions.some(a => a.type === 'play_card' && a.hand_index === handIndex);
}

function getClassicModeParams(state = currentState) {
  return (state && state.mode_config && state.mode_config.classic) || {};
}

function getModeId(state = currentState) {
  return (state && (state.game_mode || (state.mode_config && state.mode_config.mode_id))) || 'classic';
}

function isArenaWaitingForPlayers(state = currentState) {
  return state && state.match_status === 'waiting_for_players';
}

function updateArenaWaitingOverlay(state = currentState) {
  const waiting = isArenaWaitingForPlayers(state);
  if (!waiting) {
    if (arenaWaitingOverlay) arenaWaitingOverlay.style.display = 'none';
    return;
  }
  if (!arenaWaitingOverlay) {
    arenaWaitingOverlay = document.createElement('div');
    arenaWaitingOverlay.id = 'arena-waiting-overlay';
    arenaWaitingOverlay.style.cssText = 'position:fixed;inset:0;z-index:12500;background:rgba(5,3,14,.82);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:22px;pointer-events:auto';
    arenaWaitingOverlay.innerHTML = ''
      + '<section style="width:100%;max-width:360px;border-radius:18px;padding:22px 18px;background:linear-gradient(180deg,rgba(27,17,48,.98),rgba(13,8,27,.98));border:1px solid rgba(124,92,191,.45);box-shadow:0 24px 80px rgba(0,0,0,.58);text-align:center">'
      + '<div style="width:46px;height:46px;margin:0 auto 14px;border-radius:50%;border:3px solid rgba(245,146,30,.22);border-top-color:#f5921e;animation:spin .9s linear infinite"></div>'
      + '<h2 style="margin:0 0 8px;color:#fff;font:1000 22px/1.1 sans-serif">Ждем соперника</h2>'
      + '<p style="margin:0;color:#c9c1d9;font:600 13px/1.45 sans-serif">Бой начнется, когда оба игрока загрузят арену.</p>'
      + '</section>';
    document.body.appendChild(arenaWaitingOverlay);
  }
  arenaWaitingOverlay.style.display = 'flex';
}

function getPrebattleModeLabel(state = currentState) {
  const modeId = String(getModeId(state) || 'classic').toLowerCase();
  if (modeId === 'classic' || modeId.includes('classic')) return 'classic';
  if (modeId.includes('extra_arena') || modeId.includes('extraarena')) return 'extraarena';
  return modeId.replace(/[_\s-]+/g, '');
}

function isPotionCard(card) {
  return (card && String(card.card_type || '').toLowerCase() === 'potion');
}

function getRawManaCost(card) {
  const raw = card && (card.mana ?? card.mana_cost ?? 0);
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getEffectiveManaCost(card, state = currentState) {
  const classic = getClassicModeParams(state);
  if (classic.spells_free === true && isPotionCard(card)) return 0;
  return getRawManaCost(card);
}

function hasEnoughManaForCard(card, state = currentState) {
  const playerMana = Number(state?.player?.mana ?? 0);
  return getEffectiveManaCost(card, state) <= playerMana;
}

function isPowerMaxMode(state = currentState) {
  return getClassicModeParams(state).card_level_mode === 'max';
}

function isSummonReadyMode(state = currentState) {
  return getClassicModeParams(state).summon_ready_on_play === true;
}

function getModeUiMeta(state = currentState) {
  const modeId = getModeId(state);
  const classic = getClassicModeParams(state);
  if (classic.card_level_mode === 'max' || modeId.includes('powermax')) {
    return { key: 'powermax', icon: '10', label: 'PowerMax', title: 'Все карты максимального уровня' };
  }
  if (classic.spells_free === true || modeId.includes('spellstorm')) {
    return { key: 'spellstorm', icon: '0', label: 'SpellStorm', title: 'Заклинания бесплатные' };
  }
  if (classic.summon_ready_on_play === true || modeId.includes('blitzkrieg')) {
    return { key: 'blitzkrieg', icon: '⚡', label: 'Blitzkrieg', title: 'Существа готовы после выставления' };
  }
  if (classic.sudden_death_enabled === true || modeId.includes('sudden_death')) {
    return { key: 'sudden-death', icon: '-HP', label: 'Sudden Death', title: 'Герой теряет здоровье в начале своих ходов' };
  }
  if (modeId && modeId !== 'classic') {
    const label = state?.mode_config?.label || modeId;
    return { key: 'generic', icon: '✦', label: label.replace('ExtraArena ', ''), title: label };
  }
  return null;
}

function formatCompactNumber(value) {
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed)) return '0';
  return parsed.toLocaleString('ru-RU');
}

function firstLetter(value, fallback) {
  const text = String(value || fallback || '?').trim();
  return (text.charAt(0) || '?').toUpperCase();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value == null ? '' : String(value);
}

function setOptionalImage(imgId, wrapperSelector, url) {
  const img = document.getElementById(imgId);
  const wrapper = img ? img.closest(wrapperSelector) : null;
  if (!img || !wrapper) return;
  const hasUrl = Boolean(url);
  wrapper.classList.toggle('has-image', hasUrl);
  if (hasUrl) img.src = url;
  else img.removeAttribute('src');
}

function playArenaSfx(name, options = {}) {
  if (window._sfxEnabled === false) return;
  const audioId = ARENA_SFX[name] || name;
  const baseAudio = document.getElementById(audioId);
  if (!baseAudio) return;

  const volume = options.volume ?? 0.72;
  const audio = baseAudio.paused ? baseAudio : baseAudio.cloneNode(true);
  audio.volume = volume;

  if (audio !== baseAudio) {
    audio.addEventListener('ended', () => audio.remove(), { once: true });
    document.body.appendChild(audio);
  }

  try {
    audio.currentTime = 0;
  } catch (e) {}

  audio.play().catch(err => {
    console.warn('[ARENA] Не удалось проиграть SFX:', audioId, err);
    if (audio !== baseAudio) audio.remove();
  });
}

function primeArenaSfx() {
  Object.values(ARENA_SFX).forEach(id => {
    const audio = document.getElementById(id);
    if (!audio) return;
    try { audio.load(); } catch (e) {}
  });
}

function initArenaSfx() {
  window.playArenaSfx = playArenaSfx;
  primeArenaSfx();
  const unlock = () => {
    primeArenaSfx();
    document.removeEventListener('pointerdown', unlock);
    document.removeEventListener('touchstart', unlock);
  };
  document.addEventListener('pointerdown', unlock, { once: true, passive: true });
  document.addEventListener('touchstart', unlock, { once: true, passive: true });
}

function getModifierLabel(state) {
  const modeId = String(getModeId(state) || '').toLowerCase();
  const classic = getClassicModeParams(state);
  const modifiers = [];
  if (classic.card_level_mode === 'max' || modeId.includes('powermax')) modifiers.push('PowerMax');
  if (classic.spells_free === true || modeId.includes('spellstorm')) modifiers.push('SpellStorm');
  if (classic.summon_ready_on_play === true || modeId.includes('blitzkrieg')) modifiers.push('Blitzkrieg');
  if (classic.sudden_death_enabled === true || modeId.includes('sudden_death')) modifiers.push('SuddenDeath');
  return modifiers.length > 0 ? modifiers.join(' · ') : 'Без модификаторов';
}

function renderPrebattleProfile(prefix, profile, fallbackName) {
  const name = profile?.name || fallbackName;
  const title = profile?.title || '';
  const clan = profile?.clan || '';
  const titleRarity = String(profile?.title_rarity || profile?.rarity || '').toLowerCase();
  setText('prebattle-' + prefix + '-name', name);
  setText('prebattle-' + prefix + '-clan', clan);
  setText('prebattle-' + prefix + '-trophies', formatCompactNumber(profile?.trophies));
  setText('prebattle-' + prefix + '-letter', firstLetter(name, fallbackName));
  setText('prebattle-' + prefix + '-bg-fallback', firstLetter(name, fallbackName));
  setOptionalImage('prebattle-' + prefix + '-avatar', '.prebattle-avatar', profile?.avatar_url);
  setOptionalImage('prebattle-' + prefix + '-bg', '.prebattle-profile-bg', profile?.background_url);

  const titleEl = document.getElementById('prebattle-' + prefix + '-title');
  if (titleEl) {
    titleEl.textContent = title;
    titleEl.className = 'prebattle-title-tag';
    if (title) {
      titleEl.classList.add('has-title');
      if (titleRarity === 'mythic' || titleRarity === 'mythical') titleEl.classList.add('title-mythical');
      else if (titleRarity === 'limited') titleEl.classList.add('title-limited');
    }
  }
}

function renderPrebattleScreen(state) {
  const screen = document.getElementById('prebattle-screen');
  if (!screen || !state) return;

  const opponent = state.opponent || {};
  setText('prebattle-mode-label', getPrebattleModeLabel(state));
  setText('prebattle-modifier-label', getModifierLabel(state));
  renderPrebattleProfile('opponent', opponent, 'Оппонент');
  setText('prebattle-status-text', 'Противник найден');
  screen.setAttribute('aria-hidden', 'false');
  screen.classList.remove('is-hidden');
  prebattleRendered = true;
}

function hidePrebattleScreen() {
  const screen = document.getElementById('prebattle-screen');
  if (!screen) return;
  screen.setAttribute('aria-hidden', 'true');
  screen.classList.add('is-hidden');
}

function playCountdownValue(value, isVs) {
  const el = document.getElementById('countdown-value');
  if (!el) return Promise.resolve();
  if (isVs) {
    arenaHaptic('heavy', { key: 'countdown-vs', minInterval: 240 });
  } else {
    arenaHaptic('selection', { key: 'countdown-' + value, minInterval: 240 });
  }
  el.textContent = value;
  el.classList.toggle('is-vs', Boolean(isVs));
  el.classList.remove('is-swapping');
  void el.offsetWidth;
  el.classList.add('is-swapping');
  return new Promise(resolve => setTimeout(resolve, isVs ? 950 : 1050));
}

async function startPrebattleSequence() {
  if (prebattleSequenceStarted) return;
  prebattleSequenceStarted = true;
  const screen = document.getElementById('prebattle-screen');
  if (screen) {
    screen.setAttribute('aria-hidden', 'false');
    screen.classList.remove('is-hidden');
  }
  try {
    await playCountdownValue('3', false);
    await playCountdownValue('2', false);
    await playCountdownValue('1', false);
    playArenaSfx('battleStart', { volume: 0.85 });
    await playCountdownValue('VS', true);
  } finally {
    prebattleComplete = true;
    if (pendingInitialBattleState) {
      const stateToRender = pendingInitialBattleState;
      pendingInitialBattleState = null;
      renderBattleState(stateToRender);
    }
    hidePrebattleScreen();
    trySendClientReady();
  }
}

function trySendClientReady() {
  if (clientReadySent || !socket || !socket.connected || !socketJoined || !prebattleComplete) return;
  clientReadySent = true;
  console.log('[SOCKET.IO] Отправка сигнала client_ready после prebattle...');
  socket.emit('client_ready', { match_id: matchId });
}

function getArenaHeroHp(playerState) {
  const hp = playerState?.hero?.hp ?? playerState?.hp ?? 30;
  return Math.max(0, Number(hp) || 0);
}

function getArenaCardHp(card) {
  const hp = card?.hp ?? card?.hp_current ?? card?.health ?? 0;
  return Math.max(0, Number(hp) || 0);
}

function getArenaCardId(card) {
  const id = card?.instance_id ?? card?.id ?? card?.card_id;
  return id == null ? null : String(id);
}

function createArenaSoundSnapshot(playerState, opponentState) {
  const units = {};
  [
    ['player', playerState?.board || []],
    ['opponent', opponentState?.board || []]
  ].forEach(([side, board]) => {
    board.forEach(card => {
      const id = getArenaCardId(card);
      if (!id) return;
      units[id] = {
        hp: getArenaCardHp(card),
        frozen: card.is_frozen === true,
        side
      };
    });
  });

  return {
    playerHeroHp: getArenaHeroHp(playerState),
    opponentHeroHp: getArenaHeroHp(opponentState),
    units
  };
}

function playHeroHpSfx(prevHp, nextHp) {
  if (nextHp < prevHp) {
    playArenaSfx(nextHp <= 0 ? 'heroDeath' : 'heroDamage');
  } else if (nextHp > prevHp && nextHp > 0) {
    playArenaSfx('cardHeal');
  }
}

function processArenaStateSfx(playerState, opponentState) {
  recordArenaStateHaptic(playerState, opponentState);
  const next = createArenaSoundSnapshot(playerState, opponentState);
  const prev = previousArenaSoundSnapshot;
  previousArenaSoundSnapshot = next;

  if (!prev) return;

  playHeroHpSfx(prev.playerHeroHp, next.playerHeroHp);
  playHeroHpSfx(prev.opponentHeroHp, next.opponentHeroHp);

  Object.entries(prev.units).forEach(([id, oldUnit]) => {
    if (!next.units[id] && oldUnit.hp > 0) {
      playArenaSfx('cardDeath');
    }
  });

  Object.entries(next.units).forEach(([id, newUnit]) => {
    const oldUnit = prev.units[id];
    if (!oldUnit) return;
    if (newUnit.hp < oldUnit.hp) {
      playArenaSfx(newUnit.hp <= 0 ? 'cardDeath' : 'cardAttacked');
    } else if (newUnit.hp > oldUnit.hp) {
      playArenaSfx('cardHeal');
    }
    if (!oldUnit.frozen && newUnit.frozen) {
      playArenaSfx('cardFrozen');
    }
  });
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
  if (isArenaWaitingForPlayers(currentState)) return false;
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

function buildArenaAuthUrl(path) {
  if (!authToken) return path;
  const separator = path.includes('?') ? '&' : '?';
  return path + separator + '_auth=' + encodeURIComponent(authToken);
}

function arenaFetchWithTimeout(url, options = {}, timeoutMs = 6500) {
  if (!window.AbortController) return fetch(url, options);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, {...options, signal: controller.signal}).finally(() => clearTimeout(timer));
}

function makeClientActionId(prefix) {
  const randomPart = window.crypto?.randomUUID ? window.crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${matchId || 'match'}-${randomPart}`;
}

async function parseActionError(response, fallbackMessage) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (payload.state) {
    handleStateChanged({state: payload.state});
  }
  return new Error(payload.error || payload.message || fallbackMessage);
}

function isArenaAndroidShell() {
  return new URLSearchParams(location.search).get('ea_platform') === 'android_app' || !!window.ExtraArenaApp;
}

async function shareArenaResult(text, url, title) {
  const cleanText = String(text || '').trim();
  const cleanUrl = String(url || 'https://t.me/extraarena_bot').trim();
  const cleanTitle = String(title || 'ExtraArena').trim();
  if (!cleanText && !cleanUrl) return false;

  try {
    if (isArenaAndroidShell() && window.ExtraArenaApp && typeof window.ExtraArenaApp.shareText === 'function') {
      window.ExtraArenaApp.shareText(cleanText, cleanUrl, cleanTitle);
      return true;
    }
  } catch (_) {}

  const shareUrl = `https://t.me/share/url?url=${encodeURIComponent(cleanUrl)}&text=${encodeURIComponent(cleanText)}`;
  const tg = window.Telegram?.WebApp;
  if (tg?.openTelegramLink) {
    tg.openTelegramLink(shareUrl);
    return true;
  }

  try {
    if (navigator.share) {
      await navigator.share({title: cleanTitle, text: cleanText, url: cleanUrl});
      return true;
    }
  } catch (_) {}

  window.open(shareUrl, '_blank', 'noopener,noreferrer');
  return true;
}

window.shareExtraArena = shareArenaResult;

function arenaMaintenanceBlocks(data) {
  return !!(data && data.maintenance_mode && data.maintenance_mode.enabled)
    && (!data.is_admin || isArenaAndroidShell());
}

function ensureArenaHealthRoot() {
  let root = document.getElementById('arena-health-root');
  if (root) return root;
  root = document.createElement('div');
  root.id = 'arena-health-root';
  document.body.appendChild(root);
  return root;
}

function hideArenaBadConnection() {
  const banner = document.getElementById('arena-bad-connection-banner');
  if (banner) banner.remove();
}

function showArenaBadConnection(latency) {
  if (arenaBadPingDismissed || arenaHealthStopped || document.getElementById('arena-connection-modal')) return;
  const root = ensureArenaHealthRoot();
  let banner = document.getElementById('arena-bad-connection-banner');
  if (!banner) {
    banner = document.createElement('button');
    banner.id = 'arena-bad-connection-banner';
    banner.type = 'button';
    banner.style.cssText = [
      'position:fixed',
      'top:calc(10px + var(--ea-safe-top,0px))',
      'left:50%',
      'transform:translateX(-50%)',
      'z-index:12000',
      'min-height:32px',
      'padding:6px 11px',
      'border-radius:999px',
      'border:1px solid rgba(251,191,36,.42)',
      'background:linear-gradient(135deg,rgba(54,38,12,.96),rgba(33,23,9,.94))',
      'box-shadow:0 8px 26px rgba(0,0,0,.38)',
      'display:flex',
      'align-items:center',
      'gap:8px',
      'color:#fde68a',
      'font:900 11px "Exo 2",sans-serif',
      'cursor:pointer'
    ].join(';');
    banner.addEventListener('click', () => {
      arenaBadPingDismissed = true;
      hideArenaBadConnection();
    });
    root.appendChild(banner);
  }
  banner.innerHTML = '<span style="display:inline-flex;align-items:flex-end;gap:2px;height:14px"><i style="width:4px;height:6px;border-radius:4px;background:#fbbf24"></i><i style="width:4px;height:9px;border-radius:4px;background:#fbbf24"></i><i style="width:4px;height:13px;border-radius:4px;background:#f59e0b;opacity:.55"></i></span><span>Плохое соединение</span><span style="color:rgba(253,230,138,.66);font-size:10px">' + Math.round(latency) + ' мс</span>';
}

function clearArenaConnectionIssue() {
  arenaConnectionIssueSince = null;
  if (arenaConnectionIssueTimer) {
    clearTimeout(arenaConnectionIssueTimer);
    arenaConnectionIssueTimer = null;
  }
}

function showArenaConnectionModal(message) {
  stopArenaHealthPing();
  hideArenaBadConnection();
  const root = ensureArenaHealthRoot();
  if (document.getElementById('arena-connection-modal')) return;
  const modal = document.createElement('div');
  modal.id = 'arena-connection-modal';
  modal.style.cssText = 'position:fixed;inset:0;z-index:13000;background:radial-gradient(circle at 50% 22%,rgba(124,92,191,.28),transparent 34%),rgba(5,3,14,.84);backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:center;padding:22px';
  modal.innerHTML = ''
    + '<div style="width:100%;max-width:360px;border-radius:24px;padding:1px;background:linear-gradient(135deg,#ffb4ab,rgba(124,92,191,.44),rgba(45,212,191,.34));box-shadow:0 24px 80px rgba(0,0,0,.66)">'
    + '<section style="border-radius:23px;padding:22px 18px 18px;background:linear-gradient(180deg,rgba(26,16,48,.98),rgba(13,8,27,.98));border:1px solid rgba(255,255,255,.06);text-align:center">'
    + '<div style="width:72px;height:72px;border-radius:22px;margin:0 auto 14px;background:linear-gradient(135deg,#f5921e,#7c5cbf);display:grid;place-items:center;box-shadow:0 0 34px rgba(245,146,30,.32)">'
    + '<svg width="38" height="38" viewBox="0 0 38 38" fill="none"><path d="M19 5.5a13.5 13.5 0 1 0 0 27 13.5 13.5 0 0 0 0-27Z" stroke="white" stroke-width="3" opacity=".92"/><path d="M19 12v8l5 3" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    + '</div>'
    + "<h2 style=\"margin:0;color:#f0ecff;font:1000 24px/1.05 'Exo 2',sans-serif\">Соединение разорвано</h2>"
    + '<p style="margin:10px auto 18px;color:#c4b8e8;font:13px/1.55 Inter,sans-serif;max-width:300px">' + message + '</p>'
    + "<button type=\"button\" id=\"arena-connection-restart\" style=\"width:100%;height:50px;border:0;border-radius:15px;background:linear-gradient(135deg,#f5921e,#d97510);color:#201005;font:1000 16px 'Exo 2',sans-serif;cursor:pointer;box-shadow:0 12px 30px rgba(245,146,30,.32)\">Перезапустить</button>"
    + '</section></div>';
  root.appendChild(modal);
  document.getElementById('arena-connection-restart')?.addEventListener('click', () => {
    window.location.replace('/');
  });
}

function markArenaConnectionFailure() {
  if (document.getElementById('arena-connection-modal')) return;
  const now = Date.now();
  if (!arenaConnectionIssueSince) {
    arenaConnectionIssueSince = now;
  }
  const remaining = Math.max(0, ARENA_CONNECTION_FAILURE_DELAY_MS - (now - arenaConnectionIssueSince));
  if (arenaConnectionIssueTimer) clearTimeout(arenaConnectionIssueTimer);
  arenaConnectionIssueTimer = setTimeout(() => {
    showArenaConnectionModal('Похоже, связь с ареной пропала или сервер временно недоступен. Перезапусти клиент, чтобы восстановить подключение.');
  }, remaining);
}

function stopArenaHealthPing() {
  arenaHealthStopped = true;
  if (arenaHealthInterval) {
    clearInterval(arenaHealthInterval);
    arenaHealthInterval = null;
  }
  clearArenaConnectionIssue();
}

function startArenaHealthMonitor() {
  if (arenaHealthInterval || arenaHealthStopped || !authToken) return;

  const ping = async () => {
    if (document.hidden || arenaHealthStopped) return;
    const startedAt = performance.now();
    try {
      const runtimeUrl = buildArenaAuthUrl('/api/runtime/status');
      const pingUrl = runtimeUrl + (runtimeUrl.includes('?') ? '&' : '?') + 'ea_ping_ts=' + Date.now();
      const response = await arenaFetchWithTimeout(pingUrl, {cache:'no-store'});
      const latency = Math.round(performance.now() - startedAt);
      let data = null;
      try { data = await response.json(); } catch (_) {}

      if (response.ok) {
        clearArenaConnectionIssue();
        if (arenaMaintenanceBlocks(data)) {
          showArenaConnectionModal('На сервере включили технические работы. Перезапусти клиент, чтобы открыть экран обслуживания.');
          return;
        }
        if (latency > ARENA_BAD_CONNECTION_THRESHOLD_MS) showArenaBadConnection(latency);
        else hideArenaBadConnection();
        return;
      }

      if (response.status === 503 && data?.error === 'maintenance_mode') {
        showArenaConnectionModal('Сервер перешел в режим технических работ. Перезапусти клиент, чтобы открыть экран обслуживания.');
        return;
      }

      markArenaConnectionFailure();
    } catch (_) {
      markArenaConnectionFailure();
    }
  };

  ping();
  arenaHealthInterval = setInterval(ping, ARENA_HEALTH_PING_INTERVAL_MS);
}

// ============================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('[ARENA] Страница арены загружена');

  const tg = window.Telegram?.WebApp;
  if (tg) {
    try { tg.ready(); } catch (e) {}
    try { tg.expand(); } catch (e) {}
    try { tg.setHeaderColor?.('#0f0a1a'); } catch (e) {}
    try { tg.setBackgroundColor?.('#0f0a1a'); } catch (e) {}
    try { tg.setBottomBarColor?.('#0f0a1a'); } catch (e) {}
    window.ExtraArenaSafeArea?.sync?.();
  }
  
  // Извлекаем параметры из URL
  const urlParams = new URLSearchParams(window.location.search);
  matchId = urlParams.get('id');
  const _auth = urlParams.get('_auth');

  if (_auth) {
    sessionStorage.setItem('arena_auth', _auth);
    history.replaceState(null, '', '/arena?id=' + matchId);
  }
  authToken = sessionStorage.getItem('arena_auth');

  console.log('[ARENA] Match ID:', matchId);

  if (!matchId) {
    console.error('[ARENA] Отсутствует match_id в URL');
    alert('Ошибка: параметры боя не найдены');
    return;
  }

  if (!authToken) {
    console.error('[ARENA] Отсутствует _auth токен');
    alert('Ошибка: токен аутентификации не найден');
    return;
  }
  
  // Сначала грузим состояние: это дает серверу шанс лениво создать BattleEngine
  // до входа Socket.IO в комнату матча.
  loadBattleState().finally(() => initSocketIO());
  startArenaHealthMonitor();
  
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
    clearArenaConnectionIssue();
    emitJoinMatch();
    
    // ВАЖНО: НЕ отправляем client_ready здесь
    // Сигнал будет отправлен после успешной загрузки состояния боя в loadBattleState()
  });
  
  socket.on('disconnect', (reason) => {
    console.warn('[SOCKET.IO] Отключено:', reason);
    markArenaConnectionFailure();
  });

  socket.on('connect_error', (error) => {
    console.warn('[SOCKET.IO] Ошибка подключения:', error);
    markArenaConnectionFailure();
  });

  socket.io.on('reconnect_failed', () => {
    console.warn('[SOCKET.IO] Не удалось переподключиться');
    markArenaConnectionFailure();
  });
  
  socket.on('error', (error) => {
    console.error('[SOCKET.IO] Ошибка:', error);
    const message = error && (error.message || error.error || error);
    if (!socketJoined && message === 'match_not_found') {
      scheduleJoinRetry();
    }
  });
  
  socket.on('joined_match', (data) => {
    console.log('[SOCKET.IO] Вступили в матч:', data);
    socketJoined = true;
    if (socketJoinRetryTimer) {
      clearTimeout(socketJoinRetryTimer);
      socketJoinRetryTimer = null;
    }
    
    trySendClientReady();
  });
  
  socket.on('client_ready_ack', (data) => {
    console.log('[SOCKET.IO] Подтверждение готовности клиента получено:', data);
  });

  socket.on('match_waiting', (data) => {
    console.log('[SOCKET.IO] match_waiting получено:', data);
    handleStateChanged(data);
  });

  socket.on('match_ready', (data) => {
    console.log('[SOCKET.IO] match_ready получено:', data);
    handleStateChanged(data);
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
    
    // КРИТИЧНО: Принудительно сбрасываем таймер на длительность текущего режима
    const timerText = document.getElementById('turn-timer-text');
    if (timerText) {
      const state = data.state || data.state_p1 || currentState || {};
      timerText.textContent = String(Math.ceil(state.turn_duration || getClassicModeParams(state).turn_duration_seconds || 25));
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
  
   // Очищаем кеш результата при старте нового матча
  window.__battleResultEconomy = null;
  window.__resultModalShown = false;
}

function emitJoinMatch() {
  if (!socket || !socket.connected || socketJoined) return;
  socket.emit('join_match', {
    match_id: matchId,
    _auth: authToken
  });
}

function scheduleJoinRetry() {
  if (socketJoinRetryTimer) return;
  socketJoinRetryTimer = setTimeout(async () => {
    socketJoinRetryTimer = null;
    try {
      await loadBattleState();
    } finally {
      emitJoinMatch();
    }
  }, 1000);
}

// ── Helpers для безопасного мержа экономики результата ──

function _isEconomyNonEmpty(e) {
  if (!e) return false;
  return (
    (e.trophyDelta != null && e.trophyDelta !== 0) ||
    e.trophyTotal != null ||
    (e.coinsDelta != null && e.coinsDelta !== 0) ||
    e.coinsTotal != null ||
    (e.starsDelta != null && e.starsDelta !== 0) ||
    e.starsTotal != null
  );
}

function _mergeDelta(nextValue, cachedValue) {
  if (nextValue != null && nextValue !== 0) return nextValue;
  if (cachedValue != null && cachedValue !== 0) return cachedValue;
  return nextValue ?? cachedValue ?? 0;
}

function _mergeBattleResultEconomy(next) {
  const cached = window.__battleResultEconomy || {};
  const nextNonEmpty = _isEconomyNonEmpty(next);
  const cachedNonEmpty = _isEconomyNonEmpty(cached);

  if (!nextNonEmpty && cachedNonEmpty) {
    console.log('[ARENA] 🔒 Кеш экономики непустой, пропускаем пустой payload');
    return cached;
  }

  // ⚠️ Не перезаписываем непустые поля нулями/пустыми
  const merged = {
    trophyDelta: _mergeDelta(next.trophyDelta, cached.trophyDelta),
    trophyTotal: next.trophyTotal != null ? next.trophyTotal : cached.trophyTotal,
    coinsDelta: _mergeDelta(next.coinsDelta, cached.coinsDelta),
    coinsTotal: next.coinsTotal != null ? next.coinsTotal : cached.coinsTotal,
    starsDelta: _mergeDelta(next.starsDelta, cached.starsDelta),
    starsTotal: next.starsTotal != null ? next.starsTotal : cached.starsTotal,
    leagueUp: next.leagueUp ?? cached.leagueUp ?? null,
  };

  window.__battleResultEconomy = merged;
  console.log('[ARENA] 💾 Экономика сохранена в кеш:', merged);
  return merged;
}

function _readEconomyFromCache() {
  const cached = window.__battleResultEconomy || {};
  return {
    trophyDelta: cached.trophyDelta ?? 0,
    trophyTotal: cached.trophyTotal ?? null,
    coinsDelta: cached.coinsDelta ?? 0,
    coinsTotal: cached.coinsTotal ?? null,
    starsDelta: cached.starsDelta ?? 0,
    starsTotal: cached.starsTotal ?? null,
    leagueUp: cached.leagueUp ?? null,
  };
}

function handleStateChanged(eventData) {
  // Извлекаем state из события
  const newState = eventData.state || eventData.state_p1;
  
  if (!newState) {
    console.warn('[ARENA] state_changed получено без state');
    return;
  }

  if (newState.viewer_id != null && userId != null && String(newState.viewer_id) !== String(userId)) {
    console.warn('[ARENA] Игнорируем state для другого viewer_id:', newState.viewer_id, 'current:', userId);
    return;
  }
  if (!userId && newState.viewer_id != null) {
    userId = newState.viewer_id;
  }
  
  // ДОБАВЛЕНО: Подробное логирование для отслеживания изменений здоровья
  console.log('[ARENA] Обновление состояния:', newState);
  console.log('[ARENA] 🔍 здоровья tracking:');
  console.log('  - Player 1 здоровья:', newState.player1_hp || newState.player?.hp || '???');
  console.log('  - Player 2 здоровья:', newState.player2_hp || newState.opponent?.hp || '???');
  console.log('  - Current player:', newState.current_player_id);
  console.log('  - Is my turn:', newState.is_my_turn);
  console.log('  - Turn:', newState.turn);
  console.log('  - Player mana:', newState.player?.mana, '/', newState.player?.max_mana);
  console.log('  - Opponent mana:', newState.opponent?.mana, '/', newState.opponent?.max_mana);
  console.log(`[ARENA] ⏰ TIMER: turn_time_remaining=${newState.turn_time_remaining}, turn_duration=${newState.turn_duration}`);

  if (!prebattleComplete) {
    currentState = newState;
    pendingInitialBattleState = newState;
    if (!prebattleRendered) {
      renderPrebattleScreen(newState);
    }
    startPrebattleSequence();
    return;
  }
  
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
    
    const response = await fetch(`/api/battle/state?_auth=${encodeURIComponent(authToken)}&match_id=${matchId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const state = await response.json();
    console.log('[ARENA] Состояние боя загружено:', state);
    
    currentState = state;
    pendingInitialBattleState = state;
    if (!prebattleRendered) {
      renderPrebattleScreen(state);
    }
    if (!prebattleComplete) {
      startPrebattleSequence();
    } else {
      pendingInitialBattleState = null;
      renderBattleState(state);
    }
    
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
  
  // КРИТИЧНО: Устанавливаем userId из server-authoritative viewer_id
  if (!userId && state.viewer_id != null) {
    userId = state.viewer_id;
    console.log('[ARENA] Identity derived from server: userId =', userId);
  }

  const waitingForPlayers = isArenaWaitingForPlayers(state);
  updateArenaWaitingOverlay(state);

  // КРИТИЧНО: Кешируем legal_actions для использования в рендеринге
  cachedLegalActions = waitingForPlayers ? [] : (state.legal_actions || []);
  console.log('[ARENA] 📋 Legal actions:', cachedLegalActions.length, 'доступных действий');
  applyModeUi(state);

  const userIdNum = Number(userId);

  // Используем server-authoritative is_my_turn
  const isMyTurn = Boolean(state.is_my_turn) && !waitingForPlayers;
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
      hp: (state.player1_hp !== undefined && state.player1_hp !== null) ? state.player1_hp : 30,
      max_hp: (state.player1_max_hp !== undefined && state.player1_max_hp !== null) ? state.player1_max_hp : ((state.player1_hp !== undefined && state.player1_hp !== null) ? state.player1_hp : 30),
      mana: state.player1_mana || 0,
      max_mana: state.player?.max_mana || 10,
      hand: state.player1_hand || [],
      board: state.player1_board || [],
      name: state.player?.name || 'Игрок',
      title: state.player?.title || state.player1_title || '',
      rarity: state.player?.rarity || state.player1_rarity || '',
      clan: state.player?.clan || state.player1_clan || '',
      description: state.player?.description || state.player1_description || '',
      mechanics: state.player?.mechanics || state.player1_mechanics || [],
      avatar_url: state.player?.avatar_url,
      extra_pass: state.player?.extra_pass,
      hero: state.player?.hero || null
    };
    
    const p2 = {
      user_id: state.player_ids ? state.player_ids[1] : null,
      hp: (state.player2_hp !== undefined && state.player2_hp !== null) ? state.player2_hp : 30,
      max_hp: (state.player2_max_hp !== undefined && state.player2_max_hp !== null) ? state.player2_max_hp : ((state.player2_hp !== undefined && state.player2_hp !== null) ? state.player2_hp : 30),
      mana: state.player2_mana || 0,
      max_mana: state.opponent?.max_mana || 10,
      hand: state.player2_hand || [],
      board: state.player2_board || [],
      name: state.opponent?.name || 'Оппонент',
      title: state.opponent?.title || state.player2_title || '',
      rarity: state.opponent?.rarity || state.player2_rarity || '',
      clan: state.opponent?.clan || state.player2_clan || '',
      description: state.opponent?.description || state.player2_description || '',
      mechanics: state.opponent?.mechanics || state.player2_mechanics || [],
      avatar_url: state.opponent?.avatar_url,
      extra_pass: state.opponent?.extra_pass,
      hero: state.opponent?.hero || null
    };
    
    if (String(p1.user_id) === String(userIdNum)) {
      myState = p1;
      opponentStateData = p2;
    } else {
      myState = p2;
      opponentStateData = p1;
    }
  }

  // Сохраняем извлечённый оппонент в модульную переменную для openOpponentInfo
  window.__arenaOpponentState = opponentStateData;
  
  // КРИТИЧНО: Логируем здоровья при каждом обновлении для отслеживания изменений
  console.log('[ARENA] 💚 здоровья TRACKING: Мой здоровья =', myState.hp, '| Оппонент здоровья =', opponentStateData.hp);
  console.log('[ARENA] Мой state:', myState);
  console.log('[ARENA] Оппонент state:', opponentStateData);

  processArenaStateSfx(myState, opponentStateData);
  
  // Рендерим панели
  renderPlayerPanel(myState);
  renderOpponentPanel(opponentStateData);
  renderSuddenDeathBadges(state);
  
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
    // ⛔ Не перезаписываем результат, если game_over уже обработан с реальной экономикой
    if (window.__resultModalShown) {
      console.log('[ARENA] 🚫 game_over уже обработан, игнорируем late state_changed');
      return;
    }
    
    console.log('[ARENA] 🏁 Игра завершена через state_changed, показываем финальный экран');
    
    // Извлекаем данные о победителе
    const winnerId = state.winner_id ?? state.winner ?? null;
    const outcome = winnerId == null ? 'draw' : (String(winnerId) === String(userId) ? 'victory' : 'defeat');
    
    // Пытаемся достать экономику из кеша (game_over мог прийти раньше)
    const cached = _readEconomyFromCache();
    const trophyDelta = (cached.trophyDelta != null && cached.trophyDelta !== 0)
      ? cached.trophyDelta
      : parseInt(state.trophy_change || state.trophy_delta, 10) || 0;
    const trophyTotal = (cached.trophyTotal != null)
      ? cached.trophyTotal
      : parseInt(state.trophy_total || state.new_trophies, 10) || null;
    
    const coinsDelta = (cached.coinsDelta != null && cached.coinsDelta !== 0)
      ? cached.coinsDelta
      : parseInt(state.coins_change || state.coins_delta, 10) || 0;
    const coinsTotal = (cached.coinsTotal != null)
      ? cached.coinsTotal
      : parseInt(state.coins_total || state.new_coins, 10) || null;

    const starsDelta = (cached.starsDelta != null && cached.starsDelta !== 0)
      ? cached.starsDelta
      : parseInt(state.stars_delta, 10) || 0;
    const starsTotal = (cached.starsTotal != null)
      ? cached.starsTotal
      : parseInt(state.stars_total, 10) || null;

    const leagueUp = cached.leagueUp || state.league_up || null;
    if (leagueUp) {
      sessionStorage.setItem('arena_league_up', JSON.stringify(leagueUp));
    }
    
    console.log('[ARENA] 🎯 Результат игры (state_changed):', { 
      outcome, winnerId, trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal, leagueUp
    });
    
    // Показываем экран результата с задержкой для драматического эффекта
    setTimeout(() => {
      showBattleResult(outcome, trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal);
    }, 1200);
  }
}

// ============================================
// ОБНОВЛЕНИЕ ЛОГА БОЯ
// ============================================

function updateBattleLog(history) {
  const logRows = document.getElementById('battle-log-rows');
  if (!logRows) return;

  logRows.innerHTML = '';
  setText('battle-log-meta', 'Ход ' + (currentState?.turn || currentTurnCount || 0));
  
  if (!history || history.length === 0) {
    logRows.appendChild(createLogGroup('События', 'system', [{ type: 'system', text: 'Бой начался' }]));
    return;
  }

  const groups = [];
  let currentGroup = null;

  history.forEach(entry => {
    const parsed = normalizeLogEntry(entry);
    if (parsed.type === 'system' && parsed.text.includes('———')) {
      currentGroup = {
        title: parsed.text.replace(/—/g, '').trim() || 'Ход',
        side: 'system',
        entries: []
      };
      groups.push(currentGroup);
      return;
    }

    if (!currentGroup) {
      currentGroup = { title: 'События', side: parsed.type || 'system', entries: [] };
      groups.push(currentGroup);
    }

    if (currentGroup.side === 'system' && parsed.type !== 'system') {
      currentGroup.side = parsed.type;
    }
    currentGroup.entries.push(parsed);
  });

  const visibleGroups = groups.filter(group => group.entries.length > 0).reverse();
  if (visibleGroups.length === 0) {
    logRows.appendChild(createLogGroup('События', 'system', [{ type: 'system', text: 'Бой начался' }]));
  } else {
    visibleGroups.forEach(group => {
      logRows.appendChild(createLogGroup(group.title, group.side, group.entries));
    });
  }

  requestAnimationFrame(() => {
    logRows.scrollTop = 0;
  });
}

function normalizeLogEntry(entry) {
  let type = 'system';
  let text = '';

  if (Array.isArray(entry)) {
    [type, text] = entry;
  } else if (typeof entry === 'object' && entry !== null) {
    type = entry.type || 'system';
    text = entry.text || '';
  } else {
    text = String(entry || '');
    if (text.includes('Вы ') || text.includes('Ваш')) type = 'player';
    else if (text.includes('Оппонент') || text.includes('Противник')) type = 'opponent';
  }

  return {
    type: ['player', 'opponent', 'system'].includes(type) ? type : 'system',
    text: String(text || '')
  };
}

function createLogGroup(title, side, entries) {
  const group = document.createElement('section');
  group.className = 'turn-group ' + (side || 'system');

  const head = document.createElement('div');
  head.className = 'turn-head';
  const label = document.createElement('span');
  label.className = 'turn-label';
  label.textContent = title || 'События';
  const tag = document.createElement('span');
  tag.className = 'side-tag';
  tag.textContent = side === 'player' ? 'Вы' : (side === 'opponent' ? 'Соперник' : 'Система');
  head.appendChild(label);
  head.appendChild(tag);
  group.appendChild(head);

  entries.forEach(entry => {
    const row = document.createElement('div');
    row.className = 'log-entry';
    if (isHeroDamageLog(entry.text)) row.classList.add('damage-hero');

    const text = document.createElement('span');
    text.innerHTML = formatLogEntryText(entry.text);
    row.appendChild(text);

    if (row.classList.contains('damage-hero')) {
      const kind = document.createElement('span');
      kind.className = 'kind';
      kind.textContent = 'герой';
      row.appendChild(kind);
    }

    group.appendChild(row);
  });

  return group;
}

function isHeroDamageLog(text) {
  const value = String(text || '').toLowerCase();
  return value.includes('hp') || (value.includes('геро') && (value.includes('урон') || value.includes('нанос')));
}

function formatLogEntryText(text) {
  const safeText = escapeHtml(text || '');
  const damageMatch = safeText.match(/([+-]?\d+\s*(?:HP|hp|хп))/);
  if (damageMatch) {
    return safeText.replace(damageMatch[1], '<strong>' + damageMatch[1] + '</strong>');
  }
  const colonIndex = safeText.indexOf(':');
  if (colonIndex > 0 && colonIndex <= 24) {
    return '<strong>' + safeText.slice(0, colonIndex) + '</strong>' + safeText.slice(colonIndex);
  }
  const firstSpace = safeText.indexOf(' ');
  if (firstSpace > 0 && firstSpace <= 12) {
    return '<strong>' + safeText.slice(0, firstSpace) + '</strong>' + safeText.slice(firstSpace);
  }
  return safeText;
}

function applyModeUi(state) {
  const body = document.body;
  if (!body) return;
  body.classList.remove(
    'mode-powermax',
    'mode-spellstorm',
    'mode-blitzkrieg',
    'mode-sudden-death',
    'mode-extra-arena'
  );
  const modeId = getModeId(state);
  const meta = getModeUiMeta(state);
  if (modeId.startsWith('extra_arena')) body.classList.add('mode-extra-arena');
  if (meta) body.classList.add('mode-' + meta.key);
}

function renderSuddenDeathBadges(state) {
  const sd = state && state.sudden_death;
  updateSuddenDeathBadge('player-hp-block', sd && sd.enabled ? sd.player_next_damage : null);
  updateSuddenDeathBadge('opponent-hp-block', sd && sd.enabled ? sd.opponent_next_damage : null);
}

function updateSuddenDeathBadge(hpBlockId, nextDamage) {
  const block = document.getElementById(hpBlockId);
  if (!block) return;

  let badge = block.querySelector('.mode-hp-burn-badge');
  if (!nextDamage) {
    if (badge) badge.remove();
    block.classList.remove('sudden-death-hp');
    return;
  }

  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'mode-hp-burn-badge';
    block.appendChild(badge);
  }
  badge.textContent = '-' + nextDamage;
  badge.title = 'Урон в начале следующего собственного хода';
  block.classList.add('sudden-death-hp');
}

// ============================================
// РЕНДЕРИНГ ПАНЕЛЕЙ
// ============================================

function renderPlayerPanel(playerState) {
  const hpText = document.getElementById('player-hp-text');
  const hpMaxText = document.getElementById('player-hp-max-text');
  if (hpText) {
    const hp = playerState.hero?.hp ?? playerState.hp ?? 30;
    const hpValue = Math.max(0, hp);
    const maxHp = playerState.hero?.max_hp ?? playerState.max_hp ?? 30;
    hpText.textContent = hpValue;

    if (hpMaxText) {
      hpMaxText.textContent = '/' + maxHp;
    }

    // здоровья Fill Bar
    const hpFill = document.getElementById('player-hp-fill');
    if (hpFill) {
      const hpPct = Math.max(0, Math.min(100, (hpValue / maxHp) * 100));
      hpFill.style.width = hpPct + '%';
    }

    // здоровья Warning pulse (<= 7)
    const hpBlock = document.getElementById('player-hp-block');
    if (hpBlock) {
      hpBlock.classList.toggle('hp-warning', hpValue <= 7);
    }
    
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
  
  const nameText = document.getElementById('player-name-text');
  if (nameText) {
    nameText.textContent = playerState.name || 'Игрок';
  }
  
  const avatarLetter = document.getElementById('player-avatar-letter');
  if (avatarLetter) {
    const firstName = playerState.name || 'И';
    avatarLetter.textContent = firstName[0].toUpperCase();
  }

  // Title text
  const playerTitleEl = document.getElementById('player-title-text');
  if (playerTitleEl) {
    const titleText = playerState.title || currentState?.player_title || '';
    playerTitleEl.textContent = titleText;
    playerTitleEl.className = 'player-title-text';
    if (titleText) {
      playerTitleEl.classList.add('has-title');
      const rarity = (playerState.rarity || currentState?.player_rarity || '').toLowerCase();
      if (rarity === 'mythic' || rarity === 'mythical') playerTitleEl.classList.add('title-mythical');
      else if (rarity === 'limited') playerTitleEl.classList.add('title-limited');
    }
  }

  // Avatar rarity class + image
  const playerAvatar = document.getElementById('player-avatar');
  if (playerAvatar) {
    const rarity = (playerState.rarity || currentState?.player_rarity || '').toLowerCase();
    playerAvatar.className = 'player-avatar avatar-class-' + (rarity || 'starter');

    const avatarImg = document.getElementById('player-avatar-img');
    if (avatarImg) {
      const avatarUrl = playerState.avatar_url || currentState?.player?.avatar_url || '';
      if (avatarUrl) {
        avatarImg.src = avatarUrl;
        avatarImg.alt = playerState.name || '';
        playerAvatar.classList.add('has-avatar-img');
      } else {
        playerAvatar.classList.remove('has-avatar-img');
      }
    }
  }

  // ExtraPass
  const infoBlock = document.querySelector('.player-info-block');
  if (infoBlock) {
    const hasExtraPass = currentState?.extra_pass === 'active' || playerState?.extra_pass === 'active';
    if (hasExtraPass) {
      infoBlock.classList.add('extra-pass-active');
      console.log('[ARENA] 💎 ExtraPass визуал активирован для игрока');
    } else {
      infoBlock.classList.remove('extra-pass-active');
    }
  }
  
  // Mana
  const manaText = document.getElementById('player-mana-text');
  const manaMaxText = document.getElementById('player-mana-max-text');
  const manaFill = document.getElementById('player-mana-fill');
  
  if (manaText) {
    const manaValue = playerState.mana || 0;
    manaText.textContent = manaValue % 1 === 0 ? manaValue : manaValue.toFixed(1);
  }
  
  if (manaMaxText) {
    manaMaxText.textContent = `/${playerState.max_mana || 10}`;
  }
  
  if (manaFill) {
    const manaPercent = ((playerState.mana || 0) / (playerState.max_mana || 10)) * 100;
    manaFill.style.width = `${Math.min(100, Math.max(0, manaPercent))}%`;
  }

  // Restore targeting highlight
  const playerPanel = document.querySelector('.player-panel-root');
  if (playerPanel && interactionMode.type === 'TARGETING') {
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    const playerHeroId = playerState.hero?.instance_id;
    if (playActions.some(a => String(a.target_id) === String(playerHeroId) || (a.target_is_hero && !a.target_id))) {
      playerPanel.classList.add('targetable-friendly');
    }
  }

  // End-turn pulse check (no legal actions)
  checkEndTurnPulse();
}

function renderOpponentPanel(opponentState) {
  const hpText = document.getElementById('opponent-hp-text');
  const hpMaxText = document.getElementById('opponent-hp-max-text');
  if (hpText) {
    const hp = opponentState.hero?.hp ?? opponentState.hp ?? 30;
    const hpValue = Math.max(0, hp);
    const maxHp = opponentState.hero?.max_hp ?? opponentState.max_hp ?? 30;
    hpText.textContent = hpValue;

    if (hpMaxText) {
      hpMaxText.textContent = '/' + maxHp;
    }

    // здоровья Fill Bar
    const hpFill = document.getElementById('opponent-hp-fill');
    if (hpFill) {
      const hpPct = Math.max(0, Math.min(100, (hpValue / maxHp) * 100));
      hpFill.style.width = hpPct + '%';
    }

    // здоровья Warning pulse (<= 7)
    const hpBlock = document.getElementById('opponent-hp-block');
    if (hpBlock) {
      hpBlock.classList.toggle('hp-critical', hpValue <= 7);
    }
    
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
  
  const nameText = document.getElementById('opponent-name-text');
  if (nameText) {
    nameText.textContent = opponentState.name || 'Оппонент';
  }
  
  const avatarLetter = document.getElementById('opponent-avatar-letter');
  if (avatarLetter) {
    const firstName = opponentState.name || 'О';
    avatarLetter.textContent = firstName[0].toUpperCase();
  }
  
  const handCount = document.getElementById('opponent-hand-count');
  if (handCount) {
    handCount.textContent = opponentState.hand ? opponentState.hand.length : 0;
  }

  // Title text
  const opponentTitleEl = document.getElementById('opponent-title-text');
  if (opponentTitleEl) {
    const titleText = opponentState.title || currentState?.opponent_title || '';
    opponentTitleEl.textContent = titleText;
    opponentTitleEl.className = 'opponent-title-text';
    if (titleText) {
      opponentTitleEl.classList.add('has-title');
      const rarity = (opponentState.rarity || currentState?.opponent_rarity || '').toLowerCase();
      if (rarity === 'mythic' || rarity === 'mythical') opponentTitleEl.classList.add('title-mythical');
      else if (rarity === 'limited') opponentTitleEl.classList.add('title-limited');
    }
  }

  // Clan badge
  const clanBadge = document.getElementById('opponent-clan-badge');
  if (clanBadge) {
    const clan = opponentState.clan || currentState?.opponent_clan || '';
    if (clan) {
      clanBadge.textContent = clan;
      clanBadge.style.display = 'inline-block';
    } else {
      clanBadge.style.display = 'none';
    }
  }

  // Avatar rarity class + image
  const opponentAvatar = document.getElementById('opponent-avatar');
  if (opponentAvatar) {
    const rarity = (opponentState.rarity || currentState?.opponent_rarity || '').toLowerCase();
    opponentAvatar.className = 'opponent-avatar avatar-class-' + (rarity || 'starter');

    const avatarImg = document.getElementById('opponent-avatar-img');
    if (avatarImg) {
      const avatarUrl = opponentState.avatar_url || currentState?.opponent?.avatar_url || '';
      if (avatarUrl) {
        avatarImg.src = avatarUrl;
        avatarImg.alt = opponentState.name || '';
        opponentAvatar.classList.add('has-avatar-img');
      } else {
        opponentAvatar.classList.remove('has-avatar-img');
      }
    }
  }
  
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
  
  // Restore targeting highlight
  const opponentPanel = document.querySelector('.opponent-panel-root');
  if (opponentPanel && interactionMode.type === 'TARGETING') {
    const playActions = getPlayCardTargets(interactionMode.data?.handIndex ?? selectedCard?.index ?? 0);
    const opponentHeroId = opponentState.hero?.instance_id;
    if (playActions.some(a => String(a.target_id) === String(opponentHeroId) || (a.target_is_hero && !a.target_id))) {
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
  const cardMana = getRawManaCost(card);
  const effectiveMana = getEffectiveManaCost(card);
  const isFreeByMode = effectiveMana === 0 && cardMana > 0;
  const playerMana = currentState?.player?.mana || 0;
  
  if (effectiveMana > playerMana) {
    cardDiv.classList.add('insufficient-mana');
  }
  if (isFreeByMode) {
    cardDiv.classList.add('mode-free-spell');
  }
  if (isPowerMaxMode()) {
    cardDiv.classList.add('mode-powermax-card');
  }
  if (isSummonReadyMode() && cardType === 'warrior') {
    cardDiv.classList.add('mode-ready-on-play-card');
  }

  const manaDiv = document.createElement('div');
  manaDiv.className = 'mana-circle';
  if (isFreeByMode) {
    manaDiv.classList.add('mana-free');
    manaDiv.textContent = '0';
    manaDiv.title = 'Бесплатно в SpellStorm';
  } else {
    manaDiv.textContent = cardMana;
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

  // Иконки механик — внутри artWrapper, чтобы не выходить за пределы арта
  if (Array.isArray(mechanics)) {
    if (mechanics.includes('deathrattle')) {
      cardDiv.classList.add('card-deathrattle');
      const drIcon = document.createElement('div');
      drIcon.className = 'mechanic-icon deathrattle-icon';
      drIcon.textContent = '💀';
      artWrapper.appendChild(drIcon);
    }
    const hasBC = mechanics.some(m => m.startsWith('battlecry_'));
    if (hasBC) {
      const bcIcon = document.createElement('div');
      bcIcon.className = 'mechanic-icon battlecry-icon';
      bcIcon.textContent = '🔔';
      artWrapper.appendChild(bcIcon);
    }
  }

  cardDiv.appendChild(artWrapper);
  cardDiv.appendChild(manaDiv);

  if (cardType === 'potion') {
    const potionInfoBtn = document.createElement('button');
    potionInfoBtn.className = 'card-info-btn potion-info-btn';
    potionInfoBtn.textContent = 'i';
    potionInfoBtn.title = 'Информация о карте';
    potionInfoBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openCardInfo(card);
    });
    cardDiv.appendChild(potionInfoBtn);
  }

  if (isFreeByMode) {
    const freeBadge = document.createElement('div');
    freeBadge.className = 'mode-card-badge mode-card-badge-free';
    freeBadge.textContent = 'FREE';
    cardDiv.appendChild(freeBadge);
  } else if (isPowerMaxMode() && card.level != null) {
    const levelBadge = document.createElement('div');
    levelBadge.className = 'mode-card-badge mode-card-badge-level';
    levelBadge.textContent = 'LVL ' + card.level;
    cardDiv.appendChild(levelBadge);
  } else if (isSummonReadyMode() && cardType === 'warrior') {
    const readyBadge = document.createElement('div');
    readyBadge.className = 'mode-card-badge mode-card-badge-ready';
    readyBadge.textContent = '⚡';
    readyBadge.title = 'Будет готов сразу';
    cardDiv.appendChild(readyBadge);
  }

  // Статы
  if (cardType !== 'potion') {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'hand-card-stats';

    const atkDiv = document.createElement('div');
    atkDiv.className = 'card-stat attack';
    atkDiv.textContent = card.attack || card.atk || 0;

    const infoBtn = document.createElement('button');
    infoBtn.className = 'card-info-btn';
    infoBtn.textContent = 'i';
    infoBtn.title = 'Информация о карте';
    infoBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openCardInfo(card);
    });

    const hpDiv = document.createElement('div');
    hpDiv.className = 'card-stat health';
    hpDiv.textContent = card.hp || card.hp_current || 0;

    statsDiv.appendChild(atkDiv);
    statsDiv.appendChild(infoBtn);
    statsDiv.appendChild(hpDiv);
    cardDiv.appendChild(statsDiv);
  }
  
  // Имя карты
  const nameLabel = document.createElement('div');
  nameLabel.className = 'card-name-label';
  nameLabel.textContent = card.name || 'Карта';
  cardDiv.appendChild(nameLabel);
  
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

  const createIcon = (typeClass, fileName, posClass) => {
    const container = document.createElement('div');
    container.className = `status-icon-container ${typeClass} ${posClass}`;
    const img = document.createElement('img');
    img.src = effectsPath + fileName;
    img.className = 'status-icon';
    img.alt = '';
    container.appendChild(img);
    return container;
  };

  cardDiv.appendChild(createIcon('status-icon-shield',  'shield.png',      'icon-top-left'));
  cardDiv.appendChild(createIcon('status-icon-taunt',   'provocation.png', 'icon-bottom-right'));

  const frozenIcon = createIcon('status-icon-frozen', 'freeze.png', 'icon-top-right');
  if (card && card.is_frozen) {
    const counter = document.createElement('span');
    counter.className = 'freeze-counter';
    counter.textContent = card.freeze_turns || "1";
    frozenIcon.appendChild(counter);
  }
  cardDiv.appendChild(frozenIcon);

  cardDiv.appendChild(createIcon('status-icon-asleep', 'asleep.png', 'icon-top-center'));
  cardDiv.appendChild(createIcon('status-icon-target', 'target.png', 'icon-center'));
  cardDiv.appendChild(createIcon('status-icon-heal',   'toHeal.png', 'icon-center'));
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
  if (isPowerMaxMode()) {
    cardDiv.classList.add('mode-powermax-card');
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

  // Иконки механик — внутри artWrapper, чтобы не выходить за пределы арта
  if (Array.isArray(mechanics)) {
    if (mechanics.includes('deathrattle')) {
      cardDiv.classList.add('card-deathrattle');
      const drIcon = document.createElement('div');
      drIcon.className = 'mechanic-icon deathrattle-icon';
      drIcon.textContent = '💀';
      artWrapper.appendChild(drIcon);
    }
    const hasBC = mechanics.some(m => m.startsWith('battlecry_'));
    if (hasBC) {
      const bcIcon = document.createElement('div');
      bcIcon.className = 'mechanic-icon battlecry-icon';
      bcIcon.textContent = '🔔';
      artWrapper.appendChild(bcIcon);
    }
  }

  cardDiv.appendChild(artWrapper);

  if (isPowerMaxMode() && card.level != null) {
    const levelBadge = document.createElement('div');
    levelBadge.className = 'mode-card-badge mode-card-badge-level board-mode-badge';
    levelBadge.textContent = 'LVL ' + card.level;
    cardDiv.appendChild(levelBadge);
  }

  // Статы: не добавляем для зелий
  if (card.card_type !== 'potion') {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'unit-card-stats';
    
    const atkDiv = document.createElement('div');
    atkDiv.className = 'unit-stat attack';
    atkDiv.textContent = card.attack || card.atk || 0;

    const infoBtn = document.createElement('button');
    infoBtn.className = 'card-info-btn';
    infoBtn.textContent = 'i';
    infoBtn.title = 'Информация о карте';
    infoBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openCardInfo(card);
    });
    
    const hpDiv = document.createElement('div');
    hpDiv.className = 'unit-stat health';
    const hpValue = card.hp || card.hp_current || 0;
    hpDiv.textContent = hpValue;
    
    const instanceId = String(card.instance_id);
    const oldHp = previousUnitHPs[instanceId];
    if (oldHp !== undefined && hpValue < oldHp) {
      triggerDamageEffects(cardDiv, oldHp - hpValue);
    }
    previousUnitHPs[instanceId] = hpValue;
    
    statsDiv.appendChild(atkDiv);
    statsDiv.appendChild(infoBtn);
    statsDiv.appendChild(hpDiv);
    cardDiv.appendChild(statsDiv);
  }
  
  // ДОБАВЛЕНО: Имя карты на поле
  const nameLabel = document.createElement('div');
  nameLabel.className = 'card-name-label board-card-name';
  nameLabel.textContent = card.name || 'Юнит';
  cardDiv.appendChild(nameLabel);
  
  // Status overlay icons (attack target, freeze, etc.)
  const soAttack = document.createElement('div');
  soAttack.className = 'status-overlay-icon so-attack-target';
  const soAttackImg = document.createElement('img');
  soAttackImg.src = '../DesignAssets/Arena/CardEffects/target.png';
  soAttackImg.alt = 'target';
  soAttack.appendChild(soAttackImg);
  cardDiv.appendChild(soAttack);

  // Если это карта игрока, разрешаем атаку
  if (side === 'player' && card.can_attack) {
    cardDiv.style.cursor = 'pointer';
    cardDiv.addEventListener('click', (e) => {
      if (interactionMode.type === 'TARGETING') {
        return;
      }
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
  
  if (isArenaWaitingForPlayers(state)) {
    turnText.textContent = 'Ждем соперника';
    turnPlaque.classList.remove('player-turn', 'opponent-turn', 'turn-expired');
    const endTurnBtn = document.getElementById('end-turn-button');
    if (endTurnBtn) endTurnBtn.disabled = true;
    return;
  }

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
  if (activeBattleModal === 'timer') {
    renderTurnTimerModal({ ...state, turn_time_remaining: timeRemaining });
  }
  
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
    if (activeBattleModal === 'timer') {
      renderTurnTimerModal({ ...state, turn_time_remaining: timeRemaining });
    }
    
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
  if (!currentState?.is_my_turn || !canPlayCard(index) || !hasEnoughManaForCard(card)) {
    console.warn('[ARENA] ❌ Drag отменён: карта недоступна');
    arenaHaptic('warning', { key: 'card-invalid', minInterval: 180 });
    e.preventDefault();
    return;
  }
  selectedCard = { card, index };
  arenaHaptic('selection', { key: 'card-pick', minInterval: 90 });
  playArenaSfx('cardSelected', { volume: 0.62 });
  e.currentTarget.classList.add('dragging');
  
  // Получаем возможные цели для этой карты
  const playActions = getPlayCardTargets(index);
  const hasTargetingOptions = playActions.length > 0 && playActions.some(a => a.target_id !== null && a.target_id !== undefined);
  const hasNoTargetOption = playActions.some(a => a.target_id === null || a.target_id === undefined);
  const requiresTarget = hasTargetingOptions && !hasNoTargetOption;

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
  
  arenaHaptic('medium', { key: 'slot-drop', minInterval: 120 });
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
  
  const card = selectedCard.card;
  console.log('[ARENA] Карта применена на существо:', targetId);

  arenaHaptic('medium', { key: 'slot-drop-target', minInterval: 120 });
  if (card.card_type === 'potion') {
    playPotionCard(card, targetId, false);
  } else {
    playCard(card, null, targetId, false);
  }
  selectedCard = null;
}

function handlePotionHeroDrop(e) {
  e.preventDefault();
  
  if (!selectedCard) return;
  
  const panel = e.currentTarget;
  const isPlayerHero = panel.classList.contains('player-panel-root');
  const heroId = isPlayerHero
    ? currentState?.player?.hero?.instance_id
    : currentState?.opponent?.hero?.instance_id;
  const card = selectedCard.card;

  console.log('[ARENA] Карта применена на героя:', { isPlayerHero, heroId });

  arenaHaptic('medium', { key: 'slot-drop-hero', minInterval: 120 });
  if (card.card_type === 'potion') {
    playPotionCard(card, heroId, true);
  } else {
    playCard(card, null, heroId, true);
  }
  selectedCard = null;
}

// ============================================
// КЛИК ПО КАРТЕ (АЛЬТЕРНАТИВА DRAG&DROP)
// ============================================

function handleCardClick(card, index, cardEl) {
  console.log('[ARENA] 🎴 Клик по карте:', card);
  
  // LEGAL ACTIONS: Проверяем, можно ли разыграть эту карту
  if (!currentState?.is_my_turn || !canPlayCard(index) || !hasEnoughManaForCard(card)) {
    console.warn('[ARENA] ❌ Карта недоступна для розыгрыша (legal_actions)');
    arenaHaptic('warning', { key: 'card-invalid', minInterval: 180 });
    return;
  }
  
  // Сбрасываем предыдущий режим
  resetInteractionMode();
  
  cardEl.classList.add('selected');
  selectedCard = { card, index };
  arenaHaptic('selection', { key: 'card-pick', minInterval: 90 });
  playArenaSfx('cardSelected', { volume: 0.62 });
  
  // Получаем возможные цели для этой карты
  const playActions = getPlayCardTargets(index);
  const hasTargetingOptions = playActions.length > 0 && playActions.some(a => a.target_id !== null && a.target_id !== undefined);
  const hasNoTargetOption = playActions.some(a => a.target_id === null || a.target_id === undefined);
  const requiresTarget = hasTargetingOptions && !hasNoTargetOption;
  
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
        // Доверяем legal_actions: если свой герой пришёл целью, его можно выбрать.
        if (!isFreeze) {
          playerHeroTargetable = true;
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
    opponentPanel.onmouseenter = () => showDamagePreview(
      opponentPanel,
      true,
      actions.find(a => String(a.target_id) === String(opponentHeroId) || (a.target_is_hero && !a.target_id))
    );
    opponentPanel.onmouseleave = () => hideDamagePreview(opponentPanel, true);
  }
  
  // Подсвечиваем своего героя ТОЛЬКО если он в целях (лечение)
  if (playerHeroTargetable && playerPanel) {
    playerPanel.classList.add('targetable-friendly');
    console.log('[ARENA] 💚 Свой герой подсвечен для лечения');

    // Предпросмотр
    playerPanel.onmouseenter = () => showDamagePreview(
      playerPanel,
      true,
      actions.find(a => String(a.target_id) === String(playerHeroId) || (a.target_is_hero && !a.target_id))
    );
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
      unit.classList.add('attack-target', 'targetable-enemy', 'status-attack-target');
      
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
    unit.classList.remove('attack-target', 'targetable-enemy', 'status-attack-target');
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
        const clickedElement = event?.target?.closest('.player-panel-root, .opponent-panel-root');
        const isPlayerHero = clickedElement?.classList.contains('player-panel-root');
        const clickedHeroId = isPlayerHero
          ? currentState?.player?.hero?.instance_id
          : currentState?.opponent?.hero?.instance_id;

        if (a.target_id) {
          return String(a.target_id) === String(clickedHeroId);
        }
        return a.target_is_hero === true && !isPlayerHero;
      }
      return String(a.target_id) === String(targetId);
    });
    
    if (!isValidTarget && playActions.length > 0) {
      console.warn('[ARENA] ❌ Цель не в списке валидных для карты');
      arenaHaptic('warning', { key: 'target-invalid', minInterval: 160 });
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
      arenaHaptic('medium', { key: 'target-card', minInterval: 120 });
      executeTargetingPlay(heroId, true);
    } else {
      console.log('[ARENA] 🎯 Разыгрываем карту на юнита:', targetId);
      arenaHaptic('medium', { key: 'target-card', minInterval: 120 });
      executeTargetingPlay(targetId, false);
    }
    
  } else if (interactionMode.type === 'ATTACK') {
    // БЛОКИРОВКА: В режиме атаки НЕ должно быть кликов по врагам из TARGETING
    // Проверяем, что цель валидна для атаки
    if (!isValidAttackTarget(targetId, isHero)) {
      console.warn('[ARENA] ❌ Цель не валидна для атаки (taunt/legal_actions)');
      arenaHaptic('warning', { key: 'target-invalid', minInterval: 160 });
      return;
    }
    
    arenaHaptic('medium', { key: 'target-attack', minInterval: 120 });
    attack(interactionMode.data.instance_id, isHero ? null : targetId, isHero);
    resetInteractionMode();
    
  } else {
    console.warn('[ARENA] ⚠️ Клик по цели в режиме NONE - игнорируем');
    arenaHaptic('warning', { key: 'target-invalid', minInterval: 160 });
  }
}

// ============================================
// ПРЕДПРОСМОТР УРОНА (DAMAGE PREVIEW)
// ============================================

async function showDamagePreview(targetEl, isHero, targetData) {
  // Предпросмотр текста здоровья отключён — оставлена только CSS-подсветка целей
  return;
}

function hideDamagePreview(targetEl, isHero) {
  // Предпросмотр текста здоровья отключён — оставлена только CSS-подсветка целей
  return;
}

function handleAttackerClick(attackerCard) {
  console.log('[ARENA] Выбран атакующий:', attackerCard);
  
  // Если ход не наш, игнорируем
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход, атака невозможна');
    arenaHaptic('warning', { key: 'attacker-invalid', minInterval: 160 });
    return;
  }
  
  // ИСПРАВЛЕНО: Приоритет свойству can_attack самой карты
  const unitCanAttack = attackerCard.can_attack || canAttack(attackerCard.instance_id);

  if (!unitCanAttack) {
    console.warn('[ARENA] ❌ Существо не может атаковать');
    arenaHaptic('warning', { key: 'attacker-invalid', minInterval: 160 });
    return;
  }
  
  // Устанавливаем режим атаки
  interactionMode = {
    type: 'ATTACK',
    data: attackerCard
  };
  
  selectedAttacker = attackerCard;
  arenaHaptic('selection', { key: 'attacker-select', minInterval: 120 });
  playArenaSfx('cardSelected', { volume: 0.62 });
  
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
        _auth: authToken,
        hand_index: handIndex,
        card_id: card.card_id || card.id || card.instance_id,
        target_position: position,
        target_id: targetId,
        target_is_hero: targetIsHero,
        client_action_id: makeClientActionId('play_card')
      })
    });
    
    if (!response.ok) {
      throw await parseActionError(response, 'Не удалось разыграть карту');
    }
    
    const result = await response.json();
    console.log('[ARENA] Карта разыграна:', result);
    
    // Обновляем состояние
    if (result.state) {
      arenaHaptic('medium', { key: 'play-card-ok', minInterval: 140 });
      currentState = result.state;
      renderBattleState(result.state);
      
      // Рендерим поля для обоих игроков
      renderBoard('player', (currentState.player || currentState).board || []);
      renderBoard('opponent', (currentState.opponent || currentState).board || []);
    }
    
  } catch (error) {
    console.error('[ARENA] Ошибка розыгрыша карты:', error);
    arenaHaptic('error', { key: 'play-card-error', minInterval: 220 });
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
        _auth: authToken,
        hand_index: handIndex,
        card_id: card.card_id || card.id || card.instance_id,
        target_id: targetId,
        target_is_hero: targetIsHero,
        client_action_id: makeClientActionId('play_card')
      })
    });
    
    if (!response.ok) {
      throw await parseActionError(response, 'Не удалось разыграть зелье');
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
        arenaHaptic('medium', { key: 'play-card-ok', minInterval: 140 });
        currentState = result.state;
        renderBattleState(result.state);

        // ДОБАВЛЕНО: Принудительная перерисовка для обновления статуса атаки
        renderBoard('player', (currentState.player || currentState).board || []);
      }
    }, 400);
    
  } catch (error) {
    console.error('[ARENA] Ошибка розыгрыша зелья:', error);
    arenaHaptic('error', { key: 'play-card-error', minInterval: 220 });
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
        _auth: authToken,
        attacker_id: attackerId,
        target_id: targetId,
        target_is_hero: targetIsHero,
        client_action_id: makeClientActionId('attack')
      })
    });
    
    if (!response.ok) {
      throw await parseActionError(response, 'Не удалось атаковать');
    }
    
    const result = await response.json();
    console.log('[ARENA] Атака выполнена:', result);
    
    // Обновляем состояние
    if (result.state) {
      arenaHaptic('medium', { key: 'attack-ok', minInterval: 140 });
      currentState = result.state;
      renderBattleState(result.state);
      
      // Рендерим поля для обоих игроков
      renderBoard('player', (currentState.player || currentState).board || []);
      renderBoard('opponent', (currentState.opponent || currentState).board || []);
    }
    
  } catch (error) {
    console.error('[ARENA] Ошибка атаки:', error);
    arenaHaptic('error', { key: 'attack-error', minInterval: 220 });
    alert('Не удалось атаковать: ' + error.message);
  }
}

async function endTurn() {
  if (isArenaWaitingForPlayers(currentState)) {
    console.warn('[ARENA] Бой еще синхронизируется');
    return;
  }
  if (!currentState || !currentState.is_my_turn) {
    console.warn('[ARENA] Не ваш ход');
    return;
  }
  
  try {
    console.log('[ARENA] Завершение хода');
    arenaHaptic('selection', { key: 'end-turn', minInterval: 120 });
    playArenaSfx('nextMove', { volume: 0.7 });
    
    const response = await fetch('/api/battle/end-turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        _auth: authToken,
        client_action_id: makeClientActionId('end_turn')
      })
    });
    
    if (!response.ok) {
      throw await parseActionError(response, 'Не удалось завершить ход');
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
    arenaHaptic('error', { key: 'end-turn-error', minInterval: 220 });
    alert('Не удалось завершить ход: ' + error.message);
  }
}

async function surrender() {
  try {
    console.log('[ARENA] Сдача через Socket.IO');
    arenaHaptic('error', { key: 'surrender-confirm', minInterval: 260 });
    
    // Отправляем событие сдачи через Socket.IO
    socket.emit('surrender', {
      match_id: matchId,
      _auth: authToken,
      client_action_id: makeClientActionId('surrender')
    });
    
    // Ждём подтверждения от сервера
    socket.once('surrender_ack', (data) => {
      console.log('[ARENA] Сдача подтверждена:', data);

      window.__surrenderAck = {
        trophy_penalty: data.trophy_penalty || 0,
        new_trophies: data.new_trophies || null,
      };

      // Fallback: если game_over не пришел за 1500мс
      setTimeout(() => {
        const modal = document.getElementById('battle-result-modal');
        const isVisible = modal && (modal.style.display === 'flex' || modal.classList.contains('visible'));
        
        if (!isVisible) {
          console.warn('[ARENA] Server game_over timeout. Forcing local defeat screen.');
          const ack = window.__surrenderAck || {};
          showBattleResult(
            'defeat',
            ack.trophy_penalty || 0,
            ack.new_trophies || null,
            0,
            null
          );
        }
      }, 1500);
    });
    
    // Обработка ошибок
    socket.once('error', (error) => {
      console.error('[ARENA] Ошибка сдачи:', error);
      arenaHaptic('error', { key: 'surrender-error', minInterval: 260 });
      alert('Не удалось сдаться: ' + error.message);
    });
    
  } catch (error) {
    console.error('[ARENA] Ошибка сдачи:', error);
    arenaHaptic('error', { key: 'surrender-error', minInterval: 260 });
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
  const winnerId = data.winner_id ?? data.winner ?? currentState?.winner_id ?? null;
  const outcome = winnerId == null ? 'draw' : (String(winnerId) === String(userId) ? 'victory' : 'defeat');
  
  console.log('[ARENA] 🎯 Результат:', outcome, '| winnerId =', winnerId, '| myId =', userId);
  
  // КРИТИЧНО: Извлекаем данные о трофеях из state (синхронизированы с БД)
  // Приоритет: players[userId] (новый game_over) > top-level (surrender personalized) > currentState > __surrenderAck
  const myEconomy = (data.players && (data.players[String(userId)] || data.players[userId])) || {};
  const surrenderAck = window.__surrenderAck || {};
  const trophyDelta = parseInt(
    myEconomy.trophy_delta
    || data.trophy_change || data.trophy_delta
    || currentState?.trophy_change || currentState?.trophy_delta
    || surrenderAck.trophy_penalty
    , 10
  ) || 0;
  const trophyTotal = parseInt(
    myEconomy.trophy_total
    || data.trophy_total
    || currentState?.trophy_total
    || surrenderAck.new_trophies
    , 10
  ) || null;
  
  // КРИТИЧНО: Извлекаем данные о монетах из state (синхронизированы с БД)
  const coinsDelta = parseInt(
    myEconomy.coins_delta
    || data.coins_delta || data.coins_change
    || currentState?.coins_delta || currentState?.coins_change
    , 10) || 0;
  const coinsTotal = parseInt(
    myEconomy.coins_total
    || data.coins_total
    || currentState?.coins_total
    , 10) || null;

  const starsDelta = parseInt(
    myEconomy.stars_delta
    || data.stars_delta
    || currentState?.stars_delta
    , 10) || 0;
  const starsTotal = parseInt(
    myEconomy.stars_total
    || data.stars_total
    || currentState?.stars_total
    , 10) || null;

  const leagueUp = myEconomy.league_up || data.league_up || currentState?.league_up || null;
  if (leagueUp) {
    sessionStorage.setItem('arena_league_up', JSON.stringify(leagueUp));
  }

  console.log('[ARENA] 🏆 Трофеи: delta =', trophyDelta, '| total =', trophyTotal);
  console.log('[ARENA] 🪙 Монеты: delta =', coinsDelta, '| total =', coinsTotal);
  console.log('[ARENA] ⭐ Звёзды: delta =', starsDelta, '| total =', starsTotal);
  if (leagueUp) console.log('[ARENA] 🏆 Повышение лиги:', leagueUp);
  
  // Сохраняем экономику в кеш с мержем (не перезатираем непустую пустой)
  _mergeBattleResultEconomy({
    trophyDelta, trophyTotal,
    coinsDelta, coinsTotal,
    starsDelta, starsTotal,
    leagueUp,
  });
  
  // Показываем экран результата с небольшой задержкой для драматического эффекта
  setTimeout(() => {
    showBattleResult(outcome, trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal);
  }, 800);
}

function showBattleResult(outcome, trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal) {
  if (typeof outcome === 'boolean') {
    outcome = outcome ? 'victory' : 'defeat';
  }
  if (!['victory', 'defeat', 'draw'].includes(outcome)) {
    outcome = outcome ? String(outcome) : 'defeat';
    if (!['victory', 'defeat', 'draw'].includes(outcome)) outcome = 'defeat';
  }
  const isWinner = outcome === 'victory';
  const isDraw = outcome === 'draw';
  if (!window.__arenaBattleResultHaptic) {
    window.__arenaBattleResultHaptic = true;
    arenaHaptic(isWinner ? 'success' : (isDraw ? 'warning' : 'error'), { key: 'battle-result-' + outcome, minInterval: 500 });
  }
  console.log('[ARENA] 🎬 showBattleResult called:', {
    outcome, trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal,
    alreadyShown: !!window.__resultModalShown,
    cached: window.__battleResultEconomy ? 'yes' : 'no'
  });
  
  // ⛔ Если модал уже показан и новые данные пустые — пользуемся кешем
  if (window.__resultModalShown) {
    const cached = _readEconomyFromCache();
    const nextNonEmpty = _isEconomyNonEmpty({trophyDelta, trophyTotal, coinsDelta, coinsTotal, starsDelta, starsTotal});
    if (!nextNonEmpty) {
      console.log('[ARENA] 🚫 Повторный вызов с пустой экономикой, игнорируем');
      return;
    }
    // Непустая экономика при повторном вызове — разрешаем обновить модал
    console.log('[ARENA] 🔄 Обновляем модал новыми непустыми данными');
  }
  
  window.__resultModalShown = true;
  
  const modal = document.getElementById('battle-result-modal');
  const icon = document.getElementById('result-icon');
  const title = document.getElementById('result-title');
  const subtitle = document.getElementById('result-subtitle');
  const trophyDeltaEl = document.getElementById('result-trophy-delta');
  const trophyTotalEl = document.getElementById('result-trophy-total');
  const trophySection = document.getElementById('result-trophy-section');
  const coinsDeltaEl = document.getElementById('result-coins-delta');
  const coinsTotalEl = document.getElementById('result-coins-total');
  const coinsSection = document.getElementById('result-coins-section');
  const shareBtn = document.getElementById('result-share-btn');
  const rewardsGrid = document.getElementById('result-rewards-grid');
  const noRewardSection = document.getElementById('result-rewards-section');
  const card = modal ? modal.querySelector('.result-card') : null;
  
  if (!modal || !icon || !title || !card) {
    console.error('[ARENA] Элементы модального окна результата не найдены');
    alert(isWinner ? 'Победа!' : (isDraw ? 'Ничья' : 'Поражение'));
    setTimeout(() => window.location.replace('/'), 1500);
    return;
  }
  
  const opponentName = document.getElementById('opponent-name-text')?.textContent || 'Оппонент';
  card.classList.remove('victory', 'defeat', 'draw');
  card.classList.add(outcome);
  if (outcome === 'victory') {
    title.textContent = 'Победа';
    if (subtitle) subtitle.textContent = opponentName + ' повержен на арене.';
  } else if (outcome === 'draw') {
    title.textContent = 'Ничья';
    if (subtitle) subtitle.textContent = 'Оба игрока удержали арену.';
  } else {
    title.textContent = 'Поражение';
    if (subtitle) subtitle.textContent = 'Реванш доступен из меню арены.';
  }
  
  // Настройка кнопки "Поделиться"
  if (shareBtn) {
    const winnerHP = isWinner ? (document.getElementById('player-hp-text')?.textContent || '0') : (document.getElementById('opponent-hp-text')?.textContent || '0');
    const turnCount = currentTurnCount;
    const botLink = 'https://t.me/extraarena_bot';
    
    let shareText = '';
    if (isWinner) {
      shareText = `Я победил игрока ${opponentName} в @extraarena_bot! 🏆\nБитва длилась ${turnCount} ходов. Мой герой выжил с ${winnerHP} здоровья!\n\nСможешь лучше? Принимай вызов! ⚔️`;
      shareBtn.style.display = 'grid';
    } else if (isDraw) {
      shareText = `Я сыграл вничью с ${opponentName} в @extraarena_bot! ⚔️\nБитва длилась ${turnCount} ходов.\n\nПрисоединяйся к битве! 🏆`;
      shareBtn.style.display = 'none';
    } else {
      shareText = `Я сразился с ${opponentName} в @extraarena_bot! ⚔️\nБитва длилась ${turnCount} ходов. В следующий раз победа будет за мной!\n\nПрисоединяйся к битве! 🏆`;
      shareBtn.style.display = 'grid';
    }
    
    const encodedText = encodeURIComponent(shareText);
    shareBtn.href = `https://t.me/share/url?url=${botLink}&text=${encodedText}`;
    if (shareBtn.__extraArenaShareHandler) {
      shareBtn.removeEventListener('click', shareBtn.__extraArenaShareHandler);
    }
    shareBtn.__extraArenaShareHandler = (event) => {
      if (!window.shareExtraArena) return;
      event.preventDefault();
      event.stopPropagation();
      window.shareExtraArena(shareText, botLink, 'ExtraArena');
    };
    shareBtn.addEventListener('click', shareBtn.__extraArenaShareHandler);
  }
  
  // Отображаем трофеи с анимацией счетчика
  const displayTrophyTotal = trophyTotal ?? currentState?.player?.trophies ?? null;
  const hasTrophyDelta = trophyDelta !== undefined && trophyDelta !== null && trophyDelta !== 0;
  const hasTrophyTotal = displayTrophyTotal !== undefined && displayTrophyTotal !== null;

  if (hasTrophyDelta || hasTrophyTotal || isDraw) {
    if (trophySection) trophySection.style.display = 'grid';
    
    if (trophyDeltaEl) {
      if (hasTrophyDelta) {
        const deltaSign = trophyDelta > 0 ? '+' : '-';
        const deltaAbsValue = Math.abs(trophyDelta);
        trophyDeltaEl.className = 'delta trophy-delta ' + (trophyDelta > 0 ? 'positive' : 'negative');
        
        // Анимация счетчика трофеев (с абсолютным значением)
        animateCounter(trophyDeltaEl, 0, deltaAbsValue, 1000, deltaSign);
        
        trophyDeltaEl.style.display = 'block';
      } else {
        trophyDeltaEl.textContent = '0';
        trophyDeltaEl.className = 'delta trophy-delta neutral';
        trophyDeltaEl.style.display = 'block';
      }
    }
    
    if (trophyTotalEl && hasTrophyTotal) {
      // КРИТИЧНО: Анимация счетчика общих трофеев (используем state.trophy_total из БД)
      const startValue = hasTrophyDelta ? Math.max(0, displayTrophyTotal - trophyDelta) : displayTrophyTotal;
      animateCounter(trophyTotalEl, startValue, displayTrophyTotal, 1000);
    } else if (trophyTotalEl) {
      trophyTotalEl.textContent = '—';
    }
  } else if (trophySection) {
    // Скрываем секцию трофеев, только если нет ни дельты, ни общего количества
    trophySection.style.display = 'none';
  }
  
  // ДОБАВЛЕНО: Отображаем монеты с анимацией счетчика
  const hasCoinsDelta = coinsDelta !== undefined && coinsDelta !== null && coinsDelta !== 0;
  const hasCoinsTotal = coinsTotal !== undefined && coinsTotal !== null;

  if (hasCoinsDelta || hasCoinsTotal) {
    if (coinsSection) coinsSection.style.display = 'grid';
    
    if (coinsDeltaEl) {
      if (hasCoinsDelta) {
        const deltaSign = coinsDelta > 0 ? '+' : '-';
        const deltaAbsValue = Math.abs(coinsDelta);
        coinsDeltaEl.className = 'coins-delta ' + (coinsDelta > 0 ? 'positive' : 'negative');
        
        // Анимация счетчика монет (с абсолютным значением)
        animateCounter(coinsDeltaEl, 0, deltaAbsValue, 1200, deltaSign);
        
        coinsDeltaEl.style.display = '';
      } else {
        coinsDeltaEl.textContent = '+0';
        coinsDeltaEl.style.display = '';
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

  // Звёзды (Battle Pass)
  const starsDeltaEl = document.getElementById('result-stars-delta');
  const starsTotalEl = document.getElementById('result-stars-total');
  const starsSection = document.getElementById('result-stars-section');
  const hasStarsDelta = starsDelta !== undefined && starsDelta !== null && starsDelta !== 0;
  const hasStarsTotal = starsTotal !== undefined && starsTotal !== null;

  if (hasStarsDelta || hasStarsTotal) {
    if (starsSection) starsSection.style.display = 'grid';

    if (starsDeltaEl) {
      if (hasStarsDelta) {
        const deltaSign = starsDelta > 0 ? '+' : '-';
        starsDeltaEl.className = 'stars-delta positive';
        animateCounter(starsDeltaEl, 0, Math.abs(starsDelta), 1200, deltaSign);
        starsDeltaEl.style.display = '';
      } else {
        starsDeltaEl.textContent = '+0';
        starsDeltaEl.style.display = '';
      }
    }

    if (starsTotalEl && hasStarsTotal) {
      const startValue = hasStarsDelta ? Math.max(0, starsTotal - starsDelta) : starsTotal;
      animateCounter(starsTotalEl, startValue, starsTotal, 1200);
    }
  } else if (starsSection) {
    starsSection.style.display = 'none';
  }

  if (isDraw) {
    if (rewardsGrid) rewardsGrid.style.display = 'none';
    if (noRewardSection) {
      noRewardSection.style.display = 'block';
      noRewardSection.textContent = 'Награды не начислены за этот бой.';
    }
  } else if (outcome === 'defeat') {
    if (rewardsGrid) rewardsGrid.style.display = 'none';
    if (noRewardSection) {
      noRewardSection.style.display = 'block';
      noRewardSection.textContent = 'Награды не начислены за поражение.';
    }
  } else {
    if (rewardsGrid) rewardsGrid.style.display = 'grid';
    if (noRewardSection) noRewardSection.style.display = 'none';
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
  
  // Получаем здоровья блок
  const hpBlock = isPlayer 
    ? document.querySelector('.player-hp-block')
    : document.querySelector('.opponent-hp-block');
  
  if (!hpBlock) {
    console.warn('[ARENA] здоровья block not found for healing animation');
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
// CARD INFO MODAL (bottom-sheet popup)
// ============================================

function openCardInfo(card) {
  console.log('[ARENA] Card info modal:', card);
  
  const modal = document.getElementById('card-info-modal');
  if (!modal) return;

  const cardName = card.name || 'Карта';
  const cardLevel = card.level;
  
  document.getElementById('card-info-name').textContent = cardName;
  document.getElementById('card-info-level').textContent = cardLevel != null ? 'Уровень ' + cardLevel : '';
  document.getElementById('card-info-attack').textContent = card.attack != null ? card.attack : (card.atk != null ? card.atk : '—');
  
  const hp = card.hp ?? card.hp_current;
  const maxHp = card.maxHp ?? card.max_hp;
  document.getElementById('card-info-health').textContent = hp != null ? (maxHp ? hp + '/' + maxHp : hp) : '—';
  
  const rawMana = getRawManaCost(card);
  const effectiveMana = getEffectiveManaCost(card);
  const manaInfo = document.getElementById('card-info-mana');
  if (manaInfo) {
    manaInfo.textContent = effectiveMana === 0 && rawMana > 0 ? '0' : rawMana;
    manaInfo.title = effectiveMana === 0 && rawMana > 0 ? 'Бесплатно в SpellStorm' : '';
  }
  document.getElementById('card-info-description').textContent = card.description || card.text || 'Нет описания.';

  const artEl = document.getElementById('card-info-art');
  const fb = document.getElementById('card-info-art-fallback');
  if (card.image) {
    artEl.src = card.image;
    artEl.style.display = 'block';
    fb.style.display = 'none';
  } else {
    artEl.style.display = 'none';
    fb.style.display = 'flex';
    fb.textContent = card.emoji || '⚔️';
  }

  const mechEl = document.getElementById('card-info-mechanics');
  mechEl.innerHTML = '';
  const mechanics = card.mechanics || [];
  const addMechanicDetail = function(title, description, kind, iconPath) {
    const row = document.createElement('div');
    row.className = 'mechanic-detail-row';

    const icon = document.createElement('div');
    icon.className = 'mechanic-detail-icon' + (kind ? ' mechanic-kind-' + kind : '');
    const iconImg = document.createElement('img');
    iconImg.className = 'mechanic-detail-img';
    iconImg.src = iconPath || getMechanicIconPath('');
    iconImg.alt = '';
    icon.appendChild(iconImg);

    const body = document.createElement('div');
    body.className = 'mechanic-detail-body';

    const titleEl = document.createElement('div');
    titleEl.className = 'mechanic-detail-title';
    titleEl.textContent = title || 'Механика';

    const descEl = document.createElement('div');
    descEl.className = 'mechanic-detail-description';
    descEl.textContent = description || 'Описание механики отсутствует.';

    body.appendChild(titleEl);
    body.appendChild(descEl);
    row.appendChild(icon);
    row.appendChild(body);
    mechEl.appendChild(row);
  };

  if (card.mechanics_desc) {
    addMechanicDetail('МЕХАНИКА', card.mechanics_desc, 'database', getMechanicIconPath(mechanics[0] || ''));
  } else if (mechanics.length > 0) {
    mechanics.forEach(function(m) {
      const parsed = parseMechanic(m);
      if (!parsed) return;
      addMechanicDetail(parsed.label, parsed.description, parsed.kind, getMechanicIconPath(m, parsed));
    });
  }
  
  openBattleModal('card-info');
}

function closeCardInfo() {
  if (activeBattleModal === 'card-info') closeBattleModal();
}

function getMechanicIconPath(mechanic, parsed) {
  const key = String(mechanic || parsed?.label || parsed?.kind || '').toLowerCase();
  const base = '../DesignAssets/Arena/CardEffects/';
  if (key.includes('freeze') || key.includes('frozen')) return base + 'freeze.png';
  if (key.includes('shield') || key.includes('armor') || key.includes('reflect') || key.includes('bypass')) return base + 'shield.png';
  if (key.includes('taunt') || key.includes('provoc')) return base + 'provocation.png';
  if (key.includes('regen') || key.includes('heal') || key.includes('lifesteal')) return base + 'toHeal.png';
  if (key.includes('poison')) return base + 'poison.png';
  if (key.includes('stealth') || key.includes('delete') || key.includes('instant')) return base + 'stealth.png';
  if (key.includes('target') || key.includes('damage') || key.includes('aoe') || key.includes('cleave')) return base + 'target.png';
  if (key.includes('asleep') || key.includes('mana') || key.includes('drain')) return base + 'asleep.png';
  if (parsed?.kind === 'battlecry' || parsed?.kind === 'deathrattle') return base + 'target.png';
  return base + 'shield.png';
}

// ============================================
// SURRENDER HOLD-TO-ACTIVATE (1.5s)
// ============================================

function initSurrenderHold() {
  const btn = document.getElementById('surrender-hold-btn');
  const ring = document.getElementById('surrender-ring');
  if (!btn) return;
  
  const holdDuration = 1500;
  let holdStart = null;
  let rafId = null;
  let triggered = false;

  function update(ts) {
    if (!holdStart) return;
    const elapsed = ts - holdStart;
    const pct = Math.min(100, (elapsed / holdDuration) * 100);
    if (ring) ring.style.setProperty('--pct', pct + '%');
    if (elapsed >= holdDuration && !triggered) {
      triggered = true;
      cancelAnimationFrame(rafId);
      btn.classList.remove('holding');
      if (ring) ring.style.setProperty('--pct', '0%');
      openSurrenderModal();
      return;
    }
    rafId = requestAnimationFrame(update);
  }

  function start(e) {
    e.preventDefault();
    e.stopPropagation(); // не даём всплыть до player-panel-root (режим TARGETING)
    triggered = false;
    holdStart = performance.now();
    btn.classList.add('holding');
    rafId = requestAnimationFrame(update);
  }

  function end() {
    holdStart = null;
    triggered = false;
    btn.classList.remove('holding');
    cancelAnimationFrame(rafId);
    if (ring) ring.style.setProperty('--pct', '0%');
  }

  btn.addEventListener('mousedown', start);
  btn.addEventListener('touchstart', start, { passive: false });
  ['mouseup', 'mouseleave', 'touchend', 'touchcancel'].forEach(function(ev) {
    btn.addEventListener(ev, end);
  });
}

// ============================================
// END-TURN PULSE (no legal actions)
// ============================================

function checkEndTurnPulse() {
  const btn = document.getElementById('end-turn-button');
  if (!btn) return;
  
  const isMyTurn = currentState?.is_my_turn;
  const legalActions = cachedLegalActions || [];
  const realActions = legalActions.filter(a => a.type !== 'end_turn');
  const hasLegalActions = realActions.length > 0;
  
  if (isMyTurn && !hasLegalActions) {
    btn.classList.add('pulse-ready');
  } else {
    btn.classList.remove('pulse-ready');
  }
}

// ============================================
// OPPONENT INFO BUTTON
// ============================================

function openOpponentInfo() {
  const opponentState = window.__arenaOpponentState || currentState?.opponent || {};
  const hero = opponentState.hero;
  if (hero && hero.name) {
    openCardInfo({
      ...hero,
      name: hero.name,
      level: hero.level,
      attack: hero.attack ?? hero.atk ?? 0,
      hp: hero.hp ?? hero.hp_current ?? 30,
      max_hp: hero.max_hp ?? hero.maxHp ?? 30,
      mana: hero.mana ?? hero.mana_cost ?? 0,
      description: hero.description || '',
      mechanics: hero.mechanics || [],
      mechanics_desc: hero.mechanics_desc,
      card_type: hero.card_type || 'hero',
      image: hero.image,
      emoji: '🛡️'
    });
  } else {
    openCardInfo({
      name: opponentState.name || 'Оппонент',
      level: null,
      attack: null,
      hp: opponentState.hp ?? 30,
      max_hp: opponentState.max_hp ?? 30,
      mana: opponentState.mana ?? 0,
      description: 'Противник',
      mechanics: [],
      emoji: '👤'
    });
  }
}

// ============================================
// ФОНОВАЯ МУЗЫКА АРЕНЫ
// ============================================

function initArenaMusic() {
  const urlParams = new URLSearchParams(location.search);
  window._musicEnabled = urlParams.get('music') !== '0';
  window._sfxEnabled = urlParams.get('sfx') !== '0';
  initArenaSfx();

  const music = document.getElementById('arena-bg-music');
  if (!music) {
    console.warn('[ARENA] Элемент arena-bg-music не найден');
    return;
  }

  let musicStarted = false;
  let musicPausedByLifecycle = false;
  let musicManualStop = false;
  let musicWatchdogInstalled = false;

  const shouldKeepArenaMusicRunning = () => {
    return window._musicEnabled && !musicManualStop && !document.hidden;
  };

  const ensureArenaMusicWatchdog = () => {
    if (musicWatchdogInstalled) return;
    musicWatchdogInstalled = true;
    music.loop = true;

    const resumeIfNeeded = () => {
      if (!shouldKeepArenaMusicRunning()) return;
      musicStarted = false;
      startMusic();
    };

    ['ended', 'stalled', 'suspend', 'emptied'].forEach(eventName => {
      music.addEventListener(eventName, () => setTimeout(resumeIfNeeded, 250));
    });
    music.addEventListener('pause', () => {
      if (musicPausedByLifecycle) return;
      setTimeout(resumeIfNeeded, 400);
    });
    setInterval(() => {
      if (shouldKeepArenaMusicRunning() && musicStarted && (music.paused || music.ended)) {
        resumeIfNeeded();
      }
    }, 2500);
  };

  const startMusic = () => {
    if (!window._musicEnabled) return;
    ensureArenaMusicWatchdog();
    if (musicStarted && !music.paused && !music.ended) return;

    musicManualStop = false;
    music.loop = true;
    music.volume = 0.3;
    music.play().then(() => {
      console.log('[ARENA] 🎵 Фоновая музыка запущена');
      musicStarted = true;
    }).catch(err => {
      console.warn('[ARENA] Не удалось запустить музыку:', err);
    });

    document.body.removeEventListener('click', startMusic);
  };

  if (window._musicEnabled) {
    document.addEventListener('pointerdown', startMusic, { once: true, capture: true, passive: true });
    document.addEventListener('touchstart', startMusic, { once: true, capture: true, passive: true });
    document.addEventListener('keydown', startMusic, { once: true, capture: true });
    document.body.addEventListener('click', startMusic, { once: true, capture: true });
    if (isArenaAndroidShell()) setTimeout(startMusic, 150);
  }

  const pauseAllArenaMedia = (resetPosition = false) => {
    if (resetPosition) musicManualStop = true;
    document.querySelectorAll('audio, video').forEach(media => {
      try {
        media.pause();
        if (resetPosition) media.currentTime = 0;
      } catch (e) {}
    });
    musicStarted = false;
  };

  window.startArenaMusic = function() {
    startMusic();
  };

  window.stopArenaMusic = function() {
    pauseAllArenaMedia(true);
  };

  window.ExtraArenaAppPause = function() {
    musicPausedByLifecycle = musicPausedByLifecycle || musicStarted || !music.paused;
    pauseAllArenaMedia(false);
  };

  window.ExtraArenaAppResume = function() {
    if (musicPausedByLifecycle && !document.hidden && window._musicEnabled) {
      startMusic();
    }
    musicPausedByLifecycle = false;
  };

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) window.ExtraArenaAppPause();
    else window.ExtraArenaAppResume();
  });
  window.addEventListener('pagehide', () => window.ExtraArenaAppPause());
}

// ============================================
// АКТИВНЫЕ ЭФФЕКТЫ — HELPERS
// ============================================

/**
 * Парсит строку механики в читаемый объект эффекта.
 * Поддерживает форматы:
 *   - single: taunt, shield, charge, lifesteal, delete_target, ...
 *   - prefix_N: regen_1, armor_2, reflect_2, aura_atk_1, ...
 *   - prefix_X_Y: cleave_1_3, aura_atk_1_3, armor_1_3, battlecry_buff_2_3, ...
 *   - compound_word_N: deathrattle_aoe_damage_3, battlecry_heal_hero_2, ...
 * @param {string} mechanic
 * @returns {{label: string, value: string, description: string, kind: string}|null}
 */
function parseMechanic(mechanic) {
  if (!mechanic || typeof mechanic !== 'string') return null;
  var m = mechanic.trim();

  // ── Compound suffix (most specific → least specific) ──

  var drAoe = m.match(/^deathrattle_aoe_damage_(\d+)$/);
  if (drAoe) return { label: 'Предсмертный: массовый урон', value: drAoe[1], description: 'При гибели наносит ' + drAoe[1] + ' урона всем врагам', kind: 'deathrattle' };

  var drDmg = m.match(/^deathrattle_damage_(\d+)$/);
  if (drDmg) return { label: 'Предсмертный: урон', value: drDmg[1], description: 'При гибели наносит ' + drDmg[1] + ' урона всем врагам', kind: 'deathrattle' };

  var drSum = m.match(/^deathrattle_summon_(\d+)$/);
  if (drSum) return { label: 'Предсмертный: призыв', value: drSum[1], description: 'При гибели призывает существо', kind: 'deathrattle' };

  var bcHealHero = m.match(/^battlecry_heal_hero_(\d+)$/);
  if (bcHealHero) return { label: 'Эффект розыгрыша: лечение героя', value: bcHealHero[1], description: 'При розыгрыше лечит героя на ' + bcHealHero[1] + ' здоровья', kind: 'battlecry' };

  var bcHealTarget = m.match(/^battlecry_heal_target_(\d+)$/);
  if (bcHealTarget) return { label: 'Эффект розыгрыша: лечение цели', value: bcHealTarget[1], description: 'При розыгрыше лечит цель на ' + bcHealTarget[1] + ' здоровья', kind: 'battlecry' };

  var bcAoeDmg = m.match(/^battlecry_aoe_damage_(\d+)$/);
  if (bcAoeDmg) return { label: 'Эффект розыгрыша: массовый урон', value: bcAoeDmg[1], description: 'При розыгрыше наносит ' + bcAoeDmg[1] + ' урона всем врагам', kind: 'battlecry' };

  var spellDmg = m.match(/^spell_damage_(\d+)$/);
  if (spellDmg) return { label: 'Заклинание: урон', value: spellDmg[1], description: 'Наносит ' + spellDmg[1] + ' урона', kind: 'passive' };

  var spellHeal = m.match(/^spell_heal_(\d+)$/);
  if (spellHeal) return { label: 'Заклинание: лечение', value: spellHeal[1], description: 'Восстанавливает ' + spellHeal[1] + ' здоровья', kind: 'passive' };

  var spellAoe = m.match(/^spell_aoe_damage_(\d+)$/);
  if (spellAoe) return { label: 'Заклинание: массовый урон', value: spellAoe[1], description: 'Наносит ' + spellAoe[1] + ' урона всем врагам', kind: 'passive' };

  // ── Multi-number X_Y patterns ──

  var cleaveXy = m.match(/^cleave_(\d+)_(\d+)$/);
  if (cleaveXy) return { label: 'Разрубание', value: cleaveXy[1] + '×' + cleaveXy[2], description: 'Наносит ' + cleaveXy[1] + ' урона до ' + cleaveXy[2] + ' соседним врагам', kind: 'passive' };

  var auraXy = m.match(/^aura_atk_(\d+)_(\d+)$/);
  if (auraXy) return { label: 'Аура атаки', value: '+' + auraXy[1], description: '+' + auraXy[1] + ' к атаке союзных существ', kind: 'aura' };

  var armorXy = m.match(/^armor_(\d+)_(\d+)$/);
  if (armorXy) return { label: 'Броня', value: armorXy[1], description: 'Входящий урон –' + armorXy[1], kind: 'passive' };

  var startManaXy = m.match(/^start_mana_(\d+)_(\d+)$/);
  if (startManaXy) return { label: 'Стартовая мана', value: '+' + startManaXy[1], description: '+' + startManaXy[1] + ' маны в начале боя', kind: 'start' };

  var damageXy = m.match(/^damage_(\d+)_(\d+)$/);
  if (damageXy) return { label: 'Урон', value: damageXy[1], description: 'Наносит ' + damageXy[1] + ' урона', kind: 'passive' };

  var bcBuffXy = m.match(/^battlecry_buff_(\d+)_(\d+)$/);
  if (bcBuffXy) return { label: 'Эффект розыгрыша: усиление', value: '+' + bcBuffXy[1], description: 'При розыгрыше усиливает союзников на +' + bcBuffXy[1], kind: 'battlecry' };

  var buffAllXy = m.match(/^buff_all_(\d+)_(\d+)$/);
  if (buffAllXy) return { label: 'Усиление всех', value: '+' + buffAllXy[1], description: 'Усиливает всех союзников на +' + buffAllXy[1], kind: 'passive' };

  // ── Single-number patterns ──

  var manaDrain = m.match(/^mana_drain_(\d+)$/);
  if (manaDrain) return { label: 'Кража маны', value: manaDrain[1], description: 'Крадёт ' + manaDrain[1] + ' маны у врага', kind: 'passive' };

  var manaGain = m.match(/^mana_gain_(\d+)$/);
  if (manaGain) return { label: 'Прирост маны', value: '+' + manaGain[1], description: 'Даёт +' + manaGain[1] + ' маны', kind: 'passive' };

  var aura = m.match(/^aura_atk_(\d+)$/);
  if (aura) return { label: 'Аура атаки', value: '+' + aura[1], description: '+' + aura[1] + ' к атаке союзных существ', kind: 'aura' };

  var regen = m.match(/^regen_(\d+)$/);
  if (regen) return { label: 'Регенерация', value: '+' + regen[1], description: '+' + regen[1] + ' здоровья в начале своего хода', kind: 'passive' };

  var armor = m.match(/^armor_(\d+)$/);
  if (armor) return { label: 'Броня', value: armor[1], description: 'Входящий урон –' + armor[1], kind: 'passive' };

  var reflect = m.match(/^reflect_(\d+)$/);
  if (reflect) return { label: 'Отражение', value: reflect[1], description: 'Возвращает ' + reflect[1] + ' урона атакующему', kind: 'passive' };

  var startMana = m.match(/^start_mana_(\d+)$/);
  if (startMana) return { label: 'Стартовая мана', value: '+' + startMana[1], description: '+' + startMana[1] + ' маны в начале боя', kind: 'start' };

  var aoeDmg = m.match(/^aoe_damage_(\d+)$/);
  if (aoeDmg) return { label: 'Массовый урон', value: aoeDmg[1], description: 'Наносит ' + aoeDmg[1] + ' урона всем врагам', kind: 'passive' };

  var heal = m.match(/^heal_(\d+)$/);
  if (heal) return { label: 'Лечение', value: heal[1], description: 'Восстанавливает ' + heal[1] + ' здоровья', kind: 'passive' };

  var healTarget = m.match(/^heal_target_(\d+)$/);
  if (healTarget) return { label: 'Лечение цели', value: healTarget[1], description: 'Лечит цель на ' + healTarget[1] + ' здоровья', kind: 'battlecry' };

  var bcDmg = m.match(/^battlecry_damage_(\d+)$/);
  if (bcDmg) return { label: 'Эффект розыгрыша: урон', value: bcDmg[1], description: 'При розыгрыше наносит ' + bcDmg[1] + ' урона', kind: 'battlecry' };

  var bcHeal = m.match(/^battlecry_heal_(\d+)$/);
  if (bcHeal) return { label: 'Эффект розыгрыша: лечение', value: bcHeal[1], description: 'При розыгрыше лечит на ' + bcHeal[1] + ' здоровья', kind: 'battlecry' };

  var bcBuff = m.match(/^battlecry_buff_(\d+)$/);
  if (bcBuff) return { label: 'Эффект розыгрыша: усиление', value: '+' + bcBuff[1], description: 'При розыгрыше усиливает союзников на +' + bcBuff[1], kind: 'battlecry' };

  var dmg = m.match(/^damage_(\d+)$/);
  if (dmg) return { label: 'Урон', value: dmg[1], description: 'Наносит ' + dmg[1] + ' урона', kind: 'passive' };

  var cleave = m.match(/^cleave_(\d+)$/);
  if (cleave) return { label: 'Разрубание', value: cleave[1], description: 'Наносит ' + cleave[1] + ' урона соседним врагам', kind: 'passive' };

  // ── Simple exact-match (no numeric suffix) ──

  if (m === 'taunt') return { label: 'Провокация', value: '', description: 'Враг обязан атаковать эту карту', kind: 'status' };
  if (m === 'shield') return { label: 'Щит', value: '', description: 'Блокирует следующий входящий урон', kind: 'status' };
  if (m === 'permanent_shield') return { label: 'Вечный щит', value: '', description: 'Блокирует весь входящий урон', kind: 'status' };
  if (m === 'charge') return { label: 'Рывок', value: '', description: 'Может атаковать в первый ход', kind: 'status' };
  if (m === 'lifesteal') return { label: 'Вампиризм', value: '', description: 'Лечит героя на величину нанесённого урона', kind: 'passive' };
  if (m === 'freeze') return { label: 'Заморозка', value: '', description: 'Пропускает готовность к атаке', kind: 'status' };
  if (m === 'aoe_freeze') return { label: 'Массовая заморозка', value: '', description: 'Замораживает всех врагов при розыгрыше', kind: 'battlecry' };
  if (m === 'instant_kill') return { label: 'Мгновенное убийство', value: '', description: 'Уничтожает цель независимо от здоровья', kind: 'passive' };
  if (m === 'bypass_taunt') return { label: 'Обход провокации', value: '', description: 'Может игнорировать провокацию', kind: 'passive' };
  if (m === 'consume_ally') return { label: 'Поглощение союзника', value: '', description: 'Уничтожает союзника и получает его статы', kind: 'battlecry' };
  if (m === 'choose_shield_damage') return { label: 'Выбор: щит или урон', value: '', description: 'При розыгрыше: щит союзнику или урон врагу', kind: 'battlecry' };
  if (m === 'cast_random_spell') return { label: 'Уникальное заклинание', value: '', description: 'При розыгрыше выбирает один из эффектов: Техасский удар, Восстановление, Чёрный кнут или Полный покров', kind: 'battlecry' };
  if (m === 'delete_target') return { label: 'Удаление цели', value: '', description: 'Уничтожает выбранную цель', kind: 'passive' };
  if (m === 'battlecry_freeze') return { label: 'Эффект розыгрыша: заморозка', value: '', description: 'При розыгрыше замораживает цель', kind: 'battlecry' };
  if (m === 'battlecry_draw') return { label: 'Эффект розыгрыша: добор', value: '', description: 'При розыгрыше берёт карту из колоды', kind: 'battlecry' };
  if (m === 'cleave') return { label: 'Разрубание', value: '', description: 'Наносит урон соседним врагам', kind: 'passive' };
  if (m === 'deathrattle') return { label: 'Предсмертный эффект', value: '', description: 'Срабатывает при гибели', kind: 'deathrattle' };

  // ── Generic prefix fallbacks (checked last) ──

  if (m.startsWith('deathrattle_')) return { label: 'Предсмертный эффект', value: '', description: 'Срабатывает при гибели карты', kind: 'deathrattle' };
  if (m.startsWith('battlecry_')) return { label: 'Эффект розыгрыша', value: '', description: 'Срабатывает при розыгрыше карты', kind: 'battlecry' };
  if (m.startsWith('spell_')) return { label: 'Заклинание', value: '', description: 'Эффект заклинания', kind: 'passive' };

  return null;
}

/**
 * Собирает эффекты с одной карты (героя или юнита на поле).
 * @param {Object} unit — карта из state (hero или board)
 * @param {'player'|'opponent'} side
 * @param {'hero'|'board'} zone
 * @param {Object} state — полный state
 * @returns {Array}
 */
function collectUnitEffects(unit, side, zone, state) {
  if (!unit) return [];
  const results = [];
  const mechanics = unit.mechanics || [];
  const sourceName = unit.name || 'Карта';
  const sourceId = unit.instance_id || '';

  mechanics.forEach(function(mechanic) {
    const parsed = parseMechanic(mechanic);
    if (!parsed) return;

    const effect = {
      side: side,
      zone: zone,
      sourceName: sourceName,
      sourceId: sourceId,
      label: parsed.label,
      value: parsed.value,
      description: parsed.description,
      targets: [],
      kind: parsed.kind
    };

    // Collect aura targets
    if (parsed.kind === 'aura') {
      var friendlyBoard = side === 'player'
        ? (state.player && state.player.board || [])
        : (state.opponent && state.opponent.board || []);
      var targets = collectAuraTargets(sourceId, side, friendlyBoard, zone);
      effect.targets = targets;
    }

    results.push(effect);
  });

  // Frozen status (may be separate field, not in mechanics)
  if (unit.is_frozen === true && !mechanics.some(function(m) { return m === 'freeze'; })) {
    results.push({
      side: side,
      zone: zone,
      sourceName: sourceName,
      sourceId: sourceId,
      label: 'Заморозка',
      value: '',
      description: 'Пропускает готовность к атаке',
      targets: [],
      kind: 'status'
    });
  }

  return results;
}

/**
 * Собирает имена целей ауры.
 * @param {string} sourceInstanceId
 * @param {'player'|'opponent'} side
 * @param {Array} friendlyBoard
 * @param {'hero'|'board'} sourceZone
 * @returns {string[]}
 */
function collectAuraTargets(sourceInstanceId, side, friendlyBoard, sourceZone) {
  if (!friendlyBoard || friendlyBoard.length === 0) return [];
  return friendlyBoard
    .filter(function(u) {
      // Board aura doesn't buff self
      if (sourceZone === 'board' && String(u.instance_id) === String(sourceInstanceId)) return false;
      return true;
    })
    .map(function(u) {
      return u.name || 'Существо';
    });
}

/**
 * Собирает все видимые эффекты на поле.
 * @param {Object} state
 * @returns {Array}
 */
function collectVisibleEffects(state) {
  if (!state) return [];
  const all = [];

  // Player hero
  if (state.player && state.player.hero) {
    const heroEffects = collectUnitEffects(state.player.hero, 'player', 'hero', state);
    heroEffects.forEach(function(e) { all.push(e); });
  }

  // Player board
  const playerBoard = (state.player && state.player.board) || [];
  playerBoard.forEach(function(unit) {
    const unitEffects = collectUnitEffects(unit, 'player', 'board', state);
    unitEffects.forEach(function(e) { all.push(e); });
  });

  // Opponent hero (PUBLIC only)
  if (state.opponent && state.opponent.hero) {
    const oppHeroEffects = collectUnitEffects(state.opponent.hero, 'opponent', 'hero', state);
    oppHeroEffects.forEach(function(e) { all.push(e); });
  }

  // Opponent board (PUBLIC only)
  const opponentBoard = (state.opponent && state.opponent.board) || [];
  opponentBoard.forEach(function(unit) {
    const unitEffects = collectUnitEffects(unit, 'opponent', 'board', state);
    unitEffects.forEach(function(e) { all.push(e); });
  });

  return all;
}

/**
 * Собирает эффекты модификаторов боя.
 * @param {Object} modeConfig
 * @returns {{summary: Object, details: Array}|null}
 */
function collectModeEffects(modeConfig) {
  if (!modeConfig) return null;

  const classic = modeConfig.classic || {};
  const rewards = modeConfig.rewards || {};
  const modeId = modeConfig.mode_id || getModeId(currentState);
  const title = modeConfig.label || getModeUiMeta(currentState)?.label || 'Classic';
  const turnDuration = classic.turn_duration_seconds ?? currentState?.turn_duration ?? 25;
  const manaPerTurn = classic.mana_per_turn ?? 1;

  let summary = 'Стандартные правила ExtraArena без дополнительных модификаторов.';
  let status = modeId === 'classic' ? 'Стандартный' : 'Активный';
  if (classic.spells_free === true) {
    summary = 'Все заклинания стоят 0 маны.';
  } else if (classic.summon_ready_on_play === true) {
    summary = 'Существа готовы атаковать сразу после выхода на доску.';
  } else if (classic.sudden_death_enabled === true) {
    const start = classic.sudden_death_damage_start || 1;
    const step = classic.sudden_death_damage_step || 1;
    summary = 'Герои теряют здоровье каждый ход: старт ' + start + ', затем +' + step + '.';
  } else if (classic.card_level_mode === 'max') {
    summary = 'Все карты играют на максимальном уровне.';
  } else if (modeId === 'extra_arena:blitz') {
    summary = 'Короткие ходы, ускоренная мана и меньше здоровья у героев.';
  } else if (modeId === 'training') {
    summary = 'Тренировочный бой без рейтинговых наград.';
  } else if (modeId === 'friendly') {
    summary = 'Дружеский бой без изменения рейтинга.';
  }

  let levelValue = 'Стандартный';
  let levelDescription = 'Берётся текущий уровень из коллекции.';
  if (classic.card_level_mode === 'max') {
    levelValue = 'Максимальный';
    levelDescription = 'Все карты считаются максимального уровня.';
  } else if (classic.card_level_mode === 'disabled') {
    levelValue = 'Первый';
    levelDescription = 'Уровни карт в этом режиме не учитываются.';
  }

  const rewardsEnabled = rewards.enabled !== false;
  const rewardsDescription = rewardsEnabled
    ? 'Трофеи, монеты, звёзды и победы.'
    : 'Матч без изменения рейтинга и наград.';

  return {
    summary: {
      title,
      description: summary,
      status
    },
    details: [
      {
        title: 'Длина хода',
        value: turnDuration + 'с',
        description: 'Время на решение до автозавершения.'
      },
      {
        title: 'Мана за ход',
        value: '+' + manaPerTurn,
        description: 'В начале вашего хода.'
      },
      {
        title: 'Уровень карт',
        value: levelValue,
        description: levelDescription
      },
      {
        title: 'Награды',
        value: rewardsEnabled ? 'Активны' : 'Отключены',
        description: rewardsDescription
      }
    ]
  };
}

/**
 * Рендерит содержимое модалки эффектов.
 * @param {Array} effects
 * @param {Array} modeEffects
 */
function renderEffectsModal(effects, modeEffects) {
  var fieldPanel = document.getElementById('effects-field-panel');
  var modePanel = document.getElementById('effects-mode-panel');
  if (!fieldPanel || !modePanel) return;

  fieldPanel.innerHTML = '';
  modePanel.innerHTML = '';

  effects = effects || [];
  modeEffects = modeEffects || null;

  var playerHero = effects.filter(function(e) { return e.side === 'player' && e.zone === 'hero'; });
  var playerBoard = effects.filter(function(e) { return e.side === 'player' && e.zone === 'board'; });
  var opponentHero = effects.filter(function(e) { return e.side === 'opponent' && e.zone === 'hero'; });
  var opponentBoard = effects.filter(function(e) { return e.side === 'opponent' && e.zone === 'board'; });

  addEffectsSection(fieldPanel, 'Ваш герой', playerHero, 'player');
  addEffectsSection(fieldPanel, 'Ваше поле', playerBoard, 'player');
  addEffectsSection(fieldPanel, 'Герой и поле соперника', opponentHero.concat(opponentBoard), 'opponent');
  renderModeEffectsPanel(modePanel, modeEffects);

  if (!fieldPanel.children.length) {
    var fieldEmpty = document.createElement('div');
    fieldEmpty.className = 'effects-empty';
    fieldEmpty.textContent = 'На поле нет активных эффектов';
    fieldPanel.appendChild(fieldEmpty);
  }

  if (!modePanel.children.length) {
    var modeEmpty = document.createElement('div');
    modeEmpty.className = 'effects-empty';
    modeEmpty.textContent = 'Модификаторы режима не активны';
    modePanel.appendChild(modeEmpty);
  }
}

function renderModeEffectsPanel(container, modeEffects) {
  if (!container || !modeEffects) return;

  const summary = modeEffects.summary || {};
  const details = modeEffects.details || [];
  if (!summary.title && details.length === 0) return;

  const hero = document.createElement('article');
  hero.className = 'mode-summary-card';

  const heroText = document.createElement('div');
  const heroTitle = document.createElement('h2');
  heroTitle.textContent = summary.title || 'Режим игры';
  const heroDesc = document.createElement('p');
  heroDesc.textContent = summary.description || 'Особые правила этого боя.';
  heroText.appendChild(heroTitle);
  heroText.appendChild(heroDesc);

  const status = document.createElement('strong');
  status.textContent = summary.status || 'Активный';

  hero.appendChild(heroText);
  hero.appendChild(status);
  container.appendChild(hero);

  const grid = document.createElement('div');
  grid.className = 'mode-info-grid';
  details.forEach(function(item) {
    grid.appendChild(createModeInfoCard(item));
  });
  container.appendChild(grid);
}

function createModeInfoCard(item) {
  const card = document.createElement('article');
  card.className = 'mode-info-card';

  const title = document.createElement('h2');
  title.textContent = item.title || 'Параметр';

  const value = document.createElement('strong');
  value.textContent = item.value || '—';

  const desc = document.createElement('p');
  desc.textContent = item.description || '';

  card.appendChild(title);
  card.appendChild(value);
  card.appendChild(desc);
  return card;
}

function addEffectsSection(container, title, list, sideClass) {
  if (!container || !list || list.length === 0) return;

  var kicker = document.createElement('p');
  kicker.className = 'section-kicker';
  kicker.textContent = title;
  container.appendChild(kicker);

  var effectList = document.createElement('div');
  effectList.className = 'effect-list';
  list.forEach(function(effect) {
    effectList.appendChild(createEffectCard(effect, sideClass));
  });
  container.appendChild(effectList);
}

function createEffectCard(effect, sideClass) {
  var card = document.createElement('article');
  card.className = 'effect-card ' + (sideClass || effect.side || 'system');

  var textWrap = document.createElement('div');
  var title = document.createElement('h2');
  title.className = 'effect-title';
  var source = effect.sourceName || effect.label || 'Эффект';
  title.appendChild(document.createTextNode(source));
  if (effect.kind) {
    var badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = formatEffectKind(effect.kind);
    title.appendChild(badge);
  }

  var desc = document.createElement('p');
  desc.className = 'effect-desc';
  desc.textContent = [
    effect.label && effect.label !== source ? effect.label : '',
    effect.description || ''
  ].filter(Boolean).join(': ') || 'Активный эффект';

  textWrap.appendChild(title);
  textWrap.appendChild(desc);

  if (effect.targets && effect.targets.length > 0) {
    var targets = document.createElement('div');
    targets.className = 'effect-targets';
    targets.textContent = 'Затрагивает: ' + effect.targets.join(', ');
    textWrap.appendChild(targets);
  }

  var value = document.createElement('span');
  value.className = 'value-chip';
  value.textContent = effect.value !== undefined && effect.value !== null && effect.value !== ''
    ? effect.value
    : formatEffectKind(effect.kind || 'active');

  card.appendChild(textWrap);
  card.appendChild(value);
  return card;
}

function formatEffectKind(kind) {
  const labels = {
    battlecry: 'Эффект розыгрыша',
    passive: 'Пассивная способность',
    start: 'Стартовый бонус',
    aura: 'Аура',
    status: 'Статус',
    deathrattle: 'После гибели',
    mode: 'Режим',
    active: 'Активно'
  };
  return labels[kind] || String(kind || 'Активно');
}

/**
 * Экранирует HTML-символы.
 */
function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Открывает модалку активных эффектов.
 */
function openEffectsModal() {
  var effects = collectVisibleEffects(currentState);
  var modeEffects = collectModeEffects(currentState && currentState.mode_config);
  renderEffectsModal(effects, modeEffects);
  selectEffectsTab('field');
  openBattleModal('effects');
}

/**
 * Закрывает модалку активных эффектов.
 */
function closeEffectsModal() {
  closeBattleModal();
}

function openTurnTimerModal() {
  renderTurnTimerModal(currentState || {});
  openBattleModal('timer');
}

function renderTurnTimerModal(state) {
  const turnDuration = Number(state.turn_duration || getClassicModeParams(state).turn_duration_seconds || 25);
  const remaining = Math.max(0, Number(state.turn_time_remaining ?? turnDuration));
  const elapsed = Math.max(0, turnDuration - remaining);
  const progress = turnDuration > 0 ? Math.min(360, Math.max(0, (elapsed / turnDuration) * 360)) : 0;
  const isMyTurn = !!state.is_my_turn;

  setText('turn-timer-modal-meta', 'Ход ' + (state.turn || currentTurnCount || 0));
  setText('turn-timer-modal-remaining', String(Math.ceil(remaining)));
  setText('turn-timer-modal-owner', isMyTurn ? 'Ваш ход' : 'Ход противника');
  setText(
    'turn-timer-modal-summary',
    'Всего на ход: ' + Math.ceil(turnDuration) + ' сек. Прошло: ' + Math.floor(elapsed) + ' сек.'
  );

  const ring = document.getElementById('turn-timer-ring');
  if (ring) ring.style.setProperty('--timer-progress', progress + 'deg');
  renderTurnTimerHistory(state.turn_time_history || []);
}

function renderTurnTimerHistory(history) {
  const container = document.getElementById('turn-timer-history');
  if (!container) return;
  container.innerHTML = '';

  const rows = Array.isArray(history) ? history.slice(-8).reverse() : [];
  if (rows.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'timer-history-empty';
    empty.textContent = 'История времени появится после завершения первых ходов.';
    container.appendChild(empty);
    return;
  }

  rows.forEach(function(item) {
    const side = item.side === 'opponent' ? 'opponent' : 'player';
    const row = document.createElement('div');
    row.className = 'timer-history-row ' + side;

    const text = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'timer-history-title';
    title.textContent = 'Ход ' + (item.turn || '—');
    const sideEl = document.createElement('span');
    sideEl.className = 'timer-history-side';
    sideEl.textContent = side === 'player' ? 'Игрок' : 'Оппонент';
    text.appendChild(title);
    text.appendChild(sideEl);

    const value = document.createElement('strong');
    value.className = 'timer-history-value';
    value.textContent = formatTurnSeconds(item.elapsed_seconds);

    row.appendChild(text);
    row.appendChild(value);
    container.appendChild(row);
  });
}

function formatTurnSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '—';
  const rounded = Math.max(0, Math.round(seconds));
  return rounded + ' сек';
}

function openBattleModal(name) {
  const layer = document.getElementById('battle-modal-layer');
  if (!layer) return;

  activeBattleModal = name;
  layer.setAttribute('aria-hidden', 'false');
  layer.classList.add('open');
  const modalIds = {
    logs: 'battle-log-overlay',
    effects: 'effects-overlay-backdrop',
    'card-info': 'card-info-modal',
    timer: 'turn-timer-modal'
  };
  layer.querySelectorAll('.battle-modal').forEach(function(modal) {
    const isActive = modal.id === modalIds[name];
    modal.classList.toggle('is-active', isActive);
    modal.setAttribute('aria-hidden', isActive ? 'false' : 'true');
  });
}

function closeBattleModal() {
  const layer = document.getElementById('battle-modal-layer');
  activeBattleModal = null;
  if (layer) {
    layer.setAttribute('aria-hidden', 'true');
    layer.classList.remove('open');
    layer.querySelectorAll('.battle-modal').forEach(function(modal) {
      modal.classList.remove('is-active');
      modal.setAttribute('aria-hidden', 'true');
    });
  }
}

function selectEffectsTab(tabName) {
  const tabs = document.querySelectorAll('.battle-modal-tab[data-tab]');
  tabs.forEach(tab => {
    const selected = tab.dataset.tab === tabName;
    tab.setAttribute('aria-selected', selected ? 'true' : 'false');
  });

  const fieldPanel = document.getElementById('effects-field-panel');
  const modePanel = document.getElementById('effects-mode-panel');
  if (fieldPanel) fieldPanel.classList.toggle('is-active', tabName === 'field');
  if (modePanel) modePanel.classList.toggle('is-active', tabName === 'mode');
}

function bindScrollSafeClose(modal) {
  if (!modal) return;
  let scrollStarted = false;
  let startX = 0;
  let startY = 0;

  modal.querySelectorAll('.battle-modal-scroll').forEach(scroller => {
    scroller.addEventListener('pointerdown', function(e) {
      scrollStarted = false;
      startX = e.clientX;
      startY = e.clientY;
    });
    scroller.addEventListener('pointermove', function(e) {
      if (Math.abs(e.clientX - startX) > 6 || Math.abs(e.clientY - startY) > 6) {
        scrollStarted = true;
      }
    });
    scroller.addEventListener('click', function(e) {
      if (scrollStarted) {
        e.stopPropagation();
        scrollStarted = false;
      }
    });
  });
}

// ============================================
// ОБРАБОТЧИКИ UI
// ============================================

function bindUIHandlers() {
  // Opponent info button
  const opponentInfoBtn = document.getElementById('opponent-info-btn');
  if (opponentInfoBtn) {
    opponentInfoBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      openOpponentInfo();
    });
  }

  // Surrender hold-to-activate
  initSurrenderHold();

  // Глобальный слушатель для закрытия окон
  document.addEventListener('click', (e) => {
    const tooltip = document.getElementById('card-description-tooltip');

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
      endTurnBtn.classList.remove('pulse-ready');
      endTurn();
    });
  }

  // Кнопка лога боя
  const logBtn = document.getElementById('battle-log-btn');
  const logOverlay = document.getElementById('battle-log-overlay');
  const battleModalLayer = document.getElementById('battle-modal-layer');
  if (logBtn && logOverlay) {
    logBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openBattleModal('logs');
    });
  }

  // Кнопка активных эффектов
  const effectsBtn = document.getElementById('effects-btn');
  const effectsBackdrop = document.getElementById('effects-overlay-backdrop');
  if (effectsBtn && effectsBackdrop) {
    effectsBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      openEffectsModal();
    });
  }

  const timerBtn = document.getElementById('turn-timer-container');
  if (timerBtn) {
    timerBtn.setAttribute('role', 'button');
    timerBtn.setAttribute('tabindex', '0');
    timerBtn.setAttribute('title', 'Статистика времени ходов');
    timerBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      openTurnTimerModal();
    });
    timerBtn.addEventListener('keydown', function(e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      openTurnTimerModal();
    });
  }

  if (battleModalLayer) {
    battleModalLayer.addEventListener('click', function() {
      if (activeBattleModal) closeBattleModal();
    });
  }

  [
    logOverlay,
    effectsBackdrop,
    document.getElementById('card-info-modal'),
    document.getElementById('turn-timer-modal')
  ].forEach(function(modal) {
    if (!modal) return;
    bindScrollSafeClose(modal);
    modal.addEventListener('click', function(e) {
      if (e.target.closest('.battle-modal-tabs')) return;
      closeBattleModal();
    });
  });

  const effectsTabs = document.getElementById('effects-tabs');
  if (effectsTabs) {
    effectsTabs.addEventListener('click', function(e) {
      const tab = e.target.closest('.battle-modal-tab[data-tab]');
      if (!tab) return;
      e.stopPropagation();
      selectEffectsTab(tab.dataset.tab);
    });
  }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && activeBattleModal) {
      closeBattleModal();
    }
  });
  
  // Модальное окно сдачи
  const surrenderModal = document.getElementById('surrender-modal');
  if (surrenderModal) {
    const cancelBtn = surrenderModal.querySelector('[data-action="cancel"]');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => {
        closeSurrenderModal();
      });
    }
    
    const confirmBtn = surrenderModal.querySelector('[data-action="confirm"]');
    if (confirmBtn) {
      confirmBtn.addEventListener('click', () => {
        closeSurrenderModal();
        surrender();
      });
    }
    
    const overlay = surrenderModal.querySelector('.surrender-modal-overlay');
    if (overlay) {
      overlay.addEventListener('click', () => {
        closeSurrenderModal();
      });
    }
  }
  
  // Клик по герою оппонента
  const opponentPanel = document.querySelector('.opponent-panel-root');
  if (opponentPanel) {
    opponentPanel.addEventListener('click', (e) => {
      e.stopPropagation();
      
      if (interactionMode.type === 'ATTACK' || interactionMode.type === 'TARGETING') {
        handleGlobalTargetClick(null, true, e);
      }
    });
  }
  
  // Клик по своему герою (для хила)
  const playerPanel = document.querySelector('.player-panel-root');
  if (playerPanel) {
    playerPanel.addEventListener('click', (e) => {
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
      window.location.replace('/');
    });
  }
  
  const resultOverlay = document.querySelector('.result-overlay-bg');
  if (resultOverlay) {
    resultOverlay.addEventListener('click', () => {
      console.log('[ARENA] Возврат в главное меню (клик на overlay)');
      window.location.replace('/');
    });
  }
}

function openSurrenderModal() {
  const modal = document.getElementById('surrender-modal');
  const lossValue = document.getElementById('surrender-loss-trophies');
  arenaHaptic('warning', { key: 'surrender-open', minInterval: 350 });
  if (lossValue) {
    const delta = getEstimatedSurrenderTrophyDelta();
    lossValue.textContent = delta === null ? 'после боя' : String(delta);
    lossValue.classList.toggle('is-neutral', delta === 0);
  }
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

window.ExtraArenaAppBack = function() {
  const resultModal = document.getElementById('battle-result-modal');
  if (resultModal && resultModal.getAttribute('aria-hidden') !== 'true' && resultModal.style.display !== 'none') {
    window.location.replace('/');
    return true;
  }

  const surrenderModal = document.getElementById('surrender-modal');
  if (surrenderModal && surrenderModal.getAttribute('aria-hidden') !== 'true' && surrenderModal.style.display !== 'none') {
    closeSurrenderModal();
    return true;
  }

  openSurrenderModal();
  return true;
};

function getEstimatedSurrenderTrophyDelta() {
  const rewards = currentState?.mode_config?.rewards;
  if (rewards && (rewards.enabled === false || rewards.trophies === false)) return 0;
  const modeId = getModeId(currentState);
  if (modeId === 'training' || modeId === 'friendly') return 0;

  const trophyValue = currentState?.player?.trophies
    ?? currentState?.player_trophies
    ?? currentState?.trophies
    ?? currentState?.trophy_total;
  const trophies = parseInt(trophyValue, 10);
  if (!Number.isFinite(trophies)) return null;

  const tier = SURRENDER_TROPHY_TIERS.find(item => trophies >= item.min && trophies <= item.max)
    || SURRENDER_TROPHY_TIERS[SURRENDER_TROPHY_TIERS.length - 1];
  return tier.penalty;
}
