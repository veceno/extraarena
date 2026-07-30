import json
import subprocess
from pathlib import Path


def test_arena_lifts_sensitive_auth_into_same_origin_api_headers_only():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function installArenaJwtQueryAuthHeaderBridge()" in source
    assert "function looksLikeArenaJwtBearer" in source
    assert "url.origin !== window.location.origin" in source
    assert "!url.pathname.startsWith('/api/')" in source
    assert "url.searchParams.delete('_auth')" in source
    assert "headers.set('Authorization', `Bearer ${bearerToken}`)" in source
    assert "liftArenaAuthFromJsonBody" in source
    assert "headers.set('X-Telegram-Init-Data', telegramToken)" in source
    assert "Sensitive auth never stays in API URLs" in source
    build_url_block = source.split("function buildArenaAuthUrl(path)", 1)[1].split(
        "function looksLikeArenaJwtBearer",
        1,
    )[0]
    assert "'_auth='" not in build_url_block


def test_arena_missing_launch_params_show_empty_state_without_alert():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")
    missing_match_block = source.split("if (isInvalidArenaMatchId(matchId))", 1)[1].split("if (!authToken)", 1)[0]
    load_block = source.split("async function loadBattleState()", 1)[1].split(
        "// ============================================\n// РЕНДЕРИНГ СОСТОЯНИЯ",
        1,
    )[0]

    assert 'id="arena-launch-error"' in markup
    assert 'id="arena-launch-error-back"' in markup
    assert ".arena-launch-error.is-visible" in styles
    assert "#arena-battlefield-container.is-launch-blocked" in styles
    assert "function showArenaLaunchError(title, message)" in source
    assert "function normalizeArenaMatchId(value)" in source
    assert "function isInvalidArenaMatchId(value)" in source
    assert "['null', 'undefined', 'none', 'nan'].includes(normalized)" in source
    assert "showArenaLaunchError(" in missing_match_block
    assert "loadBattleState().then((loaded) =>" in source
    assert "if (loaded && !arenaLaunchBlocked) initSocketIO();" in source
    assert "return false;" in load_block
    assert "return true;" in load_block
    assert "shell.classList.add('is-launch-blocked')" in source
    assert "alert('Ошибка: параметры боя не найдены')" not in source
    assert 'if (arenaLaunchBlocked)' in source


def test_arena_blocks_external_browser_launch_without_blocking_android_shell():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    native = Path("android-app/app/src/main/java/ru/extraarena/app/MainActivity.java").read_text(encoding="utf-8")
    launch_block = source.split("if (isInvalidArenaMatchId(matchId))", 1)[1].split(
        "// Сначала грузим состояние",
        1,
    )[0]
    unsupported_block = source.split("function isUnsupportedExternalArenaBrowser", 1)[1].split(
        "function showArenaLaunchError",
        1,
    )[0]

    assert "function isArenaTelegramRuntime(tg)" in source
    assert "function isUnsupportedExternalArenaBrowser(urlParams, tg)" in source
    assert "if (isUnsupportedExternalArenaBrowser(urlParams, tg))" in launch_block
    assert "'Браузер не поддерживается'" in launch_block
    assert "Играть в арену можно только внутри Telegram или Android-клиента" in launch_block
    assert "if (!authToken)" in launch_block
    assert launch_block.index("if (isUnsupportedExternalArenaBrowser(urlParams, tg))") < launch_block.index("if (!authToken)")

    assert "if (isArenaAndroidShell()) return false;" in unsupported_block
    assert "urlParams?.get('ea_platform') === 'android_app'" in unsupported_block
    assert "isArenaTelegramRuntime(tg)" in unsupported_block
    assert "typeof tg.initData === 'string' && tg.initData.length > 0" in source
    assert "tg.initDataUnsafe?.user" in source
    assert '.appendQueryParameter("ea_platform", "android_app")' in native
    assert '.appendQueryParameter("ea_shell", "android")' in native
    assert '.appendQueryParameter("ea_telegram", "0")' in native


def test_in_game_extrapass_gifts_are_hidden_for_beta():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    shop_block = source.split("const IN_GAME_PASS_GIFTS_BETA_ENABLED = false;", 1)[1].split(
        "{/* ═══════ Section: Кейсы ═══════ */}",
        1,
    )[0]
    pass_modal_block = source.split("{/* ═══ ExtraPass Modal ═══ */}", 1)[1].split(
        "const openPaymentModal",
        1,
    )[0]

    assert "const IN_GAME_PASS_GIFTS_BETA_ENABLED = false;" in source
    assert "IN_GAME_PASS_GIFTS_BETA_ENABLED &&" in shop_block
    assert "Подарок участникам сквада" not in source
    assert "openPaymentModal('extrapass_gift'" not in source
    assert "item_type:'extrapass_gift'" not in source
    assert "Оплатить подарок" not in pass_modal_block


def test_arena_battle_action_fetches_use_auth_url_not_auth_body_payloads():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    active_blocks = [
        source.split("async function loadBattleState()", 1)[1].split("// ============================================\n// РЕНДЕРИНГ СОСТОЯНИЯ", 1)[0],
        source.split("async function playCard", 1)[1].split("async function playPotionCard", 1)[0],
        source.split("async function playPotionCard", 1)[1].split("function triggerPotionDamageFlash", 1)[0],
        source.split("async function attack", 1)[1].split("async function endTurn", 1)[0],
        source.split("async function endTurn", 1)[1].split("async function surrender", 1)[0],
    ]
    active_source = "\n".join(active_blocks)

    assert "fetch(buildArenaAuthUrl(`/api/battle/state?match_id=${encodeURIComponent(matchId)}`)" in active_source
    assert "fetch(buildArenaAuthUrl('/api/battle/play-card')," in active_source
    assert "fetch(buildArenaAuthUrl('/api/battle/attack')," in active_source
    assert "fetch(buildArenaAuthUrl('/api/battle/end-turn')," in active_source
    assert "_auth: authToken" not in active_source


def test_ready_friendly_unit_click_does_not_steal_targeting_mode():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    attack_listener = source.split("// Если это карта игрока, разрешаем атаку", 1)[1].split(
        "// ИСПРАВЛЕНО: Союзные юниты могут быть целью для хила/баффов",
        1,
    )[0]

    assert "interactionMode.type === 'TARGETING'" in attack_listener
    assert "return;" in attack_listener


def test_status_icons_use_non_overlapping_card_slots():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")
    status_block = source.split("function addStatusIcons", 1)[1].split("// ============================================\n// РЕНДЕРИНГ ПОЛЕЙ", 1)[0]

    assert "const layer = document.createElement('div')" in status_block
    assert "layer.className = 'card-status-layer'" in status_block
    assert "status-icon-shield',  'shield.png',      'icon-top-left'" in status_block
    assert "status-icon-taunt',   'provocation.png', 'icon-side-right'" in status_block
    assert "status-icon-frozen', 'freeze.png', 'icon-top-right'" in status_block
    assert "status-icon-asleep', 'asleep.png', 'icon-top-center'" in status_block
    assert "icon-bottom-right" not in status_block
    assert "card-name-label" not in status_block
    assert "unit-card-stats" not in status_block
    assert "hand-card-stats" not in status_block
    assert "card-info-btn" not in status_block

    assert ".card-status-layer" in styles
    assert ".status-icon-container.icon-top-center" in styles
    assert ".status-icon-container.icon-side-right" in styles
    assert ".status-icon-taunt .status-icon" in styles


def test_arena_status_icons_use_single_overlay_system():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")

    assert "status-overlay-icon" not in source
    assert "status-overlay-icon" not in styles
    assert ".so-" not in styles
    assert "soAttack" not in source


def test_arena_does_not_render_battlecry_bell_icon():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")

    assert "battlecry-icon" not in source
    assert "🔔" not in source
    assert ".battlecry-icon" not in styles


def test_card_info_renders_mechanic_descriptions_not_raw_codes():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    info_block = source.split("function openCardInfo(card)", 1)[1].split("function closeCardInfo()", 1)[0]
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")

    assert "card.mechanics_desc" in info_block
    assert "parseMechanic(m)" in info_block
    assert "mechanic-detail-description" in info_block
    assert "mechanic-detail-img" in info_block
    assert "chip.textContent = m" not in info_block
    assert '<article class="battle-modal card-info-modal" id="card-info-modal"' in markup
    assert 'card-info-modal-card' not in markup
    assert "hasMechanicDetails" in info_block
    assert "modal.classList.toggle('no-mechanics'" in info_block
    assert ".card-info-modal.no-mechanics" in styles
    assert ".card-info-modal.no-mechanics .section-kicker" in styles


def test_dynamic_mechanic_copy_keeps_scaling_values_readable():
    arena = Path("webapp/arena.js").read_text(encoding="utf-8")
    index = Path("webapp/index.html").read_text(encoding="utf-8")

    for source in (arena, index):
        assert "Заходит в бой с капиталом: +'" in source
        assert "' маны уже на старте'" in source
        assert "Пробивает выбранную цель на '" in source
        assert "' урона'" in source
        assert "Врывается в бой и бьёт случайного врага на '" in source
        assert "Проводит волну по врагам: '" in source
        assert "Уходит громко: при гибели наносит '" in source

    assert "startMana[1]" in arena
    assert "bcRandomDmg[1]" in arena
    assert "damageXy[1]" in arena
    assert "aoeDmg[1]" in arena
    assert "drAoe[1]" in arena
    assert "mm[1]" in index


def test_sudden_death_badges_show_current_turn_damage_not_next_turn_preview():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    badge_block = source.split("function renderSuddenDeathBadges(state)", 1)[1].split(
        "function updateSuddenDeathBadge",
        1,
    )[0]
    update_block = source.split("function updateSuddenDeathBadge", 1)[1].split(
        "function updatePlayerInfo",
        1,
    )[0]

    assert "player_turn_damage" in badge_block
    assert "opponent_turn_damage" in badge_block
    assert "player_next_damage" not in badge_block
    assert "opponent_next_damage" not in badge_block
    assert "Урон SuddenDeath на этом ходе" in update_block


def test_arena_extra_pass_visual_treats_ultra_as_pass():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    player_block = source.split("// ExtraPass", 1)[1].split("// Mana", 1)[0]
    opponent_block = source.split("const opponentInfoIsland", 1)[1].split(
        "// Restore targeting highlight",
        1,
    )[0]

    assert "function isExtraPassVisualMode(mode)" in source
    assert "function getPremiumNicknameTier(mode)" in source
    assert "if (mode === 'ultra') return 'ultra';" in source
    assert "if (mode === 'active' || mode === 'pass' || mode === 'extra_pass') return 'pass';" in source
    assert "isExtraPassVisualMode(currentState?.extra_pass)" in player_block
    assert "isExtraPassVisualMode(playerState?.extra_pass)" in player_block
    assert "isExtraPassVisualMode(opponentState?.extra_pass)" in opponent_block


def test_arena_card_sfx_resolver_contract():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    config_block = source.split("const CARD_SFX_CONFIG_DEFAULT =", 1)[1].split(
        "let cardSfxConfig",
        1,
    )[0]
    state_sfx_block = source.split("function processArenaStateSfx", 1)[1].split(
        "/**\n * Получает возможные цели",
        1,
    )[0]

    assert "const CARD_SFX_CONFIG_URL = '/DesignAssets/Sounds/arena/card_sfx_config.json';" in source
    assert "'18':" in config_block
    assert "'27':" in config_block
    assert "'29':" in config_block
    assert "'34':" in config_block
    assert "name: 'П.Е.К.К.А.'" in config_block
    assert "name: 'Скелет'" in config_block
    assert "name: 'Штурмовик'" in config_block
    assert "name: 'Крипер'" in config_block
    assert "pekka_deploy.mp3" in config_block
    assert "skeleton_death.mp3" in config_block
    assert "stormtrooper_e11_blaster.mp3" in config_block
    assert "creeper_spawn_hiss.mp3" in config_block
    assert "creeper_death_explosion.mp3" in config_block
    assert "basePolicy: 'replace'" in config_block

    for function_name in (
        "playArenaUrlSfx",
        "resolveArenaCardSfx",
        "playResolvedCardSfx",
        "playResolvedCardFeedback",
        "playArenaDeathCardSfx",
        "shouldSuppressHeroHpSfxForCardDeath",
        "processArenaSoundEvents",
    ):
        assert f"function {function_name}" in source

    assert "const suppressHeroHpSfx = shouldSuppressHeroHpSfxForCardDeath(prev, next);" in state_sfx_block
    assert "playArenaDeathCardSfx(oldUnit, 'cardDeath')" in state_sfx_block
    assert "const eventName = newUnit.hp <= 0 ? 'death' : 'damage';" in state_sfx_block
    assert "const fallbackKey = newUnit.hp <= 0 ? 'cardDeath' : 'cardAttacked';" in state_sfx_block
    assert "if (eventName === 'death') playArenaDeathCardSfx(newUnit, fallbackKey);" in state_sfx_block
    assert "else playResolvedCardFeedback(eventName, newUnit, fallbackKey);" in state_sfx_block
    assert "playResolvedCardFeedback('mechanic', newUnit, 'cardHeal', { mechanic: 'heal' })" in state_sfx_block
    assert "playResolvedCardFeedback('mechanic', newUnit, 'cardFrozen', { mechanic: 'freeze' })" in state_sfx_block
    assert "playResolvedCardFeedback('deploy', newUnit, null)" in state_sfx_block

    assert "const playedArenaSfxEventIds = new Set()" in source
    assert "function rememberArenaSfxEventId" in source
    assert "function shouldSkipRecentArenaExplicitSfx" in source
    assert "if (event.event_id != null && playedArenaSfxEventIds.has(String(event.event_id))) return;" in source
    assert "rememberArenaSfxEventId(event.event_id);" in source


def test_arena_card_sfx_config_file_and_assets_exist():
    config_path = Path("assets/audio/characters/card_sfx_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    expected_events = {
        "18": {"deploy", "attack", "damage"},
        "27": {"deploy", "damage", "death"},
        "29": {"attack", "death"},
        "34": {"deploy", "death", "mechanic:deathrattle_aoe_damage_2"},
    }

    for card_id, events in expected_events.items():
        sounds = config["cards"][card_id]["sounds"]
        assert events <= set(sounds)
        for sound in sounds.values():
            src = sound["src"]
            assert src.startswith("/DesignAssets/Sounds/arena/characters/")
            assert sound["basePolicy"] == "replace"
            assert Path(src.lstrip("/")).exists()


def test_arena_card_feedback_config_matches_default_fallback():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    config = json.loads(Path("assets/audio/characters/card_sfx_config.json").read_text(encoding="utf-8"))
    script = r"""
const fs = require('fs');
const source = fs.readFileSync('webapp/arena.js', 'utf8');
const match = source.match(/const CARD_SFX_CONFIG_DEFAULT = (\{[\s\S]*?\n\});\nlet cardSfxConfig/);
if (!match) throw new Error('CARD_SFX_CONFIG_DEFAULT block not found');
process.stdout.write(JSON.stringify(Function(`return (${match[1]});`)()));
"""
    fallback = json.loads(
        subprocess.check_output(["node", "-e", script], text=True, cwd=Path.cwd())
    )

    assert config == fallback


def test_arena_card_feedback_config_loader_merges_and_validates_external_config():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    for function_name in (
        "isPlainArenaFeedbackObject",
        "isValidArenaCardSoundConfig",
        "sanitizeArenaCardFeedbackConfig",
        "mergeArenaCardSfxConfig",
    ):
        assert f"function {function_name}" in source

    loader_block = source.split("function loadArenaCardSfxConfig", 1)[1].split(
        "function playArenaUrlSfx",
        1,
    )[0]
    validation_block = source.split("function isValidArenaCardSoundConfig", 1)[1].split(
        "function isValidArenaCardVisualConfig",
        1,
    )[0]

    assert "cardSfxConfig = mergeArenaCardSfxConfig(config);" in loader_block
    assert "CARD_SFX_CONFIG_DEFAULT" in source
    assert "sound.src.startsWith('/assets/audio/characters/')" in validation_block
    assert "sound.basePolicy !== 'replace'" in validation_block
    assert "volume >= 0 && volume <= 1" in validation_block


def test_arena_card_background_visual_reactions_contract():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")
    config = json.loads(Path("assets/audio/characters/card_sfx_config.json").read_text(encoding="utf-8"))
    config_block = source.split("const CARD_SFX_CONFIG_DEFAULT =", 1)[1].split(
        "let cardSfxConfig",
        1,
    )[0]

    assert 'id="arena-background-reaction-layer"' in markup
    assert ".arena-background-reaction-layer" in styles
    assert "@keyframes arenaBackgroundReactionFlash" in styles
    assert "--arena-bg-reaction-color" in styles
    assert "pointer-events: none" in styles

    for function_name in (
        "resolveArenaCardVisual",
        "playArenaBackgroundReaction",
        "playResolvedCardFeedback",
    ):
        assert f"function {function_name}" in source

    assert "playResolvedCardFeedback(eventName, event, fallbackKey, {" in source
    assert "effect_code: event.effect_code" in source
    assert "playResolvedCardFeedback('deploy', newUnit, null)" in source
    assert "playResolvedCardFeedback('mechanic', newUnit, 'cardHeal', { mechanic: 'heal' })" in source
    assert "playResolvedCardFeedback('mechanic', newUnit, 'cardFrozen', { mechanic: 'freeze' })" in source

    creeper_visuals = config["cards"]["34"]["visuals"]
    for event_name in ("death", "mechanic:deathrattle_aoe_damage_2"):
        visual = creeper_visuals[event_name]
        assert visual["type"] == "backgroundFlash"
        assert visual["color"] == "#ef4444"
        assert visual["durationMs"] == 3600
        assert visual["intensity"] == 0.82

    assert "visuals:" in config_block
    assert "type: 'backgroundFlash'" in config_block
    assert "color: '#ef4444'" in config_block
    assert "durationMs: 3600" in config_block
    assert "intensity: 0.82" in config_block


def test_arena_background_reaction_layer_does_not_override_modal_layers():
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")

    assert "#arena-battlefield-container > :not(.arena-background-reaction-layer)" not in styles
    assert "#arena-battlefield-container > .arena-zone-top" in styles
    assert "#arena-battlefield-container > .arena-zone-center" in styles
    assert "#arena-battlefield-container > .arena-zone-bottom" in styles

    prebattle_block = styles.split("\n.prebattle-screen {", 1)[1].split("}", 1)[0]
    modal_block = styles.split("\n.battle-modal-layer {", 1)[1].split("}", 1)[0]
    assert "position: fixed" in prebattle_block
    assert "z-index: 7000" in prebattle_block
    assert "position: fixed" in modal_block
    assert "z-index: 6500" in modal_block


def test_arena_background_reaction_invalid_colors_fallback_to_safe_rgba():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    color_block = source.split("function arenaHexColorToRgba", 1)[1].split(
        "function playArenaBackgroundReaction",
        1,
    )[0]

    assert "const fallbackColor = `rgba(239,68,68,${normalizedAlpha})`;" in color_block
    assert "if (!color) return fallbackColor;" in color_block
    assert "return fallbackColor;" in color_block
    assert "return color ||" not in color_block


def test_arena_named_card_background_visual_reactions_config():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    config = json.loads(Path("assets/audio/characters/card_sfx_config.json").read_text(encoding="utf-8"))
    config_block = source.split("const CARD_SFX_CONFIG_DEFAULT =", 1)[1].split(
        "let cardSfxConfig",
        1,
    )[0]

    expected_visuals = {
        "18": {"deploy": ("#a78bfa", 3000, 0.32)},
        "19": {
            "deploy": ("#38bdf8", 3300, 0.62),
            "mechanic:battlecry_freeze": ("#0ea5e9", 3300, 0.68),
        },
        "25": {"mechanic:instant_kill": ("#ef4444", 3400, 0.76)},
        "26": {"deploy": ("#d946ef", 3400, 0.6)},
        "27": {"death": ("#e5e7eb", 2400, 0.42)},
        "29": {"attack": ("#ef4444", 1800, 0.48, {"centerColor": "#f8fafc"})},
        "33": {
            "deploy": ("#4ade80", 3200, 0.52),
            "attack": ("#10b981", 3600, 0.58),
            "mechanic:lifesteal": ("#10b981", 3600, 0.58),
        },
        "34": {"deploy": ("#22c55e", 2800, 0.48)},
        "35": {
            "deploy": ("#5eead4", 3200, 0.5),
            "mechanic:battlecry_heal_target_5": ("#86efac", 3200, 0.54),
        },
        "36": {
            "deploy": ("#5eead4", 3200, 0.5),
            "mechanic:battlecry_heal_target_3": ("#86efac", 3200, 0.54),
        },
    }

    assert "reactionConfig.centerColor || color" in source

    for card_id, events in expected_visuals.items():
        assert f"'{card_id}':" in config_block
        for event_name, expectation in events.items():
            color, duration_ms, intensity, *extra = expectation
            visual = config["cards"][card_id]["visuals"][event_name]
            assert visual["type"] == "backgroundFlash"
            assert visual["color"] == color
            assert visual["durationMs"] == duration_ms
            assert visual["intensity"] == intensity
            for key, value in (extra[0] if extra else {}).items():
                assert visual[key] == value
                assert f"{key}: '{value}'" in config_block

            assert f"color: '{color}'" in config_block
            assert f"durationMs: {duration_ms}" in config_block
            assert f"intensity: {intensity}" in config_block


def test_arena_card_feedback_respects_sfx_setting_without_suppressing_visuals():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    play_sfx_block = source.split("function playArenaSfx", 1)[1].split(
        "function isValidArenaCardSfxConfig",
        1,
    )[0]
    play_url_sfx_block = source.split("function playArenaUrlSfx", 1)[1].split(
        "function normalizeArenaSfxToken",
        1,
    )[0]
    play_visual_block = source.split("function playArenaBackgroundReaction", 1)[1].split(
        "function playResolvedCardSfx",
        1,
    )[0]
    resolved_sfx_block = source.split("function playResolvedCardSfx", 1)[1].split(
        "function playResolvedCardFeedback",
        1,
    )[0]
    feedback_block = source.split("function playResolvedCardFeedback", 1)[1].split(
        "function trimArenaSfxEventIdQueue",
        1,
    )[0]

    assert "if (window._sfxEnabled === false) return false;" in play_sfx_block
    assert "if (window._sfxEnabled === false) return false;" in play_url_sfx_block
    assert "return true;" in play_sfx_block
    assert "const playedUrlSfx = resolved?.src" in resolved_sfx_block
    assert "playArenaUrlSfx(resolved.src" in resolved_sfx_block
    assert "if (resolved.basePolicy === 'replace') return playedUrlSfx;" in resolved_sfx_block
    assert "const playedFallbackSfx = playArenaSfx(fallbackKey, options);" in resolved_sfx_block
    assert "return Boolean(playedUrlSfx || playedFallbackSfx);" in resolved_sfx_block

    assert "window._sfxEnabled" not in play_visual_block
    assert "const playedVisual = visual ? playArenaBackgroundReaction(visual, options) : false;" in feedback_block
    assert "const playedText = playResolvedCardText(eventName, cardOrEvent, options);" in feedback_block
    assert "const playedSfx = playResolvedCardSfx(eventName, cardOrEvent, fallbackKey, options);" in feedback_block
    assert "return Boolean(playedVisual || playedText || playedSfx);" in feedback_block


def test_arena_redirect_and_bootstrap_apply_sfx_setting_before_first_feedback():
    arena_source = Path("webapp/arena.js").read_text(encoding="utf-8")
    main_source = Path("webapp/main.js").read_text(encoding="utf-8")
    dom_ready_block = arena_source.split("document.addEventListener('DOMContentLoaded'", 1)[1].split(
        "function normalizeArenaMatchId",
        1,
    )[0]
    redirect_helper = main_source.split("function buildMainArenaRedirectUrl", 1)[1].split(
        "(function installMainJwtQueryAuthHeaderBridge",
        1,
    )[0]

    assert "function appendMainArenaAudioPreferenceParams" in main_source
    assert "url.searchParams.set('sfx', currentSfxEnabled ? '1' : '0')" in main_source
    assert "url.searchParams.set('music', currentMusicEnabled ? '1' : '0')" in main_source
    assert "targetUrl = appendMainArenaAudioPreferenceParams(targetUrl);" in redirect_helper

    assert "await loadTalkieStartupSettings();" in dom_ready_block
    assert dom_ready_block.index("await loadTalkieStartupSettings();") < dom_ready_block.index("initArenaMusic();")
    assert "loadTalkieStartupSettings();" not in arena_source.split("function initTalkies", 1)[1].split(
        "function getModifierLabel",
        1,
    )[0]


def test_arena_card_text_feedback_contract():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")
    config = json.loads(Path("assets/audio/characters/card_sfx_config.json").read_text(encoding="utf-8"))
    config_block = source.split("const CARD_SFX_CONFIG_DEFAULT =", 1)[1].split(
        "let cardSfxConfig",
        1,
    )[0]

    assert 'id="arena-card-text-overlay"' in markup
    assert 'id="arena-card-target-hint"' in markup
    assert ".arena-card-text-overlay" in styles
    assert ".arena-card-target-hint" in styles
    assert "@keyframes arenaCardTextOverlayIn" in styles
    assert "pointer-events: none" in styles

    for function_name in (
        "resolveArenaCardText",
        "playArenaScreenText",
        "showArenaTargetHint",
        "hideArenaTargetHint",
        "playResolvedCardText",
    ):
        assert f"function {function_name}" in source

    screen_text_block = source.split("function playArenaScreenText", 1)[1].split(
        "function showArenaTargetHint",
        1,
    )[0]
    hint_block = source.split("function showArenaTargetHint", 1)[1].split(
        "function hideArenaTargetHint",
        1,
    )[0]
    feedback_block = source.split("function playResolvedCardFeedback", 1)[1].split(
        "function trimArenaSfxEventIdQueue",
        1,
    )[0]

    assert "window._sfxEnabled" not in screen_text_block
    assert "window._sfxEnabled" not in hint_block
    assert "const playedText = playResolvedCardText(eventName, cardOrEvent, options);" in feedback_block
    assert "return Boolean(playedVisual || playedText || playedSfx);" in feedback_block
    assert "showArenaTargetHintForCard(card, index)" in source
    assert "hideArenaTargetHint();" in source
    assert "effect_code" in source

    midoriya_text = config["cards"]["26"]["texts"]["mechanic:cast_random_spell"]
    assert midoriya_text["type"] == "screenText"
    assert midoriya_text["defaultText"] == "Случайная суперспособность"
    assert midoriya_text["durationMs"] == 1600
    assert midoriya_text["detailText"] == {
        "midoriya_texas_smash": "Техасский удар",
        "midoriya_recovery": "Восстановление",
        "midoriya_blackwhip": "Чёрный кнут",
        "midoriya_full_cowl": "Полный покров",
    }

    yuni_text = config["cards"]["36"]["texts"]["targeting:battlecry_heal_target_3"]
    assert yuni_text == {
        "type": "targetHint",
        "text": "Выбери цель для исцеления",
    }

    assert "texts:" in config_block
    assert "detailText:" in config_block
    assert "midoriya_blackwhip: 'Чёрный кнут'" in config_block
    assert "'targeting:battlecry_heal_target_3'" in config_block
    assert "text: 'Выбери цель для исцеления'" in config_block


def test_arena_card_text_feedback_named_cards_config():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    config = json.loads(Path("assets/audio/characters/card_sfx_config.json").read_text(encoding="utf-8"))
    config_block = source.split("const CARD_SFX_CONFIG_DEFAULT =", 1)[1].split(
        "let cardSfxConfig",
        1,
    )[0]

    expected_texts = {
        "8": {
            "targeting:damage_1_5": {
                "type": "targetHint",
                "text": "Выбери цель для глитч-удара",
            },
        },
        "10": {
            "mechanic:aoe_damage_2": {
                "type": "screenText",
                "text": "Импульс Бездны",
                "durationMs": 1500,
            },
        },
        "13": {
            "targeting:delete_target": {
                "type": "targetHint",
                "text": "Выбери врага для Чёрной Дыры",
            },
        },
        "19": {
            "targeting:battlecry_freeze": {
                "type": "targetHint",
                "text": "Выбери врага для ледяного захвата",
            },
        },
        "20": {
            "targeting:consume_ally": {
                "type": "targetHint",
                "text": "Выбери союзника для поглощения",
            },
        },
        "21": {
            "targeting:choose_shield_damage": {
                "type": "targetHint",
                "text": "Выбери цель для ведьмачьего знака",
            },
        },
        "22": {
            "mechanic:aoe_freeze": {
                "type": "screenText",
                "text": "Время остановлено",
                "durationMs": 1500,
            },
        },
        "25": {
            "attack": {
                "type": "screenText",
                "text": "Один удар",
                "durationMs": 1400,
            },
            "attacktargeting:instant_kill": {
                "type": "targetHint",
                "text": "Выбери цель для ваншота",
            },
        },
        "35": {
            "targeting:battlecry_heal_target_5": {
                "type": "targetHint",
                "text": "Выбери цель для заклинания Фрирен",
            },
        },
    }

    for card_id, events in expected_texts.items():
        assert f"'{card_id}':" in config_block
        for event_name, expected in events.items():
            assert config["cards"][card_id]["texts"][event_name] == expected
            assert f"'{event_name}'" in config_block
            expected_text = expected.get("text")
            if expected_text:
                assert f"text: '{expected_text}'" in config_block

    assert "function showArenaAttackHintForCard" in source
    assert "window.showArenaAttackHintForCard = showArenaAttackHintForCard;" in source
    assert "resolveArenaCardText('attackTargeting', cardContext, { mechanic })" in source
    assert "showArenaAttackHintForCard(attackerCard);" in source


def test_arena_sound_event_server_client_plumbing_contract():
    battle_source = Path("battle_engine.py").read_text(encoding="utf-8")
    battle_compact = "".join(battle_source.split())
    server_source = Path("web/server.py").read_text(encoding="utf-8")

    assert "def _sound_events_for_action" in battle_source
    assert "def _sound_card_snapshot_for_action" in battle_source
    assert 'event="deploy"' in battle_compact
    assert 'event="attack"' in battle_compact
    assert '"sound_events"' in battle_source

    assert (
        'add_static("/assets/audio/characters/"' in server_source
        or "add_static('/assets/audio/characters/'" in server_source
    )
    assert (
        'payload["sound_events"]' in server_source
        or '"sound_events": result.get("sound_events", [])' in server_source
    )


def test_timer_modal_uses_battle_modal_shell_and_history():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")

    assert '<article class="battle-modal turn-timer-modal" id="turn-timer-modal"' in markup
    assert 'id="turn-timer-history"' in markup
    assert "openTurnTimerModal()" in source
    assert "renderTurnTimerHistory" in source
    assert "turn_time_history" in source


def test_opponent_hero_info_uses_shared_card_info_with_mechanics_description():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    opponent_info_block = source.split("function openOpponentInfo()", 1)[1].split(
        "// ============================================\n// ФОНОВАЯ МУЗЫКА АРЕНЫ",
        1,
    )[0]

    assert "openCardInfo" in opponent_info_block
    assert "mechanics_desc: hero.mechanics_desc" in opponent_info_block
    assert "card_type: hero.card_type || 'hero'" in opponent_info_block


def test_state_changed_ignores_payload_for_different_viewer():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    handler_block = source.split("function handleStateChanged(eventData)", 1)[1].split(
        "function validateStateTransition",
        1,
    )[0]

    assert "newState.viewer_id != null" in handler_block
    assert "String(newState.viewer_id) !== String(userId)" in handler_block
    assert "return;" in handler_block


def test_friend_invite_requests_include_selected_deck():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "function getFriendlySelectedDeckId(profile)" in source
    assert "const selected_deck_id = Number(profile?.selected_deck || profile?.primary_deck || deckId)" in source
    assert "body: JSON.stringify({invite_id: invite.id, action: action, selected_deck_id:" in source
    assert "body: JSON.stringify({invite_id: inviteId, action: 'accept', selected_deck_id:" in source


def test_battle_pick_sheet_friendly_tab_sends_invites_and_refreshes_online_friends():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    sheet_block = source.split("const BattlePickSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// MENU SHEET",
        1,
    )[0]

    assert "const loadOnlineFriends" in sheet_block
    assert "fetch(_buildAuthUrl('/api/friends/list'), {cache:'no-store'})" in sheet_block
    assert "aria-label=\"Обновить друзей онлайн\"" in sheet_block
    assert "const sendFriendInvite" in sheet_block
    assert "fetch(_buildAuthUrl('/api/friends/invite')," in sheet_block
    assert "watchFriendlyInviteUntilBattle(d.invite_id, profile" in sheet_block
    assert "InviteInput onSend={sendFriendInvite}" in sheet_block
    assert "onlineFriends.map" in sheet_block
    assert "Дружеские игры в разработке" not in sheet_block


def test_game_mode_sheet_declares_selected_deck_playability_before_render():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    sheet_block = source.split("const GameModeSheet", 1)[1].split(
        "// ═══════════════════════════════════════════\n// MAIL SCREEN",
        1,
    )[0]
    pre_render_block = sheet_block.split(
        '  return (\n    <div className="mm-redesign">',
        1,
    )[0]

    assert "const selectedDeckValidity = getDeckPresetValidity(selectedDeck);" in pre_render_block
    assert "const deckCount = selectedDeckValidity.ownedValidCount;" in pre_render_block
    assert "const selectedDeckPlayable = selectedDeckValidity.isCompletePlayable;" in pre_render_block
    assert "!selectedDeckPlayable" in sheet_block
    assert "!selectedDeckValidity.hasHero" in sheet_block


def test_mail_screen_is_limited_to_transaction_status_and_league_mail():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    mail_block = source.split(
        "// ═══════════════════════════════════════════\n// MAIL SCREEN",
        1,
    )[1].split(
        "// ═══════════════════════════════════════════\n// PRE-BATTLE VS SCREEN",
        1,
    )[0]

    assert "const MAIL_CATEGORY_LABELS = {rewards:'Награды', system:'Статус'};" in mail_block
    assert "news:'Новости'" not in mail_block
    assert "event:'Событие'" not in mail_block
    assert "squad:'Отряд'" not in mail_block
    assert "height:'72px'" in mail_block
    assert "lineHeight:1.15" in mail_block
    assert "Здесь будут транзакции, статусы аккаунта и достижения лиг" in mail_block


def test_friend_list_payload_marks_runtime_online_friends():
    source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = source.split("async def friend_list_handler", 1)[1].split(
        "async def friend_remove_handler",
        1,
    )[0]

    assert "friend[\"online\"] = _is_user_online" in handler_block
    assert "\"online_ttl_seconds\": ONLINE_USER_TTL_SECONDS" in handler_block


def test_training_uses_four_release_models():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    server = Path("web/server.py").read_text(encoding="utf-8")
    from infrastructure.config import BOT_DIFFICULTY_PROFILES

    difficulty_block = source.split("const DIFFICULTIES = [", 1)[1].split("];", 1)[0]
    handler_block = server.split("async def match_vs_bot_handler", 1)[1].split(
        "async def",
        1,
    )[0]

    assert "tier_lite_0000" in difficulty_block
    assert "tier_easy_plus_0300" in difficulty_block
    assert "tier_medium_1200" in difficulty_block
    assert "tier_hard_4500" in difficulty_block
    assert "TrainV2" not in difficulty_block
    assert "extra-lr" not in difficulty_block
    assert "V4 Micro" in difficulty_block
    assert "V5 Lite" in difficulty_block
    assert "V5 Ultra" in difficulty_block
    assert difficulty_block.count("{id:") == 4
    assert "React.useState('tier_lite_0000')" in source
    assert "BOT_DIFFICULTY_ALIASES.get" in handler_block
    assert "BOT_DIFFICULTY_PROFILES" in handler_block
    assert "valid_difficulties = tuple(BOT_DIFFICULTY_PROFILES.keys())" in handler_block
    profile = BOT_DIFFICULTY_PROFILES["tier_lite_0000"]
    assert profile["format"] == "train_v2_classic_v1"
    assert profile["model_path"].endswith("extra-lr-v4-micro.onnx")
    # Every later UI choice uses a V5 contract; the Ultra choice enables assists.
    for tier_key in (
        "tier_easy_plus_0300",
        "tier_medium_1200",
        "tier_hard_4500",
    ):
        profile = BOT_DIFFICULTY_PROFILES[tier_key]
        assert profile["format"] == "v5"
        assert "extra-lr-v5" in profile["model_path"]
        assert profile["obs_dim"] == 7128
        assert profile["mana_draw_head"] is True
    assert BOT_DIFFICULTY_PROFILES["tier_hard_4500"]["assembler_enabled"] is True
    assert BOT_DIFFICULTY_PROFILES["tier_hard_4500"]["cardoptimum_enabled"] is True


def test_friendly_invite_uses_persistent_waiting_watcher():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "function showFriendlyBattleWait" in source
    assert "function watchFriendlyInviteUntilBattle" in source
    assert "window.__friendlyInviteWatch" in source
    assert "watchFriendlyInviteUntilBattle(d.invite_id, profile" in source
    assert "showFriendlyBattleWait('Синхронизируем бой'" in source


def test_friendly_invite_modal_is_idempotent_after_accept_started():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "function markFriendlyInviteHandled" in source
    assert "function isFriendlyInviteHandled" in source
    assert "window.__friendlyInviteResponding" in source
    assert "if (isFriendlyInviteHandled(data.invite.id)) return;" in source
    assert "if (String(window.__friendlyInviteResponding || '') === String(data.invite.id)) return;" in source
    assert "d.error === 'invite_already_responded'" in source
    assert "markFriendlyInviteHandled(invite.id)" in source


def test_friend_invite_accept_is_server_idempotent_when_match_already_exists():
    source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]

    accepted_pos = handler_block.index('invite["status"] == "accepted"')
    non_pending_pos = handler_block.index('invite["status"] != "pending"')
    assert accepted_pos < non_pending_pos
    assert "_friend_invite_status_payload(invite, user_id)" in handler_block


def test_friendly_invite_payloads_do_not_embed_raw_players_profiles():
    source = Path("web/server.py").read_text(encoding="utf-8")
    profile_block = source.split("async def _resolve_battle_profile", 1)[1].split(
        "def _extra_pass_access",
        1,
    )[0]
    status_block = source.split("async def _friend_invite_status_payload", 1)[1].split(
        "async def battle_history_handler",
        1,
    )[0]
    respond_block = source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]

    assert '"raw_profile"' not in profile_block
    assert '"players"' not in status_block
    assert '"players"' not in respond_block


def test_battle_history_api_does_not_request_unbounded_history_window():
    source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = source.split("async def battle_history_handler", 1)[1].split(
        "async def friend_invite_respond_handler",
        1,
    )[0]
    compact = "".join(handler_block.split())

    assert "/api/battles/history" in source
    assert "get_battle_history(user_id,100000)" not in compact


def test_friend_invite_accept_failures_make_invite_terminal():
    source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]
    compact = "".join(handler_block.split())

    assert "async def _fail_friend_invite_accept" in handler_block
    assert "await db.fail_friend_invite_accept" in handler_block
    assert '_fail_friend_invite_accept("invalid_deck_id"' in compact
    assert '_fail_friend_invite_accept("feature_unavailable"' in compact
    assert '_fail_friend_invite_accept("card_unavailable"' in compact
    assert '_fail_friend_invite_accept("deck_load_timeout"' in compact
    assert '_fail_friend_invite_accept("deck_load_failed"' in compact
    assert '_fail_friend_invite_accept("match_create_failed"' in compact


def test_friend_invite_accept_uses_atomic_state_machine():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    claim_block = db_source.split("async def claim_friend_invite_accept", 1)[1].split(
        "async def complete_friend_invite_accept",
        1,
    )[0]
    complete_block = db_source.split("async def complete_friend_invite_accept", 1)[1].split(
        "async def fail_friend_invite_accept",
        1,
    )[0]
    fail_block = db_source.split("async def fail_friend_invite_accept", 1)[1].split(
        "async def get_friend_invite_for_user",
        1,
    )[0]
    handler_block = server_source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]

    assert "status = 'pending'" in claim_block
    assert "expires_at > NOW()" in claim_block
    assert "status = 'accepting'" in claim_block
    assert "status = 'accepted'" in complete_block
    assert "WHERE id = $1 AND status = 'accepting'" in complete_block
    assert "from_deck_card_ids" in complete_block
    assert "to_deck_card_ids" in complete_block
    assert "fail_reason" in fail_block
    assert "await db.claim_friend_invite_accept" in handler_block
    assert "await db.complete_friend_invite_accept" in handler_block
    assert "await db.fail_friend_invite_accept" in handler_block
    assert 'await db.update_invite_status(\n            invite_id,\n            "accepted"' not in handler_block
    assert 'request.app["active_matches"].pop(match_id, None)' in handler_block


def test_accepted_friendly_invite_can_rehydrate_missing_engine():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    join_block = server_source.split("async def join_match", 1)[1].split(
        "@sio.event\nasync def leave_match",
        1,
    )[0]
    state_block = server_source.split("async def battle_state_handler", 1)[1].split(
        "async def battle_action_handler",
        1,
    )[0]

    assert "async def get_friend_invite_by_battle_id_for_user" in db_source
    assert "async def _ensure_friendly_match_engine" in server_source
    assert 'app["ensure_friendly_match_engine"] = _ensure_friendly_match_engine' in server_source
    assert "get_friend_invite_by_battle_id_for_user" in server_source
    assert "from_deck_card_ids" in server_source
    assert "to_deck_card_ids" in server_source
    assert "ensure_friendly_match_engine" in join_block
    assert "ensure_friendly_match_engine" in state_block


def test_completed_friendly_invites_do_not_rehydrate_again():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    rehydrate_lookup = db_source.split("async def get_friend_invite_by_battle_id_for_user", 1)[1].split(
        "async def expire_old_invites",
        1,
    )[0]

    assert "LEFT JOIN battle_summary" in rehydrate_lookup
    assert "bs.match_id IS NULL" in rehydrate_lookup


def test_friend_invite_terminal_status_updates_only_pending_rows():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    update_block = db_source.split("async def update_invite_status", 1)[1].split(
        "async def get_friend_invite_for_user",
        1,
    )[0]
    respond_block = server_source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]
    cancel_block = server_source.split("async def friend_invite_cancel_handler", 1)[1].split(
        "async def friend_request_send_handler",
        1,
    )[0]

    assert "status IN ('expired', 'declined', 'cancelled')" in update_block
    assert "AND status = 'pending'" in update_block
    assert "declined = await db.update_invite_status" in respond_block
    assert "cancelled = await db.update_invite_status" in cancel_block


def test_friend_invite_creation_is_idempotent_and_blocks_active_battles():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    create_block = db_source.split("async def create_friend_invite", 1)[1].split(
        "async def get_pending_invite",
        1,
    )[0]
    ensure_block = db_source.split("async def _ensure_friend_invites_table", 1)[1].split(
        "async def _ensure_generator_state_table",
        1,
    )[0]
    invite_block = server_source.split("async def friend_invite_handler", 1)[1].split(
        "async def friend_invite_status_handler",
        1,
    )[0]
    respond_block = server_source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]

    assert "friend_invites_pending_pair_uniq" in ensure_block
    assert "ON CONFLICT (from_user_id, to_user_id) WHERE status = 'pending'" in create_block
    assert "(xmax = 0) AS created" in create_block
    assert '"error": "invite_already_sent"' in create_block
    assert 'result.get("error") == "invite_already_sent"' in invite_block
    assert "active_battle_exists" in create_block
    assert "_friendly_invite_active_battle_conflict" in invite_block
    assert "_friendly_invite_active_battle_conflict" in respond_block
    assert "exclude_invite_id=invite_id" in respond_block


def test_friend_requests_respect_privacy_reopen_terminal_rows_and_notify():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = server_source.split("async def friend_request_send_handler", 1)[1].split(
        "async def friend_requests_list_handler",
        1,
    )[0]

    assert "async def create_or_reopen_friend_request" in db_source
    assert "ON CONFLICT (requester_id, addressee_id) DO UPDATE" in db_source
    assert "status IN ('declined', 'cancelled')" in db_source
    assert "already_friends" in handler_block
    assert "social_block_friend_requests" in handler_block
    assert "friend_requests_blocked" in handler_block
    assert "create_or_reopen_friend_request" in handler_block
    assert "enqueue_notification" in handler_block
    assert '"friend_requests"' in handler_block
    assert '"friend_request_received"' in handler_block


def test_friendly_battles_use_strict_selected_decks_and_expose_deck_picker():
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    index_source = Path("webapp/index.html").read_text(encoding="utf-8")
    handler_block = server_source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]
    invite_block = server_source.split("async def friend_invite_handler", 1)[1].split(
        "async def friend_invite_status_handler",
        1,
    )[0]

    assert "allow_cache_fallback=False" in handler_block
    assert "len(p1_deck_int) != DECK_SIZE" in handler_block
    assert '_fail_friend_invite_accept("deck_incomplete"' in "".join(handler_block.split())
    assert '"is_playable" in preset' in server_source
    assert "not preset.get(\"is_playable\")" in server_source
    assert "unsupported_friendly_mode_modifiers" in invite_block
    assert '"supports_mode_modifiers": False' in invite_block
    assert "function getFriendlyDeckOptions(profile)" in index_source
    assert "friendlyDeckId" in index_source
    assert "setFriendlyDeckId" in index_source
    assert "selected_deck_id: friendlyDeckId || getFriendlySelectedDeckId(profile)" in index_source
    assert "incomingFriendlyDeckId" in index_source
    assert "setIncomingFriendlyDeckId" in index_source


def test_friendly_challenge_privacy_settings_are_persisted():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    assert "social_block_friendly_invites_from_friends BOOLEAN NOT NULL DEFAULT false" in db_source
    assert "social_block_friendly_invites_from_non_friends BOOLEAN NOT NULL DEFAULT true" in db_source
    assert '"social_block_friendly_invites_from_friends"' in db_source
    assert '"social_block_friendly_invites_from_non_friends"' in db_source


def test_friendly_challenge_privacy_settings_are_exposed_by_api():
    source = Path("web/server.py").read_text(encoding="utf-8")
    assert '"social_block_friendly_invites_from_friends": settings_record.get("social_block_friendly_invites_from_friends", False)' in source
    assert '"social_block_friendly_invites_from_non_friends": settings_record.get("social_block_friendly_invites_from_non_friends", True)' in source
    assert '"social_block_friendly_invites_from_friends": False' in source
    assert '"social_block_friendly_invites_from_non_friends": True' in source


def test_friendly_invite_handler_respects_target_privacy():
    source = Path("web/server.py").read_text(encoding="utf-8")
    invite_block = source.split("async def friend_invite_handler", 1)[1].split(
        "async def friend_invite_status_handler",
        1,
    )[0]
    assert "social_block_friendly_invites_from_friends" in invite_block
    assert "social_block_friendly_invites_from_non_friends" in invite_block
    assert "friendly_invites_blocked_by_friend" in invite_block
    assert "friendly_invites_blocked_by_non_friend" in invite_block
    assert "not_friends" not in invite_block


def test_friendly_invite_handler_silently_declines_bot_ids_before_privacy_flow():
    source = Path("web/server.py").read_text(encoding="utf-8")
    invite_block = source.split("async def friend_invite_handler", 1)[1].split(
        "async def friend_invite_status_handler",
        1,
    )[0]

    assert "COALESCE(is_bot, FALSE) AS is_bot" in invite_block
    assert "asyncio.sleep(random.uniform(7, 10))" in invite_block
    assert '"status": "declined"' in invite_block
    assert '"error": "invite_declined"' in invite_block
    assert "friendly_invites_bot_not_supported" not in invite_block
    assert "бот" not in invite_block.lower()
    assert invite_block.index("COALESCE(is_bot, FALSE) AS is_bot") < invite_block.index("_resolve_match_deck_id")
    assert invite_block.index('"status": "declined"') < invite_block.index("social_block_friendly_invites_from_friends")
    assert invite_block.index('"status": "declined"') < invite_block.index("create_friend_invite")


def test_settings_ui_exposes_friendly_challenge_privacy():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    assert "blockFriendlyFromFriends" in source
    assert "blockFriendlyFromNonFriends" in source
    assert "social_block_friendly_invites_from_friends" in source
    assert "social_block_friendly_invites_from_non_friends" in source
    assert "Блокировать вызовы на дружеские бои" in source
    assert "От друзей" in source
    assert "От не-друзей" in source
    assert "friendly_invites_blocked_by_friend" in source
    assert "friendly_invites_blocked_by_non_friend" in source


def test_settings_and_extraid_guard_unhandled_frontend_paths():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    settings_block = source.split("const SettingsScreen = ({onClose, profile}) => {", 1)[1].split(
        "// ═══════════════════════════════════════════\n// HELPERS",
        1,
    )[0]
    extraid_block = source.split("const ExtraIDSheet = ({onClose, view", 1)[1].split(
        "// ═══════════════════════════════════════════\n// AI SECTION",
        1,
    )[0]

    assert "settingsLoaded" in settings_block
    assert "if (!settingsLoaded) return;" in settings_block
    assert "if (!res.ok" in settings_block
    assert "showToast?.('Не удалось сохранить настройки" in settings_block
    assert "window.__openExtraPassModal?.('basic')" in settings_block
    assert "resetSettingsToDefaults" in settings_block

    assert "openLink('https://t.me/extraarena_supbot')" in extraid_block
    assert "tg?.openTelegramLink" not in extraid_block


def test_extraid_logout_is_blocked_in_telegram_but_kept_for_other_clients():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    telegram_detector = source.split("function isTelegramGameClient()", 1)[1].split(
        "function allowLocalDevUserIdAuth",
        1,
    )[0]
    extraid_block = source.split("const ExtraIDSheet = ({onClose, view", 1)[1].split(
        "// ═══════════════════════════════════════════\n// AI SECTION",
        1,
    )[0]
    logout_block = extraid_block.split("const doLogout = async () => {", 1)[1].split(
        "var renderContent = null",
        1,
    )[0]

    assert "return !getAndroidBridge() && !!getTelegramInitData();" in telegram_detector
    assert "const isTelegramFlow = isTelegramGameClient();" in extraid_block
    assert "if (isTelegramFlow)" in logout_block
    assert "В Telegram выход из ExtraID недоступен" in logout_block
    assert "revoked = response.ok || response.status === 401;" in logout_block
    assert "Не удалось завершить серверную сессию" in logout_block
    assert "hasLocalExtraSession && !isPlatformFlow ? React.createElement('button',{key:'logout'" in extraid_block
    assert "if (isMaxFlow)" in logout_block
    assert "В MAX выход из ExtraID недоступен" in logout_block


def test_extraid_email_verification_and_password_recovery_frontend_contract():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    extraid_block = source.split("const ExtraIDSheet = ({onClose, view", 1)[1].split(
        "// ═══════════════════════════════════════════\n// INFO SHEET",
        1,
    )[0]

    assert "function takeExtraIDActionTokenFromLocation(name)" in source
    assert "extraid_verify_token" in source
    assert "extraid_reset_token" in source
    assert "extraid_cancel_token" in source
    assert "current.searchParams.get(name)" not in source
    assert "hashParams.delete(name)" in source
    assert "utf8ByteLength(password) > 72" in extraid_block
    assert "registerPayload.tg_init_data = authData" in extraid_block
    assert "fetch('/api/extraid/register'" in extraid_block
    assert "buildUiAuthUrl('/api/extraid/register'" not in extraid_block
    assert "d.error === 'email_not_verified'" in extraid_block
    assert "/api/extraid/email/resend" in extraid_block
    assert "/api/extraid/email/change" in extraid_block
    assert "changeHeaders['X-Telegram-Init-Data'] = telegramAuth" in extraid_block
    assert "changeHeaders.Authorization = 'Bearer ' + emailChangeOwnerToken" in extraid_block
    assert "Изменить email и отправить письмо" in extraid_block
    assert "/api/extraid/password-reset/request" in extraid_block
    assert "/api/extraid/password-reset/confirm" in extraid_block
    assert "/api/extraid/email/cancel" in source
    assert "view === 'verify_email'" in extraid_block
    assert "view === 'reset_password'" in extraid_block
    assert "extraIDData?.is_email_verified" in extraid_block
    assert "...(isTelegramFlow ? [{icon:" in extraid_block
    assert "isPlatformFlow ? React.createElement('div', {key:'reward'" in extraid_block
    assert "const isPlatformBound = isPlatformFlow || !!extraIDData?.linked_telegram || !!extraIDData?.linked_max;" in extraid_block
    assert "!isPlatformBound ? React.createElement('div', {key:'dangerTitle'" in extraid_block


def test_friendly_invite_client_uses_generic_decline_for_hidden_bot_ids():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    assert "friendly_invites_bot_not_supported" not in source
    assert "Ботов нельзя вызвать по ID" not in source
    assert "invite_declined" in source
    assert "Вызов отклонён" in source


def test_api_unhandled_exceptions_are_rendered_as_json_for_clients():
    source = Path("web/server.py").read_text(encoding="utf-8")
    middleware_block = source.split("async def api_json_error_middleware", 1)[1].split(
        "@web.middleware\n    async def runtime_gate_middleware",
        1,
    )[0]
    middleware_order = source.split("app.middlewares.append(admin_auth_middleware)", 1)[0]

    assert 'path.startswith("/api/")' in middleware_block
    assert "except web.HTTPException" in middleware_block
    assert '"internal_server_error"' in middleware_block
    assert "web.json_response" in middleware_block
    assert "app.middlewares.append(api_json_error_middleware)" in middleware_order


def test_sensitive_auth_responses_are_not_cacheable_and_telegram_auth_uses_header():
    source = Path("web/server.py").read_text(encoding="utf-8")

    assert 'request.headers.get("X-Telegram-Init-Data")' in source
    assert "TELEGRAM_INIT_DATA_MAX_BYTES = 16 * 1024" in source
    assert "Content-Type, Authorization, X-Telegram-Init-Data" in source
    assert "async def sensitive_auth_no_store_middleware" in source
    assert 'path.startswith("/api/extraid/")' in source
    assert 'path.startswith("/api/telegram-transfer/")' in source
    assert 'path.startswith("/api/auth/")' in source
    assert 'path.startswith("/api/rlhf/")' in source
    assert 'response.headers["Pragma"] = "no-cache"' in source


def test_friendly_invite_failed_status_stops_accept_loop():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "function friendlyInviteErrorText" in source
    assert "['declined', 'cancelled', 'expired', 'failed'].includes(d.status)" in source
    assert "d.message || friendlyInviteErrorText(d.error)" in source
    assert "d.status === 'failed'" in source
    assert "markFriendlyInviteHandled(invite.id)" in source


def test_friendly_invite_client_handles_non_json_5xx_and_stops_waiting_loop():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    watcher_block = source.split("function watchFriendlyInviteUntilBattle", 1)[1].split(
        "function openFriendlyInviteBattle",
        1,
    )[0]
    modal_block = source.split("var IncomingInviteModal", 1)[1].split(
        "const SquadToken",
        1,
    )[0]
    link_block = source.split("const acceptFromLink = async () =>", 1)[1].split(
        "acceptFromLink();",
        1,
    )[0]

    assert "async function readFriendlyInviteJson(response)" in source
    assert "network_error" in source
    assert "friendly_invite_bad_response" in source
    assert "const maxErrors = Number(options?.maxErrors || 8)" in watcher_block
    assert "watcher.errors" in watcher_block
    assert "readFriendlyInviteJson(r)" in watcher_block
    assert "hideFriendlyBattleWait()" in watcher_block
    assert "options?.onTerminal?.({status:'failed', error:'network_error'})" in watcher_block
    assert "const d = await readFriendlyInviteJson(r)" in modal_block
    assert "const sd = await readFriendlyInviteJson(sr)" in modal_block
    assert "const d = await readFriendlyInviteJson(r)" in link_block
    assert "const sd = await readFriendlyInviteJson(sr)" in link_block


def test_friendly_invites_use_unordered_pair_guard_and_cancel_reconciliation():
    index_source = Path("webapp/index.html").read_text(encoding="utf-8")
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    battle_pick_cancel = index_source.split("const cancelFriendInvite = async () =>", 1)[1].split(
        "const startBot",
        1,
    )[0]
    friends_cancel = index_source.split("const cancelInvite = async () =>", 1)[1].split(
        "const addFriend = async",
        1,
    )[0]

    assert "friend_invites_active_pair_uniq" in db_source
    assert "LEAST(from_user_id, to_user_id)" in db_source
    assert "GREATEST(from_user_id, to_user_id)" in db_source
    assert "active.status IN ('pending', 'accepting', 'accepted')" in db_source
    assert "reconcileCancelledFriendlyInvite" in index_source
    assert "await reconcileCancelledFriendlyInvite(friendInviteId, profile" in battle_pick_cancel
    assert "await reconcileCancelledFriendlyInvite(inviteId, profile" in friends_cancel


def test_friend_request_state_transitions_are_atomic():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    update_block = db_source.split("async def update_friend_request_status", 1)[1].split(
        "async def get_incoming_friend_requests",
        1,
    )[0]

    assert "AND status = 'pending'" in update_block
    assert "RETURNING id" in update_block
    assert "return row is not None" in update_block


def test_social_payloads_include_public_id_privacy_flags():
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    for start_marker, end_marker in (
        ("async def get_incoming_friend_requests", "async def get_outgoing_friend_requests"),
        ("async def get_outgoing_friend_requests", "async def get_friend_list"),
        ("async def get_friend_list", "async def remove_friendship"),
        ("async def get_recent_opponents", "async def create_payment"),
    ):
        block = db_source.split(start_marker, 1)[1].split(end_marker, 1)[0]
        assert "hide_player_id_public" in block
        assert "LEFT JOIN user_settings" in block


def test_friendly_deck_picker_reuses_playable_deck_validity():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("function normalizeFriendlyDeckOptions", 1)[1].split(
        "function getFriendlyDeckOptions",
        1,
    )[0]

    assert "getDeckPresetValidity" in block
    assert "isCompletePlayable" in block
    assert "filled >= 9" not in block


def test_arena_waits_for_both_players_before_unlocking_battle_ui():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function updateArenaWaitingOverlay" in source
    assert "match_status === 'waiting_for_players'" in source
    assert "socket.on('match_waiting'" in source
    assert "socket.on('match_ready'" in source


def test_arena_talkie_markup_uses_hp_button_picker_overlay_and_audio_aliases():
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    hp_block = markup.split('id="player-hp-block"', 1)[1].split('class="player-name-block"', 1)[0]

    assert 'id="player-hp-fill"' in hp_block
    assert 'id="talkie-button"' in hp_block
    assert hp_block.index('id="player-hp-fill"') < hp_block.index('id="talkie-button"')
    assert "../DesignAssets/Arena/Talkie.png" in hp_block
    assert 'id="talkie-popover"' in markup
    assert 'id="talkie-picker-grid"' in markup
    assert 'id="talkie-muted-toggle"' in markup
    assert 'id="talkie-limit-text"' in markup
    assert "Использовать Talkies" in markup
    assert "Если выключить - не сможешь отправлять и видеть чужие Talkies" in markup
    assert 'id="talkie-fullscreen-overlay"' in markup
    overlay_block = markup.split('id="talkie-fullscreen-overlay"', 1)[1].split("</div>", 2)[0]
    assert 'id="talkie-fullscreen-image"' in overlay_block
    assert "talkie-screen-flash" in overlay_block
    assert "talkie-burst-flash" in overlay_block
    for alias in ("happy", "neutral", "rude", "sad"):
        assert f'id="arena-sfx-talkie-{alias}"' in markup
        assert f"../DesignAssets/Arena/Talkies/sounds/{alias}.wav" in markup


def test_arena_talkie_css_is_absolute_compact_and_supports_behind_sticker_flash():
    styles = Path("webapp/arena-styles.css").read_text(encoding="utf-8")

    assert ".talkie-button" in styles
    talkie_button = styles.split(".talkie-button", 1)[1].split("}", 1)[0]
    assert "position: absolute" in talkie_button
    assert "left: 8px" in talkie_button
    assert "top: 50%" in talkie_button
    assert ".player-hp-block::before" in styles
    assert ".player-hp-block::after" in styles
    assert "flex: 0 0 34px" in styles
    player_sudden_death_hp = styles.split(".player-hp-block.sudden-death-hp", 2)[2].split("}", 1)[0]
    assert "padding-left: 8px" in player_sudden_death_hp
    assert "padding-right: 8px" in player_sudden_death_hp
    assert ".talkie-icon" in styles
    assert "brightness(0) invert(1)" in styles
    assert ".talkie-popover" in styles
    talkie_popover = styles.split(".talkie-popover", 1)[1].split("}", 1)[0]
    assert "226px" in talkie_popover
    assert "#0f0a1a" in styles
    assert "#1a1030" in styles
    assert "#4a3d6a" in styles
    assert "#7a6fa0" in styles
    assert "#f5921e" in styles
    assert ".talkie-picker-grid" in styles
    assert "grid-template-columns: repeat(4, 1fr)" in styles
    assert ".talkie-toggle-note" in styles
    mini_icon = styles.split(".talkie-sticker-btn img", 1)[1].split("}", 1)[0]
    assert "invert(1)" in mini_icon
    assert ".talkie-fullscreen-overlay" in styles
    assert ".talkie-fullscreen-overlay.is-visible" in styles
    assert ".talkie-screen-flash" in styles
    screen_flash = styles.split(".talkie-screen-flash", 1)[1].split("}", 1)[0]
    assert "rgba(255,255,255,0.48)" in screen_flash
    assert "0.5s ease-out" in screen_flash
    assert "@keyframes talkieScreenFlash" in styles
    assert ".talkie-burst-flash" in styles
    assert "@keyframes talkieBurstFlash" in styles


def test_arena_talkie_js_sends_settings_events_and_delivers_fullscreen_playback():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "const TALKIE_ENABLED_STORAGE_KEY" in source
    assert "const TALKIE_CATALOG" in source
    for talkie_id in range(1, 8):
        assert f"id: '{talkie_id}'" in source

    for sound_key, alias in (
        ("talkieHappy", "arena-sfx-talkie-happy"),
        ("talkieNeutral", "arena-sfx-talkie-neutral"),
        ("talkieRude", "arena-sfx-talkie-rude"),
        ("talkieSad", "arena-sfx-talkie-sad"),
    ):
        assert f"{sound_key}: '{alias}'" in source

    assert "function initTalkies()" in source
    assert "function renderTalkiePicker()" in source
    assert "function sendTalkie(talkieId)" in source
    assert "function showTalkieFullscreen(event)" in source
    assert "function emitTalkieSettings()" in source
    assert "function updateTalkieAvailability(state)" in source
    assert "socket.emit('battle_talkie'" in source
    assert "socket.emit('battle_talkie_settings'" in source
    assert "socket.on('battle_talkie'" in source
    assert "socket.on('battle_talkie_ack'" in source
    assert "socket.on('battle_talkie_settings_ack'" in source
    assert "localStorage.setItem(TALKIE_ENABLED_STORAGE_KEY" in source
    assert "showTalkieFullscreen(data)" in source
    assert "playArenaSfx(soundKey" in source
    assert "Используй Talkies во время своего хода" in source
    assert "image.removeAttribute('src')" in source
    assert "event?.event_id" in source
    assert "encodeURIComponent" in source
    assert "?v=${cacheKey}" in source
    assert "talkie-burst-flash" in source
    assert "burst.style.animation = 'none'" in source
    assert "talkie-screen-flash" in source
    assert "screenFlash.style.animation = 'none'" in source
    assert "updateTalkieAvailability(state)" in source
    assert "emitTalkieSettings();" in source


def test_arena_talkie_respects_player_sfx_settings_and_has_mobile_haptics():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function applyArenaSoundSettingsFromUserSettings(settings = {})" in source
    assert "if (!urlParams.has('sfx') && typeof settings.sound_sfx === 'boolean')" in source
    assert "window._sfxEnabled = settings.sound_sfx;" in source

    startup_settings_block = source.split("async function loadTalkieStartupSettings()", 1)[1].split(
        "function getTalkieExtraPassTier",
        1,
    )[0]
    assert "applyArenaSoundSettingsFromUserSettings(settings);" in startup_settings_block
    assert "applyTalkieDisableByDefault(settings?.social_disable_talkies === true)" in startup_settings_block

    assert "function playTalkieHaptic(event = {})" in source
    talkie_haptic_block = source.split("function playTalkieHaptic(event = {})", 1)[1].split(
        "function initTalkies()",
        1,
    )[0]
    assert "event?.sender_id" in talkie_haptic_block
    assert "String(userId)" in talkie_haptic_block
    assert "arenaHaptic(isOwnTalkie ? 'success' : 'medium'" in talkie_haptic_block
    assert "setTimeout(() => arenaHaptic('light'" in talkie_haptic_block
    assert "playTalkieHaptic(event);" in source


def test_global_talkie_disable_setting_is_persisted_rendered_and_applied_to_arena():
    db = Path("infrastructure/database.py").read_text(encoding="utf-8")
    server = Path("web/server.py").read_text(encoding="utf-8")
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    arena = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "social_disable_talkies BOOLEAN NOT NULL DEFAULT false" in db
    assert '"social_disable_talkies"' in db
    assert '"social_disable_talkies": False' in server
    assert '"social_disable_talkies": settings_record.get("social_disable_talkies", False)' in server

    assert "Сразу отключать Talkie" in index
    assert "social_disable_talkies" in index
    assert "extraarena.talkie.disableByDefault" in index

    assert "const TALKIE_DISABLE_BY_DEFAULT_STORAGE_KEY" in arena
    assert "function loadTalkieStartupSettings()" in arena
    assert "buildArenaAuthUrl('/api/settings')" in arena
    assert "social_disable_talkies" in arena
    assert "talkieEnabled = false" in arena
    assert "emitTalkieSettings();" in arena


def test_arena_reconnect_forces_join_even_after_previous_socket_joined():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    connect_block = source.split("socket.on('connect', () => {", 1)[1].split(
        "socket.on('disconnect'",
        1,
    )[0]
    disconnect_block = source.split("socket.on('disconnect', (reason) => {", 1)[1].split(
        "socket.on('connect_error'",
        1,
    )[0]
    emit_join_block = source.split("function emitJoinMatch", 1)[1].split(
        "function scheduleJoinRetry",
        1,
    )[0]

    assert "emitJoinMatch({ force: true })" in connect_block
    assert "socketJoined = false" in disconnect_block
    assert "force" in emit_join_block
    assert "socketJoined && !force" in emit_join_block


def test_arena_repeated_joined_match_allows_idempotent_client_ready_after_prebattle():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    joined_block = source.split("socket.on('joined_match', (data) => {", 1)[1].split(
        "socket.on('client_ready_ack'",
        1,
    )[0]

    assert "clientReadySent = false" in joined_block
    assert joined_block.index("clientReadySent = false") < joined_block.index("trySendClientReady()")


def test_arena_match_terminated_stops_retry_health_and_reconnect_ui():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    terminal_block = source.split("function enterArenaTerminalState", 1)[1].split(
        "function showArenaConnectionModal",
        1,
    )[0]
    handler_block = source.split("socket.on('match_terminated', (data) => {", 1)[1].split(
        "socket.on('state_changed'",
        1,
    )[0]

    assert "enterArenaTerminalState()" in handler_block
    assert "clearTimeout(socketJoinRetryTimer)" in terminal_block
    assert "socketJoinRetryTimer = null" in terminal_block
    assert "stopArenaHealthPing()" in terminal_block
    assert "hideArenaBadConnection()" in terminal_block
    assert "hideArenaConnectionModal()" in terminal_block
    assert "arenaTerminalState" in terminal_block


def test_arena_connection_restart_keeps_current_match_context():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    modal_block = source.split("function showArenaConnectionModal(message)", 1)[1].split(
        "function markArenaConnectionFailure",
        1,
    )[0]

    assert "window.location.reload()" in modal_block
    assert "window.location.replace('/')" not in modal_block


def test_index_startup_checks_active_battle_before_finishing_loading():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    active_block = source.split("async function checkActiveBattleOnStartup", 1)[1].split(
        "window.startBattle",
        1,
    )[0]
    startup_block = source.split("const syncProfile = async () =>", 1)[1].split(
        "syncProfile();",
        1,
    )[0]
    profile_loaded_block = startup_block.split("if (data) {", 1)[1].split(
        "} else {",
        1,
    )[0]

    assert "/api/battle/active" in active_block
    assert "buildUiAuthUrl('/api/battle/active', auth)" in active_block
    assert "window.location.replace(buildArenaRedirectUrl(data.redirect_url, auth))" in active_block
    assert "function buildArenaRedirectUrl" in active_block
    assert "checkActiveBattleOnStartup(startupAuth)" in startup_block
    assert profile_loaded_block.index("checkActiveBattleOnStartup(startupAuth)") < profile_loaded_block.index("setProfile(data)")


def test_index_afk_overlay_is_suppressed_while_user_has_active_overlay():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    afk_block = source.split("const checkAfk = () => {", 1)[1].split(
        "const interval = setInterval(checkAfk",
        1,
    )[0]

    assert "if (hasOverlay) return;" in afk_block
    assert "[profile, inBattle, afkVisible, connectionModal, maintenanceBlocked, hasOverlay, stopConnectionPing]" in source


def test_index_afk_activity_tracking_counts_common_foreground_activity():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const markInteraction = () => {", 1)[1].split(
        "return () => {",
        1,
    )[0]

    for event_name in ["click", "pointerdown", "touchstart", "keydown", "wheel", "scroll", "focus"]:
        assert f"window.addEventListener('{event_name}', markInteraction" in block
    assert "document.addEventListener('visibilitychange', markVisibleInteraction" in block
    assert "document.visibilityState === 'visible'" in block


def test_arena_haptics_are_android_only_and_cover_core_battle_actions():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function isArenaHapticsEnabled()" in source
    assert "localStorage.getItem('extra_haptics_enabled') !== 'false'" in source
    assert "function arenaHaptic(style, options = {})" in source
    assert "window.ExtraArenaApp" in source
    assert "if (!isAndroidArenaShell()) return;" in source
    assert "if (!isArenaHapticsEnabled()) return;" in source
    assert "recordArenaStateHaptic(playerState, opponentState)" in source
    assert "function hasEnoughManaForCard" in source
    assert "flushArenaStateHaptic()" in source
    assert "window.__arenaBattleResultHaptic" in source

    for marker in [
        "arenaHaptic('selection', { key: 'card-pick'",
        "arenaHaptic('warning', { key: 'card-invalid'",
        "arenaHaptic('selection', { key: 'attacker-select'",
        "arenaHaptic('medium', { key: 'target-card'",
        "arenaHaptic('medium', { key: 'target-attack'",
        "arenaHaptic('medium', { key: 'slot-drop'",
        "arenaHaptic('medium', { key: 'play-card-ok'",
        "arenaHaptic('error', { key: 'play-card-error'",
        "arenaHaptic('selection', { key: 'end-turn'",
        "arenaHaptic('heavy', { key: 'countdown-vs'",
        "arenaHaptic('warning', { key: 'surrender-open'",
        "arenaHaptic('error', { key: 'surrender-confirm'",
    ]:
        assert marker in source


def test_trophy_road_sheet_uses_auth_url_for_glory_track_and_claim():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    arena_block = source.split("const ArenaScreen = ({onMenu", 1)[1].split(
        "// ═══════════════════════════════════════════\n// CARD COLLECTION SCREEN",
        1,
    )[0]
    block = source.split("const TrophyRoadSheet = ({onClose, profile}) => {", 1)[1].split(
        "const PremiumBadgeIcon",
        1,
    )[0]

    assert "fetch(_buildAuthUrl('/api/rewards/track/glory'))" in arena_block
    assert "fetch(_buildAuthUrl('/api/rewards/track/glory'))" in block
    assert "fetch(_buildAuthUrl('/api/rewards/claim'), {" in block
    assert '"/api/rewards/track/glory?user_id="' not in source
    assert '"/api/rewards/claim?user_id="' not in source
    assert "const uid = profile?.user_id || window._userId || 0;" not in block
    assert 'body: JSON.stringify({track_type:"glory", position: pos})' in block


def test_squad_wars_beta_tab_is_hidden_behind_disabled_flag():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const SquadsScreen = ({profile,onReloadProfile,onNoticeChange,onOpenAnnouncements}) => {", 1)[1].split(
        "const NewsSheet",
        1,
    )[0]

    assert "const SQUAD_WARS_BETA_ENABLED = false;" in block
    assert "...(SQUAD_WARS_BETA_ENABLED?[['wars','Войны']]:[])" in block
    assert "['shop','Магазин'],['wars','Войны'],['search','Поиск']" not in block
    assert "SQUAD_WARS_BETA_ENABLED && clan && tab === 'wars'" in block


def test_quests_sheet_replaces_daily_login_with_drift_free_countdown():
    # Replaces the old test_daily_login_claimed_next_preview_* guard. The daily-login
    # DailyLoginSheet was removed (commit 8401464d); QuestsSheet is its replacement.
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    # DailyLoginSheet must be gone (no orphan references).
    assert "const DailyLoginSheet = " not in source
    assert "DailyLoginHelpSheet" not in source
    block = source.split("const QuestsSheet = ({onClose, status, onClaimed}) => {", 1)[1].split(
        "const CardArtLightbox",
        1,
    )[0]
    # Fixed full-viewport overlay above the main menu but below toasts (z=200, mirroring the old sheet).
    assert "position:'fixed',inset:0,zIndex:200" in block
    # Art background uses the quests.jpg asset.
    assert "/DesignAssets/Images/quests.jpg" in block
    # Sticky glass header with a "Сброс" countdown chip.
    assert ">Сброс<" in block and "fmtTimer(resetSeconds)" in block
    # Summary block: eyebrow + h2 + "Выполнено X/5" score chip.
    # Fresher mockup (2026-07-16): "дневного маршрута" headline dropped → "Забери награды".
    assert "Ежедневные задания" in block
    assert "Забери награды" in block
    assert "Забери дневной маршрут" not in block
    assert "Выполнено {doneCount}/5" in block
    # Drift-free reset countdown: re-pinned {sec, at: Date.now()} on each status refresh
    # (NOT the naive `status.reset_seconds - tick` that double-counts — see daily-quests-feature memory).
    assert "setPin({sec: status.reset_seconds, at: Date.now()})" in block
    assert "pin.sec - (Date.now() - pin.at) / 1000" in block
    assert "status.reset_seconds - tick" not in block
    assert "reset_seconds - genTick" not in block
    # Fresher mockup (2026-07-16): per-quest status chips ("В процессе"/"Готово"/"Забрано")
    # removed — the action buttons (Забрать / Ждёт / ✓ Забрано) already encode state.
    assert "{t:'В процессе'" not in block
    assert "const badge = " not in block
    assert "'Забрать'" in block
    assert ">Ждёт<" in block
    assert "✓ Забрано" in block
    # Claim posts to the new endpoint.
    assert "/api/daily-quests/claim" in block


def test_back_handler_deps_include_quests_and_pending_case_open():
    # P2 stale-closure guard: the backHandler useEffect uses `showQuests` and
    # `pendingCaseOpen` in its body (to close the Quests sheet and the pending
    # case-open modal), so both MUST appear in its dependency array — else the
    # installed window.ExtraArenaAppBack closes over stale flags and the system
    # Back button stops closing those overlays. Source-level guard so a future
    # edit that drops either dep is caught.
    import re
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    # Isolate the backHandler useEffect block: from the `const backHandler =` decl
    # to the end of its `]);` deps array.
    start = source.index("const backHandler = () => {")
    # find the closing `]);` of this useEffect — the first `]);` after the decl.
    end = source.index("]);", start)
    block = source[start:end]
    assert "if (pendingCaseOpen)" in block
    assert "if (showQuests)" in block
    # The deps array is the text after `}, [` up to the end marker.
    assert "showQuests" in block, "showQuests must be in backHandler deps (used in body)"
    assert "pendingCaseOpen" in block, "pendingCaseOpen must be in backHandler deps (used in body)"


def test_quests_sheet_surfaces_claim_errors_and_rollover_refetch():
    # P2: claim failures must surface a user-visible toast (was: silent onClaimed on every
    # non-2xx, network errors swallowed). And a rollover-refetch effect must refetch the
    # instant the drift-free resetSeconds hits 0 so yesterday's board cannot linger.
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const QuestsSheet = ({onClose, status, onClaimed}) => {", 1)[1].split(
        "const CardArtLightbox", 1
    )[0]
    # Per-error-code toasts on the failure path.
    assert "already_claimed" in block and "Награда уже забрана" in block
    assert "not_claimable" in block and "Квест ещё не выполнен" in block
    assert "unknown_quest" in block
    assert "feature_disabled" in block and "Квесты сейчас недоступны" in block
    # Generic failure + network-error toast (was swallowed by `catch(e) {}`).
    assert "Не удалось забрать награду" in block
    assert "Ошибка сети" in block
    # The catch must now invoke showToast, not be empty.
    assert "catch(e) {" not in block or block.count("window.showToast") >= 4
    # Rollover-refetch effect: re-fetches when resetSeconds hits 0.
    assert "rolledRef" in block and "hadPinRef" in block
    assert "resetSeconds <= 0" in block
    assert "rolledRef.current = true" in block


def test_quests_polling_is_gated_on_sheet_open_and_exposes_refresh_hook():
    # P2 polling load: the 30s interval must gate on showQuests (30s open / 5min closed)
    # so closed sheets don't hammer the DB per-30s. Plus an event-driven refresh hook is
    # exposed so case-open can refetch quest progress instantly.
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const fetchQuestStatus = React.useCallback", 1)[1].split(
        "const onQuestClaimed = React.useCallback", 1
    )[0]
    # Interval picks 30s while open, 5min while closed (not a fixed 30000).
    assert "const intervalMs = showQuests ? 30000 : 300000" in block
    assert "setInterval(fetchQuestStatus, intervalMs)" in block
    # The effect re-runs on showQuests (so the interval re-arms on open/close).
    assert "[fetchQuestStatus, showQuests]" in block
    # Global refresh hook exposed for event-driven refetch.
    assert "window.__refreshQuestStatus = fetchQuestStatus" in block
    # Case-open onDone calls the hook (event-driven refetch of open_case_1 progress).
    assert "window.__refreshQuestStatus?.()" in source


def test_quests_sheet_has_dialog_a11y_and_esc_to_close():
    # QuestsSheet is a real modal: labelled dialog semantics, focus capture/restore,
    # Tab trap, Escape close, and background scroll lock.
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const QuestsSheet = ({onClose, status, onClaimed}) => {", 1)[1].split(
        "const CardArtLightbox", 1
    )[0]
    assert 'role="dialog"' in block and 'aria-modal="true"' in block
    assert 'aria-labelledby="quests-sheet-title"' in block
    assert 'id="quests-sheet-title"' in block
    assert "dialogRef" in block and "Ref.current" in block
    assert "Escape" in block and "onClose" in block
    assert "e.key !== 'Tab'" in block
    assert "previousFocus.focus()" in block
    assert "document.body.style.overflow = 'hidden'" in block
    # Outer container carries the ref + role.
    assert 'ref={dialogRef} role="dialog"' in block


def test_orphaned_daily_rewards_notif_toggle_removed():
    # P3: the "Ежедневные награды" / notif_daily_rewards settings toggle was dead UI
    # (its sole producer, the daily-login scheduler, was removed). The toggle must be
    # gone from both the React settings sheet and the legacy main.js settings UI, and
    # from the notifs state/default. Server-side persistence + notifications.py maps
    # are intentionally kept dormant (column-drop is riskier).
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    assert "dailyRewards" not in source
    assert "notif_daily_rewards" not in source
    assert 'Ежедневные награды' not in source
    main_js = Path("webapp/main.js").read_text(encoding="utf-8")
    assert "notif_daily_rewards" not in main_js
    assert 'Ежедневные награды' not in main_js


def test_analytics_v2_start_captures_returnclock_launch_context():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const AnalyticsController = (() => {", 1)[1].split(
        "window.__analytics = AnalyticsController;",
        1,
    )[0]
    launch_context = block.split("const _readLaunchContext = () => {", 1)[1].split(
        "const _launchContext = _readLaunchContext();",
        1,
    )[0]
    start_block = block.split("const _startCurrentSession = () => {", 1)[1].split(
        "const _startFreshSession = () => {",
        1,
    )[0]

    assert "const ANALYTICS_VERSION = 2;" in block
    assert "params?.get(name)" in launch_context
    assert "cleanParam('ea_platform', 64)" in launch_context
    assert "source: platform || (window.Telegram?.WebApp?.initData ? 'telegram_webapp' : 'web')" in launch_context
    assert "Intl.DateTimeFormat().resolvedOptions().timeZone" in launch_context
    assert "entrypoint: cleanParam('entrypoint', 128)" in launch_context
    assert "cleanParam('rc_decision_id', 128) || cleanParam('returnclock_decision_id', 128) || cleanParam('decision_id', 128)" in launch_context
    assert "returnclockDeliveryId: cleanParam('delivery_id', 128) || cleanParam('notification_id', 128)" in launch_context
    assert "notificationId: cleanParam('notification_id', 128)" in launch_context
    assert "analytics_version: ANALYTICS_VERSION" in start_block
    assert "source: _launchContext.source" in start_block
    assert "timezone: _launchContext.timezone" in start_block
    assert "utc_offset_minutes: -new Date().getTimezoneOffset()" in start_block
    assert "const includeLaunchAttribution = !_launchAttributionConsumed;" in start_block
    assert "_launchAttributionConsumed = true;" in start_block
    assert start_block.index("_postJSONChecked('/api/analytics/session/start'") < start_block.index("_launchAttributionConsumed = true;")
    assert "_clearLaunchAttributionFromUrl();" in start_block
    assert "entrypoint: includeLaunchAttribution ? _launchContext.entrypoint : null" in start_block
    assert "returnclock_decision_id: includeLaunchAttribution ? _launchContext.returnclockDecisionId : null" in start_block
    assert "returnclock_delivery_id: includeLaunchAttribution ? _launchContext.returnclockDeliveryId : null" in start_block
    assert "notification_id: includeLaunchAttribution ? _launchContext.notificationId : null" in start_block
    assert "source: 'telegram_webapp'" not in start_block


def test_analytics_v2_session_lifecycle_preserves_short_backgrounding_and_splits_inactivity():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const AnalyticsController = (() => {", 1)[1].split(
        "window.__analytics = AnalyticsController;",
        1,
    )[0]
    visibility_block = block.split("document.addEventListener('visibilitychange', () => {", 1)[1].split(
        "});",
        1,
    )[0]
    visible_block = block.split("const _handleVisible = () => {", 1)[1].split(
        "// lifecycle",
        1,
    )[0]
    end_block = block.split(
        "const _sendEnd = (endedAtMs = null, endReason = null) => {",
        1,
    )[1].split(
        "const _startCurrentSession = () => {",
        1,
    )[0]
    fresh_block = block.split("const _startFreshSession = () => {", 1)[1].split(
        "const _handleVisible = () => {",
        1,
    )[0]

    assert "const SESSION_INACTIVITY_MS = 30 * 60 * 1000;" in block
    assert "_hiddenAt = Date.now();" in visibility_block
    assert "_flushUpdate();" in visibility_block
    assert "_sendEnd();" not in visibility_block
    assert "hiddenFor >= SESSION_INACTIVITY_MS" in visible_block
    assert "_sendEnd(hiddenStartedAt, 'background_inactivity');" in visible_block
    assert "_startFreshSession();" in visible_block
    assert "const _handlePageExit = () => _sendEnd(" in block
    assert "Number.isFinite(_hiddenAt) ? _hiddenAt : Date.now()" in block
    assert "window.addEventListener('pagehide', _handlePageExit)" in block
    assert "window.addEventListener('beforeunload', _handlePageExit)" in block
    assert "window.addEventListener('pageshow', _handleVisible)" in block
    assert "if (_ended || !_started) return;" in end_block
    assert "ended_at: new Date(clientEndedAt).toISOString()" in end_block
    assert "metadata: endReason ? { end_reason: endReason } : {}" in end_block
    assert "if (startWasConfirmed)" in end_block
    assert "else if (pendingStart)" in end_block
    assert "pendingStart.then((result) => {" in end_block
    assert "if (result?.started) _dispatchEnd(url, payload);" in end_block
    assert "if (!navigator.sendBeacon(url, blob))" in block
    assert "const beaconSafe = /(?:[?&])(?:_auth|user_id)=/.test(url);" in block
    assert "if (beaconSafe && typeof navigator.sendBeacon === 'function')" in block
    assert "_forgetSessionId(sessionId);" in end_block
    assert "_screens = currentScreen ? [{ screen: currentScreen, ts: Date.now() }] : [];" in fresh_block
    assert "_battlesPlayed = 0;" in fresh_block
    assert "_casesOpened = 0;" in fresh_block


def test_analytics_v2_fresh_session_does_not_reuse_launch_attribution():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const AnalyticsController = (() => {", 1)[1].split(
        "window.__analytics = AnalyticsController;",
        1,
    )[0]
    start_block = block.split("const _startCurrentSession = () => {", 1)[1].split(
        "const _startFreshSession = () => {",
        1,
    )[0]
    fresh_block = block.split("const _startFreshSession = () => {", 1)[1].split(
        "const _handleVisible = () => {",
        1,
    )[0]

    assert "let _launchAttributionConsumed = false;" in block
    assert "const includeLaunchAttribution = !_launchAttributionConsumed;" in start_block
    assert "_launchAttributionConsumed = true;" in start_block
    assert "_clearLaunchAttributionFromUrl();" in start_block
    assert "includeLaunchAttribution ? _launchContext.entrypoint : null" in start_block
    assert "includeLaunchAttribution ? _launchContext.returnclockDecisionId : null" in start_block
    assert "includeLaunchAttribution ? _launchContext.returnclockDeliveryId : null" in start_block
    assert "includeLaunchAttribution ? _launchContext.notificationId : null" in start_block
    assert "_launchAttributionConsumed = false" not in fresh_block


def test_analytics_v2_start_is_idempotent_and_screen_events_remain_ordered():
    source = Path("webapp/index.html").read_text(encoding="utf-8")
    block = source.split("const AnalyticsController = (() => {", 1)[1].split(
        "window.__analytics = AnalyticsController;",
        1,
    )[0]
    start_block = block.split("const _startCurrentSession = () => {", 1)[1].split(
        "const _startFreshSession = () => {",
        1,
    )[0]
    session_id_block = block.split("const _getSessionId = () => {", 1)[1].split(
        "const _forgetSessionId = (sessionId) => {",
        1,
    )[0]
    public_api = block.split("return {", 1)[1]

    assert "if (_started && !_ended) return _startRequest;" in start_block
    assert "_startRequest = _postJSONChecked('/api/analytics/session/start', payload)" in start_block
    assert "if (!result?.started) throw new Error('analytics_session_not_started');" in start_block
    assert "_startRetryTimer = setTimeout(_startCurrentSession, 5000);" in start_block
    assert "start: _startCurrentSession" in public_api
    assert "_screens.push({ screen: name, ts: Date.now() });" in public_api
    assert "_screens.slice(-200)" in block
    assert "if (_screens.length > 250) _screens = _screens.slice(-200);" in public_api
    assert "sessionStorage.getItem('ea_session_id')" not in session_id_block
    assert "if (_ended || !_started || !_startConfirmed) return;" in block


def test_clean_arena_sfx_are_registered_and_triggered():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")

    for audio_id, filename in [
        ("arena-sfx-player-turn-start", "player_turn_start.wav"),
        ("arena-sfx-victory", "victory.wav"),
        ("arena-sfx-defeat", "defeat.wav"),
        ("arena-sfx-surrender", "surrender.wav"),
        ("arena-sfx-low-time-tick", "low_time_tick.wav"),
    ]:
        assert f'id="{audio_id}"' in markup
        assert f"../DesignAssets/Sounds/arena/{filename}" in markup

    assert "playerTurnStart: 'arena-sfx-player-turn-start'" in source
    assert "victory: 'arena-sfx-victory'" in source
    assert "defeat: 'arena-sfx-defeat'" in source
    assert "surrender: 'arena-sfx-surrender'" in source
    assert "lowTimeTick: 'arena-sfx-low-time-tick'" in source
    assert "maybePlayPlayerTurnStartSfx(state);" in source
    assert "playArenaSfx('playerTurnStart', { volume: 0.5 });" in source
    assert "maybePlayLowTimeTickSfx(state, timeRemaining);" in source
    assert "if (!state.is_my_turn || turnDuration < 10) return;" in source
    assert "remainingSeconds < 5" in source
    assert "playArenaSfx('lowTimeTick', { volume: 0.46 });" in source
    assert "const resultSfx = getBattleResultSfx(outcome);" in source
    assert "playArenaSfx(resultSfx, { volume: 0.72 });" in source
