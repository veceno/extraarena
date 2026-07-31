from pathlib import Path


INDEX = Path("webapp/index.html")
ARENA_HTML = Path("webapp/arena.html")
ARENA_JS = Path("webapp/arena.js")
ARENA_CSS = Path("webapp/arena-styles.css")
SERVER = Path("web/server.py")
ONBOARDING = Path("onboarding_tutorial.py")


def test_main_onboarding_gate_uses_approved_midoria_copy_and_actions():
    source = INDEX.read_text(encoding="utf-8")
    gate_block = source.split("const OnboardingGate", 1)[1].split("const MenuTourOverlay", 1)[0]

    assert "Сразу в бой. Меню подождет." in gate_block
    assert "Цель простая: у героя напротив должно стать 0 HP." in gate_block
    assert "Начать бой" in gate_block
    assert "Уже есть аккаунт? Войди и продолжим с твоего прогресса." in gate_block
    assert "Войти в аккаунт" in gate_block
    assert "/DesignAssets/MidoriaOnboardingGuide.png" in source
    assert "Запускаем..." in gate_block
    assert "starting={onboardingStarting}" in source
    assert "startOnboardingBattle" in source
    assert "/api/onboarding/tutorial/start" in source
    assert "buildArenaRedirectUrl" in source


def test_menu_tour_highlights_only_required_sections_and_finishes_to_newbie_path():
    source = INDEX.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")
    menu_tour_block = source.split("const MenuTourOverlay", 1)[1].split("const NewbiePathPanel", 1)[0]

    assert "data-onboarding-target={i===2?'arena':i===1?'collection':undefined}" in source
    assert "data-onboarding-target={t.key === 'decks' ? 'decks' : undefined}" in source
    assert "data-onboarding-target=\"cases\"" in source
    assert "data-onboarding-target=\"wins_to_case\"" in source
    assert "Арена — место боёв. У тебя уже есть готовая стартовая колода" in server
    assert "Коллекция — все твои карты." in server
    assert "Колода — твой план на бой." in server
    # Expanded informational tour: reward / wins-to-case / cases / chat / final.
    assert "9 стартовых карт" in server
    assert "сколько побед осталось до кейса" in server
    assert "Кейсы дают новые карты и ресурсы" in server
    assert "https://t.me/extraarena_chat" in ONBOARDING.read_text(encoding="utf-8")
    assert "ONBOARDING_CHAT_URL" in server
    assert "Маршрут простой" in server
    assert "Открыть Путь новичка" in source
    assert "/api/onboarding/complete" in source
    assert "setCollectionTabIntent('decks')" in source
    assert "onboardingTabIntent={collectionTabIntent}" in source
    assert "NewbiePathPanel" in source
    assert "backdropFilter:'blur" not in menu_tour_block
    assert "data-onboarding-spotlight" in menu_tour_block
    assert "button={busy?'Сохраняю...':(step.button || 'Дальше')}" in menu_tour_block
    assert "disabled={busy}" in menu_tour_block
    assert "aria-busy={busy?'true':'false'}" in menu_tour_block
    assert "onboardingTourSavingRef" in source
    assert "setOnboardingTourSaving(true)" in source
    assert "await refreshOnboarding();" in source
    assert "busy={onboardingTourSaving}" in source


def test_newbie_path_action_reserves_mobile_layout_space_and_hides_under_overlays():
    source = INDEX.read_text(encoding="utf-8")
    css = source.split(".newbie-path-floating {", 1)[1].split(
        "@keyframes tipSwap",
        1,
    )[0]
    app_block = source.split("const App = () =>", 1)[1]
    overlay_line = source.split("const hasOverlay =", 1)[1].split(";", 1)[0]
    floating_block = source.split(
        "{onboarding?.completed && newbiePathRemaining > 0 && !hasOverlay && (",
        1,
    )[1].split("{!showAI && !showGenerator", 1)[0]

    assert "@media (max-width: 520px)" in source
    assert "position: static;" in css
    assert "right: auto;" in css
    assert "bottom: auto;" in css
    assert "width: auto;" in css
    assert "height: 46px;" in css
    assert "min-height: 46px;" in css
    assert "margin: 6px calc(12px + var(--ea-safe-right)) 6px calc(12px + var(--ea-safe-left));" in css
    assert "align-self: stretch;" in css
    assert "flex-shrink: 0;" in css
    assert "border-radius: 12px;" in css
    assert ".newbie-path-floating:focus-visible" in source
    assert 'data-newbie-path-floating="true"' in floating_block
    assert "Открыть Путь новичка: осталось задач" in floating_block
    assert "newbie-path-floating__label" in floating_block
    assert app_block.index('data-newbie-path-floating="true"') < app_block.index("{/* Main content */}")
    assert app_block.index('data-newbie-path-floating="true"') < app_block.index("<BottomNav")

    for overlay_signal in [
        "squadsOpen",
        "communityOpen",
        "showAI",
        "showGenerator",
        "showQuests",
        "showMenu",
        "showSettings",
        "showBattlePick",
        "showGameMode",
        "showMail",
        "showNews",
        "showCaseOpen",
        "showInfo",
        "showGloryPath",
        "showLeagueInfo",
        "showBattlePass",
        "showPreBattle",
        "showSupport",
        "showInvite",
        "showBattles",
        "showFriends",
        "showProfile",
        "showExtraID",
        "onboardingBlocking",
        "onboardingTourActive",
        "newbiePathOpen",
        "!!seasonResetNotice",
        "!!pendingCaseOpen",
    ]:
        assert overlay_signal in overlay_line


def test_arena_tutorial_overlay_is_wired_to_state_actions_and_feedback():
    markup = ARENA_HTML.read_text(encoding="utf-8")
    script = ARENA_JS.read_text(encoding="utf-8")
    styles = ARENA_CSS.read_text(encoding="utf-8")

    assert 'id="arena-onboarding-layer"' in markup
    assert ".arena-onboarding-layer" in styles
    assert "function updateOnboardingTutorialFromState" in script
    assert "state.is_onboarding_tutorial" in script
    assert "handleOnboardingActionPayload(result)" in script
    assert "handleOnboardingActionError(error)" in script
    assert "/api/onboarding/tutorial/action" in script
    assert "goToOnboardingMenuTour" in script
    assert "data-onboarding-target=\"hand-card:" in script
    assert "data-onboarding-target=\"board-card:" in script
    assert "selectedCard ? ['#player-board-zone .board-slot']" in script
    assert "interactionMode.type === 'ATTACK'" in script
    assert "if (allowed.type === 'complete') return ['.arena-onboarding-victory-action'];" in script
    assert "function sendOnboardingTutorialAction(action" in script
    assert "return sendOnboardingTutorialAction({" in script
    assert "shouldBypassPrebattleForOnboarding" in script
    assert "arena-onboarding-step" in styles
    assert "let onboardingFollowupMessage = '';" in script
    assert "let onboardingAutoAdvanceTimer = null;" in script
    assert "let onboardingFollowupReady = false;" in script
    assert "let onboardingMenuTourLeaving = false;" in script
    assert "function scheduleOnboardingFollowup(message, stepIndex)" in script
    assert "function scheduleOnboardingAutoAdvance(tutorial)" in script
    assert "sendOnboardingTutorialAction({ type: 'auto_continue' }" in script
    assert "Number(tutorial?.auto_advance_delay_ms || ONBOARDING_AUTO_ADVANCE_DELAY_MS)" in script
    assert "}, 2000);" in script
    assert "scheduleOnboardingFollowup(afterMessage, getOnboardingTutorial()?.step_index);" in script
    assert "onboardingFollowupMessage = normalized;" in script
    assert "onboardingFollowupReady = false;" in script
    assert "onboardingFollowupReady = true;" in script
    assert "if (onboardingFollowupReady) {" in script
    assert "currentMessage = stagedFollowupMessage;" in script
    assert "const isWaitingForFollowup = !onboardingFeedbackMessage && stagedFollowupMessage && !onboardingFollowupReady;" in script
    assert "if (!onboardingFeedbackMessage && !isWaitingForFollowup && tutorial.hint) {" in script
    assert "scheduleOnboardingAutoAdvance(getOnboardingTutorial());" in script
    assert "const displayStep = Number(tutorial.display_step ?? stepIndex);" in script
    assert "const displayStepsTotal = Number(tutorial.display_steps_total || finalStep);" in script
    assert "player_steps_total" not in script
    assert "Демо ${displayStep}/${displayStepsTotal}" in script
    assert "Шаг ${displayStep}/${displayStepsTotal}" in script
    assert "tutorial.previous_message" in script
    assert "previous.className = 'arena-onboarding-text is-stacked';" in script
    assert "current.className = 'arena-onboarding-followup';" in script
    board_card_block = script.split("function createBoardCardElement", 1)[1].split(
        "// LEGAL ACTIONS: Определяем, может ли юнит атаковать",
        1,
    )[0]
    assert "const unitHpValue = Math.max(0, Number(card.hp ?? card.hp_current ?? card.health ?? 0) || 0);" in board_card_block
    assert "cardDiv.classList.add('unit-defeated', 'card-disabled-board');" in board_card_block
    assert "status.className = 'arena-onboarding-status';" in script
    assert "status.textContent = 'Ход противника...';" in script
    assert "button.textContent = stepIndex >= finalStep ? 'В меню' : 'Понятно';" in script
    assert "tutorial.is_auto_step" in script
    assert "arena-onboarding-followup" in styles
    assert ".arena-onboarding-text.is-stacked" in styles
    assert ".arena-onboarding-status" in styles
    assert ".arena-onboarding-victory" in styles
    assert ".arena-onboarding-layer.is-victory" in styles
    assert "arena-onboarding-victory-action" in script
    assert "Учебный бой завершен" in script
    assert "Победа" in script
    assert ".board-unit-card.unit-defeated" in styles
    assert "animation: onboardingTargetPulse 2.2s ease-in-out infinite;" in styles
    assert "if (isOnboardingTutorialState(state))" in script
    assert "button.hidden = true;" in script
    assert "button.hidden = false;" in script
    assert "if (isOnboardingTutorialState()) return;" in script
    assert "if (isOnboardingTutorialState()) {" in script
    assert "if (isOnboardingTutorialState()) return;\n    showTalkieFullscreen(data);" in script


def test_arena_tutorial_click_guard_blocks_card_info_and_allows_modal_close():
    script = ARENA_JS.read_text(encoding="utf-8")
    guard_block = script.split("function installOnboardingClickGuard()", 1)[1].split(
        "async function sendOnboardingTutorialControl",
        1,
    )[0]

    assert "activeBattleModal && event.target.closest('#battle-modal-layer')" in guard_block
    assert ".card-info-btn" in guard_block
    assert guard_block.index(".card-info-btn") < guard_block.index(
        "targetMatchesOnboardingSelectors(event.target, allowedSelectors)"
    )


def test_arena_tutorial_hides_card_info_controls_and_guards_modal_open():
    script = ARENA_JS.read_text(encoding="utf-8")
    helper_block = script.split("function shouldShowCardInfoControls()", 1)[1].split(
        "function shouldBypassPrebattleForOnboarding",
        1,
    )[0]
    hand_card_block = script.split("function createHandCardElement", 1)[1].split(
        "function addStatusIcons",
        1,
    )[0]
    board_card_block = script.split("function createBoardCardElement", 1)[1].split(
        "// ДОБАВЛЕНО: Имя карты на поле",
        1,
    )[0]
    modal_block = script.split("function openCardInfo(card)", 1)[1].split(
        "function closeCardInfo",
        1,
    )[0]
    tutorial_update_block = script.split("function updateOnboardingTutorialFromState(state)", 1)[1].split(
        "function showOnboardingTutorialFeedback",
        1,
    )[0]

    assert "return !isOnboardingTutorialState();" in helper_block
    assert "const showCardInfoControls = shouldShowCardInfoControls();" in hand_card_block
    assert "cardType === 'potion' && showCardInfoControls" in hand_card_block
    assert hand_card_block.count("if (showCardInfoControls) {") >= 1
    assert "const showCardInfoControls = shouldShowCardInfoControls();" in board_card_block
    assert "if (showCardInfoControls) {" in board_card_block
    assert modal_block.index("if (isOnboardingTutorialState()) return;") < modal_block.index(
        "openBattleModal('card-info')"
    )
    assert "if (activeBattleModal === 'card-info') {" in tutorial_update_block
    assert "closeBattleModal();" in tutorial_update_block


def test_arena_tutorial_wrong_click_keeps_auto_advance_on_enemy_turn():
    script = ARENA_JS.read_text(encoding="utf-8")
    feedback_block = script.split("function showOnboardingTutorialFeedback", 1)[1].split(
        "function handleOnboardingActionPayload",
        1,
    )[0]

    assert "const shouldKeepAutoAdvance = Boolean(tutorial?.is_auto_step);" in feedback_block
    assert "if (!shouldKeepAutoAdvance) {\n    clearOnboardingAutoAdvance();" in feedback_block
    assert "scheduleOnboardingAutoAdvance(tutorial);" not in feedback_block
    assert "scheduleOnboardingAutoAdvance(getOnboardingTutorial());" not in feedback_block


def test_arena_tutorial_blocks_timer_keyboard_shortcut():
    script = ARENA_JS.read_text(encoding="utf-8")
    keydown_block = script.split("timerBtn.addEventListener('keydown', function(e)", 1)[1].split(
        "openTurnTimerModal();",
        1,
    )[0] + "openTurnTimerModal();"

    assert "if (isOnboardingTutorialState())" in keydown_block
    assert keydown_block.index("if (isOnboardingTutorialState())") < keydown_block.index("openTurnTimerModal();")


def test_arena_shell_cache_busts_battle_assets_for_telegram_webapp():
    markup = ARENA_HTML.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")

    version = "arena-newcards2606-20260627-sfx-1-returnclock-v2"
    assert f'content="{version}"' in markup
    assert f"safe-area.js?v={version}" in markup
    assert f"arena-styles.css?v={version}" in markup
    assert f"window.__EXTRA_ARENA_ASSET_VERSION__ = '{version}';" in markup
    assert f"analytics-v2.js?v={version}" in markup
    assert f"arena.js?v={version}" in markup
    assert 'src="arena.js"></script>' not in markup
    assert 'href="arena-styles.css">' not in markup

    assert 'NO_STORE_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate"}' in server
    assert '"analytics-v2.js",' in server
    assert '"arena.js",' in server
    assert '"arena-styles.css",' in server
    assert '"safe-area.js",' in server
    assert "if relative_path in BATTLE_SHELL_STATIC_FILES:" in server
    assert "return _static_text_response(file_path, headers=NO_STORE_CACHE_HEADERS)" in server


def test_arena_tutorial_sfx_use_mp3_and_do_not_overlap():
    markup = ARENA_HTML.read_text(encoding="utf-8")
    script = ARENA_JS.read_text(encoding="utf-8")

    for sound in ("start", "step", "confirm", "complete", "victory", "blocked"):
        assert f'id="arena-sfx-onboarding-{sound}"' in markup
        assert f"../DesignAssets/Sounds/onboarding/onboarding-{sound}.mp3" in markup
    assert 'type="audio/mpeg"' in markup
    assert "onboarding-card-play" not in markup
    assert "onboarding-taunt" not in markup
    assert "onboarding-turn" not in markup

    for sound_key in (
        "onboardingStart",
        "onboardingStep",
        "onboardingConfirm",
        "onboardingComplete",
        "onboardingVictory",
        "onboardingBlocked",
    ):
        assert sound_key in script
    assert "onboardingCardPlay" not in script
    assert "onboardingTaunt" not in script
    assert "onboardingTurn" not in script

    assert "let onboardingSfxCurrent = null;" in script
    assert "let onboardingVictorySfxPlayed = false;" in script
    assert "function stopOnboardingSfx()" in script
    assert "function playOnboardingVictoryCue()" in script
    assert "stopOnboardingSfx();" in script
    assert "const audio = document.getElementById(audioId);" in script
    assert "if (isOnboardingTutorialState() && !String(audioId).includes('onboarding')) return false;" in script
    assert "if (isOnboardingTutorialState()) return;" in script
    assert "playOnboardingStepCue(onboardingTutorial);" in script
    assert "playOnboardingActionCue(payload);" in script
    assert "const ONBOARDING_TUTORIAL_FINAL_STEP = 8;" in script
    assert "playOnboardingSfx('onboardingVictory'" in script
    assert "playOnboardingSfx('onboardingConfirm'" in script
    assert "playOnboardingSfx('onboardingBlocked'" in script
    assert "const actionType = String(actionBody.type || 'action');" in script
    assert "if (userId != null && actionBody.user_id == null) actionBody.user_id = userId;" in script


def test_arena_victory_menu_button_completes_tutorial_before_redirect():
    script = ARENA_JS.read_text(encoding="utf-8")
    complete_block = script.split("async function finishOnboardingTutorialForMenu()", 1)[1].split(
        "function buildOnboardingMenuTourUrl()",
        1,
    )[0]
    goto_block = script.split("async function goToOnboardingMenuTour()", 1)[1].split(
        "function isArenaAndroidShell()",
        1,
    )[0]

    assert "type: 'complete'" in complete_block
    assert "client_action_id: makeClientActionId('tutorial_complete_menu')" in complete_block
    assert "if (userId != null) actionBody.user_id = userId;" in complete_block
    assert "/api/onboarding/tutorial/action" in complete_block
    assert "event.target.closest('.arena-onboarding-victory')" in script
    assert "const tutorial = onboardingMenuTourLeaving ? onboardingTutorial : getOnboardingTutorial();" in script
    assert "if (onboardingMenuTourLeaving) {" in script
    assert "await finishOnboardingTutorialForMenu();" in goto_block
    assert "const finalTutorial = getOnboardingTutorial();" in goto_block
    assert "onboardingMenuTourLeaving = true;" in goto_block
    assert "onboardingTutorial = null;" in goto_block
    assert "setOnboardingLayerActive(false);" in goto_block
    assert "window.location.replace(buildOnboardingMenuTourUrl());" in goto_block


def test_onboarding_server_guards_ordering_and_newbie_rewards():
    server = SERVER.read_text(encoding="utf-8")
    source = INDEX.read_text(encoding="utf-8")
    claim_user_case_block = server.split("async def _claim_user_case_opening", 1)[1].split(
        "async def case_reroll_handler",
        1,
    )[0]

    assert "menu_tour_not_ready" in server
    assert "unexpected_menu_step" in server
    assert "onboarding_not_ready" in server
    assert "claim_newbie_path_task" in server
    assert "task_not_completed" in server
    assert "join_telegram_channel" in server
    assert "https://t.me/extraarena" in server
    assert "_check_telegram_channel_membership" in server
    assert "getChatMember" in server
    assert "_newbie_path_tasks_for_context" in server
    assert "/api/onboarding/newbie-path/progress" in source
    assert "telegramChannelTaskOpened" in source
    assert "window.openExternalLink?.(task.action_url)" in source
    assert "'Проверить'" in source
    assert "task.id === 'join_telegram_channel'" in source
    assert 'task_id != "view_new_card"' in server
    assert "newbie_path_task_claimed" in server
    assert "const [newbiePathClaiming, setNewbiePathClaiming] = React.useState({});" in source
    assert "const newbiePathClaimingRef = React.useRef({});" in source
    assert "const anyBusy = Object.keys(claiming || {}).length > 0;" in source
    assert "const taskBusy = !!claiming[task.id];" in source
    assert "const taskLocked = anyBusy && !taskBusy;" in source
    assert "disabled={task.claimed || taskBusy || taskLocked}" in source
    assert "Object.keys(newbiePathClaimingRef.current || {}).length > 0" in source
    assert "newbiePathClaimingRef.current[taskId]" in source
    assert "setNewbiePathClaiming(prev => ({...prev, [taskId]: true}));" in source
    assert "if (data.newbie_path) setOnboarding(prev => prev ? {...prev, newbie_path: data.newbie_path} : prev);" in source
    assert "const grantedAmount = Number(data.granted_amount || 0);" in source
    assert "window.reloadFreshProfile().then(profileData => { if (profileData) setProfile(profileData); });" in source
    assert "throw new Error(data.message || data.error || 'Не удалось забрать награду')" in source
    assert "window.showToast?.(error?.message || 'Не удалось забрать награду'" in source
    assert "except Exception:" in claim_user_case_block
    assert 'await db.mark_newbie_path_task(user_id, "open_starter_case", claimed=False)' in claim_user_case_block
    assert "newbie path user_case completion failed" in claim_user_case_block


def test_onboarding_transition_analytics_are_server_canonical_only():
    source = INDEX.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")

    assert "__analytics?.onboarding('welcome_completed'" not in source
    assert "__analytics?.onboarding('mandatory_onboarding_completed'" not in source
    assert "CANONICAL_ONBOARDING_TRANSITION_EVENTS" in server
    assert 'metadata={"source": "onboarding_gate"}' in server
    assert 'metadata={"source": "menu_tour"}' in server


def test_payload_auth_helper_accepts_query_auth_for_arena_post_actions():
    server = SERVER.read_text(encoding="utf-8")
    helper = server.split("async def require_user_id_from_payload", 1)[1].split(
        "def _require_user_id_from_init_data_str",
        1,
    )[0]

    assert "_request_auth_token(request)" in helper
    assert "request.rel_url.query.get(\"user_id\")" in helper
