from pathlib import Path


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

    assert "status-icon-shield',  'shield.png',      'icon-top-left'" in status_block
    assert "status-icon-taunt',   'provocation.png', 'icon-bottom-right'" in status_block
    assert "status-icon-frozen', 'freeze.png', 'icon-top-right'" in status_block
    assert "status-icon-asleep', 'asleep.png', 'icon-top-center'" in status_block

    assert ".status-icon-container.icon-top-center" in styles
    assert ".status-icon-taunt .status-icon" in styles


def test_card_info_renders_mechanic_descriptions_not_raw_codes():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")
    markup = Path("webapp/arena.html").read_text(encoding="utf-8")
    info_block = source.split("function openCardInfo(card)", 1)[1].split("function closeCardInfo()", 1)[0]

    assert "card.mechanics_desc" in info_block
    assert "parseMechanic(m)" in info_block
    assert "mechanic-detail-description" in info_block
    assert "mechanic-detail-img" in info_block
    assert "chip.textContent = m" not in info_block
    assert '<article class="battle-modal card-info-modal" id="card-info-modal"' in markup
    assert 'card-info-modal-card' not in markup


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


def test_friend_invite_accept_failures_make_invite_terminal():
    source = Path("web/server.py").read_text(encoding="utf-8")
    handler_block = source.split("async def friend_invite_respond_handler", 1)[1].split(
        "async def friend_invite_pending_handler",
        1,
    )[0]
    compact = "".join(handler_block.split())

    assert "async def _fail_friend_invite_accept" in handler_block
    assert 'await db.update_invite_status(invite_id, "failed"' in handler_block
    assert '_fail_friend_invite_accept("invalid_deck_id"' in compact
    assert '_fail_friend_invite_accept("feature_unavailable"' in compact
    assert '_fail_friend_invite_accept("card_unavailable"' in compact
    assert '_fail_friend_invite_accept("deck_load_timeout"' in compact
    assert '_fail_friend_invite_accept("deck_load_failed"' in compact
    assert '_fail_friend_invite_accept("match_create_failed"' in compact


def test_friendly_invite_failed_status_stops_accept_loop():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "function friendlyInviteErrorText" in source
    assert "['declined', 'cancelled', 'expired', 'failed'].includes(d.status)" in source
    assert "d.message || friendlyInviteErrorText(d.error)" in source
    assert "d.status === 'failed'" in source
    assert "markFriendlyInviteHandled(invite.id)" in source


def test_arena_waits_for_both_players_before_unlocking_battle_ui():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function updateArenaWaitingOverlay" in source
    assert "match_status === 'waiting_for_players'" in source
    assert "socket.on('match_waiting'" in source
    assert "socket.on('match_ready'" in source


def test_arena_haptics_are_android_only_and_cover_core_battle_actions():
    source = Path("webapp/arena.js").read_text(encoding="utf-8")

    assert "function arenaHaptic(style, options = {})" in source
    assert "window.ExtraArenaApp" in source
    assert "if (!isAndroidArenaShell()) return;" in source
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
