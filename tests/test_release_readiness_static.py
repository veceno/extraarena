import hashlib
from pathlib import Path


ROOT = Path(".")


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_startup_schema_migration_is_explicit_outside_development():
    config = _read("infrastructure/config.py")
    main = _read("main.py")
    run_web = _read("run_web.py")
    database = _read("infrastructure/database.py")

    assert "auto_migrate_on_start" in config
    assert "if settings.auto_migrate_on_start:" in main
    assert "await db.verify_schema_ready()" in main
    assert "if settings.auto_migrate_on_start:" in run_web
    assert "async def verify_schema_ready" in database


def test_battle_result_unique_index_migration_does_not_delete_history_rows():
    database = _read("infrastructure/database.py")
    block = database.split("async def _ensure_battle_results_table", 1)[1].split(
        "async def save_battle_result",
        1,
    )[0]

    assert "DELETE FROM battle_results" not in block
    assert "battle_results_match_id_unique_idx" in block
    assert "manual migration" in block


def test_legacy_dice_runtime_surfaces_are_removed():
    checks = {
        "main.py": ["_dice_notifications_task", "dice_ready"],
        "web/server.py": ["/api/dice", "dice_status_handler", "dice_roll_handler"],
        "infrastructure/database.py": [
            "check_dice_ready_notifications",
            "mark_dice_notification_sent",
            "notif_dice",
        ],
        "infrastructure/notifications.py": ["dice_ready"],
        "bot/handlers.py": ["_check_and_notify_dice_ready", "get_dice_status"],
        "webapp/main.js": ["loadDiceStatus", "rollDice", "notif_dice"],
    }

    for path, forbidden in checks.items():
        source = _read(path)
        for needle in forbidden:
            assert needle not in source, f"{needle!r} remains in {path}"


def test_health_readiness_contract_and_start_script_body_validation():
    server = _read("web/server.py")
    start = _read("start.sh")

    assert '"service": "extraarena-webapp"' in server
    assert 'app.router.add_get("/ready", readiness_check)' in server
    assert 'app.router.add_get("/favicon.ico", favicon_handler)' in server
    assert "components" in server
    assert "health_ready()" in start
    assert 'payload.get("status") == "ok"' in start
    assert 'payload.get("service") == "extraarena-webapp"' in start


def test_start_script_pins_single_process_web_state():
    start = _read("start.sh")

    assert 'export WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"' in start


def test_start_script_preserves_logs_and_validates_stale_pidfile():
    start = _read("start.sh")

    assert "pid_matches_project()" in start
    assert 'if pid_matches_project "$OLD_PID"; then' in start
    assert "$PID_FILE указывает на чужой или уже завершенный PID $OLD_PID" in start
    assert 'EXTRAARENA_TRUNCATE_LOG:-false' in start
    assert "os.O_APPEND" in start


def test_start_script_fails_when_readiness_never_passes():
    start = _read("start.sh")

    assert "READY=0" in start
    assert 'READY=1' in start
    assert 'if [ "$READY" -ne 1 ]; then' in start
    assert "Веб-сервер не прошёл readiness-check" in start
    assert 'tail -n 80 "$LOG_FILE"' in start
    assert 'stop_pid "$PID" "неуспешно стартовавший веб-сервер"' in start


def test_web_auth_keeps_jwt_out_of_query_and_local_storage():
    index = _read("webapp/index.html")
    arena = _read("webapp/arena.js")
    main = _read("webapp/main.js")

    assert "sessionStorage.setItem(EXTRA_ID_TOKEN_SESSION_KEY" in index
    assert "localStorage.setItem('extra_id_token'" not in index
    assert "localStorage.setItem('extra_id_token'" not in main
    assert 'localStorage.setItem("extra_id_token"' not in main
    assert 'const legacyToken = localStorage.getItem("extra_id_token")' in main
    assert 'localStorage.removeItem("extra_id_token")' in main
    assert "installMainJwtQueryAuthHeaderBridge" in main
    assert "buildMainArenaRedirectUrl" in main
    assert "auth.type === 'auth' && looksLikeJwtBearer(auth.value) && isSameOriginApiPath(path)" in index
    assert "function liftJwtAuthFromJsonBody(nextInit, headers)" in index
    assert "const bodyToken = liftJwtAuthFromJsonBody(nextInit, headers);" in index
    assert "if (looksLikeJwtBearer(sanitized._auth)) delete sanitized._auth;" in index
    assert "if (looksLikeJwtBearer(sanitized.auth)) delete sanitized.auth;" in index
    assert "headers.set('Authorization', `Bearer ${bearerToken}`)" in index
    assert "storageKey(url)" in index and "searchParams.delete('_auth')" in index
    assert "buildArenaRedirectUrl('/arena?id='" in index
    assert "authParam = data.authData" not in index

    assert "looksLikeArenaJwtBearer(authToken) && isSameOriginArenaApiPath(path)" in arena
    assert "function liftArenaJwtAuthFromJsonBody(nextInit, headers)" in arena
    assert "const bodyToken = liftArenaJwtAuthFromJsonBody(nextInit, headers);" in arena
    assert "headers.set('Authorization', `Bearer ${bearerToken}`)" in arena
    assert "sessionStorage.setItem('extra_id_token', authToken)" in arena

    assert "function liftMainJwtAuthFromJsonBody(nextInit, headers)" in main
    assert "const bodyToken = liftMainJwtAuthFromJsonBody(nextInit, headers);" in main
    assert "headers.set(\"Authorization\", `Bearer ${bearerToken}`)" in main


def test_main_webapp_uses_precompiled_bundle_instead_of_runtime_babel():
    index = _read("webapp/index.html")
    compiler = _read("scripts/precompile_webapp_index.py")
    compiled_path = ROOT / "webapp/index.compiled.js"
    compiled = compiled_path.read_text(encoding="utf-8")
    compiled_hash = hashlib.sha256(compiled_path.read_bytes()).hexdigest()[:12]

    assert "@babel/standalone" not in index
    assert 'type="text/babel"' not in index
    assert 'id="extraarena-main-jsx-source"' in index
    assert 'type="application/x-extraarena-jsx-source"' in index
    assert f'data-compiled-src="index.compiled.js?v={compiled_hash}"' in index
    assert f'<script src="index.compiled.js?v={compiled_hash}"></script>' in index
    assert "ReactDOM.createRoot(document.getElementById('root')).render(<App/>);" in index
    assert 'ReactDOM.createRoot(document.getElementById("root")).render' in compiled
    assert "esbuild@0.25.5" in compiler
    assert "--check" in compiler


def test_release_ui_placeholders_are_removed_from_visible_paths():
    index = _read("webapp/index.html")
    main = _read("webapp/main.js")

    forbidden_main = [
        "Магазин ExtraPass скоро будет доступен",
        "TODO: Открыть магазин ExtraPass",
        "TODO: Логика размещения рекламы",
        "ads-place-btn",
        "Обработчики для кнопок друзей (заглушки)",
        "Обработчики для писем (заглушки)",
    ]
    forbidden_index = [
        "Раздел в разработке",
        "Войны сквадов скоро",
        "Опрос временно недоступен",
        "Подробное описание режима появится позже",
        "ExtraPass временно недоступен",
        "Вход будет доступен",
        "Вход позже",
        "Будет доступно позже",
        "Будет доступна позже",
        "Сквадовые топы появятся отдельным обновлением",
    ]

    for needle in forbidden_main:
        assert needle not in main, f"{needle!r} remains in webapp/main.js"
    for needle in forbidden_index:
        assert needle not in index, f"{needle!r} remains in webapp/index.html"


def test_removed_shards_mechanic_is_absent_from_game_surfaces():
    checks = {
        "infrastructure/case_system.py": ["limited_shards", "shards_chance", "shards_range"],
        "infrastructure/case_config.py": ["limited_shards", "осколк"],
        "webapp/main.js": ["limited_shards", 'type: "shards"', "осколк"],
        "webapp/index.html": ["limited_shards", 'type: "shards"', "осколк"],
    }
    compiled_path = ROOT / "webapp/index.compiled.js"
    if compiled_path.exists():
        checks["webapp/index.compiled.js"] = ["limited_shards", "type:\"shards\"", "осколк"]

    for path, forbidden in checks.items():
        source = _read(path)
        for needle in forbidden:
            assert needle not in source, f"{needle!r} remains in {path}"


def test_admin_player_and_mail_rendering_escape_user_strings():
    main = _read("webapp/main.js")
    players_block = main.split("const renderPlayers = (players) => {", 1)[1].split(
        "// Переустанавливаем обработчики действий",
        1,
    )[0]
    mail_block = main.split("function renderMail(mailList) {", 1)[1].split(
        "// Вспомогательная функция для форматирования времени",
        1,
    )[0]

    assert "const playerName = escapeHtml(" in players_block
    assert "${playerName}" in players_block
    assert "${p.first_name || p.username" not in players_block

    for variable in ("mailIcon", "mailSender", "mailSubject", "mailPreview"):
        assert f"const {variable} = escapeHtml(" in mail_block
        assert f"${{{variable}}}" in mail_block
    assert "${mail.sender ||" not in mail_block
    assert "${mail.subject ||" not in mail_block
    assert "${mailContent}" not in mail_block


def test_mail_rendering_uses_delegated_click_handlers_without_clone_timeout():
    main = _read("webapp/main.js")
    mail_block = main.split("function renderMail(mailList) {", 1)[1].split(
        "// Вспомогательная функция для форматирования времени",
        1,
    )[0]

    assert "cloneNode" not in mail_block
    assert "setTimeout(" not in mail_block
    assert "ensureMailEventDelegation()" in mail_block
    assert "mailListElement.addEventListener(\"click\"" in main


def test_datetime_utcnow_is_not_used_in_backend_code():
    for path in ("infrastructure/database.py", "infrastructure/extraid_database.py"):
        assert "datetime.utcnow(" not in _read(path)


def test_dynamic_alter_table_helpers_validate_identifiers():
    database = _read("infrastructure/database.py")
    extraid_database = _read("infrastructure/extraid_database.py")

    assert "_validate_schema_identifier(table)" in database
    assert "_validate_schema_identifier(column_name)" in database
    assert "_validate_schema_identifier(table)" in extraid_database
    assert "_validate_schema_identifier(column_name)" in extraid_database


def test_profile_cosmetics_are_api_only_without_mock_fallback():
    index = _read("webapp/index.html")

    assert "MOCK_COSMETICS" not in index
    assert "setCosmeticsError" in index
    assert "cosmetics_bad_payload" in index
    assert "return [];" in index
    assert "Оформление не загружено" in index
    assert "Случайный образ!" in index


def test_extraid_mockup_is_not_publicly_served_by_static_catch_all():
    server = _read("web/server.py")
    mockup = _read("webapp/extraid-mockup.html")

    assert 'if relative_path in {"extraid-mockup.html"}:' in server
    assert "raise web.HTTPNotFound()" in server.split(
        'if relative_path in {"extraid-mockup.html"}:',
        1,
    )[1].split('if relative_path.startswith("DesignAssets/"):', 1)[0]
    assert "cdn.tailwindcss.com" in mockup
