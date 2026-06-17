from pathlib import Path
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database, _normalize_poll_options, _parse_poll_expires_at


class _AsyncBarrier:
    def __init__(self, parties: int = 2):
        self.parties = parties
        self.count = 0
        self.event = asyncio.Event()

    async def wait(self):
        self.count += 1
        if self.count >= self.parties:
            self.event.set()
        await asyncio.wait_for(self.event.wait(), timeout=1)


class _FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        while self.conn._held_locks:
            self.conn._held_locks.pop().release()


class _FakeAcquire:
    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return _FakeConn(self.state)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, state):
        self.state = state

    def acquire(self):
        return _FakeAcquire(self.state)


class _CommunityState:
    def __init__(self):
        self.polls = {
            10: {
                "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
                "options": [{"id": 1, "text": "Yes"}, {"id": 2, "text": "No"}],
            }
        }
        self.poll_votes = {}
        self.community_posts = {
            20: {
                "id": 20,
                "post_type": "idea",
                "moderation_status": "approved",
                "status": "active",
                "upvotes": 0,
                "downvotes": 0,
            },
            30: {
                "id": 30,
                "post_type": "announcement",
                "moderation_status": "approved",
                "status": "active",
                "upvotes": 0,
                "downvotes": 0,
            },
            40: {
                "id": 40,
                "post_type": "news",
                "moderation_status": "approved",
                "status": "active",
                "upvotes": 0,
                "downvotes": 0,
            },
        }
        self.community_votes = {}
        self.post_likes = set()
        self.locks = defaultdict(asyncio.Lock)
        self.race_barriers = defaultdict(_AsyncBarrier)


class _FakeConn:
    def __init__(self, state):
        self.state = state
        self._held_locks = []

    def transaction(self):
        return _FakeTransaction(self)

    async def _maybe_hold_lock(self, key):
        lock = self.state.locks[key]
        await lock.acquire()
        self._held_locks.append(lock)

    async def execute(self, query, *args):
        if "pg_advisory_xact_lock" in query:
            await self._maybe_hold_lock(args[0])
            return "SELECT 1"
        if "DELETE FROM community_votes" in query:
            self.state.community_votes.pop((args[0], args[1]), None)
            return "DELETE 1"
        if "UPDATE community_votes SET vote_type" in query:
            self.state.community_votes[(args[0], args[1])] = args[2]
            return "UPDATE 1"
        if "INSERT INTO community_votes" in query:
            key = (args[0], args[1])
            if "ON CONFLICT" in query:
                self.state.community_votes[key] = args[2]
                return "INSERT 1"
            if key in self.state.community_votes:
                raise RuntimeError("duplicate community vote")
            self.state.community_votes[key] = args[2]
            return "INSERT 1"
        if "UPDATE community_posts" in query and ("upvotes" in query or "downvotes" in query):
            self._recompute_idea_counts(args[0])
            return "UPDATE 1"
        if "DELETE FROM post_likes" in query:
            self.state.post_likes.discard((args[0], args[1]))
            return "DELETE 1"
        if "INSERT INTO post_likes" in query:
            key = (args[0], args[1])
            if key in self.state.post_likes:
                raise RuntimeError("duplicate post like")
            self.state.post_likes.add(key)
            return "INSERT 1"
        raise AssertionError(query)

    async def fetchval(self, query, *args):
        if "pg_advisory_xact_lock" in query:
            await self._maybe_hold_lock(args[0])
            return 1
        if "SELECT 1 FROM community_poll_votes" in query:
            if not self._held_locks:
                await self.state.race_barriers["poll"].wait()
            return 1 if (args[0], args[1]) in self.state.poll_votes else None
        if "SELECT COUNT(*) FROM post_likes" in query:
            return sum(1 for post_id, _ in self.state.post_likes if post_id == args[0])
        if "SELECT 1 FROM post_likes" in query:
            return 1 if (args[0], args[1]) in self.state.post_likes else None
        raise AssertionError(query)

    async def fetchrow(self, query, *args):
        if "SELECT expires_at, options FROM community_polls" in query:
            return self.state.polls.get(args[0])
        if "INSERT INTO community_poll_votes" in query and "ON CONFLICT" in query:
            key = (args[0], args[1])
            if key in self.state.poll_votes:
                return None
            self.state.poll_votes[key] = args[2]
            return {"inserted": 1}
        if "INSERT INTO post_likes" in query and "ON CONFLICT" in query:
            key = (args[0], args[1])
            if key in self.state.post_likes:
                return None
            self.state.post_likes.add(key)
            return {"inserted": 1}
        if "SELECT vote_type FROM community_votes" in query:
            if not self._held_locks:
                await self.state.race_barriers[f"vote:{args[0]}:{args[1]}"].wait()
            vote_type = self.state.community_votes.get((args[0], args[1]))
            return {"vote_type": vote_type} if vote_type else None
        if "SELECT id FROM community_posts" in query and "post_type IN ('idea', 'bug')" in query:
            post = self.state.community_posts.get(args[0])
            if not post or post["post_type"] not in ("idea", "bug"):
                return None
            if post["moderation_status"] != "approved" or post["status"] != "active":
                return None
            return {"id": post["id"]}
        if "SELECT id FROM community_posts" in query and "post_type IN ('news', 'poll')" in query:
            post = self.state.community_posts.get(args[0])
            if not post or post["post_type"] not in ("news", "poll"):
                return None
            if post["moderation_status"] != "approved" or post["status"] != "active":
                return None
            return {"id": post["id"]}
        if "SELECT upvotes, downvotes FROM community_posts" in query:
            self._recompute_idea_counts(args[0])
            post = self.state.community_posts.get(args[0])
            return {"upvotes": post["upvotes"], "downvotes": post["downvotes"]} if post else None
        if "SELECT id FROM community_posts" in query:
            post = self.state.community_posts.get(args[0])
            if not post:
                return None
            if "post_type = 'announcement'" in query and post["post_type"] != "announcement":
                return None
            if post["moderation_status"] != "approved" or post["status"] != "active":
                return None
            return {"id": post["id"]}
        if "SELECT id FROM post_likes" in query:
            if not self._held_locks:
                await self.state.race_barriers[f"like:{args[0]}:{args[1]}"].wait()
            return {"id": 1} if (args[0], args[1]) in self.state.post_likes else None
        if "SELECT" in query and "COUNT(*) FROM community_votes" in query:
            post_id = args[0]
            return {
                "likes": sum(1 for (pid, _), value in self.state.community_votes.items() if pid == post_id and value == "like"),
                "dislikes": sum(1 for (pid, _), value in self.state.community_votes.items() if pid == post_id and value == "dislike"),
            }
        raise AssertionError(query)

    def _recompute_idea_counts(self, post_id):
        post = self.state.community_posts.get(post_id)
        if post:
            post["upvotes"] = sum(1 for (pid, _), value in self.state.community_votes.items() if pid == post_id and value == "up")
            post["downvotes"] = sum(1 for (pid, _), value in self.state.community_votes.items() if pid == post_id and value == "down")


def _fake_database():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    state = _CommunityState()
    db._pool = _FakePool(state)
    return db, state


def test_normalize_poll_options_keeps_valid_choices_with_stable_ids():
    options = _normalize_poll_options([
        {"id": "10", "text": "Да"},
        {"text": "Нет"},
        "  Может быть  ",
        {"id": 99, "text": ""},
    ])

    assert options == [
        {"id": 10, "text": "Да"},
        {"id": 2, "text": "Нет"},
        {"id": 3, "text": "Может быть"},
    ]


def test_parse_poll_expires_at_accepts_iso_string():
    parsed = _parse_poll_expires_at("2026-05-26T23:24:24.425964+00:00")

    assert parsed.year == 2026
    assert parsed.tzinfo is not None


@pytest.mark.asyncio(loop_scope="function")
async def test_vote_poll_concurrent_duplicate_uses_single_insert_without_error():
    db, state = _fake_database()

    results = await asyncio.gather(
        db.vote_poll(poll_id=10, user_id=500, option_id=1),
        db.vote_poll(poll_id=10, user_id=500, option_id=1),
    )

    assert [result["success"] for result in results].count(True) == 1
    assert [result.get("error") for result in results].count("already_voted") == 1
    assert state.poll_votes == {(10, 500): 1}


@pytest.mark.asyncio(loop_scope="function")
async def test_vote_idea_concurrent_duplicate_recomputes_counts_from_votes():
    db, state = _fake_database()

    results = await asyncio.gather(
        db.vote_idea(post_id=20, user_id=501, vote_type="up"),
        db.vote_idea(post_id=20, user_id=501, vote_type="up"),
    )

    assert all(result["success"] for result in results)
    assert state.community_votes == {}
    assert state.community_posts[20]["upvotes"] == 0
    assert state.community_posts[20]["downvotes"] == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_vote_idea_rejects_missing_or_wrong_post_without_orphan_vote():
    db, state = _fake_database()

    missing = await db.vote_idea(post_id=999, user_id=501, vote_type="up")
    wrong_type = await db.vote_idea(post_id=40, user_id=501, vote_type="up")

    assert missing == {"success": False, "error": "invalid_post"}
    assert wrong_type == {"success": False, "error": "invalid_post"}
    assert state.community_votes == {}


@pytest.mark.asyncio(loop_scope="function")
async def test_react_announcement_concurrent_duplicate_keeps_reaction_counts_consistent():
    db, state = _fake_database()

    results = await asyncio.gather(
        db.react_announcement(post_id=30, user_id=502, vote_type="like"),
        db.react_announcement(post_id=30, user_id=502, vote_type="like"),
    )

    assert all(result["success"] for result in results)
    assert state.community_votes == {}
    assert results[-1]["likes"] == 0
    assert results[-1]["dislikes"] == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_toggle_post_like_concurrent_duplicate_keeps_like_count_consistent():
    db, state = _fake_database()

    results = await asyncio.gather(
        db.toggle_post_like(post_id=40, user_id=503),
        db.toggle_post_like(post_id=40, user_id=503),
    )

    assert all(result["success"] for result in results)
    assert state.post_likes == set()
    assert results[-1]["likes_count"] == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_toggle_post_like_rejects_missing_or_wrong_post_without_orphan_like():
    db, state = _fake_database()

    missing = await db.toggle_post_like(post_id=999, user_id=503)
    wrong_type = await db.toggle_post_like(post_id=20, user_id=503)

    assert missing == {"success": False, "error": "invalid_post"}
    assert wrong_type == {"success": False, "error": "invalid_post"}
    assert state.post_likes == set()


def test_news_feed_renders_poll_card_only_for_real_poll_attachment():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "hasPollAttachment" in source
    assert "hasPollAttachment(p)" in source


def test_community_rich_html_fallback_escapes_instead_of_regex_sanitizing():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const text = String(html ?? '')" in source
    assert "document.createElement('div')" in source
    assert "div.textContent = text" in source
    assert "return div.innerHTML" in source
    assert "return html.replace(/<script" not in source


def test_community_replaces_news_tab_with_rating_section():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const RatingScreen" in source
    assert "['Рейтинг','Объявления','Баги и идеи']" in source
    assert "['Новости','Объявления','Идеи']" not in source
    assert "fetch(_buildAuthUrl('/api/community/rating?scope=players'), {cache:'no-store'})" in source
    assert "if (!r.ok || d.success === false) throw new Error(d.message || d.error || 'rating_failed');" in source
    assert "/DesignAssets/Images/rating.jpg" in source
    assert "const RatingRotationTimer" in source
    assert "Пересчет через" in source
    assert "Зал боевой славы" in source
    assert "const RatingHelpPage" in source
    assert "Как работает рейтинг" in source
    assert "score = wins × winrate" in source
    assert "только уникальные предметы из user_cosmetics" in source
    assert "rating-help-btn" in source
    assert "Смотреть полный топ" in source
    assert "Смотреть предварительный топ за сегодня" in source
    assert "rating-masked-row" in source
    assert "rgba(5,2,12,0.97)" in source
    assert "Сквады" in source
    assert "Скоро" in source


def test_community_frontend_distinguishes_fetch_errors_from_empty_states():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "setPostsError" in source
    assert "setAnnouncementsError" in source
    assert "setIdeasError" in source
    assert "throw new Error(d.message || d.error || 'news_failed')" in source
    assert "throw new Error(d.message || d.error || 'announcements_failed')" in source
    assert "throw new Error(d.message || d.error || 'ideas_failed')" in source


def test_news_editor_requires_valid_poll_before_publish():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const validPollOptions = pollOptions.filter" in source
    assert "pollQuestion.trim() && validPollOptions.length >= 2" in source
    assert "disabled={!canPublish || submitting}" in source


def test_image_uploader_validates_type_and_size_before_upload():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const allowedImageTypes = new Set(['image/png','image/jpeg','image/webp'])" in source
    assert "file.size > 5 * 1024 * 1024" in source
    assert "fileRef.current.value = ''" in source


def test_global_chat_messages_requires_auth_and_clamps_limit():
    server = Path("web/server.py").read_text(encoding="utf-8")
    start = server.index("async def global_chat_messages_handler")
    end = server.index("    async def global_chat_send_handler", start)
    handler = server[start:end]

    assert "await require_user_id(request)" in handler
    assert "_community_pagination_query" in handler
    assert "int(request.rel_url.query.get(\"limit\"" not in handler


def test_legacy_community_routes_do_not_return_raw_exception_messages():
    server = Path("web/server.py").read_text(encoding="utf-8")
    database = Path("infrastructure/database.py").read_text(encoding="utf-8")
    handler_ranges = [
        ("async def community_posts_list_handler", "    async def community_post_create_handler"),
        ("async def community_post_create_handler", "    async def global_chat_messages_handler"),
        ("async def post_delete_handler", "    async def post_like_handler"),
        ("async def post_like_handler", "    async def rewards_track_handler"),
    ]
    posts_block = ""
    for start_marker, end_marker in handler_ranges:
        start = server.index(start_marker)
        end = server.index(end_marker, start)
        posts_block += server[start:end]
    db_start = database.index("async def create_community_post")
    db_end = database.index("    async def is_post_liked_by_user", db_start)
    db_block = database[db_start:db_end]

    assert "str(e)" not in posts_block
    assert "str(e)" not in db_block
    assert "internal_server_error" in db_block


def test_community_post_id_inputs_return_bad_request_before_db_calls():
    server = Path("web/server.py").read_text(encoding="utf-8")

    assert "def _community_required_int_field" in server
    for marker in (
        "async def community_news_like_handler",
        "async def community_ideas_vote_handler",
        "async def post_like_handler",
    ):
        start = server.index(marker)
        end = server.index("    async def", start + len(marker))
        block = server[start:end]
        assert "_community_required_int_field(data, \"post_id\")" in block
        assert "except AdminInputError as e" in block


def test_news_are_reachable_from_main_menu_after_rating_replaces_tab():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const NewsSheet" in source
    assert "label:'Новости'" in source
    assert "img:ICONS.menu.info,     label:'Новости'" in source
    assert 'aria-label="Меню"' in source
    assert "onNews={()=>{setShowMenu(false);setTimeout(()=>setShowNews(true),120);}}" in source


def test_news_webapp_editor_uses_runtime_admin_status():
    source = Path("webapp/index.html").read_text(encoding="utf-8")

    assert "const NewsSheet = ({onClose, isAdmin=false})" in source
    assert "<NewsSubScreen isAdmin={isAdmin}/>" in source
    assert "<NewsSheet       onClose={()=>setShowNews(false)} isAdmin={!!runtimeStatus?.is_admin}/>" in source


def test_telegram_and_web_news_paths_dual_write_to_both_feeds():
    bot_source = Path("bot/handlers.py").read_text(encoding="utf-8")
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    bot_block = bot_source.split("async def handle_news_post", 1)[1].split(
        "@router.message(Command(\"community_post\"))",
        1,
    )[0]
    server_block = server_source.split("async def community_news_create_handler", 1)[1].split(
        "async def community_news_like_handler",
        1,
    )[0]

    assert "create_news_post" in bot_block
    assert "content_html" in bot_block
    assert "create_news_entry" in server_block
    assert "button_url" in server_block


def test_telegram_news_renderer_sanitizes_links_splits_long_entries_and_handles_bad_requests():
    source = Path("bot/handlers.py").read_text(encoding="utf-8")
    render_block = source.split("def _render_news_entry", 1)[1].split(
        "def _build_news_keyboard",
        1,
    )[0]
    pages_block = source.split("def _build_news_pages", 1)[1].split(
        "def _render_news_entry",
        1,
    )[0]

    assert "NEWS_ENTRY_MAX_CHARS" in source
    assert "def _safe_news_url" in source
    assert "escape(item[\"button_url\"], quote=True)" in render_block
    assert "_safe_news_url" in render_block
    assert "entry_len > NEWS_MAX_CHARS" in pages_block
    assert "TelegramBadRequest" in source
    assert "_send_news_page" in source
    assert "answer_photo" in source


def test_beta_release_checklist_documents_config_policy_risks():
    source = Path("docs/BETA_RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "F-55 FIX_NOW/VERIFIED" in source
    assert "transaction-safe guards with concurrency regression coverage" in source
    assert "F-57 ACCEPTED_RISK/CONFIG_POLICY" in source
    assert "POLZA_AI_KEY" in source
    assert "hide/disable UGC entry points" in source
    assert "F-02 ACCEPTED_RISK" in source
    assert "single worker/drain/no horizontal scale until Redis/Postgres shared state" in source
    assert "F-12 ACCEPTED_RISK/CONSTRAINT" in source
    assert "no god-module refactor before beta" in source
    assert "minimal localized patches only" in source
    assert "WEB_CONCURRENCY=1" in source
    assert "Take and verify a beta PostgreSQL backup before deploy" in source
    assert "Rollback plan is written before deploy" in source
    assert "Staging smoke passes on the release candidate" in source
    assert "Post-deploy smoke passes on the beta host after deploy" in source
    assert "F-35 VERIFIED" in source
    assert "deterministic starting hand behavior" in source
    assert "F-54 VERIFIED" in source
    assert "bot replayability/reproducibility markers" in source
