import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure import config as app_config
from infrastructure.android_releases import (
    AndroidArtifactVerifier,
    AndroidReleaseConfig,
    AndroidReleaseError,
    AndroidReleaseService,
)
from web import server as web_server


def _config(tmp_path, **overrides):
    values = {
        "enabled": True,
        "storage_dir": tmp_path,
        "package_name": "ru.extraarena.app",
        "direct_signing_cert_sha256": "a" * 64,
        "rustore_signing_cert_sha256": "b" * 64,
        "max_bytes": 1024 * 1024,
        "chunk_bytes": 64 * 1024,
        "upload_token_ttl_seconds": 600,
    }
    values.update(overrides)
    return AndroidReleaseConfig(**values)


class ManifestDB:
    async def fetchrow(self, query, *args):
        assert "FROM android_release_channels c" in query
        return {
            "channel": "direct",
            "package_name": "ru.extraarena.app",
            "latest_release_id": "release-11",
            "latest_version_code": 11,
            "min_supported_version_code": 10,
            "latest_version_name": "0.11.0",
            "release_notes": "Optional polish",
            "published_at": datetime(2026, 7, 31, tzinfo=timezone.utc),
            "artifact_id": "artifact-11",
            "artifact_kind": "apk",
            "original_filename": "extraarena-11.apk",
            "size_bytes": 1234,
            "sha256": "c" * 64,
            "signing_cert_sha256": "a" * 64,
            "verified": True,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "update_available", "required"),
    ((9, True, True), (10, True, False), (11, False, False)),
)
async def test_manifest_preserves_required_history_after_optional_release(
    tmp_path, current, update_available, required
):
    service = AndroidReleaseService(
        ManifestDB(),
        _config(tmp_path, public_base_url="https://app.extraarena.space/game"),
    )

    manifest = await service.build_manifest(
        current_version_code=current,
        current_version_name=str(current),
        channel="direct",
    )

    assert manifest["latest_version_code"] == 11
    assert manifest["min_supported_version_code"] == 10
    assert manifest["update_available"] is update_available
    assert manifest["required"] is required
    expected_url = "https://app.extraarena.space/api/mobile/android/releases/release-11/apk"
    assert manifest["apk_url"] == expected_url
    assert manifest["update_url"] == expected_url


def test_settings_default_android_release_origin_to_configured_webapp_https(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123456:test")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("WEBAPP_URL", "https://app.extraarena.space/game")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANDROID_RELEASE_PUBLIC_BASE_URL", raising=False)
    app_config.get_settings.cache_clear()
    try:
        settings = app_config.get_settings()
    finally:
        app_config.get_settings.cache_clear()

    assert settings.android_release_public_base_url == "https://app.extraarena.space"


@pytest.mark.asyncio
async def test_enabled_manifest_db_error_fails_closed_instead_of_lower_legacy_floor(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("BOT_TOKEN", "123456:test")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("WEBAPP_URL", "https://app.extraarena.space")
    app_config.get_settings.cache_clear()

    class BrokenManifestService:
        async def build_manifest(self, **kwargs):
            raise RuntimeError("database unavailable")

    app = web_server.create_web_app(
        ManifestDB(),
        bot_token="123456:test",
        webapp_url="https://app.extraarena.space",
        android_release_storage_dir=str(tmp_path),
        android_releases_enabled=True,
        android_min_supported_version_code=1,
    )
    app["android_release_service"] = BrokenManifestService()
    client = TestClient(TestServer(app))
    try:
        await client.start_server()
        response = await client.get(
            "/api/mobile/android/releases/manifest?platform=android&channel=direct&version_code=1"
        )
        payload = await response.json()
    finally:
        await client.close()
        app_config.get_settings.cache_clear()

    assert response.status == 503
    assert payload == {"error": "android_release_manifest_unavailable"}
    assert "no-store" in response.headers["Cache-Control"]


@pytest.mark.asyncio
async def test_verifier_uses_channel_specific_certificate_pin(monkeypatch, tmp_path):
    artifact = tmp_path / "release.apk"
    artifact.write_bytes(b"PK\x03\x04placeholder")
    verifier = AndroidArtifactVerifier(_config(tmp_path))
    monkeypatch.setattr(verifier, "_resolve_command", lambda command: [command])

    async def fake_run(command, *args):
        if "badging" in args:
            return 0, "package: name='ru.extraarena.app' versionCode='49' versionName='0.5.2'"
        return 0, (
            "Verifies\n"
            f"V3.0 Signer: certificate SHA-256 digest: {'b' * 64}\n"
            f"V2 Signer: certificate SHA-256 digest: {'b' * 64}\n"
        )

    monkeypatch.setattr(verifier, "_run", fake_run)

    direct = await verifier.verify(artifact, "apk", channel="direct")
    rustore = await verifier.verify(artifact, "apk", channel="rustore")
    aab = await verifier.verify(artifact, "aab", channel="rustore")

    assert direct["verified"] is False
    assert "signing_certificate_mismatch" in direct["errors"]
    assert rustore["verified"] is True
    assert rustore["signing_cert_sha256"] == "b" * 64
    assert rustore["signing_certificates_sha256"] == ["b" * 64]
    assert aab["verified"] is False
    assert "aab_verifier_not_configured" in aab["errors"]


class PublishValidationDB:
    def __init__(self, row):
        self.row = row

    async def fetchrow(self, query, *args):
        if "FROM android_releases r" in query:
            return dict(self.row)
        raise AssertionError(query)


class ExpiredUploadCleanupDB:
    def __init__(self):
        self.status = "uploading"
        self.storage_key = "uploads/expired-upload.part"

    async def fetch(self, query, *args):
        assert "token_expires_at <= NOW()" in query
        assert "status IN ('expired', 'aborted')" in query
        if self.status == "uploading" or self.storage_key == "uploads/expired-upload.part":
            return [{"upload_id": "expired-upload", "status": self.status, "created_by": 1}]
        return []

    async def fetchrow(self, query, *args):
        assert "THEN 'expired'" in query
        if self.status not in {"uploading", "expired"} or not self.storage_key:
            return None
        previous_status = self.status
        self.status = "expired"
        return {
            "temp_storage_key": self.storage_key,
            "previous_status": previous_status,
        }

    async def execute(self, query, *args):
        assert "SET temp_storage_key = $3" in query
        assert args == (
            "expired-upload",
            self.storage_key,
            "expired/expired-upload.part",
            "expired",
        )
        self.storage_key = args[2]


@pytest.mark.asyncio
async def test_expired_upload_cleanup_is_bounded_and_removes_partial_file(tmp_path):
    partial = tmp_path / "uploads" / "expired-upload.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial-apk")
    db = ExpiredUploadCleanupDB()
    service = AndroidReleaseService(db, _config(tmp_path))

    result = await service.cleanup_expired_uploads(limit=10)

    assert result == {"expired": 1, "reconciled": 0, "removed_files": 1, "failures": 0}
    assert db.status == "expired"
    assert db.storage_key == "expired/expired-upload.part"
    assert not partial.exists()


@pytest.mark.asyncio
async def test_expired_upload_cleanup_retries_unlink_after_status_was_committed(
    tmp_path,
    monkeypatch,
):
    partial = tmp_path / "uploads" / "expired-upload.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial-apk")
    db = ExpiredUploadCleanupDB()
    service = AndroidReleaseService(db, _config(tmp_path))
    original_unlink = Path.unlink
    attempts = 0

    def fail_once(path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated unlink interruption")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)

    first_result = await service.cleanup_expired_uploads(limit=10)

    assert first_result == {"expired": 1, "reconciled": 0, "removed_files": 0, "failures": 1}
    assert db.status == "expired"
    assert db.storage_key == "uploads/expired-upload.part"
    assert partial.exists()

    result = await service.cleanup_expired_uploads(limit=10)

    assert result == {"expired": 0, "reconciled": 0, "removed_files": 1, "failures": 0}
    assert db.storage_key == "expired/expired-upload.part"
    assert not partial.exists()


class MultipleExpiredUploadCleanupDB:
    def __init__(self):
        self.rows = {
            upload_id: {
                "status": "uploading",
                "storage_key": f"uploads/{upload_id}.part",
            }
            for upload_id in ("blocked-upload", "healthy-upload")
        }

    async def fetch(self, query, *args):
        return [
            {"upload_id": upload_id, "status": row["status"], "created_by": 1}
            for upload_id, row in self.rows.items()
        ]

    async def fetchrow(self, query, upload_id):
        row = self.rows[upload_id]
        if row["storage_key"] != f"uploads/{upload_id}.part":
            return None
        previous_status = row["status"]
        row["status"] = "expired"
        return {
            "temp_storage_key": row["storage_key"],
            "previous_status": previous_status,
        }

    async def execute(self, query, upload_id, storage_key, tombstone_key, terminal_status):
        row = self.rows[upload_id]
        assert row["storage_key"] == storage_key
        assert terminal_status == "expired"
        row["storage_key"] = tombstone_key


@pytest.mark.asyncio
async def test_expired_upload_cleanup_failure_does_not_starve_later_candidates(
    tmp_path,
    monkeypatch,
):
    blocked = tmp_path / "uploads" / "blocked-upload.part"
    healthy = tmp_path / "uploads" / "healthy-upload.part"
    blocked.parent.mkdir(parents=True)
    blocked.write_bytes(b"blocked")
    healthy.write_bytes(b"healthy")
    db = MultipleExpiredUploadCleanupDB()
    service = AndroidReleaseService(db, _config(tmp_path))
    original_unlink = Path.unlink

    def reject_blocked(path, *args, **kwargs):
        if path.name == "blocked-upload.part":
            raise OSError("persistent unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_blocked)

    result = await service.cleanup_expired_uploads(limit=10)

    assert result == {"expired": 2, "reconciled": 0, "removed_files": 1, "failures": 1}
    assert blocked.exists()
    assert not healthy.exists()
    assert db.rows["blocked-upload"]["storage_key"] == "uploads/blocked-upload.part"
    assert db.rows["healthy-upload"]["storage_key"] == "expired/healthy-upload.part"


@pytest.mark.asyncio
async def test_aab_is_explicitly_unpublishable_and_retire_checks_version(tmp_path):
    base = {
        "release_id": "release-49",
        "channel": "rustore",
        "version_code": 49,
        "version_name": "0.5.2",
        "status": "staged",
        "artifact_id": "artifact-49",
        "artifact_kind": "aab",
        "artifact_package_name": "ru.extraarena.app",
        "verified": True,
        "latest_release_id": "release-48",
        "latest_version_code": 48,
        "min_supported_version_code": 45,
    }
    service = AndroidReleaseService(PublishValidationDB(base), _config(tmp_path))

    with pytest.raises(AndroidReleaseError, match="android_aab_publish_not_supported"):
        await service.publish_release(
            release_id="release-49",
            required=False,
            expected_version_code=49,
            admin_user_id=1,
            dry_run=True,
        )
    with pytest.raises(AndroidReleaseError, match="android_release_version_confirmation_mismatch"):
        await service.retire_release(
            release_id="release-49",
            expected_version_code=50,
            admin_user_id=1,
            reason="wrong stale target",
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_republishing_required_current_as_optional_cannot_clear_required_policy(tmp_path):
    current = {
        "release_id": "release-49",
        "channel": "direct",
        "version_code": 49,
        "version_name": "0.5.2",
        "status": "published",
        "required_at_publish": True,
        "artifact_id": "artifact-49",
        "artifact_kind": "apk",
        "artifact_package_name": "ru.extraarena.app",
        "signing_cert_sha256": "a" * 64,
        "verified": True,
        "latest_release_id": "release-49",
        "latest_version_code": 49,
        "min_supported_version_code": 49,
    }
    service = AndroidReleaseService(PublishValidationDB(current), _config(tmp_path))

    preview = await service.publish_release(
        release_id="release-49",
        required=False,
        expected_version_code=49,
        admin_user_id=1,
        dry_run=True,
    )

    assert preview["required_requested"] is False
    assert preview["required_at_publish"] is True
    assert preview["min_supported_version_code_after"] == 49


@pytest.mark.asyncio
async def test_rustore_publish_requires_exact_store_rollout_confirmation(tmp_path):
    staged = {
        "release_id": "rustore-49",
        "channel": "rustore",
        "version_code": 49,
        "version_name": "0.6.0",
        "status": "staged",
        "artifact_id": "artifact-rustore-49",
        "artifact_kind": "apk",
        "artifact_package_name": "ru.extraarena.app",
        "signing_cert_sha256": "b" * 64,
        "verified": True,
        "latest_release_id": "rustore-48",
        "latest_version_code": 48,
        "min_supported_version_code": 45,
    }
    service = AndroidReleaseService(PublishValidationDB(staged), _config(tmp_path))

    with pytest.raises(AndroidReleaseError, match="android_rustore_release_not_confirmed_live"):
        await service.publish_release(
            release_id="rustore-49",
            required=True,
            expected_version_code=49,
            admin_user_id=1,
            dry_run=True,
        )

    preview = await service.publish_release(
        release_id="rustore-49",
        required=True,
        expected_version_code=49,
        store_release_confirmed=True,
        admin_user_id=1,
        dry_run=True,
    )
    assert preview["store_release_confirmed"] is True
    assert preview["min_supported_version_code_after"] == 49


class ConcurrentUploadDB:
    def __init__(self):
        self.row = {
            "upload_id": "upload-1",
            "channel": "direct",
            "artifact_kind": "apk",
            "original_filename": "release.apk",
            "expected_size_bytes": 8,
            "expected_sha256": hashlib.sha256(b"AAAABBBB").hexdigest(),
            "expected_version_code": 49,
            "expected_version_name": "0.5.2",
            "received_bytes": 0,
            "temp_storage_key": "uploads/upload-1.part",
            "token_hash": "unused",
            "status": "uploading",
            "token_expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        self._guard = asyncio.Lock()

    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            return dict(self.row)
        if "UPDATE android_release_uploads" in query and "received_bytes = received_bytes" in query:
            async with self._guard:
                _, written, expected_offset = args
                if self.row["status"] != "uploading" or self.row["received_bytes"] != expected_offset:
                    return None
                self.row["received_bytes"] += written
                return dict(self.row)
        if "WITH abort_candidate AS" in query:
            if self.row["status"] not in {"uploading", "failed", "expired"}:
                return None
            previous_status = self.row["status"]
            self.row.update(
                status="aborted",
                token_hash=f"aborted:{self.row['upload_id']}",
            )
            return {**self.row, "previous_status": previous_status}
        raise AssertionError(query)

    async def execute(self, query, *args):
        if "SET temp_storage_key = $3" in query:
            self.row["temp_storage_key"] = args[2]
            return "UPDATE 1"
        raise AssertionError(query)


class StrictUploadLookupDB(ConcurrentUploadDB):
    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            if str(args[0]) != str(self.row["upload_id"]):
                return None
        return await super().fetchrow(query, *args)


async def _one_chunk(value: bytes):
    await asyncio.sleep(0.01)
    yield value


@pytest.mark.asyncio
async def test_invalid_upload_tickets_create_no_lock_state_or_inodes_and_valid_lock_is_evicted(tmp_path):
    db = StrictUploadLookupDB()
    valid_token = "valid-scoped-upload-token"
    db.row["token_hash"] = hashlib.sha256(valid_token.encode("utf-8")).hexdigest()
    service = AndroidReleaseService(db, _config(tmp_path))

    for index in range(50):
        with pytest.raises(AndroidReleaseError, match="android_release_upload_not_found"):
            await service.append_upload_chunk(
                upload_id=f"attacker-{index}",
                offset=0,
                chunks=_one_chunk(b"AAAA"),
                upload_token="invalid-ticket",
            )
    with pytest.raises(AndroidReleaseError, match="android_release_upload_auth_required"):
        await service.append_upload_chunk(
            upload_id="upload-1",
            offset=0,
            chunks=_one_chunk(b"AAAA"),
            upload_token="invalid-ticket",
        )

    assert service._upload_locks == {}
    assert service._upload_lock_refs == {}
    assert not (tmp_path / "locks").exists()

    uploaded = await service.append_upload_chunk(
        upload_id="upload-1",
        offset=0,
        chunks=_one_chunk(b"AAAA"),
        upload_token=valid_token,
    )

    assert uploaded["received_bytes"] == 4
    assert service._upload_locks == {}
    assert service._upload_lock_refs == {}
    assert len(list((tmp_path / "locks").glob("*.lock"))) == 1


@pytest.mark.asyncio
async def test_cross_service_same_offset_writers_cannot_truncate_winner(tmp_path):
    db = ConcurrentUploadDB()
    config = _config(tmp_path)
    first_service = AndroidReleaseService(db, config)
    second_service = AndroidReleaseService(db, config)

    results = await asyncio.gather(
        first_service.append_upload_chunk(
            upload_id="upload-1", offset=0, chunks=_one_chunk(b"AAAA")
        ),
        second_service.append_upload_chunk(
            upload_id="upload-1", offset=0, chunks=_one_chunk(b"BBBB")
        ),
        return_exceptions=True,
    )

    successes = [result for result in results if isinstance(result, dict)]
    conflicts = [result for result in results if isinstance(result, AndroidReleaseError)]
    stored = (tmp_path / "uploads" / "upload-1.part").read_bytes()
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].code == "android_release_upload_offset_conflict"
    assert db.row["received_bytes"] == 4
    assert stored in {b"AAAA", b"BBBB"}


@pytest.mark.asyncio
async def test_cancelled_append_rolls_file_back_to_committed_offset(tmp_path):
    db = ConcurrentUploadDB()
    service = AndroidReleaseService(db, _config(tmp_path))

    async def cancelled_chunk():
        yield b"AAAA"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.append_upload_chunk(
            upload_id="upload-1",
            offset=0,
            chunks=cancelled_chunk(),
        )

    assert db.row["received_bytes"] == 0
    assert (tmp_path / "uploads" / "upload-1.part").read_bytes() == b""


class AmbiguousAppendDB(ConcurrentUploadDB):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.injected = False
        self.fail_outcome_readback = False

    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            if self.fail_outcome_readback:
                self.fail_outcome_readback = False
                raise ConnectionError("simulated append outcome readback outage")
            return dict(self.row)
        if "received_bytes = received_bytes" in query and not self.injected:
            self.injected = True
            if self.mode == "before_commit":
                raise asyncio.CancelledError
            if self.mode == "unknown_before_commit":
                self.fail_outcome_readback = True
                raise ConnectionError("simulated ambiguous append response")
            if self.mode == "after_commit":
                await super().fetchrow(query, *args)
                raise asyncio.CancelledError
        return await super().fetchrow(query, *args)


@pytest.mark.asyncio
async def test_append_cancellation_before_commit_truncates_fsynced_suffix(tmp_path):
    db = AmbiguousAppendDB("before_commit")
    service = AndroidReleaseService(db, _config(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        await service.append_upload_chunk(
            upload_id="upload-1",
            offset=0,
            chunks=_one_chunk(b"AAAA"),
        )

    assert db.row["received_bytes"] == 0
    assert (tmp_path / "uploads" / "upload-1.part").read_bytes() == b""


@pytest.mark.asyncio
async def test_append_cancellation_after_commit_keeps_fsynced_suffix(tmp_path):
    db = AmbiguousAppendDB("after_commit")
    service = AndroidReleaseService(db, _config(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        await service.append_upload_chunk(
            upload_id="upload-1",
            offset=0,
            chunks=_one_chunk(b"AAAA"),
        )

    partial = tmp_path / "uploads" / "upload-1.part"
    assert db.row["received_bytes"] == 4
    assert partial.read_bytes() == b"AAAA"
    result = await service.append_upload_chunk(
        upload_id="upload-1",
        offset=4,
        chunks=_one_chunk(b"BBBB"),
    )
    assert result["received_bytes"] == 8
    assert partial.read_bytes() == b"AAAABBBB"


@pytest.mark.asyncio
async def test_append_unknown_outcome_preserves_then_fresh_read_repairs_orphan_tail(tmp_path):
    db = AmbiguousAppendDB("unknown_before_commit")
    service = AndroidReleaseService(db, _config(tmp_path))

    with pytest.raises(ConnectionError, match="ambiguous append"):
        await service.append_upload_chunk(
            upload_id="upload-1",
            offset=0,
            chunks=_one_chunk(b"AAAA"),
        )

    partial = tmp_path / "uploads" / "upload-1.part"
    assert db.row["received_bytes"] == 0
    assert partial.read_bytes() == b"AAAA"
    result = await service.append_upload_chunk(
        upload_id="upload-1",
        offset=0,
        chunks=_one_chunk(b"BBBB"),
    )
    assert result["received_bytes"] == 4
    assert partial.read_bytes() == b"BBBB"


@pytest.mark.asyncio
async def test_abort_waits_for_inflight_append_on_shared_lifecycle_lock(tmp_path):
    db = ConcurrentUploadDB()
    config = _config(tmp_path)
    writer = AndroidReleaseService(db, config)
    aborter = AndroidReleaseService(db, config)
    append_started = asyncio.Event()
    allow_append = asyncio.Event()

    async def held_chunk():
        append_started.set()
        await allow_append.wait()
        yield b"AAAA"

    append_task = asyncio.create_task(
        writer.append_upload_chunk(upload_id="upload-1", offset=0, chunks=held_chunk())
    )
    await append_started.wait()
    abort_task = asyncio.create_task(
        aborter.abort_upload(upload_id="upload-1", admin_user_id=1)
    )
    await asyncio.sleep(0.02)
    assert abort_task.done() is False

    allow_append.set()
    appended, aborted = await asyncio.gather(append_task, abort_task)

    assert appended["received_bytes"] == 4
    assert aborted["aborted"] is True
    assert db.row["status"] == "aborted"
    assert not (tmp_path / "uploads" / "upload-1.part").exists()


@pytest.mark.asyncio
async def test_cancelled_flock_waiter_cannot_acquire_and_leak_lock_after_holder_exits(tmp_path):
    config = _config(tmp_path)
    holder_service = AndroidReleaseService(object(), config)
    waiter_service = AndroidReleaseService(object(), config)
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_lock():
        async with holder_service._upload_lifecycle_lock("lock-cancellation"):
            holder_entered.set()
            await release_holder.wait()

    async def wait_for_lock():
        async with waiter_service._upload_lifecycle_lock("lock-cancellation"):
            return True

    holder = asyncio.create_task(hold_lock())
    await holder_entered.wait()
    waiter = asyncio.create_task(wait_for_lock())
    await asyncio.sleep(0.12)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_holder.set()
    await holder
    # The cancelled waiter must not leave a background blocking flock call
    # which later acquires the OS lock without any coroutine able to release it.
    assert await asyncio.wait_for(wait_for_lock(), timeout=1.0) is True


class AbortCleanupDB(ConcurrentUploadDB):
    async def fetch(self, query, *args):
        if self.row["status"] == "aborted" and self.row["temp_storage_key"].startswith("uploads/"):
            return [{"upload_id": self.row["upload_id"], "status": "aborted", "created_by": 1}]
        return []

    async def fetchrow(self, query, *args):
        if "WITH cleanup_candidate AS" in query:
            if self.row["status"] != "aborted":
                return None
            return {
                "temp_storage_key": self.row["temp_storage_key"],
                "previous_status": "aborted",
            }
        return await super().fetchrow(query, *args)


@pytest.mark.asyncio
async def test_aborted_upload_unlink_failure_is_tombstoned_and_retried(tmp_path, monkeypatch):
    db = AbortCleanupDB()
    service = AndroidReleaseService(db, _config(tmp_path))
    partial = tmp_path / "uploads" / "upload-1.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    original_unlink = Path.unlink
    failed_once = False

    def fail_first_unlink(path, *args, **kwargs):
        nonlocal failed_once
        if not failed_once and path == partial:
            failed_once = True
            raise OSError("simulated aborted-upload unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_unlink)

    aborted = await service.abort_upload(upload_id="upload-1", admin_user_id=1)

    assert aborted["aborted"] is True
    assert aborted["cleanup_pending"] is True
    assert db.row["status"] == "aborted"
    assert db.row["temp_storage_key"] == "uploads/upload-1.part"
    assert partial.exists()

    cleanup = await service.cleanup_expired_uploads(limit=10)

    assert cleanup == {"expired": 0, "reconciled": 0, "removed_files": 1, "failures": 0}
    assert db.row["temp_storage_key"] == "aborted/upload-1.part"
    assert not partial.exists()


class FinalizeDB:
    def __init__(self):
        self.upload = {
            "upload_id": "upload-finalize",
            "channel": "direct",
            "artifact_kind": "apk",
            "original_filename": "release.apk",
            "expected_size_bytes": 4,
            "expected_sha256": hashlib.sha256(b"PK12").hexdigest(),
            "expected_version_code": 49,
            "expected_version_name": "0.5.2",
            "release_notes": "Race regression",
            "received_bytes": 4,
            "temp_storage_key": "uploads/upload-finalize.part",
            "token_hash": "unused",
            "status": "uploading",
            "token_expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
            "release_id": None,
        }
        self.release = None
        self.finalize_writes = 0

    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            return dict(self.upload)
        if "SET status = 'finalizing'" in query:
            if self.upload["status"] not in {"uploading", "failed"}:
                return None
            self.upload["status"] = "finalizing"
            return dict(self.upload)
        if "SET status = 'failed'" in query:
            if self.upload["status"] not in {"uploading", "failed"}:
                return None
            self.upload["status"] = "failed"
            return dict(self.upload)
        if "WITH eligible_upload AS" in query:
            assert self.upload["status"] == "finalizing"
            release_id = args[0]
            self.upload.update(status="finalized", release_id=release_id)
            self.release = {
                "release_id": release_id,
                "channel": "direct",
                "version_code": 49,
                "version_name": "0.5.2",
                "status": "staged",
                "artifact_id": args[6],
                "artifact_kind": "apk",
                "sha256": args[11],
                "verified": True,
            }
            self.finalize_writes += 1
            return {"release_id": release_id}
        if "FROM android_releases r" in query:
            return dict(self.release) if self.release else None
        raise AssertionError(query)


class AlwaysVerifiedArtifact:
    async def verify(self, path, artifact_kind, *, channel="direct"):
        return {
            "verified": True,
            "artifact_kind": artifact_kind,
            "channel": channel,
            "package_name": "ru.extraarena.app",
            "version_code": 49,
            "version_name": "0.5.2",
            "signing_cert_sha256": "a" * 64,
            "errors": [],
        }


class TransientApkVerifier(AlwaysVerifiedArtifact):
    def __init__(self):
        self.calls = 0

    async def verify(self, path, artifact_kind, *, channel="direct"):
        self.calls += 1
        if self.calls == 1:
            return {
                "verified": False,
                "artifact_kind": artifact_kind,
                "channel": channel,
                "package_name": None,
                "version_code": None,
                "version_name": None,
                "signing_cert_sha256": None,
                "errors": ["aapt_unavailable"],
            }
        return await super().verify(path, artifact_kind, channel=channel)


@pytest.mark.asyncio
async def test_failed_apk_verification_is_retryable_without_burning_release_version(tmp_path):
    db = FinalizeDB()
    verifier = TransientApkVerifier()
    service = AndroidReleaseService(db, _config(tmp_path), verifier=verifier)
    partial = tmp_path / "uploads" / "upload-finalize.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"PK12")

    with pytest.raises(AndroidReleaseError) as rejected:
        await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    assert rejected.value.code == "android_release_apk_verification_failed"
    assert rejected.value.details["retryable"] is True
    assert rejected.value.details["replacement_allowed"] is True
    assert db.upload["status"] == "failed"
    assert db.upload["release_id"] is None
    assert db.release is None
    assert partial.read_bytes() == b"PK12"
    assert not (tmp_path / "artifacts").exists()

    finalized = await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    assert finalized["release"]["version_code"] == 49
    assert finalized["release"]["verified"] is True
    assert verifier.calls == 2
    assert db.finalize_writes == 1


class AmbiguousFinalizeDB(FinalizeDB):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.injected = False
        self.fail_outcome_readback = False

    async def fetch(self, query, *args):
        if self.upload["status"] == "finalizing":
            return [{"upload_id": self.upload["upload_id"], "status": "finalizing", "created_by": 1}]
        return []

    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            if self.fail_outcome_readback:
                self.fail_outcome_readback = False
                raise ConnectionError("simulated outcome readback outage")
            return dict(self.upload)
        if "WITH eligible_upload AS" in query and not self.injected:
            self.injected = True
            if self.mode == "before_commit":
                raise asyncio.CancelledError
            if self.mode == "unknown_before_commit":
                self.fail_outcome_readback = True
                raise ConnectionError("simulated ambiguous commit response")
            if self.mode == "after_commit":
                await super().fetchrow(query, *args)
                raise asyncio.CancelledError
        return await super().fetchrow(query, *args)


class MultiArtifactFinalizeDB:
    def __init__(self):
        self.uploads = {}
        self.release = None
        self.artifacts = {}

    async def execute(self, query, *args):
        if "INSERT INTO android_release_channels" in query:
            return "INSERT 0 1"
        if "SET temp_storage_key = $3" in query:
            upload = self.uploads[str(args[0])]
            upload["temp_storage_key"] = args[2]
            return "UPDATE 1"
        raise AssertionError(query)

    async def fetchrow(self, query, *args):
        if query.strip().startswith("SELECT * FROM android_release_uploads"):
            row = self.uploads.get(str(args[0]))
            return dict(row) if row else None
        if "INSERT INTO android_release_uploads" in query:
            row = {
                "upload_id": args[0],
                "channel": args[1],
                "artifact_kind": args[2],
                "original_filename": args[3],
                "expected_size_bytes": args[4],
                "expected_sha256": args[5],
                "expected_version_code": args[6],
                "expected_version_name": args[7],
                "release_notes": args[8],
                "received_bytes": 0,
                "temp_storage_key": args[9],
                "token_hash": args[10],
                "status": "uploading",
                "created_by": args[11],
                "token_expires_at": args[12],
                "release_id": None,
            }
            self.uploads[str(row["upload_id"])] = row
            return dict(row)
        if "SET status = 'finalizing'" in query:
            upload = self.uploads[str(args[0])]
            if upload["status"] not in {"uploading", "failed"}:
                return None
            upload["status"] = "finalizing"
            return dict(upload)
        if "SET status = 'failed'" in query:
            upload = self.uploads[str(args[0])]
            allowed = (
                {"uploading", "failed"}
                if "status IN ('uploading', 'failed')" in query
                else {"finalizing"}
            )
            if upload["status"] not in allowed:
                return None
            upload["status"] = "failed"
            return dict(upload)
        if "WITH abort_candidate AS" in query:
            upload = self.uploads[str(args[0])]
            if upload["status"] not in {"uploading", "failed", "expired"}:
                return None
            previous_status = upload["status"]
            upload.update(
                status="aborted",
                token_hash=f"aborted:{upload['upload_id']}",
            )
            return {**upload, "previous_status": previous_status}
        if "WITH eligible_upload AS" in query:
            assert "ON CONFLICT (channel, version_code) DO UPDATE" in query
            assert "ON CONFLICT (release_id, artifact_kind) DO NOTHING" in query
            upload = self.uploads[str(args[16])]
            if upload["status"] != "finalizing":
                return None
            if self.release is None:
                self.release = {
                    "release_id": args[0],
                    "channel": args[1],
                    "version_code": args[2],
                    "version_name": args[3],
                    "status": "staged",
                    "required_at_publish": False,
                    "release_notes": args[4],
                    "latest_release_id": None,
                    "latest_version_code": 0,
                    "min_supported_version_code": 0,
                }
            elif (
                self.release["status"] != "staged"
                or self.release["channel"] != args[1]
                or self.release["version_code"] != args[2]
                or self.release["version_name"] != args[3]
            ):
                return None
            kind = str(args[7])
            if kind in self.artifacts:
                return None
            self.artifacts[kind] = {
                "artifact_id": args[6],
                "artifact_kind": kind,
                "original_filename": args[8],
                "storage_key": args[9],
                "size_bytes": args[10],
                "sha256": args[11],
                "artifact_package_name": args[12],
                "artifact_version_code": args[2],
                "artifact_version_name": args[3],
                "signing_cert_sha256": args[13],
                "verified": args[14],
                "verifier_result": args[15],
            }
            upload.update(status="finalized", release_id=self.release["release_id"])
            return {"release_id": self.release["release_id"]}
        if "WHERE r.channel = $1 AND r.version_code = $2" in query:
            if not self.release:
                return None
            if self.release["channel"] != args[0] or self.release["version_code"] != args[1]:
                return None
            return {
                "release_id": self.release["release_id"],
                "status": self.release["status"],
                "version_name": self.release["version_name"],
                "release_notes": self.release["release_notes"],
                "artifact_kind_exists": str(args[2]) in self.artifacts,
            }
        if "FROM android_releases r" in query and "WHERE r.release_id = $1" in query:
            if not self.release or self.release["release_id"] != str(args[0]):
                return None
            preferred = self.artifacts.get("apk") or self.artifacts.get("aab") or {}
            return {
                **self.release,
                **preferred,
                "artifact_kinds": sorted(self.artifacts, key=lambda item: item != "apk"),
            }
        raise AssertionError(query)


class ArtifactKindVerifier:
    async def verify(self, path, artifact_kind, *, channel="direct"):
        verified = artifact_kind == "apk"
        return {
            "verified": verified,
            "artifact_kind": artifact_kind,
            "channel": channel,
            "package_name": "ru.extraarena.app",
            "version_code": 49,
            "version_name": "0.5.2",
            "signing_cert_sha256": "a" * 64 if verified else None,
            "errors": [] if verified else ["aab_verifier_not_configured"],
        }


async def _prepare_complete_and_finalize(service, db, tmp_path, artifact_kind):
    payload = b"PK" + artifact_kind.encode("ascii")
    prepared = await service.prepare_upload(
        channel="direct",
        artifact_kind=artifact_kind,
        filename=f"release.{artifact_kind}",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        version_code=49,
        version_name="0.5.2",
        release_notes="Multi-artifact release",
        admin_user_id=1,
    )
    upload_id = prepared["upload"]["upload_id"]
    upload = db.uploads[upload_id]
    partial = tmp_path / upload["temp_storage_key"]
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(payload)
    upload["received_bytes"] = len(payload)
    finalized = await service.finalize_upload(upload_id=upload_id, admin_user_id=1)
    return prepared, finalized


@pytest.mark.asyncio
async def test_aab_first_then_apk_attaches_to_same_staged_release_and_becomes_publishable(tmp_path):
    db = MultiArtifactFinalizeDB()
    service = AndroidReleaseService(db, _config(tmp_path), verifier=ArtifactKindVerifier())

    _, aab = await _prepare_complete_and_finalize(service, db, tmp_path, "aab")
    apk_prepared, apk = await _prepare_complete_and_finalize(service, db, tmp_path, "apk")

    release_id = aab["release"]["release_id"]
    assert apk_prepared["attach_to_release_id"] == release_id
    assert apk["release"]["release_id"] == release_id
    assert apk["release"]["artifact_kinds"] == ["apk", "aab"]
    assert set(db.artifacts) == {"aab", "apk"}
    preview = await service.publish_release(
        release_id=release_id,
        required=False,
        expected_version_code=49,
        admin_user_id=1,
        dry_run=True,
    )
    assert preview["artifact"]["artifact_kind"] == "apk"
    assert preview["artifact"]["verified"] is True
    with pytest.raises(AndroidReleaseError, match="android_release_artifact_kind_exists"):
        await service.prepare_upload(
            channel="direct",
            artifact_kind="apk",
            filename="duplicate.apk",
            size_bytes=4,
            sha256="d" * 64,
            version_code=49,
            version_name="0.5.2",
            admin_user_id=1,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_apk_first_then_aab_attaches_without_consuming_second_version(tmp_path):
    db = MultiArtifactFinalizeDB()
    service = AndroidReleaseService(db, _config(tmp_path), verifier=ArtifactKindVerifier())

    _, apk = await _prepare_complete_and_finalize(service, db, tmp_path, "apk")
    aab_prepared, aab = await _prepare_complete_and_finalize(service, db, tmp_path, "aab")

    release_id = apk["release"]["release_id"]
    assert aab_prepared["attach_to_release_id"] == release_id
    assert aab["release"]["release_id"] == release_id
    assert set(db.artifacts) == {"apk", "aab"}
    assert aab["release"]["artifact_kind"] == "apk"
    assert aab["release"]["artifact_kinds"] == ["apk", "aab"]

    db.release["status"] = "published"
    with pytest.raises(AndroidReleaseError, match="android_release_version_exists"):
        await service.prepare_upload(
            channel="direct",
            artifact_kind="aab",
            filename="late.aab",
            size_bytes=4,
            sha256="e" * 64,
            version_code=49,
            version_name="0.5.2",
            admin_user_id=1,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_concurrent_same_kind_finalize_conflict_restores_abortable_partial(tmp_path):
    db = MultiArtifactFinalizeDB()
    service = AndroidReleaseService(db, _config(tmp_path), verifier=ArtifactKindVerifier())
    payloads = (b"PKa1", b"PKa2")
    upload_ids = []
    for index, payload in enumerate(payloads):
        prepared = await service.prepare_upload(
            channel="direct",
            artifact_kind="apk",
            filename=f"release-{index}.apk",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            version_code=49,
            version_name="0.5.2",
            release_notes="Concurrent same-kind race",
            admin_user_id=1,
        )
        upload_id = prepared["upload"]["upload_id"]
        upload_ids.append(upload_id)
        upload = db.uploads[upload_id]
        partial = tmp_path / upload["temp_storage_key"]
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(payload)
        upload["received_bytes"] = len(payload)

    await service.finalize_upload(upload_id=upload_ids[0], admin_user_id=1)
    with pytest.raises(AndroidReleaseError, match="android_release_artifact_kind_exists"):
        await service.finalize_upload(upload_id=upload_ids[1], admin_user_id=1)

    losing = db.uploads[upload_ids[1]]
    losing_partial = tmp_path / f"uploads/{upload_ids[1]}.part"
    assert losing["status"] == "failed"
    assert losing_partial.read_bytes() == payloads[1]
    assert len(list((tmp_path / "artifacts").rglob("*.apk"))) == 1

    aborted = await service.abort_upload(upload_id=upload_ids[1], admin_user_id=1)

    assert aborted["aborted"] is True
    assert aborted["cleanup_pending"] is False
    assert losing["status"] == "aborted"
    assert not losing_partial.exists()


@pytest.mark.asyncio
async def test_double_finalize_across_services_is_idempotent(tmp_path):
    db = FinalizeDB()
    config = _config(tmp_path)
    partial = tmp_path / "uploads" / "upload-finalize.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"PK12")
    first = AndroidReleaseService(db, config, verifier=AlwaysVerifiedArtifact())
    second = AndroidReleaseService(db, config, verifier=AlwaysVerifiedArtifact())

    results = await asyncio.gather(
        first.finalize_upload(upload_id="upload-finalize", admin_user_id=1),
        second.finalize_upload(upload_id="upload-finalize", admin_user_id=1),
    )

    assert db.finalize_writes == 1
    assert sum(bool(result.get("already_finalized")) for result in results) == 1
    assert len(list((tmp_path / "artifacts").rglob("*.apk"))) == 1


@pytest.mark.asyncio
async def test_finalize_cancellation_before_commit_preserves_final_and_background_reconciles(tmp_path):
    db = AmbiguousFinalizeDB("before_commit")
    service = AndroidReleaseService(db, _config(tmp_path), verifier=AlwaysVerifiedArtifact())
    partial = tmp_path / "uploads" / "upload-finalize.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"PK12")

    with pytest.raises(asyncio.CancelledError):
        await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    assert db.upload["status"] == "finalizing"
    assert not partial.exists()
    final_files = list((tmp_path / "artifacts").rglob("*.apk"))
    assert len(final_files) == 1
    assert final_files[0].read_bytes() == b"PK12"

    cleanup = await service.cleanup_expired_uploads(limit=10)

    assert cleanup == {"expired": 0, "reconciled": 1, "removed_files": 0, "failures": 0}
    assert db.upload["status"] == "finalized"
    assert db.finalize_writes == 1
    assert final_files[0].is_file()


@pytest.mark.asyncio
async def test_finalize_cancellation_after_commit_keeps_db_owned_final_and_retry_is_idempotent(tmp_path):
    db = AmbiguousFinalizeDB("after_commit")
    service = AndroidReleaseService(db, _config(tmp_path), verifier=AlwaysVerifiedArtifact())
    partial = tmp_path / "uploads" / "upload-finalize.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"PK12")

    with pytest.raises(asyncio.CancelledError):
        await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    final_files = list((tmp_path / "artifacts").rglob("*.apk"))
    assert db.upload["status"] == "finalized"
    assert db.finalize_writes == 1
    assert len(final_files) == 1
    assert final_files[0].read_bytes() == b"PK12"

    retry = await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    assert retry["already_finalized"] is True
    assert db.finalize_writes == 1
    assert final_files[0].is_file()


@pytest.mark.asyncio
async def test_finalize_unknown_outcome_never_removes_final_artifact(tmp_path):
    db = AmbiguousFinalizeDB("unknown_before_commit")
    service = AndroidReleaseService(db, _config(tmp_path), verifier=AlwaysVerifiedArtifact())
    partial = tmp_path / "uploads" / "upload-finalize.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"PK12")

    with pytest.raises(ConnectionError, match="ambiguous commit"):
        await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    final_files = list((tmp_path / "artifacts").rglob("*.apk"))
    assert db.upload["status"] == "finalizing"
    assert not partial.exists()
    assert len(final_files) == 1
    assert final_files[0].read_bytes() == b"PK12"

    retry = await service.finalize_upload(upload_id="upload-finalize", admin_user_id=1)

    assert retry["release"]["release_id"] == db.upload["release_id"]
    assert db.finalize_writes == 1
    assert final_files[0].is_file()


class _AsyncContext:
    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __init__(self, value):
        self.value = value


class RetiredDuringPublishConnection:
    def __init__(self, pre_read):
        self.pre_read = pre_read
        self.execute_calls = []

    def transaction(self):
        return _AsyncContext(None)

    async def fetchrow(self, query, *args):
        if "FROM android_release_channels" in query:
            return {
                "channel": "direct",
                "latest_version_code": 48,
                "min_supported_version_code": 45,
            }
        if "FOR UPDATE OF r" in query:
            return {**self.pre_read, "status": "retired"}
        raise AssertionError(query)

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"


class PublishRaceDB(PublishValidationDB):
    def __init__(self, row):
        super().__init__(row)
        self.connection = RetiredDuringPublishConnection(row)
        self._pool = self

    def acquire(self):
        return _AsyncContext(self.connection)


@pytest.mark.asyncio
async def test_publish_rechecks_retired_status_inside_transaction(tmp_path):
    staged = {
        "release_id": "release-49",
        "channel": "direct",
        "version_code": 49,
        "version_name": "0.5.2",
        "status": "staged",
        "artifact_id": "artifact-49",
        "artifact_kind": "apk",
        "artifact_package_name": "ru.extraarena.app",
        "signing_cert_sha256": "a" * 64,
        "verified": True,
        "latest_release_id": "release-48",
        "latest_version_code": 48,
        "min_supported_version_code": 45,
    }
    db = PublishRaceDB(staged)
    service = AndroidReleaseService(db, _config(tmp_path))

    with pytest.raises(AndroidReleaseError, match="android_release_retired"):
        await service.publish_release(
            release_id="release-49",
            required=False,
            expected_version_code=49,
            admin_user_id=1,
        )

    assert db.connection.execute_calls == []


def test_database_schema_contains_durable_android_release_tables():
    source = (__import__("pathlib").Path(__file__).parents[1] / "infrastructure" / "database.py").read_text()
    assert "SCHEMA_VERSION = 55" in source
    assert "CREATE TABLE IF NOT EXISTS android_release_channels" in source
    assert "CREATE TABLE IF NOT EXISTS android_releases" in source
    assert "CREATE TABLE IF NOT EXISTS android_release_artifacts" in source
    assert "CREATE TABLE IF NOT EXISTS android_release_uploads" in source
    assert "'uploading', 'finalizing', 'finalized'" in source
    assert "'failed', 'expired'" in source
    assert "CREATE TRIGGER android_release_floor_monotonic" in source
