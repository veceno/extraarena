from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Mapping
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)

ANDROID_RELEASE_CHANNELS = ("direct", "rustore")
ANDROID_ARTIFACT_KINDS = ("apk", "aab")
ANDROID_RELEASE_STATUSES = ("staged", "published", "superseded", "retired")
ANDROID_UPLOAD_STATUSES = (
    "uploading",
    "finalizing",
    "finalized",
    "aborted",
    "failed",
    "expired",
)
_ANDROID_FINALIZE_NAMESPACE = uuid.UUID("bd6a7844-d314-42fd-821c-f8702671c51a")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SERVER_COMPUTED_SHA256 = "0" * 64
ANDROID_UPLOAD_LOCK_REGISTRY_LIMIT = 1024
_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")
_AAPT_PACKAGE_RE = re.compile(
    r"package:\s+name='(?P<package>[^']+)'\s+versionCode='(?P<code>[^']+)'\s+versionName='(?P<name>[^']*)'"
)
_CERT_DIGEST_RE = re.compile(
    r"^\s*(?:(?:Signer\s+#\d+)|(?:V(?:2(?:\.0)?|3(?:\.[012])?|4(?:\.0)?)\s+Signer:))"
    r"\s+certificate\s+SHA-256\s+digest:\s*(?P<digest>[0-9A-Fa-f:]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class AndroidReleaseError(ValueError):
    def __init__(self, code: str, *, status: int = 400, details: Mapping[str, Any] | None = None):
        super().__init__(code)
        self.code = str(code)
        self.status = int(status)
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class AndroidReleaseConfig:
    enabled: bool
    storage_dir: Path
    public_base_url: str = ""
    package_name: str = "ru.extraarena.app"
    direct_signing_cert_sha256: str = ""
    rustore_signing_cert_sha256: str = ""
    apksigner_command: str = "apksigner"
    aapt_command: str = "aapt2"
    max_bytes: int = 1024 * 1024 * 1024
    chunk_bytes: int = 8 * 1024 * 1024
    upload_token_ttl_seconds: int = 60 * 60

    def __post_init__(self) -> None:
        if not _PACKAGE_RE.fullmatch(self.package_name):
            raise ValueError("invalid_android_package_name")
        if int(self.max_bytes) <= 0:
            raise ValueError("invalid_android_release_max_bytes")
        if int(self.chunk_bytes) <= 0 or int(self.chunk_bytes) > int(self.max_bytes):
            raise ValueError("invalid_android_release_chunk_bytes")
        if int(self.upload_token_ttl_seconds) < 60:
            raise ValueError("invalid_android_upload_token_ttl")
        public_base = str(self.public_base_url or "").strip()
        if public_base:
            if "\\" in public_base or any(
                ord(character) < 32 or ord(character) == 127 for character in public_base
            ):
                raise ValueError("invalid_android_release_public_base_url")
            try:
                parsed = urlsplit(public_base)
                _ = parsed.port
            except ValueError as exc:
                raise ValueError("invalid_android_release_public_base_url") from exc
            if (
                parsed.scheme.lower() != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("invalid_android_release_public_base_url")
            object.__setattr__(
                self,
                "public_base_url",
                urlunsplit(("https", parsed.netloc, "", "", "")),
            )


def normalize_cert_sha256(value: Any) -> str:
    return re.sub(r"[^0-9a-f]", "", str(value or "").lower())


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _row_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("verifier_result",):
        raw = data.get(key)
        if isinstance(raw, str):
            try:
                data[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return _json_value(data)


async def _sha256_file(path: Path) -> str:
    def _digest() -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return await asyncio.to_thread(_digest)


class AndroidArtifactVerifier:
    """Verify APK metadata and signature without invoking a shell."""

    def __init__(self, config: AndroidReleaseConfig):
        self.config = config

    @staticmethod
    def _resolve_command(command: str) -> list[str] | None:
        parts = shlex.split(str(command or "").strip())
        if not parts:
            return None
        executable = parts[0]
        if not os.path.isabs(executable):
            executable = shutil.which(executable) or ""
        if not executable or not Path(executable).is_file():
            return None
        return [executable, *parts[1:]]

    @staticmethod
    async def _run(command: list[str], *args: str) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return 124, "verifier_timeout"
        return int(process.returncode or 0), stdout.decode("utf-8", errors="replace")

    async def verify(self, path: Path, artifact_kind: str, *, channel: str = "direct") -> dict[str, Any]:
        result: dict[str, Any] = {
            "verified": False,
            "artifact_kind": artifact_kind,
            "channel": channel,
            "package_name": None,
            "version_code": None,
            "version_name": None,
            "signing_cert_sha256": None,
            "errors": [],
        }
        try:
            def _read_signature() -> bytes:
                with path.open("rb") as stream:
                    return stream.read(4)

            signature = await asyncio.to_thread(_read_signature)
        except OSError:
            result["errors"].append("artifact_unreadable")
            return result
        if not signature.startswith(b"PK"):
            result["errors"].append("invalid_zip_signature")
            return result
        if artifact_kind != "apk":
            result["errors"].append("aab_verifier_not_configured")
            return result

        aapt = self._resolve_command(self.config.aapt_command)
        apksigner = self._resolve_command(self.config.apksigner_command)
        if not aapt:
            result["errors"].append("aapt_unavailable")
        if not apksigner:
            result["errors"].append("apksigner_unavailable")
        if not aapt or not apksigner:
            return result

        aapt_status, aapt_output = await self._run(aapt, "dump", "badging", str(path))
        if aapt_status != 0:
            result["errors"].append("aapt_verification_failed")
            return result
        package_match = _AAPT_PACKAGE_RE.search(aapt_output)
        if not package_match:
            result["errors"].append("aapt_metadata_missing")
            return result
        try:
            version_code = int(package_match.group("code"))
        except (TypeError, ValueError):
            result["errors"].append("invalid_apk_version_code")
            return result
        result.update(
            package_name=package_match.group("package"),
            version_code=version_code,
            version_name=package_match.group("name"),
        )

        signer_status, signer_output = await self._run(
            apksigner,
            "verify",
            "--verbose",
            "--print-certs",
            str(path),
        )
        if signer_status != 0:
            result["errors"].append("apk_signature_invalid")
            return result
        certs = list(
            dict.fromkeys(
                cert
                for cert in (
                    normalize_cert_sha256(match.group("digest"))
                    for match in _CERT_DIGEST_RE.finditer(signer_output)
                )
                if _SHA256_RE.fullmatch(cert)
            )
        )
        if not certs:
            result["errors"].append("apk_signing_certificate_missing")
            return result
        expected_cert = normalize_cert_sha256(
            self.config.direct_signing_cert_sha256
            if channel == "direct"
            else self.config.rustore_signing_cert_sha256
        )
        selected_cert = expected_cert if expected_cert and expected_cert in certs else certs[0]
        result["signing_cert_sha256"] = selected_cert
        result["signing_certificates_sha256"] = certs

        if result["package_name"] != self.config.package_name:
            result["errors"].append("package_name_mismatch")
        if expected_cert and expected_cert not in certs:
            result["errors"].append("signing_certificate_mismatch")
        result["verified"] = not result["errors"]
        return result


class AndroidReleaseService:
    def __init__(
        self,
        db: Any,
        config: AndroidReleaseConfig,
        *,
        verifier: AndroidArtifactVerifier | None = None,
    ):
        self.db = db
        self.config = config
        self.verifier = verifier or AndroidArtifactVerifier(config)
        self._upload_locks: dict[str, asyncio.Lock] = {}
        self._upload_lock_refs: dict[str, int] = {}
        self._upload_locks_guard = asyncio.Lock()

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise AndroidReleaseError("android_releases_disabled", status=503)

    @staticmethod
    def _channel(value: Any) -> str:
        channel = str(value or "direct").strip().lower()
        if channel not in ANDROID_RELEASE_CHANNELS:
            raise AndroidReleaseError("unsupported_android_release_channel")
        return channel

    @staticmethod
    def _artifact_kind(value: Any) -> str:
        kind = str(value or "apk").strip().lower()
        if kind not in ANDROID_ARTIFACT_KINDS:
            raise AndroidReleaseError("unsupported_android_artifact_kind")
        return kind

    @staticmethod
    def _version_code(value: Any) -> int:
        try:
            code = int(value)
        except (TypeError, ValueError) as exc:
            raise AndroidReleaseError("invalid_android_version_code") from exc
        if code <= 0:
            raise AndroidReleaseError("invalid_android_version_code")
        return code

    def _path_for_key(self, storage_key: str) -> Path:
        root = self.config.storage_dir.expanduser().resolve()
        candidate = (root / str(storage_key or "")).resolve()
        if candidate == root or root not in candidate.parents:
            raise AndroidReleaseError("invalid_android_storage_key", status=500)
        return candidate

    async def _ensure_channel(self, channel: str) -> None:
        await self.db.execute(
            """
            INSERT INTO android_release_channels (channel, package_name)
            VALUES ($1, $2)
            ON CONFLICT (channel) DO UPDATE
            SET package_name = EXCLUDED.package_name,
                updated_at = NOW()
            """,
            channel,
            self.config.package_name,
        )

    async def cleanup_expired_uploads(self, *, limit: int = 100) -> dict[str, int]:
        """Reconcile finalization and retry deletion of app-owned upload tombstones."""
        self._require_enabled()
        safe_limit = max(1, min(int(limit), 500))
        candidates = await self.db.fetch(
            """
            SELECT upload_id, status, created_by
            FROM android_release_uploads
            WHERE status = 'finalizing'
               OR (status IN ('uploading', 'failed') AND token_expires_at <= NOW())
               OR (
                   status IN ('expired', 'aborted')
                   AND temp_storage_key = 'uploads/' || upload_id || '.part'
               )
            ORDER BY CASE WHEN status = 'finalizing' THEN 0 ELSE 1 END,
                     token_expires_at ASC
            LIMIT $1
            """,
            safe_limit,
        )
        expired = 0
        reconciled = 0
        removed_files = 0
        failures = 0
        for candidate in candidates:
            upload_id = str(candidate.get("upload_id") or "")
            if not upload_id:
                continue
            try:
                async with self._upload_lifecycle_lock(upload_id):
                    if str(candidate.get("status") or "") == "finalizing":
                        result = await self._finalize_upload_locked(
                            upload_id=upload_id,
                            admin_user_id=int(candidate.get("created_by") or 0),
                            dry_run=False,
                        )
                        if result.get("release"):
                            reconciled += 1
                        continue
                    row = await self.db.fetchrow(
                        """
                        WITH cleanup_candidate AS (
                            SELECT upload_id, status AS previous_status
                            FROM android_release_uploads
                            WHERE upload_id = $1
                              AND (
                                  (status IN ('uploading', 'failed') AND token_expires_at <= NOW())
                                  OR (
                                      status IN ('expired', 'aborted')
                                      AND temp_storage_key = 'uploads/' || upload_id || '.part'
                                  )
                              )
                            FOR UPDATE
                        )
                        UPDATE android_release_uploads AS uploads
                        SET status = CASE
                                WHEN cleanup_candidate.previous_status IN ('uploading', 'failed')
                                    THEN 'expired'
                                ELSE uploads.status
                            END,
                            token_hash = CASE
                                WHEN cleanup_candidate.previous_status IN ('uploading', 'failed')
                                    THEN 'expired:' || uploads.upload_id
                                ELSE uploads.token_hash
                            END,
                            updated_at = NOW()
                        FROM cleanup_candidate
                        WHERE uploads.upload_id = cleanup_candidate.upload_id
                        RETURNING uploads.temp_storage_key, cleanup_candidate.previous_status
                        """,
                        upload_id,
                    )
                    if not row:
                        continue
                    if str(row.get("previous_status") or "") == "uploading":
                        expired += 1
                    storage_key = str(row.get("temp_storage_key") or "")
                    terminal_status = (
                        "aborted"
                        if str(row.get("previous_status") or "") == "aborted"
                        else "expired"
                    )
                    removed_files += int(
                        await self._remove_upload_partial_and_tombstone(
                            upload_id=upload_id,
                            storage_key=storage_key,
                            terminal_status=terminal_status,
                        )
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                logger.warning(
                    "Failed to clean expired Android upload %s; it will be retried",
                    upload_id,
                    exc_info=True,
                )
        return {
            "expired": expired,
            "reconciled": reconciled,
            "removed_files": removed_files,
            "failures": failures,
        }

    async def _remove_upload_partial_and_tombstone(
        self,
        *,
        upload_id: str,
        storage_key: str,
        terminal_status: str,
    ) -> bool:
        """Delete only the canonical partial path, then persist a retry tombstone."""
        expected_key = f"uploads/{upload_id}.part"
        tombstone_key = f"{terminal_status}/{upload_id}.part"
        if storage_key == tombstone_key:
            return False
        if storage_key != expected_key or terminal_status not in {"aborted", "expired"}:
            raise AndroidReleaseError("android_release_upload_storage_mismatch", status=409)
        path = self._path_for_key(storage_key)
        existed = path.is_file()
        # unlink(2) is a short, non-blocking metadata operation. Keeping it in
        # this task avoids the same cancel-after-dispatch ambiguity as flock.
        path.unlink(missing_ok=True)
        # Change the path only after unlink succeeds. A crash or DB failure
        # leaves the row eligible for another idempotent cleanup pass.
        await self.db.execute(
            """
            UPDATE android_release_uploads
            SET temp_storage_key = $3, updated_at = NOW()
            WHERE upload_id = $1 AND status = $4 AND temp_storage_key = $2
            """,
            str(upload_id),
            storage_key,
            tombstone_key,
            terminal_status,
        )
        return existed

    @asynccontextmanager
    async def _upload_lifecycle_lock(self, upload_id: str) -> AsyncIterator[None]:
        """Serialize every filesystem/DB transition for one upload.

        The asyncio lock covers concurrent tasks using this service instance;
        the advisory lock covers independent web workers sharing storage.
        Keeping append, finalize, and abort on the same lock prevents a partial
        file from being moved or removed while another worker is writing it.
        """
        normalized_upload_id = str(upload_id)
        async with self._upload_locks_guard:
            lock = self._upload_locks.get(normalized_upload_id)
            if lock is None:
                if len(self._upload_locks) >= ANDROID_UPLOAD_LOCK_REGISTRY_LIMIT:
                    raise AndroidReleaseError(
                        "android_release_upload_lock_capacity_reached",
                        status=503,
                    )
                lock = asyncio.Lock()
                self._upload_locks[normalized_upload_id] = lock
                self._upload_lock_refs[normalized_upload_id] = 0
            self._upload_lock_refs[normalized_upload_id] += 1
        try:
            async with lock:
                lock_name = hashlib.sha256(normalized_upload_id.encode("utf-8")).hexdigest()
                lock_path = self._path_for_key(f"locks/{lock_name}.lock")
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_stream = lock_path.open("a+b")
                acquired = False
                try:
                    while not acquired:
                        try:
                            fcntl.flock(
                                lock_stream.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                            acquired = True
                        except BlockingIOError:
                            await asyncio.sleep(0.05)
                    yield
                finally:
                    if acquired:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
                    lock_stream.close()
        finally:
            # Keep the registry proportional to active holders/waiters. The
            # per-upload lock file remains the cross-process rendezvous point,
            # but it is created only after the caller has authenticated/read a
            # real upload row.
            async with self._upload_locks_guard:
                remaining = self._upload_lock_refs.get(normalized_upload_id, 1) - 1
                if remaining <= 0:
                    self._upload_lock_refs.pop(normalized_upload_id, None)
                    if self._upload_locks.get(normalized_upload_id) is lock:
                        self._upload_locks.pop(normalized_upload_id, None)
                else:
                    self._upload_lock_refs[normalized_upload_id] = remaining

    @staticmethod
    def _public_upload(row: Mapping[str, Any]) -> dict[str, Any]:
        data = _row_dict(row) or {}
        data.pop("token_hash", None)
        data.pop("temp_storage_key", None)
        return data

    async def list_releases(
        self,
        *,
        channel: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_retired: bool = True,
    ) -> dict[str, Any]:
        self._require_enabled()
        normalized_channel = self._channel(channel) if channel else None
        safe_limit = max(1, min(int(limit), 200))
        safe_offset = max(0, int(offset))
        rows = await self.db.fetch(
            """
            SELECT r.*, a.artifact_id, a.artifact_kind, a.original_filename,
                   a.size_bytes, a.sha256, a.package_name AS artifact_package_name,
                   a.version_code AS artifact_version_code,
                   a.version_name AS artifact_version_name,
                   a.signing_cert_sha256, a.verified, a.verifier_result,
                   ARRAY(
                       SELECT aa.artifact_kind FROM android_release_artifacts aa
                       WHERE aa.release_id = r.release_id
                       ORDER BY (aa.artifact_kind = 'apk') DESC, aa.artifact_kind
                   ) AS artifact_kinds,
                   c.latest_release_id, c.latest_version_code, c.min_supported_version_code
            FROM android_releases r
            JOIN android_release_channels c ON c.channel = r.channel
            LEFT JOIN LATERAL (
                SELECT * FROM android_release_artifacts aa
                WHERE aa.release_id = r.release_id
                ORDER BY (aa.artifact_kind = 'apk') DESC, aa.created_at DESC
                LIMIT 1
            ) a ON TRUE
            WHERE ($1::text IS NULL OR r.channel = $1::text)
              AND ($2::boolean OR r.status <> 'retired')
            ORDER BY r.created_at DESC, r.version_code DESC
            LIMIT $3 OFFSET $4
            """,
            normalized_channel,
            bool(include_retired),
            safe_limit,
            safe_offset,
        )
        return {
            "releases": [_row_dict(row) for row in rows],
            "limit": safe_limit,
            "offset": safe_offset,
            "channel": normalized_channel,
        }

    async def read_release(self, release_id: str) -> dict[str, Any]:
        self._require_enabled()
        row = await self.db.fetchrow(
            """
            SELECT r.*, a.artifact_id, a.artifact_kind, a.original_filename,
                   a.storage_key, a.size_bytes, a.sha256,
                   a.package_name AS artifact_package_name,
                   a.version_code AS artifact_version_code,
                   a.version_name AS artifact_version_name,
                   a.signing_cert_sha256, a.verified, a.verifier_result,
                   ARRAY(
                       SELECT aa.artifact_kind FROM android_release_artifacts aa
                       WHERE aa.release_id = r.release_id
                       ORDER BY (aa.artifact_kind = 'apk') DESC, aa.artifact_kind
                   ) AS artifact_kinds,
                   c.latest_release_id, c.latest_version_code, c.min_supported_version_code
            FROM android_releases r
            JOIN android_release_channels c ON c.channel = r.channel
            LEFT JOIN LATERAL (
                SELECT * FROM android_release_artifacts aa
                WHERE aa.release_id = r.release_id
                ORDER BY (aa.artifact_kind = 'apk') DESC, aa.created_at DESC
                LIMIT 1
            ) a ON TRUE
            WHERE r.release_id = $1
            """,
            str(release_id),
        )
        if not row:
            raise AndroidReleaseError("android_release_not_found", status=404)
        return _row_dict(row) or {}

    async def prepare_upload(
        self,
        *,
        channel: str,
        artifact_kind: str,
        filename: str,
        size_bytes: int,
        sha256: str,
        version_code: int,
        version_name: str,
        release_notes: str = "",
        admin_user_id: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._require_enabled()
        normalized_channel = self._channel(channel)
        kind = self._artifact_kind(artifact_kind)
        clean_filename = Path(str(filename or "").strip()).name
        if not clean_filename or len(clean_filename) > 200:
            raise AndroidReleaseError("invalid_android_artifact_filename")
        if not clean_filename.lower().endswith(f".{kind}"):
            raise AndroidReleaseError("android_artifact_extension_mismatch")
        try:
            expected_size = int(size_bytes)
        except (TypeError, ValueError) as exc:
            raise AndroidReleaseError("invalid_android_artifact_size") from exc
        if expected_size <= 0 or expected_size > int(self.config.max_bytes):
            raise AndroidReleaseError("android_artifact_size_out_of_range", status=413)
        expected_sha = str(sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(expected_sha):
            raise AndroidReleaseError("invalid_android_artifact_sha256")
        expected_code = self._version_code(version_code)
        expected_name = str(version_name or "").strip()
        if not expected_name or len(expected_name) > 80:
            raise AndroidReleaseError("invalid_android_version_name")
        notes = str(release_notes or "").strip()
        if len(notes) > 20_000:
            raise AndroidReleaseError("android_release_notes_too_long")

        existing = await self.db.fetchrow(
            """
            SELECT r.release_id, r.status, r.version_name, r.release_notes,
                   EXISTS (
                       SELECT 1 FROM android_release_artifacts a
                       WHERE a.release_id = r.release_id AND a.artifact_kind = $3
                   ) AS artifact_kind_exists
            FROM android_releases r
            WHERE r.channel = $1 AND r.version_code = $2
            """,
            normalized_channel,
            expected_code,
            kind,
        )
        if existing:
            if str(existing.get("status")) != "staged":
                raise AndroidReleaseError("android_release_version_exists", status=409)
            if str(existing.get("version_name") or "") != expected_name:
                raise AndroidReleaseError("android_release_version_name_conflict", status=409)
            if bool(existing.get("artifact_kind_exists")):
                raise AndroidReleaseError("android_release_artifact_kind_exists", status=409)
            # Release-level metadata belongs to the existing staged row; an
            # attached artifact must not silently rewrite it.
            notes = str(existing.get("release_notes") or "")
        preview = {
            "dry_run": bool(dry_run),
            "channel": normalized_channel,
            "artifact_kind": kind,
            "filename": clean_filename,
            "size_bytes": expected_size,
            "sha256": expected_sha,
            "version_code": expected_code,
            "version_name": expected_name,
            "release_notes": notes,
            "chunk_bytes": int(self.config.chunk_bytes),
            "publishable_kind": kind == "apk",
            "direct_publishable_kind": kind == "apk",
            "attach_to_release_id": str(existing.get("release_id")) if existing else None,
        }
        if dry_run:
            return preview

        await self._ensure_channel(normalized_channel)
        upload_id = str(uuid.uuid4())
        upload_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(upload_token.encode("utf-8")).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(self.config.upload_token_ttl_seconds))
        temp_key = f"uploads/{upload_id}.part"
        temp_path = self._path_for_key(temp_key)
        await asyncio.to_thread(temp_path.parent.mkdir, parents=True, exist_ok=True)
        row = await self.db.fetchrow(
            """
            INSERT INTO android_release_uploads (
                upload_id, channel, artifact_kind, original_filename,
                expected_size_bytes, expected_sha256, expected_version_code,
                expected_version_name, release_notes, received_bytes,
                temp_storage_key, token_hash, status, created_by,
                token_expires_at, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, 0,
                $10, $11, 'uploading', $12, $13, NOW(), NOW()
            )
            RETURNING *
            """,
            upload_id,
            normalized_channel,
            kind,
            clean_filename,
            expected_size,
            expected_sha,
            expected_code,
            expected_name,
            notes,
            temp_key,
            token_hash,
            int(admin_user_id),
            expires_at,
        )
        return {
            **preview,
            "dry_run": False,
            "upload": self._public_upload(row),
            "upload_token": upload_token,
            "upload_url": f"/api/admin/android-releases/uploads/{upload_id}",
        }

    async def _upload_row(self, upload_id: str) -> dict[str, Any]:
        row = await self.db.fetchrow(
            "SELECT * FROM android_release_uploads WHERE upload_id = $1",
            str(upload_id),
        )
        if not row:
            raise AndroidReleaseError("android_release_upload_not_found", status=404)
        return dict(row)

    @staticmethod
    def _upload_expired(row: Mapping[str, Any]) -> bool:
        expires = row.get("token_expires_at")
        if isinstance(expires, str):
            try:
                expires = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            except ValueError:
                return True
        if not isinstance(expires, datetime):
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires <= datetime.now(timezone.utc)

    async def authenticate_upload(self, upload_id: str, upload_token: str) -> dict[str, Any]:
        self._require_enabled()
        row = await self._upload_row(upload_id)
        presented = hashlib.sha256(str(upload_token or "").encode("utf-8")).hexdigest()
        if not upload_token or not hmac.compare_digest(presented, str(row.get("token_hash") or "")):
            raise AndroidReleaseError("android_release_upload_auth_required", status=401)
        if self._upload_expired(row):
            raise AndroidReleaseError("android_release_upload_expired", status=410)
        return row

    async def upload_status(self, upload_id: str) -> dict[str, Any]:
        self._require_enabled()
        return self._public_upload(await self._upload_row(upload_id))

    async def append_upload_chunk(
        self,
        *,
        upload_id: str,
        offset: int,
        chunks: AsyncIterable[bytes],
        upload_token: str | None = None,
    ) -> dict[str, Any]:
        self._require_enabled()
        try:
            expected_offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise AndroidReleaseError("invalid_android_upload_offset") from exc
        # Reject unknown IDs and bad scoped tickets before allocating a local
        # lock or creating a persistent flock inode. Re-authenticate after the
        # lock as well so expiry/revocation and the winner's offset are fresh.
        if upload_token is not None:
            await self.authenticate_upload(upload_id, upload_token)
        else:
            await self._upload_row(upload_id)
        async with self._upload_lifecycle_lock(upload_id):
            # Re-read only after acquiring the cross-process lock. A waiter
            # must observe the winner's committed offset before it decides
            # whether it may touch the partial file.
            row = (
                await self.authenticate_upload(upload_id, upload_token)
                if upload_token is not None
                else await self._upload_row(upload_id)
            )
            if str(row.get("status")) != "uploading":
                raise AndroidReleaseError("android_release_upload_not_writable", status=409)
            received = int(row.get("received_bytes") or 0)
            if expected_offset != received:
                raise AndroidReleaseError(
                    "android_release_upload_offset_conflict",
                    status=409,
                    details={"expected_offset": received},
                )
            path = self._path_for_key(str(row.get("temp_storage_key") or ""))
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            actual_size = path.stat().st_size if path.exists() else 0
            if actual_size > received:
                # A previous writer may have fsynced bytes and then lost the
                # database UPDATE response. This fresh row read is authoritative:
                # an oversized suffix is not committed and may be discarded.
                os.truncate(path, received)
                actual_size = received
            if actual_size != received:
                raise AndroidReleaseError("android_release_upload_storage_mismatch", status=409)

            stream = await asyncio.to_thread(path.open, "ab")
            written = 0
            try:
                async for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray)) or not chunk:
                        continue
                    written += len(chunk)
                    if written > int(self.config.chunk_bytes):
                        raise AndroidReleaseError("android_release_chunk_too_large", status=413)
                    if received + written > int(row.get("expected_size_bytes") or 0):
                        raise AndroidReleaseError("android_release_upload_exceeds_expected_size", status=413)
                    await asyncio.to_thread(stream.write, bytes(chunk))
                if written <= 0:
                    raise AndroidReleaseError("android_release_empty_chunk")
                await asyncio.to_thread(stream.flush)
                await asyncio.to_thread(os.fsync, stream.fileno())
            except BaseException:
                await asyncio.to_thread(stream.close)
                await asyncio.to_thread(os.truncate, path, received)
                raise
            else:
                await asyncio.to_thread(stream.close)

            try:
                updated = await self.db.fetchrow(
                    """
                    UPDATE android_release_uploads
                    SET received_bytes = received_bytes + $2, updated_at = NOW()
                    WHERE upload_id = $1 AND status = 'uploading' AND received_bytes = $3
                    RETURNING *
                    """,
                    str(upload_id),
                    written,
                    received,
                )
            except BaseException:
                outcome = await self._read_upload_outcome(upload_id)
                if outcome is not None and str(outcome.get("status") or "") == "uploading":
                    committed_offset = int(outcome.get("received_bytes") or 0)
                    if committed_offset == received:
                        os.truncate(path, received)
                    # If committed_offset == received + written, the UPDATE
                    # committed and the fsynced suffix belongs to it. Any other
                    # state is left intact for a later fresh-read reconciliation.
                raise
            if not updated:
                outcome = await self._read_upload_outcome(upload_id)
                if outcome is not None and int(outcome.get("received_bytes") or 0) == received:
                    os.truncate(path, received)
                raise AndroidReleaseError("android_release_upload_offset_conflict", status=409)
            return self._public_upload(updated)

    def _finalize_identity(self, row: Mapping[str, Any]) -> tuple[str, str, str, Path]:
        upload_id = str(row.get("upload_id") or "")
        channel = self._channel(row.get("channel"))
        kind = self._artifact_kind(row.get("artifact_kind"))
        version_code = int(row.get("expected_version_code") or 0)
        extension = ".apk" if kind == "apk" else ".aab"
        release_id = str(uuid.uuid5(_ANDROID_FINALIZE_NAMESPACE, f"release:{upload_id}"))
        artifact_id = str(uuid.uuid5(_ANDROID_FINALIZE_NAMESPACE, f"artifact:{upload_id}"))
        final_key = f"artifacts/{channel}/{version_code}/{artifact_id}{extension}"
        return release_id, artifact_id, final_key, self._path_for_key(final_key)

    async def _read_upload_outcome(self, upload_id: str) -> dict[str, Any] | None:
        """Best-effort fresh read after an ambiguous database operation."""
        try:
            return await self._upload_row(upload_id)
        except (Exception, asyncio.CancelledError):
            logger.warning(
                "Android upload %s outcome is unknown; preserving filesystem state",
                upload_id,
                exc_info=True,
            )
            return None

    async def _recover_finalize_business_conflict(
        self,
        *,
        row: Mapping[str, Any],
        final_path: Path,
    ) -> bool:
        """Return a definitely-uncommitted artifact to an abortable state."""
        upload_id = str(row.get("upload_id") or "")
        outcome = await self._read_upload_outcome(upload_id)
        if outcome is None or str(outcome.get("status") or "") != "finalizing":
            # A finalized or unknown outcome owns the final path; never move it.
            return False
        partial_path = self._path_for_key(str(outcome.get("temp_storage_key") or ""))
        final_exists = final_path.is_file()
        partial_exists = partial_path.is_file()
        if final_exists and not partial_exists:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final_path, partial_path)
        elif not partial_exists:
            logger.error(
                "Android upload %s conflict could not be reconciled: storage is missing or ambiguous",
                upload_id,
            )
            return False
        elif final_exists:
            logger.error(
                "Android upload %s conflict left both partial and final paths; preserving both",
                upload_id,
            )
            return False
        try:
            updated = await self.db.fetchrow(
                """
                UPDATE android_release_uploads
                SET status = 'failed', updated_at = NOW()
                WHERE upload_id = $1 AND status = 'finalizing'
                RETURNING *
                """,
                upload_id,
            )
        except (Exception, asyncio.CancelledError):
            # The partial path remains complete. A later reconciliation can
            # retry this CAS without losing or duplicating bytes.
            logger.warning(
                "Android upload %s conflict file was restored but failed-state write is ambiguous",
                upload_id,
                exc_info=True,
            )
            raise
        return bool(updated)

    async def _mark_apk_verification_failed(
        self,
        *,
        row: Mapping[str, Any],
        final_path: Path,
    ) -> None:
        """Keep rejected APK bytes retryable without creating a release tuple."""
        upload_id = str(row.get("upload_id") or "")
        status = str(row.get("status") or "")
        if status == "finalizing":
            recovered = await self._recover_finalize_business_conflict(
                row=row,
                final_path=final_path,
            )
            if not recovered:
                raise AndroidReleaseError(
                    "android_release_upload_storage_mismatch",
                    status=409,
                )
            return
        updated = await self.db.fetchrow(
            """
            UPDATE android_release_uploads
            SET status = 'failed', updated_at = NOW()
            WHERE upload_id = $1 AND status IN ('uploading', 'failed')
            RETURNING *
            """,
            upload_id,
        )
        if not updated:
            raise AndroidReleaseError(
                "android_release_upload_finalize_conflict",
                status=409,
            )

    async def _finalize_preview(self, upload_id: str) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
        row = await self._upload_row(upload_id)
        status = str(row.get("status") or "")
        if status not in {"uploading", "failed", "finalizing"}:
            if status == "finalized" and row.get("release_id"):
                release = await self.read_release(str(row["release_id"]))
                return row, Path(), str(release.get("sha256") or ""), {"verified": bool(release.get("verified"))}
            raise AndroidReleaseError("android_release_upload_not_finalizable", status=409)
        expected_size = int(row.get("expected_size_bytes") or 0)
        received = int(row.get("received_bytes") or 0)
        if received != expected_size:
            raise AndroidReleaseError(
                "android_release_upload_incomplete",
                status=409,
                details={"received_bytes": received, "expected_size_bytes": expected_size},
            )
        partial_path = self._path_for_key(str(row.get("temp_storage_key") or ""))
        _, _, _, final_path = self._finalize_identity(row)
        if status == "finalizing":
            partial_exists = partial_path.is_file()
            final_exists = final_path.is_file()
            # os.replace is atomic. Seeing both or neither means that storage
            # no longer matches any valid state, so never guess which to erase.
            if partial_exists == final_exists:
                raise AndroidReleaseError("android_release_upload_storage_mismatch", status=409)
            path = final_path if final_exists else partial_path
        else:
            path = partial_path
        if not path.is_file() or path.stat().st_size != expected_size:
            raise AndroidReleaseError("android_release_upload_storage_mismatch", status=409)
        digest = await _sha256_file(path)
        expected_digest = str(row.get("expected_sha256") or "")
        if expected_digest != SERVER_COMPUTED_SHA256 and not hmac.compare_digest(digest, expected_digest):
            raise AndroidReleaseError(
                "android_release_artifact_hash_mismatch",
                status=422,
                details={"actual_sha256": digest},
            )
        verification = await self.verifier.verify(
            path,
            str(row.get("artifact_kind") or ""),
            channel=self._channel(row.get("channel")),
        )
        derived_code = verification.get("version_code")
        derived_name = verification.get("version_name")
        if derived_code is not None and int(derived_code) != int(row.get("expected_version_code") or 0):
            raise AndroidReleaseError("android_release_version_code_mismatch", status=422)
        if derived_name is not None and str(derived_name) != str(row.get("expected_version_name") or ""):
            raise AndroidReleaseError("android_release_version_name_mismatch", status=422)
        return row, path, digest, verification

    async def finalize_upload(
        self,
        *,
        upload_id: str,
        admin_user_id: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._require_enabled()
        # Admin-controlled IDs still should not grow the lock registry or lock
        # directory when the upload does not exist.
        await self._upload_row(upload_id)
        async with self._upload_lifecycle_lock(upload_id):
            return await self._finalize_upload_locked(
                upload_id=upload_id,
                admin_user_id=admin_user_id,
                dry_run=dry_run,
            )

    async def _finalize_upload_locked(
        self,
        *,
        upload_id: str,
        admin_user_id: int,
        dry_run: bool,
    ) -> dict[str, Any]:
        row, path, digest, verification = await self._finalize_preview(upload_id)
        if str(row.get("status")) == "finalized" and row.get("release_id"):
            return {
                "dry_run": bool(dry_run),
                "already_finalized": True,
                "release": await self.read_release(str(row["release_id"])),
            }
        preview = {
            "dry_run": bool(dry_run),
            "upload_id": str(upload_id),
            "sha256": digest,
            "verification": verification,
            "publishable": bool(verification.get("verified")),
            "verification_errors": list(verification.get("errors") or []),
        }
        if dry_run:
            return preview

        release_id, artifact_id, final_key, final_path = self._finalize_identity(row)
        channel = self._channel(row.get("channel"))
        kind = self._artifact_kind(row.get("artifact_kind"))
        version_code = int(row.get("expected_version_code") or 0)
        extension = ".apk" if kind == "apk" else ".aab"
        if kind == "apk" and not bool(verification.get("verified")):
            # An APK is never staged unless package/signature verification
            # succeeds. Keep the complete bytes in a typed failed state so a
            # transient verifier outage can be retried, or the upload can be
            # aborted and replaced without consuming (channel, version_code).
            await self._mark_apk_verification_failed(
                row=row,
                final_path=final_path,
            )
            raise AndroidReleaseError(
                "android_release_apk_verification_failed",
                status=422,
                details={
                    "verification": verification,
                    "retryable": True,
                    "replacement_allowed": True,
                },
            )
        if str(row.get("status") or "") in {"uploading", "failed"}:
            try:
                transitioned = await self.db.fetchrow(
                    """
                    UPDATE android_release_uploads
                    SET status = 'finalizing', updated_at = NOW()
                    WHERE upload_id = $1 AND status IN ('uploading', 'failed')
                    RETURNING *
                    """,
                    str(upload_id),
                )
            except BaseException:
                # A cancelled/failed UPDATE may have committed. Read it back,
                # but do not touch either filesystem location in this path.
                await self._read_upload_outcome(upload_id)
                raise
            if not transitioned:
                outcome = await self._upload_row(upload_id)
                if str(outcome.get("status") or "") == "finalized" and outcome.get("release_id"):
                    return {
                        **preview,
                        "dry_run": False,
                        "already_finalized": True,
                        "release": await self.read_release(str(outcome["release_id"])),
                    }
                if str(outcome.get("status") or "") != "finalizing":
                    raise AndroidReleaseError("android_release_upload_finalize_conflict", status=409)
            else:
                row = dict(transitioned)

        # The durable finalizing marker is committed before the atomic move.
        # Synchronous metadata operations avoid releasing the lifecycle lock
        # while a cancelled worker thread may still be moving the file.
        if path != final_path:
            if final_path.exists():
                raise AndroidReleaseError("android_release_upload_storage_mismatch", status=409)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, final_path)
        try:
            created = await self.db.fetchrow(
                """
                WITH eligible_upload AS (
                    SELECT upload_id FROM android_release_uploads
                    WHERE upload_id = $17 AND status = 'finalizing'
                    FOR UPDATE
                ), selected_release AS (
                    INSERT INTO android_releases (
                        release_id, channel, version_code, version_name, status,
                        release_notes, created_by, created_at, updated_at
                    )
                    SELECT $1, $2, $3, $4, 'staged', $5, $6, NOW(), NOW()
                    FROM eligible_upload
                    ON CONFLICT (channel, version_code) DO UPDATE
                    SET updated_at = android_releases.updated_at
                    WHERE android_releases.status = 'staged'
                      AND android_releases.version_name = EXCLUDED.version_name
                    RETURNING release_id
                ), created_artifact AS (
                    INSERT INTO android_release_artifacts (
                        artifact_id, release_id, artifact_kind, original_filename,
                        storage_key, size_bytes, sha256, package_name, version_code,
                        version_name, signing_cert_sha256, verified, verifier_result,
                        created_at
                    )
                    SELECT $7, release_id, $8, $9, $10, $11, $12, $13, $3,
                           $4, $14, $15, $16::jsonb, NOW()
                    FROM selected_release
                    ON CONFLICT (release_id, artifact_kind) DO NOTHING
                    RETURNING release_id
                )
                UPDATE android_release_uploads
                SET status = 'finalized', release_id = (SELECT release_id FROM created_artifact),
                    finalized_at = NOW(), updated_at = NOW()
                WHERE upload_id = (SELECT upload_id FROM eligible_upload)
                  AND EXISTS (SELECT 1 FROM created_artifact)
                RETURNING release_id
                """,
                release_id,
                channel,
                version_code,
                str(row.get("expected_version_name") or ""),
                str(row.get("release_notes") or ""),
                int(admin_user_id),
                artifact_id,
                kind,
                str(row.get("original_filename") or f"release{extension}"),
                final_key,
                int(row.get("expected_size_bytes") or 0),
                digest,
                str(verification.get("package_name") or self.config.package_name),
                normalize_cert_sha256(verification.get("signing_cert_sha256")),
                bool(verification.get("verified")),
                json.dumps(verification, ensure_ascii=False, separators=(",", ":")),
                str(upload_id),
            )
            if not created:
                outcome = await self._read_upload_outcome(upload_id)
                if outcome and str(outcome.get("status") or "") == "finalized" and outcome.get("release_id"):
                    return {
                        **preview,
                        "dry_run": False,
                        "already_finalized": True,
                        "release": await self.read_release(str(outcome["release_id"])),
                    }
                conflict = await self.db.fetchrow(
                    """
                    SELECT r.release_id, r.status, r.version_name, r.release_notes,
                           EXISTS (
                               SELECT 1 FROM android_release_artifacts a
                               WHERE a.release_id = r.release_id AND a.artifact_kind = $3
                           ) AS artifact_kind_exists
                    FROM android_releases r
                    WHERE r.channel = $1 AND r.version_code = $2
                    """,
                    channel,
                    version_code,
                    kind,
                )
                failure = AndroidReleaseError(
                    "android_release_upload_finalize_conflict",
                    status=409,
                )
                if conflict:
                    if str(conflict.get("status")) != "staged":
                        failure = AndroidReleaseError("android_release_version_exists", status=409)
                    elif str(conflict.get("version_name") or "") != str(row.get("expected_version_name") or ""):
                        failure = AndroidReleaseError("android_release_version_name_conflict", status=409)
                    elif bool(conflict.get("artifact_kind_exists")):
                        failure = AndroidReleaseError("android_release_artifact_kind_exists", status=409)
                await self._recover_finalize_business_conflict(
                    row=row,
                    final_path=final_path,
                )
                raise failure
        except BaseException as exc:
            # fetchrow may have committed and then lost/cancelled its response.
            # A fresh read distinguishes a known commit from a resumable
            # finalizing row. If readback also fails, the outcome is unknown;
            # keeping the deterministic final path is the only safe choice.
            outcome = await self._read_upload_outcome(upload_id)
            if outcome and str(outcome.get("status") or "") == "finalized":
                logger.info(
                    "Android upload %s finalized despite an ambiguous response",
                    upload_id,
                )
            if getattr(exc, "sqlstate", None) == "23505":
                if outcome and str(outcome.get("status") or "") == "finalizing":
                    await self._recover_finalize_business_conflict(
                        row=outcome,
                        final_path=final_path,
                    )
                raise AndroidReleaseError("android_release_version_exists", status=409) from exc
            raise
        resolved_release_id = str(created["release_id"])
        return {
            **preview,
            "dry_run": False,
            "release": await self.read_release(resolved_release_id),
        }

    async def abort_upload(
        self,
        *,
        upload_id: str,
        admin_user_id: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._require_enabled()
        await self._upload_row(upload_id)
        async with self._upload_lifecycle_lock(upload_id):
            return await self._abort_upload_locked(
                upload_id=upload_id,
                admin_user_id=admin_user_id,
                dry_run=dry_run,
            )

    async def _abort_upload_locked(
        self,
        *,
        upload_id: str,
        admin_user_id: int,
        dry_run: bool,
    ) -> dict[str, Any]:
        row = await self._upload_row(upload_id)
        preview = {
            "dry_run": bool(dry_run),
            "upload": self._public_upload(row),
            "will_delete_partial_bytes": int(row.get("received_bytes") or 0),
        }
        if dry_run:
            return preview
        status = str(row.get("status") or "")
        if status in {"finalizing", "finalized"}:
            raise AndroidReleaseError("android_release_upload_already_finalized", status=409)
        if status == "aborted":
            transitioned = row
        else:
            try:
                transitioned = await self.db.fetchrow(
                    """
                    WITH abort_candidate AS (
                        SELECT upload_id, status AS previous_status
                        FROM android_release_uploads
                        WHERE upload_id = $1
                          AND status IN ('uploading', 'failed', 'expired')
                        FOR UPDATE
                    )
                    UPDATE android_release_uploads AS uploads
                    SET status = 'aborted',
                        token_hash = 'aborted:' || uploads.upload_id,
                        aborted_at = NOW(), updated_at = NOW()
                    FROM abort_candidate
                    WHERE uploads.upload_id = abort_candidate.upload_id
                    RETURNING uploads.*, abort_candidate.previous_status
                    """,
                    str(upload_id),
                )
            except BaseException:
                outcome = await self._read_upload_outcome(upload_id)
                if not outcome or str(outcome.get("status") or "") != "aborted":
                    # Unknown or uncommitted abort: deleting the partial could
                    # destroy a still-writable upload, so leave it untouched.
                    raise
                transitioned = outcome
                storage_key = str(transitioned.get("temp_storage_key") or "")
                try:
                    await self._remove_upload_partial_and_tombstone(
                        upload_id=str(upload_id),
                        storage_key=storage_key,
                        terminal_status="aborted",
                    )
                except Exception:
                    logger.warning(
                        "Android upload %s was aborted but cleanup remains pending",
                        upload_id,
                        exc_info=True,
                    )
                raise
            if not transitioned:
                outcome = await self._upload_row(upload_id)
                if str(outcome.get("status") or "") in {"finalizing", "finalized"}:
                    raise AndroidReleaseError("android_release_upload_already_finalized", status=409)
                if str(outcome.get("status") or "") != "aborted":
                    raise AndroidReleaseError("android_release_upload_abort_conflict", status=409)
                transitioned = outcome

        storage_key = str(transitioned.get("temp_storage_key") or "")
        cleanup_pending = False
        try:
            await self._remove_upload_partial_and_tombstone(
                upload_id=str(upload_id),
                storage_key=storage_key,
                terminal_status="aborted",
            )
        except Exception:
            cleanup_pending = True
            logger.warning(
                "Android upload %s was aborted but cleanup remains pending",
                upload_id,
                exc_info=True,
            )
        return {
            **preview,
            "dry_run": False,
            "aborted": True,
            "cleanup_pending": cleanup_pending,
            "admin_user_id": int(admin_user_id),
        }

    def _validate_publishable(self, release: Mapping[str, Any], *, expected_version_code: int | None = None) -> None:
        if expected_version_code is not None and int(release.get("version_code") or 0) != int(expected_version_code):
            raise AndroidReleaseError("android_release_version_confirmation_mismatch", status=409)
        if str(release.get("status")) == "retired":
            raise AndroidReleaseError("android_release_retired", status=409)
        if not release.get("artifact_id"):
            raise AndroidReleaseError("android_release_artifact_missing", status=409)
        if str(release.get("artifact_kind")) != "apk":
            raise AndroidReleaseError("android_aab_publish_not_supported", status=422)
        if not bool(release.get("verified")):
            raise AndroidReleaseError(
                "android_release_artifact_not_verified",
                status=422,
                details={"verification": release.get("verifier_result") or {}},
            )
        if str(release.get("artifact_package_name") or "") != self.config.package_name:
            raise AndroidReleaseError("android_release_package_mismatch", status=422)
        if str(release.get("channel")) == "direct":
            expected_cert = normalize_cert_sha256(self.config.direct_signing_cert_sha256)
            if not expected_cert:
                raise AndroidReleaseError("android_direct_signing_certificate_not_configured", status=503)
            if not hmac.compare_digest(
                expected_cert,
                normalize_cert_sha256(release.get("signing_cert_sha256")),
            ):
                raise AndroidReleaseError("android_release_signing_certificate_mismatch", status=422)
        elif str(release.get("channel")) == "rustore":
            expected_cert = normalize_cert_sha256(self.config.rustore_signing_cert_sha256)
            if expected_cert and not hmac.compare_digest(
                expected_cert,
                normalize_cert_sha256(release.get("signing_cert_sha256")),
            ):
                raise AndroidReleaseError("android_release_signing_certificate_mismatch", status=422)

    async def publish_release(
        self,
        *,
        release_id: str,
        required: bool,
        admin_user_id: int,
        expected_version_code: int | None = None,
        store_release_confirmed: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._require_enabled()
        release = await self.read_release(release_id)
        self._validate_publishable(release, expected_version_code=expected_version_code)
        if str(release.get("channel")) == "rustore" and not store_release_confirmed:
            raise AndroidReleaseError("android_rustore_release_not_confirmed_live", status=409)
        latest_code = int(release.get("latest_version_code") or 0)
        version_code = int(release.get("version_code") or 0)
        if str(release.get("status")) != "published" and version_code <= latest_code:
            raise AndroidReleaseError("android_release_version_not_newer", status=409)
        floor_before = int(release.get("min_supported_version_code") or 0)
        effective_required = bool(required or release.get("required_at_publish"))
        floor_after = max(floor_before, version_code if effective_required else floor_before)
        preview = {
            "dry_run": bool(dry_run),
            "release_id": str(release_id),
            "channel": release.get("channel"),
            "version_code": version_code,
            "version_name": release.get("version_name"),
            "required_requested": bool(required),
            "store_release_confirmed": bool(store_release_confirmed),
            "required_at_publish": effective_required,
            "latest_version_code_before": latest_code,
            "latest_version_code_after": max(latest_code, version_code),
            "min_supported_version_code_before": floor_before,
            "min_supported_version_code_after": floor_after,
            "artifact": {
                "artifact_kind": release.get("artifact_kind"),
                "sha256": release.get("sha256"),
                "size_bytes": release.get("size_bytes"),
                "package_name": release.get("artifact_package_name"),
                "signing_cert_sha256": release.get("signing_cert_sha256"),
                "verified": release.get("verified"),
            },
        }
        if dry_run:
            return preview

        pool = getattr(self.db, "_pool", None)
        if pool is None:
            raise AndroidReleaseError("android_release_transaction_unavailable", status=503)
        async with pool.acquire() as connection:
            async with connection.transaction():
                channel = await connection.fetchrow(
                    "SELECT * FROM android_release_channels WHERE channel = $1 FOR UPDATE",
                    str(release.get("channel")),
                )
                locked_release = await connection.fetchrow(
                    """
                    SELECT r.release_id, r.channel, r.status, r.version_code,
                           r.required_at_publish,
                           a.artifact_id, a.artifact_kind,
                           a.package_name AS artifact_package_name,
                           a.signing_cert_sha256, a.verified, a.verifier_result
                    FROM android_releases r
                    LEFT JOIN LATERAL (
                        SELECT * FROM android_release_artifacts aa
                        WHERE aa.release_id = r.release_id
                        ORDER BY (aa.artifact_kind = 'apk') DESC, aa.created_at DESC
                        LIMIT 1
                    ) a ON TRUE
                    WHERE r.release_id = $1
                    FOR UPDATE OF r
                    """,
                    str(release_id),
                )
                if not channel or not locked_release:
                    raise AndroidReleaseError("android_release_not_found", status=404)
                # The pre-read only powers the confirmation preview. Security
                # decisions must be repeated against the transaction-locked
                # row so a concurrent retire/artifact change cannot be
                # overwritten by this publish.
                self._validate_publishable(
                    locked_release,
                    expected_version_code=expected_version_code,
                )
                if str(locked_release.get("channel")) == "rustore" and not store_release_confirmed:
                    raise AndroidReleaseError("android_rustore_release_not_confirmed_live", status=409)
                locked_latest = int(channel.get("latest_version_code") or 0)
                if str(locked_release.get("status")) != "published" and version_code <= locked_latest:
                    raise AndroidReleaseError("android_release_version_not_newer", status=409)
                locked_floor = int(channel.get("min_supported_version_code") or 0)
                locked_effective_required = bool(required or locked_release.get("required_at_publish"))
                locked_floor_after = max(
                    locked_floor,
                    version_code if locked_effective_required else locked_floor,
                )
                preview.update(
                    required_at_publish=locked_effective_required,
                    latest_version_code_before=locked_latest,
                    latest_version_code_after=max(locked_latest, version_code),
                    min_supported_version_code_before=locked_floor,
                    min_supported_version_code_after=locked_floor_after,
                )
                await connection.execute(
                    """
                    UPDATE android_releases
                    SET status = 'superseded', updated_at = NOW()
                    WHERE channel = $1 AND status = 'published' AND release_id <> $2
                    """,
                    str(release.get("channel")),
                    str(release_id),
                )
                await connection.execute(
                    """
                    UPDATE android_releases
                    SET status = 'published', required_at_publish = $2,
                        required_floor_after_publish = $3, published_by = $4,
                        published_at = COALESCE(published_at, NOW()), updated_at = NOW()
                    WHERE release_id = $1
                    """,
                    str(release_id),
                    locked_effective_required,
                    locked_floor_after,
                    int(admin_user_id),
                )
                await connection.execute(
                    """
                    UPDATE android_release_channels
                    SET latest_release_id = $2, latest_version_code = $3,
                        min_supported_version_code = $4, updated_at = NOW()
                    WHERE channel = $1
                    """,
                    str(release.get("channel")),
                    str(release_id),
                    version_code,
                    locked_floor_after,
                )
        return {**preview, "dry_run": False, "release": await self.read_release(release_id)}

    async def retire_release(
        self,
        *,
        release_id: str,
        expected_version_code: int,
        admin_user_id: int,
        reason: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._require_enabled()
        release = await self.read_release(release_id)
        if int(release.get("version_code") or 0) != self._version_code(expected_version_code):
            raise AndroidReleaseError("android_release_version_confirmation_mismatch", status=409)
        reason_text = str(reason or "").strip()
        if not reason_text:
            raise AndroidReleaseError("android_release_retire_reason_required")
        channel = str(release.get("channel"))
        floor = int(release.get("min_supported_version_code") or 0)
        is_current = str(release.get("latest_release_id") or "") == str(release_id)
        fallback = None
        if is_current:
            row = await self.db.fetchrow(
                """
                SELECT release_id, version_code, version_name, status
                FROM android_releases
                WHERE channel = $1 AND release_id <> $2
                  AND status IN ('published', 'superseded')
                  AND version_code >= $3
                ORDER BY version_code DESC
                LIMIT 1
                """,
                channel,
                str(release_id),
                floor,
            )
            fallback = _row_dict(row)
            if not fallback:
                raise AndroidReleaseError("android_release_retire_would_break_required_floor", status=409)
        preview = {
            "dry_run": bool(dry_run),
            "release_id": str(release_id),
            "version_code": int(release.get("version_code") or 0),
            "was_current": is_current,
            "fallback": fallback,
            "min_supported_version_code": floor,
            "required_floor_unchanged": True,
            "reason": reason_text,
        }
        if dry_run:
            return preview
        if str(release.get("status")) == "retired":
            return {**preview, "dry_run": False, "already_retired": True}

        pool = getattr(self.db, "_pool", None)
        if pool is None:
            raise AndroidReleaseError("android_release_transaction_unavailable", status=503)
        async with pool.acquire() as connection:
            async with connection.transaction():
                locked_channel = await connection.fetchrow(
                    "SELECT * FROM android_release_channels WHERE channel = $1 FOR UPDATE",
                    channel,
                )
                if not locked_channel:
                    raise AndroidReleaseError("android_release_channel_not_found", status=404)
                locked_current = str(locked_channel.get("latest_release_id") or "") == str(release_id)
                locked_fallback = None
                if locked_current:
                    locked_fallback = await connection.fetchrow(
                        """
                        SELECT release_id, version_code FROM android_releases
                        WHERE channel = $1 AND release_id <> $2
                          AND status IN ('published', 'superseded')
                          AND version_code >= $3
                        ORDER BY version_code DESC
                        LIMIT 1
                        FOR UPDATE
                        """,
                        channel,
                        str(release_id),
                        int(locked_channel.get("min_supported_version_code") or 0),
                    )
                    if not locked_fallback:
                        raise AndroidReleaseError("android_release_retire_would_break_required_floor", status=409)
                    await connection.execute(
                        "UPDATE android_releases SET status = 'published', updated_at = NOW() WHERE release_id = $1",
                        str(locked_fallback["release_id"]),
                    )
                    await connection.execute(
                        """
                        UPDATE android_release_channels
                        SET latest_release_id = $2, latest_version_code = $3, updated_at = NOW()
                        WHERE channel = $1
                        """,
                        channel,
                        str(locked_fallback["release_id"]),
                        int(locked_fallback["version_code"]),
                    )
                await connection.execute(
                    """
                    UPDATE android_releases
                    SET status = 'retired', retired_by = $2, retired_at = NOW(),
                        retire_reason = $3, updated_at = NOW()
                    WHERE release_id = $1
                    """,
                    str(release_id),
                    int(admin_user_id),
                    reason_text,
                )
        return {**preview, "dry_run": False, "retired": True}

    async def build_manifest(
        self,
        *,
        current_version_code: int,
        current_version_name: str,
        channel: str = "direct",
        platform: str = "android",
    ) -> dict[str, Any] | None:
        self._require_enabled()
        normalized_channel = self._channel(channel)
        row = await self.db.fetchrow(
            """
            SELECT c.channel, c.package_name, c.latest_release_id,
                   c.latest_version_code, c.min_supported_version_code,
                   r.version_name AS latest_version_name, r.release_notes,
                   r.published_at, a.artifact_id, a.artifact_kind,
                   a.original_filename, a.size_bytes, a.sha256,
                   a.signing_cert_sha256, a.verified
            FROM android_release_channels c
            LEFT JOIN android_releases r ON r.release_id = c.latest_release_id
            LEFT JOIN LATERAL (
                SELECT * FROM android_release_artifacts aa
                WHERE aa.release_id = r.release_id AND aa.artifact_kind = 'apk'
                ORDER BY aa.created_at DESC LIMIT 1
            ) a ON TRUE
            WHERE c.channel = $1
            """,
            normalized_channel,
        )
        data = _row_dict(row)
        if not data or not data.get("latest_release_id"):
            return None
        current_code = max(0, int(current_version_code or 0))
        latest_code = int(data.get("latest_version_code") or 0)
        floor = int(data.get("min_supported_version_code") or 0)
        update_available = current_code < latest_code
        required = str(platform).lower() == "android" and current_code < floor
        download_path = None
        if (
            normalized_channel == "direct"
            and data.get("artifact_id")
            and data.get("artifact_kind") == "apk"
            and bool(data.get("verified"))
        ):
            download_path = f"/api/mobile/android/releases/{data['latest_release_id']}/apk"
        absolute_download = download_path
        if download_path and self.config.public_base_url:
            absolute_download = f"{self.config.public_base_url.rstrip('/')}{download_path}"
        message = (
            "Это обязательное обновление ExtraArena. Установи новую версию, чтобы продолжить игру."
            if required
            else "Доступно обновление ExtraArena с новыми возможностями."
        )
        return {
            "schema_version": 1,
            "platform": str(platform).lower(),
            "channel": normalized_channel,
            "package_name": data.get("package_name") or self.config.package_name,
            "current_version_code": current_code,
            "current_version_name": str(current_version_name or ""),
            "latest_version_code": latest_code,
            "latest_version_name": data.get("latest_version_name") or "",
            "min_supported_version_code": floor,
            "update_available": update_available,
            "required": required,
            "update_required": required,
            "release_id": data.get("latest_release_id"),
            "release_notes": data.get("release_notes") or "",
            "published_at": data.get("published_at"),
            "update_url": absolute_download,
            "apk_url": absolute_download,
            "download_path": download_path,
            "artifact": (
                {
                    "kind": data.get("artifact_kind"),
                    "filename": data.get("original_filename"),
                    "size_bytes": data.get("size_bytes"),
                    "sha256": data.get("sha256"),
                    "signing_cert_sha256": data.get("signing_cert_sha256"),
                    "verified": bool(data.get("verified")),
                    "download_path": download_path,
                }
                if data.get("artifact_id")
                else None
            ),
            "message": message,
        }

    async def resolve_download(self, release_id: str) -> dict[str, Any]:
        self._require_enabled()
        row = await self.db.fetchrow(
            """
            SELECT r.release_id, r.version_code, r.version_name, r.status,
                   a.original_filename, a.storage_key, a.size_bytes, a.sha256,
                   a.signing_cert_sha256, a.verified
            FROM android_releases r
            JOIN android_release_artifacts a ON a.release_id = r.release_id
            WHERE r.release_id = $1 AND r.channel = 'direct'
              AND r.status IN ('published', 'superseded')
              AND a.artifact_kind = 'apk' AND a.verified = TRUE
            ORDER BY a.created_at DESC
            LIMIT 1
            """,
            str(release_id),
        )
        data = _row_dict(row)
        if not data:
            raise AndroidReleaseError("android_release_download_not_found", status=404)
        path = self._path_for_key(str(data.get("storage_key") or ""))
        if not path.is_file() or path.stat().st_size != int(data.get("size_bytes") or -1):
            raise AndroidReleaseError("android_release_artifact_unavailable", status=503)
        return {**data, "path": path}
