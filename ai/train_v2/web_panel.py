from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, unquote

from ai.train_v2.run_index import build_run_index
from ai.train_v2.profile_registry import build_profile_registry
from ai.train_v2.report import load_run_report


@dataclass
class WebPanelConfig:
    runs_dir: str = "ai/train_v2/runs"
    releases_dir: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765


def _resolve_roots(config: WebPanelConfig) -> list[Path]:
    roots: list[Path] = [Path(config.runs_dir).resolve()]
    if config.releases_dir:
        roots.append(Path(config.releases_dir).resolve())
    return roots


def _is_path_allowed(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def read_safe_text_file(
    path: str,
    *,
    roots: list[str],
    max_bytes: int = 2_000_000,
) -> tuple[str, str]:
    p = Path(path)
    resolved_roots = [Path(r).resolve() for r in roots]
    if not _is_path_allowed(p, resolved_roots):
        raise PermissionError(f"Access denied: {path}")
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes > {max_bytes}")
    suffix = p.suffix.lower()
    if suffix not in (".md", ".json", ".txt", ".log", ".js", ".css", ".html"):
        raise ValueError(f"Disallowed file type: {suffix}")
    content = p.read_text(encoding="utf-8")
    if suffix == ".json":
        content_type = "application/json"
    elif suffix == ".css":
        content_type = "text/css"
    elif suffix == ".js":
        content_type = "application/javascript"
    elif suffix == ".html":
        content_type = "text/html"
    else:
        content_type = "text/plain; charset=utf-8"
    return content, content_type


def _latest_by_mtime(candidates: list[dict], key: str = "path") -> dict | None:
    if not candidates:
        return None
    best = None
    best_mtime = -1
    for c in candidates:
        try:
            mtime = Path(c[key]).stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best = c
        except Exception:
            continue
    return best


def collect_panel_data(
    *,
    runs_dir: str,
    releases_dir: str | None = None,
) -> dict:
    runs_dir_p = Path(runs_dir)
    data: dict[str, Any] = {
        "version": "train_v2_web_panel_v1",
        "runs_dir": str(runs_dir_p.resolve()) if runs_dir_p.exists() else runs_dir,
        "releases_dir": None,
        "run_index": {"root": runs_dir, "runs": 0, "rows": []},
        "profile_registry": {"version": "train_v2_profile_registry_v1", "profiles": 0, "ok": 0, "errors": 0, "rows": [], "best": None},
        "leaderboard": None,
        "release_bundles": [],
        "acceptance_reports": [],
        "shadow_packs": [],
        "lifecycle": {},
        "artifacts": [],
    }

    if releases_dir:
        releases_dir_p = Path(releases_dir)
        if releases_dir_p.exists():
            data["releases_dir"] = str(releases_dir_p.resolve())

    # Run index
    try:
        if runs_dir_p.exists():
            data["run_index"] = build_run_index(runs_dir)
    except Exception as exc:
        data["run_index_error"] = str(exc)

    # Profile registry
    try:
        if runs_dir_p.exists():
            data["profile_registry"] = build_profile_registry([runs_dir])
    except Exception as exc:
        data["profile_registry_error"] = str(exc)

    # Leaderboard
    lb_path = runs_dir_p / "leaderboard.json"
    if lb_path.is_file():
        try:
            data["leaderboard"] = json.loads(lb_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Release bundles
    if releases_dir:
        releases_dir_p = Path(releases_dir)
        if releases_dir_p.exists():
            for manifest_path in releases_dir_p.rglob("release_manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bundle_dir = manifest_path.parent
                    archive_candidates = list(bundle_dir.parent.glob(bundle_dir.name + ".tar.gz"))
                    data["release_bundles"].append({
                        "model_name": manifest.get("model_name", bundle_dir.name),
                        "bundle_dir": str(bundle_dir.resolve()),
                        "created_at": manifest.get("created_at"),
                        "missing": manifest.get("missing", []),
                        "files_count": len(manifest.get("files", [])),
                        "archive_exists": len(archive_candidates) > 0,
                    })
                except Exception as exc:
                    data["release_bundles"].append({
                        "model_name": manifest_path.parent.name,
                        "bundle_dir": str(manifest_path.parent.resolve()),
                        "error": str(exc),
                    })

    # Acceptance reports
    acceptance_roots = [runs_dir_p]
    if releases_dir:
        releases_dir_p = Path(releases_dir)
        if releases_dir_p.exists():
            acceptance_roots.append(releases_dir_p)

    for root in acceptance_roots:
        if not root.exists():
            continue
        for ag_path in root.rglob("acceptance_gate.json"):
            try:
                ag = json.loads(ag_path.read_text(encoding="utf-8"))
                data["acceptance_reports"].append({
                    "status": ag.get("status", "unknown"),
                    "score": ag.get("score"),
                    "path": str(ag_path.resolve()),
                    "dir": str(ag_path.parent.resolve()),
                })
            except Exception:
                pass

    # Shadow packs
    for root in acceptance_roots:
        if not root.exists():
            continue
        for manifest_path in root.rglob("shadow_evidence/*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                summary = manifest.get("summary", {})
                data["shadow_packs"].append({
                    "steps": summary.get("steps"),
                    "match_rate": summary.get("match_rate"),
                    "overlay_latency_ms_p95": summary.get("overlay_latency_ms_p95"),
                    "path": str(manifest_path.resolve()),
                    "dir": str(manifest_path.parent.resolve()),
                })
            except Exception:
                pass

    # Lifecycle summary
    ri = data["run_index"]
    pr = data["profile_registry"]
    rb = data["release_bundles"]
    ag = data["acceptance_reports"]
    sp = data["shadow_packs"]

    acceptance_pass = sum(1 for a in ag if a.get("status") == "pass")
    acceptance_warn = sum(1 for a in ag if a.get("status") == "warn")
    acceptance_fail = sum(1 for a in ag if a.get("status") == "fail")

    data["lifecycle"] = {
        "runs": ri.get("runs", 0),
        "profiles_ok": pr.get("ok", 0),
        "profiles_errors": pr.get("errors", 0),
        "release_bundles": len(rb),
        "acceptance_pass": acceptance_pass,
        "acceptance_warn": acceptance_warn,
        "acceptance_fail": acceptance_fail,
        "shadow_packs": len(sp),
        "latest_release": _latest_by_mtime(rb, key="bundle_dir"),
        "latest_acceptance": _latest_by_mtime(ag, key="path"),
        "best_profile": pr.get("best"),
    }

    # Artifacts index
    artifacts: list[dict] = []
    for b in rb:
        artifacts.append({
            "kind": "release_bundle",
            "name": b.get("model_name", "unknown"),
            "path": b.get("bundle_dir"),
            "display_path": b.get("bundle_dir"),
            "status": "ok" if not b.get("error") else "error",
            "score": None,
            "created_at": b.get("created_at"),
        })
    for a in ag:
        artifacts.append({
            "kind": "acceptance_gate",
            "name": a.get("status", "unknown"),
            "path": a.get("path"),
            "display_path": a.get("dir"),
            "status": a.get("status", "unknown"),
            "score": a.get("score"),
            "created_at": None,
        })
    for s in sp:
        artifacts.append({
            "kind": "shadow_pack",
            "name": f"steps={s.get('steps')}",
            "path": s.get("path"),
            "display_path": s.get("dir"),
            "status": "ok",
            "score": s.get("match_rate"),
            "created_at": None,
        })
    for row in pr.get("rows", []):
        artifacts.append({
            "kind": "profile",
            "name": row.get("model_name", "unknown"),
            "path": row.get("profile_path"),
            "display_path": row.get("profile_path"),
            "status": row.get("status", "unknown"),
            "score": row.get("score"),
            "created_at": None,
        })
    if data["leaderboard"]:
        lb = data["leaderboard"]
        artifacts.append({
            "kind": "leaderboard",
            "name": "leaderboard",
            "path": str(lb_path.resolve()) if lb_path.exists() else None,
            "display_path": str(runs_dir_p / "leaderboard.json"),
            "status": "ok",
            "score": lb.get("best", {}).get("score") if lb.get("best") else None,
            "created_at": None,
        })

    data["artifacts"] = artifacts

    return data


class TrainV2PanelHandler(BaseHTTPRequestHandler):
    panel_config: WebPanelConfig

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, content: str, content_type: str, status: int = 200) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _serve_static(self, name: str, content_type: str) -> None:
        static_dir = Path(__file__).parent / "web_panel_static"
        path = static_dir / name
        if not path.is_file():
            self._send_error(f"Static file not found: {name}", status=404)
            return
        content = path.read_text(encoding="utf-8")
        self._send_text(content, content_type)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self._serve_static("index.html", "text/html")
        elif path == "/app.js":
            self._serve_static("app.js", "application/javascript")
        elif path == "/style.css":
            self._serve_static("style.css", "text/css")
        elif path == "/api/summary":
            try:
                data = collect_panel_data(
                    runs_dir=self.panel_config.runs_dir,
                    releases_dir=self.panel_config.releases_dir,
                )
                self._send_json(data)
            except Exception as exc:
                self._send_error(str(exc), status=500)
        elif path == "/api/snapshot":
            try:
                data = collect_panel_data(
                    runs_dir=self.panel_config.runs_dir,
                    releases_dir=self.panel_config.releases_dir,
                )
                snapshot = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "summary": data,
                }
                self._send_json(snapshot)
            except Exception as exc:
                self._send_error(str(exc), status=500)
        elif path == "/api/run-report":
            dir_param = query.get("dir", [""])[0]
            if not dir_param:
                self._send_error("Missing dir parameter", status=400)
                return
            run_dir = Path(unquote(dir_param))
            roots = _resolve_roots(self.panel_config)
            if not _is_path_allowed(run_dir, roots):
                self._send_error("Access denied", status=403)
                return
            if not run_dir.is_dir():
                self._send_error("Run directory not found", status=404)
                return
            try:
                report = load_run_report(str(run_dir))
                self._send_json(report)
            except Exception as exc:
                self._send_error(str(exc), status=500)
        elif path == "/api/file" or path == "/api/artifact":
            file_param = query.get("path", [""])[0]
            if not file_param:
                self._send_error("Missing path parameter", status=400)
                return
            try:
                roots = [str(r) for r in _resolve_roots(self.panel_config)]
                content, content_type = read_safe_text_file(
                    unquote(file_param),
                    roots=roots,
                )
                if path == "/api/artifact":
                    self._send_json({
                        "path": unquote(file_param),
                        "content_type": content_type,
                        "content": content,
                    })
                else:
                    self._send_text(content, content_type)
            except PermissionError as exc:
                self._send_error(str(exc), status=403)
            except FileNotFoundError as exc:
                self._send_error(str(exc), status=404)
            except ValueError as exc:
                self._send_error(str(exc), status=400)
            except Exception as exc:
                self._send_error(str(exc), status=500)
        else:
            self._send_error("Not found", status=404)


def make_handler(config: WebPanelConfig):
    class Handler(TrainV2PanelHandler):
        panel_config = config
    return Handler


def run_web_panel(config: WebPanelConfig) -> None:
    handler = make_handler(config)
    httpd = HTTPServer((config.host, config.port), handler)
    print(f"TrainV2 web panel: http://{config.host}:{config.port}")
    httpd.serve_forever()


def _main() -> None:
    parser = argparse.ArgumentParser(description="TrainV2 lightweight web panel")
    parser.add_argument("--runs-dir", default="ai/train_v2/runs")
    parser.add_argument("--releases-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = WebPanelConfig(
        runs_dir=args.runs_dir,
        releases_dir=args.releases_dir,
        host=args.host,
        port=args.port,
    )
    run_web_panel(config)


if __name__ == "__main__":
    _main()
