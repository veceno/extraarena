#!/usr/bin/env python3
"""Lightweight localhost dashboard for TrainV3 training runs."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATEST_FILE = ROOT / "TrainV3" / "runs" / "latest_trainv3_training_run.txt"
FALLBACK_LATEST_FILES = (
    ROOT / "TrainV3" / "runs" / "latest_phase26_noassist_easy_gate_run.txt",
    ROOT / "TrainV3" / "runs" / "latest_phase25_clean_noassist_run.txt",
)
METRICS_TAIL_BYTES = 768 * 1024
PROCESS_PATTERNS = (
    "run_phase26_noassist_easy_gate.py",
    "run_phase25_clean_noassist_foundation.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run", type=Path, default=None, help="Pin dashboard to a run directory.")
    parser.add_argument("--latest-file", type=Path, default=DEFAULT_LATEST_FILE)
    parser.add_argument("--refresh", type=int, default=10, help="Browser refresh interval in seconds.")
    args = parser.parse_args(argv)

    refresh_seconds = max(5, int(args.refresh))
    run_dir = None if args.run is None else args.run.expanduser().resolve()
    latest_file = args.latest_file.expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrainV3Dashboard/1.1"

        def log_message(self, fmt: str, *values: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                query = parse_qs(parsed.query)
                local_run_dir = run_dir
                if "run" in query and query["run"]:
                    local_run_dir = Path(query["run"][0]).expanduser().resolve()
                payload = build_status(local_run_dir, latest_file=latest_file)
                self._send_json(payload)
                return
            if parsed.path in {"/", "/index.html"}:
                self._send_html(render_page(refresh_seconds))
                return
            self.send_error(404, "not found")

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"TRAINV3_DASHBOARD {url}", flush=True)
    print(f"TRAINV3_DASHBOARD_LATEST {latest_file}", flush=True)
    if run_dir is not None:
        print(f"TRAINV3_DASHBOARD_RUN {run_dir}", flush=True)
    try:
        httpd.serve_forever(poll_interval=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def build_status(run_dir: Path | None, *, latest_file: Path = DEFAULT_LATEST_FILE) -> dict[str, Any]:
    resolved_run = resolve_run_dir(run_dir, latest_file=latest_file)
    now = time.time()
    status: dict[str, Any] = {
        "generated_at": iso_time(now),
        "latest_file": str(latest_file),
        "run_dir": None if resolved_run is None else str(resolved_run),
        "exists": bool(resolved_run and resolved_run.exists()),
    }
    if resolved_run is None or not resolved_run.exists():
        status["error"] = "run directory not found"
        status["process"] = find_training_process()
        return status

    config = read_first_json_file(
        resolved_run,
        ("run_config.json", "phase26_config.json", "phase25_config.json", "phase2_config.json", "phase1_config.json"),
    )
    metrics = read_metrics_tail(resolved_run / "metrics.jsonl")
    data_rows = [row for row in metrics if row.get("type") != "summary"]
    summary_rows = [row for row in metrics if row.get("type") == "summary"]
    latest_metric = data_rows[-1] if data_rows else {}
    target_updates = int(config.get("updates") or latest_metric.get("updates") or 0)
    current_update = int(latest_metric.get("update") or 0)
    curriculum = latest_metric.get("v5_curriculum_metadata") or {}
    resume_source_update = int(curriculum.get("resume_source_update") or 0)
    cumulative_update = resume_source_update + current_update
    cumulative_target_update = resume_source_update + target_updates
    tail_rows = data_rows[-min(50, len(data_rows)) :]
    avg_update_seconds = mean(
        float(row.get("collect_seconds", 0.0)) + float(row.get("update_seconds", 0.0))
        for row in tail_rows
    )
    eta_seconds = None
    if target_updates > 0 and current_update > 0 and avg_update_seconds > 0:
        eta_seconds = max(0.0, (target_updates - current_update) * avg_update_seconds)

    checkpoint = latest_checkpoint(resolved_run / "checkpoints")
    resume_checkpoint = checkpoint_status(config.get("resume_checkpoint"))
    metrics_path = resolved_run / "metrics.jsonl"
    metrics_age_seconds = None
    if metrics_path.exists():
        metrics_age_seconds = max(0.0, now - metrics_path.stat().st_mtime)

    status.update(
        {
            "run_name": config.get("run_name"),
            "run_id": config.get("run_id"),
            "phase": config.get("phase") or latest_metric.get("phase"),
            "model_name": "extra-lr-v5-adaptive",
            "target_updates": target_updates,
            "current_update": current_update,
            "resume_source_update": resume_source_update,
            "cumulative_update": cumulative_update,
            "cumulative_target_update": cumulative_target_update,
            "progress_fraction": None if target_updates <= 0 else current_update / target_updates,
            "cumulative_progress_fraction": (
                None if cumulative_target_update <= 0 else cumulative_update / cumulative_target_update
            ),
            "remaining_updates": max(0, target_updates - current_update),
            "avg_update_seconds_tail": avg_update_seconds,
            "eta_seconds": eta_seconds,
            "eta_human": format_duration(eta_seconds),
            "metrics_age_seconds": metrics_age_seconds,
            "last_metric": summarize_metric(latest_metric),
            "tail": summarize_tail(tail_rows),
            "checkpoint": checkpoint,
            "resume_checkpoint": resume_checkpoint,
            "config": {
                "env_count": config.get("env_count"),
                "steps_per_update": config.get("steps_per_update"),
                "minibatch_size": config.get("minibatch_size"),
                "checkpoint_every": config.get("checkpoint_every"),
                "ppo_minibatch_plan": config.get("ppo_minibatch_plan"),
                "policy_padding_mode": config.get("policy_padding_mode"),
                "resume_checkpoint": config.get("resume_checkpoint"),
                "trace_manifest_reused": config.get("trace_manifest_reused"),
                "clean_room_noassist": config.get("clean_room_noassist"),
                "contaminated_prior_data_excluded": config.get("contaminated_prior_data_excluded"),
                "v4_1_included": config.get("v4_1_included"),
                "runtime_opponent_mode": config.get("runtime_opponent_mode"),
                "runtime_opponents": config.get("runtime_opponents"),
                "max_opponent_actions": config.get("max_opponent_actions"),
                "phase": config.get("phase"),
                "config_file": config.get("_config_file"),
            },
            "summary": summary_rows[-1] if summary_rows else None,
            "process": find_training_process(),
        }
    )
    return status


def resolve_run_dir(run_dir: Path | None, *, latest_file: Path) -> Path | None:
    if run_dir is not None:
        return run_dir
    candidate = next((path for path in (latest_file, *FALLBACK_LATEST_FILES) if path.exists()), None)
    if candidate is None:
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid json: {exc}"}


def read_first_json_file(run_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        path = run_dir / name
        if path.exists():
            data = read_json_file(path)
            data["_config_file"] = name
            return data
    return {}


def read_metrics_tail(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > METRICS_TAIL_BYTES:
            handle.seek(size - METRICS_TAIL_BYTES)
            handle.readline()
        raw = handle.read().decode("utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_metric(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "update",
        "total_env_transitions",
        "collect_seconds",
        "update_seconds",
        "prepare_seconds",
        "approx_kl",
        "entropy",
        "clip_fraction",
        "loss",
        "planned_padding_expansion_ratio",
        "planned_padded_total_bytes",
        "minibatch_plan_kind",
        "v5_mode",
        "assist_mode",
        "trace_manifest_id",
        "phase",
        "runtime_opponent_mode",
        "runtime_opponent_actions_per_transition",
        "runtime_opponent_lane_metrics",
        "terminal_rate",
        "reset_rate",
        "mean_reward",
        "mean_learner_action_reward",
        "mean_opponent_response_reward",
        "mean_legal_actions",
    )
    return {key: row.get(key) for key in keys if key in row}


def summarize_tail(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "avg_collect_seconds": mean(float(row.get("collect_seconds", 0.0)) for row in rows),
        "avg_update_seconds": mean(float(row.get("update_seconds", 0.0)) for row in rows),
        "avg_approx_kl_abs": mean(abs(float(row.get("approx_kl", 0.0))) for row in rows),
        "avg_entropy": mean(float(row.get("entropy", 0.0)) for row in rows),
        "avg_padding_expansion": mean(float(row.get("planned_padding_expansion_ratio", 0.0)) for row in rows),
        "avg_terminal_rate": mean(float(row.get("terminal_rate", 0.0)) for row in rows),
        "avg_opponent_actions_per_transition": mean(
            float(row.get("runtime_opponent_actions_per_transition", 0.0)) for row in rows
        ),
        "avg_mean_reward": mean(float(row.get("mean_reward", 0.0)) for row in rows),
    }


def latest_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("trainv3_rust_legal_update_*.npz"))
    if not candidates:
        return None
    path = candidates[-1]
    update = parse_checkpoint_update(path.name)
    return {
        "path": str(path),
        "update": update,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "modified_at": iso_time(path.stat().st_mtime),
    }


def checkpoint_status(value: Any) -> dict[str, Any] | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    payload: dict[str, Any] = {
        "path": str(path),
        "update": parse_checkpoint_update(path.name),
        "exists": path.exists(),
    }
    if path.exists():
        payload["size_mb"] = path.stat().st_size / (1024 * 1024)
        payload["modified_at"] = iso_time(path.stat().st_mtime)
    return payload


def parse_checkpoint_update(name: str) -> int | None:
    match = re.search(r"update_(\d+)\.npz$", name)
    return None if match is None else int(match.group(1))


def find_training_process() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid,ppid,%cpu,%mem,rss,etime,command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines()[1:]:
        if not any(pattern in line for pattern in PROCESS_PATTERNS):
            continue
        if "phase25_training_dashboard.py" in line:
            continue
        if "tmux new-session" in line or "zsh -c" in line or "tee TrainV3/runs" in line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, cpu, mem, rss, elapsed, command = parts
        return {
            "pid": int(pid),
            "ppid": int(ppid),
            "cpu_percent": float(cpu),
            "mem_percent": float(mem),
            "rss_mb": int(rss) / 1024,
            "elapsed": elapsed,
            "command": command,
        }
    return None


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def iso_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp))


def render_page(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TrainV3 Training Dashboard</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101214;
      color: #e8ecef;
    }}
    body {{ margin: 0; padding: 24px; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }}
    .muted {{ color: #9aa5ad; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #30363d; border-radius: 8px; padding: 14px; background: #15191d; }}
    .label {{ font-size: 12px; color: #9aa5ad; text-transform: uppercase; }}
    .value {{ font-size: 24px; margin-top: 4px; overflow-wrap: anywhere; }}
    .bar {{ height: 12px; background: #272c31; border-radius: 999px; overflow: hidden; }}
    .fill {{ height: 100%; width: 0%; background: #4da3ff; transition: width 200ms linear; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #30363d; vertical-align: top; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #15191d; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }}
    .ok {{ color: #75d99d; }}
    .warn {{ color: #ffd479; }}
    .lane-table td:nth-child(n+2), .lane-table th:nth-child(n+2) {{ text-align: right; }}
  </style>
</head>
<body>
<main>
  <h1>TrainV3 Training Dashboard</h1>
  <div class="muted" id="subtitle">Loading...</div>
  <div class="grid" id="cards"></div>
  <div class="card">
    <div class="label">Progress</div>
    <div class="bar"><div class="fill" id="progress-fill"></div></div>
  </div>
  <div class="card">
    <div class="label">Latest Metric</div>
    <table id="metric-table"></table>
  </div>
  <div class="card">
    <div class="label">Runtime Opponent Lanes</div>
    <table id="lane-table" class="lane-table"></table>
  </div>
  <div class="card">
    <div class="label">Run Config</div>
    <table id="config-table"></table>
  </div>
  <div class="card">
    <div class="label">Raw Status</div>
    <pre id="raw"></pre>
  </div>
</main>
<script>
const refreshSeconds = {refresh_seconds};
function fmt(n, digits = 2) {{
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
  return Number(n).toFixed(digits);
}}
function pct(x) {{
  if (x === null || x === undefined) return "-";
  return (Number(x) * 100).toFixed(1) + "%";
}}
function addCard(label, value, cls = "") {{
  return `<div class="card"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`;
}}
function tableFrom(obj) {{
  return Object.entries(obj || {{}}).map(([k, v]) => `<tr><th>${{k}}</th><td><code>${{escapeHtml(formatValue(v))}}</code></td></tr>`).join("");
}}
function laneTableFrom(obj) {{
  const rows = Object.entries(obj || {{}}).sort((a, b) => a[0].localeCompare(b[0]));
  if (!rows.length) return `<tr><td class="muted">No per-lane telemetry yet</td></tr>`;
  const header = `<tr><th>lane</th><th>slots</th><th>reward</th><th>learner</th><th>opp response</th><th>opp act/row</th><th>term</th></tr>`;
  const body = rows.map(([lane, v]) => `<tr>
    <td><code>${{escapeHtml(lane)}}</code></td>
    <td>${{v.env_slots ?? "-"}}</td>
    <td>${{fmt(v.mean_reward, 4)}}</td>
    <td>${{fmt(v.mean_learner_action_reward, 4)}}</td>
    <td>${{fmt(v.mean_opponent_response_reward, 4)}}</td>
    <td>${{fmt(v.opponent_actions_per_transition, 2)}}</td>
    <td>${{pct(v.terminal_rate)}}</td>
  </tr>`).join("");
  return header + body;
}}
function formatValue(v) {{
  if (v === null || v === undefined) return "-";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}}
function escapeHtml(s) {{
  return s.replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
async function refresh() {{
  const res = await fetch("/api/status", {{cache: "no-store"}});
  const data = await res.json();
  document.getElementById("subtitle").textContent = `${{data.run_dir || "no run"}} · generated ${{data.generated_at || "-"}}`;
  const process = data.process;
  const running = process ? "running" : "not found";
  document.getElementById("cards").innerHTML = [
    addCard("Run update", `${{data.current_update || 0}} / ${{data.target_updates || 0}}`),
    addCard("Total update", `${{data.cumulative_update || data.current_update || 0}} / ${{data.cumulative_target_update || data.target_updates || 0}}`),
    addCard("Phase", data.phase || "-"),
    addCard("Progress", pct(data.progress_fraction)),
    addCard("ETA", data.eta_human || "-"),
    addCard("Avg update", fmt(data.avg_update_seconds_tail, 1) + "s"),
    addCard("Entropy", fmt(data.last_metric?.entropy, 3)),
    addCard("Abs KL tail", fmt(data.tail?.avg_approx_kl_abs, 4)),
    addCard("Term rate", pct(data.last_metric?.terminal_rate)),
    addCard("Opp act/row", fmt(data.last_metric?.runtime_opponent_actions_per_transition, 2)),
    addCard("Mean reward", fmt(data.last_metric?.mean_reward, 4)),
    addCard("Padding", fmt(data.last_metric?.planned_padding_expansion_ratio, 2) + "x"),
    addCard("Checkpoint", data.checkpoint?.update ?? "-", data.checkpoint ? "ok" : "warn"),
    addCard("Resume from", data.resume_checkpoint?.update ?? "-", data.resume_checkpoint ? "ok" : ""),
    addCard("Process", running, process ? "ok" : "warn"),
    addCard("RSS", process ? fmt(process.rss_mb, 0) + " MB" : "-")
  ].join("");
  document.getElementById("progress-fill").style.width = `${{Math.max(0, Math.min(100, Number(data.progress_fraction || 0) * 100))}}%`;
  document.getElementById("metric-table").innerHTML = tableFrom(data.last_metric);
  document.getElementById("lane-table").innerHTML = laneTableFrom(data.last_metric?.runtime_opponent_lane_metrics);
  document.getElementById("config-table").innerHTML = tableFrom(data.config);
  document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
}}
refresh();
setInterval(refresh, refreshSeconds * 1000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
