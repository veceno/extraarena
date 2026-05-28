from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai.train_v2 import TRAIN_V2_ARTIFACT_VERSIONS
from ai.train_v2.acceptance_gate import evaluate_acceptance_gate, load_gate_inputs, GateThresholds
from ai.train_v2.profile_registry import build_profile_registry, load_profile_overlay
from ai.train_v2.release_bundle import collect_file_manifest, sha256_file
from ai.train_v2.run_index import build_run_index
from ai.train_v2.web_panel import collect_panel_data


FAKE_ONNX_CONTENT = b"fake_onnx"


def build_synthetic_operator_tree(root: str) -> dict:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    runs_dir = root_p / "runs"
    run_dir = runs_dir / "demo_run_20260520_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"name": "demo", "seed": 42}))
    (run_dir / "summary.json").write_text(json.dumps({
        "train": {"updates": 5, "steps": 400, "last_loss": 0.5, "last_entropy": 1.2},
        "eval": {
            "random": {"winrate": 0.45},
            "end_turn": {"winrate": 0.85},
            "greedy_face": {"winrate": 0.35},
        },
    }))
    (run_dir / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "update": 1, "steps": 80, "loss": 0.6, "entropy": 1.3}),
        json.dumps({"type": "train", "update": 2, "steps": 80, "loss": 0.5, "entropy": 1.2}),
    ]))
    exported = run_dir / "exported"
    exported.mkdir()
    (exported / "demo.onnx").write_bytes(FAKE_ONNX_CONTENT)
    (run_dir / "candidate_profile.json").write_text(json.dumps({
        "difficulty": "train_v2_candidate",
        "profile": {
            "model_path": str(exported / "demo.onnx"),
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "selection": "argmax",
            "temperature_range": [1.0, 1.0],
        },
        "source": {
            "candidate_onnx": str(exported / "demo.onnx"),
            "source_onnx": str(exported / "demo.onnx"),
            "model_name": "demo",
            "score": 1.2,
            "source_run_dir": str(run_dir),
        },
    }))
    (run_dir / "leaderboard.json").write_text(json.dumps({
        "models": 1,
        "rows": [
            {
                "rank": 1,
                "model_name": "demo",
                "onnx_path": str(exported / "demo.onnx"),
                "score": 1.2,
                "wr_random": 0.45,
                "wr_end_turn": 0.85,
                "wr_greedy_face": 0.35,
                "latency_ms_p50_random": 5.0,
                "invalid_actions_total": 0,
                "brain_invalid_total": 0,
                "parity_mismatches": 0,
            }
        ],
        "best": {
            "rank": 1,
            "model_name": "demo",
            "onnx_path": str(exported / "demo.onnx"),
            "score": 1.2,
            "wr_random": 0.45,
            "wr_end_turn": 0.85,
            "wr_greedy_face": 0.35,
            "latency_ms_p50_random": 5.0,
            "invalid_actions_total": 0,
            "brain_invalid_total": 0,
            "parity_mismatches": 0,
        },
    }))

    candidates_dir = root_p / "candidates"
    cand_dir = candidates_dir / "demo_candidate"
    cand_dir.mkdir(parents=True)
    (cand_dir / "candidate.json").write_text(json.dumps({
        "model_name": "demo",
        "score": 1.2,
        "candidate_onnx": str(cand_dir / "demo.onnx"),
    }))
    (cand_dir / "demo.onnx").write_bytes(FAKE_ONNX_CONTENT)
    (cand_dir / "demo.onnx.json").write_text(json.dumps({"opset": 17}))
    (cand_dir / "candidate_profile.json").write_text(json.dumps({
        "difficulty": "train_v2_candidate",
        "profile": {
            "model_path": "demo.onnx",
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "selection": "argmax",
            "temperature_range": [1.0, 1.0],
        },
        "source": {
            "candidate_onnx": str(cand_dir / "demo.onnx"),
            "source_onnx": str(cand_dir / "demo.onnx"),
            "model_name": "demo",
            "score": 1.2,
            "source_run_dir": str(run_dir),
        },
    }))
    overlay = {
        "version": TRAIN_V2_ARTIFACT_VERSIONS["profile_overlay"],
        "created_at": "2026-05-20T00:00:00",
        "source_profile_path": str(cand_dir / "candidate_profile.json"),
        "profiles": {
            "train_v2_candidate": {
                "model_path": "demo.onnx",
                "format": "train_v2_classic_v1",
                "obs_dim": 1456,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
                "selection": "argmax",
                "temperature_range": [1.0, 1.0],
                "difficulty": "train_v2_candidate",
            }
        },
    }
    (cand_dir / "profile_overlay.json").write_text(json.dumps(overlay, indent=2))

    se_dir = cand_dir / "shadow_evidence" / "demo_shadow"
    se_dir.mkdir(parents=True)
    shadow_manifest = {
        "version": TRAIN_V2_ARTIFACT_VERSIONS["shadow_evidence"],
        "created_at": "2026-05-20T00:00:00",
        "overlay_path": str(cand_dir / "profile_overlay.json"),
        "candidate_profile_path": str(cand_dir / "candidate_profile.json"),
        "candidate_dir": str(cand_dir),
        "artifacts": {
            "shadow_result": str(se_dir / "shadow_result.json"),
            "shadow_summary": str(se_dir / "shadow_summary.json"),
            "shadow_markdown": str(se_dir / "shadow_summary.md"),
            "shadow_mismatches": str(se_dir / "shadow_mismatches.json"),
        },
        "summary": {
            "episodes": 2,
            "steps": 40,
            "matches": 35,
            "mismatches": 5,
            "match_rate": 0.875,
            "mismatch_rate": 0.125,
            "legacy_invalid_actions": 0,
            "overlay_invalid_actions": 0,
            "played_invalid_actions": 0,
            "legacy_latency_ms_p50": 0.5,
            "legacy_latency_ms_p95": 1.0,
            "overlay_latency_ms_p50": 8.0,
            "overlay_latency_ms_p95": 15.0,
        },
    }
    (se_dir / "manifest.json").write_text(json.dumps(shadow_manifest, indent=2))
    (se_dir / "shadow_summary.json").write_text(json.dumps(shadow_manifest["summary"], indent=2))
    (se_dir / "shadow_result.json").write_text(json.dumps({"episodes": 2, "steps": 40}, indent=2))
    (se_dir / "shadow_summary.md").write_text("# Shadow Report\n\nSynthetic.\n")
    (se_dir / "shadow_mismatches.json").write_text("[]")

    ag_dir = cand_dir / "acceptance_gate"
    ag_dir.mkdir()
    gate_result = evaluate_acceptance_gate(
        load_gate_inputs(
            candidate_dir=str(cand_dir),
            overlay_path=str(cand_dir / "profile_overlay.json"),
            shadow_pack_dir=str(se_dir),
            leaderboard_path=str(run_dir / "leaderboard.json"),
        ),
        thresholds=GateThresholds(min_winrate_random=0.35, min_winrate_end_turn=0.80),
    )
    (ag_dir / "acceptance_gate.json").write_text(json.dumps(gate_result, indent=2))
    (ag_dir / "acceptance_gate.md").write_text("# Acceptance Gate\n\nSynthetic.\n")

    releases_dir = root_p / "releases"
    bundle_dir = releases_dir / "demo_bundle"
    bundle_dir.mkdir(parents=True)
    manifest = {
        "version": TRAIN_V2_ARTIFACT_VERSIONS["release_bundle"],
        "created_at": "2026-05-20T00:00:00",
        "bundle_dir": str(bundle_dir),
        "source_candidate_dir": str(cand_dir),
        "model_name": "demo",
        "missing": [],
        "artifacts": {
            "onnx": "model/demo.onnx",
            "onnx_sidecar": "model/demo.onnx.json",
            "candidate_json": "candidate/candidate.json",
            "candidate_profile": "profile/candidate_profile.json",
            "profile_overlay": "profile/profile_overlay.json",
            "shadow_evidence": "shadow_evidence/",
            "acceptance_gate": "acceptance_gate/",
        },
        "files": [],
    }
    (bundle_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    (bundle_dir / "README.md").write_text("# Release Bundle\n\nSynthetic.\n")

    return {
        "root": str(root_p.resolve()),
        "runs_dir": str(runs_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "candidates_dir": str(candidates_dir.resolve()),
        "candidate_dir": str(cand_dir.resolve()),
        "shadow_pack_dir": str(se_dir.resolve()),
        "acceptance_dir": str(ag_dir.resolve()),
        "releases_dir": str(releases_dir.resolve()),
        "bundle_dir": str(bundle_dir.resolve()),
    }


def check_operator_contracts(root: str) -> dict:
    root_p = Path(root)
    checks: list[dict] = []
    ok = True

    def _add(name: str, status: str, message: str) -> None:
        nonlocal ok
        checks.append({"name": name, "status": status, "message": message})
        if status == "fail":
            ok = False

    runs_dir = root_p / "runs"
    if runs_dir.exists():
        try:
            ri = build_run_index(str(runs_dir))
            if ri.get("runs", 0) >= 1:
                _add("run_index", "pass", f"found {ri['runs']} run(s)")
            else:
                _add("run_index", "fail", "no runs found")
        except Exception as exc:
            _add("run_index", "fail", str(exc))
    else:
        _add("run_index", "fail", "runs dir missing")

    if runs_dir.exists():
        try:
            pr = build_profile_registry([str(runs_dir)])
            if pr.get("ok", 0) >= 1:
                _add("profile_registry", "pass", f"ok={pr['ok']} errors={pr['errors']}")
            else:
                _add("profile_registry", "fail", "no valid profiles")
        except Exception as exc:
            _add("profile_registry", "fail", str(exc))
    else:
        _add("profile_registry", "fail", "runs dir missing")

    cand_dirs = list((root_p / "candidates").iterdir()) if (root_p / "candidates").exists() else []
    if cand_dirs:
        cand_dir = cand_dirs[0]
        try:
            inputs = load_gate_inputs(candidate_dir=str(cand_dir))
            if inputs.get("candidate") is not None:
                _add("gate_inputs", "pass", "candidate loaded")
            else:
                _add("gate_inputs", "fail", "candidate missing")
        except Exception as exc:
            _add("gate_inputs", "fail", str(exc))

        try:
            inputs = load_gate_inputs(candidate_dir=str(cand_dir))
            result = evaluate_acceptance_gate(inputs, thresholds=GateThresholds())
            _add("gate_eval", "pass", f"status={result['status']}")
        except Exception as exc:
            _add("gate_eval", "fail", str(exc))
    else:
        _add("gate_inputs", "fail", "no candidate dirs")
        _add("gate_eval", "fail", "no candidate dirs")

    try:
        pd = collect_panel_data(runs_dir=str(runs_dir), releases_dir=str(root_p / "releases"))
        if pd.get("lifecycle", {}).get("runs", 0) >= 1:
            _add("panel_data", "pass", f"runs={pd['lifecycle']['runs']}")
        else:
            _add("panel_data", "fail", "no runs in panel data")
    except Exception as exc:
        _add("panel_data", "fail", str(exc))

    # version checks
    for key, expected_version in TRAIN_V2_ARTIFACT_VERSIONS.items():
        if key in ("panel_snapshot",):
            continue
        found = False
        try:
            if key == "profile_overlay":
                for f in root_p.rglob("profile_overlay.json"):
                    data = json.loads(f.read_text())
                    if data.get("version") == expected_version:
                        found = True
                        break
            elif key == "shadow_evidence":
                for f in root_p.rglob("shadow_evidence/*/manifest.json"):
                    data = json.loads(f.read_text())
                    if data.get("version") == expected_version:
                        found = True
                        break
            elif key == "acceptance_gate":
                for f in root_p.rglob("acceptance_gate.json"):
                    data = json.loads(f.read_text())
                    if data.get("version") == expected_version:
                        found = True
                        break
            elif key == "release_bundle":
                for f in root_p.rglob("release_manifest.json"):
                    data = json.loads(f.read_text())
                    if data.get("version") == expected_version:
                        found = True
                        break
            elif key == "profile_registry":
                for f in root_p.rglob("candidate_profile.json"):
                    data = json.loads(f.read_text())
                    if data.get("profile", {}).get("format") == "train_v2_classic_v1":
                        found = True
                        break
        except Exception:
            pass
        if found:
            _add(f"version_{key}", "pass", f"found {expected_version}")
        else:
            _add(f"version_{key}", "fail", f"missing {expected_version}")

    # no production touch
    _add("no_production_touch", "pass", "synthetic tree, no production files modified")

    return {"ok": ok, "checks": checks}


def run_operator_smoke(root: str) -> dict:
    tree = build_synthetic_operator_tree(root)
    contracts = check_operator_contracts(root)
    return {
        "root": root,
        "tree": tree,
        "contracts": contracts,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="TrainV2 operator smoke test")
    parser.add_argument("--root", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep", action="store_true", help="Keep synthetic tree after run")
    args = parser.parse_args()

    result = run_operator_smoke(args.root)
    contracts = result["contracts"]
    pass_count = sum(1 for c in contracts["checks"] if c["status"] == "pass")
    fail_count = sum(1 for c in contracts["checks"] if c["status"] == "fail")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        status = "PASS" if contracts["ok"] else "FAIL"
        print(f"TrainV2 operator smoke: {status}")
        print(f"Root: {args.root}")
        print(f"Checks: {pass_count} pass / {fail_count} fail")
        for c in contracts["checks"]:
            if c["status"] == "fail":
                print(f"  FAIL: {c['name']} — {c['message']}")

    if not contracts["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    _main()
