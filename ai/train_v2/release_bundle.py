from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ReleaseBundleConfig:
    candidate_dir: str
    output_dir: str
    name: str | None = None
    overlay_path: str | None = None
    profile_path: str | None = None
    shadow_pack_dir: str | None = None
    acceptance_dir: str | None = None
    include_shadow: bool = True
    include_acceptance: bool = True
    create_archive: bool = False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_file_manifest(root: str) -> list[dict]:
    entries: list[dict] = []
    root_p = Path(root)
    for p in sorted(root_p.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root_p).as_posix()
        if rel == "release_manifest.json":
            continue
        entries.append({
            "path": rel,
            "size": p.stat().st_size,
            "sha256": sha256_file(str(p)),
        })
    return entries


def _load_json_if_exists(p: Path) -> dict | None:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _find_first_onnx(candidate_dir: Path) -> str | None:
    onnx_files = sorted(candidate_dir.rglob("*.onnx"))
    if onnx_files:
        return str(onnx_files[0].resolve())
    return None


def _find_overlay_recursive(candidate_dir: Path) -> str | None:
    candidates: list[tuple[Path, float]] = []
    for p in candidate_dir.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("version") == "train_v2_profile_overlay_v1":
                candidates.append((p, p.stat().st_mtime))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return str(candidates[0][0].resolve())


def _find_shadow_pack(candidate_dir: Path) -> str | None:
    shadow_dir = candidate_dir / "shadow_evidence"
    if not shadow_dir.is_dir():
        return None
    packs = [d for d in shadow_dir.iterdir() if d.is_dir()]
    if not packs:
        return None
    packs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest = packs[0]
    if (latest / "manifest.json").exists():
        return str(latest.resolve())
    return None


def _find_acceptance_dir(candidate_dir: Path) -> str | None:
    candidates: list[tuple[Path, float]] = []
    for p in candidate_dir.rglob("acceptance_gate.json"):
        candidates.append((p.parent, p.stat().st_mtime))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return str(candidates[0][0].resolve())


def discover_release_inputs(config: ReleaseBundleConfig) -> dict:
    cdir = Path(config.candidate_dir)
    missing: list[str] = []

    candidate_json_path = cdir / "candidate.json"
    candidate = _load_json_if_exists(candidate_json_path)
    if candidate is None:
        missing.append("candidate_json")

    onnx_path: str | None = None
    if candidate and candidate.get("candidate_onnx"):
        onnx_path = str(Path(candidate["candidate_onnx"]).resolve())
    if not onnx_path:
        onnx_path = _find_first_onnx(cdir)
    if not onnx_path:
        missing.append("onnx")

    onnx_sidecar: str | None = None
    if onnx_path:
        sidecar = Path(onnx_path).with_suffix(".onnx.json")
        if sidecar.exists():
            onnx_sidecar = str(sidecar.resolve())
        else:
            missing.append("onnx_sidecar")
    else:
        missing.append("onnx_sidecar")

    profile_path: str | None = None
    if config.profile_path:
        p = Path(config.profile_path)
        if p.exists():
            profile_path = str(p.resolve())
        else:
            missing.append("profile")
    else:
        prof = cdir / "candidate_profile.json"
        if prof.exists():
            profile_path = str(prof.resolve())
        else:
            missing.append("profile")

    overlay_path: str | None = None
    if config.overlay_path:
        p = Path(config.overlay_path)
        if p.exists():
            overlay_path = str(p.resolve())
        else:
            missing.append("overlay")
    else:
        overlay_path = _find_overlay_recursive(cdir)
        if not overlay_path:
            missing.append("overlay")

    shadow_pack: str | None = None
    if config.shadow_pack_dir:
        p = Path(config.shadow_pack_dir)
        if p.exists():
            shadow_pack = str(p.resolve())
        else:
            missing.append("shadow_pack")
    elif config.include_shadow:
        shadow_pack = _find_shadow_pack(cdir)
        if not shadow_pack:
            missing.append("shadow_pack")

    acceptance_dir: str | None = None
    if config.acceptance_dir:
        p = Path(config.acceptance_dir)
        if p.exists():
            acceptance_dir = str(p.resolve())
        else:
            missing.append("acceptance_dir")
    elif config.include_acceptance:
        acceptance_dir = _find_acceptance_dir(cdir)
        if not acceptance_dir:
            missing.append("acceptance_dir")

    # deduplicate missing preserving order
    seen = set()
    uniq = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            uniq.append(m)

    return {
        "candidate_dir": str(cdir.resolve()),
        "candidate_json": str(candidate_json_path.resolve()) if candidate_json_path.exists() else None,
        "onnx": onnx_path,
        "onnx_sidecar": onnx_sidecar,
        "profile": profile_path,
        "overlay": overlay_path,
        "shadow_pack": shadow_pack,
        "acceptance_dir": acceptance_dir,
        "missing": uniq,
    }


def _copy_dir_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        if item.is_file():
            shutil.copy2(item, dst / item.name)
        elif item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)


def format_release_readme(
    bundle: dict,
    inputs: dict,
    acceptance: dict | None = None,
) -> str:
    model_name = bundle.get("model_name", "unknown")
    missing = inputs.get("missing", [])
    shadow_included = inputs.get("shadow_pack") is not None
    acceptance_included = inputs.get("acceptance_dir") is not None

    verdict_status = "n/a"
    gate_score = "n/a"
    if acceptance:
        verdict_status = acceptance.get("status", "n/a").upper()
        gate_score = acceptance.get("score", "n/a")

    candidate_score = "n/a"
    if inputs.get("candidate_json"):
        try:
            cand = json.loads(Path(inputs["candidate_json"]).read_text(encoding="utf-8"))
            candidate_score = cand.get("score", "n/a")
        except Exception:
            pass

    shadow_steps = "n/a"
    shadow_match_rate = "n/a"
    overlay_latency = "n/a"
    overlay_invalid = "n/a"
    if shadow_included and inputs.get("shadow_pack"):
        try:
            summary_path = Path(inputs["shadow_pack"]) / "shadow_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                shadow_steps = summary.get("steps", "n/a")
                shadow_match_rate = summary.get("match_rate", "n/a")
                overlay_latency = summary.get("overlay_latency_ms_p95", "n/a")
                overlay_invalid = summary.get("overlay_invalid_actions", "n/a")
        except Exception:
            pass

    lines: list[str] = []
    lines.append("# TrainV2 Release Candidate Bundle")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- Acceptance: {verdict_status}")
    lines.append(f"- Gate score: {gate_score}")
    lines.append(f"- Model: {model_name}")
    lines.append(f"- Candidate score: {candidate_score}")
    lines.append("")
    lines.append("## Contents")
    lines.append(f"- ONNX model: {'included' if inputs.get('onnx') else 'missing'}")
    lines.append(f"- Candidate profile: {'included' if inputs.get('profile') else 'missing'}")
    lines.append(f"- Overlay: {'included' if inputs.get('overlay') else 'missing'}")
    lines.append(f"- Shadow evidence: {'included' if shadow_included else 'missing'}")
    lines.append(f"- Acceptance gate: {'included' if acceptance_included else 'missing'}")
    lines.append("")
    lines.append("## Shadow Summary")
    lines.append(f"- Steps: {shadow_steps}")
    lines.append(f"- Match rate: {shadow_match_rate}")
    lines.append(f"- Overlay latency p95: {overlay_latency}")
    lines.append(f"- Overlay invalid actions: {overlay_invalid}")
    lines.append("")
    lines.append("## Safety")
    lines.append("This bundle is an opt-in artifact. It does not modify production configs.")
    lines.append("")
    lines.append("## Manual Use")
    lines.append("Review `profile/profile_overlay.json` and validate before manual connection.")
    lines.append("")

    if missing:
        lines.append("## Missing Artifacts")
        for m in missing:
            lines.append(f"- {m}")
        lines.append("")

    return "\n".join(lines)


def build_release_bundle(config: ReleaseBundleConfig) -> dict:
    inputs = discover_release_inputs(config)
    hard_missing = [m for m in inputs["missing"] if m in ("candidate_json", "onnx")]
    if hard_missing:
        raise FileNotFoundError(f"Missing required artifacts: {', '.join(hard_missing)}")

    candidate = json.loads(Path(inputs["candidate_json"]).read_text(encoding="utf-8"))
    model_name = candidate.get("model_name") or Path(inputs["onnx"]).stem

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = config.name or f"{model_name}_{timestamp}"

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle_dir = out / base_name
    suffix = 0
    while bundle_dir.exists():
        suffix += 1
        bundle_dir = out / f"{base_name}_{suffix}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # model/
    model_dir = bundle_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["onnx"], model_dir / Path(inputs["onnx"]).name)
    if inputs.get("onnx_sidecar"):
        shutil.copy2(inputs["onnx_sidecar"], model_dir / Path(inputs["onnx_sidecar"]).name)

    # candidate/
    cand_dir = bundle_dir / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["candidate_json"], cand_dir / "candidate.json")

    # profile/
    prof_dir = bundle_dir / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)
    if inputs.get("profile"):
        shutil.copy2(inputs["profile"], prof_dir / "candidate_profile.json")
    if inputs.get("overlay"):
        shutil.copy2(inputs["overlay"], prof_dir / "profile_overlay.json")

    # shadow_evidence/
    if config.include_shadow and inputs.get("shadow_pack"):
        shadow_dir = bundle_dir / "shadow_evidence"
        _copy_dir_contents(Path(inputs["shadow_pack"]), shadow_dir)

    # acceptance_gate/
    if config.include_acceptance and inputs.get("acceptance_dir"):
        acc_dir = bundle_dir / "acceptance_gate"
        _copy_dir_contents(Path(inputs["acceptance_dir"]), acc_dir)

    # acceptance for README
    acceptance = None
    if config.include_acceptance and inputs.get("acceptance_dir"):
        ag_path = Path(inputs["acceptance_dir"]) / "acceptance_gate.json"
        if ag_path.exists():
            acceptance = json.loads(ag_path.read_text(encoding="utf-8"))

    readme = format_release_readme(
        {"model_name": model_name},
        inputs,
        acceptance=acceptance,
    )
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    manifest = {
        "version": "train_v2_release_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir.resolve()),
        "source_candidate_dir": inputs["candidate_dir"],
        "model_name": model_name,
        "missing": inputs["missing"],
        "artifacts": {
            "onnx": f"model/{Path(inputs['onnx']).name}",
            "onnx_sidecar": f"model/{Path(inputs['onnx_sidecar']).name}" if inputs.get("onnx_sidecar") else None,
            "candidate_json": "candidate/candidate.json",
            "candidate_profile": "profile/candidate_profile.json" if inputs.get("profile") else None,
            "profile_overlay": "profile/profile_overlay.json" if inputs.get("overlay") else None,
            "shadow_evidence": "shadow_evidence/" if (config.include_shadow and inputs.get("shadow_pack")) else None,
            "acceptance_gate": "acceptance_gate/" if (config.include_acceptance and inputs.get("acceptance_dir")) else None,
        },
        "files": collect_file_manifest(str(bundle_dir)),
    }

    manifest_path = bundle_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    archive_path: str | None = None
    if config.create_archive:
        archive_base = bundle_dir.parent / bundle_dir.name
        # shutil.make_archive creates archive at archive_base with format gztar
        archive_file = shutil.make_archive(
            base_name=str(archive_base),
            format="gztar",
            root_dir=str(bundle_dir.parent),
            base_dir=bundle_dir.name,
        )
        archive_path = str(Path(archive_file).resolve())

    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "readme_path": str(readme_path.resolve()),
        "archive_path": archive_path,
        "status": "created",
        "missing": inputs["missing"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build TrainV2 release candidate bundle")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--shadow-pack", default=None)
    parser.add_argument("--acceptance-dir", default=None)
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--no-acceptance", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = ReleaseBundleConfig(
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        name=args.name,
        overlay_path=args.overlay,
        profile_path=args.profile,
        shadow_pack_dir=args.shadow_pack,
        acceptance_dir=args.acceptance_dir,
        include_shadow=not args.no_shadow,
        include_acceptance=not args.no_acceptance,
        create_archive=args.archive,
    )

    result = build_release_bundle(config)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Release bundle: {result['bundle_dir']}")
        print(f"Manifest: {result['manifest_path']}")
        if result["archive_path"]:
            print(f"Archive: {result['archive_path']}")
        if result["missing"]:
            print(f"Missing: {', '.join(result['missing'])}")


if __name__ == "__main__":
    _main()
