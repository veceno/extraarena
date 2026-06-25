# TrainV2 Artifact Reference

This document describes the TrainV2 operator artifact chain, version strings, and safety notes.

> **Important:** Smoke artifacts are synthetic and not suitable for inference. Real artifacts must come from actual training runs, ONNX exports, and shadow comparisons.

---

## Artifact Chain

```text
suite_summary.json
  → candidate.json
    → candidate_profile.json
      → profile_overlay.json
        → shadow_evidence/
          manifest.json
          shadow_summary.json
          shadow_result.json
          shadow_summary.md
          shadow_mismatches.json
        → acceptance_gate/
          acceptance_gate.json
          acceptance_gate.md
      → release_bundle/
        release_manifest.json
        README.md
        model/
        candidate/
        profile/
        shadow_evidence/
        acceptance_gate/
  → panel snapshot (train_v2_panel_snapshot_v1)
```

---

## Version Registry

| Artifact | Version String |
|---|---|
| Profile registry | `train_v2_profile_registry_v1` |
| Profile overlay | `train_v2_profile_overlay_v1` |
| Shadow evidence | `train_v2_shadow_evidence_v1` |
| Acceptance gate | `train_v2_acceptance_gate_v1` |
| Release bundle | `train_v2_release_bundle_v1` |
| Panel snapshot | `train_v2_panel_snapshot_v1` |

---

## Artifacts

### `candidate.json`

- **Created by:** suite promotion / manual
- **Purpose:** links a run to a candidate ONNX model
- **Key fields:** `model_name`, `score`, `candidate_onnx`, `source_onnx`
- **Safety:** read-only reference, no production impact

### `candidate_profile.json`

- **Created by:** `ai.train_v2.candidate_profile`
- **Purpose:** TrainV2 profile pack ready for overlay generation
- **Key fields:** `difficulty`, `profile.model_path`, `profile.format`, `source`
- **Safety:** does not modify production configs automatically

### `profile_overlay.json`

- **Created by:** `ai.train_v2.profile_registry`
- **Version:** `train_v2_profile_overlay_v1`
- **Purpose:** portable overlay for local shadow comparison
- **Key fields:** `profiles.<difficulty>`, `model_path`, `format`
- **Safety:** read-only, must be manually connected for production use

### `shadow_evidence/manifest.json`

- **Created by:** `ai.train_v2.shadow_report`
- **Version:** `train_v2_shadow_evidence_v1`
- **Purpose:** evidence pack from legacy vs overlay comparison
- **Key fields:** `summary.episodes`, `summary.steps`, `summary.match_rate`, `summary.mismatches`, `summary.overlay_latency_ms_p95`
- **Safety:** read-only, does not modify production profiles

### `acceptance_gate.json`

- **Created by:** `ai.train_v2.acceptance_gate`
- **Version:** `train_v2_acceptance_gate_v1`
- **Purpose:** PASS/WARN/FAIL recommendation based on thresholds
- **Key fields:** `status`, `score`, `checks[]`, `summary`
- **Safety:** read-only decision layer, does not auto-promote to production

### `release_manifest.json`

- **Created by:** `ai.train_v2.release_bundle`
- **Version:** `train_v2_release_bundle_v1`
- **Purpose:** release candidate bundle manifest
- **Key fields:** `model_name`, `artifacts.*`, `files[]`, `missing[]`
- **Safety:** self-contained read-only artifact tree

### Panel snapshot

- **Created by:** `ai.train_v2.operator snapshot`
- **Version:** `train_v2_panel_snapshot_v1`
- **Purpose:** exported panel state for sharing or archiving
- **Key fields:** `generated_at`, `data`
- **Safety:** read-only JSON export

---

## Production Safety

None of these artifacts modify production game server configs automatically. Connection to production requires explicit manual steps:

1. Review `profile_overlay.json`
2. Validate ONNX runtime behavior
3. Update `BOT_DIFFICULTY_PROFILES` or equivalent production config manually
4. Deploy through your normal release process

The operator pipeline is intentionally read-only to keep production safe.

---

## Audit (2026-06-25)

Verified against current source. No changes needed — content matches current code.

Checked:
- Artifact chain ordering (suite.py, candidate_profile.py, profile_registry.py, shadow_report.py, acceptance_gate.py, release_bundle.py).
- Version registry (ai/train_v2/__init__.py TRAIN_V2_ARTIFACT_VERSIONS).
- candidate.json key fields (suite.py promote_candidate meta block).
- candidate_profile.json key fields (candidate_profile.py build_train_v2_profile).
- profile_overlay.json key fields (profile_registry.py write_profile_overlay).
- shadow_evidence/ files and manifest summary keys (shadow_report.py summarize_shadow_result + write_shadow_evidence_pack).
- acceptance_gate.json fields (acceptance_gate.py evaluate_acceptance_gate return).
- release_manifest.json fields (release_bundle.py build_release_bundle manifest dict).
- Panel snapshot version + operator CLI (ai/train_v2/operator.py write_panel_snapshot, subcommands: panel, snapshot, doctor).
- BOT_DIFFICULTY_PROFILES existence (infrastructure/config.py:243).
- TrainV3 coexistence: TrainV3/README.md still imports codecs from `ai.train_v2` (e.g. model_mlx.ActionConditionedPolicy); TrainV2 remains the codec/operator pipeline, TrainV3 is the Rust-acceleration boundary, so the historical TrainV2 framing stays valid.
