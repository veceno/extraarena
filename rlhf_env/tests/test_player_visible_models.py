from pathlib import Path

from rlhf_env.server import player_visible_model_specs


def test_player_registry_exposes_only_v5_models():
    specs = [
        {"name": "v3", "kind": "legacy_onnx"},
        {"name": "v4", "kind": "action_onnx"},
        {"name": "candidate", "kind": "v5"},
        {"name": "random", "kind": "random"},
        {"name": "end_turn", "kind": "end_turn"},
    ]
    assert player_visible_model_specs(specs) == [
        {"name": "candidate", "kind": "v5"},
    ]


def test_player_form_has_no_non_v5_fallback_and_filters_cached_registry():
    root = Path(__file__).resolve().parents[1]
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "static" / "rlhf.js").read_text(encoding="utf-8")

    assert 'value="end_turn"' not in html
    assert 'id="p2_model" disabled' in html
    assert '/static/rlhf.js?v=20260716-v5-only' in html
    assert '.filter((m) => m.kind === "v5")' in js
    assert "extra-lr-v4-max" not in js
    assert "Baselines" not in js
