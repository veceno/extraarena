from rlhf_env.tools.mcp_cli import build_action, fmt_legal


def test_build_action_preserves_mana_draw():
    assert build_action({"type": "mana_draw"}) == {"type": "mana_draw"}


def test_fmt_legal_labels_mana_draw_separately():
    rendered = fmt_legal(
        [{"type": "end_turn"}, {"type": "mana_draw"}],
        {"player": {}, "opponent": {}},
    )
    assert "[0] end_turn" in rendered
    assert "[1] mana_draw" in rendered
