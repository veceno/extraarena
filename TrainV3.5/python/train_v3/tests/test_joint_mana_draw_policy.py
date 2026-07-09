"""Regression tests for the live V5 joint candidate/mana-draw policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "TrainV3.5" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_phaseA_random_bootstrap import _select_from_joint_padded  # noqa: E402


def test_joint_argmax_selects_legal_mana_draw_and_uses_joint_log_prob() -> None:
    actions, log_probs, selected, mana_draw = _select_from_joint_padded(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float32),
        counts=np.asarray([2], dtype=np.uintp),
        ids=np.asarray([17, 23], dtype=np.uintp),
        mana_draw_logits=np.asarray([5.0], dtype=np.float32),
        mana_draw_legal=np.asarray([True]),
        rng=None,
    )
    assert actions.tolist() == [23]  # valid ignored placeholder for Rust FFI
    assert selected.tolist() == [1]
    assert mana_draw.tolist() == [True]
    expected = 5.0 - np.log(np.exp(1.0) + np.exp(2.0) + np.exp(5.0))
    assert log_probs[0] == np.float32(expected)


def test_joint_argmax_masks_illegal_mana_draw_even_at_huge_logit() -> None:
    actions, _log_probs, selected, mana_draw = _select_from_joint_padded(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float32),
        counts=np.asarray([2], dtype=np.uintp),
        ids=np.asarray([17, 23], dtype=np.uintp),
        mana_draw_logits=np.asarray([100.0], dtype=np.float32),
        mana_draw_legal=np.asarray([False]),
        rng=None,
    )
    assert actions.tolist() == [23]
    assert selected.tolist() == [1]
    assert mana_draw.tolist() == [False]
