"""
Catalog audit for TrainV2 — validates card encoding, mechanics coverage, and identity leakage.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np

from core.converter import card_from_db
from core.state import MECHANICS_LIST, CardInstance, CardType

from ai.train_v2.classic_rl_env import _load_cards_db
from ai.train_v2.classic_card_shape_v1 import (
    CARD_SHAPE_DIM,
    _SCALAR_PATTERNS,
    encode_card_shape,
)
from ai.train_v2.classic_obs_v1 import OBS_DIM
from ai.train_v2.classic_actions_v1 import ACTION_FEATURE_DIM


def load_current_catalog() -> dict[int, dict]:
    return _load_cards_db()


def collect_catalog_mechanics(cards_data: dict[int, dict]) -> dict:
    raw_set: set[str] = set()
    families: set[str] = set()
    unknown: set[str] = set()
    unparsed: set[str] = set()

    cards_count = len(cards_data)
    heroes = warriors = potions = 0
    for cid, item in sorted(cards_data.items()):
        ct = item.get("card_type", "warrior")
        if ct == "hero":
            heroes += 1
        elif ct == "potion":
            potions += 1
        else:
            warriors += 1

        mechs = item.get("mechanics", [])
        if isinstance(mechs, list):
            for m in mechs:
                if m:
                    raw_set.add(m)

    for m in sorted(raw_set):
        family = _match_family(m)
        if family:
            families.add(family)
        else:
            unknown.add(m)

    for m in sorted(raw_set):
        if not _has_numeric_suffix(m):
            continue
        card = _synthetic_card(m)
        enc = encode_card_shape(card)
        if np.sum(np.abs(enc[47:64])) < 1e-8:
            unparsed.add(m)

    return {
        "cards": cards_count,
        "heroes": heroes,
        "warriors": warriors,
        "potions": potions,
        "raw_mechanics": sorted(raw_set),
        "mechanic_families": sorted(families),
        "unknown_families": sorted(unknown),
        "unparsed_scalars": sorted(unparsed),
    }


def audit_catalog(cards_data: dict[int, dict] | None = None) -> dict:
    if cards_data is None:
        cards_data = load_current_catalog()

    mech_info = collect_catalog_mechanics(cards_data)

    all_encode = True
    nan_inf_cards: list[int] = []
    id_checked = True
    warnings: list[str] = []
    errors: list[str] = []

    for cid, item in sorted(cards_data.items()):
        try:
            card = card_from_db(item, level=1)
        except Exception as e:
            errors.append(f"card_from_db failed for id={cid}: {e}")
            continue

        try:
            enc = encode_card_shape(card)
        except Exception as e:
            errors.append(f"encode_card_shape failed for id={cid}: {e}")
            continue

        if np.any(np.isnan(enc)) or np.any(np.isinf(enc)):
            all_encode = False
            nan_inf_cards.append(cid)

        try:
            clone = copy.deepcopy(card)
            clone.card_id += 100000
            clone.name = "CLONE"
            clone.instance_id = uuid4()
            enc_clone = encode_card_shape(clone)
            if not np.array_equal(enc, enc_clone):
                id_checked = False
                warnings.append(f"identity leakage for card id={cid}: encoding differs with changed identity")
        except Exception as e:
            id_checked = False
            warnings.append(f"identity leakage check failed for id={cid}: {e}")

    if mech_info["unknown_families"]:
        warnings.append(f"unknown mechanic families: {mech_info['unknown_families']}")

    if mech_info["unparsed_scalars"]:
        warnings.append(f"unparsed scalar mechanics: {mech_info['unparsed_scalars']}")

    summary = {
        "total": mech_info["cards"],
        "heroes": mech_info["heroes"],
        "warriors": mech_info["warriors"],
        "potions": mech_info["potions"],
    }

    return {
        "summary": summary,
        "mechanics": mech_info,
        "encoding": {
            "card_shape_dim": CARD_SHAPE_DIM,
            "obs_dim": OBS_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "all_cards_encode": all_encode,
            "nan_or_inf_cards": nan_inf_cards,
            "identity_leakage_checked": id_checked,
        },
        "warnings": warnings,
        "errors": errors,
    }


def _match_family(mechanic: str) -> str | None:
    for family in MECHANICS_LIST:
        if mechanic == family or mechanic.startswith(family + "_"):
            return family
    return None


def _has_numeric_suffix(mechanic: str) -> bool:
    import re
    return bool(re.search(r"_\d+", mechanic))


def _synthetic_card(mechanic: str) -> CardInstance:
    return CardInstance(
        instance_id=uuid4(),
        card_id=0,
        name="synthetic",
        card_type=CardType.WARRIOR,
        mana_cost=0,
        attack=0,
        hp=1,
        max_hp=1,
        mechanics=[mechanic],
        is_ready=False,
    )


def _main():
    parser = argparse.ArgumentParser(description="Audit TrainV2 card catalog")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = audit_catalog()
    summary = result["summary"]
    mech = result["mechanics"]
    enc = result["encoding"]

    if args.json:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return

    print(f"Catalog: {summary['total']} cards ({summary['heroes']} heroes, {summary['warriors']} warriors, {summary['potions']} potions)")
    print(f"Raw mechanics: {len(mech['raw_mechanics'])} unique")
    print(f"Mechanic families: {len(mech['mechanic_families'])} matched / {len(mech['unknown_families'])} unknown")
    if mech["unknown_families"]:
        print(f"  Unknown: {mech['unknown_families']}")
    print(f"Unparsed scalars: {len(mech['unparsed_scalars'])}")
    if mech["unparsed_scalars"]:
        print(f"  Unparsed: {mech['unparsed_scalars']}")
    print(f"All cards encode: {'yes' if enc['all_cards_encode'] else 'no'} (NaN/Inf: {enc['nan_or_inf_cards']})")
    print(f"Identity leakage checked: {enc['identity_leakage_checked']}")
    print(f"Dims: card_shape={enc['card_shape_dim']}, obs={enc['obs_dim']}, action_feat={enc['action_feature_dim']}")
    print(f"Warnings: {len(result['warnings'])}")
    for w in result["warnings"]:
        print(f"  - {w}")
    print(f"Errors: {len(result['errors'])}")
    for e in result["errors"]:
        print(f"  - {e}")


if __name__ == "__main__":
    _main()
