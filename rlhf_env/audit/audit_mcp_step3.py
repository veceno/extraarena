"""Шаг 3: проверка NDJSON (F01, decision_source, accepted/won) + сравнение схем MCP vs браузер."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def analyse_dataset(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "rows": len(rows),
        "decision_sources": sorted({r.get("decision_source") for r in rows}),
        "accepted_values": sorted({str(r.get("accepted")) for r in rows}),
        "won_values": sorted({str(r.get("won")) for r in rows}),
        "winner_user_ids": sorted({str(r.get("winner_user_id")) for r in rows}),
        "is_bot_values": sorted({str(r.get("is_bot")) for r in rows}),
        "visibility_values": sorted({r.get("visibility") for r in rows}),
    }
    # F01: opponent-of-actor hand hidden
    f1_ok = 0
    f1_leak = 0
    f1_actor_hand_visible = 0
    for r in rows:
        sj = r.get("state_json", {})
        acting = r.get("acting_user_id")
        p1 = sj.get("p1", {})
        p2 = sj.get("p2", {})
        p1_hand = p1.get("hand", [])
        p2_hand = p2.get("hand", [])
        if acting == 1000:
            opp_hand = p2_hand
            own_hand = p1_hand
        elif acting == 2000:
            opp_hand = p1_hand
            own_hand = p2_hand
        else:
            continue
        opp_hidden = bool(opp_hand) and all(isinstance(h, dict) and h.get("hidden") is True for h in opp_hand)
        opp_has_content = bool(opp_hand) and any(isinstance(h, dict) and not h.get("hidden") for h in opp_hand)
        if opp_hidden and not opp_has_content:
            f1_ok += 1
        elif opp_has_content:
            f1_leak += 1
        # actor's own hand should be visible (have card content)
        own_visible = bool(own_hand) and any(isinstance(h, dict) and not h.get("hidden") for h in own_hand)
        if own_visible:
            f1_actor_hand_visible += 1
    summary["f1_opp_hidden_ok"] = f1_ok
    summary["f1_opp_leak"] = f1_leak
    summary["f1_actor_hand_visible"] = f1_actor_hand_visible

    # human rows: decision_source='human', is_bot=False
    human_rows = [r for r in rows if r.get("decision_source") == "human"]
    summary["human_rows"] = len(human_rows)
    summary["human_rows_is_bot_false"] = sum(1 for r in human_rows if r.get("is_bot") is False)
    summary["human_rows_accepted"] = sorted({str(r.get("accepted")) for r in human_rows})

    # schema keys consistency
    keysets = sorted({tuple(sorted(r.keys())) for r in rows})
    summary["distinct_keysets"] = len(keysets)
    if keysets:
        summary["sample_keyset"] = list(keysets[0])
    return summary


def main() -> int:
    # find latest MCP dataset (from step2 sessions dir, group a91ca8d60010) — but better: scan all sessions, pick latest
    sess = REPO / "rlhf_env" / "sessions"
    candidates = sorted(sess.glob("*/dataset.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = {}
    if candidates:
        out["mcp_latest_dataset"] = str(candidates[0])
        out["mcp_summary"] = analyse_dataset(candidates[0])
    # browser dataset comparison: any pre-existing dataset.jsonl from a browser session?
    # look for one not produced by these audit runs (heuristic: older mtime)
    all_ds = sorted(sess.glob("*/dataset.jsonl"), key=lambda p: p.stat().st_mtime)
    out["all_dataset_count"] = len(all_ds)
    if len(all_ds) > 1:
        # oldest as "browser-like" reference
        out["reference_oldest_dataset"] = str(all_ds[0])
        out["reference_summary"] = analyse_dataset(all_ds[0])
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())