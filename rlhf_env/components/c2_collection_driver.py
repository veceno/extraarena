"""Phase-C C2 human-trace observer and harvester.

The browser/web process owns human matches.  MCP processes have independent
in-memory match managers, so this component never creates or advances a human
series through MCP.  It observes completed ``human-vs-rl`` groups persisted in
the shared sessions directory, deep-validates them, verifies V5 provenance,
and returns their group directories to the offline replay bridge.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

_BATTLES_PER_SPEC_CAP = 1000


class McpCollectionClient(Protocol):
    def list_v5_groups(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def get_v5_dataset_summary(self, group_id: str) -> Dict[str, Any]: ...
    def get_v5_trace(self, group_id: str, battle_id: str, what: str) -> Dict[str, Any]: ...
    def validate_v5_traces(self, group_id: str) -> Dict[str, Any]: ...


@dataclass
class C2CollectionResult:
    status: str = "skipped"  # ok | skipped
    mana_draw_row_count: int = 0
    battle_count: int = 0
    groups_collected: int = 0
    stopped_reason: str = ""
    reason: str = ""
    group_ids: List[str] = field(default_factory=list)
    group_dirs: List[str] = field(default_factory=list)
    rejected_groups: List[Dict[str, Any]] = field(default_factory=list)
    series_plans: List[Dict[str, Any]] = field(default_factory=list)


class C2CollectionDriver:
    """Harvest fresh completed human-vs-V5 groups; never drive human battles."""

    def __init__(
        self,
        v5_checkpoint_path: str,
        *,
        mana_draw_floor: int = 5000,
        battle_cap: int = 5000,
        battles_per_series: int = 1000,
        expected_catalog_hash: str | None = None,
        consumed_group_ids: Optional[set[str]] = None,
    ) -> None:
        if battles_per_series <= 0:
            raise ValueError("battles_per_series must be > 0")
        self.v5_checkpoint_path = str(v5_checkpoint_path)
        self.mana_draw_floor = int(mana_draw_floor)
        self.battle_cap = int(battle_cap)
        self.battles_per_series = int(battles_per_series)
        self.expected_catalog_hash = expected_catalog_hash
        self.expected_weights_hash = self._checkpoint_hash(self.v5_checkpoint_path)
        self.consumed_group_ids = consumed_group_ids if consumed_group_ids is not None else set()

    @staticmethod
    def _checkpoint_hash(path: str) -> str | None:
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def plan_series_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        produced = 0
        while produced < self.battle_cap:
            count = min(self.battles_per_series, _BATTLES_PER_SPEC_CAP, self.battle_cap - produced)
            specs.append(self.plan_series_spec(len(specs), battles_planned=count))
            produced += count
        return specs

    def plan_series_spec(self, start_index: int = 0, *, battles_planned: Optional[int] = None) -> Dict[str, Any]:
        count = min(max(int(battles_planned or self.battles_per_series), 1), _BATTLES_PER_SPEC_CAP)
        return {
            "p1_actor_type": "human",
            "p2_model": {"name": "v5-deploy", "path": self.v5_checkpoint_path, "kind": "v5"},
            "battles_planned": count,
            "starting_player": "random",
            "seed": 1000 + int(start_index),
        }

    def collect(self, mcp_client: McpCollectionClient) -> C2CollectionResult:
        result = C2CollectionResult(series_plans=self.plan_series_specs())
        listed = mcp_client.list_v5_groups(battle_tag="human-vs-rl", limit=10_000)
        groups = listed.get("groups", []) if isinstance(listed, dict) else []
        for group in groups:
            gid = str(group.get("group_id") or "")
            if not gid or gid in self.consumed_group_ids or not group.get("finished_at"):
                continue
            summary = mcp_client.get_v5_dataset_summary(gid)
            accepted, detail = self._harvest_group(mcp_client, gid, summary)
            if not accepted:
                result.rejected_groups.append({"group_id": gid, **detail})
                continue
            battles = int(detail["battle_count"])
            draws = int(detail["mana_draw_row_count"])
            result.group_ids.append(gid)
            result.group_dirs.append(str(detail["group_dir"]))
            result.groups_collected += 1
            result.battle_count += battles
            result.mana_draw_row_count += draws
            self.consumed_group_ids.add(gid)
            if result.mana_draw_row_count >= self.mana_draw_floor:
                result.stopped_reason = "floor"
                break
            if result.battle_count >= self.battle_cap:
                result.stopped_reason = "cap"
                break

        if result.groups_collected:
            result.status = "ok"
            result.stopped_reason = result.stopped_reason or "available"
        else:
            result.status = "skipped"
            result.stopped_reason = "waiting_for_human_data"
            result.reason = "no fresh completed human-vs-rl groups passed Phase-C gates"
        return result

    def _harvest_group(
        self, mcp_client: McpCollectionClient, gid: str, summary: Any,
    ) -> tuple[bool, Dict[str, Any]]:
        if not isinstance(summary, dict) or summary.get("error"):
            return False, {"reason": "summary_unavailable"}
        if not summary.get("group_dir"):
            return False, {"reason": "group_dir_missing"}
        finished = int(summary.get("battles_finished", 0) or 0)
        battle_ids = [str(b) for b in (summary.get("battle_ids") or [])]
        if finished <= 0 or len(battle_ids) != finished:
            return False, {"reason": "incomplete_manifest", "finished": finished, "battle_ids": battle_ids}
        if int(summary.get("current_card_count", 0) or 0) != 50:
            return False, {"reason": "wrong_card_catalog_size", "current_card_count": summary.get("current_card_count")}
        if self.expected_catalog_hash and summary.get("current_catalog_hash") != self.expected_catalog_hash:
            return False, {"reason": "catalog_hash_mismatch"}
        deep = mcp_client.validate_v5_traces(gid)
        if (not isinstance(deep, dict) or deep.get("error") or
                int(deep.get("checked", 0)) != finished or int(deep.get("ok", 0)) != finished or deep.get("broken")):
            return False, {"reason": "deep_trace_validation_failed", "validation": deep}

        warnings = [w for b in (summary.get("battles") or []) for w in (b.get("policy_warnings") or [])]
        if warnings or any(bool(b.get("degraded")) for b in (summary.get("battles") or [])):
            return False, {"reason": "policy_fallback", "policy_warnings": warnings}

        mana_draw_rows = 0
        for bid in battle_ids:
            meta_resp = mcp_client.get_v5_trace(gid, bid, "meta")
            meta = meta_resp.get("data", {}) if isinstance(meta_resp, dict) else {}
            policy = meta.get("bot_policy") or {}
            if meta.get("battle_tag") != "human-vs-rl" or meta.get("p1_actor_type") != "human":
                return False, {"reason": "wrong_actor_topology", "battle_id": bid}
            if policy.get("kind") != "v5":
                return False, {"reason": "wrong_policy_kind", "battle_id": bid, "policy": policy}
            if self.expected_weights_hash and policy.get("weights_hash") != self.expected_weights_hash:
                return False, {"reason": "weights_hash_mismatch", "battle_id": bid, "policy": policy}
            if meta.get("catalog_hash") != summary.get("current_catalog_hash"):
                return False, {"reason": "trace_catalog_hash_mismatch", "battle_id": bid}
            action_resp = mcp_client.get_v5_trace(gid, bid, "actions")
            rows = action_resp.get("data", []) if isinstance(action_resp, dict) else []
            mana_draw_rows += sum(
                1 for row in rows
                if row.get("action_type") == "mana_draw" and row.get("decision_source") == "human"
            )
        return True, {
            "battle_count": finished,
            "mana_draw_row_count": mana_draw_rows,
            "group_dir": summary.get("group_dir"),
        }


__all__ = ["C2CollectionDriver", "C2CollectionResult", "McpCollectionClient"]
