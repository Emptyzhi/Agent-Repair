"""Compare checkpoint scores before and after repair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_nodes(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    details = result.get("details") or []
    nodes: dict[str, dict[str, Any]] = {}
    for sample in details:
        for node in sample.get("nodes", []):
            nodes[str(node.get("id"))] = {
                "score": float(node.get("score", 0)),
                "analysis": str(node.get("analysis", "")),
            }
    return nodes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint_id",
        "before",
        "after",
        "delta",
        "was_low_before",
        "collateral_damage",
        "before_analysis",
        "after_analysis",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def compare(before: dict[str, Any], after: dict[str, Any], threshold: float, damage_threshold: float) -> dict[str, Any]:
    before_nodes = flatten_nodes(before)
    after_nodes = flatten_nodes(after)
    checkpoint_ids = sorted(set(before_nodes) | set(after_nodes))

    rows = []
    for cp_id in checkpoint_ids:
        b = before_nodes.get(cp_id, {"score": None, "analysis": ""})
        a = after_nodes.get(cp_id, {"score": None, "analysis": ""})
        before_score = b["score"]
        after_score = a["score"]
        delta = None if before_score is None or after_score is None else after_score - before_score
        was_low = before_score is not None and before_score < threshold
        collateral = bool(
            before_score is not None
            and after_score is not None
            and before_score >= threshold
            and delta <= -abs(damage_threshold)
        )
        rows.append(
            {
                "checkpoint_id": cp_id,
                "before": before_score,
                "after": after_score,
                "delta": delta,
                "was_low_before": was_low,
                "collateral_damage": collateral,
                "before_analysis": b.get("analysis", ""),
                "after_analysis": a.get("analysis", ""),
            }
        )

    target_rows = [row for row in rows if row["was_low_before"]]
    non_target_rows = [row for row in rows if not row["was_low_before"]]
    collateral_rows = [row for row in rows if row["collateral_damage"]]

    def avg_delta(items: list[dict[str, Any]]) -> float | None:
        vals = [row["delta"] for row in items if row["delta"] is not None]
        return None if not vals else sum(vals) / len(vals)

    return {
        "before_root_score": before.get("gpt_score"),
        "after_root_score": after.get("gpt_score"),
        "root_delta": None
        if before.get("gpt_score") is None or after.get("gpt_score") is None
        else after.get("gpt_score") - before.get("gpt_score"),
        "threshold": threshold,
        "damage_threshold": damage_threshold,
        "num_checkpoints": len(rows),
        "num_low_before": len(target_rows),
        "num_collateral_damage": len(collateral_rows),
        "target_avg_delta": avg_delta(target_rows),
        "non_target_avg_delta": avg_delta(non_target_rows),
        "collateral_damage_ids": [row["checkpoint_id"] for row in collateral_rows],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--damage-threshold", type=float, default=1.0)
    args = parser.parse_args()

    report = compare(load_json(args.before), load_json(args.after), args.threshold, args.damage_threshold)
    dump_json(args.out / "checkpoint_diff.json", report)
    write_csv(args.out / "checkpoint_diff.csv", report["rows"])
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

