"""Apply a non-regression gate to checkpoint-diff repair results.

The gate is intentionally simple and paper-friendly:

1. Accept only if root score improves by at least ``--min-root-delta``.
2. Accept only if the repaired root score is at least ``--min-after-score``.
3. Reject if any non-target checkpoint has collateral damage.
4. Reject if target checkpoint average delta is below ``--min-target-delta``.

This turns checkpoint-level diffs into a reusable accept/reject artifact for
selective rerun experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "non_regression_gate"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_case(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("case must be label=path")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("case label cannot be empty")
    return label, Path(path.strip())


def decide(
    label: str,
    diff: dict[str, Any],
    min_root_delta: float,
    min_after_score: float,
    min_target_delta: float,
) -> dict[str, Any]:
    reasons = []
    root_delta = float(diff.get("root_delta") or 0.0)
    after_root_score = float(diff.get("after_root_score") or 0.0)
    target_avg_delta = diff.get("target_avg_delta")
    target_avg_delta_value = None if target_avg_delta is None else float(target_avg_delta)
    num_collateral_damage = int(diff.get("num_collateral_damage") or 0)

    if root_delta < min_root_delta:
        reasons.append(f"root_delta {root_delta:.2f} < required {min_root_delta:.2f}")
    if after_root_score < min_after_score:
        reasons.append(f"after_root_score {after_root_score:.2f} < required {min_after_score:.2f}")
    if num_collateral_damage > 0:
        reasons.append(f"{num_collateral_damage} collateral-damage checkpoints detected")
    if target_avg_delta_value is None:
        reasons.append("target_avg_delta is unavailable")
    elif target_avg_delta_value < min_target_delta:
        reasons.append(f"target_avg_delta {target_avg_delta_value:.2f} < required {min_target_delta:.2f}")

    decision = "ACCEPT" if not reasons else "REJECT"
    return {
        "case": label,
        "decision": decision,
        "before_root_score": diff.get("before_root_score"),
        "after_root_score": diff.get("after_root_score"),
        "root_delta": diff.get("root_delta"),
        "num_checkpoints": diff.get("num_checkpoints"),
        "num_low_before": diff.get("num_low_before"),
        "target_avg_delta": diff.get("target_avg_delta"),
        "non_target_avg_delta": diff.get("non_target_avg_delta"),
        "num_collateral_damage": diff.get("num_collateral_damage"),
        "collateral_damage_ids": diff.get("collateral_damage_ids", []),
        "reason": "; ".join(reasons) if reasons else "passes root, target, and non-regression gates",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-root-delta", type=float, default=0.0)
    parser.add_argument("--min-after-score", type=float, default=7.0)
    parser.add_argument("--min-target-delta", type=float, default=0.0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    rows = []
    for label, path in args.case:
        diff = load_json(path)
        row = decide(
            label,
            diff,
            min_root_delta=args.min_root_delta,
            min_after_score=args.min_after_score,
            min_target_delta=args.min_target_delta,
        )
        row["diff_path"] = str(path)
        rows.append(row)

    accepted = [row for row in rows if row["decision"] == "ACCEPT"]
    rejected = [row for row in rows if row["decision"] == "REJECT"]
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "gate": {
            "min_root_delta": args.min_root_delta,
            "min_after_score": args.min_after_score,
            "min_target_delta": args.min_target_delta,
            "reject_on_collateral_damage": True,
        },
        "num_cases": len(rows),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "rows": rows,
    }
    dump_json(out_dir / "non_regression_gate_report.json", report)
    write_csv(out_dir / "non_regression_gate_report.csv", rows)
    print(json.dumps({"out_dir": str(out_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
