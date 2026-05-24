"""Unified closed-loop controller for GTA repair candidates.

This script standardizes the experiment loop around existing repair modules:

1. optionally run a task-specific generator adapter,
2. locate the repaired candidate directory,
3. score it with the official GTA evaluator if needed,
4. compare before/after checkpoint scores,
5. apply a non-regression gate,
6. write a controller trace suitable for paper tables.

The goal is not to hide task-specific repair logic yet. Instead, it creates a
single harness boundary around heterogeneous repair adapters, making the
pipeline auditable and comparable across tasks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from apply_non_regression_gate import decide as gate_decide
from compare_checkpoint_scores import compare as compare_scores
from verify_opencompass_artifact_gate import verify as verify_artifact_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "closed_loop_repair_controller"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_generator_stdout(stdout: str) -> Path | None:
    text = stdout.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    raw = data.get("structured_repair_dir")
    return Path(raw) if raw else None


def write_markdown_trace(path: Path, trace: dict[str, Any]) -> None:
    gate = trace.get("gate_decision", {})
    artifact_gate = trace.get("artifact_gate_decision") or {}
    diff = trace.get("checkpoint_diff", {})
    lines = [
        f"# Closed Loop Controller Trace: {trace['case_label']}",
        "",
        f"- task_id: {trace['task_id']}",
        f"- created: {trace['created']}",
        f"- baseline_result: `{trace['baseline_result']}`",
        f"- candidate_dir: `{trace['candidate_dir']}`",
        f"- repaired_result: `{trace['repaired_result']}`",
        f"- checkpoint_diff: `{trace['checkpoint_diff_path']}`",
        f"- gate_decision: **{gate.get('decision')}**",
        f"- gate_reason: {gate.get('reason')}",
        f"- artifact_gate_decision: **{artifact_gate.get('decision', '-')}**",
        f"- artifact_gate_reason: {artifact_gate.get('reason', '-')}",
        "",
        "## Scores",
        "",
        "| before | after | delta | low checkpoints | collateral damage |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            str(diff.get(key))
            for key in [
                "before_root_score",
                "after_root_score",
                "root_delta",
                "num_low_before",
                "num_collateral_damage",
            ]
        )
        + " |",
        "",
        "## Controller Steps",
        "",
    ]
    for step in trace.get("steps", []):
        lines.append(f"- {step['name']}: {step['status']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--generator-script", type=Path)
    parser.add_argument("--repaired-result", type=Path)
    parser.add_argument("--checkpoint-diff", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--damage-threshold", type=float, default=1.0)
    parser.add_argument("--min-root-delta", type=float, default=0.0)
    parser.add_argument("--min-after-score", type=float, default=7.0)
    parser.add_argument("--min-target-delta", type=float, default=0.0)
    parser.add_argument(
        "--artifact-gate-prediction",
        type=Path,
        help="Optional OpenCompass prediction JSON used to infer required artifacts for the repaired candidate.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Optional artifact directory for Artifact Gate. Defaults to candidate-dir when --artifact-gate-prediction is set.",
    )
    parser.add_argument("--min-artifact-score", type=float, default=1.0)
    parser.add_argument(
        "--artifact-score-threshold",
        type=float,
        default=8.0,
        help="Official score threshold used by Artifact Gate to flag false-high cases.",
    )
    parser.add_argument("--no-score", action="store_true", help="Require --repaired-result and skip official scoring.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"task{args.task_id}_{args.case_label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    candidate_dir = args.candidate_dir
    if args.generator_script:
        command = [sys.executable, str(args.generator_script)]
        run = run_command(command, ROOT)
        dump_json(out_dir / "generator_run.json", run)
        steps.append(
            {
                "name": "repair_generator_adapter",
                "status": "ok" if run["returncode"] == 0 else "failed",
                "artifact": str(out_dir / "generator_run.json"),
            }
        )
        if run["returncode"] != 0:
            raise SystemExit(f"generator failed: {args.generator_script}")
        parsed_dir = parse_generator_stdout(run["stdout"])
        candidate_dir = candidate_dir or parsed_dir

    if candidate_dir is None:
        raise SystemExit("candidate_dir is required when no generator emits structured_repair_dir")
    candidate_dir = candidate_dir.resolve()
    if not candidate_dir.exists():
        raise SystemExit(f"candidate_dir does not exist: {candidate_dir}")

    repaired_result = args.repaired_result
    if repaired_result is None:
        if args.no_score:
            raise SystemExit("--no-score requires --repaired-result")
        repaired_result = out_dir / "official_rescore_result.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "score_candidate_with_official_evaluator.py"),
            "--task-id",
            str(args.task_id),
            "--candidate-dir",
            str(candidate_dir),
            "--out",
            str(repaired_result),
        ]
        run = run_command(command, ROOT)
        dump_json(out_dir / "official_score_run.json", run)
        steps.append(
            {
                "name": "official_rescore",
                "status": "ok" if run["returncode"] == 0 else "failed",
                "artifact": str(out_dir / "official_score_run.json"),
            }
        )
        if run["returncode"] != 0:
            raise SystemExit("official rescore failed")
    else:
        repaired_result = repaired_result.resolve()
        steps.append({"name": "official_rescore", "status": "reused", "artifact": str(repaired_result)})

    artifact_gate_report = None
    artifact_gate_decision = None
    artifact_gate_report_path = None
    artifact_gate_decision_path = None
    if args.artifact_gate_prediction:
        artifact_dir = (args.artifact_dir or candidate_dir).resolve()
        artifact_gate_report = verify_artifact_gate(
            args.artifact_gate_prediction.resolve(),
            repaired_result.resolve(),
            args.artifact_score_threshold,
            artifact_dir,
        )
        artifact_gate_report_path = out_dir / "artifact_gate_report.json"
        dump_json(artifact_gate_report_path, artifact_gate_report)
        artifact_score = float(artifact_gate_report.get("artifact_score") or 0.0)
        if artifact_gate_report.get("false_high_score"):
            decision = "REJECT"
            reason = "official score is high but required artifacts are missing"
        elif artifact_score < args.min_artifact_score:
            decision = "REJECT"
            reason = f"artifact_score {artifact_score:.2f} is below required {args.min_artifact_score:.2f}"
        else:
            decision = "ACCEPT"
            reason = f"artifact_score {artifact_score:.2f} satisfies required {args.min_artifact_score:.2f}"
        artifact_gate_decision = {
            "decision": decision,
            "reason": reason,
            "artifact_score": artifact_score,
            "false_high_score": artifact_gate_report.get("false_high_score"),
            "expected_artifact_count": artifact_gate_report.get("expected_artifact_count"),
            "passed_artifact_count": artifact_gate_report.get("passed_artifact_count"),
            "report": str(artifact_gate_report_path),
        }
        artifact_gate_decision_path = out_dir / "artifact_gate_decision.json"
        dump_json(artifact_gate_decision_path, artifact_gate_decision)
        steps.append(
            {
                "name": "artifact_gate",
                "status": artifact_gate_decision["decision"],
                "artifact": str(artifact_gate_report_path),
            }
        )

    if args.checkpoint_diff:
        diff_path = args.checkpoint_diff.resolve()
        diff = load_json(diff_path)
        steps.append({"name": "checkpoint_diff", "status": "reused", "artifact": str(diff_path)})
    else:
        diff = compare_scores(
            load_json(args.baseline_result.resolve()),
            load_json(repaired_result),
            threshold=args.threshold,
            damage_threshold=args.damage_threshold,
        )
        diff_path = out_dir / "checkpoint_diff.json"
        dump_json(diff_path, diff)
        steps.append({"name": "checkpoint_diff", "status": "computed", "artifact": str(diff_path)})

    gate = gate_decide(
        args.case_label,
        diff,
        min_root_delta=args.min_root_delta,
        min_after_score=args.min_after_score,
        min_target_delta=args.min_target_delta,
    )
    gate_path = out_dir / "gate_decision.json"
    dump_json(gate_path, gate)
    steps.append({"name": "non_regression_gate", "status": gate["decision"], "artifact": str(gate_path)})

    final_decision = gate["decision"]
    final_reason = gate.get("reason")
    if artifact_gate_decision and artifact_gate_decision["decision"] == "REJECT":
        final_decision = "REJECT"
        final_reason = f"artifact gate rejected candidate: {artifact_gate_decision['reason']}"

    trace = {
        "schema_version": "0.1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "task_id": args.task_id,
        "case_label": args.case_label,
        "baseline_result": str(args.baseline_result.resolve()),
        "candidate_dir": str(candidate_dir),
        "repaired_result": str(repaired_result),
        "checkpoint_diff_path": str(diff_path),
        "gate_decision_path": str(gate_path),
        "artifact_gate_report_path": None if artifact_gate_report_path is None else str(artifact_gate_report_path),
        "artifact_gate_decision_path": None if artifact_gate_decision_path is None else str(artifact_gate_decision_path),
        "checkpoint_diff": {k: v for k, v in diff.items() if k != "rows"},
        "gate_decision": gate,
        "artifact_gate": artifact_gate_report,
        "artifact_gate_decision": artifact_gate_decision,
        "final_decision": final_decision,
        "final_reason": final_reason,
        "steps": steps,
    }
    dump_json(out_dir / "controller_trace.json", trace)
    write_markdown_trace(out_dir / "controller_trace.md", trace)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "decision": final_decision,
                "non_regression_decision": gate["decision"],
                "artifact_gate_decision": None if artifact_gate_decision is None else artifact_gate_decision["decision"],
                **trace["checkpoint_diff"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
