"""v3.1 preservation-search retry for Gemini held-out20 Full-ours.

This runner turns the non-regression module into an active search aid:

1. official root score is the primary selector;
2. artifact gate stays a hard validity constraint for final selection;
3. collateral damage is fed back into retry prompts instead of acting as a
   final rejection rule;
4. the with-non-regression branch gets preservation feedback, while the
   ablation branch gets only artifact/target feedback.

The goal is to make the preservation module score-improving rather than purely
score-preserving.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
OPENCOMPASS_DIR = ROOT / "vendor" / "GTA" / "opencompass"
KB_PATH = ROOT / "docs" / "EXPERIMENT_KB_v3.1.md"
DEFAULT_REGISTRY = ROOT / "configs" / "heldout20" / "heldout20_router_registry_gemini.json"
DEFAULT_ROUTES = ROOT / "configs" / "heldout20" / "repair_module_routes.csv"
DEFAULT_SOURCE_DATASET = OPENCOMPASS_DIR / "data" / "gta_dataset_v2" / "end.json"
DEFAULT_COMPARE_ROWS = ROOT / "configs" / "heldout20" / "heldout20_full_ours_rows.json"
DEFAULT_OUT_ROOT = ROOT / "runs" / "v3.1" / "full_ours_gemini_preservation_search"
DEFAULT_TASK_IDS = [7, 14, 45, 61, 62, 71, 79, 82, 88, 90, 92, 94, 98, 108, 111, 121, 122, 127, 133, 151]

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_checkpoint_scores import compare as compare_scores  # noqa: E402
from repair_evaluator import (  # noqa: E402
    build_checkpoint_evidence_map as repair_build_checkpoint_evidence_map,
    evaluate_candidate as evaluate_repair_candidate,
    retry_feedback_from_report as repair_retry_feedback_from_report,
    select_candidate_by_repair_evaluator,
)
from run_heldout20_full_ours_candidate_generation import (  # noqa: E402
    build_generation_prompt,
    call_llm_json,
    default_model,
    dump_json,
    expected_files_from_prediction,
    flatten_nodes,
    load_json,
    load_source_task,
    materialize_candidate_files,
    read_score,
    registry_rows,
    route_rows,
    run_cmd,
    write_csv,
)
from scorer_runtime import scorer_python  # noqa: E402
from tool_repair_planner import (  # noqa: E402
    build_tool_plan_prompt,
    call_tool_planner,
    execute_tool_plan,
    format_evidence_for_prompt,
    resource_map,
)
from verify_opencompass_artifact_gate import first_sample, infer_expected_artifacts, verify as verify_artifact_gate  # noqa: E402

MAX_ATTEMPTS = 3
QUALITY_FLOOR = 7.0
MIN_REFINEMENT_ATTEMPTS_AFTER_BASELINE = 2
FALLBACK_EPSILON_ROOT = 0.2
FALLBACK_EPSILON_CP = 0.25


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def default_base_url() -> str:
    if first_env("DEEPSEEK_API_KEY", "deepseek_api_key"):
        return os.environ.get("DEEPSEEK_OPENAI_BASE_URL", "https://api.deepseek.com")
    if first_env("GLM_API_KEY", "glm_api_key"):
        return os.environ.get("GLM_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    if first_env("DASHSCOPE_API_KEY"):
        return os.environ.get("EVAL_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def load_source_task(task_id: int) -> dict[str, Any]:
    data = load_json(DEFAULT_SOURCE_DATASET)
    return data[str(task_id)]


def low_checkpoint_diagnostics(result: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    diagnostics = []
    for node in flatten_nodes(result):
        score = float(node.get("score", 0))
        if score >= threshold:
            continue
        diagnostics.append(
            {
                "checkpoint_id": node.get("id"),
                "score": score,
                "analysis": str(node.get("analysis", ""))[:1200],
            }
        )
    return diagnostics


def summarize_artifact_feedback(report: dict[str, Any], artifact_gate: dict[str, Any]) -> dict[str, Any]:
    failed_checks = []
    for check in report.get("checks", []):
        if not check.get("passed"):
            failed_checks.append(
                {
                    "artifact_id": check.get("artifact_id"),
                    "expected_name": check.get("expected_name"),
                    "expected_ext": check.get("expected_ext"),
                    "why_failed": check.get("why_failed"),
                    "partial": check.get("partial"),
                }
            )
    return {
        "decision": artifact_gate["decision"],
        "reason": artifact_gate["reason"],
        "artifact_score": artifact_gate["artifact_score"],
        "false_high_score": report.get("false_high_score"),
        "score_threshold": report.get("score_threshold"),
        "expected_artifact_count": report.get("expected_artifact_count"),
        "passed_artifact_count": report.get("passed_artifact_count"),
        "partial_artifact_count": report.get("partial_artifact_count"),
        "failed_checks": failed_checks[:8],
    }


def load_selection_config(path: Path | None, allow_fallback: bool = False) -> dict[str, Any]:
    if path is None and not allow_fallback:
        raise FileNotFoundError("A calibrated --selection-config is required. Use --allow-fallback-selection-config only for exploratory runs.")
    if path is not None and not path.exists() and not allow_fallback:
        raise FileNotFoundError(f"Selection config not found: {path}")
    config = {
        "epsilon_root": FALLBACK_EPSILON_ROOT,
        "epsilon_cp": FALLBACK_EPSILON_CP,
        "selection_config_source": "fallback",
    }
    if path and path.exists():
        data = load_json(path)
        source = data.get("selection_config") if isinstance(data.get("selection_config"), dict) else data
        config["epsilon_root"] = float(source.get("epsilon_root", config["epsilon_root"]))
        config["epsilon_cp"] = float(source.get("epsilon_cp", config["epsilon_cp"]))
        config["selection_config_source"] = str(source.get("selection_config_source") or source.get("source") or path)
    return config


def scorer_command(
    *,
    backend: str,
    model: str,
    task_id: int,
    candidate_dir: Path,
    out: Path,
) -> list[str]:
    if backend == "gemini":
        return [
            str(scorer_python()),
            str(ROOT / "scripts" / "score_candidate_with_gta2_gemini_evaluator.py"),
            "--task-id",
            str(task_id),
            "--candidate-dir",
            str(candidate_dir),
            "--out",
            str(out),
            "--model",
            model,
        ]
    if backend == "responses":
        return [
            str(scorer_python()),
            str(ROOT / "scripts" / "score_candidate_with_gta2_responses_evaluator.py"),
            "--task-id",
            str(task_id),
            "--candidate-dir",
            str(candidate_dir),
            "--out",
            str(out),
            "--model",
            model,
        ]
    if backend == "legacy":
        return [
            str(scorer_python()),
            str(ROOT / "scripts" / "score_candidate_with_official_evaluator.py"),
            "--task-id",
            str(task_id),
            "--candidate-dir",
            str(candidate_dir),
            "--out",
            str(out),
        ]
    raise ValueError(f"Unsupported scorer backend: {backend}")


def baseline_scorer_command(
    *,
    backend: str,
    model: str,
    task_id: int,
    prediction: Path,
    out: Path,
) -> list[str] | None:
    if backend == "gemini":
        return [
            str(scorer_python()),
            str(ROOT / "scripts" / "score_prediction_with_gta2_gemini_evaluator.py"),
            "--task-id",
            str(task_id),
            "--prediction",
            str(prediction),
            "--out",
            str(out),
            "--model",
            model,
        ]
    if backend == "responses":
        return [
            str(scorer_python()),
            str(ROOT / "scripts" / "score_prediction_with_gta2_responses_evaluator.py"),
            "--task-id",
            str(task_id),
            "--prediction",
            str(prediction),
            "--out",
            str(out),
            "--model",
            model,
        ]
    if backend == "legacy":
        return None
    raise ValueError(f"Unsupported scorer backend: {backend}")


def resolve_baseline_result(
    *,
    task_id: int,
    reg: dict[str, Any],
    task_dir: Path,
    scorer_backend: str,
    scorer_model: str,
) -> Path:
    baseline_result = Path(reg["baseline_result"]).resolve()
    baseline_prediction = Path(reg["baseline_prediction"]).resolve()
    command = baseline_scorer_command(
        backend=scorer_backend,
        model=scorer_model,
        task_id=task_id,
        prediction=baseline_prediction,
        out=task_dir / "baseline_gta2_rescore" / "official_rescore_result.json",
    )
    if command is None:
        return baseline_result

    out = Path(command[command.index("--out") + 1])
    run_record = out.parent / "official_rescore_run.json"
    rescore_run = run_cmd(command)
    dump_json(run_record, rescore_run)
    if rescore_run["returncode"] != 0 or not out.exists():
        raise RuntimeError(f"Baseline GTA-2 scorer failed for task {task_id}. See {run_record}")
    return out


def _choose_artifact(expected_files: list[str], preferred: tuple[str, ...]) -> str:
    for name in expected_files:
        if Path(name).suffix.lower() in preferred:
            return Path(name).name
    return Path(expected_files[0]).name if expected_files else "repaired_final_answer.md"


def checkpoint_weight_map(task: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}

    def walk(node: dict[str, Any], inherited: float) -> None:
        children = node.get("sub_tasks") or []
        if not children:
            cp_id = str(node.get("id") or "")
            if cp_id:
                weights[cp_id] = inherited
            return
        total = sum(float(child.get("weight", 1.0) or 1.0) for child in children) or 1.0
        for child in children:
            child_weight = float(child.get("weight", 1.0) or 1.0) / total
            walk(child, inherited * child_weight)

    root = {"id": "root", "sub_tasks": task.get("sub_tasks", [])}
    walk(root, 1.0)
    return weights


def _evidence_units_for_text(cp_id: str, text: str, expected_files: list[str]) -> list[dict[str, Any]]:
    lowered = text.lower()
    units = []
    if "csv" in lowered or "table" in lowered or "spreadsheet" in lowered:
        artifact = _choose_artifact(expected_files, (".csv", ".xlsx"))
        units.append({"artifact": artifact, "unit_id": "csv.schema", "type": "csv_schema", "reason": "checkpoint mentions table/csv evidence"})
        units.append({"artifact": artifact, "unit_id": "table.generated", "type": "table", "reason": "checkpoint mentions table/csv evidence"})
    if "ocr" in lowered or "text extraction" in lowered or "extract" in lowered:
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html")), "unit_id": "section.ocr_evidence", "type": "section", "reason": "checkpoint mentions OCR or extraction"})
    if "visual" in lowered or "image" in lowered or "figure" in lowered or "comparison" in lowered:
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html", ".png", ".jpg", ".jpeg")), "unit_id": "section.visual_comparison", "type": "section", "reason": "checkpoint mentions visual/image evidence"})
    if "source" in lowered or "reference" in lowered or "citation" in lowered or "ground" in lowered:
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html")), "unit_id": "section.source_appendix", "type": "section", "reason": "checkpoint mentions source grounding"})
    if "method" in lowered or "calculation" in lowered or "formula" in lowered:
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html")), "unit_id": "section.methods", "type": "section", "reason": "checkpoint mentions method/calculation"})
    if "file" in lowered or "deliverable" in lowered or "final answer" in lowered:
        units.append({"artifact": "repaired_final_answer.md", "unit_id": "final_answer.deliverables", "type": "final_answer", "reason": "checkpoint mentions final deliverable listing"})
    if not units:
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html", ".csv", ".xlsx")), "unit_id": f"section.checkpoint_{re.sub(r'[^a-zA-Z0-9]+', '_', cp_id).strip('_')}", "type": "section", "reason": "fallback checkpoint-local section"})
    return units


def build_checkpoint_evidence_map(
    task: dict[str, Any],
    baseline: dict[str, Any],
    expected_files: list[str],
    threshold: float,
    route: dict[str, Any],
    tool_evidence: str | None,
) -> dict[str, Any]:
    by_id = {str(node.get("id")): node for node in flatten_nodes(baseline)}
    entries = {}
    for cp_id, node in by_id.items():
        score = float(node.get("score", 0.0))
        analysis = str(node.get("analysis", ""))
        status = "target_low" if score < threshold else "protected_high"
        entries[cp_id] = {
            "checkpoint_id": cp_id,
            "status": status,
            "baseline_score": score,
            "evidence_units": _evidence_units_for_text(cp_id, f"{cp_id}\n{analysis}", expected_files),
            "analysis_excerpt": analysis[:700],
        }
    return {
        "map_version": "v1_heuristic",
        "threshold": threshold,
        "route_name": route.get("route_name"),
        "routed_modules": route.get("routed_modules"),
        "expected_files": [Path(name).name for name in expected_files],
        "tool_evidence_available": bool(tool_evidence and "No tool evidence" not in tool_evidence),
        "checkpoints": entries,
    }


def protected_units_from_evidence_map(evidence_map: dict[str, Any]) -> list[str]:
    units = []
    for cp in (evidence_map.get("checkpoints") or {}).values():
        if cp.get("status") != "protected_high":
            continue
        for unit in cp.get("evidence_units", []) or []:
            unit_id = str(unit.get("unit_id") or "")
            artifact = str(unit.get("artifact") or "")
            if unit_id:
                units.append(unit_id)
                if artifact:
                    units.append(f"{Path(artifact).name}:{unit_id}")
    return sorted(set(units))


def validate_evidence_map(evidence_map: dict[str, Any], artifact_spec: dict[str, Any]) -> dict[str, Any]:
    actual_units = set()
    for artifact_name, artifact in (artifact_spec.get("artifacts") or {}).items():
        for unit in artifact.get("units", []) or []:
            unit_id = str(unit.get("unit_id") or "")
            if unit_id:
                actual_units.add(unit_id)
                actual_units.add(f"{Path(artifact_name).name}:{unit_id}")
        for column in artifact.get("schema", []) or []:
            column_unit = f"column.{re.sub(r'[^a-zA-Z0-9]+', '_', str(column).strip().lower()).strip('_')}"
            actual_units.add(column_unit)
            actual_units.add(f"{Path(artifact_name).name}:{column_unit}")

    unresolved = []
    protected = []
    for cp_id, cp in (evidence_map.get("checkpoints") or {}).items():
        for unit in cp.get("evidence_units", []) or []:
            unit_id = str(unit.get("unit_id") or "")
            artifact = Path(str(unit.get("artifact") or "")).name
            qualified = f"{artifact}:{unit_id}" if artifact and unit_id else unit_id
            row = {"checkpoint_id": cp_id, "status": cp.get("status"), "artifact": artifact, "unit_id": unit_id}
            if unit_id not in actual_units and qualified not in actual_units:
                unresolved.append(row)
            elif cp.get("status") == "protected_high":
                protected.append(row)
    return {
        "protected_units": protected,
        "unresolved_evidence_units": unresolved,
    }


def summarize_preservation_feedback(
    diff: dict[str, Any],
    threshold: float,
    epsilon_cp: float,
    checkpoint_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = diff.get("rows") or []
    target_rows = [row for row in rows if row.get("was_low_before")]
    protected_rows = [row for row in rows if not row.get("was_low_before") and row.get("delta") is not None and row["delta"] < 0]
    protected_damage_score = 0.0
    max_single_damage = 0.0
    protected_damage_ids = []
    for row in protected_rows:
        delta = float(row["delta"])
        checkpoint_id = str(row.get("checkpoint_id") or "")
        weight = float((checkpoint_weights or {}).get(checkpoint_id, 1.0))
        damage = max(0.0, -delta - epsilon_cp)
        if damage <= 0:
            continue
        protected_damage_score += damage * weight
        max_single_damage = max(max_single_damage, damage)
        protected_damage_ids.append(row.get("checkpoint_id"))
    return {
        "before_root_score": diff.get("before_root_score"),
        "after_root_score": diff.get("after_root_score"),
        "root_delta": diff.get("root_delta"),
        "target_avg_delta": diff.get("target_avg_delta"),
        "non_target_avg_delta": diff.get("non_target_avg_delta"),
        "num_collateral_damage": diff.get("num_collateral_damage"),
        "collateral_damage_ids": diff.get("collateral_damage_ids", []),
        "protected_damage_score": protected_damage_score,
        "max_single_damage": max_single_damage,
        "protected_damage_ids": protected_damage_ids,
        "epsilon_cp": epsilon_cp,
        "target_examples": target_rows[:6],
        "protected_examples": protected_rows[:6],
    }


def summarize_artifact_only_feedback(report: dict[str, Any], artifact_gate: dict[str, Any]) -> dict[str, Any]:
    base = summarize_artifact_feedback(report, artifact_gate)
    return {
        "artifact_feedback": base,
        "selection_hint": "fix the artifact issues first; do not worry about protected checkpoints in this branch",
    }


def build_preservation_prompt(
    task_id: int,
    task: dict[str, Any],
    baseline_score: float,
    diagnostics: list[dict[str, Any]],
    route: dict[str, Any],
    registry: dict[str, Any],
    expected_files: list[str],
    feedback: dict[str, Any],
    tool_evidence: str | None = None,
    checkpoint_evidence_map: dict[str, Any] | None = None,
) -> str:
    origin_prompt = ""
    for dialog in task.get("dialogs", []):
        if dialog.get("role") == "user":
            origin_prompt = str(dialog.get("content", ""))
            break
    return f"""
GTA-2 held-out task id: {task_id}

Original user task:
{origin_prompt}

Checkpoint tree:
{json.dumps(task.get("sub_tasks", []), ensure_ascii=False, indent=2)}

Baseline official score: {baseline_score}

Low-checkpoint diagnostics from the official evaluator:
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

Frozen route metadata:
{json.dumps(route, ensure_ascii=False, indent=2)}

Baseline artifact gate metadata:
{json.dumps({k: registry.get(k) for k in ["artifact_gate_decision", "artifact_score", "artifact_false_high", "failure_type"]}, ensure_ascii=False, indent=2)}

Expected deliverable filenames to materialize:
{json.dumps(expected_files, ensure_ascii=False, indent=2)}

Controller-executed tool evidence cache:
{tool_evidence or "No tool evidence cache was requested or available."}

Checkpoint-to-evidence map:
{json.dumps(checkpoint_evidence_map or {}, ensure_ascii=False, indent=2)}

Previous candidate feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

This is a preservation retry. Keep the repaired target evidence, reduce the protected-checkpoint damage, and only make minimal edits needed to restore collateral checkpoints.

Return JSON with:
- final_answer_markdown: concise final answer that names every materialized deliverable.
- deliverables: object mapping each expected filename to complete content.
- preserve_units: evidence units that must not be rewritten.
- patches: patch-style edits over artifact units. Prefer add_section, update_section, add_or_update_section, or update_table_cells.
- do_not_modify: protected units such as csv.schema, figure captions, existing calculations, and protected evidence map units.
- repair_rationale: list of short reasons tied to checkpoint diagnostics, route modules, and preservation feedback.

Rules:
- Do not claim a file exists unless it appears in deliverables.
- If a PDF/DOCX/PPTX/XLSX is expected, provide markdown/text/table content for that file; the runner will materialize the binary container.
- Do not rewrite the whole artifact unless the feedback says the artifact is missing or unreadable.
- Prefer patch-style repair: preserve protected evidence units, restore damaged units, and only add/update target evidence.
""".strip()


def build_retry_prompt(
    task_id: int,
    task: dict[str, Any],
    baseline_score: float,
    diagnostics: list[dict[str, Any]],
    route: dict[str, Any],
    registry: dict[str, Any],
    expected_files: list[str],
    feedback: dict[str, Any],
    branch_name: str,
    tool_evidence: str | None = None,
    checkpoint_evidence_map: dict[str, Any] | None = None,
    retry_mode: str = "target_retry",
) -> str:
    origin_prompt = ""
    for dialog in task.get("dialogs", []):
        if dialog.get("role") == "user":
            origin_prompt = str(dialog.get("content", ""))
            break
    return f"""
GTA-2 held-out task id: {task_id}

Branch: {branch_name}

Retry mode: {retry_mode}

Original user task:
{origin_prompt}

Checkpoint tree:
{json.dumps(task.get("sub_tasks", []), ensure_ascii=False, indent=2)}

Baseline official score: {baseline_score}

Low-checkpoint diagnostics from the official evaluator:
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

Frozen route metadata:
{json.dumps(route, ensure_ascii=False, indent=2)}

Baseline artifact gate metadata:
{json.dumps({k: registry.get(k) for k in ["artifact_gate_decision", "artifact_score", "artifact_false_high", "failure_type"]}, ensure_ascii=False, indent=2)}

Expected deliverable filenames to materialize:
{json.dumps(expected_files, ensure_ascii=False, indent=2)}

Controller-executed tool evidence cache:
{tool_evidence or "No tool evidence cache was requested or available."}

Checkpoint-to-evidence map:
{json.dumps(checkpoint_evidence_map or {}, ensure_ascii=False, indent=2)}

Previous candidate feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

This is a retry branch that should improve the official root score.
If the feedback includes protected checkpoint damage, repair those checkpoints without sacrificing the target repair.
If the feedback is artifact-only, focus on the missing or invalid artifacts and keep the rest stable.

Return JSON with:
- final_answer_markdown: concise final answer that names every materialized deliverable.
- deliverables: object mapping each expected filename to complete content.
- preserve_units: evidence units that must not be rewritten.
- patches: patch-style edits over artifact units. Prefer add_section, update_section, add_or_update_section, or update_table_cells.
- do_not_modify: protected units such as csv.schema, figure captions, existing calculations, and protected evidence map units.
- repair_rationale: list of short reasons tied to checkpoint diagnostics, route modules, and branch feedback.

Rules:
- Do not claim a file exists unless it appears in deliverables.
- If a PDF/DOCX/PPTX/XLSX is expected, provide markdown/text/table content for that file; the runner will materialize the binary container.
- For target_retry, prioritize missing low-checkpoint evidence and artifact validity.
- For preservation_retry, keep the target gain, do not rewrite the whole artifact, and restore only the damaged protected evidence units.
""".strip()


def summarize_gate(report: dict[str, Any], min_artifact_score: float) -> dict[str, Any]:
    artifact_score = float(report.get("artifact_score") or 0)
    if report.get("false_high_score"):
        decision = "REJECT"
        reason = "official score is high but required artifacts are missing"
    elif artifact_score < min_artifact_score:
        decision = "REJECT"
        reason = f"artifact_score {artifact_score:.2f} below {min_artifact_score:.2f}"
    else:
        decision = "ACCEPT"
        reason = f"artifact_score {artifact_score:.2f} satisfies {min_artifact_score:.2f}"
    return {"decision": decision, "reason": reason, "artifact_score": artifact_score}


def candidate_metrics(
    baseline_score: float,
    score: float,
    diff: dict[str, Any],
    threshold: float,
    epsilon_cp: float,
    checkpoint_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = diff.get("rows") or []
    protected_rows = [row for row in rows if not row.get("was_low_before") and row.get("delta") is not None and row["delta"] < 0]
    protected_damage_score = 0.0
    max_single_damage = 0.0
    protected_damage_ids = []
    for row in protected_rows:
        delta = float(row["delta"])
        checkpoint_id = str(row.get("checkpoint_id") or "")
        weight = float((checkpoint_weights or {}).get(checkpoint_id, 1.0))
        damage = max(0.0, -delta - epsilon_cp)
        if damage <= 0:
            continue
        protected_damage_score += damage * weight
        max_single_damage = max(max_single_damage, damage)
        protected_damage_ids.append(row.get("checkpoint_id"))
    target_avg_delta = diff.get("target_avg_delta")
    target_gain = 0.0 if target_avg_delta is None else float(target_avg_delta)
    root_gain = float(score) - float(baseline_score)
    return {
        "root_after": float(score),
        "root_gain": root_gain,
        "target_gain": target_gain,
        "protected_damage_score": protected_damage_score,
        "max_single_damage": max_single_damage,
        "num_collateral_damage": diff.get("num_collateral_damage"),
        "collateral_damage_ids": diff.get("collateral_damage_ids", []),
        "protected_damage_ids": protected_damage_ids,
        "epsilon_cp": epsilon_cp,
    }


def select_root_primary(
    candidates: list[dict[str, Any]],
    baseline_score: float,
    selection_config: dict[str, Any],
    damage_aware: bool = True,
) -> dict[str, Any]:
    if not candidates:
        return make_baseline_candidate(baseline_score)
    epsilon_root = float(selection_config.get("epsilon_root", 0.0) or 0.0)
    root_best = max(float(c["root_after"]) for c in candidates)
    near_best = [c for c in candidates if root_best - float(c["root_after"]) <= epsilon_root]
    if damage_aware:
        selected = min(
            near_best,
            key=lambda c: (
                float(c.get("protected_damage_score", 0.0) or 0.0),
                float(c.get("max_single_damage", 0.0) or 0.0),
                -float(c.get("target_gain", 0.0) or 0.0),
                -float(c.get("root_after", 0.0) or 0.0),
            ),
        )
        rule = "root_primary_calibrated_tie_band"
    else:
        selected = max(
            candidates,
            key=lambda c: (
                float(c.get("root_after", 0.0) or 0.0),
                float(c.get("target_gain", 0.0) or 0.0),
            ),
        )
        rule = "root_primary_no_non_regression_ablation"
        near_best = [selected]
    if selected["root_after"] <= baseline_score:
        return make_baseline_candidate(baseline_score)
    selected = dict(selected)
    selected["near_best_pool_size"] = len(near_best)
    selected["selection_root_best"] = root_best
    selected["epsilon_root"] = epsilon_root
    selected["selection_rule"] = rule
    return selected


def make_baseline_candidate(baseline_score: float) -> dict[str, Any]:
    return {
        "attempt_index": 0,
        "label": "baseline",
        "artifact_pass": True,
        "root_after": float(baseline_score),
        "root_gain": 0.0,
        "target_gain": 0.0,
        "protected_damage_score": 0.0,
        "max_single_damage": 0.0,
        "num_collateral_damage": 0,
        "collateral_damage_ids": [],
        "protected_damage_ids": [],
        "utility": 0.0,
        "official_score": float(baseline_score),
        "artifact_gate_decision": "BASELINE",
        "artifact_score": 1.0,
        "artifact_false_high": False,
        "non_regression_decision": "BASELINE",
        "candidate_dir": "",
        "rescore_result": "",
        "checkpoint_diff": None,
        "artifact_report": None,
        "near_best_pool_size": 1,
        "selection_root_best": float(baseline_score),
        "epsilon_root": 0.0,
        "selection_rule": "baseline",
    }


def build_retry_feedback(
    attempt: dict[str, Any],
    threshold: float,
    include_preservation: bool,
    epsilon_cp: float,
    evidence_validation: dict[str, Any] | None = None,
    checkpoint_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if attempt.get("attempt_index") == 0 or attempt.get("artifact_report") is None:
        payload = {
            "artifact_feedback": {
                "decision": "BASELINE_RETAINED",
                "reason": "baseline currently dominates the Pareto frontier",
                "artifact_score": 1.0,
                "false_high_score": False,
                "expected_artifact_count": 0,
                "passed_artifact_count": 0,
                "partial_artifact_count": 0,
                "failed_checks": [],
            },
            "preservation_feedback": {
                "before_root_score": attempt.get("root_after"),
                "after_root_score": attempt.get("root_after"),
                "root_delta": 0.0,
                "target_avg_delta": 0.0,
                "non_target_avg_delta": 0.0,
                "num_collateral_damage": 0,
                "collateral_damage_ids": [],
                "protected_damage_score": 0.0,
                "max_single_damage": 0.0,
                "protected_damage_ids": [],
                "target_examples": [],
                "protected_examples": [],
            },
            "utility": attempt.get("utility", 0.0),
            "root_after": attempt.get("root_after"),
            "retry_mode": "target_retry",
            "evidence_validation": evidence_validation or {},
            "selection_hint": "baseline currently wins; only retry if you can produce a strictly better candidate without collateral damage",
        }
        if not include_preservation:
            payload["preservation_feedback"] = {
                "status": "omitted_for_ablation",
                "note": "This branch should not receive non-regression preservation guidance.",
            }
            payload["selection_hint"] = "focus on target repair and artifact validity only"
        return payload
    payload = {
        "artifact_feedback": summarize_artifact_feedback(attempt["artifact_report"], attempt["artifact_gate_decision"]),
        "utility": attempt.get("utility", 0.0),
        "root_after": attempt.get("root_after"),
        "evidence_validation": evidence_validation or attempt.get("evidence_validation", {}),
    }
    if include_preservation:
        preservation_feedback = summarize_preservation_feedback(
            attempt["checkpoint_diff"],
            threshold,
            epsilon_cp,
            checkpoint_weights=checkpoint_weights,
        )
        payload["preservation_feedback"] = preservation_feedback
        root_delta = float(preservation_feedback.get("root_delta") or 0.0)
        if preservation_feedback.get("protected_damage_score", 0.0) > 0 and root_delta > 0:
            payload["retry_mode"] = "preservation_retry"
        else:
            payload["retry_mode"] = "target_retry"
        if float(attempt.get("root_after") or 0.0) < QUALITY_FLOOR:
            payload["selection_hint"] = (
                "the candidate beats baseline but is still below the quality floor; improve the official root score, "
                "cover missing core deliverables/checkpoints, and preserve existing gains"
            )
        elif preservation_feedback.get("protected_damage_score", 0.0) > 0:
            payload["selection_hint"] = (
                "preservation_retry: keep the target gain, restore damaged protected checkpoints/evidence units, "
                "and avoid rewriting the whole artifact"
            )
        else:
            payload["selection_hint"] = (
                "target_retry: keep the current successful content, add missing low-checkpoint evidence, "
                "and improve the official root score without changing protected evidence units"
            )
    else:
        payload["preservation_feedback"] = {
            "status": "omitted_for_ablation",
            "note": "This branch should not receive non-regression preservation guidance.",
        }
        payload["retry_mode"] = "target_retry"
        payload["selection_hint"] = "focus on target repair and artifact validity only"
    repair_eval_report = attempt.get("repair_eval_report")
    if isinstance(repair_eval_report, dict):
        repair_feedback = repair_retry_feedback_from_report(
            repair_eval_report,
            include_preservation=include_preservation,
        )
        payload["repair_evaluator_feedback"] = repair_feedback
        if include_preservation:
            payload["retry_mode"] = repair_feedback.get("retry_mode", payload.get("retry_mode", "target_retry"))
            payload["selection_hint"] = repair_feedback.get("selection_hint", payload.get("selection_hint", ""))
    return payload


def compare_with_original(compare_rows_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_rows = {int(row["task_id"]): row for row in load_json(compare_rows_path)}
    out = []
    for row in rows:
        original = original_rows[int(row["task_id"])]
        out.append(
            {
                "task_id": row["task_id"],
                "original_full_ours_score": original.get("full_ours_score"),
                "original_without_non_regression_gate_score": original.get("without_non_regression_gate_score"),
                "pareto_full_ours_score": row.get("full_ours_score"),
                "pareto_without_artifact_gate_score": row.get("without_artifact_gate_score"),
                "pareto_without_non_regression_gate_score": row.get("without_non_regression_gate_score"),
                "delta_full_ours": None
                if original.get("full_ours_score") is None or row.get("full_ours_score") is None
                else float(row["full_ours_score"]) - float(original["full_ours_score"]),
                "delta_without_non_regression_gate": None
                if original.get("without_non_regression_gate_score") is None or row.get("without_non_regression_gate_score") is None
                else float(row["without_non_regression_gate_score"]) - float(original["without_non_regression_gate_score"]),
                "original_accept": original.get("full_ours_accept"),
                "pareto_accept": row.get("full_ours_accept"),
                "attempts_used": row.get("attempts_used"),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Full ours preservation search", "full_ours_score", "full_ours_accept"),
        ("Ours w/o artifact gate preservation search", "without_artifact_gate_score", "without_artifact_gate_accept"),
        ("Ours w/o non-regression preservation search", "without_non_regression_gate_score", "without_non_regression_gate_accept"),
        ("Ours w/o evidence map preservation search", "without_evidence_map_score", "without_evidence_map_accept"),
        ("Ours w/o patch repair preservation search", "without_patch_repair_score", "without_patch_repair_accept"),
    ]
    out = []
    for method, score_key, accept_key in specs:
        scores = [float(row[score_key]) for row in rows if row.get(score_key) is not None]
        out.append(
            {
                "method": method,
                "num_tasks": len(scores),
                "mean_score": None if not scores else mean(scores),
                "success_at_8": None if not scores else sum(score >= 8 for score in scores) / len(scores),
                "perfect_at_10": None if not scores else sum(score >= 10 for score in scores) / len(scores),
                "num_accept": sum(1 for row in rows if row.get(accept_key) is True or row.get("status") == "skip_high_baseline"),
            }
        )
    return out


def branch_needs_retry_full(
    best_valid: dict[str, Any],
    best_any: dict[str, Any],
    baseline_score: float,
    attempt_count: int,
) -> bool:
    if best_valid["attempt_index"] == 0:
        return True
    if best_any["root_after"] <= baseline_score:
        return True
    repair_report = best_any.get("repair_eval_report")
    if isinstance(repair_report, dict):
        if repair_report.get("retry_mode") == "preservation_retry":
            return True
        if not repair_report.get("target_gain_observed", False):
            return True
    if attempt_count < MIN_REFINEMENT_ATTEMPTS_AFTER_BASELINE:
        return True
    if best_any["root_after"] < QUALITY_FLOOR:
        return True
    if best_any["protected_damage_score"] > 0:
        return True
    return False


def branch_needs_retry_no_nr(
    best_valid: dict[str, Any],
    best_any: dict[str, Any],
    baseline_score: float,
    attempt_count: int,
) -> bool:
    if best_valid["attempt_index"] == 0:
        return True
    if best_any["root_after"] <= baseline_score:
        return True
    if attempt_count < MIN_REFINEMENT_ATTEMPTS_AFTER_BASELINE:
        return True
    if best_any["root_after"] < QUALITY_FLOOR:
        return True
    return False


def run_attempt(
    task_id: int,
    attempt_index: int,
    prompt: str,
    task_dir: Path,
    baseline_result: Path,
    baseline_prediction: Path,
    baseline_score: float,
    threshold: float,
    min_artifact_score: float,
    model: str,
    max_tokens: int,
    scorer_backend: str,
    scorer_model: str,
    selection_config: dict[str, Any],
    checkpoint_evidence_map: dict[str, Any] | None,
    enable_patch_repair: bool,
    checkpoint_weights: dict[str, float] | None,
) -> dict[str, Any]:
    attempt_dir = task_dir / f"attempt{attempt_index}"
    candidate_dir = attempt_dir / "candidate"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    payload = call_llm_json(prompt, model=model, max_tokens=max_tokens)
    dump_json(attempt_dir / "generation_payload.json", payload)
    (attempt_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")

    answer = str(payload.get("final_answer_markdown", "")).strip() or "Generated held-out20 repair candidate."
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "repaired_final_answer.md").write_text(answer, encoding="utf-8")

    expected_files = expected_files_from_prediction(baseline_prediction)
    protected_units = protected_units_from_evidence_map(checkpoint_evidence_map or {})
    materialized = materialize_candidate_files(
        candidate_dir,
        payload,
        expected_files,
        protected_units=protected_units,
        enable_patch_repair=enable_patch_repair,
    )
    created_files = materialized["created_files"]
    dump_json(attempt_dir / "materialized_files.json", created_files)
    patch_plan = load_json(Path(materialized["patch_plan_path"]))
    evidence_validation = validate_evidence_map(checkpoint_evidence_map or {}, materialized["artifact_spec"])
    dump_json(attempt_dir / "evidence_map_validation.json", evidence_validation)

    rescore = attempt_dir / "official_rescore_result.json"
    rescore_run = run_cmd(
        scorer_command(
            backend=scorer_backend,
            model=scorer_model,
            task_id=task_id,
            candidate_dir=candidate_dir,
            out=rescore,
        )
    )
    dump_json(attempt_dir / "official_rescore_run.json", rescore_run)
    if scorer_backend == "legacy" and (rescore_run["returncode"] != 0 or not rescore.exists()):
        rescore = baseline_result
    elif rescore_run["returncode"] != 0 or not rescore.exists():
        raise RuntimeError(f"GTA-2 scorer failed for task {task_id} attempt {attempt_index}. See {attempt_dir / 'official_rescore_run.json'}")

    artifact_report = verify_artifact_gate(baseline_prediction, rescore, 8.0, candidate_dir)
    dump_json(attempt_dir / "artifact_gate_report.json", artifact_report)
    artifact_gate = summarize_gate(artifact_report, min_artifact_score)
    dump_json(attempt_dir / "artifact_gate_decision.json", artifact_gate)

    baseline = load_json(baseline_result)
    after = load_json(rescore)
    diff = compare_scores(baseline, after, threshold=threshold, damage_threshold=float(selection_config["epsilon_cp"]))
    dump_json(attempt_dir / "checkpoint_diff.json", diff)

    metrics = candidate_metrics(
        baseline_score,
        read_score(rescore) or baseline_score,
        diff,
        threshold,
        float(selection_config["epsilon_cp"]),
        checkpoint_weights=checkpoint_weights,
    )
    repair_eval_report = evaluate_repair_candidate(
        task_id=task_id,
        checkpoint_tree=[],
        baseline_official_result=baseline,
        candidate_dir=candidate_dir,
        expected_files=expected_files,
        artifact_spec=materialized["artifact_spec"],
        patch_plan=patch_plan,
        evidence_map=checkpoint_evidence_map,
        artifact_validity_report=artifact_report,
        min_artifact_score=min_artifact_score,
        threshold=threshold,
    )
    dump_json(attempt_dir / "repair_eval_report.json", repair_eval_report)
    retry_feedback = repair_retry_feedback_from_report(
        repair_eval_report,
        include_preservation=checkpoint_evidence_map is not None,
    )
    dump_json(attempt_dir / "retry_feedback.json", retry_feedback)
    attempt = {
        "attempt_index": attempt_index,
        "candidate_dir": str(candidate_dir),
        "generation_payload": str(attempt_dir / "generation_payload.json"),
        "generation_prompt": str(attempt_dir / "generation_prompt.txt"),
        "official_rescore_result": str(rescore),
        "scorer_backend": scorer_backend,
        "scorer_model": scorer_model,
        "official_score": read_score(rescore),
        "artifact_gate_decision": artifact_gate,
        "artifact_report": artifact_report,
        "artifact_validity": repair_eval_report["artifact_validity"],
        "checkpoint_diff": diff,
        "artifact_pass": bool(repair_eval_report["artifact_pass"]),
        "patch_mode": materialized["patch_mode"],
        "artifact_spec": materialized["artifact_spec_path"],
        "patch_plan": materialized["patch_plan_path"],
        "repair_eval_report": repair_eval_report,
        "repair_eval_report_path": str(attempt_dir / "repair_eval_report.json"),
        "retry_feedback": retry_feedback,
        "retry_feedback_path": str(attempt_dir / "retry_feedback.json"),
        "target_checkpoint_status": repair_eval_report["target_checkpoint_status"],
        "protected_checkpoint_status": repair_eval_report["protected_checkpoint_status"],
        "candidate_selection_signals": repair_eval_report["candidate_selection_signals"],
        "violated_protected_units": materialized["violated_protected_units"],
        "unresolved_evidence_units": evidence_validation["unresolved_evidence_units"],
        "evidence_validation": evidence_validation,
        **metrics,
        "materialized_files": created_files,
    }
    return attempt


def select_branch_results(
    attempts: list[dict[str, Any]],
    baseline_score: float,
    selection_config: dict[str, Any],
    damage_aware: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_valid = select_candidate_by_repair_evaluator(
        attempts,
        baseline_score=baseline_score,
        require_artifact_pass=True,
        damage_aware=damage_aware,
        epsilon_root=float(selection_config["epsilon_root"]),
    )
    best_any = select_candidate_by_repair_evaluator(
        attempts,
        baseline_score=baseline_score,
        require_artifact_pass=False,
        damage_aware=damage_aware,
        epsilon_root=float(selection_config["epsilon_root"]),
    )
    return best_valid, best_any


def run_branch(
    *,
    task_id: int,
    branch_name: str,
    initial_attempt: dict[str, Any],
    task: dict[str, Any],
    baseline_score: float,
    diagnostics: list[dict[str, Any]],
    route: dict[str, Any],
    reg: dict[str, Any],
    expected_files: list[str],
    task_dir: Path,
    model: str,
    max_tokens: int,
    scorer_backend: str,
    scorer_model: str,
    threshold: float,
    min_artifact_score: float,
    include_preservation: bool,
    tool_evidence: str | None,
    selection_config: dict[str, Any],
    checkpoint_evidence_map: dict[str, Any] | None,
    enable_patch_repair: bool,
    checkpoint_weights: dict[str, float] | None,
) -> dict[str, Any]:
    branch_dir = task_dir / branch_name
    branch_dir.mkdir(parents=True, exist_ok=True)
    branch_evidence_map = checkpoint_evidence_map if include_preservation else None
    attempts = [initial_attempt]
    best_valid, best_any = select_branch_results(attempts, baseline_score, selection_config, damage_aware=include_preservation)
    for attempt_index in range(2, MAX_ATTEMPTS + 1):
        needs_retry = (
            branch_needs_retry_full(best_valid, best_any, baseline_score, len(attempts))
            if include_preservation
            else branch_needs_retry_no_nr(best_valid, best_any, baseline_score, len(attempts))
        )
        if not needs_retry:
            break
        feedback_source = best_any
        if include_preservation and best_any["attempt_index"] == 0:
            improved_damaged = [
                attempt
                for attempt in attempts
                if attempt.get("attempt_index") != 0
                and float(attempt.get("root_after") or 0.0) > baseline_score
                and float(attempt.get("protected_damage_score") or 0.0) > 0
            ]
            if improved_damaged:
                feedback_source = max(improved_damaged, key=lambda attempt: float(attempt.get("root_after") or 0.0))
        use_preservation_feedback = (
            include_preservation
            and feedback_source["attempt_index"] != 0
            and feedback_source["root_after"] > baseline_score
        )
        feedback = build_retry_feedback(
            feedback_source,
            threshold,
            include_preservation=use_preservation_feedback,
            epsilon_cp=float(selection_config["epsilon_cp"]),
            evidence_validation=feedback_source.get("evidence_validation"),
            checkpoint_weights=checkpoint_weights,
        )
        prompt = build_retry_prompt(
            task_id=task_id,
            task=task,
            baseline_score=float(baseline_score),
            diagnostics=diagnostics,
            route=route,
            registry=reg,
            expected_files=expected_files,
            feedback=feedback,
            branch_name=branch_name,
            tool_evidence=tool_evidence,
            checkpoint_evidence_map=branch_evidence_map,
            retry_mode=str(feedback.get("retry_mode") or "target_retry"),
        )
        attempt = run_attempt(
            task_id=task_id,
            attempt_index=attempt_index,
            prompt=prompt,
            task_dir=branch_dir,
            baseline_result=Path(reg["baseline_result"]).resolve(),
            baseline_prediction=Path(reg["baseline_prediction"]).resolve(),
            baseline_score=baseline_score,
            threshold=threshold,
            min_artifact_score=min_artifact_score,
            model=model,
            max_tokens=max_tokens,
            scorer_backend=scorer_backend,
            scorer_model=scorer_model,
            selection_config=selection_config,
            checkpoint_evidence_map=branch_evidence_map,
            enable_patch_repair=enable_patch_repair,
            checkpoint_weights=checkpoint_weights,
        )
        attempt["retry_mode"] = feedback.get("retry_mode", "target_retry")
        attempts.append(attempt)
        dump_json(branch_dir / "attempts.json", attempts)
        best_valid, best_any = select_branch_results(attempts, baseline_score, selection_config, damage_aware=include_preservation)

    dump_json(branch_dir / "attempts.json", attempts)
    dump_json(
        branch_dir / "candidate_selection_trace.json",
        {
            "branch_name": branch_name,
            "damage_aware": include_preservation,
            "best_valid_attempt_index": best_valid.get("attempt_index"),
            "best_any_attempt_index": best_any.get("attempt_index"),
            "best_valid_selection_rule": best_valid.get("selection_rule"),
            "best_any_selection_rule": best_any.get("selection_rule"),
            "best_valid_selection_trace": best_valid.get("selection_trace"),
            "best_any_selection_trace": best_any.get("selection_trace"),
            "attempt_signals": [
                {
                    "attempt_index": attempt.get("attempt_index"),
                    "root_after": attempt.get("root_after"),
                    "artifact_pass": attempt.get("artifact_pass"),
                    "patch_mode": attempt.get("patch_mode"),
                    "repair_eval_report": attempt.get("repair_eval_report_path"),
                    "candidate_selection_signals": attempt.get("candidate_selection_signals"),
                }
                for attempt in attempts
            ],
        },
    )
    return {
        "branch_name": branch_name,
        "attempts": attempts,
        "best_valid": best_valid,
        "best_any": best_any,
    }


def run_task(
    task_id: int,
    reg: dict[str, Any],
    route: dict[str, Any],
    out_dir: Path,
    model: str,
    max_tokens: int,
    scorer_backend: str,
    scorer_model: str,
    threshold: float,
    min_artifact_score: float,
    enable_tool_planner: bool,
    selection_config: dict[str, Any],
    run_extra_ablations: bool,
) -> dict[str, Any]:
    task_dir = out_dir / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    baseline_prediction = Path(reg["baseline_prediction"]).resolve()
    baseline_result = resolve_baseline_result(
        task_id=task_id,
        reg=reg,
        task_dir=task_dir,
        scorer_backend=scorer_backend,
        scorer_model=scorer_model,
    )
    baseline_score = read_score(baseline_result) or 0.0
    baseline = load_json(baseline_result)
    task = load_source_task(task_id)
    checkpoint_weights = checkpoint_weight_map(task)
    diagnostics = low_checkpoint_diagnostics(baseline, threshold)
    expected_files = expected_files_from_prediction(baseline_prediction)
    tool_evidence = None
    if enable_tool_planner:
        resources = resource_map(task, DEFAULT_SOURCE_DATASET.parent)
        plan_prompt = build_tool_plan_prompt(
            task_id=task_id,
            task=task,
            diagnostics=diagnostics,
            route=route,
            expected_files=expected_files,
            available_resources=sorted(set(resources)),
        )
        (task_dir / "tool_repair_plan_prompt.txt").write_text(plan_prompt, encoding="utf-8")
        plan = call_tool_planner(llm_json=call_llm_json, prompt=plan_prompt, model=model)
        evidence = execute_tool_plan(plan=plan, task=task, dataset_dir=DEFAULT_SOURCE_DATASET.parent, out_dir=task_dir)
        tool_evidence = format_evidence_for_prompt(evidence)

    checkpoint_evidence_map = repair_build_checkpoint_evidence_map(
        checkpoint_tree=task.get("sub_tasks", []),
        baseline_official_result=baseline,
        expected_files=expected_files,
        threshold=threshold,
        route=route,
        tool_evidence=tool_evidence,
    )
    dump_json(task_dir / "checkpoint_evidence_map.json", checkpoint_evidence_map)

    initial_prompt = build_generation_prompt(
        task_id,
        task,
        float(baseline_score),
        diagnostics,
        route,
        reg,
        expected_files,
        tool_evidence=tool_evidence,
        checkpoint_evidence_map=checkpoint_evidence_map,
    )
    initial_attempt = run_attempt(
        task_id=task_id,
        attempt_index=1,
        prompt=initial_prompt,
        task_dir=task_dir / "shared_initial",
        baseline_result=baseline_result,
        baseline_prediction=baseline_prediction,
        baseline_score=baseline_score,
        threshold=threshold,
        min_artifact_score=min_artifact_score,
        model=model,
        max_tokens=max_tokens,
        scorer_backend=scorer_backend,
        scorer_model=scorer_model,
        selection_config=selection_config,
        checkpoint_evidence_map=checkpoint_evidence_map,
        enable_patch_repair=True,
        checkpoint_weights=checkpoint_weights,
    )
    dump_json(task_dir / "shared_initial" / "attempt.json", initial_attempt)

    full_branch = run_branch(
        task_id=task_id,
        branch_name="full_ours",
        initial_attempt=initial_attempt,
        task=task,
        baseline_score=baseline_score,
        diagnostics=diagnostics,
        route=route,
        reg=reg,
        expected_files=expected_files,
        task_dir=task_dir,
        model=model,
        max_tokens=max_tokens,
        scorer_backend=scorer_backend,
        scorer_model=scorer_model,
        threshold=threshold,
        min_artifact_score=min_artifact_score,
        include_preservation=True,
        tool_evidence=tool_evidence,
        selection_config=selection_config,
        checkpoint_evidence_map=checkpoint_evidence_map,
        enable_patch_repair=True,
        checkpoint_weights=checkpoint_weights,
    )
    no_nr_branch = run_branch(
        task_id=task_id,
        branch_name="without_non_regression",
        initial_attempt=initial_attempt,
        task=task,
        baseline_score=baseline_score,
        diagnostics=diagnostics,
        route=route,
        reg=reg,
        expected_files=expected_files,
        task_dir=task_dir,
        model=model,
        max_tokens=max_tokens,
        scorer_backend=scorer_backend,
        scorer_model=scorer_model,
        threshold=threshold,
        min_artifact_score=min_artifact_score,
        include_preservation=False,
        tool_evidence=tool_evidence,
        selection_config=selection_config,
        checkpoint_evidence_map=checkpoint_evidence_map,
        enable_patch_repair=True,
        checkpoint_weights=checkpoint_weights,
    )

    no_evidence_branch = None
    no_patch_branch = None
    if run_extra_ablations:
        no_evidence_prompt = build_generation_prompt(
            task_id,
            task,
            float(baseline_score),
            diagnostics,
            route,
            reg,
            expected_files,
            tool_evidence=tool_evidence,
            checkpoint_evidence_map=None,
        )
        no_evidence_initial = run_attempt(
            task_id=task_id,
            attempt_index=1,
            prompt=no_evidence_prompt,
            task_dir=task_dir / "without_evidence_map_initial",
            baseline_result=baseline_result,
            baseline_prediction=baseline_prediction,
            baseline_score=baseline_score,
            threshold=threshold,
            min_artifact_score=min_artifact_score,
            model=model,
            max_tokens=max_tokens,
            scorer_backend=scorer_backend,
            scorer_model=scorer_model,
            selection_config=selection_config,
            checkpoint_evidence_map=None,
            enable_patch_repair=True,
            checkpoint_weights=checkpoint_weights,
        )
        no_evidence_branch = run_branch(
            task_id=task_id,
            branch_name="without_evidence_map",
            initial_attempt=no_evidence_initial,
            task=task,
            baseline_score=baseline_score,
            diagnostics=diagnostics,
            route=route,
            reg=reg,
            expected_files=expected_files,
            task_dir=task_dir,
            model=model,
            max_tokens=max_tokens,
            scorer_backend=scorer_backend,
            scorer_model=scorer_model,
            threshold=threshold,
            min_artifact_score=min_artifact_score,
            include_preservation=True,
            tool_evidence=tool_evidence,
            selection_config=selection_config,
            checkpoint_evidence_map=None,
            enable_patch_repair=True,
            checkpoint_weights=checkpoint_weights,
        )

        no_patch_initial = run_attempt(
            task_id=task_id,
            attempt_index=1,
            prompt=initial_prompt,
            task_dir=task_dir / "without_patch_repair_initial",
            baseline_result=baseline_result,
            baseline_prediction=baseline_prediction,
            baseline_score=baseline_score,
            threshold=threshold,
            min_artifact_score=min_artifact_score,
            model=model,
            max_tokens=max_tokens,
            scorer_backend=scorer_backend,
            scorer_model=scorer_model,
            selection_config=selection_config,
            checkpoint_evidence_map=checkpoint_evidence_map,
            enable_patch_repair=False,
            checkpoint_weights=checkpoint_weights,
        )
        no_patch_branch = run_branch(
            task_id=task_id,
            branch_name="without_patch_repair",
            initial_attempt=no_patch_initial,
            task=task,
            baseline_score=baseline_score,
            diagnostics=diagnostics,
            route=route,
            reg=reg,
            expected_files=expected_files,
            task_dir=task_dir,
            model=model,
            max_tokens=max_tokens,
            scorer_backend=scorer_backend,
            scorer_model=scorer_model,
            threshold=threshold,
            min_artifact_score=min_artifact_score,
            include_preservation=True,
            tool_evidence=tool_evidence,
            selection_config=selection_config,
            checkpoint_evidence_map=checkpoint_evidence_map,
            enable_patch_repair=False,
            checkpoint_weights=checkpoint_weights,
        )

    full_attempts = full_branch["attempts"]
    no_nr_attempts = no_nr_branch["attempts"]
    full_valid = full_branch["best_valid"]
    full_any = full_branch["best_any"]
    no_nr_valid = no_nr_branch["best_valid"]
    no_nr_any = no_nr_branch["best_any"]

    full_selected = full_valid
    artifact_free_selected = full_any
    non_regression_free_selected = no_nr_valid
    no_evidence_selected = no_evidence_branch["best_valid"] if no_evidence_branch else None
    no_patch_selected = no_patch_branch["best_valid"] if no_patch_branch else None
    retry_mode_counts: dict[str, int] = {}
    for attempt in full_attempts:
        for key in [attempt.get("retry_mode")]:
            if key:
                retry_mode_counts[str(key)] = retry_mode_counts.get(str(key), 0) + 1

    row = {
        "task_id": task_id,
        "status": "preservation_search_retry",
        "scorer_backend": scorer_backend,
        "scorer_model": scorer_model,
        "baseline_result": str(baseline_result),
        "baseline_prediction": str(baseline_prediction),
        "baseline_score": baseline_score,
        "final_score": full_selected["root_after"],
        "full_ours_score": full_selected["root_after"],
        "without_artifact_gate_score": artifact_free_selected["root_after"],
        "without_non_regression_gate_score": non_regression_free_selected["root_after"],
        "without_evidence_map_score": None if no_evidence_selected is None else no_evidence_selected["root_after"],
        "without_patch_repair_score": None if no_patch_selected is None else no_patch_selected["root_after"],
        "full_ours_accept": full_selected["attempt_index"] != 0,
        "without_artifact_gate_accept": artifact_free_selected["attempt_index"] != 0,
        "without_non_regression_gate_accept": non_regression_free_selected["attempt_index"] != 0,
        "without_evidence_map_accept": None if no_evidence_selected is None else no_evidence_selected["attempt_index"] != 0,
        "without_patch_repair_accept": None if no_patch_selected is None else no_patch_selected["attempt_index"] != 0,
        "artifact_gate_decision": "ACCEPT" if full_selected["attempt_index"] != 0 else "BASELINE_RETAINED",
        "artifact_score": None if full_selected["attempt_index"] == 0 else full_selected["artifact_gate_decision"]["artifact_score"],
        "artifact_false_high": False if full_selected["attempt_index"] == 0 else full_selected["artifact_report"].get("false_high_score"),
        "non_regression_decision": "ACCEPT" if full_selected["attempt_index"] != 0 else "BASELINE_RETAINED",
        "num_collateral_damage": None if full_selected["attempt_index"] == 0 else full_selected["num_collateral_damage"],
        "protected_damage_score": full_selected.get("protected_damage_score", 0.0),
        "max_single_damage": full_selected.get("max_single_damage", 0.0),
        "near_best_pool_size": full_selected.get("near_best_pool_size", 1),
        "epsilon_root": selection_config["epsilon_root"],
        "epsilon_cp": selection_config["epsilon_cp"],
        "selection_config_source": selection_config["selection_config_source"],
        "selection_rule": full_selected.get("selection_rule"),
        "selection_trace": full_selected.get("selection_trace"),
        "checkpoint_weights": checkpoint_weights,
        "retry_mode_counts": retry_mode_counts,
        "patch_mode": full_selected.get("patch_mode", "baseline"),
        "checkpoint_evidence_map": str(task_dir / "checkpoint_evidence_map.json"),
        "violated_protected_units": full_selected.get("violated_protected_units", []),
        "unresolved_evidence_units": full_selected.get("unresolved_evidence_units", []),
        "attempts_used": len(full_attempts),
        "selected_attempt_index": full_selected["attempt_index"],
        "artifact_free_selected_attempt_index": artifact_free_selected["attempt_index"],
        "non_regression_free_selected_attempt_index": non_regression_free_selected["attempt_index"],
        "attempt_scores": [a["root_after"] for a in full_attempts],
        "attempt_utilities": [a.get("utility", 0.0) for a in full_attempts],
        "attempt_protected_damage_scores": [a.get("protected_damage_score", 0.0) for a in full_attempts],
        "attempt_artifact_decisions": [a["artifact_gate_decision"]["decision"] for a in full_attempts],
        "candidate_dir": full_selected.get("candidate_dir", ""),
        "rescore_result": full_selected.get("official_rescore_result", ""),
        "route_name": route.get("route_name"),
        "routed_modules": route.get("routed_modules"),
        "expected_files": expected_files,
        "tool_planner_enabled": enable_tool_planner,
        "tool_evidence_cache": str(task_dir / "tool_evidence_cache.json") if enable_tool_planner else "",
        "full_branch_attempts": len(full_attempts),
        "no_non_regression_branch_attempts": len(no_nr_attempts),
        "no_evidence_map_branch_attempts": None if no_evidence_branch is None else len(no_evidence_branch["attempts"]),
        "no_patch_repair_branch_attempts": None if no_patch_branch is None else len(no_patch_branch["attempts"]),
        "full_branch_best_any_score": full_any["root_after"],
        "no_non_regression_branch_best_any_score": no_nr_any["root_after"],
        "run_extra_ablations": run_extra_ablations,
    }
    dump_json(task_dir / "selected_attempt.json", full_selected)
    dump_json(task_dir / "full_ours_branch.json", full_branch)
    dump_json(task_dir / "without_non_regression_branch.json", no_nr_branch)
    if no_evidence_branch:
        dump_json(task_dir / "without_evidence_map_branch.json", no_evidence_branch)
    if no_patch_branch:
        dump_json(task_dir / "without_patch_repair_branch.json", no_patch_branch)
    dump_json(task_dir / "row.json", row)
    return row


def main() -> None:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Required experiment KB is missing: {KB_PATH}")
    kb_text = KB_PATH.read_text(encoding="utf-8")
    if "Version v3.1 uses this logic" not in kb_text:
        raise RuntimeError(f"Unexpected v3.1 KB contents: {KB_PATH}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--task-ids", nargs="*", type=int, default=DEFAULT_TASK_IDS)
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--scorer-backend", choices=["legacy", "gemini", "responses"], default="legacy")
    parser.add_argument("--scorer-model", default="gemini-2.5-pro")
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--min-artifact-score", type=float, default=1.0)
    parser.add_argument("--enable-tool-planner", action="store_true", help="Ask the model for an allowlisted local tool plan and inject executed evidence into repair prompts.")
    parser.add_argument("--selection-config", type=Path, default=None, help="JSON file with calibrated epsilon_root and epsilon_cp. Missing file uses marked fallback values.")
    parser.add_argument("--allow-fallback-selection-config", action="store_true", help="Allow exploratory fallback epsilon values when no calibrated selection config is available.")
    parser.add_argument("--run-extra-ablations", action="store_true", help="Also run w/o evidence-map and w/o patch-repair branches.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--compare-with", type=Path, default=DEFAULT_COMPARE_ROWS)
    args = parser.parse_args()

    regs = registry_rows(args.registry)
    routes = route_rows(args.routes)
    selection_config = load_selection_config(args.selection_config, allow_fallback=args.allow_fallback_selection_config)
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        out_dir / "experiment_kb.json",
        {
            "kb_path": str(KB_PATH),
            "kb_version": "v3.1",
            "kb_excerpt": kb_text[:4000],
            "selection_config": selection_config,
            "run_extra_ablations": args.run_extra_ablations,
            "generation_model": args.model,
            "scorer_backend": args.scorer_backend,
            "scorer_model": args.scorer_model,
        },
    )

    rows = []
    run_error: Exception | None = None
    total_tasks = len(args.task_ids)
    try:
        for index, task_id in enumerate(args.task_ids, 1):
            print(f"[{index}/{total_tasks}] start task {task_id}", flush=True)
            row = run_task(
                task_id,
                regs[task_id],
                routes[task_id],
                out_dir,
                args.model,
                args.max_tokens,
                args.scorer_backend,
                args.scorer_model,
                args.threshold,
                args.min_artifact_score,
                args.enable_tool_planner,
                selection_config,
                args.run_extra_ablations,
            )
            rows.append(row)
            print(
                f"[{index}/{total_tasks}] done task {task_id} "
                f"baseline={row.get('baseline_score')} full={row.get('full_ours_score')} "
                f"no_artifact={row.get('without_artifact_gate_score')} "
                f"no_nonreg={row.get('without_non_regression_gate_score')} "
                f"attempts={row.get('attempts_used')} selected={row.get('selected_attempt_index')} "
                f"artifact={row.get('artifact_gate_decision')} nonreg={row.get('non_regression_decision')}",
                flush=True,
            )
            write_csv(out_dir / "heldout20_pareto_preservation_rows.csv", rows)
            dump_json(out_dir / "heldout20_pareto_preservation_rows.json", rows)
    except Exception as exc:
        run_error = exc
    finally:
        summaries = summarize(rows)
        comparison = compare_with_original(args.compare_with, rows)
        write_csv(out_dir / "heldout20_pareto_preservation_method_summary.csv", summaries)
        dump_json(out_dir / "heldout20_pareto_preservation_method_summary.json", summaries)
        write_csv(out_dir / "heldout20_pareto_preservation_comparison.csv", comparison)
        dump_json(out_dir / "heldout20_pareto_preservation_comparison.json", comparison)
        summary = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "out_dir": str(out_dir),
            "registry": str(args.registry),
            "routes": str(args.routes),
            "compare_with": str(args.compare_with),
            "generation_model": args.model,
            "scorer_backend": args.scorer_backend,
            "scorer_model": args.scorer_model,
            "num_tasks": len(rows),
            "task_ids": args.task_ids,
            "completed_task_ids": [row["task_id"] for row in rows],
            "status": "partial" if run_error is not None else "complete",
            "error": None
            if run_error is None
            else {
                "type": type(run_error).__name__,
                "message": str(run_error),
            },
            "method_summary": summaries,
            "note": "Root-primary preservation search with artifact gate hard and non-regression as retry feedback on held-out20.",
        }
        dump_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
