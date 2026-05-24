"""Preservation-aware repair evaluator for GTA-2 repair loops.

This module is an internal repair controller. It does not replace the GTA-2
official scorer and it does not freely assign 0-10 checkpoint scores. Instead
it emits stable process signals: artifact validity, evidence-unit changes,
patch violations, protected-unit risk, coarse checkpoint status, retry mode,
and candidate-selection features.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable


TARGET_LOW = "target_low"
PROTECTED_HIGH = "protected_high"
NEUTRAL = "neutral"

COARSE_STATUSES = {
    "fully_satisfied",
    "mostly_satisfied",
    "partially_satisfied",
    "minimally_satisfied",
    "not_satisfied",
    "insufficient_evidence",
    "unresolved",
    "evidence_added",
    "protected_evidence_unchanged",
    "protected_evidence_present",
    "risk_detected",
}
GOOD_TARGET_STATUSES = {"fully_satisfied", "mostly_satisfied", "evidence_added"}
TARGET_RETRY_STATUSES = {"partially_satisfied", "minimally_satisfied", "not_satisfied", "insufficient_evidence", "unresolved"}
RISK_RANK = {"low": 0, "medium": 1, "high": 2}


StatusJudge = Callable[[dict[str, Any], list[dict[str, Any]], str], dict[str, Any] | str]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_unit_id(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "generated"


def _content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _choose_artifact(expected_files: list[str], preferred: tuple[str, ...]) -> str:
    for name in expected_files:
        if Path(name).suffix.lower() in preferred:
            return Path(name).name
    return Path(expected_files[0]).name if expected_files else "repaired_final_answer.md"


def _flatten_checkpoint_tree(checkpoint_tree: Any) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        cp_id = str(node.get("id") or node.get("checkpoint_id") or "").strip()
        if cp_id:
            nodes[cp_id] = node
        for child in node.get("sub_tasks") or node.get("children") or []:
            walk(child)

    walk(checkpoint_tree)
    return nodes


def flatten_official_result_nodes(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for sample in result.get("details") or []:
        if not isinstance(sample, dict):
            continue
        for node in sample.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            cp_id = str(node.get("id") or node.get("checkpoint_id") or "").strip()
            if not cp_id:
                continue
            score = node.get("score")
            nodes[cp_id] = {
                "score": float(score) if isinstance(score, (int, float)) else None,
                "analysis": str(node.get("analysis") or ""),
            }
    return nodes


def evidence_units_for_checkpoint(cp_id: str, text: str, expected_files: list[str]) -> list[dict[str, Any]]:
    lowered = text.lower()
    units: list[dict[str, Any]] = []
    if any(token in lowered for token in ["csv", "table", "spreadsheet", "xlsx", "record"]):
        artifact = _choose_artifact(expected_files, (".csv", ".xlsx"))
        units.append({"artifact": artifact, "unit_id": "csv.schema", "type": "csv_schema", "reason": "checkpoint mentions table/csv evidence"})
        units.append({"artifact": artifact, "unit_id": "table.generated", "type": "table", "reason": "checkpoint mentions table/csv evidence"})
    if any(token in lowered for token in ["ocr", "text extraction", "extract", "abstract", "methodology", "key result"]):
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html")), "unit_id": "section.ocr_evidence", "type": "section", "reason": "checkpoint mentions extraction evidence"})
    if any(token in lowered for token in ["visual", "image", "figure", "chart", "comparison", "illustration", "cover"]):
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html", ".png", ".jpg", ".jpeg", ".pptx")), "unit_id": "section.visual_comparison", "type": "section", "reason": "checkpoint mentions visual/image evidence"})
    if any(token in lowered for token in ["source", "reference", "citation", "ground", "url", "bibliographic"]):
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html")), "unit_id": "section.source_appendix", "type": "section", "reason": "checkpoint mentions source grounding"})
    if any(token in lowered for token in ["method", "calculation", "formula", "ranked", "metric"]):
        units.append({"artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html", ".xlsx", ".csv")), "unit_id": "section.methods", "type": "section", "reason": "checkpoint mentions methods/calculation"})
    if any(token in lowered for token in ["file", "deliverable", "final answer", "named", "download"]):
        units.append({"artifact": "repaired_final_answer.md", "unit_id": "final_answer.deliverables", "type": "final_answer", "reason": "checkpoint mentions final deliverable listing"})
    if not units:
        units.append(
            {
                "artifact": _choose_artifact(expected_files, (".pdf", ".docx", ".html", ".csv", ".xlsx", ".pptx")),
                "unit_id": f"section.checkpoint_{_safe_unit_id(cp_id)}",
                "type": "section",
                "reason": "fallback checkpoint-local section",
            }
        )
    return _dedupe_units(units)


def _dedupe_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for unit in units:
        key = (Path(str(unit.get("artifact") or "")).name, str(unit.get("unit_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(unit)
    return out


def build_checkpoint_evidence_map(
    *,
    checkpoint_tree: Any,
    baseline_official_result: dict[str, Any],
    expected_files: list[str],
    threshold: float = 7.0,
    protected_threshold: float | None = None,
    route: dict[str, Any] | None = None,
    tool_evidence: str | None = None,
) -> dict[str, Any]:
    protected_cutoff = threshold if protected_threshold is None else protected_threshold
    tree_nodes = _flatten_checkpoint_tree(checkpoint_tree)
    score_nodes = flatten_official_result_nodes(baseline_official_result)
    cp_ids = sorted(set(tree_nodes) | set(score_nodes))
    entries: dict[str, Any] = {}
    for cp_id in cp_ids:
        tree_node = tree_nodes.get(cp_id, {})
        score_node = score_nodes.get(cp_id, {})
        score = score_node.get("score")
        if isinstance(score, (int, float)) and score < threshold:
            status = TARGET_LOW
        elif isinstance(score, (int, float)) and score >= protected_cutoff:
            status = PROTECTED_HIGH
        else:
            status = NEUTRAL
        requirement = str(tree_node.get("requirements") or tree_node.get("requirement") or tree_node.get("description") or "")
        analysis = str(score_node.get("analysis") or "")
        entries[cp_id] = {
            "checkpoint_id": cp_id,
            "status": status,
            "baseline_score": score,
            "requirements": requirement,
            "analysis_excerpt": analysis[:700],
            "evidence_units": evidence_units_for_checkpoint(cp_id, f"{cp_id}\n{requirement}\n{analysis}", expected_files),
        }
    return {
        "map_version": "repair_evaluator_v1",
        "threshold": threshold,
        "protected_threshold": protected_cutoff,
        "route_name": (route or {}).get("route_name"),
        "routed_modules": (route or {}).get("routed_modules"),
        "expected_files": [Path(name).name for name in expected_files],
        "tool_evidence_available": bool(tool_evidence and "No tool evidence" not in tool_evidence),
        "checkpoints": entries,
    }


def protected_units_from_evidence_map(evidence_map: dict[str, Any]) -> list[str]:
    units: set[str] = set()
    for cp in (evidence_map.get("checkpoints") or {}).values():
        if cp.get("status") != PROTECTED_HIGH:
            continue
        for unit in cp.get("evidence_units") or []:
            unit_id = str(unit.get("unit_id") or "")
            artifact = Path(str(unit.get("artifact") or "")).name
            if not unit_id:
                continue
            units.add(unit_id)
            if artifact:
                units.add(f"{artifact}:{unit_id}")
    return sorted(units)


def _artifact_lookup(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {Path(str(name)).name: artifact for name, artifact in (spec.get("artifacts") or {}).items() if isinstance(artifact, dict)}


def _unit_lookup(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for artifact_name, artifact in _artifact_lookup(spec).items():
        for unit in artifact.get("units") or []:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "")
            if not unit_id:
                continue
            row = {
                "artifact": artifact_name,
                "unit_id": unit_id,
                "type": unit.get("type"),
                "title": unit.get("title"),
                "content": str(unit.get("content") or ""),
                "hash": _content_hash(str(unit.get("content") or "")),
            }
            out.setdefault(unit_id, row)
            out[f"{artifact_name}:{unit_id}"] = row
        schema = artifact.get("schema") or []
        if schema:
            row = {
                "artifact": artifact_name,
                "unit_id": "csv.schema",
                "type": "csv_schema",
                "title": "CSV Schema",
                "content": ",".join(str(item) for item in schema),
                "hash": _content_hash(",".join(str(item) for item in schema)),
                "schema": [str(item) for item in schema],
            }
            out.setdefault("csv.schema", row)
            out[f"{artifact_name}:csv.schema"] = row
    return out


def _resolve_unit(spec: dict[str, Any], artifact: str, unit_id: str) -> dict[str, Any] | None:
    lookup = _unit_lookup(spec)
    artifact_name = Path(artifact).name
    if artifact_name and f"{artifact_name}:{unit_id}" in lookup:
        return lookup[f"{artifact_name}:{unit_id}"]
    return lookup.get(unit_id)


def _unit_key(artifact: str, unit_id: str) -> str:
    artifact_name = Path(artifact).name
    return f"{artifact_name}:{unit_id}" if artifact_name else unit_id


def _artifact_validity(
    *,
    candidate_dir: Path | None,
    expected_files: list[str],
    artifact_validity_report: dict[str, Any] | None,
    min_artifact_score: float,
) -> dict[str, Any]:
    fail_reasons: list[str] = []
    if artifact_validity_report:
        if artifact_validity_report.get("false_high_score"):
            fail_reasons.append("official score is high but required artifacts are missing")
        for check in artifact_validity_report.get("checks") or []:
            if isinstance(check, dict) and not check.get("passed"):
                fail_reasons.append(str(check.get("why_failed") or check.get("reason") or check.get("expected_name") or "artifact check failed"))
        artifact_score = artifact_validity_report.get("artifact_score")
        if isinstance(artifact_score, (int, float)) and float(artifact_score) < min_artifact_score:
            fail_reasons.append(f"artifact_score {float(artifact_score):.2f} below {min_artifact_score:.2f}")
        return {
            "artifact_pass": not fail_reasons,
            "artifact_fail_reasons": fail_reasons,
            "artifact_score": artifact_score,
            "source": "artifact_validity_report",
        }

    if candidate_dir is not None:
        for name in expected_files:
            target = candidate_dir / Path(name).name
            if not target.exists():
                fail_reasons.append(f"missing expected artifact: {Path(name).name}")
            elif target.is_file() and target.stat().st_size <= 0:
                fail_reasons.append(f"empty expected artifact: {Path(name).name}")
    return {
        "artifact_pass": not fail_reasons,
        "artifact_fail_reasons": fail_reasons,
        "artifact_score": 1.0 if not fail_reasons else 0.0,
        "source": "candidate_dir_expected_files",
    }


def _normalize_status(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("status") or value.get("verdict")
    status = str(value or "").strip().lower()
    status = status.replace("-", "_").replace(" ", "_")
    return status if status in COARSE_STATUSES else "unresolved"


def _evidence_text(units: list[dict[str, Any]]) -> str:
    parts = []
    for unit in units:
        content = str(unit.get("content") or "").strip()
        if content:
            parts.append(f"[{unit.get('artifact')}:{unit.get('unit_id')}]\n{content[:2000]}")
    return "\n\n".join(parts)


def _patch_violations(patch_plan: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in patch_plan.get("violated_protected_units") or []:
        if isinstance(row, dict):
            violations.append(row)
        else:
            violations.append({"unit": str(row), "reason": "protected unit violation"})
    for row in patch_plan.get("patches") or []:
        if not isinstance(row, dict):
            continue
        if row.get("status") not in {None, "applied"}:
            violations.append(
                {
                    "patch_index": row.get("index"),
                    "reason": row.get("reason") or f"patch {row.get('status')}",
                    "patch": row.get("patch"),
                }
            )
    return violations


def _schema_changes(
    baseline_artifact_spec: dict[str, Any] | None,
    artifact_spec: dict[str, Any],
    evidence_map: dict[str, Any],
) -> list[dict[str, Any]]:
    if not baseline_artifact_spec:
        return []
    changes = []
    baseline_artifacts = _artifact_lookup(baseline_artifact_spec)
    candidate_artifacts = _artifact_lookup(artifact_spec)
    protected_artifacts = {
        Path(str(unit.get("artifact") or "")).name
        for cp in (evidence_map.get("checkpoints") or {}).values()
        if cp.get("status") == PROTECTED_HIGH
        for unit in cp.get("evidence_units") or []
        if unit.get("type") in {"csv_schema", "table"} or unit.get("unit_id") == "csv.schema"
    }
    for artifact_name in protected_artifacts:
        if not artifact_name:
            continue
        before = [str(item) for item in (baseline_artifacts.get(artifact_name) or {}).get("schema") or []]
        after = [str(item) for item in (candidate_artifacts.get(artifact_name) or {}).get("schema") or []]
        if before and before != after:
            changes.append({"artifact": artifact_name, "before_schema": before, "after_schema": after})
    return changes


def evaluate_candidate(
    *,
    task_id: int,
    checkpoint_tree: Any,
    baseline_official_result: dict[str, Any],
    candidate_dir: Path | str | None,
    expected_files: list[str],
    artifact_spec: dict[str, Any],
    patch_plan: dict[str, Any] | None = None,
    evidence_map: dict[str, Any] | None = None,
    baseline_artifact_spec: dict[str, Any] | None = None,
    artifact_validity_report: dict[str, Any] | None = None,
    min_artifact_score: float = 1.0,
    threshold: float = 7.0,
    protected_threshold: float | None = None,
    route: dict[str, Any] | None = None,
    tool_evidence: str | None = None,
    status_judge: StatusJudge | None = None,
) -> dict[str, Any]:
    candidate_path = Path(candidate_dir).resolve() if candidate_dir else None
    patch_plan = patch_plan or {}
    evidence_map = evidence_map or build_checkpoint_evidence_map(
        checkpoint_tree=checkpoint_tree,
        baseline_official_result=baseline_official_result,
        expected_files=expected_files,
        threshold=threshold,
        protected_threshold=protected_threshold,
        route=route,
        tool_evidence=tool_evidence,
    )
    artifact_validity = _artifact_validity(
        candidate_dir=candidate_path,
        expected_files=expected_files,
        artifact_validity_report=artifact_validity_report,
        min_artifact_score=min_artifact_score,
    )

    target_status: dict[str, str] = {}
    protected_status: dict[str, str] = {}
    evidence_unit_changes: list[dict[str, Any]] = []
    protected_hash_changed: list[dict[str, Any]] = []
    section_deleted: list[dict[str, Any]] = []

    patch_violations = _patch_violations(patch_plan)
    candidate_lookup = _unit_lookup(artifact_spec)
    baseline_lookup = _unit_lookup(baseline_artifact_spec or {})

    for cp_id, cp in (evidence_map.get("checkpoints") or {}).items():
        unit_rows = []
        cp_has_added = False
        cp_has_missing = False
        cp_has_changed = False
        cp_has_unchanged = False
        for unit in cp.get("evidence_units") or []:
            artifact = Path(str(unit.get("artifact") or "")).name
            unit_id = str(unit.get("unit_id") or "")
            key = _unit_key(artifact, unit_id)
            candidate_unit = _resolve_unit(artifact_spec, artifact, unit_id)
            baseline_unit = _resolve_unit(baseline_artifact_spec or {}, artifact, unit_id) if baseline_artifact_spec else None
            if candidate_unit is None:
                status = "missing"
                cp_has_missing = True
                if cp.get("status") == PROTECTED_HIGH:
                    section_deleted.append({"checkpoint_id": cp_id, "artifact": artifact, "unit_id": unit_id, "reason": "protected evidence unit missing"})
            elif baseline_unit is None:
                status = "added" if baseline_artifact_spec else "present"
                cp_has_added = cp_has_added or status == "added"
            elif candidate_unit.get("hash") != baseline_unit.get("hash"):
                status = "changed"
                cp_has_changed = True
                if cp.get("status") == PROTECTED_HIGH:
                    protected_hash_changed.append({"checkpoint_id": cp_id, "artifact": artifact, "unit_id": unit_id, "reason": "protected unit content changed"})
            else:
                status = "unchanged"
                cp_has_unchanged = True
            row = {
                "checkpoint_id": cp_id,
                "checkpoint_status": cp.get("status"),
                "artifact": artifact,
                "unit_id": unit_id,
                "qualified_unit": key,
                "type": unit.get("type"),
                "status": status,
            }
            evidence_unit_changes.append(row)
            if candidate_unit:
                unit_rows.append(candidate_unit)

        if cp.get("status") == TARGET_LOW:
            if status_judge:
                target_status[cp_id] = _normalize_status(status_judge(cp, unit_rows, _evidence_text(unit_rows)))
            elif cp_has_added or cp_has_changed or (unit_rows and not cp_has_missing):
                target_status[cp_id] = "evidence_added"
            else:
                target_status[cp_id] = "insufficient_evidence"
        elif cp.get("status") == PROTECTED_HIGH:
            if cp_has_missing or cp_has_changed:
                protected_status[cp_id] = "risk_detected"
            elif cp_has_unchanged:
                protected_status[cp_id] = "protected_evidence_unchanged"
            elif unit_rows:
                protected_status[cp_id] = "protected_evidence_present"
            else:
                protected_status[cp_id] = "unresolved"

    csv_schema_changed = _schema_changes(baseline_artifact_spec, artifact_spec, evidence_map)
    hard_violations = bool(patch_violations or protected_hash_changed or csv_schema_changed or section_deleted)
    target_gain_observed = any(status in GOOD_TARGET_STATUSES for status in target_status.values())
    target_missing = any(status in TARGET_RETRY_STATUSES for status in target_status.values()) or (bool(target_status) and not target_gain_observed)
    patch_mode = str(patch_plan.get("patch_mode") or "unknown")

    if not artifact_validity["artifact_pass"] or hard_violations:
        collateral_damage_risk = "high"
    elif patch_mode in {"fallback_full_generation", "disabled_full_generation"} and protected_status:
        collateral_damage_risk = "medium"
    elif any(status in {"risk_detected", "unresolved"} for status in protected_status.values()):
        collateral_damage_risk = "medium"
    else:
        collateral_damage_risk = "low"

    if not artifact_validity["artifact_pass"]:
        retry_mode = "target_retry"
    elif hard_violations:
        retry_mode = "preservation_retry"
    elif target_missing:
        retry_mode = "target_retry"
    elif target_gain_observed and collateral_damage_risk in {"medium", "high"}:
        retry_mode = "preservation_retry"
    else:
        retry_mode = "target_retry"

    target_status_score = sum(1 for status in target_status.values() if status in GOOD_TARGET_STATUSES)
    selection_signals = {
        "valid_for_selection": artifact_validity["artifact_pass"],
        "artifact_hard_fail": not artifact_validity["artifact_pass"],
        "protected_hard_violation": hard_violations,
        "protected_unit_risk": collateral_damage_risk,
        "target_gain_observed": target_gain_observed,
        "target_status_score": target_status_score,
        "collateral_damage_risk": collateral_damage_risk,
        "collateral_damage_risk_rank": RISK_RANK[collateral_damage_risk],
        "patch_mode": patch_mode,
        "structured_patch": patch_mode == "structured_patch",
        "retry_mode": retry_mode,
    }

    return {
        "task_id": task_id,
        "repair_evaluator_version": "preservation_aware_v1",
        "artifact_validity": artifact_validity,
        "artifact_pass": artifact_validity["artifact_pass"],
        "artifact_fail_reasons": artifact_validity["artifact_fail_reasons"],
        "target_checkpoint_status": target_status,
        "protected_checkpoint_status": protected_status,
        "evidence_unit_changes": evidence_unit_changes,
        "patch_violations": patch_violations,
        "protected_unit_hash_changed": protected_hash_changed,
        "csv_schema_changed": csv_schema_changed,
        "section_deleted": section_deleted,
        "collateral_damage_risk": collateral_damage_risk,
        "target_gain_observed": target_gain_observed,
        "retry_mode": retry_mode,
        "candidate_selection_signals": selection_signals,
        "checkpoint_evidence_map": evidence_map,
    }


def retry_feedback_from_report(report: dict[str, Any], *, include_preservation: bool = True) -> dict[str, Any]:
    protected_payload = {
        "protected_checkpoint_status": report.get("protected_checkpoint_status", {}),
        "patch_violations": report.get("patch_violations", []),
        "protected_unit_hash_changed": report.get("protected_unit_hash_changed", []),
        "csv_schema_changed": report.get("csv_schema_changed", []),
        "section_deleted": report.get("section_deleted", []),
        "collateral_damage_risk": report.get("collateral_damage_risk"),
    }
    if not include_preservation:
        protected_payload = {"status": "omitted_for_ablation"}
    return {
        "artifact_validity": report.get("artifact_validity", {}),
        "target_checkpoint_status": report.get("target_checkpoint_status", {}),
        "protected_feedback": protected_payload,
        "target_gain_observed": report.get("target_gain_observed", False),
        "retry_mode": "target_retry" if not include_preservation else report.get("retry_mode", "target_retry"),
        "selection_hint": _selection_hint(report, include_preservation=include_preservation),
    }


def _selection_hint(report: dict[str, Any], *, include_preservation: bool) -> str:
    if not report.get("artifact_pass"):
        return "target_retry: fix missing or invalid artifacts before candidate selection"
    if include_preservation and report.get("retry_mode") == "preservation_retry":
        return "preservation_retry: keep target gains and restore protected evidence units; prefer patch over full rewrite"
    if not report.get("target_gain_observed"):
        return "target_retry: add or ground evidence for target checkpoints"
    return "candidate is repair-evaluator valid; preserve protected evidence and avoid unnecessary rewrites"


def make_baseline_selection_candidate(baseline_score: float) -> dict[str, Any]:
    return {
        "attempt_index": 0,
        "label": "baseline",
        "artifact_pass": True,
        "root_after": float(baseline_score),
        "root_gain": 0.0,
        "target_gain": 0.0,
        "official_score": float(baseline_score),
        "repair_eval_report": {
            "artifact_pass": True,
            "target_gain_observed": False,
            "collateral_damage_risk": "low",
            "retry_mode": "target_retry",
            "candidate_selection_signals": {
                "valid_for_selection": True,
                "artifact_hard_fail": False,
                "protected_hard_violation": False,
                "protected_unit_risk": "low",
                "target_gain_observed": False,
                "target_status_score": 0,
                "collateral_damage_risk": "low",
                "collateral_damage_risk_rank": 0,
                "patch_mode": "baseline",
                "structured_patch": False,
                "retry_mode": "target_retry",
            },
        },
        "candidate_selection_signals": {
            "valid_for_selection": True,
            "artifact_hard_fail": False,
            "protected_hard_violation": False,
            "protected_unit_risk": "low",
            "target_gain_observed": False,
            "target_status_score": 0,
            "collateral_damage_risk": "low",
            "collateral_damage_risk_rank": 0,
            "patch_mode": "baseline",
            "structured_patch": False,
            "retry_mode": "target_retry",
        },
        "selection_rule": "baseline",
    }


def select_candidate_by_repair_evaluator(
    candidates: list[dict[str, Any]],
    *,
    baseline_score: float,
    require_artifact_pass: bool = True,
    damage_aware: bool = True,
    epsilon_root: float = 0.0,
) -> dict[str, Any]:
    baseline = make_baseline_selection_candidate(baseline_score)
    pool = [cand for cand in candidates if cand.get("attempt_index") != 0]
    if require_artifact_pass:
        pool = [cand for cand in pool if _signals(cand).get("valid_for_selection") and cand.get("artifact_pass", True)]
    if not pool:
        selected = dict(baseline)
        selected["selection_trace"] = {
            "selection_rule": "root_primary_repair_evaluator_tie_band",
            "require_artifact_pass": require_artifact_pass,
            "damage_aware": damage_aware,
            "epsilon_root": float(epsilon_root or 0.0),
            "R_best": float(baseline_score),
            "near_best": ["baseline"],
            "excluded_by_root_gap": [],
            "selected": "baseline",
            "selection_reason": "no candidate remained after artifact gate",
        }
        return selected

    valid_pool = [baseline] + pool
    epsilon = max(0.0, float(epsilon_root or 0.0))
    r_best = max(float(cand.get("root_after", 0.0) or 0.0) for cand in valid_pool)
    near_best = [
        cand
        for cand in valid_pool
        if r_best - float(cand.get("root_after", 0.0) or 0.0) <= epsilon
    ]
    excluded_by_root_gap = [
        cand
        for cand in valid_pool
        if r_best - float(cand.get("root_after", 0.0) or 0.0) > epsilon
    ]

    def key(cand: dict[str, Any]) -> tuple[Any, ...]:
        signals = _signals(cand)
        protected_violation = bool(signals.get("protected_hard_violation")) if damage_aware else False
        risk_rank = int(signals.get("collateral_damage_risk_rank", 0) or 0) if damage_aware else 0
        structured_patch_penalty = 0 if signals.get("structured_patch") else 1
        return (
            protected_violation,
            risk_rank,
            -int(signals.get("target_status_score", 0) or 0),
            -int(bool(signals.get("target_gain_observed"))),
            structured_patch_penalty,
            -float(cand.get("target_gain", 0.0) or 0.0),
            -float(cand.get("root_after", 0.0) or 0.0),
        )

    selected = min(near_best, key=key)
    trace = {
        "selection_rule": "root_primary_repair_evaluator_tie_band",
        "require_artifact_pass": require_artifact_pass,
        "damage_aware": damage_aware,
        "epsilon_root": epsilon,
        "R_best": r_best,
        "near_best": [_candidate_label(cand) for cand in near_best],
        "excluded_by_root_gap": [_candidate_label(cand) for cand in excluded_by_root_gap],
        "selected": _candidate_label(selected),
        "selection_reason": (
            "repair evaluator tie-break within epsilon_root"
            if len(near_best) > 1
            else "root gap exceeds epsilon_root; selected root-best candidate"
        ),
    }
    if float(selected.get("root_after", 0.0) or 0.0) <= float(baseline_score):
        selected_baseline = dict(baseline)
        selected_baseline["selection_trace"] = {
            **trace,
            "selected": "baseline",
            "selection_reason": "baseline has the best or non-worse official root score",
        }
        return selected_baseline
    selected = dict(selected)
    selected["selection_rule"] = "root_primary_repair_evaluator_tie_band"
    selected["selection_trace"] = trace
    selected["selection_pool_size"] = len(pool)
    selected["selection_root_best"] = r_best
    selected["near_best_pool_size"] = len(near_best)
    return selected


def _candidate_label(candidate: dict[str, Any]) -> str:
    if int(candidate.get("attempt_index", -1)) == 0:
        return "baseline"
    return f"attempt{candidate.get('attempt_index')}"


def _signals(candidate: dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate.get("candidate_selection_signals"), dict):
        return candidate["candidate_selection_signals"]
    report = candidate.get("repair_eval_report")
    if isinstance(report, dict) and isinstance(report.get("candidate_selection_signals"), dict):
        return report["candidate_selection_signals"]
    return {
        "valid_for_selection": bool(candidate.get("artifact_pass", True)),
        "artifact_hard_fail": not bool(candidate.get("artifact_pass", True)),
        "protected_hard_violation": bool(candidate.get("violated_protected_units")),
        "protected_unit_risk": "medium" if candidate.get("violated_protected_units") else "low",
        "target_gain_observed": float(candidate.get("target_gain", 0.0) or 0.0) > 0,
        "target_status_score": 1 if float(candidate.get("target_gain", 0.0) or 0.0) > 0 else 0,
        "collateral_damage_risk": "medium" if candidate.get("violated_protected_units") else "low",
        "collateral_damage_risk_rank": 1 if candidate.get("violated_protected_units") else 0,
        "structured_patch": candidate.get("patch_mode") == "structured_patch",
        "retry_mode": "target_retry",
    }
