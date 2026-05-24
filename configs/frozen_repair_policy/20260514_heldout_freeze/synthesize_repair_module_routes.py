"""Synthesize repair-module routes from baseline diagnostics.

This script is a lightweight router for the next controller stage. It consumes
the baseline diagnosis expansion registry and turns each task into a reusable
module route based on:

- baseline root score,
- number/fraction of low checkpoints,
- failure type,
- existing diagnosis text.

The output is intentionally interpretable so it can be used both as experiment
metadata and as a method-table artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "baseline_diagnosis_expansion.json"
DEFAULT_OUT = ROOT / "runs" / "repair_module_routes"


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
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def checkpoint_stats(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    nodes = []
    for sample in result.get("details", []):
        nodes.extend(sample.get("nodes", []))
    low = [node for node in nodes if float(node.get("score", 0)) < threshold]
    return {
        "baseline_score": result.get("gpt_score"),
        "num_checkpoints": len(nodes),
        "num_low_before": len(low),
        "low_checkpoint_ids": [node.get("id") for node in low],
        "low_fraction": 0.0 if not nodes else len(low) / len(nodes),
    }


def route_modules(failure_type: str, stats: dict[str, Any], suggested: list[str]) -> tuple[str, list[str], str]:
    score = float(stats.get("baseline_score") or 0.0)
    num_low = int(stats.get("num_low_before") or 0)
    failure = failure_type.lower()

    if score >= 7.0 and 0 < num_low <= 2:
        modules = list(dict.fromkeys([*suggested, "NonRegressionPreservation"]))
        return "preservation_patch", modules, "high baseline with few low checkpoints; preserve sibling evidence"

    if "visual render" in failure:
        return (
            "visual_artifact_generation",
            ["ImageDescriptionRepair", "VisualAssetGenerationRepair", "RenderMetadataRepair"],
            "visual deliverables and render metadata are missing",
        )

    if (
        "document artifact" in failure
        or "document artifact package" in failure
        or "docx report" in failure
        or "report materialization" in failure
    ):
        return (
            "document_package_generation",
            list(dict.fromkeys([*suggested, "SourceGroundedDocumentRepair", "DocumentPackageRepair"])),
            "document/report artifacts and source-grounded sections are missing or not visible",
        )

    if "local file analysis" in failure:
        return (
            "local_data_report_generation",
            ["LocalDataExtractionRepair", "RankingComputationRepair", "ReportAssemblyRepair"],
            "local input parsing, ranking, and report assembly are required",
        )

    if "video artifact" in failure:
        return (
            "video_presentation_generation",
            ["ScenarioSpecificationRepair", "VideoArtifactGenerationRepair", "HtmlPresentationRepair"],
            "browser-playable MP4 artifacts and standalone HTML presentation are missing",
        )

    if "multi-artifact" in failure or "intent-only deliverable" in failure:
        return (
            "multi_artifact_package_generation",
            list(dict.fromkeys([*suggested, "MultiArtifactPackageRepair"])),
            "multiple promised deliverables are absent and must be materialized as a package",
        )

    if "limited-turn" in failure or "timeout" in failure:
        return (
            "timeout_aware_static_site_generation",
            [
                "TimeoutAwareStaticSiteAdapter",
                "ArtifactPackageRepair",
                "GeometryInteractionVisibilityRepair",
                "QuizCoverageRepair",
                "BrowserCompatibilityEvidenceRepair",
            ],
            "timeout or limited-turn baseline requires bounded static-site artifact materialization",
        )

    if "pptx" in failure or "powerpoint" in failure:
        return (
            "presentation_artifact_delivery",
            [
                "PptxDeckMaterializationRepair",
                "CitationManifestRepair",
                "EvidencePreservingAnswerRepair",
                "NonRegressionPreservation",
            ],
            "presentation file delivery is missing while sibling evidence must be preserved",
        )

    if "minor evidence" in failure:
        return (
            "skip_or_light_evidence_patch",
            ["EvidenceGroundingRepair"],
            "baseline is already high; route to lightweight evidence patch or skip",
        )

    if "evidence completeness" in failure or "report-table" in failure or "typicality evidence" in failure:
        return (
            "high_baseline_evidence_refinement",
            [
                "EvidenceCompletenessRepair",
                "DocxUsageTableRepair",
                "SourceBibliographyRepair",
                "NonRegressionPreservation",
            ],
            "high-baseline report needs explicit evidence/table completion without sibling regression",
        )

    if suggested:
        return "registry_suggested_route", suggested, "fall back to registry-provided suggested modules"

    return "manual_review", ["ManualReview"], "no rule matched"


def route_row(item: dict[str, Any], threshold: float) -> dict[str, Any]:
    baseline_path = Path(item["baseline_result"])
    baseline = load_json(baseline_path)
    stats = checkpoint_stats(baseline, threshold)
    route_name, modules, rationale = route_modules(
        str(item.get("failure_type", "")),
        stats,
        list(item.get("suggested_modules", [])),
    )
    repaired_score = None
    if item.get("repaired_result") and Path(item["repaired_result"]).exists():
        repaired_score = load_json(Path(item["repaired_result"])).get("gpt_score")

    return {
        "task_id": item.get("task_id"),
        "category": item.get("category"),
        "status": item.get("status"),
        "failure_type": item.get("failure_type"),
        **stats,
        "route_name": route_name,
        "routed_modules": modules,
        "suggested_modules": item.get("suggested_modules", []),
        "rationale": rationale,
        "repaired_score": repaired_score,
        "score_delta": None
        if repaired_score is None or stats.get("baseline_score") is None
        else float(repaired_score) - float(stats["baseline_score"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=7.0)
    args = parser.parse_args()

    registry = load_json(args.registry)
    rows = [route_row(item, args.threshold) for item in registry.get("runs", [])]
    out_dir = args.out_dir / __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_json(out_dir / "repair_module_routes.json", rows)
    write_csv(out_dir / "repair_module_routes.csv", rows)
    summary = {
        "num_tasks": len(rows),
        "num_closed_loop": sum(1 for row in rows if str(row.get("status", "")).startswith("closed_loop")),
        "routes": sorted({row["route_name"] for row in rows}),
        "out_dir": str(out_dir),
    }
    dump_json(out_dir / "repair_module_routes_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
