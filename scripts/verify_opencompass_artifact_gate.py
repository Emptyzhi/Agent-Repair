"""Detect false-high GTA/OpenCompass scores caused by missing artifacts.

This verifier reads an OpenCompass prediction JSON, optionally the official
result JSON, and checks whether required deliverables were actually materialized
as files. It is deliberately deterministic and conservative: text that merely
contains HTML/CSV/DOCX-like content does not count as a real artifact.

Typical use:

    uv run python scripts/verify_opencompass_artifact_gate.py ^
      --prediction vendor/GTA/opencompass/outputs/.../predictions/.../gta_bench_end_single.json ^
      --result vendor/GTA/opencompass/outputs/.../results/.../gta_bench_end_single.json ^
      --out runs/artifact_gate/task88.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


GENERATOR_BY_EXT = {
    ".csv": "CsvFileGenerator",
    ".docx": "DocxFileGenerator",
    ".html": "HtmlFileGenerator",
    ".pdf": "PdfFileGenerator",
    ".pptx": "PptxFileGenerator",
    ".xlsx": "XlsxFileGenerator",
    ".png": None,
    ".jpg": None,
    ".jpeg": None,
}

EXT_ALIASES = {
    "csv": ".csv",
    "docx": ".docx",
    "word": ".docx",
    "microsoft word": ".docx",
    "html": ".html",
    "web page": ".html",
    "pdf": ".pdf",
    "powerpoint": ".pptx",
    "pptx": ".pptx",
    "excel": ".xlsx",
    "xlsx": ".xlsx",
}


@dataclass
class ExpectedArtifact:
    ext: str
    name: str | None = None
    source: str = "inferred"
    delivery: str = "file"  # "file" or "inline"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def first_sample(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "0" in data and isinstance(data["0"], dict):
        return data["0"]
    if isinstance(data, dict) and len(data) == 1:
        item = next(iter(data.values()))
        if isinstance(item, dict):
            return item
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError("Unsupported prediction JSON shape")


def official_score(result_path: Path | None) -> float | None:
    if result_path is None or not result_path.exists():
        return None
    data = load_json(result_path)
    if isinstance(data, dict):
        value = data.get("gpt_score")
        if isinstance(value, (int, float)):
            return float(value)
        if scores := data.get("sample_scores"):
            if isinstance(scores, list) and scores:
                return float(scores[0])
    return None


def stringify_task_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("origin_prompt", "full_tree"):
        value = item.get(key)
        if value is not None:
            parts.append(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
    if parts:
        return "\n".join(parts)
    # Fallback only for unusual traces without origin_prompt/full_tree. Do not
    # normally infer required deliverables from prediction text because generated
    # HTML/CSS/references can contain incidental words such as "word-break" or
    # "PDF" that are not task requirements.
    for key in ("gold", "prediction"):
        value = item.get(key)
        if value is not None:
            parts.append(json.dumps(value, ensure_ascii=False)[:4000])
    return "\n".join(parts)


def infer_expected_artifacts(item: dict[str, Any]) -> list[ExpectedArtifact]:
    text = stringify_task_text(item)
    found: dict[tuple[str, str | None], ExpectedArtifact] = {}
    lower = text.lower()
    inline_html = (
        "without attaching a separate file" in lower
        or "without attaching separate file" in lower
        or "supplied as plain text" in lower
        or "provide the complete markup" in lower
        or "complete markup should be supplied" in lower
    )

    # Explicit filenames are the strongest signal.
    # Keep this conservative. A broad "spaces allowed" pattern easily turns
    # prose such as "A CSV file named leads.csv" into a bogus filename.
    for match in re.finditer(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9][A-Za-z0-9_.-]{0,100}\.(?:csv|docx|html|pdf|pptx|xlsx|png|jpe?g))(?![A-Za-z0-9_.-])", text, re.I):
        name = match.group(1).strip(" .,:;\"'`")
        ext = Path(name).suffix.lower()
        prefix = text[max(0, match.start() - 40) : match.start()].lower().replace("\\", "/")
        if any(token in prefix for token in ["resources/files/", "resources/images/", "resources/image/", "input file "]):
            continue
        # GTA image tasks often name the input resource as a numeric file
        # (e.g. 20.jpg, 47.jpeg, 107.png). These are resources to inspect, not
        # deliverables that the agent should generate. Keep non-numeric image
        # filenames such as mediterranean_render.png or illustration_1.png.
        if ext in {".png", ".jpg", ".jpeg"} and re.fullmatch(r"\d+\.(?:png|jpe?g)", name, re.I):
            continue
        if ext in GENERATOR_BY_EXT:
            found[(ext, name.lower())] = ExpectedArtifact(ext=ext, name=name, source="explicit_filename", delivery="file")

    # Generic deliverable type mentions are weaker but useful when no filename is
    # named, e.g. "produce a professional HTML document".
    for phrase, ext in EXT_ALIASES.items():
        if phrase in lower and any(token in lower for token in ["deliverable", "file", "document", "report", "workbook", "web page", "html"]):
            # Do not treat ordinary word-count instructions as Word document
            # requirements.
            if phrase == "word" and "word document" not in lower and "microsoft word" not in lower:
                continue
            delivery = "inline" if ext == ".html" and inline_html else "file"
            found.setdefault((ext, None), ExpectedArtifact(ext=ext, name=None, source=f"type_mention:{phrase}", delivery=delivery))

    explicit_exts = {spec.ext for spec in found.values() if spec.name is not None}
    for key, spec in list(found.items()):
        if spec.name is None and spec.ext in explicit_exts:
            del found[key]

    # Avoid over-requiring source images that are task inputs, not deliverables.
    return sorted(found.values(), key=lambda spec: (spec.ext, spec.name or ""))


def flatten_assistant_outputs(item: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for run in item.get("assistant_outputs", []):
        if isinstance(run, list):
            turns.extend(turn for turn in run if isinstance(turn, dict))
        elif isinstance(run, dict):
            turns.append(run)
    return turns


def prediction_text(item: dict[str, Any]) -> str:
    return json.dumps(item.get("prediction", ""), ensure_ascii=False)


def attached_file_paths(item: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for value in item.get("attached_files", []) or []:
        if isinstance(value, str):
            paths.append(Path(value))
        elif isinstance(value, dict):
            candidate = value.get("path") or value.get("file") or value.get("filename")
            if candidate:
                paths.append(Path(str(candidate)))
    return paths


def artifact_dir_paths(artifact_dir: Path | None) -> list[Path]:
    if artifact_dir is None or not artifact_dir.exists():
        return []
    return [path for path in artifact_dir.rglob("*") if path.is_file()]


def collect_tool_summaries(item: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for turn in flatten_assistant_outputs(item):
        for summary in turn.get("tool_calls_summary", []) or []:
            if isinstance(summary, dict):
                summaries.append(summary)
    return summaries


def tool_attempts_for_ext(summaries: list[dict[str, Any]], ext: str) -> list[dict[str, Any]]:
    tool = GENERATOR_BY_EXT.get(ext)
    if not tool:
        return []
    return [summary for summary in summaries if summary.get("name") == tool]


def file_matches_expected(path: Path, expected: ExpectedArtifact) -> bool:
    if path.suffix.lower() != expected.ext:
        return False
    if expected.name is None:
        return True
    return path.name.lower() == expected.name.lower()


def file_looks_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def check_expected_artifact(
    expected: ExpectedArtifact,
    attached: list[Path],
    summaries: list[dict[str, Any]],
    pred_text: str,
) -> dict[str, Any]:
    if expected.delivery == "inline":
        ok = bool(re.search(r"<html[\s>].*</html>", pred_text, re.IGNORECASE | re.DOTALL))
        return {
            "expected": asdict(expected),
            "passed": ok,
            "partial": False,
            "attached_matches": [],
            "successful_attempt_count": 0,
            "failed_attempt_count": 0,
            "failed_attempt_examples": [],
            "reasons": [] if ok else ["inline_markup_not_found"],
        }

    attached_matches = [str(path) for path in attached if file_matches_expected(path, expected)]
    existing_matches = [path for path in attached if file_matches_expected(path, expected) and path.exists()]
    nonempty_matches = [path for path in existing_matches if file_looks_nonempty(path)]
    attempts = tool_attempts_for_ext(summaries, expected.ext)
    successful_attempts = [attempt for attempt in attempts if attempt.get("success") is True]
    failed_attempts = [attempt for attempt in attempts if attempt.get("success") is False]

    # A matching attached existing file is the strongest pass. If an AgentLego
    # generator explicitly succeeded but OpenCompass failed to attach the path,
    # count it as partial rather than full pass.
    passed = bool(nonempty_matches)
    partial = bool(successful_attempts) and not passed

    reasons = []
    if not attached_matches:
        reasons.append("no_matching_attached_file")
    elif not existing_matches:
        reasons.append("matching_attached_file_path_does_not_exist")
    elif not nonempty_matches:
        reasons.append("matching_attached_file_is_empty")
    if attempts and not successful_attempts:
        reasons.append("generator_attempt_failed")
    if not attempts and GENERATOR_BY_EXT.get(expected.ext) and not passed:
        reasons.append("generator_never_called")

    return {
        "expected": asdict(expected),
        "passed": passed,
        "partial": partial,
        "attached_matches": attached_matches,
        "successful_attempt_count": len(successful_attempts),
        "failed_attempt_count": len(failed_attempts),
        "failed_attempt_examples": failed_attempts[:2],
        "reasons": reasons,
    }


def verify(
    prediction_path: Path,
    result_path: Path | None,
    score_threshold: float,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    item = first_sample(load_json(prediction_path))
    expected = infer_expected_artifacts(item)
    attached_from_prediction = attached_file_paths(item)
    supplied_artifacts = artifact_dir_paths(artifact_dir)
    attached = attached_from_prediction + supplied_artifacts
    summaries = collect_tool_summaries(item)
    pred_text = prediction_text(item)
    checks = [check_expected_artifact(spec, attached, summaries, pred_text) for spec in expected]
    required_count = len(checks)
    passed_count = sum(1 for check in checks if check["passed"])
    partial_count = sum(1 for check in checks if check["partial"])
    artifact_score = 1.0 if required_count == 0 else passed_count / required_count
    score = official_score(result_path)
    false_high = bool(score is not None and score >= score_threshold and required_count > 0 and artifact_score < 1.0)

    generator_failures = [
        summary
        for summary in summaries
        if str(summary.get("name", "")).endswith("FileGenerator") and summary.get("success") is False
    ]

    return {
        "prediction_path": str(prediction_path),
        "result_path": None if result_path is None else str(result_path),
        "official_score": score,
        "score_threshold": score_threshold,
        "expected_artifact_count": required_count,
        "passed_artifact_count": passed_count,
        "partial_artifact_count": partial_count,
        "artifact_score": artifact_score,
        "false_high_score": false_high,
        "attached_files": [str(path) for path in attached_from_prediction],
        "artifact_dir": None if artifact_dir is None else str(artifact_dir),
        "artifact_dir_files": [str(path) for path in supplied_artifacts],
        "generator_failure_count": len(generator_failures),
        "generator_failure_examples": generator_failures[:3],
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--artifact-dir", type=Path, help="Optional repaired/output artifact directory to verify in addition to attached_files.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--score-threshold", type=float, default=8.0)
    parser.add_argument("--fail-on-false-high", action="store_true", help="Exit with code 2 when a high official score lacks required artifacts.")
    args = parser.parse_args()

    report = verify(
        args.prediction.resolve(),
        args.result.resolve() if args.result else None,
        args.score_threshold,
        args.artifact_dir.resolve() if args.artifact_dir else None,
    )
    out = args.out or args.prediction.resolve().parent / "artifact_gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "official_score": report["official_score"],
        "artifact_score": report["artifact_score"],
        "false_high_score": report["false_high_score"],
        "expected_artifact_count": report["expected_artifact_count"],
        "passed_artifact_count": report["passed_artifact_count"],
    }, ensure_ascii=False, indent=2))
    if args.fail_on_false_high and report["false_high_score"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
