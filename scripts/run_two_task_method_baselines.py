"""Run method-level baselines for a held-out task subset.

This runner is intentionally separate from the Full Ours repair pipeline. It
does not call the repair evaluator, evidence map, patch-preservation controller,
or non-regression logic. It only:

1. reuses/fetches the GTA-2 official baseline score for diagnosis context;
2. asks a generation model to produce a method-specific candidate;
3. materializes the expected deliverables through the shared artifact builder;
4. reports the selected candidate with the unchanged GTA-2 official scorer.

The default method set is for filling main-table baselines, not module
ablations.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "heldout20" / "heldout20_router_registry_gemini.json"
DEFAULT_SOURCE_DATASET = ROOT / "vendor" / "GTA" / "opencompass" / "data" / "gta_dataset_v2" / "end.json"
DEFAULT_OUT_ROOT = ROOT / "runs" / "v3.1" / "method_baselines_yunwu_strict_gta2"
DEFAULT_TASK_IDS = [7, 14, 45, 61, 62, 71, 79, 82, 88, 90, 92, 94, 98, 108, 111, 121, 122, 127, 133, 151]
DEFAULT_METHODS = [
    "prompt_only_self_refine",
    "evaluator_feedback_retry",
    "diagnosis_only_repair",
    "agentfixer_guided_independent_repair",
    "full_retry",
    "full_reflexion",
]
LOW_CHECKPOINT_THRESHOLD = 7.0

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from provider_runtime import extract_first_json_object, strip_code_fences  # noqa: E402
from run_heldout20_full_ours_candidate_generation import (  # noqa: E402
    dump_json,
    expected_files_from_prediction,
    flatten_nodes,
    load_json,
    materialize_candidate_files,
    read_score,
    write_csv,
)
from scorer_runtime import scorer_python  # noqa: E402


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def exception_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    status_code = getattr(exc, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def transient_retry_delay(attempt: int, status_code: int | None) -> int | None:
    if status_code not in {408, 409, 429, 500, 502, 503, 504}:
        return None
    if status_code in {429, 503}:
        return min(300, 20 * attempt)
    return min(120, 5 * attempt)


def registry_rows(path: Path) -> dict[int, dict[str, Any]]:
    data = load_json(path)
    runs = data.get("runs", data) if isinstance(data, dict) else data
    return {int(row["task_id"]): row for row in runs}


def load_source_task(task_id: int) -> dict[str, Any]:
    data = load_json(DEFAULT_SOURCE_DATASET)
    return data[str(task_id)]


def task_prompt(task: dict[str, Any]) -> str:
    for dialog in task.get("dialogs", []) or []:
        if isinstance(dialog, dict) and dialog.get("role") == "user":
            content = dialog.get("content")
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content or "")
    return str(task.get("prompt") or task.get("query") or "")


def first_sample(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "prediction" in payload:
        return payload
    if isinstance(payload, dict):
        for key in sorted(payload.keys(), key=lambda value: str(value)):
            value = payload[key]
            if isinstance(value, dict) and ("prediction" in value or "assistant_outputs" in value):
                return value
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    raise KeyError("No sample with prediction/assistant_outputs found")


def extract_final_answer(prediction_entry: dict[str, Any]) -> str:
    prediction = prediction_entry.get("assistant_outputs") or prediction_entry.get("prediction") or []
    flat: list[Any] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, list):
            for item in obj:
                collect(item)
        else:
            flat.append(obj)

    collect(prediction)
    for item in reversed(flat):
        if isinstance(item, dict) and item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def low_checkpoint_diagnostics(result: dict[str, Any], threshold: float = LOW_CHECKPOINT_THRESHOLD) -> list[dict[str, Any]]:
    diagnostics = []
    for node in flatten_nodes(result):
        try:
            score = float(node.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        if score >= threshold:
            continue
        analysis = str(node.get("analysis", ""))
        text = analysis.lower()
        failure_type = "unknown"
        if "missing" in text or "file" in text or "artifact" in text:
            failure_type = "missing_or_incomplete_artifact"
        elif "citation" in text or "source" in text or "ground" in text:
            failure_type = "insufficient_grounding"
        elif "calculation" in text or "quantitative" in text or "dimension" in text:
            failure_type = "calculation_or_quantitative_gap"
        elif "format" in text or "layout" in text or "section" in text:
            failure_type = "format_or_structure_gap"
        diagnostics.append(
            {
                "checkpoint_id": node.get("id"),
                "score": score,
                "failure_type": failure_type,
                "evaluator_analysis": analysis[:1400],
            }
        )
    return diagnostics


def resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def attachment_paths(prediction_entry: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for raw in prediction_entry.get("attached_files") or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = resolve_project_path(raw)
        if path.exists() and path.is_file():
            paths.append(path)
    return paths


def extract_text_attachment(path: Path, max_chars: int = 6000) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            chunks = []
            for page in reader.pages[:8]:
                chunks.append(page.extract_text() or "")
                if sum(len(chunk) for chunk in chunks) >= max_chars:
                    break
            return "\n".join(chunks).strip()[:max_chars]
        if suffix == ".docx":
            from docx import Document

            doc = Document(str(path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()[:max_chars]
        if suffix in {".txt", ".md", ".csv", ".html"}:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception as exc:
        return f"[Could not extract attachment text from {path.name}: {type(exc).__name__}: {exc}]"
    return ""


def attachment_context_and_images(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    context_chunks = []
    image_parts: list[dict[str, Any]] = []
    for path in paths:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        if mime.startswith("image/"):
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            context_chunks.append(f"- {path.name}: image attached to the model input.")
            continue
        text = extract_text_attachment(path)
        if text:
            context_chunks.append(f"### {path.name}\n{text}")
        else:
            context_chunks.append(f"- {path.name}: original input file present, but no text excerpt was extracted.")
    return "\n\n".join(context_chunks), image_parts


def json_contract(expected_files: list[str]) -> str:
    return f"""
Return only valid JSON with this shape:
{{
  "final_answer_markdown": "Concise final response that names every produced deliverable.",
  "deliverables": {{
    "filename.ext": "Complete textual/markdown/table content for that file"
  }},
  "rationale": ["short method-specific notes"]
}}

Expected deliverable filenames:
{json.dumps(expected_files, ensure_ascii=False, indent=2)}

Rules:
- Every expected filename must appear as a key in deliverables.
- For PDF/DOCX/PPTX/XLSX deliverables, provide the source text/markdown/table content; the runner will materialize the actual container.
- Do not claim a file exists unless it appears in deliverables.
- Do not mention hidden repair modules, artifact gates, evidence maps, or non-regression controllers.
""".strip()


def build_method_prompt(
    *,
    method: str,
    task_id: int,
    origin_prompt: str,
    original_answer: str,
    baseline_score: float,
    diagnostics: list[dict[str, Any]],
    expected_files: list[str],
    attachment_context: str,
    registry_row: dict[str, Any] | None = None,
) -> str:
    shared = f"""
GTA-2 task id: {task_id}

Original user task:
{origin_prompt}

Original input attachments/excerpts:
{attachment_context or "No attachment excerpts were available."}

{json_contract(expected_files)}
""".strip()
    if method == "prompt_only_self_refine":
        return f"""
You are performing the Prompt-only self-refine baseline.
Use only the original task, original input attachments/excerpts, and the previous final answer.
Do not use evaluator feedback, checkpoint scores, repair modules, evidence maps, artifact verifiers, non-regression gates, or external tools.

Previous final answer:
{original_answer}

{shared}
""".strip()
    if method == "evaluator_feedback_retry":
        feedback = [
            {
                "checkpoint_id": item["checkpoint_id"],
                "score": item["score"],
                "evaluator_analysis": item["evaluator_analysis"],
            }
            for item in diagnostics
        ]
        return f"""
You are performing the Evaluator-feedback retry baseline.
Use the original task, previous final answer, original input attachments/excerpts, and low-checkpoint evaluator feedback.
Do not use repair modules, evidence maps, artifact verifiers, non-regression gates, or external tools.

Previous final answer:
{original_answer}

Baseline official root score: {baseline_score}

Low-checkpoint evaluator feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

{shared}
""".strip()
    if method == "diagnosis_only_repair":
        return f"""
You are performing the Diagnosis-only repair baseline.
Use the original task, previous final answer, original input attachments/excerpts, and the structured failure diagnosis.
Do not use task-specific repair modules, evidence maps, artifact verifiers, non-regression gates, or external tools.

Previous final answer:
{original_answer}

Structured diagnosis:
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

{shared}
""".strip()
    if method == "agentfixer_guided_independent_repair":
        fix_context = {
            "failure_type": (registry_row or {}).get("failure_type"),
            "artifact_gate_decision": (registry_row or {}).get("artifact_gate_decision"),
            "artifact_score": (registry_row or {}).get("artifact_score"),
            "suggested_modules": (registry_row or {}).get("suggested_modules", []),
            "diagnostics": diagnostics,
        }
        return f"""
You are performing the AgentFixer-guided independent repair baseline.
Use the original task, previous final answer, original input attachments/excerpts, and the AgentFixer-style failure classification below.
Do not use Full Ours repair evaluator, evidence maps, patch-preservation controller, artifact gate decisions as filters, or non-regression gates.
Generate an independent repaired candidate guided only by the failure class and fix recommendations.

Previous final answer:
{original_answer}

AgentFixer-style failure context:
{json.dumps(fix_context, ensure_ascii=False, indent=2)}

{shared}
""".strip()
    if method == "full_retry":
        return f"""
You are performing the ReAct + full retry baseline in a single clean retry.
Solve the original task from scratch using only the original task and original input attachments/excerpts.
Do not use the previous final answer, evaluator feedback, checkpoint scores, repair modules, evidence maps, artifact verifiers, or non-regression gates.

{shared}
""".strip()
    if method == "full_reflexion":
        failure_summary = {
            "baseline_root_score": baseline_score,
            "main_failures": diagnostics[:8],
            "reflection": (
                "The previous run likely failed because required deliverables and concrete evidence were incomplete. "
                "The next attempt should complete the requested artifacts, include source-grounded content, and make the final answer point to produced files."
            ),
        }
        return f"""
You are performing the Full Reflexion baseline.
Use the original task, original input attachments/excerpts, and the concise reflection below to rerun the task from scratch.
Do not use repair modules, evidence maps, artifact verifiers, non-regression gates, or protected-unit controllers.

Reflection memory:
{json.dumps(failure_summary, ensure_ascii=False, indent=2)}

{shared}
""".strip()
    raise ValueError(f"Unknown method: {method}")


def call_gpt_json(prompt: str, image_parts: list[dict[str, Any]], model: str, max_tokens: int) -> dict[str, Any]:
    api_key = first_env("OPENAI_API_KEY", "OPEN_API_KEY", "GPT_API_KEY", "GPT4O_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY/OPEN_API_KEY/GPT_API_KEY/GPT4O_API_KEY is required.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_headers: dict[str, str] = {}
    app_code = first_env("OPENAI_APP_CODE")
    if app_code or "yunwu.ai" in base_url.lower():
        default_headers["APP-Code"] = app_code or "APP-10012fcb"
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)

    def make_content(include_images: bool) -> str | list[dict[str, Any]]:
        if include_images and image_parts:
            return [{"type": "text", "text": prompt}, *image_parts]
        return prompt

    last_text = ""
    last_error = ""
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        kwargs: dict[str, Any] = {}
        if attempt == 1:
            kwargs["response_format"] = {"type": "json_object"}
        include_images = bool(image_parts) and attempt == 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate GTA-2 baseline candidate artifacts. Return only valid JSON.",
                    },
                    {"role": "user", "content": make_content(include_images)},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if image_parts and include_images:
                continue
            delay = transient_retry_delay(attempt, exception_status_code(exc))
            if delay is None or attempt >= max_attempts:
                raise
            print(
                f"[generation retry] model={model} attempt={attempt}/{max_attempts} "
                f"delay={delay}s error={last_error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            continue
        last_text = response.choices[0].message.content or ""
        parsed = extract_first_json_object(last_text)
        if parsed is not None:
            return parsed
    return {"final_answer_markdown": strip_code_fences(last_text), "deliverables": {}, "rationale": [f"model returned non-json; last_error={last_error}"]}


def score_candidate_command(task_id: int, candidate_dir: Path, out: Path, scorer_model: str) -> list[str]:
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
        scorer_model,
    ]


def score_prediction_command(task_id: int, prediction: Path, out: Path, scorer_model: str) -> list[str]:
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
        scorer_model,
    ]


def run_cmd(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def latest_existing_baseline_rescore(task_id: int) -> Path | None:
    candidates = list((ROOT / "runs" / "v3.1").glob(f"*/**/task{task_id}/baseline_gta2_rescore/official_rescore_result.json"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def ensure_baseline_rescore(
    *,
    task_id: int,
    prediction_path: Path,
    task_dir: Path,
    scorer_model: str,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    existing = None if force else latest_existing_baseline_rescore(task_id)
    if existing is not None:
        result = load_json(existing)
        dump_json(
            task_dir / "baseline_gta2_rescore" / "baseline_rescore_reused.json",
            {"source": str(existing), "gpt_score": result.get("gpt_score")},
        )
        return existing, result

    out = task_dir / "baseline_gta2_rescore" / "official_rescore_result.json"
    run = run_cmd(score_prediction_command(task_id, prediction_path, out, scorer_model))
    dump_json(task_dir / "baseline_gta2_rescore" / "score_prediction_run.json", run)
    if run["returncode"] != 0 or not out.exists():
        raise RuntimeError(f"Baseline rescore failed for task {task_id}. See {task_dir / 'baseline_gta2_rescore' / 'score_prediction_run.json'}")
    return out, load_json(out)


def parse_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    methods = [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Known methods: {DEFAULT_METHODS}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="*", type=int, default=DEFAULT_TASK_IDS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--scorer-model", default="gpt-5.2")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--force-baseline-rescore", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    rows_by_task = registry_rows(args.registry)
    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_by_method: dict[str, list[float]] = {"vanilla_react_lagent": []}
    for method in methods:
        summary_by_method[method] = []

    for task_id in args.task_ids:
        if task_id not in rows_by_task:
            raise KeyError(f"Task {task_id} not found in {args.registry}")
        row = rows_by_task[task_id]
        task_dir = run_dir / f"task{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        task = load_source_task(task_id)
        baseline_prediction = resolve_project_path(str(row["baseline_prediction"]))
        prediction_entry = first_sample(load_json(baseline_prediction))
        origin_prompt = str(prediction_entry.get("origin_prompt") or task_prompt(task))
        original_answer = extract_final_answer(prediction_entry)
        expected_files = expected_files_from_prediction(baseline_prediction)
        attach_paths = attachment_paths(prediction_entry)
        attach_context, image_parts = attachment_context_and_images(attach_paths)

        baseline_result_path, baseline_result = ensure_baseline_rescore(
            task_id=task_id,
            prediction_path=baseline_prediction,
            task_dir=task_dir,
            scorer_model=args.scorer_model,
            force=args.force_baseline_rescore,
        )
        baseline_score = float(baseline_result.get("gpt_score") or 0.0)
        diagnostics = low_checkpoint_diagnostics(baseline_result)
        dump_json(task_dir / "baseline_context.json", {
            "task_id": task_id,
            "baseline_prediction": str(baseline_prediction),
            "baseline_result": str(baseline_result_path),
            "baseline_score": baseline_score,
            "expected_files": expected_files,
            "attached_files": [str(path) for path in attach_paths],
            "num_low_checkpoints": len(diagnostics),
        })
        dump_json(task_dir / "diagnostics.json", diagnostics)

        vanilla_row = {
            "task_id": task_id,
            "method": "vanilla_react_lagent",
            "generation_model": "baseline",
            "scorer_model": args.scorer_model,
            "official_root_score": baseline_score,
            "baseline_score": baseline_score,
            "score_delta_vs_baseline": 0.0,
            "candidate_dir": "",
            "result_path": str(baseline_result_path),
            "expected_files": expected_files,
            "num_low_checkpoints": len(diagnostics),
        }
        all_rows.append(vanilla_row)
        summary_by_method["vanilla_react_lagent"].append(baseline_score)

        for method in methods:
            method_dir = task_dir / method
            candidate_dir = method_dir / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            prompt = build_method_prompt(
                method=method,
                task_id=task_id,
                origin_prompt=origin_prompt,
                original_answer=original_answer,
                baseline_score=baseline_score,
                diagnostics=diagnostics,
                expected_files=expected_files,
                attachment_context=attach_context,
                registry_row=row,
            )
            (method_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
            payload = call_gpt_json(prompt, image_parts, args.model, args.max_tokens)
            dump_json(method_dir / "generation_payload.json", payload)
            answer = str(payload.get("final_answer_markdown") or "").strip()
            if not answer:
                answer = f"Generated {method} candidate for task {task_id}. See produced deliverables: {', '.join(expected_files)}."
                payload["final_answer_markdown"] = answer
            (candidate_dir / "repaired_final_answer.md").write_text(answer, encoding="utf-8")
            materialized = materialize_candidate_files(candidate_dir, payload, expected_files, enable_patch_repair=False)
            dump_json(method_dir / "materialized_files.json", materialized)

            score_out = method_dir / "official_rescore_result.json"
            score_run = run_cmd(score_candidate_command(task_id, candidate_dir, score_out, args.scorer_model))
            dump_json(method_dir / "score_candidate_run.json", score_run)
            if score_run["returncode"] != 0 or not score_out.exists():
                raise RuntimeError(f"Candidate score failed for task {task_id} method {method}. See {method_dir / 'score_candidate_run.json'}")
            result = load_json(score_out)
            score = read_score(score_out)
            method_row = {
                "task_id": task_id,
                "method": method,
                "generation_model": args.model,
                "scorer_model": args.scorer_model,
                "official_root_score": score,
                "baseline_score": baseline_score,
                "score_delta_vs_baseline": None if score is None else score - baseline_score,
                "candidate_dir": str(candidate_dir),
                "result_path": str(score_out),
                "expected_files": expected_files,
                "num_low_checkpoints": len(diagnostics),
                "patch_mode": materialized.get("patch_mode"),
                "created_files": materialized.get("created_files"),
                "gpt_score_raw": result.get("gpt_score"),
            }
            dump_json(method_dir / "row.json", method_row)
            all_rows.append(method_row)
            if score is not None:
                summary_by_method[method].append(float(score))
            print(json.dumps(method_row, ensure_ascii=False, indent=2))

    write_csv(run_dir / "baseline_rows.csv", all_rows)
    dump_json(run_dir / "baseline_rows.json", all_rows)
    method_summary = []
    for method, scores in summary_by_method.items():
        method_summary.append(
            {
                "method": method,
                "num_tasks": len(scores),
                "mean_score": None if not scores else mean(scores),
                "success_at_8": None if not scores else sum(score >= 8 for score in scores) / len(scores),
                "perfect_at_10": None if not scores else sum(score >= 10 for score in scores) / len(scores),
            }
        )
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(run_dir),
        "task_ids": args.task_ids,
        "methods": ["vanilla_react_lagent", *methods],
        "generation_model": args.model,
        "scorer_model": args.scorer_model,
        "method_summary": method_summary,
        "note": "Method-level baselines only; no module ablations were run.",
    }
    dump_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
