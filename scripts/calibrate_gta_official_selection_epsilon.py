"""Calibrate epsilon_root / epsilon_cp with the official GTA-2 scorer.

This script intentionally routes scoring through the GTA GitHub
`agent_app_eval/score_with_gpt52.py` entrypoint and the repo-local official
`opencompass.datasets.gta_bench_v2.GPTEvaluator`.

For each fixed candidate instance it creates a GTA-style eval input:

  <run>/eval_inputs/task<T>/candidate<C>/repeat<R>/
    final.txt
    files/
      ... copied candidate deliverables ...
    eval_pack.json

It then scores that one-task pack with GPT-5.2 and computes:

  epsilon_root = max(score_resolution_root, 2 * P75(sigma_root))
  epsilon_cp   = max(score_resolution_cp,   2 * P75(sigma_cp))

where sigma_root is computed per fixed (task, candidate) across repeats, and
sigma_cp is computed per fixed (task, candidate, checkpoint_id) across repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GTA_REPO = ROOT / "vendor" / "GTA"
OFFICIAL_SCORE_SCRIPT = GTA_REPO / "agent_app_eval" / "score_with_gpt52.py"
OFFICIAL_COMMAND_SHIMS = ROOT / "tools" / "official-command-shims"
LIBREOFFICE_PROGRAM_DIR = Path(r"C:\Program Files\LibreOffice\program")
DEFAULT_TASK_IDS = [7, 14, 45, 61, 62, 71, 79, 82, 88, 90, 92, 94, 98, 108, 111, 121, 122, 127, 133, 151]
DEFAULT_RUN_FAMILY = ROOT / "runs" / "v3.1" / "full_ours_deepseek_v4_pro_preservation_search"
DEFAULT_DATASET_END_JSON = GTA_REPO / "opencompass" / "data" / "gta_dataset_v2" / "end.json"
SCORE_RESOLUTION_ROOT = 0.01
SCORE_RESOLUTION_CP = 0.01
ANSWER_NAMES = {
    "repaired_final_answer.md",
    "diagnosis_only_answer.md",
    "prompt_only_answer.md",
    "final.txt",
}


def scorer_python() -> Path:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_run_root(family: Path) -> Path:
    candidates = [path for path in family.iterdir() if path.is_dir()] if family.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {family}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def unique_existing(paths: list[str]) -> list[Path]:
    out = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        key = str(path.resolve()).lower()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append(path)
    return out


def select_task_candidates(run_root: Path, task_id: int, per_task: int) -> list[dict[str, Any]]:
    task_dir = run_root / f"task{task_id}"
    paths: list[str] = []
    attempts_path = task_dir / "full_ours" / "attempts.json"
    if attempts_path.exists():
        attempts = load_json(attempts_path)
        if isinstance(attempts, list):
            paths.extend(str(attempt.get("candidate_dir") or "") for attempt in attempts if isinstance(attempt, dict))
    paths.extend(str(path) for path in task_dir.glob("*/*/candidate"))
    paths.extend(str(path) for path in task_dir.glob("*/*/*/candidate"))
    selected = unique_existing(paths)[:per_task]
    if len(selected) < per_task:
        raise RuntimeError(f"Task {task_id} has only {len(selected)} candidate dirs under {task_dir}; need {per_task}")
    return [
        {
            "task_id": task_id,
            "candidate_index": index,
            "candidate_dir": str(path.resolve()),
            "candidate_id": f"task{task_id}_cand{index}_{re.sub(r'[^a-zA-Z0-9]+', '_', path.parent.parent.name + '_' + path.parent.name).strip('_')}",
        }
        for index, path in enumerate(selected, 1)
    ]


def find_answer(candidate_dir: Path) -> Path:
    for name in ANSWER_NAMES:
        direct = candidate_dir / name
        if direct.exists():
            return direct
        nested = list(candidate_dir.rglob(name))
        if nested:
            return nested[0]
    raise FileNotFoundError(f"No answer file found under {candidate_dir}")


def candidate_files(candidate_dir: Path, answer_path: Path) -> list[Path]:
    files: list[Path] = []
    answer_resolved = answer_path.resolve()
    for path in sorted(candidate_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == answer_resolved:
            continue
        if path.name.startswith("."):
            continue
        files.append(path)
    return files


def load_task(task_id: int, dataset_path: Path) -> dict[str, Any]:
    data = load_json(dataset_path)
    key = str(task_id)
    if key not in data:
        raise KeyError(f"Task id {task_id} not found in {dataset_path}")
    task = data[key]
    if not isinstance(task, dict):
        raise TypeError(f"Task {task_id} is not a dict in {dataset_path}")
    return task


def task_prompt(task: dict[str, Any]) -> str:
    dialogs = task.get("dialogs") or []
    if isinstance(dialogs, list) and dialogs:
        first = dialogs[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
    return str(task.get("prompt") or task.get("query") or "")


def stage_official_eval_input(
    *,
    candidate: dict[str, Any],
    repeat_index: int,
    out_dir: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    task_id = int(candidate["task_id"])
    candidate_dir = Path(candidate["candidate_dir"]).resolve()
    answer_path = find_answer(candidate_dir)

    repeat_dir = out_dir / f"task{task_id}" / f"candidate{candidate['candidate_index']}" / f"repeat{repeat_index}"
    builder_input_dir = repeat_dir / "official_builder_input"
    manus_result_dir = builder_input_dir / "manus_result"
    manus_task_dir = manus_result_dir / str(task_id)
    manus_task_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for src in candidate_files(candidate_dir, answer_path):
        # The official Manus/OpenClaw builders list top-level result files.
        # This staging only places candidate deliverables into that official
        # input shape; conversion/attachment selection is delegated to the
        # official builder below.
        dst = manus_task_dir / src.name
        shutil.copy2(src, dst)
        copied_files.append(
            {
                "source": str(src),
                "staged": str(dst.resolve()),
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size,
            }
        )

    task_ids_file = builder_input_dir / "selected_task_ids.txt"
    task_ids_file.write_text(f"{task_id}\n", encoding="utf-8")
    empty_runs_dir = builder_input_dir / "runs"
    empty_runs_dir.mkdir(parents=True, exist_ok=True)
    pack_path = (repeat_dir / "eval_pack.json").resolve()
    builder_artifacts_dir = repeat_dir / "official_builder_artifacts"
    builder_times_dir = repeat_dir / "official_builder_times"
    builder_run_path = repeat_dir / "official_builder_run.json"
    builder_script = GTA_REPO / "agent_app_eval" / "examples" / "build_gpt52_pack_from_manus_results.py"
    command = [
        str(scorer_python()),
        str(builder_script),
        "--dataset-end-json",
        str(dataset_path.resolve()),
        "--task-ids",
        str(task_ids_file.resolve()),
        "--runs-dir",
        str(empty_runs_dir.resolve()),
        "--manus-result-dir",
        str(manus_result_dir.resolve()),
        "--out-pack",
        str(pack_path),
        "--out-times-dir",
        str(builder_times_dir.resolve()),
        "--out-artifacts-dir",
        str(builder_artifacts_dir.resolve()),
    ]
    builder_env = os.environ.copy()
    path_entries = [
        str(OFFICIAL_COMMAND_SHIMS),
        str(LIBREOFFICE_PROGRAM_DIR),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps"),
        builder_env.get("PATH", ""),
    ]
    builder_env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)
    proc = subprocess.run(command, cwd=GTA_REPO, text=True, capture_output=True, check=False, env=builder_env)
    dump_json(
        builder_run_path,
        {
            "command": command,
            "cwd": str(GTA_REPO),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
    )
    if proc.returncode != 0 or not pack_path.exists():
        raise RuntimeError(f"Official builder failed for task {task_id} candidate {candidate['candidate_index']} repeat {repeat_index}. See {builder_run_path}")

    pack = load_json(pack_path)
    tasks = pack.get("tasks") if isinstance(pack, dict) else []
    attachments = []
    if isinstance(tasks, list) and tasks:
        first_task = tasks[0] if isinstance(tasks[0], dict) else {}
        attachments = first_task.get("attached_files") or []

    input_manifest = {
        **candidate,
        "repeat": repeat_index,
        "official_builder_script": str(builder_script),
        "official_builder_input_dir": str(builder_input_dir),
        "official_builder_artifacts_dir": str(builder_artifacts_dir),
        "official_builder_run_path": str(builder_run_path),
        "pack_path": str(pack_path),
        "answer_path": str(answer_path),
        "answer_sha256": sha256_file(answer_path),
        "copied_candidate_files": copied_files,
        "attached_files_from_official_builder": attachments,
        "num_attachments": len(attachments),
    }
    dump_json(repeat_dir / "eval_input_manifest.json", input_manifest)
    return input_manifest


def score_env(model: str, base_url: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["EVAL_OPENAI_MODEL"] = model
    if base_url:
        env["EVAL_OPENAI_BASE_URL"] = base_url
        env["OPENAI_BASE_URL"] = base_url
    if env.get("OPENAI_API_KEY") and not env.get("EVAL_OPENAI_API_KEY"):
        env["EVAL_OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    if env.get("EVAL_OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["EVAL_OPENAI_API_KEY"]
    env.pop("EVAL_DISABLE_FILE_UPLOAD", None)
    env.pop("EVAL_USE_CHAT_COMPLETIONS", None)
    opencompass_path = str((GTA_REPO / "opencompass").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = opencompass_path if not existing_pythonpath else opencompass_path + os.pathsep + existing_pythonpath
    env.setdefault("EVAL_DEBUG", "false")
    return env


def normalize_official_state(state_path: Path, task_id: int, out_path: Path) -> dict[str, Any]:
    state = load_json(state_path)
    task_results = state.get("task_results") if isinstance(state, dict) else {}
    rec = task_results.get(str(task_id)) if isinstance(task_results, dict) else None
    if not isinstance(rec, dict):
        raise RuntimeError(f"Official score state has no task result for {task_id}: {state_path}")
    if rec.get("status") != "ok":
        raise RuntimeError(f"Official score failed for task {task_id}: {rec.get('error')}")
    score = rec.get("score")
    detail = rec.get("detail")
    result = {
        "gpt_score": score,
        "sample_scores": [score] if isinstance(score, (int, float)) else [],
        "details": [detail] if isinstance(detail, dict) else [],
        "official_score_state": str(state_path),
        "official_task_record": rec,
    }
    dump_json(out_path, result)
    return result


def run_gemini_pack_score(pack_path: Path, out_path: Path, run_path: Path, model: str) -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY/gemini_api_key is required for --backend gemini.")

    import sys as _sys
    import types as py_types
    from google import genai
    from google.genai import types

    pack = load_json(pack_path)
    tasks = pack.get("tasks") or []
    if len(tasks) != 1:
        raise ValueError(f"Gemini calibration expects one-task packs, got {len(tasks)} in {pack_path}")
    task = tasks[0]
    task_id = int(task.get("task_id"))
    if str(GTA_REPO / "opencompass") not in _sys.path:
        _sys.path.insert(0, str(GTA_REPO / "opencompass"))
    from opencompass.datasets.gta_bench_v2 import GPTEvaluator  # type: ignore

    gemini_client = genai.Client(api_key=key)
    upload_logs: list[dict[str, Any]] = []
    call_logs: list[dict[str, Any]] = []

    def gemini_call(self: Any, prompt: str, attached_files: Any) -> str:
        def flatten_files(obj: Any) -> list[Any]:
            if obj is None:
                return []
            if isinstance(obj, list):
                out = []
                for item in obj:
                    out.extend(flatten_files(item))
                return out
            if isinstance(obj, tuple):
                out = []
                for item in obj:
                    out.extend(flatten_files(item))
                return out
            return [obj]

        contents: list[Any] = []
        for item in flatten_files(attached_files):
            path = item.get("path") if isinstance(item, dict) else item
            if not isinstance(path, str) or not path.strip():
                continue
            ftype = self._infer_file_type(path)
            final_path = self._prepare_upload_file(path)
            if not final_path:
                upload_logs.append({"path": path, "status": "skipped_by_official_prepare"})
                continue
            if ftype == "audio":
                contents.append(f"[Attached audio file exists: {os.path.basename(path)}]")
                upload_logs.append({"path": path, "status": "audio_existence_note"})
                continue
            if ftype == "video":
                for frame in self._extract_video_frames(path):
                    uploaded_frame = gemini_client.files.upload(file=str(frame))
                    contents.append(uploaded_frame)
                    upload_logs.append(
                        {
                            "path": str(frame),
                            "source_video": path,
                            "status": "ok",
                            "name": getattr(uploaded_frame, "name", None),
                            "uri": getattr(uploaded_frame, "uri", None),
                            "mime_type": getattr(uploaded_frame, "mime_type", None),
                        }
                    )
                continue
            uploaded = gemini_client.files.upload(file=str(final_path))
            contents.append(uploaded)
            upload_logs.append(
                {
                    "path": str(path),
                    "prepared_path": str(final_path),
                    "status": "ok",
                    "name": getattr(uploaded, "name", None),
                    "uri": getattr(uploaded, "uri", None),
                    "mime_type": getattr(uploaded, "mime_type", None),
                }
            )

        contents.append(prompt)
        response = None
        for attempt in range(1, 7):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                break
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                transient = any(token in message for token in ["429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand"])
                call_logs.append({"attempt": attempt, "error": message, "transient": transient})
                if not transient or attempt == 6:
                    raise
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
        if response is None:
            raise RuntimeError("Gemini call failed without a response.")
        raw_text = response.text or ""
        call_logs.append({"raw_text": raw_text})
        return raw_text

    evaluator = GPTEvaluator(mode="every", debug=False)
    evaluator._call_openai = py_types.MethodType(gemini_call, evaluator)

    def select_trace(assistant_outputs_field: Any) -> Any:
        if isinstance(assistant_outputs_field, list) and assistant_outputs_field:
            first = assistant_outputs_field[0]
            if isinstance(first, list):
                return first
        return assistant_outputs_field

    trace = select_trace(task.get("assistant_outputs"))
    try:
        scores = evaluator.score(
            assistant_outputs=[trace],
            checkpoints=[task.get("full_tree")],
            origin_prompt=[task.get("origin_prompt")],
            attached_files=[task.get("attached_files") or []],
            indices=[task_id],
        )
    except Exception:
        dump_json(
            run_path,
            {
                "backend": "gemini",
                "model": model,
                "pack_path": str(pack_path),
                "uploaded_files": upload_logs,
                "gemini_calls": call_logs,
                "official_gptevaluator_score_loop": True,
                "only_model_call_layer_replaced": True,
            },
        )
        raise
    result = dict(scores)
    result["gemini_score_state"] = {
        "model": model,
        "pack_path": str(pack_path),
        "uploaded_files": upload_logs,
        "num_upload_operations": len(upload_logs),
        "official_gptevaluator_score_loop": True,
        "only_model_call_layer_replaced": True,
    }
    dump_json(out_path, result)
    dump_json(
        run_path,
        {
            "backend": "gemini",
            "model": model,
            "pack_path": str(pack_path),
            "result_path": str(out_path),
            "upload_logs": upload_logs,
            "call_logs": call_logs,
            "official_gptevaluator_score_loop": True,
            "only_model_call_layer_replaced": True,
        },
    )
    return result


def run_rescore(
    candidate: dict[str, Any],
    repeat_index: int,
    out_dir: Path,
    dataset_path: Path,
    model: str,
    base_url: str | None,
    fail_on_upload_warning: bool,
    backend: str,
) -> dict[str, Any]:
    repeat_dir = out_dir / f"task{candidate['task_id']}" / f"candidate{candidate['candidate_index']}" / f"repeat{repeat_index}"
    normalized_path = (repeat_dir / "official_rescore_result.json").resolve()
    official_state_path = (repeat_dir / "official_gta_score_state.json").resolve()
    official_summary_path = (repeat_dir / "official_gta_score_summary.json").resolve()
    run_path = (repeat_dir / "score_run.json").resolve()
    if normalized_path.exists():
        return {"status": "cached", "result_path": str(normalized_path)}

    manifest = stage_official_eval_input(
        candidate=candidate,
        repeat_index=repeat_index,
        out_dir=out_dir,
        dataset_path=dataset_path,
    )
    if backend == "gemini":
        try:
            run_gemini_pack_score(Path(manifest["pack_path"]), normalized_path, run_path, model)
        except Exception as exc:
            dump_json(
                run_path,
                {
                    "backend": "gemini",
                    "model": model,
                    "eval_input_manifest": manifest,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return {
                "status": "error",
                "result_path": str(normalized_path),
                "run_path": str(run_path),
                "returncode": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"status": "ok", "result_path": str(normalized_path), "run_path": str(run_path)}

    command = [
        str(scorer_python()),
        str(OFFICIAL_SCORE_SCRIPT),
        "--in-pack",
        manifest["pack_path"],
        "--out",
        str(official_state_path),
        "--summary-out",
        str(official_summary_path),
        "--overwrite",
    ]
    proc = subprocess.run(command, cwd=GTA_REPO, text=True, capture_output=True, check=False, env=score_env(model, base_url))
    payload = {
        "command": command,
        "cwd": str(GTA_REPO),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "eval_input_manifest": manifest,
    }
    dump_json(run_path, payload)
    upload_warning = "[Upload Warning] Failed to upload" in proc.stdout or "[Upload Warning] Failed to upload" in proc.stderr
    if upload_warning and fail_on_upload_warning:
        return {
            "status": "error",
            "error_type": "upload_warning",
            "result_path": str(normalized_path),
            "official_state_path": str(official_state_path),
            "run_path": str(run_path),
            "returncode": proc.returncode,
        }
    if proc.returncode != 0 or not official_state_path.exists():
        return {
            "status": "error",
            "result_path": str(normalized_path),
            "official_state_path": str(official_state_path),
            "run_path": str(run_path),
            "returncode": proc.returncode,
        }
    try:
        normalize_official_state(official_state_path, int(candidate["task_id"]), normalized_path)
    except Exception as exc:
        return {
            "status": "error",
            "result_path": str(normalized_path),
            "official_state_path": str(official_state_path),
            "run_path": str(run_path),
            "returncode": proc.returncode,
            "normalization_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "ok",
        "result_path": str(normalized_path),
        "official_state_path": str(official_state_path),
        "run_path": str(run_path),
    }


def leaf_scores(result: dict[str, Any]) -> dict[str, float]:
    details = result.get("details") or []
    if not details:
        return {}
    scores = {}
    for node in details[0].get("nodes", []) or []:
        if "id" in node and isinstance(node.get("score"), (int, float)):
            scores[str(node["id"])] = float(node["score"])
    return scores


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def p75(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if value is not None and not math.isnan(float(value)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * 0.75
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return clean[int(pos)]
    fraction = pos - lower
    return clean[lower] * (1 - fraction) + clean[upper] * fraction


def compute_calibration(candidates: list[dict[str, Any]], out_dir: Path, repeats: int, model: str, backend: str) -> dict[str, Any]:
    candidate_rows = []
    checkpoint_rows = []
    errors = []

    for candidate in candidates:
        root_scores = []
        cp_values: dict[str, list[float]] = {}
        result_paths = []
        for repeat_index in range(1, repeats + 1):
            result_path = out_dir / f"task{candidate['task_id']}" / f"candidate{candidate['candidate_index']}" / f"repeat{repeat_index}" / "official_rescore_result.json"
            if not result_path.exists():
                errors.append({"candidate": candidate, "repeat": repeat_index, "error": "missing result"})
                continue
            result = load_json(result_path)
            if isinstance(result.get("gpt_score"), (int, float)):
                root_scores.append(float(result["gpt_score"]))
                result_paths.append(str(result_path))
            for cp_id, score in leaf_scores(result).items():
                cp_values.setdefault(cp_id, []).append(score)
        candidate_rows.append(
            {
                **candidate,
                "num_repeats": len(root_scores),
                "root_scores": root_scores,
                "sigma_root": sample_std(root_scores),
                "result_paths": result_paths,
            }
        )
        for cp_id, scores in sorted(cp_values.items()):
            checkpoint_rows.append(
                {
                    "task_id": candidate["task_id"],
                    "candidate_index": candidate["candidate_index"],
                    "candidate_id": candidate["candidate_id"],
                    "checkpoint_id": cp_id,
                    "num_repeats": len(scores),
                    "checkpoint_scores": scores,
                    "sigma_cp": sample_std(scores),
                }
            )

    sigma_root_values = [row["sigma_root"] for row in candidate_rows if row["num_repeats"] >= 2]
    sigma_cp_values = [row["sigma_cp"] for row in checkpoint_rows if row["num_repeats"] >= 2]
    sigma_root_p75 = p75(sigma_root_values)
    sigma_cp_p75 = p75(sigma_cp_values)
    epsilon_root = max(SCORE_RESOLUTION_ROOT, 2 * sigma_root_p75)
    epsilon_cp = max(SCORE_RESOLUTION_CP, 2 * sigma_cp_p75)

    return {
        "selection_config": {
            "epsilon_root": epsilon_root,
            "epsilon_cp": epsilon_cp,
            "selection_config_source": f"gta2_{backend}_{re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')}_attached_files_repeated_rescore_p75",
            "calibration_model": model,
            "calibration_backend": backend,
            "official_score_script": str(OFFICIAL_SCORE_SCRIPT),
            "official_evaluator": "vendor/GTA/opencompass/opencompass/datasets/gta_bench_v2.py",
            "score_resolution_root": SCORE_RESOLUTION_ROOT,
            "score_resolution_cp": SCORE_RESOLUTION_CP,
            "sigma_root_calibrated_p75": sigma_root_p75,
            "sigma_cp_calibrated_p75": sigma_cp_p75,
            "std_estimator": "sample_std_ddof_1",
            "percentile_method": "linear_interpolation_type7",
            "file_upload": "official_attached_files" if backend == "openai_official" else "gemini_files_api",
        },
        "summary": {
            "num_candidate_instances": len(candidate_rows),
            "num_checkpoint_instances": len(checkpoint_rows),
            "num_root_sigma_values": len(sigma_root_values),
            "num_cp_sigma_values": len(sigma_cp_values),
            "errors": errors,
        },
        "candidate_rows": candidate_rows,
        "checkpoint_rows": checkpoint_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run-root", type=Path, default=None)
    parser.add_argument("--dataset-end-json", type=Path, default=DEFAULT_DATASET_END_JSON)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "official_gta2_selection_epsilon_calibration")
    parser.add_argument("--task-ids", nargs="*", type=int, default=DEFAULT_TASK_IDS)
    parser.add_argument("--candidates-per-task", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--backend", choices=["openai_official", "gemini"], default="openai_official")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-upload-warning",
        action="store_true",
        help="Do not fail a run when the official evaluator prints upload warnings. Use only for debugging text-only fallbacks.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compute epsilon from completed rescoring jobs even if some jobs failed.",
    )
    args = parser.parse_args()

    if args.backend == "gemini" and args.model == "gpt-5.2":
        args.model = "gemini-2.5-pro"

    if args.candidate_run_root is None:
        args.candidate_run_root = latest_run_root(DEFAULT_RUN_FAMILY)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.run_dir if args.run_dir is not None else args.out_dir / timestamp
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for task_id in args.task_ids:
        candidates.extend(select_task_candidates(args.candidate_run_root, task_id, args.candidates_per_task))
    dump_json(
        out_dir / "candidate_manifest.json",
        {
            "candidate_run_root": str(args.candidate_run_root),
            "dataset_end_json": str(args.dataset_end_json),
            "official_score_script": str(OFFICIAL_SCORE_SCRIPT),
            "backend": args.backend,
            "model": args.model,
            "candidates": candidates,
        },
    )

    if args.prepare_only:
        for candidate in candidates:
            for repeat_index in range(1, args.repeats + 1):
                stage_official_eval_input(
                    candidate=candidate,
                    repeat_index=repeat_index,
                    out_dir=out_dir,
                    dataset_path=args.dataset_end_json,
                )
        print(json.dumps({"out_dir": str(out_dir), "num_candidates": len(candidates), "prepare_only": True}, ensure_ascii=False, indent=2))
        return

    if args.backend == "gemini":
        if not any(os.environ.get(name) for name in ["GEMINI_API_KEY", "gemini_api_key"]):
            raise EnvironmentError("GEMINI_API_KEY/gemini_api_key is required for Gemini calibration.")
    elif not any(os.environ.get(name) for name in ["OPENAI_API_KEY", "EVAL_OPENAI_API_KEY"]):
        raise EnvironmentError("OPENAI_API_KEY or EVAL_OPENAI_API_KEY is required for official GTA-2 GPT-5.2 calibration.")

    jobs = [(candidate, repeat_index) for candidate in candidates for repeat_index in range(1, args.repeats + 1)]
    run_records = []
    total = len(jobs)
    base_url = args.base_url.strip() or None

    def run_job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
        candidate, repeat_index = job
        record = run_rescore(
            candidate,
            repeat_index,
            out_dir,
            args.dataset_end_json,
            args.model,
            base_url,
            fail_on_upload_warning=not args.allow_upload_warning,
            backend=args.backend,
        )
        return {**candidate, "repeat": repeat_index, **record}

    if args.workers <= 1:
        for done, job in enumerate(jobs, 1):
            candidate, repeat_index = job
            print(f"[{done}/{total}] task={candidate['task_id']} candidate={candidate['candidate_index']} repeat={repeat_index}", flush=True)
            record = run_job(job)
            run_records.append(record)
            dump_json(out_dir / "run_records.json", run_records)
            if record.get("status") == "error" and record.get("run_path"):
                run_payload = load_json(Path(record["run_path"]))
                error_text = str(run_payload.get("stderr", "")) + "\n" + str(run_payload.get("stdout", ""))
                if "AuthenticationError" in error_text or "invalid_api_key" in error_text or "Incorrect API key" in error_text:
                    raise RuntimeError(f"Stopping calibration because authentication failed. See {record['run_path']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_job, job): job for job in jobs}
            for done, future in enumerate(as_completed(future_map), 1):
                candidate, repeat_index = future_map[future]
                record = future.result()
                print(
                    f"[{done}/{total}] task={candidate['task_id']} candidate={candidate['candidate_index']} repeat={repeat_index} status={record.get('status')}",
                    flush=True,
                )
                run_records.append(record)
                dump_json(out_dir / "run_records.json", run_records)
                if record.get("status") == "error" and record.get("run_path"):
                    run_payload = load_json(Path(record["run_path"]))
                    error_text = str(run_payload.get("stderr", "")) + "\n" + str(run_payload.get("stdout", ""))
                    if "AuthenticationError" in error_text or "invalid_api_key" in error_text or "Incorrect API key" in error_text:
                        raise RuntimeError(f"Stopping calibration because authentication failed. See {record['run_path']}")

    failed_records = [record for record in run_records if record.get("status") not in {"ok", "cached"}]
    if failed_records and not args.allow_partial:
        dump_json(out_dir / "run_failures.json", failed_records)
        raise RuntimeError(
            f"Stopping calibration because {len(failed_records)} / {len(run_records)} rescoring jobs failed. "
            f"See {out_dir / 'run_failures.json'}"
        )

    calibration = compute_calibration(candidates, out_dir, args.repeats, args.model, args.backend)
    dump_json(out_dir / "calibration_full.json", calibration)
    dump_json(out_dir / "selection_config.json", calibration["selection_config"])
    print(json.dumps({"out_dir": str(out_dir), **calibration["selection_config"], **calibration["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
