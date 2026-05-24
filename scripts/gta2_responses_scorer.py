"""GTA-2 official evaluator with Responses inline file inputs.

This keeps the official GTA-2 GPTEvaluator scoring loop intact and replaces
only the model-call layer.  Yunwu's /v1/files route is not available in the
current account group, so attachments are sent as inline data URLs through
/v1/responses instead of uploaded first as file_ids.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import types as py_types
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
GTA_REPO = ROOT / "vendor" / "GTA"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def _responses_client() -> OpenAI:
    key = _first_env("EVAL_OPENAI_API_KEY", "OPENAI_API_KEY")
    base_url = _first_env("EVAL_OPENAI_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1"
    kwargs: dict[str, Any] = {"api_key": key, "base_url": base_url, "timeout": 180.0}
    app_code = _first_env("OPENAI_APP_CODE")
    if app_code or "yunwu.ai" in base_url.lower():
        kwargs["default_headers"] = {"APP-Code": app_code or "APP-10012fcb"}
    return OpenAI(**kwargs)


def _data_url(path: Path, fallback_mime: str = "application/octet-stream") -> str:
    mime = mimetypes.guess_type(str(path))[0] or fallback_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _flatten_files(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        out: list[Any] = []
        for item in obj:
            out.extend(_flatten_files(item))
        return out
    if isinstance(obj, tuple):
        out: list[Any] = []
        for item in obj:
            out.extend(_flatten_files(item))
        return out
    return [obj]


def _path_from_attachment(item: Any) -> str | None:
    if isinstance(item, dict):
        value = item.get("path") or item.get("file") or item.get("filename")
        return str(value) if value else None
    if isinstance(item, str) and item.strip():
        return item.strip()
    return None


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    try:
        payload = response.model_dump()
    except Exception:
        return str(response)
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def run_responses_pack_score(pack_path: Path, out_path: Path, run_path: Path, model: str) -> dict[str, Any]:
    key = _first_env("EVAL_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("EVAL_OPENAI_API_KEY/OPENAI_API_KEY is required for responses scoring.")

    if str(GTA_REPO / "opencompass") not in sys.path:
        sys.path.insert(0, str(GTA_REPO / "opencompass"))
    from opencompass.datasets.gta_bench_v2 import GPTEvaluator  # type: ignore

    pack = load_json(pack_path)
    tasks = pack.get("tasks") or []
    if len(tasks) != 1:
        raise ValueError(f"Responses scoring expects one-task packs, got {len(tasks)} in {pack_path}")
    task = tasks[0]
    task_id = int(task.get("task_id"))

    client = _responses_client()
    attachment_logs: list[dict[str, Any]] = []
    call_logs: list[dict[str, Any]] = []

    def responses_call(self: Any, prompt: str, attached_files: Any) -> str:
        content_blocks: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for item in _flatten_files(attached_files):
            raw_path = _path_from_attachment(item)
            if not raw_path:
                continue
            ftype = self._infer_file_type(raw_path)
            final_path = self._prepare_upload_file(raw_path)
            if not final_path:
                attachment_logs.append({"path": raw_path, "status": "skipped_by_official_prepare"})
                continue
            final = Path(final_path)
            try:
                if ftype == "audio":
                    content_blocks.append(
                        {
                            "type": "input_text",
                            "text": f"[Attached audio file exists: {os.path.basename(raw_path)}]",
                        }
                    )
                    attachment_logs.append({"path": raw_path, "prepared_path": str(final), "status": "audio_existence_note"})
                elif ftype == "video":
                    for frame in self._extract_video_frames(raw_path):
                        frame_path = Path(frame)
                        content_blocks.append({"type": "input_image", "image_url": _data_url(frame_path, "image/jpeg")})
                        attachment_logs.append(
                            {
                                "path": str(frame_path),
                                "source_video": raw_path,
                                "status": "inline_image_frame",
                                "size_bytes": frame_path.stat().st_size,
                            }
                        )
                elif ftype == "image":
                    content_blocks.append({"type": "input_image", "image_url": _data_url(final, "image/png")})
                    attachment_logs.append({"path": raw_path, "prepared_path": str(final), "status": "inline_image", "size_bytes": final.stat().st_size})
                else:
                    content_blocks.append(
                        {
                            "type": "input_file",
                            "filename": final.name,
                            "file_data": _data_url(final, "application/pdf"),
                        }
                    )
                    attachment_logs.append({"path": raw_path, "prepared_path": str(final), "status": "inline_file", "size_bytes": final.stat().st_size})
            except Exception as exc:
                attachment_logs.append({"path": raw_path, "prepared_path": str(final), "status": "inline_failed", "error": f"{type(exc).__name__}: {exc}"})

        for attempt in range(1, 5):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": [{"role": "user", "content": content_blocks}],
                    "temperature": 0.2,
                }
                try:
                    response = client.responses.create(**kwargs)
                except Exception as exc:
                    message = str(exc)
                    if "temperature" not in message.lower():
                        raise
                    kwargs.pop("temperature", None)
                    response = client.responses.create(**kwargs)
                text = _extract_response_text(response)
                call_logs.append({"attempt": attempt, "num_content_blocks": len(content_blocks), "raw_text": text})
                return text
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                transient = any(token in message for token in ["429", "500", "502", "503", "504", "timeout", "temporarily", "rate"])
                call_logs.append({"attempt": attempt, "error": message, "transient": transient})
                if not transient or attempt == 4:
                    raise
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
        raise RuntimeError("Responses scorer failed without returning text.")

    evaluator = GPTEvaluator(mode="every", debug=False)
    evaluator._call_openai = py_types.MethodType(responses_call, evaluator)

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
                "backend": "responses_inline_file",
                "model": model,
                "pack_path": str(pack_path),
                "attachment_logs": attachment_logs,
                "call_logs": call_logs,
                "official_gptevaluator_score_loop": True,
                "only_model_call_layer_replaced": True,
            },
        )
        raise

    result = dict(scores)
    result["responses_score_state"] = {
        "model": model,
        "pack_path": str(pack_path),
        "attachment_logs": attachment_logs,
        "num_inline_attachments": len(attachment_logs),
        "official_gptevaluator_score_loop": True,
        "only_model_call_layer_replaced": True,
    }
    dump_json(out_path, result)
    dump_json(
        run_path,
        {
            "backend": "responses_inline_file",
            "model": model,
            "pack_path": str(pack_path),
            "result_path": str(out_path),
            "attachment_logs": attachment_logs,
            "call_logs": call_logs,
            "official_gptevaluator_score_loop": True,
            "only_model_call_layer_replaced": True,
        },
    )
    return result
