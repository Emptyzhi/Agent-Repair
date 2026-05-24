"""Score an OpenCompass prediction with GTA-2 GPTEvaluator and Gemini backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibrate_gta_official_selection_epsilon import run_gemini_pack_score
from score_candidate_with_official_evaluator import load_source_task


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    args = parser.parse_args()

    payload = load_json(args.prediction)
    sample = first_sample(payload)
    source_task = load_source_task(args.task_id)
    origin_prompt = sample.get("origin_prompt") or source_task["dialogs"][0]["content"]
    if isinstance(origin_prompt, list):
        origin_prompt = str(origin_prompt[0]) if origin_prompt else ""
    task = {
        "task_id": args.task_id,
        "origin_prompt": str(origin_prompt),
        "assistant_outputs": sample.get("assistant_outputs") or sample.get("prediction") or [],
        "full_tree": sample.get("full_tree") or source_task["sub_tasks"],
        "attached_files": sample.get("attached_files") or [],
    }
    pack = {
        "schema_version": "gta2_gemini_prediction_pack_v1",
        "generated_from": str(args.prediction.resolve()),
        "tasks": [task],
    }
    run_dir = args.out.resolve().parent
    pack_path = run_dir / "eval_pack.json"
    dump_json(pack_path, pack)
    run_path = run_dir / "gta2_gemini_score_run.json"
    result = run_gemini_pack_score(pack_path, args.out.resolve(), run_path, args.model)
    summary = {
        "task_id": args.task_id,
        "prediction": str(args.prediction.resolve()),
        "model": args.model,
        "official_score": result.get("gpt_score"),
        "out": str(args.out.resolve()),
        "pack_path": str(pack_path),
    }
    dump_json(run_dir / "official_rescore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
