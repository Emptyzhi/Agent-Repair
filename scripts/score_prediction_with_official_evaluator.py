"""Score an OpenCompass prediction JSON with the GTA official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from score_candidate_with_official_evaluator import (
    dump_json,
    load_json,
    load_source_task,
    score_with_official_evaluator,
)


def first_sample(payload: dict[str, Any]) -> dict[str, Any]:
    if "prediction" in payload:
        return payload
    for key in sorted(payload.keys(), key=lambda value: str(value)):
        value = payload[key]
        if isinstance(value, dict) and "prediction" in value:
            return value
    raise KeyError("No sample with a prediction field found")


def extract_trace(sample: dict[str, Any]) -> list[dict[str, Any]]:
    prediction = sample.get("prediction")
    if isinstance(prediction, list) and prediction:
        first = prediction[0]
        if isinstance(first, list):
            return first
        if isinstance(first, dict):
            return prediction
    assistant_outputs = sample.get("assistant_outputs")
    if isinstance(assistant_outputs, list) and assistant_outputs:
        first = assistant_outputs[0]
        if isinstance(first, list):
            return first
    raise ValueError("Unsupported prediction JSON shape")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = load_json(args.prediction)
    sample = first_sample(payload)
    trace = extract_trace(sample)
    source_task = load_source_task(args.task_id)
    checkpoint_tree = sample.get("full_tree") or source_task["sub_tasks"]
    origin_prompt = sample.get("origin_prompt") or [source_task["dialogs"][0]["content"]]
    if isinstance(origin_prompt, list):
        origin_prompt_text = str(origin_prompt[0]) if origin_prompt else ""
    else:
        origin_prompt_text = str(origin_prompt)
    attached_files = sample.get("attached_files") or []

    result = score_with_official_evaluator(
        trace=trace,
        checkpoint_tree=checkpoint_tree,
        origin_prompt=origin_prompt_text,
        sample_index=args.task_id,
        attached_files=attached_files,
    )
    dump_json(args.out, result)
    summary = {
        "task_id": args.task_id,
        "prediction": str(args.prediction),
        "official_score": result.get("gpt_score"),
        "out": str(args.out),
    }
    dump_json(args.out.parent / "official_rescore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
