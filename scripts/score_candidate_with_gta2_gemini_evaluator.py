"""Score a candidate with GTA-2 official builder/evaluator and Gemini backend.

The scoring protocol intentionally reuses the official GTA builder and
`GPTEvaluator`; only the model-call layer is replaced with Gemini.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibrate_gta_official_selection_epsilon import (
    DEFAULT_DATASET_END_JSON,
    run_gemini_pack_score,
    stage_official_eval_input,
)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--dataset-end-json", type=Path, default=DEFAULT_DATASET_END_JSON)
    args = parser.parse_args()

    run_dir = args.out.resolve().parent
    manifest = stage_official_eval_input(
        candidate={
            "task_id": args.task_id,
            "candidate_index": 1,
            "candidate_dir": str(args.candidate_dir.resolve()),
            "candidate_id": f"task{args.task_id}_candidate",
        },
        repeat_index=1,
        out_dir=run_dir,
        dataset_path=args.dataset_end_json,
    )
    run_path = run_dir / "gta2_gemini_score_run.json"
    result = run_gemini_pack_score(Path(manifest["pack_path"]), args.out.resolve(), run_path, args.model)
    summary = {
        "task_id": args.task_id,
        "candidate_dir": str(args.candidate_dir.resolve()),
        "model": args.model,
        "official_score": result.get("gpt_score"),
        "out": str(args.out.resolve()),
        "pack_path": manifest["pack_path"],
        "official_builder_script": manifest["official_builder_script"],
    }
    dump_json(run_dir / "official_rescore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
