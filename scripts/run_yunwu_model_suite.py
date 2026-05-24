"""Run one generation model's strict GTA-2 experiment suite through Yunwu.

The suite is sequential per model:

1. method-level baselines;
2. Full Ours with all four module ablations.

Launch three copies of this script in parallel with
``launch_yunwu_strict_gta2_matrix.py`` to run DeepSeek/Gemini/GPT-4o at the
same time while keeping each model's output directories and logs separate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_IDS = [7, 14, 45, 61, 62, 71, 79, 82, 88, 90, 92, 94, 98, 108, 111, 121, 122, 127, 133, 151]
DEFAULT_OUT_ROOT = ROOT / "runs" / "v3.1" / "yunwu_strict_gta2_matrix"
DIRECT_PROVIDER_ENV_KEYS = [
    "GEMINI_API_KEY",
    "gemini_api_key",
    "DEEPSEEK_API_KEY",
    "deepseek_api_key",
    "DASHSCOPE_API_KEY",
    "GLM_API_KEY",
    "glm_api_key",
    "AIHUBMIX_API_KEY",
    "aihubmix_api_key",
]


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def yunwu_env() -> dict[str, str]:
    env = os.environ.copy()
    api_key = env.get("OPENAI_API_KEY") or env.get("OPEN_API_KEY") or env.get("GPT_API_KEY") or env.get("GPT4O_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY/OPEN_API_KEY/GPT_API_KEY/GPT4O_API_KEY is required for Yunwu runs.")
    for key in DIRECT_PROVIDER_ENV_KEYS:
        env.pop(key, None)
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
    env["OPENAI_APP_CODE"] = env.get("OPENAI_APP_CODE") or "APP-10012fcb"
    env["EVAL_OPENAI_API_KEY"] = api_key
    env["EVAL_OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
    env["EVAL_OPENAI_MODEL"] = env.get("EVAL_OPENAI_MODEL") or "gpt-5.2"
    env["EVAL_DEBUG"] = "0"
    env["YUNWU_GEMINI_NATIVE"] = "1"
    env["GEMINI_REST_BASE_URL"] = "https://yunwu.ai/v1beta"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(name: str, command: list[str], out_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.out.log"
    stderr_path = out_dir / f"{name}.err.log"
    started = datetime.now().isoformat(timespec="seconds")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=stdout, stderr=stderr, check=False)
    record = {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    dump_json(out_dir / f"{name}.run.json", record)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with returncode {proc.returncode}; see {stderr_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-ids", nargs="*", type=int, default=DEFAULT_TASK_IDS)
    parser.add_argument("--scorer-model", default="gpt-5.2")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--skip-method-baselines", action="store_true")
    parser.add_argument("--skip-full-ours", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    env = yunwu_env()
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        python = Path(sys.executable)

    task_args = [str(task_id) for task_id in args.task_ids]
    manifest: dict[str, Any] = {
        "model_label": args.model_label,
        "model": args.model,
        "task_ids": args.task_ids,
        "scorer_model": args.scorer_model,
        "yunwu_base_url": env["OPENAI_BASE_URL"],
        "direct_provider_keys_removed": DIRECT_PROVIDER_ENV_KEYS,
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
    }
    dump_json(out_root / "suite_manifest.json", manifest)

    if not args.skip_method_baselines:
        command = [
            str(python),
            str(ROOT / "scripts" / "run_two_task_method_baselines.py"),
            "--model",
            args.model,
            "--scorer-model",
            args.scorer_model,
            "--methods",
            args.methods,
            "--max-tokens",
            str(args.max_tokens),
            "--force-baseline-rescore",
            "--out-dir",
            str(out_root / "method_baselines"),
        ]
        if task_args:
            command.extend(["--task-ids", *task_args])
        manifest["steps"].append(run_step("method_baselines", command, out_root / "logs", env))
        dump_json(out_root / "suite_manifest.json", manifest)

    if not args.skip_full_ours:
        command = [
            str(python),
            str(ROOT / "scripts" / "v3.1" / "run_full_ours_gemini_preservation_search.py"),
            "--model",
            args.model,
            "--scorer-backend",
            "responses",
            "--scorer-model",
            args.scorer_model,
            "--max-tokens",
            str(args.max_tokens),
            "--allow-fallback-selection-config",
            "--run-extra-ablations",
            "--out-dir",
            str(out_root / "full_ours_with_ablations"),
        ]
        if task_args:
            command.extend(["--task-ids", *task_args])
        manifest["steps"].append(run_step("full_ours_with_ablations", command, out_root / "logs", env))
        dump_json(out_root / "suite_manifest.json", manifest)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "complete"
    dump_json(out_root / "suite_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
