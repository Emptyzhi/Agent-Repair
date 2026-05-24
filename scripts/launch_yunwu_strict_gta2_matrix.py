"""Launch the strict GTA-2 Yunwu matrix for three generation models.

This launcher starts one suite process per generation model. Each suite then
runs method baselines followed by Full Ours and the module ablations. The child
environment is pinned to the Yunwu OpenAI-compatible relay and direct provider
keys are removed so Gemini/DeepSeek model names cannot silently use native APIs.
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
DEFAULT_MODEL_SPECS = [
    "deepseek_v4_pro=deepseek-v4-pro",
    "gemini_2_5_pro=gemini-2.5-pro",
    "gpt_4o=gpt-4o",
]
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


def first_env(env: dict[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name)
        if value:
            return value.strip()
    return ""


def parse_model_specs(raw_specs: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Model spec must be label=model, got: {raw}")
        label, model = raw.split("=", 1)
        label = label.strip()
        model = model.strip()
        if not label or not model:
            raise ValueError(f"Model spec must be label=model, got: {raw}")
        specs.append({"label": label, "model": model})
    return specs


def yunwu_child_env(scorer_model: str) -> dict[str, str]:
    env = os.environ.copy()
    api_key = first_env(env, "OPENAI_API_KEY", "OPEN_API_KEY", "GPT_API_KEY", "GPT4O_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY/OPEN_API_KEY/GPT_API_KEY/GPT4O_API_KEY is required for Yunwu runs.")
    removed_present = [key for key in DIRECT_PROVIDER_ENV_KEYS if env.get(key)]
    for key in DIRECT_PROVIDER_ENV_KEYS:
        env.pop(key, None)
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
    env["OPENAI_APP_CODE"] = env.get("OPENAI_APP_CODE") or "APP-10012fcb"
    env["EVAL_OPENAI_API_KEY"] = api_key
    env["EVAL_OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
    env["EVAL_OPENAI_MODEL"] = scorer_model
    env["EVAL_DEBUG"] = "0"
    env["YUNWU_GEMINI_NATIVE"] = "1"
    env["GEMINI_REST_BASE_URL"] = "https://yunwu.ai/v1beta"
    env["PYTHONIOENCODING"] = "utf-8"
    env["_YUNWU_REMOVED_DIRECT_PROVIDER_KEYS"] = ",".join(removed_present)
    return env


def python_exe() -> Path:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def make_suite_command(
    *,
    model_label: str,
    model: str,
    task_ids: list[int],
    scorer_model: str,
    methods: str,
    max_tokens: int,
    out_root: Path,
    skip_method_baselines: bool,
    skip_full_ours: bool,
) -> list[str]:
    command = [
        str(python_exe()),
        str(ROOT / "scripts" / "run_yunwu_model_suite.py"),
        "--model-label",
        model_label,
        "--model",
        model,
        "--scorer-model",
        scorer_model,
        "--methods",
        methods,
        "--max-tokens",
        str(max_tokens),
        "--out-root",
        str(out_root),
    ]
    if task_ids:
        command.extend(["--task-ids", *[str(task_id) for task_id in task_ids]])
    if skip_method_baselines:
        command.append("--skip-method-baselines")
    if skip_full_ours:
        command.append("--skip-full-ours")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="*", type=int, default=DEFAULT_TASK_IDS)
    parser.add_argument("--model-spec", action="append", dest="model_specs", default=None, help="label=model; repeatable.")
    parser.add_argument("--scorer-model", default="gpt-5.2")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--skip-method-baselines", action="store_true")
    parser.add_argument("--skip-full-ours", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_specs = parse_model_specs(args.model_specs or DEFAULT_MODEL_SPECS)
    env = yunwu_child_env(args.scorer_model)
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.out_root / run_label).resolve()
    logs_root = run_root / "launcher_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "task_ids": args.task_ids,
        "model_specs": model_specs,
        "scorer_model": args.scorer_model,
        "methods": args.methods,
        "max_tokens": args.max_tokens,
        "yunwu_base_url": env["OPENAI_BASE_URL"],
        "openai_app_code": env["OPENAI_APP_CODE"],
        "direct_provider_env_keys_removed": DIRECT_PROVIDER_ENV_KEYS,
        "direct_provider_env_keys_present_before_removal": env.get("_YUNWU_REMOVED_DIRECT_PROVIDER_KEYS", "").split(",")
        if env.get("_YUNWU_REMOVED_DIRECT_PROVIDER_KEYS")
        else [],
        "dry_run": args.dry_run,
        "processes": [],
    }

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for spec in model_specs:
        model_label = spec["label"]
        model = spec["model"]
        model_out_root = run_root / model_label
        command = make_suite_command(
            model_label=model_label,
            model=model,
            task_ids=args.task_ids,
            scorer_model=args.scorer_model,
            methods=args.methods,
            max_tokens=args.max_tokens,
            out_root=model_out_root,
            skip_method_baselines=args.skip_method_baselines,
            skip_full_ours=args.skip_full_ours,
        )
        stdout_path = logs_root / f"{model_label}.out.log"
        stderr_path = logs_root / f"{model_label}.err.log"
        record: dict[str, Any] = {
            "model_label": model_label,
            "model": model,
            "command": command,
            "out_root": str(model_out_root),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        if args.dry_run:
            record["pid"] = None
            record["status"] = "dry_run"
        else:
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=creationflags,
            )
            stdout.close()
            stderr.close()
            record["pid"] = proc.pid
            record["status"] = "started"
            record["started"] = datetime.now().isoformat(timespec="seconds")
        manifest["processes"].append(record)

    dump_json(run_root / "matrix_launch_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
