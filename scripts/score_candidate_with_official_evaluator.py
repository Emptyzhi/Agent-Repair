"""Score an arbitrary repaired candidate with the official GTA evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from scorer_runtime import scorer_python


ROOT = Path(__file__).resolve().parents[1]
OPENCOMPASS_DIR = ROOT / "vendor" / "GTA" / "opencompass"
TASK_DATASETS = {
    15: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_one" / "end.json", "15"),
    1: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task1" / "end.json", "0"),
    12: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task12" / "end.json", "0"),
    20: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task20" / "end.json", "0"),
    22: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task22" / "end.json", "0"),
    21: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task21" / "end.json", "0"),
    24: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task24" / "end.json", "0"),
    28: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task28" / "end.json", "0"),
    57: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task57" / "end.json", "0"),
    65: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task65" / "end.json", "0"),
    98: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task98" / "end.json", "0"),
    107: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task107" / "end.json", "0"),
    122: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task122" / "end.json", "0"),
    47: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task47" / "end.json", "0"),
    66: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task66" / "end.json", "0"),
    70: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task70" / "end.json", "0"),
    71: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task71" / "end.json", "0"),
    88: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task88" / "end.json", "0"),
    93: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task93" / "end.json", "0"),
    95: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task95" / "end.json", "0"),
    129: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task129" / "end.json", "0"),
    108: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task108" / "end.json", "0"),
    46: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task46" / "end.json", "0"),
    53: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task53" / "end.json", "0"),
    58: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task58" / "end.json", "0"),
    63: (OPENCOMPASS_DIR / "data" / "gta_dataset_v2_task63" / "end.json", "0"),
}
DEFAULT_SOURCE_DATASET = OPENCOMPASS_DIR / "data" / "gta_dataset_v2" / "end.json"
ANSWER_NAMES = [
    "repaired_final_answer.md",
    "diagnosis_only_answer.md",
    "prompt_only_answer.md",
    "final.txt",
]


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def configure_openai_compatible_defaults() -> None:
    eval_key = first_env("EVAL_OPENAI_API_KEY")
    eval_base = first_env("EVAL_OPENAI_BASE_URL")
    if eval_key and eval_base:
        os.environ["EVAL_OPENAI_API_KEY"] = eval_key
        os.environ["OPENAI_API_KEY"] = eval_key
        os.environ["EVAL_OPENAI_BASE_URL"] = eval_base
        os.environ["OPENAI_BASE_URL"] = eval_base
        os.environ.setdefault("EVAL_OPENAI_MODEL", "deepseek-v4-pro")
        os.environ.setdefault("EVAL_DISABLE_FILE_UPLOAD", "true")
        os.environ.setdefault("EVAL_DEBUG", "false")
        return

    deepseek_key = first_env("DEEPSEEK_API_KEY", "deepseek_api_key")
    dashscope_key = first_env("DASHSCOPE_API_KEY")
    glm_key = first_env("GLM_API_KEY", "glm_api_key")

    if dashscope_key:
        os.environ.setdefault("EVAL_OPENAI_API_KEY", dashscope_key)
        os.environ.setdefault("OPENAI_API_KEY", dashscope_key)
        os.environ.setdefault("EVAL_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        os.environ.setdefault("EVAL_OPENAI_MODEL", os.environ.get("DASHSCOPE_MODEL", "qwen3-max"))
        os.environ.setdefault("EVAL_FILE_PURPOSE", "file-extract")
    elif glm_key:
        base_url = os.environ.get("GLM_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
        model = first_env("GLM_MODEL", "glm_model") or "glm-4.6v-flashx"
        os.environ["GLM_API_KEY"] = glm_key
        os.environ["OPENAI_API_KEY"] = glm_key
        os.environ["EVAL_OPENAI_API_KEY"] = glm_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["EVAL_OPENAI_BASE_URL"] = base_url
        os.environ["EVAL_OPENAI_MODEL"] = model
        os.environ.setdefault("GLM_OPENAI_CHAT_COMPLETIONS", f"{base_url}/chat/completions")
        os.environ.setdefault("GLM_THINKING", "enabled")
    elif deepseek_key:
        base_url = os.environ.get("DEEPSEEK_OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
        model = first_env("DEEPSEEK_MODEL", "deepseek_model") or "deepseek-v4-pro"
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        os.environ["OPENAI_API_KEY"] = deepseek_key
        os.environ["EVAL_OPENAI_API_KEY"] = deepseek_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["EVAL_OPENAI_BASE_URL"] = base_url
        os.environ["EVAL_OPENAI_MODEL"] = model
        os.environ.setdefault("EVAL_DISABLE_FILE_UPLOAD", "true")

    os.environ.setdefault("EVAL_DEBUG", "false")


def ensure_project_venv() -> None:
    project_python = scorer_python().resolve()
    try:
        current_python = Path(sys.executable).resolve()
    except Exception:
        return
    if project_python.exists() and current_python != project_python:
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env.setdefault("PYTHONNOUSERSITE", "1")
        os.execve(str(project_python), [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_source_task(task_id: int) -> dict[str, Any]:
    source = load_json(DEFAULT_SOURCE_DATASET)
    key = str(task_id)
    if key not in source:
        raise KeyError(f"Task id {task_id} not found in {DEFAULT_SOURCE_DATASET}")
    return source[key]


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_answer(candidate_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    for name in ANSWER_NAMES:
        direct = candidate_dir / name
        if direct.exists():
            return direct
        nested = list(candidate_dir.rglob(name))
        if nested:
            return nested[0]
    raise FileNotFoundError(f"No answer file found under {candidate_dir}")


def find_supported_attachments(candidate_dir: Path) -> list[str]:
    supported = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
    return [str(path) for path in candidate_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported]


def score_with_official_evaluator(
    trace: list[dict[str, str]],
    checkpoint_tree: Any,
    origin_prompt: str,
    sample_index: int,
    attached_files: list[str] | None = None,
) -> dict[str, Any]:
    ensure_project_venv()
    __import__("torch")
    sys.path.insert(0, str(OPENCOMPASS_DIR))
    old_cwd = Path.cwd()
    os.chdir(OPENCOMPASS_DIR)
    configure_openai_compatible_defaults()
    try:
        from opencompass.datasets.gta_bench_v2 import GPTEvaluator

        evaluator = GPTEvaluator(mode="every", proxy=os.environ.get("EVAL_PROXY"))
        return evaluator.score(
            predictions=[trace],
            checkpoints=[checkpoint_tree],
            origin_prompt=[origin_prompt],
            assistant_outputs=[trace],
            attached_files=[attached_files or []],
            indices=[sample_index],
        )
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--answer", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--attach-files",
        action="store_true",
        help="Upload supported local artifacts to the evaluator. Off by default because DashScope Responses compatibility rejects some file/image blocks.",
    )
    args = parser.parse_args()

    if args.task_id in TASK_DATASETS:
        dataset_path, key = TASK_DATASETS[args.task_id]
        task = load_json(dataset_path)[key]
    else:
        task = load_source_task(args.task_id)
    answer_path = find_answer(args.candidate_dir.resolve(), args.answer)
    answer = answer_path.read_text(encoding="utf-8-sig", errors="ignore")
    trace = [{"role": "assistant", "content": answer}]
    attachments = find_supported_attachments(args.candidate_dir.resolve()) if args.attach_files else []
    result = score_with_official_evaluator(
        trace=trace,
        checkpoint_tree=task["sub_tasks"],
        origin_prompt=task["dialogs"][0]["content"],
        sample_index=args.task_id,
        attached_files=attachments,
    )
    out = args.out or args.candidate_dir / "official_rescore_result.json"
    dump_json(out, result)
    summary = {
        "task_id": args.task_id,
        "candidate_dir": str(args.candidate_dir),
        "answer_path": str(answer_path),
        "official_score": result.get("gpt_score"),
        "out": str(out),
    }
    dump_json(out.parent / "official_rescore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
