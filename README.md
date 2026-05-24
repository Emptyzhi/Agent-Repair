# Agent-Repair

Clean workspace for preservation-aware agent repair experiments on GTA-2.

This repo keeps only the code and small frozen inputs needed to run the current
Full Ours pipeline.  Historical experiment outputs stay out of git.

## What Is Included

- `scripts/repair_evaluator.py`: internal preservation-aware repair evaluator.
- `scripts/v3.1/run_full_ours_gemini_preservation_search.py`: Full Ours runner.
- `scripts/gta2_responses_scorer.py`: GTA-2 `GPTEvaluator` loop with Yunwu
  `/v1/responses` inline file inputs.
- `scripts/score_*_gta2_responses_evaluator.py`: candidate/prediction scoring
  wrappers for `gpt-5.2`.
- `configs/heldout20/`: frozen route and baseline registry snapshots with
  repo-relative paths.
- `baselines/gemini_2_5_pro_lagent/`: small baseline prediction/result JSONs
  plus required scoring attachments copied from the old workspace.
- `tests/test_repair_evaluator.py`: selection and repair evaluator unit tests.

## External Dependency

The official GTA-2 source and dataset are not committed.  Place or clone the
official repo at:

```text
vendor/GTA
```

The runner expects:

```text
vendor/GTA/opencompass/data/gta_dataset_v2/end.json
vendor/GTA/agent_app_eval/examples/build_gpt52_pack_from_manus_results.py
vendor/GTA/opencompass/opencompass/datasets/gta_bench_v2.py
```

## Setup

```powershell
cd G:\project\Agent-Repair
uv sync
```

Set Yunwu/OpenAI-compatible evaluator variables:

```powershell
$env:OPENAI_API_KEY = "<your-yunwu-key>"
$env:OPENAI_BASE_URL = "https://yunwu.ai/v1"
$env:EVAL_OPENAI_API_KEY = $env:OPENAI_API_KEY
$env:EVAL_OPENAI_BASE_URL = $env:OPENAI_BASE_URL
$env:OPENAI_APP_CODE = "APP-10012fcb"
```

## Smoke Tests

```powershell
python -m unittest tests.test_repair_evaluator -v
python -m py_compile scripts\gta2_responses_scorer.py scripts\score_candidate_with_gta2_responses_evaluator.py scripts\score_prediction_with_gta2_responses_evaluator.py scripts\v3.1\run_full_ours_gemini_preservation_search.py
```

## Run Task 98 and 108

```powershell
python scripts\v3.1\run_full_ours_gemini_preservation_search.py `
  --task-ids 98 108 `
  --model gpt-4o `
  --scorer-backend responses `
  --scorer-model gpt-5.2 `
  --allow-fallback-selection-config `
  --out-dir runs\v3.1\full_ours_gpt4o_responses_gpt52_2tasks
```

The final reported scores still come from the GTA-2 checkpoint tree and
`GPTEvaluator`; only the model-call transport is switched to `/v1/responses`
with inline file data because Yunwu's `/v1/files` route is unavailable in the
current account group.
