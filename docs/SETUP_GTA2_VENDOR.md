# GTA-2 Vendor Setup

`vendor/GTA` is intentionally ignored by git because it contains the upstream
benchmark source, datasets, and generated artifacts.  To run official scoring,
the clean repo must have this directory populated locally.

Expected layout:

```text
vendor/GTA/
  agent_app_eval/
    examples/build_gpt52_pack_from_manus_results.py
  opencompass/
    data/gta_dataset_v2/end.json
    opencompass/datasets/gta_bench_v2.py
```

For local migration from the old workspace, copying `G:\project\GTA-2\vendor\GTA`
to `G:\project\Agent-Repair\vendor\GTA` is enough.

The committed baseline registry under `configs/heldout20/` no longer points to
old absolute paths.  It uses repo-relative baseline JSONs and attachment files
under `baselines/gemini_2_5_pro_lagent/`.
