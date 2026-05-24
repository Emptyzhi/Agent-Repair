# GTA-2 Experiment KB v3.1

Last updated: 2026-05-19

Read this file before starting, rerunning, or summarizing any experiment.

## Scope

This KB is the current source of truth for:

- main-table requirements
- held-out20 experiment design
- ablation rules
- versioned launch layout

If a later message conflicts with this file, update this file first, then run.

## Main Table Requirements

Use the normal held-out20 benchmark as the main table.

Do not mix polluted-task stress tests into the main table. Those are appendix / robustness analyses only.

Main rows:

- Vanilla ReAct/Lagent
- ReAct + full retry
- Prompt-only self-refine
- Evaluator feedback retry
- Full Reflexion
- AgentFixer-guided independent repair
- Diagnosis-only repair
- Full ours
- Ours w/o artifact gate
- Ours w/o non-regression gate

Table rules:

- Compare methods on their final selected candidate, not on an intermediate attempt.
- Report the official root score for the selected candidate.
- Also report artifact-aware score for the same selected candidate.
- If the final candidate fails artifact gate, roll back to the task baseline score for artifact-aware reporting.
- Do not let a stale intermediate attempt override the final selected candidate.
- Keep model/backend and source run dir explicit in the table notes.

## Latest Experiment Design

Version v3.1 uses this logic:

1. Artifact gate remains a hard validity check.
2. Non-regression is no longer a final veto.
3. Non-regression provides preservation feedback during retry.
4. Final selection is root-score first, with stability only as a near-tie signal.
5. If a candidate is below baseline, retry with explicit checkpoint feedback.
6. If a candidate is above baseline, do not stop immediately. Run at least one refinement attempt.
7. If a candidate is above baseline but below the quality floor (`official root score < 7.0`), continue retry/refinement up to the attempt budget.
8. Full-ours refinement uses preservation feedback; the w/o-non-regression ablation may use the same attempt budget but must not receive preservation feedback.
9. Final evaluation uses the same artifact-aware rule for every method.

## Versioning Rule

Use versioned experiment folders under `scripts/`.

Current convention:

- KB: `docs/EXPERIMENT_KB_v3.1.md`
- launcher folder: `scripts/v3.1/`
- output folders: `runs/<method>_v3.1/` or `runs/v3.1/<method>/`

If the experiment design changes materially, bump both the KB version and the scripts folder version together.

## Required Pre-Run Checklist

Before any run:

1. Read this file.
2. Confirm the exact method version.
3. Confirm the baseline source run dir.
4. Confirm whether the run is main-table or appendix-only.
5. Confirm whether the final score is raw official or artifact-aware.
6. Save outputs under the matching versioned run directory.

## Current Interpretation

Current focus:

- main table should not reward fake gains from invalid artifacts
- non-regression should help preserve quality, not suppress clear root-score gains
- full-ours and ablations must share one candidate pool and one scoring policy

Operational note:

- the official scorer must run under `G:\project\GTA-2\.venv\Scripts\python.exe`; the bundled runtime used in some reruns does not ship `torch`, which caused the 2026-05-19 full DeepSeek rerun to silently fall back to baseline scores after scorer failure
