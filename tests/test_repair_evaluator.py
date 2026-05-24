from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from repair_evaluator import evaluate_candidate, select_candidate_by_repair_evaluator  # noqa: E402


BASELINE_RESULT = {
    "gpt_score": 8.0,
    "details": [
        {
            "nodes": [
                {"id": "cp1", "score": 5.0, "analysis": "missing evidence"},
                {"id": "cp2", "score": 9.0, "analysis": "visual comparison is good"},
                {"id": "cp3", "score": 9.0, "analysis": "csv table is good"},
            ]
        }
    ],
}

TREE = [
    {"id": "cp1", "requirements": "Add source evidence section."},
    {"id": "cp2", "requirements": "Preserve visual comparison section."},
    {"id": "cp3", "requirements": "Preserve CSV table schema."},
]


def spec(units, schema=None):
    return {
        "artifacts": {
            "report.pdf": {"name": "report.pdf", "suffix": ".pdf", "schema": [], "units": units},
            "data.csv": {"name": "data.csv", "suffix": ".csv", "schema": schema or [], "units": []},
        }
    }


def selection_candidate(
    attempt_index: int,
    root_after: float,
    *,
    protected_risk: bool,
    artifact_pass: bool = True,
) -> dict:
    risk = "high" if protected_risk else "low"
    return {
        "attempt_index": attempt_index,
        "root_after": root_after,
        "target_gain": root_after,
        "artifact_pass": artifact_pass,
        "patch_mode": "structured_patch",
        "candidate_selection_signals": {
            "valid_for_selection": artifact_pass,
            "artifact_hard_fail": not artifact_pass,
            "protected_hard_violation": protected_risk,
            "protected_unit_risk": risk,
            "target_gain_observed": True,
            "target_status_score": 1,
            "collateral_damage_risk": risk,
            "collateral_damage_risk_rank": 2 if protected_risk else 0,
            "structured_patch": True,
            "retry_mode": "preservation_retry" if protected_risk else "target_retry",
        },
    }


class RepairEvaluatorTests(unittest.TestCase):
    def test_artifact_missing_fails_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_candidate(
                task_id=1,
                checkpoint_tree=TREE,
                baseline_official_result=BASELINE_RESULT,
                candidate_dir=Path(tmp),
                expected_files=["missing.pdf"],
                artifact_spec=spec([]),
            )
        self.assertFalse(report["artifact_pass"])
        self.assertIn("missing expected artifact", report["artifact_fail_reasons"][0])

    def test_csv_schema_change_is_reported_for_protected_unit(self):
        evidence_map = {
            "checkpoints": {
                "cp3": {
                    "status": "protected_high",
                    "evidence_units": [{"artifact": "data.csv", "unit_id": "csv.schema", "type": "csv_schema"}],
                }
            }
        }
        report = evaluate_candidate(
            task_id=1,
            checkpoint_tree=TREE,
            baseline_official_result=BASELINE_RESULT,
            candidate_dir=None,
            expected_files=["data.csv"],
            baseline_artifact_spec=spec([], schema=["a", "b"]),
            artifact_spec=spec([], schema=["a", "c"]),
            evidence_map=evidence_map,
        )
        self.assertTrue(report["csv_schema_changed"])
        self.assertEqual(report["retry_mode"], "preservation_retry")

    def test_protected_section_hash_change_is_reported(self):
        evidence_map = {
            "checkpoints": {
                "cp2": {
                    "status": "protected_high",
                    "evidence_units": [{"artifact": "report.pdf", "unit_id": "section.visual_comparison", "type": "section"}],
                }
            }
        }
        before = spec([{"unit_id": "section.visual_comparison", "content": "keep me"}])
        after = spec([{"unit_id": "section.visual_comparison", "content": "rewritten"}])
        report = evaluate_candidate(
            task_id=1,
            checkpoint_tree=TREE,
            baseline_official_result=BASELINE_RESULT,
            candidate_dir=None,
            expected_files=["report.pdf"],
            baseline_artifact_spec=before,
            artifact_spec=after,
            evidence_map=evidence_map,
        )
        self.assertTrue(report["protected_unit_hash_changed"])
        self.assertEqual(report["protected_checkpoint_status"]["cp2"], "risk_detected")

    def test_deleted_protected_section_is_reported(self):
        evidence_map = {
            "checkpoints": {
                "cp2": {
                    "status": "protected_high",
                    "evidence_units": [{"artifact": "report.pdf", "unit_id": "section.visual_comparison", "type": "section"}],
                }
            }
        }
        before = spec([{"unit_id": "section.visual_comparison", "content": "keep me"}])
        after = spec([])
        report = evaluate_candidate(
            task_id=1,
            checkpoint_tree=TREE,
            baseline_official_result=BASELINE_RESULT,
            candidate_dir=None,
            expected_files=["report.pdf"],
            baseline_artifact_spec=before,
            artifact_spec=after,
            evidence_map=evidence_map,
        )
        self.assertTrue(report["section_deleted"])
        self.assertEqual(report["collateral_damage_risk"], "high")

    def test_target_evidence_added_sets_target_gain(self):
        evidence_map = {
            "checkpoints": {
                "cp1": {
                    "status": "target_low",
                    "evidence_units": [{"artifact": "report.pdf", "unit_id": "section.source_appendix", "type": "section"}],
                }
            }
        }
        before = spec([])
        after = spec([{"unit_id": "section.source_appendix", "content": "new evidence"}])
        report = evaluate_candidate(
            task_id=1,
            checkpoint_tree=TREE,
            baseline_official_result=BASELINE_RESULT,
            candidate_dir=None,
            expected_files=["report.pdf"],
            baseline_artifact_spec=before,
            artifact_spec=after,
            evidence_map=evidence_map,
        )
        self.assertEqual(report["target_checkpoint_status"]["cp1"], "evidence_added")
        self.assertTrue(report["target_gain_observed"])

    def test_status_judge_cannot_emit_freeform_numeric_score(self):
        evidence_map = {
            "checkpoints": {
                "cp1": {
                    "status": "target_low",
                    "evidence_units": [{"artifact": "report.pdf", "unit_id": "section.source_appendix", "type": "section"}],
                }
            }
        }
        after = spec([{"unit_id": "section.source_appendix", "content": "new evidence"}])
        report = evaluate_candidate(
            task_id=1,
            checkpoint_tree=TREE,
            baseline_official_result=BASELINE_RESULT,
            candidate_dir=None,
            expected_files=["report.pdf"],
            artifact_spec=after,
            evidence_map=evidence_map,
            status_judge=lambda _cp, _units, _text: {"status": "8.7"},
        )
        self.assertEqual(report["target_checkpoint_status"]["cp1"], "unresolved")

    def test_root_gap_dominates_protected_risk(self):
        high_root_risky = selection_candidate(1, 9.29, protected_risk=True)
        low_root_safe = selection_candidate(2, 6.70, protected_risk=False)
        selected = select_candidate_by_repair_evaluator(
            [high_root_risky, low_root_safe],
            baseline_score=1.0,
            require_artifact_pass=True,
            damage_aware=True,
            epsilon_root=0.3,
        )
        self.assertEqual(selected["attempt_index"], 1)
        self.assertEqual(selected["selection_trace"]["near_best"], ["attempt1"])
        self.assertIn("attempt2", selected["selection_trace"]["excluded_by_root_gap"])

    def test_near_best_uses_repair_evaluator_tie_break(self):
        high_root_risky = selection_candidate(1, 9.10, protected_risk=True)
        low_root_safe = selection_candidate(2, 9.00, protected_risk=False)
        selected = select_candidate_by_repair_evaluator(
            [high_root_risky, low_root_safe],
            baseline_score=1.0,
            require_artifact_pass=True,
            damage_aware=True,
            epsilon_root=0.2,
        )
        self.assertEqual(selected["attempt_index"], 2)
        self.assertEqual(set(selected["selection_trace"]["near_best"]), {"attempt1", "attempt2"})

    def test_unresolved_or_risky_unit_does_not_hard_reject_root_best(self):
        risky = selection_candidate(1, 9.50, protected_risk=True)
        safe = selection_candidate(2, 8.70, protected_risk=False)
        selected = select_candidate_by_repair_evaluator(
            [risky, safe],
            baseline_score=1.0,
            require_artifact_pass=True,
            damage_aware=True,
            epsilon_root=0.2,
        )
        self.assertEqual(selected["attempt_index"], 1)

    def test_artifact_fail_is_hard_filter(self):
        invalid_high = selection_candidate(1, 9.50, protected_risk=False, artifact_pass=False)
        valid_lower = selection_candidate(2, 7.00, protected_risk=False, artifact_pass=True)
        selected = select_candidate_by_repair_evaluator(
            [invalid_high, valid_lower],
            baseline_score=1.0,
            require_artifact_pass=True,
            damage_aware=True,
            epsilon_root=0.2,
        )
        self.assertEqual(selected["attempt_index"], 2)


if __name__ == "__main__":
    unittest.main()
