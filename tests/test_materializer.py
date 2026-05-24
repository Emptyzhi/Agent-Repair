from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_heldout20_full_ours_candidate_generation import _render_units, build_artifact_spec  # noqa: E402


class MaterializerTests(unittest.TestCase):
    def test_same_stem_docx_pdf_share_rich_content(self):
        rich_report = "\n".join(
            [
                "# Full Report",
                "Step-by-step protocol details.",
                "Safe space layout details.",
                "Medication dosage and timing details.",
            ]
            * 20
        )
        payload = {
            "final_answer_markdown": "Short final answer that names the files.",
            "deliverables": {
                "repaired_report.docx": rich_report,
            },
        }

        spec = build_artifact_spec(payload, ["repaired_report.docx", "repaired_report.pdf"])
        docx_text = _render_units(spec["artifacts"]["repaired_report.docx"]["units"])
        pdf_text = _render_units(spec["artifacts"]["repaired_report.pdf"]["units"])

        self.assertIn("Step-by-step protocol details.", docx_text)
        self.assertIn("Step-by-step protocol details.", pdf_text)
        self.assertEqual(docx_text, pdf_text)


if __name__ == "__main__":
    unittest.main()
