"""Integrity regressions for the publication audit; fixtures are not research data."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from verify_results import bootstrap, check_archives, paired_values, read_scores, score

HERE = Path(__file__).resolve().parent


class PublicationIntegrityTests(unittest.TestCase):
    def test_missing_archives_cannot_produce_a_reproduction_report(self):
        data = json.loads((HERE / "results_snapshot.json").read_text())
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "Required archives missing"):
                check_archives(data, Path(root))

    def test_pairing_rejects_equal_counts_with_different_ids_and_missing_pairs(self):
        with self.assertRaisesRegex(ValueError, "Unmatched IDs"):
            paired_values({"a": 0.1, "b": 0.2}, {"a": 0.1, "c": 0.2}, 2)
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            paired_values({"a": 0.1, "b": 0.2}, {"a": 0.1, "b": 0.2}, 3)

    def test_invalid_scores_are_rejected_and_empty_predictions_score_zero(self):
        for value in (None, True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score(value)
        self.assertEqual(score(0), 0)

    def test_duplicate_rows_fail_in_both_archived_schemas(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fixture.json"
            for row in ({"track_id": "a", "stems": {"other": {"note_f": 0.5}}},
                        {"track_id": "a", "stem": "other", "note_f": 0.5}):
                path.write_text(json.dumps({"per_track": [row, row]}))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    read_scores(path)

    def test_bootstrap_uses_paired_differences(self):
        _, a, b = paired_values({"a": 0.3, "b": 0.9}, {"b": 0.7, "a": 0.1}, 2)
        result = bootstrap([x-y for x, y in zip(a, b)])
        self.assertAlmostEqual(result["mean"], 0.2)
        for endpoint in result["ci"]:
            self.assertAlmostEqual(endpoint, 0.2)

    def test_legacy_key_metric_against_versioned_primary_implementation(self):
        # Load only the actual historical scorer definitions; avoid importing the
        # service/model environment merely to compare 24 x 24 key relationships.
        import mir_eval
        self.assertEqual(mir_eval.__version__, "0.8.2")
        path = HERE.parent / "eval" / "stats_significance.py"
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if
                    isinstance(node, ast.FunctionDef) and node.name in {"parse", "mirex"}
                    or isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "FLAT" for t in node.targets)]
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        namespace = {"NOTES": notes}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        legacy = namespace["mirex"]
        differences = []
        for i, ref_tonic in enumerate(notes):
            for ref_mode in ("major", "minor"):
                for j, est_tonic in enumerate(notes):
                    for est_mode in ("major", "minor"):
                        ref, est = f"{ref_tonic} {ref_mode}", f"{est_tonic} {est_mode}"
                        x, y = legacy(ref, est), mir_eval.key.weighted_score(ref, est)
                        if x != y:
                            self.assertEqual(ref_mode, est_mode)
                            self.assertEqual((j-i) % 12, 5)
                            self.assertEqual((x, y), (0.5, 0.0))
                            differences.append((ref, est))
        self.assertEqual(len(differences), 24)


if __name__ == "__main__":
    unittest.main()
