"""Integrity regressions for the publication audit; fixtures are not research data."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from verify_results import (
    bootstrap,
    check_archives,
    check_snapshot,
    paired_values,
    read_scores,
    score,
)

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
            for row in (
                {"track_id": "a", "stems": {"other": {"note_f": 0.5}}},
                {"track_id": "a", "stem": "other", "note_f": 0.5},
            ):
                path.write_text(json.dumps({"per_track": [row, row]}))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    read_scores(path)

    def test_bootstrap_uses_paired_differences(self):
        _, a, b = paired_values({"a": 0.3, "b": 0.9}, {"b": 0.7, "a": 0.1}, 2)
        result = bootstrap([x - y for x, y in zip(a, b)])
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
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"parse", "mirex"}
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "FLAT" for t in node.targets)
        ]
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
                            self.assertEqual((j - i) % 12, 5)
                            self.assertEqual((x, y), (0.5, 0.0))
                            differences.append((ref, est))
        self.assertEqual(len(differences), 24)

    def test_snapshot_check_covers_every_published_section(self):
        """A corrupted value in any section make_figures.py publishes must fail, not only the
        four transcription rows (Codex review of PR #29)."""
        import copy

        good = json.loads((HERE / "results_snapshot.json").read_text())
        check_snapshot(good)
        cases = {
            "separation f1": lambda d: d["separation"][0].__setitem__("drums_f1", 99),
            "separation si-sdr": lambda d: d["separation"][1].__setitem__(
                "bass_si_sdr", float("nan")
            ),
            "bass delta": lambda d: d["bass_reference"].__setitem__("paired_delta", 0.5),
            "bass ci": lambda d: d["bass_reference"].__setitem__("paired_ci", [0.2, 0.3]),
            "contrast mean": lambda d: d["paired_contrasts"][1].__setitem__("mean", 0.1),
            "contrast ci": lambda d: d["paired_contrasts"][0].__setitem__("ci", [0.5, 0.6]),
            "contrast support": lambda d: d["paired_contrasts"][0].__setitem__("n", 150),
            "separator delta": lambda d: d["separator_paired_delta"]["drums_onset_f1"].__setitem__(
                "mean", 0.2
            ),
        }
        for name, corrupt in cases.items():
            bad = copy.deepcopy(good)
            corrupt(bad)
            with self.subTest(case=name), self.assertRaises(ValueError):
                check_snapshot(bad)

    def test_archive_check_recomputes_bass_and_separator_sections(self):
        """A snapshot edit that stays internally consistent must still fail against the archives."""
        import copy

        evidence = HERE / "evidence"
        good = json.loads((HERE / "results_snapshot.json").read_text())
        report = check_archives(good, evidence)
        self.assertEqual(report["bass_reference"]["n"], 143)
        self.assertEqual(
            set(report["separator"]["paired"]), {"drums_onset_f1", "other_f1", "bass_f1"}
        )
        bad = copy.deepcopy(good)
        bad["bass_reference"]["basic_pitch"] = 0.7890
        bad["bass_reference"]["paired_delta"] = round(bad["bass_reference"]["yourmt3"] - 0.7890, 4)
        check_snapshot(bad)  # internally consistent
        with self.assertRaisesRegex(ValueError, "bass_reference"):
            check_archives(bad, evidence)
        bad = copy.deepcopy(good)
        for r in bad["separation"]:
            if r["id"] == "S1":
                r["drums_f1"] = 0.5850
        bad["separator_paired_delta"]["drums_onset_f1"]["mean"] = round(0.5741 - 0.5850, 4)
        check_snapshot(bad)
        with self.assertRaisesRegex(ValueError, "separation S1 drums_f1"):
            check_archives(bad, evidence)


if __name__ == "__main__":
    unittest.main()
