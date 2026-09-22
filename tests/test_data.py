"""Dataset integrity, isolation and deterministic reproduction."""

from pathlib import Path
import os
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mlcourse import metalearning as ml
from mlcourse.labs import load_meta_dataset
from mlcourse.uci import CONCRETE_COLUMNS, sources, load_dataset, normalise, validate_frame
from mlcourse.regression import concrete_split, concrete_groups


class DatasetIntegrityTests(unittest.TestCase):

    def test_recipe_groups_ignore_age_and_target(self):
        frame = pd.DataFrame(np.ones((4, 9)), columns=CONCRETE_COLUMNS)
        frame["age"] = [1, 7, 28, 90]
        frame["strength_mpa"] = [10, 20, 30, 40]
        self.assertEqual(concrete_groups(frame).nunique(), 1)
        frame.loc[3, "cement"] = 2
        self.assertEqual(concrete_groups(frame).nunique(), 2)

    def test_splits_are_deterministic_disjoint_and_exhaustive(self):
        frame = pd.DataFrame(np.ones((120, 9)), columns=CONCRETE_COLUMNS)
        frame["cement"] = np.repeat(np.arange(40), 3)
        frame["age"] = np.tile([1, 7, 28], 40)
        train, val, test, groups = concrete_split(frame)
        self.assertEqual(sorted(np.concatenate([train, val, test]).tolist()), list(range(120)))
        for a, b in [(train, val), (train, test), (val, test)]:
            self.assertTrue(set(groups.iloc[a]).isdisjoint(groups.iloc[b]))
        for a, b in zip((train, val, test), concrete_split(frame)[:3]):
            np.testing.assert_array_equal(a, b)

    def test_source_digest_failure_precedes_parsing(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            normalise(b"changed", sources(ROOT)[0])

    def test_incomplete_or_missing_source_is_rejected(self):
        spec = sources(ROOT)[0]
        frame = pd.DataFrame(np.ones((1030, 9)), columns=CONCRETE_COLUMNS)
        validate_frame(frame, spec)
        with self.assertRaises(ValueError):
            validate_frame(frame.iloc[:-1], spec)
        frame.loc[0, "water"] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(frame, spec)

    def test_absent_cache_never_downloads(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            shutil.copyfile(ROOT / "data/uci_sources.json", root / "data/uci_sources.json")
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
                with self.assertRaises(FileNotFoundError):
                    load_dataset(165, root)


class MetaDatasetTests(unittest.TestCase):

    def test_default_is_offline_and_ordered(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            table = ml.load_meta_dataset(ROOT)
        self.assertEqual(table.OpenMLID.tolist(), ml.read_manifest(ROOT).openml_id.tolist())
        self.assertEqual(set(ml.FEATURES), set((c for c in table if c.startswith("mf_"))))

    def test_cache_tampering_and_missing_fail_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                ROOT / "data/ho09", root / "data/ho09", ignore=shutil.ignore_patterns("raw")
            )
            csv = root / "data/ho09/meta_dataset.csv"
            csv.write_bytes(csv.read_bytes() + b"\n")
            with self.assertRaises(ValueError), patch(
                "urllib.request.urlopen", side_effect=AssertionError
            ):
                ml.load_meta_dataset(root)
            csv.unlink()
            with self.assertRaises(FileNotFoundError):
                ml.load_meta_dataset(root)

    def test_training_only_preprocessing_and_unknown_category(self):
        train = pd.DataFrame({"x": [0.0, 2.0, np.nan], "cat": ["a", "b", "a"]})
        test = pd.DataFrame({"x": [np.nan, 10000.0], "cat": ["unseen", "a"]})
        transformer = ml.preprocessing(train).fit(train)
        transformed = transformer.transform(test)
        self.assertEqual(transformed[0, 0], 1.0)
        np.testing.assert_array_equal(transformed[0, 1:], [0.0, 0.0])
        self.assertEqual(transformer.named_transformers_["numeric"].statistics_[0], 1.0)

    def test_test_rows_do_not_change_metafeatures_and_ties_are_exact(self):
        y = pd.Series(["a", "b"] * 30)
        X = pd.DataFrame({"x": np.zeros(60)})
        _, test = train_test_split(np.arange(60), test_size=0.3, random_state=42, stratify=y)
        entry = SimpleNamespace(openml_id=999, name="fixture")
        original = ml.summarise_one_dataset(entry, X, y)
        X.loc[test, "x"] = 10000.0
        changed = ml.summarise_one_dataset(entry, X, y)
        self.assertEqual(
            {k: original[k] for k in ml.FEATURES}, {k: changed[k] for k in ml.FEATURES}
        )
        self.assertEqual(original["BestModel"], "Tie")
        self.assertEqual(original["BestModelBinary"], 0)

    def test_failed_build_does_not_replace_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta_dataset.csv"
            p.write_text("published cache")
            with patch.object(ml, "load_raw", side_effect=RuntimeError("unavailable source")):
                with self.assertRaises(RuntimeError):
                    ml.rebuild_meta_dataset(ROOT, output_dir=tmp)
            self.assertEqual(p.read_text(), "published cache")

    def test_corrupt_raw_bytes_fail_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = next(ml.read_manifest(ROOT).itertuples(index=False))
            (Path(tmp) / f"{entry.openml_id}.pq").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                ml.load_raw(entry, tmp)

    def test_each_meta_prediction_holds_out_query(self):
        table = ml.load_meta_dataset(ROOT)
        _, pred, score, _ = ml.fit_meta_model(table)
        altered = table.copy()
        altered.loc[0, "BestModelBinary"] = 1 - int(table.loc[0, "BestModelBinary"])
        _, other, _, _ = ml.fit_meta_model(altered)
        self.assertEqual(pred.loc[0, "Predicted"], other.loc[0, "Predicted"])
        self.assertEqual(pred.loc[0, "Baseline"], other.loc[0, "Baseline"])
        self.assertTrue(0 <= score <= 1)

    def test_readable_meta_data_preserves_cached_outcomes(self):
        from mlcourse.metalearning import load_meta_dataset as load_internal

        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            shown, raw = (load_meta_dataset(), load_internal(ROOT))
        self.assertEqual(len(shown), 12)
        self.assertFalse(any((c.startswith("mf_") for c in shown)))
        np.testing.assert_array_equal(shown.tree_wins, raw.BestModelBinary)
        np.testing.assert_allclose(shown.accuracy_gap, shown.tree_accuracy - shown.stump_accuracy)


class DataWorkflowTests(unittest.TestCase):
    def test_preparation_failure_preserves_all_published_data(self):
        import prepare_data

        def fetch(root, output_dir, dataset_ids):
            target = output_dir / "ho05/concrete_compressive_strength.csv"
            target.parent.mkdir(parents=True)
            target.write_text("replacement")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data"
            existing = destination / "ho05/concrete_compressive_strength.csv"
            existing.parent.mkdir(parents=True)
            existing.write_text("published")
            with patch.object(prepare_data, "fetch_datasets", side_effect=fetch), patch.object(
                prepare_data, "rebuild_meta_dataset", side_effect=RuntimeError("unavailable")
            ):
                with self.assertRaises(RuntimeError):
                    prepare_data.prepare(
                        ["concrete", "meta-learning"], destination, Path(directory) / "raw"
                    )
            self.assertEqual(existing.read_text(), "published")
            self.assertFalse((destination / "ho09").exists())

    def test_preparation_uses_consistent_data_folders(self):
        import prepare_data

        def rebuild(root, raw_dir, output_dir, refresh):
            output_dir.mkdir(parents=True)
            (output_dir / "meta_dataset.csv").write_text("candidate")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(prepare_data, "rebuild_meta_dataset", side_effect=rebuild):
                prepare_data.prepare(["meta-learning"], destination, destination / "raw")
            self.assertEqual((destination / "ho09/meta_dataset.csv").read_text(), "candidate")

    def test_supplied_data_requires_explicit_publication(self):
        import prepare_data
        import contextlib
        import io

        with patch.object(prepare_data, "prepare") as prepare, contextlib.redirect_stderr(
            io.StringIO()
        ):
            for destination in [ROOT / "data", ROOT / "data/ho09", ROOT]:
                with self.assertRaises(SystemExit):
                    prepare_data.main(["--dataset", "all", "--output-dir", str(destination)])
            prepare.assert_not_called()
            prepare_data.main(["--dataset", "meta-learning", "--publish"])
            self.assertEqual(prepare.call_args.args[1], ROOT / "data")

    def test_shared_validator_checks_every_supplied_dataset(self):
        from validate_materials import validate_supplied_data

        validate_supplied_data(ROOT)

    def test_missing_supplied_uci_snapshot_fails_validation(self):
        from validate_materials import validate_uci_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / "data")
            (root / "data/ho05/concrete_compressive_strength.csv").unlink()
            with self.assertRaisesRegex(ValueError, "Missing supplied dataset"):
                validate_uci_data(root)

    def test_shared_validator_rejects_cache_tampering(self):
        from validate_materials import validate_meta_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "data/ho09", root / "data/ho09")
            path = root / "data/ho09/meta_dataset.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Cache hash mismatch"):
                validate_meta_dataset(root)


if __name__ == "__main__":
    unittest.main()
