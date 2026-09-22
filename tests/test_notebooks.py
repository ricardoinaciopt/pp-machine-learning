"""Setup, notebook contracts, execution selection and interactive tools."""

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
from mlcourse import setup
from mlcourse.labs import cluster_summary
from mlcourse.regression import plot_concrete_surfaces


class NotebookContractTests(unittest.TestCase):

    def test_root_independent_of_working_directory(self):
        with patch.dict(os.environ, {"MLCOURSE_ROOT": str(ROOT)}):
            self.assertEqual(setup.find_root("/tmp"), ROOT)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(setup.find_root("/tmp"), ROOT)

    def test_linter_detects_solution_drift(self):
        import tempfile
        import nbformat

        sys.path.insert(0, str(ROOT / "scripts"))
        from validate_materials import validate_notebooks as validate

        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            for source in ROOT.glob("hands-on/ho*/**/notebook.ipynb"):
                target = copy / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            path = copy / "hands-on/ho03/solutions/notebook.ipynb"
            book = nbformat.read(path, as_version=4)
            book.cells[3].source = book.cells[3].source.replace("## 1.", "## 99.")
            nbformat.write(book, path)
            self.assertTrue(
                any(("structure or code mismatch" in error for error in validate(copy)))
            )

    def test_local_setup_never_installs(self):
        with patch.object(setup.importlib.util, "find_spec", return_value=None), patch.object(
            setup, "in_colab", return_value=False
        ), patch.object(setup.subprocess, "run") as install:
            with self.assertRaisesRegex(RuntimeError, "Course packages are missing"):
                setup.setup_notebook()
            install.assert_not_called()

    def test_colab_installs_only_missing_dependencies(self):
        fake = types.ModuleType("google.colab")
        fake.output = types.SimpleNamespace(enable_custom_widget_manager=lambda: None)
        with patch.object(
            setup.importlib.util,
            "find_spec",
            side_effect=lambda name: None if name == "ipywidgets" else object(),
        ), patch.object(setup, "in_colab", return_value=True), patch.object(
            setup.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)
        ) as install, patch.dict(
            sys.modules, {"google.colab": fake}
        ):
            setup.setup_notebook()
            command = install.call_args.args[0]
            self.assertEqual(command[5:], [setup.DEPENDENCIES["ipywidgets"]])

    def test_colab_complete_environment_does_not_install(self):
        fake = types.ModuleType("google.colab")
        fake.output = types.SimpleNamespace(enable_custom_widget_manager=lambda: None)
        with patch.object(setup.importlib.util, "find_spec", return_value=object()), patch.object(
            setup, "in_colab", return_value=True
        ), patch.object(setup.subprocess, "run") as install, patch.dict(
            sys.modules, {"google.colab": fake}
        ):
            setup.setup_notebook()
            install.assert_not_called()

    def test_noise_is_not_an_additional_cluster(self):
        scores = cluster_summary(np.array([[0, 0], [1, 1], [100, 100]]), [0, 0, -1])
        self.assertEqual(scores["clusters"], 1)
        self.assertEqual(scores["noise_count"], 1)
        self.assertTrue(np.isnan(scores["silhouette"]))

    def test_notebook_conventions(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from validate_materials import validate_notebooks as validate

        self.assertEqual(validate(), [])


class InteractiveSemanticsTests(unittest.TestCase):

    def test_threshold_changes_predictions_without_refitting(self):
        from mlcourse.labs import make_imbalance_data, split_classification
        from mlcourse.widgets import interactive_threshold
        from sklearn.linear_model import LogisticRegression
        from IPython.utils.capture import capture_output
        import matplotlib.pyplot as plt

        with capture_output(), patch.object(plt, "show", side_effect=lambda: plt.close("all")):
            lab = interactive_threshold(*split_classification(*make_imbalance_data()))
            with patch.object(
                LogisticRegression, "fit", side_effect=AssertionError("threshold refitted model")
            ):
                high = lab.render(threshold=0.5, class_weight=None).iloc[0]
                low = lab.render(threshold=0.2, class_weight=None).iloc[0]
            self.assertGreater(low.recall, high.recall)
            self.assertLess(low.precision, high.precision)

    def test_svm_control_relevance_and_callbacks(self):
        from mlcourse.labs import load_course_data, split_classification
        from mlcourse.widgets import interactive_svm
        from IPython.utils.capture import capture_output
        import matplotlib.pyplot as plt

        data = load_course_data("ds2")
        with capture_output(), patch.object(plt, "show", side_effect=lambda: plt.close("all")):
            lab = interactive_svm(*split_classification(data.iloc[:, :2], data.iloc[:, -1]))
            lab.controls["kernel"].value = "linear"
            self.assertTrue(lab.controls["gamma"].disabled)
            self.assertTrue(lab.controls["degree"].disabled)
            lab.controls["kernel"].value = "poly"
            self.assertFalse(lab.controls["degree"].disabled)
            result = lab.render(kernel="poly", C=1.0, gamma=0.1, degree=2)
            self.assertTrue(0 <= result["test_accuracy"] <= 1)
            self.assertEqual(lab.controls["degree"].style.description_width, "180px")

    def test_surface_controls_produce_two_matching_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        X = pd.DataFrame({"cement": np.linspace(100, 500, 30), "age": np.tile([7, 28, 90], 10)})
        y = pd.Series(np.linspace(10, 70, 30))
        for depth, leaf in [(1, 1), (6, 10)]:
            fig = plot_concrete_surfaces(X, y, depth, leaf)
            self.assertEqual(len(fig.axes), 3)
            self.assertEqual(
                fig.axes[0].collections[0].get_clim(), fig.axes[1].collections[0].get_clim()
            )
            plt.close(fig)


class ExecutionWorkflowTests(unittest.TestCase):
    def test_select_all_pair_and_single(self):
        from check_notebooks import select_notebooks

        self.assertEqual(len(select_notebooks()), 18)
        pair = select_notebooks(ho=9)
        self.assertEqual(len(pair), 2)
        self.assertEqual(select_notebooks([pair[0]]), [pair[0]])

    def test_missing_pair_and_unknown_notebook_fail(self):
        from check_notebooks import select_notebooks

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                select_notebooks(root=directory)
        with self.assertRaises(ValueError):
            select_notebooks([ROOT / "README.md"])
        with self.assertRaises(ValueError):
            select_notebooks([ROOT / "hands-on/ho01/notebook.ipynb"], ho=1)

    def test_missing_notebook_reports_validation_error(self):
        from validate_materials import validate_notebooks

        with tempfile.TemporaryDirectory() as directory:
            errors = validate_notebooks(Path(directory))
        self.assertTrue(any("invalid or missing notebook" in error for error in errors))

    def test_broken_image_reference_reports_source(self):
        from validate_materials import validate_media

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hands-on/ho02/handout.qmd"
            source.parent.mkdir(parents=True)
            source.write_text("![Plot](missing.png)")
            self.assertEqual(
                validate_media(root), ["hands-on/ho02/handout.qmd: missing image missing.png"]
            )


class MaintenanceWorkflowTests(unittest.TestCase):
    def test_failed_validation_prevents_rendering(self):
        import build_materials as build
        import subprocess
        import contextlib
        import io
        with patch.object(build.shutil, 'which', return_value='/quarto'), patch.object(
            build.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'validation')
        ) as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                build.main([])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.args[0][1].endswith('validate_materials.py'))

    def test_full_build_checks_before_rendering(self):
        import build_materials as build
        with patch.object(build.shutil, 'which', return_value='/quarto'), patch.object(
            build.subprocess, 'run'
        ) as run:
            build.main(['--full'])
            self.assertEqual(run.call_args_list[0].args[0][-1], '--full')
            self.assertEqual(run.call_args_list[1].args[0], ['/quarto', 'render', '--to', 'all'])
            self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

    def test_failed_test_suite_prevents_notebook_execution(self):
        import validate_materials as validation
        import subprocess
        import contextlib
        import io
        with patch.object(validation, 'validate', return_value=[]), patch.object(
            validation.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'tests')
        ) as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validation.main(['--full'])
            self.assertEqual(run.call_count, 1)
            self.assertIn('unittest', run.call_args.args[0])

    def test_quick_validation_does_not_execute_notebooks(self):
        import validate_materials as validation
        with patch.object(validation, 'validate', return_value=[]), patch.object(
            validation.subprocess, 'run'
        ) as run:
            validation.main([])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
