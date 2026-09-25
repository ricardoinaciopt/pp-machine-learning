"""Shared notebook setup. Local environments are never modified implicitly."""

from pathlib import Path
import importlib.util
import os
import subprocess
import sys

# Use the shared course requirements only for packages missing in Colab.
_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "ipywidgets": "ipywidgets",
    "IPython": "ipython",
}
_REQUIREMENTS = {
    line.split("==")[0]: line
    for line in (Path(__file__).resolve().parents[2] / "requirements.txt")
    .read_text()
    .splitlines()
    if line and not line.startswith("#")
}
DEPENDENCIES = {module: _REQUIREMENTS[name] for module, name in _DISTRIBUTIONS.items()}


def in_colab():
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def find_root(start=None):
    explicit = os.environ.get("MLCOURSE_ROOT")
    bases = [Path(explicit)] if explicit else []
    bases += [Path(start or Path.cwd()), Path(__file__).resolve().parents[2]]
    for base in bases:
        for candidate in (base.resolve(), *base.resolve().parents):
            if (candidate / "src/mlcourse").is_dir() and (
                candidate / "hands-on"
            ).is_dir():
                return candidate
    raise FileNotFoundError(
        "Course files not found. Set MLCOURSE_ROOT to the extracted course repository and rerun setup."
    )


def setup_notebook(start=None):
    root = find_root(start)
    missing = [
        spec
        for name, spec in DEPENDENCIES.items()
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        if not in_colab():
            raise RuntimeError(
                f'Course packages are missing. Run: "{sys.executable}" -m pip install -r "{root / "requirements.txt"}"'
            )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "Course package installation failed. Check the Colab connection, then rerun setup."
            )
        importlib.invalidate_caches()
    if not (root / "data").is_dir():
        raise FileNotFoundError(
            "Course data not found. Restore the repository data folder and rerun setup."
        )
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    if in_colab():
        from google.colab import output

        output.enable_custom_widget_manager()
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option("display.float_format", lambda value: f"{value:.3f}")
    pd.set_option("display.max_rows", 14)
    pd.set_option("display.max_columns", 10)
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 90,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )
    return root
