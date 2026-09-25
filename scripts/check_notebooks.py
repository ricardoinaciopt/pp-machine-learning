"""Execute notebooks in fresh kernels, without data downloads."""

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """import socket, urllib.request

def deny_network(*args, **kwargs):
    raise AssertionError("Default laboratory attempted a network download")
socket.create_connection = deny_network
socket.socket.connect = deny_network
urllib.request.urlopen = deny_network
"""
COLAB = """import sys, types, importlib.machinery
try:
    import google
except ModuleNotFoundError:
    google = types.ModuleType("google")
    google.__path__ = []
    sys.modules["google"] = google
colab = types.ModuleType("google.colab")
colab.__spec__ = importlib.machinery.ModuleSpec("google.colab", loader=None)
colab.output = types.SimpleNamespace(enable_custom_widget_manager=lambda: None)
sys.modules["google.colab"] = colab
google.colab = colab
"""


def run(path, output, simulate_colab=False, save_solutions=False):
    book = nbformat.read(path, as_version=4)
    book.cells.insert(
        0, nbformat.v4.new_code_cell(PRELUDE + (COLAB if simulate_colab else ""))
    )
    NotebookClient(
        book,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    for bundle in book.metadata.get("widgets", {}).values():
        for model in bundle.get("state", {}).values():
            for result in model.get("state", {}).get("outputs", []):
                if result.get("output_type") == "error":
                    raise RuntimeError(
                        f"Widget callback failed in {path}: {result.get('evalue')}"
                    )
    book.cells.pop(0)
    relative = path.relative_to(ROOT)
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(book, destination)
    if save_solutions and "solutions" in path.parts:
        book.metadata.pop("widgets", None)
        for cell in book.cells:
            if cell.cell_type == "code":
                if "interactive" in cell.metadata.get("tags", []) or cell.metadata.get(
                    "jupyter", {}
                ).get("source_hidden"):
                    cell.outputs = []
                cell.metadata.pop("execution", None)
                cell.execution_count = None
                for result in cell.outputs:
                    if "execution_count" in result:
                        result.execution_count = None
        nbformat.write(book, path)
    print(f"PASS {relative}", flush=True)


def select_notebooks(paths=(), ho=None, root=ROOT):
    root = Path(root).resolve()
    canonical = sorted(root.glob("hands-on/ho*/**/notebook.ipynb"))
    if paths and ho is not None:
        raise ValueError("Choose notebook paths or --ho, not both")
    if paths:
        selected = list(dict.fromkeys(Path(path).resolve() for path in paths))
        if any(path not in canonical for path in selected):
            raise ValueError(
                "Each path must identify a canonical notebook in this repository"
            )
    else:
        selected = [
            path for path in canonical if ho is None or f"ho{ho:02d}" in path.parts
        ]
        expected = 18 if ho is None else 2
        if len(selected) != expected:
            raise ValueError(f"Expected {expected} notebooks; found {len(selected)}")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Specific notebook paths; defaults to all pairs.",
    )
    parser.add_argument(
        "--ho",
        type=int,
        choices=range(1, 10),
        help="Execute one student/solution pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep executed copies here; otherwise use temporary files.",
    )
    parser.add_argument(
        "--simulate-colab",
        action="store_true",
        help="Exercise the Colab setup branch locally; this does not run hosted Colab.",
    )
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Refresh selected solutions with bounded static outputs.",
    )
    args = parser.parse_args(argv)
    try:
        paths = select_notebooks(args.notebooks, args.ho)
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir:
        output = args.output_dir.resolve()
        if (
            output == ROOT
            or ROOT / "hands-on" == output
            or ROOT / "hands-on" in output.parents
        ):
            parser.error(
                "Executed copies must be separate from the canonical notebooks"
            )
    with tempfile.TemporaryDirectory(prefix="ac1-notebooks-") as directory:
        output = args.output_dir.resolve() if args.output_dir else Path(directory)
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                (
                    path,
                    pool.submit(
                        run, path, output, args.simulate_colab, args.save_solutions
                    ),
                )
                for path in paths
            ]
            for path, job in jobs:
                try:
                    job.result()
                except Exception as error:
                    failures.append(f"{path.relative_to(ROOT)}: {error}")
        if failures:
            parser.exit(1, "\n".join(failures) + "\n")
    print(f"Executed {len(paths)} notebooks successfully; data downloads were blocked.")


if __name__ == "__main__":
    main()
