"""Check notebook pairs, teaching data and handout references without network access."""

import argparse
import subprocess
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_notebooks(root=ROOT):
    root = Path(root)
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    for number in range(1, 10):
        pair = []
        for solution in (False, True):
            path = (
                root
                / f"hands-on/ho{number:02d}"
                / ("solutions/notebook.ipynb" if solution else "notebook.ipynb")
            )
            try:
                book = nbformat.read(path, as_version=4)
                nbformat.validate(book)
            except Exception as error:
                errors.append(f"{path.relative_to(root)}: invalid or missing notebook: {error}")
                continue
            label = str(path.relative_to(root))
            check(path.stat().st_size < 1500000, f"{label}: oversized notebook")
            for index, cell in enumerate(book.cells):
                if cell.metadata.get("role") != "task":
                    continue
                block = book.cells[index : index + 3]
                check(
                    len(block) == 3
                    and [c.metadata.get("role") for c in block]
                    == ["task", "experiment", "response"],
                    f"{label}: broken experiment sequence",
                )
                check("**Prediction:**" in cell.source, f"{label}: missing prediction space")
                if len(block) == 3:
                    check(
                        all(
                            (
                                c.metadata.get("experiment_id")
                                == cell.metadata.get("experiment_id")
                                for c in block
                            )
                        ),
                        f"{label}: mismatched experiment identifiers",
                    )
                    check(
                        "**Observation:**" in block[2].source
                        and "**Explanation:**" in block[2].source,
                        f"{label}: missing response spaces",
                    )
            images = 0
            structure = []
            check(
                sum((c.metadata.get("role") == "bootstrap" for c in book.cells)) == 1,
                f"{label}: missing/duplicated bootstrap",
            )
            check(
                sum((c.metadata.get("role") == "setup" for c in book.cells)) == 1,
                f"{label}: missing/duplicated setup",
            )
            for i, cell in enumerate(book.cells):
                source = cell.source
                location = f"{label} cell {i}"
                check("—" not in source, f"{location}: em dash")
                check(
                    not re.search(
                        "Predict\\s*→|Diagnose\\s*→|Proposed interpretation|optional but important",
                        source,
                        re.I,
                    ),
                    f"{location}: forbidden framework or essay wording",
                )
                if cell.cell_type == "markdown":
                    plain = re.sub("```.*?```|`[^`]+`", "", source, flags=re.S)
                    check(
                        not re.search(
                            "\\b\\w+\\(\\)|\\b(?:max_depth|min_samples_leaf|n_neighbors|random_state|min_samples|class_weight)\\b",
                            plain,
                        ),
                        f"{location}: unformatted Python API",
                    )
                    if number > 1:
                        check(
                            not re.search(
                                "Open-data migration|rebuild.*OpenML|checksum|requirements[^ ]*\\.txt|frozen cache",
                                source,
                                re.I,
                            ),
                            f"{location}: maintenance prose",
                        )
                    if cell.metadata.get("role") == "task":
                        structure.append(("task", source.split("**Prediction:**")[0]))
                    structure.append(
                        (
                            "markdown",
                            re.findall(
                                "^#{1,3} .+|^\\*\\*(?:Prediction|Observation|Explanation):\\*\\*.*",
                                source,
                                re.M,
                            ),
                        )
                    )
                elif cell.cell_type == "code":
                    try:
                        tree = ast.parse(source)
                    except SyntaxError as error:
                        errors.append(f"{location}: Python syntax error: {error.msg}")
                        continue
                    if any(
                        (isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
                    ):
                        check(
                            cell.metadata.get("jupyter", {}).get("source_hidden"),
                            f"{location}: helper cell not collapsed",
                        )
                    structure.append(("code", source))
                    check(
                        not re.search(":\\.2f|round\\([^\\n]+,\\s*2\\)", source),
                        f"{location}: two-decimal metric formatting",
                    )
                    for line in source.splitlines():
                        check(
                            not re.match(
                                "\\s*#\\s*(?:import |from \\w+ import |print\\(|\\w+\\s*=)", line
                            ),
                            f"{location}: commented executable code",
                        )
                    for output in cell.get("outputs", []):
                        images += "image/png" in output.get("data", {})
                        check(output.output_type != "error", f"{location}: stored execution error")
                    check(solution or not cell.get("outputs"), f"{location}: stored student output")
                    if cell.metadata.get("role") in ("setup", "bootstrap"):
                        check(
                            cell.metadata.get("jupyter", {}).get("source_hidden"),
                            f"{location}: setup not collapsed",
                        )
            check(images <= 12, f"{label}: excessive stored figures ({images})")
            pair.append(structure)
        check(
            len(pair) == 2 and pair[0] == pair[1],
            f"HO{number:02d}: student/solution structure or code mismatch",
        )
    return errors


def validate_meta_dataset(root=ROOT):
    folder = Path(root) / "data/ho09"
    with (folder / "openml_datasets.csv").open() as file:
        manifest = list(csv.DictReader(file))
    with (folder / "meta_dataset.csv").open() as file:
        rows = list(csv.DictReader(file))
    provenance = json.loads((folder / "meta_dataset.provenance.json").read_text())
    require(len(manifest) == len(rows) == provenance["row_count"] == 12, "Expected twelve rows")
    ids = [int(row["openml_id"]) for row in manifest]
    require(ids == sorted(set(ids)), "IDs must be unique and ordered")
    require([int(row["OpenMLID"]) for row in rows] == ids, "Dataset IDs differ")
    require([row["Dataset"] for row in rows] == [row["name"] for row in manifest], "Names differ")
    for field, path in [
        ("manifest_sha256", folder / "openml_datasets.csv"),
        ("meta_dataset_sha256", folder / "meta_dataset.csv"),
        ("implementation_sha256", Path(root) / "src/mlcourse/metalearning.py"),
    ]:
        require(provenance[field] == digest(path), f"Cache hash mismatch: {field}")
    for entry, row in zip(manifest, rows):
        require(len(entry["sha256"]) == 64 and entry["reason_for_inclusion"], "Incomplete manifest")
        n, train, test = (int(row[k]) for k in ["NInstances", "NTrain", "NTest"])
        require(n == int(entry["n_instances"]) == train + test, "Partition sizes differ")
        require(test == math.ceil(n * provenance["protocol"]["test_size"]), "Wrong test size")
        require(float(row["mf_n_instances"]) == train, "Meta-features must describe training data")
        require(
            float(row["mf_n_features_raw"]) == int(entry["n_features"]), "Feature count differs"
        )
        require(float(row["mf_n_classes"]) == int(entry["n_classes"]), "Class count differs")
        require(
            all((math.isfinite(float(v)) for k, v in row.items() if k.startswith("mf_"))),
            "Non-finite feature",
        )
        for label in ["DT", "DS"]:
            correct = int(row["Correct" + label])
            require(0 <= correct <= test, "Invalid correct count")
            require(
                math.isclose(float(row["Accuracy" + label]), correct / test, abs_tol=1e-10),
                "Wrong accuracy",
            )
        gap = (int(row["CorrectDT"]) - int(row["CorrectDS"])) / test
        require(
            math.isclose(float(row["AccuracyGap_DT_minus_DS"]), gap, abs_tol=1e-10), "Wrong gap"
        )
        require(int(row["BestModelBinary"]) == int(gap > 0), "Wrong binary label")
        require(int(row["Best"]) == (-1 if gap > 0 else 1 if gap < 0 else 0), "Wrong ternary label")
        require(
            row["BestModel"]
            == ("Decision tree" if gap > 0 else "Decision stump" if gap < 0 else "Tie"),
            "Wrong winner",
        )
        for field in ["BalancedAccuracyDT", "BalancedAccuracyDS", "AccuracyMajority"]:
            require(0 <= float(row[field]) <= 1, f"Invalid {field}")
    return rows


def validate_uci_data(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "data/uci_sources.json").read_text())
    require([s["uci_id"] for s in manifest["datasets"]] == [165, 292], "Wrong UCI source set")
    for spec in manifest["datasets"]:
        require(
            spec["license"] == "CC BY 4.0" and spec["citation"] and spec["changes"],
            "Missing attribution",
        )
        require(
            len(spec["source_sha256"]) == len(spec["normalised_sha256"]) == 64, "Missing hashes"
        )
        inspection = json.loads((root / "data" / spec["folder"] / "inspection.json").read_text())
        require(
            inspection["rows"] == spec["rows"] and inspection["columns"] == spec["columns"],
            "Inspection mismatch",
        )
        require(
            (root / "data" / spec["folder"] / spec["filename"]).is_file(),
            f"Missing supplied dataset: {spec['filename']}",
        )
        for base in ["data", ".local-data"]:
            path = root / base / spec["folder"] / spec["filename"]
            if not path.exists():
                continue
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == spec["normalised_sha256"],
                f"Snapshot checksum mismatch: {path}",
            )
            with path.open() as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                require(
                    reader.fieldnames == spec["columns"] and len(rows) == spec["rows"],
                    "Snapshot schema mismatch",
                )
            require(
                all((math.isfinite(float(v)) for row in rows for v in row.values())),
                "Non-finite values",
            )


def validate_supplied_data(root=ROOT):
    """Check all declared inputs, including the supplied result archive."""
    root = Path(root)
    with (root / "data/manifest.csv").open() as file:
        for entry in csv.DictReader(file):
            for name in entry["local_file"].split(";"):
                require((root / name).is_file(), f"Missing teaching data: {name}")
    require((root / "data/ho03/Iris.csv").is_file(), "Missing Iris data")
    with zipfile.ZipFile(root / "data/ho06/results.zip") as archive:
        require(archive.testzip() is None, "Corrupt supplied results archive")
        require(any(name.endswith(".csv") for name in archive.namelist()), "Empty results archive")
    legacy = json.loads((root / "data/legacy_provenance.json").read_text())
    for entry in [legacy["hotel_reservations"], *legacy["synthetic_datasets"]]:
        path = root / entry["file"]
        require(digest(path) == entry["sha256"], f"Snapshot checksum mismatch: {entry['file']}")
        with path.open() as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            require(
                reader.fieldnames == entry["columns"] and len(rows) == entry["rows"],
                f"Snapshot schema mismatch: {entry['file']}",
            )


def validate_sources(root=ROOT):
    errors = []
    ignored = {".git", ".quarto", ".local-data", ".venv", "__pycache__", "dist", "raw"}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or ignored.intersection(path.relative_to(root).parts):
            continue
        name = str(path.relative_to(root))
        if any(part in path.name.lower() for part in ["copy of", "_old", "_back", "(1)"]):
            errors.append(f"Archive-style filename: {name}")
        if path.suffix in {".md", ".qmd", ".py", ".yml", ".yaml", ".txt", ".csv"}:
            text = path.read_text(errors="replace")
            if re.search(r"20\d{2}\s*[/–-]\s*20\d{2}", text):
                errors.append(f"Academic year found: {name}")
    return errors


def validate_media(root=ROOT):
    errors = []
    for path in sorted(Path(root).glob("hands-on/**/*.qmd")):
        for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if reference.startswith(("http://", "https://")):
                continue
            target = (
                reference[1 : reference.index(">")]
                if reference.startswith("<")
                else reference.split()[0]
            )
            if not (path.parent / target).is_file():
                errors.append(f"{path.relative_to(root)}: missing image {target}")
    return errors


def validate(root=ROOT):
    root = Path(root)
    errors = validate_notebooks(root) + validate_sources(root) + validate_media(root)
    for name, check in [
        ("Meta-dataset", validate_meta_dataset),
        ("UCI data", validate_uci_data),
        ("Supplied data", validate_supplied_data),
    ]:
        try:
            check(root)
        except Exception as error:
            errors.append(f"{name}: {error}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Also run unit tests and all notebooks locally and with simulated Colab setup.')
    args = parser.parse_args(argv)
    errors = validate()
    if errors:
        parser.exit(1, '\n'.join('ERROR: ' + error for error in errors) + '\n')
    print('Validated notebook pairs, supplied data, provenance and handout image references.', flush=True)
    if args.full:
        commands = [
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            [sys.executable, str(ROOT / 'scripts/check_notebooks.py')],
            [sys.executable, str(ROOT / 'scripts/check_notebooks.py'), '--simulate-colab'],
        ]
        try:
            for command in commands:
                subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            parser.exit(error.returncode, 'Full validation stopped because a check failed.\n')
        print('Full validation passed. Simulated Colab checks do not test the hosted service.')


if __name__ == '__main__':
    main()
