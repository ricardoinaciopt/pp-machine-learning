"""Explicitly reproduce teaching datasets; ordinary notebooks use supplied data."""

import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlcourse.uci import fetch_datasets
from mlcourse.metalearning import rebuild_meta_dataset

DATASETS = {"concrete": 165, "wholesale": 292, "meta-learning": None}


def prepare(datasets, destination, raw_dir, refresh=False, root=ROOT):
    """Complete every selected dataset before replacing any destination files."""
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix="ac1-data-") as directory:
        staged = Path(directory)
        ids = [DATASETS[name] for name in datasets if DATASETS[name] is not None]
        if ids:
            fetch_datasets(root, output_dir=staged, dataset_ids=ids)
        if "meta-learning" in datasets:
            rebuild_meta_dataset(root, raw_dir=raw_dir, output_dir=staged / "ho09", refresh=refresh)
        # Publish provenance last so an interrupted file pair fails integrity checks.
        files = sorted(
            staged.rglob("*"), key=lambda p: (p.name.endswith(".provenance.json"), str(p))
        )
        for source in files:
            if not source.is_file():
                continue
            target = destination / source.relative_to(staged)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(source.read_bytes())
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Prepared {target}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        choices=[*DATASETS, "all"],
        help="Repeat to select several datasets.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir", type=Path, help="Comparison data root; defaults to .local-data/."
    )
    destination.add_argument(
        "--publish", action="store_true", help="Replace verified snapshots under data/."
    )
    parser.add_argument(
        "--raw-dir", type=Path, help="OpenML download cache; defaults to .local-data/ho09/raw/."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download frozen OpenML sources instead of reusing verified bytes.",
    )
    args = parser.parse_args(argv)
    datasets = list(DATASETS) if "all" in args.dataset else list(dict.fromkeys(args.dataset))
    if (args.refresh or args.raw_dir) and "meta-learning" not in datasets:
        parser.error("--refresh and --raw-dir apply to the meta-learning dataset")
    output = (ROOT / "data" if args.publish else args.output_dir or ROOT / ".local-data").resolve()
    supplied = (ROOT / "data").resolve()
    if not args.publish and (
        output == supplied or supplied in output.parents or output in supplied.parents
    ):
        parser.error(
            "Use --publish to replace supplied data; choose a separate comparison directory otherwise"
        )
    raw = (args.raw_dir or ROOT / ".local-data/ho09/raw").resolve()
    prepare(datasets, output, raw, args.refresh)
    print("Preparation complete. No files were committed or uploaded.")


if __name__ == "__main__":
    main()
