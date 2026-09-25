"""Verified UCI retrieval, local-only caching until redistribution is approved."""

from pathlib import Path
import hashlib
import io
import json
import tempfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONCRETE_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength_mpa",
]
SPENDING_COLUMNS = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources(root=ROOT):
    return json.loads((Path(root) / "data/uci_sources.json").read_text())["datasets"]


def normalise(raw, spec):
    if digest(raw) != spec["source_sha256"]:
        raise ValueError(
            f"UCI {spec['uci_id']}: source checksum changed; review before updating the manifest"
        )
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.columns.tolist() != spec["source_columns"]:
        raise ValueError("Unexpected UCI column names or order")
    frame.columns = spec["columns"]
    validate_frame(frame, spec)
    return frame


def validate_frame(frame, spec):
    if frame.columns.tolist() != spec["columns"] or frame.shape != (
        spec["rows"],
        len(spec["columns"]),
    ):
        raise ValueError("Unexpected dataset schema or dimensions")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        raise ValueError("Unexpected non-numeric column")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(
            "Missing or infinite values are not present in the frozen UCI source"
        )
    if (frame.to_numpy() < 0).any():
        raise ValueError("Negative ingredient, spending, or category values")
    if spec["uci_id"] == 165:
        if not frame.age.between(1, 365).all() or not (frame.strength_mpa > 0).all():
            raise ValueError("Invalid age or strength")
    elif set(frame.Channel) != {1, 2} or set(frame.Region) != {1, 2, 3}:
        raise ValueError("Unexpected Channel/Region codes")


def inspect(frame, spec):
    report = {
        "uci_id": spec["uci_id"],
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "ranges": {
            c: {"min": float(frame[c].min()), "max": float(frame[c].max())}
            for c in frame
        },
    }
    if spec["uci_id"] == 165:
        report.update(
            ingredient_groups=len(frame[CONCRETE_COLUMNS[:7]].drop_duplicates()),
            repeated_predictor_rows_beyond_first=int(
                frame[CONCRETE_COLUMNS[:8]].duplicated().sum()
            ),
        )
    else:
        report["category_counts"] = {
            c: {str(k): int(v) for k, v in frame[c].value_counts().sort_index().items()}
            for c in ["Channel", "Region"]
        }
    return report


def fetch_datasets(root=ROOT, output_dir=None, dataset_ids=None):
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / ".local-data"
    prepared = []
    for spec in sources(root):
        if dataset_ids and spec["uci_id"] not in dataset_ids:
            continue
        with urllib.request.urlopen(spec["data_url"], timeout=90) as response:
            raw = response.read()
        frame = normalise(raw, spec)
        content = frame.to_csv(
            index=False, float_format="%.12g", lineterminator="\n"
        ).encode()
        if digest(content) != spec["normalised_sha256"]:
            raise ValueError(
                "Normalised CSV digest differs; check parser/serialization version"
            )
        report = inspect(frame, spec)
        prepared.append((spec, content, report))
    # Finish validation of all requested datasets before writing any of them.
    for spec, content, report in prepared:
        path = output_dir / spec["folder"] / spec["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temp = Path(file.name)
            file.write(content)
        temp.replace(path)
        (path.parent / "inspection.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(f"Verified UCI {spec['uci_id']}: {path}", flush=True)
    return [report for _, _, report in prepared]


def load_dataset(dataset_id, root=ROOT):
    """Prefer distributable data, then the ignored local cache; never download implicitly."""
    root = Path(root)
    spec = next(s for s in sources(root) if s["uci_id"] == dataset_id)
    for folder in ["data", ".local-data"]:
        path = root / folder / spec["folder"] / spec["filename"]
        if path.exists():
            if digest(path.read_bytes()) != spec["normalised_sha256"]:
                raise ValueError(f"Dataset differs from verified UCI snapshot: {path}")
            frame = pd.read_csv(path)
            validate_frame(frame, spec)
            return frame
    raise FileNotFoundError(
        f"UCI {dataset_id} is not cached. Restore the supplied data folder, or run python scripts/prepare_data.py --dataset {'concrete' if dataset_id == 165 else 'wholesale'} --publish from the repository root."
    )
