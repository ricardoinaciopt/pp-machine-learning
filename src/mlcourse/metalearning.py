"""HO9's frozen, small algorithm-selection experiment."""

from pathlib import Path
import hashlib
import json
import platform
import tempfile
import urllib.request
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier

RANDOM_STATE = 42
PROTOCOL = {
    "version": 1,
    "seed": RANDOM_STATE,
    "test_size": 0.30,
    "split": "stratified holdout; full dataset; original row order",
    "metafeatures": "training partition only; correlation on encoded training predictors",
    "preprocessing": "numeric median; categorical most-frequent then one-hot; fit on training only",
    "models": "DecisionTreeClassifier defaults; stump max_depth=1; both random_state=42",
    "winner": "test correct-count difference; exact ties map to stump-or-tie",
    "meta_evaluation": "leave-one-dataset-out; depth-2 tree; most-frequent baseline",
}
FEATURES = [
    "mf_n_instances",
    "mf_n_features_raw",
    "mf_n_features_encoded",
    "mf_n_classes",
    "mf_feature_instance_ratio",
    "mf_instance_feature_ratio",
    "mf_numeric_ratio",
    "mf_missing_rate",
    "mf_class_entropy",
    "mf_majority_class_prop",
    "mf_minority_class_prop",
    "mf_imbalance_ratio",
    "mf_mean_abs_feature_corr",
]
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(root=ROOT):
    manifest = pd.read_csv(
        Path(root) / "data/ho09/openml_datasets.csv", keep_default_na=False
    )
    if len(manifest) != 12 or manifest.openml_id.nunique() != 12:
        raise ValueError("HO9 requires exactly 12 distinct frozen OpenML IDs")
    if list(manifest.openml_id) != sorted(manifest.openml_id):
        raise ValueError("Manifest must be ordered by OpenML ID")
    return manifest


def preprocessing(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def compute_simple_metafeatures(X, y):
    """Describe only the labelled training data available at recommendation time."""
    n, p = X.shape
    encoded = preprocessing(X).fit_transform(X)
    corr = pd.DataFrame(encoded).corr().abs().to_numpy()
    values = corr[np.triu_indices_from(corr, k=1)]
    finite = values[np.isfinite(values)]
    probs = y.value_counts() / len(y)
    probs = probs[probs > 0]
    return dict(
        zip(
            FEATURES,
            [
                n,
                p,
                encoded.shape[1],
                len(probs),
                p / n,
                n / p,
                len(X.select_dtypes(include="number").columns) / p,
                float(X.isna().to_numpy().mean()),
                float(-(probs * np.log2(probs)).sum()),
                float(probs.max()),
                float(probs.min()),
                float(probs.max() / probs.min()),
                float(finite.mean()) if len(finite) else 0.0,
            ],
        )
    )


def load_raw(entry, raw_dir, refresh=False):
    """Read verified bytes, downloading only in this explicit rebuild path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{entry.openml_id}.pq"
    if refresh or not path.exists():
        # Verify identity as well as the frozen byte digest. No substitute datasets.
        with urllib.request.urlopen(
            f"https://www.openml.org/api/v1/json/data/{entry.openml_id}", timeout=90
        ) as response:
            description = json.load(response)["data_set_description"]
        for key, expected in [
            ("id", entry.openml_id),
            ("name", entry.name),
            ("version", entry.version),
            ("default_target_attribute", entry.target),
        ]:
            if str(description[key]) != str(expected):
                raise ValueError(f"OpenML {entry.openml_id}: changed {key}")
        with urllib.request.urlopen(entry.parquet_url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise ValueError(
                f"OpenML {entry.openml_id}: source checksum changed; review before refreezing"
            )
        _atomic_write(path, data)
    if sha256(path) != entry.sha256:
        raise ValueError(f"Raw cache checksum mismatch: {path}")
    frame = pd.read_parquet(path)
    if (
        frame.shape != (entry.n_instances, entry.n_features + 1)
        or entry.target not in frame
    ):
        raise ValueError(f"OpenML {entry.openml_id}: unexpected schema")
    y = frame.pop(entry.target)
    if y.isna().any() or y.nunique() != entry.n_classes or y.value_counts().min() < 2:
        raise ValueError(f"OpenML {entry.openml_id}: invalid classification target")
    # Remove unused category levels; sklearn learns categories from training rows.
    for col in frame.select_dtypes(exclude="number"):
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, y.astype(str)


def summarise_one_dataset(entry, X, y):
    train, test = train_test_split(
        np.arange(len(y)),
        test_size=PROTOCOL["test_size"],
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test, y_train, y_test = (
        X.iloc[train],
        X.iloc[test],
        y.iloc[train],
        y.iloc[test],
    )
    scores, correct, balanced = {}, {}, {}
    for label, depth in [("DT", None), ("DS", 1)]:
        model = Pipeline(
            [
                ("prepare", preprocessing(X_train)),
                (
                    "tree",
                    DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
                ),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        correct[label] = int(np.sum(pred == y_test.to_numpy()))
        scores[label] = correct[label] / len(test)
        balanced[label] = balanced_accuracy_score(y_test, pred)
    difference = correct["DT"] - correct["DS"]
    majority = y_train.value_counts().sort_index().idxmax()
    return {
        "OpenMLID": entry.openml_id,
        "Dataset": entry.name,
        "Source": "OpenML",
        "NInstances": len(y),
        "NTrain": len(train),
        "NTest": len(test),
        "TrainIndexSHA256": hashlib.sha256(
            np.asarray(train, dtype="<i8").tobytes()
        ).hexdigest(),
        "TestIndexSHA256": hashlib.sha256(
            np.asarray(test, dtype="<i8").tobytes()
        ).hexdigest(),
        "CorrectDT": correct["DT"],
        "CorrectDS": correct["DS"],
        "AccuracyDT": scores["DT"],
        "AccuracyDS": scores["DS"],
        "BalancedAccuracyDT": balanced["DT"],
        "BalancedAccuracyDS": balanced["DS"],
        "AccuracyMajority": float((y_test == majority).mean()),
        "AccuracyGap_DT_minus_DS": difference / len(test),
        "Best": -int(np.sign(difference)),
        "BestModel": (
            "Decision tree"
            if difference > 0
            else "Decision stump" if difference < 0 else "Tie"
        ),
        "BestModelBinary": int(difference > 0),
        **compute_simple_metafeatures(X_train, y_train),
    }


def _atomic_write(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        file.write(data)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def rebuild_meta_dataset(root=ROOT, raw_dir=None, output_dir=None, refresh=False):
    root = Path(root)
    destination = Path(output_dir) if output_dir else root / "data/ho09"
    raw_dir = Path(raw_dir) if raw_dir else root / "data/ho09/raw"
    rows = []
    for entry in read_manifest(root).itertuples(index=False):
        X, y = load_raw(entry, raw_dir, refresh=refresh)
        rows.append(summarise_one_dataset(entry, X, y))
        print(f"Verified {entry.openml_id}: {entry.name}", flush=True)
    table = pd.DataFrame(rows)
    if not np.isfinite(table[FEATURES].to_numpy()).all():
        raise ValueError("Non-finite meta-features")
    csv_bytes = table.to_csv(
        index=False, float_format="%.12g", lineterminator="\n"
    ).encode()
    provenance = {
        "protocol": PROTOCOL,
        "manifest_sha256": sha256(root / "data/ho09/openml_datasets.csv"),
        "implementation_sha256": sha256(Path(__file__)),
        "meta_dataset_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow"]
        },
        "row_count": len(table),
    }
    destination.mkdir(parents=True, exist_ok=True)
    # Compute every row first; a failed dataset never replaces the published cache.
    # Provenance is committed last; an interrupted pair fails checksum validation.
    _atomic_write(destination / "meta_dataset.csv", csv_bytes)
    _atomic_write(
        destination / "meta_dataset.provenance.json",
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
    )
    return table


def load_meta_dataset(root=ROOT):
    """Offline-only default. A missing/stale cache is an error, never a download."""
    root = Path(root)
    folder = root / "data/ho09"
    try:
        provenance = json.loads((folder / "meta_dataset.provenance.json").read_text())
        checks = {
            "manifest_sha256": folder / "openml_datasets.csv",
            "meta_dataset_sha256": folder / "meta_dataset.csv",
            "implementation_sha256": Path(__file__),
        }
        for key, path in checks.items():
            if provenance[key] != sha256(path):
                raise ValueError(f"HO9 cache is stale or corrupt: {key}")
        if provenance["protocol"] != PROTOCOL:
            raise ValueError("HO9 cache protocol differs")
        table = pd.read_csv(folder / "meta_dataset.csv")
        manifest = read_manifest(root)
        if (
            table.OpenMLID.tolist() != manifest.openml_id.tolist()
            or table.Dataset.tolist() != manifest.name.tolist()
        ):
            raise ValueError("HO9 cache does not match the frozen manifest")
        if not np.isfinite(table[FEATURES].to_numpy()).all():
            raise ValueError("Non-finite cached meta-features")
        return table
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "HO9 needs its committed cache. Restore data/ho09 or explicitly run python scripts/prepare_data.py --dataset meta-learning --publish"
        ) from error


def get_meta_feature_columns(meta_data):
    return FEATURES.copy()


def fit_meta_model(meta_data):
    """Every prediction holds out one entire dataset, with a matched dummy baseline."""
    X, y = meta_data[FEATURES], meta_data["BestModelBinary"]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
            ),
        ]
    )
    cv = LeaveOneOut()
    pred = cross_val_predict(model, X, y, cv=cv)
    baseline = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)
    labels = {0: "Stump or tie", 1: "Decision tree"}
    predictions = pd.DataFrame(
        {
            "Dataset": meta_data.Dataset.to_numpy(),
            "Actual": y.map(labels).to_numpy(),
            "Predicted": [labels[v] for v in pred],
            "Baseline": [labels[v] for v in baseline],
        }
    )
    predictions.attrs["baseline_accuracy"] = accuracy_score(y, baseline)
    model.fit(X, y)
    return (
        model,
        predictions,
        accuracy_score(y, pred),
        confusion_matrix(y, pred, labels=[0, 1]),
    )


def plot_accuracy_gap(meta_data):
    from matplotlib import pyplot as plt

    ordered = meta_data.sort_values("AccuracyGap_DT_minus_DS")
    plt.figure(figsize=(9, 5))
    plt.barh(ordered.Dataset, ordered.AccuracyGap_DT_minus_DS)
    plt.axvline(0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Test accuracy: full tree minus stump")
    plt.tight_layout()
    plt.show()


def plot_feature_relationship(meta_data, feature_name, target_name):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(meta_data[feature_name], meta_data[target_name])
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.tight_layout()
    plt.show()


def correlation_table(meta_data, target_name="AccuracyGap_DT_minus_DS"):
    # Descriptive associations only: no uncorrected significance-screening exercise.
    rows = [
        {
            "Meta-feature": feature,
            "Pearson correlation": meta_data[feature].corr(meta_data[target_name]),
        }
        for feature in FEATURES
        if meta_data[feature].nunique() > 1 and meta_data[target_name].nunique() > 1
    ]
    result = pd.DataFrame(rows, columns=["Meta-feature", "Pearson correlation"])
    result["abs(correlation)"] = result["Pearson correlation"].abs()
    return result.sort_values("abs(correlation)", ascending=False).reset_index(
        drop=True
    )
