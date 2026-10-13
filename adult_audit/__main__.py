"""Run with: python -m adult_audit --data-dir data --output-dir artifacts"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from .data import FEATURES, load_adult
from .metrics import binary_metrics, group_metrics
from .model import select_model, select_threshold

EXPECTED_SHA256 = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_dir: Path, output_dir: Path):
    train_path, test_path = data_dir / "adult.data", data_dir / "adult.test"
    for path in (train_path, test_path):
        if sha256(path) != EXPECTED_SHA256[path.name]:
            raise ValueError(f"Unexpected checksum for {path}; use the bundled UCI files")
    original_train, test = load_adult(train_path), load_adult(test_path)
    if len(original_train) != 32561 or len(test) != 16281:
        raise ValueError("Expected the official UCI train and test split sizes")
    train, validation = train_test_split(
        original_train, test_size=0.2, stratify=original_train["target"], random_state=42,
    )
    # Position-based subgroup indexing must align with probability arrays.
    train, validation, test = (part.reset_index(drop=True) for part in (train, validation, test))
    model, selected_c, trials = select_model(train, validation)
    validation_prob = model.predict_proba(validation[FEATURES])[:, 1]
    threshold, validation_f1 = select_threshold(validation["target"], validation_prob)

    baseline = DummyClassifier(strategy="prior").fit(train[FEATURES], train["target"])
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    baseline_prob = baseline.predict_proba(test[FEATURES])[:, 1]
    report = {
        "reproducibility": {"seed": 42, "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "data": {
            "source": "https://archive.ics.uci.edu/dataset/2/adult",
            "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
            "fit_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "features": FEATURES, "excluded_fields": ["income", "target", "sex", "race", "fnlwgt", "education"],
        },
        "selection": {"criterion": "validation log loss", "candidates": trials,
                      "selected_C": selected_c, "threshold_criterion": "validation F1",
                      "threshold": threshold, "validation_f1": validation_f1},
        "test": binary_metrics(test["target"], test_prob, threshold),
        "majority_prior_baseline": binary_metrics(test["target"], baseline_prob, threshold),
        "subgroups": group_metrics(test, test_prob, threshold),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    joblib.dump({"pipeline": model, "threshold": threshold, "features": FEATURES}, output_dir / "model.joblib")
    return report


def main():
    parser = argparse.ArgumentParser(description="Reproduce the Adult income classification audit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = run(args.data_dir, args.output_dir)
    print(json.dumps({"selected_C": report["selection"]["selected_C"],
                      "threshold": report["selection"]["threshold"],
                      "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
