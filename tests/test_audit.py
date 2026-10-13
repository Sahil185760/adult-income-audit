from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from adult_audit.data import COLUMNS, FEATURES, load_adult
from adult_audit.metrics import group_metrics
from adult_audit.model import make_pipeline, select_threshold


def test_official_splits_parse_and_are_distinct():
    train = load_adult(Path("data/adult.data"))
    test = load_adult(Path("data/adult.test"))
    assert (len(train), len(test)) == (32561, 16281)
    assert set(train.target) == set(test.target) == {0, 1}
    assert train.target.mean() != test.target.mean()
    assert train[COLUMNS].isna().any().any()  # '?' must become missing data.
    assert "sex" not in FEATURES and "race" not in FEATURES
    assert "income" not in FEATURES and "target" not in FEATURES


def test_unexpected_label_is_rejected(tmp_path):
    row = ["1"] * (len(COLUMNS) - 1) + ["UNKNOWN"]
    path = tmp_path / "bad.data"
    path.write_text(", ".join(row) + "\n")
    with pytest.raises(ValueError, match="Unexpected income label"):
        load_adult(path)


def test_preprocessor_fits_only_train_and_handles_unknown_category():
    frame = load_adult(Path("data/adult.data"))
    train, validation = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.target)
    model = make_pipeline(1.0).fit(train[FEATURES], train.target)
    other = validation[FEATURES].iloc[:1].copy()
    other["workclass"] = "A-category-never-seen-before"
    probability = model.predict_proba(other)[:, 1]
    assert len(probability) == 1 and 0 <= probability[0] <= 1
    learned_medians = model.named_steps["prepare"].named_transformers_["numerical"].named_steps["impute"].statistics_
    np.testing.assert_allclose(learned_medians, train[["age", "education_num", "capital_gain",
                                                        "capital_loss", "hours_per_week"]].median().to_numpy())


def test_threshold_tie_favors_higher_value():
    y = np.array([0, 1])
    probabilities = np.array([0.0, 1.0])
    threshold, score = select_threshold(y, probabilities)
    assert threshold == 0.90 and score == 1.0


def test_group_rates_use_matching_rows():
    frame = pd.DataFrame({"sex": ["A", "B", "A", "B"],
                          "race": ["X", "X", "Y", "Y"], "target": [1, 0, 0, 1]})
    result = group_metrics(frame, np.array([0.9, 0.9, 0.1, 0.1]), 0.5)
    assert result["sex"][0]["true_positive_rate"] == 1.0
    assert result["sex"][1]["false_positive_rate"] == 1.0
    assert result["race"][1]["selection_rate"] == 0.0
