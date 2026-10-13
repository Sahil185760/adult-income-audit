"""Dataset schema and parsing. The original UCI test file starts with a comment."""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "native_country"]
FEATURES = NUMERIC + CATEGORICAL
AUDIT_GROUPS = ["sex", "race"]


def load_adult(path: Path) -> pd.DataFrame:
    """Read one official UCI split, rejecting unexpected labels or malformed rows."""
    frame = pd.read_csv(
        path, header=None, names=COLUMNS, comment="|", skipinitialspace=True,
        na_values=["?"], skip_blank_lines=True,
    )
    if frame.empty or frame.isna().all(axis=1).any():
        raise ValueError(f"Empty or malformed Adult file: {path}")
    labels = frame["income"].astype("string").str.strip().str.rstrip(".")
    if labels.isna().any() or not labels.isin(["<=50K", ">50K"]).all():
        raise ValueError(f"Unexpected income label in {path}")
    frame["target"] = labels.eq(">50K").astype(int)
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[NUMERIC].isna().all(axis=1).any():
        raise ValueError(f"Numeric fields missing for an entire row in {path}")
    return frame

