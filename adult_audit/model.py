"""Model construction and validation-only choices."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC


def make_pipeline(c: float) -> Pipeline:
    numerical = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("numerical", numerical, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ])),
        ("classifier", LogisticRegression(C=c, max_iter=1000, random_state=42)),
    ])


def select_model(train, validation, candidates=(0.1, 1.0, 10.0)):
    """Choose regularization by validation log loss; test data is never read here."""
    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = validation[FEATURES], validation["target"]
    trials = []
    models = []
    for c in candidates:
        model = make_pipeline(c).fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        trials.append({"C": c, "validation_log_loss": float(log_loss(y_val, probability))})
        models.append(model)
    best = int(np.argmin([row["validation_log_loss"] for row in trials]))
    return models[best], trials[best]["C"], trials


def select_threshold(y_true, probabilities):
    """Maximize validation F1 on a fixed grid; ties favor the higher threshold."""
    thresholds = np.arange(0.10, 0.901, 0.01)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    best = max(range(len(scores)), key=lambda i: (scores[i], thresholds[i]))
    return round(float(thresholds[best]), 2), float(scores[best])

