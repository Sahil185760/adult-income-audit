# Model card: Adult income classifier

## Purpose

This is a learning and review artifact for binary classification and fairness auditing. It estimates the probability that a record in the UCI Adult dataset has annual income above the dataset's $50,000 threshold. It is **not intended for decisions about individuals**.

## Model and data

Regularized logistic regression over imputed and standardized numeric fields and one-hot encoded categorical fields. `C` and the decision threshold are selected using only a validation subset of the official training file. Evaluation uses the official test file once. The code excludes recorded sex and race from predictors, but correlated predictors remain. The dataset derives from selected 1994 U.S. Census records and contains missing values and dated demographic categories. See the README for feature list and source citation.

## Evaluation

At threshold 0.33, the 16,281-row held-out test set gives ROC AUC 0.904, average precision 0.757, precision 0.624, recall 0.751, and F1 0.682. The baseline has ROC AUC 0.500 and F1 0.000 at that threshold. Exact metrics and subgroup counts are in [reports/benchmark.json](reports/benchmark.json).

## Risks and limitations

- Differences in selection and error rates appear across recorded sex and race categories. Excluding those fields does not remove proxy effects or historical bias.
- The dataset's recorded sex is binary and its race taxonomy is dated. It cannot represent all identities or support a complete fairness assessment.
- Small groups yield noisy estimates. The report contains counts so rates can be interpreted in context.
- The label reflects historical income and social conditions, not merit or qualification. The model must not be used to rank, screen, or allocate opportunities to people.
- There is no validation on current, external, or temporally shifted data; no causal fairness analysis; and no deployment monitoring.

## Reproducibility

The CLI verifies both original file hashes, uses a fixed seed, and records library versions in the report. The serialized model is generated locally and is not committed. Read the [README](README.md) for the exact command.

