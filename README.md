# Adult income: a reproducible classification and fairness audit

An educational machine learning project built around the [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course). It predicts whether income in the 1994 U.S. Census derived [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) exceeds $50,000. The point is to show careful data handling and evaluation, including where an apparently strong model falls short. **This model is not suitable for employment, lending, eligibility, or other decisions about people.**

## What this project demonstrates

| Course concept | Implementation |
| --- | --- |
| Numerical and categorical data | Median imputation and scaling for five numeric fields; mode imputation and one-hot encoding for five categorical fields. |
| Logistic regression and regularization | `LogisticRegression` with three candidate `C` values, selected by validation log loss. |
| Generalization and overfitting | Official UCI test split stays untouched until final evaluation; the original train split is divided into fit and validation sets. All preprocessing is fitted inside the pipeline on fit rows only. |
| Classification | Validation-only F1 threshold choice; test precision, recall, F1, ROC AUC, average precision, log loss, Brier score, and confusion matrix. |
| Fairness | Test-set selection, true positive, and false positive rates by the dataset's recorded sex and race categories. |
| Production systems | Input checks, fixed source checksums, deterministic seed, machine-readable report, model artifact, and tests. These are foundations for review, not a deployment claim. |

Relevant lessons: [logistic regression](https://developers.google.com/machine-learning/crash-course/logistic-regression), [classification](https://developers.google.com/machine-learning/crash-course/classification), [datasets and overfitting](https://developers.google.com/machine-learning/crash-course/overfitting), [fairness](https://developers.google.com/machine-learning/crash-course/fairness), [production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems).

## Run it

Use Python 3.9 or newer. The raw UCI splits are included under `data/` for offline reproduction and are attributed below. `requirements.txt` pins the versions used for the checked-in benchmark.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m adult_audit --data-dir data --output-dir artifacts
```

The run writes `artifacts/report.json` and `artifacts/model.joblib`. The checked-in [benchmark report](reports/benchmark.json) is the output of the same command from the environment recorded inside it. Loading a joblib file can execute code; load only an artifact you created or trust.

## Protocol

The official `adult.data` file supplies 32,561 rows. A stratified split with seed 42 uses 26,048 for fitting and 6,513 for validation. The official `adult.test` file supplies 16,281 held-out rows. The choice of `C` (0.1, 1, 10) minimizes validation log loss. The threshold is selected from 0.10 through 0.90 in increments of 0.01 to maximize validation F1, with ties going to the larger threshold. The chosen model remains fitted on the fit rows; it is **not refitted** after choosing the threshold, so its probability scale remains the one used for threshold selection. The test set is evaluated once.

Sex and race are excluded from model features and retained for audit only. `fnlwgt` is a census sampling weight, so it is excluded rather than treated as an individual characteristic. `education` is excluded because `education_num` encodes the same education level. Other features, such as relationship and occupation, can still act as proxies for protected attributes.

## Held-out results

| Measure | Logistic model | Prior-only baseline |
| --- | ---: | ---: |
| ROC AUC | 0.904 | 0.500 |
| Average precision | 0.757 | 0.236 |
| F1 at threshold 0.33 | 0.682 | 0.000 |
| Precision | 0.624 | 0.000 |
| Recall | 0.751 | 0.000 |
| Log loss | 0.321 | 0.547 |

The baseline always estimates the fit-set positive rate and therefore predicts the negative class at the selected threshold. The model's confusion matrix is TN 10,694; FP 1,741; FN 956; TP 2,890. These values describe this historical dataset and split; they are not a claim of real-world performance.

## Audit findings and limits

For the dataset's recorded sex categories, the test selection rate is **11.9% for Female** and **36.7% for Male**; true positive rate is **65.6%** and **76.9%**, respectively. The full report also breaks out recorded race categories and sample counts. Some groups have under 200 rows, so their rates are especially unstable. Different base rates, historical social inequality, dataset selection, and proxy features can all affect these comparisons. Group metrics identify disparities; they do not, by themselves, establish a causal explanation or certify fairness.

The dataset reflects selected records from the 1994 Census. Its income threshold is nominal and unadjusted for inflation, its categories are dated and limited, and the sample is not representative of current populations. There is no temporal or external validation. The official test split may share collection biases with the training split. No causal analysis, intersectional audit, privacy assessment, or deployment monitoring is provided. See [MODEL_CARD.md](MODEL_CARD.md) for intended use and risks.

## Data provenance and license

Data: Becker, B. & Kohavi, R. (1996), *Adult* [dataset], UCI Machine Learning Repository, [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20). The [UCI page](https://archive.ics.uci.edu/dataset/2/adult) lists the data as **CC BY 4.0**. The original train and test files are included unmodified. SHA-256 checksums are recorded in `adult_audit/__main__.py` and checked on every run. Code in this repository is MIT licensed; the data retain their CC BY 4.0 license.

## Repository layout

```text
adult_audit/       parsing, preprocessing, selection, metrics, CLI
data/              original UCI Adult train and test files
reports/           checked-in benchmark report
tests/             parsing, leakage boundary, unseen category, threshold, and audit checks
MODEL_CARD.md      intended use and limitations
```
