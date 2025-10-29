---
change-id: 0001_baseline_svm
title: Baseline SVM spam-classification (Phase 1)
author: TBD
status: proposed
---

# Summary

This proposal requests the addition of a Phase 1 baseline feature to train and evaluate a simple spam vs. ham classifier using a Support Vector Machine (SVM). The pipeline will fetch the Packt-provided SMS/spam dataset, perform preprocessing (tokenization and TF-IDF), train an SVM baseline, and produce serialized model artifacts and evaluation reports.

# Motivation

Automated spam detection is a common requirement for security and user-experience tooling. A small, reproducible ML baseline helps validate data ingestion, preprocessing, evaluation methodology, and CI smoke tests before investing in more advanced models (logistic regression, tuning, or productionization). This Phase 1 work aims to deliver a working, documented baseline so we can measure improvements in Phase 2.

# Implementation Plan

1. Create a reproducible training script (Python) at `tools/ml/train_baseline_svm.py` that:
   - Downloads and caches the dataset from:
     `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv`
   - Loads the CSV into pandas and performs a sanity check (non-empty text, expected row counts).
   - Preprocesses text with simple cleaning, tokenization, and TF-IDF vectorization.
   - Splits data into train/validation/test (e.g., 70/15/15) with a fixed random seed.
   - Trains an SVM classifier (scikit-learn's SVC with a linear or RBF kernel) and saves the model with `joblib` to `models/svm_baseline.joblib`.
   - Evaluates on the test split and writes `reports/evaluation.json` containing accuracy, precision, recall, F1, and optionally ROC-AUC, plus path to a confusion matrix image under `reports/`.

2. Add a `tools/ml/requirements.txt` (or update top-level requirements) listing pinned versions: pandas, scikit-learn, joblib, matplotlib, seaborn.

3. Add a small CLI wrapper or README showing how to run the training script and where outputs are produced.

4. Add tests:
   - Unit tests for preprocessing and TF-IDF vectorization under `tests/`.
   - A smoke/integration test that runs the training script on a small subsample (or with `--quick` flag) and verifies that `reports/evaluation.json` and the model artifact are produced.

5. Add a minimal OpenSpec spec delta (if not already present) documenting the requirement and one or more scenarios (dataset fetched, model trained, metrics produced). Reference: `openspec/changes/add-spam-classification-ml/specs/spam/spec.md`.

# Expected Outcome

- A reproducible baseline training pipeline that produces `models/svm_baseline.joblib` and `reports/evaluation.json`.
- A short evaluation report including accuracy, precision, recall, F1, and confusion matrix for the SVM baseline.
- Unit and smoke tests that exercise preprocessing and the end-to-end training run.
- Clear instructions in README for reproducing the baseline locally.

# Verification Steps

1. Run the data fetch/sanity script and confirm the CSV is downloaded to `data/raw/` and row/column checks pass.
2. Execute the training script:

   - Command (example):

     python tools/ml/train_baseline_svm.py --data-url "<csv-url>" --output-dir . --quick

   - Expected: `models/svm_baseline.joblib` and `reports/evaluation.json` are created.

3. Check `reports/evaluation.json` contains numeric fields: accuracy, precision, recall, f1.
4. Confirm the smoke test passes in CI: the smoke run should complete within a short time limit and produce the expected artifacts.
5. Review the artifacts and the spec entry in `openspec/specs/ml/spam-classification/spec.md` (or the change delta) and approve the proposal before Phase 2 work begins.

# Notes

- This proposal intentionally scopes Phase 1 to a prototype baseline (SVM). Phase 2 will replace or augment this with logistic regression and tuning as needed. The dataset is public; still include a citation in the README when distributing artifacts.
