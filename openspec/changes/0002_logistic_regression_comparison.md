---
change-id: 0002_logistic_regression_comparison
title: Compare Logistic Regression vs SVM for spam-classification (Phase 2)
author: TBD
status: proposed
---

# Summary

This change proposal adds Phase 2 of the spam-classification project: a structured comparison between Logistic Regression and the Phase 1 SVM baseline. The work includes training both models on the Packt SMS spam dataset, evaluating them on identical splits, and producing visualizations and reports that make model trade-offs clear to reviewers.

# Motivation

Phase 1 produced a working SVM baseline and evaluation artifacts. To guide model selection for future work, we must compare a Logistic Regression model against the SVM baseline using consistent preprocessing, training, and evaluation. Logistic Regression often yields interpretable coefficients and probability estimates which can be valuable for downstream decision logic and calibration.

# Implementation Plan

1. Data & preprocessing (reuse Phase 1 pipeline):
   - Reuse the dataset download and caching logic from Phase 1 (`data/raw/sms_spam_no_header.csv`).
   - Use the same preprocessing pipeline (lowercase, remove punctuation). Optionally add TF-IDF as an experimental variant.
   - Ensure train/val/test splits are identical across models (use fixed random seed and store split indices for reproducibility).

2. Training:
   - Implement `src/train_compare_models.py` which trains:
     - Logistic Regression (`sklearn.linear_model.LogisticRegression`) with solver='liblinear' or 'saga' depending on scale.
     - SVM baseline (re-run Phase 1 pipeline or load Phase 1 model if desired for parity).
   - Support hyperparameter options (regularization C for LogisticRegression / kernel/C for SVM) and a `--grid` flag to run a small grid search for each model.

3. Evaluation & visualization:
   - Compute metrics for each model: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
   - Produce direct comparison visualizations saved to `reports/phase2/`:
     - Side-by-side bar chart of metrics (accuracy, precision, recall, F1) for both models.
     - ROC curves (both models) on the same plot.
     - Calibration plot for Logistic Regression (reliability diagram) to inspect probability calibration.
     - Confusion matrices for both models.
     - Coefficient-based top-positive/top-negative words for Logistic Regression (words with largest positive/negative coefficients) saved as horizontal bar charts.

4. Artifacts & outputs:
   - Serialized model artifacts under `models/` with clear filenames: `models/logreg_phase2.joblib`, `models/svm_phase2.joblib` (or reuse baseline name if re-trained).
   - `reports/phase2/evaluation_summary.json` listing metrics for both models and paths to generated plots.
   - A short Markdown report `reports/phase2/README.md` summarizing key findings and recommended next steps.

5. Tests & reproducibility:
   - Add a smoke mode (`--quick`) that trains both models on a sample subset and validates that artifacts and `evaluation_summary.json` are produced.
   - Unit tests for any new preprocessing or evaluation helpers.

# Expected Outcome

- A reproducible comparison showing numeric metrics and visualizations for Logistic Regression vs SVM on identical datasets and splits.
- Easy-to-read artifacts in `reports/phase2/` and serialized models under `models/` so reviewers can reproduce and inspect results.
- A short recommendation in `reports/phase2/README.md` explaining which model to prefer given observed metrics and trade-offs.

# Verification Steps

1. Run the compare training script in quick mode to verify artifacts are produced:

   python src/train_compare_models.py --quick --output-dir .

   Expected: `models/logreg_phase2.joblib`, `models/svm_phase2.joblib`, and `reports/phase2/evaluation_summary.json` are created.

2. Run a full comparison without `--quick` (may take longer) and inspect `reports/phase2/` for the visualizations:
   - `metrics_comparison.png` (bar chart)
   - `roc_curves.png`
   - `calibration_logreg.png`
   - `confusion_matrix_logreg.png`, `confusion_matrix_svm.png`
   - `top_coeff_words_logreg.png`

3. Confirm that train/val/test splits match Phase 1 splits (compare a stored `splits.json` file or reproduce with fixed seed).
4. Review `reports/phase2/README.md` and accept the proposal if the results and recommended model meet acceptance criteria.

# Files to add (examples)

- `src/train_compare_models.py` — main script to train and compare Logistic Regression and SVM.
- `tools/ml/requirements.txt` — pin versions if not already present.
- `reports/phase2/README.md` — human-readable summary of findings.
- Unit tests under `tests/` for any new helper functions.

# Tasks (starter checklist)

- [ ] 1. Implement `src/train_compare_models.py` using scikit-learn (LogisticRegression and SVM) with `--quick` support.
- [ ] 2. Produce visualizations and write `reports/phase2/evaluation_summary.json`.
- [ ] 3. Add unit & smoke tests and update `openspec/changes/0002_logistic_regression_comparison.md` if needed.
- [ ] 4. Open PR linking this change and request review; include key plots and a short summary.

# Notes

- This proposal assumes Phase 1 artifacts and data download code remain available. Use consistent preprocessing and vectorization across models to ensure a fair comparison.
- Logistic Regression offers interpretable coefficients and probability outputs. If calibration is poor, consider Platt scaling or isotonic regression as follow-up steps.
