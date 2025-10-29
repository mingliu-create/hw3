## Change ID
add-spam-classification-ml

## Title
Spam Email Classification (baseline + roadmap to logistic regression)

## Target capability
ml/spam-classification

## Why
Many security workflows need automated spam detection for emails and messages. This change proposes a small, low-risk machine learning project to build a baseline spam classifier (SVM baseline), gather evaluation metrics, and then iterate to a logistic regression model and improvements.

Primary goals:
- Build a working baseline classifier and a reproducible training pipeline.
- Use a publicly-available dataset for quick iteration and reproducibility.
- Produce evaluation artifacts (metrics, confusion matrix, sample predictions) to guide improvements.

## Data Source
We will use the dataset hosted at:
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## What Changes
- ADDED: `ml/spam-classification` training scripts and docs under `tools/ml/` (or `scripts/`) that produce a baseline SVM model and evaluation report.
- ADDED: A reproducible pipeline to fetch and preprocess the CSV dataset, split train/validation/test, train a baseline SVM, and output metrics (accuracy, precision, recall, F1) and an artifact (serialized model and sample predictions).
- ADDED (Phase 2): Train and evaluate a logistic regression classifier and compare with baseline.

## Phases
1. Phase 1 — Baseline (SVM): fetch dataset, EDA, preprocessing (tokenization, TF-IDF or simple bag-of-words), train an SVM baseline, and produce evaluation report.
2. Phase 2 — Logistic regression: implement logistic regression model(s), hyperparameter tuning, compare results with SVM baseline.
3. Phase 3 — Hardening: add tests, CI job for training smoke-test, packaging model artifacts, and optional deployment instructions.

## Impact
- Affects only dev tooling and documentation; no production runtime changes.
- New files under `tools/ml/` (or `scripts/ml/`) and `openspec/changes/add-spam-classification-ml/` for the proposal/specs.

## Risks & Mitigations
- Risk: Dataset licensing/quality — Mitigation: dataset is public from Packt; include citation and sanity-check script.
- Risk: ML model not robust — Mitigation: clearly mark this as prototype and include evaluation criteria and thresholds before any production use.

## Acceptance Criteria
- A reproducible script/pipeline that downloads the CSV, trains a baseline SVM, and writes evaluation metrics to disk.
- The change includes `tasks.md` with implementation, test steps, and verification guidance.
- At least one spec requirement with a scenario that describes the baseline behavior and artifacts produced.

## Files to Create / Update
- `tools/ml/train_baseline_svm.py` (or `scripts/ml/train_baseline_svm.py`) — training script
- `tools/ml/requirements.txt` or `requirements.txt` (if using Python)
- `openspec/changes/add-spam-classification-ml/tasks.md` — implementation checklist
- `openspec/changes/add-spam-classification-ml/specs/spam/spec.md` — spec delta describing the requirement and scenario(s)

## Notes
- This proposal focuses on reproducibility and clear evaluation. It intentionally keeps the scope small (baseline first) to allow rapid experimentation.
