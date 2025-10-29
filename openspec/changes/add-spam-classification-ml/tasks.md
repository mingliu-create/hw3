## 1. Implementation
- [ ] 1.1 Create `tools/ml/train_baseline_svm.py` to fetch the dataset, preprocess text, train an SVM classifier, and save model + metrics.
- [ ] 1.2 Add a small `tools/ml/requirements.txt` (or update top-level `requirements.txt`) listing exact package versions (e.g., scikit-learn, pandas, numpy).
- [ ] 1.3 Add data fetching and sanity-check script that downloads from the provided Packt CSV URL.
- [ ] 1.4 Add a README with usage examples and expected outputs.

## 2. Evaluation
- [ ] 2.1 Produce evaluation outputs: accuracy, precision, recall, F1, confusion matrix, and a CSV with sample predictions.
- [ ] 2.2 Create a small notebook or script that reproduces EDA and basic model analysis.

## 3. Tests
- [ ] 3.1 Add unit tests for preprocessing functions (e.g., tokenization, TF-IDF vectorization).
- [ ] 3.2 Add a smoke/integration test that runs the training script with a small subset of the dataset and verifies it completes and writes metrics.

## 4. Documentation & PR
- [ ] 4.1 Document the pipeline and how to run locally in `openspec/changes/add-spam-classification-ml/README.md`.
- [ ] 4.2 Open a PR linking this proposal, include sample outputs and evaluation summary.
