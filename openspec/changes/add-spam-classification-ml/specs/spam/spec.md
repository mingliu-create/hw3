## ADDED Requirements

### Requirement: Baseline Spam Classification Pipeline
The project SHALL provide a reproducible baseline spam classification pipeline that downloads the dataset from the specified Packt CSV URL, preprocesses the text data, trains a baseline SVM model, and outputs evaluation artifacts (metrics, confusion matrix, and sample predictions).

#### Scenario: Download and sanity-check dataset
- **WHEN** the repository provides the data fetch script
- **THEN** running the script downloads `sms_spam_no_header.csv`, verifies row counts and basic column sanity (non-empty text entries), and writes a local cached copy.

#### Scenario: Train baseline SVM and produce metrics
- **WHEN** a contributor runs the baseline training script with default arguments
- **THEN** the script trains an SVM on the training split, saves a serialized model artifact, and writes an `evaluation.json` containing accuracy, precision, recall, F1, and a path to a confusion matrix image and `predictions_sample.csv`.

### Requirement: Roadmap to Logistic Regression
The project SHOULD include a follow-up phase to implement logistic regression training, hyperparameter tuning, and a comparison report against the SVM baseline.

#### Scenario: Logistic regression roadmap exists
- **WHEN** Phase 1 completes and artifacts are available
- **THEN** a documented `Phase 2` plan shall be included describing data transformations, tuning strategy, and evaluation comparison criteria.
