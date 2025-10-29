# Project Context

## Purpose
This repository stores the project's OpenSpec-driven documentation, proposed changes, and (optionally) small tooling and reference implementations. The purpose of this repository is to keep the "single source of truth" about capabilities in `openspec/specs/`, and manage proposals for changing that truth in `openspec/changes/`.

Primary goals:
- Make requirements explicit and executable as specs
- Keep change proposals discoverable, reviewable, and auditable
- Provide lightweight tooling and conventions to help contributors create high-quality spec deltas

## Tech Stack (assumptions and recommendations)
The repository primarily contains Markdown-based specification artifacts (OpenSpec format). Typical supporting tools and runtimes recommended for contributors:
- OpenSpec tool (project CLI) — used for listing, validating, and archiving changes
- Git (GitHub/GitLab) — PR-based workflow
- ripgrep (`rg`) — for fast full-text search across specs
- Node.js + npm or Python (optional) — for any small automation or tooling
- Prettier / Markdownlint — for consistent Markdown formatting

If your project already uses a specific runtime (TypeScript, Python, Go, etc.), list it here and we can add language-specific conventions.

## Project Conventions

### Code & Markdown Style
- Use consistent Markdown headings and the OpenSpec scenario formatting (`#### Scenario:`) for all requirements.
- Keep lines wrapped at ~100 chars where helpful for diffs.
- Run `prettier --write` or your repository's formatter on any changed files before opening a PR.
- Use meaningful commit messages and reference change IDs when implementing proposals (see Git Workflow).

### Architecture Patterns
- Capabilities are modeled as single-purpose specs under `openspec/specs/<capability>/spec.md`.
- Changes are proposed under `openspec/changes/<change-id>/` and should be small, reviewable, and scoped to a single intent when possible.
- Prefer incremental changes: add new requirements for new behaviour rather than modifying existing requirements unless the change is explicitly breaking.

### Testing Strategy
- For spec validation, run `openspec validate <change-id> --strict` (preferred) prior to requesting approval.
- Implementation code should include unit tests and at least one integration/acceptance test that verifies the behaviour described in the spec scenario(s).
- When adding tooling or automation, include a tiny smoke test in the `changes/<change-id>/` tasks or CI job.

### Git Workflow
- Branch naming: `change/<change-id>-short-desc` or `feature/<short-desc>` when implementing an approved change.
- Commits should reference the change id in the commit message (e.g., `add-openspec-cli: scaffold CLI commands`).
- Create a PR targeting `main` (or the default branch). Link the proposal directory (e.g., `openspec/changes/add-openspec-cli`) in the PR description.
- Do not implement major changes until the proposal has been reviewed and approved (see OpenSpec workflow in `openspec/AGENTS.md`).

## Domain Context
- This repository is spec-centric: the source of truth for what the system must do is stored in `openspec/specs/`.
- Implementation code (if present) is considered secondary to the spec; code must implement the spec, not the other way around.

## Important Constraints
- All normative requirements MUST use SHALL/MUST and include at least one `#### Scenario:`.
- Follow the ADDED / MODIFIED / REMOVED delta format described in `openspec/AGENTS.md`.
- When proposing breaking changes, include a migration plan and mark the change as **BREAKING** in `proposal.md`.

```markdown
# Project Context

## Purpose
This repository stores the project's OpenSpec-driven documentation, proposed changes, and (optionally) small tooling and reference implementations. The purpose of this repository is to keep the "single source of truth" about capabilities in `openspec/specs/`, and manage proposals for changing that truth in `openspec/changes/`.

Primary goals:
- Make requirements explicit and executable as specs
- Keep change proposals discoverable, reviewable, and auditable
- Provide lightweight tooling and conventions to help contributors create high-quality spec deltas

## Tech Stack (assumptions and recommendations)
The repository primarily contains Markdown-based specification artifacts (OpenSpec format). Typical supporting tools and runtimes recommended for contributors:
- OpenSpec tool (project CLI) — used for listing, validating, and archiving changes
- Git (GitHub/GitLab) — PR-based workflow
- ripgrep (`rg`) — for fast full-text search across specs
- Node.js + npm or Python (optional) — for any small automation or tooling
- Prettier / Markdownlint — for consistent Markdown formatting

If your project already uses a specific runtime (TypeScript, Python, Go, etc.), list it here and we can add language-specific conventions.

## Project Conventions

### Code & Markdown Style
- Use consistent Markdown headings and the OpenSpec scenario formatting (`#### Scenario:`) for all requirements.
- Keep lines wrapped at ~100 chars where helpful for diffs.
- Run `prettier --write` or your repository's formatter on any changed files before opening a PR.
- Use meaningful commit messages and reference change IDs when implementing proposals (see Git Workflow).

### Architecture Patterns
- Capabilities are modeled as single-purpose specs under `openspec/specs/<capability>/spec.md`.
- Changes are proposed under `openspec/changes/<change-id>/` and should be small, reviewable, and scoped to a single intent when possible.
- Prefer incremental changes: add new requirements for new behaviour rather than modifying existing requirements unless the change is explicitly breaking.

### Testing Strategy
- For spec validation, run `openspec validate <change-id> --strict` (preferred) prior to requesting approval.
- Implementation code should include unit tests and at least one integration/acceptance test that verifies the behaviour described in the spec scenario(s).
- When adding tooling or automation, include a tiny smoke test in the `changes/<change-id>/` tasks or CI job.

### Git Workflow
- Branch naming: `change/<change-id>-short-desc` or `feature/<short-desc>` when implementing an approved change.
- Commits should reference the change id in the commit message (e.g., `add-openspec-cli: scaffold CLI commands`).
- Create a PR targeting `main` (or the default branch). Link the proposal directory (e.g., `openspec/changes/add-openspec-cli`) in the PR description.
- Do not implement major changes until the proposal has been reviewed and approved (see OpenSpec workflow in `openspec/AGENTS.md`).

## Domain Context
- This repository is spec-centric: the source of truth for what the system must do is stored in `openspec/specs/`.
- Implementation code (if present) is considered secondary to the spec; code must implement the spec, not the other way around.

## Important Constraints
- All normative requirements MUST use SHALL/MUST and include at least one `#### Scenario:`.
- Follow the ADDED / MODIFIED / REMOVED delta format described in `openspec/AGENTS.md`.
- When proposing breaking changes, include a migration plan and mark the change as **BREAKING** in `proposal.md`.

## External Dependencies
- Document any external APIs, services, or libraries that changes depend on inside the proposal and `design.md` (if required).

## How to Work With the Assistant
- When you ask the assistant to create a proposal, include a short goal, the target capability (if known), and whether the change is breaking.
- Example prompt: "Help me create a change proposal to add two-factor auth for the `auth` capability; change-id suggestion: `add-two-factor-auth`."
- The assistant will scaffold `proposal.md`, `tasks.md`, and the minimal spec deltas. It will also run local checks where possible.

---

If any of the assumptions above (Node.js, CLI availability, CI environment) are incorrect for this project, tell me what the real stack is and I will update this file accordingly.

## Spam classification project

### Overview
This workspace will include a small, reproducible machine learning project to classify spam vs. ham (non-spam) messages. The project follows a phased approach:
- Phase 1 — Baseline: build a simple SVM baseline to validate the pipeline and data ingestion.
- Phase 2 — Model iteration: train and compare logistic regression (and other models), run hyperparameter tuning, and produce comparison reports.
- Phase 3 — Hardening: add tests, CI smoke jobs, packaging of model artifacts, and optional Streamlit demo for quick inspection.

This work is intentionally scoped as a research/prototype effort — evaluation and metrics must inform any production decisions.

### Tech Stack
- Python 3.8+ (recommended)
- pandas for data loading and preprocessing
- scikit-learn for model training (SVM, LogisticRegression) and evaluation
- matplotlib / seaborn for plots (confusion matrix, ROC curves)
- joblib for model serialization
- Streamlit for an optional lightweight demo UI
- OpenSpec for documenting requirements, scenarios, and approval workflow

### Dataset
- Primary dataset (Phase 1): `sms_spam_no_header.csv` from Packt's example repo:
	https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- The dataset will be downloaded and cached under `data/raw/` by the pipeline. Include a citation and a simple sanity-check step that validates row counts and non-empty text fields before training.

### Directory structure (recommended)
```
.
├── tools/ml/                      # training scripts and CLI wrappers
│   ├── train_baseline_svm.py
│   ├── evaluate.py
│   └── requirements.txt
├── data/
│   ├── raw/                       # cached downloads (do not commit large files)
│   └── processed/                 # train/val/test splits
├── models/                        # serialized model artifacts (joblib)
├── notebooks/                     # exploratory notebooks (EDA)
├── reports/                       # evaluation reports, plots
├── tests/                         # unit and integration tests for preprocessing and pipeline
└── openspec/                      # specs and changes (existing)
```

### Conventions
- Use a Python virtual environment and commit `tools/ml/requirements.txt` with pinned versions for reproducibility.
- Data: always download into `data/raw/` and never commit raw data to Git. Use `.gitignore` to exclude large files.
- Reproducibility: set and document random seeds for train/test splits and model training.
- Artifact formats: serialize models with `joblib.dump()`; store metrics in `reports/evaluation.json` with fields: accuracy, precision, recall, f1, roc_auc, confusion_matrix_path.
- Experiment naming: use `experiment_<date>_<short-desc>` and persist a small `experiments.csv` to track parameters and metrics.

### Goals & Acceptance Criteria
- Phase 1 (SVM baseline):
	- Script downloads the Packt CSV, preprocesses text (simple tokenization + TF-IDF), trains an SVM, and writes `models/svm_baseline.joblib` and `reports/evaluation.json`.
	- `reports/evaluation.json` contains accuracy, precision, recall, and F1. A confusion matrix image is produced under `reports/`.
	- A smoke test runs the training script on a small subset and asserts the metrics file is created.
- Phase 2 (Logistic Regression):
	- Implement logistic regression training, compare metrics to SVM baseline, and produce a brief comparison report.
- Phase 3 (Hardening):
	- Add unit tests for core preprocessing steps and a CI job that runs the smoke training/test and fails on regressions.

### How this maps to OpenSpec
- Create a capability `ml/spam-classification` under `openspec/specs/` describing the expected artifacts and scenarios (data fetched, model trained, metrics produced). Use the `ADDED Requirements` format and include `#### Scenario:` entries to make the requirement testable and reviewable.

If you'd like, I can now scaffold the Python training script, `requirements.txt`, a README, and a small smoke test that trains on a subset of the dataset. Tell me if you prefer `venv` or `pipenv/poetry` and whether to pin package versions.

```
