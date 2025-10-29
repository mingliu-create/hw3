#!/usr/bin/env python3
"""Train a baseline SVM spam classifier.

Downloads the Packt CSV, preprocesses text, vectorizes using CountVectorizer,
trains a LinearSVC, evaluates on a test split and writes artifacts:
- models/svm_baseline.joblib
- models/vectorizer.joblib
- reports/evaluation.json
- reports/confusion_matrix.png

Usage:
    python src/train_baseline_svm.py --data-url <csv-url> --output-dir . [--quick]

The --quick flag will sample a small subset for smoke tests.
"""

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


DEFAULT_CSV = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def preprocess_text(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    # lowercase
    s = s.str.lower()
    # remove punctuation (keep word chars and whitespace)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    # collapse whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_and_save_confusion(cm, labels, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train baseline SVM spam classifier")
    parser.add_argument("--data-url", default=DEFAULT_CSV, help="CSV URL to download dataset")
    parser.add_argument("--output-dir", default=".", help="Output directory for models/reports")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test on a sample of the data")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size when --quick is used")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    phase1_dir = reports_dir / "phase1"
    data_dir = out_dir / "data" / "raw"

    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    ensure_dir(phase1_dir)
    ensure_dir(data_dir)

    print("Downloading dataset from:", args.data_url)
    try:
        df = pd.read_csv(args.data_url, header=None, encoding="latin-1", usecols=[0, 1])
    except Exception as e:
        print("Failed to download or read CSV:", e)
        raise

    # dataset expected to be two columns: label, text
    df.columns = ["label", "text"]
    print(f"Downloaded dataset with {len(df)} rows")

    # quick mode: sample fewer rows for smoke tests
    if args.quick:
        n = min(args.sample_size, len(df))
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
        print(f"Quick mode: sampled {n} rows")

    # persist raw csv to cache
    raw_path = data_dir / "sms_spam_no_header.csv"
    try:
        df.to_csv(raw_path, index=False, header=False)
        print("Cached raw dataset to:", raw_path)
    except Exception:
        print("Warning: could not write raw dataset cache")

    # basic sanity checks
    non_empty = df['text'].str.strip().astype(bool).sum()
    if non_empty == 0:
        raise RuntimeError("No non-empty text entries found in dataset")

    df['text_clean'] = preprocess_text(df['text'])

    X = df['text_clean']
    y = df['label']

    # train / val / test split: 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    relative_val_size = 0.15 / (1.0 - 0.15)  # 0.15 of original => fraction of trainval
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=relative_val_size, stratify=y_trainval, random_state=42)

    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Vectorize
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Use a linear SVM classifier (via SGDClassifier with hinge loss for stability in many environments)
    clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
    print("Training classifier...")
    clf.fit(X_train_vec, y_train)

    # Predict on train/val/test
    y_train_pred = clf.predict(X_train_vec)
    y_val_pred = clf.predict(X_val_vec)
    y_test_pred = clf.predict(X_test_vec)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    cls_report = classification_report(y_test, y_test_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=list(sorted(df['label'].unique())))

    eval_obj = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "classification_report": cls_report,
        "labels": list(sorted(df['label'].unique()))
    }

    eval_path = reports_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=2)
    print("Wrote evaluation metrics to:", eval_path)

    cm_path = phase1_dir / "confusion_matrix.png"
    plot_and_save_confusion(cm, eval_obj['labels'], cm_path)
    print("Saved confusion matrix to:", cm_path)

    # Plot train/val/test accuracy bar chart
    try:
        acc_fig_path = phase1_dir / "accuracy_train_val_test.png"
        plt.figure(figsize=(6, 4))
        acc_vals = [train_acc, val_acc, test_acc]
        acc_names = ["train", "val", "test"]
        sns.barplot(x=acc_names, y=acc_vals, palette="muted")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Train / Val / Test Accuracy")
        plt.tight_layout()
        plt.savefig(acc_fig_path)
        plt.close()
        print("Saved accuracy plot to:", acc_fig_path)
    except Exception as e:
        print("Warning: failed to save accuracy plot:", e)

    # Top indicative words by mean frequency difference (spam - ham) in training set
    try:
        feature_names = vectorizer.get_feature_names_out()
        # compute mean feature frequency per class on the training set
        X_train_arr = X_train_vec.toarray()
        classes = np.array(y_train)
        unique_labels = np.unique(classes)
        # handle binary spam/ham expectation
        if len(unique_labels) == 2:
            lab1, lab2 = unique_labels
            mean1 = X_train_arr[classes == lab1].mean(axis=0)
            mean2 = X_train_arr[classes == lab2].mean(axis=0)
            # we want top words more indicative of spam vs ham
            # decide which label is 'spam' if present
            if 'spam' in unique_labels:
                spam_label = 'spam'
                other_label = [l for l in unique_labels if l != 'spam'][0]
            else:
                # pick second as positive
                spam_label = lab2
                other_label = lab1

            mean_spam = X_train_arr[classes == spam_label].mean(axis=0)
            mean_other = X_train_arr[classes == other_label].mean(axis=0)
            diff = mean_spam - mean_other

            top_n = 20
            top_spam_idx = np.argsort(-diff)[:top_n]
            top_ham_idx = np.argsort(diff)[:top_n]

            def save_top_words(idx_list, labels, out_path):
                words = [feature_names[i] for i in idx_list]
                scores = diff[idx_list]
                plt.figure(figsize=(6, max(4, len(words) * 0.25)))
                sns.barplot(x=scores, y=words, palette="vlag")
                plt.xlabel("Mean frequency difference")
                plt.title(labels)
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()

            spam_words_path = phase1_dir / "top_words_spam.png"
            ham_words_path = phase1_dir / "top_words_ham.png"
            save_top_words(top_spam_idx, f"Top words indicative of {spam_label}", spam_words_path)
            save_top_words(top_ham_idx, f"Top words indicative of {other_label}", ham_words_path)
            print("Saved top words plots to:", spam_words_path, ham_words_path)
    except Exception as e:
        print("Warning: failed to compute/save top indicative words:", e)

    # Save model and vectorizer
    model_path = models_dir / "svm_baseline.joblib"
    vect_path = models_dir / "vectorizer.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vect_path)
    print("Saved model to:", model_path)
    print("Saved vectorizer to:", vect_path)


if __name__ == "__main__":
    main()
