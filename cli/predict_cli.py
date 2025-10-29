#!/usr/bin/env python3
"""Simple CLI to load trained model and vectorizer and predict a text label.

Usage:
    python cli/predict_cli.py --text "Free entry in 2 a wkly comp" \
        --model models/svm_baseline.joblib --vectorizer models/vectorizer.joblib
"""

import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser(description="Quick CLI to predict spam/ham with trained model")
    parser.add_argument("--model", default="models/svm_baseline.joblib", help="Path to trained model joblib")
    parser.add_argument("--vectorizer", default="models/vectorizer.joblib", help="Path to vectorizer joblib")
    parser.add_argument("--text", help="Text to classify; if omitted, prompt interactively")
    args = parser.parse_args()

    model_path = Path(args.model)
    vect_path = Path(args.vectorizer)

    if not model_path.exists() or not vect_path.exists():
        print("Model or vectorizer not found. Run training first to produce these files:")
        print("  models/svm_baseline.joblib and models/vectorizer.joblib")
        return

    clf = joblib.load(model_path)
    vect = joblib.load(vect_path)

    if args.text:
        text = args.text
    else:
        text = input("Enter text to classify: ")

    X = vect.transform([text])
    pred = clf.predict(X)[0]
    print(f"Prediction: {pred}")

    # decision score (not a probability for SGDClassifier)
    try:
        score = clf.decision_function(X)
        print("Decision score:", score[0])
    except Exception:
        pass


if __name__ == "__main__":
    main()
