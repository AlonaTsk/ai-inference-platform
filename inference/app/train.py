"""Train and evaluate the ticket classifier.

Usage: python -m app.train --out model/
Writes model.joblib and metrics.json. The test split is held out before
fitting, so the TF-IDF vocabulary never sees test tickets (no leakage).
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.data import LABELS, generate

SEED = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model")
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    texts, labels = generate(n=args.samples, seed=SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED)

    model = build_pipeline()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "synthetic support tickets (see app/data.py)",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            l: {k: round(report[l][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for l in LABELS
        },
        "confusion_matrix": {"labels": LABELS,
                             "rows_true_cols_pred": confusion_matrix(y_test, pred, labels=LABELS).tolist()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(classification_report(y_test, pred, labels=LABELS, digits=3, zero_division=0))
    print(f"saved model and metrics to {out}/")


if __name__ == "__main__":
    main()
