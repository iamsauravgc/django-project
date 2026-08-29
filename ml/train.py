"""
Training and evaluation for the emotion detector.

Runs a most-frequent baseline, trains LinearSVC and SGDClassifier, tunes the
SVC's C parameter, saves the tuned SVC, and reports the final test macro-F1.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pickle

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from ml.preprocess import CORE_EMOTIONS, MODEL_DIR, prepare_data

LABELS = CORE_EMOTIONS


def macro_f1(model, X, y) -> float:
    return f1_score(y, model.predict(X), average="macro", labels=LABELS)


def baseline(data):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(data["X_train"], data["y_train"])
    score = macro_f1(model, data["X_dev"], data["y_dev"])
    print(f"Baseline (most_frequent) - dev macro-F1: {score:.4f}")
    return model


def train_svc(data):
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(data["X_train"], data["y_train"])
    score = macro_f1(model, data["X_dev"], data["y_dev"])
    print(f"LinearSVC              - dev macro-F1: {score:.4f}")
    return model


def train_sgd(data):
    model = SGDClassifier(
        loss="log_loss", class_weight="balanced", random_state=42, max_iter=1000, tol=1e-3
    )
    model.fit(data["X_train"], data["y_train"])
    score = macro_f1(model, data["X_dev"], data["y_dev"])
    print(f"SGDClassifier          - dev macro-F1: {score:.4f}")
    return model


def tune(data) -> LinearSVC:
    grid = GridSearchCV(
        LinearSVC(class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10]},
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(data["X_train"], data["y_train"])
    print(f"Tuned LinearSVC - best C={grid.best_params_['C']} (cv macro-F1 {grid.best_score_:.4f})")
    return grid.best_estimator_


def final_eval(model, data):
    y_true = data["y_test"]
    y_pred = model.predict(data["X_test"])
    print("\nFinal test evaluation")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print(f"Test macro-F1: {f1_score(y_true, y_pred, average='macro', labels=LABELS):.4f}")


def save_artifacts(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_DIR / 'model.pkl'}")


def main():
    data = prepare_data()

    baseline(data)
    svc = train_svc(data)
    train_sgd(data)

    model = tune(data)
    save_artifacts(model)

    # keep the untuned SVC result available for reference
    _ = svc
    final_eval(model, data)


if __name__ == "__main__":
    main()
