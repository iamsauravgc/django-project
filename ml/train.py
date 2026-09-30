import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

import ml.preprocess as preprocess
from ml.preprocess import (
    CORE_EMOTIONS,
    DATA_DIR,
    MODEL_DIR,
    BoostedClassifier,
    SoftEnsemble,
    class_weights,
    load_tsv,
)

LABELS = CORE_EMOTIONS
C_GRID = [0.03, 0.1, 0.3, 1, 3]
SVC_PARAMS = {"class_weight": "balanced", "random_state": 42, "max_iter": 10000}
WEAK_CLASSES = ("anger", "surprise", "disgust")
WEAK_MULTIPLIERS = [1.0, 1.2, 1.5]

# All comparison is on the dev set; no test access until the final block.
VARIANTS = [
    {"name": "A word, stopwords (previous)", "stop": True, "kind": "word",
     "word_max": 10000, "word_ngram": (1, 2)},
    {"name": "B word, no stopwords", "stop": False, "kind": "word",
     "word_max": 10000, "word_ngram": (1, 2)},
    {"name": "C word+char, stopwords", "stop": True, "kind": "union",
     "word_max": 10000, "word_ngram": (1, 2), "char_max": 30000, "char_ngram": (3, 5)},
    {"name": "D word+char, no stopwords", "stop": False, "kind": "union",
     "word_max": 10000, "word_ngram": (1, 2), "char_max": 30000, "char_ngram": (3, 5)},
    {"name": "E word+char, no stop, wide", "stop": False, "kind": "union",
     "word_max": 30000, "word_ngram": (1, 2), "char_max": 60000, "char_ngram": (3, 5)},
    {"name": "G word(1,3)+char, no stop, wide", "stop": False, "kind": "union",
     "word_max": 30000, "word_ngram": (1, 3), "char_max": 60000, "char_ngram": (3, 5)},
    {"name": "H word(1,3)+char(2,6), wide", "stop": False, "kind": "union",
     "word_max": 30000, "word_ngram": (1, 3), "char_max": 80000, "char_ngram": (2, 6)},
    {"name": "I word(1,3)+2 chars, wide", "stop": False, "kind": "union",
     "word_max": 30000, "word_ngram": (1, 3), "char_max": 60000, "char_ngram": (3, 5),
     "char2_max": 60000, "char2_ngram": (2, 6)},
]


def macro_f1(model, X, y) -> float:
    return f1_score(y, model.predict(X), average="macro", labels=LABELS)


def f1_from_preds(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=LABELS)


def weak_scores(model, X, y) -> str:
    scores = f1_score(y, model.predict(X), labels=LABELS, average=None, zero_division=0)
    per = dict(zip(LABELS, scores))
    return " ".join(f"{k}={per[k]:.3f}" for k in ("surprise", "anger", "disgust"))


def load_splits(remove_stopwords: bool):
    kw = {"remove_stopwords": remove_stopwords}
    return {
        "train": load_tsv(DATA_DIR / "train.tsv", **kw),
        "dev": load_tsv(DATA_DIR / "dev.tsv", **kw),
    }


def build_vectorizer(spec: dict):
    word = TfidfVectorizer(
        max_features=spec["word_max"], ngram_range=spec["word_ngram"],
        min_df=2, sublinear_tf=True,
    )
    if spec["kind"] == "word":
        return word
    transformers = [("word", word)]
    for i, key in enumerate(("char", "char2"), start=1):
        if f"{key}_max" in spec:
            transformers.append((key, TfidfVectorizer(
                analyzer="char_wb", ngram_range=spec[f"{key}_ngram"],
                max_features=spec[f"{key}_max"], min_df=2, sublinear_tf=True,
            )))
    return FeatureUnion(transformers)


def weak_class_weights(y_train, multiplier: float) -> dict:
    """Balanced class weights with the weak classes upweighted by a multiplier."""
    weights = class_weights(y_train)
    if multiplier != 1.0:
        for label in WEAK_CLASSES:
            weights[label] *= multiplier
    return weights


def boost_sweep(probs, classes, y_dev) -> tuple[dict, float]:
    """Coordinate search for per-class decision boosts on dev.

    Three greedy passes over every class; probabilities stay untouched,
    only the argmax decision is biased. Returns (boosts, dev_macro_f1).
    """
    classes = list(classes)
    boosts = {c: 1.0 for c in classes}
    # fine grid: 0.05 steps to 1.0, 0.1 steps above
    grid = ([round(v, 2) for v in np.arange(0.6, 1.01, 0.05)]
            + [round(v, 2) for v in np.arange(1.1, 3.01, 0.1)])

    def score() -> float:
        weights = np.array([boosts[c] for c in classes])
        pred = np.asarray(classes)[np.argmax(probs * weights, axis=1)]
        return f1_from_preds(y_dev, pred)

    for _ in range(4):
        for c in classes:
            best_v, best_s = boosts[c], score()
            for v in grid:
                boosts[c] = v
                s = score()
                if s > best_s:
                    best_v, best_s = v, s
            boosts[c] = best_v
    return boosts, score()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true",
                        help="compare and tune on dev; skip the final test evaluation")
    parser.add_argument("--refit", action="store_true",
                        help="after all dev choices, refit the winner on train+dev "
                             "before the final test evaluation")
    args = parser.parse_args()

    frames_cache = {flag: load_splits(flag) for flag in (True, False)}

    # Baseline on dev (most-frequent ignores feature values)
    frames = frames_cache[True]
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(frames["train"]), 1)), frames["train"]["label"])
    print(f"Baseline (most_frequent) - dev macro-F1: "
          f"{macro_f1(baseline, np.zeros((len(frames['dev']), 1)), frames['dev']['label']):.4f}\n")

    best = None  # (dev_f1, spec, vec, X_train, X_dev)
    for spec in VARIANTS:
        frames = frames_cache[spec["stop"]]
        vec = build_vectorizer(spec)
        X_train = vec.fit_transform(frames["train"]["text"])
        X_dev = vec.transform(frames["dev"]["text"])
        model = LinearSVC(**SVC_PARAMS)
        model.fit(X_train, frames["train"]["label"])
        score = macro_f1(model, X_dev, frames["dev"]["label"])
        print(f"{spec['name']:34s} dev macro-F1: {score:.4f}  "
              f"{weak_scores(model, X_dev, frames['dev']['label'])}")
        if best is None or score > best[0]:
            best = (score, spec, vec, X_train, X_dev)

    best_dev, spec, vec, X_train, X_dev = best
    frames = frames_cache[spec["stop"]]
    y_train = frames["train"]["label"]
    y_dev = frames["dev"]["label"]
    print(f"\nBest feature config: {spec['name']}, remove_stopwords={spec['stop']} "
          f"(dev {best_dev:.4f})")

    # Model comparison on the winner features (decision-log model kept for production)
    sgd = SGDClassifier(loss="log_loss", class_weight="balanced",
                        random_state=42, max_iter=1000, tol=1e-3)
    sgd.fit(X_train, y_train)
    print(f"SGDClassifier (reference)  dev macro-F1: {macro_f1(sgd, X_dev, y_dev):.4f}")

    # C selected by dev score (planning brief step 10)
    best_c, c_model, c_dev = None, None, -1.0
    for c in C_GRID:
        model = LinearSVC(C=c, **SVC_PARAMS)
        model.fit(X_train, y_train)
        score = macro_f1(model, X_dev, y_dev)
        print(f"C={c:<6}                       dev macro-F1: {score:.4f}")
        if score > c_dev:
            best_c, c_model, c_dev = c, model, score
    print(f"Best C: {best_c} (dev {c_dev:.4f})")

    # Weak-class weight multiplier sweep on dev
    best_m, plain, plain_dev = 1.0, c_model, c_dev
    for m in WEAK_MULTIPLIERS[1:]:
        model = LinearSVC(C=best_c, max_iter=SVC_PARAMS["max_iter"], random_state=42,
                          class_weight=weak_class_weights(y_train, m))
        model.fit(X_train, y_train)
        score = macro_f1(model, X_dev, y_dev)
        print(f"Plain + weak x{m:<4}              dev macro-F1: {score:.4f}")
        if score > plain_dev:
            best_m, plain, plain_dev = m, model, score
    print(f"Best plain: weak x{best_m} (dev {plain_dev:.4f})")

    # Plain vs calibrated on dev
    production, production_dev, kind = plain, plain_dev, f"plain (weak x{best_m})"
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            LinearSVC(C=best_c, class_weight=plain.class_weight,
                      random_state=42, max_iter=SVC_PARAMS["max_iter"]),
            cv=3, method=method,
        )
        calibrated.fit(X_train, y_train)
        score = macro_f1(calibrated, X_dev, y_dev)
        print(f"Calibrated ({method:8s})         dev macro-F1: {score:.4f}")
        if score > production_dev:
            production, production_dev, kind = calibrated, score, f"calibrated {method} (weak x{best_m})"
    print(f"Production model: {kind} LinearSVC (dev {production_dev:.4f})")

    # Per-class decision boost search on the single production model
    single_boosts, single_boosted = boost_sweep(
        production.predict_proba(X_dev), production.classes_, y_dev)
    print(f"Per-class boosts (single): {single_boosts}")
    print(f"Boosted dev macro-F1: {single_boosted:.4f} (from {production_dev:.4f})")

    # Soft ensemble of calibrated LinearSVCs at nearby C values
    member_cs = sorted({0.05, 0.1, 0.3, best_c})
    members = []
    for c in member_cs:
        member = CalibratedClassifierCV(
            LinearSVC(C=c, class_weight=plain.class_weight,
                      random_state=42, max_iter=SVC_PARAMS["max_iter"]),
            cv=3, method="sigmoid",
        )
        member.fit(X_train, y_train)
        members.append(member)
    ensemble = SoftEnsemble(members)
    ens_boosts, ens_boosted = boost_sweep(
        ensemble.predict_proba(X_dev), ensemble.classes_, y_dev)
    print(f"Soft ensemble C={member_cs} + boosts: dev {ens_boosted:.4f}")

    # SGD + same boost search, for reference only (LinearSVC stays per decision log)
    sgd_cal = CalibratedClassifierCV(sgd, cv=3, method="sigmoid")
    sgd_cal.fit(X_train, y_train)
    _, sgd_boosted = boost_sweep(sgd_cal.predict_proba(X_dev), sgd_cal.classes_, y_dev)
    print(f"SGD calibrated + boosts: dev {sgd_boosted:.4f} (reference only)")

    # Final production pick by boosted dev score (ensemble must win clearly:
    # at display precision it has tied the single model, so prefer the simpler
    # decision-log model on anything less than a 0.0005 margin)
    if ens_boosted > single_boosted + 0.0005:
        final, final_boosts = ensemble, ens_boosts
        kind = f"soft ensemble C={member_cs} (boosted)"
    else:
        final, final_boosts = production, single_boosts
        kind = f"{kind} (boosted)"
    final_dev = max(ens_boosted, single_boosted)
    print(f"Production: {kind} - dev {final_dev:.4f}")

    if args.dev_only:
        print("\n--dev-only: artifacts not saved, test set not evaluated.")
        return

    # Optional standard final step: refit the exact winner on train+dev.
    # Every hyperparameter above was chosen on dev before this line and the
    # test set stays untouched, so the final evaluation remains unbiased.
    save_vec, save_model = vec, final
    if args.refit:
        combined = pd.concat([frames["train"], frames["dev"]], ignore_index=True)
        save_vec = build_vectorizer(spec)
        X_full = save_vec.fit_transform(combined["text"])
        save_model = clone(final)
        save_model.fit(X_full, combined["label"])
        print(f"\nRefit winner on train+dev ({len(combined):,} rows, "
              f"{X_full.shape[1]:,} features)")

    # Save fitted production vectorizer + production model (boosted wrapper)
    use_boosts = any(v != 1.0 for v in final_boosts.values())
    saved = BoostedClassifier(save_model, final_boosts) if use_boosts else save_model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(save_vec, f)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(saved, f)
    print(f"\nSaved {MODEL_DIR / 'model.pkl'} ({'boosted wrapper' if use_boosts else 'plain'}) "
          f"and {MODEL_DIR / 'vectorizer.pkl'}")
    (MODEL_DIR / "training_scope.txt").write_text(
        ("train+dev" if args.refit else "train") + "\n", encoding="utf-8")

    if preprocess.DEFAULT_REMOVE_STOPWORDS != spec["stop"]:
        print(
            f"WARNING: ml/preprocess.DEFAULT_REMOVE_STOPWORDS is "
            f"{preprocess.DEFAULT_REMOVE_STOPWORDS} but the artifacts were trained with "
            f"remove_stopwords={spec['stop']}. Set DEFAULT_REMOVE_STOPWORDS = {spec['stop']} "
            f"in ml/preprocess.py and re-run so serving matches training."
        )

    # Final test evaluation - once, after every choice is made
    test = load_tsv(DATA_DIR / "test.tsv", remove_stopwords=spec["stop"])
    X_test = save_vec.transform(test["text"])
    classes = np.asarray(save_model.classes_)
    probs = save_model.predict_proba(X_test)
    raw = classes[np.argmax(probs, axis=1)]
    print("\nFinal test evaluation")
    print(classification_report(test["label"], raw, labels=LABELS, digits=4))
    print(f"Test macro-F1 (raw argmax): {f1_from_preds(test['label'], raw):.4f}")
    if use_boosts:
        weights = np.array([final_boosts.get(c, 1.0) for c in classes])
        boosted = probs * weights
        boosted = boosted / boosted.sum(axis=1, keepdims=True)
        y_pred = classes[np.argmax(boosted, axis=1)]
        print(f"\nWith decision boosts {final_boosts}")
        print(classification_report(test["label"], y_pred, labels=LABELS, digits=4))
        print(f"Test macro-F1 (served): {f1_from_preds(test['label'], y_pred):.4f}")


if __name__ == "__main__":
    main()
