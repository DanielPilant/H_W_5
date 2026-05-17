"""
train_models.py - Step 3 (Offline Training) of the Real-Time Spam Filter project.

Loads `emails_master.csv`, splits 70/30 stratified, fits a StandardScaler,
trains Logistic Regression / SVM / Random Forest, prints classification
reports + a side-by-side comparison table, and pickles each model + the
scaler for use by the real-time consumer.
"""

import pickle
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
INPUT_CSV = "emails_master.csv"
FEATURE_PREFIXES = ("word_freq_", "char_freq_", "capital_")
TARGET_NAMES = ["ham", "spam"]
SPAM_LABEL = 1

MODEL_FILES = {
    "logistic": "model_logistic.pkl",
    "svm": "model_svm.pkl",
    "rf": "model_rf.pkl",
}
SCALER_FILE = "scaler.pkl"


# ---------------------------------------------------------------------------
def load_data():
    print(f"Loading data from '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV)

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    if len(feature_cols) != 57:
        print(f"Warning: expected 57 feature columns, found {len(feature_cols)}")

    X = df[feature_cols].values
    y = df["label"].astype(int).values
    print(f"  rows={len(df)}  features={len(feature_cols)}  classes={dict(pd.Series(y).value_counts())}")
    return X, y, feature_cols


def build_models():
    return {
        "logistic": LogisticRegression(C=1.0, max_iter=1000),
        "svm":      SVC(kernel="rbf", C=1.0, probability=True),
        "rf":       RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
    }


def print_comparison_table(results):
    """Pretty-print a Classifier x {Accuracy, Precision, Recall, F1} grid."""
    header = f"{'Classifier':<22} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("Model comparison (precision/recall/F1 reported for class 'spam')")
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:>10.4f} {m['precision']:>12.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
def main():
    X, y, _ = load_data()

    print("Splitting (70/30, stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name.upper()} ===")
        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0
        print(f"Training time: {train_time:.4f}s")

        y_pred = model.predict(X_test_s)
        print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=SPAM_LABEL, zero_division=0),
            "recall":    recall_score(y_test, y_pred, pos_label=SPAM_LABEL, zero_division=0),
            "f1":        f1_score(y_test, y_pred, pos_label=SPAM_LABEL, zero_division=0),
        }

        with open(MODEL_FILES[name], "wb") as f:
            pickle.dump(model, f)
        print(f"Saved -> {MODEL_FILES[name]}")

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved -> {SCALER_FILE}")

    print_comparison_table(results)
    print("Offline training complete.")


if __name__ == "__main__":
    main()
