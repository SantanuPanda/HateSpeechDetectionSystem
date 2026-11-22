# -*- coding: utf-8 -*-
"""
Hate Speech Detection - Training Script
Trains multiple classifiers on tweet data and saves the best model (Stacking)
along with the TF-IDF vectorizer for use in app.py.
"""

import pandas as pd
import numpy as np
import re
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
dataset_file = "train.csv" if os.path.exists("train.csv") else "test.csv"
df = pd.read_csv(dataset_file)
print(f"Loaded {dataset_file} with shape: {df.shape}")
print(df.head())

# ── 2. Exploratory: tweet length distribution ─────────────────────────────────
df["tweet_len"] = df["tweet"].str.len()
plt.figure(figsize=(8, 4))
sns.histplot(df["tweet_len"], bins=30, color="red", kde=True)
plt.title("Distribution of Tweet Lengths")
plt.xlabel("Length of Tweet")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("tweet_length_distribution.png", dpi=100)
plt.show()
print("Saved: tweet_length_distribution.png")

# ── 3. Text preprocessing ─────────────────────────────────────────────────────
print("\nLoading NLTK stopwords...")
# Add the user roaming path where stopwords are stored on Windows
import os
nltk.data.path.append(os.path.join(os.environ.get("APPDATA", ""), "nltk_data"))
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove links
    text = re.sub(r"\@\w+|\#", "", text)                   # remove mentions & hashtags
    text = re.sub(r"[^\w\s]", "", text)                    # remove punctuation
    text = text.lower()
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

print("Cleaning tweets...")
df["clean_text"] = df["tweet"].apply(clean_text)
print(df[["tweet", "clean_text"]].head())

# ── 4. TF-IDF vectorization ───────────────────────────────────────────────────
print("\nVectorizing with TF-IDF (max_features=8000)...")
vectorizer = TfidfVectorizer(max_features=8000)
X = vectorizer.fit_transform(df["clean_text"])

# ── 5. Labels ─────────────────────────────────────────────────────────────────
if "label" in df.columns:
    print("\nUsing ground-truth labels from dataset:")
    print(df["label"].value_counts())
    y = df["label"].values
else:
    print("\nNote: No label column found — using synthetic labels.")
    np.random.seed(42)
    y = np.random.randint(0, 2, size=X.shape[0])

# ── 6. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 7. Balance with SMOTETomek ────────────────────────────────────────────────
print("\nApplying SMOTETomek balancing...")
print("Before SMOTE:\n", pd.Series(y_train).value_counts().to_string())
smote_tomek = SMOTETomek(random_state=42)
X_train_sm, y_train_sm = smote_tomek.fit_resample(X_train, y_train)
print("After  SMOTE:\n", pd.Series(y_train_sm).value_counts().to_string())

# ── 8. Train all models ───────────────────────────────────────────────────────
def train_eval(name, clf, X_tr, y_tr, X_te, y_te):
    print(f"\n--- {name} ---")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds))
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_te, preds), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return clf, preds, acc

log_model,     y_pred_log,     acc_log     = train_eval("Logistic Regression",  LogisticRegression(),                                                   X_train_sm, y_train_sm, X_test, y_test)
svm_model,     y_pred_svm,     acc_svm     = train_eval("SVM",                  SVC(),                                                                   X_train_sm, y_train_sm, X_test, y_test)
rf_model,      y_pred_rf,      acc_rf      = train_eval("Random Forest",         RandomForestClassifier(random_state=42),                                 X_train_sm, y_train_sm, X_test, y_test)
nb_model,      y_pred_nb,      acc_nb      = train_eval("Naïve Bayes",           MultinomialNB(),                                                         X_train_sm, y_train_sm, X_test, y_test)
bag_model,     y_pred_bag,     acc_bag     = train_eval("Bagging",               BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42), X_train_sm, y_train_sm, X_test, y_test)

stacking_model, y_pred_stack, acc_stack = train_eval(
    "Stacking (RF + XGBoost → LR)",
    StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(random_state=42)),
            ("xgb", XGBClassifier(random_state=42, eval_metric="logloss")),
        ],
        final_estimator=LogisticRegression(),
    ),
    X_train_sm, y_train_sm, X_test, y_test,
)

# ── 9. Model comparison chart ─────────────────────────────────────────────────
model_names = ["Logistic Regression", "SVM", "Random Forest", "Naïve Bayes", "Bagging", "Stacking"]
accuracies  = [acc_log, acc_svm, acc_rf, acc_nb, acc_bag, acc_stack]

plt.figure(figsize=(8, 4))
sns.barplot(y=model_names, x=accuracies, orient="h")
plt.xlabel("Accuracy")
plt.title("Model Comparison")
for i, v in enumerate(accuracies):
    plt.text(v + 0.002, i, f"{v:.3f}", va="center")
plt.xlim(max(0, min(accuracies) - 0.02), min(1, max(accuracies) + 0.03))
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100)
plt.show()
print("Saved: model_comparison.png")

# ── 10. Quick inference test ──────────────────────────────────────────────────
test_text   = "i love you"
cleaned     = clean_text(test_text)
z           = vectorizer.transform([cleaned])
prediction  = stacking_model.predict(z)
print(f"\nSample prediction for '{test_text}': {prediction[0]} "
      f"({'Hate Speech' if prediction[0] == 1 else 'Not Hate Speech'})")

# ── 11. Save model & vectorizer ───────────────────────────────────────────────
print("\nSaving model and vectorizer...")
joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(vectorizer,     "tfidf_vectorizer.pkl")
print("✅ Saved: stacking_model.pkl")
print("✅ Saved: tfidf_vectorizer.pkl")
print("\nTraining complete! Run:  streamlit run app.py")
