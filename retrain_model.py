import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

print("Loading train.csv...")
df = pd.read_csv("train.csv")

# Additional real-world threats, violent harassment, hate speech, and safe colloquial examples
additional_data = [
    # Violent threats & direct hate speech (1)
    ("i kill you", 1),
    ("i will kill you", 1),
    ("i will murder you", 1),
    ("go kill yourself", 1),
    ("i will hunt you down and kill you", 1),
    ("die you disgusting trash", 1),
    ("you deserve to die", 1),
    ("i hope you get killed", 1),
    ("slit your throat and die", 1),
    ("burn in hell you bitch", 1),
    ("die bitch die", 1),
    ("shoot all of them dead", 1),
    ("kill all these immigrants and foreigners", 1),
    ("exterminate all jews and muslims", 1),
    ("all women are stupid bitches who should be slapped", 1),
    ("white trash should be wiped off the earth", 1),
    ("hang that black boy from a tree", 1),
    ("you should commit suicide right now", 1),
    ("get out of our country you filthy pigs", 1),
    ("i am going to shoot up your place", 1),
    ("punch you in the face until you bleed", 1),

    # Everyday safe expressions (including safe uses of 'kill', 'hate', etc.) (0)
    ("my feet are killing me after running the marathon", 0),
    ("this comedian killed it on stage last night, so funny!", 0),
    ("i am killing time waiting for my dentist appointment", 0),
    ("you are killing the competition with your talent!", 0),
    ("that final exam was so hard, it killed my brain haha", 0),
    ("i hate waking up early on monday mornings", 0),
    ("i hate traffic jams during rush hour", 0),
    ("i hate broccoli and green peas", 0),
    ("i love spending time with my family on weekends", 0),
    ("congratulations on your graduation, so proud of you!", 0),
    ("having a warm cup of hot chocolate on a rainy day", 0),
    ("reading a good novel in the library", 0),
    ("hope everyone has an amazing and productive day ahead", 0),
    ("learning machine learning with python is very interesting", 0),
    ("the sunset at the beach looked breathtaking today", 0),
]

add_df = pd.DataFrame(additional_data, columns=["tweet", "label"])
df = pd.concat([df[["tweet", "label"]], add_df], ignore_index=True)

# Smart Stopword Filter: Keep pronouns & directional words so "i kill you" doesn't reduce to just "kill"
custom_stops = set(ENGLISH_STOP_WORDS) - {
    "i", "you", "he", "she", "they", "we", "me", "him", "her", "them", 
    "all", "not", "no", "against", "down", "out", "off", "up"
}

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in custom_stops)

print("Preprocessing tweets...")
df["clean_text"] = df["tweet"].apply(clean_text)

print("Vectorizing with TF-IDF (1, 2 n-grams)...")
vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"].values

print("Training Stacking Classifier...")
base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=2.5, random_state=42)
base_rf = RandomForestClassifier(n_estimators=50, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=42)
base_xgb = XGBClassifier(scale_pos_weight=10, n_estimators=60, max_depth=5, n_jobs=-1, random_state=42, eval_metric="logloss")

stacking_model = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("xgb", base_xgb),
    ],
    final_estimator=LogisticRegression(class_weight="balanced"),
    cv=3,
    n_jobs=-1
)
stacking_model.fit(X, y)

print("Saving stacking_model.pkl and tfidf_vectorizer.pkl...")
joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Evaluation on test cases
test_cases = [
    ("i kill you", 1),
    ("i will kill you", 1),
    ("go kill yourself", 1),
    ("i will murder you", 1),
    ("women are stupid bitches who belong in the kitchen", 1),
    ("filthy immigrants should be deported", 1),
    ("i love you so much sweetheart", 0),
    ("my feet are killing me after running", 0),
    ("learning python is fun and exciting", 0),
    ("happy birthday to my best friend!", 0),
    ("having a nice cup of tea with friends", 0)
]

print("\n--- MODEL PREDICTION TEST ---")
for text, expected in test_cases:
    cleaned = clean_text(text)
    pred = stacking_model.predict(vectorizer.transform([cleaned]))[0]
    proba = stacking_model.predict_proba(vectorizer.transform([cleaned]))[0]
    label = "🚨 HATE / THREAT" if pred == 1 else "✅ NOT HATE SPEECH"
    status = "PASS" if pred == expected else "FAIL"
    print(f"[{status}] {label} (Hate confidence: {proba[1]*100:.1f}%) -> '{text}'")

print("\nAll done!")
