import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def train():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model_pipeline.fit(train_df["clean_text"], train_df["label"])
    preds = model_pipeline.predict(test_df["clean_text"])

    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="macro")
    print(f"[OK] Accuracy: {acc:.4f} | Macro F1-Score: {f1:.4f}")
    print(classification_report(test_df["label"], preds, target_names=["Negative", "Neutral", "Positive"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/sentiment_baseline.pkl")
    print("[OK] Đã lưu mô hình tại: models/sentiment_baseline.pkl")

if __name__ == "__main__":
    train()
