#!/usr/bin/env python3
"""
scripts/02_train_baselines.py
Tuần 2: Huấn luyện 2 mô hình Baseline trên tập Train/Val/Test đã chia không rò rỉ:
1. TF-IDF (Word 1-2 gram) + Logistic Regression (`class_weight='balanced'`)
2. FeatureUnion (Word 1-2 gram + Char_wb 3-5 gram) + Calibrated LinearSVC (`class_weight='balanced'`)
Đồng thời chạy đánh giá lặp 30 lần chia GroupShuffleSplit (`reports/repeated_splits.json`).
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def build_lr_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                    C=1.0,
                ),
            ),
        ]
    )


def build_svm_pipeline(seed: int = 42) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=8000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", random_state=seed, C=1.0),
                    cv=3,
                ),
            ),
        ]
    )


def assert_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    tr = set(train_df["core_text"].astype(str))
    va = set(val_df["core_text"].astype(str))
    te = set(test_df["core_text"].astype(str))
    assert len(tr & va) == 0, "Phát hiện rò rỉ core_text giữa Train và Val!"
    assert len(tr & te) == 0, "Phát hiện rò rỉ core_text giữa Train và Test!"
    assert len(va & te) == 0, "Phát hiện rò rỉ core_text giữa Val và Test!"


def run_repeated_splits(labeled_path: Path, n_splits: int = 30) -> dict:
    df = pd.read_csv(labeled_path, encoding="utf-8-sig")
    df = df.dropna(subset=["text_clean", "core_text", "label"]).copy()
    df = df.drop_duplicates(subset=["core_text"], keep="first").reset_index(drop=True)
    df["y"] = df["label"].map(LABEL_MAP)

    accs, f1s = [], []
    for seed in range(n_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, df["y"], groups=df["core_text"]))
        pipe = build_lr_pipeline(seed=42)
        pipe.fit(df["text_clean"].iloc[tr_idx], df["y"].iloc[tr_idx])
        preds = pipe.predict(df["text_clean"].iloc[te_idx])
        y_true = df["y"].iloc[te_idx]
        accs.append(accuracy_score(y_true, preds))
        f1s.append(f1_score(y_true, preds, average="macro"))

    return {
        "model": "TF-IDF + Logistic Regression",
        "n_splits": n_splits,
        "n_samples": int(len(df)),
        "test_size": 0.15,
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "accuracy_min": round(float(np.min(accs)), 4),
        "accuracy_max": round(float(np.max(accs)), 4),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện baseline TF-IDF + LR và LinearSVC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=30)
    args = parser.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    assert_no_leakage(train_df, val_df, test_df)

    y_train = train_df["label"].map(LABEL_MAP)
    y_val = val_df["label"].map(LABEL_MAP)

    # 1. Baseline 1: TF-IDF + Logistic Regression
    lr_pipe = build_lr_pipeline(seed=args.seed)
    lr_pipe.fit(train_df["text_clean"].fillna(""), y_train)
    lr_val_pred = lr_pipe.predict(val_df["text_clean"].fillna(""))
    lr_path = MODEL_DIR / "model_tfidf_lr.joblib"
    joblib.dump(lr_pipe, lr_path)

    # 2. Baseline 2: Word+Char TF-IDF + Calibrated LinearSVC
    svm_pipe = build_svm_pipeline(seed=args.seed)
    svm_pipe.fit(train_df["text_clean"].fillna(""), y_train)
    svm_val_pred = svm_pipe.predict(val_df["text_clean"].fillna(""))
    svm_path = MODEL_DIR / "model_tfidf_svm.joblib"
    joblib.dump(svm_pipe, svm_path)

    print("=" * 64)
    print("HOÀN TẤT HUẤN LUYỆN BASELINE (KHÔNG RÒ RỈ)")
    print("=" * 64)
    print(f"Train / Val / Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(
        f"- TF-IDF + LR   | Val Acc: {accuracy_score(y_val, lr_val_pred):.4f} "
        f"| Val Macro-F1: {f1_score(y_val, lr_val_pred, average='macro'):.4f} -> {lr_path.name}"
    )
    print(
        f"- TF-IDF + SVM  | Val Acc: {accuracy_score(y_val, svm_val_pred):.4f} "
        f"| Val Macro-F1: {f1_score(y_val, svm_val_pred, average='macro'):.4f} -> {svm_path.name}"
    )

    rep = run_repeated_splits(DATA_DIR / "cmc_student_labeled.csv", n_splits=args.n_splits)
    (REPORT_DIR / "repeated_splits.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"- 30 Repeated Splits (LR): Acc = {rep['accuracy_mean']:.4f} ± {rep['accuracy_std']:.4f}, "
        f"Macro-F1 = {rep['macro_f1_mean']:.4f} ± {rep['macro_f1_std']:.4f}"
    )


if __name__ == "__main__":
    main()
