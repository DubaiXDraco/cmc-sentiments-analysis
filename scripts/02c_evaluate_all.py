#!/usr/bin/env python3
"""
scripts/02c_evaluate_all.py
Tuần 2: Đánh giá đồng bộ 3 mô hình (TF-IDF+LR, TF-IDF+LinearSVC, Fine-tuned PhoBERT)
trên các tập Val, Semi-Auto Test, Gold Real Test (120 mẫu thật kiểm định thủ công)
và Challenge Test Set (đã loại câu trùng với Train/Val).
Đo độ trễ suy luận thực tế (ms/câu), dung lượng đĩa (MB) và xuất báo cáo/biểu đồ.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import LABELS, SentimentEngine
from src.preprocess import make_loose_key

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def measure_dir_or_file_mb(path: Path) -> float:
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)
    return 0.0


def measure_latency_ms(engine: SentimentEngine, sample_texts: List[str], n_runs: int = 50) -> float:
    for t in sample_texts[:5]:
        engine.predict(t)
    t0 = time.perf_counter()
    for i in range(n_runs):
        engine.predict(sample_texts[i % len(sample_texts)])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return round(elapsed_ms / n_runs, 2)


def eval_split(engine: SentimentEngine, df: pd.DataFrame, text_col: str, label_col: str) -> Dict:
    results = engine.predict_batch(df[text_col].fillna("").tolist(), batch_size=32)
    y_pred = [r.label for r in results]
    y_true = df[label_col].astype(str).tolist()
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    f1 = round(float(f1_score(y_true, y_pred, average="macro")), 4)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    report = classification_report(y_true, y_pred, labels=LABELS, digits=4)
    return {
        "n_samples": int(len(df)),
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "preds": y_pred,
    }


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    gold_df = pd.read_csv(DATA_DIR / "gold_real_test.csv", encoding="utf-8-sig")
    chal_raw = pd.read_csv(DATA_DIR / "challenge_test_set.csv", encoding="utf-8-sig")

    # Loại bỏ câu thử thách trùng/gần trùng với Train/Val để bảo đảm độc lập
    seen_keys = set(
        pd.concat([train_df["text"], val_df["text"]], ignore_index=True).apply(make_loose_key)
    )
    in_train_val = chal_raw["text"].apply(make_loose_key).isin(seen_keys)
    chal_df = chal_raw[~in_train_val].reset_index(drop=True)

    # Bảo đảm Gold Real Test không giao với Train/Val
    tv_ids = set(train_df["id"].astype(str)) | set(val_df["id"].astype(str))
    assert not (set(gold_df["id"].astype(str)) & tv_ids), "Gold Real Test bị rò rỉ với Train/Val!"

    model_specs = [
        ("tfidf_lr", "TF-IDF + Logistic Regression", MODELS_DIR / "model_tfidf_lr.joblib"),
        ("tfidf_svm", "TF-IDF (Word+Char) + LinearSVC", MODELS_DIR / "model_tfidf_svm.joblib"),
        ("phobert", "Fine-tuned PhoBERT (v2)", MODELS_DIR / "cmc_phobert_sentiment"),
    ]

    sample_texts = gold_df["text"].fillna("").tolist()[:20]
    comparison_rows = []
    full_eval = {}

    for backend_key, display_name, artifact_path in model_specs:
        engine = SentimentEngine(backend=backend_key, models_dir=MODELS_DIR)
        if engine.backend != backend_key:
            raise RuntimeError(f"Model `{backend_key}` chưa sẵn sàng (chuyển về {engine.backend}).")

        val_res = eval_split(engine, val_df, "text", "label")
        test_res = eval_split(engine, test_df, "text", "label")
        gold_res = eval_split(engine, gold_df, "text", "gold_label")
        chal_res = eval_split(engine, chal_df, "text", "true_label")

        latency_ms = measure_latency_ms(engine, sample_texts, n_runs=50)
        size_mb = measure_dir_or_file_mb(artifact_path)

        full_eval[backend_key] = {
            "display_name": display_name,
            "size_mb": size_mb,
            "latency_ms_per_text": latency_ms,
            "val": {k: v for k, v in val_res.items() if k not in ("report", "preds")},
            "test": {k: v for k, v in test_res.items() if k not in ("report", "preds")},
            "gold_real_test": {k: v for k, v in gold_res.items() if k not in ("report", "preds")},
            "challenge_test": {k: v for k, v in chal_res.items() if k not in ("report", "preds")},
        }

        comparison_rows.append(
            {
                "model_key": backend_key,
                "model_name": display_name,
                "val_acc": val_res["accuracy"],
                "val_macro_f1": val_res["macro_f1"],
                "test_acc": test_res["accuracy"],
                "test_macro_f1": test_res["macro_f1"],
                "gold_acc": gold_res["accuracy"],
                "gold_macro_f1": gold_res["macro_f1"],
                "challenge_acc": chal_res["accuracy"],
                "challenge_macro_f1": chal_res["macro_f1"],
                "latency_ms": latency_ms,
                "size_mb": size_mb,
            }
        )

        if backend_key == "tfidf_lr":
            (REPORT_DIR / "classification_report_baseline.txt").write_text(
                f"=== GOLD REAL TEST ({len(gold_df)} mẫu) - {display_name} ===\n"
                f"Accuracy: {gold_res['accuracy']:.4f} | Macro-F1: {gold_res['macro_f1']:.4f}\n\n"
                f"{gold_res['report']}\n"
                f"=== SEMI-AUTO TEST ({len(test_df)} mẫu) ===\n"
                f"Accuracy: {test_res['accuracy']:.4f} | Macro-F1: {test_res['macro_f1']:.4f}\n\n"
                f"{test_res['report']}",
                encoding="utf-8",
            )
            lr_gold_cm = np.array(gold_res["confusion_matrix"])
            lr_chal = chal_res
        elif backend_key == "phobert":
            (REPORT_DIR / "classification_report_phobert.txt").write_text(
                f"=== GOLD REAL TEST ({len(gold_df)} mẫu) - {display_name} ===\n"
                f"Accuracy: {gold_res['accuracy']:.4f} | Macro-F1: {gold_res['macro_f1']:.4f}\n\n"
                f"{gold_res['report']}\n"
                f"=== SEMI-AUTO TEST ({len(test_df)} mẫu) ===\n"
                f"Accuracy: {test_res['accuracy']:.4f} | Macro-F1: {test_res['macro_f1']:.4f}\n\n"
                f"{test_res['report']}",
                encoding="utf-8",
            )
            phobert_gold_cm = np.array(gold_res["confusion_matrix"])
            phobert_chal = chal_res

    # Lưu challenge_test_summary.txt
    chal_summary = [
        f"Challenge test set: {len(chal_df)} câu độc lập (từ {len(chal_raw)} câu viết tay, loại {int(in_train_val.sum())} câu trùng Train/Val)",
        f"TF-IDF + LR   -> Accuracy: {lr_chal['accuracy']:.4f} | Macro-F1: {lr_chal['macro_f1']:.4f}",
        f"PhoBERT (v2)  -> Accuracy: {phobert_chal['accuracy']:.4f} | Macro-F1: {phobert_chal['macro_f1']:.4f}",
        "",
        "Chi tiết phân tích trên PhoBERT:",
        phobert_chal["report"],
    ]
    chal_df_copy = chal_df.copy()
    chal_df_copy["lr_pred"] = lr_chal["preds"]
    chal_df_copy["phobert_pred"] = phobert_chal["preds"]
    wrong_pb = chal_df_copy[chal_df_copy["true_label"] != chal_df_copy["phobert_pred"]]
    chal_summary.append(f"Số câu PhoBERT sai ({len(wrong_pb)}/{len(chal_df)}):")
    chal_summary.append(wrong_pb[["text", "true_label", "lr_pred", "phobert_pred"]].to_string())
    (REPORT_DIR / "challenge_test_summary.txt").write_text("\n".join(chal_summary), encoding="utf-8")

    # Lưu JSON & CSV tổng hợp
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_stats = json.loads((REPORT_DIR / "split_stats.json").read_text(encoding="utf-8"))
    eval_summary = {
        "n_train": split_stats["n_train"],
        "n_val": split_stats["n_val"],
        "n_test": split_stats["n_test"],
        "n_gold_real_test": int(len(gold_df)),
        "n_challenge_test": int(len(chal_df)),
        "n_unique_core_text": split_stats["n_unique_core_text"],
        "leakage_check": split_stats["leakage_check"],
        "models": full_eval,
    }
    (REPORT_DIR / "evaluation_results.json").write_text(
        json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Vẽ Confusion Matrix Comparison (Gold Real Test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(lr_gold_cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
    axes[0].set_title(f"TF-IDF + LR (Gold Acc={full_eval['tfidf_lr']['gold_real_test']['accuracy']:.3f})")
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Nhãn thật")

    sns.heatmap(phobert_gold_cm, annot=True, fmt="d", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title(f"Fine-tuned PhoBERT (Gold Acc={full_eval['phobert']['gold_real_test']['accuracy']:.3f})")
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Nhãn thật")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix_comparison.png", dpi=150)
    plt.close(fig)

    # Vẽ Model Benchmark Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(comp_df))
    w = 0.25
    axes[0].bar(x - w, comp_df["test_macro_f1"], width=w, label="Semi-Auto Test F1", color="#3B82F6")
    axes[0].bar(x, comp_df["gold_macro_f1"], width=w, label="Gold Real Test F1", color="#10B981")
    axes[0].bar(x + w, comp_df["challenge_macro_f1"], width=w, label="Challenge Test F1", color="#F59E0B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["TF-IDF + LR", "TF-IDF + SVM", "PhoBERT (v2)"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("So sánh Macro-F1 trên 3 tập kiểm thử độc lập")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(["TF-IDF + LR", "TF-IDF + SVM", "PhoBERT (v2)"], comp_df["latency_ms"], color=["#6366F1", "#8B5CF6", "#EC4899"])
    axes[1].set_ylabel("Độ trễ CPU (ms / câu)")
    axes[1].set_title("Tốc độ suy luận CPU (thấp hơn = nhanh hơn)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for idx_b, val in enumerate(comp_df["latency_ms"]):
        axes[1].text(idx_b, val, f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_benchmark.png", dpi=150)
    plt.close(fig)

    print("=" * 72)
    print("BẢNG SO SÁNH HIỆU NĂNG & ĐỘ TRỄ TẤT CẢ MÔ HÌNH")
    print("=" * 72)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
