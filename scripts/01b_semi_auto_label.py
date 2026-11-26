#!/usr/bin/env python3
"""
scripts/01b_semi_auto_label.py
Tuần 2: Gán nhãn bán tự động (Semi-Automatic Labeling) bằng pretrained
`wonrax/phobert-base-vietnamese-sentiment` + hậu xử lý tối thiểu cho bài đăng tiện ích
(tìm đồ, tìm trọ/ở ghép không mang phàn nàn), loại trùng `core_text`, chia tập không rò rỉ
Train/Val/Test (70/15/15) bằng GroupShuffleSplit, và trích batch gán nhãn nhóm 200 mẫu.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import segment_for_phobert

HF_CACHE_DIR = ROOT / ".hf_cache"
CLEAN_CSV = ROOT / "data" / "cmc_student_clean.csv"
LABELED_CSV = ROOT / "data" / "cmc_student_labeled.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
VAL_CSV = ROOT / "data" / "val.csv"
TEST_CSV = ROOT / "data" / "test.csv"
TEAM_BATCH_CSV = ROOT / "data" / "annotation_batch_for_team.csv"
SPLIT_STATS_JSON = ROOT / "reports" / "split_stats.json"

SILVER_MODEL_ID = "wonrax/phobert-base-vietnamese-sentiment"

LABEL_VI = {"Negative": "Tiêu cực", "Neutral": "Bình thường", "Positive": "Tích cực"}

LOST_FOUND_RE = re.compile(
    r"(?:để quên|nhặt được|đánh rơi|tìm lại thẻ|xin lại|rơi thẻ|rơi chìa|thất lạc)",
    flags=re.IGNORECASE,
)
RENTAL_AD_RE = re.compile(
    r"(?:ở ghép|tìm trọ|phòng trọ|vệ sinh khép kín|pass phòng|nhượng phòng)",
    flags=re.IGNORECASE,
)
COMPLAINT_GUARD_RE = re.compile(
    r"(?:tệ|kém|chán|bực|lừa đảo|thất vọng|đắt|nóng|ồn|bẩn|hỏng|không hài lòng|mất cắp|trộm|ức chế|xàm|vô duyên)",
    flags=re.IGNORECASE,
)


def _normalize_hf_label(raw_lbl: str) -> str:
    u = str(raw_lbl).strip().upper()
    if u in ("NEG", "NEGATIVE", "LABEL_0", "TIÊU CỰC"):
        return "Negative"
    if u in ("POS", "POSITIVE", "LABEL_2", "TÍCH CỰC"):
        return "Positive"
    if u in ("NEU", "NEUTRAL", "LABEL_1", "BÌNH THƯỜNG"):
        return "Neutral"
    raise ValueError(f"Không nhận diện được nhãn từ pretrained model: {raw_lbl}")


def run_silver_labeling(
    texts: List[str],
    model_id: str = SILVER_MODEL_ID,
    batch_size: int = 32,
    max_length: int = 128,
) -> pd.DataFrame:
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)
    model.eval()

    raw_id2label = getattr(model.config, "id2label", {0: "NEG", 1: "POS", 2: "NEU"})
    idx_to_std: Dict[int, str] = {
        int(k): _normalize_hf_label(v) for k, v in raw_id2label.items()
    }

    seg_texts = [segment_for_phobert(t) for t in texts]
    all_rows = []

    for start in range(0, len(seg_texts), batch_size):
        batch = seg_texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row_prob in probs:
            p_map = {idx_to_std[i]: float(row_prob[i]) for i in range(len(row_prob))}
            best = max(("Negative", "Neutral", "Positive"), key=lambda k: p_map[k])
            all_rows.append(
                {
                    "silver_label_raw": best,
                    "silver_confidence": round(p_map[best], 4),
                    "silver_prob_neg": round(p_map["Negative"], 4),
                    "silver_prob_neu": round(p_map["Neutral"], 4),
                    "silver_prob_pos": round(p_map["Positive"], 4),
                }
            )
    return pd.DataFrame(all_rows)


def apply_domain_utility_rules(df: pd.DataFrame) -> pd.DataFrame:
    labels: List[str] = []
    sources: List[str] = []
    for _, row in df.iterrows():
        text_lower = str(row.get("text_clean") or row.get("text") or "").lower()
        raw_lbl = str(row["silver_label_raw"])
        is_utility = (
            LOST_FOUND_RE.search(text_lower) is not None
            or RENTAL_AD_RE.search(text_lower) is not None
        )
        has_complaint = COMPLAINT_GUARD_RE.search(text_lower) is not None
        if is_utility and not has_complaint and raw_lbl != "Neutral":
            labels.append("Neutral")
            sources.append("semi_auto_domain_rule")
        else:
            labels.append(raw_lbl)
            sources.append("semi_auto_phobert")

    out = df.copy()
    out["label"] = labels
    out["label_vi"] = out["label"].map(LABEL_VI)
    out["label_source"] = sources
    return out


def split_leakage_free(df: pd.DataFrame, seed: int = 42):
    df_dedup = df.drop_duplicates(subset=["core_text"], keep="first").reset_index(drop=True)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, tmp_idx = next(gss1.split(df_dedup, df_dedup["label"], groups=df_dedup["core_text"]))
    train_df = df_dedup.iloc[tr_idx].reset_index(drop=True)
    tmp_df = df_dedup.iloc[tmp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, te_idx = next(gss2.split(tmp_df, tmp_df["label"], groups=tmp_df["core_text"]))
    val_df = tmp_df.iloc[val_idx].reset_index(drop=True)
    test_df = tmp_df.iloc[te_idx].reset_index(drop=True)

    tr_c = set(train_df["core_text"])
    va_c = set(val_df["core_text"])
    te_c = set(test_df["core_text"])
    assert len(tr_c & va_c) == 0, "Rò rỉ core_text Train-Val!"
    assert len(tr_c & te_c) == 0, "Rò rỉ core_text Train-Test!"
    assert len(va_c & te_c) == 0, "Rò rỉ core_text Val-Test!"

    return df_dedup, train_df, val_df, test_df


def build_team_annotation_batch(df_dedup: pd.DataFrame, n_total: int = 200, seed: int = 42) -> pd.DataFrame:
    per_class = n_total // 3
    sampled_parts = []
    for lbl in ("Negative", "Neutral", "Positive"):
        sub = df_dedup[df_dedup["label"] == lbl]
        k = min(len(sub), per_class + (1 if lbl == "Neutral" and n_total % 3 else 0))
        sampled_parts.append(sub.sample(n=k, random_state=seed))
    batch = pd.concat(sampled_parts, ignore_index=True)
    if len(batch) < n_total:
        rem = df_dedup[~df_dedup["id"].isin(set(batch["id"]))]
        extra = rem.sample(n=min(len(rem), n_total - len(batch)), random_state=seed)
        batch = pd.concat([batch, extra], ignore_index=True)
    batch = batch.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pd.DataFrame(
        {
            "id": batch["id"],
            "type": batch["type"],
            "source": batch["source"],
            "topic": batch["topic"],
            "text": batch["text"],
            "suggested_label": batch["label"],
            "annotator_1": "",
            "annotator_2": "",
            "notes": "",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-automatic PhoBERT labeling & leakage-free split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    print(f"Đọc dữ liệu sạch: {len(df)} mẫu từ {CLEAN_CSV.name}")

    silver_df = run_silver_labeling(
        df["text_clean"].fillna("").tolist(),
        model_id=SILVER_MODEL_ID,
        batch_size=args.batch_size,
    )
    df_merged = pd.concat([df.reset_index(drop=True), silver_df], axis=1)
    df_labeled = apply_domain_utility_rules(df_merged)
    df_labeled.to_csv(LABELED_CSV, index=False, encoding="utf-8-sig")

    df_dedup, train_df, val_df, test_df = split_leakage_free(df_labeled, seed=args.seed)
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    team_batch = build_team_annotation_batch(df_dedup, n_total=200, seed=args.seed)
    team_batch.to_csv(TEAM_BATCH_CSV, index=False, encoding="utf-8-sig")

    split_stats = {
        "silver_model": SILVER_MODEL_ID,
        "seed": args.seed,
        "n_labeled_total": int(len(df_labeled)),
        "n_unique_core_text": int(len(df_dedup)),
        "n_removed_core_duplicates": int(len(df_labeled) - len(df_dedup)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "label_source_counts": df_labeled["label_source"].value_counts().to_dict(),
        "label_dist_all_clean": df_labeled["label"].value_counts().to_dict(),
        "label_dist_dedup": df_dedup["label"].value_counts().to_dict(),
        "label_dist_train": train_df["label"].value_counts().to_dict(),
        "label_dist_val": val_df["label"].value_counts().to_dict(),
        "label_dist_test": test_df["label"].value_counts().to_dict(),
        "leakage_check": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
        },
        "team_annotation_batch_size": int(len(team_batch)),
    }
    SPLIT_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_STATS_JSON.write_text(json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print("HOÀN TẤT GÁN NHÃN BÁN TỰ ĐỘNG & CHIA TẬP KHÔNG RÒ RỈ")
    print("=" * 64)
    print(f"- Tổng mẫu sạch có nhãn : {len(df_labeled)}")
    print(f"- Sau loại trùng core   : {len(df_dedup)}")
    print(f"- Train / Val / Test    : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"- Phân bố nhãn (dedup)  : {split_stats['label_dist_dedup']}")
    print(f"- Nguồn nhãn            : {split_stats['label_source_counts']}")
    print(f"- Batch gán nhãn nhóm   : {TEAM_BATCH_CSV.name} ({len(team_batch)} mẫu)")


if __name__ == "__main__":
    main()
