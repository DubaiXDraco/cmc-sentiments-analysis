#!/usr/bin/env python3
"""
scripts/01c_cohen_kappa.py
Tính độ đồng thuận Cohen's kappa giữa hai thành viên gán nhãn độc lập
trên file `data/annotation_batch_for_team.csv` (hoặc trên `data/gold_real_test.csv`
giữa nhãn bán tự động và nhãn kiểm định thủ công).
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEAM_BATCH = ROOT / "data" / "annotation_batch_for_team.csv"
DEFAULT_GOLD_TEST = ROOT / "data" / "gold_real_test.csv"


def evaluate_agreement(df: pd.DataFrame, col_a: str, col_b: str) -> None:
    sub = df.dropna(subset=[col_a, col_b]).copy()
    sub[col_a] = sub[col_a].astype(str).str.strip()
    sub[col_b] = sub[col_b].astype(str).str.strip()
    sub = sub[(sub[col_a] != "") & (sub[col_b] != "")]
    if sub.empty:
        print(f"Chưa có cặp nhãn hoàn chỉnh cho `{col_a}` và `{col_b}`.")
        return

    agree = float((sub[col_a] == sub[col_b]).mean())
    kappa = float(cohen_kappa_score(sub[col_a], sub[col_b]))
    labels = sorted(set(sub[col_a]) | set(sub[col_b]))
    cm = confusion_matrix(sub[col_a], sub[col_b], labels=labels)
    print(f"Số mẫu đối chiếu ({col_a} vs {col_b}): {len(sub)}")
    print(f"Tỷ lệ đồng thuận (Agreement): {agree:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print("Ma trận đối chiếu:")
    print(pd.DataFrame(cm, index=labels, columns=labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính Cohen's kappa gán nhãn.")
    parser.add_argument("--file", type=str, default=str(DEFAULT_GOLD_TEST))
    parser.add_argument("--col-a", type=str, default="label")
    parser.add_argument("--col-b", type=str, default="gold_label")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    evaluate_agreement(df, args.col_a, args.col_b)


if __name__ == "__main__":
    main()
