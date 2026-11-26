import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anonymize import contains_pii

DATA_DIR = ROOT / "data"
OFFICIAL_FANPAGE_SOURCES = {"TruongDaiHocCMC", "ctsv.cmcuni"}


def test_no_raw_pii_json_in_project():
    raw_jsons = list(ROOT.rglob("dataset_*.json"))
    assert raw_jsons == [], f"Tìm thấy JSON thô chứa PII trong project: {raw_jsons}"


def test_no_official_pr_posts_in_clean_dataset():
    df = pd.read_csv(DATA_DIR / "cmc_student_clean.csv", encoding="utf-8-sig")
    pr_mask = (df["type"] == "post") & (df["source"].isin(OFFICIAL_FANPAGE_SOURCES))
    assert int(pr_mask.sum()) == 0, "Vẫn còn bài đăng PR của Fanpage chính thức trong tập sạch!"


def test_zero_residual_pii_in_all_data_csvs():
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in ("text", "text_clean", "core_text"):
            if col in df.columns:
                bad = df[col].fillna("").apply(contains_pii)
                assert int(bad.sum()) == 0, f"Phát hiện PII sót trong {csv_path.name} cột {col}"


def test_no_core_text_overlap_between_splits():
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")

    tr = set(train_df["core_text"].astype(str))
    va = set(val_df["core_text"].astype(str))
    te = set(test_df["core_text"].astype(str))

    assert not (tr & va), "Rò rỉ core_text giữa Train và Val"
    assert not (tr & te), "Rò rỉ core_text giữa Train và Test"
    assert not (va & te), "Rò rỉ core_text giữa Val và Test"

    for split_df in (train_df, val_df, test_df):
        assert len(split_df) > 0
        assert {"Negative", "Neutral", "Positive"}.issubset(set(split_df["label"]))


def test_gold_real_test_strict_isolation():
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    gold_df = pd.read_csv(DATA_DIR / "gold_real_test.csv", encoding="utf-8-sig")

    train_val_ids = set(train_df["id"].astype(str)) | set(val_df["id"].astype(str))
    test_ids = set(test_df["id"].astype(str))
    gold_ids = set(gold_df["id"].astype(str))

    assert gold_ids.issubset(test_ids), "Gold Real Test phải là tập con chặt của test.csv"
    assert not (gold_ids & train_val_ids), "Gold Real Test không được giao với Train/Val"
    assert len(gold_df) >= 100
