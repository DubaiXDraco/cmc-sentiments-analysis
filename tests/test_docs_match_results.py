"""Kiểm tra tính nhất quán tuyệt đối giữa tài liệu (README, Báo cáo, Data Card) và các file kết quả JSON."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

INGEST = json.loads((REPORTS / "ingestion_stats.json").read_text(encoding="utf-8"))
SPLITS = json.loads((REPORTS / "split_stats.json").read_text(encoding="utf-8"))
AUDIT = json.loads((REPORTS / "labeling_audit.json").read_text(encoding="utf-8"))
REP = json.loads((REPORTS / "repeated_splits.json").read_text(encoding="utf-8"))
EVAL = json.loads((REPORTS / "evaluation_results.json").read_text(encoding="utf-8"))

DOCS = {
    "README": (ROOT / "README.md").read_text(encoding="utf-8"),
    "report": (REPORTS / "Bao_cao_toan_bo_du_an.md").read_text(encoding="utf-8"),
    "data_card": (ROOT / "data" / "DATA_CARD.md").read_text(encoding="utf-8"),
}


def test_dataset_counts_match_docs():
    split_str = f"{SPLITS['n_train']} / {SPLITS['n_val']} / {SPLITS['n_test']}"
    for name, txt in DOCS.items():
        assert str(INGEST["after_exact_text_clean_dedup"]) in txt, name
        assert str(INGEST["excluded_official_fanpage_posts"]) in txt, name
        assert str(SPLITS["n_unique_core_text"]) in txt, name
        assert split_str in txt, name


def test_labeling_audit_matches_docs():
    agree_str = f"{AUDIT['semi_vs_gold_agreement']:.4f}"
    kappa_str = f"{AUDIT['semi_vs_gold_cohen_kappa']:.4f}"
    for name, txt in DOCS.items():
        assert str(AUDIT["n_gold_samples"]) in txt, name
        assert agree_str in txt, name
        assert kappa_str in txt, name


def test_model_metrics_match_docs():
    for model_key in ("tfidf_lr", "tfidf_svm", "phobert"):
        m = EVAL["models"][model_key]
        for split_key in ("val", "test", "gold_real_test", "challenge_test"):
            acc = f"{m[split_key]['accuracy']:.4f}"
            f1 = f"{m[split_key]['macro_f1']:.4f}"
            for name in ("README", "report"):
                assert acc in DOCS[name], (model_key, split_key, "acc", acc, name)
                assert f1 in DOCS[name], (model_key, split_key, "f1", f1, name)


def test_repeated_splits_and_leakage_match_docs():
    acc_mean = f"{REP['accuracy_mean']:.4f}"
    acc_std = f"{REP['accuracy_std']:.4f}"
    f1_mean = f"{REP['macro_f1_mean']:.4f}"
    f1_std = f"{REP['macro_f1_std']:.4f}"
    for name in ("README", "report"):
        assert acc_mean in DOCS[name] and acc_std in DOCS[name], name
        assert f1_mean in DOCS[name] and f1_std in DOCS[name], name
    assert EVAL["leakage_check"] == {"train_val_overlap": 0, "train_test_overlap": 0, "val_test_overlap": 0}
