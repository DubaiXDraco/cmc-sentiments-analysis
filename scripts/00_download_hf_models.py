#!/usr/bin/env python3
"""
scripts/00_download_hf_models.py
Tải trước 2 mô hình HuggingFace vào thư mục cache cục bộ `.hf_cache/`:
1. `wonrax/phobert-base-vietnamese-sentiment` (gán nhãn bán tự động Tuần 2)
2. `vinai/phobert-base-v2` (backbone để fine-tune Tuần 2)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = ROOT / ".hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    ("wonrax/phobert-base-vietnamese-sentiment", None),
    ("vinai/phobert-base-v2", 3),
]


def main() -> None:
    for model_id, num_labels in MODELS:
        print(f"Đang tải tokenizer & model: {model_id} -> {HF_CACHE}")
        AutoTokenizer.from_pretrained(model_id)
        if num_labels is not None:
            AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
        else:
            AutoModelForSequenceClassification.from_pretrained(model_id)
        print(f"✓ Hoàn tất tải: {model_id}")


if __name__ == "__main__":
    main()
