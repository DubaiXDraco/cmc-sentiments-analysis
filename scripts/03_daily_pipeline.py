#!/usr/bin/env python3
"""
scripts/03_daily_pipeline.py
Tuần 3: Pipeline cập nhật cảm xúc hàng ngày (hỗ trợ 3 chế độ):
- `--mode replay --date YYYY-MM-DD`: phát lại dữ liệu thật của ngày cụ thể từ `cmc_student_clean.csv`
- `--mode json --input <path.json>`: nạp file JSON mới (lọc PR Fanpage, ẩn danh PII)
- `--mode playwright --url <url>`: thu thập từ trang công khai qua Playwright và ẩn danh PII

Bảo đảm idempotent nhờ `processed_ids.json`, xử lý ngoại lệ nhẹ nhàng và ghi log chi tiết.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anonymize import anonymize_id, anonymize_text
from src.inference import SentimentEngine
from src.preprocess import clean_text

CLEAN_CSV = ROOT / "data" / "cmc_student_clean.csv"
DAILY_DIR = ROOT / "daily_results"
OFFICIAL_FANPAGE_SOURCES = {"TruongDaiHocCMC", "ctsv.cmcuni"}


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    with open(log_path, mode="a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_processed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception:
        pass
    return set()


def _save_processed_ids(path: Path, ids: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_replay_records(target_date: str, clean_csv: Path) -> pd.DataFrame:
    if not clean_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu nguồn: {clean_csv}")
    df = pd.read_csv(clean_csv, encoding="utf-8-sig")
    sub = df[df["date"].astype(str) == str(target_date)].copy()
    return sub


def _load_json_records(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {json_path}")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("File JSON đầu vào phải là danh sách bản ghi.")

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("groupTitle") or item.get("pageName") or item.get("source") or "External")
        rec_type = str(item.get("type") or "post")
        if rec_type == "post" and source in OFFICIAL_FANPAGE_SOURCES:
            continue
        raw_text = str(item.get("text") or item.get("message") or item.get("content") or "").strip()
        if not raw_text:
            continue
        anon = anonymize_text(raw_text)
        cleaned = clean_text(anon)
        if len(cleaned) < 5:
            continue
        raw_id = str(item.get("id") or item.get("postId") or f"json_{json_path.stem}_{idx}")
        rows.append(
            {
                "id": anonymize_id(raw_id),
                "type": rec_type,
                "source": source,
                "timestamp": str(item.get("time") or item.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "date": str(item.get("date") or datetime.now().strftime("%Y-%m-%d")),
                "likes": int(item.get("likes", 0) or 0),
                "text": anon,
            }
        )
    return pd.DataFrame(rows)


def run_daily_pipeline(
    mode: str = "replay",
    target_date: Optional[str] = None,
    input_path: Optional[Path] = None,
    url: Optional[str] = None,
    backend: str = "tfidf_lr",
    out_dir: Path = DAILY_DIR,
    clean_csv: Path = CLEAN_CSV,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    processed_path = out_dir / "processed_ids.json"

    date_label = target_date or datetime.now().strftime("%Y-%m-%d")
    _append_log(log_path, f"START mode={mode} date={date_label} backend={backend}")

    try:
        if mode == "replay":
            df_in = _load_replay_records(date_label, clean_csv)
        elif mode == "json":
            if input_path is None:
                raise ValueError("Chế độ json yêu cầu tham số --input")
            df_in = _load_json_records(Path(input_path))
        elif mode == "playwright":
            if not url:
                raise ValueError("Chế độ playwright yêu cầu tham số --url")
            import asyncio
            import importlib
            crawler_mod = importlib.import_module("scripts.03c_crawler_playwright")
            crawled = asyncio.run(crawler_mod.crawl_public_page(url))
            df_in = pd.DataFrame(crawled)
        else:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")
    except Exception as exc:
        _append_log(log_path, f"ERROR loading records: {type(exc).__name__}: {exc}")
        return {"status": "error", "error": str(exc), "new_rows": 0, "skipped_duplicates": 0}

    if df_in.empty:
        _append_log(log_path, f"DONE date={date_label}: không có dữ liệu mới (0 dòng).")
        return {"status": "empty", "date": date_label, "new_rows": 0, "skipped_duplicates": 0}

    processed_ids = _load_processed_ids(processed_path)
    df_in["id"] = df_in["id"].astype(str)
    mask_new = ~df_in["id"].isin(processed_ids)
    skipped = int((~mask_new).sum())
    df_new = df_in[mask_new].copy().reset_index(drop=True)

    if df_new.empty:
        _append_log(
            log_path,
            f"DONE date={date_label}: 0 bản ghi mới (bỏ qua {skipped} ID đã xử lý - idempotent).",
        )
        return {
            "status": "no_new_records",
            "date": date_label,
            "new_rows": 0,
            "skipped_duplicates": skipped,
        }

    engine = SentimentEngine(backend=backend)
    preds = engine.predict_batch(df_new["text"].fillna("").tolist(), batch_size=32)

    df_new["sentiment"] = [r.label for r in preds]
    df_new["sentiment_vi"] = [r.label_vi for r in preds]
    df_new["confidence"] = [r.confidence for r in preds]
    df_new["topic"] = [r.topic for r in preds]
    df_new["model_backend"] = [r.backend for r in preds]

    out_csv = out_dir / f"sentiment_{date_label}.csv"
    keep_cols = [
        "id", "type", "source", "timestamp", "date", "likes",
        "topic", "text", "sentiment", "sentiment_vi", "confidence", "model_backend",
    ]
    df_save = df_new[[c for c in keep_cols if c in df_new.columns]].copy()

    if out_csv.exists():
        existing = pd.read_csv(out_csv, encoding="utf-8-sig")
        combined = pd.concat([existing, df_save], ignore_index=True).drop_duplicates(subset=["id"])
        combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    else:
        df_save.to_csv(out_csv, index=False, encoding="utf-8-sig")

    processed_ids.update(df_save["id"].astype(str).tolist())
    _save_processed_ids(processed_path, processed_ids)

    dist = df_save["sentiment"].value_counts().to_dict()
    _append_log(
        log_path,
        f"SUCCESS date={date_label}: +{len(df_save)} bản ghi mới (bỏ qua {skipped} trùng) | "
        f"Phân bố={dist} | Lưu={out_csv.name}",
    )
    return {
        "status": "success",
        "date": date_label,
        "new_rows": int(len(df_save)),
        "skipped_duplicates": skipped,
        "distribution": dist,
        "output_file": str(out_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline cập nhật cảm xúc sinh viên CMC hàng ngày.")
    parser.add_argument("--mode", choices=["replay", "json", "playwright"], default="replay")
    parser.add_argument("--date", type=str, default=None, help="Ngày YYYY-MM-DD cho chế độ replay.")
    parser.add_argument("--input", type=str, default=None, help="File JSON đầu vào cho chế độ json.")
    parser.add_argument("--url", type=str, default=None, help="URL công khai cho chế độ playwright.")
    parser.add_argument("--backend", choices=["tfidf_lr", "tfidf_svm", "phobert"], default="tfidf_lr")
    args = parser.parse_args()

    res = run_daily_pipeline(
        mode=args.mode,
        target_date=args.date,
        input_path=Path(args.input) if args.input else None,
        url=args.url,
        backend=args.backend,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
