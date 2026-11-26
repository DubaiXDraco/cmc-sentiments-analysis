#!/usr/bin/env python3
"""
scripts/01_ingest_and_clean.py
Tuần 1: Thu thập/nạp dữ liệu JSON Facebook công khai từ cộng đồng CMC,
loại bỏ bài PR chính thức của Fanpage trường, ẩn danh hóa PII,
làm sạch văn bản, gán chủ đề (6 nhóm) và lưu thống kê phễu dữ liệu.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anonymize import anonymize_id, anonymize_text, contains_pii
from src.preprocess import clean_text, make_core_text
from src.topics import classify_topic

DEFAULT_RAW_DIR = Path(r"C:\Users\bao long\.gemini\antigravity\scratch\cmc-sentiment-analysis\data")
OUT_CSV = ROOT / "data" / "cmc_student_clean.csv"
STATS_JSON = ROOT / "reports" / "ingestion_stats.json"

OFFICIAL_FANPAGE_SOURCES = {"TruongDaiHocCMC", "ctsv.cmcuni"}
VN_TZ = timezone(timedelta(hours=7))


def parse_likes(val: Any) -> int:
    if val is None:
        return 0
    try:
        return max(0, int(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return 0


def parse_timestamp(val: Any) -> Tuple[str, str]:
    """Chuẩn hóa timestamp về giờ Việt Nam (YYYY-MM-DD HH:MM:SS) và date (YYYY-MM-DD)."""
    if val is None or str(val).strip() == "":
        return "", ""
    s = str(val).strip()
    try:
        if s.isdigit():
            dt = datetime.fromtimestamp(int(s), tz=timezone.utc).astimezone(VN_TZ)
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
            if pd.isna(dt):
                return "", ""
            dt = dt.to_pydatetime().astimezone(VN_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%Y-%m-%d")
    except Exception:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            return f"{m.group(1)} 00:00:00", m.group(1)
        return "", ""


def extract_all_comments(post: Dict[str, Any]) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for key in ("comments", "topComments"):
        items = post.get(key)
        if isinstance(items, list):
            queue.extend(items)
    shared = post.get("sharedPost")
    if isinstance(shared, dict):
        for key in ("comments", "topComments"):
            items = shared.get(key)
            if isinstance(items, list):
                queue.extend(items)

    collected: List[Dict[str, Any]] = []
    while queue:
        c = queue.pop(0)
        if not isinstance(c, dict):
            continue
        collected.append(c)
        for reply_key in ("comments", "replies", "topComments"):
            replies = c.get(reply_key)
            if isinstance(replies, list):
                queue.extend(replies)
    return collected


def ingest_json_files(json_files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw_records: List[Dict[str, Any]] = []
    excluded_official_pr = 0
    empty_text_dropped = 0
    total_raw_items = 0

    for jp in sorted(json_files):
        posts = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(posts, list):
            continue
        for post_idx, post in enumerate(posts, start=1):
            if not isinstance(post, dict):
                continue

            source = str(post.get("groupTitle") or post.get("pageName") or "Unknown")
            total_raw_items += 1

            # Tách bài đăng chính thức của Fanpage (thông báo/PR) khỏi dữ liệu cảm xúc SV
            post_text = str(post.get("text") or post.get("message") or "").strip()
            shared = post.get("sharedPost")
            if isinstance(shared, dict):
                shared_text = str(shared.get("text") or shared.get("message") or "").strip()
                if shared_text and shared_text not in post_text:
                    post_text = f"{post_text} {shared_text}".strip() if post_text else shared_text

            if source in OFFICIAL_FANPAGE_SOURCES:
                excluded_official_pr += 1
            elif not post_text:
                empty_text_dropped += 1
            else:
                raw_id = str(
                    post.get("legacyId")
                    or post.get("postId")
                    or post.get("id")
                    or f"post_{jp.stem}_{post_idx}"
                )
                ts, dt = parse_timestamp(post.get("time") or post.get("date") or post.get("timestamp"))
                likes = parse_likes(
                    post.get("likesCount", post.get("likes", post.get("reactionLikeCount", 0)))
                )
                raw_records.append(
                    {
                        "id": anonymize_id(raw_id),
                        "type": "post",
                        "source": source,
                        "timestamp": ts,
                        "date": dt,
                        "likes": likes,
                        "raw_text": post_text,
                    }
                )

            # Giữ toàn bộ bình luận của sinh viên trên cả Group và Fanpage
            for c_idx, comment in enumerate(extract_all_comments(post), start=1):
                total_raw_items += 1
                c_text = str(comment.get("text") or comment.get("message") or "").strip()
                if not c_text:
                    empty_text_dropped += 1
                    continue
                c_raw_id = str(
                    comment.get("commentId")
                    or comment.get("id")
                    or f"comment_{jp.stem}_{post_idx}_{c_idx}"
                )
                ts, dt = parse_timestamp(
                    comment.get("date") or comment.get("time") or post.get("time")
                )
                c_likes = parse_likes(
                    comment.get("likesCount", comment.get("likes", comment.get("reactionLikeCount", 0)))
                )
                raw_records.append(
                    {
                        "id": anonymize_id(c_raw_id),
                        "type": "comment",
                        "source": source,
                        "timestamp": ts,
                        "date": dt,
                        "likes": c_likes,
                        "raw_text": c_text,
                    }
                )

    df = pd.DataFrame(raw_records)
    n_student_nonempty = len(df)

    # Ẩn danh hóa PII & làm sạch văn bản
    df["text"] = df["raw_text"].apply(anonymize_text)
    df["text_clean"] = df["text"].apply(clean_text)
    df["core_text"] = df["text"].apply(make_core_text)
    df["topic"] = df["text"].apply(classify_topic)

    # Lọc bỏ văn bản quá ngắn sau làm sạch (< 5 ký tự hoặc < 2 từ)
    mask_valid = (df["text_clean"].str.len() >= 5) & (df["text_clean"].str.split().str.len() >= 2)
    df = df[mask_valid].copy()
    n_after_length_filter = len(df)

    # Loại trùng lặp exact text_clean
    df = df.drop_duplicates(subset=["text_clean"], keep="first").reset_index(drop=True)
    n_after_exact_dedup = len(df)

    # Bảo đảm không còn PII trong cột text hoặc text_clean
    pii_in_text = int(df["text"].apply(contains_pii).sum())
    pii_in_clean = int(df["text_clean"].apply(contains_pii).sum())
    assert pii_in_text == 0, f"Phát hiện {pii_in_text} dòng còn PII trong cột text!"
    assert pii_in_clean == 0, f"Phát hiện {pii_in_clean} dòng còn PII trong text_clean!"

    # Sắp xếp ổn định theo thời gian
    df = df.sort_values(["timestamp", "id"], kind="mergesort").reset_index(drop=True)

    out_cols = ["id", "type", "source", "timestamp", "date", "likes", "topic", "text", "text_clean", "core_text"]
    df_out = df[out_cols].copy()

    stats = {
        "n_json_files": len(json_files),
        "total_raw_items_scanned": int(total_raw_items),
        "excluded_official_fanpage_posts": int(excluded_official_pr),
        "dropped_empty_text": int(empty_text_dropped),
        "student_nonempty_records": int(n_student_nonempty),
        "after_length_filter": int(n_after_length_filter),
        "after_exact_text_clean_dedup": int(n_after_exact_dedup),
        "unique_core_text_count": int(df_out["core_text"].nunique()),
        "pii_residual_count": 0,
        "type_distribution": df_out["type"].value_counts().to_dict(),
        "source_distribution": df_out["source"].value_counts().to_dict(),
        "topic_distribution": df_out["topic"].value_counts().to_dict(),
        "date_min": str(df_out["date"].replace("", pd.NA).dropna().min()),
        "date_max": str(df_out["date"].replace("", pd.NA).dropna().max()),
    }
    return df_out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest & anonymize real CMC Facebook JSON datasets.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out", type=str, default=str(OUT_CSV))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("dataset_*.json"))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file dataset_*.json tại {input_dir}")

    df_clean, stats = ingest_json_files(json_files)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print("HOÀN TẤT NẠP, ẨN DANH HÓA VÀ LÀM SẠCH DỮ LIỆU THẬT CMC")
    print("=" * 64)
    for k, v in stats.items():
        print(f"- {k}: {v}")
    print(f"- Đã lưu dữ liệu sạch: {out_path}")


if __name__ == "__main__":
    main()
