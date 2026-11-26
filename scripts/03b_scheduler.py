#!/usr/bin/env python3
"""
scripts/03b_scheduler.py
Tuần 3: Lập lịch chạy tự động hàng ngày bằng thư viện `schedule`.
Hỗ trợ chạy vòng lặp hàng ngày lúc 08:00 hoặc chế độ demo kiểm thử ngắn (`--demo-interval-sec`).
"""

import argparse
from datetime import datetime
import importlib
from pathlib import Path
import sys
import time
from typing import Callable, Optional

import pandas as pd
import schedule

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_CSV = ROOT / "data" / "cmc_student_clean.csv"


def get_latest_available_date(clean_csv: Path = CLEAN_CSV) -> str:
    if clean_csv.exists():
        df = pd.read_csv(clean_csv, encoding="utf-8-sig")
        dates = sorted(d for d in df["date"].dropna().astype(str).unique() if d)
        if dates:
            return dates[-1]
    return datetime.now().strftime("%Y-%m-%d")


def register_daily_schedule(
    sched: schedule.Scheduler,
    job_fn: Callable[[], object],
    at_time: str = "08:00",
    interval_sec: Optional[int] = None,
) -> schedule.Job:
    """Đăng ký lịch chạy hàng ngày (hoặc theo giây trong chế độ kiểm thử)."""
    if interval_sec is not None and interval_sec > 0:
        return sched.every(interval_sec).seconds.do(job_fn)
    return sched.every().day.at(at_time).do(job_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bộ lập lịch tự động pipeline cảm xúc hàng ngày.")
    parser.add_argument("--at", type=str, default="08:00", help="Giờ chạy hàng ngày (HH:MM).")
    parser.add_argument("--demo-interval-sec", type=int, default=None, help="Chạy thử mỗi N giây.")
    parser.add_argument("--max-runs", type=int, default=0, help="Dừng sau N lần chạy (0 = vô hạn).")
    parser.add_argument("--backend", choices=["tfidf_lr", "tfidf_svm", "phobert"], default="tfidf_lr")
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    daily_mod = importlib.import_module("scripts.03_daily_pipeline")
    sched = schedule.Scheduler()
    run_counter = {"count": 0}
    target_date = args.date or get_latest_available_date()

    def _job():
        run_counter["count"] += 1
        print(f"[Scheduler] Kích hoạt lần #{run_counter['count']} lúc {datetime.now().strftime('%H:%M:%S')}")
        daily_mod.run_daily_pipeline(mode="replay", target_date=target_date, backend=args.backend)

    register_daily_schedule(
        sched,
        _job,
        at_time=args.at,
        interval_sec=args.demo_interval_sec,
    )
    print(
        f"[Scheduler] Đã đăng ký job ({'mỗi ' + str(args.demo_interval_sec) + 's' if args.demo_interval_sec else 'hàng ngày lúc ' + args.at})."
    )

    while True:
        sched.run_pending()
        if args.max_runs > 0 and run_counter["count"] >= args.max_runs:
            print(f"[Scheduler] Hoàn tất {run_counter['count']} lần chạy kiểm thử. Thoát.")
            break
        time.sleep(0.25)


if __name__ == "__main__":
    main()
