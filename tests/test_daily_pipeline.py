import importlib
import sys
from pathlib import Path

import pandas as pd
import schedule

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

daily_mod = importlib.import_module("scripts.03_daily_pipeline")
sched_mod = importlib.import_module("scripts.03b_scheduler")


def test_daily_pipeline_replay_and_idempotency(tmp_path):
    clean_csv = ROOT / "data" / "cmc_student_clean.csv"
    df = pd.read_csv(clean_csv, encoding="utf-8-sig")
    sample_date = str(df["date"].dropna().iloc[-1])

    out_dir = tmp_path / "daily_out"
    res1 = daily_mod.run_daily_pipeline(
        mode="replay", target_date=sample_date, backend="tfidf_lr", out_dir=out_dir, clean_csv=clean_csv
    )
    assert res1["status"] == "success"
    assert res1["new_rows"] > 0

    out_file = out_dir / f"sentiment_{sample_date}.csv"
    assert out_file.exists()
    rows_first = len(pd.read_csv(out_file, encoding="utf-8-sig"))

    # Chạy lại cùng ngày -> idempotent, không ghi trùng
    res2 = daily_mod.run_daily_pipeline(
        mode="replay", target_date=sample_date, backend="tfidf_lr", out_dir=out_dir, clean_csv=clean_csv
    )
    assert res2["status"] == "no_new_records"
    assert res2["new_rows"] == 0
    assert res2["skipped_duplicates"] == rows_first
    assert len(pd.read_csv(out_file, encoding="utf-8-sig")) == rows_first


def test_daily_pipeline_empty_and_missing_input(tmp_path):
    clean_csv = ROOT / "data" / "cmc_student_clean.csv"
    out_dir = tmp_path / "daily_out"

    res_empty = daily_mod.run_daily_pipeline(
        mode="replay", target_date="1999-01-01", backend="tfidf_lr", out_dir=out_dir, clean_csv=clean_csv
    )
    assert res_empty["status"] == "empty"
    assert res_empty["new_rows"] == 0

    res_missing = daily_mod.run_daily_pipeline(
        mode="json", input_path=tmp_path / "does_not_exist.json", backend="tfidf_lr", out_dir=out_dir
    )
    assert res_missing["status"] == "error"


def test_schedule_job_registration():
    sched = schedule.Scheduler()
    job = sched_mod.register_daily_schedule(sched, lambda: None, at_time="08:00")
    assert len(sched.jobs) == 1
    assert job.at_time.hour == 8 and job.at_time.minute == 0
