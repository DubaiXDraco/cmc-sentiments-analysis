import os
import sys
import time
import logging
import schedule
from datetime import datetime
from data_pipeline import run_pipeline
from train_baseline import train

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/daily_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def job():
    print(f"[{datetime.now()}] Đang chạy job cập nhật dữ liệu & dự đoán hàng ngày...")
    try:
        run_pipeline()
        train()
        logging.info("Cập nhật dữ liệu và dự đoán cảm xúc thành công.")
    except Exception as e:
        logging.error(f"Lỗi khi chạy pipeline: {e}")

if __name__ == "__main__":
    job()
    schedule.every().day.at("23:00").do(job)
    print("[OK] Đã kích hoạt lịch chạy tự động hàng ngày vào 23:00. Nhấn Ctrl+C để dừng.")
    while True:
        schedule.run_pending()
        time.sleep(60)
